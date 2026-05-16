# warp/recognition/icon_matcher.py
#
# Matches cropped icon images against SETS item icon library.
#
# SETS stores downloaded item images in:
#   <config_folder>/images/<quote_plus(item_name)>.png
#
# RECOGNITION STRATEGY (in priority order):
#
#   Stage 1 — Multi-scale template matching (primary, no training needed)
#     For each slot crop:
#       a) Resize crop to MATCH_SIZE×MATCH_SIZE
#       b) For each template in index: cv2.matchTemplate (TM_CCOEFF_NORMED)
#       c) Best match above TEMPLATE_THRESHOLD wins
#     Advantages over pHash+histogram:
#       - Sensitive to icon shape, not just color distribution
#       - Robust to STO's icon rendering at different UI scales
#       - Works immediately from the SETS image cache (no ML training)
#
#   Stage 2 — Color histogram fallback
#     When template matching confidence is low, use HSV histogram correlation
#     as a secondary signal to break ties or rescue near-misses.
#
#   Stage 3 — ML classifier (optional, downloaded from HF Hub)
#     ONNX EfficientNet-B0 trained on SETS icon library.
#     Activated when both Stage 1+2 fail (conf < ML_TRIGGER_THRESHOLD).
#
# The public match() method returns (name, confidence, thumbnail_QImage).

from __future__ import annotations

import logging
import json
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np

log = logging.getLogger(__name__)

# ── Tunable thresholds ─────────────────────────────────────────────────────────
MATCH_SIZE          = 64     # resize crop + template to this before matching
TEMPLATE_THRESHOLD  = 0.55   # min TM_CCOEFF_NORMED score to accept a match
HIST_WEIGHT         = 0.20   # weight of histogram score when blending with template
HIST_THRESHOLD      = 0.50   # min histogram correlation to contribute
ML_PRIMARY_THRESHOLD= 0.50   # ML conf >= this → ML is the source of truth
VIRTUAL_OVERRIDE_CONF = 0.40 # when ML returns a real icon with conf >= this,
                             # suppress virtual (__empty__/__inactive__)
                             # session/template candidates
ML_TRIGGER_THRESHOLD= 0.50   # if combined conf below this, try ML stage (legacy)
FUSION_THRESHOLD    = 0.75   # P8: run ML and fuse scores when template < this (legacy)
HIST_BINS           = [18, 16] # H×S bins for _hist_hsv — must match everywhere

HF_REPO_ID          = 'sets-sto/icon-classifier'
HF_MODEL_FILENAME   = 'icon_classifier.onnx'
HF_LABELS_FILE      = 'label_map.json'
# Sentinel file written after a failed availability check.
# Prevents repeated 401/404 download attempts across sessions.
HF_UNAVAILABLE_FILE = 'model_unavailable.flag'
# How many hours to wait before retrying after a failed check
HF_RETRY_HOURS      = 24


class SETSIconMatcher:
    """
    Multi-stage icon recognition against the SETS image cache.

    match(crop_bgr) -> (item_name, confidence, thumbnail_QImage, used_session)
      name=''  if no match above threshold.
      used_session=True when autonomous ML/template recognition failed and the
      result came from confirmed training-data crops (session examples).
      Callers should log this as a training gap for future ML improvement.
    """

    # Session examples: confirmed crops added by user during this session.
    # Shared across all instances so every match() call benefits.
    _session_examples: list[dict] = []   # {name, tmpl64, hist_hsv, orig}

    # Guard: prevent re-seeding from training data on every new matcher instance.
    _seeded_from_training_data: bool = False

    def __init__(self, sets_app, sync_client=None):
        self._sets        = sets_app
        self._index: list[dict] = []   # {name, tmpl64, hist_hsv, path}
        self._ml_session  = None
        self._ml_disabled = False      # True after first failed download attempt
        self._label_map: dict[int, str] = {}
        # Metric-learning path: when icon_embedder.pt is present, _ml_session is
        # the embedder model and _gallery_* hold the k-NN search index. When
        # _ml_kind=='classifier' (legacy softmax), _gallery_* stay None.
        self._ml_kind: str = ''        # 'embedder' | 'classifier' | ''
        self._gallery_emb = None       # np.ndarray (N, D) float32, L2-normed
        self._gallery_lbl = None       # np.ndarray (N,) int32 — indices into _label_map
        # Diagnostic: source of the most recent match() decision.
        # Values: 'ml' (embedder/classifier), 'template' (wiki PNG histogram),
        # 'session' (confirmed training crop), 'knowledge' (pHash override),
        # 'none' (no signal above threshold), '' (no match attempted).
        # Read by warp_importer to expose match source in autodetect logs.
        self._last_match_src: str = ''
        # Per-stage raw scores from the most recent match() call. Filled in
        # before every return path (knowledge / no-candidates / final winner).
        # Consumed by RecognitionWorker to build the per-image match summary
        # table. Keys: 'embed', 'soft', 'session', 'template', 'knowledge'.
        self._last_stage_scores: dict[str, float] = {
            'embed': 0.0, 'soft': 0.0, 'session': 0.0,
            'template': 0.0, 'knowledge': 0.0,
        }
        self._sync_client = sync_client  # WARPSyncClient | None
        self._build_index()

    # ── Public ─────────────────────────────────────────────────────────────────

    def match(
        self,
        crop_bgr: np.ndarray,
        candidate_names: set[str] | None = None,
    ) -> tuple[str, float, object, bool]:
        """
        Match a slot crop against the SETS icon library.

        candidate_names: optional set of allowed item names.
          When provided, only entries in this set are considered.

        ML-primary design (2026-05-15):
          Stage 0 — community pHash knowledge override (hard override, trust=1.0)
          Stage 1 — ML classifier (local PyTorch / HF ONNX) — PRIMARY SOURCE
                    when ml_conf >= ML_PRIMARY_THRESHOLD AND result is in candidate_names
          Fallback (only when Stage 1 is uncertain / out of candidates):
            Stage 2 — template matching + histogram (SETS wiki-icon cache)
            Stage 3 — session examples (confirmed training-data crops)
            Stage 4 — last resort: weak ML result (better than nothing)

        Rationale: ML is trained on real game-screenshot crops (via
        sync.py → admin_train.py), so it generalizes to actual rendered
        icons including virtual states (__empty__, __inactive__). Template
        matching against wiki PNGs and session examples suffer from HSV-
        distribution mismatch on dimly-rendered cells, producing false
        positives (e.g. filled icon → __empty__). Treating ML as primary
        eliminates that class of error; the fallback chain only kicks in
        for items genuinely missing from the model's label_map.

        Returns:
            (item_name, confidence, thumbnail_QImage, used_session)
            item_name='' and confidence=0.0 if nothing matched.
            used_session=True means Stage 3 (session example) rescued the
            result — a training gap signal for the caller.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            self._last_match_src = ''
            self._last_stage_scores = {'embed': 0.0, 'soft': 0.0,
                                       'session': 0.0, 'template': 0.0,
                                       'knowledge': 0.0}
            return '', 0.0, None, False

        import cv2
        self._last_match_src = ''
        self._last_stage_scores = {'embed': 0.0, 'soft': 0.0,
                                   'session': 0.0, 'template': 0.0,
                                   'knowledge': 0.0}

        crop64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                            interpolation=cv2.INTER_AREA)
        q_hist = self._hist_hsv(crop64)

        # Stage 0: community pHash knowledge override (hard override)
        if self._sync_client is not None:
            try:
                from warp.knowledge.sync_client import _compute_phash
                phash     = _compute_phash(crop64)
                overrides = self._sync_client.get_knowledge()
                if phash in overrides:
                    name = overrides[phash]
                    # Defense-in-depth: never let knowledge.json hard-override a
                    # crop to a virtual class (__empty__ / __inactive__) or a
                    # leftover dev-test entry. Such entries pollute Stage 0 and
                    # used to silently turn real icons into empty slots at
                    # conf=1.0. Skip the override — fall through to ML/template.
                    if name.startswith('__') or name == 'Test Item Name':
                        log.debug(f'WARPSync: pHash override {name!r} suppressed (virtual/test)')
                    elif candidate_names is not None and name not in candidate_names:
                        log.debug(f'WARPSync: pHash override {name!r} rejected — not valid for slot')
                    else:
                        log.debug(f'WARPSync: knowledge override → {name!r}')
                        self._last_match_src = 'knowledge'
                        self._last_stage_scores['knowledge'] = 1.0
                        return name, 1.0, self._bgr_to_qimage(crop_bgr), False
            except Exception as e:
                log.debug(f'WARPSync: override lookup failed: {e}')

        # Stage 1: ML classifier — always consulted (one of three signals)
        ml_name, ml_conf = ('', 0.0)
        if not self._ml_disabled:
            ml_name, ml_conf = self._classify_ml(crop64)

        # Stage 2: template matching + histogram against wiki PNGs
        auto_name  = ''
        auto_score = 0.0
        auto_entry = None
        for entry in self._index:
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            res      = cv2.matchTemplate(crop64, entry['tmpl64'],
                                         cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            if tm_score < TEMPLATE_THRESHOLD * 0.7:
                continue
            h_score = max(0.0, float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL)))
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT
            if combined > auto_score:
                auto_score = combined
                auto_name  = entry['name']
                auto_entry = entry

        # Stage 3: session examples (confirmed training-data crops)
        sess_name, sess_score, sess_entry = self._best_session_match(
            crop64, q_hist, candidate_names)

        # Record raw per-stage scores for the summary table.
        if self._ml_kind == 'embedder':
            self._last_stage_scores['embed'] = float(ml_conf)
        elif self._ml_kind == 'classifier':
            self._last_stage_scores['soft']  = float(ml_conf)
        self._last_stage_scores['template'] = float(auto_score)
        self._last_stage_scores['session']  = float(sess_score)

        # Combine all signals — strongest wins. No hard threshold here;
        # caller (warp_importer) applies MIN_ACCEPT_CONF as final gate.
        # Anti-virtual-bias rule: when ML returned a real icon with decent
        # confidence (>= VIRTUAL_OVERRIDE_CONF), suppress virtual session /
        # template matches (__empty__/__inactive__). This is the Bug 2 fix —
        # session-virtual was beating real ML on filled icons due to HSV
        # histogram bias of dim cells. ML is still NOT mandatory to win;
        # template/session with a real icon name can outscore it.
        ml_real = bool(ml_name) and not ml_name.startswith('__')
        suppress_virtual = ml_real and ml_conf >= VIRTUAL_OVERRIDE_CONF

        def _virtual(n: str) -> bool:
            return bool(n) and n.startswith('__')

        candidates = []
        if sess_name and not (suppress_virtual and _virtual(sess_name)):
            candidates.append(('session', sess_name, sess_score, sess_entry))
        if auto_name and not (suppress_virtual and _virtual(auto_name)):
            candidates.append(('template', auto_name, auto_score, auto_entry))
        if ml_name and (candidate_names is None or ml_name in candidate_names):
            candidates.append(('ml', ml_name, ml_conf, None))
        if not candidates:
            self._last_match_src = 'none'
            return '', 0.0, None, False
        src, name, score, entry = max(candidates, key=lambda x: x[2])
        # Disambiguate ML source by model kind so logs distinguish the
        # ArcFace embedder from the legacy softmax classifier.
        if src == 'ml' and self._ml_kind == 'embedder':
            self._last_match_src = 'embed'
        elif src == 'ml':
            self._last_match_src = 'soft'
        else:
            self._last_match_src = src
        if entry is not None:
            thumb = self._bgr_to_qimage(entry.get('orig'))
        else:
            thumb = self._thumb_for_name(name)
        return name, score, thumb, (src == 'session')

    def _thumb_for_name(self, name: str) -> object:
        """Return a QImage thumbnail for an item name by looking it up in the
        wiki PNG index. Returns None for virtual items (__empty__/__inactive__)
        or when the name is not in the index."""
        if not name or name.startswith('__'):
            return None
        for entry in self._index:
            if entry['name'] == name:
                return self._bgr_to_qimage(entry.get('orig'))
        return None

    def _best_session_match(
        self,
        crop64: np.ndarray,
        q_hist: np.ndarray,
        candidate_names: set[str] | None,
    ) -> tuple[str, float, dict | None]:
        """Return (name, score, entry) for the best session example match."""
        import cv2
        expected_shape = tuple(HIST_BINS)
        sess_name  = ''
        sess_score = 0.0
        sess_entry = None
        for entry in self._session_examples:
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            if entry['hist_hsv'].shape != expected_shape:
                continue
            res      = cv2.matchTemplate(crop64, entry['tmpl64'],
                                         cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            h_score  = max(0.0, float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL)))
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT
            if combined > sess_score:
                sess_score = combined
                sess_name  = entry['name']
                sess_entry = entry
        return sess_name, sess_score, sess_entry

    def classify_patch(self, patch_bgr: np.ndarray) -> tuple[str, float]:
        """Classify a single BGR patch using ML only (fast path for dense scanning)."""
        import cv2
        if patch_bgr is None or patch_bgr.size == 0:
            return '', 0.0
        crop64 = cv2.resize(patch_bgr, (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
        return self._classify_ml(crop64)

    def classify_ml_batch(
        self,
        thumbnails: list    # list[QImage | None]
    ) -> tuple[list[str], list[float]]:
        """Stage 3 batch classifier (ONNX EfficientNet-B0)."""
        session = self._get_ml_session()
        if session is None:
            return [''] * len(thumbnails), [0.0] * len(thumbnails)

        import cv2
        names, confs = [], []
        for thumb in thumbnails:
            arr = self._qimage_to_bgr(thumb)
            if arr is None:
                names.append(''); confs.append(0.0)
                continue
            name, conf = self._classify_ml(
                cv2.resize(arr, (MATCH_SIZE, MATCH_SIZE))
            )
            names.append(name)
            confs.append(conf)

        return names, confs

    # ── Index building ──────────────────────────────────────────────────────────

    def _build_index(self):
        """
        Load all PNG files from the SETS images directory and build
        a template + histogram index for fast matching.
        """
        images_dir = self._get_images_dir()
        if images_dir is None or not images_dir.exists():
            log.warning(
                'WARP: SETS images directory not found — '
                'icon matching disabled. '
                f'Expected: {images_dir}'
            )
            return

        import cv2
        count = 0
        for png in images_dir.glob('*.png'):
            name = unquote_plus(png.stem)
            orig = cv2.imread(str(png))
            if orig is None:
                continue

            tmpl64 = cv2.resize(orig, (MATCH_SIZE, MATCH_SIZE),
                                 interpolation=cv2.INTER_AREA)
            self._index.append({
                'name':     name,
                'tmpl64':   tmpl64,
                'hist_hsv': self._hist_hsv(tmpl64),
                'orig':     orig,      # kept for thumbnail generation
            })
            count += 1

        log.info(f'WARP: indexed {count} icons from {images_dir}')

    def _get_images_dir(self) -> Path | None:
        if self._sets is None:
            return None
        try:
            # Primary: config sub-folder key
            return Path(self._sets.config['config_subfolders']['images'])
        except Exception:
            pass
        try:
            # Fallback: derive from config_folder
            base = Path(self._sets.config['config_folder'])
            candidate = base / 'images'
            if candidate.exists():
                return candidate
        except Exception:
            pass
        return None

    # ── Feature helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _hist_hsv(icon_bgr: np.ndarray) -> np.ndarray:
        """
        Normalised HSV histogram.
        Using H(18 bins) × S(16 bins) — ignores Value to be lighting-robust.
        """
        import cv2
        hsv  = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist

    # ── ML helpers ──────────────────────────────────────────────────────────────

    def _classify_ml(self, crop64: np.ndarray) -> tuple[str, float]:
        """Run local PyTorch classifier on a 64x64 BGR crop.
        Falls back to ONNX session for HuggingFace-downloaded model.

        Preprocessing must match admin_train.py CropDataset.__getitem__:
          1. BGR → RGB  (training uses cv2.COLOR_BGR2RGB)
          2. /255.0
          3. ImageNet mean/std normalization  (training uses T.Normalize)
        Missing either step produces a completely wrong input distribution
        (model was trained on normalized RGB, but would receive raw BGR).
        """
        import cv2
        model = self._get_ml_session()
        if model is None:
            return '', 0.0
        # Metric-learning path: model is an Embedder, _gallery_* hold the k-NN index.
        if self._ml_kind == 'embedder':
            return self._classify_ml_embed(crop64)
        rgb = cv2.cvtColor(cv2.resize(crop64, (224, 224)), cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        # ImageNet normalization (same as T.Normalize in admin_train.py)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
        try:
            if hasattr(model, 'run'):  # onnxruntime session (HuggingFace model)
                out   = model.run(None, {'input': inp})[0][0]
                probs = self._softmax(out)
            else:                      # PyTorch model (locally trained)
                import torch
                t = torch.from_numpy(inp)
                with torch.no_grad():
                    out = model(t)[0]
                probs = self._softmax(out.numpy())
            top = int(np.argmax(probs))
            return self._label_map.get(top, ''), float(probs[top])
        except Exception as e:
            log.debug(f'WARP: ML classify error: {e}')
            return '', 0.0

    def _classify_ml_embed(self, crop64: np.ndarray) -> tuple[str, float]:
        """Embed a crop and return the nearest-neighbour label from the gallery.

        Confidence is the cosine similarity to the nearest gallery embedding,
        clamped to [0, 1] — same range as the softmax classifier's confidence,
        so the rest of the fallback chain treats both models interchangeably.
        """
        import cv2
        if self._gallery_emb is None or self._gallery_lbl is None:
            return '', 0.0
        rgb = cv2.cvtColor(cv2.resize(crop64, (224, 224)), cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
        try:
            import torch
            t = torch.from_numpy(inp)
            with torch.no_grad():
                emb = self._ml_session(t).numpy()[0]    # (D,) already L2-normed
            sims = self._gallery_emb @ emb              # (N,) cosine similarity
            top = int(np.argmax(sims))
            best_lbl = int(self._gallery_lbl[top])
            conf = float(max(0.0, min(1.0, sims[top])))
            return self._label_map.get(best_lbl, ''), conf
        except Exception as e:
            log.debug(f'WARP: ML embed error: {e}')
            return '', 0.0

    def _get_ml_session(self):
        if self._ml_disabled:
            return None
        if self._ml_session:
            return self._ml_session

        sets_root  = self._find_sets_root()
        models_dir = sets_root / 'warp' / 'models'

        # Priority 0: metric-learning embedder (icon_embedder.pt + gallery index)
        # Uses embedder_label_map.json so its class space stays disjoint from
        # the softmax classifier's label_map.json (different class counts).
        emb_path     = models_dir / 'icon_embedder.pt'
        gallery_path = models_dir / 'embedding_index.npz'
        emb_label    = models_dir / 'embedder_label_map.json'
        if emb_path.exists() and gallery_path.exists() and emb_label.exists():
            try:
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from torchvision.models import efficientnet_b0
                with open(emb_label, encoding='utf-8') as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
                # Match admin_train_metric.py architecture: backbone with no classifier,
                # plus a Linear projection to EMBED_DIM with L2-normalize on output.
                gallery = np.load(str(gallery_path))
                embed_dim = int(gallery['embeddings'].shape[1])
                backbone = efficientnet_b0(weights=None)
                in_features = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()

                class Embedder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.backbone = backbone
                        self.proj = nn.Linear(in_features, embed_dim)
                    def forward(self, x):
                        f = self.backbone(x)
                        return F.normalize(self.proj(f), dim=1)

                model = Embedder()
                model.load_state_dict(torch.load(str(emb_path), map_location='cpu',
                                                  weights_only=True))
                model.eval()
                self._ml_session = model
                self._ml_kind = 'embedder'
                self._gallery_emb = gallery['embeddings'].astype(np.float32)
                self._gallery_lbl = gallery['labels'].astype(np.int32)
                log.info(f'WARP: metric-learning embedder loaded '
                         f'({len(self._label_map)} classes, '
                         f'gallery={len(self._gallery_emb)}, dim={embed_dim})')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: embedder load failed: {e} — falling back to classifier')

        # Priority 1: locally trained PyTorch model (.pt)
        pt_path    = models_dir / 'icon_classifier.pt'
        label_path = models_dir / 'label_map.json'
        if pt_path.exists() and label_path.exists():
            try:
                import torch
                from torchvision.models import efficientnet_b0
                import torch.nn as nn
                with open(label_path, encoding='utf-8') as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
                n_classes = len(self._label_map)
                model = efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, n_classes)
                model.load_state_dict(torch.load(str(pt_path), map_location='cpu',
                                                  weights_only=True))
                model.eval()
                self._ml_session = model
                self._ml_kind = 'classifier'
                log.info(f'WARP: local PyTorch icon classifier loaded ({n_classes} classes)')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: local .pt load failed: {e}')

        # Priority 2: ONNX model from HuggingFace Hub
        model_path = models_dir / HF_MODEL_FILENAME
        hf_label   = models_dir / HF_LABELS_FILE
        flag_path  = models_dir / HF_UNAVAILABLE_FILE

        if model_path.exists() and hf_label.exists():
            try:
                import onnxruntime as ort
                self._ml_session = ort.InferenceSession(str(model_path))
                with open(hf_label, encoding='utf-8') as f:
                    raw = json.load(f)
                    self._label_map = {int(k): v for k, v in raw.items()}
                self._ml_kind = 'classifier'
                log.info('WARP: HuggingFace ONNX icon classifier loaded')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: HF ONNX load failed: {e}')
                self._ml_disabled = True
                return None

        # Check sentinel
        if flag_path.exists():
            import time
            age_h = (time.time() - flag_path.stat().st_mtime) / 3600
            if age_h < HF_RETRY_HOURS:
                self._ml_disabled = True
                return None
            flag_path.unlink(missing_ok=True)

        # Attempt HuggingFace download
        if not self._check_repo_exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            flag_path.touch()
            self._ml_disabled = True
            return None

        if not self._download_model(model_path, hf_label):
            models_dir.mkdir(parents=True, exist_ok=True)
            flag_path.touch()
            self._ml_disabled = True
            return None

        try:
            import onnxruntime as ort
            self._ml_session = ort.InferenceSession(str(model_path))
            with open(hf_label, encoding='utf-8') as f:
                raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
            log.info('WARP: HuggingFace ONNX icon classifier loaded')
            return self._ml_session
        except Exception as e:
            log.warning(f'WARP: HF ONNX load failed: {e}')
            self._ml_disabled = True
            return None

    @classmethod
    def add_session_example(cls, crop_bgr: 'np.ndarray', name: str) -> None:
        """
        Add a user-confirmed crop to the in-memory session index.
        Immediately improves recognition for the rest of this session
        without any retraining.
        """
        import cv2
        if crop_bgr is None or crop_bgr.size == 0 or not name.strip():
            return
        tmpl64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                             interpolation=cv2.INTER_AREA)
        hist = cls._hist_hsv(tmpl64)
        cls._session_examples.append({
            'name':     name,
            'tmpl64':   tmpl64,
            'hist_hsv': hist,
            'orig':     crop_bgr,
        })
        log.debug(f'WARP: session example added for {name!r} '
                  f'({len(cls._session_examples)} total)')

    @classmethod
    def seed_from_training_data(cls, training_data_dir) -> int:
        """
        Load all confirmed icon crops from annotations.json as session examples.
        Guarded by _seeded_from_training_data — runs only once per process
        lifetime (reset by reset_ml_session).
        Returns the number of crops loaded (0 if already seeded).
        """
        if cls._seeded_from_training_data:
            return 0

        import json
        import cv2
        from pathlib import Path

        training_data_dir = Path(training_data_dir)
        ann_path = training_data_dir / 'annotations.json'
        if not ann_path.exists():
            return 0
        try:
            data = json.loads(ann_path.read_text(encoding='utf-8'))
        except Exception as e:
            log.warning(f'WARP: seed_from_training_data: {e}')
            return 0

        # These slots have no crop PNGs — skip them
        _TEXT_SLOTS = frozenset({
            'Ship Name', 'Ship Type', 'Ship Tier',
            'Primary Specialization', 'Secondary Specialization',
        })
        crops_dir = training_data_dir / 'crops'
        count = 0
        for _fname, annotations in data.items():
            for ann in annotations:
                if ann.get('state') != 'confirmed':
                    continue
                name = ann.get('name', '').strip()
                slot = ann.get('slot', '')
                if not name or slot in _TEXT_SLOTS:
                    continue

                # Primary: explicit crop_name field (newer annotations)
                crop_path = None
                crop_name = ann.get('crop_name', '')
                if crop_name:
                    p = training_data_dir / crop_name
                    if p.exists():
                        crop_path = p

                # Fallback: reconstruct filename from slot + name + ann_id
                # (matches TrainingDataManager._export_crop naming convention)
                if crop_path is None:
                    ann_id = ann.get('ann_id', '')
                    if ann_id:
                        safe_slot = slot.replace(' ', '_').lower()
                        safe_name = name.replace(' ', '_').lower()[:40]
                        fname = f'{safe_slot}__{safe_name}__{ann_id}.png'
                        p = crops_dir / fname
                        if p.exists():
                            crop_path = p

                if crop_path is None:
                    continue
                img = cv2.imread(str(crop_path))
                if img is None:
                    continue
                cls.add_session_example(img, name)
                count += 1

        cls._seeded_from_training_data = True
        log.info(f'WARP: training data seed: {count} session examples from {len(data)} screenshots '
                 f'(path: {training_data_dir})')
        return count

    @classmethod
    def reset_ml_session(cls):
        """
        Force reload of the ML model on next inference call.
        Called after local training completes.
        """
        # New SETSIconMatcher() instances will reload fresh from disk.
        # Existing instances keep their old model until garbage-collected.
        # (_shared_* attributes don't exist; instance attrs are _ml_session etc.)
        cls._session_examples         = []
        cls._seeded_from_training_data = False
        log.info('WARP: ML session reset -- will reload on next match')

    def _check_repo_exists(self) -> bool:
        """
        Do a lightweight HEAD request to check if the HF repo exists.
        Returns False silently on 401/404 or any network error.
        """
        try:
            import urllib.request
            url = f'https://huggingface.co/{HF_REPO_ID}'
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=6) as r:
                return r.status == 200
        except Exception:
            return False

    def _download_model(self, dest: Path, label_path: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME,
                            local_dir=str(dest.parent))
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_LABELS_FILE,
                            local_dir=str(dest.parent))
            return dest.exists()
        except Exception as e:
            log.warning(f'WARP: model download failed: {e}')
            return False

    # ── Misc helpers ────────────────────────────────────────────────────────────

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(6):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def _bgr_to_qimage(self, img_bgr: np.ndarray | None):
        if img_bgr is None:
            return None
        try:
            import cv2
            from PySide6.QtGui import QImage
            rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            return QImage(rgb.data, w, h, 3 * w,
                          QImage.Format.Format_RGB888).copy()
        except Exception:
            return None

    def _qimage_to_bgr(self, qimg) -> np.ndarray | None:
        if qimg is None:
            return None
        try:
            import cv2
            from PySide6.QtGui import QImage
            q   = qimg.convertToFormat(QImage.Format.Format_RGB888)
            w, h = q.width(), q.height()
            arr  = np.frombuffer(q.bits(), dtype=np.uint8).reshape((h, w, 3)).copy()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
