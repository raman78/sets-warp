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
ML_TRIGGER_THRESHOLD= 0.50   # if combined conf below this, try ML stage
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

        Two-phase design — autonomous recognition first, confirmed data as fallback:

          Phase A (autonomous):
            Stage 0 — community pHash knowledge override
            Stage 1 — template matching + histogram (SETS wiki-icon cache)
            Stage 2 — ML classifier (local PyTorch / HF ONNX)

          Phase B (fallback — only when Phase A confidence < TEMPLATE_THRESHOLD):
            Stage 3 — session examples (confirmed training-data crops)

        Returns:
            (item_name, confidence, thumbnail_QImage, used_session)
            item_name='' and confidence=0.0 if nothing matched.
            used_session=True means Phase A failed and Phase B rescued the result.
            Callers should treat used_session=True as a training gap.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return '', 0.0, None, False

        import cv2

        crop64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                            interpolation=cv2.INTER_AREA)
        q_hist = self._hist_hsv(crop64)

        # ── Phase A — autonomous recognition ──────────────────────────────────

        # Stage 0: community pHash knowledge override
        if self._sync_client is not None:
            try:
                from warp.knowledge.sync_client import _compute_phash
                phash     = _compute_phash(crop64)
                overrides = self._sync_client.get_knowledge()
                if phash in overrides:
                    name = overrides[phash]
                    if candidate_names is not None and name not in candidate_names:
                        log.debug(f'WARPSync: pHash override {name!r} rejected — not valid for slot')
                    else:
                        log.debug(f'WARPSync: knowledge override → {name!r}')
                        return name, 1.0, self._bgr_to_qimage(crop_bgr), False
            except Exception as e:
                log.debug(f'WARPSync: override lookup failed: {e}')

        # Stages 1+2: template matching + histogram
        auto_name  = ''
        auto_score = 0.0
        auto_entry = None

        for entry in self._index:
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            res      = cv2.matchTemplate(crop64, entry['tmpl64'],
                                         cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            if tm_score < TEMPLATE_THRESHOLD * 0.7:   # early reject
                continue
            h_score = float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL))
            if h_score < 0:
                h_score = 0.0
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT
            if combined > auto_score:
                auto_score = combined
                auto_name  = entry['name']
                auto_entry = entry

        # Stage 3 (autonomous ML): triggered when template confidence is low
        if auto_score < TEMPLATE_THRESHOLD and not self._ml_disabled:
            ml_name, ml_conf = self._classify_ml(crop64)
            if ml_name and ml_conf > auto_score:
                # Reject ML result if it's not valid for this slot
                if candidate_names is None or ml_name in candidate_names:
                    auto_name  = ml_name
                    auto_score = ml_conf
                    auto_entry = None

        # ── Phase B — session examples (confirmed crops from training data) ────
        # Only runs when autonomous recognition is not confident enough.
        # This is the fallback path — finding a match here is a training gap.
        if auto_score < TEMPLATE_THRESHOLD:
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

            if sess_score > auto_score and sess_name:
                thumb = self._bgr_to_qimage(sess_entry.get('orig')) if sess_entry else None
                return sess_name, sess_score, thumb, True   # training gap

        # Phase A result (autonomous)
        thumb = None
        if auto_entry is not None and auto_score >= TEMPLATE_THRESHOLD:
            thumb = self._bgr_to_qimage(auto_entry.get('orig'))
        return auto_name, auto_score, thumb, False

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
        """
        import cv2
        model = self._get_ml_session()
        if model is None:
            return '', 0.0
        inp = cv2.resize(crop64, (224, 224)).astype(np.float32) / 255.0
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

    def _get_ml_session(self):
        if self._ml_disabled:
            return None
        if self._ml_session:
            return self._ml_session

        sets_root  = self._find_sets_root()
        models_dir = sets_root / 'warp' / 'models'

        # Priority 1: locally trained PyTorch model (.pt)
        pt_path    = models_dir / 'icon_classifier.pt'
        label_path = models_dir / 'label_map.json'
        if pt_path.exists() and label_path.exists():
            try:
                import torch
                from torchvision.models import efficientnet_b0
                import torch.nn as nn
                with open(label_path) as f:
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
                with open(hf_label) as f:
                    raw = json.load(f)
                    self._label_map = {int(k): v for k, v in raw.items()}
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
            with open(hf_label) as f:
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
        # Walk all live instances via a class-level weak reference dict
        # Simple approach: just clear the module-level session cache flag
        # Each new SETSIconMatcher() will reload fresh from disk.
        # Existing instances need _ml_session cleared.
        cls._shared_ml_session        = None
        cls._shared_label_map         = {}
        cls._shared_ml_disabled       = False
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
