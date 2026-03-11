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

HF_REPO_ID          = 'sets-sto/icon-classifier'
HF_MODEL_FILENAME   = 'icon_classifier.onnx'
HF_LABELS_FILE      = 'label_map.json'


class SETSIconMatcher:
    """
    Multi-stage icon recognition against the SETS image cache.

    match(crop_bgr) -> (item_name, confidence, thumbnail_QImage)
      name=''  if no match above threshold.
    """

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
        self, crop_bgr: np.ndarray
    ) -> tuple[str, float, object]:   # object = QImage | None
        """
        Match a slot crop against the SETS icon library.

        Priority:
          0. Community knowledge override  (phash → confirmed item_name)
          1. Template matching + histogram (local SETS image cache)
          2. ML classifier fallback        (ONNX, optional)

        Returns:
            (item_name, confidence, thumbnail_QImage)
            item_name='' and confidence=0.0 if nothing matched.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return '', 0.0, None

        import cv2

        # Resize crop once for consistent matching
        crop64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                            interpolation=cv2.INTER_AREA)
        q_hist = self._hist_hsv(crop64)

        # ── Stage 0: community knowledge override ─────────────────────────────
        if self._sync_client is not None:
            try:
                from warp.knowledge.sync_client import _compute_phash
                phash     = _compute_phash(crop64)
                overrides = self._sync_client.get_knowledge()
                if phash in overrides:
                    name = overrides[phash]
                    log.debug(f'WARPSync: knowledge override → {name!r}')
                    return name, 1.0, self._bgr_to_qimage(crop_bgr)
            except Exception as e:
                log.debug(f'WARPSync: override lookup failed: {e}')

        best_name  = ''
        best_score = 0.0
        best_entry = None

        for entry in self._index:
            # ── Stage 1: template match ───────────────────────────────────────
            res     = cv2.matchTemplate(crop64, entry['tmpl64'],
                                        cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            if tm_score < TEMPLATE_THRESHOLD * 0.7:   # early reject
                continue

            # ── Stage 2: histogram correlation ───────────────────────────────
            h_score = float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL
            ))
            if h_score < 0:
                h_score = 0.0

            # Blend: template dominates, histogram refines
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT

            if combined > best_score:
                best_score = combined
                best_name  = entry['name']
                best_entry = entry

        # Hard threshold on final score
        if best_score < TEMPLATE_THRESHOLD and not self._ml_disabled:
            # ── Stage 3: ML fallback ──────────────────────────────────────────
            ml_name, ml_conf = self._classify_ml(crop64)
            if ml_name and ml_conf > best_score:
                best_name  = ml_name
                best_score = ml_conf

        # Build thumbnail
        thumb = None
        if best_entry is not None and best_score >= TEMPLATE_THRESHOLD:
            thumb = self._bgr_to_qimage(best_entry.get('orig'))

        return best_name, best_score, thumb

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
        Using H(18 bins) × S(8 bins) — ignores Value to be lighting-robust.
        """
        import cv2
        hsv  = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist

    # ── ML helpers ──────────────────────────────────────────────────────────────

    def _classify_ml(self, crop64: np.ndarray) -> tuple[str, float]:
        """Run ONNX classifier on a 64×64 BGR crop."""
        import cv2
        session = self._get_ml_session()
        if session is None:
            return '', 0.0

        inp = cv2.resize(crop64, (224, 224)).astype(np.float32) / 255.0
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
        try:
            out   = session.run(None, {'input': inp})[0][0]
            probs = self._softmax(out)
            top   = int(np.argmax(probs))
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
        model_path = sets_root / 'warp' / 'models' / HF_MODEL_FILENAME
        label_path = sets_root / 'warp' / 'models' / HF_LABELS_FILE

        if not model_path.exists():
            if not self._download_model(model_path, label_path):
                self._ml_disabled = True   # don't retry on every call
                return None

        if not label_path.exists():
            self._ml_disabled = True
            return None

        try:
            import onnxruntime as ort
            self._ml_session = ort.InferenceSession(str(model_path))
            with open(label_path) as f:
                raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
            log.info('WARP: ML classifier loaded')
            return self._ml_session
        except Exception as e:
            log.warning(f'WARP: ML load failed: {e}')
            self._ml_disabled = True
            return None

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
