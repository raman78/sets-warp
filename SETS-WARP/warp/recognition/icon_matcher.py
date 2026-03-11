# warp/recognition/icon_matcher.py
# Matches cropped icon images against SETS item icon library.
#
# SETS stores downloaded item images in:
#   <config_folder>/.config/images/<quote_plus(item_name)>.png
#
# This module:
#   1. Indexes all PNG files in that directory at startup
#   2. Builds a perceptual hash + histogram lookup table
#   3. For each crop: finds best match by pHash distance + histogram correlation
#   4. Falls back to ONNX ML classifier for low-confidence items
#
# Integration: receives SETS `sets_app` instance for cache/config access.

from __future__ import annotations

import logging
import json
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np

log = logging.getLogger(__name__)

PHASH_THRESHOLD  = 14    # max pHash distance to accept (0-64; lower = stricter)
HIST_THRESHOLD   = 0.65  # min histogram correlation to accept (0-1)
PHASH_MAX        = 64

HF_REPO_ID        = 'sets-sto/icon-classifier'
HF_MODEL_FILENAME = 'icon_classifier.onnx'
HF_LABELS_FILE    = 'label_map.json'


class SETSIconMatcher:
    """
    Two-stage icon recognition:
      Stage 1 — pHash + histogram vs SETS image cache (fast, offline)
      Stage 2 — ONNX EfficientNet-B0 (downloaded from HF Hub on first use)
    """

    def __init__(self, sets_app):
        self._sets      = sets_app
        self._index: list[dict] = []     # list of {name, phash, hist}
        self._ml_session = None
        self._label_map: dict[int, str] = {}
        self._build_index()

    # ── Public ─────────────────────────────────────────────────────────────

    def match(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Stage 1: match crop against SETS icon index.
        Returns (item_name, confidence).  name='' if no match.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return '', 0.0

        import cv2
        icon = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        qh   = self._phash(icon)
        qhist= self._hist(icon)

        best_name = ''
        best_conf = 0.0

        for entry in self._index:
            dist = bin(int(qh) ^ int(entry['phash'])).count('1')
            if dist > PHASH_THRESHOLD:
                continue
            corr = cv2.compareHist(qhist, entry['hist'], cv2.HISTCMP_CORREL)
            if corr < HIST_THRESHOLD:
                continue
            conf = 1.0 - dist / PHASH_MAX
            if conf > best_conf:
                best_conf = conf
                best_name = entry['name']

        return best_name, best_conf

    def classify_ml_batch(
        self,
        thumbnails: list   # list of QImage | None
    ) -> tuple[list[str], list[float]]:
        """Stage 2: ONNX ML batch classifier."""
        session = self._get_ml_session()
        if session is None:
            return [''] * len(thumbnails), [0.0] * len(thumbnails)

        import cv2
        names, confs = [], []
        for thumb in thumbnails:
            arr = self._qimage_to_bgr(thumb)
            if arr is None:
                names.append(''); confs.append(0.0); continue

            inp = cv2.resize(arr, (224, 224)).astype(np.float32) / 255.0
            inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
            out = session.run(None, {'input': inp})[0][0]
            probs = self._softmax(out)
            top   = int(np.argmax(probs))
            names.append(self._label_map.get(top, ''))
            confs.append(float(probs[top]))

        return names, confs

    # ── Index building ──────────────────────────────────────────────────────

    def _build_index(self):
        """
        Scans SETS images directory and builds pHash + histogram index.
        Item name is recovered from filename via unquote_plus().
        """
        images_dir = self._get_images_dir()
        if images_dir is None or not images_dir.exists():
            log.warning('WARP: SETS images directory not found — template matching disabled')
            return

        import cv2
        count = 0
        for png in images_dir.glob('*.png'):
            name = unquote_plus(png.stem)
            img  = cv2.imread(str(png))
            if img is None:
                continue
            icon = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            self._index.append({
                'name':  name,
                'phash': self._phash(icon),
                'hist':  self._hist(icon),
            })
            count += 1

        log.info(f'WARP: indexed {count} icons from {images_dir}')

    def _get_images_dir(self) -> Path | None:
        if self._sets is None:
            return None
        try:
            return Path(self._sets.config['config_subfolders']['images'])
        except Exception:
            return None

    # ── Hash helpers ────────────────────────────────────────────────────────

    def _phash(self, icon_bgr: np.ndarray) -> int:
        """64-bit perceptual hash (DCT-based)."""
        import cv2
        gray  = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        small = cv2.resize(gray, (32, 32))
        dct   = np.fft.rfft2(small)
        low   = np.abs(dct[:8, :8]).flatten()
        mean  = low.mean()
        bits  = low > mean
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def _hist(self, icon_bgr: np.ndarray) -> np.ndarray:
        """Normalised BGR histogram for colour matching."""
        import cv2
        hist = cv2.calcHist([icon_bgr], [0, 1, 2], None,
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    # ── ML helpers ──────────────────────────────────────────────────────────

    def _get_ml_session(self):
        if self._ml_session:
            return self._ml_session

        sets_root  = self._find_sets_root()
        model_path = sets_root / 'warp' / 'models' / HF_MODEL_FILENAME
        label_path = sets_root / 'warp' / 'models' / HF_LABELS_FILE

        if not model_path.exists():
            if not self._download_model(model_path):
                return None

        if not label_path.exists():
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
            return None

    def _download_model(self, dest: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME,
                local_dir=str(dest.parent))
            # Also download label map
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_LABELS_FILE,
                local_dir=str(dest.parent))
            return dest.exists()
        except Exception as e:
            log.warning(f'WARP: model download failed: {e}')
            return False

    # ── Misc helpers ────────────────────────────────────────────────────────

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(6):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def _qimage_to_bgr(self, qimg) -> np.ndarray | None:
        if qimg is None:
            return None
        try:
            import cv2
            from PySide6.QtGui import QImage
            q = qimg.convertToFormat(QImage.Format.Format_RGB888)
            w, h = q.width(), q.height()
            ptr = q.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
