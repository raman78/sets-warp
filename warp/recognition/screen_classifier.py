# warp/recognition/screen_classifier.py
#
# Screen-type classifier for STO screenshots.
#
# Classification pipeline (in priority order):
#
#   Stage 1 — ONNX MobileNetV3-Small  (trained on user corrections)
#     Fast (~50 ms CPU), 8-class softmax output.
#     Loaded from:  <sets_root>/warp/models/screen_classifier.onnx
#     Label map:    <sets_root>/warp/models/screen_classifier_labels.json
#     Used only when model exists AND confidence ≥ CONF_THRESHOLD.
#
#   Stage 2 — Session k-NN  (in-memory, current session only)
#     Each user correction this session is stored as a 224×224 HSV histogram.
#     Cosine nearest-neighbour among session examples.
#     Used when Stage 1 absent or low-confidence AND ≥ MIN_SESSION_EXAMPLES total.
#
#   Stage 3 — OCR fallback  (TextExtractor)
#     Original keyword-matching on two scan regions.
#     Always available, used when Stages 1+2 fail or are uncertain.
#
# Public API:
#   classifier = ScreenTypeClassifier(models_dir)
#   stype, conf = classifier.classify(img_bgr)          # full pipeline
#   classifier.add_session_example(img_bgr, stype)      # call on user correction
#
# Training data is NOT managed here — see screen_type_trainer.py.

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

# ── Constants ──────────────────────────────────────────────────────────────────
SCREEN_TYPES = [
    'SPACE_EQ', 'GROUND_EQ', 'TRAITS',
    'BOFFS', 'SPECIALIZATIONS', 'SPACE_MIXED', 'GROUND_MIXED',
]

INPUT_SIZE        = 224     # MobileNetV3 input
CONF_THRESHOLD    = 0.70    # min softmax confidence to trust ML result
SESSION_THRESHOLD = 0.65    # min cosine similarity to trust k-NN session result
MIN_SESSION_EXAMPLES = 2    # need at least this many session examples to use k-NN

MODEL_FILENAME  = 'screen_classifier.onnx'
LABELS_FILENAME = 'screen_classifier_labels.json'

# ImageNet normalisation (MobileNetV3 pre-trained)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_224(img_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE),
                      interpolation=cv2.INTER_AREA)


def _to_chw_float(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → normalised CHW float32 (1, 3, H, W)."""
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return rgb.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


def _hist_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Compute a flattened, L2-normalised HSV histogram (36×32 bins)."""
    import cv2
    small = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class ScreenTypeClassifier:
    """
    Three-stage screen-type classifier.

    Thread-safe for read operations (classify).  add_session_example may be
    called from the main thread after a user correction.
    """

    # Class-level session memory — shared across all instances in the process
    _session_examples: list[dict] = []   # {stype, hist}

    def __init__(self, models_dir: Path):
        self._models_dir  = Path(models_dir)
        self._session      = None   # onnxruntime.InferenceSession | None
        self._label_map: dict[int, str] = {}
        self._ml_disabled = False
        self._try_load_model()

    # ── Public API ──────────────────────────────────────────────────────────────

    def classify(self, img_bgr: np.ndarray) -> tuple[str, float]:
        """
        Classify screenshot.
        Returns (screen_type_str, confidence_0_to_1).
        screen_type_str is '' if nothing confident enough.
        """
        if img_bgr is None or img_bgr.size == 0:
            return '', 0.0

        # Stage 1: ONNX model
        ml_stype, ml_conf = self._classify_ml(img_bgr)
        if ml_conf >= CONF_THRESHOLD:
            log.debug(f'ScreenClassifier ML: {ml_stype} ({ml_conf:.2f})')
            return ml_stype, ml_conf

        # Stage 2: session k-NN
        knn_stype, knn_conf = self._classify_session(img_bgr)
        if knn_conf >= SESSION_THRESHOLD:
            log.debug(f'ScreenClassifier k-NN: {knn_stype} ({knn_conf:.2f})')
            return knn_stype, knn_conf

        # Return best available even if below threshold — caller decides
        if ml_conf > knn_conf and ml_stype:
            return ml_stype, ml_conf
        if knn_stype:
            return knn_stype, knn_conf

        return '', 0.0

    @classmethod
    def add_session_example(cls, img_bgr: np.ndarray, stype: str) -> None:
        """
        Register a user-confirmed screen type for this session.
        Immediately improves k-NN classification for subsequent screenshots.
        """
        if img_bgr is None or img_bgr.size == 0 or stype not in SCREEN_TYPES:
            return
        hist = _hist_hsv(img_bgr)
        cls._session_examples.append({'stype': stype, 'hist': hist})
        log.debug(
            f'ScreenClassifier: session example added for {stype!r} '
            f'({len(cls._session_examples)} total)')

    @classmethod
    def clear_session(cls) -> None:
        cls._session_examples = []

    def reload_model(self) -> None:
        """Call after training a new model to load the updated weights."""
        self._session    = None
        self._ml_disabled = False
        self._label_map  = {}
        self._try_load_model()

    # ── Stage 1: ONNX ───────────────────────────────────────────────────────────

    # ── Stage 1: PyTorch model ──────────────────────────────────────────────────

    def _try_load_model(self) -> None:
        pt_path     = self._models_dir / 'screen_classifier.pt'
        labels_path = self._models_dir / 'screen_classifier_labels.json'
        meta_path   = self._models_dir / 'screen_classifier_meta.json'
        if not pt_path.exists():
            _slog.debug('ScreenClassifier: no .pt model found, using OCR fallback')
            return
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            import torch.nn as nn
            # Load metadata
            n_classes = 7  # default
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                n_classes = meta.get('n_classes', 7)
            # Load label map
            if labels_path.exists():
                with open(labels_path) as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
            else:
                self._label_map = {i: s for i, s in enumerate(SCREEN_TYPES)}
            # Rebuild model architecture and load weights
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)
            model.load_state_dict(torch.load(str(pt_path), map_location='cpu',
                                              weights_only=True))
            model.eval()
            self._session = model  # reuse _session field as model holder
            _slog.info(f'ScreenClassifier: PyTorch model loaded — {len(self._label_map)} classes: {list(self._label_map.values())}')
        except Exception as e:
            _slog.warning(f'ScreenClassifier: model load failed: {e}')
            self._ml_disabled = True

    def _classify_ml(self, img_bgr: np.ndarray) -> tuple[str, float]:
        if self._session is None or self._ml_disabled:
            return '', 0.0
        try:
            import torch
            small  = _resize_224(img_bgr)
            tensor = _to_chw_float(small)  # (1, 3, H, W) numpy
            t      = torch.from_numpy(tensor)
            with torch.no_grad():
                logits = self._session(t)[0]  # (n_classes,)
            probs  = _softmax(logits.numpy())
            idx    = int(np.argmax(probs))
            conf   = float(probs[idx])
            name   = self._label_map.get(idx, SCREEN_TYPES[idx] if idx < len(SCREEN_TYPES) else '')
            # _slog.debug(f'ScreenClassifier: probs={[f"{p:.2f}" for p in probs]} → {name} ({conf:.2f})')
            return name, conf
        except Exception as e:
            _slog.warning(f'ScreenClassifier ML inference error: {e}')
            return '', 0.0


    # ── Stage 2: session k-NN ───────────────────────────────────────────────────

    def _classify_session(self, img_bgr: np.ndarray) -> tuple[str, float]:
        examples = self._session_examples
        if len(examples) < MIN_SESSION_EXAMPLES:
            return '', 0.0
        q_hist = _hist_hsv(img_bgr)
        # Weighted vote: each example contributes its cosine similarity
        scores: dict[str, float] = {}
        for ex in examples:
            sim = max(0.0, _cosine(q_hist, ex['hist']))
            scores[ex['stype']] = scores.get(ex['stype'], 0.0) + sim
        best_stype = max(scores, key=lambda k: scores[k])
        # Normalise by number of examples to get 0-1 confidence estimate
        total = sum(scores.values())
        conf  = scores[best_stype] / total if total > 0 else 0.0
        return best_stype, conf


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
