# warp/warp_importer.py
# Main import pipeline — fully integrated with SETS.
#
# Key integration points:
#   - Uses sets_app.cache.images       (already-downloaded item icons)
#   - Uses sets_app.config             (images folder path)
#   - Uses sets_app.cache.equipment    (item name → build_key lookup)
#   - Uses sets_app.cache.traits       (trait name lookup)
#   - Uses sets_app.cache.starship_traits
#   - Uses iofunc.get_image_file_name  (item name → PNG filename)
#
# Pipeline:
#   1. LayoutDetector  — finds section bboxes (OCR labels + anchor fallback)
#   2. IconExtractor   — crops icon regions
#   3. SETSIconMatcher — matches crops against SETS icon image cache (pHash)
#   4. ML fallback     — ONNX EfficientNet from HF Hub (optional)
#   5. TextExtractor   — OCR ship name/tier from top-left block

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np

log = logging.getLogger(__name__)

SCREENSHOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

# Confidence threshold below which ML fallback is tried
TEMPLATE_CONF_THRESHOLD = 0.72

# ── Slot specification for layout detection ────────────────────────────────────
# (warp_slot_name, max_icons)
SPACE_SLOTS = [
    ('Fore Weapons', 5),
    ('Aft Weapons',  5),
    ('Experimental Weapon', 1),
    ('Devices',      6),
    ('Hangars',      2),
    ('Deflector',    1),
    ('Sec-Def',      1),
    ('Engines',      1),
    ('Warp Core',    1),
    ('Shield',       1),
    ('Universal Consoles',   3),
    ('Engineering Consoles', 5),
    ('Science Consoles',     5),
    ('Tactical Consoles',    5),
    ('Personal Space Traits',10),
    ('Starship Traits',       7),
    ('Reputation Traits',     5),
    ('Active Rep Traits',     5),
]

GROUND_SLOTS = [
    ('Body Armor',         1),
    ('EV Suit',            1),
    ('Personal Shield',    1),
    ('Weapons',            2),
    ('Kit',                1),
    ('Kit Modules',        6),
    ('Ground Devices',     3),
    ('Personal Ground Traits',10),
    ('Ground Rep Traits',   5),
]

SLOT_SPECS = {
    'SPACE':         SPACE_SLOTS,
    'GROUND':        GROUND_SLOTS,
    'SPACE_SKILLS':  [],
    'GROUND_SKILLS': [],
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RecognisedItem:
    slot:        str
    slot_index:  int
    name:        str
    confidence:  float
    thumbnail:   Any   = None      # QImage | None
    source_file: str   = ''
    bbox:        tuple = field(default_factory=tuple)


@dataclass
class ImportResult:
    build_type: str
    ship_name:  str = ''
    ship_type:  str = ''
    ship_tier:  str = ''
    items:      list = field(default_factory=list)   # list[RecognisedItem]
    errors:     list = field(default_factory=list)
    warnings:   list = field(default_factory=list)


# ── Main importer ──────────────────────────────────────────────────────────────

class WarpImporter:
    """
    Processes a folder of STO screenshots and returns an ImportResult.
    Reads item icons directly from SETS image cache/disk.
    """

    def __init__(
        self,
        build_type: str = 'SPACE',
        sets_app=None,
        progress_callback: Callable[[int, str], None] | None = None,
    ):
        self._build_type = build_type
        self._sets       = sets_app
        self._progress   = progress_callback or (lambda p, m: None)

        self._layout_detector  = None
        self._icon_matcher     = None
        self._text_extractor   = None

    def process_folder(self, folder: Path) -> ImportResult:
        result = ImportResult(build_type=self._build_type)

        shots = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in SCREENSHOT_EXTENSIONS
        ])
        if not shots:
            result.errors.append(f'No screenshots found in {folder}')
            return result

        self._progress(5, f'Found {len(shots)} screenshot(s). Initialising…')
        self._init_subsystems()
        self._progress(15, 'Recognition ready.')

        all_items: list[RecognisedItem] = []
        ship_found = False

        for idx, path in enumerate(shots):
            pct = 15 + int(70 * idx / len(shots))
            self._progress(pct, f'Analysing: {path.name}')
            try:
                items, meta = self._process_single(path)
                all_items.extend(items)
                if not ship_found and meta.get('ship_name'):
                    result.ship_name = meta['ship_name']
                    result.ship_type = meta['ship_type']
                    result.ship_tier = meta['ship_tier']
                    ship_found = True
            except Exception as exc:
                log.warning(f'WARP: error on {path.name}: {exc}')
                result.warnings.append(f'{path.name}: {exc}')

        self._progress(86, 'Merging results…')
        result.items = self._merge(all_items)

        self._progress(93, 'ML fallback for low-confidence items…')
        result.items = self._ml_fallback(result.items)

        self._progress(100, f'Done — {len(result.items)} item(s) recognised.')
        return result

    # ── Single screenshot ──────────────────────────────────────────────────

    def _process_single(self, path: Path) -> tuple[list[RecognisedItem], dict]:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f'Cannot read image: {path.name}')

        sections = self._layout_detector.detect(img, self._build_type)
        meta     = self._text_extractor.extract_ship_info(img)
        items: list[RecognisedItem] = []

        for slot_name, icon_bboxes in sections.items():
            for i, bbox in enumerate(icon_bboxes):
                x, y, w, h = bbox
                crop = img[y:y+h, x:x+w]
                name, conf = self._icon_matcher.match(crop)
                thumb = self._to_qimage(crop)
                items.append(RecognisedItem(
                    slot=slot_name, slot_index=i,
                    name=name, confidence=conf,
                    thumbnail=thumb, source_file=path.name, bbox=bbox,
                ))
        return items, meta

    # ── Dedup / merge ──────────────────────────────────────────────────────

    def _merge(self, items: list[RecognisedItem]) -> list[RecognisedItem]:
        best: dict[tuple, RecognisedItem] = {}
        for item in items:
            key = (item.slot, item.slot_index)
            if key not in best or item.confidence > best[key].confidence:
                best[key] = item
        return list(best.values())

    # ── ML fallback ────────────────────────────────────────────────────────

    def _ml_fallback(self, items: list[RecognisedItem]) -> list[RecognisedItem]:
        low = [i for i in items if i.confidence < TEMPLATE_CONF_THRESHOLD]
        if not low:
            return items
        try:
            names, confs = self._icon_matcher.classify_ml_batch(
                [i.thumbnail for i in low])
            for item, (n, c) in zip(low, zip(names, confs)):
                if c > item.confidence:
                    item.name       = n
                    item.confidence = c
        except Exception as exc:
            log.debug(f'WARP ML fallback skipped: {exc}')
        return items

    # ── Init ───────────────────────────────────────────────────────────────

    def _init_subsystems(self):
        from warp.recognition.layout_detector import LayoutDetector
        from warp.recognition.icon_matcher    import SETSIconMatcher
        from warp.recognition.text_extractor  import TextExtractor
        self._layout_detector = LayoutDetector()
        self._icon_matcher    = SETSIconMatcher(self._sets)
        self._text_extractor  = TextExtractor()

    # ── QImage helper ──────────────────────────────────────────────────────

    def _to_qimage(self, crop_bgr: np.ndarray) -> Any:
        try:
            import cv2
            from PySide6.QtGui import QImage
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            return QImage(
                rgb.tobytes(), w, h, ch * w,
                QImage.Format.Format_RGB888).copy()  # .copy() detaches from buffer
        except Exception:
            return None
