# warp/recognition/layout_detector.py
# Detects equipment/trait section bounding boxes in STO screenshots.
#
# Strategy:
#   Primary:  EasyOCR — finds section label text ("Fore Weapons", "Deflector" etc.)
#             and derives icon row positions relative to label position.
#   Fallback: Calibrated relative anchor positions (fraction of image size).
#             Defaults derived from screenshot analysis at 1920×1080.
#             Users can create warp/training_data/anchors.json to override.
#
# STO UI layout notes (from screenshot analysis):
#   Status tab has section labels on the LEFT side (x ≈ 36% of image width).
#   Icons are in a horizontal row starting ≈47% x, aligned with label y.
#   Icons are approximately 40×52px at 1080p (box_width=49, box_height=64 in SETS).

from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Anchor table (1920×1080 reference, Status tab) ────────────────────────────
# Each value: (rel_x, rel_y, icon_w_rel, icon_h_rel, max_icons, direction)
#   rel_*    : fraction of image width/height
#   direction: 'h' = horizontal row

SPACE_ANCHORS = {
    'Fore Weapons':          (0.464, 0.051, 0.026, 0.059, 5, 'h'),
    'Aft Weapons':           (0.464, 0.322, 0.026, 0.059, 5, 'h'),
    'Experimental Weapon':   (0.464, 0.244, 0.026, 0.059, 1, 'h'),
    'Devices':               (0.464, 0.384, 0.026, 0.059, 6, 'h'),
    'Hangars':               (0.464, 0.451, 0.026, 0.059, 2, 'h'),
    'Deflector':             (0.464, 0.112, 0.026, 0.059, 1, 'h'),
    'Sec-Def':               (0.464, 0.176, 0.026, 0.059, 1, 'h'),
    'Engines':               (0.464, 0.112, 0.026, 0.059, 1, 'h'),
    'Warp Core':             (0.464, 0.176, 0.026, 0.059, 1, 'h'),
    'Shield':                (0.464, 0.244, 0.026, 0.059, 1, 'h'),
    'Universal Consoles':    (0.562, 0.051, 0.026, 0.059, 3, 'h'),
    'Engineering Consoles':  (0.562, 0.130, 0.026, 0.059, 5, 'h'),
    'Science Consoles':      (0.562, 0.244, 0.026, 0.059, 5, 'h'),
    'Tactical Consoles':     (0.562, 0.358, 0.026, 0.059, 5, 'h'),
    'Personal Space Traits': (0.695, 0.051, 0.026, 0.059, 5, 'h'),
    'Starship Traits':       (0.695, 0.200, 0.026, 0.059, 5, 'h'),
    'Reputation Traits':     (0.695, 0.340, 0.026, 0.059, 5, 'h'),
    'Active Rep Traits':     (0.695, 0.440, 0.026, 0.059, 5, 'h'),
}

GROUND_ANCHORS = {
    'Body Armor':            (0.464, 0.051, 0.026, 0.059, 1, 'h'),
    'EV Suit':               (0.464, 0.112, 0.026, 0.059, 1, 'h'),
    'Personal Shield':       (0.464, 0.176, 0.026, 0.059, 1, 'h'),
    'Weapons':               (0.464, 0.244, 0.026, 0.059, 2, 'h'),
    'Kit':                   (0.464, 0.322, 0.026, 0.059, 1, 'h'),
    'Kit Modules':           (0.464, 0.384, 0.026, 0.059, 6, 'h'),
    'Ground Devices':        (0.464, 0.451, 0.026, 0.059, 3, 'h'),
    'Personal Ground Traits':(0.695, 0.051, 0.026, 0.059, 5, 'h'),
    'Ground Rep Traits':     (0.695, 0.340, 0.026, 0.059, 5, 'h'),
}

ANCHORS = {'SPACE': SPACE_ANCHORS, 'GROUND': GROUND_ANCHORS}

# All slot names for OCR fuzzy matching
ALL_SLOT_NAMES = list(SPACE_ANCHORS.keys()) + list(GROUND_ANCHORS.keys())

# OCR confidence threshold
OCR_CONF_THRESHOLD = 0.45

# Calibration file path (relative to SETS root)
CALIBRATION_FILE = Path('warp') / 'training_data' / 'anchors.json'


class LayoutDetector:
    """
    Detects section bounding boxes for all slots in an STO screenshot.

    Returns dict: { slot_name: [(x, y, w, h), …] }
    """

    def __init__(self):
        self._ocr = None
        self._calibration = self._load_calibration()

    def detect(self, img: np.ndarray, build_type: str) -> dict[str, list]:
        """
        Detect icon positions for the given build type.
        Returns { slot_name: [(x, y, w, h), ...] }
        """
        h, w = img.shape[:2]
        ocr_result = self._detect_via_ocr(img, build_type)
        anchor_result = self._detect_via_anchors(img, build_type)

        # Merge: OCR wins when available, anchors fill the rest
        from warp.warp_importer import SLOT_SPECS
        merged: dict[str, list] = {}
        for slot_name, _ in SLOT_SPECS.get(build_type, []):
            if slot_name in ocr_result:
                merged[slot_name] = ocr_result[slot_name]
            elif slot_name in anchor_result:
                merged[slot_name] = anchor_result[slot_name]

        return merged

    # ── OCR path ────────────────────────────────────────────────────────────

    def _detect_via_ocr(self, img: np.ndarray, build_type: str) -> dict[str, list]:
        try:
            ocr = self._get_ocr()
            results = ocr.readtext(img)
        except Exception as e:
            log.debug(f'LayoutDetector OCR error: {e}')
            return {}

        h, w = img.shape[:2]
        sections: dict[str, list] = {}

        for (bbox_pts, text, conf) in results:
            if conf < OCR_CONF_THRESHOLD:
                continue
            text_clean = text.strip()
            matched = self._fuzzy_match_slot(text_clean)
            if not matched:
                continue

            label_y = int(np.mean([pt[1] for pt in bbox_pts]))
            bboxes  = self._icon_row_from_label(label_y, matched, w, h)
            if bboxes:
                sections[matched] = bboxes

        return sections

    def _fuzzy_match_slot(self, text: str) -> str | None:
        matches = get_close_matches(text, ALL_SLOT_NAMES, n=1, cutoff=0.72)
        return matches[0] if matches else None

    def _icon_row_from_label(
        self, label_y: int, slot_name: str, img_w: int, img_h: int
    ) -> list:
        """Compute icon bboxes given a detected label Y position."""
        icon_h = max(30, int(img_h * 0.055))
        icon_w = max(30, int(img_w * 0.026))
        gap    = max(2, int(icon_w * 0.08))

        # Icons start at ≈47% x in Status tab layout
        start_x = int(img_w * 0.464)
        icon_y  = label_y - icon_h // 2

        anchor = ANCHORS.get('SPACE', {}).get(slot_name) or \
                 ANCHORS.get('GROUND', {}).get(slot_name)
        max_icons = anchor[4] if anchor else 5

        return [(start_x + i * (icon_w + gap), icon_y, icon_w, icon_h)
                for i in range(max_icons)]

    # ── Anchor path ─────────────────────────────────────────────────────────

    def _detect_via_anchors(self, img: np.ndarray, build_type: str) -> dict[str, list]:
        h, w = img.shape[:2]
        table = self._get_anchor_table(build_type)
        result: dict[str, list] = {}

        for slot_name, v in table.items():
            rel_x, rel_y, iw_rel, ih_rel, max_icons, direction = v
            ix = int(img_w * iw_rel) if (img_w := w) else 30
            iy = int(h   * ih_rel)
            sx = int(w   * rel_x)
            sy = int(h   * rel_y)
            gap = max(2, int(ix * 0.08))
            bboxes = []
            for i in range(max_icons):
                if direction == 'h':
                    bboxes.append((sx + i * (ix + gap), sy, ix, iy))
                else:
                    bboxes.append((sx, sy + i * (iy + gap), ix, iy))
            result[slot_name] = bboxes

        return result

    def _get_anchor_table(self, build_type: str) -> dict:
        if self._calibration and build_type in self._calibration:
            return self._calibration[build_type]
        return ANCHORS.get(build_type, {})

    # ── Init / helpers ──────────────────────────────────────────────────────

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr

    def _load_calibration(self) -> dict | None:
        p = Path(__file__).resolve()
        for _ in range(6):
            cfile = p / CALIBRATION_FILE
            if cfile.exists():
                try:
                    with open(cfile) as f:
                        return json.load(f)
                except Exception:
                    pass
            p = p.parent
        return None
