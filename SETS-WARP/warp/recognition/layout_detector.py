# warp/recognition/layout_detector.py
#
# Detect equipment slot bounding boxes in STO Status-tab screenshots.
#
# Recognition strategy (in priority order):
#
#   1. OCR labels (EasyOCR)
#      The Status tab shows section labels ("Fore Weapons", "Deflector", etc.)
#      on the left side. We find them, fuzzy-match to canonical slot names,
#      then compute icon positions relative to label Y.
#
#   2. Position-based (no labels visible)
#      When labels are absent (cropped screenshot, different UI scale) we use
#      the FIXED VERTICAL ORDER of slots combined with the ship profile to
#      assign icons to slots purely by Y position.
#
#   3. Anchor fallback
#      If both above fail, use pre-calibrated relative anchors (1920×1080 base).
#      User can override via warp/training_data/anchors.json.
#
# Key insight: slot order is ALWAYS fixed (see SPACE_SLOT_ORDER in warp_importer).
# Ship type determines WHICH optional slots are present, not their order.
# Fore/Aft weapons share the same icon appearance — slot position determines which
# side they're on, not the item icon itself.

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

OCR_CONF_THRESHOLD = 0.40
LABEL_FUZZY_CUTOFF = 0.68
CALIBRATION_FILE   = Path('warp') / 'training_data' / 'anchors.json'

# ── Icon geometry (1920×1080 reference) ───────────────────────────────────────
# Status tab: section labels at x≈36% width, icons start at x≈46.4% width
# Icon size: ~40×52px at 1080p → 0.0208 w × 0.0481 h (relative)
# Gap between icons in a row: ~4px → 0.002 w
ICON_W_REL   = 0.0208
ICON_H_REL   = 0.0481
ICON_GAP_REL = 0.0022
ICONS_START_X_REL = 0.464   # left edge of first icon in a row


# ── Anchor table ───────────────────────────────────────────────────────────────
# (rel_x, rel_y, max_icons)  —  1920×1080 reference, Status tab layout
# rel_x: left edge of icon row
# rel_y: top edge of icon row
# Coordinates measured from actual STO screenshots.

SPACE_ANCHORS: dict[str, tuple[float, float, int]] = {
    'Fore Weapons':         (0.464, 0.043, 5),
    'Deflector':            (0.464, 0.108, 1),
    'Sec-Def':              (0.464, 0.163, 1),
    'Engines':              (0.464, 0.163, 1),   # y moves up if no Sec-Def
    'Warp Core':            (0.464, 0.218, 1),
    'Shield':               (0.464, 0.273, 1),
    'Aft Weapons':          (0.464, 0.328, 5),
    'Experimental':         (0.464, 0.383, 1),
    'Devices':              (0.464, 0.438, 6),
    'Universal Consoles':   (0.562, 0.043, 3),
    'Engineering Consoles': (0.562, 0.120, 5),
    'Science Consoles':     (0.562, 0.230, 5),
    'Tactical Consoles':    (0.562, 0.340, 5),
    'Hangars':              (0.562, 0.450, 4),
}

GROUND_ANCHORS: dict[str, tuple[float, float, int]] = {
    'Body Armor':       (0.464, 0.043, 1),
    'EV Suit':          (0.464, 0.100, 1),
    'Personal Shield':  (0.464, 0.157, 1),
    'Weapons':          (0.464, 0.214, 2),
    'Kit':              (0.464, 0.300, 1),
    'Kit Modules':      (0.464, 0.357, 6),
    'Ground Devices':   (0.464, 0.443, 3),
}

ANCHORS = {'SPACE': SPACE_ANCHORS, 'GROUND': GROUND_ANCHORS}


class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.

    detect(img, build_type, ship_profile) → dict[slot_name, list[(x,y,w,h)]]

    ship_profile: dict from warp_importer.ship_type_to_profile()
    """

    def __init__(self):
        self._ocr         = None
        self._calibration = self._load_calibration()

    def detect(
        self,
        img:          np.ndarray,
        build_type:   str,
        ship_profile: dict | None = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

        from warp.warp_importer import SLOT_ORDER, SLOT_LABEL_ALIASES

        h, w = img.shape[:2]
        slot_defs = SLOT_ORDER.get(build_type, [])
        profile   = ship_profile or {}

        # Slots to look for (respects ship profile — skip slots with max=0)
        active_slots = [
            s for s in slot_defs
            if profile.get(s['name'], s['max']) > 0
        ]

        # ── Strategy 1: OCR labels ─────────────────────────────────────────
        ocr_found = self._detect_via_ocr(img, active_slots, build_type, profile)
        if len(ocr_found) >= max(2, len(active_slots) // 3):
            log.debug(f'LayoutDetector: OCR found {len(ocr_found)} slots')
            return self._fill_gaps(ocr_found, active_slots, img, profile)

        # ── Strategy 2: Position-based (fixed vertical order) ─────────────
        pos_found = self._detect_via_position(img, active_slots, profile)
        if pos_found:
            log.debug(f'LayoutDetector: position-based {len(pos_found)} slots')
            return pos_found

        # ── Strategy 3: Anchor fallback ────────────────────────────────────
        log.debug('LayoutDetector: using anchor fallback')
        return self._detect_via_anchors(img, build_type, profile)

    # ── OCR detection ──────────────────────────────────────────────────────────

    def _detect_via_ocr(
        self,
        img:         np.ndarray,
        active_slots: list[dict],
        build_type:  str,
        profile:     dict,
    ) -> dict[str, list]:

        from warp.warp_importer import SLOT_LABEL_ALIASES

        try:
            results = self._get_ocr().readtext(img)
        except Exception as e:
            log.debug(f'OCR error: {e}')
            return {}

        h, w    = img.shape[:2]
        found:  dict[str, list] = {}
        slot_names = [s['name'] for s in active_slots]
        # Also build lookup from alias → canonical
        all_labels = list(SLOT_LABEL_ALIASES.keys()) + slot_names

        for (bbox_pts, text, conf) in results:
            if conf < OCR_CONF_THRESHOLD:
                continue
            text_clean = text.strip()

            canonical = self._match_label(text_clean, SLOT_LABEL_ALIASES, slot_names)
            if not canonical:
                continue
            if canonical not in slot_names:
                continue

            label_y   = int(np.mean([pt[1] for pt in bbox_pts]))
            max_icons = profile.get(canonical,
                         next((s['max'] for s in active_slots
                               if s['name'] == canonical), 1))
            bboxes = self._icon_row(w, h, label_y, max_icons)
            if bboxes:
                found[canonical] = bboxes

        return found

    def _match_label(
        self, text: str,
        aliases:    dict[str, str],
        slot_names: list[str],
    ) -> str | None:
        # Direct alias lookup (case-insensitive)
        for alias, canonical in aliases.items():
            if text.lower() == alias.lower():
                return canonical
        # Fuzzy match against alias keys + canonical names
        candidates = list(aliases.keys()) + slot_names
        matches = get_close_matches(text, candidates, n=1, cutoff=LABEL_FUZZY_CUTOFF)
        if not matches:
            return None
        hit = matches[0]
        return aliases.get(hit, hit)

    def _icon_row(
        self, img_w: int, img_h: int, label_y: int, max_icons: int
    ) -> list[tuple[int, int, int, int]]:
        iw  = max(28, int(img_w * ICON_W_REL))
        ih  = max(28, int(img_h * ICON_H_REL))
        gap = max(2,  int(img_w * ICON_GAP_REL))
        sx  = int(img_w * ICONS_START_X_REL)
        iy  = label_y - ih // 2
        return [(sx + i * (iw + gap), iy, iw, ih) for i in range(max_icons)]

    # ── Position-based detection ───────────────────────────────────────────────
    # When no labels are visible we distribute the known FIXED slot order
    # evenly across the left column's vertical space.

    def _detect_via_position(
        self,
        img:          np.ndarray,
        active_slots: list[dict],
        profile:      dict,
    ) -> dict[str, list]:

        h, w = img.shape[:2]
        iw   = max(28, int(w * ICON_W_REL))
        ih   = max(28, int(h * ICON_H_REL))
        gap  = max(2,  int(w * ICON_GAP_REL))
        sx   = int(w * ICONS_START_X_REL)

        # Estimate vertical band where slots live: ~4% to ~90% of height
        y_start = int(h * 0.04)
        y_end   = int(h * 0.90)
        total_y = y_end - y_start

        if not active_slots:
            return {}

        # Each "row" takes one step_y pixels
        step_y  = total_y // max(1, len(active_slots))
        result: dict[str, list] = {}

        for i, slot_def in enumerate(active_slots):
            name      = slot_def['name']
            max_icons = profile.get(name, slot_def['max'])
            if max_icons == 0:
                continue
            row_y = y_start + i * step_y
            result[name] = [(sx + j * (iw + gap), row_y, iw, ih)
                            for j in range(max_icons)]

        return result

    # ── Anchor fallback ────────────────────────────────────────────────────────

    def _detect_via_anchors(
        self,
        img:        np.ndarray,
        build_type: str,
        profile:    dict,
    ) -> dict[str, list]:

        h, w   = img.shape[:2]
        iw     = max(28, int(w * ICON_W_REL))
        ih     = max(28, int(h * ICON_H_REL))
        gap    = max(2,  int(w * ICON_GAP_REL))

        table  = self._calibration.get(build_type) if self._calibration \
                 else ANCHORS.get(build_type, {})

        result: dict[str, list] = {}
        for slot_name, (rx, ry, anchor_max) in table.items():
            max_icons = profile.get(slot_name, anchor_max)
            if max_icons == 0:
                continue
            sx = int(w * rx)
            sy = int(h * ry)
            result[slot_name] = [
                (sx + i * (iw + gap), sy, iw, ih)
                for i in range(max_icons)
            ]

        return result

    # ── Gap filler ─────────────────────────────────────────────────────────────
    # OCR found some slots but not all — fill missing ones by interpolating
    # Y positions from the known fixed order.

    def _fill_gaps(
        self,
        found:       dict[str, list],
        active_slots: list[dict],
        img:         np.ndarray,
        profile:     dict,
    ) -> dict[str, list]:

        h, w = img.shape[:2]
        iw   = max(28, int(w * ICON_W_REL))
        ih   = max(28, int(h * ICON_H_REL))
        gap  = max(2,  int(w * ICON_GAP_REL))
        sx   = int(w * ICONS_START_X_REL)

        result = dict(found)

        # Build sorted list of (slot_order_index, Y) from OCR hits
        order_map = {s['name']: i for i, s in enumerate(active_slots)}
        anchored: list[tuple[int, int]] = []   # (order_idx, y)
        for name, bboxes in found.items():
            if bboxes and name in order_map:
                anchored.append((order_map[name], bboxes[0][1]))
        anchored.sort()

        if len(anchored) < 2:
            return result   # not enough anchors to interpolate

        # Linear interpolation between nearest known Y values
        for slot_def in active_slots:
            name = slot_def['name']
            if name in result:
                continue
            idx       = order_map[name]
            max_icons = profile.get(name, slot_def['max'])
            if max_icons == 0:
                continue

            # Find surrounding anchors
            before = [(i, y) for (i, y) in anchored if i < idx]
            after  = [(i, y) for (i, y) in anchored if i > idx]

            if before and after:
                i0, y0 = before[-1]
                i1, y1 = after[0]
                # Interpolate
                t   = (idx - i0) / max(1, i1 - i0)
                row_y = int(y0 + t * (y1 - y0))
            elif before:
                # Extrapolate below last anchor
                if len(before) >= 2:
                    step = (before[-1][1] - before[-2][1]) / max(1, before[-1][0] - before[-2][0])
                else:
                    step = ih + gap
                row_y = int(before[-1][1] + (idx - before[-1][0]) * step)
            elif after:
                if len(after) >= 2:
                    step = (after[1][1] - after[0][1]) / max(1, after[1][0] - after[0][0])
                else:
                    step = ih + gap
                row_y = int(after[0][1] - (after[0][0] - idx) * step)
            else:
                continue

            result[name] = [(sx + j * (iw + gap), row_y, iw, ih)
                            for j in range(max_icons)]

        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr

    def _load_calibration(self) -> dict | None:
        p = Path(__file__).resolve().parent
        for _ in range(6):
            cfile = p / CALIBRATION_FILE
            if cfile.exists():
                try:
                    return json.loads(cfile.read_text())
                except Exception:
                    pass
            p = p.parent
        return None
