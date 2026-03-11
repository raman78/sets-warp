# warp/recognition/layout_detector.py
#
# Detect equipment slot bounding boxes in STO Status-tab screenshots.
#
<<<<<<< HEAD
# CALIBRATED from actual screenshots (March 2026):
#   - Resolute 1507×906, Genius 1247×964, Simonz 862×1150, Union 1476×900
#
# KEY FINDINGS:
#   - Icons are RIGHT-ALIGNED in the slot panel (not left-aligned!)
#   - Slot row height: ~58–66 px depending on window scale
#   - Icon cell width: ~50 px (≈ 0.80 × row_height)
#   - Icon face: ~47×47 px
#   - Rows separated by dark bands (~6–8 px each)
#   - Panel right edge detectable by rightmost dense column
#
# Detection strategy:
#   1. Image-analysis (primary): detect dark separators + right edge automatically
#   2. OCR labels (secondary): if analysis fails, fall back to label positions
#   3. Anchor fallback (last resort): use calibrated relative positions
=======
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
>>>>>>> origin/main

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

<<<<<<< HEAD
# Slot order for space builds
SPACE_SLOT_ORDER_STANDARD = [
    'Fore Weapons',
    'Deflector',
    'Impulse',
    'Warp Core',
    'Shields',
    'Aft Weapons',
    'Devices',
    'Universal Consoles',
    'Engineering Consoles',
    'Science Consoles',
    'Tactical Consoles',
]

SPACE_SLOT_ORDER_CARRIER = SPACE_SLOT_ORDER_STANDARD + ['Hangar']

# Default icon counts per slot (overridden by ship profile)
SLOT_DEFAULT_COUNTS = {
    'Fore Weapons':         5,
    'Deflector':            1,
    'Impulse':              1,
    'Warp Core':            1,
    'Shields':              1,
    'Aft Weapons':          4,
    'Devices':              4,
    'Universal Consoles':   2,
    'Engineering Consoles': 4,
    'Science Consoles':     2,
    'Tactical Consoles':    4,
    'Hangar':               1,
}

# OCR label aliases
SLOT_LABEL_ALIASES = {
    'fore weapons': 'Fore Weapons',
    'fore':         'Fore Weapons',
    'deflector':    'Deflector',
    'impulse':      'Impulse',
    'engines':      'Impulse',
    'warp core':    'Warp Core',
    'warp':         'Warp Core',
    'shields':      'Shields',
    'shield':       'Shields',
    'aft weapons':  'Aft Weapons',
    'aft':          'Aft Weapons',
    'devices':      'Devices',
    'universal consoles': 'Universal Consoles',
    'universal':    'Universal Consoles',
    'engineering consoles': 'Engineering Consoles',
    'engineering':  'Engineering Consoles',
    'science consoles': 'Science Consoles',
    'science':      'Science Consoles',
    'tactical consoles': 'Tactical Consoles',
    'tactical':     'Tactical Consoles',
    'hangar':       'Hangar',
    'hangars':      'Hangar',
}

=======
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

>>>>>>> origin/main

class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.

<<<<<<< HEAD
    detect(img, build_type, ship_profile) -> dict[slot_name, list[(x,y,w,h)]]

    ship_profile: dict  slot_name -> max_icons  (from warp_importer)
=======
    detect(img, build_type, ship_profile) → dict[slot_name, list[(x,y,w,h)]]

    ship_profile: dict from warp_importer.ship_type_to_profile()
>>>>>>> origin/main
    """

    def __init__(self):
        self._ocr         = None
        self._calibration = self._load_calibration()

<<<<<<< HEAD
    # ── Public API ─────────────────────────────────────────────────────────────

=======
>>>>>>> origin/main
    def detect(
        self,
        img:          np.ndarray,
        build_type:   str,
        ship_profile: dict | None = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

<<<<<<< HEAD
        profile = ship_profile or {}
        slot_order = (SPACE_SLOT_ORDER_CARRIER
                      if profile.get('Hangar', 0) > 0
                      else SPACE_SLOT_ORDER_STANDARD)

        # Strategy 1: pixel analysis
        result = self._detect_via_pixel_analysis(img, slot_order, profile)
        if result and len(result) >= len(slot_order) * 0.6:
            log.debug(f'LayoutDetector: pixel analysis found {len(result)} slots')
            return result

        # Strategy 2: OCR labels
        ocr_result = self._detect_via_ocr(img, slot_order, profile)
        if ocr_result and len(ocr_result) >= 2:
            log.debug(f'LayoutDetector: OCR found {len(ocr_result)} slots')
            return self._fill_gaps(ocr_result, slot_order, img, profile)

        # Strategy 3: anchor fallback
        log.debug('LayoutDetector: using anchor fallback')
        return self._detect_via_anchors(img, slot_order, profile)

    # ── Strategy 1: Pixel analysis ─────────────────────────────────────────────

    def _detect_via_pixel_analysis(
        self,
        img:        np.ndarray,
        slot_order: list[str],
        profile:    dict,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Detect slot rows by finding dark horizontal separator bands
        and icon columns by locating the rightmost dense content area.
        """
        h, w = img.shape[:2]

        # Step 1: find panel right edge
        panel_right = self._find_panel_right_edge(img)
        if panel_right < w * 0.3:
            log.debug(f'panel_right={panel_right} too small, aborting pixel analysis')
            return {}

        # Step 2: find dark row separators
        x_start = max(0, panel_right - int(w * 0.25))
        row_seps = self._find_row_separators(img, x_start, panel_right)
        if len(row_seps) < 3:
            log.debug(f'Only {len(row_seps)} row separators found')
            return {}

        # Step 3: derive row bounds
        row_bounds = []
        for i in range(len(row_seps) - 1):
            y_top = row_seps[i]
            y_bot = row_seps[i + 1]
            if y_bot - y_top >= 30:
                row_bounds.append((y_top, y_bot))

        if not row_bounds:
            return {}

        # Step 4: compute icon geometry from measured row height
        row_h_avg = sum(b - a for a, b in row_bounds) / len(row_bounds)
        cell_w    = max(30, int(row_h_avg * 0.80))
        icon_w    = max(26, cell_w - 4)
        icon_h    = max(26, int(row_h_avg * 0.78))

        log.debug(f'row_h_avg={row_h_avg:.1f} cell_w={cell_w} '
                  f'panel_right={panel_right} n_rows={len(row_bounds)}')

        # Step 5: assign slot names to rows
        n_expected = len(slot_order)
        if len(row_bounds) > n_expected:
            row_bounds = row_bounds[:n_expected]

        result: dict[str, list[tuple[int, int, int, int]]] = {}
        for i, (y_top, y_bot) in enumerate(row_bounds):
            if i >= len(slot_order):
                break
            slot_name = slot_order[i]
            n_icons   = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n_icons == 0:
                continue

            row_cy = (y_top + y_bot) // 2
            iy     = row_cy - icon_h // 2

            # Icons are RIGHT-ALIGNED: build right-to-left, then reverse
            bboxes = []
            for j in range(n_icons):
                ix = panel_right - (j + 1) * cell_w + 2
                bboxes.append((max(0, ix), iy, icon_w, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes

        return result

    def _find_panel_right_edge(self, img: np.ndarray) -> int:
        """
        Find the right edge of the slot icon column.

        Uses "slot-row consistency": scans right-to-left and looks for the
        first column that has bright icon content in MOST of the expected
        slot row bands. This avoids being fooled by:
          - Item tooltip overlays (bright but concentrated in one area)
          - UI chrome at image edges
          - Left-side info panels (ship picture, mastery)

        Falls back to simple rightmost-content scan if consistency fails.
        """
        h, w = img.shape[:2]

        # Define ~10 equidistant Y bands covering the slot area
        # (slots typically span y ≈ 3%–90% of image height)
        y_top_pct, y_bot_pct = 0.03, 0.90
        n_bands = 10
        band_h  = int(h * (y_bot_pct - y_top_pct) / n_bands)
        y_bands = [
            (int(h * y_top_pct + i * band_h),
             int(h * y_top_pct + (i + 1) * band_h))
            for i in range(n_bands)
        ]

        min_hits = max(5, n_bands * 7 // 10)   # at least 70% of bands

        # Scan right-to-left; stop at 20% of image width from left
        for x in range(w - 2, max(w // 5, 50), -1):
            hit_count = sum(
                1 for (y1, y2) in y_bands
                if any(
                    sum(int(c) for c in img[y, x]) / 3 > 50
                    for y in range(y1, y2, 4)
                )
            )
            if hit_count >= min_hits:
                return x

        # Fallback: simple rightmost bright column
        for x in range(w - 2, max(w // 5, 50), -1):
            for y in range(int(h * 0.03), int(h * 0.90), 5):
                if sum(int(c) for c in img[y, x]) / 3 > 50:
                    return x

        return int(w * 0.90)

    def _find_row_separators(
        self, img: np.ndarray, x_start: int, x_end: int
    ) -> list[int]:
        """
        Find Y positions of dark horizontal separator bands between slot rows.
        Returns sorted list of separator midpoints (including y=0 and y=h).
        """
        h = img.shape[0]
        x_step = max(1, (x_end - x_start) // 25)

        # Compute average brightness per row
        row_avgs: list[float] = []
        for y in range(h):
            vals = [
                sum(int(c) for c in img[y, x]) / 3
                for x in range(x_start, x_end, x_step)
                if x < img.shape[1]
            ]
            row_avgs.append(sum(vals) / max(1, len(vals)))

        # 5-point moving average
        smoothed = list(row_avgs)
        for y in range(2, h - 2):
            smoothed[y] = sum(row_avgs[y-2:y+3]) / 5

        # Dark threshold: 25% of max brightness, never more than 30
        max_bright = max(smoothed[10:h-10]) if h > 20 else 100.0
        dark_thr   = min(30.0, max_bright * 0.25)

        # Collect dark runs
        dark_runs: list[tuple[int, int]] = []
        in_dark, ds = False, 0
        for y, avg in enumerate(smoothed):
            if avg < dark_thr and not in_dark:
                in_dark, ds = True, y
            elif avg >= dark_thr and in_dark:
                in_dark = False
                if y - ds >= 2:
                    dark_runs.append((ds, y))
        if in_dark:
            dark_runs.append((ds, h - 1))

        # Merge runs closer than 4 px
        merged: list[tuple[int, int]] = []
        for s, e in dark_runs:
            if merged and s - merged[-1][1] < 4:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append([s, e])

        seps = [int((s + e) / 2) for s, e in merged]

        # Ensure we have boundaries at 0 and h
        if not seps or seps[0] > 15:
            seps = [0] + seps
        if not seps or seps[-1] < h - 40:
            seps = seps + [h]

        return sorted(seps)

    # ── Strategy 2: OCR labels ─────────────────────────────────────────────────

    def _detect_via_ocr(
        self,
        img:        np.ndarray,
        slot_order: list[str],
        profile:    dict,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

        try:
            results = self._get_ocr().readtext(img)
        except Exception as e:
            log.debug(f'OCR error: {e}')
            return {}

        h, w        = img.shape[:2]
        panel_right = self._find_panel_right_edge(img)
        row_h_est   = int(h * 0.068)
        cell_w      = max(30, int(row_h_est * 0.80))
        icon_w      = cell_w - 4
        icon_h      = max(26, int(row_h_est * 0.78))

        found: dict[str, list[tuple[int, int, int, int]]] = {}

        for (bbox_pts, text, conf) in results:
            if conf < OCR_CONF_THRESHOLD:
                continue
            canonical = self._match_label(text.strip().lower())
            if not canonical or canonical not in slot_order:
                continue

            label_y = int(np.mean([pt[1] for pt in bbox_pts]))
            n_icons = profile.get(canonical, SLOT_DEFAULT_COUNTS.get(canonical, 1))
            if n_icons == 0:
                continue

            iy = label_y - icon_h // 2
            bboxes = []
            for j in range(n_icons):
                ix = panel_right - (j + 1) * cell_w + 2
                bboxes.append((max(0, ix), iy, icon_w, icon_h))
            bboxes.reverse()
            found[canonical] = bboxes

        return found

    def _match_label(self, text_lower: str) -> str | None:
        if text_lower in SLOT_LABEL_ALIASES:
            return SLOT_LABEL_ALIASES[text_lower]
        matches = get_close_matches(
            text_lower, list(SLOT_LABEL_ALIASES.keys()),
            n=1, cutoff=LABEL_FUZZY_CUTOFF
        )
        if matches:
            return SLOT_LABEL_ALIASES.get(matches[0])
        return None

    # ── Strategy 3: Anchor fallback ────────────────────────────────────────────
    # Calibrated from Resolute 1507×906 pixel measurements.
    # y_rel = separator_midpoint_y / image_height
    # All icons are right-aligned to panel_right.

    SPACE_ANCHORS_REL: dict[str, tuple[float, int]] = {
        'Fore Weapons':         (0.036, 5),
        'Deflector':            (0.107, 1),
        'Impulse':              (0.178, 1),
        'Warp Core':            (0.249, 1),
        'Shields':              (0.325, 1),
        'Aft Weapons':          (0.401, 4),
        'Devices':              (0.475, 4),
        'Universal Consoles':   (0.547, 2),
        'Engineering Consoles': (0.620, 4),
        'Science Consoles':     (0.695, 2),
        'Tactical Consoles':    (0.768, 4),
        'Hangar':               (0.840, 1),
    }

    def _detect_via_anchors(
        self,
        img:        np.ndarray,
        slot_order: list[str],
        profile:    dict,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

        h, w        = img.shape[:2]
        panel_right = self._find_panel_right_edge(img)
        row_h_est   = int(h * 0.072)
        cell_w      = max(30, int(row_h_est * 0.80))
        icon_w      = cell_w - 4
        icon_h      = max(26, int(row_h_est * 0.78))

        cal = (self._calibration or {}).get('SPACE', {})

        result: dict[str, list[tuple[int, int, int, int]]] = {}
        for slot_name in slot_order:
            if slot_name in cal:
                y_rel, n_default = cal[slot_name]
            elif slot_name in self.SPACE_ANCHORS_REL:
                y_rel, n_default = self.SPACE_ANCHORS_REL[slot_name]
            else:
                continue

            n_icons = profile.get(slot_name, n_default)
            if n_icons == 0:
                continue

            iy = int(h * y_rel) - icon_h // 2
            bboxes = []
            for j in range(n_icons):
                ix = panel_right - (j + 1) * cell_w + 2
                bboxes.append((max(0, ix), iy, icon_w, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
=======
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
>>>>>>> origin/main

        return result

    # ── Gap filler ─────────────────────────────────────────────────────────────
<<<<<<< HEAD

    def _fill_gaps(
        self,
        found:      dict[str, list],
        slot_order: list[str],
        img:        np.ndarray,
        profile:    dict,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

        h, w        = img.shape[:2]
        panel_right = self._find_panel_right_edge(img)
        row_h_est   = int(h * 0.068)
        cell_w      = max(30, int(row_h_est * 0.80))
        icon_w      = cell_w - 4
        icon_h      = max(26, int(row_h_est * 0.78))

        result    = dict(found)
        order_map = {name: i for i, name in enumerate(slot_order)}

        anchored: list[tuple[int, int]] = []
        for name, bboxes in found.items():
            if bboxes and name in order_map:
                cy = bboxes[0][1] + bboxes[0][3] // 2
                anchored.append((order_map[name], cy))
        anchored.sort()

        if len(anchored) < 2:
            return result

        for slot_name in slot_order:
            if slot_name in result:
                continue
            idx     = order_map.get(slot_name)
            if idx is None:
                continue
            n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n_icons == 0:
                continue

            before = [(i, y) for i, y in anchored if i < idx]
            after  = [(i, y) for i, y in anchored if i > idx]

            if before and after:
                i0, y0 = before[-1]; i1, y1 = after[0]
                t  = (idx - i0) / max(1, i1 - i0)
                cy = int(y0 + t * (y1 - y0))
            elif before:
                step = ((before[-1][1] - before[-2][1])
                        / max(1, before[-1][0] - before[-2][0])
                        if len(before) >= 2 else row_h_est)
                cy = int(before[-1][1] + (idx - before[-1][0]) * step)
            elif after:
                step = ((after[1][1] - after[0][1])
                        / max(1, after[1][0] - after[0][0])
                        if len(after) >= 2 else row_h_est)
                cy = int(after[0][1] - (after[0][0] - idx) * step)
            else:
                continue

            iy = cy - icon_h // 2
            bboxes = []
            for j in range(n_icons):
                ix = panel_right - (j + 1) * cell_w + 2
                bboxes.append((max(0, ix), iy, icon_w, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
=======
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
>>>>>>> origin/main

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
