# warp/recognition/layout_detector.py
#
# Detect equipment slot bounding boxes in STO Status-tab screenshots.
#
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

from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

OCR_CONF_THRESHOLD = 0.40
LABEL_FUZZY_CUTOFF = 0.68
CALIBRATION_FILE   = Path('warp') / 'training_data' / 'anchors.json'

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


class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.

    detect(img, build_type, ship_profile) -> dict[slot_name, list[(x,y,w,h)]]

    ship_profile: dict  slot_name -> max_icons  (from warp_importer)
    """

    def __init__(self):
        self._ocr         = None
        self._calibration = self._load_calibration()

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        img:          np.ndarray,
        build_type:   str,
        ship_profile: dict | None = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:

        # Dispatch non-equipment build types to specialized detectors
        if build_type in ('SPACE_TRAITS', 'GROUND_TRAITS'):
            return self._detect_traits(img, build_type)
        if build_type == 'BOFFS':
            return self._detect_boffs(img)
        if build_type == 'SPEC':
            return self._detect_spec(img)

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

    # ── Traits detection ───────────────────────────────────────────────────────
    # Traits are displayed as horizontal grids of icons under a text header.
    # We OCR headers then crop icon rows beneath each.

    def _detect_traits(
        self,
        img:        np.ndarray,
        build_type: str,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Detect trait icon grids by finding section headers via OCR,
        then extracting icon bboxes from the row below each header.

        Works for both the standalone Traits tab and the STOCD overlay
        (where traits occupy the middle column).
        """
        h, w = img.shape[:2]

        # Section headers we look for (lowercase)
        if 'GROUND' in build_type:
            section_map = {
                'personal ground traits': 'Personal Ground Traits',
                'ground reputation':       'Ground Reputation',
                'active ground rep':       'Active Ground Rep',
            }
        else:
            section_map = {
                'personal space traits':  'Personal Space Traits',
                'starship traits':        'Starship Traits',
                'space reputation':       'Space Reputation',
                'active space rep':       'Active Space Rep',
                # custom ship traits shown as "[Name] Traits"
            }

        try:
            ocr_out = self._get_ocr().readtext(img)
        except Exception as e:
            log.debug(f'Traits OCR failed: {e}')
            return {}

        # Find header positions
        headers: list[tuple[str, int, int]] = []  # (canonical, cy, x_right)
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3:
                continue
            tl, tr = bbox[0], bbox[1]
            y_center = int((tl[1] + bbox[2][1]) / 2)
            text_low = text.lower().strip()

            # Exact or fuzzy match against known headers
            matched = None
            for kw, canonical in section_map.items():
                if kw in text_low or text_low in kw:
                    matched = canonical
                    break
            # Custom "[Ship] Traits" header
            if matched is None and text_low.endswith('traits') and len(text_low) > 6:
                matched = text.strip()   # keep as-is

            if matched:
                x_right = int(max(p[0] for p in bbox))
                headers.append((matched, y_center, x_right))

        if not headers:
            log.debug('Traits: no section headers found via OCR')
            return {}

        headers.sort(key=lambda x: x[1])   # sort by y position

        # Estimate icon size from image dimensions
        # Typical trait icon: ~40–55px on a 1366×768 screenshot
        icon_est = max(32, int(h * 0.055))

        result: dict[str, list[tuple[int, int, int, int]]] = {}

        for i, (section, hy, x_right) in enumerate(headers):
            # Icon row starts ~20px below the header text
            row_y = hy + int(icon_est * 0.5)
            row_y_end = (headers[i + 1][1] - 10) if i + 1 < len(headers) else (row_y + icon_est + 20)

            # Scan that horizontal strip for icon clusters
            strip = img[max(0, row_y): min(h, row_y_end), :]
            if strip.size == 0:
                continue

            bboxes = self._find_icon_bboxes_in_strip(
                strip,
                y_offset=max(0, row_y),
                icon_size=icon_est,
            )
            if bboxes:
                result[section] = bboxes

        return result

    def _find_icon_bboxes_in_strip(
        self,
        strip:     np.ndarray,
        y_offset:  int,
        icon_size: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        Find individual icon bboxes in a horizontal strip.
        Uses brightness variance to locate icon cells.
        """
        import cv2

        sh, sw = strip.shape[:2]
        if sh == 0 or sw == 0:
            return []

        # Threshold: bright pixels (icons have colored art vs dark background)
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Column brightness profile — find icon-dense columns
        col_bright = np.sum(mask, axis=0).astype(float) / 255

        # Find runs of bright columns (icons)
        in_icon = False
        start = 0
        bboxes = []
        min_w = max(20, icon_size // 2)

        for x in range(sw):
            bright = col_bright[x] > sh * 0.2
            if bright and not in_icon:
                in_icon = True
                start = x
            elif not bright and in_icon:
                in_icon = False
                run_w = x - start
                if run_w >= min_w:
                    # Center vertically in strip
                    y = max(0, (sh - icon_size) // 2)
                    bboxes.append((start, y_offset + y, run_w, min(icon_size, sh)))
                start = x
        # Close last run
        if in_icon:
            run_w = sw - start
            if run_w >= min_w:
                y = max(0, (sh - icon_size) // 2)
                bboxes.append((start, y_offset + y, run_w, min(icon_size, sh)))

        return bboxes

    # ── BOffs detection ────────────────────────────────────────────────────────
    # BOffs panel (right side of Status or dedicated Boff tab):
    # profession headers (Tactical / Engineering / Science / ...) with
    # ability icons listed under each seat.

    def _detect_boffs(
        self,
        img: np.ndarray,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Detect Boff ability icons grouped by profession.
        Uses OCR to find profession headers, then extracts icon rows.
        """
        h, w = img.shape[:2]

        PROFESSION_MAP = {
            'tactical':      'Boff Tactical',
            'engineering':   'Boff Engineering',
            'science':       'Boff Science',
            'operations':    'Boff Operations',
            'intelligence':  'Boff Intelligence',
            'command':       'Boff Command',
            'pilot':         'Boff Pilot',
            'miracle worker':'Boff Miracle Worker',
            'temporal':      'Boff Temporal',
            'medical':       'Boff Science',   # Medical maps to Science abilities
        }

        try:
            # Right 45% of image where boff panel lives
            x_start = int(w * 0.55)
            roi = img[:, x_start:]
            ocr_out = self._get_ocr().readtext(roi)
        except Exception as e:
            log.debug(f'BOffs OCR failed: {e}')
            return {}

        # Find profession headers
        headers: list[tuple[str, int]] = []  # (canonical_name, y_center)
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3:
                continue
            text_low = text.lower().strip()
            for kw, canonical in PROFESSION_MAP.items():
                if kw in text_low:
                    y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                    headers.append((canonical, y_center))
                    break

        if not headers:
            log.debug('BOffs: no profession headers found')
            return {}

        headers.sort(key=lambda x: x[1])

        # Merge duplicate professions (keep first occurrence per canonical)
        seen: set[str] = set()
        merged: list[tuple[str, int]] = []
        for name, y in headers:
            if name not in seen:
                seen.add(name)
                merged.append((name, y))
        headers = merged

        x_start = int(w * 0.55)
        icon_est = max(32, int(h * 0.055))
        result: dict[str, list[tuple[int, int, int, int]]] = {}

        for i, (section, hy) in enumerate(headers):
            # Boff abilities start right below the profession header
            row_y     = hy + int(icon_est * 0.3)
            row_y_end = (headers[i + 1][1] - 5) if i + 1 < len(headers) else (h)
            row_y_end = min(h, row_y_end)

            # Scan the entire right panel for this section
            strip = img[max(0, row_y): row_y_end, x_start:]
            if strip.size == 0:
                continue

            bboxes = self._find_icon_bboxes_in_strip(strip, row_y, icon_est)
            # Offset x to account for roi crop
            bboxes = [(x + x_start, y, ww, hh) for (x, y, ww, hh) in bboxes]
            if bboxes:
                result.setdefault(section, []).extend(bboxes)

        return result

    # ── Spec detection ─────────────────────────────────────────────────────────
    # Primary/Secondary spec shown as large icon + text label at bottom.

    def _detect_spec(
        self,
        img: np.ndarray,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Detect Primary and Secondary specialization icons.
        They appear as large icons (~60px) with text labels at bottom of screen.
        """
        h, w = img.shape[:2]

        # Specs are always in bottom 25% of screen
        bottom = img[int(h * 0.75):, :]
        y_off  = int(h * 0.75)

        try:
            ocr_out = self._get_ocr().readtext(bottom)
        except Exception as e:
            log.debug(f'Spec OCR failed: {e}')
            return {}

        KNOWN_SPECS = {
            'temporal operative', 'strategist', 'intelligence', 'commando',
            'miracle worker', 'pilot', 'command', 'constable', 'emergence',
        }

        found: list[tuple[int, int, int, int, str]] = []  # (x,y,w,h, label)
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3:
                continue
            text_low = text.lower().strip()
            is_spec = any(s in text_low for s in KNOWN_SPECS)
            if is_spec:
                # Look for an icon to the left of the text
                tx = int(bbox[0][0])
                ty = int(bbox[0][1]) + y_off
                # Estimate icon to the left: ~60px wide
                icon_size = max(40, int(h * 0.07))
                ix = max(0, tx - icon_size - 5)
                iy = max(0, ty - icon_size // 4)
                found.append((ix, iy, icon_size, icon_size, text.strip()))

        result: dict[str, list[tuple[int, int, int, int]]] = {}
        if len(found) >= 1:
            result['Primary Specialization'] = [(found[0][0], found[0][1], found[0][2], found[0][3])]
        if len(found) >= 2:
            result['Secondary Specialization'] = [(found[1][0], found[1][1], found[1][2], found[1][3])]

        return result



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

        return result

    # ── Gap filler ─────────────────────────────────────────────────────────────

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
