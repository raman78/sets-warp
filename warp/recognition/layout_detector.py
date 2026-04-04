# warp/recognition/layout_detector.py
#
# Detect equipment slot bounding boxes in STO Status-tab screenshots.
# Now with Dynamic Layout Learning — learns from user-confirmed data.
#
# Detection strategy:
#   1. Learned Layouts: Match current screen against known confirmed patterns (anchors.json)
#   2. Pixel analysis: detect dark separators + right edge automatically
#   3. OCR labels (fallback): if analysis fails, fall back to label positions
#   4. Default Anchors (last resort): use calibrated relative positions

from __future__ import annotations

import json
import logging
import os
from difflib import get_close_matches
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

OCR_CONF_THRESHOLD = 0.40
LABEL_FUZZY_CUTOFF = 0.68
# Calibration file is stored in training_data
CALIBRATION_FILE   = Path('warp') / 'training_data' / 'anchors.json'

# Slot order for space builds
# Slot names must match warp_importer.py SPACE_SLOT_ORDER exactly
SPACE_SLOT_ORDER_STANDARD = [
    'Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield',
    'Aft Weapons', 'Devices', 'Universal Consoles', 'Engineering Consoles',
    'Science Consoles', 'Tactical Consoles',
]
SPACE_SLOT_ORDER_CARRIER = SPACE_SLOT_ORDER_STANDARD + ['Hangars']

GROUND_SLOT_ORDER = [
    'Kit Modules', 'Kit', 'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons',
    'Ground Devices',
]

SLOT_DEFAULT_COUNTS = {
    'Fore Weapons': 5, 'Deflector': 1, 'Engines': 1, 'Warp Core': 1, 'Shield': 1,
    'Aft Weapons': 4, 'Devices': 4, 'Universal Consoles': 2, 'Engineering Consoles': 4,
    'Science Consoles': 2, 'Tactical Consoles': 4, 'Hangar': 1,
    # Ground slots
    'Body Armor': 1, 'EV Suit': 1, 'Personal Shield': 1, 'Weapons': 2,
    'Kit': 1, 'Kit Modules': 6, 'Ground Devices': 3,
}

SLOT_LABEL_ALIASES = {
    'fore weapons': 'Fore Weapons', 'fore': 'Fore Weapons', 'deflector': 'Deflector',
    'impulse': 'Engines', 'engines': 'Engines', 'warp core': 'Warp Core',
    'warp': 'Warp Core', 'shields': 'Shield', 'shield': 'Shield',
    'aft weapons': 'Aft Weapons', 'aft': 'Aft Weapons', 'devices': 'Devices',
    'universal consoles': 'Universal Consoles', 'universal': 'Universal Consoles',
    'engineering consoles': 'Engineering Consoles', 'engineering': 'Engineering Consoles',
    'science consoles': 'Science Consoles', 'science': 'Science Consoles',
    'tactical consoles': 'Tactical Consoles', 'tactical': 'Tactical Consoles',
    'hangar': 'Hangar', 'hangars': 'Hangar',
}

class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.
    Learns new layouts automatically from confirmed annotations.
    """

    def __init__(self):
        self._ocr = None
        self._calibration = self._load_calibration()
        self._community_anchors: list | None = None  # instance cache for community_anchors.json (P11)

    def detect(self, img: np.ndarray, build_type: str, ship_profile: dict | None = None) -> dict[str, list[tuple[int, int, int, int]]]:
        if build_type in ('SPACE_TRAITS', 'GROUND_TRAITS'):
            return self._detect_traits(img, build_type)
        if build_type == 'BOFFS':
            return self._detect_boffs(img)
        if build_type == 'SPEC':
            return self._detect_spec(img)

        profile = ship_profile or {}
        if build_type == 'GROUND':
            slot_order = GROUND_SLOT_ORDER
        else:
            slot_order = (SPACE_SLOT_ORDER_CARRIER if profile.get('Hangar', 0) > 0 else SPACE_SLOT_ORDER_STANDARD)

        # Strategy 1: Learned Layouts — tried FIRST because they contain
        # user-confirmed slot counts, more reliable than ShipDB generic fallback
        learned = self._detect_via_learned_layouts(img, build_type, slot_order, profile)
        if learned:
            _slog.info(f'LayoutDetector: Strategy 1 (learned) → {len(learned)} slot groups, {sum(len(v) for v in learned.values())} bboxes')
            for slot, boxes in learned.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return learned

        # Strategy 2: Pixel analysis (fallback — uses ShipDB profile counts)
        result = self._detect_via_pixel_analysis(img, slot_order, profile)
        if result and len(result) >= len(slot_order) * 0.7:
            _slog.info(f'LayoutDetector: Strategy 2 (pixel) → {len(result)} slot groups, {sum(len(v) for v in result.values())} bboxes')
            for slot, boxes in result.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return result

        # Strategy 3: OCR labels
        ocr_result = self._detect_via_ocr(img, slot_order, profile)
        if ocr_result and len(ocr_result) >= 2:
            filled = self._fill_gaps(ocr_result, slot_order, img, profile)
            _slog.info(f'LayoutDetector: Strategy 3 (OCR) → {len(filled)} slot groups, {sum(len(v) for v in filled.values())} bboxes')
            for slot, boxes in filled.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return filled

        # Strategy 4: Hardcoded anchor fallback
        fallback = self._detect_via_anchors(img, slot_order, profile)
        _slog.info(f'LayoutDetector: Strategy 4 (anchors) → {len(fallback)} slot groups, {sum(len(v) for v in fallback.values())} bboxes')
        for slot, boxes in fallback.items():
            for b in boxes:
                _slog.info(f'  [{slot}] bbox={b}')
        return fallback
    # ── Learning Logic ────────────────────────────────────────────────────────

    def remove_layout(self, source_file: str) -> bool:
        """Remove all learned layout entries for source_file from anchors.json."""
        if not source_file or not self._calibration or 'learned' not in self._calibration:
            return False
        before = len(self._calibration['learned'])
        self._calibration['learned'] = [
            e for e in self._calibration['learned']
            if e.get('source_file') != source_file
        ]
        removed = before - len(self._calibration['learned'])
        if removed:
            self._save_calibration()
            _slog.info(f'LayoutDetector: removed {removed} layout entries for {source_file!r}')
            return True
        return False

    def learn_layout(self, screen_type: str, img_size: tuple[int, int], annotations: list[dict], source_file: str = ''):
        """
        Record a confirmed layout to anchors.json.

        Stores full relative geometry per slot:
          x0_rel   — leftmost icon X / image width
          y_rel    — icon row center Y / image height
          w_rel    — icon width / image width
          h_rel    — icon height / image height
          step_rel — X step between consecutive icons / image width
          count    — number of icons stored (for this ship)

        All values are relative so the layout scales correctly to different
        window sizes and resolutions without any estimation.
        """
        if not annotations: return
        h, w = img_size
        aspect = round(w / h, 3)

        # Group annotations by slot, keeping only confirmed ones
        from collections import defaultdict
        slot_bboxes: dict[str, list] = defaultdict(list)
        for ann in annotations:
            bbox = ann.get('bbox')
            slot = ann.get('slot')
            if not bbox or not slot:
                continue
            slot_bboxes[slot].append(bbox)

        if not slot_bboxes:
            return

        slot_map = {}
        for slot, bboxes in slot_bboxes.items():
            # Sort left-to-right
            bboxes_s = sorted(bboxes, key=lambda b: b[0])
            bw = int(round(sum(b[2] for b in bboxes_s) / len(bboxes_s)))
            bh = int(round(sum(b[3] for b in bboxes_s) / len(bboxes_s)))
            cy = int(round(sum(b[1] + b[3] / 2 for b in bboxes_s) / len(bboxes_s)))
            x0 = bboxes_s[0][0]

            if len(bboxes_s) > 1:
                steps = [bboxes_s[i+1][0] - bboxes_s[i][0] for i in range(len(bboxes_s) - 1)]
                step = int(round(sum(steps) / len(steps)))
            else:
                step = bw + max(2, int(bw * 0.08))   # sensible gap when only 1 icon

            slot_map[slot] = {
                'x0_rel':   round(x0 / w, 5),
                'y_rel':    round(cy / h, 5),
                'w_rel':    round(bw / w, 5),
                'h_rel':    round(bh / h, 5),
                'step_rel': round(step / w, 5),
                'count':    len(bboxes_s),
            }

        if not slot_map:
            return

        if not self._calibration:
            self._calibration = {}
        if 'learned' not in self._calibration:
            self._calibration['learned'] = []

        entry = {
            'type':        screen_type,
            'aspect':      aspect,
            'slots':       slot_map,
            'res':         f'{w}x{h}',
            'timestamp':   int(__import__('time').time()),
            'source_file': source_file,
        }

        # Avoid exact duplicates
        total = len(self._calibration['learned'])
        for existing in self._calibration['learned']:
            if (existing['type'] == screen_type
                    and existing['res'] == entry['res']
                    and existing['slots'] == slot_map):
                _slog.debug(f'LayoutDetector: learn_layout {screen_type} {w}x{h} — duplicate, skipping')
                return

        self._calibration['learned'].append(entry)

        # P3: LRU cap — keep at most 200 entries, evict oldest
        MAX_LEARNED = 200
        if len(self._calibration['learned']) > MAX_LEARNED:
            self._calibration['learned'] = self._calibration['learned'][-MAX_LEARNED:]
            _slog.info(f'LayoutDetector: LRU eviction — trimmed to {MAX_LEARNED} entries')

        self._save_calibration()
        _slog.info(
            f'LayoutDetector: saved layout [{screen_type}] {w}x{h} '
            f'({len(slot_map)} slots, total={len(self._calibration["learned"])})'
        )

    def _detect_via_learned_layouts(self, img, build_type, slot_order, profile):
        """Find the best matching learned layout by scoring pixel brightness.

        P3 improvement: instead of blindly picking the most recent layout,
        score each candidate by checking whether bright pixels (icons) exist
        at the predicted slot positions.  The layout whose predicted positions
        best match actual icon regions in the image wins.
        """
        if not self._calibration or 'learned' not in self._calibration:
            return None

        h, w = img.shape[:2]
        aspect = round(w / h, 3)

        # Filter by screen type and similar aspect ratio
        candidates = [e for e in self._calibration['learned']
                      if e['type'] == build_type and abs(e['aspect'] - aspect) < 0.05]

        if not candidates:
            # Strategy 1b: try community anchors (P11)
            community = self._load_community_anchors()
            candidates = [e for e in community
                          if e.get('type') == build_type and abs(e.get('aspect', 0) - aspect) < 0.05]
            if not candidates:
                return None

        # ── Score each candidate by pixel brightness at predicted Y rows ─────
        # Convert to grayscale once for fast brightness sampling
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_score = -1
        best_entry = None

        for entry in candidates:
            score = 0
            checked = 0
            for slot_name, geo in entry['slots'].items():
                if isinstance(geo, (int, float)):
                    continue  # old-format entry
                # Predicted center Y and X region for this slot
                cy = int(geo['y_rel'] * h)
                x0 = int(geo['x0_rel'] * w)
                bw = max(1, int(geo['w_rel'] * w))
                bh = max(1, int(geo['h_rel'] * h))
                step = max(bw, int(geo['step_rel'] * w))
                count = geo.get('count', 1)

                # Sample a small horizontal strip at predicted Y
                y1 = max(0, cy - bh // 4)
                y2 = min(h, cy + bh // 4)
                for j in range(min(count, 8)):  # max 8 icons
                    ix = x0 + j * step
                    ix2 = min(w, ix + bw)
                    if ix >= w or y1 >= y2:
                        continue
                    patch = gray[y1:y2, ix:ix2]
                    if patch.size == 0:
                        continue
                    checked += 1
                    avg_brightness = float(patch.mean())
                    if avg_brightness > 40:  # icon region (brighter than dark BG)
                        score += 1

            # Normalise: fraction of predicted positions that have bright pixels
            norm_score = score / max(checked, 1)
            if norm_score > best_score or (norm_score == best_score and
                    entry.get('timestamp', 0) > (best_entry or {}).get('timestamp', 0)):
                best_score = norm_score
                best_entry = entry

        if best_entry is None:
            return None

        _slog.info(f'LayoutDetector: Strategy 1 (learned) — scored {len(candidates)} layouts '
                   f'for [{build_type}] aspect={aspect}, best score={best_score:.2f} '
                   f'({best_entry["res"]})')

        # ── Build result from best layout ────────────────────────────────────
        result = {}
        for slot_name in slot_order:
            geo = best_entry['slots'].get(slot_name)
            if geo is None:
                continue
            if isinstance(geo, (int, float)):
                continue

            n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n_icons == 0:
                continue

            x0   = int(geo['x0_rel']   * w)
            cy   = int(geo['y_rel']    * h)
            bw   = max(1, int(geo['w_rel']   * w))
            bh   = max(1, int(geo['h_rel']   * h))
            step = max(bw, int(geo['step_rel'] * w))

            bboxes = []
            for j in range(n_icons):
                ix = x0 + j * step
                iy = cy - bh // 2
                bboxes.append((max(0, ix), max(0, iy), bw, bh))
            result[slot_name] = bboxes

        return result if result else None

    # ── Original Logic (truncated for brevity, but kept in final write) ────────

    def _load_community_anchors(self) -> list:
        """Load community_anchors.json (P11) from warp/models/, cached in-memory."""
        if self._community_anchors is not None:
            return self._community_anchors
        try:
            p = Path('warp') / 'models' / 'community_anchors.json'
            if not p.exists():
                self._community_anchors = []
                return []
            data = json.loads(p.read_text(encoding='utf-8'))
            self._community_anchors = data.get('entries', [])
            _slog.info(f'LayoutDetector: loaded {len(self._community_anchors)} community anchor entries')
        except Exception as e:
            _slog.debug(f'LayoutDetector: community anchors unavailable: {e}')
            self._community_anchors = []
        return self._community_anchors

    @staticmethod
    def reset_community_anchors_cache() -> None:
        """Invalidate in-memory community anchors cache on all instances (called by ModelUpdater)."""
        # Walk all live LayoutDetector instances via gc — simpler than a class-level ref
        import gc
        for obj in gc.get_objects():
            if type(obj).__name__ == 'LayoutDetector' and hasattr(obj, '_community_anchors'):
                obj._community_anchors = None

    def _detect_traits(self, img, build_type):
        h, w = img.shape[:2]
        section_map = {
            'personal ground traits': 'Personal Ground Traits', 'ground reputation': 'Ground Reputation',
            'active ground rep': 'Active Ground Rep'
        } if 'GROUND' in build_type else {
            'personal space traits': 'Personal Space Traits', 'starship traits': 'Starship Traits',
            'space reputation': 'Space Reputation', 'active space rep': 'Active Space Rep'
        }
        try: ocr_out = self._get_ocr().readtext(img)
        except: return {}
        headers = []
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3: continue
            text_low = text.lower().strip()
            matched = next((can for kw, can in section_map.items() if kw in text_low or text_low in kw), None)
            if matched: headers.append((matched, int((bbox[0][1] + bbox[2][1]) / 2), int(max(p[0] for p in bbox))))
        if not headers: return {}
        headers.sort(key=lambda x: x[1])
        icon_est = max(32, int(h * 0.055))
        result = {}
        for i, (section, hy, xr) in enumerate(headers):
            row_y = hy + int(icon_est * 0.5)
            row_y_end = (headers[i + 1][1] - 10) if i + 1 < len(headers) else (row_y + icon_est + 20)
            strip = img[max(0, row_y): min(h, row_y_end), :]
            if strip.size == 0: continue
            bboxes = self._find_icon_bboxes_in_strip(strip, max(0, row_y), icon_est)
            if bboxes: result[section] = bboxes
        return result

    def _find_icon_bboxes_in_strip(self, strip, y_offset, icon_size):
        import cv2
        sh, sw = strip.shape[:2]
        if sh == 0 or sw == 0: return []
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        col_bright = np.sum(mask, axis=0).astype(float) / 255
        in_icon, start, bboxes = False, 0, []
        min_w = max(20, icon_size // 2)
        for x in range(sw):
            bright = col_bright[x] > sh * 0.2
            if bright and not in_icon: in_icon, start = True, x
            elif not bright and in_icon:
                in_icon, run_w = False, x - start
                if run_w >= min_w: bboxes.append((start, y_offset + max(0, (sh - icon_size) // 2), run_w, min(icon_size, sh)))
        if in_icon:
            run_w = sw - start
            if run_w >= min_w: bboxes.append((start, y_offset + max(0, (sh - icon_size) // 2), run_w, min(icon_size, sh)))
        return bboxes

    # Boff profession → canonical slot name
    _PROF_MAP = {
        'tactical':      'Boff Tactical',
        'engineering':   'Boff Engineering',
        'science':       'Boff Science',
        'operations':    'Boff Operations',
        'intelligence':  'Boff Intelligence',
        'command':       'Boff Command',
        'pilot':         'Boff Pilot',
        'miracle worker':'Boff Miracle Worker',
        'temporal':      'Boff Temporal',
        'medical':       'Boff Science',
    }

    def _detect_boffs(self, img):
        h, w = img.shape[:2]
        x_start = int(w * 0.55)
        icon_est = max(32, int(h * 0.055))

        # ── Strategy A: OCR finds profession header labels ────────────────────
        headers = []
        try:
            ocr_out = self._get_ocr().readtext(img[:, x_start:])
            for (bbox, text, conf) in ocr_out:
                if conf < 0.3:
                    continue
                text_low = text.lower().strip()
                kw = next((k for k in self._PROF_MAP if k in text_low), None)
                if kw:
                    headers.append((self._PROF_MAP[kw], int((bbox[0][1] + bbox[2][1]) / 2)))
        except Exception:
            pass

        if headers:
            headers.sort(key=lambda x: x[1])
            seen, merged = set(), []
            for n, y in headers:
                if n not in seen:
                    seen.add(n)
                    merged.append((n, y))
            result = {}
            for i, (section, hy) in enumerate(merged):
                row_y     = hy + int(icon_est * 0.3)
                row_y_end = min(h, (merged[i + 1][1] - 5) if i + 1 < len(merged) else h)
                strip = img[max(0, row_y): row_y_end, x_start:]
                if strip.size == 0:
                    continue
                bboxes = self._find_icon_bboxes_in_strip(strip, row_y, icon_est)
                if bboxes:
                    result.setdefault(section, []).extend(
                        [(bx + x_start, by, bw, bh) for (bx, by, bw, bh) in bboxes]
                    )
            if result:
                _slog.debug(f'LayoutDetector: _detect_boffs via OCR — {len(result)} sections')
                return result
            _slog.debug('LayoutDetector: _detect_boffs OCR found headers but no icons — trying color fallback')

        # ── Strategy B: color-based profession detection (no OCR needed) ─────
        _slog.debug('LayoutDetector: _detect_boffs — no OCR headers, using icon color classification')
        panel = img[:, x_start:]
        all_bboxes = self._find_icon_bboxes_in_strip(panel, 0, icon_est)
        if not all_bboxes:
            return {}

        result = {}
        for bx, by, bw, bh in all_bboxes:
            crop = panel[by: by + bh, bx: bx + bw]
            if crop.size == 0:
                continue
            prof = self._classify_boff_profession(crop)
            if prof:
                slot = self._PROF_MAP.get(prof)
                if slot:
                    result.setdefault(slot, []).append(
                        (bx + x_start, by, bw, bh)
                    )

        _slog.debug(
            f'LayoutDetector: _detect_boffs color — {len(result)} sections, '
            f'{sum(len(v) for v in result.values())} icons'
        )
        return result

    @staticmethod
    def _classify_boff_profession(crop_bgr) -> str | None:
        """
        Classify Boff profession from accent glow color in the icon.

        All STO Boff icons share a dark navy-blue background (H 85-120).
        Profession glow color is identified as an ACCENT on top of that background:

          Tactical       — red accent      H  0-15 / 165-180, bright (V≥80)
          Command        — dark-red accent  H  0-15 / 165-180, dim   (V<80)
          Engineering    — amber accent     H 15-30, dominant over blue
          Temporal       — amber + strong mid-blue (H 105-115) alongside amber
          Intelligence   — purple accent    H 115-145
          Miracle Worker — green accent     H 48-72
          Pilot          — cyan accent      H 78-88  (slightly below the bg range)
          Science        — no accent (pure background blue) → default

        Returns the lowercase profession key (matches _PROF_MAP) or None.
        """
        import cv2

        # Sample only the OUTER BORDER RING — profession glow is in the frame,
        # not the center (which carries ability-specific art that varies per ability).
        ih, iw = crop_bgr.shape[:2]
        b = max(3, int(min(ih, iw) * 0.22))
        top    = crop_bgr[:b, :].reshape(-1, 3)
        bottom = crop_bgr[-b:, :].reshape(-1, 3)
        left   = crop_bgr[b:-b, :b].reshape(-1, 3)
        right  = crop_bgr[b:-b, -b:].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right])

        hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat_mask = (hsv[:, 1] > 80) & (hsv[:, 2] > 60)
        if sat_mask.sum() < 8:
            return None

        hues = hsv[sat_mask, 0]   # 0-180
        vals = hsv[sat_mask, 2]
        mean_v = float(vals.mean())

        # 36-bin hue histogram (5° per bin)
        hist, _ = np.histogram(hues, bins=36, range=(0, 180))

        # Helper: count pixels whose hue falls in [lo, hi] degrees (OpenCV 0-180)
        def _h(lo, hi):
            return int(hist[lo // 5: hi // 5 + 1].sum())

        # Hue bands (OpenCV H: 0-180 = half of 360°)
        red_lo    = _h(0,    9)    # pure red H0-9 (real 0-18°); H10+ is orange/amber
        red_hi    = _h(160, 175)   # dark red / maroon (high-H side of wrap-around)
        red_total = red_lo + red_hi
        amber     = _h(10,  30)    # amber/gold = Engineering & Temporal (H10+ = orange-amber)
        mid_blue  = _h(105, 120)   # Temporal's distinctive mid-blue (H105-120, NOT H90-100 bg)
        green     = _h(48,  72)    # green = Miracle Worker
        purple    = _h(115, 145)   # purple/violet = Intelligence
        bg_blue   = _h(85, 120)    # common background navy (shared by all)

        total = int(hist.sum()) or 1

        # ── Command / Tactical (red accent) ───────────────────────────────────
        # Command peaks at H 160-175 (dark maroon), Tactical at H 0-9 (pure red)
        if red_total / total >= 0.05:
            return 'command' if red_hi > red_lo else 'tactical'

        # ── Intelligence (purple accent — unique to this profession) ──────────
        if purple / total >= 0.07 or purple >= 40:
            return 'intelligence'

        # ── Miracle Worker (green accent) ─────────────────────────────────────
        if green / total >= 0.10 or green >= 25:
            return 'miracle worker'

        # ── Engineering vs Temporal (both amber, Temporal also has strong mid-blue)
        if amber / total >= 0.12 or amber >= 40:
            # Temporal: amber is prominent AND significant mid-blue (H105-120) also present
            # Engineering has mid_blue ≈ 0; Temporal has mid_blue = 30-50% of amber
            if mid_blue >= 40 and mid_blue >= amber * 0.28:
                return 'temporal'
            return 'engineering'

        # ── Pilot vs Science (both pure blue; Pilot peaks at H95, Science at H100+)
        # hist bin 19 = H95-99, bin 20 = H100-104
        if int(hist[19]) > int(hist[20]) and int(hist[19]) >= 30:
            return 'pilot'

        # ── Science (default: icon is dominated by background blue, no accent)
        return 'science'

    def _detect_spec(self, img):
        h, w = img.shape[:2]
        y_off = int(h * 0.75)
        try: ocr_out = self._get_ocr().readtext(img[y_off:, :])
        except: return {}
        KNOWN = {'temporal operative', 'strategist', 'intelligence', 'commando', 'miracle worker', 'pilot', 'command', 'constable', 'emergence'}
        found = []
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3: continue
            if any(s in text.lower() for s in KNOWN):
                icon_sz = max(40, int(h * 0.07))
                found.append((max(0, int(bbox[0][0]) - icon_sz - 5), max(0, int(bbox[0][1]) + y_off - icon_sz // 4), icon_sz, icon_sz))
        res = {}
        if len(found) >= 1: res['Primary Specialization'] = [found[0]]
        if len(found) >= 2: res['Secondary Specialization'] = [found[1]]
        return res

    def _count_icons_in_row(self, img, y_top, y_bot, panel_right, cell_w) -> int:
        """Count icons in a row by scanning horizontally for icon-shaped bright patches.
        Scans centre strip of the row. Stops at first dark cell (gap between slot groups)."""
        import numpy as np
        row_h = y_bot - y_top
        # Use inner 50% of row height to avoid separator lines
        y1 = max(0, y_top + row_h // 4)
        y2 = min(img.shape[0], y_bot - row_h // 4)
        count = 0
        max_icons = 8  # STO max slots per row
        consecutive_dark = 0
        for j in range(max_icons):
            x2 = panel_right - j * cell_w
            x1 = max(0, x2 - int(cell_w * 0.85))  # inner 85% of cell
            if x1 >= x2 or x1 < 0:
                break
            strip = img[y1:y2, x1:x2]
            if strip.size == 0:
                break
            avg = float(np.mean(strip))
            if avg > 45:  # icon present — STO icons are distinctly brighter than background
                count += 1
                consecutive_dark = 0
            else:
                consecutive_dark += 1
                if consecutive_dark >= 2:
                    break  # two dark cells = end of slot group
        return max(1, count)

    def _detect_via_pixel_analysis(self, img, slot_order, profile):
        h, w = img.shape[:2]
        panel_right = self._find_panel_right_edge(img)
        if panel_right < w * 0.3: return {}
        row_seps = self._find_row_separators(img, max(0, panel_right - int(w * 0.25)), panel_right)
        if len(row_seps) < 3: return {}
        row_bounds = [(row_seps[i], row_seps[i+1]) for i in range(len(row_seps)-1) if row_seps[i+1]-row_seps[i] >= 30]
        if not row_bounds: return {}
        row_h_avg = sum(b-a for a, b in row_bounds) / len(row_bounds)
        cell_w, icon_w, icon_h = max(30, int(row_h_avg * 0.80)), max(26, int(row_h_avg * 0.80)-4), max(26, int(row_h_avg * 0.78))
        result = {}
        for i, (y_top, y_bot) in enumerate(row_bounds):
            if i >= len(slot_order): break
            slot_name = slot_order[i]
            # Count icons by pixel brightness — more reliable than profile for unknown ships
            pixel_count = self._count_icons_in_row(img, y_top, y_bot, panel_right, cell_w)
            profile_count = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if profile_count <= 1:
                # Single mandatory slot (Deflector, Engines, Warp Core, Shield)
                # Pixel count unreliable here — use profile exactly
                n_icons = profile_count
            else:
                # Multi-slot row: pixel analysis can undercount (empty slots, misalignment)
                # Use max of pixel count and profile count so ShipDB is the floor,
                # but allow pixel count to exceed profile by 1 (T6-X tier upgrades)
                n_icons = min(max(pixel_count, profile_count), profile_count + 1)
            if n_icons == 0: continue
            _slog.info(f'LayoutDetector: row {i} [{slot_name}] pixel_count={pixel_count} profile={profile_count} → using {n_icons}')
            iy, bboxes = (y_top + y_bot) // 2 - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, icon_w, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
        return result

    def _find_panel_right_edge(self, img: np.ndarray) -> int:
        h, w = img.shape[:2]
        y_bands = [(int(h * 0.03 + i * int(h * 0.87 / 10)), int(h * 0.03 + (i + 1) * int(h * 0.87 / 10))) for i in range(10)]
        for x in range(w - 2, max(w // 5, 50), -1):
            if sum(1 for (y1, y2) in y_bands if any(sum(int(c) for c in img[y, x]) / 3 > 50 for y in range(y1, y2, 4))) >= 7: return x
        return int(w * 0.90)

    def _find_row_separators(self, img, x_start, x_end):
        h, w = img.shape[0], img.shape[1]
        x_step = max(1, (x_end - x_start) // 25)
        row_avgs = [sum(sum(int(c) for c in img[y, x]) / 3 for x in range(x_start, x_end, x_step) if x < w) / max(1, (x_end-x_start)//x_step) for y in range(h)]
        smoothed = [sum(row_avgs[max(0, y-2):min(h, y+3)]) / 5 for y in range(h)]
        dark_thr = min(30.0, max(smoothed[10:h-10] if h > 20 else [100.0]) * 0.25)
        dark_runs, in_dark, ds = [], False, 0
        for y, avg in enumerate(smoothed):
            if avg < dark_thr and not in_dark: in_dark, ds = True, y
            elif avg >= dark_thr and in_dark:
                in_dark = False
                if y - ds >= 2: dark_runs.append((ds, y))
        if in_dark: dark_runs.append((ds, h - 1))
        merged = []
        for s, e in dark_runs:
            if merged and s - merged[-1][1] < 4: merged[-1] = (merged[-1][0], e)
            else: merged.append([s, e])
        seps = sorted([int((s + e) / 2) for s, e in merged])
        if not seps or seps[0] > 15: seps = [0] + seps
        if not seps or seps[-1] < h - 40: seps = seps + [h]
        return sorted(seps)

    def _detect_via_ocr(self, img, slot_order, profile):
        try: results = self._get_ocr().readtext(img)
        except: return {}
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.068)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))
        found = {}
        for (bbox_pts, text, conf) in results:
            if conf < OCR_CONF_THRESHOLD: continue
            can = self._match_label(text.strip().lower())
            if not can or can not in slot_order: continue
            n_icons = profile.get(can, SLOT_DEFAULT_COUNTS.get(can, 1))
            if n_icons == 0: continue
            iy, bboxes = int(np.mean([pt[1] for pt in bbox_pts])) - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse(); found[can] = bboxes
        return found

    def _match_label(self, text_lower: str) -> str | None:
        if text_lower in SLOT_LABEL_ALIASES: return SLOT_LABEL_ALIASES[text_lower]
        matches = get_close_matches(text_lower, list(SLOT_LABEL_ALIASES.keys()), n=1, cutoff=LABEL_FUZZY_CUTOFF)
        return SLOT_LABEL_ALIASES.get(matches[0]) if matches else None

    SPACE_ANCHORS_REL: dict[str, tuple[float, int]] = {
        'Fore Weapons': (0.036, 5), 'Deflector': (0.107, 1), 'Engines': (0.178, 1), 'Warp Core': (0.249, 1), 'Shield': (0.325, 1),
        'Aft Weapons': (0.401, 4), 'Devices': (0.475, 4), 'Universal Consoles': (0.547, 2), 'Engineering Consoles': (0.620, 4),
        'Science Consoles': (0.695, 2), 'Tactical Consoles': (0.768, 4), 'Hangar': (0.840, 1),
    }

    def _detect_via_anchors(self, img, slot_order, profile):
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.072)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))
        cal = (self._calibration or {}).get('SPACE', {})
        result = {}
        for slot_name in slot_order:
            y_rel, n_default = cal.get(slot_name, self.SPACE_ANCHORS_REL.get(slot_name, (None, 0)))
            if y_rel is None: continue
            n_icons = profile.get(slot_name, n_default)
            if n_icons == 0: continue
            iy, bboxes = int(h * y_rel) - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse(); result[slot_name] = bboxes
        return result

    def _fill_gaps(self, found, slot_order, img, profile):
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.068)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))
        result, order_map = dict(found), {name: i for i, name in enumerate(slot_order)}
        anchored = sorted([(order_map[name], bboxes[0][1] + bboxes[0][3] // 2) for name, bboxes in found.items() if bboxes and name in order_map])
        if len(anchored) < 2: return result
        for slot_name in slot_order:
            if slot_name in result: continue
            idx = order_map.get(slot_name)
            n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if idx is None or n_icons == 0: continue
            before, after = [i for i in anchored if i[0] < idx], [i for i in anchored if i[0] > idx]
            if before and after: cy = int(before[-1][1] + (idx - before[-1][0]) / max(1, after[0][0] - before[-1][0]) * (after[0][1] - before[-1][1]))
            elif before: cy = int(before[-1][1] + (idx - before[-1][0]) * ((before[-1][1] - before[-2][1]) / max(1, before[-1][0] - before[-2][0]) if len(before) >= 2 else row_h_est))
            elif after: cy = int(after[0][1] - (after[0][0] - idx) * ((after[1][1] - after[0][1]) / max(1, after[1][0] - after[0][0]) if len(after) >= 2 else row_h_est))
            else: continue
            iy, bboxes = cy - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse(); result[slot_name] = bboxes
        return result

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
                try: return json.loads(cfile.read_text())
                except: pass
            p = p.parent
        return None

    def _save_calibration(self):
        p = Path(__file__).resolve().parent
        cfile = None
        for _ in range(6):
            if (p / 'pyproject.toml').exists():
                cfile = p / CALIBRATION_FILE
                break
            p = p.parent
        if cfile:
            cfile.parent.mkdir(parents=True, exist_ok=True)
            cfile.write_text(json.dumps(self._calibration, indent=2))
