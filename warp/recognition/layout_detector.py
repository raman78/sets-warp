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
CALIBRATION_FILE        = Path('warp') / 'training_data' / 'anchors.json'
CANONICAL_LAYOUT_FILE   = Path('warp') / 'training_data' / 'canonical_layout.json'
# Minimum brightness score for canonical layout to be accepted
_CANONICAL_MIN_SCORE    = 0.35

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
    # Traits (used by full scan OCR)
    'personal space traits': 'Personal Space Traits',
    'personal traits':       'Personal Space Traits',
    'starship traits':       'Starship Traits',
    'reputation':            'Space Reputation',
    'space reputation':      'Space Reputation',
    'active space rep':      'Active Space Rep',
    'active reputation':     'Active Space Rep',
    'personal ground traits':'Personal Ground Traits',
    'ground reputation':     'Ground Reputation',
    'active ground rep':     'Active Ground Rep',
    # Boffs
    'stations':              'Boff Tactical',
    'space stations':        'Boff Tactical',
    'standard away team':    'Boff Tactical',
}

# ── Full-scan constants ───────────────────────────────────────────────────────
_SCAN_CONF_MIN  = 0.45   # min ML confidence to keep a sliding-window detection
_SCAN_NMS_IOU   = 0.50   # IoU threshold for greedy NMS
_SCAN_ROW_GAP   = 0.50   # fraction of icon_est: max Y gap within a row

# Equipment item type → slot name (avoids circular import from warp_importer)
_EQ_TYPE_TO_SLOT: dict[str, str] = {
    'Ship Fore Weapon': 'Fore Weapons', 'Ship Weapon': 'Fore Weapons',
    'Experimental Weapon': 'Experimental',
    'Ship Aft Weapon': 'Aft Weapons',
    'Ship Deflector Dish': 'Deflector', 'Ship Secondary Deflector': 'Sec-Def',
    'Impulse Engine': 'Engines',
    'Warp Engine': 'Warp Core', 'Singularity Engine': 'Warp Core',
    'Ship Shields': 'Shield',
    'Ship Device': 'Devices',
    'Ship Engineering Console': 'Engineering Consoles',
    'Ship Science Console': 'Science Consoles',
    'Ship Tactical Console': 'Tactical Consoles',
    'Universal Console': 'Universal Consoles',
    'Hangar Bay': 'Hangars',
    'Ground Weapon': 'Weapons',
    'Body Armor': 'Body Armor', 'EV Suit': 'EV Suit',
    'Personal Shield': 'Personal Shield',
    'Kit': 'Kit', 'Kit Module': 'Kit Modules',
    'Ground Device': 'Ground Devices',
}

# Valid equipment types per slot (mirrors warp_importer.SLOT_VALID_TYPES)
_SCAN_SLOT_VALID_TYPES: dict[str, frozenset] = {
    'Fore Weapons':           frozenset({'Ship Fore Weapon', 'Ship Weapon', 'Experimental Weapon'}),
    'Aft Weapons':            frozenset({'Ship Aft Weapon',  'Ship Weapon', 'Experimental Weapon'}),
    'Experimental':           frozenset({'Experimental Weapon'}),
    'Deflector':              frozenset({'Ship Deflector Dish', 'Ship Secondary Deflector'}),
    'Sec-Def':                frozenset({'Ship Secondary Deflector'}),
    'Engines':                frozenset({'Impulse Engine'}),
    'Warp Core':              frozenset({'Warp Engine', 'Singularity Engine'}),
    'Shield':                 frozenset({'Ship Shields'}),
    'Devices':                frozenset({'Ship Device'}),
    'Engineering Consoles':   frozenset({'Ship Engineering Console', 'Universal Console'}),
    'Science Consoles':       frozenset({'Ship Science Console',     'Universal Console'}),
    'Tactical Consoles':      frozenset({'Ship Tactical Console',    'Universal Console'}),
    'Universal Consoles':     frozenset({'Universal Console', 'Ship Tactical Console',
                                         'Ship Engineering Console', 'Ship Science Console'}),
    'Hangars':                frozenset({'Hangar Bay'}),
    'Weapons':                frozenset({'Ground Weapon'}),
    'Kit Modules':            frozenset({'Kit Module'}),
    'Kit':                    frozenset({'Kit'}),
    'Body Armor':             frozenset({'Body Armor'}),
    'EV Suit':                frozenset({'EV Suit'}),
    'Personal Shield':        frozenset({'Personal Shield'}),
    'Ground Devices':         frozenset({'Ground Device'}),
}

# Trait category key → marker string used in type scoring
_TRAIT_KEY_TO_MARKER: dict[tuple, str] = {
    ('space',  'personal'):   '__trait_space_personal',
    ('space',  'starship'):   '__trait_space_starship',
    ('space',  'rep'):        '__trait_space_rep',
    ('space',  'active_rep'): '__trait_space_active_rep',
    ('ground', 'personal'):   '__trait_ground_personal',
    ('ground', 'rep'):        '__trait_ground_rep',
    ('ground', 'active_rep'): '__trait_ground_active_rep',
}

# Slot name → trait marker (reverse of above + starship)
_TRAIT_SLOT_MARKER: dict[str, str] = {
    'Personal Space Traits':  '__trait_space_personal',
    'Starship Traits':        '__trait_space_starship',
    'Space Reputation':       '__trait_space_rep',
    'Active Space Rep':       '__trait_space_active_rep',
    'Personal Ground Traits': '__trait_ground_personal',
    'Ground Reputation':      '__trait_ground_rep',
    'Active Ground Rep':      '__trait_ground_active_rep',
}

_BOFF_SLOT_NAMES = frozenset({
    'Boff Tactical', 'Boff Engineering', 'Boff Science', 'Boff Operations',
    'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
})

_FULL_SCAN_SLOT_NAMES = (
    list(SPACE_SLOT_ORDER_STANDARD)
    + ['Hangars', 'Experimental', 'Sec-Def']
    + list(GROUND_SLOT_ORDER)
    + ['Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
       'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep']
    + sorted(_BOFF_SLOT_NAMES)
)


def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _nms_boxes(dets: list, iou_thr: float = _SCAN_NMS_IOU) -> list:
    """Greedy NMS. dets = [(x,y,w,h,name,conf,...), ...]. Returns filtered list."""
    kept = []
    for det in sorted(dets, key=lambda d: -d[5]):
        x, y, w, h = det[:4]
        dominated = any(
            _iou_1d(x, x+w, k[0], k[0]+k[2]) * _iou_1d(y, y+h, k[1], k[1]+k[3]) > iou_thr
            for k in kept
        )
        if not dominated:
            kept.append(det)
    return kept


def _cluster_rows_by_y(dets: list, icon_est: int) -> list[list]:
    """Group detections into rows by Y proximity (gap < icon_est * _SCAN_ROW_GAP)."""
    gap = icon_est * _SCAN_ROW_GAP
    rows: list[list] = []
    for det in sorted(dets, key=lambda d: d[1]):
        for row in rows:
            if abs(det[1] - row[0][1]) <= gap:
                row.append(det)
                break
        else:
            rows.append([det])
    return rows


def _get_item_type(item_name: str, eq_cache: dict, trait_cache: dict,
                   starship_traits: dict, boff_cache: dict) -> str:
    """Return type marker for item_name from any of the caches."""
    for cat_items in (eq_cache or {}).values():
        entry = cat_items.get(item_name)
        if entry is not None:
            return entry.get('type', '') if isinstance(entry, dict) else ''
    for (env, cat), marker in _TRAIT_KEY_TO_MARKER.items():
        if env == 'space' and cat == 'starship':
            if item_name in (starship_traits or {}):
                return marker
        else:
            if item_name in (trait_cache or {}).get(env, {}).get(cat, {}):
                return marker
    for env in ('space', 'ground'):
        for prof_name, rank_lists in (boff_cache or {}).get(env, {}).items():
            if not isinstance(rank_lists, list):
                continue
            for rank_dict in rank_lists:
                if isinstance(rank_dict, dict) and item_name in rank_dict:
                    return f'__boff_{prof_name.lower()}'
    return ''


def _score_row_for_slot(row_types: list[str], slot_name: str,
                        ocr_labels: dict, row_cx: float, row_cy: float,
                        icon_est: int) -> float:
    """Score how well a row of icon types fits a slot (0–1)."""
    if not row_types:
        return 0.0

    if slot_name in _TRAIT_SLOT_MARKER:
        needed = _TRAIT_SLOT_MARKER[slot_name]
        type_score = sum(1 for t in row_types if t == needed) / len(row_types)
    elif slot_name in _BOFF_SLOT_NAMES:
        # 'Boff Tactical' → 'tactical', 'Boff Miracle Worker' → 'miracle worker'
        prof_key = slot_name.split(' ', 1)[1].lower()
        exact = sum(1 for t in row_types if t == f'__boff_{prof_key}')
        generic = sum(1 for t in row_types if t.startswith('__boff') and t != f'__boff_{prof_key}')
        type_score = (exact + 0.2 * generic) / len(row_types)
    else:
        valid = _SCAN_SLOT_VALID_TYPES.get(slot_name)
        if not valid:
            return 0.0
        type_score = sum(1 for t in row_types if t in valid) / len(row_types)

    if type_score < 0.1:
        return 0.0

    ocr_score = 0.0
    if slot_name in ocr_labels:
        ocx, ocy = ocr_labels[slot_name]
        dist = ((ocx - row_cx) ** 2 + (ocy - row_cy) ** 2) ** 0.5
        if dist <= icon_est * 3:
            ocr_score = 1.0
        elif dist <= icon_est * 7:
            ocr_score = 0.5

    return 0.65 * type_score + 0.35 * ocr_score


class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.
    Learns new layouts automatically from confirmed annotations.
    """

    def __init__(self):
        self._ocr = None
        self._calibration = self._load_calibration()
        self._community_anchors: list | None = None  # instance cache for community_anchors.json (P11)

    def detect(self, img: np.ndarray, build_type: str, ship_profile: dict | None = None,
               icon_matcher=None, app_cache=None) -> dict[str, list[tuple[int, int, int, int]]]:
        if build_type in ('SPACE_TRAITS', 'GROUND_TRAITS'):
            return self._detect_traits(img, build_type)
        if build_type in ('BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS'):
            if icon_matcher is not None and app_cache is not None:
                full = self._detect_via_full_scan(img, build_type, icon_matcher, app_cache)
                if full and len(full) >= 2:
                    return full
            return self._detect_boffs(img)
        if build_type == 'SPEC':
            return {}

        profile = ship_profile or {}
        if build_type == 'GROUND':
            slot_order = GROUND_SLOT_ORDER
        else:
            slot_order = (SPACE_SLOT_ORDER_CARRIER if profile.get('Hangars', 0) > 0 else SPACE_SLOT_ORDER_STANDARD)

        # Full scan path for MIXED — try after learned layouts (which are always most accurate)
        if build_type in ('SPACE_MIXED', 'GROUND_MIXED') and icon_matcher is not None and app_cache is not None:
            learned = self._detect_via_learned_layouts(img, build_type, slot_order, profile)
            if learned and len(learned) >= 5:
                _slog.info(f'LayoutDetector: Strategy 1 (learned/MIXED) → {len(learned)} slot groups')
                return learned
            full = self._detect_via_full_scan(img, build_type, icon_matcher, app_cache)
            if full and len(full) >= 3:
                return full
            # Fall through to standard strategies if full scan returns too few slots

        # Strategy 1: Learned Layouts — tried FIRST because they contain
        # user-confirmed slot counts, more reliable than ShipDB generic fallback
        learned = self._detect_via_learned_layouts(img, build_type, slot_order, profile)
        if learned:
            # Supplement with pixel analysis for any profile slots missing from learned layout.
            # This handles optional slots (Hangars, Universal Consoles, Sec-Def, Experimental)
            # that may not have been annotated when the layout template was created.
            missing = [s for s in slot_order if s not in learned and profile.get(s, 0) > 0]
            if missing:
                pixel = self._detect_via_pixel_analysis(img, slot_order, profile)
                for slot in missing:
                    if pixel.get(slot):
                        learned[slot] = pixel[slot]
                        _slog.info(f'LayoutDetector: Strategy 1 supplement [{slot}] '
                                   f'from pixel analysis ({len(pixel[slot])} bboxes)')
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

        # Strategy 2.5: Canonical layout + Y-offset scan
        # Uses aggregate learned Y positions when pixel analysis under-covers the screen
        canonical = self._detect_via_canonical_layout(img, build_type, slot_order, profile)
        if canonical and len(canonical) >= max(3, int(len(slot_order) * 0.6)):
            _slog.info(f'LayoutDetector: Strategy 2.5 (canonical) → {len(canonical)} slot groups, {sum(len(v) for v in canonical.values())} bboxes')
            for slot, boxes in canonical.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return canonical

        # Strategy 3: OCR labels
        ocr_result = self._detect_via_ocr(img, slot_order, profile)
        if ocr_result and len(ocr_result) >= 2:
            filled = self._fill_gaps(ocr_result, slot_order, img, profile)
            _slog.info(f'LayoutDetector: Strategy 3 (OCR) → {len(filled)} slot groups, {sum(len(v) for v in filled.values())} bboxes')
            for slot, boxes in filled.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return filled

        # Strategy 4: Anchor fallback — uses canonical learned values if available,
        # otherwise falls back to hardcoded SPACE_ANCHORS_REL
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

    # ── Canonical Layout (aggregate from anchors.json) ────────────────────────

    @classmethod
    def build_canonical_layout(cls) -> dict:
        """
        Aggregate all learned entries from anchors.json into canonical_layout.json.
        Computes median Y/W/H per slot per screen type.
        Called after learn_layout() and as a one-time bootstrap.
        Returns the canonical dict (empty on failure).
        """
        import statistics as _st

        # Locate anchors.json
        p = Path(__file__).resolve().parent
        cal_data = None
        for _ in range(6):
            cfile = p / CALIBRATION_FILE
            if cfile.exists():
                try:
                    cal_data = json.loads(cfile.read_text(encoding='utf-8'))
                except Exception:
                    pass
                break
            p = p.parent
        if not cal_data:
            return {}

        learned = cal_data.get('learned', [])
        if not learned:
            return {}

        from collections import defaultdict
        type_slots: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {'y': [], 'w': [], 'h': []}))

        for entry in learned:
            btype = entry.get('type')
            if not btype:
                continue
            for slot_name, geo in entry.get('slots', {}).items():
                if not isinstance(geo, dict):
                    continue
                sd = type_slots[btype][slot_name]
                if 'y_rel' in geo:
                    sd['y'].append(geo['y_rel'])
                if 'w_rel' in geo:
                    sd['w'].append(geo['w_rel'])
                if 'h_rel' in geo:
                    sd['h'].append(geo['h_rel'])

        canonical: dict = {'version': 1, 'types': {}}
        for btype, slot_data in type_slots.items():
            slots_out = {}
            for slot_name, vals in slot_data.items():
                if len(vals['y']) < 2:
                    continue
                slots_out[slot_name] = {
                    'y_rel': round(_st.median(vals['y']), 5),
                    'y_std': round(_st.stdev(vals['y']), 5),
                    'w_rel': round(_st.median(vals['w']), 5) if vals['w'] else 0.028,
                    'h_rel': round(_st.median(vals['h']), 5) if vals['h'] else 0.055,
                    'n':     len(vals['y']),
                }
            if slots_out:
                canonical['types'][btype] = {
                    'n_samples': sum(1 for e in learned if e.get('type') == btype),
                    'slots':     slots_out,
                }

        # Save next to anchors.json
        p2 = Path(__file__).resolve().parent
        for _ in range(6):
            if (p2 / 'pyproject.toml').exists():
                out = p2 / CANONICAL_LAYOUT_FILE
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(canonical, indent=2), encoding='utf-8')
                _slog.info(
                    f'LayoutDetector: canonical_layout.json saved '
                    f'({len(canonical["types"])} types, '
                    f'{sum(len(v["slots"]) for v in canonical["types"].values())} slot entries)'
                )
                break
            p2 = p2.parent

        return canonical

    def _load_canonical_layout(self) -> dict | None:
        """Load canonical_layout.json. Returns None if missing/corrupt."""
        p = Path(__file__).resolve().parent
        for _ in range(6):
            cfile = p / CANONICAL_LAYOUT_FILE
            if cfile.exists():
                try:
                    return json.loads(cfile.read_text(encoding='utf-8'))
                except Exception:
                    return None
            p = p.parent
        return None

    def _detect_via_canonical_layout(
        self,
        img,
        build_type: str,
        slot_order: list,
        profile: dict,
    ) -> dict | None:
        """
        Strategy 2.5: canonical layout + vertical offset scan.

        Loads the aggregate canonical Y positions (median across all learned entries),
        then searches for the best Y offset by scoring pixel brightness at predicted
        icon rows. Generates bboxes from the best-fit offset.

        Triggered when pixel analysis produces < 70% coverage.
        """
        canonical = self._load_canonical_layout()
        if not canonical:
            return None
        type_data = canonical.get('types', {}).get(build_type)
        if not type_data or not type_data.get('slots'):
            return None

        can_slots: dict = type_data['slots']

        import cv2
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panel_right = self._find_panel_right_edge(img)

        # High-coverage "anchor" slots used for brightness scoring
        ANCHOR_CHECK = {
            'Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield', 'Aft Weapons',
            'Boff Engineering', 'Boff Science', 'Boff Tactical',
            'Personal Space Traits', 'Space Reputation', 'Starship Traits',
            'Primary Specialization', 'Secondary Specialization',
            'Kit Modules', 'Kit', 'Body Armor', 'Personal Shield',
        }

        def _score(dy: float) -> float:
            sc = ck = 0
            for slot_name, geo in can_slots.items():
                if slot_name not in ANCHOR_CHECK:
                    continue
                cy = int((geo['y_rel'] + dy) * h)
                bw = max(18, int(geo['w_rel'] * w))
                bh = max(18, int(geo['h_rel'] * h))
                y1 = max(0, cy - bh // 4)
                y2 = min(h, cy + bh // 4)
                if y1 >= y2:
                    continue
                # Check at panel right edge (STO icons are right-aligned)
                x0 = max(0, panel_right - 5 * bw)
                patch = gray[y1:y2, x0:panel_right]
                if patch.size == 0:
                    continue
                ck += 1
                if float(patch.mean()) > 40:
                    sc += 1
            return sc / max(ck, 1)

        # Clamp dy range: don't allow shifts that push any slot above the image top
        min_y_rel = min((geo['y_rel'] for geo in can_slots.values()), default=0.0)

        # Scan Y offsets -0.20 … +0.20 in 0.01 steps
        best_dy, best_score = 0.0, _score(0.0)
        for dy_i in range(-20, 21):
            if dy_i == 0:
                continue
            dy = dy_i / 100.0
            if min_y_rel + dy < 0:  # would push topmost slot above image top
                continue
            s = _score(dy)
            if s > best_score:
                best_score, best_dy = s, dy

        if best_score < _CANONICAL_MIN_SCORE:
            _slog.debug(
                f'LayoutDetector: canonical [{build_type}] score={best_score:.2f} '
                f'< {_CANONICAL_MIN_SCORE} — skipping'
            )
            return None

        _slog.info(
            f'LayoutDetector: Strategy 2.5 (canonical) [{build_type}] '
            f'dy={best_dy:+.2f} score={best_score:.2f}'
        )

        row_h_est = int(h * 0.068)
        cell_w    = max(30, int(row_h_est * 0.80))

        result = {}
        for slot_name in slot_order:
            geo = can_slots.get(slot_name)
            if geo is None:
                continue
            cy   = int((geo['y_rel'] + best_dy) * h)
            bh   = max(26, int(geo['h_rel'] * h))
            iy   = max(0, cy - bh // 2)
            n    = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n == 0:
                continue
            bboxes = []
            for j in range(n):
                bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, bh))
            bboxes.reverse()
            result[slot_name] = bboxes

        return result if len(result) >= 3 else None

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

        GAP_FACTOR = 2.5   # gap > 2.5× median step = new column (e.g. split Boff Tactical)
        slot_map = {}
        for slot, bboxes in slot_bboxes.items():
            # Sort left-to-right
            bboxes_s = sorted(bboxes, key=lambda b: b[0])
            bw = int(round(sum(b[2] for b in bboxes_s) / len(bboxes_s)))
            bh = int(round(sum(b[3] for b in bboxes_s) / len(bboxes_s)))
            cy = int(round(sum(b[1] + b[3] / 2 for b in bboxes_s) / len(bboxes_s)))

            if len(bboxes_s) == 1:
                step = bw + max(2, int(bw * 0.08))
                slot_map[slot] = {
                    'x0_rel':   round(bboxes_s[0][0] / w, 5),
                    'y_rel':    round(cy / h, 5),
                    'w_rel':    round(bw / w, 5),
                    'h_rel':    round(bh / h, 5),
                    'step_rel': round(step / w, 5),
                    'count':    1,
                }
                continue

            steps = [bboxes_s[i+1][0] - bboxes_s[i][0] for i in range(len(bboxes_s) - 1)]
            median_step = sorted(steps)[len(steps) // 2]
            split_indices = [i for i, s in enumerate(steps) if s > GAP_FACTOR * median_step]

            if not split_indices:
                # Single contiguous run — use flat format (backward compatible)
                step = int(round(sum(steps) / len(steps)))
                slot_map[slot] = {
                    'x0_rel':   round(bboxes_s[0][0] / w, 5),
                    'y_rel':    round(cy / h, 5),
                    'w_rel':    round(bw / w, 5),
                    'h_rel':    round(bh / h, 5),
                    'step_rel': round(step / w, 5),
                    'count':    len(bboxes_s),
                }
            else:
                # Multiple columns (e.g. same Boff profession in left+right column)
                runs = []
                prev = 0
                for si in split_indices + [len(bboxes_s) - 1]:
                    chunk = bboxes_s[prev:si + 1]
                    chunk_steps = [chunk[j+1][0] - chunk[j][0] for j in range(len(chunk) - 1)]
                    chunk_step = int(round(sum(chunk_steps) / len(chunk_steps))) if chunk_steps else (bw + max(2, int(bw * 0.08)))
                    runs.append({
                        'x0_rel':   round(chunk[0][0] / w, 5),
                        'step_rel': round(chunk_step / w, 5),
                        'count':    len(chunk),
                    })
                    prev = si + 1
                slot_map[slot] = {
                    'y_rel': round(cy / h, 5),
                    'w_rel': round(bw / w, 5),
                    'h_rel': round(bh / h, 5),
                    'runs':  runs,
                }
                _slog.info(f'LayoutDetector: learn_layout [{slot}] split into {len(runs)} runs: {[(r["count"], round(r["x0_rel"],3)) for r in runs]}')

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
        total_bboxes = sum(
            v['count'] if 'count' in v else sum(r['count'] for r in v.get('runs', []))
            for v in slot_map.values()
        )
        _slog.info(
            f'LayoutDetector: saved layout [{screen_type}] {w}x{h} '
            f'({len(slot_map)} slot groups, {total_bboxes} bboxes'
            + (f', src={source_file}' if source_file else '')
            + f', total entries={len(self._calibration["learned"])})'
        )
        # Rebuild canonical layout so Strategy 2.5 benefits from new data
        try:
            LayoutDetector.build_canonical_layout()
        except Exception as _ce:
            _slog.debug(f'LayoutDetector: canonical rebuild failed: {_ce}')

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
                bw = max(1, int(geo['w_rel'] * w))
                bh = max(1, int(geo['h_rel'] * h))
                cy = int(geo['y_rel'] * h)
                y1 = max(0, cy - bh // 4)
                y2 = min(h, cy + bh // 4)
                # Normalise to runs — handles both flat and multi-run formats
                runs = geo.get('runs') or [{'x0_rel': geo['x0_rel'], 'step_rel': geo['step_rel'], 'count': geo.get('count', 1)}]
                for run in runs:
                    x0   = int(run['x0_rel'] * w)
                    step = max(bw, int(run['step_rel'] * w))
                    for j in range(min(run['count'], 8)):
                        ix  = x0 + j * step
                        ix2 = min(w, ix + bw)
                        if ix >= w or y1 >= y2:
                            continue
                        patch = gray[y1:y2, ix:ix2]
                        if patch.size == 0:
                            continue
                        checked += 1
                        if float(patch.mean()) > 40:  # icon region (brighter than dark BG)
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
            if geo is None or isinstance(geo, (int, float)):
                continue

            cy = int(geo['y_rel'] * h)
            bw = max(1, int(geo['w_rel'] * w))
            bh = max(1, int(geo['h_rel'] * h))
            iy = max(0, cy - bh // 2)

            bboxes = []
            if 'runs' in geo:
                # Multi-column layout — use stored run counts (authoritative)
                for run in geo['runs']:
                    x0   = int(run['x0_rel'] * w)
                    step = max(bw, int(run['step_rel'] * w))
                    for j in range(run['count']):
                        bboxes.append((max(0, x0 + j * step), iy, bw, bh))
            else:
                # Single-run layout — respect ship profile count
                n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, geo.get('count', 1)))
                if n_icons == 0:
                    continue
                x0   = int(geo['x0_rel'] * w)
                step = max(bw, int(geo['step_rel'] * w))
                for j in range(n_icons):
                    bboxes.append((max(0, x0 + j * step), iy, bw, bh))

            if bboxes:
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
        # Trait icons are ~44–55 px absolute regardless of screen height.
        # h * 0.055 underestimates at low-res windows (gives 30 px at h=560).
        icon_est = max(44, int(h * 0.065))
        result = {}
        for i, (section, hy, xr) in enumerate(headers):
            row_y = hy + int(icon_est * 0.5)
            row_y_end = (headers[i + 1][1] - 10) if i + 1 < len(headers) else (row_y + icon_est * 4)
            strip = img[max(0, row_y): min(h, row_y_end), :]
            if strip.size == 0: continue
            # Detect individual icon rows within the section strip —
            # a section may overflow to 2+ rows (e.g. 11 personal space traits).
            row_centers = self._find_icon_rows_in_strip(strip, icon_est)
            _slog.debug(f'LayoutDetector._detect_traits: section={section!r} '
                        f'strip y={max(0,row_y)}..{min(h,row_y_end)} '
                        f'(h={strip.shape[0]}) icon_est={icon_est} '
                        f'row_centers={row_centers}')
            all_bboxes = []
            for rc in row_centers:
                r0 = max(0, rc - icon_est // 2)
                r1 = min(strip.shape[0], rc + icon_est // 2 + 1)
                row_strip = strip[r0:r1, :]
                bboxes = self._find_icon_bboxes_in_strip(row_strip, max(0, row_y) + r0, icon_est)
                _slog.debug(f'LayoutDetector._detect_traits:   rc={rc} '
                            f'row_strip h={row_strip.shape[0]} → {len(bboxes)} bboxes')
                all_bboxes.extend(bboxes)
            if all_bboxes:
                result[section] = all_bboxes
        return result

    def _find_icon_rows_in_strip(self, strip: np.ndarray, icon_est: int) -> list[int]:
        """
        Find Y-centers of icon rows within a section strip.
        Returns list of Y offsets (relative to strip top), one per row found.
        Handles multi-row sections (e.g. 11 personal space traits = 2 rows).
        """
        sh, sw = strip.shape[:2]
        if sh < icon_est // 2:
            return [sh // 2]
        import cv2
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        # Row brightness: how many bright pixels per row
        row_bright = np.sum(mask, axis=1).astype(float) / max(sw, 1)
        # Smooth with a narrow kernel to reduce per-pixel noise
        kernel = max(3, icon_est // 10)
        smoothed = np.convolve(row_bright, np.ones(kernel) / kernel, mode='same')
        # Use 90th-percentile instead of max to avoid a single bright separator line
        # inflating the threshold above actual icon rows.
        peak = float(np.percentile(smoothed, 90))
        _slog.debug(f'LayoutDetector._find_icon_rows: sh={sh} peak={peak:.1f}')
        if peak < 5:          # almost no bright pixels → empty strip
            return [sh // 2]
        threshold = peak * 0.30
        _slog.debug(f'LayoutDetector._find_icon_rows: threshold={threshold:.1f}')
        # Find bright runs (candidate icon rows), merge gaps < icon_est//3
        min_sep = max(icon_est // 2, 20)
        centers: list[int] = []
        in_bright, run_start = False, 0
        for y in range(sh):
            if smoothed[y] >= threshold and not in_bright:
                in_bright, run_start = True, y
            elif smoothed[y] < threshold and in_bright:
                in_bright = False
                center = (run_start + y) // 2
                run_len = y - run_start
                # Filter out thin text labels (< icon_est/2 tall)
                accepted = run_len >= icon_est // 2 and (not centers or center - centers[-1] >= min_sep)
                _slog.debug(f'LayoutDetector._find_icon_rows:   run y={run_start}..{y} '
                            f'len={run_len} center={center} → {"OK" if accepted else "skip"}')
                if accepted:
                    centers.append(center)
        if in_bright:
            center = (run_start + sh) // 2
            run_len = sh - run_start
            accepted = run_len >= icon_est // 2 and (not centers or center - centers[-1] >= min_sep)
            _slog.debug(f'LayoutDetector._find_icon_rows:   run y={run_start}..{sh} '
                        f'len={run_len} center={center} → {"OK(tail)" if accepted else "skip(tail)"}')
            if accepted:
                centers.append(center)
        return centers if centers else [sh // 2]

    def _find_icon_bboxes_in_strip(self, strip, y_offset, icon_size):
        import cv2
        sh, sw = strip.shape[:2]
        if sh == 0 or sw == 0: return []
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        col_bright = np.sum(mask, axis=0).astype(float) / 255
        in_icon, start, bboxes = False, 0, []
        min_w = max(20, icon_size // 2)
        # Threshold: 10% of strip height (was 20%). Dark-bordered icons can have
        # columns with only 3–5 bright pixels; 20% broke their runs into sub-min_w
        # fragments that were discarded, causing the first icon to be missed.
        for x in range(sw):
            bright = col_bright[x] > sh * 0.1
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
        """Detect BOFF ability icons using structural knowledge of BOFFS screen.

        BOFFS layout: 2 columns (left: max 3 seats, right: max 2 seats).
        Each seat has up to 4 ability icons in a horizontal row.

        Approach:
        1. Find icon row bands via vertical brightness/variance profile
        2. Column split at ~55% of image width (consistent across BOFFS screens)
        3. Template-slide a 4-icon pattern across each (row, column) cell
        4. Classify profession per seat via color analysis
        """
        import cv2
        from collections import Counter
        h, w = img.shape[:2]

        icon_w = max(20, round(w * 0.078))
        icon_h = max(28, round(h * 0.110))
        spacing = max(24, round(w * 0.093))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 1: Find icon row bands ---
        # Compute per-row stats using a small vertical window for stability
        win = max(3, icon_h // 8)
        row_bands: list[tuple[int, int]] = []
        in_band = False
        band_start = 0
        skip_y = round(h * 0.12)  # skip header area

        for y in range(skip_y, h - win):
            strip = gray[y:y + win, :]
            mn = float(strip.mean())
            sd = float(strip.std())
            # Icon rows: moderate brightness + high variance (NOT name plates)
            is_icon = 30 < mn < 85 and sd > 30
            if is_icon:
                if not in_band:
                    band_start = y
                    in_band = True
            else:
                if in_band and y - band_start > icon_h * 0.5:
                    row_bands.append((band_start, y))
                in_band = False
        if in_band and h - band_start > icon_h * 0.5:
            row_bands.append((band_start, h))

        if not row_bands:
            _slog.debug('LayoutDetector: _detect_boffs — no row bands found')
            return {}

        # Merge close bands (gap < icon_h/3 → part of same row)
        merged: list[tuple[int, int]] = []
        for y1, y2 in row_bands:
            if merged and y1 - merged[-1][1] < icon_h // 3:
                merged[-1] = (merged[-1][0], y2)
            else:
                merged.append((y1, y2))
        row_bands = merged

        # BOFFS has max 3 row positions — drop extra bands (name plate noise)
        row_bands = row_bands[:3]

        # Expand narrow bands to at least icon_h (band edges may miss icon top/bottom)
        expanded: list[tuple[int, int]] = []
        for y1, y2 in row_bands:
            if y2 - y1 < icon_h:
                center = (y1 + y2) // 2
                y1 = max(0, center - icon_h // 2)
                y2 = min(h, y1 + icon_h)
            expanded.append((y1, y2))
        row_bands = expanded

        _slog.debug(
            f'LayoutDetector: _detect_boffs — {len(row_bands)} row bands: '
            + ', '.join(f'y={y1}-{y2}' for y1, y2 in row_bands)
        )

        # --- Step 2: Column split ---
        col_split = round(w * 0.55)
        columns = [(0, col_split), (col_split, w)]

        # --- Step 3: Template-slide 4 icons per cell ---
        result: dict[str, list] = {}

        for band_y1, band_y2 in row_bands:
            for col_x1, col_x2 in columns:
                cell_w = col_x2 - col_x1
                if cell_w < spacing * 2:
                    continue

                # Use vertical center of band for the horizontal profile
                mid_y1 = band_y1 + max(0, (band_y2 - band_y1 - icon_h) // 2)
                mid_y2 = min(mid_y1 + icon_h, band_y2)
                cell = gray[mid_y1:mid_y2, col_x1:col_x2]
                if cell.size == 0:
                    continue

                # Score using horizontal std profile (icon artwork has high variance;
                # profession indicators and empty areas have low variance).
                h_std = cell.std(axis=0).astype(float)
                h_mean = cell.mean(axis=0).astype(float)

                # Slide 4-icon template: score = sum of (mean * std) at 4 positions.
                # Profession indicators at each row's left edge create a false peak.
                # Strategy: find all scores, then pick the rightmost strong peak
                # (correct icons are always to the right of the indicators).
                search_end = max(1, cell_w - 3 * spacing - icon_w + 1)
                all_scores: list[tuple[float, int]] = []
                for x_start in range(search_end):
                    score = 0.0
                    for k in range(4):
                        x = x_start + k * spacing
                        x_end = min(x + icon_w, cell_w)
                        if x < cell_w:
                            m = float(h_mean[x:x_end].mean())
                            s = float(h_std[x:x_end].mean())
                            score += m * s
                    all_scores.append((score, x_start))

                if not all_scores:
                    continue

                all_scores.sort(reverse=True)
                best_score, best_x = all_scores[0]

                # If the global peak is in the indicator zone (very left of cell),
                # look for a rightward peak — real icons are always to the right.
                if best_x < icon_w // 2:
                    for sc, xs in all_scores:
                        if xs > best_x + icon_w and sc > best_score * 0.60:
                            best_x = xs
                            best_score = sc
                            break

                if best_score < 500:
                    continue

                # Refine: find exact Y within band for each icon
                icon_y = mid_y1
                # Try to find the best y alignment by scanning vertically
                best_y_score = 0.0
                for try_y in range(band_y1, max(band_y1 + 1, band_y2 - icon_h + 1)):
                    strip = gray[try_y:try_y + icon_h,
                                 col_x1 + best_x:min(col_x1 + best_x + icon_w, col_x2)]
                    if strip.size > 0:
                        sc = float(strip.std())
                        if sc > best_y_score:
                            best_y_score = sc
                            icon_y = try_y

                # Build bboxes for 4 icon positions with state classification
                bboxes = []
                for k in range(4):
                    ix = col_x1 + best_x + k * spacing
                    crop_g = gray[icon_y:icon_y + icon_h, ix:ix + icon_w]
                    std = float(crop_g.std()) if crop_g.size > 0 else 0
                    state = 'active' if std > 30 else (
                        'inactive' if std > 8 else 'empty')
                    bboxes.append((ix, icon_y, icon_w, icon_h, state))

                # Classify profession by majority vote — only on active crops.
                profs: list[str] = []
                for ix, iy, iw, ih, state in bboxes:
                    if state != 'active':
                        continue
                    crop = img[iy:iy + ih, ix:ix + iw]
                    if crop.size > 0:
                        prof = self._classify_boff_profession(crop)
                        if prof:
                            profs.append(prof)

                if not profs:
                    continue

                majority = Counter(profs).most_common(1)[0][0]
                slot = self._PROF_MAP.get(majority)
                if not slot:
                    continue

                result.setdefault(slot, []).extend(bboxes)

        _slog.debug(
            f'LayoutDetector: _detect_boffs — {len(result)} sections, '
            f'{sum(len(v) for v in result.values())} bboxes'
        )
        return result

    def _fill_boff_gaps(self, bboxes_abs: list, img, icon_est: int,
                        x_min: int = 0, max_slots: int = 4) -> list:
        """
        Given absolute-coordinate bboxes of active BOFF icons in one seat row,
        fill in empty/inactive positions at expected grid intervals.

        Returns list of (x, y, w, h, state) 5-tuples where state is
        'active', 'empty', or 'inactive'.
        """
        if not bboxes_abs:
            return []

        sorted_bx = sorted(bboxes_abs, key=lambda b: b[0])
        xs = [b[0] for b in sorted_bx]

        # Step estimate: minimum positive X-gap, or icon_est for a single icon
        if len(xs) >= 2:
            gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            pos_gaps = [g for g in gaps if g > 0]
            step = min(pos_gaps) if pos_gaps else icon_est
        else:
            step = icon_est

        x0, y0, w0, h0 = sorted_bx[0]
        result: list = []
        consumed: set = set()

        for slot_i in range(max_slots):
            x_exp = x0 + slot_i * step
            if x_exp + w0 > img.shape[1]:
                break

            # Find active bbox closest to expected position (within 45% of step)
            match_i = None
            for i, (bx, _, _, _) in enumerate(sorted_bx):
                if i not in consumed and abs(bx - x_exp) < step * 0.45:
                    match_i = i
                    break

            if match_i is not None:
                result.append((*sorted_bx[match_i], 'active'))
                consumed.add(match_i)
            else:
                y1 = max(0, y0)
                y2 = min(img.shape[0], y0 + h0)
                x1 = max(0, int(x_exp))
                x2 = min(img.shape[1], x1 + w0)
                crop = img[y1:y2, x1:x2]
                state = self._classify_cell(crop) if crop.size > 0 else 'empty'
                if state == 'active':
                    state = 'empty'  # no matched bbox here — treat as empty
                result.append((int(x_exp), y0, w0, h0, state))

        virtual_n = sum(1 for r in result if r[4] != 'active')
        if virtual_n:
            _slog.debug(f'LayoutDetector: _fill_boff_gaps — {virtual_n} virtual positions added '
                        f'({len(result) - virtual_n} active)')
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
        """Detect Primary / Secondary Specialization icons on a SPEC screen.

        STO marks each selected spec with a diamond icon to the right of the
        spec name.  The diamond contains a Roman numeral:
          I  (single bar) → Primary Specialization
          II (two bars)   → Secondary Specialization

        Detection steps:
          1. OCR full image → find spec name label positions.
          2. For each label, sample the region immediately right of the text.
          3. If the region has significant brightness variance → icon present.
          4. Horizontal bright-cluster count in the interior of the icon:
               1 cluster → "I" → Primary
               2 clusters → "II" → Secondary
        """
        import cv2
        h, w = img.shape[:2]
        try:
            ocr_out = self._get_ocr().readtext(img)
        except:
            return {}

        KNOWN = {'temporal operative', 'strategist', 'intelligence', 'commando',
                 'miracle worker', 'pilot', 'command', 'constable', 'emergence'}

        # Collect candidate rows: (tx1, ty1, tx2, ty2)
        candidates = []
        for (pts, text, conf) in ocr_out:
            if conf < 0.3:
                continue
            if not any(s in text.lower() for s in KNOWN):
                continue
            tx1 = int(min(p[0] for p in pts))
            ty1 = int(min(p[1] for p in pts))
            tx2 = int(max(p[0] for p in pts))
            ty2 = int(max(p[1] for p in pts))
            candidates.append((tx1, ty1, tx2, ty2))

        results = []  # (ty1, is_primary, bbox_xywh)
        for (tx1, ty1, tx2, ty2) in candidates:
            row_h = max(ty2 - ty1, 12)
            # Sample region to the right of the text where the diamond icon sits
            ix1 = tx2 + 2
            ix2 = min(w, tx2 + row_h * 2)
            iy1 = max(0, ty1 - 2)
            iy2 = min(h, ty2 + 2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            icon_crop = img[iy1:iy2, ix1:ix2]
            gray = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2GRAY)

            # Skip if no significant content (spec not selected — no icon present)
            if int(gray.max()) - int(gray.min()) < 30:
                continue

            # Threshold: pixels clearly brighter than the crop's mean
            bright = (gray > gray.mean() + 15).astype('uint8')

            # Horizontal center strip, interior 60% (avoids diamond border edges)
            cy = bright.shape[0] // 2
            sh = max(1, bright.shape[0] // 5)
            strip = bright[cy - sh: cy + sh + 1, :]
            iw = strip.shape[1]
            interior = strip[:, iw // 5: 4 * iw // 5]
            col_on = interior.mean(axis=0) > 0.3

            # Count 0→1 transitions = number of distinct bright clusters
            clusters = sum(1 for i in range(1, len(col_on))
                           if col_on[i] and not col_on[i - 1])

            is_primary = (clusters <= 1)
            # Bbox covers text + icon
            bbox_xywh = (tx1, ty1, ix2 - tx1, ty2 - ty1)
            results.append((ty1, is_primary, bbox_xywh))

        if not results:
            return {}

        res = {}
        for (_ty, is_primary, bbox) in results:
            slot = 'Primary Specialization' if is_primary else 'Secondary Specialization'
            if slot not in res:
                res[slot] = [bbox]
        return res

    @staticmethod
    def _classify_cell(crop_bgr) -> str:
        """
        Classify a single slot cell crop as 'active', 'empty', or 'inactive'.

        Uses the inner 60% of the crop to avoid border contamination.

        active   — has a visible icon (bright content)
        inactive — locked / unavailable slot:
                     BOFFS: dark navy-blue with X pattern (blue-saturated)
                     EQ/Traits: near-black with 'LOCK' text (higher brightness variance)
        empty    — slot exists but nothing is slotted (uniform near-black, thin border only)
        """
        import cv2
        if crop_bgr is None or crop_bgr.size == 0:
            return 'active'  # unknown → treat as active (safe fallback)
        ih, iw = crop_bgr.shape[:2]
        mx = max(1, int(iw * 0.20))
        my = max(1, int(ih * 0.20))
        inner = crop_bgr[my:ih - my, mx:iw - mx]
        if inner.size == 0:
            inner = crop_bgr
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        mean_v = hsv[:, 2].mean()
        std_v  = hsv[:, 2].std()
        mean_s = hsv[:, 1].mean()
        mean_h = hsv[:, 0].mean()

        if mean_v > 45:
            return 'active'
        # BOFFS inactive: navy-blue X pattern — distinctly saturated blue
        if mean_s > 40 and 95 < mean_h < 130:
            return 'inactive'
        # LOCK (EQ/Traits): near-black but text pixels raise brightness variance
        if std_v > 10:
            return 'inactive'
        return 'empty'

    def _count_icons_in_row(self, img, y_top, y_bot, panel_right, cell_w,
                            slot_name: str = '') -> tuple[int, list[str]]:
        """
        Count active icons in a row, scanning right-to-left.

        Returns (count, cell_states) where cell_states is a list of
        'active' | 'empty' | 'inactive' for each scanned cell position
        (index 0 = rightmost cell).

        Empty and inactive cells are skipped in the count but do NOT
        stop the scan — only two consecutive background cells stop it.
        A background cell is any dark cell that lies outside the known
        slot grid (distinguished from empty/inactive by context: once
        we exit the grid there is no more slot structure).
        """
        import cv2
        row_h = y_bot - y_top
        y1 = max(0, y_top + row_h // 4)
        y2 = min(img.shape[0], y_bot - row_h // 4)
        count = 0
        max_icons = 8
        consecutive_bg = 0   # counts cells that look like plain background (not a slot)
        cell_states: list[str] = []
        for j in range(max_icons):
            x2 = panel_right - j * cell_w
            x1 = max(0, x2 - int(cell_w * 0.85))
            if x1 >= x2 or x1 < 0:
                break
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                break
            state = self._classify_cell(crop)
            cell_states.append(state)
            if state == 'active':
                count += 1
                consecutive_bg = 0
            else:
                # empty/inactive — still a grid cell, don't increment bg counter
                # but check if it truly looks like featureless background:
                # background has even lower brightness and zero border structure
                avg = float(crop.mean())
                if avg < 8:          # essentially pure black — outside grid
                    consecutive_bg += 1
                    if consecutive_bg >= 2:
                        break
                # else: empty/inactive slot within grid — keep scanning
        if slot_name and any(s in ('empty', 'inactive') for s in cell_states):
            non_active = [(i, s) for i, s in enumerate(cell_states) if s != 'active']
            _slog.info(
                f'LayoutDetector: [{slot_name}] {count} active + '
                + ', '.join(f'{s} at pos {i}' for i, s in non_active)
            )
        return max(1, count), cell_states

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
            pixel_count, _ = self._count_icons_in_row(img, y_top, y_bot, panel_right, cell_w, slot_name)
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

    # ── Full scan (MIXED screens) ────────────────────────────────────────────────

    def _ocr_section_labels(self, img) -> dict[str, tuple[float, float]]:
        """Run full-image OCR, return {slot_name: (center_x, center_y)} for each found label."""
        try:
            results = self._get_ocr().readtext(img)
        except Exception:
            return {}
        labels: dict[str, tuple[float, float]] = {}
        for (bbox_pts, text, conf) in results:
            if conf < OCR_CONF_THRESHOLD:
                continue
            slot = self._match_label(text.strip().lower())
            if slot:
                cx = float(np.mean([pt[0] for pt in bbox_pts]))
                cy = float(np.mean([pt[1] for pt in bbox_pts]))
                labels[slot] = (cx, cy)
        return labels

    def _detect_via_full_scan(self, img: np.ndarray, build_type: str,
                               icon_matcher, app_cache) -> dict[str, list[tuple]]:
        """Full-image detection for MIXED screens: OCR labels + dense icon scan + fusion.

        Phase 1 — OCR: find slot label positions across the full image.
        Phase 2 — Dense scan: sliding window + EfficientNet classify every patch.
        Phase 3 — NMS + row clustering + per-row slot scoring.
        Phase 4 — Output dict {slot_name: [(x,y,w,h), ...]} like other strategies.
        """
        from collections import Counter
        h, w = img.shape[:2]
        icon_est = max(32, int(h * 0.060))
        stride   = max(8, icon_est // 2)

        eq_cache        = getattr(app_cache, 'equipment',      {}) or {}
        trait_cache     = getattr(app_cache, 'traits',         {}) or {}
        starship_traits = getattr(app_cache, 'starship_traits',{}) or {}
        boff_cache      = getattr(app_cache, 'boff_abilities', {}) or {}

        # Phase 1: OCR section labels
        ocr_labels = self._ocr_section_labels(img)
        _slog.info(f'LayoutDetector FullScan: OCR labels → {list(ocr_labels.keys())}')

        # Phase 2: Sliding window
        raw_dets = []
        for y in range(0, h - icon_est + 1, stride):
            for x in range(0, w - icon_est + 1, stride):
                patch = img[y:y+icon_est, x:x+icon_est]
                if float(patch.std()) < 12.0:  # skip uniform background
                    continue
                name, conf = icon_matcher.classify_patch(patch)
                if conf >= _SCAN_CONF_MIN and name:
                    raw_dets.append((x, y, icon_est, icon_est, name, conf))

        if not raw_dets:
            _slog.info('LayoutDetector FullScan: no detections — scan failed')
            return {}

        dets = _nms_boxes(raw_dets)
        _slog.info(f'LayoutDetector FullScan: {len(raw_dets)} raw → {len(dets)} after NMS')

        # Enrich detections with item type (computed once, reused across scoring).
        # For unrecognised patches in BOFF screens, try color classification as fallback.
        _is_boff_screen = build_type in ('BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS')
        enriched = []
        for det in dets:
            itype = _get_item_type(det[4], eq_cache, trait_cache, starship_traits, boff_cache)
            if not itype and _is_boff_screen:
                x, y, bw, bh = det[:4]
                crop = img[y:y+bh, x:x+bw]
                if crop.size > 0:
                    prof = self._classify_boff_profession(crop)
                    if prof:
                        itype = f'__boff_{prof}'
            enriched.append(det + (itype,))
        # each entry: (x, y, w, h, name, conf, item_type)

        # Phase 3: Cluster into rows
        rows = _cluster_rows_by_y(enriched, icon_est)
        _slog.info(f'LayoutDetector FullScan: {len(rows)} rows')

        # Phase 4: Score each row against candidate slot names
        result: dict[str, list] = {}
        used_slots: set[str] = set()

        # Only consider slot names relevant to this build type
        boff_slots = sorted(_BOFF_SLOT_NAMES)
        if 'SPACE' in build_type or build_type in ('BOFFS',):
            candidates = (
                list(SPACE_SLOT_ORDER_STANDARD) + ['Hangars', 'Experimental', 'Sec-Def']
                + ['Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep']
                + boff_slots
            )
        else:
            candidates = (
                list(GROUND_SLOT_ORDER)
                + ['Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep']
                + boff_slots
            )

        for row in sorted(rows, key=lambda r: min(it[1] for it in r)):
            row_types = [it[6] for it in row]
            row_cx    = sum(it[0] + it[2] / 2 for it in row) / len(row)
            row_cy    = sum(it[1] + it[3] / 2 for it in row) / len(row)

            best_slot, best_score = '', 0.0
            for slot_name in candidates:
                score = _score_row_for_slot(row_types, slot_name, ocr_labels,
                                            row_cx, row_cy, icon_est)
                if score > best_score:
                    best_score = score
                    best_slot  = slot_name

            if not best_slot or best_score < 0.30:
                continue

            # Multi-row slots (traits, boffs) — allow multiple rows per slot
            is_multi = best_slot in _TRAIT_SLOT_MARKER or best_slot in _BOFF_SLOT_NAMES
            if best_slot in used_slots and not is_multi:
                continue  # single-row slot already filled
            used_slots.add(best_slot)

            bboxes = [(it[0], it[1], it[2], it[3]) for it in sorted(row, key=lambda it: it[0])]
            result.setdefault(best_slot, []).extend(bboxes)

        _slog.info(f'LayoutDetector FullScan: {len(result)} slots → {list(result.keys())}')
        return result

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

        # Load canonical learned Y values; fall back to hardcoded SPACE_ANCHORS_REL
        canonical = self._load_canonical_layout()
        can_slots = {}
        if canonical:
            # Use build_type='SPACE' as best general fallback for equipment screens
            can_slots = canonical.get('types', {}).get('SPACE', {}).get('slots', {})

        cal = (self._calibration or {}).get('SPACE', {})
        result = {}
        for slot_name in slot_order:
            # Priority: canonical learned > hardcoded
            if slot_name in can_slots:
                y_rel    = can_slots[slot_name]['y_rel']
                n_default = SLOT_DEFAULT_COUNTS.get(slot_name, 1)
            else:
                anchor = cal.get(slot_name, self.SPACE_ANCHORS_REL.get(slot_name))
                if anchor is None:
                    continue
                y_rel, n_default = anchor if isinstance(anchor, tuple) else (anchor, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            n_icons = profile.get(slot_name, n_default)
            if n_icons == 0:
                continue
            iy = int(h * y_rel) - icon_h // 2
            bboxes = []
            for j in range(n_icons):
                bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
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
