"""
OCR-anchored GROUND EQ panel geometry detector.

Ground EQ topology (top→bottom):
  row 0   Kit Modules   — single row, up to 6 cells, LEFT-shifted from main block
  row 1   Kit           — 1 cell in left column
  row 2   Body Armor    — 1 cell in left column      (OCR label: "Body")
          EV Suit       — 1 cell in right column     (same y as Body)
  row 3   Personal Shield — 1 cell in left column    (OCR label: "Shields")
  row 4   Weapons       — 2 cells STACKED in left column (NOT side-by-side)
  row 5+  Ground Devices — multi-row, both columns   (OCR label: "Devices")

OCR anchors (7 labels above the icons):
  "Kit Modules"   → km_label_x0, km_label_cy
  "Kit"           → col_left_x, slot_rows['Kit']
  "Body"          → col_left_x, slot_rows['Body Armor']
  "EV Suit"       → col_right_x, slot_rows['EV Suit']     (optional)
  "Shields"       → col_left_x, slot_rows['Personal Shield']
                    (filter out HUD "Shields:" stat, x usually <200)
  "Weapons"       → col_left_x, slot_rows['Weapons'][0]
  "Devices"       → col_left_x, slot_rows['Ground Devices'][0]

Geometry derivation (no GT, no annotations.json):
  row_pitch  = median of consecutive label-cy gaps in the EQ column
  cell_h     ≈ row_pitch × 0.60        (empirical, 12 GT screens)
  cell_w     ≈ cell_h    × 0.78        (ground icons taller than wide)
  km_pitch   ≈ cell_w    + 3           (KM cells packed tight)
  weapons_stack_pitch ≈ cell_h + 7     (2 weapon cells in left col)

Cell projection (one cy per icon, NOT one cy per label):
  label_cy → row_cy ≈ label_cy + cell_h × 0.50  (label sits above the icon)

This module is pure detection — no annotations.json access. Caller passes only
an image. Returns None when OCR yields no usable EQ labels.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import cv2
import numpy as np

# Lazy import to avoid circular dependency
_TEXT_EXTRACTOR = None


def _get_easyocr_reader():
    global _TEXT_EXTRACTOR
    if _TEXT_EXTRACTOR is None:
        from warp.recognition.text_extractor import TextExtractor
        _TEXT_EXTRACTOR = TextExtractor()
    return _TEXT_EXTRACTOR._get_ocr()


# ----------------------------------------------------------------------------
# Empirical constants (derived from 12 GT-annotated ground screens)
# ----------------------------------------------------------------------------

CELL_H_RATIO            = 0.615  # cell_h / row_pitch — empirically tuned
                                 # so projection covers ~1 px beyond the
                                 # visible icon edge on all slots.
CELL_W_RATIO            = 0.78   # cell_w / cell_h  (ground icons are tall)
KM_PITCH_OFFSET         = 3      # km_pitch = cell_w + 3
WEAPONS_STACK_OFFSET    = 7      # weapons_stack_pitch = cell_h + 7

# Slot column nudge (all slots EXCEPT Ground Devices): OCR label x sits ~1 px
# left of the actual icon column, so we add a small right-shift. Expressed
# as a fraction of cell_w so it scales with UI resolution.
SLOT_X_NUDGE_RATIO      = 0.025  # ≈ 1 px at cell_w ~40

# Devices column inset relative to the Body/EV Suit column axis:
#   L col shifts +cell_w × ratio right of col_left_x,
#   R col shifts -cell_w × ratio left of (col_right_x + cell_w),
#   row cy shifts up by cell_h × ratio.
# Empirical observation from GT — STO ground UI places Device icons slightly
# inset and lifted from the rest of EQ. Expressed as a fraction of cell size
# (same pattern as space DX_RATIO) so the inset scales with UI resolution.
DEVICES_INSET_X_RATIO   = 0.05   # ≈ 2 px at cell_w ~40
DEVICES_INSET_Y_RATIO   = 0.025  # ≈ 1 px at cell_h ~44

# Distance between left-column x and right-column x (Body Armor → EV Suit
# axis), expressed as a multiple of cell_w. Empirical median from GT:
#   sweelinck-ground.png  46 / 35 = 1.314
#   image27.png           44 / 35 = 1.257
# Median ≈ 1.28, i.e. one full cell + ~28% of cell_w gap between columns.
# Used as fallback when the OCR merged "Body EV Suit" into one token and we
# cannot read a real EV Suit x0. With the old value of 1.0 the right column
# touched / overlapped the left column.
BODY_EV_PITCH_RATIO     = 1.28
# row_cy = label_cy + row_pitch * LABEL_TO_ROW_RATIO. Empirical median 0.46
# across 12 GT ground screens — the label sits roughly half a row-pitch
# above the icon center.
LABEL_TO_ROW_RATIO      = 0.46

KM_MAX_CELLS            = 6      # STO max kit modules
COL_LEFT_MIN_X          = 200    # filter stat-bar "Shields:" (x near 30-100)

# Slot key on the SETS side ↔ canonical row index ↔ OCR keyword(s)
SLOT_KIT_MODULES        = 'Kit Modules'
SLOT_KIT                = 'Kit'
SLOT_BODY_ARMOR         = 'Body Armor'
SLOT_EV_SUIT            = 'EV Suit'
SLOT_PERSONAL_SHIELD    = 'Personal Shield'
SLOT_WEAPONS            = 'Weapons'
SLOT_GROUND_DEVICES     = 'Ground Devices'

# OCR keyword → canonical slot (lowercased single-token match)
OCR_KEYWORD_TO_SLOT = {
    'kit modules':   SLOT_KIT_MODULES,
    'kit':           SLOT_KIT,
    'body':          SLOT_BODY_ARMOR,
    'ev suit':       SLOT_EV_SUIT,
    'shields':       SLOT_PERSONAL_SHIELD,   # without trailing colon
    'weapons':       SLOT_WEAPONS,
    'devices':       SLOT_GROUND_DEVICES,
}

# Canonical row order (visual top→bottom).
ROW_ORDER = [
    SLOT_KIT_MODULES,
    SLOT_KIT,
    SLOT_BODY_ARMOR,        # same row as EV Suit, listed first
    SLOT_PERSONAL_SHIELD,
    SLOT_WEAPONS,
    SLOT_GROUND_DEVICES,
]


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------

@dataclass
class GroundEQGeometry:
    row_pitch:            int
    cell_h:               float
    cell_w:               float
    km_pitch:             float
    weapons_stack_pitch:  float

    km_label_cy:          Optional[int]
    km_label_x0:          Optional[int]

    col_left_x:           int
    col_right_x:          Optional[int]

    # {slot_name: label_cy} — only slots whose OCR label was found
    slot_label_cys:       dict = field(default_factory=dict)

    mode:                 str  = 'OCR_FULL'


# ----------------------------------------------------------------------------
# OCR helpers
# ----------------------------------------------------------------------------

def _run_ocr(img: np.ndarray) -> list[dict]:
    reader = _get_easyocr_reader()
    if reader is None:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        raw = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception:
        return []
    out = []
    for box, text, conf in raw:
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            'text': text,
            'low':  text.lower().strip(),
            'conf': float(conf),
            'cx':   int(sum(xs) / 4),
            'cy':   int(sum(ys) / 4),
            'x0':   int(min(xs)),
            'y0':   int(min(ys)),
            'x1':   int(max(xs)),
            'y1':   int(max(ys)),
            'w':    int(max(xs) - min(xs)),
            'h':    int(max(ys) - min(ys)),
        })
    return out


def _match_label(tok: dict) -> Optional[str]:
    """Return canonical slot name if token text matches an EQ label.

    Filters:
      * "Shields:" (trailing colon) is a HUD stat label — reject.
      * "shields" with x0 < COL_LEFT_MIN_X is the HUD stat — reject.
    """
    low = tok['low'].rstrip(':').strip()
    # Reject HUD stat lines that begin with "shields:"
    if tok['low'].endswith(':') and low == 'shields':
        return None
    if low in OCR_KEYWORD_TO_SLOT:
        # "shields" must sit in the EQ column, not the stat bar
        if low == 'shields' and tok['x0'] < COL_LEFT_MIN_X:
            return None
        return OCR_KEYWORD_TO_SLOT[low]
    # OCR sometimes merges "Body" + "EV Suit" into one token.
    if 'body' in low and ('ev' in low or 'suit' in low):
        # We can't reliably split a merged token here; flag as Body only.
        return SLOT_BODY_ARMOR
    return None


# ----------------------------------------------------------------------------
# Multi-candidate hit collection + scoring
# ----------------------------------------------------------------------------

# Canonical row index (1..5) for left-column slots — used to weight gaps by
# how many physical rows separate two labels.
_CANONICAL_ROW_IDX = {
    SLOT_KIT:             1,
    SLOT_BODY_ARMOR:      2,
    SLOT_PERSONAL_SHIELD: 3,
    SLOT_WEAPONS:         4,
    SLOT_GROUND_DEVICES:  5,
}


def _collect_candidates(ocr: list[dict]) -> dict[str, list[dict]]:
    """Return ALL OCR tokens per canonical slot (multi-candidate).

    Merged 'Body EV Suit' tokens emit two virtual hits — one for Body Armor
    (at original x0) and one for EV Suit (at x0 + w × 0.4, marked `_split`).
    """
    out: dict[str, list[dict]] = {}
    for tok in ocr:
        low = tok['low'].rstrip(':').strip()
        if tok['low'].endswith(':') and low == 'shields':
            continue
        if low in OCR_KEYWORD_TO_SLOT:
            if low == 'shields' and tok['x0'] < COL_LEFT_MIN_X:
                continue
            out.setdefault(OCR_KEYWORD_TO_SLOT[low], []).append(tok)
            continue
        if 'body' in low and ('ev' in low or 'suit' in low):
            out.setdefault(SLOT_BODY_ARMOR, []).append(tok)
            ev_tok = dict(tok)
            ev_tok['x0'] = int(tok['x0'] + tok['w'] * 0.4)
            ev_tok['_split'] = True
            out.setdefault(SLOT_EV_SUIT, []).append(ev_tok)
    return out


def _score_combo(combo: dict[str, dict]) -> tuple[float, dict]:
    """Score a slot→token assignment. Returns (score, derived) or (-inf, {}).

    Higher score = better. Rejection conditions:
      - fewer than 2 left-column hits (cannot derive row_pitch)
      - non-monotonic cy ordering vs canonical top→bottom row index
    """
    left = [(s, combo[s]) for s in _CANONICAL_ROW_IDX if s in combo]
    if len(left) < 2:
        return float('-inf'), {}

    sorted_by_cy = sorted(left, key=lambda x: x[1]['cy'])
    order = [_CANONICAL_ROW_IDX[s] for s, _ in sorted_by_cy]
    if any(a >= b for a, b in zip(order, order[1:])):
        return float('-inf'), {}

    # Per-row pitch: skip pairs where prev is Weapons (stack inflates gap).
    per_row_gaps: list[float] = []
    for (ps, pt), (cs, ct) in zip(sorted_by_cy, sorted_by_cy[1:]):
        if ps == SLOT_WEAPONS:
            continue
        rows = _CANONICAL_ROW_IDX[cs] - _CANONICAL_ROW_IDX[ps]
        gap = ct['cy'] - pt['cy']
        if rows <= 0 or gap <= 0:
            continue
        per_row_gaps.append(gap / rows)
    if not per_row_gaps:
        return float('-inf'), {}

    mean_pitch = sum(per_row_gaps) / len(per_row_gaps)
    if mean_pitch <= 0:
        return float('-inf'), {}
    if len(per_row_gaps) > 1:
        pitch_std = (sum((g - mean_pitch) ** 2 for g in per_row_gaps)
                     / len(per_row_gaps)) ** 0.5
    else:
        pitch_std = 0.0

    # col_left consistency: x0 stdev across real (non-split) left-col hits.
    x0s = [t['x0'] for s, t in left if not t.get('_split')]
    if len(x0s) >= 2:
        mean_x = sum(x0s) / len(x0s)
        x0_std = (sum((x - mean_x) ** 2 for x in x0s) / len(x0s)) ** 0.5
        col_left_x = int(round(mean_x))
    elif x0s:
        x0_std = 0.0
        col_left_x = x0s[0]
    else:
        return float('-inf'), {}

    coverage = (len(left)
                + (1 if SLOT_EV_SUIT in combo else 0)
                + (1 if SLOT_KIT_MODULES in combo else 0))
    has_real_ev = (SLOT_EV_SUIT in combo
                   and not combo[SLOT_EV_SUIT].get('_split'))
    conf_sum = sum(t['conf'] for _, t in left)

    # Normalize stdevs by mean_pitch so the score is resolution-agnostic.
    norm_pitch_std = pitch_std / mean_pitch
    norm_x0_std = x0_std / mean_pitch

    score = (
        10.0 * coverage
        - 20.0 * norm_pitch_std
        - 10.0 * norm_x0_std
        + 5.0 * (1 if has_real_ev else 0)
        + 1.0 * conf_sum
    )
    return score, {'row_pitch': int(round(mean_pitch)),
                   'col_left_x': col_left_x}


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def detect_ground_eq_geometry(img: np.ndarray) -> Optional[GroundEQGeometry]:
    """Full pipeline. Returns None if OCR finds no usable EQ labels.

    Uses multi-candidate scoring: each canonical slot may have multiple OCR
    tokens (e.g. duplicate "Body" hits from HUD/tooltips), and the optimal
    layout is selected by enumerating combinations and ranking on monotonic
    cy ordering + per-row gap uniformity + col_left consistency. This is
    more robust than picking max-conf per slot, which fails when a wrong
    OCR token (high conf, wrong position) sits below the real label.
    """
    if img is None or img.size == 0:
        return None
    ocr = _run_ocr(img)
    if not ocr:
        return None

    candidates = _collect_candidates(ocr)
    if not candidates:
        return None

    # Cap per slot to top-3 by confidence to bound enumeration:
    # 7 slots × (1 skip + 3 cands) = 4^7 = 16384 combos worst case.
    slot_keys = list(candidates.keys())
    options: list[list[Optional[dict]]] = []
    for s in slot_keys:
        tops = sorted(candidates[s], key=lambda t: -t['conf'])[:3]
        options.append([None] + tops)

    best_score = float('-inf')
    best_combo: dict[str, dict] = {}
    best_extras: dict = {}
    for choice in itertools.product(*options):
        combo = {s: t for s, t in zip(slot_keys, choice) if t is not None}
        score, extras = _score_combo(combo)
        if score > best_score:
            best_score = score
            best_combo = combo
            best_extras = extras

    if not best_combo or best_score == float('-inf'):
        return None

    hits = best_combo
    row_pitch = best_extras['row_pitch']
    col_left_x = best_extras['col_left_x']

    cell_h = row_pitch * CELL_H_RATIO
    cell_w = cell_h * CELL_W_RATIO
    km_pitch = cell_w + KM_PITCH_OFFSET
    weapons_stack_pitch = cell_h + WEAPONS_STACK_OFFSET

    # col_right_x — prefer real EV Suit label x0; for synthetic-split EV Suit
    # estimate from col_left_x + cell_w × BODY_EV_PITCH_RATIO (one full cell
    # plus the inter-column gap observed in GT).
    col_right_x: Optional[int] = None
    if SLOT_EV_SUIT in hits:
        ev = hits[SLOT_EV_SUIT]
        if ev.get('_split'):
            col_right_x = int(round(col_left_x + cell_w * BODY_EV_PITCH_RATIO))
        else:
            col_right_x = ev['x0']

    km_label = hits.get(SLOT_KIT_MODULES)
    km_label_cy = km_label['cy'] if km_label else None
    km_label_x0 = km_label['x0'] if km_label else None

    # Real slot_label_cys: keep all chosen hits (including synthetic EV Suit
    # so it projects a right column for Devices/Body row pairing).
    slot_label_cys = {s: h['cy'] for s, h in hits.items()}

    return GroundEQGeometry(
        row_pitch=row_pitch,
        cell_h=cell_h,
        cell_w=cell_w,
        km_pitch=km_pitch,
        weapons_stack_pitch=weapons_stack_pitch,
        km_label_cy=km_label_cy,
        km_label_x0=km_label_x0,
        col_left_x=col_left_x,
        col_right_x=col_right_x,
        slot_label_cys=slot_label_cys,
        mode='OCR_FULL' if len(hits) >= 6 else 'OCR_PARTIAL',
    )


# ----------------------------------------------------------------------------
# Cell projection
# ----------------------------------------------------------------------------

def _row_cy(label_cy: int, row_pitch: int) -> int:
    """Center-y of an icon row given the label cy above it."""
    return int(round(label_cy + row_pitch * LABEL_TO_ROW_RATIO))


def project_cells(geom: GroundEQGeometry, img_w: int, img_h: int
                  ) -> dict[str, list[tuple[int, int, int, int]]]:
    """Project bboxes (x, y, w, h) for each slot from geometry.

    Returned bboxes use TOP-LEFT coords (matches annotations.json convention).
    Cells whose projected center falls outside the image are dropped.
    """
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    cw = int(round(geom.cell_w))
    ch = int(round(geom.cell_h))

    def _push(slot: str, cx: float, cy: float) -> None:
        x = int(round(cx - cw / 2))
        y = int(round(cy - ch / 2))
        if (cx < 0 or cy < 0 or cx >= img_w or cy >= img_h or
                x + cw <= 0 or y + ch <= 0 or x >= img_w or y >= img_h):
            return
        out.setdefault(slot, []).append((x, y, cw, ch))

    # Per-slot x-nudge (all slots except Devices): OCR label x sits ~1 px
    # left of the actual icon column.
    slot_nudge = geom.cell_w * SLOT_X_NUDGE_RATIO

    # Kit Modules — 1 row, up to KM_MAX_CELLS, stop before col_left_x.
    # Boundary rule: keep cells whose CENTER sits strictly left of col_left_x
    # (cell may extend a few px into col_left — STO packs them tight).
    if geom.km_label_cy is not None and geom.km_label_x0 is not None:
        km_row_cy = _row_cy(geom.km_label_cy, geom.row_pitch)
        # First-cell cx aligned with KM label x0 + cell_w/2 (label sits over
        # the icon block left edge).
        cx0 = geom.km_label_x0 + slot_nudge + cw / 2
        for j in range(KM_MAX_CELLS):
            cx = cx0 + j * geom.km_pitch
            if cx >= geom.col_left_x:
                break
            _push(SLOT_KIT_MODULES, cx, km_row_cy)

    col_left_cx = geom.col_left_x + slot_nudge + cw / 2
    col_right_cx = (geom.col_right_x + slot_nudge + cw / 2
                    if geom.col_right_x is not None else None)

    # Kit
    if SLOT_KIT in geom.slot_label_cys:
        cy = _row_cy(geom.slot_label_cys[SLOT_KIT], geom.row_pitch)
        _push(SLOT_KIT, col_left_cx, cy)

    # Body Armor (left col) + EV Suit (right col, same y as Body)
    if SLOT_BODY_ARMOR in geom.slot_label_cys:
        cy = _row_cy(geom.slot_label_cys[SLOT_BODY_ARMOR], geom.row_pitch)
        _push(SLOT_BODY_ARMOR, col_left_cx, cy)
        if col_right_cx is not None:
            _push(SLOT_EV_SUIT, col_right_cx, cy)

    # Personal Shield
    if SLOT_PERSONAL_SHIELD in geom.slot_label_cys:
        cy = _row_cy(geom.slot_label_cys[SLOT_PERSONAL_SHIELD], geom.row_pitch)
        _push(SLOT_PERSONAL_SHIELD, col_left_cx, cy)

    # Weapons — 2 cells stacked at col_left
    if SLOT_WEAPONS in geom.slot_label_cys:
        cy0 = _row_cy(geom.slot_label_cys[SLOT_WEAPONS], geom.row_pitch)
        _push(SLOT_WEAPONS, col_left_cx, cy0)
        _push(SLOT_WEAPONS, col_left_cx, cy0 + geom.weapons_stack_pitch)

    # Ground Devices — 2 cols × multiple rows.  We don't know the row count
    # from OCR alone; project up to 3 rows × 2 cols. Out-of-image cells are
    # dropped by _push.
    # Inset axis: L col shifts +cell_w × ratio right of col_left_x;
    # R col shifts -cell_w × ratio left; cy shifts up by cell_h × ratio.
    if SLOT_GROUND_DEVICES in geom.slot_label_cys:
        cy0 = _row_cy(geom.slot_label_cys[SLOT_GROUND_DEVICES], geom.row_pitch)
        device_row_pitch = geom.cell_h + WEAPONS_STACK_OFFSET
        inset_x = geom.cell_w * DEVICES_INSET_X_RATIO
        inset_y = geom.cell_h * DEVICES_INSET_Y_RATIO
        devices_left_cx = geom.col_left_x + inset_x + cw / 2
        devices_right_cx = (geom.col_right_x - inset_x + cw / 2
                            if geom.col_right_x is not None else None)
        for r in range(3):
            cy = cy0 + r * device_row_pitch - inset_y
            _push(SLOT_GROUND_DEVICES, devices_left_cx, cy)
            if devices_right_cx is not None:
                _push(SLOT_GROUND_DEVICES, devices_right_cx, cy)

    return out
