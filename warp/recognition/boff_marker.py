"""
BOFF profession-marker detector — production module.

Locates a BOFF panel in a screenshot by detecting the small profession-
coloured badges that sit on the LEFT of each seat name bar (which itself
sits BELOW the seat's 4 ability icons). From the badge grid we recover:

  - the canonical 3+2 (or 2+2 / 2+1) seat layout,
  - per-seat ability slot bboxes via bible-driven projection.

This module is detection-only. It MUST NOT read `annotations.json` or
any user ground truth (CORE RULE: WARP detection must derive its output
from pixels).

Public API:
    detect_panel(img) -> Optional[PanelResult]

PanelResult is a TypedDict-like dict:
    {
      'col_a':   list[(x, y, w, h, code, spec_code | None)],  # left column
      'col_b':   list[(x, y, w, h, code, spec_code | None)],  # right column
      'score':   float,                            # RANSAC score
      'seats':   list[(side, mx, my, mw, mh, seat_code, spec_code | None)],
      'slots':   list[(seat_idx, slot_idx, x, y, w, h, seat_code)],
    }

Algorithm prototype + 2026-04-30 baseline (36 GT screens):
  - Panel anchor:  100% (36/36)
  - Seat hit:      100% (177/177)
  - Slot IoU≥.30:  99.9%
  - Slot IoU≥.50:  99.9%
  - Slot IoU≥.70:  96.0%

Algorithm reference: docs/BOFF_DETECTION.md.
Diagnostic prototype:  tests/diag_boff_markers.py.
"""
from __future__ import annotations

import statistics as st
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

# Seat-type bands (main zone of the marker, 75-95% of bar width).
# Sampled across 25+ GT seats. Used as DETECTION seeds.
MAIN_BANDS: list[tuple[str, int, int, int, int, int, int, str]] = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, code
    ('TAC',   0,   6,   125, 255,  85, 255, 'T'),  # red
    ('TAC', 174, 180,   125, 255,  85, 255, 'T'),  # red wrap
    ('ENG',  18,  30,   100, 220, 160, 255, 'E'),  # saturated gold
    ('SCI', 102, 114,   160, 255, 140, 255, 'S'),  # blue
    ('UNI',  18,  30,    25,  95, 195, 255, 'U'),  # pale cream
]

# Spec-stripe bands (narrow right edge, 5-25% of bar width). NOT used
# for detection — applied post-hoc to label each detected seat with
# its specialization.
STRIPE_BANDS: list[tuple[str, int, int, int, int, int, int, str]] = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, spec_code
    # Spec codes are human-readable abbreviations so seat keys like
    # `Boff Seat L[U+Plt]_96` are intuitive in logs and review UIs
    # (the single-letter codes O/P/Y/C/L did NOT match first letters
    # and confused everyone).
    ('CMD',  12,  19,    80, 200, 130, 200, 'Cmd'),  # Command — orange
    ('INT', 120, 135,   130, 200, 150, 220, 'Int'),  # Intelligence — purple
    ('TMP',  25,  35,   140, 255, 215, 255, 'Tem'),  # Temporal — bright gold
    ('PIL',  86, 100,    80, 140, 215, 255, 'Plt'),  # Pilot — light cyan
    ('MW',   32,  44,   220, 255, 195, 255, 'MW'),   # Miracle Worker — lime
]


# ---------------------------------------------------------------------------
# Bible — measured on Stations.png ("idealny rozkład").
# All values are panel-internal pixels at the reference scale where one
# full marker bar is 29 pixels wide. At any other UI scale we rescale
# everything by `k = detected_marker_w / _BIBLE_MARKER_W`.
# ---------------------------------------------------------------------------

_BIBLE_MARKER_W   = 29
_BIBLE_SLOT_W     = 29
_BIBLE_SLOT_H     = 37
_BIBLE_GAP_FIRST  = 3      # marker_right_edge → slot1_left_edge (X)
_BIBLE_GAP_SLOT   = 2      # slot_right_edge   → next_slot_left   (X)
_BIBLE_STRIDE_X   = _BIBLE_SLOT_W + _BIBLE_GAP_SLOT   # = 31


# ---------------------------------------------------------------------------
# Code → human label maps (for callers that want pretty names)
# ---------------------------------------------------------------------------

SEAT_CODE_LABEL = {
    'T': 'Tactical',
    'E': 'Engineering',
    'S': 'Science',
    'U': 'Universal',
}
SPEC_CODE_LABEL = {
    'Cmd': 'Command',
    'Int': 'Intelligence',
    'Tem': 'Temporal',
    'Plt': 'Pilot',
    'MW':  'Miracle Worker',
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def estimate_icon_dims(img: np.ndarray) -> tuple[int, int]:
    """Initial wide range for the marker size — refined later from
    detected marker sizes."""
    h, _w = img.shape[:2]
    icon_h = max(int(round(h * 0.045)), 24)
    icon_w = max(int(round(icon_h * 0.78)), 22)
    return icon_w, icon_h


def _refine_dims_from_markers(markers, icon_w, icon_h):
    if len(markers) < 4:
        return icon_w, icon_h
    hs = sorted(m[3] for m in markers)
    ws = sorted(m[2] for m in markers)
    med_h = hs[len(hs) // 2]
    med_w = ws[len(ws) // 2]
    if 16 <= med_h <= 90 and 4 <= med_w <= 90:
        return max(med_w, 16), max(med_h, 20)
    return icon_w, icon_h


def _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    return ((H >= h_lo) & (H <= h_hi)
            & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255


def _merge_close_bboxes(boxes, gap_x, overlap_y_frac=0.6):
    """Merge bboxes whose x-gap is ≤ gap_x and whose y-overlap is at
    least overlap_y_frac of the shorter bbox height."""
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for b in boxes[1:]:
        m = merged[-1]
        gx = b[0] - (m[0] + m[2])
        oy = max(0, min(m[1] + m[3], b[1] + b[3]) - max(m[1], b[1]))
        short = max(1, min(m[3], b[3]))
        if gx <= gap_x and (oy / short) >= overlap_y_frac:
            x0 = min(m[0], b[0])
            y0 = min(m[1], b[1])
            x1 = max(m[0] + m[2], b[0] + b[2])
            y1 = max(m[1] + m[3], b[1] + b[3])
            m[0] = x0; m[1] = y0
            m[2] = x1 - x0; m[3] = y1 - y0
            m[4] = m[4] + b[4]
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def detect_markers(img: np.ndarray, icon_w: int, icon_h: int):
    """Detect seat-type markers. Returns list of (x, y, w, h, code).

    Per band:
      1. HSV mask + small horizontal CLOSE.
      2. Connected components on BOTH raw and CLOSE-d masks (dual-mask CC).
      3. _merge_close_bboxes() to glue same-colour fragments.
      4. Size + aspect + fill-density + uniformity + Canny-edge filters.
      5. Cross-band IoU dedupe + size-outlier cull.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    abs_min_w, abs_min_h = 10, 12
    abs_max_w, abs_max_h = 90, 90
    min_w = max(abs_min_w, int(icon_w * 0.45))
    # Empirical max marker_w in GT baseline = 43 px; UI scale is user-set
    # and not proportional to resolution, so cap must accommodate the full
    # observed range regardless of icon_w estimate.
    max_w = min(abs_max_w, max(int(icon_w * 1.6), 44))
    min_h = max(abs_min_h, int(icon_h * 0.45))
    max_h = min(abs_max_h, max(int(icon_h * 1.35), 44))
    ar_min, ar_max = 0.30, 1.8
    fill_min = 0.70

    kx = max(3, int(round(icon_w * 0.22)))
    ky = max(2, icon_h // 12)

    uni_v_max = 28
    uni_h_max = 6
    edge_max = 0.07
    edge_inset = 2
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

    # Merge bands sharing the same name (e.g. red TAC wraps 0/180).
    by_name: dict[str, dict] = {}
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in MAIN_BANDS:
        by_name.setdefault(
            name, {'code': code, 'ranges': []},
        )['ranges'].append((h_lo, h_hi, s_lo, s_hi, v_lo, v_hi))

    out: list[tuple[int, int, int, int, str]] = []
    seen_rects: list[tuple[int, int, int, int]] = []
    for _name, info in by_name.items():
        code = info['code']
        m = None
        for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in info['ranges']:
            part = _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
            m = part if m is None else cv2.bitwise_or(m, part)

        m_closed = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
        )
        gap_x = max(2, int(round(icon_w * 0.35)))
        merged: list[tuple] = []
        for src in (m, m_closed):
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                src, connectivity=8)
            raw = [tuple(int(v) for v in stats[i, [0, 1, 2, 3, 4]])
                   for i in range(1, n)]
            merged.extend(_merge_close_bboxes(
                raw, gap_x=gap_x, overlap_y_frac=0.55))

        for x, y, w, h, area in merged:
            if w < min_w or w > max_w:
                continue
            if h < min_h or h > max_h:
                continue
            ar = w / max(h, 1)
            if ar < ar_min or ar > ar_max:
                continue
            sel = m[y:y + h, x:x + w] > 0
            if sel.sum() < 20:
                continue
            crop_v = hsv[y:y + h, x:x + w, 2][sel]
            crop_h = hsv[y:y + h, x:x + w, 0][sel]
            v_std = float(np.std(crop_v))
            if v_std > uni_v_max:
                continue
            h_std = min(
                float(np.std(crop_h)),
                float(np.std((crop_h.astype(np.int32) + 90) % 180)),
            )
            if h_std > uni_h_max:
                continue
            # Strong-uniformity bypass for fill/edge: a flat colour bar
            # (low v_std, low h_std) IS a marker. Spec stripes (e.g. MW
            # lime on Engineering) break the main-zone CC short and add
            # a transition that bumps Canny edge_frac just past the
            # baseline — relax both thresholds when uniformity is high.
            strong_uniform = v_std <= 20 and h_std <= 4
            fill_thr = 0.60 if strong_uniform else fill_min
            edge_thr = 0.12 if strong_uniform else edge_max
            if area < (w * h) * fill_thr:
                continue
            ix0 = x + edge_inset; iy0 = y + edge_inset
            ix1 = x + w - edge_inset; iy1 = y + h - edge_inset
            if ix1 > ix0 and iy1 > iy0:
                edge_crop = edges[iy0:iy1, ix0:ix1]
                edge_frac = float(edge_crop.sum() / 255.0) / (
                    (ix1 - ix0) * (iy1 - iy0))
                if edge_frac > edge_thr:
                    continue
            dup = False
            for (px, py, pw, ph) in seen_rects:
                if x < px + pw and px < x + w and y < py + ph and py < y + h:
                    ix = min(x + w, px + pw) - max(x, px)
                    iy = min(y + h, py + ph) - max(y, py)
                    iou = (ix * iy) / max(w * h + pw * ph - ix * iy, 1)
                    if iou > 0.4:
                        dup = True
                        break
            if dup:
                continue
            seen_rects.append((x, y, w, h))
            out.append((int(x), int(y), int(w), int(h), code))

    if len(out) >= 4:
        med_w = sorted(m[2] for m in out)[len(out) // 2]
        med_h = sorted(m[3] for m in out)[len(out) // 2]
        out = [m for m in out
               if m[2] >= 0.65 * med_w and m[3] >= 0.65 * med_h]
    return out


# ---------------------------------------------------------------------------
# Spec-stripe / full bar
# ---------------------------------------------------------------------------

def classify_stripe(hsv, marker, _icon_w_unused=None):
    """Identify the specialization stripe on the right edge of a marker.
    Returns (spec_code, score) or (None, score)."""
    x, y, w, h, _ = marker
    sx0 = x + w - max(2, int(w * 0.10))
    sx1 = x + w + max(6, int(w * 0.40))
    sy0 = y + max(0, int(h * 0.10))
    sy1 = y + h - max(0, int(h * 0.10))
    H, W = hsv.shape[:2]
    sx0 = max(0, sx0); sx1 = min(W, sx1)
    sy0 = max(0, sy0); sy1 = min(H, sy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return (None, 0.0)
    crop = hsv[sy0:sy1, sx0:sx1]
    total = crop.shape[0] * crop.shape[1]
    if total <= 0:
        return (None, 0.0)
    best_code, best_count = None, 0
    for _, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in STRIPE_BANDS:
        m = ((crop[:, :, 0] >= h_lo) & (crop[:, :, 0] <= h_hi)
             & (crop[:, :, 1] >= s_lo) & (crop[:, :, 1] <= s_hi)
             & (crop[:, :, 2] >= v_lo) & (crop[:, :, 2] <= v_hi))
        c = int(m.sum())
        if c > best_count:
            best_count = c; best_code = code
    score = best_count / total
    if score < 0.15:
        return (None, score)
    return (best_code, score)


def full_bar_extent(hsv, marker):
    """Find the rightmost x where the colour bar (main zone OR spec
    stripe) still has a strong column-fill, starting from the marker
    bbox. Returns (full_w_including_stripe, has_spec, stripe_width).

    Two-phase walk past the main zone:
      Phase 1: skip ≤6 dim cols (the gap between main and stripe).
      Phase 2: extend through contiguous bright cols (the stripe itself).

    This stops cleanly at the END of the spec stripe — beyond which
    there's another gap and then the seat name bar (same hue as marker)
    that a naive "rightmost ≥35%" rule would grab.
    """
    x, y, w, h, _code = marker
    spec, _score = classify_stripe(hsv, marker)
    if spec is None:
        return (w, False, 0)
    H_im, W_im = hsv.shape[:2]
    look_ahead = max(4, int(round(w * 0.35)))
    sx0 = max(0, x)
    sx1 = min(W_im, x + w + look_ahead)
    sy0 = max(0, y + max(0, int(h * 0.15)))
    sy1 = min(H_im, y + h - max(0, int(h * 0.15)))
    if sy1 <= sy0 or sx1 <= sx0:
        return (w, False, 0)
    crop = hsv[sy0:sy1, sx0:sx1]
    H = crop[:, :, 0]; S = crop[:, :, 1]; V = crop[:, :, 2]
    main_m = np.zeros(crop.shape[:2], dtype=bool)
    for _n, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _c in MAIN_BANDS:
        main_m |= ((H >= h_lo) & (H <= h_hi)
                   & (S >= s_lo) & (S <= s_hi)
                   & (V >= v_lo) & (V <= v_hi))
    stripe_m = np.zeros(crop.shape[:2], dtype=bool)
    for _n, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _c in STRIPE_BANDS:
        stripe_m |= ((H >= h_lo) & (H <= h_hi)
                     & (S >= s_lo) & (S <= s_hi)
                     & (V >= v_lo) & (V <= v_hi))
    full_m = main_m | stripe_m
    col_frac = full_m.mean(axis=0)
    main_end_in_crop = (x + w) - sx0
    last_col = main_end_in_crop - 1
    n_cols = col_frac.shape[0]
    ci = main_end_in_crop
    while (ci < n_cols
           and col_frac[ci] < 0.35
           and ci - main_end_in_crop < 6):
        ci += 1
    while ci < n_cols and col_frac[ci] >= 0.35:
        last_col = ci
        ci += 1
    full_w = max(w, last_col + 1)
    has_spec = full_w > w + 1
    stripe_w = full_w - w if has_spec else 0
    return (full_w, has_spec, stripe_w)


def annotate_specs(img, markers):
    """Return [(x, y, w, h, seat_code, spec_code_or_None, score)]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for x, y, w, h, code in markers:
        spec, score = classify_stripe(hsv, (x, y, w, h, code))
        out.append((x, y, w, h, code, spec, round(score, 3)))
    return out


# ---------------------------------------------------------------------------
# Panel selection — RANSAC-style 3+2 grid search
# ---------------------------------------------------------------------------

def best_panel(markers, icon_w, icon_h):
    """Pick the best 2-column anchor grid. Returns (col_a, col_b, score)
    or None if no plausible panel was found."""
    if len(markers) < 4:
        return None

    cx_arr = [m[0] + m[2] / 2 for m in markers]
    cy_arr = [m[1] + m[3] / 2 for m in markers]
    n = len(markers)

    pitch_y_candidates = [
        max(icon_h * 1.6, 24),
        max(icon_h * 2.0, 30),
        max(icon_h * 2.4, 36),
        max(icon_h * 3.0, 50),
        max(icon_h * 3.6, 70),
    ]
    x_tol = max(icon_w * 0.7, 8)

    best = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(cy_arr[i] - cy_arr[j]) > max(icon_h * 0.7, 12):
                continue
            xa, xb = cx_arr[i], cx_arr[j]
            if xb <= xa + 3 * icon_w:
                continue
            if xb - xa > 9 * icon_w:
                continue

            for pitch_y in pitch_y_candidates:
                y_tol = pitch_y * 0.30
                anchor_y = (cy_arr[i] + cy_arr[j]) / 2
                col_a, col_b = [], []
                for k in range(n):
                    cx, cy = cx_arr[k], cy_arr[k]
                    r = round((cy - anchor_y) / pitch_y)
                    if abs(r) > 3:
                        continue
                    expected_y = anchor_y + r * pitch_y
                    if abs(cy - expected_y) > y_tol:
                        continue
                    if abs(cx - xa) <= x_tol:
                        col_a.append((r, markers[k]))
                    elif abs(cx - xb) <= x_tol:
                        col_b.append((r, markers[k]))

                def _keep_one(col):
                    seen: dict[int, tuple] = {}
                    for r, m in col:
                        if r not in seen:
                            seen[r] = m
                    return [m for _, m in sorted(seen.items())]
                col_a = _keep_one(col_a)
                col_b = _keep_one(col_b)

                if len(col_a) < 2 or len(col_b) < 1:
                    continue
                if len(col_a) > 3 or len(col_b) > 3:
                    continue

                aligned = 0
                yas = [m[1] + m[3] / 2 for m in col_a]
                ybs = [m[1] + m[3] / 2 for m in col_b]
                for yb in ybs:
                    if any(abs(yb - ya) < y_tol for ya in yas):
                        aligned += 1
                if aligned < 1:
                    continue

                n_total = len(col_a) + len(col_b)
                canon_table = {5: 1.5, 4: 1.0, 3: 0.4, 2: 0.0, 6: 0.6}
                canon = canon_table.get(n_total, 0.0)

                codes = {m[4] for m in col_a} | {m[4] for m in col_b}
                div = 0.6 if len(codes) >= 2 else -0.8
                layout = 0.3 if len(col_a) >= len(col_b) else 0.0

                pitch_score = 0.0
                if len(col_a) >= 2:
                    ys = sorted(yas)
                    diffs = [ys[k + 1] - ys[k] for k in range(len(ys) - 1)]
                    if len(diffs) >= 2:
                        m = st.mean(diffs)
                        d = st.stdev(diffs)
                        pitch_score = 0.4 * (1.0 - min(d / max(m, 1), 1.0))

                score = (0.6 * aligned + canon + div + layout + pitch_score)
                if best is None or score > best[2]:
                    best = (col_a, col_b, score)
    return best


# ---------------------------------------------------------------------------
# Bible-driven slot projection
# ---------------------------------------------------------------------------

def project_seat_slots(panel, n_abilities=4, hsv=None):
    """Project 4 ability-icon bboxes per detected seat marker.

    Geometry comes from the bible (panel-internal pixels). Detection
    contributes a single scale factor `k = full_marker_w / 29`, so
    every grid distance scales coherently — slot 4 cannot drift away
    from slot 1 just because `med_marker_w` was a pixel low.

    Float-domain math: bible distances are kept as floats, the only
    rounding happens once per slot at the final pixel position. This
    eliminates cumulative error from `i * round(stride)` and the
    visual gap alternation that comes from independently rounding
    slot_w vs stride.

    Returns list of (seat_idx, slot_idx, x, y, w, h, seat_code).
    """
    if panel is None:
        return []
    a, b, _score = panel
    all_m = a + b
    if not all_m:
        return []

    if hsv is not None:
        widths = sorted(full_bar_extent(hsv, m)[0] for m in all_m)
    else:
        widths = sorted(m[2] for m in all_m)
    det_w = widths[len(widths) // 2]
    k = det_w / _BIBLE_MARKER_W

    stride_f    = k * _BIBLE_STRIDE_X
    gap_first_f = k * _BIBLE_GAP_FIRST
    ab_w        = max(1, int(round(k * _BIBLE_SLOT_W)))

    med_h    = sorted(m[3] for m in all_m)[len(all_m) // 2]
    ab_h     = max(1, int(round(med_h / 0.63)))
    gap_y    = int(round(med_h * 0.20))

    def col_anchor(col):
        if hsv is None:
            return max(m[0] + m[2] for m in col)
        return max(m[0] + full_bar_extent(hsv, m)[0] for m in col)
    a_anchor = col_anchor(a) if a else 0
    b_anchor = col_anchor(b) if b else 0

    out = []
    for seat_idx, m in enumerate(all_m):
        mx, my, mw, mh, code = m
        slot_y = my - ab_h - gap_y
        anchor = a_anchor if seat_idx < len(a) else b_anchor
        slot_x0_f = anchor + gap_first_f
        for k_idx in range(n_abilities):
            x = int(round(slot_x0_f + k_idx * stride_f))
            out.append((seat_idx, k_idx, x, slot_y, ab_w, ab_h, code))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_panel(img: np.ndarray) -> Optional[dict]:
    """Detect the BOFF panel in `img`. Returns None when no plausible
    panel is found.

    Output dict:
      'col_a':   list[(x, y, w, h, code, spec_code | None)] — left column
                 markers (top→bottom). `spec_code` is the spec-stripe code
                 ('Cmd'/'Int'/'Tem'/'Plt'/'MW') when detected, else None.
      'col_b':   list[(x, y, w, h, code, spec_code | None)] — right column
      'score':   float — RANSAC score of the chosen 2-column grid
      'seats':   list[(side, mx, my, mw, mh, seat_code, spec_code | None)]
                 — 'L' or 'R'
      'slots':   list[(seat_idx, slot_idx, x, y, w, h, seat_code)]
                 — `seat_idx` indexes into `col_a + col_b`.
    """
    if img is None or img.size == 0:
        return None

    icon_w, icon_h = estimate_icon_dims(img)
    markers = detect_markers(img, icon_w, icon_h)
    if len(markers) < 3:
        return None

    icon_w, icon_h = _refine_dims_from_markers(markers, icon_w, icon_h)
    panel = best_panel(markers, icon_w, icon_h)
    if panel is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    slots = project_seat_slots(panel, hsv=hsv)
    a, b, score = panel
    # Augment column markers with spec stripe (per-seat) — 6-tuple
    # (mx, my, mw, mh, code, spec_code | None). Detection rate ~36% on
    # 36 GT screens; missing specs leave the slot at None.
    a = [(mx, my, mw, mh, c, classify_stripe(hsv, (mx, my, mw, mh, c))[0])
         for (mx, my, mw, mh, c) in a]
    b = [(mx, my, mw, mh, c, classify_stripe(hsv, (mx, my, mw, mh, c))[0])
         for (mx, my, mw, mh, c) in b]
    seats = (
        [('L', mx, my, mw, mh, c, sp) for (mx, my, mw, mh, c, sp) in a]
        + [('R', mx, my, mw, mh, c, sp) for (mx, my, mw, mh, c, sp) in b]
    )
    return {
        'col_a':  a,
        'col_b':  b,
        'score':  float(score),
        'seats':  seats,
        'slots':  slots,
    }
