"""
OCR-anchored EQ panel geometry detector.

Produces a single source of truth for the 6-cell × N-row EQ matrix:
  * panel_x_start  — left edge of the matrix (pixel-derived)
  * panel_right    — right edge of the matrix (from single-slot icon right-edges,
                     math-extrapolated when none detected)
  * final_dx       — cell pitch in x direction
  * row_pitch      — cell pitch in y direction
  * row_cys        — list of visible row center-Y values, top→bottom
  * mode           — 'v8' if right-edge anchored on real icons, 'MATH_FALLBACK' otherwise

Pipeline:
  1. EasyOCR on full image → label tokens.
  2. Classify tokens against EQ-row keyword tables, cluster 2-line labels.
  3. X-cluster canonical-named hits by x1; keep the LARGEST cluster as the real
     EQ label column. Discards off-panel hits (HUD "Shields", specialization
     "Miracle Worker" / "Temporal Operative", etc.).
  4. detect_stripe_start (HSV gradient) per label → panel_x_start (median).
  5. row_pitch = median of cy-gaps / canonical-step-count between EQ-column hits.
  6. est_dx = row_pitch × DX_RATIO  (0.725 — see comment below)
  7. detect_v8_adaptive_bg (RTL adaptive-bg scan) on canonical single-slot rows
     (Deflector / Engines / Warp Core / Shields) → panel_right (median).
     Math fallback: panel_x_start + 6 × est_dx.
  8. final_dx = (panel_right - panel_x_start) / 6.
  9. Visible row cys = OCR-column cys with linear interpolation between
     consecutive cys whose gap is a multiple of row_pitch. No extrapolation
     beyond the highest/lowest OCR anchor — honest visualisation of what was
     actually seen.

DX_RATIO derivation:
  Statistical measurement across 38 GT-annotated EQ screens (ratio of GT-median
  dx to OCR-derived row_pitch). Median 0.725, stdev ~0.03. Used only as a
  default-multiplier for `est_dx` — final_dx is recomputed from the v8 right
  edge whenever available. This replaces an older `row_pitch/1.5 + 3` formula
  which underestimated dx by ~1.3 px/cell on big-icon panels.

This module is pure detection — no annotations.json access, no ground-truth
look-up. Caller passes only an image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import cv2
import numpy as np

# Lazy import to avoid circular dependency during module load
_TEXT_EXTRACTOR = None


def _get_easyocr_reader():
    """Return a shared EasyOCR reader (via TextExtractor)."""
    global _TEXT_EXTRACTOR
    if _TEXT_EXTRACTOR is None:
        from warp.recognition.text_extractor import TextExtractor
        _TEXT_EXTRACTOR = TextExtractor()
    return _TEXT_EXTRACTOR._get_ocr()


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# dx/row_pitch ratio: median 0.725 across 38 EQ-GT screens.
DX_RATIO = 0.725

# X-cluster tolerance (px) when grouping label x1 values.
X_CLUSTER_TOL = 30

# Single-slot rows whose right edge is reliable for anchoring panel_right.
TARGET_SINGLE_SLOTS = {'Deflector', 'Engines', 'Warp Core', 'Shields'}

# Canonical EQ row order (top→bottom in STO UI).
STD_ORDER = {
    'Fore Weapons':         0,
    'Aft Weapons':          1,
    'Experimental':         2,
    'Deflector':            3,
    'Sec-Def':              4,
    'Engines':              5,
    'Warp Core':            6,
    'Shields':              7,
    'Devices':              8,
    'Universal Consoles':   9,
    'Engineering Consoles': 10,
    'Science Consoles':     11,
    'Tactical Consoles':    12,
    'Hangars':              13,
}

# German UI variants (OCR sometimes returns these on localized clients).
GERMAN_ORDER = {
    'Bug Waffen':       0,
    'Heck Waffen':      1,
    'Deflektor':        3,
    'Antriebe':         5,
    'Warp Antrieb':     6,
    'Schilde':          7,
    'Geraete':          8,
}

# Single-token first-line keywords for the row-label OCR classifier.
SINGLE_LINE_KW = {
    'deflector':    'Deflector',
    'secondary':    'Sec-Def',
    'impulse':      'Engines',
    'engines':      'Engines',
    'singularity':  'Warp Core',
    'devices':      'Devices',
    'hangars':      'Hangars',
    'hangar':       'Hangars',
    'shields':      'Shields',
    'shield':       'Shields',
    'experimental': 'Experimental',
}

FIRST_LINE_KW = {
    'fore':         'Fore',
    'aft':          'Aft',
    'alt':          'Aft',
    'aff':          'Aft',
    'universal':    'Universal',
    'engineering':  'Engineering',
    'science':      'Science',
    'tactical':     'Tactical',
    'warp':         'Warp',
    'secondary':    'Secondary',
    'experimental': 'Experimental',
}

SECOND_LINE_KW = {
    'weapons':   'Weapons',
    'weapon':    'Weapons',
    'consoles':  'Consoles',
    'console':   'Consoles',
    'core':      'Core',
    'deflector': 'Deflector',
}

COMPOSITE = {
    ('Fore',         'Weapons'):   'Fore Weapons',
    ('Aft',          'Weapons'):   'Aft Weapons',
    ('Universal',    'Consoles'):  'Universal Consoles',
    ('Engineering',  'Consoles'):  'Engineering Consoles',
    ('Science',      'Consoles'):  'Science Consoles',
    ('Tactical',     'Consoles'):  'Tactical Consoles',
    ('Warp',         'Core'):      'Warp Core',
    ('Secondary',    'Deflector'): 'Sec-Def',
    ('Experimental', 'Weapons'):   'Experimental',
}

_FUZZY_CUTOFF = 0.65

COL_W = 3  # column slice width (px) for stripe-start / right-edge scans


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------

@dataclass
class EQGeometry:
    panel_x_start: int
    panel_right:   int
    final_dx:      float
    row_pitch:     int
    row_cys:       list   # sorted top→bottom, visible rows
    mode:          str    # 'v8' or 'MATH_FALLBACK'
    eq_label_cys:  dict = field(default_factory=dict)  # {canonical_idx: cy}

    @property
    def panel_width(self) -> int:
        return self.panel_right - self.panel_x_start


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


def _fuzzy_best(low: str, kw_dict: dict) -> tuple[Optional[str], float]:
    from difflib import SequenceMatcher
    n_best, r_best = None, 0.0
    for kw, name in kw_dict.items():
        if abs(len(kw) - len(low)) > 2:
            continue
        r = SequenceMatcher(None, low, kw).ratio()
        if r > r_best:
            r_best = r
            n_best = name
    return n_best, r_best


def _classify_tokens(tokens: list[dict]) -> list[dict]:
    """Tag each token with its keyword role (single / first / second)."""
    for t in tokens:
        t['kw_role'] = None
        t['kw_name'] = None
        t['kw_alt_first']  = None
        t['kw_alt_second'] = None
        low = ''.join(ch for ch in t['low'] if ch.isalnum())
        if not low:
            continue
        if low in SINGLE_LINE_KW:
            t['kw_role'] = 'single'
            t['kw_name'] = SINGLE_LINE_KW[low]
            t['kw_alt_first']  = FIRST_LINE_KW.get(low)
            t['kw_alt_second'] = SECOND_LINE_KW.get(low)
            continue
        if low in FIRST_LINE_KW:
            t['kw_role'] = 'first'
            t['kw_name'] = FIRST_LINE_KW[low]
            t['kw_alt_second'] = SECOND_LINE_KW.get(low)
            continue
        if low in SECOND_LINE_KW:
            t['kw_role'] = 'second'
            t['kw_name'] = SECOND_LINE_KW[low]
            t['kw_alt_first']  = FIRST_LINE_KW.get(low)
            continue
        if not (3 <= len(low) <= 15):
            continue
        n1, r1 = _fuzzy_best(low, SINGLE_LINE_KW)
        n2, r2 = _fuzzy_best(low, FIRST_LINE_KW)
        n3, r3 = _fuzzy_best(low, SECOND_LINE_KW)
        choices = [('single', n1, r1), ('first', n2, r2), ('second', n3, r3)]
        choices = [c for c in choices if c[1] is not None and c[2] >= _FUZZY_CUTOFF]
        if not choices:
            continue
        choices.sort(key=lambda c: -c[2])
        role, name, _ = choices[0]
        t['kw_role'] = role
        t['kw_name'] = name
    return tokens


def _cluster_2line_labels(tokens: list[dict]) -> list[dict]:
    """Pair first-line + second-line tokens vertically aligned (x-overlap >40%,
    y-gap within ~1.6× line height) into composite hits."""
    firsts = []
    for t in tokens:
        if t['kw_role'] == 'first':
            firsts.append((t, t['kw_name']))
        elif t.get('kw_alt_first'):
            firsts.append((t, t['kw_alt_first']))
    seconds = []
    for t in tokens:
        if t['kw_role'] == 'second':
            seconds.append((t, t['kw_name']))
        elif t.get('kw_alt_second'):
            seconds.append((t, t['kw_alt_second']))

    composites = []
    used_first  = set()
    used_second = set()
    for i, (f, fname) in enumerate(firsts):
        if i in used_first:
            continue
        for j, (s, sname) in enumerate(seconds):
            if j in used_second:
                continue
            if f is s:
                continue
            row_name = COMPOSITE.get((fname, sname))
            if not row_name:
                continue
            ox = max(0, min(f['x1'], s['x1']) - max(f['x0'], s['x0']))
            min_w = max(1, min(f['w'], s['w']))
            x_overlap = ox / min_w
            line_h = max(f['h'], s['h'])
            y_gap = s['y0'] - f['y1']
            if x_overlap > 0.4 and -line_h * 0.5 <= y_gap <= line_h * 1.6:
                composites.append({
                    'row': row_name,
                    'cx': (f['cx'] + s['cx']) // 2,
                    'cy': (f['cy'] + s['cy']) // 2,
                    'x0': min(f['x0'], s['x0']),
                    'y0': min(f['y0'], s['y0']),
                    'x1': max(f['x1'], s['x1']),
                    'y1': max(f['y1'], s['y1']),
                })
                used_first.add(i)
                used_second.add(j)
                if f['kw_role'] == 'single':
                    f['kw_role'] = '_consumed'
                if s['kw_role'] == 'single':
                    s['kw_role'] = '_consumed'
                break
    return composites


def _collect_single_hits(tokens: list[dict]) -> list[dict]:
    """Single-line keyword hits → composite-form dict per row (best conf)."""
    by_row: dict = {}
    for t in tokens:
        if t['kw_role'] != 'single':
            continue
        row = t['kw_name']
        if row not in by_row or t['conf'] > by_row[row]['conf']:
            by_row[row] = {
                'row': row,
                'cx': t['cx'], 'cy': t['cy'],
                'x0': t['x0'], 'y0': t['y0'],
                'x1': t['x1'], 'y1': t['y1'],
                'conf': t['conf'],
            }
    return list(by_row.values())


def _warp_core_fallback(all_hits: list[dict], classified: list[dict]) -> list[dict]:
    """If no Warp Core composite, accept a standalone 'Warp' first-line token."""
    if any(h.get('row') == 'Warp Core' for h in all_hits):
        return all_hits
    for t in classified:
        if t.get('kw_role') == 'first' and t.get('kw_name') == 'Warp':
            all_hits = list(all_hits) + [{
                'row': 'Warp Core',
                'cx': t['cx'], 'cy': t['cy'],
                'x0': t['x0'], 'y0': t['y0'],
                'x1': t['x1'], 'y1': t['y1'],
            }]
            break
    return all_hits


def _canonical_idx(row_text: str) -> Optional[int]:
    row = (row_text or '').strip()
    if row in STD_ORDER:
        return STD_ORDER[row]
    if row in GERMAN_ORDER:
        return GERMAN_ORDER[row]
    if 'hangar' in row.lower():
        return 13
    return None


def _cluster_by_x1(hits: list[dict], tol: int = X_CLUSTER_TOL) -> list[list[dict]]:
    if not hits:
        return []
    s = sorted(hits, key=lambda h: h['x1'])
    groups = [[s[0]]]
    for h in s[1:]:
        if h['x1'] - groups[-1][-1]['x1'] <= tol:
            groups[-1].append(h)
        else:
            groups.append([h])
    return groups


# ----------------------------------------------------------------------------
# Pixel-level helpers
# ----------------------------------------------------------------------------

def _detect_stripe_start(img_hsv: np.ndarray, y0: int, y1: int,
                         x_from: int, x_to: int) -> Optional[int]:
    """Gradient-relative panel-content start. Walk x left→right inside a y-band
    and return first column where V drops ≥25 below baseline or S jumps ≥30
    above baseline (or absolute S ≥ 60). Baseline = first 4 columns from x_from
    (right of label, still on label-bg)."""
    H, W = img_hsv.shape[:2]
    y0 = max(0, y0)
    y1 = min(H, y1)
    x_from = max(0, x_from)
    x_to   = min(W, x_to)
    if y1 <= y0 or x_to - x_from < COL_W + 4:
        return None
    band = img_hsv[y0:y1, x_from:x_to]
    S_chan = band[..., 1]
    V_chan = band[..., 2]
    base_v = float(V_chan[:, :4].mean())
    base_s = float(S_chan[:, :4].mean())
    for x in range(0, band.shape[1] - COL_W):
        col_v = V_chan[:, x:x + COL_W]
        col_s = S_chan[:, x:x + COL_W]
        mv = float(col_v.mean())
        ms = float(col_s.mean())
        v_dropped = (base_v - mv) >= 25
        s_jumped  = (ms - base_s) >= 30 or ms >= 60
        if v_dropped or s_jumped:
            return x_from + x
    return None


def _detect_right_edge_adaptive_bg(img_hsv: np.ndarray, y0: int, y1: int,
                                   x_from: int, x_to: int) -> Optional[int]:
    """Adaptive-background right-edge scan. Walk x right→left across a y-band;
    take bg_v = min of per-column V means across the search band (the darkest
    column = inter-cell stripe or post-icon background), then return the
    rightmost x with 2 consecutive columns above bg_v + 12. Designed for
    single-slot icon rows (Deflector / Engines / Warp Core / Shields) — the
    icon's right edge is a strong brightness cliff above the surrounding
    panel-bg / ship-image. Benchmark vs GT (38 screens, 4 single-slot rows
    each): median Δ=0 px, mean |Δ|=2.6 px, 97 % within ±10 px."""
    H, W = img_hsv.shape[:2]
    y0, y1 = max(0, y0), min(H, y1)
    x_from, x_to = max(0, x_from), min(W, x_to)
    if y1 <= y0 or x_to - x_from < 7:
        return None
    band = img_hsv[y0:y1, x_from:x_to]
    V_chan = band[..., 2]
    col_means = V_chan.mean(axis=0)
    bg_v = float(np.min(col_means))
    threshold = bg_v + 12.0
    consecutive = 0
    for x in range(len(col_means) - 1, -1, -1):
        if col_means[x] > threshold:
            consecutive += 1
            if consecutive >= 2:
                # +3 = off-by-one (return one PAST rightmost bright column)
                # + 2 px to absorb antialiased icon edge that falls below
                # the brightness threshold. Empirically aligns scan result
                # with GT bbox right-edge to within ±2 px on 38 GT screens.
                return x_from + x + 3
        else:
            consecutive = 0
    return None


# ----------------------------------------------------------------------------
# Row reconstruction
# ----------------------------------------------------------------------------

def _rows_from_filtered_hits(filtered: list[dict], row_pitch: int) -> list[int]:
    """Compute visible row cy positions from filtered OCR hits.
    Between consecutive OCR cys, insert round(gap/row_pitch) - 1 interpolated
    rows. No extrapolation beyond first/last OCR cy."""
    if not filtered:
        return []
    s = sorted(filtered, key=lambda h: h['cy'])
    rows = [s[0]['cy']]
    for i in range(1, len(s)):
        prev_cy = s[i - 1]['cy']
        curr_cy = s[i]['cy']
        gap = curr_cy - prev_cy
        steps = max(1, round(gap / row_pitch))
        for k in range(1, steps):
            rows.append(int(prev_cy + k * gap / steps))
        rows.append(curr_cy)
    return rows


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def detect_eq_geometry(img: np.ndarray) -> Optional[EQGeometry]:
    """Full pipeline. Returns None when OCR yields no usable EQ labels."""
    if img is None or img.size == 0:
        return None
    H, W = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ocr = _run_ocr(img)
    if not ocr:
        return None
    classified = _classify_tokens(ocr)
    all_hits = _cluster_2line_labels(classified) + _collect_single_hits(classified)
    all_hits = _warp_core_fallback(all_hits, classified)
    if not all_hits:
        return None

    # Canonical hits only (named EQ labels)
    canonical = [h for h in all_hits if _canonical_idx(h.get('row', '')) is not None]
    if not canonical:
        return None

    # X-cluster: keep the largest cluster of canonical hits as the EQ column
    groups = _cluster_by_x1(canonical, tol=X_CLUSTER_TOL)
    eq_column = max(groups, key=len)

    # Deduplicate by canonical idx (average cy if two hits share idx)
    eq_by_idx: dict = {}
    for h in eq_column:
        idx = _canonical_idx(h['row'])
        if idx is None:
            continue
        if idx in eq_by_idx:
            merged = dict(h)
            merged['cy'] = int((eq_by_idx[idx]['cy'] + h['cy']) / 2)
            eq_by_idx[idx] = merged
        else:
            eq_by_idx[idx] = h
    if not eq_by_idx:
        return None

    # panel_x_start from EQ-column hits only
    x_starts: list[int] = []
    for h in eq_by_idx.values():
        y0_b, y1_b = h['cy'] - 15, h['cy'] + 15
        x_pix = _detect_stripe_start(img_hsv, y0_b, y1_b, h['x1'] + 2, h['x1'] + 150)
        if x_pix:
            x_starts.append(x_pix)
    if not x_starts:
        return None
    panel_x_start = int(median(x_starts))

    # row_pitch from canonical-idx-aware cy gaps
    items = sorted(eq_by_idx.items())
    pitches: list[float] = []
    for i in range(1, len(items)):
        idx_prev, h_prev = items[i - 1]
        idx_curr, h_curr = items[i]
        steps = idx_curr - idx_prev
        if steps <= 0:
            continue
        pitches.append((h_curr['cy'] - h_prev['cy']) / steps)
    if not pitches:
        return None
    row_pitch = int(round(median(pitches)))
    est_dx = row_pitch * DX_RATIO

    # Visible rows
    row_cys = _rows_from_filtered_hits(list(eq_by_idx.values()), row_pitch)

    # panel_right via right-edge scan. Primary: single-slot rows (guaranteed
    # 1 icon at the right edge). Secondary: any other OCR-anchored row. STO
    # is right-justified — every populated row's rightmost cell sits at
    # panel_right, so multi-cell rows are equally valid anchors when the
    # canonical single-slot rows weren't OCR-resolved.
    # Narrow search range for primary (single-slot rows): tight margin
    # avoids catching adjacent UI elements when the canonical anchors
    # are present. STO right-justifies, so single-slot icons sit exactly
    # at panel_right.
    x_search_start = int(panel_x_start + 4.5 * est_dx)
    x_search_end_tight = min(W - 1, int(panel_x_start + 6.05 * est_dx))
    # Wider range for the multi-cell fallback: est_dx can underestimate
    # true dx by up to ~1.5 px/cell (DX_RATIO stdev ≈ 0.03), so by
    # 6 cells the tight bound may sit ~6-9 px short of the real right
    # edge. +0.15× row_pitch (~7-8 px) gives enough headroom while
    # stopping well before UI glow / adjacent panel edges.
    x_search_end_wide = min(W - 1,
                            int(panel_x_start + 6 * est_dx + 0.15 * row_pitch))

    def _scan_rows(slot_indices, x_end):
        out = []
        for idx in slot_indices:
            if idx not in eq_by_idx:
                continue
            cy = eq_by_idx[idx]['cy']
            r = _detect_right_edge_adaptive_bg(img_hsv, cy - 8, cy + 8,
                                               x_search_start, x_end)
            if r is not None:
                out.append(r)
        return out

    primary_idx = [STD_ORDER[s] for s in TARGET_SINGLE_SLOTS if s in STD_ORDER]
    rights = _scan_rows(primary_idx, x_search_end_tight)
    mode = 'v8' if rights else None

    if not rights:
        # Fallback: scan any remaining OCR-anchored rows (multi-cell).
        secondary_idx = [i for i in eq_by_idx.keys() if i not in primary_idx]
        rights = _scan_rows(secondary_idx, x_search_end_wide)
        if rights:
            mode = 'v8_multicell'

    if rights:
        panel_right = int(median(rights))
        final_dx = (panel_right - panel_x_start) / 6.0
    else:
        panel_right = int(panel_x_start + 6 * est_dx)
        final_dx = float(est_dx)
        mode = 'MATH_FALLBACK'

    return EQGeometry(
        panel_x_start=panel_x_start,
        panel_right=panel_right,
        final_dx=final_dx,
        row_pitch=row_pitch,
        row_cys=row_cys,
        mode=mode,
        eq_label_cys={idx: h['cy'] for idx, h in eq_by_idx.items()},
    )
