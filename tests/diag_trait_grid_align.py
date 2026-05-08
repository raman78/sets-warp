"""Trait grid alignment — phase 3 diagnostic.

Three measurements to validate the detector strategy proposed in phase 2:

D) Cross-section column alignment
   Take the section with most boxes (richest, usually Personal Traits) as
   PIVOT. Measure x-shift of every other section's column centers vs the
   pivot's column centers. If shift is consistently small (<= a few px),
   we can lock the column grid from one section and snap others.

E) Inter-section vertical gap signature
   Measure (y_top of section N+1) - (y_bottom of section N) normalised by
   icon_h. Compare against the within-section row gaps from phase 2.
   Tests whether a single gap-threshold can split sections reliably.

F) Visual header-strip detection without GT
   Slide a 1-row-tall analysis window down the trait x-band, scoring
   "header-strip likelihood" by the brightness+edge signature found in
   phase 2 (header brighter than below + Canny edges > 0.08). For each
   GT section, measure how close the nearest detected header peak is
   to the true header band.

Read-only. Output → tests/_diag_out/trait_grid_align/.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'trait_grid_align'
OUT_VIZ = OUT_DIR / 'viz'
OUT_JSON = OUT_DIR / 'summary.json'

TRAIT_SLOTS = {
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
}

# Order matters for inter-section gap analysis (top → bottom in STO UI)
SPACE_ORDER = ['Personal Space Traits', 'Starship Traits',
               'Space Reputation', 'Active Space Rep']
GROUND_ORDER = ['Personal Ground Traits', 'Ground Reputation',
                'Active Ground Rep']


def find_screen_files(fnames):
    candidates = [
        Path('/home/raman/STO_screens'),
        ROOT / 'warp' / 'training_data' / 'migrated_images',
        ROOT / 'warp' / 'training_data' / 'screen_types',
    ]
    out = {}
    targets = set(fnames)
    for d in candidates:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if p.is_file() and p.name in targets and p.name not in out:
                out[p.name] = p
    return out


def cluster_1d(values, tol):
    if not values:
        return []
    vs = sorted(values)
    cl = [[vs[0]]]
    for v in vs[1:]:
        if v - cl[-1][-1] <= tol:
            cl[-1].append(v)
        else:
            cl.append([v])
    return cl


def column_centers(boxes):
    if not boxes:
        return []
    icon_w = st.median(b[2] for b in boxes)
    cxs = [b[0] + b[2] / 2 for b in boxes]
    cl = cluster_1d(cxs, tol=max(8, icon_w * 0.4))
    return sorted(st.median(c) for c in cl)


def section_y_range(boxes):
    return (min(b[1] for b in boxes), max(b[1] + b[3] for b in boxes))


# ── D: cross-section column alignment ──
def measure_column_alignment(boxes_by_slot):
    """For each screen, pick richest section as pivot, measure x-shift
    of other sections' column centers vs pivot's.

    Returns: list of dicts (one per non-pivot section).
    """
    if len(boxes_by_slot) < 2:
        return None, []
    # Pivot = section with most boxes (and at least 5 cols if possible)
    candidates = sorted(boxes_by_slot.items(), key=lambda kv: -len(kv[1]))
    pivot_slot = None
    pivot_cols = []
    for slot, boxes in candidates:
        cols = column_centers(boxes)
        if len(cols) == 5:
            pivot_slot = slot
            pivot_cols = cols
            break
    if pivot_slot is None:
        # No section has 5 cols → take the richest
        pivot_slot, pivot_boxes = candidates[0]
        pivot_cols = column_centers(pivot_boxes)
    if len(pivot_cols) < 2:
        return pivot_slot, []
    diffs = []
    for slot, boxes in boxes_by_slot.items():
        if slot == pivot_slot:
            continue
        cols = column_centers(boxes)
        # For each col in this section, find nearest pivot col and record
        # signed shift
        for c in cols:
            nearest = min(pivot_cols, key=lambda p: abs(p - c))
            diffs.append({
                'slot': slot, 'col_x': c, 'pivot_x': nearest,
                'shift': c - nearest,
                'abs_shift': abs(c - nearest),
            })
    return pivot_slot, diffs


# ── E: inter-section vertical gap signature ──
def measure_inter_section_gaps(boxes_by_slot, order):
    """For each consecutive pair (sec_N, sec_N+1) present in this screen,
    return gap_px and gap / icon_h."""
    present = [(s, boxes_by_slot[s]) for s in order if s in boxes_by_slot]
    if len(present) < 2:
        return []
    out = []
    for i in range(len(present) - 1):
        a_slot, a_boxes = present[i]
        b_slot, b_boxes = present[i + 1]
        a_y0, a_y1 = section_y_range(a_boxes)
        b_y0, b_y1 = section_y_range(b_boxes)
        gap = b_y0 - a_y1
        ih = st.median(a_boxes[k][3] for k in range(len(a_boxes)))
        out.append({
            'a': a_slot, 'b': b_slot,
            'gap_px': gap,
            'gap_ratio': round(gap / ih, 2) if ih > 0 else None,
        })
    return out


# ── F: visual header-strip detection without GT ──
def detect_header_peaks(img, x0, x1, y0, y1, icon_h):
    """Slide a (icon_h * 0.7) tall window down the trait x-band, scoring
    by mean brightness + Canny edge density. Returns list of (cy, score).
    """
    H, W = img.shape[:2]
    x0 = max(0, x0)
    x1 = min(W, x1)
    y0 = max(0, y0)
    y1 = min(H, y1)
    if x1 - x0 < 20 or y1 - y0 < icon_h:
        return []
    strip = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160) / 255.0
    # Per-row mean and edge density
    row_mean = gray.mean(axis=1)
    row_edge = edges.mean(axis=1)
    win = max(4, int(icon_h * 0.7))
    half = win // 2
    n = len(row_mean)
    if n < win:
        return []
    # Score = brightness * edge density (peaks where both are high)
    cum_mean = np.cumsum(np.concatenate([[0.0], row_mean]))
    cum_edge = np.cumsum(np.concatenate([[0.0], row_edge]))
    scores = np.zeros(n)
    for i in range(half, n - half):
        m = (cum_mean[i + half] - cum_mean[i - half]) / win
        e = (cum_edge[i + half] - cum_edge[i - half]) / win
        scores[i] = m * e
    # Also require this band to be brighter than the band BELOW it
    # (header-vs-below contrast). Add penalty if not.
    contrast = np.zeros(n)
    for i in range(half, n - half - win):
        m = (cum_mean[i + half] - cum_mean[i - half]) / win
        below_lo = i + half
        below_hi = below_lo + win
        if below_hi >= n:
            continue
        m_below = (cum_mean[below_hi] - cum_mean[below_lo]) / win
        contrast[i] = m - m_below
    final = scores * np.maximum(0.0, contrast)
    if final.max() < 0.5:
        return []
    threshold = final.max() * 0.3
    min_sep = max(int(icon_h * 1.0), 12)
    peaks = []
    for i in range(1, n - 1):
        if (final[i] >= threshold and
                final[i] >= final[i - 1] and final[i] >= final[i + 1]):
            if not peaks or i - peaks[-1][0] >= min_sep:
                peaks.append((i, final[i]))
            elif final[i] > peaks[-1][1]:
                peaks[-1] = (i, final[i])
    return [(p + y0, sc) for p, sc in peaks]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
    with open(ANN_PATH) as f:
        data = json.load(f)
    targets = {}
    for fn, ann in data.items():
        if not isinstance(ann, list):
            continue
        boxes_by_slot = defaultdict(list)
        for a in ann:
            if a.get('state') != 'confirmed':
                continue
            slot = a.get('slot')
            if slot in TRAIT_SLOTS:
                bb = a.get('bbox')
                if bb and len(bb) == 4:
                    boxes_by_slot[slot].append(tuple(bb))
        if boxes_by_slot:
            targets[fn] = boxes_by_slot
    paths = find_screen_files(targets.keys())
    print(f'screens: {len(targets)}  files found: {len(paths)}')

    # Aggregators
    all_align_shifts = []           # (D) list of abs shift values
    align_by_slot = defaultdict(list)
    inter_section_gaps = []         # (E) list of dicts
    header_recall = []              # (F) list of dicts per section
    header_recall_by_slot = defaultdict(list)

    summary = {'screens': {}}

    for fname, boxes_by_slot in targets.items():
        path = paths.get(fname)
        if not path:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        scr = {'size': [W, H]}

        # D: column alignment
        pivot_slot, diffs = measure_column_alignment(boxes_by_slot)
        if pivot_slot is not None:
            scr['pivot_slot'] = pivot_slot
            scr['column_align_diffs'] = diffs
            for d in diffs:
                all_align_shifts.append(d['abs_shift'])
                align_by_slot[d['slot']].append(d['abs_shift'])

        # E: inter-section gaps (probe both space + ground orderings)
        gaps_space = measure_inter_section_gaps(boxes_by_slot, SPACE_ORDER)
        gaps_ground = measure_inter_section_gaps(boxes_by_slot, GROUND_ORDER)
        if gaps_space or gaps_ground:
            scr['inter_section_gaps'] = {
                'space': gaps_space, 'ground': gaps_ground,
            }
            for g in gaps_space + gaps_ground:
                if g['gap_ratio'] is not None:
                    inter_section_gaps.append(g)

        # F: visual header detection
        all_boxes = [b for boxes in boxes_by_slot.values() for b in boxes]
        if all_boxes:
            x_min = min(b[0] for b in all_boxes)
            x_max = max(b[0] + b[2] for b in all_boxes)
            icon_h = st.median(b[3] for b in all_boxes)
            peaks = detect_header_peaks(img, x_min - 4, x_max + 4,
                                        0, H, icon_h)
            scr['detected_header_peaks'] = peaks
            # For each GT section, find nearest detected peak above first row
            section_recall = []
            for slot, boxes in boxes_by_slot.items():
                y_top = min(b[1] for b in boxes)
                # GT header expected ~icon_h above y_top
                gt_header_y = y_top - icon_h * 0.5
                # nearest peak ABOVE the icon row, within reasonable range
                cands = [(abs(p - gt_header_y), p)
                         for p, _ in peaks if p < y_top + 4]
                cands.sort()
                if cands:
                    err = cands[0][0]
                    section_recall.append({
                        'slot': slot,
                        'gt_header_y': round(gt_header_y, 1),
                        'nearest_peak_y': round(cands[0][1], 1),
                        'err_px': round(err, 1),
                        'err_ratio': round(err / icon_h, 2),
                    })
                    header_recall.append({'slot': slot, 'err_ratio': err / icon_h})
                    header_recall_by_slot[slot].append(err / icon_h)
                else:
                    section_recall.append({
                        'slot': slot, 'gt_header_y': round(gt_header_y, 1),
                        'nearest_peak_y': None, 'err_px': None,
                    })
                    header_recall_by_slot[slot].append(None)
            scr['header_recall'] = section_recall

        summary['screens'][fname] = scr

    # ── Aggregates ──
    def stats(xs, ndigits=2):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return {'n': len(xs), 'med': round(st.median(xs), ndigits),
                'min': round(min(xs), ndigits),
                'max': round(max(xs), ndigits),
                'p90': round(st.quantiles(xs, n=10)[8], ndigits)
                if len(xs) >= 10 else None}

    agg = {}
    # D
    agg['column_align'] = {
        'all_abs_shift_px': stats(all_align_shifts),
        'by_section': {s: stats(v) for s, v in align_by_slot.items()},
        'within_5px_pct': round(100 * sum(1 for x in all_align_shifts if x <= 5)
                                / max(1, len(all_align_shifts)), 1),
        'within_10px_pct': round(100 * sum(1 for x in all_align_shifts if x <= 10)
                                 / max(1, len(all_align_shifts)), 1),
    }
    # E
    pair_buckets = defaultdict(list)
    for g in inter_section_gaps:
        pair_buckets[(g['a'], g['b'])].append(g['gap_ratio'])
    agg['inter_section_gaps'] = {
        f'{a} → {b}': stats(v) for (a, b), v in pair_buckets.items()
    }
    # F
    agg['header_detection'] = {}
    for slot, errs in header_recall_by_slot.items():
        non_none = [e for e in errs if e is not None]
        agg['header_detection'][slot] = {
            'n': len(errs),
            'found_pct': round(100 * len(non_none) / max(1, len(errs)), 1),
            'err_ratio': stats(non_none),
            'within_05_icon_h_pct': round(100 * sum(1 for e in non_none if e <= 0.5)
                                          / max(1, len(non_none)), 1),
            'within_10_icon_h_pct': round(100 * sum(1 for e in non_none if e <= 1.0)
                                          / max(1, len(non_none)), 1),
        }

    summary['aggregates'] = agg
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {OUT_JSON}')

    # ── Console report ──
    print('\n=== D) CROSS-SECTION COLUMN ALIGNMENT ===')
    print(f'  all_abs_shift_px: {agg["column_align"]["all_abs_shift_px"]}')
    print(f'  within ±5px:  {agg["column_align"]["within_5px_pct"]}%')
    print(f'  within ±10px: {agg["column_align"]["within_10px_pct"]}%')
    print('  by section:')
    for slot, s in agg['column_align']['by_section'].items():
        print(f'    {slot:28s} {s}')
    print('\n=== E) INTER-SECTION GAP RATIO (gap / icon_h_of_upper) ===')
    for pair, s in agg['inter_section_gaps'].items():
        print(f'  {pair:48s} {s}')
    print('\n=== F) VISUAL HEADER DETECTION (no OCR) ===')
    for slot, r in agg['header_detection'].items():
        print(f'  {slot:28s}  n={r["n"]:3d}  found={r["found_pct"]}%  '
              f'err_ratio={r["err_ratio"]}  '
              f'within_0.5h={r["within_05_icon_h_pct"]}%  '
              f'within_1.0h={r["within_10_icon_h_pct"]}%')


if __name__ == '__main__':
    main()
