"""Trait anchors — phase 2 diagnostic.

Measures three approaches to anchor trait sections WITHOUT relying on
external OCR (which the user said is "często, ale nie zawsze").

A) Header-bar visual signature
   STO trait sections are preceded by a darker decorative strip with the
   label text. Per GT section, sample the strip ABOVE the first row of
   icons and characterise it: brightness, contrast, width, edge density.

B) Section-anchored column projection
   For each GT section, project bright pixels in its actual y-band only
   (not full image). Measure how reliably this finds 5 columns.

C) Inter-row divider in Starship Traits
   User reported: when Starship Traits spans 2 rows, there is a thin
   horizontal bar + ship-name text between the rows. Measure the gap
   between consecutive rows of Starship Traits and compare to gaps in
   other sections.

Pure read-only. Output → tests/_diag_out/trait_anchors/.
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
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'trait_anchors'
OUT_VIZ = OUT_DIR / 'viz'
OUT_JSON = OUT_DIR / 'summary.json'

TRAIT_SLOTS = {
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
}


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
    clusters = [[vs[0]]]
    for v in vs[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def split_rows(boxes, gap_factor=0.6):
    """Split boxes into rows by y-clustering. Returns list of row-boxes
    sorted top-to-bottom. Within-row tolerance scales with icon height."""
    if not boxes:
        return []
    icon_h = st.median(b[3] for b in boxes)
    tol = max(8, icon_h * gap_factor)
    cys = sorted(((b[1] + b[3] / 2), b) for b in boxes)
    rows = [[cys[0]]]
    for cy, b in cys[1:]:
        if cy - rows[-1][-1][0] <= tol:
            rows[-1].append((cy, b))
        else:
            rows.append([(cy, b)])
    return [[b for _, b in row] for row in rows]


def measure_header_strip(img, section_boxes, h):
    """Sample horizontal strip just ABOVE the first row of `section_boxes`.

    Returns dict of intensity / contrast metrics, or None if out of bounds.
    """
    rows = split_rows(section_boxes)
    if not rows:
        return None
    first_row = rows[0]
    y_top = min(b[1] for b in first_row)
    x_min = min(b[0] for b in first_row)
    x_max = max(b[0] + b[2] for b in first_row)
    icon_h = st.median(b[3] for b in first_row)
    # strip = the icon_h tall band immediately above the first row
    strip_y0 = max(0, int(y_top - icon_h * 0.9))
    strip_y1 = max(strip_y0 + 1, int(y_top - 2))
    strip_x0 = max(0, int(x_min - 4))
    strip_x1 = min(img.shape[1], int(x_max + 4))
    if strip_y1 - strip_y0 < 6 or strip_x1 - strip_x0 < 20:
        return None
    strip = img[strip_y0:strip_y1, strip_x0:strip_x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    # Compare to a same-sized strip BELOW the row (to confirm the header
    # has a distinct signature, not just "any band looks the same").
    below_y0 = min(img.shape[0], y_top + int(icon_h * 1.05))
    below_y1 = min(img.shape[0], below_y0 + (strip_y1 - strip_y0))
    below_metrics = None
    if below_y1 - below_y0 >= 6:
        below = img[below_y0:below_y1, strip_x0:strip_x1]
        bgray = cv2.cvtColor(below, cv2.COLOR_BGR2GRAY)
        below_metrics = {
            'mean': float(bgray.mean()),
            'std': float(bgray.std()),
        }
    return {
        'y0': strip_y0, 'y1': strip_y1, 'x0': strip_x0, 'x1': strip_x1,
        'mean': float(gray.mean()),
        'std': float(gray.std()),
        # text-presence proxy: edge-pixel density (Canny)
        'edge_density': float(cv2.Canny(gray, 80, 160).mean() / 255.0),
        'below': below_metrics,
    }


def section_anchored_columns(img, section_boxes, expected=5):
    """Project bright pixels in this section's y-band only.

    Returns: detected peak xs, recall vs GT col centers within ±10px.
    """
    if not section_boxes:
        return None
    rows = split_rows(section_boxes)
    if not rows:
        return None
    # Section y-band = top of first row to bottom of last row
    y0 = min(b[1] for b in section_boxes)
    y1 = max(b[1] + b[3] for b in section_boxes)
    # Section x-band: tight around bbox cx-range, with small margin
    icon_w = st.median(b[2] for b in section_boxes)
    x0 = max(0, int(min(b[0] for b in section_boxes) - icon_w * 0.4))
    x1 = min(img.shape[1], int(max(b[0] + b[2] for b in section_boxes) + icon_w * 0.4))
    if x1 - x0 < 20 or y1 - y0 < 10:
        return None
    strip = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    col_b = mask.sum(axis=0).astype(float) / 255.0
    if col_b.max() < 1:
        return None
    k = max(3, (x1 - x0) // 80)
    col_b = np.convolve(col_b, np.ones(k) / k, mode='same')
    threshold = col_b.max() * 0.4
    min_sep = max(15, int(icon_w * 0.7))
    peaks = []
    for i in range(1, len(col_b) - 1):
        if (col_b[i] >= threshold and
                col_b[i] >= col_b[i - 1] and col_b[i] >= col_b[i + 1]):
            if not peaks or i - peaks[-1] >= min_sep:
                peaks.append(i)
            elif col_b[i] > col_b[peaks[-1]]:
                peaks[-1] = i
    detected_xs = [p + x0 for p in peaks]
    # GT cols
    gt_cxs = sorted(set(round(b[0] + b[2] / 2, 1) for b in section_boxes))
    # cluster GT cxs (in case of multi-row alignment noise)
    gt_clusters = cluster_1d(gt_cxs, tol=max(8, icon_w * 0.4))
    gt_centers = [st.median(c) for c in gt_clusters]
    matched = sum(1 for g in gt_centers
                  if any(abs(g - d) <= 10 for d in detected_xs))
    return {
        'gt_cols': len(gt_centers),
        'detected_peaks': len(detected_xs),
        'detected_xs': detected_xs,
        'matched': matched,
        'recall': round(matched / len(gt_centers), 3) if gt_centers else None,
        'has_5_peaks': len(detected_xs) == 5,
    }


def analyze_inter_row_gaps(section_boxes, slot_name):
    """For multi-row sections, measure the gap between consecutive rows
    (top of next row - bottom of prev row). User reports Starship Traits
    has a wider gap (ship name strip) than within-section gaps elsewhere.
    """
    rows = split_rows(section_boxes)
    if len(rows) < 2:
        return None
    icon_h = st.median(b[3] for b in section_boxes)
    gaps = []
    for i in range(len(rows) - 1):
        prev_bot = max(b[1] + b[3] for b in rows[i])
        next_top = min(b[1] for b in rows[i + 1])
        gaps.append(next_top - prev_bot)
    return {
        'slot': slot_name,
        'n_rows': len(rows),
        'icon_h': round(icon_h, 1),
        'gaps_px': gaps,
        'gaps_ratio': [round(g / icon_h, 2) for g in gaps],
    }


def viz(img, fname, sections, header_metrics, col_results, out_path):
    canvas = img.copy()
    section_colors = {
        'Personal Space Traits':  (255, 200, 0),
        'Starship Traits':        (0, 200, 255),
        'Space Reputation':       (180, 255, 100),
        'Active Space Rep':       (200, 100, 255),
        'Personal Ground Traits': (255, 200, 0),
        'Ground Reputation':      (180, 255, 100),
        'Active Ground Rep':      (200, 100, 255),
    }
    for slot, boxes in sections.items():
        col = section_colors.get(slot, (200, 200, 200))
        for b in boxes:
            x, y, w, h = b
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)
        # Draw header strip
        hm = header_metrics.get(slot)
        if hm:
            cv2.rectangle(canvas, (hm['x0'], hm['y0']),
                          (hm['x1'], hm['y1']), col, 1)
            label = f"{slot}: m={hm['mean']:.0f} s={hm['std']:.0f} e={hm['edge_density']:.2f}"
            cv2.putText(canvas, label, (hm['x0'], max(10, hm['y0'] - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        # Draw section-anchored col peaks (magenta short ticks at row top)
        cr = col_results.get(slot)
        if cr and boxes:
            y_top = min(b[1] for b in boxes)
            for x in cr['detected_xs']:
                cv2.line(canvas, (int(x), y_top - 6), (int(x), y_top - 1),
                         (255, 0, 255), 1)
    cv2.putText(canvas, fname, (4, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), canvas)


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
    headers_by_slot = defaultdict(list)  # slot → list of metric dicts
    below_by_slot   = defaultdict(list)
    col_recall_by_slot = defaultdict(list)
    col_5peaks_by_slot = defaultdict(list)
    inter_row_by_slot  = defaultdict(list)

    summary = {'screens': {}}

    for fname, boxes_by_slot in targets.items():
        path = paths.get(fname)
        if not path:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        scr = {'size': [W, H], 'sections': {}}
        header_metrics = {}
        col_results = {}
        for slot, boxes in boxes_by_slot.items():
            entry = {'n_boxes': len(boxes)}
            # A: header strip
            hm = measure_header_strip(img, list(boxes), H)
            if hm:
                header_metrics[slot] = hm
                headers_by_slot[slot].append({
                    'mean': hm['mean'], 'std': hm['std'],
                    'edge_density': hm['edge_density'],
                })
                if hm['below']:
                    below_by_slot[slot].append(hm['below'])
                entry['header_strip'] = hm
            # B: section-anchored cols
            cr = section_anchored_columns(img, list(boxes))
            if cr:
                col_results[slot] = cr
                col_recall_by_slot[slot].append(cr['recall'])
                col_5peaks_by_slot[slot].append(cr['has_5_peaks'])
                entry['column_projection'] = cr
            # C: inter-row gaps (multi-row sections)
            ig = analyze_inter_row_gaps(list(boxes), slot)
            if ig:
                inter_row_by_slot[slot].append(ig)
                entry['inter_row'] = ig
            scr['sections'][slot] = entry
        summary['screens'][fname] = scr
        viz(img, fname, boxes_by_slot, header_metrics, col_results,
            OUT_VIZ / fname)

    # ── Aggregates ──
    def stats(xs):
        if not xs:
            return None
        return {'n': len(xs), 'med': round(st.median(xs), 2),
                'min': round(min(xs), 2), 'max': round(max(xs), 2)}

    agg = {'header_strip': {}, 'col_projection': {}, 'inter_row': {}}
    for slot, lst in headers_by_slot.items():
        agg['header_strip'][slot] = {
            'mean':  stats([m['mean'] for m in lst]),
            'std':   stats([m['std']  for m in lst]),
            'edge':  stats([m['edge_density'] for m in lst]),
            'n':     len(lst),
        }
    for slot, lst in below_by_slot.items():
        agg['header_strip'].setdefault(slot, {})['below'] = {
            'mean': stats([m['mean'] for m in lst]),
            'std':  stats([m['std']  for m in lst]),
        }
    for slot in col_recall_by_slot:
        recall = col_recall_by_slot[slot]
        peaks5 = col_5peaks_by_slot[slot]
        agg['col_projection'][slot] = {
            'n_screens': len(recall),
            'recall_med': round(st.median(recall), 3) if recall else None,
            'recall_mean': round(sum(recall) / len(recall), 3) if recall else None,
            'pct_5_peaks': round(100 * sum(peaks5) / len(peaks5), 1) if peaks5 else None,
        }
    for slot, lst in inter_row_by_slot.items():
        all_gaps_ratio = [g for ig in lst for g in ig['gaps_ratio']]
        agg['inter_row'][slot] = {
            'n_screens_with_multi_row': len(lst),
            'gap_ratio_to_icon_h': stats(all_gaps_ratio),
            'samples': lst[:5],  # keep first 5 for inspection
        }

    summary['aggregates'] = agg
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {OUT_JSON}')

    # Console report
    print('\n=== A) HEADER STRIP signature (above first icon row) ===')
    for slot, m in agg['header_strip'].items():
        print(f'  {slot:28s}  n={m["n"]:3d}  '
              f'mean={m["mean"]}  std={m["std"]}  edge={m["edge"]}')
        if 'below' in m:
            print(f'    below-strip (sanity)  mean={m["below"]["mean"]} std={m["below"]["std"]}')
    print('\n=== B) SECTION-ANCHORED column projection ===')
    for slot, r in agg['col_projection'].items():
        print(f'  {slot:28s}  n={r["n_screens"]:3d}  '
              f'recall_med={r["recall_med"]}  recall_mean={r["recall_mean"]}  '
              f'pct_exactly_5_peaks={r["pct_5_peaks"]}%')
    print('\n=== C) INTER-ROW gap ratio (gap / icon_h) ===')
    for slot, r in agg['inter_row'].items():
        print(f'  {slot:28s}  n={r["n_screens_with_multi_row"]:3d}  '
              f'gap_ratio={r["gap_ratio_to_icon_h"]}')


if __name__ == '__main__':
    main()
