"""
Empirical check: does the BOFF ability-slot gap (2 px in the bible) scale
with UI scale, or does it stay fixed at 2 px regardless of marker size?

For every GT seat with ≥2 confirmed BOFF ability bboxes in a row, we
measure:
  - gt_w     = median bbox width across the row
  - gt_stride = median (next.x - this.x) between adjacent same-row bboxes
  - gt_gap   = stride - gt_w  (the actual visible gap between icons)

Then we group rows by detected marker_w on the same screen and check:
  - is `gt_gap` ≈ 2 px on every UI scale (fixed-px gap), OR
  - is `gt_gap / marker_w` ≈ 2/29 (gap scales)?

Prints per-screen rows and aggregate stats grouped by marker_w bucket.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

from tests.diag_boff_markers import (   # noqa: E402
    detect_markers,
    estimate_icon_dims,
    find_screen_files,
    VIRTUAL,
)

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'


def main():
    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))
    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        boffs = [a for a in entries
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('bbox')]
        n_real = sum(1 for a in boffs if a.get('name', '') not in VIRTUAL)
        if n_real >= 5:
            candidates.append((fname, boffs))

    paths = find_screen_files([c[0] for c in candidates])

    rows = []
    for fname, boffs in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        iw0, ih0 = estimate_icon_dims(img)
        markers = detect_markers(img, iw0, ih0)
        if len(markers) < 3:
            continue
        widths = sorted(m[2] for m in markers)
        m_w = widths[len(widths) // 2]

        # Cluster GT bboxes into rows by Y proximity (tol = 0.5*median_h).
        bboxes = [tuple(a['bbox']) for a in boffs
                  if a.get('name', '') not in VIRTUAL or True]
        if len(bboxes) < 2:
            continue
        med_bh = sorted(b[3] for b in bboxes)[len(bboxes) // 2]
        ytol = max(2, 0.5 * med_bh)

        # Sort by Y then X.
        bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        # Greedy row-cluster.
        row_clusters = []
        cur = [bboxes[0]]
        for b in bboxes[1:]:
            if abs(b[1] - cur[-1][1]) <= ytol:
                cur.append(b)
            else:
                row_clusters.append(cur)
                cur = [b]
        row_clusters.append(cur)

        # Within each row, group by X-column (left vs right column of panel).
        # Simple split: x-gap > 3*median_w marks a column boundary.
        for row in row_clusters:
            if len(row) < 2:
                continue
            row = sorted(row, key=lambda b: b[0])
            ws = [b[2] for b in row]
            row_med_w = st.median(ws)
            # Split by x-gap > 1.8*med_w → different seat/column.
            seats = [[row[0]]]
            for prev, cur_b in zip(row, row[1:]):
                gap_x = cur_b[0] - (prev[0] + prev[2])
                if gap_x > 1.8 * row_med_w:
                    seats.append([cur_b])
                else:
                    seats[-1].append(cur_b)

            for seat in seats:
                if len(seat) < 2:
                    continue
                seat = sorted(seat, key=lambda b: b[0])
                strides = [seat[i + 1][0] - seat[i][0]
                           for i in range(len(seat) - 1)]
                gt_w   = st.median(b[2] for b in seat)
                stride = st.median(strides)
                gap    = stride - gt_w
                rows.append({
                    'file':   fname,
                    'mw':     m_w,
                    'n':      len(seat),
                    'gt_w':   gt_w,
                    'stride': stride,
                    'gap':    gap,
                })

    # Per-screen / per-row print.
    rows.sort(key=lambda r: (r['mw'], r['file']))
    print(f'{"file":<42s} {"mw":>3s} {"n":>2s} '
          f'{"gt_w":>5s} {"stride":>6s} {"gap":>4s} '
          f'{"stride/mw":>9s} {"gap/mw":>7s}')
    for r in rows:
        print(f'{r["file"][:42]:<42s} {r["mw"]:>3d} {r["n"]:>2d} '
              f'{r["gt_w"]:>5.1f} {r["stride"]:>6.1f} {r["gap"]:>4.1f} '
              f'{r["stride"] / r["mw"]:>9.3f} '
              f'{r["gap"] / r["mw"]:>7.3f}')

    # Group by marker_w bucket.
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['mw']].append(r)

    print('\nGrouped by marker_w bucket:')
    print(f'{"mw":>3s} {"n_rows":>6s} '
          f'{"med_gt_w":>9s} {"med_stride":>10s} {"med_gap":>8s} '
          f'{"med_stride/mw":>13s} {"med_gap/mw":>10s}')
    for mw in sorted(buckets):
        rs   = buckets[mw]
        gws  = [r['gt_w']   for r in rs]
        strs = [r['stride'] for r in rs]
        gaps = [r['gap']    for r in rs]
        print(f'{mw:>3d} {len(rs):>6d} '
              f'{st.median(gws):>9.2f} '
              f'{st.median(strs):>10.2f} '
              f'{st.median(gaps):>8.2f} '
              f'{st.median(strs) / mw:>13.3f} '
              f'{st.median(gaps) / mw:>10.3f}')

    # Aggregate.
    all_gaps      = [r['gap']               for r in rows]
    all_gap_ratio = [r['gap']    / r['mw']  for r in rows]
    all_str_ratio = [r['stride'] / r['mw']  for r in rows]
    all_w_ratio   = [r['gt_w']   / r['mw']  for r in rows]

    print('\nAggregate over all rows:')
    print(f'  n            = {len(rows)}')
    print(f'  gap (px)     min={min(all_gaps):.1f} max={max(all_gaps):.1f} '
          f'median={st.median(all_gaps):.2f} stdev={st.stdev(all_gaps):.2f}')
    print(f'  gap/mw       min={min(all_gap_ratio):.4f} '
          f'max={max(all_gap_ratio):.4f} '
          f'median={st.median(all_gap_ratio):.4f} '
          f'stdev={st.stdev(all_gap_ratio):.4f}')
    print(f'  stride/mw    min={min(all_str_ratio):.4f} '
          f'max={max(all_str_ratio):.4f} '
          f'median={st.median(all_str_ratio):.4f} '
          f'stdev={st.stdev(all_str_ratio):.4f}')
    print(f'  gt_w/mw      min={min(all_w_ratio):.4f} '
          f'max={max(all_w_ratio):.4f} '
          f'median={st.median(all_w_ratio):.4f} '
          f'stdev={st.stdev(all_w_ratio):.4f}')

    print('\nVerdict heuristic:')
    cv_abs = st.stdev(all_gaps)      / st.mean(all_gaps)
    cv_rel = st.stdev(all_gap_ratio) / st.mean(all_gap_ratio)
    print(f'  CV (gap absolute) : {cv_abs:.3f}')
    print(f'  CV (gap / mw)     : {cv_rel:.3f}')
    if cv_abs < cv_rel * 0.7:
        print('  → gap is likely FIXED across UI scales (use 2 px directly)')
    elif cv_rel < cv_abs * 0.7:
        print('  → gap likely SCALES with marker_w (use 2/29 ratio)')
    else:
        print('  → inconclusive')


if __name__ == '__main__':
    main()
