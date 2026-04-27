"""
BOFF panel geometry measurement — read-only.

Pulls all confirmed BOFF ability bboxes from annotations.json and measures:
  pitch_x   — within-row icon spacing (centre-to-centre)
  pitch_y   — row-to-row spacing       (centre-to-centre)
  col_gap   — right-col left-edge minus left-col right-edge (the empty band
              between the two columns)
  icon_w/h  — median bbox dimensions
  panel_w/h — bounding box of all BOFF cells
  ratios    — pitch_y / icon_h, col_gap / pitch_x (used by `grid_from_anchor`)

Validates whether the constants hard-coded in
`tests/diag_boff_scan.grid_from_anchor()`:
    pitch_y = icon_h × 2.2
    col_gap = pitch_x × 2.5
hold across all 34 screens or are calibrated to a single ekran.

Output goes to `tests/_diag_out/boff_geometry.{txt,json}`.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))
ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR  = ROOT / 'tests' / '_diag_out'
OUT_TXT  = OUT_DIR / 'boff_geometry.txt'
OUT_JSON = OUT_DIR / 'boff_geometry.json'

VIRTUAL = frozenset({'__empty__', '__inactive__'})


def cluster_1d(values, gap_threshold):
    """Group sorted values into clusters where consecutive gap < threshold."""
    if not values:
        return []
    vs = sorted(values)
    out = [[vs[0]]]
    for v in vs[1:]:
        if v - out[-1][-1] < gap_threshold:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def measure_screen(fname, entries, image_dims=None):
    """Measure geometry from BOFF GT bboxes of one screen.

    Returns dict with keys: pitch_x, pitch_y, col_gap, icon_w, icon_h,
    panel_w, panel_h, n_rows_left, n_rows_right, n_boffs, plus normalized
    versions (suffix _norm) when image_dims is provided.
    """
    boffs = [a for a in entries
             if a.get('state') == 'confirmed'
             and a.get('slot', '').startswith('Boff ')
             and a.get('name') not in VIRTUAL
             and a.get('bbox')]
    if len(boffs) < 4:
        return None

    bboxes = [a['bbox'] for a in boffs]   # [x, y, w, h]
    cxs = [b[0] + b[2] / 2 for b in bboxes]
    cys = [b[1] + b[3] / 2 for b in bboxes]
    icon_ws = [b[2] for b in bboxes]
    icon_hs = [b[3] for b in bboxes]

    icon_w_med = st.median(icon_ws)
    icon_h_med = st.median(icon_hs)

    # Cluster columns: gap > 1.5 × icon_w means we're across columns
    col_clusters = cluster_1d(cxs, gap_threshold=icon_w_med * 1.5)
    if len(col_clusters) != 2:
        # 1 column visible (or odd structure) — skip column metrics
        col_left = col_clusters[0] if col_clusters else []
        col_right = []
    else:
        col_left, col_right = col_clusters

    # Cluster rows within each column: gap > 0.5 × icon_h means new row
    def _row_clusters(col_xs, all_bboxes):
        col_xs_set = set(col_xs)
        ys = [b[1] + b[3] / 2 for b in all_bboxes
              if (b[0] + b[2] / 2) in col_xs_set]
        return cluster_1d(ys, gap_threshold=icon_h_med * 0.5)

    rows_l = _row_clusters(col_left, bboxes)
    rows_r = _row_clusters(col_right, bboxes)

    # pitch_x: within a row of 4 abilities, median of deltas of sorted cxs
    pitch_xs = []
    for col in (col_left, col_right):
        # group bboxes by row using y-distance < 0.5 icon_h
        rows = cluster_1d([(b[1] + b[3] / 2) for b in bboxes
                           if (b[0] + b[2] / 2) in set(col)],
                          gap_threshold=icon_h_med * 0.5)
        for row_ys in rows:
            row_set = set(row_ys)
            row_bboxes = [b for b in bboxes
                          if (b[1] + b[3] / 2) in row_set
                          and (b[0] + b[2] / 2) in set(col)]
            if len(row_bboxes) < 2:
                continue
            row_bboxes.sort(key=lambda b: b[0])
            row_cxs = [b[0] + b[2] / 2 for b in row_bboxes]
            for a, b in zip(row_cxs, row_cxs[1:]):
                pitch_xs.append(b - a)
    pitch_x_med = st.median(pitch_xs) if pitch_xs else None

    # pitch_y: median of consecutive row-centre deltas (in either column)
    pitch_ys = []
    for col_rows in (rows_l, rows_r):
        if len(col_rows) < 2:
            continue
        row_centers = sorted(st.mean(r) for r in col_rows)
        for a, b in zip(row_centers, row_centers[1:]):
            pitch_ys.append(b - a)
    pitch_y_med = st.median(pitch_ys) if pitch_ys else None

    # col_gap: right-col leftmost x - left-col rightmost x (right-edge of left
    # icon to left-edge of right icon)
    col_gap = None
    if col_left and col_right:
        left_right_edges = [(b[0] + b[2]) for b in bboxes
                            if (b[0] + b[2] / 2) in set(col_left)]
        right_left_edges = [b[0] for b in bboxes
                            if (b[0] + b[2] / 2) in set(col_right)]
        if left_right_edges and right_left_edges:
            col_gap = min(right_left_edges) - max(left_right_edges)

    # Panel bbox
    panel_x0 = min(b[0]                for b in bboxes)
    panel_y0 = min(b[1]                for b in bboxes)
    panel_x1 = max(b[0] + b[2]         for b in bboxes)
    panel_y1 = max(b[1] + b[3]         for b in bboxes)
    panel_w  = panel_x1 - panel_x0
    panel_h  = panel_y1 - panel_y0

    out = {
        'file':         fname,
        'n_boffs':      len(bboxes),
        'n_rows_left':  len(rows_l),
        'n_rows_right': len(rows_r),
        'icon_w':       icon_w_med,
        'icon_h':       icon_h_med,
        'pitch_x':      pitch_x_med,
        'pitch_y':      pitch_y_med,
        'col_gap':      col_gap,
        'panel_w':      panel_w,
        'panel_h':      panel_h,
        'panel_x0':     panel_x0,
        'panel_y0':     panel_y0,
    }

    # Ratios used by grid_from_anchor
    if pitch_x_med:
        out['pitch_x_over_icon_w'] = pitch_x_med / icon_w_med
    if pitch_y_med:
        out['pitch_y_over_icon_h'] = pitch_y_med / icon_h_med
    if col_gap is not None and pitch_x_med:
        out['col_gap_over_pitch_x'] = col_gap / pitch_x_med

    if image_dims:
        iw, ih = image_dims
        out['iw'] = iw
        out['ih'] = ih
        out['icon_w_norm'] = icon_w_med / iw
        out['icon_h_norm'] = icon_h_med / ih
        out['panel_w_norm'] = panel_w / iw
        out['panel_h_norm'] = panel_h / ih
        if pitch_x_med:
            out['pitch_x_norm'] = pitch_x_med / iw
        if pitch_y_med:
            out['pitch_y_norm'] = pitch_y_med / ih
        if col_gap is not None:
            out['col_gap_norm'] = col_gap / iw

    return out


def fmt_dist(name, vals, fmt='{:.3f}'):
    if not vals:
        return f'  {name:24s}  (no data)'
    median = st.median(vals)
    mn, mx = min(vals), max(vals)
    stdev = st.stdev(vals) if len(vals) > 1 else 0.0
    return (f'  {name:24s}  n={len(vals):2d}  '
            f'median={fmt.format(median)}  '
            f'mean={fmt.format(st.mean(vals))}  '
            f'std={fmt.format(stdev)}  '
            f'min={fmt.format(mn)}  max={fmt.format(mx)}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))

    # Load image dims by reading them lazily (skip if file unavailable)
    import cv2
    candidate_dirs = [
        Path('/home/raman/STO_screens'),
        ROOT / 'warp' / 'training_data' / 'migrated_images',
        ROOT / 'warp' / 'training_data',
    ]
    img_idx = {}
    fnames_set = {f for f, e in ann.items() if isinstance(e, list)}
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if (p.is_file()
                    and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                    and p.name in fnames_set
                    and p.name not in img_idx):
                img_idx[p.name] = p

    rows = []
    skipped = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        image_dims = None
        path = img_idx.get(fname)
        if path is not None:
            img = cv2.imread(str(path))
            if img is not None:
                image_dims = (img.shape[1], img.shape[0])
        m = measure_screen(fname, entries, image_dims=image_dims)
        if m is None:
            continue
        if m['n_boffs'] < 5:
            skipped.append((fname, m['n_boffs']))
            continue
        rows.append(m)

    print(f'Screens with ≥5 BOFF GT: {len(rows)}')
    print(f'Screens skipped (<5 BOFF): {len(skipped)}')
    print()

    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    print('=== absolute pixel measurements ===')
    print(fmt_dist('icon_w (px)',         col('icon_w'),     '{:5.1f}'))
    print(fmt_dist('icon_h (px)',         col('icon_h'),     '{:5.1f}'))
    print(fmt_dist('pitch_x (px)',        col('pitch_x'),    '{:5.1f}'))
    print(fmt_dist('pitch_y (px)',        col('pitch_y'),    '{:5.1f}'))
    print(fmt_dist('col_gap (px)',        col('col_gap'),    '{:5.1f}'))
    print(fmt_dist('panel_w (px)',        col('panel_w'),    '{:5.0f}'))
    print(fmt_dist('panel_h (px)',        col('panel_h'),    '{:5.0f}'))
    print()
    print('=== ratios (these drive grid_from_anchor constants) ===')
    print(fmt_dist('pitch_x / icon_w',    col('pitch_x_over_icon_w'), '{:5.3f}'))
    print(fmt_dist('pitch_y / icon_h',    col('pitch_y_over_icon_h'), '{:5.3f}'))
    print(fmt_dist('col_gap / pitch_x',   col('col_gap_over_pitch_x'),'{:5.3f}'))
    print()
    print('=== normalised by image dims (if image found) ===')
    print(fmt_dist('icon_w_norm  (×iw)',  col('icon_w_norm'),  '{:7.4f}'))
    print(fmt_dist('icon_h_norm  (×ih)',  col('icon_h_norm'),  '{:7.4f}'))
    print(fmt_dist('pitch_x_norm (×iw)',  col('pitch_x_norm'), '{:7.4f}'))
    print(fmt_dist('pitch_y_norm (×ih)',  col('pitch_y_norm'), '{:7.4f}'))
    print(fmt_dist('col_gap_norm (×iw)',  col('col_gap_norm'), '{:7.4f}'))
    print(fmt_dist('panel_w_norm (×iw)',  col('panel_w_norm'), '{:7.4f}'))
    print(fmt_dist('panel_h_norm (×ih)',  col('panel_h_norm'), '{:7.4f}'))
    print()

    print('=== row layout ===')
    n_rows_pairs = [(r['n_rows_left'], r['n_rows_right']) for r in rows]
    from collections import Counter
    counter = Counter(n_rows_pairs)
    for (l, r_), n in sorted(counter.items(), key=lambda x: -x[1]):
        print(f'  ({l}, {r_})    {n}/{len(rows)}')
    print()

    print('=== per-screen (for outliers) ===')
    rows.sort(key=lambda r: r.get('pitch_y_over_icon_h') or 0)
    for r in rows:
        py = r.get('pitch_y_over_icon_h')
        cg = r.get('col_gap_over_pitch_x')
        print(f'  {r["file"][:42]:<42s}  '
              f'n={r["n_boffs"]:2d}  '
              f'L={r["n_rows_left"]} R={r["n_rows_right"]}  '
              f'icon=({r["icon_w"]:4.0f}×{r["icon_h"]:4.0f})  '
              f'pitch=({(r["pitch_x"] or 0):5.1f},{(r["pitch_y"] or 0):5.1f})  '
              f'col_gap={(r["col_gap"] or 0):5.1f}  '
              f'py/h={py:5.3f}  cg/px={(cg if cg else 0):5.3f}')

    OUT_JSON.write_text(json.dumps({'screens': rows}, indent=2), encoding='utf-8')
    print(f'\nFull dump: {OUT_JSON}')


if __name__ == '__main__':
    main()