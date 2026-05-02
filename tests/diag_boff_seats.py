"""
BOFF seat-bar detector — read-only prototype.

Each BOFF seat in STO has a curved dark name bar BELOW its 4 ability icons.
We detect those bars by HSV thresholding (dark + saturated) and filter by
shape (wide + short). The 5 detected bars (3 in left col, 2 in right col)
are the anchor for the panel — much more uniform than scanning ability icons.

This replaces dense_scan + sweep_for_boff for window selection. The grid
extrapolation (grid_from_anchor) still consumes the resulting anchor.

Validation: per-screen, compare detected bar y-centres to GT BOFF row
y-centres (median per row). Hit if any detected bar's y is within
icon_h/2 of any GT row centre.

Output: tests/_diag_out/boff_seats.{txt,json}.
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))
ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR  = ROOT / 'tests' / '_diag_out'
OUT_JSON = OUT_DIR / 'boff_seats.json'
OUT_VIZ  = OUT_DIR / 'boff_seats_viz'

VIRTUAL = frozenset({'__empty__', '__inactive__'})


def find_screen_files(fnames):
    candidate_dirs = [
        Path('/home/raman/STO_screens'),
        ROOT / 'warp' / 'training_data' / 'migrated_images',
        ROOT / 'warp' / 'training_data',
    ]
    out = {}
    targets = set(fnames)
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if (p.is_file()
                    and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                    and p.name in targets and p.name not in out):
                out[p.name] = p
    return out


def detect_bars(img, icon_h_est, debug=False):
    """Detect dark+saturated horizontal bars (BOFF seat name bars).

    Returns list of (x, y, w, h) bounding boxes, sorted by y.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Dark + saturated mask
    # V < 70 catches dark; S > 35 excludes pure black/grey UI
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]
    mask = ((V < 70) & (S > 35)).astype(np.uint8) * 255

    # Bar shape: width ≈ 4×pitch_x ≈ 4 × icon_w ≈ 120-200 px
    # height ≈ icon_h × 0.5 ≈ 18-30 px
    # Horizontal close to merge bar interior, vertical erode to drop
    # thin horizontal UI lines.
    kx = max(8, icon_h_est // 2)
    ky = 1
    mask_h = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
    )
    # Drop very thin vertical strips (e.g. single saturated columns)
    mask_h = cv2.morphologyEx(
        mask_h, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, max(2, icon_h_est // 6))),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_h, connectivity=8)
    bars = []
    min_w = max(60, icon_h_est * 2)         # at least 2× icon width
    max_w = max(600, icon_h_est * 12)       # not too wide
    min_h = max(6, icon_h_est // 4)
    max_h = icon_h_est                       # not taller than an icon
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if ww < min_w or ww > max_w:
            continue
        if hh < min_h or hh > max_h:
            continue
        ar = ww / max(hh, 1)
        if ar < 3 or ar > 12:
            continue
        bars.append((int(x), int(y), int(ww), int(hh)))

    bars.sort(key=lambda b: b[1])
    if debug:
        return bars, mask, mask_h
    return bars


def cluster_bars_to_seats(bars, icon_h_est):
    """Cluster bars by y (rows) and x (columns) — return per-row groups.

    Returns list of (x_centre, y_centre, w, h) per detected seat.
    """
    if not bars:
        return []
    # Group by y proximity (gap < icon_h)
    bars_sorted = sorted(bars, key=lambda b: b[1])
    rows = [[bars_sorted[0]]]
    for b in bars_sorted[1:]:
        if b[1] - rows[-1][-1][1] < icon_h_est:
            rows[-1].append(b)
        else:
            rows.append([b])

    seats = []
    for row in rows:
        for b in row:
            seats.append((b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]))
    return seats


def gt_row_centres(boffs, icon_h_est):
    """Return list of (col, y_centre) for each GT seat row.

    col is 'L' or 'R' based on x-position split at the gap.
    """
    if not boffs:
        return []
    cxs = sorted(b[0] + b[2] / 2 for b in boffs)
    # column split: largest gap in cxs
    gaps = [(cxs[i + 1] - cxs[i], i) for i in range(len(cxs) - 1)]
    gap_max, gap_idx = max(gaps)
    split_x = (cxs[gap_idx] + cxs[gap_idx + 1]) / 2

    by_row = {}
    for b in boffs:
        cx = b[0] + b[2] / 2
        cy = b[1] + b[3] / 2
        col = 'L' if cx < split_x else 'R'
        # bucket y to nearest icon_h
        ykey = round(cy / icon_h_est) * icon_h_est
        by_row.setdefault((col, ykey), []).append((cx, cy))

    rows = []
    for (col, _), members in by_row.items():
        y_med = st.median(m[1] for m in members)
        rows.append((col, y_med))
    rows.sort(key=lambda r: r[1])
    return rows


def evaluate_screen(fname, img, boffs):
    """Compute detection vs GT for one screen.

    Returns dict with hit_rows, total_rows, n_bars_detected, etc.
    """
    h, w = img.shape[:2]
    icon_h_est = max(28, int(round(h * 0.047)))

    bars = detect_bars(img, icon_h_est)
    seats = cluster_bars_to_seats(bars, icon_h_est)

    gt_rows = gt_row_centres(boffs, icon_h_est)
    n_rows = len(gt_rows)

    # Bar y-centres (a bar should match the row's BOFF row centre + ~icon_h/2,
    # since bars are below the abilities). Tolerance: icon_h.
    bar_ys = [s[1] for s in seats]
    hit_rows = 0
    for col, gt_y in gt_rows:
        # name bar is ~icon_h*1.0 below the ability row centre
        expected_bar_y = gt_y + icon_h_est * 0.7
        if any(abs(by - expected_bar_y) < icon_h_est for by in bar_ys):
            hit_rows += 1

    return {
        'file':           fname,
        'iw':             w,
        'ih':             h,
        'icon_h_est':     icon_h_est,
        'n_bars':         len(bars),
        'n_seats':        len(seats),
        'gt_n_rows':      n_rows,
        'hit_rows':       hit_rows,
        'hit_pct':        round(hit_rows / max(n_rows, 1) * 100, 1),
        'bars':           [list(b) for b in bars],
        'gt_rows':        [(c, round(y, 1)) for c, y in gt_rows],
    }


def viz(fname, img, bars, gt_rows):
    out = img.copy()
    for x, y, w, h in bars:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    for col, y in gt_rows:
        cv2.line(out, (0, int(y)), (out.shape[1], int(y)), (0, 0, 255), 1)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_VIZ / fname.replace('/', '_')), out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))

    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        boffs = [a['bbox'] for a in entries
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('name') not in VIRTUAL
                 and a.get('bbox')]
        if len(boffs) >= 5:
            candidates.append((fname, boffs))

    paths = find_screen_files([c[0] for c in candidates])
    print(f'Screens with ≥5 BOFF GT: {len(candidates)}, '
          f'findable on disk: {len(paths)}')

    rows = []
    total_gt = 0
    total_hit = 0
    for fname, boffs in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        m = evaluate_screen(fname, img, boffs)
        rows.append(m)
        total_gt  += m['gt_n_rows']
        total_hit += m['hit_rows']

        viz(fname, img, m['bars'], m['gt_rows'])

    rows.sort(key=lambda r: r['hit_pct'])
    print('\nPer-screen results (sorted by hit %):')
    for r in rows:
        bar = '█' * int(r['hit_pct'] / 10)
        print(f'  {r["file"][:42]:<42s}  '
              f'rows={r["hit_rows"]}/{r["gt_n_rows"]} '
              f'({r["hit_pct"]:5.1f}%)  '
              f'bars={r["n_bars"]:3d}  {bar}')

    print('\n' + '=' * 80)
    print(f'Total rows hit: {total_hit}/{total_gt} '
          f'({total_hit / max(total_gt, 1) * 100:.1f}%)')
    print(f'Mean bars detected: {st.mean(r["n_bars"] for r in rows):.1f}')
    print(f'Median bars detected: {st.median(r["n_bars"] for r in rows):.1f}')
    print(f'Visualisations: {OUT_VIZ}')

    OUT_JSON.write_text(json.dumps({'screens': rows}, indent=2),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
