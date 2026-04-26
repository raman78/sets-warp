"""
Sample HSV at the LEFT edge of each BOFF seat across several screens.

For each GT-confirmed BOFF panel:
1. Group abilities into rows (by y proximity) and columns (left/right via x split).
2. For each (col, row) seat, find leftmost ability x.
3. Sample a strip just left of it (x_left - 0.6*icon_w .. x_left - 0.05*icon_w),
   y = row centre +/- icon_h/2.
4. Print median HSV and dominant H bins.

Goal: characterise the profession marker colours so we can build a detector.
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))
ANN = ROOT / 'warp' / 'training_data' / 'annotations.json'

VIRTUAL = frozenset({'__empty__', '__inactive__'})
SCREEN_DIR = Path('/home/raman/STO_screens')

TARGETS = [
    'Ambassador-broadside.png',
    'Pumwl1.png',
    'Chronos-broadside.png',
    'Nautilus.png',
    'broadside.png',
]


def find_image(fname):
    for p in SCREEN_DIR.rglob(fname):
        if p.is_file():
            return p
    for p in (ROOT / 'warp' / 'training_data').rglob(fname):
        if p.is_file():
            return p
    return None


def group_seats(boffs):
    """Return [(col, row_idx, [bboxes sorted by x], y_centre)]."""
    if not boffs:
        return []
    cxs = sorted(b[0] + b[2] / 2 for b in boffs)
    gaps = [(cxs[i + 1] - cxs[i], i) for i in range(len(cxs) - 1)]
    _, gi = max(gaps)
    split_x = (cxs[gi] + cxs[gi + 1]) / 2

    icon_h = st.median(b[3] for b in boffs)

    def col_of(b):
        return 'L' if (b[0] + b[2] / 2) < split_x else 'R'

    by_col = {'L': [], 'R': []}
    for b in boffs:
        by_col[col_of(b)].append(b)

    seats = []
    for col, items in by_col.items():
        items.sort(key=lambda b: (b[1], b[0]))
        rows = [[items[0]]]
        for b in items[1:]:
            if abs((b[1] + b[3] / 2) - (rows[-1][0][1] + rows[-1][0][3] / 2)) < icon_h * 0.6:
                rows[-1].append(b)
            else:
                rows.append([b])
        for r_idx, row in enumerate(rows):
            row.sort(key=lambda b: b[0])
            yc = st.median(b[1] + b[3] / 2 for b in row)
            seats.append((col, r_idx, row, yc))
    return seats


def sample_hsv(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv


def describe_hsv(hsv):
    if hsv is None or hsv.size == 0:
        return 'empty'
    H = hsv[:, :, 0].flatten()
    S = hsv[:, :, 1].flatten()
    V = hsv[:, :, 2].flatten()
    # mask: only "coloured" pixels (S>50, V>60) — exclude background
    m = (S > 50) & (V > 60)
    if m.sum() < 5:
        return f'no-colour (Smed={int(np.median(S))}, Vmed={int(np.median(V))})'
    Hc = H[m]
    # Dominant H bin (10° buckets)
    bins = np.bincount(Hc // 10, minlength=18)
    top_bin = int(bins.argmax())
    pct = bins[top_bin] / m.sum() * 100
    return (f'Hmed={int(np.median(Hc))} (peak {top_bin*10}-{top_bin*10+10}, '
            f'{pct:.0f}%) Smed={int(np.median(S[m]))} Vmed={int(np.median(V[m]))} '
            f'n_col={int(m.sum())}/{m.size}')


def main():
    ann = json.loads(ANN.read_text(encoding='utf-8'))
    for fname in TARGETS:
        if fname not in ann:
            print(f'\n[skip] {fname} — not in annotations')
            continue
        boffs = [a['bbox'] for a in ann[fname]
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('name') not in VIRTUAL
                 and a.get('bbox')]
        if not boffs:
            print(f'\n[skip] {fname} — no BOFFs')
            continue

        path = find_image(fname)
        if path is None:
            print(f'\n[skip] {fname} — file not found')
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f'\n[skip] {fname} — imread failed')
            continue

        seats = group_seats(boffs)
        icon_h = int(round(st.median(b[3] for b in boffs)))
        icon_w = int(round(st.median(b[2] for b in boffs)))

        print(f'\n=== {fname}  iw={img.shape[1]} ih={img.shape[0]}'
              f'  icon_w={icon_w} icon_h={icon_h}  seats={len(seats)} ===')

        for col, r, row, yc in seats:
            x_left = row[0][0]
            # Sample LEFT-of-name-bar: name bar sits BELOW the ability row,
            # roughly at y = yc + icon_h * 0.6 to yc + icon_h * 1.4.
            sx0 = x_left - int(icon_w * 0.9)
            sx1 = x_left - int(icon_w * 0.05)
            sy0 = int(yc + icon_h * 0.6)
            sy1 = int(yc + icon_h * 1.4)
            hsv = sample_hsv(img, sx0, sx1, sy0, sy1)
            print(f'  {col} r{r} yc={int(yc)} x_left={x_left}  '
                  f'name_bar=({sx0}..{sx1}, {sy0}..{sy1})  -> {describe_hsv(hsv)}')


if __name__ == '__main__':
    main()
