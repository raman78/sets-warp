"""
Sample BOFF spec-stripe HSV at user-labelled (col, row, spec) seats.

Goal: resolve the Pilot-vs-Command TBD in the spec-stripe palette
(both currently catalogued as "orange, H 8-18"). User supplied
ground-truth (col, row, spec) labels for ~17 seats across 9 screens.
For each labelled seat we walk the right edge of the marker bar and
print the HSV distribution of the coloured pixels — then group by
spec label to see whether Pilot and Command separate cleanly.
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

ANN     = ROOT / 'warp' / 'training_data' / 'annotations.json'
SCREENS = Path('/home/raman/STO_screens/screeny')
VIRTUAL = frozenset({'__empty__', '__inactive__'})

# (filename, col, row_idx_0based, spec)  — from user message 2026-04-26
GT = [
    ('Screenshot_2025-09-25_104040.png', 'L', 1, 'pilot'),
    ('Screenshot_2025-09-08_194542.png', 'L', 2, 'command'),
    ('Screenshot_2025-09-25_111530.png', 'L', 2, 'command'),
    ('Screenshot_2026-03-05_181717.png', 'L', 0, 'temporal'),
    ('Screenshot_2026-03-05_181717.png', 'L', 1, 'intel'),
    ('20250719145458_1.jpg',              'L', 1, 'command'),
    ('20250719145458_1.jpg',              'R', 1, 'pilot'),
    ('FaW_seating.jpg',                   'L', 1, 'command'),
    ('FaW_seating.jpg',                   'R', 1, 'pilot'),
    ('Screenshot_2026-02-06_024815.png',  'R', 1, 'mw'),
    ('Screenshot_2026-02-06_024815.png',  'L', 2, 'intel'),
    ('Stations.png',                      'L', 0, 'pilot'),
    ('Stations.png',                      'L', 1, 'command'),
    ('Tessa_Info_3.jpg',                  'L', 0, 'temporal'),
    ('Tessa_Info_3.jpg',                  'L', 2, 'temporal'),
]


def find_image(fname):
    for p in SCREENS.rglob(fname):
        if p.is_file():
            return p
    return None


def group_seats(boffs):
    cxs = sorted(b[0] + b[2] / 2 for b in boffs)
    if len(cxs) < 2:
        return []
    gaps = [(cxs[i + 1] - cxs[i], i) for i in range(len(cxs) - 1)]
    _, gi = max(gaps)
    split_x = (cxs[gi] + cxs[gi + 1]) / 2
    icon_h = st.median(b[3] for b in boffs)

    by_col = {'L': [], 'R': []}
    for b in boffs:
        col = 'L' if (b[0] + b[2] / 2) < split_x else 'R'
        by_col[col].append(b)

    seats = []
    for col, items in by_col.items():
        items.sort(key=lambda b: b[1])
        if not items:
            continue
        rows = [[items[0]]]
        for b in items[1:]:
            if abs((b[1] + b[3] / 2) - (rows[-1][0][1] + rows[-1][0][3] / 2)) < icon_h * 0.6:
                rows[-1].append(b)
            else:
                rows.append([b])
        for r_idx, row in enumerate(rows):
            row.sort(key=lambda b: b[0])
            yc = st.median(b[1] + b[3] / 2 for b in row)
            seats.append((col, r_idx, row[0][0], yc))
    return seats


def sample_stripe(img, x_left, yc, icon_w, icon_h):
    """Sample the rightmost ~25% of the marker bar.

    Marker spans x in [x_left - 0.9*icon_w, x_left - 0.05*icon_w], so the
    right-edge stripe is the last 25% of that range.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bar_x0 = x_left - int(round(icon_w * 0.9))
    bar_x1 = x_left - int(round(icon_w * 0.05))
    bar_w  = bar_x1 - bar_x0
    sx0 = bar_x1 - int(round(bar_w * 0.25))
    sx1 = bar_x1
    sy0 = int(round(yc + icon_h * 0.7))
    sy1 = int(round(yc + icon_h * 1.3))
    h, w = img.shape[:2]
    sx0, sx1 = max(0, sx0), min(w, sx1)
    sy0, sy1 = max(0, sy0), min(h, sy1)
    crop = hsv[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return None
    H = crop[:, :, 0].flatten()
    S = crop[:, :, 1].flatten()
    V = crop[:, :, 2].flatten()
    m = (S > 60) & (V > 70)
    if m.sum() < 5:
        return None
    return {
        'h_med': int(np.median(H[m])),
        's_med': int(np.median(S[m])),
        'v_med': int(np.median(V[m])),
        'h_lo':  int(np.percentile(H[m], 10)),
        'h_hi':  int(np.percentile(H[m], 90)),
        'n':     int(m.sum()),
        'box':   (sx0, sy0, sx1, sy1),
    }


def main():
    ann = json.loads(ANN.read_text(encoding='utf-8'))
    by_spec = {}
    print(f'{"file":<42s} {"col":<3s} {"row":<3s} {"spec":<8s} '
          f'{"Hmed":>4s} {"H10-H90":>9s} {"Smed":>5s} {"Vmed":>5s}  n')
    print('-' * 100)

    for fname, col, row, spec in GT:
        if fname not in ann:
            print(f'{fname:<42s} -- not in annotations')
            continue
        boffs = [a['bbox'] for a in ann[fname]
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('name') not in VIRTUAL
                 and a.get('bbox')]
        if not boffs:
            print(f'{fname:<42s} -- no GT BOFFs')
            continue
        path = find_image(fname)
        if path is None:
            print(f'{fname:<42s} -- not on disk')
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f'{fname:<42s} -- imread failed')
            continue
        seats  = group_seats(boffs)
        icon_h = int(round(st.median(b[3] for b in boffs)))
        icon_w = int(round(st.median(b[2] for b in boffs)))
        match = next((s for s in seats if s[0] == col and s[1] == row), None)
        if match is None:
            print(f'{fname:<42s} {col} r{row}  -- seat not found '
                  f'(have {[(c, r) for c, r, _, _ in seats]})')
            continue
        _, _, x_left, yc = match
        stat = sample_stripe(img, x_left, yc, icon_w, icon_h)
        if stat is None:
            print(f'{fname:<42s} {col:<3s} {row:<3d} {spec:<8s} -- no coloured pixels')
            continue
        print(f'{fname:<42s} {col:<3s} {row:<3d} {spec:<8s} '
              f'{stat["h_med"]:>4d} {stat["h_lo"]:>3d}-{stat["h_hi"]:<3d}  '
              f'{stat["s_med"]:>5d} {stat["v_med"]:>5d}  {stat["n"]}')
        by_spec.setdefault(spec, []).append(stat)

    print()
    print('=== summary by spec ===')
    for spec, stats in sorted(by_spec.items()):
        hs = [s['h_med'] for s in stats]
        ss = [s['s_med'] for s in stats]
        vs = [s['v_med'] for s in stats]
        print(f'  {spec:<10s} n={len(stats):2d}  '
              f'H med={int(st.median(hs)):>3d} (range {min(hs)}-{max(hs)})  '
              f'S med={int(st.median(ss)):>3d}  '
              f'V med={int(st.median(vs)):>3d}')


if __name__ == '__main__':
    main()
