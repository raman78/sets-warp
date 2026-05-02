"""
Inspect horizontal HSV profile across the BOFF seat name-bar marker.

Per user observation: a specialization seat shows main profession colour
across ~95% of the marker bar width, with a thin (~5%) coloured stripe
indicating the spec (MW, Intel, Temporal, Pilot, Command) — likely on
one of the bar edges.

Universal seats also have their own marker colour.

For each screen, for each GT seat, walk x from leftmost ability minus
~1.0 icon_w towards x_left, at the name-bar y, and dump (H, S, V)
samples per pixel column to spot multi-zone structure.
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

# Pick screens likely to have spec seats:
TARGETS = [
    'Ambassador-broadside.png',
    'Pumwl1.png',
    'Chronos-broadside.png',
    'Nautilus.png',
    'broadside.png',
    'Tessa_Info_3.jpg',
    'Stations.png',
    'image1.png',  # has 4 boffs — possibly mixed
    'TEST.png',
    'image3.png',
]


def find_image(fname):
    for d in (SCREEN_DIR, ROOT / 'warp' / 'training_data' / 'migrated_images',
              ROOT / 'warp' / 'training_data'):
        if not d.exists():
            continue
        for p in d.rglob(fname):
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


def hue_label(h, s, v):
    if s < 50 and v < 80:
        return '·'   # background/dark
    if s < 30:
        return '_'   # grey/white
    if h < 8 or h >= 170:
        return 'R'   # red (Tactical/Command)
    if 8 <= h < 18:
        return 'O'   # orange
    if 18 <= h < 36:
        return 'Y'   # yellow/gold (Engineering)
    if 36 <= h < 50:
        return 'L'   # lime
    if 50 <= h < 75:
        return 'G'   # green (Miracle Worker)
    if 75 <= h < 92:
        return 'C'   # cyan (Pilot)
    if 92 <= h < 118:
        return 'B'   # blue (Science / Temporal)
    if 118 <= h < 145:
        return 'P'   # purple (Intel)
    if 145 <= h < 170:
        return 'M'   # magenta
    return '?'


def main():
    ann = json.loads(ANN.read_text(encoding='utf-8'))
    for fname in TARGETS:
        if fname not in ann:
            continue
        boffs = [a['bbox'] for a in ann[fname]
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('name') not in VIRTUAL
                 and a.get('bbox')]
        if not boffs:
            continue

        path = find_image(fname)
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue

        seats = group_seats(boffs)
        icon_h = int(round(st.median(b[3] for b in boffs)))
        icon_w = int(round(st.median(b[2] for b in boffs)))

        print(f'\n=== {fname}  iw={img.shape[1]} ih={img.shape[0]}'
              f'  icon_w={icon_w} icon_h={icon_h} ===')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for col, r, x_left, yc in seats:
            # Name bar y range — sample 3 rows in the middle of the bar
            ny0 = int(yc + icon_h * 0.7)
            ny1 = int(yc + icon_h * 1.3)
            yc_mid = (ny0 + ny1) // 2
            # Walk x from x_left - 1.2*icon_w to x_left
            xs0 = max(0, x_left - int(icon_w * 1.2))
            xs1 = max(0, x_left - 1)

            label_row = []
            hsv_row = []
            for x in range(xs0, xs1):
                # average a 3x3 box around (x, yc_mid)
                y0, y1 = max(0, yc_mid - 1), min(hsv.shape[0], yc_mid + 2)
                xs_, xe_ = max(0, x - 1), min(hsv.shape[1], x + 2)
                patch = hsv[y0:y1, xs_:xe_].reshape(-1, 3)
                m = (patch[:, 1] > 30) & (patch[:, 2] > 40)
                if m.sum() == 0:
                    label_row.append('·')
                    hsv_row.append((0, 0, 0))
                    continue
                med = np.median(patch[m], axis=0).astype(int)
                label_row.append(hue_label(int(med[0]), int(med[1]), int(med[2])))
                hsv_row.append(tuple(int(v) for v in med))

            # Find runs of distinct colour labels
            runs = []
            for ch in label_row:
                if runs and runs[-1][0] == ch:
                    runs[-1] = (ch, runs[-1][1] + 1)
                else:
                    runs.append((ch, 1))

            print(f'  {col} r{r} y={int(yc_mid)} '
                  f'x[{xs0}..{xs1}]  ({xs1-xs0}px)')
            print(f'    labels: {"".join(label_row)}')
            # Show colour runs ≥3 px wide
            run_str = ' '.join(f'{ch}×{n}' for ch, n in runs if n >= 2)
            print(f'    runs:   {run_str}')


if __name__ == '__main__':
    main()
