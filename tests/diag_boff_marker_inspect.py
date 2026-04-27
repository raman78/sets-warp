"""
Per-seat visual inspector for BOFF marker detector failures.

For each (file, col, row, expected_spec) case we:
  1. Locate the GT seat (its leftmost ability x_left + y centre yc).
  2. Run detect_markers() on the full image.
  3. Crop a generous patch around the seat (left of x_left, top to
     well below the name bar) and draw on it:
       - the GT ability row bboxes (white, dashed)
       - every detected marker in that crop, color-coded by seat code
         (T/E/S/U) and labelled with seat code + spec code
       - the expected marker region (where the seat's name-bar marker
         should live, computed from x_left/yc/icon dims): red dashed box
       - the classify_stripe() sample window for whichever marker the
         post-hoc classifier picked: yellow dashed box
  4. Upscale the crop ×4 (nearest-neighbour) and save as PNG so each
     pixel is visible.

Output: tests/_diag_out/boff_marker_inspect/<case>.png

Cases come from the user-labelled GT in diag_boff_spec_stripe_gt.py:
the four seats where the post-hoc spec classifier returned "-" because
the seat-type detector found a non-marker CC for that seat.
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from diag_boff_markers import (  # noqa: E402
    detect_markers,
    annotate_specs,
    estimate_icon_dims,
    MAIN_BANDS,
    STRIPE_BANDS,
)

ANN     = ROOT / 'warp' / 'training_data' / 'annotations.json'
SCREENS = Path('/home/raman/STO_screens/screeny')
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'boff_marker_inspect'
VIRTUAL = frozenset({'__empty__', '__inactive__'})

CASES = [
    ('Screenshot_2026-03-05_181717.png', 'L', 0, 'temporal'),
    ('20250719145458_1.jpg',              'R', 1, 'pilot'),
    ('Tessa_Info_3.jpg',                  'L', 0, 'temporal'),
    ('Tessa_Info_3.jpg',                  'L', 2, 'temporal'),
]

# Same palette the main detector viz uses, in BGR.
SEAT_COLOURS = {
    'T': (40,  40,  255),
    'E': (40,  220, 255),
    'S': (255, 100, 40),
    'U': (200, 255, 255),
}
SPEC_COLOURS = {
    'O': (40,  140, 240),  # Command
    'P': (220, 60,  220),  # Intelligence
    'Y': (40,  240, 240),  # Temporal
    'C': (255, 220, 100),  # Pilot
    'L': (60,  240, 40),   # MW
}


def find(fname):
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
        if not items:
            continue
        items.sort(key=lambda b: b[1])
        rows = [[items[0]]]
        for b in items[1:]:
            if abs((b[1] + b[3] / 2) - (rows[-1][0][1] + rows[-1][0][3] / 2)) < icon_h * 0.6:
                rows[-1].append(b)
            else:
                rows.append([b])
        for ri, row in enumerate(rows):
            row.sort(key=lambda b: b[0])
            seats.append((col, ri, row))
    return seats


def draw_dashed_rect(img, p0, p1, color, dash=4, thickness=1):
    x0, y0 = p0
    x1, y1 = p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thickness)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thickness)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thickness)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thickness)


def render_case(fname, col, row, spec, ann_data, out_dir):
    if fname not in ann_data:
        return f'{fname}: missing in annotations'
    boffs = [a['bbox'] for a in ann_data[fname]
             if a.get('state') == 'confirmed'
             and a.get('slot', '').startswith('Boff ')
             and a.get('name') not in VIRTUAL
             and a.get('bbox')]
    if not boffs:
        return f'{fname}: no GT BOFFs'
    path = find(fname)
    if path is None:
        return f'{fname}: not on disk'
    img = cv2.imread(str(path))
    if img is None:
        return f'{fname}: imread failed'

    seats = group_seats(boffs)
    icon_h_real = int(round(st.median(b[3] for b in boffs)))
    icon_w_real = int(round(st.median(b[2] for b in boffs)))
    icon_w_est, icon_h_est = estimate_icon_dims(img)

    match = next((s for s in seats if s[0] == col and s[1] == row), None)
    if match is None:
        return f'{fname} {col} r{row}: seat not found'
    _, _, ability_row = match
    x_left = ability_row[0][0]
    yc = st.median(b[1] + b[3] / 2 for b in ability_row)

    # Crop window: from a bit left of the marker bar to past the
    # rightmost ability, vertically from above the row to well below
    # the name bar.
    crop_x0 = max(0, x_left - int(icon_w_real * 1.6))
    crop_x1 = min(img.shape[1], ability_row[-1][0] + ability_row[-1][2] + int(icon_w_real * 0.4))
    crop_y0 = max(0, int(yc - icon_h_real * 1.0))
    crop_y1 = min(img.shape[0], int(yc + icon_h_real * 2.4))
    crop = img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    # Run detection on full image, filter to crop window
    markers = detect_markers(img, icon_w_est, icon_h_est)
    annotated = annotate_specs(img, markers, icon_w_est)

    def in_crop(x, y, w, h):
        return (x + w >= crop_x0 and x <= crop_x1
                and y + h >= crop_y0 and y <= crop_y1)

    visible = [m for m in annotated if in_crop(m[0], m[1], m[2], m[3])]

    # Same heuristic the verifier uses to pick "the marker for this seat"
    cands = [m for m in annotated
             if abs((m[1] + m[3] / 2) - (yc + icon_h_est * 1.0)) < icon_h_est * 1.0
             and (x_left - icon_w_est * 1.8) < (m[0] + m[2] / 2) < x_left]
    cands.sort(key=lambda m: abs((m[0] + m[2] / 2) - x_left))
    chosen = cands[0] if cands else None

    # GT ability bboxes
    for bx, by, bw, bh in ability_row:
        draw_dashed_rect(
            crop,
            (bx - crop_x0, by - crop_y0),
            (bx - crop_x0 + bw, by - crop_y0 + bh),
            (255, 255, 255), dash=3,
        )

    # Expected marker region (where the seat's actual name-bar marker
    # SHOULD be). Bar geometry: x in [x_left - 0.9 icon_w, x_left - 0.05 icon_w],
    # y in [yc + 0.6 icon_h, yc + 1.4 icon_h].
    ex0 = x_left - int(round(icon_w_real * 0.9))
    ex1 = x_left - int(round(icon_w_real * 0.05))
    ey0 = int(round(yc + icon_h_real * 0.6))
    ey1 = int(round(yc + icon_h_real * 1.4))
    draw_dashed_rect(
        crop,
        (ex0 - crop_x0, ey0 - crop_y0),
        (ex1 - crop_x0, ey1 - crop_y0),
        (60, 60, 255), dash=4, thickness=2,
    )

    # Detected markers in this crop
    for x, y, w, h, code, sp, score in visible:
        c = SEAT_COLOURS.get(code, (255, 255, 255))
        cv2.rectangle(
            crop,
            (x - crop_x0, y - crop_y0),
            (x - crop_x0 + w, y - crop_y0 + h),
            c, 1,
        )
        label = f'{code}'
        if sp:
            label = f'{code}+{sp}({score:.2f})'
        cv2.putText(
            crop, label,
            (x - crop_x0, max(8, y - crop_y0 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA,
        )

    # classify_stripe sample window for the chosen marker
    if chosen is not None:
        x, y, w, h, *_ = chosen
        sx0 = x + w - max(2, int(w * 0.10))
        sx1 = x + w + max(6, int(w * 0.40))
        sy0 = y + max(0, int(h * 0.10))
        sy1 = y + h - max(0, int(h * 0.10))
        draw_dashed_rect(
            crop,
            (sx0 - crop_x0, sy0 - crop_y0),
            (sx1 - crop_x0, sy1 - crop_y0),
            (40, 240, 240), dash=2, thickness=1,
        )

    # Upscale ×4 nearest-neighbour for inspection
    big = cv2.resize(
        crop, (crop.shape[1] * 4, crop.shape[0] * 4),
        interpolation=cv2.INTER_NEAREST,
    )

    # Header banner
    banner_h = 56
    banner = np.full((banner_h, big.shape[1], 3), 30, dtype=np.uint8)
    chosen_str = (
        f'detector picked seat={chosen[4]} '
        f'spec={chosen[5] or "-"} score={chosen[6]:.2f}  '
        f'@ x={chosen[0]} y={chosen[1]} {chosen[2]}x{chosen[3]}'
        if chosen else 'no marker candidate near this seat'
    )
    cv2.putText(banner, f'{fname}  {col} r{row}  expected spec: {spec}',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, chosen_str, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (200, 200, 200), 1, cv2.LINE_AA)

    legend_h = 28
    legend = np.full((legend_h, big.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        legend,
        'white dashes = GT abilities  |  '
        'red dashes = expected marker bar  |  '
        'yellow dashes = stripe sample  |  '
        'coloured rects = detected markers (T/E/S/U)',
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (220, 220, 220), 1, cv2.LINE_AA,
    )

    out_img = np.vstack([banner, big, legend])
    safe = fname.replace('.', '_').replace('/', '_')
    out_path = out_dir / f'{safe}__{col}_r{row}_{spec}.png'
    cv2.imwrite(str(out_path), out_img)

    # Also dump the HSV row through the expected marker bar centre
    txt_path = out_dir / f'{safe}__{col}_r{row}_{spec}.txt'
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ymid = (ey0 + ey1) // 2
    lines = [
        f'{fname}  {col} r{row}  expected={spec}',
        f'image: {img.shape[1]}x{img.shape[0]}',
        f'icon_w_real={icon_w_real} icon_h_real={icon_h_real}  '
        f'(estimate {icon_w_est}x{icon_h_est})',
        f'x_left={x_left}  yc={yc:.1f}',
        f'expected marker bar: x=[{ex0},{ex1}]  y=[{ey0},{ey1}]',
        f'detector chose: {chosen}',
        '',
        f'HSV pixel walk at y={ymid} from x={ex0 - 4} to x={ex1 + 4}:',
    ]
    for px in range(max(0, ex0 - 4), min(img.shape[1], ex1 + 4)):
        H, S, V = hsv[ymid, px]
        lines.append(f'  x={px:4d}  H={int(H):3d}  S={int(S):3d}  V={int(V):3d}')
    txt_path.write_text('\n'.join(lines), encoding='utf-8')

    return f'{out_path.name}  ({chosen_str})'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANN.read_text(encoding='utf-8'))
    print(f'Inspecting {len(CASES)} BOFF marker cases')
    print(f'Output: {OUT_DIR}\n')
    for fname, col, row, spec in CASES:
        msg = render_case(fname, col, row, spec, ann, OUT_DIR)
        print(f'  {msg}')


if __name__ == '__main__':
    main()
