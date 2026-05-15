"""Trace which filter rejects the Engineering CC at x=252 y=519 on
overview.png inside detect_markers()."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warp.recognition import boff_marker as bm  # noqa: E402

IMG = Path('warp/training_data/screen_types/SPACE_MIXED/overview.png')


def main() -> None:
    img = cv2.imread(str(IMG))
    icon_w, icon_h = bm.estimate_icon_dims(img)
    print(f'icon_w={icon_w} icon_h={icon_h}')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ENG band (gold)
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = 18, 30, 100, 220, 160, 255
    raw_m = bm._colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
    kx = max(3, int(round(icon_w * 0.22)))
    ky = max(2, icon_h // 12)
    closed_m = cv2.morphologyEx(
        raw_m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
    )
    print(f'kx={kx} ky={ky}')

    # The CC of interest (already known)
    target_box = (252, 519, 28, 28)
    print(f'Target Eng marker bbox: {target_box}')

    # Recompute thresholds exactly as detect_markers does
    abs_min_w, abs_min_h = 10, 12
    abs_max_w, abs_max_h = 90, 90
    min_w = max(abs_min_w, int(icon_w * 0.45))
    max_w = min(abs_max_w, max(int(icon_w * 1.6), 44))
    min_h = max(abs_min_h, int(icon_h * 0.45))
    max_h = min(abs_max_h, max(int(icon_h * 1.35), 44))
    ar_min, ar_max = 0.30, 1.8
    fill_min = 0.70
    uni_v_max = 28
    uni_h_max = 6
    edge_max = 0.07
    edge_inset = 2
    print(f'thresholds: w[{min_w},{max_w}] h[{min_h},{max_h}] ar[{ar_min},{ar_max}] fill≥{fill_min} v_std≤{uni_v_max} h_std≤{uni_h_max} edge≤{edge_max}')

    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

    # Check candidates from BOTH raw and closed masks
    for label, src in [('raw', raw_m), ('closed', closed_m)]:
        n, _, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        # Find CC near target
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i, [0, 1, 2, 3, 4]])
            # Overlap with target box?
            tx, ty, tw, th = target_box
            ix = min(x + w, tx + tw) - max(x, tx)
            iy = min(y + h, ty + th) - max(y, ty)
            if ix <= 0 or iy <= 0:
                continue
            print(f'\n[{label}] CC x={x} y={y} w={w} h={h} area={area}')
            ar = w / max(h, 1)
            print(f'  ar={ar:.2f}, fill={area/(w*h):.3f}')
            if w < min_w or w > max_w:
                print(f'  REJECT: w {w} out of [{min_w},{max_w}]')
            if h < min_h or h > max_h:
                print(f'  REJECT: h {h} out of [{min_h},{max_h}]')
            if ar < ar_min or ar > ar_max:
                print(f'  REJECT: ar {ar:.2f} out of [{ar_min},{ar_max}]')
            if area < (w * h) * fill_min:
                print(f'  REJECT: fill {area/(w*h):.3f} < {fill_min}')
            sel = raw_m[y:y + h, x:x + w] > 0
            print(f'  sel.sum (raw) = {sel.sum()}')
            if sel.sum() >= 20:
                crop_v = hsv[y:y + h, x:x + w, 2][sel]
                crop_h = hsv[y:y + h, x:x + w, 0][sel]
                v_std = float(np.std(crop_v))
                h_std_raw = float(np.std(crop_h))
                h_std_wrap = float(np.std((crop_h.astype(np.int32) + 90) % 180))
                h_std = min(h_std_raw, h_std_wrap)
                print(f'  v_std={v_std:.2f} (max {uni_v_max}), h_std={h_std:.2f} (max {uni_h_max})')
                if v_std > uni_v_max:
                    print('  REJECT: v_std')
                if h_std > uni_h_max:
                    print('  REJECT: h_std')
            ix0 = x + edge_inset; iy0 = y + edge_inset
            ix1 = x + w - edge_inset; iy1 = y + h - edge_inset
            if ix1 > ix0 and iy1 > iy0:
                ec = edges[iy0:iy1, ix0:ix1]
                ef = float(ec.sum() / 255.0) / ((ix1 - ix0) * (iy1 - iy0))
                print(f'  edge_frac={ef:.3f} (max {edge_max})')
                if ef > edge_max:
                    print('  REJECT: edge')

    # Now also try _merge_close_bboxes on raw CCs and see what merged blob
    # the target ends up in.
    print('\n--- After _merge_close_bboxes (raw mask) ---')
    n, _, stats, _ = cv2.connectedComponentsWithStats(raw_m, connectivity=8)
    raw_ccs = [tuple(int(v) for v in stats[i, [0, 1, 2, 3, 4]]) for i in range(1, n)]
    gap_x = max(2, int(round(icon_w * 0.35)))
    print(f'gap_x={gap_x}')
    merged = bm._merge_close_bboxes(raw_ccs, gap_x=gap_x, overlap_y_frac=0.55)
    for x, y, w, h, area in merged:
        tx, ty, tw, th = target_box
        ix = min(x + w, tx + tw) - max(x, tx)
        iy = min(y + h, ty + th) - max(y, ty)
        if ix <= 0 or iy <= 0:
            continue
        print(f'  merged: x={x} y={y} w={w} h={h} area={area} ar={w/h:.2f} fill={area/(w*h):.3f}')


if __name__ == '__main__':
    main()
