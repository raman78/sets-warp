"""Diagnose why Engineering+MW marker on overview.png (right col row 2)
is missed by boff_marker.detect_panel().

Read-only diagnostic. Inspects:
 1) all raw markers from detect_markers()
 2) HSV stats inside the expected Eng+MW bbox (right col, level of left
    col Tactical marker at y≈522)
 3) Engineering MAIN_BANDS mask hit count + spec stripe (MW) match
"""
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
    print(f'img: {img.shape}')
    icon_w, icon_h = bm.estimate_icon_dims(img)
    print(f'estimate_icon_dims: icon_w={icon_w} icon_h={icon_h}')

    markers = bm.detect_markers(img, icon_w, icon_h)
    print(f'\ndetect_markers: {len(markers)} markers')
    for i, m in enumerate(markers):
        x, y, w, h, code = m
        print(f'  [{i}] x={x} y={y} w={w} h={h} code={code}')

    panel = bm.detect_panel(img)
    if panel is None:
        print('detect_panel: None')
        return
    print(f"\npanel score={panel['score']:.3f}")
    print(f"col_a ({len(panel['col_a'])}):")
    for m in panel['col_a']:
        print(f'  {m}')
    print(f"col_b ({len(panel['col_b'])}):")
    for m in panel['col_b']:
        print(f'  {m}')

    # Refine icon dims (same as in detect_panel)
    iw, ih = bm._refine_dims_from_markers(markers, icon_w, icon_h)
    print(f'\nrefined icon_w={iw} icon_h={ih}')

    # Expected Eng+MW marker: right column, row matching left-col Tactical
    # (the "missing 5th seat"). From summary: left col row 1 (Tactical) is
    # at x=65 y=522 w=30 h=24. Right col anchor x=252.
    # Predict: x≈252, y≈522. Probe a window around it.
    probe_cx = 252 + 30 // 2
    probe_cy = 522 + 24 // 2
    probe_w = 60   # generous window
    probe_h = 50
    px0 = max(0, probe_cx - probe_w // 2)
    py0 = max(0, probe_cy - probe_h // 2)
    px1 = px0 + probe_w
    py1 = py0 + probe_h

    print(f'\nProbe window: x=[{px0}, {px1}) y=[{py0}, {py1})')
    crop = img[py0:py1, px0:px1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    print(f'  HSV mean: H={hsv[:,:,0].mean():.1f} S={hsv[:,:,1].mean():.1f} V={hsv[:,:,2].mean():.1f}')
    print(f'  HSV  max: H={int(hsv[:,:,0].max())} S={int(hsv[:,:,1].max())} V={int(hsv[:,:,2].max())}')

    # Test each MAIN_BAND mask coverage in the probe
    print('\nMAIN_BAND mask coverage in probe window:')
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in bm.MAIN_BANDS:
        m = bm._colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
        n = int((m > 0).sum())
        if n > 0:
            print(f'  {name}({code}) H={h_lo}-{h_hi} S={s_lo}-{s_hi} V={v_lo}-{v_hi}: {n} px')

    # Test STRIPE bands too (MW would be here)
    print('\nSTRIPE_BAND mask coverage in probe window:')
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in bm.STRIPE_BANDS:
        m = bm._colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
        n = int((m > 0).sum())
        if n > 0:
            print(f'  {name}({code}) H={h_lo}-{h_hi} S={s_lo}-{s_hi} V={v_lo}-{v_hi}: {n} px')

    # Now run the full-image ENG mask + CC and see what bbox appears near probe
    print('\n--- Engineering full-image mask CC analysis ---')
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    eng_m = bm._colour_mask(hsv_full, 18, 30, 100, 220, 160, 255)
    print(f'ENG total mask px: {int((eng_m > 0).sum())}')

    kx = max(3, int(round(iw * 0.22)))
    ky = max(2, ih // 12)
    m_closed = cv2.morphologyEx(
        eng_m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(m_closed, connectivity=8)
    print(f'ENG closed CCs: {n - 1}')

    print('ENG CCs near probe (x within 100px, y within 80px):')
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx = x + w / 2
        cy = y + h / 2
        if abs(cx - probe_cx) < 100 and abs(cy - probe_cy) < 80:
            ar = w / max(h, 1)
            fill = area / max(w * h, 1)
            print(f'  x={x} y={y} w={w} h={h} area={area} ar={ar:.2f} fill={fill:.2f}')

    # Lime/MW main mask -- maybe the bar is dominantly MW lime not Eng gold
    print('\n--- MW lime full-image mask CC analysis ---')
    mw_m = bm._colour_mask(hsv_full, 32, 44, 220, 255, 195, 255)
    print(f'MW lime total mask px: {int((mw_m > 0).sum())}')
    m_closed = cv2.morphologyEx(
        mw_m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(m_closed, connectivity=8)
    print(f'MW closed CCs: {n - 1}')
    print('MW CCs near probe:')
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx = x + w / 2
        cy = y + h / 2
        if abs(cx - probe_cx) < 100 and abs(cy - probe_cy) < 80:
            ar = w / max(h, 1)
            fill = area / max(w * h, 1)
            print(f'  x={x} y={y} w={w} h={h} area={area} ar={ar:.2f} fill={fill:.2f}')

    # Dump a sample of pixel HSV from the actual marker area
    # (use known left-col row 2 Tactical for y-reference: x=65 y=522 w=30 h=24
    #  so marker midline y≈534; right col same row)
    sample_y = 534
    print(f'\nHSV row scan at y={sample_y} from x=240 to x=300:')
    for sx in range(240, 301, 4):
        h_, s_, v_ = hsv_full[sample_y, sx]
        b_, g_, r_ = img[sample_y, sx]
        print(f'  x={sx}: BGR=({b_:3d},{g_:3d},{r_:3d}) HSV=({h_:3d},{s_:3d},{v_:3d})')


if __name__ == '__main__':
    main()
