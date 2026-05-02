"""
Step-by-step visual walk of detect_panel() on overview-carrier.png.

Layout: 3+2 BOFF panel in lower-left of a 1528x755 build screen, with
another small image pasted under the 2nd right-column seat (in the
"empty 3rd seat" slot), creating colour noise.

Outputs (under tests/_diag_out/marker_carrier_walk/):
  step1_main_masks.png   — combined HSV main-band mask per seat colour
  step2_raw_ccs.png      — every connected-component bbox passed through
                           the full filter chain (pre-dedupe), labelled
                           with the reason it was rejected (or kept).
  step3_kept_markers.png — final detect_markers() output.
  step4_panel.png        — best_panel() result with col_a/col_b grid.
  step5_slots.png        — project_seat_slots() projection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

from warp.recognition.boff_marker import (  # noqa: E402
    MAIN_BANDS,
    STRIPE_BANDS,
    _colour_mask,
    _merge_close_bboxes,
    _refine_dims_from_markers,
    best_panel,
    detect_markers,
    estimate_icon_dims,
    full_bar_extent,
    project_seat_slots,
)

IMG_PATH = Path('/home/raman/STO_screens/screeny/overview-carrier.png')
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'marker_carrier_walk'

SEAT_BGR = {
    'T': (40, 40, 255),
    'E': (40, 220, 255),
    'S': (255, 100, 40),
    'U': (200, 255, 255),
}


def write(name, img):
    p = OUT_DIR / name
    cv2.imwrite(str(p), img)
    print(f'  wrote {p.name}  ({img.shape[1]}x{img.shape[0]})')


def step1_masks(img):
    """Visualise the per-band HSV masks (main bands only)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    composite = img.copy()
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in MAIN_BANDS:
        m = _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
        col = SEAT_BGR.get(code, (0, 255, 0))
        layer = np.zeros_like(img)
        layer[m > 0] = col
        composite = cv2.addWeighted(composite, 1.0, layer, 0.7, 0)
    cv2.putText(composite, 'step1: HSV main-band masks (T=red E=gold S=blue U=cream)',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    write('step1_main_masks.png', composite)


def step2_raw_ccs(img, icon_w, icon_h):
    """Run the dual-mask CC pipeline and collect EVERY candidate with
    the reason it was kept or rejected. Visualise on the image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out_img = img.copy()

    abs_min_w, abs_min_h = 10, 12
    abs_max_w, abs_max_h = 90, 90
    min_w = max(abs_min_w, int(icon_w * 0.45))
    max_w = min(abs_max_w, max(int(icon_w * 1.25), 36))
    min_h = max(abs_min_h, int(icon_h * 0.45))
    max_h = min(abs_max_h, max(int(icon_h * 1.35), 44))
    ar_min, ar_max = 0.30, 1.8
    fill_min = 0.70
    uni_v_max = 28
    uni_h_max = 6
    edge_max = 0.07
    edge_inset = 2
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

    by_name = {}
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in MAIN_BANDS:
        by_name.setdefault(name, {'code': code, 'ranges': []})['ranges'].append(
            (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi))

    kx = max(3, int(round(icon_w * 0.22)))
    ky = max(2, icon_h // 12)

    counts = {'total': 0, 'kept': 0, 'rej_size': 0, 'rej_ar': 0,
              'rej_fill': 0, 'rej_unifV': 0, 'rej_unifH': 0, 'rej_edge': 0}

    for _name, info in by_name.items():
        code = info['code']
        m = None
        for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in info['ranges']:
            part = _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
            m = part if m is None else cv2.bitwise_or(m, part)
        m_closed = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        gap_x = max(2, int(round(icon_w * 0.35)))
        merged = []
        for src in (m, m_closed):
            n, _, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
            raw = [tuple(int(v) for v in stats[i, [0, 1, 2, 3, 4]]) for i in range(1, n)]
            merged.extend(_merge_close_bboxes(raw, gap_x=gap_x, overlap_y_frac=0.55))

        for x, y, w, h, area in merged:
            counts['total'] += 1
            why = None
            if w < min_w or w > max_w or h < min_h or h > max_h:
                why = 'size'
                counts['rej_size'] += 1
            elif (w / max(h, 1)) < ar_min or (w / max(h, 1)) > ar_max:
                why = 'ar'
                counts['rej_ar'] += 1
            elif area < (w * h) * fill_min:
                why = 'fill'
                counts['rej_fill'] += 1
            else:
                sel = m[y:y + h, x:x + w] > 0
                if sel.sum() < 20:
                    why = 'fill'
                    counts['rej_fill'] += 1
                else:
                    crop_v = hsv[y:y + h, x:x + w, 2][sel]
                    crop_h = hsv[y:y + h, x:x + w, 0][sel]
                    if float(np.std(crop_v)) > uni_v_max:
                        why = 'unifV'
                        counts['rej_unifV'] += 1
                    else:
                        h_std = min(
                            float(np.std(crop_h)),
                            float(np.std((crop_h.astype(np.int32) + 90) % 180)),
                        )
                        if h_std > uni_h_max:
                            why = 'unifH'
                            counts['rej_unifH'] += 1
                        else:
                            ix0 = x + edge_inset; iy0 = y + edge_inset
                            ix1 = x + w - edge_inset; iy1 = y + h - edge_inset
                            if ix1 > ix0 and iy1 > iy0:
                                edge_crop = edges[iy0:iy1, ix0:ix1]
                                edge_frac = float(edge_crop.sum() / 255.0) / (
                                    (ix1 - ix0) * (iy1 - iy0))
                                if edge_frac > edge_max:
                                    why = 'edge'
                                    counts['rej_edge'] += 1
            if why is None:
                counts['kept'] += 1
                col = SEAT_BGR.get(code, (0, 255, 0))
                cv2.rectangle(out_img, (x, y), (x + w, y + h), col, 2)
                cv2.putText(out_img, code, (x, max(0, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(out_img, (x, y), (x + w, y + h), (140, 140, 140), 1)
                cv2.putText(out_img, f'{code}/{why}', (x, max(0, y - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180),
                            1, cv2.LINE_AA)

    label = (f'step2: raw CCs (kept {counts["kept"]}/{counts["total"]}) — '
             f'size:{counts["rej_size"]} ar:{counts["rej_ar"]} '
             f'fill:{counts["rej_fill"]} unifV:{counts["rej_unifV"]} '
             f'unifH:{counts["rej_unifH"]} edge:{counts["rej_edge"]}')
    cv2.putText(out_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    write('step2_raw_ccs.png', out_img)
    return counts


def step3_kept_markers(img, icon_w, icon_h):
    markers = detect_markers(img, icon_w, icon_h)
    out = img.copy()
    for x, y, w, h, code in markers:
        col = SEAT_BGR.get(code, (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, code, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(out, f'step3: detect_markers final = {len(markers)} markers',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    write('step3_kept_markers.png', out)
    return markers


def step4_panel(img, markers, icon_w, icon_h):
    panel = best_panel(markers, icon_w, icon_h)
    out = img.copy()
    if panel is None:
        cv2.putText(out, 'step4: best_panel() returned None',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        write('step4_panel.png', out)
        return None
    a, b, score = panel
    for tag, col_list, side_col in (('L', a, (0, 255, 0)), ('R', b, (255, 0, 200))):
        for i, m in enumerate(col_list):
            x, y, w, h, code = m
            cv2.rectangle(out, (x, y), (x + w, y + h), side_col, 2)
            cv2.putText(out, f'{tag}{i}:{code}', (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, side_col, 1, cv2.LINE_AA)
    cv2.putText(out, f'step4: best_panel score={score:.2f}  '
                f'L={len(a)} R={len(b)}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    write('step4_panel.png', out)
    return panel


def step5_slots(img, panel):
    if panel is None:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    slots = project_seat_slots(panel, hsv=hsv)
    a, b, _ = panel
    out = img.copy()
    for m in a:
        x, y, w, h, c = m
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for m in b:
        x, y, w, h, c = m
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 200), 1)
    for seat_idx, slot_idx, sx, sy, sw, sh, scode in slots:
        col = SEAT_BGR.get(scode, (0, 255, 0))
        cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), col, 2)
        cv2.putText(out, f'{seat_idx}.{slot_idx}', (sx, sy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    cv2.putText(out, f'step5: project_seat_slots = {len(slots)} slots',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    write('step5_slots.png', out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print(f'imread failed: {IMG_PATH}')
        return
    print(f'Image: {img.shape[1]}x{img.shape[0]}')

    icon_w, icon_h = estimate_icon_dims(img)
    print(f'estimate_icon_dims: {icon_w}x{icon_h}')

    step1_masks(img)
    counts = step2_raw_ccs(img, icon_w, icon_h)
    print(f'step2 counts: {counts}')
    markers = step3_kept_markers(img, icon_w, icon_h)
    print(f'step3 markers: {len(markers)}')
    if markers:
        icon_w2, icon_h2 = _refine_dims_from_markers(markers, icon_w, icon_h)
        print(f'refined dims: {icon_w2}x{icon_h2}')
        for m in markers:
            print(f'  marker {m}')
    else:
        icon_w2, icon_h2 = icon_w, icon_h
    panel = step4_panel(img, markers, icon_w2, icon_h2)
    if panel is not None:
        a, b, score = panel
        print(f'step4 panel: score={score:.2f}  L={len(a)}  R={len(b)}')
    step5_slots(img, panel)


if __name__ == '__main__':
    main()
