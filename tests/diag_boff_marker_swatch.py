"""
Generate a visual color reference for the BOFF seat marker palette.

Renders two stacked panels:
  1. Seat-type swatches  : Tactical / Engineering / Science / Universal
  2. Spec-stripe swatches: Intelligence / Temporal / Miracle Worker /
                           Pilot-or-Command

Colors come from HSV values sampled across 25+ ground-truth seats during
the 2026-04-26 detector work (see memory/boff_marker_detection_2026-04-26.md).
Each swatch shows a representative center HSV, not a band edge.

Output: docs/images/boff_seat_marker_colors.png
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
OUT  = ROOT / 'docs' / 'images' / 'boff_seat_marker_colors.png'


def hsv_to_bgr(h, s, v):
    """OpenCV-HSV (H 0-180, S/V 0-255) → BGR pixel."""
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


# (label, hue, sat, val, code) — representative center of each band.
SEAT_TYPES = [
    ('Tactical',    0,   192, 106, 'T'),
    ('Engineering', 23,  155, 187, 'E'),
    ('Science',     108, 180, 150, 'S'),
    ('Universal',   26,  70,  213, 'U'),
]

# Spec stripes (narrow ~5-8 px stripe on the RIGHT edge of the marker).
SPEC_STRIPES = [
    ('Intelligence',     130, 180, 200, 'P'),  # purple
    ('Temporal',         108, 200, 220, 'B'),  # blue
    ('Miracle Worker',   55,  200, 200, 'G'),  # green / lime
    ('Pilot or Command', 12,  220, 230, 'O'),  # orange
]


def render_panel(title, rows, cell_w=320, cell_h=72, pad=12, font_scale=0.6):
    """Render one labelled panel of swatches; return BGR image."""
    n = len(rows)
    w = cell_w + pad * 2
    title_h = 36
    h = title_h + n * (cell_h + pad) + pad
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    cv2.putText(img, title, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (40, 40, 40), 2, cv2.LINE_AA)

    y = title_h
    for label, H, S, V, code in rows:
        bgr = hsv_to_bgr(H, S, V)
        x0 = pad
        x1 = x0 + cell_w
        y0 = y
        y1 = y + cell_h
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)

        # Choose readable text colour: white on dark, black on light.
        luma = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
        txt_col = (255, 255, 255) if luma < 140 else (20, 20, 20)

        line1 = f'{label}  [{code}]'
        line2 = f'H={H:>3}  S={S:>3}  V={V:>3}'
        cv2.putText(img, line1, (x0 + 12, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.05,
                    txt_col, 2, cv2.LINE_AA)
        cv2.putText(img, line2, (x0 + 12, y0 + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.05,
                    txt_col, 1, cv2.LINE_AA)
        y = y1 + pad

    return img


def hstack_with_pad(panels, pad=16, bg=245):
    h = max(p.shape[0] for p in panels)
    out_panels = []
    for p in panels:
        if p.shape[0] < h:
            extra = np.full((h - p.shape[0], p.shape[1], 3), bg, dtype=np.uint8)
            p = np.vstack([p, extra])
        out_panels.append(p)
        out_panels.append(np.full((h, pad, 3), bg, dtype=np.uint8))
    out_panels.pop()
    return np.hstack(out_panels)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seats   = render_panel('Seat type (main zone, 75-95% width)', SEAT_TYPES)
    stripes = render_panel('Spec stripe (right edge, 5-25% width)', SPEC_STRIPES)
    img = hstack_with_pad([seats, stripes])
    cv2.imwrite(str(OUT), img)
    print(f'wrote {OUT}  ({img.shape[1]}x{img.shape[0]})')


if __name__ == '__main__':
    main()
