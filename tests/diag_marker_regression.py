"""Quick regression check after relaxing fill/edge thresholds in
boff_marker.detect_markers under the strong-uniformity bypass.

Reports: per-image marker count, panel score, seat count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warp.recognition import boff_marker as bm  # noqa: E402

ROOTS = [
    Path('warp/training_data/screen_types/SPACE_MIXED'),
    Path('warp/training_data/screen_types/GROUND_MIXED'),
    Path('warp/training_data/screen_types/SPACE_BOFFS'),
    Path('warp/training_data/screen_types/GROUND_BOFFS'),
    Path('warp/training_data/screen_types/BOFFS'),
]


def main() -> None:
    total = 0
    panel_ok = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.glob('*.png')):
            total += 1
            img = cv2.imread(str(p))
            if img is None:
                continue
            iw, ih = bm.estimate_icon_dims(img)
            markers = bm.detect_markers(img, iw, ih)
            res = bm.detect_panel(img)
            n_m = len(markers)
            if res is None:
                print(f'{p.name}: markers={n_m} panel=None')
            else:
                panel_ok += 1
                n_a = len(res['col_a']); n_b = len(res['col_b'])
                print(f'{p.name}: markers={n_m} panel={res["score"]:.2f} '
                      f'col_a={n_a} col_b={n_b}')
    print(f'\nTotal: {total}, panels detected: {panel_ok}')


if __name__ == '__main__':
    main()
