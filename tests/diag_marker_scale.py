"""
Empirical check: does STO render BOFF marker bars at a fixed pixel width,
or does the UI scale with screen resolution?

Reads all 36 GT screens, runs the marker detector, and prints a table:

  filename, image_w x image_h, median_marker_w, full_bar_w, marker_w/image_w

Interpretation:
  - marker_w ≈ const across resolutions → STO uses FIXED-PX UI
  - marker_w / image_w ≈ const → STO SCALES UI proportionally
  - neither → mixed (user-configurable scale, dpi, etc.)
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import cv2

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

from tests.diag_boff_markers import (   # noqa: E402
    detect_markers,
    estimate_icon_dims,
    find_screen_files,
    full_bar_extent,
    VIRTUAL,
)

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'


def main():
    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))
    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        gt_full = [a for a in entries
                   if a.get('state') == 'confirmed'
                   and a.get('slot', '').startswith('Boff ')
                   and a.get('bbox')]
        n_real = sum(1 for a in gt_full if a.get('name', '') not in VIRTUAL)
        if n_real >= 5:
            candidates.append(fname)

    paths = find_screen_files(candidates)

    rows = []
    for fname in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        H, W = img.shape[:2]
        iw0, ih0 = estimate_icon_dims(img)
        markers = detect_markers(img, iw0, ih0)
        if len(markers) < 3:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        widths   = sorted(m[2] for m in markers)
        full_ws  = sorted(full_bar_extent(hsv, m)[0] for m in markers)
        m_w      = widths[len(widths) // 2]
        f_w      = full_ws[len(full_ws) // 2]
        rows.append((fname, W, H, m_w, f_w, m_w / W, f_w / W))

    rows.sort(key=lambda r: r[1])

    print(f'{"file":<42s} {"W":>5s} {"H":>5s} {"mw":>4s} {"fw":>4s} '
          f'{"mw/W":>7s} {"fw/W":>7s}')
    for r in rows:
        print(f'{r[0][:42]:<42s} {r[1]:>5d} {r[2]:>5d} '
              f'{r[3]:>4d} {r[4]:>4d} {r[5]:>7.4f} {r[6]:>7.4f}')

    mws  = [r[3] for r in rows]
    fws  = [r[4] for r in rows]
    rmws = [r[5] for r in rows]
    rfws = [r[6] for r in rows]
    print()
    print(f'marker_w     : min={min(mws)} max={max(mws)} '
          f'median={st.median(mws):.1f} stdev={st.stdev(mws):.2f}')
    print(f'full_bar_w   : min={min(fws)} max={max(fws)} '
          f'median={st.median(fws):.1f} stdev={st.stdev(fws):.2f}')
    print(f'marker_w / W : min={min(rmws):.4f} max={max(rmws):.4f} '
          f'median={st.median(rmws):.4f} stdev={st.stdev(rmws):.4f}')
    print(f'full_w   / W : min={min(rfws):.4f} max={max(rfws):.4f} '
          f'median={st.median(rfws):.4f} stdev={st.stdev(rfws):.4f}')

    print()
    print('Verdict heuristic:')
    cv_abs = st.stdev(mws)  / st.mean(mws)
    cv_rel = st.stdev(rmws) / st.mean(rmws)
    print(f'  CV (marker_w absolute) : {cv_abs:.3f}')
    print(f'  CV (marker_w / W)      : {cv_rel:.3f}')
    if cv_abs < cv_rel * 0.7:
        print('  → STO appears to use FIXED-PX UI (marker width stable across resolutions)')
    elif cv_rel < cv_abs * 0.7:
        print('  → STO SCALES UI proportionally (marker_w / W stable)')
    else:
        print('  → mixed / inconclusive — neither is clearly stable')


if __name__ == '__main__':
    main()
