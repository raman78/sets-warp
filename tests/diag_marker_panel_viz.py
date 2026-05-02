"""
Quick batch visualizer for boff_marker.detect_panel.

For each input image, runs the production marker detector and saves an
annotated PNG with col_a / col_b markers + projected slot bboxes. Images
where the detector returns None get a "_FAIL" overlay.

Usage:
    .venv/bin/python tests/diag_marker_panel_viz.py [DIR]

Output: tests/_diag_out/marker_panel_viz/<stem>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

from warp.recognition.boff_marker import detect_panel

SEAT_BGR = {
    'T': (40, 40, 255),
    'E': (40, 220, 255),
    'S': (255, 100, 40),
    'U': (200, 255, 255),
}
EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def annotate(img, panel):
    out = img.copy()
    a = panel['col_a']
    b = panel['col_b']
    for x, y, w, h, code, spec in a + b:
        col = SEAT_BGR.get(code, (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = f'{code}' + (f'+{spec}' if spec else '')
        cv2.putText(out, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    for seat_idx, slot_idx, sx, sy, sw, sh, scode in panel['slots']:
        col = SEAT_BGR.get(scode, (0, 255, 0))
        cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), col, 1)
    score = panel['score']
    cv2.putText(out, f'score={score:.2f} seats={len(panel["seats"])}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def fail_overlay(img):
    out = img.copy()
    cv2.putText(out, 'NO BOFF PANEL DETECTED',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def main():
    src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/raman/STO_screens/screeny')
    out_dir = ROOT / 'tests' / '_diag_out' / 'marker_panel_viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        print(f'No images under {src_dir}')
        return

    ok, miss = 0, 0
    failures: list[str] = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f'SKIP {p.name}: unreadable')
            continue
        panel = detect_panel(img)
        if panel is None:
            miss += 1
            failures.append(p.name)
            cv2.imwrite(str(out_dir / f'{p.stem}_FAIL.png'), fail_overlay(img))
            print(f'FAIL  {p.name}')
        else:
            ok += 1
            cv2.imwrite(str(out_dir / f'{p.stem}.png'), annotate(img, panel))
            print(f'OK    {p.name}  score={panel["score"]:.2f}  seats={len(panel["seats"])}')

    total = ok + miss
    print(f'\n=== {ok}/{total} panels detected ({100 * ok / max(total, 1):.1f}%) — {miss} failures ===')
    if failures:
        print('Failures:')
        for n in failures:
            print(f'  {n}')


if __name__ == '__main__':
    main()
