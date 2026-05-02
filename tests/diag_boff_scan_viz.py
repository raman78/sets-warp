"""Visualize EQ-anchored sliding-window scan results.

Reads tests/_diag_out/boff_scan/scan.json — uses the per-screen
`detections` array dumped by `diag_boff_scan.py` (the EfficientNet
classify_patch + NMS results enriched with `__boff_*` types). NO
re-running the detector here.

Colors:
  green  - GT BOFF bboxes
  blue   - GT EQ bboxes
  yellow - picked sliding window
  red    - predicted BOFF-typed dets (whole image; brighter inside window)
  gray   - other GT slot bboxes (traits, rep, etc.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EQ_PREFIXES = (
    'Fore Weapon', 'Aft Weapon', 'Deflector', 'Engines', 'Shield', 'Warp Core',
    'Singularity Core', 'Devices', 'Hangar', 'Secondary Deflector',
    'Experimental',
)
CONSOLE_SUBSTR = ('Console',)  # matches Engineering/Science/Tactical/Universal Consoles
BOFF_PREFIXES = ('Boff ',)
TRAIT_PREFIXES = ('Personal Space Traits', 'Personal Ground Traits',
                  'Personal Trait')
STARSHIP_TRAIT_PREFIXES = ('Starship Trait',)
REP_PREFIXES = ('Space Reputation', 'Ground Reputation', 'Active Space Rep',
                'Active Ground Rep', 'Active Reputation', 'Reputation')
SHIP_INFO_PREFIXES = ('Ship Name', 'Ship Tier', 'Ship Type')

OUT_DIR = ROOT / 'tests' / '_diag_out' / 'boff_scan' / 'viz'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCAN_JSON = ROOT / 'tests' / '_diag_out' / 'boff_scan' / 'scan.json'
ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
SCREEN_DIR = ROOT / 'warp' / 'training_data'

TARGETS = {
    'Ambassador-broadside.png',
}


def categorize(slot: str) -> str:
    if any(slot.startswith(p) for p in BOFF_PREFIXES):
        return 'boff'
    if any(slot.startswith(p) for p in EQ_PREFIXES):
        return 'eq'
    if any(sub in slot for sub in CONSOLE_SUBSTR):
        return 'console'
    if any(slot.startswith(p) for p in STARSHIP_TRAIT_PREFIXES):
        return 'starship_trait'
    if any(slot.startswith(p) for p in TRAIT_PREFIXES):
        return 'trait'
    if any(slot.startswith(p) for p in REP_PREFIXES):
        return 'rep'
    if any(slot.startswith(p) for p in SHIP_INFO_PREFIXES):
        return 'shipinfo'
    return 'other'


CAT_STYLE = {
    'boff':           ((0, 255, 0),     2),  # bright green
    'eq':             ((255, 100, 0),   2),  # blue/cyan-ish
    'console':        ((255, 200, 0),   2),  # cyan
    'trait':          ((255, 0, 255),   2),  # magenta
    'starship_trait': ((180, 50, 255),  2),  # pink-purple
    'rep':            ((0, 200, 255),   2),  # orange
    'shipinfo':       ((200, 200, 200), 1),  # light gray
    'other':          ((140, 140, 140), 1),  # gray
}


def find_screen(fname: str) -> Path | None:
    for sub in ('screen_types/SPACE_MIXED', 'screen_types/SPACE_BOFFS',
                'screen_types/SPACE_EQ', 'screen_types/UNKNOWN',
                'screen_types/GROUND_MIXED'):
        p = SCREEN_DIR / sub / fname
        if p.exists():
            return p
    for p in SCREEN_DIR.rglob(fname):
        return p
    return None


def draw(img: np.ndarray, ann: list, picked: tuple, preds: list) -> np.ndarray:
    out = img.copy()
    counts: dict[str, int] = {}
    n_empty = 0
    n_inactive = 0
    for a in ann:
        bbox = a.get('bbox')
        if not bbox or len(bbox) < 4:
            continue
        slot = str(a.get('slot', ''))
        name = str(a.get('name', ''))
        cat = categorize(slot)
        counts[cat] = counts.get(cat, 0) + 1
        if name == '__empty__':
            n_empty += 1
        elif name == '__inactive__':
            n_inactive += 1
        color, thick = CAT_STYLE.get(cat, CAT_STYLE['other'])
        x, y, w, h = [int(v) for v in bbox[:4]]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)

    if picked:
        x, y, w, h = picked
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 3)

    # preds: only draw detections inside the picked window
    col = (0, 0, 255)
    for p in preds:
        if not p.get('in_window'):
            continue
        b = p.get('bbox')
        if not b or len(b) < 4:
            continue
        x, y, w, h = [int(v) for v in b[:4]]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        t = str(p.get('type', ''))
        if t.startswith('__boff_'):
            label = t.replace('__boff_', '')[:4]
            cv2.putText(out, label, (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255),
                        2, cv2.LINE_AA)
            cv2.putText(out, label, (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    legend = [
        (f"GT BOFF ({counts.get('boff', 0)})",                CAT_STYLE['boff'][0]),
        (f"GT EQ ({counts.get('eq', 0)})",                    CAT_STYLE['eq'][0]),
        (f"GT Console ({counts.get('console', 0)})",          CAT_STYLE['console'][0]),
        (f"GT Personal Trait ({counts.get('trait', 0)})",     CAT_STYLE['trait'][0]),
        (f"GT Starship Trait ({counts.get('starship_trait', 0)})",
                                                              CAT_STYLE['starship_trait'][0]),
        (f"GT Reputation ({counts.get('rep', 0)})",           CAT_STYLE['rep'][0]),
        (f"GT Ship Info ({counts.get('shipinfo', 0)})",       CAT_STYLE['shipinfo'][0]),
        (f"GT other ({counts.get('other', 0)})",              CAT_STYLE['other'][0]),
        (f"(of which __empty__: {n_empty}, __inactive__: {n_inactive})",
                                                              (200, 200, 200)),
        ('Picked window',                                     (0, 255, 255)),
        (f'Predicted ({len(preds)})',                         (0, 0, 255)),
    ]
    for i, (text, col) in enumerate(legend):
        y = 18 + i * 22
        cv2.rectangle(out, (8, y - 12), (28, y + 4), col, -1)
        cv2.putText(out, text, (34, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (34, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    if not SCAN_JSON.exists():
        print(f'Run tests/diag_boff_scan.py first to produce {SCAN_JSON}')
        return 2
    scan = json.loads(SCAN_JSON.read_text())
    per_screen = {r['file']: r for r in scan.get('per_screen', [])}
    detections = {r['file']: r for r in scan.get('detections', [])}
    ann_all = json.loads(ANN_PATH.read_text())

    written = 0
    for fname in sorted(TARGETS):
        ann = ann_all.get(fname)
        if not isinstance(ann, list):
            print(f'[skip] {fname}: no annotations')
            continue
        path = find_screen(fname)
        if not path:
            print(f'[skip] {fname}: file not found on disk')
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f'[skip] {fname}: cv2 read failed')
            continue
        rec = per_screen.get(fname, {})
        win = tuple(rec.get('window', [])) if rec.get('window') else None
        det_rec = detections.get(fname, {})
        boff_dets = det_rec.get('boff_dets', [])
        preds = []
        for d in boff_dets:
            b = d.get('bbox')
            if not b:
                continue
            in_window = False
            if win:
                wx, wy, ww, wh = win
                bx, by, bw, bh = b
                in_window = (bx >= wx and by >= wy
                             and bx + bw <= wx + ww
                             and by + bh <= wy + wh)
            preds.append({**d, 'in_window': in_window})
        out = draw(img, ann, win, preds)
        dest = OUT_DIR / (fname.replace('/', '_').rsplit('.', 1)[0] + '.png')
        cv2.imwrite(str(dest), out)
        written += 1
        cnts: dict[str, int] = {}
        n_empty = 0
        n_inactive = 0
        for a in ann:
            if not isinstance(a, dict):
                continue
            cnts[categorize(str(a.get('slot', '')))] = cnts.get(
                categorize(str(a.get('slot', ''))), 0) + 1
            nm = str(a.get('name', ''))
            if nm == '__empty__':
                n_empty += 1
            elif nm == '__inactive__':
                n_inactive += 1
        cat_str = ' '.join(f'{k}={v}' for k, v in sorted(cnts.items()))
        print(f'[ok ] {fname:38s} | {cat_str:80s} | empty={n_empty} inactive={n_inactive} '
              f'total={sum(cnts.values()):3d} win={win} preds={len(preds)}')
    print(f'\n{written} images written to {OUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
