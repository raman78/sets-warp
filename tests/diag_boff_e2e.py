"""
End-to-end BOFF recognition diagnostic.

Runs LayoutDetector.detect() + candidate-narrowing + IconMatcher.match()
exactly as warp_importer does for `Boff Seat *` slots — on every annotated
screen with ≥4 real BOFF abilities. Compares predictions to ground truth
annotations via IoU.

Metrics (per screen + rolled up):
  - Seat coverage: fraction of GT BOFF annotations that land inside a
    predicted seat bbox (IoU ≥ 0.3).
  - Base-profession classification accuracy on matched seat crops.
  - Ability-name accuracy (exact match) on matched seat crops.
  - Final slot-name remap accuracy (post-match → SETS slot).

Read-only. Does not modify code or annotations.

Usage:
  PYTHONPATH=/home/raman/PycharmProjects/sets-warp \\
    python tests/diag_boff_e2e.py [--limit N]
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, '/home/raman/PycharmProjects/sets-warp')
from warp.recognition.layout_detector import LayoutDetector
from warp.recognition.icon_matcher import SETSIconMatcher

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
TD = ROOT / 'warp' / 'training_data'
ANN_PATH = TD / 'annotations.json'
SCREEN_TYPES_PATH = TD / 'screen_types.json'
SCREEN_TYPES_CONFIRMED_PATH = TD / 'screen_types_user_confirmed.json'
CARGO = ROOT / 'data' / 'cargo' / 'boff_abilities.json'
OUT = ROOT / 'tests' / '_diag_out' / 'boff_e2e'

IOU_THRESHOLD = 0.30
VIRTUAL = frozenset({'__empty__', '__inactive__'})

PROF_MAP_KEY_TO_CANON = {   # same as warp_importer.py:972
    'tactical': 'Tactical', 'engineering': 'Engineering', 'science': 'Science',
    'intelligence': 'Intelligence', 'command': 'Command', 'pilot': 'Pilot',
    'miracle worker': 'Miracle Worker', 'temporal': 'Temporal Operative',
}
PROF_TO_SLOT = {   # same as warp_importer.py:1038
    'Tactical': 'Boff Tactical', 'Engineering': 'Boff Engineering',
    'Science': 'Boff Science', 'Intelligence': 'Boff Intelligence',
    'Command': 'Boff Command', 'Pilot': 'Boff Pilot',
    'Miracle Worker': 'Boff Miracle Worker',
    'Temporal Operative': 'Boff Temporal', 'Temporal': 'Boff Temporal',
}
SPEC_PROFS = {'Intelligence', 'Command', 'Pilot',
              'Miracle Worker', 'Temporal Operative', 'Temporal'}


class MockCache:
    """Minimal cache with boff_abilities + equipment (empty)."""

    def __init__(self):
        self.boff_abilities: dict = {}
        self.boff_abilities['all'] = {}   # flat for warp_importer-compat
        if CARGO.exists():
            cargo = json.loads(CARGO.read_text(encoding='utf-8'))
            for scope in ('space', 'ground'):
                if scope not in cargo:
                    continue
                for prof, ranks in cargo[scope].items():
                    for rank_dict in ranks:
                        if not rank_dict:
                            continue
                        for name in rank_dict:
                            entry = {'name': name, 'profession': prof}
                            self.boff_abilities['all'][name] = entry
        self.equipment = {}


class MockApp:
    def __init__(self):
        self.cache = MockCache()
        # Wire the real SETS images directory so IconMatcher builds its
        # template + HSV-histogram index (production path). Without this,
        # the matcher silently runs ML-only.
        self.config = {
            'config_subfolders': {
                'images': str(ROOT / '.config' / 'images'),
            }
        }


def iou(a, b):
    """bbox a,b as (x,y,w,h) or (x,y,w,h,state) — extra elements ignored."""
    ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
    bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def narrow_candidates(crop_bgr, boff_cache):
    """Replicate warp_importer.py:968-986 narrowing logic.
    Returns (candidates_set_or_none, base_prof_key_or_none, base_prof_canon_or_none).
    """
    base_prof_key = LayoutDetector._classify_boff_profession(crop_bgr)
    if not base_prof_key:
        return None, None, None
    base_prof = PROF_MAP_KEY_TO_CANON.get(base_prof_key)
    if not base_prof:
        return None, base_prof_key, None
    allowed = {base_prof} | SPEC_PROFS
    candidates = {name for name, info in boff_cache.items()
                  if info.get('profession') in allowed}
    return candidates, base_prof_key, base_prof


def load_screen_types() -> dict:
    types = {}
    if SCREEN_TYPES_CONFIRMED_PATH.exists():
        try:
            data = json.loads(SCREEN_TYPES_CONFIRMED_PATH.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                types.update(data)
        except Exception:
            pass
    if SCREEN_TYPES_PATH.exists():
        data = json.loads(SCREEN_TYPES_PATH.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            for k, v in data.items():
                types.setdefault(k, v)   # confirmed takes priority
    return types


def find_images(names: list[str]) -> dict[str, Path]:
    candidates = [
        Path('/home/raman/STO_screens'),
        TD / 'migrated_images',
        TD,
    ]
    idx = {}
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            if p.name not in idx and p.name in set(names):
                idx[p.name] = p
    return idx


def find_seat_for_bbox(seat_bboxes, gt_bbox, thr=IOU_THRESHOLD):
    """Given dict {seat_id: [bbox, ...]}, return (seat_id, pred_bbox, iou)
    for the best-matching seat/bbox, or (None, None, 0) if none ≥ threshold."""
    best = (None, None, 0.0)
    for sid, boxes in seat_bboxes.items():
        for pb in boxes:
            v = iou(pb, gt_bbox)
            if v > best[2]:
                best = (sid, pb, v)
    if best[2] < thr:
        return (None, None, best[2])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='test only first N screens (0 = all)')
    ap.add_argument('--screen-filter', type=str, default='',
                    help='substring match on filename (optional)')
    ap.add_argument('--use-gt-bbox', action='store_true',
                    help='skip detector; crop at GT bbox directly (matcher ceiling)')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))
    screen_types = load_screen_types()

    # Pick screens with ≥4 real BOFF annotations
    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        real_boff = [a for a in entries
                     if a.get('state') == 'confirmed'
                     and a.get('slot', '').startswith('Boff ')
                     and a.get('name') not in VIRTUAL
                     and a.get('bbox')]
        if len(real_boff) >= 4:
            candidates.append((fname, real_boff))
    candidates.sort(key=lambda x: -len(x[1]))

    if args.screen_filter:
        candidates = [c for c in candidates if args.screen_filter in c[0]]
    if args.limit:
        candidates = candidates[:args.limit]

    paths = find_images([c[0] for c in candidates])
    print(f'Screens with ≥4 BOFF GT: {len(candidates)}   '
          f'findable on disk: {len(paths)}')

    app = MockApp()
    matcher = SETSIconMatcher(app)
    detector = LayoutDetector()
    boff_cache = app.cache.boff_abilities['all']

    # Rollup
    roll = {
        'screens_tested':    0,
        'gt_total':          0,
        'gt_covered':        0,   # GT has matching seat bbox (IoU ≥ 0.3)
        'base_prof_correct': 0,   # on covered GT
        'ability_correct':   0,   # on covered GT
        'slot_correct':      0,   # on covered GT (final_slot_name == gt.slot)
        'bproof_confusion':  defaultdict(lambda: defaultdict(int)),
        'per_screen':        [],
    }

    for idx, (fname, real_boff) in enumerate(candidates, 1):
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue

        if args.use_gt_bbox:
            # Matcher-ceiling mode: skip detector, crop at GT directly
            seat_bboxes = None   # signals to branch below
        else:
            screen_type = screen_types.get(fname) or 'SPACE_MIXED'
            try:
                layout = detector.detect(img, screen_type)
            except Exception as e:
                print(f'[{idx}] {fname}: detect() raised {e}')
                continue
            seat_bboxes = {k: v for k, v in layout.items() if k.startswith('Boff Seat')}

        per_screen = {
            'file': fname,
            'screen_type': 'GT' if args.use_gt_bbox else (screen_types.get(fname) or 'SPACE_MIXED'),
            'gt_n': len(real_boff),
            'seats_detected': 0 if args.use_gt_bbox else len(seat_bboxes),
            'covered': 0, 'base_prof_correct': 0,
            'ability_correct': 0, 'slot_correct': 0,
        }

        for gt in real_boff:
            roll['gt_total'] += 1
            gt_bbox = gt['bbox']
            gt_name = gt.get('name', '')
            gt_slot = gt.get('slot', '')
            # ground-truth profession (lower-case) from slot name
            gt_prof_key = gt_slot.replace('Boff ', '').lower().strip()

            if args.use_gt_bbox:
                pb = gt_bbox
            else:
                sid, pb, _iou_v = find_seat_for_bbox(seat_bboxes, gt_bbox)
                if sid is None:
                    continue
            roll['gt_covered'] += 1
            per_screen['covered'] += 1

            # Crop from predicted bbox to feed the same classifier/matcher.
            # BOFF bboxes may be 5-tuples (x, y, w, h, state); ignore state.
            x, y, w, h = pb[0], pb[1], pb[2], pb[3]
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue

            # Base-prof classification
            cands, base_prof_key, base_prof = narrow_candidates(crop, boff_cache)
            pred_prof_key = base_prof_key or 'none'
            roll['bproof_confusion'][gt_prof_key][pred_prof_key] += 1
            if pred_prof_key == gt_prof_key:
                roll['base_prof_correct'] += 1
                per_screen['base_prof_correct'] += 1

            # Icon matching
            try:
                name, conf, _, _ = matcher.match(crop, candidate_names=cands)
            except Exception:
                name, conf = '', 0.0
            if name and name == gt_name:
                roll['ability_correct'] += 1
                per_screen['ability_correct'] += 1

            # Slot remap
            if name and name in boff_cache:
                prof = boff_cache[name].get('profession', '')
                final_slot = PROF_TO_SLOT.get(prof, '')
                if final_slot == gt_slot:
                    roll['slot_correct'] += 1
                    per_screen['slot_correct'] += 1

        roll['screens_tested'] += 1
        roll['per_screen'].append(per_screen)
        print(f'[{idx:3d}] {fname[:50]:<50s}  '
              f'seats={per_screen["seats_detected"]:2d}  '
              f'GT={per_screen["gt_n"]:2d}  '
              f'cov={per_screen["covered"]:2d}  '
              f'bp={per_screen["base_prof_correct"]:2d}  '
              f'ab={per_screen["ability_correct"]:2d}  '
              f'sl={per_screen["slot_correct"]:2d}')

    # Print rollup
    n = roll['gt_total'] or 1
    c = roll['gt_covered']
    print('\n' + '=' * 70)
    print(f'Screens tested:           {roll["screens_tested"]}')
    print(f'GT annotations total:     {roll["gt_total"]}')
    print(f'Seat coverage (IoU≥0.3):  {c} / {n} = {c/n*100:.1f}%')
    cov = c or 1
    print(f'Base-prof correct:        {roll["base_prof_correct"]}/{c} = '
          f'{roll["base_prof_correct"]/cov*100:.1f}%  (on covered)')
    print(f'Ability name correct:     {roll["ability_correct"]}/{c} = '
          f'{roll["ability_correct"]/cov*100:.1f}%  (on covered)')
    print(f'Slot-name correct:        {roll["slot_correct"]}/{c} = '
          f'{roll["slot_correct"]/cov*100:.1f}%  (on covered)')
    # End-to-end: correct out of ALL GT (including uncovered)
    print(f'End-to-end slot correct:  {roll["slot_correct"]}/{n} = '
          f'{roll["slot_correct"]/n*100:.1f}%  (vs total GT)')

    # Base-prof confusion (on covered crops)
    profs_order = ['tactical', 'engineering', 'science', 'command',
                   'intelligence', 'pilot', 'temporal', 'miracle worker']
    print('\nBase-prof confusion on covered seat crops '
          '(rows=gt, cols=pred):')
    cols = [p[:3] for p in profs_order] + ['non']
    print(' ' * 18 + ''.join(f'{c:>6}' for c in cols))
    for gp in profs_order:
        row = roll['bproof_confusion'].get(gp, {})
        vals = [row.get(pp, 0) for pp in profs_order]
        vals.append(row.get('none', 0))
        n_row = sum(vals)
        line = f'{gp[:17]:<18}' + ''.join(f'{v or ".":>6}' for v in vals)
        print(f'{line}  (n={n_row})')

    # Write JSON dump for later analysis
    out_path = OUT / 'rollup.json'
    serializable = {
        k: (dict(v) if isinstance(v, defaultdict)
            else ([dict(e) if isinstance(e, dict) else e for e in v]
                  if isinstance(v, list) else v))
        for k, v in roll.items()
    }
    # bproof_confusion is nested defaultdict → plain dict
    serializable['bproof_confusion'] = {
        k: dict(v) for k, v in roll['bproof_confusion'].items()
    }
    out_path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    print(f'\nFull dump: {out_path}')


if __name__ == '__main__':
    main()
