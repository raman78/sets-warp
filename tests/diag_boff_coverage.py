"""
Detector-coverage diagnostic for BOFF seats.

Goal: break down the 26% of GT BOFFs that detector never covers (IoU<0.3 to
any predicted seat bbox) into named failure classes, so we know where to
spend engineering effort next.

Failure classes
---------------
  no_seats       : detect() returned 0 BOFF seats on this screen
  mirror_layout  : detector found seats, but on the OPPOSITE horizontal half
                   from the uncovered GT — classic STO mirror UI
  middle_image   : uncovered GT is in the middle third of the image (34-66%
                   width) — outside the 34% left/right strip heuristic
  same_region    : uncovered GT is in the SAME horizontal half as covered
                   GT — detector found the panel but missed rows / cols
  other          : none of the above

Read-only; writes JSON dump + per-class sample list.

Usage:
  PYTHONPATH=/home/raman/PycharmProjects/sets-warp \\
    /home/raman/PycharmProjects/sets-warp/.venv/bin/python tests/diag_boff_coverage.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, '/home/raman/PycharmProjects/sets-warp')
from warp.recognition.layout_detector import LayoutDetector

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
TD = ROOT / 'warp' / 'training_data'
ANN_PATH = TD / 'annotations.json'
SCREEN_TYPES_PATH = TD / 'screen_types.json'
SCREEN_TYPES_CONFIRMED_PATH = TD / 'screen_types_user_confirmed.json'
OUT = ROOT / 'tests' / '_diag_out' / 'boff_coverage'

IOU_THRESHOLD = 0.30
VIRTUAL = frozenset({'__empty__', '__inactive__'})


def iou(a, b):
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
                types.setdefault(k, v)
    return types


def find_images(names):
    candidates = [
        Path('/home/raman/STO_screens'),
        TD / 'migrated_images',
        TD,
    ]
    idx = {}
    names_set = set(names)
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            if p.name not in idx and p.name in names_set:
                idx[p.name] = p
    return idx


def _center_x(bb):
    return bb[0] + bb[2] / 2.0


def classify_miss(uncovered, covered, detected_seats, img_w, img_h):
    """Assign a failure class to a single uncovered GT bbox.

    Hierarchy (first match wins):
      panel_crop    : image is tiny (iw<500) — cropped BOFF panel, edge case
      no_seats      : detector returned 0 BOFF seats on this screen
      near_detector : nearest detected seat is within 10% of image width
                      — detector found right panel but missed a row/col/bbox
      mirror_layout : nearest detected seat is on opposite horizontal half
                      AND far from GT — classic wrong-side detection
      middle_image  : GT is middle (33-66%) AND no detected seat nearby
                      — BOFFs placed in center of full screen
      other
    """
    if img_w < 500:
        return 'panel_crop'
    if not detected_seats:
        return 'no_seats'

    u_cx = _center_x(uncovered)
    u_norm = u_cx / img_w

    seat_cxs = [_center_x(b) for b in detected_seats]
    nearest_dx = min(abs(u_cx - sx) for sx in seat_cxs)
    if nearest_dx < img_w * 0.10:
        return 'near_detector'

    u_left = u_norm < 0.5
    det_norms = [sx / img_w for sx in seat_cxs]
    det_left_any  = any(n < 0.5 for n in det_norms)
    det_right_any = any(n >= 0.5 for n in det_norms)
    if u_left and not det_left_any:
        return 'mirror_layout'
    if (not u_left) and not det_right_any:
        return 'mirror_layout'

    if 0.33 <= u_norm <= 0.66:
        return 'middle_image'

    return 'other'


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))
    screen_types = load_screen_types()

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

    paths = find_images([c[0] for c in candidates])
    detector = LayoutDetector()

    by_class = defaultdict(list)   # class -> list of (file, gt_bbox_info)
    per_screen_summary = []
    total_gt = 0
    total_covered = 0

    for fname, real_boff in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        screen_type = screen_types.get(fname) or 'SPACE_MIXED'
        try:
            layout = detector.detect(img, screen_type)
        except Exception as e:
            print(f'{fname}: detect() raised {e}')
            continue

        seats = {k: v for k, v in layout.items() if k.startswith('Boff Seat')}
        all_seat_bboxes = [b for boxes in seats.values() for b in boxes]

        covered_gt, uncovered_gt = [], []
        for gt in real_boff:
            gt_bb = gt['bbox']
            best = max((iou(sb, gt_bb) for sb in all_seat_bboxes), default=0.0)
            if best >= IOU_THRESHOLD:
                covered_gt.append(gt_bb)
            else:
                uncovered_gt.append(gt)

        total_gt       += len(real_boff)
        total_covered  += len(covered_gt)

        classes_here = defaultdict(int)
        for gt in uncovered_gt:
            cls = classify_miss(gt['bbox'], covered_gt, all_seat_bboxes, iw, ih)
            classes_here[cls] += 1
            by_class[cls].append({
                'file': fname,
                'gt_bbox': gt['bbox'],
                'gt_slot': gt.get('slot', ''),
                'gt_name': gt.get('name', ''),
                'img_w': iw, 'img_h': ih,
                'gt_cx_norm': _center_x(gt['bbox']) / iw,
                'n_seats_detected': len(all_seat_bboxes),
                'seat_cx_norms': sorted(round(_center_x(b) / iw, 3)
                                        for b in all_seat_bboxes)[:8],
            })

        per_screen_summary.append({
            'file': fname,
            'gt_n': len(real_boff),
            'covered': len(covered_gt),
            'uncovered': len(uncovered_gt),
            'seats_detected': len(all_seat_bboxes),
            'classes': dict(classes_here),
            'img_w': iw,
            'img_h': ih,
        })

    # ── Report ───────────────────────────────────────────────────────────────
    total_uncovered = total_gt - total_covered
    print(f'Screens tested:     {len(per_screen_summary)}')
    print(f'GT BOFF total:      {total_gt}')
    print(f'Covered (IoU≥0.3):  {total_covered} = {total_covered/total_gt*100:.1f}%')
    print(f'Uncovered:          {total_uncovered} = {total_uncovered/total_gt*100:.1f}%')
    print()
    print('Uncovered breakdown by failure class:')
    for cls in ('no_seats', 'mirror_layout', 'middle_image',
                'near_detector', 'panel_crop', 'other'):
        n = len(by_class.get(cls, []))
        pct_of_uncov = n / max(total_uncovered, 1) * 100
        pct_of_gt    = n / max(total_gt, 1) * 100
        print(f'  {cls:<16s} {n:4d}   '
              f'{pct_of_uncov:5.1f}% of uncovered   '
              f'{pct_of_gt:5.1f}% of all GT')

    # Per-class screen list (unique file names per class)
    print()
    print('Screens touched by each class (unique files):')
    for cls in ('no_seats', 'mirror_layout', 'middle_image',
                'near_detector', 'panel_crop', 'other'):
        items = by_class.get(cls, [])
        files = sorted({i['file'] for i in items})
        if not files:
            continue
        print(f'  {cls}:  ({len(files)} files, {len(items)} GT boxes)')
        for f in files:
            n_in_class = sum(1 for i in items if i['file'] == f)
            ps = next((s for s in per_screen_summary if s['file'] == f), None)
            seats = ps['seats_detected'] if ps else '?'
            print(f'    {f}  (class_n={n_in_class}, seats_detected={seats})')

    # Dump JSON
    dump = {
        'totals': {
            'screens_tested': len(per_screen_summary),
            'gt_total': total_gt,
            'covered': total_covered,
            'uncovered': total_uncovered,
        },
        'by_class_counts': {k: len(v) for k, v in by_class.items()},
        'by_class_details': {k: v for k, v in by_class.items()},
        'per_screen': per_screen_summary,
    }
    (OUT / 'coverage.json').write_text(json.dumps(dump, indent=2),
                                       encoding='utf-8')
    print(f'\nFull dump: {OUT / "coverage.json"}')


if __name__ == '__main__':
    main()
