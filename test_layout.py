#!/usr/bin/env python3
"""
test_layout.py — Layout detection accuracy benchmark.

Runs layout_detector.detect() on screenshots that have confirmed WARP CORE
annotations and measures how well detected bboxes match the ground truth.

Metrics per screenshot:
  - slot recall   : confirmed slots found by detect() / total confirmed slots
  - count accuracy: slots where detected count == confirmed count
  - IoU           : mean IoU between best-matched detected/confirmed bbox pairs

Usage:
    python test_layout.py
    python test_layout.py --min-slots 5 --iou-thresh 0.30
    python test_layout.py --data-dir warp/training_data
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────────────

def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _best_iou(detected_boxes, confirmed_box):
    """Return the highest IoU between confirmed_box and any detected box."""
    if not detected_boxes:
        return 0.0
    return max(_iou(confirmed_box, d) for d in detected_boxes)


# Screen type → build_type passed to layout_detector.detect()
# (same mapping as WarpCoreWindow._STYPE_TO_BUILD — no full scan in test context)
_STYPE_TO_BUILD = {
    'SPACE_EQ':        'SPACE',
    'GROUND_EQ':       'GROUND',
    'TRAITS':          'SPACE_TRAITS',
    'BOFFS':           'BOFFS',
    'SPECIALIZATIONS': 'SPEC',
    'SPACE_MIXED':     'SPACE',
    'GROUND_MIXED':    'GROUND',
}


# Slots that layout_detector never returns (position-only text fields).
# Exclude from accuracy computation — they are always "missed" by design.
_POSITION_ONLY_SLOTS = frozenset({'Ship Name', 'Ship Type', 'Ship Tier'})


def _build_profile(confirmed_anns):
    """Count confirmed bboxes per slot → ship profile for detect()."""
    counts = defaultdict(int)
    for a in confirmed_anns:
        counts[a['slot']] += 1
    return dict(counts)


# ── core evaluation ───────────────────────────────────────────────────────────

def evaluate_screenshot(img_path, confirmed_anns, build_type, iou_thresh):
    """
    Run detect() on one screenshot and compare against confirmed_anns.
    Returns a dict with per-slot and aggregate stats, or None on error.
    """
    import cv2
    from warp.recognition.layout_detector import LayoutDetector

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    profile = _build_profile(confirmed_anns)

    # Suppress internal LayoutDetector log spam during test
    try:
        from src.setsdebug import log as _l
        _l.setLevel(50)  # CRITICAL — silence during batch run
    except Exception:
        pass

    try:
        detected = LayoutDetector().detect(img, build_type, profile)
    except Exception as e:
        print(f'  ERROR: detect() raised {e}')
        return None
    finally:
        try:
            _l.setLevel(20)  # restore INFO
        except Exception:
            pass

    # Group confirmed annotations by slot
    confirmed_by_slot = defaultdict(list)
    for a in confirmed_anns:
        confirmed_by_slot[a['slot']].append(tuple(a['bbox']))

    confirmed_slots = set(confirmed_by_slot)
    detected_slots  = set(detected)

    slot_hit   = 0    # confirmed slot present in detected
    slot_miss  = 0    # confirmed slot absent from detected
    count_ok   = 0    # slots with matching bbox count
    iou_scores = []

    for slot, gt_boxes in confirmed_by_slot.items():
        det_boxes = detected.get(slot, [])
        if det_boxes:
            slot_hit += 1
        else:
            slot_miss += 1
            iou_scores.extend([0.0] * len(gt_boxes))
            continue

        if len(det_boxes) == len(gt_boxes):
            count_ok += 1

        for gt_box in gt_boxes:
            iou_scores.append(_best_iou(det_boxes, gt_box))

    mean_iou   = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    iou_hits   = sum(1 for s in iou_scores if s >= iou_thresh)
    slot_recall = slot_hit / len(confirmed_slots) if confirmed_slots else 0.0

    return {
        'confirmed_slots': len(confirmed_slots),
        'detected_slots':  len(detected_slots),
        'slot_hit':        slot_hit,
        'slot_miss':       slot_miss,
        'slot_recall':     slot_recall,
        'count_ok':        count_ok,
        'mean_iou':        mean_iou,
        'iou_hits':        iou_hits,
        'total_gt_boxes':  len(iou_scores),
        'missed_slots':    sorted(confirmed_slots - detected_slots),
        'extra_slots':     sorted(detected_slots - confirmed_slots),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Layout detection accuracy benchmark')
    ap.add_argument('--data-dir',   default='warp/training_data',
                    help='Path to training_data directory')
    ap.add_argument('--min-slots',  type=int, default=3,
                    help='Minimum confirmed slots per screenshot to include (default 3)')
    ap.add_argument('--iou-thresh', type=float, default=0.30,
                    help='IoU threshold for counting a bbox as correctly located (default 0.30)')
    ap.add_argument('--verbose',    action='store_true',
                    help='Print per-screenshot details')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ann_file  = data_dir / 'annotations.json'
    stype_file = data_dir / 'screen_types.json'

    if not ann_file.exists():
        print(f'ERROR: {ann_file} not found'); sys.exit(1)
    if not stype_file.exists():
        print(f'ERROR: {stype_file} not found'); sys.exit(1)

    with open(ann_file)  as f: annotations  = json.load(f)
    with open(stype_file) as f: screen_types = json.load(f)

    # Aggregate counters
    agg = defaultdict(lambda: defaultdict(int))  # build_type → metric → count
    results_per_file = []
    skipped = 0

    for filename, ann_list in sorted(annotations.items()):
        confirmed = [a for a in ann_list if a.get('state') == 'confirmed'
                     and a.get('slot') and a.get('bbox')
                     and a.get('slot') not in _POSITION_ONLY_SLOTS]
        if len(confirmed) < args.min_slots:
            continue

        stype = screen_types.get(filename, 'UNKNOWN')
        build_type = _STYPE_TO_BUILD.get(stype)
        if not build_type:
            skipped += 1
            continue

        # Locate image file
        img_candidates = list(data_dir.glob(filename))
        if not img_candidates:
            # Try screenshots/ subdirectory
            img_candidates = list((data_dir / 'screenshots').glob(filename))
        if not img_candidates:
            img_candidates = list(data_dir.rglob(filename))
        if not img_candidates:
            skipped += 1
            continue
        img_path = img_candidates[0]

        result = evaluate_screenshot(img_path, confirmed, build_type, args.iou_thresh)
        if result is None:
            skipped += 1
            continue

        result['file']       = filename
        result['build_type'] = build_type
        result['stype']      = stype
        results_per_file.append(result)

        b = agg[build_type]
        b['files']           += 1
        b['confirmed_slots'] += result['confirmed_slots']
        b['slot_hit']        += result['slot_hit']
        b['slot_miss']       += result['slot_miss']
        b['count_ok']        += result['count_ok']
        b['iou_hits']        += result['iou_hits']
        b['total_gt_boxes']  += result['total_gt_boxes']
        b['iou_sum']         += result['mean_iou']

        if args.verbose:
            recall = result['slot_recall']
            miou   = result['mean_iou']
            status = 'OK ' if recall >= 0.80 and miou >= args.iou_thresh else 'WARN'
            print(f'[{status}] {filename}  stype={stype}  '
                  f'slots={result["slot_hit"]}/{result["confirmed_slots"]}  '
                  f'iou={miou:.2f}')
            if result['missed_slots']:
                print(f'       missed : {result["missed_slots"]}')
            if result['extra_slots']:
                print(f'       extra  : {result["extra_slots"]}')

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print(f'Layout detection accuracy  (iou_thresh={args.iou_thresh})')
    print('=' * 70)

    total_files = sum(v['files'] for v in agg.values())
    if total_files == 0:
        print('No screenshots with enough confirmed annotations found.')
        print(f'(skipped={skipped}  min-slots={args.min_slots})')
        sys.exit(0)

    for bt, b in sorted(agg.items()):
        slot_recall = b['slot_hit'] / b['confirmed_slots'] if b['confirmed_slots'] else 0
        count_acc   = b['count_ok'] / b['confirmed_slots'] if b['confirmed_slots'] else 0
        iou_acc     = b['iou_hits'] / b['total_gt_boxes']  if b['total_gt_boxes']  else 0
        mean_iou    = b['iou_sum']  / b['files']           if b['files']           else 0
        print(f'  {bt:<16}  files={b["files"]:3d}  '
              f'slot_recall={slot_recall:.1%}  '
              f'count_acc={count_acc:.1%}  '
              f'bbox_iou≥{args.iou_thresh:.2f}={iou_acc:.1%}  '
              f'mean_iou={mean_iou:.2f}')

    print('-' * 70)
    all_gt    = sum(v['total_gt_boxes'] for v in agg.values())
    all_hits  = sum(v['slot_hit'] for v in agg.values())
    all_conf  = sum(v['confirmed_slots'] for v in agg.values())
    all_iou_h = sum(v['iou_hits'] for v in agg.values())
    print(f'  TOTAL            files={total_files:3d}  '
          f'slot_recall={all_hits/all_conf:.1%}  '
          f'bbox_iou≥{args.iou_thresh:.2f}={all_iou_h/all_gt:.1%}'
          if all_gt else '  TOTAL  (no bboxes)')
    if skipped:
        print(f'  skipped={skipped} (no image found or unknown screen type)')
    print()

    # Worst performers
    worst = sorted(results_per_file, key=lambda r: r['slot_recall'])[:5]
    if worst and worst[0]['slot_recall'] < 1.0:
        print('Worst slot recall:')
        for r in worst:
            if r['slot_recall'] < 1.0:
                print(f'  {r["slot_recall"]:.0%}  {r["file"]}  missed={r["missed_slots"]}')
        print()


if __name__ == '__main__':
    main()
