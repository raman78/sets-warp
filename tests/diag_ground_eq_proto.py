"""Prototype validation for ground_eq_geometry.

For every ground-annotated screen:
  1. Run detect_ground_eq_geometry() (pure detection, no GT)
  2. Project cell bboxes via project_cells()
  3. Compare against GT annotations from annotations.json (greedy IoU match)
  4. Save a side-by-side overlay PNG:
        green = projected cell, red = GT bbox, dashed = no-match
  5. Print summary stats (per slot + per screen + overall)

Output PNGs go to tests/_diag_out/ground_eq_proto/ (gitignored).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warp.recognition.ground_eq_geometry import (  # noqa: E402
    detect_ground_eq_geometry, project_cells, _run_ocr, _match_label,
    SLOT_KIT_MODULES, SLOT_KIT, SLOT_BODY_ARMOR, SLOT_EV_SUIT,
    SLOT_PERSONAL_SHIELD, SLOT_WEAPONS, SLOT_GROUND_DEVICES,
)

ANN_PATH = Path('warp/training_data/annotations.json')
SCREENS_ROOT = Path('warp/training_data/screen_types')
OUT_DIR = Path('tests/_diag_out/ground_eq_proto')

GROUND_SLOTS = {SLOT_KIT_MODULES, SLOT_KIT, SLOT_BODY_ARMOR, SLOT_EV_SUIT,
                SLOT_PERSONAL_SHIELD, SLOT_WEAPONS, SLOT_GROUND_DEVICES}

IOU_THRESHOLD = 0.30


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(projected: list[tuple], gt: list[tuple]
                  ) -> tuple[list[tuple], list[int], list[int]]:
    """Return (matches, unmatched_proj_idx, unmatched_gt_idx).
    matches = [(proj_idx, gt_idx, iou), ...]
    """
    pairs = []
    for i, p in enumerate(projected):
        for j, g in enumerate(gt):
            iou = _bbox_iou(p, g)
            if iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matches = []
    for iou, i, j in pairs:
        if iou < IOU_THRESHOLD:
            break
        if i in used_p or j in used_g:
            continue
        matches.append((i, j, iou))
        used_p.add(i)
        used_g.add(j)
    up = [i for i in range(len(projected)) if i not in used_p]
    ug = [j for j in range(len(gt)) if j not in used_g]
    return matches, up, ug


def _load_gt(ann_data: dict, screen_name: str) -> dict[str, list[tuple]]:
    """Return {slot: [(x, y, w, h), ...]} of confirmed GT for a screen."""
    out: defaultdict[str, list[tuple]] = defaultdict(list)
    anns = ann_data.get(screen_name)
    if not isinstance(anns, list):
        return {}
    for a in anns:
        if not isinstance(a, dict):
            continue
        s = a.get('slot')
        if s not in GROUND_SLOTS or a.get('state') != 'confirmed':
            continue
        bb = a.get('bbox')
        if isinstance(bb, list) and len(bb) == 4:
            out[s].append(tuple(bb))
    return dict(out)


SLOT_COLORS = {
    SLOT_KIT_MODULES:     (0, 255, 255),   # yellow
    SLOT_KIT:             (255, 200, 0),   # orange-blue
    SLOT_BODY_ARMOR:      (255, 128, 0),   # orange
    SLOT_EV_SUIT:         (255, 0, 255),   # magenta
    SLOT_PERSONAL_SHIELD: (0, 255, 0),     # green
    SLOT_WEAPONS:         (255, 255, 0),   # cyan
    SLOT_GROUND_DEVICES:  (0, 200, 255),   # amber
}


def _draw_bbox(img, b, color, thickness=2, dashed=False):
    x, y, w, h = b
    if not dashed:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    else:
        # Simple dashed approximation: 4 short corner segments
        L = 6
        cv2.line(img, (x, y), (x + L, y), color, thickness)
        cv2.line(img, (x, y), (x, y + L), color, thickness)
        cv2.line(img, (x + w - L, y), (x + w, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + L), color, thickness)
        cv2.line(img, (x, y + h - L), (x, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x + L, y + h), color, thickness)
        cv2.line(img, (x + w, y + h - L), (x + w, y + h), color, thickness)
        cv2.line(img, (x + w - L, y + h), (x + w, y + h), color, thickness)


def _render(img, geom, projected, gt, matches, screen_name,
            ocr_label_boxes=None) -> np.ndarray:
    out = img.copy()

    # White rectangles around OCR-detected EQ-label tokens
    if ocr_label_boxes:
        for slot, (x0, y0, x1, y1) in ocr_label_boxes:
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
            cv2.putText(out, slot, (x0, max(0, y0 - 2)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # Faint label markers (cy of each OCR anchor) on the left edge
    if geom is not None:
        for slot, cy in geom.slot_label_cys.items():
            cv2.line(out, (0, cy), (20, cy), (0, 255, 0), 1)
            cv2.putText(out, slot[:4], (24, cy + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        if geom.km_label_cy is not None:
            cv2.line(out, (0, geom.km_label_cy), (20, geom.km_label_cy),
                     (0, 255, 255), 1)

    # GT bboxes (red, slightly thicker)
    for slot, bbs in gt.items():
        for bb in bbs:
            _draw_bbox(out, bb, (0, 0, 255), thickness=2)

    # Projected bboxes
    matched_proj = {i for i, _, _ in matches}
    for idx, (slot, bb) in enumerate(projected):
        color = SLOT_COLORS.get(slot, (255, 255, 255))
        dashed = idx not in matched_proj
        _draw_bbox(out, bb, color, thickness=2, dashed=dashed)

    # Header text
    if geom is not None:
        info = (f'{screen_name}  pitch={geom.row_pitch} '
                f'cell={int(geom.cell_w)}x{int(geom.cell_h)} '
                f'col_L={geom.col_left_x} col_R={geom.col_right_x} '
                f'mode={geom.mode}')
    else:
        info = f'{screen_name}  NO GEOMETRY'
    cv2.putText(out, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANN_PATH.read_text())

    # Discover ground screens — any annotation with a GROUND_SLOTS slot
    ground_screens: list[str] = []
    for name, anns in ann.items():
        if not isinstance(anns, list):
            continue
        for a in anns:
            if isinstance(a, dict) and a.get('slot') in GROUND_SLOTS \
                    and a.get('state') == 'confirmed':
                ground_screens.append(name)
                break

    # Per-slot accumulators
    per_slot_proj = defaultdict(int)
    per_slot_gt = defaultdict(int)
    per_slot_match = defaultdict(int)
    per_slot_iou_sum = defaultdict(float)
    no_geom_screens: list[str] = []

    print(f'Found {len(ground_screens)} ground-annotated screens\n')

    for name in sorted(ground_screens):
        # Find the screenshot file
        path = None
        for sub in SCREENS_ROOT.iterdir():
            if not sub.is_dir():
                continue
            cand = sub / name
            if cand.exists():
                path = cand
                break
        if path is None:
            print(f'  SKIP (file not found): {name}')
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Run OCR separately to draw white rectangles around matched labels
        ocr_tokens = _run_ocr(img)
        ocr_label_boxes = []
        for tok in ocr_tokens:
            slot = _match_label(tok)
            if slot is None:
                continue
            ocr_label_boxes.append(
                (slot, (tok['x0'], tok['y0'], tok['x1'], tok['y1'])))

        geom = detect_ground_eq_geometry(img)
        if geom is None:
            no_geom_screens.append(name)
            print(f'  {name}: GEOM = None  (size={w}x{h})')
            cv2.imwrite(
                str(OUT_DIR / f'{Path(name).stem}.png'),
                _render(img, None, [], _load_gt(ann, name), [], name,
                        ocr_label_boxes=ocr_label_boxes))
            continue

        cells = project_cells(geom, w, h)
        projected_flat: list[tuple] = []   # [(slot, bbox), ...]
        for slot, bbs in cells.items():
            for bb in bbs:
                projected_flat.append((slot, bb))

        gt = _load_gt(ann, name)

        # Per-slot greedy IoU match
        screen_matches = []
        screen_proj_match = 0
        screen_proj_total = 0
        screen_gt_total = 0
        for slot in GROUND_SLOTS:
            proj_bbs = [bb for s, bb in projected_flat if s == slot]
            gt_bbs = gt.get(slot, [])
            per_slot_proj[slot] += len(proj_bbs)
            per_slot_gt[slot] += len(gt_bbs)
            screen_proj_total += len(proj_bbs)
            screen_gt_total += len(gt_bbs)
            if not proj_bbs or not gt_bbs:
                continue
            matches, _, _ = _greedy_match(proj_bbs, gt_bbs)
            per_slot_match[slot] += len(matches)
            screen_proj_match += len(matches)
            for _, _, iou in matches:
                per_slot_iou_sum[slot] += iou
            # Record matches for visualization (using flat indices)
            slot_proj_idx_offset = next(
                i for i, (s, _) in enumerate(projected_flat) if s == slot)
            for pi, gi, iou in matches:
                screen_matches.append((slot_proj_idx_offset + pi, gi, iou))

        # Visualization
        viz = _render(img, geom, projected_flat, gt, screen_matches, name,
                      ocr_label_boxes=ocr_label_boxes)
        cv2.imwrite(str(OUT_DIR / f'{Path(name).stem}.png'), viz)

        rate = (screen_proj_match / screen_gt_total * 100
                if screen_gt_total else 0.0)
        print(f'  {name:55s}  proj={screen_proj_total:2d} '
              f'gt={screen_gt_total:2d} match={screen_proj_match:2d} '
              f'({rate:5.1f}%)')

    # Summary
    print('\n=== Per-slot summary (IoU≥0.30) ===')
    tot_match = tot_gt = tot_proj = 0
    tot_iou = 0.0
    for slot in [SLOT_KIT_MODULES, SLOT_KIT, SLOT_BODY_ARMOR, SLOT_EV_SUIT,
                 SLOT_PERSONAL_SHIELD, SLOT_WEAPONS, SLOT_GROUND_DEVICES]:
        m = per_slot_match[slot]
        g = per_slot_gt[slot]
        p = per_slot_proj[slot]
        recall = (m / g * 100) if g else 0.0
        precision = (m / p * 100) if p else 0.0
        mean_iou = (per_slot_iou_sum[slot] / m) if m else 0.0
        print(f'  {slot:20s}  match={m:3d}/gt={g:3d} '
              f'(recall={recall:5.1f}%)  '
              f'proj={p:3d} (precision={precision:5.1f}%)  '
              f'mean_iou={mean_iou:.3f}')
        tot_match += m
        tot_gt += g
        tot_proj += p
        tot_iou += per_slot_iou_sum[slot]

    overall_recall = (tot_match / tot_gt * 100) if tot_gt else 0.0
    overall_precision = (tot_match / tot_proj * 100) if tot_proj else 0.0
    overall_iou = (tot_iou / tot_match) if tot_match else 0.0
    print(f'  {"OVERALL":20s}  match={tot_match}/gt={tot_gt} '
          f'(recall={overall_recall:.1f}%)  proj={tot_proj} '
          f'(precision={overall_precision:.1f}%)  mean_iou={overall_iou:.3f}')

    if no_geom_screens:
        print(f'\nNo-geometry screens: {len(no_geom_screens)}')
        for n in no_geom_screens:
            print(f'  {n}')

    print(f'\nVisualizations → {OUT_DIR}/')


if __name__ == '__main__':
    main()
