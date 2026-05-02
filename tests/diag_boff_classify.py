"""
BOFF ability-slot CONTENT classifier diagnostic.

Read-only. Reuses the marker detector + per-seat ability-slot projection
from `tests/diag_boff_markers.py`, then for each projected bbox runs the
production `SETSIconMatcher.classify_patch()` (ML-only fast path) and
compares to the user-confirmed GT name.

Goal: separate "where are the slots?" (localizer, already 99.9% IoU≥.3)
from "what's in each slot?" (classifier accuracy on detected slots).

GT comes from `warp/training_data/annotations.json` — we keep both real
ability names and the virtual labels `__empty__` / `__inactive__`.

Metrics:
  - Top-1 accuracy (overall + per-state: real / empty / inactive)
  - Confusion of state buckets (real → real | empty → empty | …)
  - For "real" GT: rate of *exact* name match
  - Conf histogram of predictions (binned)

Output:
  tests/_diag_out/boff_classify/classify.json   — per-screen + rollup
  stdout                                        — per-screen line + totals

Usage:
  PYTHONPATH=/home/raman/PycharmProjects/sets-warp \\
    python tests/diag_boff_classify.py [--limit N] [--screen-filter sub]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2

warnings.filterwarnings('ignore', category=UserWarning)

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

# Reused localizer/projection helpers — keep classify diag focused on
# matcher behaviour, not on re-implementing detection.
from tests.diag_boff_markers import (   # noqa: E402
    detect_markers,
    best_panel,
    project_seat_slots,
    find_screen_files,
    estimate_icon_dims,
    VIRTUAL,
)
from warp.recognition.icon_matcher import SETSIconMatcher   # noqa: E402

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
CARGO    = ROOT / 'data' / 'cargo' / 'boff_abilities.json'
IMG_DIR  = ROOT / '.config' / 'images'

OUT_DIR  = ROOT / 'tests' / '_diag_out' / 'boff_classify'
OUT_JSON = OUT_DIR / 'classify.json'

IOU_THRESHOLD = 0.30


# ── Mock SETS app — wires real images dir + boff cache (production path) ──

class _MockCache:
    def __init__(self):
        self.boff_abilities: dict = {'all': {}}
        if CARGO.exists():
            cargo = json.loads(CARGO.read_text(encoding='utf-8'))
            for scope in ('space', 'ground'):
                for prof, ranks in cargo.get(scope, {}).items():
                    for rank_dict in ranks or []:
                        if not rank_dict:
                            continue
                        for name in rank_dict:
                            self.boff_abilities['all'][name] = {
                                'name': name, 'profession': prof}
        self.equipment = {}


class _MockApp:
    def __init__(self):
        self.cache  = _MockCache()
        self.config = {'config_subfolders': {'images': str(IMG_DIR)}}


# ── IoU helper (local copy — keeps file independent) ──────────────────────

def _iou(a, b):
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── State bucketing ───────────────────────────────────────────────────────

def _state_of(name: str) -> str:
    if name == '__empty__':
        return 'empty'
    if name == '__inactive__':
        return 'inactive'
    return 'real'


# ── Per-screen evaluation ─────────────────────────────────────────────────

def evaluate_screen(fname, img, gt_full, matcher):
    """gt_full: [(bbox, name)] — confirmed BOFF annotations incl. virtual.

    Returns (per_screen_dict, list_of_pred_records) where each pred record
    is {gt_name, gt_state, pred_name, pred_state, conf, iou}.
    """
    iw0, ih0 = estimate_icon_dims(img)
    markers = detect_markers(img, iw0, ih0)
    if len(markers) >= 3:
        ws = sorted(m[2] for m in markers)
        hs = sorted(m[3] for m in markers)
        iw0 = max(iw0, int(round(ws[len(ws) // 2] / 0.85)))
        ih0 = max(ih0, int(round(hs[len(hs) // 2] / 0.80)))
    panel = best_panel(markers, iw0, ih0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    proj_slots = project_seat_slots(panel, hsv=hsv)

    H, W = img.shape[:2]
    used_gt = set()
    preds = []

    for (_si, _sk, x, y, sw, sh, _c) in proj_slots:
        # Match this projected bbox to its best-IoU GT (greedy, no reuse).
        best_i, best_iou = -1, 0.0
        for gi, (gb, _gn) in enumerate(gt_full):
            if gi in used_gt:
                continue
            v = _iou((x, y, sw, sh), gb)
            if v > best_iou:
                best_iou, best_i = v, gi

        if best_iou < IOU_THRESHOLD or best_i < 0:
            continue
        used_gt.add(best_i)
        gt_name = gt_full[best_i][1]

        # Crop the projected patch (clamped) and classify ML-only.
        cx0 = max(0, x); cy0 = max(0, y)
        cx1 = min(W, x + sw); cy1 = min(H, y + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        patch = img[cy0:cy1, cx0:cx1]
        pred_name, pred_conf = matcher.classify_patch(patch)

        preds.append({
            'gt_name':    gt_name,
            'gt_state':   _state_of(gt_name),
            'pred_name':  pred_name,
            'pred_state': _state_of(pred_name),
            'conf':       round(float(pred_conf), 4),
            'iou':        round(float(best_iou), 3),
        })

    # Aggregate per-screen.
    state_correct = defaultdict(int)
    state_total   = defaultdict(int)
    name_correct  = 0
    name_total    = 0
    cm = defaultdict(lambda: defaultdict(int))   # gt_state → pred_state
    for p in preds:
        gs, ps = p['gt_state'], p['pred_state']
        state_total[gs] += 1
        cm[gs][ps] += 1
        if gs == ps:
            state_correct[gs] += 1
        if gs == 'real':
            name_total += 1
            if p['pred_name'] == p['gt_name']:
                name_correct += 1

    summary = {
        'file':           fname,
        'projected':      len(proj_slots),
        'matched_to_gt':  len(preds),
        'state_total':    dict(state_total),
        'state_correct':  dict(state_correct),
        'name_total':     name_total,
        'name_correct':   name_correct,
        'cm':             {k: dict(v) for k, v in cm.items()},
    }
    return summary, preds


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='test only first N screens (0 = all)')
    ap.add_argument('--screen-filter', type=str, default='')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))
    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        gt_full = [(a['bbox'], a.get('name', ''))
                   for a in entries
                   if a.get('state') == 'confirmed'
                   and a.get('slot', '').startswith('Boff ')
                   and a.get('bbox')]
        n_real = sum(1 for _, n in gt_full if n not in VIRTUAL)
        if n_real >= 5:
            candidates.append((fname, gt_full))

    if args.screen_filter:
        candidates = [c for c in candidates if args.screen_filter in c[0]]
    if args.limit:
        candidates = candidates[:args.limit]

    paths = find_screen_files([c[0] for c in candidates])
    print(f'Screens with ≥5 BOFF GT: {len(candidates)}, '
          f'findable on disk: {len(paths)}')

    matcher = SETSIconMatcher(_MockApp())

    rows         = []
    all_preds    = []
    state_total  = defaultdict(int)
    state_corr   = defaultdict(int)
    name_total   = 0
    name_correct = 0
    cm_all       = defaultdict(lambda: defaultdict(int))

    for fname, gt_full in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        s, preds = evaluate_screen(fname, img, gt_full, matcher)
        rows.append(s)
        all_preds.extend(preds)
        for k, v in s['state_total'].items():
            state_total[k] += v
        for k, v in s['state_correct'].items():
            state_corr[k] += v
        name_total   += s['name_total']
        name_correct += s['name_correct']
        for gs, sub in s['cm'].items():
            for ps, n in sub.items():
                cm_all[gs][ps] += n

        per_state = ' '.join(
            f'{k}={s["state_correct"].get(k, 0)}/{s["state_total"].get(k, 0)}'
            for k in ('real', 'empty', 'inactive')
            if s['state_total'].get(k))
        print(f'  {fname[:40]:<40s}  '
              f'matched={s["matched_to_gt"]}/{s["projected"]:<2d}  '
              f'{per_state}  '
              f'name={s["name_correct"]}/{s["name_total"]}')

    print('\n' + '=' * 80)
    print('Per-state state-bucket accuracy (top-1 matches GT bucket):')
    for k in ('real', 'empty', 'inactive'):
        tot = state_total.get(k, 0)
        if not tot:
            continue
        cor = state_corr.get(k, 0)
        print(f'  {k:<8s}: {cor}/{tot} ({cor / tot * 100:.1f}%)')

    overall_tot = sum(state_total.values())
    overall_cor = sum(state_corr.values())
    print(f'  overall : {overall_cor}/{overall_tot} '
          f'({overall_cor / max(overall_tot, 1) * 100:.1f}%)')

    print('\nReal-ability EXACT name accuracy:')
    print(f'  {name_correct}/{name_total} '
          f'({name_correct / max(name_total, 1) * 100:.1f}%)')

    print('\nConfusion matrix (rows = GT state, cols = predicted state):')
    states = ['real', 'empty', 'inactive']
    print(f'  {"":>10s}  ' + '  '.join(f'{s:>9s}' for s in states))
    for gs in states:
        row = '  '.join(f'{cm_all[gs].get(ps, 0):>9d}' for ps in states)
        print(f'  {gs:>10s}  {row}')

    # Conf bins (separate for correct vs incorrect at state level).
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    correct_bins   = [0] * (len(bins) - 1)
    incorrect_bins = [0] * (len(bins) - 1)
    for p in all_preds:
        c = p['conf']
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                if p['gt_state'] == p['pred_state']:
                    correct_bins[i]   += 1
                else:
                    incorrect_bins[i] += 1
                break
    print('\nConfidence histogram (state-bucket):')
    for i in range(len(bins) - 1):
        print(f'  [{bins[i]:.2f}-{bins[i + 1]:.2f})  '
              f'correct={correct_bins[i]:<5d}  '
              f'incorrect={incorrect_bins[i]:<5d}')

    OUT_JSON.write_text(json.dumps({
        'screens': rows,
        'totals': {
            'state_total':   dict(state_total),
            'state_correct': dict(state_corr),
            'name_total':    name_total,
            'name_correct':  name_correct,
            'cm':             {k: dict(v) for k, v in cm_all.items()},
        },
    }, indent=2), encoding='utf-8')
    print(f'\nResults JSON: {OUT_JSON}')


if __name__ == '__main__':
    main()
