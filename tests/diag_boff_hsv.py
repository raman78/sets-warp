"""
Diagnostic for _classify_boff_profession (layout_detector.py:1406).

Goal: understand WHY production HSV classifier confuses BOFF professions.
Read-only — does not modify code. Outputs:
  tests/_diag_out/boff_hsv/
    features.csv                    one row per BOFF crop (true, pred, features...)
    per_profession_stats.txt        median + p10/p90 per feature per true profession
    decision_trace.csv              which rule fired per crop
    confusion/<true>__<pred>/       up to 10 sample crops per cell
    summary.txt                     top-line accuracy + confusion matrix

Usage: PYTHONPATH=/home/raman/PycharmProjects/sets-warp python tests/diag_boff_hsv.py
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, '/home/raman/PycharmProjects/sets-warp')
from warp.recognition.layout_detector import LayoutDetector

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
TD   = ROOT / 'warp' / 'training_data'
OUT  = ROOT / 'tests' / '_diag_out' / 'boff_hsv'
CONF = OUT / 'confusion'
ANN  = TD / 'annotations.json'

MAX_SAMPLES_PER_CELL = 10
VIRTUAL_NAMES = frozenset({'__empty__', '__inactive__'})


def extract_features(crop_bgr):
    """Mirror of _classify_boff_profession internals — returns feature dict.
    The `pred` field is set by calling the production classifier (ground truth
    for the diagnostic — rules drift independently of feature extraction)."""
    ih, iw = crop_bgr.shape[:2]
    b = max(3, int(min(ih, iw) * 0.22))
    top    = crop_bgr[:b, :].reshape(-1, 3)
    bottom = crop_bgr[-b:, :].reshape(-1, 3)
    left   = crop_bgr[b:-b, :b].reshape(-1, 3)
    right  = crop_bgr[b:-b, -b:].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right])

    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat_mask = (hsv[:, 1] > 80) & (hsv[:, 2] > 60)
    sat_count = int(sat_mask.sum())

    feats = {
        'crop_w':     iw,
        'crop_h':     ih,
        'border_b':   b,
        'border_px':  border.shape[0],
        'sat_count':  sat_count,
        'red_lo':     0,   # H  0- 9
        'red_hi':     0,   # H160-175
        'red_total':  0,
        'amber':      0,   # H 10- 30
        'mid_blue':   0,   # H105-120
        'green':      0,   # H 48- 72
        'purple':     0,   # H115-145
        'bg_blue':    0,   # H 85-120
        'bin19':      0,   # H 95- 99 — pilot peak
        'bin20':      0,   # H100-104 — science peak
        'total_hue':  0,
        'mean_v':     0.0,
        'rule':       '',
        'pred':       '',
    }

    if sat_count < 8:
        feats['total_hue'] = 0
        # production returns None for sat<8
        feats['pred'] = LayoutDetector._classify_boff_profession(crop_bgr)
        feats['rule'] = 'sat_lt_8'
        return feats

    hues = hsv[sat_mask, 0]
    vals = hsv[sat_mask, 2]
    feats['mean_v'] = float(vals.mean())
    hist, _ = np.histogram(hues, bins=36, range=(0, 180))
    total = int(hist.sum()) or 1
    feats['total_hue'] = total

    def h(lo, hi):
        return int(hist[lo // 5: hi // 5 + 1].sum())

    feats['red_lo']    = h(0, 9)
    feats['red_hi']    = h(160, 175)
    feats['red_total'] = feats['red_lo'] + feats['red_hi']
    feats['amber']     = h(10, 30)
    feats['mid_blue']  = h(105, 120)
    feats['green']     = h(48, 72)
    feats['purple']    = h(115, 145)
    feats['bg_blue']   = h(85, 120)
    feats['bin19']     = int(hist[19])
    feats['bin20']     = int(hist[20])

    # Pred comes from production classifier (not inline-replicated rules)
    feats['pred'] = LayoutDetector._classify_boff_profession(crop_bgr)
    feats['rule'] = ''   # rule trace no longer available; kept empty for CSV compat
    return feats


def pct(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    CONF.mkdir(parents=True)

    if not ANN.exists():
        print(f'No annotations at {ANN}')
        return

    data = json.loads(ANN.read_text(encoding='utf-8'))

    rows = []
    confusion_crops = defaultdict(list)   # (true, pred) -> [crop_path]
    per_prof_feats = defaultdict(lambda: defaultdict(list))
    pred_counts = defaultdict(lambda: defaultdict(int))
    none_count = 0

    for _fname, anns in data.items():
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if ann.get('state') != 'confirmed':
                continue
            slot = ann.get('slot', '')
            if not slot.startswith('Boff '):
                continue
            name = (ann.get('name') or '').strip()
            if name in VIRTUAL_NAMES:
                continue   # skip inactive/empty placeholder crops
            true_prof = slot.replace('Boff ', '').lower()
            crop_name = ann.get('crop_name')
            if not crop_name:
                continue
            crop_path = TD / crop_name
            if not crop_path.exists():
                continue
            img = cv2.imread(str(crop_path))
            if img is None:
                continue

            f = extract_features(img)
            pred = f['pred'] if f['pred'] is not None else 'NONE'
            row = {'crop': str(crop_path.relative_to(TD)), 'true': true_prof, **f}
            rows.append(row)

            pred_counts[true_prof][pred] += 1
            if f['pred'] is None:
                none_count += 1

            for key in ('sat_count', 'red_lo', 'red_hi', 'red_total', 'amber',
                        'mid_blue', 'green', 'purple', 'bg_blue',
                        'bin19', 'bin20', 'total_hue', 'mean_v',
                        'crop_w', 'crop_h'):
                per_prof_feats[true_prof][key].append(f[key])

            cell = (true_prof, pred)
            if len(confusion_crops[cell]) < MAX_SAMPLES_PER_CELL:
                confusion_crops[cell].append(crop_path)

    if not rows:
        print('No crops matched.')
        return

    # --- features.csv ----------------------------------------------------------
    csv_path = OUT / 'features.csv'
    keys = ['crop', 'true', 'pred', 'rule', 'crop_w', 'crop_h', 'border_b',
            'border_px', 'sat_count', 'total_hue', 'mean_v',
            'red_lo', 'red_hi', 'red_total', 'amber', 'mid_blue', 'green',
            'purple', 'bg_blue', 'bin19', 'bin20']
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})

    # --- per_profession_stats.txt ---------------------------------------------
    stats_path = OUT / 'per_profession_stats.txt'
    with stats_path.open('w', encoding='utf-8') as fh:
        fh.write(f'# Per-profession feature distributions  (N = {len(rows)})\n\n')
        for prof in sorted(per_prof_feats):
            feats = per_prof_feats[prof]
            n = len(feats['sat_count'])
            fh.write(f'=== {prof}  (n={n}) ===\n')
            for key in ('crop_w', 'crop_h', 'sat_count', 'total_hue',
                        'red_lo', 'red_hi', 'red_total', 'amber', 'mid_blue',
                        'green', 'purple', 'bg_blue', 'bin19', 'bin20',
                        'mean_v'):
                vals = feats[key]
                if not vals:
                    continue
                fh.write(f'  {key:<12s} median={np.median(vals):8.1f}   '
                         f'p10={pct(vals, 10):7.1f}  p90={pct(vals, 90):7.1f}   '
                         f'mean={np.mean(vals):8.1f}\n')
            fh.write('\n')

    # --- summary.txt (confusion matrix + top-line) ----------------------------
    summary_path = OUT / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as fh:
        total_n = len(rows)
        correct = sum(r['true'] == r['pred'] for r in rows)
        fh.write(f'Tested {total_n} BOFF crops.\n')
        fh.write(f'Accuracy: {correct}/{total_n} = {correct/total_n*100:.1f}%\n')
        fh.write(f'Returned NONE (sat<8): {none_count}\n\n')

        fh.write('Confusion matrix (rows=true, cols=pred):\n')
        all_preds = sorted({p for prof in pred_counts for p in pred_counts[prof]})
        col_w = 12
        header = ' ' * 18 + ''.join(f'{p[:col_w-1]:>{col_w}}' for p in all_preds)
        fh.write(header + '\n')
        for true_prof in sorted(pred_counts):
            n_true = sum(pred_counts[true_prof].values())
            line = f'{true_prof[:17]:<18}'
            for p in all_preds:
                c = pred_counts[true_prof].get(p, 0)
                cell = f'{c}' if c else '.'
                line += f'{cell:>{col_w}}'
            line += f'  (n={n_true})'
            fh.write(line + '\n')

    # --- decision_trace.csv (just true, pred, rule) ---------------------------
    trace_path = OUT / 'decision_trace.csv'
    with trace_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'true', 'pred', 'rule'])
        for r in rows:
            w.writerow([r['crop'], r['true'], r['pred'], r['rule']])

    # --- confusion sample crops -----------------------------------------------
    for (tp, pp), paths in confusion_crops.items():
        cell_dir = CONF / f'{tp}__{pp}'
        cell_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, cell_dir / p.name)

    # Console output: compact summary
    total_n = len(rows)
    correct = sum(r['true'] == r['pred'] for r in rows)
    print(f'Tested {total_n} BOFF crops.')
    print(f'Accuracy: {correct}/{total_n} = {correct/total_n*100:.1f}%')
    print(f'Returned NONE (sat<8): {none_count}')
    print()
    print('Per-profession correctness:')
    for prof in sorted(pred_counts):
        n_true = sum(pred_counts[prof].values())
        c = pred_counts[prof].get(prof, 0)
        print(f'  {prof:<15s} {c:4d}/{n_true:<4d} ({c/n_true*100:5.1f}%)')
    print()
    print(f'Details written to: {OUT}')


if __name__ == '__main__':
    main()
