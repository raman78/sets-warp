"""
Threshold sweep for _classify_boff_profession (layout_detector.py).

Uses features already on disk from diag_boff_hsv.py — no re-scanning crops.
Sweeps rule parameters, reports per-config accuracy + per-profession deltas.

Requires: tests/_diag_out/boff_hsv/features.csv  (run diag_boff_hsv.py first)

Output: tests/_diag_out/boff_hsv/sweep.txt
"""
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
OUT  = ROOT / 'tests' / '_diag_out' / 'boff_hsv'
FEAT = OUT / 'features.csv'
SWEEP_OUT = OUT / 'sweep.txt'

PROFS = ('command', 'engineering', 'intelligence', 'miracle worker',
         'pilot', 'science', 'tactical', 'temporal')


@dataclass(frozen=True)
class Config:
    """Rule-parameter knobs; defaults replicate current production."""
    red_ratio:           float = 0.05       # red_total/total threshold
    purple_ratio:        float = 0.07
    purple_abs:          float = 40
    green_ratio:         float = 0.10
    green_abs:           float = 25
    amber_ratio:         float = 0.12
    amber_abs:           float = 40
    # NEW: amber must dominate over bg_blue (False in production)
    amber_vs_bg_min:     float = 0.0        # require amber > bg_blue * this (0 = off)
    # Temporal-branch:
    temp_midblue_min:    float = 40         # absolute mid_blue threshold
    temp_midblue_ratio:  float = 0.28       # mid_blue >= amber * ratio
    temp_midblue_strong: float = 1e9        # mid_blue >= this alone → temporal (1e9 = off)
    # Pilot:
    pilot_bin19_min:     float = 30
    # Reorder:
    pilot_before_amber:  bool  = False


def classify(row: dict, cfg: Config) -> str:
    """Parameterized replica of _classify_boff_profession."""
    red_lo    = int(row['red_lo'])
    red_hi    = int(row['red_hi'])
    red_total = int(row['red_total'])
    amber     = int(row['amber'])
    mid_blue  = int(row['mid_blue'])
    green     = int(row['green'])
    purple    = int(row['purple'])
    bg_blue   = int(row['bg_blue'])
    bin19     = int(row['bin19'])
    bin20     = int(row['bin20'])
    total     = int(row['total_hue']) or 1

    # Skip classification if sat<8 (production returns None → fall through
    # to whatever slot voting does). Here we map None → 'science' (most common).
    if int(row['sat_count']) < 8:
        return 'science'

    # 1. Red (command / tactical)
    if red_total / total >= cfg.red_ratio:
        return 'command' if red_hi > red_lo else 'tactical'

    # 2. Purple (intelligence)
    if purple / total >= cfg.purple_ratio or purple >= cfg.purple_abs:
        return 'intelligence'

    # 3. Green (miracle worker)
    if green / total >= cfg.green_ratio or green >= cfg.green_abs:
        return 'miracle worker'

    # (optional) Pilot before amber
    if cfg.pilot_before_amber:
        if bin19 > bin20 and bin19 >= cfg.pilot_bin19_min:
            return 'pilot'

    # 4. Amber (engineering / temporal) with optional bg dominance guard
    amber_hit = amber / total >= cfg.amber_ratio or amber >= cfg.amber_abs
    if cfg.amber_vs_bg_min > 0:
        amber_hit = amber_hit and amber > bg_blue * cfg.amber_vs_bg_min
    if amber_hit:
        if (mid_blue >= cfg.temp_midblue_min
                and mid_blue >= amber * cfg.temp_midblue_ratio):
            return 'temporal'
        if mid_blue >= cfg.temp_midblue_strong:
            return 'temporal'
        return 'engineering'

    # 5. Pilot (after amber, if not already checked)
    if not cfg.pilot_before_amber:
        if bin19 > bin20 and bin19 >= cfg.pilot_bin19_min:
            return 'pilot'

    # 6. Default: science
    return 'science'


def evaluate(rows: list, cfg: Config) -> dict:
    """Run config on all rows → counts + confusion."""
    conf = defaultdict(lambda: defaultdict(int))
    correct = 0
    for r in rows:
        t = r['true']
        p = classify(r, cfg)
        conf[t][p] += 1
        if t == p:
            correct += 1
    per_prof = {}
    for prof in PROFS:
        n = sum(conf[prof].values())
        c = conf[prof].get(prof, 0)
        per_prof[prof] = (c, n)
    return {
        'total':    sum(sum(d.values()) for d in conf.values()),
        'correct':  correct,
        'per_prof': per_prof,
        'confusion': {t: dict(d) for t, d in conf.items()},
    }


def fmt_pp(per_prof):
    parts = []
    for prof in PROFS:
        c, n = per_prof[prof]
        if n:
            parts.append(f'{prof[:3]}:{c}/{n}')
    return '  '.join(parts)


def main():
    if not FEAT.exists():
        print(f'Missing {FEAT} — run diag_boff_hsv.py first.')
        return

    rows = []
    with FEAT.open(encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    report = []
    append = report.append

    # 0. Baseline (production-equivalent)
    base_cfg = Config()
    base = evaluate(rows, base_cfg)
    append(f'# BOFF profession sweep on {base["total"]} real crops\n')
    append(f'## Baseline (production defaults)')
    append(f'Accuracy: {base["correct"]}/{base["total"]} '
           f'= {base["correct"]/base["total"]*100:.1f}%')
    append(fmt_pp(base['per_prof']))
    append('')
    baseline_pp = base['per_prof']

    def compare(title: str, cfg: Config):
        r = evaluate(rows, cfg)
        acc = r['correct'] / r['total'] * 100
        base_acc = base['correct'] / base['total'] * 100
        delta = acc - base_acc
        line = f'acc={acc:5.1f}% ({delta:+5.1f})  |  '
        parts = []
        for prof in PROFS:
            c, n = r['per_prof'][prof]
            bc, bn = baseline_pp[prof]
            if n == 0:
                continue
            d = c - bc
            mark = '' if d == 0 else f' ({d:+d})'
            parts.append(f'{prof[:3]}:{c}/{n}{mark}')
        append(f'{title}')
        append(f'  {line}')
        append('  ' + '  '.join(parts))
        return r, acc

    # --- Sweep A: amber_vs_bg_min (should reduce science→eng/temp) ----------
    append('## Sweep A: require amber > bg_blue * X  (reduces Sci false-positives)')
    for x in (0.0, 0.30, 0.40, 0.45, 0.50, 0.60, 0.75):
        compare(f'A  amber_vs_bg_min={x:.2f}',
                Config(amber_vs_bg_min=x))
    append('')

    # --- Sweep B: temp_midblue_strong (absolute override) -------------------
    append('## Sweep B: add mid_blue >= Z alone → temporal  (recovers temp→eng)')
    for z in (1e9, 100, 90, 80, 70, 60):
        compare(f'B  temp_midblue_strong={z:g}',
                Config(temp_midblue_strong=z))
    append('')

    # --- Sweep C: temp_midblue_ratio loosening ------------------------------
    append('## Sweep C: loosen temp_midblue_ratio  (easier temporal trigger)')
    for y in (0.28, 0.25, 0.22, 0.20, 0.18):
        compare(f'C  temp_midblue_ratio={y:.2f}',
                Config(temp_midblue_ratio=y))
    append('')

    # --- Sweep D: pilot_before_amber ---------------------------------------
    append('## Sweep D: check pilot before amber')
    compare('D  pilot_before_amber=True',
            Config(pilot_before_amber=True))
    append('')

    # --- Sweep E: combo — best-of-class candidates --------------------------
    append('## Sweep E: combinations')
    combos = [
        ('E1 A+B',      Config(amber_vs_bg_min=0.45, temp_midblue_strong=80)),
        ('E2 A+B+D',    Config(amber_vs_bg_min=0.45, temp_midblue_strong=80,
                               pilot_before_amber=True)),
        ('E3 A+B+C',    Config(amber_vs_bg_min=0.45, temp_midblue_strong=80,
                               temp_midblue_ratio=0.22)),
        ('E4 A+B+C+D',  Config(amber_vs_bg_min=0.45, temp_midblue_strong=80,
                               temp_midblue_ratio=0.22, pilot_before_amber=True)),
        ('E5 stronger A', Config(amber_vs_bg_min=0.50, temp_midblue_strong=80,
                                 pilot_before_amber=True)),
        ('E6 weaker A',   Config(amber_vs_bg_min=0.35, temp_midblue_strong=80,
                                 pilot_before_amber=True)),
    ]
    for title, cfg in combos:
        compare(title, cfg)
    append('')

    # --- Finer grid around top combos ---------------------------------------
    append('## Sweep F: fine grid around best region')
    best_acc = 0.0
    best_title = ''
    best_cfg = None
    for avs in (0.30, 0.35, 0.40, 0.45, 0.50):
        for tms in (70, 75, 80, 85, 90):
            for tmr in (0.20, 0.22, 0.25):
                for pba in (False, True):
                    cfg = Config(amber_vs_bg_min=avs,
                                 temp_midblue_strong=tms,
                                 temp_midblue_ratio=tmr,
                                 pilot_before_amber=pba)
                    r = evaluate(rows, cfg)
                    acc = r['correct'] / r['total'] * 100
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = cfg
                        best_title = (f'avs={avs:.2f}  tms={tms}  tmr={tmr:.2f}  '
                                      f'pba={pba}')
    append(f'Best in grid: {best_title}')
    r = evaluate(rows, best_cfg)
    append(f'  acc={r["correct"]}/{r["total"]} = {r["correct"]/r["total"]*100:.2f}%')
    parts = []
    for prof in PROFS:
        c, n = r['per_prof'][prof]
        bc, bn = baseline_pp[prof]
        if n == 0:
            continue
        d = c - bc
        mark = '' if d == 0 else f' ({d:+d})'
        parts.append(f'{prof[:3]}:{c}/{n}{mark}')
    append('  ' + '  '.join(parts))
    append('')
    append(f'Best Config as dict: {asdict(best_cfg)}')
    append('')

    # --- Per-cell confusion delta for best vs baseline ----------------------
    append('## Confusion delta: best vs baseline (cells that changed)')
    base_conf = base['confusion']
    best_r = evaluate(rows, best_cfg)
    best_conf = best_r['confusion']
    changed_cells = []
    for t in PROFS:
        for p in PROFS:
            bc = base_conf.get(t, {}).get(p, 0)
            wc = best_conf.get(t, {}).get(p, 0)
            if bc != wc:
                changed_cells.append((t, p, bc, wc, wc - bc))
    changed_cells.sort(key=lambda x: -abs(x[4]))
    for t, p, bc, wc, d in changed_cells:
        mark = 'CORRECT' if t == p else 'wrong'
        append(f'  {t:<15s} → {p:<15s}  {bc} → {wc}  (delta {d:+d})  [{mark}]')
    append('')

    text = '\n'.join(report)
    SWEEP_OUT.write_text(text, encoding='utf-8')

    print(text[-2000:] if len(text) > 2000 else text)
    print(f'\nFull report: {SWEEP_OUT}')


if __name__ == '__main__':
    main()
