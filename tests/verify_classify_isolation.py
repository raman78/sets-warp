"""
CORE RULE verification — confirm marker detection + classify_patch path
does NOT touch warp/training_data/ at runtime.

Two layers of evidence:

  1. Static state assertions:
       - SETSIconMatcher._session_examples is empty
       - SETSIconMatcher._seeded_from_training_data is False
     before AND after the full pipeline runs.

  2. sys.addaudithook trace: every open() call during the run is
     logged, then we assert that no file under warp/training_data/
     was opened. The screen images live in /home/raman/STO_screens/
     so reading them does not violate the rule.

If this script exits 0, the production-style path is clean to wire
into layout_detector.py.

Usage:
  PYTHONPATH=/home/raman/PycharmProjects/sets-warp \\
    python tests/verify_classify_isolation.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

TRAINING_DATA_DIR = ROOT / 'warp' / 'training_data'
FORBIDDEN_PREFIX  = str(TRAINING_DATA_DIR.resolve())

# Audit hook log lives outside the matcher's reach.
opened_paths: list[str] = []


def _audit_hook(event: str, args):
    """Track every file open. Tolerant of non-string args."""
    if event != 'open':
        return
    if not args:
        return
    p = args[0]
    if isinstance(p, (str, bytes, Path)):
        try:
            opened_paths.append(str(p))
        except Exception:
            pass


sys.addaudithook(_audit_hook)


def _check_state(matcher_cls, label: str):
    se  = getattr(matcher_cls, '_session_examples', None)
    sd  = getattr(matcher_cls, '_seeded_from_training_data', None)
    assert se == [], (
        f'{label}: _session_examples is not empty (n={len(se)}) — '
        f'something seeded session examples on the production path')
    assert sd is False, (
        f'{label}: _seeded_from_training_data is True — '
        f'seed_from_training_data was called')
    print(f'  ✓ {label}: session_examples=[], seeded_from_training_data=False')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=3,
                    help='number of GT screens to test (default 3)')
    args = ap.parse_args()

    # Imports happen AFTER the audit hook is installed so even import-time
    # file reads are visible.
    from warp.recognition.icon_matcher import SETSIconMatcher
    from tests.diag_boff_classify import (
        _MockApp, evaluate_screen, ANN_PATH, CARGO,
    )
    from tests.diag_boff_markers import find_screen_files, VIRTUAL

    print('Pre-instantiation state:')
    _check_state(SETSIconMatcher, 'before SETSIconMatcher()')

    # Annotations.json is loaded HERE (by this verifier) only to pick
    # candidate screens for the test. The matcher itself will NOT see it.
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
    candidates = candidates[:args.limit]
    paths = find_screen_files([c[0] for c in candidates])

    matcher = SETSIconMatcher(_MockApp())
    print(f'\nMatcher constructed. Index size: {len(matcher._index)}')
    _check_state(SETSIconMatcher, 'after SETSIconMatcher()')

    # Snapshot opens up to this point — these are setup-time reads
    # (model weights, SETS image cache, cargo). They are EXPECTED.
    setup_opens = list(opened_paths)
    print(f'  setup-time file opens: {len(setup_opens)}')

    # Mark the boundary so we only audit the inference phase.
    boundary = len(opened_paths)

    print(f'\nRunning marker→projection→classify_patch on '
          f'{len(candidates)} screens…')
    n_real_correct = 0
    n_real_total   = 0
    for fname, gt_full in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        s, preds = evaluate_screen(fname, img, gt_full, matcher)
        for p in preds:
            if p['gt_state'] == 'real':
                n_real_total += 1
                if p['pred_name'] == p['gt_name']:
                    n_real_correct += 1
        print(f'  {fname[:50]:<50s}  '
              f'real={s["name_correct"]}/{s["name_total"]}')

    print(f'\nReal-ability exact-name accuracy: '
          f'{n_real_correct}/{n_real_total} '
          f'({n_real_correct / max(n_real_total, 1) * 100:.1f}%)')

    print('\nPost-inference state:')
    _check_state(SETSIconMatcher, 'after classify_patch')

    inference_opens = opened_paths[boundary:]
    print(f'\nFile opens during inference phase: {len(inference_opens)}')

    # Forbidden = anything under training_data/ EXCEPT it's only forbidden
    # because that path holds annotations + confirmed crops. The matcher
    # must not look there at runtime.
    forbidden = [p for p in inference_opens
                 if FORBIDDEN_PREFIX in str(Path(p).resolve())]
    if forbidden:
        print('  ✗ CORE RULE VIOLATION — these files were opened by the '
              'inference path:')
        for f in sorted(set(forbidden))[:20]:
            print(f'      {f}')
        sys.exit(1)
    print(f'  ✓ no inference-time read under {FORBIDDEN_PREFIX}')

    # Bonus: list distinct top-level dirs touched during inference, to
    # make it visible WHAT the matcher reads.
    seen_dirs = sorted({str(Path(p).parent) for p in inference_opens
                        if Path(p).is_absolute()})
    if seen_dirs:
        print('\nDirs read during inference:')
        for d in seen_dirs[:15]:
            print(f'    {d}')

    print('\n✓ classify_patch path is isolated from annotations.json '
          'and confirmed crops.')


if __name__ == '__main__':
    main()
