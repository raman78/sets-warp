"""
EQ-anchored sliding-window scan for BOFF panel detection — v2 (icon-based).

This rewrite replaces the brightness-statistics primitive `_detect_boffs()`
with the production icon-recognition path:

  1. Dense full-image scan with `IconMatcher.classify_patch()` (EfficientNet)
  2. NMS + enrich each detection with item type via `_get_item_type`
  3. Filter to `__boff_*` typed detections (the actual BOFF icons we want)
  4. EQ-anchored sliding window over non-EQ region
  5. Score window = (# of BOFF dets inside) + structural-fit bonus for
     the canonical 3+2 layout (or smaller 2+1 / 2+2 / 3+1)

Why this rewrite:
  Previous prototype called `_detect_boffs()` per window — that function uses
  brightness statistics only and never verifies that what it found is an icon.
  It happily returns "seats" inside ship-renderings, text blocks, and noise.
  By doing actual icon classification first, we only consider patches that
  EfficientNet recognises as BOFF abilities — no false positives from noise.

Read-only. GT EQ used as anchor (CORE RULE: production must use detected EQ).

Usage:
  PYTHONPATH=/home/raman/PycharmProjects/sets-warp \\
    /home/raman/PycharmProjects/sets-warp/.venv/bin/python tests/diag_boff_scan.py

Optional:
  --limit N           test only first N screens
  --screen-filter S   substring match on filename
  --verbose           print per-screen window decisions and dets
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, '/home/raman/PycharmProjects/sets-warp')
from warp.recognition.layout_detector import (  # noqa: E402
    _get_item_type, _nms_boxes, LayoutDetector,
)
from warp.recognition.icon_matcher import SETSIconMatcher  # noqa: E402

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
TD = ROOT / 'warp' / 'training_data'
ANN_PATH = TD / 'annotations.json'
SCREEN_TYPES_PATH = TD / 'screen_types.json'
SCREEN_TYPES_CONFIRMED_PATH = TD / 'screen_types_user_confirmed.json'
CARGO = ROOT / '.config' / 'cargo' / 'boff_abilities.json'
OUT = ROOT / 'tests' / '_diag_out' / 'boff_scan'

IOU_THRESHOLD = 0.30
# match() with candidate_names=boff returns conf 0.4-0.6 on real BOFF crops
# and 0.0-0.01 on noise. 0.30 floor catches all real BOFFs while rejecting noise.
# (classify_patch ML-only returns 0.001-0.05 even on real BOFFs — too weak alone.)
SCAN_CONF_MIN = 0.30
# Aggressive NMS: stride icon/3 generates many overlapping dets per real icon.
# 0.25 IoU merges all dets within ~25% pixel overlap — keeps best per location.
SCAN_NMS_IOU  = 0.25

VIRTUAL = frozenset({'__empty__', '__inactive__'})

EQ_PREFIXES = (
    'Fore Weapon', 'Aft Weapon',
    'Deflector', 'Secondary Deflector',
    'Engines', 'Impulse Engines',
    'Shield', 'Shields',
    'Warp Core', 'Singularity Core',
    'Devices',
    'Tactical Console', 'Engineering Console', 'Science Console',
    'Universal Console',
    'Hangar', 'Experimental',
)


def is_eq_slot(slot: str) -> bool:
    return any(slot.startswith(p) for p in EQ_PREFIXES)


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
    return inter / (aw * ah + bw * bh - inter)


def boxes_overlap(a, b, margin=0):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return (ax1 - margin < bx2 and ax2 + margin > bx1
            and ay1 - margin < by2 and ay2 + margin > by1)


def union_bbox(boxes):
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[0] + b[2] for b in boxes]
    ys2 = [b[1] + b[3] for b in boxes]
    return (min(xs1), min(ys1),
            max(xs2) - min(xs1), max(ys2) - min(ys1))


def load_screen_types():
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
    candidates = [Path('/home/raman/STO_screens'), TD / 'migrated_images', TD]
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


# ── MockApp wires production IconMatcher with real SETS images dir + boff cache ──

class _MockCache:
    def __init__(self):
        self.equipment = {}
        self.traits = {}
        self.starship_traits = {}
        self.boff_abilities = {}
        if CARGO.exists():
            try:
                self.boff_abilities = json.loads(CARGO.read_text(encoding='utf-8'))
            except Exception:
                self.boff_abilities = {}


class _MockApp:
    def __init__(self):
        self.cache = _MockCache()
        self.config = {
            'config_subfolders': {
                'images': str(ROOT / '.config' / 'images'),
            }
        }


def _build_boff_name_set(boff_cache):
    names = set()
    for env in ('space', 'ground'):
        for prof, ranks in (boff_cache or {}).get(env, {}).items():
            if not isinstance(ranks, list):
                continue
            for rd in ranks:
                if isinstance(rd, dict):
                    names.update(rd.keys())
    return names


def dense_scan(matcher, img, boff_names, eq_bbox=None):
    """Full-image dense scan via `match()` restricted to BOFF candidates.

    Why match() not classify_patch:
      classify_patch (ML-only) softmaxes over 3249 classes — even real BOFF
      crops return conf 0.001-0.05, useless as a presence signal. match()
      runs template matching against the SETS image cache; restricted to the
      ~228 BOFF templates it returns 0.4-0.6 on real BOFFs and 0.00-0.01 on
      noise, giving a clean discriminator.

    EQ exclusion: skip patches whose center falls inside the EQ region
    (padded by icon_est) — those are equipment by definition, not BOFFs.

    Each output entry: (x, y, w, h, name, conf, item_type='__boff_<prof>')
    """
    h, w = img.shape[:2]
    # Icon size based on observed standalone-MIXED dims (~2.1% × 4.7%).
    # NON-square — real BOFF icons are slightly taller than wide (32×38).
    # Square patches stretched 32→37 break template alignment.
    icon_w = max(20, int(round(w * 0.021)))
    icon_h = max(28, int(round(h * 0.047)))
    # Stride 1/3 of icon to ensure ≥1 well-aligned probe per icon position.
    stride = max(6, icon_w // 3)

    cache = matcher._sets.cache

    # EQ exclusion bounds (with margin = icon_w)
    eq_x0 = eq_y0 = eq_x1 = eq_y1 = -1
    if eq_bbox:
        eq_x0 = eq_bbox[0] - icon_w
        eq_y0 = eq_bbox[1] - icon_h
        eq_x1 = eq_bbox[0] + eq_bbox[2] + icon_w
        eq_y1 = eq_bbox[1] + eq_bbox[3] + icon_h

    raw = []
    for y in range(0, h - icon_h + 1, stride):
        for x in range(0, w - icon_w + 1, stride):
            cx = x + icon_w // 2
            cy = y + icon_h // 2
            if eq_x0 <= cx < eq_x1 and eq_y0 <= cy < eq_y1:
                continue
            patch = img[y:y + icon_h, x:x + icon_w]
            if float(patch.std()) < 12.0:
                continue
            name, conf, _, _ = matcher.match(patch, candidate_names=boff_names)
            if conf >= SCAN_CONF_MIN and name:
                raw.append((x, y, icon_w, icon_h, name, conf))

    dets = _nms_boxes(raw, iou_thr=SCAN_NMS_IOU)

    # Profession cross-check: real BOFF icon must have BOTH a template match
    # AND a matching profession color. Eliminates noise patches whose nearest
    # template match happens to be a generic-looking BOFF icon.
    enriched = []
    n_color_rejected = 0
    for d in dets:
        itype = _get_item_type(
            d[4], cache.equipment, cache.traits,
            cache.starship_traits, cache.boff_abilities,
        )
        # Cross-check profession color (only for BOFF-typed dets)
        color_match = True
        if itype.startswith('__boff_'):
            x, y, dw, dh = d[:4]
            crop = img[y:y + dh, x:x + dw]
            color_prof = LayoutDetector._classify_boff_profession(crop)
            tmpl_prof = itype.replace('__boff_', '')
            # 'temporal' ↔ 'temporal_operative' alias; 'miracle worker' has space
            color_match = (color_prof == tmpl_prof
                           or (color_prof and tmpl_prof.startswith(color_prof))
                           or (tmpl_prof and color_prof and color_prof.startswith(tmpl_prof.split(' ')[0])))
        if not color_match:
            n_color_rejected += 1
            continue
        enriched.append(d + (itype,))
    if n_color_rejected:
        print(f'    color cross-check rejected {n_color_rejected}/{len(dets)} dets')
    return enriched, icon_w


def cluster_rows_xy(dets, icon_est):
    """Group detections into rows by Y (gap < icon_est * 0.5)."""
    gap = icon_est * 0.5
    rows = []
    for d in sorted(dets, key=lambda d: d[1]):
        for r in rows:
            if abs(d[1] - r[0][1]) <= gap:
                r.append(d)
                break
        else:
            rows.append([d])
    return rows


def split_columns(row, icon_est):
    """Split a row's dets into (left_dets, right_dets) by gap >= icon_est.

    BOFF row has up to 8 icons total: 4 left ability + 4 right ability with a
    rank-icon gap between L and R groups. Traits/rep have no gap.
    """
    if not row:
        return [], []
    s = sorted(row, key=lambda d: d[0])
    if len(s) == 1:
        return s, []
    # Find biggest x-gap between consecutive icons
    best_i = 0
    best_gap = 0
    for i in range(len(s) - 1):
        right_edge_left = s[i][0] + s[i][2]
        gap = s[i + 1][0] - right_edge_left
        if gap > best_gap:
            best_gap = gap
            best_i = i
    if best_gap >= icon_est * 0.6:
        return s[:best_i + 1], s[best_i + 1:]
    return s, []


HIGH_CONF = 0.40   # threshold for "anchor" (still reported in debug)


def score_window(boff_dets, window, icon_est):
    """Score a candidate BOFF window against canonical structural rules.

    Rules (gameplay-fixed, see todo_boff_tier_grid.md):
      - 2 columns: Left primary (1-3 rows) + Right spec (0-2 rows)
      - left_rows >= right_rows  (hard)
      - 4 abilities per seat (max), ie. up to 8 icons per row
      - total seats: 2-5; total icons: 4-20

    Score = sum(conf) + structural bonuses.
    Window must contain at least HIGH_CONF_MIN_N anchors above HIGH_CONF —
    a wall of weak matches in noise area should NOT beat a sparse-but-strong
    match in the real BOFF panel.

    Returns (score, debug_dict) — score < 0 disqualifies.
    """
    wx, wy, ww, wh = window
    inside = [d for d in boff_dets
              if d[0] >= wx and d[1] >= wy
              and d[0] + d[2] <= wx + ww and d[1] + d[3] <= wy + wh]
    n = len(inside)
    if n < 4:
        return -1.0, {'reason': f'n={n}<4'}
    # No upper cap: physical max is 20 BOFF abilities, but noise/empty seats
    # also template-match weakly. We rely on n_hi anchors + structural rules
    # to discriminate, not on raw count.

    n_hi = sum(1 for d in inside if d[5] >= HIGH_CONF)
    rows = cluster_rows_xy(inside, icon_est)
    if not rows:
        return -1.0, {'reason': 'no rows'}

    left_rows = []
    right_rows = []
    row_info = []
    for r in rows:
        L, R = split_columns(r, icon_est)
        info = {'n': len(r), 'L': len(L), 'R': len(R), 'y': min(d[1] for d in r)}
        row_info.append(info)
        if L:
            left_rows.append(r)
        if R:
            right_rows.append(r)

    nL = len(left_rows)
    nR = len(right_rows)

    if nL < 1:
        return -1.0, {'reason': f'no left rows', 'rows': row_info}
    if nL > 3:
        return -1.0, {'reason': f'left_rows={nL}>3', 'rows': row_info}
    if nR > 2:
        return -1.0, {'reason': f'right_rows={nR}>2', 'rows': row_info}
    if nR > nL:
        return -1.0, {'reason': f'R>L ({nR}>{nL})', 'rows': row_info}

    # Per-row icon-count cap: real BOFF row has 4 left + 4 right icons max.
    for info in row_info:
        if info['n'] > 9:
            return -1.0, {'reason': f"row n={info['n']}>9", 'rows': row_info}

    sum_conf = sum(d[5] for d in inside)

    shape_bonus = 5.0 if (nL, nR) in {(2, 1), (3, 1), (2, 2), (3, 2)} else 0.0
    paired_rows = sum(1 for info in row_info if info['L'] and info['R'])
    pair_bonus = 1.0 * paired_rows
    # Reward rows that contain at least one hi-conf anchor — evidence spread
    # across multiple rows is much stronger than a cluster in one area.
    hi_row_ys = set()
    for row in rows:
        if any(d[5] >= HIGH_CONF for d in row):
            hi_row_ys.add(min(d[1] for d in row))
    hi_bonus = 3.0 * len(hi_row_ys)

    score = sum_conf + shape_bonus + pair_bonus + hi_bonus
    return score, {'n': n, 'n_hi': n_hi, 'nL': nL, 'nR': nR,
                   'rows': row_info, 'sum_conf': round(sum_conf, 2),
                   'shape_bonus': shape_bonus, 'pair_bonus': pair_bonus,
                   'hi_bonus': hi_bonus}


def sweep_for_boff(boff_dets, img_shape, eq_bbox, icon_est, verbose=False):
    """Sweep BOFF-sized windows over non-EQ region; pick best by score."""
    h, w = img_shape[:2]
    # panel_h = 3 × row_pitch, where row_pitch ≈ 2.5 × icon_h (icon_h ≈ h×0.047)
    # This covers exactly 3 BOFF ability rows (canonical T6 layout).
    icon_h_est = max(int(h * 0.047), icon_est)
    row_pitch  = int(icon_h_est * 2.5)
    panel_w = max(int(w * 0.22), icon_est * 6)
    panel_h = max(3 * row_pitch, icon_est * 4)
    stride_x = max(20, int(w * 0.05))
    stride_y = max(20, int(h * 0.05))
    margin = icon_est

    y_start = int(h * 0.05)

    best = (None, -1.0, None)  # (window, score, debug)
    n_eval = 0
    for y in range(y_start, h - panel_h + 1, stride_y):
        for x in range(0, w - panel_w + 1, stride_x):
            window = (x, y, panel_w, panel_h)
            if eq_bbox and boxes_overlap(window, eq_bbox, margin=margin):
                continue
            s, dbg = score_window(boff_dets, window, icon_est)
            n_eval += 1
            if verbose:
                if s > 0:
                    print(f'    OK  win=({x:4},{y:4})  '
                          f"n={dbg.get('n')} hi={dbg.get('n_hi')} L={dbg.get('nL')} R={dbg.get('nR')} "
                          f'score={s:.2f}')
                elif dbg.get('n', 0) >= 2:
                    print(f'    rej win=({x:4},{y:4}) {dbg.get("reason","")} '
                          f"n={dbg.get('n', '-')}")
            if s > best[1]:
                best = (window, s, dbg)

    return best[0], best[1], best[2] if best[0] else {}, n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--screen-filter', type=str, default='')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    matcher = SETSIconMatcher(_MockApp())
    boff_names = _build_boff_name_set(matcher._sets.cache.boff_abilities)
    print(f'BOFF candidate set: {len(boff_names)} names')

    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))

    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        real_boff = [a for a in entries
                     if a.get('state') == 'confirmed'
                     and a.get('slot', '').startswith('Boff ')
                     and a.get('name') not in VIRTUAL
                     and a.get('bbox')]
        eq = [a for a in entries
              if a.get('state') == 'confirmed'
              and is_eq_slot(a.get('slot', ''))
              and a.get('bbox')]
        if len(real_boff) >= 4:
            candidates.append((fname, real_boff, eq))
    candidates.sort(key=lambda x: -len(x[1]))

    if args.screen_filter:
        candidates = [c for c in candidates if args.screen_filter in c[0]]
    if args.limit:
        candidates = candidates[:args.limit]

    paths = find_images([c[0] for c in candidates])
    print(f'Screens with ≥4 BOFF GT: {len(candidates)}   '
          f'findable on disk: {len(paths)}')

    total_gt = 0
    total_covered = 0
    rows = []
    no_eq_count = 0
    per_screen_dump = []

    for idx, (fname, real_boff, eq) in enumerate(candidates, 1):
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        eq_boxes = [a['bbox'] for a in eq]
        eq_bbox = union_bbox(eq_boxes)
        if eq_bbox is None:
            no_eq_count += 1

        if args.verbose:
            print(f'\n[{idx}] {fname}  iw={iw} ih={ih}  '
                  f'GT_BOFF={len(real_boff)} GT_EQ={len(eq)}  '
                  f'eq_bbox={eq_bbox}')

        dets, icon_est = dense_scan(matcher, img, boff_names, eq_bbox=eq_bbox)
        boff_dets = [d for d in dets if d[6].startswith('__boff_')]

        if args.verbose:
            other_dets = [d for d in dets if not d[6].startswith('__boff_')]
            print(f'    dense_scan: {len(dets)} total dets '
                  f'({len(boff_dets)} BOFF-typed, {len(other_dets)} other)  '
                  f'icon_est={icon_est}')

        window, score, dbg, n_eval = sweep_for_boff(
            boff_dets, img.shape, eq_bbox, icon_est, verbose=args.verbose)

        # Coverage: how many GT BOFF bboxes are covered by predicted BOFF dets
        # (only those inside the chosen window).
        pred_inside = []
        if window:
            wx, wy, ww, wh = window
            pred_inside = [(d[0], d[1], d[2], d[3]) for d in boff_dets
                           if d[0] >= wx and d[1] >= wy
                           and d[0] + d[2] <= wx + ww
                           and d[1] + d[3] <= wy + wh]

        covered = 0
        for gt in real_boff:
            best_iou = max((iou(pb, gt['bbox']) for pb in pred_inside), default=0.0)
            if best_iou >= IOU_THRESHOLD:
                covered += 1

        total_gt += len(real_boff)
        total_covered += covered

        n_pred = len(pred_inside)
        rows.append({
            'file': fname,
            'iw': iw, 'ih': ih,
            'gt_n': len(real_boff),
            'eq_n': len(eq),
            'window': window,
            'score': round(score, 2),
            'n_pred': n_pred,
            'n_dets_total': len(dets),
            'n_dets_boff': len(boff_dets),
            'covered': covered,
            'cov_pct': round(covered / max(len(real_boff), 1) * 100, 1),
        })
        per_screen_dump.append({
            'file': fname,
            'window': list(window) if window else None,
            'score': round(score, 2),
            'icon_est': icon_est,
            'boff_dets': [
                {'bbox': [d[0], d[1], d[2], d[3]],
                 'name': d[4], 'conf': round(d[5], 3), 'type': d[6]}
                for d in boff_dets
            ],
            'all_dets_n': len(dets),
        })

        win_str = (f'win=({window[0]:4},{window[1]:4},{window[2]:3},{window[3]:3})'
                   if window else 'win=NONE                          ')
        print(f'[{idx:3d}] {fname[:42]:<42s}  '
              f'GT={len(real_boff):2d}  '
              f'EQ={len(eq):2d}  '
              f'{win_str}  '
              f'dets={len(boff_dets):2d}/{len(dets):3d}  '
              f'in_win={n_pred:2d}  '
              f'cov={covered:2d}/{len(real_boff):2d} ({covered/max(len(real_boff),1)*100:5.1f}%)')

    print('\n' + '=' * 80)
    print(f'Screens tested:           {len(rows)}')
    print(f'Screens without GT EQ:    {no_eq_count}')
    print(f'GT BOFF total:            {total_gt}')
    print(f'Covered (IoU≥0.30):       {total_covered} = '
          f'{total_covered/max(total_gt,1)*100:.1f}%')
    print(f'Baseline (current code):  327 = 74.0%')

    rows.sort(key=lambda r: r['cov_pct'])
    print('\nPer-screen coverage (this run):')
    for r in rows:
        bar = '█' * int(r['cov_pct'] / 5)
        print(f'  {r["file"][:42]:<42s}  '
              f'{r["covered"]:2d}/{r["gt_n"]:2d} ({r["cov_pct"]:5.1f}%) {bar}')

    (OUT / 'scan.json').write_text(json.dumps({
        'totals': {
            'screens_tested': len(rows),
            'gt_total': total_gt,
            'covered': total_covered,
            'no_eq_anchor': no_eq_count,
        },
        'per_screen': rows,
        'detections': per_screen_dump,
    }, indent=2), encoding='utf-8')
    print(f'\nFull dump: {OUT / "scan.json"}')


if __name__ == '__main__':
    main()
