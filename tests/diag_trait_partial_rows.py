"""Trait partial-row position — phase 4 diagnostic.

Question: when a section's row is incomplete (< 5 icons), which columns
of the 5-col grid are occupied?

This determines whether we can discriminate, by position alone, between:
  - "row 2 of Starship Traits"  (partial continuation, expected fill pattern)
  - "first row of next section" (e.g. Reputation with empty/locked slots)

Per screen with annotated traits:
  1. Lock the 5-col grid using the densest row in the screen.
  2. For every row of every section, classify each of the 5 grid slots as
     OCCUPIED (within ±0.5*col_dx of a GT bbox center) or EMPTY.
  3. Aggregate per (slot, position-in-section) to see fill patterns.

Read-only. Output → tests/_diag_out/trait_partial_rows/.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'trait_partial_rows'
OUT_JSON = OUT_DIR / 'summary.json'

TRAIT_SLOTS = {
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
}


def cluster_1d(values, tol):
    if not values:
        return []
    vs = sorted(values)
    cl = [[vs[0]]]
    for v in vs[1:]:
        if v - cl[-1][-1] <= tol:
            cl[-1].append(v)
        else:
            cl.append([v])
    return cl


def split_rows(boxes, gap_factor=0.4):
    if not boxes:
        return []
    icon_h = st.median(b[3] for b in boxes)
    tol = max(8, icon_h * gap_factor)
    cys = sorted(((b[1] + b[3] / 2), b) for b in boxes)
    rows = [[cys[0]]]
    for cy, b in cys[1:]:
        if cy - rows[-1][-1][0] <= tol:
            rows[-1].append((cy, b))
        else:
            rows.append([(cy, b)])
    return [[b for _, b in row] for row in rows]


def column_centers(boxes):
    if not boxes:
        return []
    icon_w = st.median(b[2] for b in boxes)
    cxs = [b[0] + b[2] / 2 for b in boxes]
    cl = cluster_1d(cxs, tol=max(8, icon_w * 0.4))
    return sorted(st.median(c) for c in cl)


def lock_grid(all_boxes_by_slot):
    """Pick the densest single row across all sections. Use its column
    centers as the canonical 5-col grid."""
    best_row = None
    best_n = 0
    icon_w_all = st.median(
        b[2] for boxes in all_boxes_by_slot.values() for b in boxes
    ) if all_boxes_by_slot else 30
    for slot, boxes in all_boxes_by_slot.items():
        for row in split_rows(list(boxes)):
            if len(row) > best_n:
                best_n = len(row)
                best_row = row
    if best_row is None:
        return [], icon_w_all
    return column_centers(best_row), icon_w_all


def classify_row(row_boxes, grid_cols, col_dx):
    """For each grid col, return True/False = occupied/empty."""
    if not grid_cols:
        return []
    cxs = [b[0] + b[2] / 2 for b in row_boxes]
    occ = []
    tol = col_dx * 0.5 if col_dx else 20
    for gc in grid_cols:
        occ.append(any(abs(cx - gc) <= tol for cx in cxs))
    return occ


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANN_PATH) as f:
        data = json.load(f)
    targets = {}
    for fn, ann in data.items():
        if not isinstance(ann, list):
            continue
        boxes_by_slot = defaultdict(list)
        for a in ann:
            if a.get('state') != 'confirmed':
                continue
            slot = a.get('slot')
            if slot in TRAIT_SLOTS:
                bb = a.get('bbox')
                if bb and len(bb) == 4:
                    boxes_by_slot[slot].append(tuple(bb))
        if boxes_by_slot:
            targets[fn] = boxes_by_slot

    # Aggregators
    # For each (slot, row_idx_within_section), record list of fill patterns
    # like [True,True,True,True,True], or [True,False,True,False,True].
    fill_patterns = defaultdict(list)  # (slot, row_idx) → list[tuple[bool,...]]
    # Also record raw row-icon-count distribution per (slot, row_idx).
    row_count_dist = defaultdict(list)  # (slot, row_idx) → list[int]
    # Single-row partial sections (Reputation/Active Rep with <5 icons)
    single_row_partial_layout = defaultdict(list)  # slot → list[tuple[bool×5]]

    summary = {'screens': {}}

    for fname, boxes_by_slot in targets.items():
        grid, icon_w = lock_grid(boxes_by_slot)
        # Use grid column step
        if len(grid) >= 2:
            col_dx = st.median(grid[i + 1] - grid[i] for i in range(len(grid) - 1))
        else:
            col_dx = icon_w * 1.4
        # Skip screens where grid is incomplete (avoid noise; we want to
        # measure POSITION, not detection)
        if len(grid) < 4:
            continue
        scr = {'grid_cols': [round(g, 1) for g in grid],
               'col_dx': round(col_dx, 1),
               'sections': {}}
        for slot, boxes in boxes_by_slot.items():
            rows = split_rows(list(boxes))
            sec = {'n_rows': len(rows), 'rows': []}
            for ri, row in enumerate(rows):
                occ = classify_row(row, grid, col_dx)
                fill_patterns[(slot, ri)].append(tuple(occ))
                row_count_dist[(slot, ri)].append(sum(occ))
                # Single-row Reputation/Active Rep with partial fill?
                if (slot in ('Space Reputation', 'Ground Reputation',
                             'Active Space Rep', 'Active Ground Rep')
                        and len(rows) == 1 and sum(occ) < len(grid)):
                    single_row_partial_layout[slot].append(tuple(occ))
                sec['rows'].append({
                    'row_idx': ri,
                    'n_icons': len(row),
                    'occupied': occ,
                    'pattern_str': ''.join('1' if o else '0' for o in occ),
                })
            scr['sections'][slot] = sec
        summary['screens'][fname] = scr

    # ── Aggregates ──
    # Pattern frequencies per (slot, row_idx)
    pattern_summary = {}
    for (slot, ri), patterns in fill_patterns.items():
        counter = defaultdict(int)
        for p in patterns:
            counter[''.join('1' if o else '0' for o in p)] += 1
        ranked = sorted(counter.items(), key=lambda kv: -kv[1])
        pattern_summary.setdefault(slot, {})[f'row_{ri}'] = {
            'n_screens': len(patterns),
            'top_patterns': ranked[:8],
            'left_anchored_consecutive_pct': round(100 * sum(
                v for k, v in counter.items()
                if all(c == '1' for c in k[:k.count('1')])
                and all(c == '0' for c in k[k.count('1'):])
            ) / max(1, len(patterns)), 1),
        }

    # Row count distribution
    count_summary = {}
    for (slot, ri), counts in row_count_dist.items():
        count_summary.setdefault(slot, {})[f'row_{ri}'] = {
            'n': len(counts),
            'min': min(counts),
            'max': max(counts),
            'med': st.median(counts),
            'distribution': {c: counts.count(c) for c in sorted(set(counts))},
        }

    # Single-row partial Reputation
    rep_summary = {}
    for slot, patterns in single_row_partial_layout.items():
        counter = defaultdict(int)
        for p in patterns:
            counter[''.join('1' if o else '0' for o in p)] += 1
        ranked = sorted(counter.items(), key=lambda kv: -kv[1])
        n = sum(counter.values())
        rep_summary[slot] = {
            'n': n,
            'top_patterns': ranked[:6],
            'left_anchored_consecutive_pct': round(100 * sum(
                v for k, v in counter.items()
                if all(c == '1' for c in k[:k.count('1')])
                and all(c == '0' for c in k[k.count('1'):])
            ) / max(1, n), 1),
        }

    summary['aggregates'] = {
        'fill_patterns': pattern_summary,
        'row_counts': count_summary,
        'reputation_partial_rows': rep_summary,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # ── Console report ──
    print('=== ROW COUNT DISTRIBUTION (slot → row_idx → n_icons hist) ===')
    for slot in sorted(count_summary):
        print(f'\n  {slot}')
        for row_key, s in sorted(count_summary[slot].items()):
            print(f'    {row_key}: n={s["n"]:3d}  min={s["min"]} max={s["max"]} '
                  f'med={s["med"]}  dist={s["distribution"]}')

    print('\n\n=== FILL PATTERNS PER (slot, row_idx) ===')
    print('  Pattern is "occupied?" for each of 5 grid columns left→right.')
    for slot in sorted(pattern_summary):
        print(f'\n  {slot}')
        for row_key, s in sorted(pattern_summary[slot].items()):
            print(f'    {row_key}: n={s["n_screens"]:3d}  '
                  f'left_anchored_consecutive={s["left_anchored_consecutive_pct"]}%')
            for pat, cnt in s['top_patterns']:
                print(f'      {pat}  ×{cnt}')

    print('\n\n=== REPUTATION (single-row, partial) FILL PATTERNS ===')
    for slot in sorted(rep_summary):
        s = rep_summary[slot]
        print(f'\n  {slot}: n={s["n"]}  '
              f'left_anchored_consecutive={s["left_anchored_consecutive_pct"]}%')
        for pat, cnt in s['top_patterns']:
            print(f'    {pat}  ×{cnt}')


if __name__ == '__main__':
    main()
