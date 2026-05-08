"""Trait grid detector — prototype v1 (read-only diagnostic).

End-to-end pipeline based on phase 1-4 findings:

  1. Find icon-sized bright connected components (size prior from
     phase 1: icon_h ≈ 0.05–0.08 × H).
  2. Cluster CCs into horizontal rows (CCs within 0.5× icon_h of same y).
  3. Keep only rows with ≥3 CCs at consistent x-spacing
     (col_dx ≈ 0.13–0.18 × W from phase 1).
  4. Cluster rows into row-groups by Y gap (gap < 0.4× icon_h ⇒ same
     group; phase 2 found within-section row gap = 0.06×, inter-section
     gap = 0.9×).
  5. Lock 5-col grid from densest row-group (phase 3: 93.6% within ±5px
     across sections).
  6. Snap each row to the 5-col grid; emit bboxes.
  7. Label row-groups top-down with overflow rules from phase 4:
       - row-group of 1 icon at col 1   → Personal overflow
       - row-group of 2 icons at cols 1-2 → Starship overflow
       - else → next section in canonical order

Validation:
  - per GT bbox, find best-IoU match in detected bboxes
  - report IoU≥0.30 / IoU≥0.50 hit rates per slot
  - report section-label accuracy (right slot for the row?)

Output → tests/_diag_out/trait_detector_v1/.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'trait_detector_v1'
OUT_VIZ = OUT_DIR / 'viz'
OUT_JSON = OUT_DIR / 'summary.json'

TRAIT_SLOTS = {
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
}
SPACE_ORDER = ['Personal Space Traits', 'Starship Traits',
               'Space Reputation', 'Active Space Rep']
GROUND_ORDER = ['Personal Ground Traits', 'Ground Reputation',
                'Active Ground Rep']
MIXED_ORDER = SPACE_ORDER + GROUND_ORDER
SLOT_CAPS = {
    'Personal Space Traits': 11, 'Starship Traits': 7,
    'Space Reputation': 5, 'Active Space Rep': 5,
    'Personal Ground Traits': 11,
    'Ground Reputation': 5, 'Active Ground Rep': 5,
}

# Phase 1 priors. icon_h/H ranges from 0.031 (full-screen) to 0.5 (cropped
# panel). Use a wide CC-detect range here; precision comes from the row+grid
# stages, not the size prior.
ICON_H_FRAC_LO = 0.025
ICON_H_FRAC_HI = 0.300
# col_dx is proportional to icon_w, NOT screen_w (cropped panels skew screen
# ratios). STO trait icons sit ~touching each other: col_dx ≈ 1.05-1.4 × icon_w.
COL_DX_VS_ICON_W_LO = 1.00
COL_DX_VS_ICON_W_HI = 1.50
ICON_AR_LO     = 0.55  # w/h
ICON_AR_HI     = 0.95


# ── ML matcher integration ────────────────────────────────────────────────
class _MockApp:
    """Minimal sets_app stub for IconMatcher. Provides config dict pointing
    at the real SETS images directory."""
    def __init__(self, images_dir: Path):
        self.config = {
            'config_subfolders': {'images': str(images_dir)},
            'config_folder': str(images_dir.parent),
        }


def build_name_to_section(traits_cache: dict, ship_traits: dict) -> dict:
    """Build {icon_name: section_label} from cached cargo data."""
    m: dict[str, str] = {}
    sec_for = {
        ('space', 'personal'):    'Personal Space Traits',
        ('ground', 'personal'):   'Personal Ground Traits',
        ('space', 'rep'):         'Space Reputation',
        ('ground', 'rep'):        'Ground Reputation',
        ('space', 'active_rep'):  'Active Space Rep',
        ('ground', 'active_rep'): 'Active Ground Rep',
    }
    for env in ('space', 'ground'):
        env_d = traits_cache.get(env, {})
        for kind in ('personal', 'rep', 'active_rep'):
            sec = sec_for[(env, kind)]
            for name in env_d.get(kind, {}):
                m[name] = sec
    for name in ship_traits:
        m[name] = 'Starship Traits'
    return m


def init_matcher() -> tuple[object, dict]:
    """Load IconMatcher with mock cache; return (matcher, name_to_section)."""
    images_dir = ROOT / '.config' / 'images'
    cache_dir = ROOT / '.config' / 'cache'
    traits = json.load(open(cache_dir / 'traits.json'))
    ship_traits = json.load(open(cache_dir / 'starship_traits.json'))
    name_to_section = build_name_to_section(traits, ship_traits)
    from warp.recognition.icon_matcher import SETSIconMatcher
    matcher = SETSIconMatcher(_MockApp(images_dir))
    return matcher, name_to_section


def classify_group_section(group, img, matcher, name_to_section,
                           grid_cols, icon_w, icon_h, max_probes=4):
    """Probe up to `max_probes` icons in a group via classify_patch.
    Returns (section_name | None, confidence_sum, votes_dict).

    Strategy: pick icons spread across columns/rows to avoid bias.
    For each probe, look up the predicted name's section. Sum confidence
    per section, pick winner.
    """
    H, W = img.shape[:2]
    # Flatten group: list of (col_idx, bbox)
    icons = [(ci, b) for row in group for ci, b in row]
    if not icons:
        return None, 0.0, {}
    # Pick spread sample: even-spaced indices
    step = max(1, len(icons) // max_probes)
    sample = icons[::step][:max_probes]
    votes: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for ci, b in sample:
        x, y, w, h = b
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = img[y0:y1, x0:x1]
        try:
            name, conf = matcher.classify_patch(patch)
        except Exception:
            name, conf = '', 0.0
        if not name or conf <= 0:
            continue
        sec = name_to_section.get(name)
        if sec is None:
            continue
        votes[sec] += conf
        counts[sec] += 1
    if not votes:
        return None, 0.0, dict(votes)
    best_sec = max(votes, key=votes.get)
    return best_sec, votes[best_sec], dict(votes)


def merge_starship_overflow(groups, icon_h, gap_lo=0.5, gap_hi=1.5):
    """Detect [5] followed by single-row [≤2 cols⊆{0,1}] pairs with vertical
    gap ≈ 0.6-1.3×icon_h (the ship-name divider) and merge them.

    Returns new list of groups; each group still = list[row], where
    row = list[(col_idx, bbox)].
    """
    if len(groups) < 2:
        return list(groups)
    merged = []
    skip_next = False
    for i, g in enumerate(groups):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(groups):
            ng = groups[i + 1]
            # Current must be a single row of 5; next must be single row of ≤2 cols⊆{0,1}
            if (len(g) == 1 and len(g[0]) == 5
                    and len(ng) == 1):
                ncols = sorted(ci for ci, _ in ng[0])
                if len(ncols) <= 2 and all(c in (0, 1) for c in ncols):
                    # Check vertical gap signature
                    g_bot = max(b[1] + b[3] for _, b in g[0])
                    ng_top = min(b[1] for _, b in ng[0])
                    gap = ng_top - g_bot
                    if gap_lo * icon_h <= gap <= gap_hi * icon_h:
                        merged.append([g[0], ng[0]])
                        skip_next = True
                        continue
        merged.append(g)
    return merged


def find_screen_files(fnames):
    candidates = [
        Path('/home/raman/STO_screens'),
        ROOT / 'warp' / 'training_data' / 'migrated_images',
        ROOT / 'warp' / 'training_data' / 'screen_types',
    ]
    out = {}
    targets = set(fnames)
    for d in candidates:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if p.is_file() and p.name in targets and p.name not in out:
                out[p.name] = p
    return out


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── Step 1: candidate icon CCs ──
def detect_icon_ccs(img):
    """Find icon-sized bright connected components."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_lo = int(H * ICON_H_FRAC_LO)
    # Tiny cropped panels: a single icon row can fill 50-65% of image height,
    # blowing past the standard 30% cap. Relax for small images.
    h_hi_frac = 0.65 if H < 250 else ICON_H_FRAC_HI
    h_hi = int(H * h_hi_frac)
    ccs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < h_lo or h > h_hi:
            continue
        if w < int(h * ICON_AR_LO) or w > int(h * ICON_AR_HI) + 4:
            continue
        ccs.append((x, y, w, h))
    return ccs


# ── Step 2-3: build candidate trait rows ──
def find_trait_rows(ccs, W):
    """Cluster CCs into horizontal rows; keep rows with ≥3 CCs at consistent
    x-spacing in the icon-relative col_dx prior. Returns list of rows; each
    row is sorted bboxes left→right.

    Critical: only icons that are TIGHTLY GROUPED (tight cy AND tight icon
    height, since EQ/trait/BOFF icons mix on full screens) form a row.
    """
    if not ccs:
        return []
    # Sort by y to cluster rows. Use a tight per-row tolerance (icon_h
    # of THIS group), not global median, because mixed icon sizes from
    # EQ + traits + BOFFs would otherwise merge into giant rows.
    ccs_by_y = sorted(ccs, key=lambda b: b[1] + b[3] / 2)
    rows = []
    cur = [ccs_by_y[0]]
    for b in ccs_by_y[1:]:
        cy = b[1] + b[3] / 2
        cur_cys = [c[1] + c[3] / 2 for c in cur]
        cur_cy_med = st.median(cur_cys)
        cur_h_med = st.median(c[3] for c in cur)
        # Per-row tol scales with the current row's median icon height
        if abs(cy - cur_cy_med) <= max(6, cur_h_med * 0.35):
            cur.append(b)
        else:
            rows.append(cur)
            cur = [b]
    rows.append(cur)
    # For each row, find ALL non-overlapping sub-sequences of CCs with
    # consistent x-spacing (dx ≈ icon_w × [LO, HI]). Two trait panels at
    # the same Y but different x-bands (e.g. PGT left, GR right on
    # image26.png) must each emit their own chain. Sub-row CCs must also
    # have similar widths.
    def best_chain_from(row, cxs, ws, used, start):
        chain = [start]
        for j in range(start + 1, len(row)):
            if j in used:
                continue
            dx_last = cxs[j] - cxs[chain[-1]]
            chain_w_med = st.median(ws[k] for k in chain)
            dx_lo = chain_w_med * COL_DX_VS_ICON_W_LO
            dx_hi = chain_w_med * COL_DX_VS_ICON_W_HI
            if dx_last > dx_hi:
                break
            if dx_last < dx_lo:
                continue
            if abs(ws[j] - chain_w_med) > chain_w_med * 0.25:
                continue
            if len(chain) >= 2:
                dxs = [cxs[chain[k + 1]] - cxs[chain[k]]
                       for k in range(len(chain) - 1)]
                med_dx = st.median(dxs)
                if abs(dx_last - med_dx) > med_dx * 0.20:
                    continue
            chain.append(j)
        return chain

    out = []
    for row in rows:
        row = sorted(row, key=lambda b: b[0] + b[2] / 2)
        if len(row) < 3:
            continue
        cxs = [b[0] + b[2] / 2 for b in row]
        ws = [b[2] for b in row]
        used = set()
        # Repeatedly extract the longest chain from unused CCs
        for _ in range(8):  # at most 8 panels per row (safety bound)
            best_chain = []
            for i in range(len(row)):
                if i in used:
                    continue
                chain = best_chain_from(row, cxs, ws, used, i)
                if len(chain) > len(best_chain):
                    best_chain = chain
            if len(best_chain) < 3:
                break
            out.append([row[k] for k in best_chain])
            used.update(best_chain)
    return out


# ── Step 4: cluster rows into row-groups ──
def cluster_row_groups(rows):
    """Cluster trait rows into vertically-grouped row-groups.

    Returns list of row-groups; each group = list[row], sorted top→bottom.
    """
    if not rows:
        return []
    # Use median icon_h across all kept rows
    icon_h = st.median(b[3] for r in rows for b in r)
    # Sort rows by their cy
    rows_sorted = sorted(rows, key=lambda r: r[0][1] + r[0][3] / 2)
    groups = [[rows_sorted[0]]]
    for r in rows_sorted[1:]:
        prev = groups[-1][-1]
        prev_cy = prev[0][1] + prev[0][3] / 2
        prev_bot = max(b[1] + b[3] for b in prev)
        cur_top = min(b[1] for b in r)
        gap = cur_top - prev_bot
        if gap < icon_h * 0.4:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


# ── Step 5: lock 5-col grid from densest row-group ──
def lock_grids_multi(row_groups, max_panels=4):
    """Find ALL distinct trait grids (panels). STO users can arrange trait
    sections anywhere — left/right, top/bottom, multiple panels — so we
    cluster rows by grid signature and emit one panel per cluster.

    Signature = (col_dx, first_col_x) bucketed to ±15% / ±0.5×icon_w.

    Returns list of dicts; each panel has: cols, col_dx, icon_w, icon_h,
    cluster_rows (rows that voted for this panel), y_top, y_bot.
    Sorted by panel strength (n_with_5, n_rows, max_row_size).
    """
    all_rows = [r for g in row_groups for r in g]
    if not all_rows:
        return []
    sigs = []
    for r in all_rows:
        cxs = sorted(b[0] + b[2] / 2 for b in r)
        if len(cxs) < 2:
            continue
        dx = st.median(cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1))
        x0 = cxs[0]
        iw = st.median(b[2] for b in r)
        ih = st.median(b[3] for b in r)
        sigs.append({'dx': dx, 'x0': x0, 'iw': iw, 'ih': ih,
                     'row': r, 'n': len(cxs)})
    if not sigs:
        return []

    def matches(a, b):
        if abs(a['dx'] - b['dx']) > min(a['dx'], b['dx']) * 0.15:
            return False
        return abs(a['x0'] - b['x0']) <= max(a['iw'], b['iw']) * 0.5

    clusters = []
    for s in sigs:
        placed = False
        for c in clusters:
            if matches(c[0], s):
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])

    def cluster_score(c):
        sizes = [s['n'] for s in c]
        n_with_5 = sum(1 for n in sizes if n == 5)
        return (n_with_5, len(c), max(sizes))

    clusters.sort(key=cluster_score, reverse=True)
    panels = []
    for c in clusters[:max_panels]:
        if not any(s['n'] >= 4 for s in c):
            continue
        fives = [s for s in c if s['n'] == 5]
        pool = fives if fives else [s for s in c if s['n'] >= 4]
        if not pool:
            continue
        iw_med = st.median(s['iw'] for s in pool)
        pool.sort(key=lambda s: abs(s['iw'] - iw_med))
        ref = pool[0]
        cols = sorted(b[0] + b[2] / 2 for b in ref['row'])
        col_dx = st.median(cols[i + 1] - cols[i] for i in range(len(cols) - 1))
        while len(cols) < 5:
            cols.append(cols[-1] + col_dx)
        # y-extent across the entire cluster
        y_top = min(b[1] for s in c for b in s['row'])
        y_bot = max(b[1] + b[3] for s in c for b in s['row'])
        panels.append({
            'cols': cols[:5],
            'col_dx': float(col_dx),
            'icon_w': float(ref['iw']),
            'icon_h': float(ref['ih']),
            'cluster_rows': [s['row'] for s in c],
            'y_top': int(y_top),
            'y_bot': int(y_bot),
        })
    return panels


def lock_grid(row_groups):
    """Backward-compat wrapper: returns (cols, col_dx, icon_w, icon_h)."""
    panels = lock_grids_multi(row_groups, max_panels=1)
    if not panels:
        return None
    p = panels[0]
    return p['cols'], p['col_dx'], p['icon_w'], p['icon_h']


# ── Step 6-7: snap rows to grid, label sections top-down ──
def assign_sections(row_groups, grid_cols, col_dx, icon_w, icon_h,
                    is_ground=False):
    """For each row in each row-group, snap to grid and label section.

    Returns dict: slot_name → list[bbox_xywh].
    """
    order = GROUND_ORDER if is_ground else SPACE_ORDER

    # Flatten: list of (row_top_y, row_bottom_y, occupied_cols: set[int]).
    flat_rows = []
    for g in row_groups:
        for r in g:
            cxs = [b[0] + b[2] / 2 for b in r]
            occ = set()
            tol = col_dx * 0.5
            for ci, gc in enumerate(grid_cols):
                if any(abs(cx - gc) <= tol for cx in cxs):
                    occ.add(ci)
            y_top = int(min(b[1] for b in r))
            y_bot = int(max(b[1] + b[3] for b in r))
            flat_rows.append({
                'y_top': y_top, 'y_bot': y_bot, 'occ': occ,
                'cy': (y_top + y_bot) / 2,
            })
    flat_rows.sort(key=lambda r: r['cy'])

    # Walk rows top-down, assigning to sections.
    # State machine: current_section_idx, used_rows_in_current_section.
    # Overflow rules from phase 4:
    #   - row with N=1 at col 0 → continue current section (Personal overflow)
    #   - row with N=2 at cols {0,1} → continue current section (Starship overflow)
    #   - else: advance to next section
    result = defaultdict(list)
    sec_idx = 0
    if not flat_rows:
        return result

    # Helper to build a bbox at grid (col_idx, row)
    def bbox_at(row, ci):
        cx = grid_cols[ci]
        cy = (row['y_top'] + row['y_bot']) / 2
        x = int(cx - icon_w / 2)
        y = int(cy - icon_h / 2)
        return (x, y, int(round(icon_w)), int(round(icon_h)))

    def emit(row, slot):
        for ci in sorted(row['occ']):
            result[slot].append(bbox_at(row, ci))

    # First row → first section
    cur_slot = order[0] if order else None
    if cur_slot is None:
        return result
    emit(flat_rows[0], cur_slot)
    rows_in_cur = 1

    for row in flat_rows[1:]:
        n = len(row['occ'])
        is_personal_overflow = (n == 1 and row['occ'] == {0})
        is_starship_overflow = (n == 2 and row['occ'] == {0, 1})
        if is_personal_overflow and cur_slot in (
                'Personal Space Traits', 'Personal Ground Traits'):
            emit(row, cur_slot)
            rows_in_cur += 1
            continue
        if is_starship_overflow and cur_slot == 'Starship Traits':
            emit(row, cur_slot)
            rows_in_cur += 1
            continue
        # Otherwise: advance to next section in canonical order
        sec_idx += 1
        if sec_idx >= len(order):
            break
        cur_slot = order[sec_idx]
        emit(row, cur_slot)
        rows_in_cur = 1
    return result


# ── Heuristic: is this image space, ground, or mixed? ──
def guess_mode(boxes_by_slot):
    """Use GT to decide space/ground/mixed (validation only)."""
    has_ground = any('Ground' in s for s in boxes_by_slot)
    has_space = any('Space' in s or 'Starship' in s for s in boxes_by_slot)
    if has_ground and has_space:
        return 'mixed'
    if has_ground:
        return 'ground'
    return 'space'


def resweep_rows_in_band(ccs, grid_cols, col_dx, icon_w, icon_h,
                         y_min, y_max):
    """Pass 2: within the trait y-band, snap every CC of icon-size to the
    nearest grid col. Re-cluster into rows. Returns list of rows (each
    row = list[(col_idx, bbox)] sorted by col_idx)."""
    tol_x = col_dx * 0.5
    candidates = []
    for b in ccs:
        x, y, w, h = b
        cy = y + h / 2
        if cy < y_min or cy > y_max:
            continue
        # Width must roughly match grid icon_w
        if w < icon_w * 0.65 or w > icon_w * 1.35:
            continue
        if h < icon_h * 0.65 or h > icon_h * 1.35:
            continue
        cx = x + w / 2
        # snap to nearest grid col
        ci = min(range(len(grid_cols)), key=lambda i: abs(grid_cols[i] - cx))
        if abs(grid_cols[ci] - cx) > tol_x:
            continue
        candidates.append((cy, ci, b))
    if not candidates:
        return []
    # Cluster by cy with tol = 0.4 * icon_h
    candidates.sort(key=lambda t: t[0])
    rows = [[candidates[0]]]
    for c in candidates[1:]:
        prev_cy = st.median(x[0] for x in rows[-1])
        if abs(c[0] - prev_cy) <= max(6, icon_h * 0.4):
            rows[-1].append(c)
        else:
            rows.append([c])
    # Per row: dedupe by col_idx (keep CC closest to grid col)
    result = []
    for row in rows:
        by_col = {}
        for cy, ci, b in row:
            cx = b[0] + b[2] / 2
            err = abs(grid_cols[ci] - cx)
            if ci not in by_col or err < by_col[ci][2]:
                by_col[ci] = (cy, b, err)
        sorted_cols = sorted(by_col.items())
        result.append([(ci, t[1]) for ci, t in sorted_cols])
    return result


def cluster_resweep_groups(rows, icon_h):
    """Cluster resweep rows into row-groups by Y-gap (< 0.4× icon_h)."""
    if not rows:
        return []
    # rows: list of list[(col_idx, bbox)]; sort by row cy
    def row_cy(r):
        return st.median(b[1] + b[3] / 2 for _, b in r)
    rows_sorted = sorted(rows, key=row_cy)
    groups = [[rows_sorted[0]]]
    for r in rows_sorted[1:]:
        prev = groups[-1][-1]
        prev_bot = max(b[1] + b[3] for _, b in prev)
        cur_top = min(b[1] for _, b in r)
        if cur_top - prev_bot < icon_h * 0.4:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def label_groups(groups, order, slot_caps):
    """Top-down section labelling using row-group structure.

    Phase 4 finding: Starship Traits' 2-row case has a wide inter-row
    divider (the ship-name strip) at ~0.9× icon_h. cluster_resweep_groups
    splits this into TWO row-groups. The detector must merge them back
    by recognising the second group as an overflow.

    Overflow patterns (phase 4):
      - Personal Traits overflow row: [1@col 0]   (100% of n=16 cases)
      - Starship Traits overflow row: [2@cols 0-1] (94% of n=36 cases)
    """
    labels = []
    sec_idx = 0
    prev_label = None
    for gi, g in enumerate(groups):
        # Single-row continuation check
        if prev_label is not None and len(g) == 1:
            row = g[0]
            cols_used = sorted(ci for ci, _ in row)
            n = len(cols_used)
            if (n == 1 and cols_used == [0]
                    and prev_label in ('Personal Space Traits',
                                       'Personal Ground Traits')):
                labels.append(prev_label)
                continue
            # Starship overflow: 2 icons at cols 0-1 (94% of cases) OR
            # 1 icon at col 0 (partial fill, 6%).
            if (prev_label == 'Starship Traits'
                    and ((n == 2 and cols_used == [0, 1])
                         or (n == 1 and cols_used == [0]))):
                labels.append(prev_label)
                continue
        if sec_idx >= len(order):
            labels.append(None)
            continue
        slot = order[sec_idx]
        labels.append(slot)
        prev_label = slot
        sec_idx += 1
    return labels


def emit_bboxes(group, slot, grid_cols, icon_w, icon_h):
    """Convert a row-group into bboxes."""
    out = []
    for row in group:
        # Use median row cy
        cy = st.median(b[1] + b[3] / 2 for _, b in row)
        for ci, b in row:
            cx = grid_cols[ci]
            x = int(round(cx - icon_w / 2))
            y = int(round(cy - icon_h / 2))
            out.append((x, y, int(round(icon_w)), int(round(icon_h))))
    return out


# ── Master detect function ──
def detect_traits(img, mode='space', matcher=None, name_to_section=None):
    """End-to-end. Returns {slot_name: [bbox,...]}, plus debug info.

    mode: kept for back-compat, but ML probe drives section labeling.
    matcher / name_to_section: when provided, each group is classified by
    probing icons through icon_matcher.classify_patch and mapping names to
    sections via name_to_section. When None, falls back to canonical order
    (legacy behavior).
    """
    H, W = img.shape[:2]
    ccs = detect_icon_ccs(img)
    rows = find_trait_rows(ccs, W)
    groups = cluster_row_groups(rows)
    panels = lock_grids_multi(groups, max_panels=4)
    if not panels:
        return {}, {'reason': 'no_grid', 'n_ccs': len(ccs),
                    'n_rows_pass1': len(rows), '_all_ccs': ccs,
                    '_merged_groups': []}

    sections = defaultdict(list)
    all_merged = []
    all_labels = []
    all_votes = []
    debug_panels = []

    for pi, p in enumerate(panels):
        cols = p['cols']
        col_dx = p['col_dx']
        icon_w = p['icon_w']
        icon_h = p['icon_h']
        # y-band: panel's own y-extent + slack below for overflow rows
        y_min = p['y_top'] - icon_h * 0.5
        y_max = p['y_bot'] + icon_h * 2.5
        rs_rows = resweep_rows_in_band(ccs, cols, col_dx, icon_w, icon_h,
                                       y_min, y_max)
        rs_groups = cluster_resweep_groups(rs_rows, icon_h)
        merged = merge_starship_overflow(rs_groups, icon_h)

        panel_labels = []
        panel_votes = []

        if matcher is not None and name_to_section is not None:
            for g in merged:
                is_starship_struct = (
                    len(g) == 2 and len(g[0]) == 5
                    and 1 <= len(g[1]) <= 2
                    and all(ci in (0, 1) for ci, _ in g[1])
                )
                if is_starship_struct:
                    slot = 'Starship Traits'
                    votes = {'Starship Traits': float('inf')}
                else:
                    slot, _, votes = classify_group_section(
                        g, img, matcher, name_to_section,
                        cols, icon_w, icon_h)
                panel_labels.append(slot)
                panel_votes.append({k: round(v, 3) for k, v in votes.items()})
                if slot is None:
                    continue
                sections[slot].extend(emit_bboxes(g, slot, cols, icon_w, icon_h))
        else:
            if mode == 'ground':
                order = GROUND_ORDER
            elif mode == 'mixed':
                order = MIXED_ORDER
            else:
                order = SPACE_ORDER
            panel_labels = label_groups(merged, order, SLOT_CAPS)
            for g, slot in zip(merged, panel_labels):
                if slot is None:
                    continue
                sections[slot].extend(emit_bboxes(g, slot, cols, icon_w, icon_h))

        all_merged.extend(merged)
        all_labels.extend(panel_labels)
        all_votes.extend(panel_votes)
        debug_panels.append({
            'cols': [round(c, 1) for c in cols],
            'col_dx': round(col_dx, 1),
            'icon_w': round(icon_w, 1),
            'icon_h': round(icon_h, 1),
            'y_band': [int(y_min), int(y_max)],
            'n_groups': len(merged),
            'group_row_counts': [[len(r) for r in g] for g in merged],
            'group_labels': panel_labels,
        })

    p0 = panels[0]
    return dict(sections), {
        'n_ccs': len(ccs), 'n_rows_pass1': len(rows),
        'n_groups_pass1': len(groups),
        'n_panels': len(panels),
        'panels': debug_panels,
        'grid_cols': [round(c, 1) for c in p0['cols']],
        'col_dx': round(p0['col_dx'], 1),
        'icon_w': round(p0['icon_w'], 1),
        'icon_h': round(p0['icon_h'], 1),
        'group_labels': all_labels,
        'group_votes': all_votes,
        '_all_ccs': ccs,
        '_merged_groups': all_merged,
    }


# ── Validation ──
def validate(detected, gt):
    """For each GT bbox, find best-IoU match in detected. Returns per-slot
    hit counts at IoU≥0.30 and ≥0.50, and section-label accuracy.

    section-label accuracy: a GT bbox is "correctly labelled" if the
    detected bbox with best IoU is in the SAME slot.
    """
    slot_stats = defaultdict(lambda: {
        'n_gt': 0, 'iou30': 0, 'iou50': 0, 'correct_slot': 0,
    })
    # Flatten detected with slot labels
    det_flat = []
    for slot, boxes in detected.items():
        for b in boxes:
            det_flat.append((slot, b))
    for gt_slot, gt_boxes in gt.items():
        for gb in gt_boxes:
            slot_stats[gt_slot]['n_gt'] += 1
            # Best IoU match across ALL detected
            best_iou, best_slot = 0.0, None
            for ds, db in det_flat:
                iou = bbox_iou(gb, db)
                if iou > best_iou:
                    best_iou, best_slot = iou, ds
            if best_iou >= 0.30:
                slot_stats[gt_slot]['iou30'] += 1
            if best_iou >= 0.50:
                slot_stats[gt_slot]['iou50'] += 1
            if best_iou >= 0.30 and best_slot == gt_slot:
                slot_stats[gt_slot]['correct_slot'] += 1
    return dict(slot_stats)


def viz(img, detected, gt, debug, fname, out_path, all_ccs=None,
        rs_groups=None, rs_labels=None):
    """Diagnostic visualization.

    Layers (bottom→top):
      - all CCs found: cyan thin (raw candidates before row-filter)
      - GT: green thin
      - detected per-section: thick colored
      - grid columns: magenta vertical lines
      - per-group label + n_rows × n_cols printed near group y-center
    """
    canvas = img.copy()
    # All CCs in cyan (1px) — what survived the size+AR filter
    if all_ccs:
        for x, y, w, h in all_ccs:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 80), 1)
    # GT in green
    for slot, boxes in gt.items():
        for b in boxes:
            x, y, w, h = b
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # Detected in colors per section
    palette = {
        'Personal Space Traits':  (255, 200, 0),
        'Starship Traits':        (0, 200, 255),
        'Space Reputation':       (180, 255, 100),
        'Active Space Rep':       (200, 100, 255),
        'Personal Ground Traits': (255, 100, 0),
        'Ground Reputation':      (100, 255, 200),
        'Active Ground Rep':      (255, 0, 200),
    }
    for slot, boxes in detected.items():
        col = palette.get(slot, (200, 200, 200))
        for b in boxes:
            x, y, w, h = b
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
    # Grid cols — draw per panel, in band-only Y range
    panels_dbg = debug.get('panels', [])
    if panels_dbg:
        for pi, pd in enumerate(panels_dbg):
            yb = pd.get('y_band', [0, canvas.shape[0]])
            for cx in pd.get('cols', []):
                cv2.line(canvas, (int(cx), int(yb[0])),
                         (int(cx), int(yb[1])), (255, 0, 255), 1)
    elif 'grid_cols' in debug and debug['grid_cols']:
        for cx in debug['grid_cols']:
            cv2.line(canvas, (int(cx), 0), (int(cx), canvas.shape[0]),
                     (255, 0, 255), 1)
    # Per-group annotation: label + size + votes
    if rs_groups is not None and rs_labels is not None:
        votes_list = debug.get('group_votes', [])
        for gi, g in enumerate(rs_groups):
            label = rs_labels[gi] if gi < len(rs_labels) else None
            counts = [len(r) for r in g]
            xs = [b[0] for row in g for _, b in row] if g and g[0] and isinstance(g[0][0], tuple) else []
            ys = [b[1] for row in g for _, b in row] if g and g[0] and isinstance(g[0][0], tuple) else []
            if not xs:
                continue
            x0, y0 = min(xs), min(ys)
            tag = f'g{gi}: {label or "NONE"}  rows={counts}'
            if gi < len(votes_list) and votes_list[gi]:
                vs = votes_list[gi]
                top = sorted(vs.items(), key=lambda kv: -kv[1])[:2]
                tag += '  votes=' + ','.join(f'{k[:4]}{v:.1f}' for k, v in top)
            cv2.putText(canvas, tag, (max(2, x0 - 5), max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # Header info
    info = (f'{fname}  ccs={debug.get("n_ccs", "?")} '
            f'p1rows={debug.get("n_rows_pass1", "?")} '
            f'p2rows={debug.get("n_rows_pass2", "?")} '
            f'icon_h={debug.get("icon_h", "?")}')
    cv2.putText(canvas, info, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, fname, (4, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), canvas)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
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
    paths = find_screen_files(targets.keys())
    print(f'screens: {len(targets)}  files: {len(paths)}')

    print('Loading IconMatcher + trait cache...')
    try:
        matcher, name_to_section = init_matcher()
        print(f'  name_to_section: {len(name_to_section)} icon names mapped')
    except Exception as e:
        print(f'  matcher init failed: {e}; falling back to canonical order')
        matcher, name_to_section = None, None

    summary = {'screens': {}}
    overall = defaultdict(lambda: {
        'n_gt': 0, 'iou30': 0, 'iou50': 0, 'correct_slot': 0,
    })

    for fname, gt in targets.items():
        path = paths.get(fname)
        if not path:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        mode = guess_mode(gt)
        detected, debug = detect_traits(img, mode=mode,
                                         matcher=matcher,
                                         name_to_section=name_to_section)
        stats = validate(detected, gt)
        debug_safe = {k: v for k, v in debug.items() if not k.startswith('_')}
        summary['screens'][fname] = {
            'mode': mode,
            'debug': debug_safe,
            'detected_counts': {s: len(v) for s, v in detected.items()},
            'gt_counts': {s: len(v) for s, v in gt.items()},
            'per_slot': stats,
        }
        for slot, s in stats.items():
            for k, v in s.items():
                overall[slot][k] += v
        viz(img, detected, gt, debug, fname, OUT_VIZ / fname,
            all_ccs=debug.get('_all_ccs'),
            rs_groups=debug.get('_merged_groups'),
            rs_labels=debug.get('group_labels'))

    # Aggregate
    total_gt = total_iou30 = total_iou50 = total_correct = 0
    print('\n=== PER-SLOT RESULTS ===')
    print(f'  {"slot":28s} {"n_gt":>5s} {"IoU≥30":>9s} {"IoU≥50":>9s} {"slot OK":>9s}')
    for slot in sorted(overall.keys()):
        s = overall[slot]
        n = s['n_gt']
        if n == 0:
            continue
        total_gt += n
        total_iou30 += s['iou30']
        total_iou50 += s['iou50']
        total_correct += s['correct_slot']
        print(f'  {slot:28s} {n:5d} '
              f'{s["iou30"]:5d} ({100 * s["iou30"] / n:4.1f}%) '
              f'{s["iou50"]:5d} ({100 * s["iou50"] / n:4.1f}%) '
              f'{s["correct_slot"]:5d} ({100 * s["correct_slot"] / n:4.1f}%)')
    print(f'\n  {"TOTAL":28s} {total_gt:5d} '
          f'{total_iou30:5d} ({100 * total_iou30 / max(1, total_gt):.1f}%) '
          f'{total_iou50:5d} ({100 * total_iou50 / max(1, total_gt):.1f}%) '
          f'{total_correct:5d} ({100 * total_correct / max(1, total_gt):.1f}%)')

    summary['totals'] = {
        'n_gt': total_gt, 'iou30': total_iou30, 'iou50': total_iou50,
        'correct_slot': total_correct,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
