# warp/recognition/trait_grid.py
#
# Structure-driven trait-grid detector.
#
# Strategy 0 for SPACE_TRAITS / GROUND_TRAITS / *_MIXED traits:
#   1. Find icon-sized bright connected components.
#   2. Cluster CCs into horizontal rows; extract ALL non-overlapping
#      consistent-spacing chains per row (two side-by-side trait panels
#      at the same Y must each emit their own chain).
#   3. Cluster rows into row-groups by Y-gap.
#   4. Cluster row signatures (col_dx, x0) into panels — STO users may
#      arrange trait sections anywhere on screen.
#   5. Per panel: resweep CCs in y-band, snap to grid, re-cluster groups.
#   6. Pre-merge Starship-Traits overflow ([5] + [≤2 cols 0-1] separated
#      by the ship-name divider gap ≈ 0.6-1.3×icon_h).
#   7. Classify each row-group INDEPENDENTLY by probing icons through
#      icon_matcher.classify_patch and mapping the predicted name to its
#      section via app_cache.traits / app_cache.starship_traits.
#
# CRITICAL RULE — never assume canonical section order. STO users compose
# screenshots arbitrarily (PGT first, then PST; reputation only; reputation
# above traits; etc.). Each row-group is classified on its own merits.
# OCR labels may be absent. Source: feedback_trait_groups_independent.md.

from __future__ import annotations

import statistics as st
from collections import defaultdict

import cv2
import numpy as np

try:
    from src.setsdebug import log as _slog
except Exception:
    import logging
    _slog = logging.getLogger(__name__)


# Section labels
TRAIT_SECTIONS = (
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
)

# Phase-1 priors. icon_h/H ranges 0.025–0.30 on full screens; cropped trait
# panels can have a single row filling 50-65% of image height.
ICON_H_FRAC_LO = 0.025
ICON_H_FRAC_HI = 0.300
# col_dx is proportional to icon_w, NOT screen_w (cropped panels skew screen
# ratios). STO trait icons sit ~touching: col_dx ≈ 1.0–1.5 × icon_w.
COL_DX_VS_ICON_W_LO = 1.00
COL_DX_VS_ICON_W_HI = 1.50
ICON_AR_LO = 0.55  # w/h
ICON_AR_HI = 0.95


# ── Step 1: candidate icon CCs ─────────────────────────────────────────────
def _detect_icon_ccs(img):
    """Find icon-sized bright connected components."""
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
        ccs.append((int(x), int(y), int(w), int(h)))
    return ccs


# ── Step 2-3: build candidate trait rows (multi-chain per Y) ───────────────
def _find_trait_rows(ccs):
    """Cluster CCs into horizontal rows; extract ALL non-overlapping chains
    of ≥3 CCs at consistent x-spacing (dx ≈ icon_w × [LO, HI]).

    Two trait panels at the same Y but different x-bands (e.g. PGT left,
    GR right on image26.png) must each emit their own chain.
    """
    if not ccs:
        return []
    ccs_by_y = sorted(ccs, key=lambda b: b[1] + b[3] / 2)
    rows = []
    cur = [ccs_by_y[0]]
    for b in ccs_by_y[1:]:
        cy = b[1] + b[3] / 2
        cur_cys = [c[1] + c[3] / 2 for c in cur]
        cur_cy_med = st.median(cur_cys)
        cur_h_med = st.median(c[3] for c in cur)
        if abs(cy - cur_cy_med) <= max(6, cur_h_med * 0.35):
            cur.append(b)
        else:
            rows.append(cur)
            cur = [b]
    rows.append(cur)

    def chain_from(row, cxs, ws, used, start):
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
        for _ in range(8):  # bound: at most 8 panels per Y line
            best = []
            for i in range(len(row)):
                if i in used:
                    continue
                ch = chain_from(row, cxs, ws, used, i)
                if len(ch) > len(best):
                    best = ch
            if len(best) < 3:
                break
            out.append([row[k] for k in best])
            used.update(best)
    return out


# ── Step 4: cluster rows into row-groups by Y-gap ──────────────────────────
def _cluster_row_groups(rows):
    if not rows:
        return []
    icon_h = st.median(b[3] for r in rows for b in r)
    rows_sorted = sorted(rows, key=lambda r: r[0][1] + r[0][3] / 2)
    groups = [[rows_sorted[0]]]
    for r in rows_sorted[1:]:
        prev = groups[-1][-1]
        prev_bot = max(b[1] + b[3] for b in prev)
        cur_top = min(b[1] for b in r)
        if cur_top - prev_bot < icon_h * 0.4:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


# ── Step 5: lock per-panel grids ───────────────────────────────────────────
def _lock_grids_multi(row_groups, max_panels=4):
    """Cluster rows by (col_dx, first_col_x) signature; emit one grid per
    panel cluster. Returns list of dicts: {cols, col_dx, icon_w, icon_h,
    cluster_rows, y_top, y_bot}, sorted by panel strength."""
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
        y_top = min(b[1] for s in c for b in s['row'])
        y_bot = max(b[1] + b[3] for s in c for b in s['row'])
        panels.append({
            'cols': [float(c) for c in cols[:5]],
            'col_dx': float(col_dx),
            'icon_w': float(ref['iw']),
            'icon_h': float(ref['ih']),
            'y_top': int(y_top),
            'y_bot': int(y_bot),
        })
    return panels


# ── Step 6: per-panel resweep + group clustering ───────────────────────────
def _resweep_rows_in_band(ccs, grid_cols, col_dx, icon_w, icon_h, y_min, y_max):
    """Within (y_min, y_max), snap every icon-sized CC to nearest grid col.
    Returns list[row], row = list[(col_idx, bbox)] sorted by col_idx."""
    tol_x = col_dx * 0.5
    candidates = []
    for b in ccs:
        x, y, w, h = b
        cy = y + h / 2
        if cy < y_min or cy > y_max:
            continue
        if w < icon_w * 0.65 or w > icon_w * 1.35:
            continue
        if h < icon_h * 0.65 or h > icon_h * 1.35:
            continue
        cx = x + w / 2
        ci = min(range(len(grid_cols)), key=lambda i: abs(grid_cols[i] - cx))
        if abs(grid_cols[ci] - cx) > tol_x:
            continue
        candidates.append((cy, ci, b))
    if not candidates:
        return []
    candidates.sort(key=lambda t: t[0])
    rows = [[candidates[0]]]
    for c in candidates[1:]:
        prev_cy = st.median(x[0] for x in rows[-1])
        if abs(c[0] - prev_cy) <= max(6, icon_h * 0.4):
            rows[-1].append(c)
        else:
            rows.append([c])
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


def _cluster_resweep_groups(rows, icon_h):
    if not rows:
        return []

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


def _merge_starship_overflow(groups, icon_h, gap_lo=0.5, gap_hi=1.5):
    """Detect [5] + [≤2 cols⊆{0,1}] pairs separated by the ship-name divider
    gap ≈ 0.6-1.3×icon_h, merge them. Phase-2 finding: ship-name divider is
    a unique structural signal for Starship Traits."""
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
            if (len(g) == 1 and len(g[0]) == 5 and len(ng) == 1):
                ncols = sorted(ci for ci, _ in ng[0])
                if len(ncols) <= 2 and all(c in (0, 1) for c in ncols):
                    g_bot = max(b[1] + b[3] for _, b in g[0])
                    ng_top = min(b[1] for _, b in ng[0])
                    gap = ng_top - g_bot
                    if gap_lo * icon_h <= gap <= gap_hi * icon_h:
                        merged.append([g[0], ng[0]])
                        skip_next = True
                        continue
        merged.append(g)
    return merged


# ── Step 7: section classification (ML probe per group) ────────────────────
def _build_name_to_section(app_cache):
    """Map every known trait/starship-trait name → section label.

    app_cache.traits[env][trait_type][name]: env={space,ground},
    trait_type={personal,rep,active_rep}.
    app_cache.starship_traits[name]: flat dict.
    """
    m = {}
    sec_for = {
        ('space', 'personal'):    'Personal Space Traits',
        ('ground', 'personal'):   'Personal Ground Traits',
        ('space', 'rep'):         'Space Reputation',
        ('ground', 'rep'):        'Ground Reputation',
        ('space', 'active_rep'):  'Active Space Rep',
        ('ground', 'active_rep'): 'Active Ground Rep',
    }
    traits = getattr(app_cache, 'traits', None) or {}
    for env in ('space', 'ground'):
        env_d = traits.get(env, {}) or {}
        for kind in ('personal', 'rep', 'active_rep'):
            sec = sec_for[(env, kind)]
            for name in (env_d.get(kind) or {}):
                m[name] = sec
    ship_traits = getattr(app_cache, 'starship_traits', None) or {}
    for name in ship_traits:
        m[name] = 'Starship Traits'
    return m


def _classify_group_section(group, img, icon_matcher, name_to_section,
                            max_probes=4):
    """Probe up to max_probes icons via classify_patch. Sum confidence per
    section, pick winner. Returns (section, votes_dict) or (None, {})."""
    H, W = img.shape[:2]
    icons = [(ci, b) for row in group for ci, b in row]
    if not icons:
        return None, {}
    step = max(1, len(icons) // max_probes)
    sample = icons[::step][:max_probes]
    votes = defaultdict(float)
    for _ci, b in sample:
        x, y, w, h = b
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = img[y0:y1, x0:x1]
        try:
            name, conf = icon_matcher.classify_patch(patch)
        except Exception:
            name, conf = '', 0.0
        if not name or conf <= 0:
            continue
        sec = name_to_section.get(name)
        if sec is None:
            continue
        votes[sec] += conf
    if not votes:
        return None, dict(votes)
    return max(votes, key=votes.get), dict(votes)


def _emit_bboxes(group, grid_cols, icon_w, icon_h):
    out = []
    for row in group:
        cy = st.median(b[1] + b[3] / 2 for _, b in row)
        for ci, _b in row:
            cx = grid_cols[ci]
            x = int(round(cx - icon_w / 2))
            y = int(round(cy - icon_h / 2))
            out.append((x, y, int(round(icon_w)), int(round(icon_h))))
    return out


# ── Master entry point ─────────────────────────────────────────────────────
_SPACE_TRAIT_SLOTS = frozenset({
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
})
_GROUND_TRAIT_SLOTS = frozenset({
    'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
})


def detect_traits(img, icon_matcher, app_cache, build_type: str | None = None):
    """Detect trait icon bboxes per section.

    Returns dict[slot_name → list[bbox]] where bbox = (x, y, w, h).
    Empty dict on failure (caller should fall back to OCR-header strategy).

    Section classification is ML-driven: each row-group is probed
    independently — we never rely on canonical section order.

    `build_type` (optional) scopes the output: GROUND_MIXED / GROUND /
    GROUND_TRAITS keep only ground-trait slots; SPACE_MIXED / SPACE /
    SPACE_TRAITS keep only space-trait slots. Cross-environment leakage
    (e.g. emitting "Starship Traits" on a ground panel) is impossible.
    """
    if icon_matcher is None or app_cache is None:
        return {}

    allowed_slots: frozenset[str] | None = None
    if build_type:
        bt = build_type.upper()
        if 'GROUND' in bt:
            allowed_slots = _GROUND_TRAIT_SLOTS
        elif 'SPACE' in bt:
            allowed_slots = _SPACE_TRAIT_SLOTS

    name_to_section = _build_name_to_section(app_cache)
    if not name_to_section:
        _slog.warning('trait_grid: empty name_to_section — cache.traits missing?')
        return {}

    ccs = _detect_icon_ccs(img)
    rows = _find_trait_rows(ccs)
    groups = _cluster_row_groups(rows)
    panels = _lock_grids_multi(groups, max_panels=4)
    if not panels:
        _slog.debug(f'trait_grid: no panels (n_ccs={len(ccs)}, n_rows={len(rows)})')
        return {}

    sections: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    panel_summaries = []

    for pi, p in enumerate(panels):
        cols = p['cols']
        col_dx = p['col_dx']
        icon_w = p['icon_w']
        icon_h = p['icon_h']
        y_min = p['y_top'] - icon_h * 0.5
        y_max = p['y_bot'] + icon_h * 2.5
        rs_rows = _resweep_rows_in_band(ccs, cols, col_dx, icon_w, icon_h,
                                        y_min, y_max)
        rs_groups = _cluster_resweep_groups(rs_rows, icon_h)
        merged = _merge_starship_overflow(rs_groups, icon_h)

        labels = []
        for g in merged:
            # Structural shortcut: a 2-row group with [5,≤2] cols 0-1 IS a
            # Starship Traits overflow pair.
            is_starship_struct = (
                len(g) == 2 and len(g[0]) == 5
                and 1 <= len(g[1]) <= 2
                and all(ci in (0, 1) for ci, _ in g[1])
            )
            if is_starship_struct:
                slot = 'Starship Traits'
            else:
                slot, _votes = _classify_group_section(
                    g, img, icon_matcher, name_to_section)
            # Drop slots that don't belong to the requested build_type
            # group (prevents space traits from leaking onto ground panels
            # and vice versa).
            if slot is not None and allowed_slots is not None \
                    and slot not in allowed_slots:
                slot = None
            labels.append(slot)
            if slot is None:
                continue
            sections[slot].extend(_emit_bboxes(g, cols, icon_w, icon_h))

        panel_summaries.append({
            'pi': pi,
            'cols': [round(c, 1) for c in cols],
            'iw': round(icon_w, 1),
            'ih': round(icon_h, 1),
            'rows': [[len(r) for r in g] for g in merged],
            'labels': labels,
        })

    _slog.info(f'trait_grid: n_panels={len(panels)} '
               f'n_sections={len(sections)} '
               f'n_bboxes={sum(len(v) for v in sections.values())}')
    for ps in panel_summaries:
        _slog.debug(f'  panel{ps["pi"]} cols={ps["cols"]} iw={ps["iw"]} '
                    f'rows={ps["rows"]} labels={ps["labels"]}')

    return dict(sections)
