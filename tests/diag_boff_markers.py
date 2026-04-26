"""
BOFF profession-marker detector — read-only prototype v2.

Each BOFF seat has a small profession-coloured badge on the LEFT of its
name bar (which sits BELOW the 4 ability icons). The badge colour matches
the seat's profession:
  - Tactical:    red    H~0,    S>150, V>80
  - Engineering: yellow H~22,   S>50,  V>150  (S can be low for pale gold)
  - Science:     blue   H~108,  S>150, V>100

These markers are far more uniform than ability icons:
  - solid colour (high S OR high V),
  - fixed size (~icon_w × icon_h),
  - present at every seat regardless of which abilities are slotted.

Approach:
  1. Build narrow-band colour mask for each of the 3 colours separately.
  2. Find CCs sized roughly icon_w × icon_h.
  3. Cluster CCs into columns by x.
  4. Score column-pair candidates expecting 2 cols × (2..3) markers,
     col-gap ≈ 4 * pitch_x, pitch_y ≈ 2.4 * icon_h.
  5. Best pair = BOFF panel anchor.

Validation: per screen with ≥5 GT BOFFs, compare detected anchor centre
to GT panel centre; also count markers landing on a GT seat.
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR  = ROOT / 'tests' / '_diag_out'
OUT_JSON = OUT_DIR / 'boff_markers.json'
OUT_VIZ  = OUT_DIR / 'boff_markers_viz'

VIRTUAL = frozenset({'__empty__', '__inactive__'})

# Colour bands (OpenCV HSV: H 0-180).
# Sampled across 25 GT seats in 5 screens.
COLOUR_BANDS = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, code
    ('TAC', 0,   6,   160, 255, 90,  255, 'T'),     # red
    ('TAC', 174, 180, 160, 255, 90,  255, 'T'),     # red wrap-around
    ('ENG', 18,  30,  120, 200, 160, 210, 'E'),     # saturated gold
    ('SCI', 102, 114, 160, 255, 110, 255, 'S'),     # blue
    ('UNI', 18,  30,  40,  110, 195, 255, 'U'),     # pale cream yellow (low S, high V)
]


def find_screen_files(fnames):
    candidate_dirs = [
        Path('/home/raman/STO_screens'),
        ROOT / 'warp' / 'training_data' / 'migrated_images',
        ROOT / 'warp' / 'training_data',
    ]
    out = {}
    targets = set(fnames)
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if (p.is_file()
                    and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                    and p.name in targets and p.name not in out):
                out[p.name] = p
    return out


def estimate_icon_dims(img):
    """Initial wide range. Refined later from detected marker sizes."""
    h, w = img.shape[:2]
    # Allow both full-screen (~4.7% of h) and panel-crop (~11% of h)
    icon_h = max(int(round(h * 0.045)), 24)
    icon_w = max(int(round(icon_h * 0.78)), 22)
    return icon_w, icon_h


def refine_dims_from_markers(markers, icon_w, icon_h):
    """If we have ≥4 markers, use their median size — otherwise fallback."""
    if len(markers) < 4:
        return icon_w, icon_h
    hs = sorted(m[3] for m in markers)
    ws = sorted(m[2] for m in markers)
    med_h = hs[len(hs) // 2]
    med_w = ws[len(ws) // 2]
    # Markers are roughly icon-sized (height); accept if reasonable
    if 16 <= med_h <= 90 and 4 <= med_w <= 90:
        return max(med_w, 16), max(med_h, 20)
    return icon_w, icon_h


def colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    return ((H >= h_lo) & (H <= h_hi)
            & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255


def detect_markers(img, icon_w, icon_h):
    """Return list of (x, y, w, h, code)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Marker CC shape — solid blob roughly icon-sized
    min_w = max(6, int(icon_w * 0.25))
    max_w = max(int(icon_w * 1.4), 40)
    min_h = max(8, int(icon_h * 0.4))
    max_h = max(int(icon_h * 1.6), 50)

    out = []
    seen_rects = []  # to dedupe overlaps across colour bands
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in COLOUR_BANDS:
        m = colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
        # consolidate
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(2, icon_h // 8))),
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < min_w or w > max_w:
                continue
            if h < min_h or h > max_h:
                continue
            # squareish-ish blob — reject very thin lines
            ar = w / max(h, 1)
            if ar < 0.15 or ar > 3.5:
                continue
            # density check: filled fraction > 0.30
            if area < (w * h) * 0.25:
                continue
            # dedupe overlap
            dup = False
            for (px, py, pw, ph) in seen_rects:
                if x < px + pw and px < x + w and y < py + ph and py < y + h:
                    ix = min(x + w, px + pw) - max(x, px)
                    iy = min(y + h, py + ph) - max(y, py)
                    iou = (ix * iy) / max(w * h + pw * ph - ix * iy, 1)
                    if iou > 0.4:
                        dup = True
                        break
            if dup:
                continue
            seen_rects.append((x, y, w, h))
            out.append((int(x), int(y), int(w), int(h), code))
    return out


def cluster_columns(markers, x_tol):
    if not markers:
        return []
    ms = sorted(markers, key=lambda m: m[0])
    cols = [[ms[0]]]
    for m in ms[1:]:
        last_x = st.median(c[0] for c in cols[-1])
        if abs(m[0] - last_x) <= x_tol:
            cols[-1].append(m)
        else:
            cols.append([m])
    return cols


def score_pair(col_a, col_b, icon_w, icon_h):
    # Each column must have at least 2 markers (BOFF panel: L≥2, R≥2 except T0)
    if len(col_a) < 2 or len(col_b) < 2:
        return -1.0
    if len(col_a) > 4 or len(col_b) > 4:
        return -1.0
    xa = st.median(m[0] for m in col_a)
    xb = st.median(m[0] for m in col_b)
    if xb <= xa:
        return -1.0
    dx = xb - xa
    dx_norm = dx / max(icon_w, 1)
    if not (3.5 <= dx_norm <= 9.0):
        return -1.0

    # Need ≥2 row alignments
    yas = sorted(m[1] + m[3] / 2 for m in col_a)
    ybs = sorted(m[1] + m[3] / 2 for m in col_b)
    aligned = 0
    for yb in ybs:
        if any(abs(yb - ya) < icon_h * 0.6 for ya in yas):
            aligned += 1
    if aligned < 2:
        return -1.0

    # Pitch_y check on the larger column
    big = col_a if len(col_a) >= len(col_b) else col_b
    ys = sorted(m[1] + m[3] / 2 for m in big)
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    med_gap = st.median(gaps)
    gap_norm = med_gap / icon_h
    if not (1.6 <= gap_norm <= 3.2):
        return -1.0
    gap_score = 1.0 - abs(gap_norm - 2.38) / 1.0

    # Prefer 3+2 = 5 (canonical T6 BOFF panel)
    n_total = len(col_a) + len(col_b)
    canon_score = 1.0 - abs(n_total - 5) * 0.25

    # Bigger column on the LEFT is canonical (3 left, 2 right). If swapped,
    # mild penalty (mirror layouts exist but are minority).
    layout_bonus = 0.2 if len(col_a) >= len(col_b) else 0.0

    # Profession diversity (real BOFF panel has 2-3 different professions)
    codes = {m[4] for m in col_a} | {m[4] for m in col_b}
    diversity_bonus = 0.3 if len(codes) >= 2 else -0.5

    score = (1.5 * aligned
             + 0.6 * gap_score
             + 0.5 * canon_score
             + diversity_bonus
             + layout_bonus)
    return score


def best_panel(markers, icon_w, icon_h):
    """Pick a 2-row anchor pair (markers in same row, separated by col_gap).
    For each candidate pair, count how many OTHER markers fall in the
    panel-sized bounding box around them. Best score wins.

    Panel BBox: width ≈ col_gap + 2*icon_w; height ≈ 3*pitch_y + icon_h.
    """
    if len(markers) < 4:
        return None

    cx_arr = [m[0] + m[2] / 2 for m in markers]
    cy_arr = [m[1] + m[3] / 2 for m in markers]
    n = len(markers)

    # We DON'T trust icon_h from image — use a wide range of pitch_y
    # candidates and pick the one with most inliers.
    pitch_y_candidates = [
        max(icon_h * 1.6, 24),
        max(icon_h * 2.0, 30),
        max(icon_h * 2.4, 36),
        max(icon_h * 3.0, 50),
        max(icon_h * 3.6, 70),
    ]
    x_tol = max(icon_w * 0.7, 8)

    best = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Row anchor: similar y, dx in panel range
            if abs(cy_arr[i] - cy_arr[j]) > max(icon_h * 0.7, 12):
                continue
            xa, xb = cx_arr[i], cx_arr[j]
            if xb <= xa + 3 * icon_w:
                continue
            if xb - xa > 9 * icon_w:
                continue

            for pitch_y in pitch_y_candidates:
                y_tol = pitch_y * 0.30

                # Collect inliers in each column at expected row positions
                # Allow 0..3 extra rows below and 0..3 above the anchor row
                anchor_y = (cy_arr[i] + cy_arr[j]) / 2
                col_a, col_b = [], []
                for k in range(n):
                    cx, cy = cx_arr[k], cy_arr[k]
                    # Closest expected row index
                    r = round((cy - anchor_y) / pitch_y)
                    if abs(r) > 3:
                        continue
                    expected_y = anchor_y + r * pitch_y
                    if abs(cy - expected_y) > y_tol:
                        continue
                    if abs(cx - xa) <= x_tol:
                        col_a.append((r, markers[k]))
                    elif abs(cx - xb) <= x_tol:
                        col_b.append((r, markers[k]))

                # Dedupe by row index
                def keep_one(col):
                    seen = {}
                    for r, m in col:
                        if r not in seen:
                            seen[r] = m
                    return [m for _, m in sorted(seen.items())]
                col_a = keep_one(col_a)
                col_b = keep_one(col_b)

                if len(col_a) < 2 or len(col_b) < 1:
                    continue
                # Canonical max: L=3, R=2. Reject impossible configurations.
                if len(col_a) > 3 or len(col_b) > 3:
                    continue

                aligned = 0
                yas = [m[1] + m[3] / 2 for m in col_a]
                ybs = [m[1] + m[3] / 2 for m in col_b]
                for yb in ybs:
                    if any(abs(yb - ya) < y_tol for ya in yas):
                        aligned += 1
                if aligned < 1:
                    continue

                n_total = len(col_a) + len(col_b)
                # Strong canonical preference: 3+2=5 best, 2+2=4 next.
                canon_table = {5: 1.5, 4: 1.0, 3: 0.4, 2: 0.0, 6: 0.6}
                canon = canon_table.get(n_total, 0.0)

                codes = {m[4] for m in col_a} | {m[4] for m in col_b}
                # BOFF panels nearly always have ≥ 2 different professions
                div = 0.6 if len(codes) >= 2 else -0.8
                layout = 0.3 if len(col_a) >= len(col_b) else 0.0

                # Pitch-y consistency: if col_a has ≥ 2 markers, std of
                # row-index-derived y diffs should be small.
                pitch_score = 0.0
                if len(col_a) >= 2:
                    ys = sorted(yas)
                    diffs = [ys[k + 1] - ys[k] for k in range(len(ys) - 1)]
                    if len(diffs) >= 2:
                        m = st.mean(diffs)
                        d = st.stdev(diffs)
                        pitch_score = 0.4 * (1.0 - min(d / max(m, 1), 1.0))

                score = (0.6 * aligned + canon + div + layout + pitch_score)
                if best is None or score > best[2]:
                    best = (col_a, col_b, score)
    return best


def gt_panel_extent(boffs):
    """Return (x0, y0, x1, y1) bounding all GT BOFF abilities."""
    x0 = min(b[0] for b in boffs)
    y0 = min(b[1] for b in boffs)
    x1 = max(b[0] + b[2] for b in boffs)
    y1 = max(b[1] + b[3] for b in boffs)
    return x0, y0, x1, y1


def gt_seat_centres(boffs, icon_h):
    if not boffs:
        return []
    cxs = sorted(b[0] + b[2] / 2 for b in boffs)
    if len(cxs) < 2:
        return []
    gaps = [(cxs[i + 1] - cxs[i], i) for i in range(len(cxs) - 1)]
    _, gi = max(gaps)
    split_x = (cxs[gi] + cxs[gi + 1]) / 2

    rows = {'L': {}, 'R': {}}
    for b in boffs:
        cx = b[0] + b[2] / 2
        cy = b[1] + b[3] / 2
        col = 'L' if cx < split_x else 'R'
        ykey = round(cy / icon_h)
        rows[col].setdefault(ykey, []).append((b[0], cy))
    out = []
    for col, ys in rows.items():
        for _, members in ys.items():
            yc = st.median(m[1] for m in members)
            xl = min(m[0] for m in members)
            out.append((col, yc, xl))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def evaluate_screen(fname, img, boffs):
    h, w = img.shape[:2]
    icon_w, icon_h = estimate_icon_dims(img)
    markers = detect_markers(img, icon_w, icon_h)
    panel = best_panel(markers, icon_w, icon_h)

    gt_seats = gt_seat_centres(boffs, icon_h)
    gx0, gy0, gx1, gy1 = gt_panel_extent(boffs)

    # Per-seat hit: marker found within y±0.7*icon_h and at x within
    # [x_left - 1.6*icon_w, x_left] of GT seat.
    seat_hit = 0
    for col, yc, xl in gt_seats:
        for mx, my, mw, mh, _ in markers:
            mcx = mx + mw / 2
            mcy = my + mh / 2
            if (abs(mcy - (yc + icon_h * 1.0)) < icon_h * 0.9
                    and (xl - icon_w * 1.8) < mcx < xl + icon_w * 0.3):
                seat_hit += 1
                break

    # Panel-anchor hit: panel centre within GT bbox
    panel_info = None
    panel_in_gt = False
    if panel is not None:
        a, b, s = panel
        all_m = a + b
        px0 = min(m[0] for m in all_m)
        py0 = min(m[1] for m in all_m)
        px1 = max(m[0] + m[2] for m in all_m)
        py1 = max(m[1] + m[3] for m in all_m)
        pcx = (px0 + px1) / 2
        pcy = (py0 + py1) / 2
        gcx = (gx0 + gx1) / 2
        gcy = (gy0 + gy1) / 2
        # tolerance: within 1.5×panel_width of GT centre
        gt_w = gx1 - gx0
        gt_h = gy1 - gy0
        panel_in_gt = (abs(pcx - gcx) < gt_w * 0.7
                       and abs(pcy - gcy) < gt_h * 0.7)
        panel_info = {
            'col_a_x':  float(st.median(m[0] for m in a)),
            'col_b_x':  float(st.median(m[0] for m in b)),
            'col_a_n':  len(a),
            'col_b_n':  len(b),
            'score':    round(s, 3),
            'pcx':      round(pcx, 1),
            'pcy':      round(pcy, 1),
            'gcx':      round(gcx, 1),
            'gcy':      round(gcy, 1),
            'in_gt':    panel_in_gt,
        }

    return {
        'file':       fname,
        'iw':         w,
        'ih':         h,
        'icon_w':     icon_w,
        'icon_h':     icon_h,
        'n_markers':  len(markers),
        'gt_seats':   len(gt_seats),
        'seat_hits':  seat_hit,
        'hit_pct':    round(seat_hit / max(len(gt_seats), 1) * 100, 1),
        'panel':      panel_info,
        'panel_ok':   panel_in_gt,
        'markers':    [list(m) for m in markers],
    }


def viz(fname, img, markers, panel, boffs):
    out = img.copy()
    code_color = {
        'T': (0, 0, 255),       # red
        'E': (0, 200, 255),     # gold
        'S': (255, 100, 0),     # blue
        'U': (200, 255, 255),   # cream (Universal)
        'X': (255, 0, 200),     # magenta (legacy)
    }
    for x, y, w, h, code in markers:
        c = code_color.get(code, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
    if panel is not None:
        a, b, _ = panel
        all_m = a + b
        x0 = min(m[0] for m in all_m) - 4
        y0 = min(m[1] for m in all_m) - 4
        x1 = max(m[0] + m[2] for m in all_m) + 4
        y1 = max(m[1] + m[3] for m in all_m) + 4
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if boffs:
        gx0, gy0, gx1, gy1 = gt_panel_extent(boffs)
        cv2.rectangle(out, (gx0, gy0), (gx1, gy1), (255, 255, 255), 1)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_VIZ / fname.replace('/', '_')), out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANN_PATH.read_text(encoding='utf-8'))

    candidates = []
    for fname, entries in ann.items():
        if not isinstance(entries, list):
            continue
        boffs = [a['bbox'] for a in entries
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('name') not in VIRTUAL
                 and a.get('bbox')]
        if len(boffs) >= 5:
            candidates.append((fname, boffs))

    paths = find_screen_files([c[0] for c in candidates])
    print(f'Screens with ≥5 BOFF GT: {len(candidates)}, '
          f'findable: {len(paths)}')

    rows = []
    total_seats = 0
    total_hits = 0
    panel_ok_count = 0
    for fname, boffs in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        m = evaluate_screen(fname, img, boffs)
        rows.append(m)
        total_seats += m['gt_seats']
        total_hits  += m['seat_hits']
        if m['panel_ok']:
            panel_ok_count += 1

        # Reproduce same dims for viz
        iw0, ih0 = estimate_icon_dims(img)
        markers_full = detect_markers(img, iw0, ih0)
        panel = best_panel(markers_full, iw0, ih0)
        viz(fname, img, markers_full, panel, boffs)

    rows.sort(key=lambda r: (r['panel_ok'], r['hit_pct']))
    print('\nPer-screen results (sorted by panel_ok, hit %):')
    for r in rows:
        bar = '█' * int(r['hit_pct'] / 10)
        panel_ok = '✓' if r['panel_ok'] else ('?' if r['panel'] else '✗')
        print(f'  {r["file"][:38]:<38s}  '
              f'seats={r["seat_hits"]}/{r["gt_seats"]} '
              f'({r["hit_pct"]:5.1f}%)  '
              f'mks={r["n_markers"]:3d}  '
              f'panel={panel_ok}  {bar}')

    print('\n' + '=' * 80)
    print(f'Total seats hit: {total_hits}/{total_seats} '
          f'({total_hits / max(total_seats, 1) * 100:.1f}%)')
    print(f'Panel anchored on GT: {panel_ok_count}/{len(rows)} '
          f'({panel_ok_count / max(len(rows), 1) * 100:.1f}%)')
    print(f'Mean markers detected: {st.mean(r["n_markers"] for r in rows):.1f}')
    print(f'Median markers detected: {st.median(r["n_markers"] for r in rows):.1f}')
    print(f'Visualisations: {OUT_VIZ}')

    OUT_JSON.write_text(json.dumps({'screens': rows}, indent=2),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
