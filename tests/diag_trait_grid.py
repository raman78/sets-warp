"""Trait grid — read-only diagnostic.

Goal: validate structural assumptions about TRAIT panels in STO screenshots
before designing a visual detector (analogous to BOFF marker detector).

Per annotated screen, group GT trait bboxes by section and report:
  - columns per section row (should be 5)
  - column x-position stability across sections (do all sections share the
    same 5 column x-centers?)
  - icon size distribution per section
  - section vertical separation (header gap between sections)
  - per-section row count vs. profile cap

Also probes structural anchors a future detector could use without OCR:
  - horizontal "header bar" signature above each section (dark strip with
    text) detected by row-projection on the trait x-band
  - bright-pixel column projection within trait x-band → 5 expected peaks

Pure read-only: no production code touched. Output → tests/_diag_out/trait_grid/.
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
OUT_DIR = ROOT / 'tests' / '_diag_out' / 'trait_grid'
OUT_VIZ = OUT_DIR / 'viz'
OUT_JSON = OUT_DIR / 'summary.json'

TRAIT_SLOTS_SPACE = {
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
}
TRAIT_SLOTS_GROUND = {
    'Personal Ground Traits',
    'Ground Reputation', 'Active Ground Rep',
}
TRAIT_SLOTS = TRAIT_SLOTS_SPACE | TRAIT_SLOTS_GROUND

# Canonical caps from warp_importer.py / docs/sto_slots_rules.md
SLOT_CAPS = {
    'Personal Space Traits': 11,
    'Starship Traits': 7,
    'Space Reputation': 5,
    'Active Space Rep': 5,
    'Personal Ground Traits': 11,
    'Ground Reputation': 5,
    'Active Ground Rep': 5,
}

# Column count (canonical assumption to be validated)
EXPECTED_COLS = 5


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


def cluster_1d(values, tol):
    """Greedy 1D clustering: sort, merge points within `tol`. Returns list
    of (center, members)."""
    if not values:
        return []
    vs = sorted(values)
    clusters = [[vs[0]]]
    for v in vs[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(st.median(c), c) for c in clusters]


def analyze_section(boxes, expected_cap):
    """Group bboxes into rows × columns. Returns dict of metrics."""
    if not boxes:
        return None
    # icon size stats
    ws = [b[2] for b in boxes]
    hs = [b[3] for b in boxes]
    icon_w = st.median(ws)
    icon_h = st.median(hs)
    # row clustering by y (cy)
    cys = [b[1] + b[3] / 2 for b in boxes]
    row_tol = max(8, icon_h * 0.4)
    rows = cluster_1d(cys, row_tol)
    # column clustering by x (cx) across all boxes in this section
    cxs = [b[0] + b[2] / 2 for b in boxes]
    col_tol = max(8, icon_w * 0.4)
    cols = cluster_1d(cxs, col_tol)
    # spacings
    col_centers = sorted(c[0] for c in cols)
    row_centers = sorted(r[0] for r in rows)
    col_dx = [col_centers[i + 1] - col_centers[i] for i in range(len(col_centers) - 1)]
    row_dy = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)]
    return {
        'n_boxes': len(boxes),
        'cap': expected_cap,
        'icon_w': round(icon_w, 1),
        'icon_h': round(icon_h, 1),
        'n_rows': len(rows),
        'n_cols': len(cols),
        'col_centers': [round(c, 1) for c in col_centers],
        'row_centers': [round(r, 1) for r in row_centers],
        'col_dx': [round(d, 1) for d in col_dx],
        'row_dy': [round(d, 1) for d in row_dy],
        'col_dx_med': round(st.median(col_dx), 1) if col_dx else None,
        'row_dy_med': round(st.median(row_dy), 1) if row_dy else None,
    }


def cross_section_columns(section_metrics):
    """Are column centers consistent across all sections of this screen?"""
    all_cols = []
    for m in section_metrics.values():
        if m and m.get('col_centers'):
            all_cols.append(m['col_centers'])
    if len(all_cols) < 2:
        return None
    # Take union and recluster
    flat = [c for lst in all_cols for c in lst]
    # Use a small abs tol — column should be within a few px across sections
    clusters = cluster_1d(flat, tol=8)
    return {
        'union_centers': [round(c[0], 1) for c in clusters],
        'n_unique_cols': len(clusters),
        'per_section_cols': all_cols,
    }


def detect_visual_columns(img, x_min, x_max, y_min, y_max, expected=5):
    """Project bright pixels along columns within the trait x-band.

    Returns list of detected column centers (peaks).
    """
    H, W = img.shape[:2]
    x0 = max(0, int(x_min))
    x1 = min(W, int(x_max))
    y0 = max(0, int(y_min))
    y1 = min(H, int(y_max))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    strip = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    col_bright = mask.sum(axis=0).astype(float) / 255.0
    if col_bright.max() < 1:
        return []
    # Smooth
    k = max(3, (x1 - x0) // 100)
    col_bright = np.convolve(col_bright, np.ones(k) / k, mode='same')
    threshold = col_bright.max() * 0.4
    # Find peaks: local maxima above threshold separated by min_sep
    min_sep = max(15, (x1 - x0) // (expected + 2))
    peaks = []
    for i in range(1, len(col_bright) - 1):
        if (col_bright[i] >= threshold and
                col_bright[i] >= col_bright[i - 1] and
                col_bright[i] >= col_bright[i + 1]):
            if not peaks or i - peaks[-1] >= min_sep:
                peaks.append(i)
            elif col_bright[i] > col_bright[peaks[-1]]:
                peaks[-1] = i
    return [p + x0 for p in peaks]


def viz_screen(img, fname, sections, x_band, cross, vis_cols, out_path):
    canvas = img.copy()
    # Trait x-band bounds (vertical lines)
    if x_band:
        cv2.line(canvas, (int(x_band[0]), 0), (int(x_band[0]), canvas.shape[0]),
                 (255, 255, 0), 1)
        cv2.line(canvas, (int(x_band[1]), 0), (int(x_band[1]), canvas.shape[0]),
                 (255, 255, 0), 1)
    # GT bboxes per section, color-coded
    section_colors = {
        'Personal Space Traits':  (255, 200, 0),
        'Starship Traits':        (0, 200, 255),
        'Space Reputation':       (180, 255, 100),
        'Active Space Rep':       (200, 100, 255),
        'Personal Ground Traits': (255, 200, 0),
        'Ground Reputation':      (180, 255, 100),
        'Active Ground Rep':      (200, 100, 255),
    }
    for slot, boxes in sections.items():
        col = section_colors.get(slot, (200, 200, 200))
        for b in boxes:
            x, y, w, h = b
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)
        if boxes:
            # Label section at first bbox
            x, y, _, _ = boxes[0]
            cv2.putText(canvas, slot, (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    # Cross-section column union → vertical guide lines
    if cross and cross.get('union_centers'):
        for cx in cross['union_centers']:
            cv2.line(canvas, (int(cx), 0), (int(cx), canvas.shape[0]),
                     (0, 255, 0), 1)
    # Visual-projection columns → magenta tick at top
    for vc in vis_cols:
        cv2.drawMarker(canvas, (int(vc), 8), (255, 0, 255),
                       markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=10)
    cv2.putText(canvas, fname, (4, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), canvas)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
    with open(ANN_PATH) as f:
        data = json.load(f)
    # Filter screens with confirmed trait annotations
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
    print(f'screens with trait GT: {len(targets)}')
    paths = find_screen_files(targets.keys())
    print(f'image files found: {len(paths)} / {len(targets)}')

    summary = {
        'screens': {},
        'global': {
            'col_count_per_section_hist': defaultdict(int),
            'rows_per_section': defaultdict(list),
            'cross_section_n_unique_cols_hist': defaultdict(int),
            'icon_w_per_screen_h': [],
            'col_dx_per_screen_w': [],
        },
    }
    glob = summary['global']

    for fname, boxes_by_slot in targets.items():
        path = paths.get(fname)
        if not path:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        screen_summary = {
            'size': [W, H],
            'sections': {},
        }
        section_metrics = {}
        # Determine trait x-band from all trait boxes (for visual projection)
        all_trait_boxes = [b for boxes in boxes_by_slot.values() for b in boxes]
        if all_trait_boxes:
            min_x = min(b[0] for b in all_trait_boxes)
            max_x = max(b[0] + b[2] for b in all_trait_boxes)
            x_band = (min_x - 10, max_x + 10)
        else:
            x_band = None

        for slot, boxes in boxes_by_slot.items():
            cap = SLOT_CAPS.get(slot, 0)
            metrics = analyze_section(list(boxes), cap)
            if metrics:
                section_metrics[slot] = metrics
                screen_summary['sections'][slot] = metrics
                glob['col_count_per_section_hist'][metrics['n_cols']] += 1
                glob['rows_per_section'][slot].append(metrics['n_rows'])
                glob['icon_w_per_screen_h'].append(metrics['icon_w'] / H)
                if metrics['col_dx_med']:
                    glob['col_dx_per_screen_w'].append(metrics['col_dx_med'] / W)

        # Cross-section column consistency
        cross = cross_section_columns(section_metrics)
        if cross:
            screen_summary['cross_section_columns'] = cross
            glob['cross_section_n_unique_cols_hist'][cross['n_unique_cols']] += 1

        # Visual-projection columns within trait x-band, full image height
        vis_cols = []
        if x_band:
            vis_cols = detect_visual_columns(
                img, x_band[0], x_band[1], 0, H, expected=EXPECTED_COLS,
            )
            screen_summary['visual_projection_cols'] = vis_cols

        summary['screens'][fname] = screen_summary

        # Viz
        viz_screen(img, fname, boxes_by_slot, x_band, cross, vis_cols,
                   OUT_VIZ / fname)

    # Compress globals to plain dicts
    glob['col_count_per_section_hist'] = dict(glob['col_count_per_section_hist'])
    glob['cross_section_n_unique_cols_hist'] = dict(glob['cross_section_n_unique_cols_hist'])
    glob['rows_per_section'] = {k: v for k, v in glob['rows_per_section'].items()}
    glob['icon_w_per_screen_h_med'] = round(st.median(glob['icon_w_per_screen_h']), 4) \
        if glob['icon_w_per_screen_h'] else None
    glob['icon_w_per_screen_h_min'] = round(min(glob['icon_w_per_screen_h']), 4) \
        if glob['icon_w_per_screen_h'] else None
    glob['icon_w_per_screen_h_max'] = round(max(glob['icon_w_per_screen_h']), 4) \
        if glob['icon_w_per_screen_h'] else None
    glob['col_dx_per_screen_w_med'] = round(st.median(glob['col_dx_per_screen_w']), 4) \
        if glob['col_dx_per_screen_w'] else None
    # rows-per-section summary
    rows_summary = {}
    for slot, lst in glob['rows_per_section'].items():
        if not lst:
            continue
        rows_summary[slot] = {
            'n_screens': len(lst),
            'min': min(lst), 'max': max(lst), 'med': st.median(lst),
        }
    glob['rows_per_section_summary'] = rows_summary

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {OUT_JSON}')
    print(f'Wrote {sum(1 for _ in OUT_VIZ.glob("*"))} viz images to {OUT_VIZ}')

    # ── Console report ──
    print('\n=== GLOBAL ===')
    print(f'col_count_per_section_hist: {glob["col_count_per_section_hist"]}')
    print(f'cross_section_n_unique_cols_hist: {glob["cross_section_n_unique_cols_hist"]}')
    print(f'icon_w / screen_h: med={glob["icon_w_per_screen_h_med"]} '
          f'min={glob["icon_w_per_screen_h_min"]} max={glob["icon_w_per_screen_h_max"]}')
    print(f'col_dx / screen_w (med): {glob["col_dx_per_screen_w_med"]}')
    print('\nRows per section:')
    for slot, s in rows_summary.items():
        print(f'  {slot:28s} n={s["n_screens"]:3d} '
              f'min={s["min"]} max={s["max"]} med={s["med"]}')


if __name__ == '__main__':
    main()
