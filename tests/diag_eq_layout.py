"""
SPACE_EQ layout diagnostic — analysis + visualization.

Purpose (2026-05-08): start designing a structure-driven SPACE_EQ detector
analogous to BOFF marker detector. Before writing any new detector, look at
what the current code does on a real screenshot and verify two empirical
claims the user made:

  (a) Each EQ row has a faint dark-blueish horizontal background bar that
      extends right from the OCR label across the full row width — visible
      even on slots that have no icon. Pattern: thin / thick / thin
      horizontal stripes (top to bottom).

  (b) Even with 0 OCR matches, a single detected single-slot row (Deflector,
      Shields, Impulse, Warp Core, Sec-Def) is enough to anchor the rest of
      the grid because row pitch + ship-tier profile fully determine the
      layout.

This script does NOT modify production code. It loads one screenshot, runs
the existing LayoutDetector, draws GT bboxes from annotations.json beside
detector output, and samples HSV column profiles at non-icon x positions of
each known row to visualize the row-background stripe.

Output:
  tests/_diag_out/eq_layout/<stem>__overlay.png — GT (green) + detector (red)
  tests/_diag_out/eq_layout/<stem>__rowbg.png   — row-background HSV profile
  tests/_diag_out/eq_layout/<stem>__report.txt  — text dump

Usage:
  python tests/diag_eq_layout.py [screenshot_filename]
  default: broadside.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/raman/PycharmProjects/sets-warp')
sys.path.insert(0, str(ROOT))

from warp.recognition.layout_detector import (  # noqa: E402
    LayoutDetector, SPACE_SLOT_ORDER_STANDARD, SLOT_DEFAULT_COUNTS,
)

SCREENS_ROOT = Path('/home/raman/STO_screens')
ANN_PATH     = ROOT / 'warp' / 'training_data' / 'annotations.json'
OUT_DIR      = ROOT / 'tests' / '_diag_out' / 'eq_layout'

# EQ-only slots — we ignore traits, BOFFs, ship-name etc. for this pass.
EQ_SLOTS = set(SPACE_SLOT_ORDER_STANDARD) | {'Hangars', 'Experimental', 'Sec-Def'}

# Visual styles
GT_COLOR        = (60,  220, 60)    # green
DET_COLOR       = (40,  40,  240)   # red
LABEL_BG        = (20,  20,  20)
LABEL_FG        = (240, 240, 240)


def find_screenshot(name: str) -> Path | None:
    for p in SCREENS_ROOT.rglob(name):
        if p.is_file():
            return p
    return None


def load_gt_eq_bboxes(fname: str) -> dict[str, list[tuple[int, int, int, int]]]:
    if not ANN_PATH.exists():
        return {}
    data = json.loads(ANN_PATH.read_text())
    items = data.get(fname, [])
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for it in items:
        slot = it.get('slot', '')
        if slot not in EQ_SLOTS:
            continue
        bb = it.get('bbox')
        if not bb or len(bb) != 4:
            continue
        out.setdefault(slot, []).append(tuple(int(v) for v in bb))
    return out


def synth_profile_from_gt(gt: dict) -> dict:
    """Build a ShipDB-style slot-count profile from GT counts (counts what
    the user actually annotated). Used so the detector doesn't have to
    guess from a real ShipDB lookup."""
    prof: dict = {}
    for slot, boxes in gt.items():
        prof[slot] = max(SLOT_DEFAULT_COUNTS.get(slot, 1), len(boxes))
    # Make sure all standard EQ slots have at least their default count.
    for s in SPACE_SLOT_ORDER_STANDARD:
        prof.setdefault(s, SLOT_DEFAULT_COUNTS.get(s, 1))
    if 'Hangars' in gt:
        prof['Hangars'] = max(1, len(gt['Hangars']))
    return prof


def draw_box(canvas, bbox, color, label: str = '', thickness: int = 2):
    x, y, w, h = bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    if label:
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 6, y), LABEL_BG, -1)
        cv2.putText(canvas, label, (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def overlay_image(img, gt: dict, det: dict, out_path: Path):
    canvas = img.copy()
    for slot, boxes in gt.items():
        for b in boxes:
            draw_box(canvas, b, GT_COLOR, slot, 2)
    for slot, boxes in det.items():
        if slot not in EQ_SLOTS:
            continue
        for b in boxes:
            draw_box(canvas, b, DET_COLOR, '', 1)
    # Legend strip
    h, w = canvas.shape[:2]
    legend_h = 30
    pad = np.full((legend_h, w, 3), 25, dtype=np.uint8)
    cv2.putText(pad, 'GT (green)  /  detector (red)',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    full = np.vstack([pad, canvas])
    cv2.imwrite(str(out_path), full)


def sample_row_background(img, gt: dict) -> tuple[np.ndarray, dict]:
    """For each GT row, sample a vertical strip across the full row width
    at x positions that fall BETWEEN icons (gaps). Compute the mean HSV
    column profile and look for a thin/thick/thin stripe pattern.

    Returns: (visualization image, per-row stats dict).
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Group GT bboxes into row clusters by Y-overlap. Each cluster shares
    # the same row band and the same gap pattern on the X axis.
    rows: list[dict] = []
    flat = [(slot, b) for slot, bxs in gt.items() for b in bxs]
    flat.sort(key=lambda t: t[1][1])
    for slot, b in flat:
        x, y, w, h = b
        cy = y + h // 2
        attached = False
        for r in rows:
            if abs(cy - r['cy']) <= max(8, r['h'] // 3):
                r['items'].append((slot, b))
                r['cy'] = int(np.mean([it[1][1] + it[1][3] // 2 for it in r['items']]))
                r['y_top']  = min(r['y_top'], y)
                r['y_bot']  = max(r['y_bot'], y + h)
                attached = True
                break
        if not attached:
            rows.append({
                'cy':    cy,
                'h':     h,
                'y_top': y,
                'y_bot': y + h,
                'items': [(slot, b)],
            })

    stats: dict = {}
    panels: list[np.ndarray] = []

    for r in rows:
        items = sorted(r['items'], key=lambda t: t[1][0])
        slot_label = items[0][0]
        x_left  = min(b[0]            for _, b in items)
        x_right = max(b[0] + b[2]     for _, b in items)
        # Extend ±20% horizontally to capture left-of-icons (label area)
        # and right-of-icons (empty tail) — these are the "no-icon" zones
        # where the stripe should be visible.
        ext = max(40, int(0.2 * (x_right - x_left)))
        x0 = max(0, x_left  - ext)
        x1 = min(W, x_right + ext)

        # Pad row vertically to capture the full thin/thick/thin stripe band.
        # Empirically each row icon is ~32px tall; stripe band may extend
        # a few px above and below.
        pad_v = max(6, r['h'] // 4)
        y0 = max(0, r['y_top'] - pad_v)
        y1 = min(H, r['y_bot'] + pad_v)

        # Extract the row strip and compute the per-row mean HSV column
        # profile at x positions BETWEEN icons (gaps).
        gap_xs: list[int] = []
        prev_right = x0
        for _, b in items:
            ix0 = b[0]
            if ix0 - prev_right > 4:
                gap_xs.extend(range(prev_right + 2, ix0 - 1))
            prev_right = b[0] + b[2]
        if x1 - prev_right > 4:
            gap_xs.extend(range(prev_right + 2, x1 - 1))

        if not gap_xs:
            continue

        # Sample the gap columns to get a vertical HSV profile (mean of all gap cols).
        gap_cols = hsv[y0:y1, gap_xs, :]            # (rowH, n_gap, 3)
        if gap_cols.size == 0:
            continue
        v_profile = gap_cols.mean(axis=1)            # (rowH, 3)  H S V means

        # Look for stripes via V channel: a stripe is a row-band where V
        # is low AND H is in blueish range (~95..130) AND S is low.
        v   = v_profile[:, 2]
        h_  = v_profile[:, 0]
        s_  = v_profile[:, 1]
        # Lower V than icon area + bluish hue + low-medium sat = candidate stripe
        is_stripe = (v < 70) & (h_ > 90) & (h_ < 135)
        # Continuous bands
        bands: list[tuple[int, int]] = []
        in_band = False
        bs = 0
        for i, on in enumerate(is_stripe):
            if on and not in_band:
                bs = i; in_band = True
            elif not on and in_band:
                bands.append((bs, i)); in_band = False
        if in_band:
            bands.append((bs, len(is_stripe)))

        stats[slot_label] = {
            'cy':       r['cy'],
            'row_h':    r['h'],
            'n_items':  len(items),
            'gap_cols': len(gap_xs),
            'bands':    bands,
            'mean_v':   float(v.mean()),
            'mean_h':   float(h_.mean()),
            'mean_s':   float(s_.mean()),
        }

        # Build a panel: row strip on the left, profile chart on the right.
        row_strip = img[y0:y1, x0:x1].copy()
        # Mark gap columns with a faint blue overlay
        overlay = row_strip.copy()
        for gx in gap_xs:
            cv2.line(overlay, (gx - x0, 0), (gx - x0, row_strip.shape[0]),
                     (255, 100, 0), 1)
        row_strip = cv2.addWeighted(row_strip, 0.7, overlay, 0.3, 0)
        # Draw band markers
        for bs_, be_ in bands:
            cv2.rectangle(row_strip,
                          (0, bs_), (row_strip.shape[1] - 1, be_),
                          (40, 240, 40), 1)

        # V profile chart
        chart_w = 220
        chart_h = row_strip.shape[0]
        chart = np.full((chart_h, chart_w, 3), 25, dtype=np.uint8)
        for i, val in enumerate(v):
            x_ = int((val / 255.0) * (chart_w - 2))
            cv2.line(chart, (1, i), (1 + x_, i), (200, 200, 200), 1)
        # Hue overlay in red
        for i, hv in enumerate(h_):
            x_ = int((hv / 180.0) * (chart_w - 2))
            cv2.circle(chart, (1 + x_, i), 0, (60, 60, 240), -1)

        # Slot label header
        header = np.full((22, row_strip.shape[1] + chart_w, 3), 35, dtype=np.uint8)
        cv2.putText(
            header,
            f'{slot_label}  cy={r["cy"]}  h={r["h"]}  bands={len(bands)}  '
            f'meanV={v.mean():.1f}  meanH={h_.mean():.1f}',
            (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )

        panel = np.hstack([row_strip, chart])
        panel = np.vstack([header, panel])
        panels.append(panel)

    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8), stats

    max_w = max(p.shape[1] for p in panels)
    norm = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 25, dtype=np.uint8)
            p = np.hstack([p, pad])
        norm.append(p)
    return np.vstack(norm), stats


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'broadside.png'
    img_path = find_screenshot(fname)
    if img_path is None:
        print(f'NOT FOUND: {fname} under {SCREENS_ROOT}')
        sys.exit(1)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'cv2 failed to load: {img_path}')
        sys.exit(1)
    H, W = img.shape[:2]
    print(f'Loaded {img_path} ({W}x{H})')

    gt = load_gt_eq_bboxes(fname)
    print(f'GT EQ slots: {sorted(gt.keys())}')
    for s in sorted(gt):
        print(f'  {s:24s}  n={len(gt[s])}')

    profile = synth_profile_from_gt(gt)
    detector = LayoutDetector()
    det_layout = detector.detect(img, 'SPACE_EQ', profile)
    eq_det = {k: v for k, v in det_layout.items() if k in EQ_SLOTS}
    print(f'\nDetector returned slots: {sorted(eq_det.keys())}')
    for s in sorted(eq_det):
        print(f'  {s:24s}  n={len(eq_det[s])}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(fname).stem

    overlay_image(img, gt, eq_det, OUT_DIR / f'{stem}__overlay.png')
    rowbg_img, rowbg_stats = sample_row_background(img, gt)
    cv2.imwrite(str(OUT_DIR / f'{stem}__rowbg.png'), rowbg_img)

    report = OUT_DIR / f'{stem}__report.txt'
    with report.open('w') as f:
        f.write(f'screenshot: {img_path}\nsize: {W}x{H}\n\n')
        f.write('=== GT EQ slots ===\n')
        for s in sorted(gt):
            for b in gt[s]:
                f.write(f'  {s:24s}  bbox={b}\n')
        f.write('\n=== Detector EQ slots ===\n')
        for s in sorted(eq_det):
            for b in eq_det[s]:
                f.write(f'  {s:24s}  bbox={b}\n')
        f.write('\n=== Row-background HSV stripe analysis ===\n')
        for s, st in rowbg_stats.items():
            f.write(f'  {s:24s}  cy={st["cy"]:4d}  rowH={st["row_h"]:3d}  '
                    f'gap_cols={st["gap_cols"]:4d}  bands={len(st["bands"])}  '
                    f'meanV={st["mean_v"]:.1f}  meanH={st["mean_h"]:.1f}  '
                    f'meanS={st["mean_s"]:.1f}  band_runs={st["bands"]}\n')

    print(f'\nWrote:\n  {OUT_DIR / f"{stem}__overlay.png"}'
          f'\n  {OUT_DIR / f"{stem}__rowbg.png"}'
          f'\n  {report}')


if __name__ == '__main__':
    main()
