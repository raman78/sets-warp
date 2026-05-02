"""
BOFF profession-marker detector — read-only prototype v3.

Each BOFF seat has a profession-coloured badge on the LEFT of its name
bar (which sits BELOW the 4 ability icons). The badge has two zones:
  - main zone (75-95% of bar width): seat type — TAC / ENG / SCI / UNI
  - spec stripe (right 5-25% of bar): specialization — Command,
    Intelligence, Temporal, Pilot, Miracle Worker (only on spec seats)

HSV bands are sampled across 25+ GT seats; stripe bands verified
2026-04-26 against 15 user-labelled seats (see
`tests/diag_boff_spec_stripe_gt.py`).

Approach:
  1. Build narrow-band colour mask per band (4 main + 5 stripe).
  2. Find CCs sized roughly icon_w × icon_h (main) or narrower (stripe).
  3. IoU-dedupe overlapping CCs across bands.
  4. RANSAC-style pair search: for each (m_i, m_j) at similar y and
     dx in [3..9]·icon_w, sweep 5 pitch_y candidates, count inliers.
  5. Score: prefers 3+2=5, profession diversity, L≥R layout.

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
OUT_DIR        = ROOT / 'tests' / '_diag_out'
OUT_JSON       = OUT_DIR / 'boff_markers.json'
OUT_VIZ        = OUT_DIR / 'boff_markers_viz'
OUT_INSPECTOR  = OUT_DIR / 'boff_spec_inspector'

VIRTUAL = frozenset({'__empty__', '__inactive__'})

# Seat-type bands (main zone of the marker, 75-95% of bar width).
# Used as DETECTION seeds. Sampled across 25+ GT seats.
MAIN_BANDS = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, code
    ('TAC',   0,   6,   125, 255,  85, 255, 'T'),  # red
    ('TAC', 174, 180,   125, 255,  85, 255, 'T'),  # red wrap
    ('ENG',  18,  30,   100, 220, 160, 255, 'E'),  # saturated gold
    ('SCI', 102, 114,   160, 255, 140, 255, 'S'),  # blue
    ('UNI',  18,  30,    25,  95, 195, 255, 'U'),  # pale cream
]

# Spec-stripe bands (narrow right edge, 5-25% of bar width). NOT used
# for detection — applied post-hoc to label each detected seat with its
# specialization. Verified against 15 user-labelled seats (2026-04-26).
STRIPE_BANDS = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, spec_code
    ('CMD',  12,  19,    80, 200, 130, 200, 'O'),  # Command — orange
    ('INT', 120, 135,   130, 200, 150, 220, 'P'),  # Intelligence — purple
    ('TMP',  25,  35,   140, 255, 215, 255, 'Y'),  # Temporal — bright gold
    ('PIL',  86, 100,    80, 140, 215, 255, 'C'),  # Pilot — light cyan
    ('MW',   32,  44,   220, 255, 195, 255, 'L'),  # Miracle Worker — lime
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


def _merge_close_bboxes(boxes, gap_x, overlap_y_frac=0.6):
    """Merge bboxes whose x-gap is ≤ gap_x and whose y-overlap is at
    least overlap_y_frac of the shorter bbox height. Used to glue
    fragmented same-colour CCs (e.g. a marker bar broken by a darker
    inner pixel column) into a single bar.

    Each input/output box: (x, y, w, h, area).
    """
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for b in boxes[1:]:
        m = merged[-1]
        # x-gap from m's right edge to b's left edge
        gx = b[0] - (m[0] + m[2])
        # y-overlap fraction (against shorter)
        oy = max(0, min(m[1] + m[3], b[1] + b[3]) - max(m[1], b[1]))
        short = max(1, min(m[3], b[3]))
        if gx <= gap_x and (oy / short) >= overlap_y_frac:
            x0 = min(m[0], b[0])
            y0 = min(m[1], b[1])
            x1 = max(m[0] + m[2], b[0] + b[2])
            y1 = max(m[1] + m[3], b[1] + b[3])
            m[0] = x0
            m[1] = y0
            m[2] = x1 - x0
            m[3] = y1 - y0
            m[4] = m[4] + b[4]  # area sums (approximate)
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


def detect_markers(img, icon_w, icon_h):
    """Detect seat-type markers. Returns list of (x, y, w, h, code).

    Per band:
      1. HSV mask + morph-CLOSE (horizontal kernel ~icon_w*0.3 to glue
         fragmented bars into one CC).
      2. CC analysis → raw bboxes.
      3. _merge_close_bboxes() → merge same-colour CCs that are close
         in x and overlap in y, healing remaining gaps.
      4. Size + aspect + fill-density filter.
      5. Cross-band IoU dedupe (>0.4).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Size: a real marker is roughly icon_w × icon_h. Absolute floors
    # reject tiny noise blobs irrespective of icon-size estimation;
    # icon-relative bounds reject oversized background patches.
    abs_min_w, abs_min_h = 10, 12
    abs_max_w, abs_max_h = 90, 90
    min_w = max(abs_min_w, int(icon_w * 0.45))
    max_w = min(abs_max_w, max(int(icon_w * 1.25), 36))
    min_h = max(abs_min_h, int(icon_h * 0.45))
    max_h = min(abs_max_h, max(int(icon_h * 1.35), 44))
    # Aspect ratio: real bars are roughly square or portrait
    # (w/h ≈ 0.4..1.6). Wide squat rectangles = UI false-positives.
    ar_min, ar_max = 0.30, 1.8
    # Fill density: solid colour blobs only — noise has lower density.
    fill_min = 0.70

    # Horizontal close kernel to glue stripe fragments inside one mask.
    # Keep small — `_merge_close_bboxes` handles real gaps; an aggressive
    # CLOSE merges the marker with adjacent same-hue UI artifacts and
    # bloats the bbox past the size limits.
    kx = max(3, int(round(icon_w * 0.22)))
    ky = max(2, icon_h // 12)

    # Per-channel uniformity thresholds inside masked pixels.
    # Real marker bars have very flat colour: V stddev ~10-20.
    # Icons/UI texture inside the same hue band: V stddev ~35-60.
    uni_v_max = 28
    uni_h_max = 6
    # Edge density via Canny on the bbox INTERIOR (3-px inset to skip
    # the bar's own border). Real bars: ~0% interior edges.
    # Ability icons / textured UI: 15-35%.
    edge_max = 0.07
    edge_inset = 2
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

    # Merge bands with the same name (e.g. red TAC wraps 0/180 across two
    # entries) into one combined mask, otherwise each half captures only
    # part of the marker and fill-density / size filters reject it.
    by_name = {}
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in MAIN_BANDS:
        by_name.setdefault(name, {'code': code, 'ranges': []})['ranges'].append(
            (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi))

    out = []
    seen_rects = []
    for name, info in by_name.items():
        code = info['code']
        m = None
        for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in info['ranges']:
            part = colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
            m = part if m is None else cv2.bitwise_or(m, part)
        # Run CC on BOTH the raw mask and a CLOSE-d copy and process
        # each independently. CLOSE glues genuinely-fragmented bars;
        # the raw mask preserves clean markers that CLOSE would merge
        # with adjacent same-hue UI strips (e.g. dark-red name bars
        # sitting 3-5 px to the right of the marker on Pumwl1).
        # Combining bboxes BEFORE size filtering would let the bloated
        # post-CLOSE blob swallow the clean pre-CLOSE one — instead we
        # keep them separate; the IoU dedupe at the end drops doubles.
        m_closed = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
        )
        gap_x = max(2, int(round(icon_w * 0.35)))
        merged = []
        for src in (m, m_closed):
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                src, connectivity=8)
            raw = [tuple(int(v) for v in stats[i, [0, 1, 2, 3, 4]])
                   for i in range(1, n)]
            merged.extend(_merge_close_bboxes(
                raw, gap_x=gap_x, overlap_y_frac=0.55))

        for x, y, w, h, area in merged:
            if w < min_w or w > max_w:
                continue
            if h < min_h or h > max_h:
                continue
            ar = w / max(h, 1)
            if ar < ar_min or ar > ar_max:
                continue
            if area < (w * h) * fill_min:
                continue
            # Colour-uniformity: real bars are flat colour; icons /
            # textured UI inside the same hue band are not. Compute
            # stddev of V (and H) over masked pixels inside the CC bbox.
            sel = m[y:y + h, x:x + w] > 0
            if sel.sum() < 20:
                continue
            crop_v = hsv[y:y + h, x:x + w, 2][sel]
            crop_h = hsv[y:y + h, x:x + w, 0][sel]
            if float(np.std(crop_v)) > uni_v_max:
                continue
            # Hue is angular; for the red bands that wrap 0/180 a naive
            # std over raw H is huge. Use the smaller of {std(H),
            # std((H+90)%180)} so red wrap is handled.
            h_std = min(
                float(np.std(crop_h)),
                float(np.std((crop_h.astype(np.int32) + 90) % 180)),
            )
            if h_std > uni_h_max:
                continue
            # Edge density inside the bbox interior (skip border pixels
            # of the bar itself).
            ix0 = x + edge_inset
            iy0 = y + edge_inset
            ix1 = x + w - edge_inset
            iy1 = y + h - edge_inset
            if ix1 > ix0 and iy1 > iy0:
                edge_crop = edges[iy0:iy1, ix0:ix1]
                edge_frac = float(edge_crop.sum() / 255.0) / (
                    (ix1 - ix0) * (iy1 - iy0))
                if edge_frac > edge_max:
                    continue
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

    # Size-outlier filter: drop markers whose w or h is < 65% of the
    # median across all detections. Real BOFF markers within a screen
    # are nearly identical in size; tiny detections are red-coloured
    # UI artifacts (e.g. empire's 13×18 blob at top-right).
    if len(out) >= 4:
        med_w = sorted(m[2] for m in out)[len(out) // 2]
        med_h = sorted(m[3] for m in out)[len(out) // 2]
        out = [m for m in out
               if m[2] >= 0.65 * med_w and m[3] >= 0.65 * med_h]
    return out


def classify_stripe(hsv, marker, icon_w):
    """Identify the specialization stripe on the right edge of a marker.

    Samples the rightmost ~25% of the marker bar and counts pixels per
    stripe band. Returns (spec_code, score) or (None, 0) if no band
    dominates. Score = fraction of right-edge pixels matching the band.
    """
    x, y, w, h, _ = marker
    # The main-zone CC bbox is 75-95% of the bar; the spec stripe is
    # the remaining 5-25% on the right. We sample starting just before
    # the CC's right edge (in case the CC over-extended) and continuing
    # past it by a fraction of the CC width — this scales to whatever
    # bar size we have, rather than relying on an estimated icon_w.
    sx0 = x + w - max(2, int(w * 0.10))
    sx1 = x + w + max(6, int(w * 0.40))
    sy0 = y + max(0, int(h * 0.10))
    sy1 = y + h - max(0, int(h * 0.10))
    H, W = hsv.shape[:2]
    sx0 = max(0, sx0); sx1 = min(W, sx1)
    sy0 = max(0, sy0); sy1 = min(H, sy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return (None, 0.0)
    crop = hsv[sy0:sy1, sx0:sx1]
    total = crop.shape[0] * crop.shape[1]
    if total <= 0:
        return (None, 0.0)

    best_code = None
    best_count = 0
    for _, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in STRIPE_BANDS:
        m = ((crop[:, :, 0] >= h_lo) & (crop[:, :, 0] <= h_hi)
             & (crop[:, :, 1] >= s_lo) & (crop[:, :, 1] <= s_hi)
             & (crop[:, :, 2] >= v_lo) & (crop[:, :, 2] <= v_hi))
        c = int(m.sum())
        if c > best_count:
            best_count = c
            best_code = code
    score = best_count / total
    # Require ≥15% of the strip to match — below that it's likely noise.
    if score < 0.15:
        return (None, score)
    return (best_code, score)


def stripe_band_scores(hsv, marker):
    """Per-band fractions for the spec-stripe sample region of one marker.

    Returns: (sample_rect, [(code, fraction), ...]) where sample_rect
    is (sx0, sy0, sx1, sy1) in image coords.
    """
    x, y, w, h, _ = marker
    sx0 = x + w - max(2, int(w * 0.10))
    sx1 = x + w + max(6, int(w * 0.40))
    sy0 = y + max(0, int(h * 0.10))
    sy1 = y + h - max(0, int(h * 0.10))
    H, W = hsv.shape[:2]
    sx0 = max(0, sx0); sx1 = min(W, sx1)
    sy0 = max(0, sy0); sy1 = min(H, sy1)
    rect = (sx0, sy0, sx1, sy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return rect, []
    crop = hsv[sy0:sy1, sx0:sx1]
    total = crop.shape[0] * crop.shape[1]
    if total <= 0:
        return rect, []
    out = []
    for _, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in STRIPE_BANDS:
        m = ((crop[:, :, 0] >= h_lo) & (crop[:, :, 0] <= h_hi)
             & (crop[:, :, 1] >= s_lo) & (crop[:, :, 1] <= s_hi)
             & (crop[:, :, 2] >= v_lo) & (crop[:, :, 2] <= v_hi))
        out.append((code, int(m.sum()) / total))
    return rect, out


CODE_LABEL = {
    'T': 'Tac', 'E': 'Eng', 'S': 'Sci', 'U': 'Uni',
    'O': 'Cmd', 'P': 'Int', 'Y': 'Temp', 'C': 'Pilot', 'L': 'MW',
}


def viz_spec_inspector(fname, img, panel):
    """Save a per-screen inspector that shows each panel marker zoomed
    4× with the spec-stripe sample region highlighted, plus per-band
    fractions and median HSV of the sampled pixels.
    """
    if panel is None:
        return
    a, b, _ = panel
    all_m = a + b
    if not all_m:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]
    ZOOM = 4
    PAD = 4

    tiles = []
    for x, y, w, h, code in all_m:
        rect, bands = stripe_band_scores(hsv, (x, y, w, h, code))
        sx0, sy0, sx1, sy1 = rect
        cx0 = max(0, x - PAD)
        cy0 = max(0, y - PAD)
        cx1 = min(W, sx1 + PAD)
        cy1 = min(H, y + h + PAD)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        zoom = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM,
                          interpolation=cv2.INTER_NEAREST)
        # Sample region
        zx0 = (sx0 - cx0) * ZOOM
        zy0 = (sy0 - cy0) * ZOOM
        zx1 = (sx1 - cx0) * ZOOM
        zy1 = (sy1 - cy0) * ZOOM
        cv2.rectangle(zoom, (zx0, zy0), (zx1 - 1, zy1 - 1), (0, 255, 0), 1)
        # Marker bbox outline
        mx0 = (x - cx0) * ZOOM
        my0 = (y - cy0) * ZOOM
        mx1 = mx0 + w * ZOOM
        my1 = my0 + h * ZOOM
        cv2.rectangle(zoom, (mx0, my0), (mx1 - 1, my1 - 1), (255, 0, 255), 1)

        bands_sorted = sorted(bands, key=lambda r: -r[1])
        best_code, best_score = bands_sorted[0] if bands_sorted else ('—', 0.0)
        accepted_lbl = (CODE_LABEL.get(best_code, best_code)
                        if best_score >= 0.15 else '—')
        seat_lbl = CODE_LABEL.get(code, code)

        sample = hsv[sy0:sy1, sx0:sx1].reshape(-1, 3) if sx1 > sx0 else None
        if sample is not None and len(sample):
            med_h = int(np.median(sample[:, 0]))
            med_s = int(np.median(sample[:, 1]))
            med_v = int(np.median(sample[:, 2]))
            hsv_str = f'med HSV {med_h:>3d},{med_s:>3d},{med_v:>3d}'
        else:
            hsv_str = 'med HSV —'

        text_lines = [
            f'{seat_lbl} -> {accepted_lbl}  ({best_score:.2f})',
            hsv_str,
            ' '.join(f'{CODE_LABEL.get(c, c)}:{s:.2f}'
                     for c, s in bands_sorted[:3]),
        ]

        line_h = 16
        text_h = line_h * len(text_lines) + 6
        tile_w = max(zoom.shape[1], 200)
        tile_h = zoom.shape[0] + text_h
        tile = np.full((tile_h, tile_w, 3), 25, dtype=np.uint8)
        x_off = (tile_w - zoom.shape[1]) // 2
        tile[:zoom.shape[0], x_off:x_off + zoom.shape[1]] = zoom
        for i, ln in enumerate(text_lines):
            cv2.putText(tile, ln, (4, zoom.shape[0] + 14 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                        cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        return
    max_h = max(t.shape[0] for t in tiles)
    sep = np.full((max_h, 8, 3), 60, dtype=np.uint8)
    padded = []
    for i, t in enumerate(tiles):
        if t.shape[0] < max_h:
            pad = np.full((max_h - t.shape[0], t.shape[1], 3), 25,
                          dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)
        if i < len(tiles) - 1:
            padded.append(sep)
    canvas = np.hstack(padded)
    header = np.full((28, canvas.shape[1], 3), 50, dtype=np.uint8)
    cv2.putText(header, fname, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([header, canvas])

    OUT_INSPECTOR.mkdir(parents=True, exist_ok=True)
    out_name = fname.replace('/', '_')
    if not out_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        out_name += '.png'
    cv2.imwrite(str(OUT_INSPECTOR / out_name), canvas)


def full_bar_extent(hsv, marker):
    """Find the rightmost x where the colour bar (main zone OR spec
    stripe) still has a strong column-fill, starting from the marker
    bbox. Returns (full_w_including_stripe, has_spec, stripe_width).

    Only runs the extension scan if `classify_stripe` already detected
    a spec on this marker — otherwise ability icons just to the right
    of the marker frequently match some HSV band and inflate full_w.
    """
    x, y, w, h, code = marker
    spec, _score = classify_stripe(hsv, marker, w)
    if spec is None:
        return (w, False, 0)
    H_im, W_im = hsv.shape[:2]
    # Stripe is ≤30% of full bar in practice — keep look-ahead tight.
    look_ahead = max(4, int(round(w * 0.35)))
    sx0 = max(0, x)
    sx1 = min(W_im, x + w + look_ahead)
    sy0 = max(0, y + max(0, int(h * 0.15)))
    sy1 = min(H_im, y + h - max(0, int(h * 0.15)))
    if sy1 <= sy0 or sx1 <= sx0:
        return (w, False, 0)
    crop = hsv[sy0:sy1, sx0:sx1]
    H = crop[:, :, 0]
    S = crop[:, :, 1]
    V = crop[:, :, 2]
    main_m = np.zeros(crop.shape[:2], dtype=bool)
    for _name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _c in MAIN_BANDS:
        main_m |= ((H >= h_lo) & (H <= h_hi)
                   & (S >= s_lo) & (S <= s_hi)
                   & (V >= v_lo) & (V <= v_hi))
    stripe_m = np.zeros(crop.shape[:2], dtype=bool)
    for _name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _c in STRIPE_BANDS:
        stripe_m |= ((H >= h_lo) & (H <= h_hi)
                     & (S >= s_lo) & (S <= s_hi)
                     & (V >= v_lo) & (V <= v_hi))
    full_m = main_m | stripe_m
    # Column fill fraction in the marker y-range
    col_frac = full_m.mean(axis=0)
    # Walk right from end of main zone. The +spec stripe sits past a
    # short dark gap (~2-4 dim cols) — so we skip up to 6 dim cols
    # before requiring a bright run. We then take the END of THAT run
    # only — beyond which there's another gap and then the seat name
    # bar (which has the same profession hue as the marker and would
    # otherwise be grabbed by a "rightmost ≥35% fill" rule).
    main_end_in_crop = (x + w) - sx0
    last_col = main_end_in_crop - 1
    n_cols = col_frac.shape[0]
    ci = main_end_in_crop
    # Phase 1: skip up to 6 dim cols (the gap between main and stripe).
    while (ci < n_cols
           and col_frac[ci] < 0.35
           and ci - main_end_in_crop < 6):
        ci += 1
    # Phase 2: extend through contiguous bright cols (the stripe itself).
    while ci < n_cols and col_frac[ci] >= 0.35:
        last_col = ci
        ci += 1
    full_w = max(w, last_col + 1)
    has_spec = full_w > w + 1
    stripe_w = full_w - w if has_spec else 0
    return (full_w, has_spec, stripe_w)


# Bible — measured on Stations.png ("idealny rozkład"). All values are
# in panel-internal pixels (i.e. how many image pixels the panel UI
# would occupy at the reference scale where one full marker bar is 29
# pixels wide). At any other resolution we just rescale by a single
# factor `k = detected_marker_w / _BIBLE_MARKER_W`.
_BIBLE_MARKER_W   = 29
_BIBLE_SLOT_W     = 29
_BIBLE_SLOT_H     = 37
_BIBLE_GAP_FIRST  = 3      # marker_right_edge → slot1_left_edge (X)
_BIBLE_GAP_SLOT   = 2      # slot_right_edge   → next_slot_left   (X)
_BIBLE_STRIDE_X   = _BIBLE_SLOT_W + _BIBLE_GAP_SLOT   # = 31


def project_seat_slots(panel, n_abilities=4, hsv=None):
    """Project 4 ability-icon bboxes per detected seat marker.

    Geometry comes from the bible (panel-internal pixels). Detection
    contributes a single scale factor `k = full_marker_w / 29`, so
    every grid distance scales coherently — slot 4 cannot drift away
    from slot 1 just because `med_marker_w` was a pixel low.

    X-axis (per column):
      anchor   = max(marker_x + full_bar_extent) over markers in column
      slot_x[i] = anchor + k * (BIBLE_GAP_FIRST + i * BIBLE_STRIDE_X)
      slot_w   = round(k * BIBLE_SLOT_W)
      slot_h   = round(k * BIBLE_SLOT_H)

    Y-axis is left as the empirical formula (`ab_h / 0.63`-style was
    already 98.4% IoU≥.5; not touching what works). Vertical bible
    constants (51 px row-to-row) were not used because the marker
    height itself is not specified in the bible.

    Returns list of (seat_idx, slot_idx, x, y, w, h, seat_code).
    """
    if panel is None:
        return []
    a, b, _ = panel
    all_m = a + b
    if not all_m:
        return []

    # Scale factor — prefer full bar width (constant 29 panel-px both
    # for plain and +spec markers); fall back to raw marker.w when no
    # HSV is supplied.
    if hsv is not None:
        widths = sorted(full_bar_extent(hsv, m)[0] for m in all_m)
    else:
        widths = sorted(m[2] for m in all_m)
    det_w = widths[len(widths) // 2]
    k = det_w / _BIBLE_MARKER_W

    # Float-domain bible distances. We do NOT round any intermediate
    # value — all bible constants are kept as floats and the only
    # rounding happens once per slot at the final pixel position. This
    # eliminates cumulative error from `i * round(stride)` and visual
    # gap alternation from independent rounding of slot_w vs stride.
    stride_f    = k * _BIBLE_STRIDE_X    # = k * (29 + 2)
    gap_first_f = k * _BIBLE_GAP_FIRST   # = k * 3
    ab_w        = max(1, int(round(k * _BIBLE_SLOT_W)))

    # Y axis — keep the proven empirical formula.
    med_h    = sorted(m[3] for m in all_m)[len(all_m) // 2]
    ab_h     = max(1, int(round(med_h / 0.63)))
    gap_y    = int(round(med_h * 0.20))

    def col_anchor(col):
        if hsv is None:
            return max(m[0] + m[2] for m in col)
        return max(m[0] + full_bar_extent(hsv, m)[0] for m in col)
    a_anchor = col_anchor(a) if a else 0
    b_anchor = col_anchor(b) if b else 0

    out = []
    for seat_idx, m in enumerate(all_m):
        mx, my, mw, mh, code = m
        slot_y = my - ab_h - gap_y
        anchor = a_anchor if seat_idx < len(a) else b_anchor
        slot_x0_f = anchor + gap_first_f
        for k_idx in range(n_abilities):
            x = int(round(slot_x0_f + k_idx * stride_f))
            out.append((seat_idx, k_idx, x, slot_y, ab_w, ab_h, code))
    return out


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def annotate_specs(img, markers, icon_w):
    """Return [(x, y, w, h, seat_code, spec_code_or_None, score)]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for x, y, w, h, code in markers:
        spec, score = classify_stripe(hsv, (x, y, w, h, code), icon_w)
        out.append((x, y, w, h, code, spec, round(score, 3)))
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
    """Cluster GT BOFF abilities into seats (one seat = 4 abilities in
    one (col, row) cell). Uses tolerance-based 1-D clustering on cy
    rather than `round(cy/icon_h)` — the latter splits a single row
    into two ykeys when cy falls on a rounding boundary (e.g.
    cy=285 with icon_h=38 → 7.5, banker's rounding picks 7 or 8
    depending on the last digit of cy)."""
    if not boffs:
        return []
    cxs = sorted(b[0] + b[2] / 2 for b in boffs)
    if len(cxs) < 2:
        return []
    gaps = [(cxs[i + 1] - cxs[i], i) for i in range(len(cxs) - 1)]
    _, gi = max(gaps)
    split_x = (cxs[gi] + cxs[gi + 1]) / 2

    # Group by column first, then cluster cy with tolerance 0.6·icon_h.
    cols = {'L': [], 'R': []}
    for b in boffs:
        cx = b[0] + b[2] / 2
        cy = b[1] + b[3] / 2
        col = 'L' if cx < split_x else 'R'
        cols[col].append((b[0], cy))

    tol = max(1.0, 0.6 * icon_h)
    out = []
    for col, members in cols.items():
        if not members:
            continue
        members.sort(key=lambda m: m[1])
        cluster = [members[0]]
        for m in members[1:]:
            if m[1] - cluster[-1][1] <= tol:
                cluster.append(m)
            else:
                out.append((col, st.median(c[1] for c in cluster),
                            min(c[0] for c in cluster)))
                cluster = [m]
        out.append((col, st.median(c[1] for c in cluster),
                    min(c[0] for c in cluster)))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def evaluate_screen(fname, img, boffs, gt_full=None):
    """gt_full: optional [(bbox, name)] including __empty__/__inactive__."""
    h, w = img.shape[:2]
    icon_w, icon_h = estimate_icon_dims(img)
    markers = detect_markers(img, icon_w, icon_h)
    # Re-derive icon dims from actual marker sizes before panel anchor:
    # marker_w ≈ 0.85·icon_w, marker_h ≈ 0.80·icon_h. Heuristic estimate
    # is often 30-50% too small when image is small/cropped.
    if len(markers) >= 3:
        ws = sorted(m[2] for m in markers)
        hs = sorted(m[3] for m in markers)
        med_mw = ws[len(ws) // 2]
        med_mh = hs[len(hs) // 2]
        icon_w = max(icon_w, int(round(med_mw / 0.85)))
        icon_h = max(icon_h, int(round(med_mh / 0.80)))
    panel = best_panel(markers, icon_w, icon_h)
    annotated = annotate_specs(img, markers, icon_w)
    n_specced = sum(1 for *_, sp, _s in annotated if sp is not None)

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

    # --- Per-slot projection + IoU vs GT (incl. virtual slots) ---
    hsv_eval = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    proj_slots = project_seat_slots(panel, hsv=hsv_eval)
    slot_eval = {
        'projected':   len(proj_slots),
        'gt_total':    len(gt_full) if gt_full is not None else 0,
        'iou30':       0,
        'iou50':       0,
        'iou70':       0,
        'mean_iou':    0.0,
        'matched_real':     0,
        'matched_empty':    0,
        'matched_inactive': 0,
        'gt_unmatched':     0,
    }
    if gt_full and proj_slots:
        used_gt = set()
        ious = []
        for (_si, _sk, x, y, sw, sh, _c) in proj_slots:
            best_i = -1; best_iou = 0.0
            for gi, (gb, _gn) in enumerate(gt_full):
                if gi in used_gt:
                    continue
                iou = _bbox_iou((x, y, sw, sh), gb)
                if iou > best_iou:
                    best_iou = iou; best_i = gi
            ious.append(best_iou)
            if best_iou >= 0.30 and best_i >= 0:
                used_gt.add(best_i)
                slot_eval['iou30'] += 1
                if best_iou >= 0.50:
                    slot_eval['iou50'] += 1
                if best_iou >= 0.70:
                    slot_eval['iou70'] += 1
                gn = gt_full[best_i][1]
                if gn == '__empty__':
                    slot_eval['matched_empty'] += 1
                elif gn == '__inactive__':
                    slot_eval['matched_inactive'] += 1
                else:
                    slot_eval['matched_real'] += 1
        slot_eval['mean_iou'] = round(
            sum(ious) / max(len(ious), 1), 3)
        slot_eval['gt_unmatched'] = len(gt_full) - len(used_gt)

    return {
        'file':       fname,
        'iw':         w,
        'ih':         h,
        'icon_w':     icon_w,
        'icon_h':     icon_h,
        'n_markers':  len(markers),
        'n_specced':  n_specced,
        'gt_seats':   len(gt_seats),
        'seat_hits':  seat_hit,
        'hit_pct':    round(seat_hit / max(len(gt_seats), 1) * 100, 1),
        'panel':      panel_info,
        'panel_ok':   panel_in_gt,
        'markers':    [list(m) for m in markers],
        'annotated':  [list(m) for m in annotated],
        'slots':      [list(s) for s in proj_slots],
        'slot_eval':  slot_eval,
    }


def viz(fname, img, markers, panel, boffs, gt_full=None):
    out = img.copy()
    code_color = {
        # Seat-type main zone
        'T': (0,   0,   255),  # red       — Tactical
        'E': (0,   200, 255),  # gold      — Engineering
        'S': (255, 100, 0),    # blue      — Science
        'U': (200, 255, 255),  # cream     — Universal
        # Spec stripe
        'O': (0,   140, 220),  # orange    — Command
        'P': (200, 0,   200),  # purple    — Intelligence
        'Y': (0,   220, 220),  # bright Y  — Temporal
        'C': (255, 220, 100),  # cyan      — Pilot
        'L': (50,  220, 0),    # lime      — Miracle Worker
    }
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for x, y, w, h, code in markers:
        c = code_color.get(code, (255, 255, 255))
        full_w, has_spec, _sw = full_bar_extent(hsv_full, (x, y, w, h, code))
        if has_spec:
            # Single combined frame: main + stripe (magenta border)
            cv2.rectangle(out, (x, y), (x + full_w, y + h),
                          (255, 0, 255), 1)
            # Faint inner frame to indicate where main zone ends
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 1)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 1)
    if panel is not None:
        a, b, _ = panel
        all_m = a + b
        # Envelope anchored on FULL bar (main + spec stripe) end.
        # Per column: x = max(marker.x + full_bar_w) — plain markers
        # have full_w == w; +spec markers have full_w > w. Taking max
        # canonicalises both kinds to the same bar-end x.
        med_w = sorted(m[2] for m in all_m)[len(all_m) // 2]
        med_h = sorted(m[3] for m in all_m)[len(all_m) // 2]
        ab_w = int(round(med_w / 0.88))
        ab_h = int(round(med_h / 0.80))
        x0 = max(m[0] + full_bar_extent(hsv_full, m)[0] for m in a)
        y0 = min(m[1] for m in all_m) - int(round(ab_h * 1.6))
        x1 = max(m[0] + full_bar_extent(hsv_full, m)[0] for m in b) + 4 * ab_w
        y1 = max(m[1] for m in all_m)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 1)
    # Projected ability slots (cyan, thin).
    PROJ_COL = (255, 200, 0)
    proj = project_seat_slots(panel, hsv=hsv_full)
    for _si, _sk, sx, sy, sw, sh, _c in proj:
        cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), PROJ_COL, 1)
    # GT individual slots — only real items (white). Virtual slots
    # (__empty__/__inactive__) are skipped: stats in slot_eval already
    # report how many were covered by projections.
    if gt_full:
        for (gb, gn) in gt_full:
            if gn in VIRTUAL:
                continue
            gx, gy, gw, gh = gb
            cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh),
                          (255, 255, 255), 1)
    elif boffs:
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
        # Include empty / inactive slots — they are real BOFF positions
        # and define the full panel envelope (3+2 seats × 4 abilities).
        boffs = [a['bbox'] for a in entries
                 if a.get('state') == 'confirmed'
                 and a.get('slot', '').startswith('Boff ')
                 and a.get('bbox')]
        # Full GT incl. virtual-slot tags for IoU-by-state breakdown.
        gt_full = [(a['bbox'], a.get('name', ''))
                   for a in entries
                   if a.get('state') == 'confirmed'
                   and a.get('slot', '').startswith('Boff ')
                   and a.get('bbox')]
        # Count "real" (non-virtual) for the seat threshold.
        n_real = sum(1 for a in entries
                     if a.get('state') == 'confirmed'
                     and a.get('slot', '').startswith('Boff ')
                     and a.get('name') not in VIRTUAL
                     and a.get('bbox'))
        if n_real >= 5:
            candidates.append((fname, boffs, gt_full))

    paths = find_screen_files([c[0] for c in candidates])
    print(f'Screens with ≥5 BOFF GT: {len(candidates)}, '
          f'findable: {len(paths)}')

    rows = []
    total_seats = 0
    total_hits = 0
    panel_ok_count = 0
    for fname, boffs, gt_full in candidates:
        if fname not in paths:
            continue
        img = cv2.imread(str(paths[fname]))
        if img is None:
            continue
        m = evaluate_screen(fname, img, boffs, gt_full=gt_full)
        rows.append(m)
        total_seats += m['gt_seats']
        total_hits  += m['seat_hits']
        if m['panel_ok']:
            panel_ok_count += 1

        # Reproduce same dims for viz (mirror evaluate_screen refine).
        iw0, ih0 = estimate_icon_dims(img)
        markers_full = detect_markers(img, iw0, ih0)
        if len(markers_full) >= 3:
            ws_ = sorted(mk[2] for mk in markers_full)
            hs_ = sorted(mk[3] for mk in markers_full)
            iw0 = max(iw0, int(round(ws_[len(ws_) // 2] / 0.85)))
            ih0 = max(ih0, int(round(hs_[len(hs_) // 2] / 0.80)))
        panel = best_panel(markers_full, iw0, ih0)
        viz(fname, img, markers_full, panel, boffs, gt_full=gt_full)
        viz_spec_inspector(fname, img, panel)

    rows.sort(key=lambda r: (r['panel_ok'], r['hit_pct']))
    print('\nPer-screen results (sorted by panel_ok, hit %):')
    for r in rows:
        panel_ok = '✓' if r['panel_ok'] else ('?' if r['panel'] else '✗')
        se = r['slot_eval']
        slot_pct = (se['iou30'] / max(se['gt_total'], 1) * 100)
        print(f'  {r["file"][:38]:<38s}  '
              f'seats={r["seat_hits"]}/{r["gt_seats"]} '
              f'({r["hit_pct"]:5.1f}%)  '
              f'mks={r["n_markers"]:3d} '
              f'spec={r["n_specced"]:2d}  '
              f'panel={panel_ok}  '
              f'slots IoU≥.3={se["iou30"]}/{se["gt_total"]} '
              f'({slot_pct:5.1f}%)  '
              f'mIoU={se["mean_iou"]:.2f}')

    # Spec-code histogram across all detected markers
    spec_hist = {}
    for r in rows:
        for *_, sp, _s in r['annotated']:
            if sp:
                spec_hist[sp] = spec_hist.get(sp, 0) + 1

    # Slot-projection aggregates
    slot_iou30  = sum(r['slot_eval']['iou30']  for r in rows)
    slot_iou50  = sum(r['slot_eval']['iou50']  for r in rows)
    slot_iou70  = sum(r['slot_eval']['iou70']  for r in rows)
    slot_total  = sum(r['slot_eval']['gt_total']  for r in rows)
    slot_proj   = sum(r['slot_eval']['projected'] for r in rows)
    m_real      = sum(r['slot_eval']['matched_real']     for r in rows)
    m_empty     = sum(r['slot_eval']['matched_empty']    for r in rows)
    m_inactive  = sum(r['slot_eval']['matched_inactive'] for r in rows)

    print('\n' + '=' * 80)
    print(f'Total seats hit: {total_hits}/{total_seats} '
          f'({total_hits / max(total_seats, 1) * 100:.1f}%)')
    print(f'Panel anchored on GT: {panel_ok_count}/{len(rows)} '
          f'({panel_ok_count / max(len(rows), 1) * 100:.1f}%)')
    print(f'Mean markers detected: {st.mean(r["n_markers"] for r in rows):.1f}')
    print(f'Median markers detected: {st.median(r["n_markers"] for r in rows):.1f}')
    print(f'Markers with spec stripe identified: '
          f'{sum(r["n_specced"] for r in rows)}  '
          f'({", ".join(f"{c}:{n}" for c, n in sorted(spec_hist.items()))})')
    print()
    print('Ability slot projection (IoU vs GT bbox, per-state):')
    print(f'  Projected slots:   {slot_proj}')
    print(f'  GT slots:          {slot_total}')
    print(f'  IoU ≥ 0.30:        {slot_iou30} '
          f'({slot_iou30 / max(slot_total, 1) * 100:.1f}% of GT)')
    print(f'  IoU ≥ 0.50:        {slot_iou50} '
          f'({slot_iou50 / max(slot_total, 1) * 100:.1f}%)')
    print(f'  IoU ≥ 0.70:        {slot_iou70} '
          f'({slot_iou70 / max(slot_total, 1) * 100:.1f}%)')
    print(f'  Matched (state):   real={m_real}  empty={m_empty}  '
          f'inactive={m_inactive}')
    print(f'Visualisations:    {OUT_VIZ}')
    print(f'Spec inspector:    {OUT_INSPECTOR}')

    OUT_JSON.write_text(json.dumps({'screens': rows}, indent=2),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
