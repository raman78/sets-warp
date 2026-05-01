# BOFF panel detection

Prototype: `tests/diag_boff_markers.py`. The detector localises the BOFF
panel on STO screenshots by finding the small profession-coloured bars
that sit on the LEFT of every seat's name strip (just below the row of
ability icons). It does NOT depend on the icons themselves, so it is
robust to empty/inactive slots and to text-rendering changes.

Latest baseline (36 GT screens, 185 seats): **98.9% seat hit, 100%
panel anchor**, 60/167 markers also enriched with a specialization
code via the post-hoc stripe classifier.

Per-seat ability-slot projection from those markers (4 abilities/seat):
**99.9% IoU≥.30, 99.9% IoU≥.50, 96.0% IoU≥.70** vs ground truth on 705
GT slots. Content classification of the projected slots via the
production `SETSIconMatcher.classify_patch()` (EfficientNet-B0):
**99.8% bucket accuracy on real abilities, 94.9% exact-name**, but
`__inactive__` is misclassified as `__empty__` ~96% of the time
(see "Slot content classification" below). Diagnostic:
`tests/diag_boff_classify.py`.

## Why colour markers (not OCR / icon templates)

Every BOFF seat — empty, inactive, locked, or fully populated — has a
solid-coloured profession bar on the LEFT of its name strip. The bar
is the most visually stable feature of the panel:

- present at every tier (Ensign … Commander), every panel size
- position within the panel is geometrically constrained (2 columns,
  ≤3 seats left + ≤2 right for the canonical 3+2 layout)
- not affected by missing icons, unreadable ability text, theme tint,
  or background art
- distinguishes profession (Tac/Eng/Sci/Uni) by hue alone, with a
  narrow specialization stripe on the right edge for spec seats
  (Cmd/Int/Temp/Pilot/MW)

OCR alone is unreliable on STO's stylised rank labels (0 keyword hits
on Ambassador-broadside / image.png — see `boff_detection_2026-04-25`
research). Icon-template matching has its own purpose (item ID inside
each slot) but cannot anchor the panel.

## Pipeline

```
img ─► estimate_icon_dims ─► detect_markers ─► best_panel ─► annotate_specs
                              │                  │             │
                              ▼                  ▼             ▼
                        list[(x,y,w,h,code)]  (col_a, col_b)  add spec
                                                              code+score
```

### 1. Icon-dim estimate

`estimate_icon_dims(img)` returns a coarse `(icon_w, icon_h)` based on
image size, covering both full-screen captures (~4.7% of h) and
panel-only crops (~11% of h). After detection, both dims are refined
from the median marker size (`marker_h / 0.80 ≈ icon_h`,
`marker_w / 0.88 ≈ icon_w`).

### 2. `detect_markers`

For each main band in `MAIN_BANDS`:

1. **Build colour mask** with `cv2.inRange`. Bands sharing the same
   profession name (TAC has two: `H 0-6` + `H 174-180`) are combined
   with `bitwise_or` *before* connected-component analysis. Splitting
   them captures only ~50% of the marker pixels and breaks fill
   density.
2. **Dual-mask CC**: run `connectedComponentsWithStats` on BOTH
   - the raw mask, and
   - a morph-CLOSE'd copy (`kx ≈ 0.22·icon_w, ky ≈ icon_h//12`).

   The raw mask preserves clean markers that CLOSE would merge with
   adjacent same-hue UI strips (e.g. dark-red name bars sitting 3-5 px
   to the right of the marker on Pumwl1). The CLOSE-d mask glues
   genuinely fragmented bars. Each mask is processed independently;
   `_merge_close_bboxes` runs per mask to coalesce any fragments
   *within* that mask only.
3. **Filter** every CC bbox:
    - size: `min_w=10, min_h=12`; relative `[0.45..1.25]·icon_w`,
      `[0.45..1.35]·icon_h`, capped at `[10..36] × [12..44]`
    - aspect ratio in `[0.30, 1.8]`
    - **fill density ≥ 0.70** of mask pixels inside the bbox
      (rejects FP markers that contain UI decorations like yellow
      arrows — those have ~33% non-mask pixels)
    - colour uniformity on red-only pixels: V std ≤ 28; H std ≤ 6
      with wrap-aware `min(std(H), std((H+90) mod 180))`
    - **Canny edge density on bbox interior (2-px inset) ≤ 0.07**
      (rejects textured icons inside the same hue band; real markers
      ≤ 0.068, FPs with internal triangles ≥ 0.09)
4. **Cross-band IoU dedupe**: overlap >0.4 → keep first.
5. **Size-outlier filter**: after all bands, drop any marker whose
   `w` or `h` is < 65% of the median `w/h` across detections (cuts
   tiny noise blobs that pass band filters but don't match the panel).

### 3. `best_panel` — RANSAC 3+2 grid

For every pair of detected markers `(m_i, m_j)` in the same row
(similar y, dx in `[3..9]·icon_w`):

1. Sweep five `pitch_y` candidates (`1.6, 2.0, 2.4, 3.0, 3.6 × icon_h`).
2. For each candidate, collect inliers per row index in both columns
   with `y_tol = 0.30·pitch_y`.
3. Dedupe by row index (keep one marker per row per column).
4. Score:
    - aligned-rows count
    - canonical n-total: `{5: 1.5, 4: 1.0, 3: 0.4, 2: 0.0, 6: 0.6}`
    - profession diversity: `≥2 codes → +0.6, else −0.8`
    - bigger column on the LEFT: `+0.3` (canonical 3 left + 2 right)
    - pitch-y consistency on the larger column

The pair with the highest score wins. Output is `(col_a, col_b, score)`
with markers in row order (top to bottom).

### 4. Spec stripe (`classify_stripe` / `annotate_specs`)

Post-hoc per-marker classifier. Samples a thin strip at the right
edge of each marker (10% inside the bbox + up to 40% past the right
edge — scales with marker width). For each `STRIPE_BANDS` entry,
counts pixels passing the `inRange` test. The band with the highest
count wins if its score (≥15% of strip pixels) clears the threshold;
otherwise marker has no spec. Verified against user-labelled GT on
60+ markers across 36 screens.

`full_bar_extent(hsv, marker)` extends the marker bbox to include the
spec stripe. Walking right from `marker.x + marker.w`:

1. **Phase 1** — skip up to 6 dim cols (`col_fill < 0.35`). This is
   the gap between the main coloured zone and the spec stripe (~2-4
   dim cols in practice).
2. **Phase 2** — extend through the contiguous bright run (the stripe
   itself), then stop on the next dim col.

Stopping at the **end of the first bright run after the gap** is
important: the seat name bar starts a few pixels further right and
shares the same profession hue as the marker, so a "rightmost ≥35%
fill in look-ahead" rule would grab the name bar's start and inflate
`full_w` by 5-10 px on Tac/Eng panels. That, in turn, would drag the
column anchor right by the same amount, shifting the projected slot
column.

### 5. Per-seat ability-slot projection (`project_seat_slots`)

Markers tell us WHERE each seat is; we project the four ability bboxes
above each marker geometrically. The X-axis grid is **bible-driven** —
all distances are measured once on a reference screen (Stations.png,
"idealny rozkład") in panel-internal pixels and rescaled by a single
factor at any other resolution.

```
# Bible (panel-internal px, reference scale where marker_bar_w == 29):
BIBLE_MARKER_W   = 29
BIBLE_SLOT_W     = 29   # ability icon width
BIBLE_GAP_FIRST  = 3    # marker right-edge → first slot left-edge
BIBLE_GAP_SLOT   = 2    # slot right-edge   → next slot left-edge
BIBLE_STRIDE_X   = BIBLE_SLOT_W + BIBLE_GAP_SLOT   # = 31

k           = full_bar_w_median / BIBLE_MARKER_W   # single scale factor

# Float-domain bible distances. NO intermediate rounding.
stride_f    = k * BIBLE_STRIDE_X
gap_first_f = k * BIBLE_GAP_FIRST
ab_w        = round(k * BIBLE_SLOT_W)              # rounded once

slot_x[i]   = round(col_anchor + gap_first_f + i * stride_f)   # round once per slot
slot_w      = ab_w
slot_h      = round(med_marker_h / 0.63)           # empirical Y
slot_y      = marker.y - slot_h - round(0.20 * med_marker_h)
```

**Empirical scale validation** (36 GT screens, `tests/diag_marker_scale.py`):
detected `marker_w` clusters in 11 discrete buckets between 24 and
43 px while `marker_w / image_w` varies 5×. STO renders UI at a
**user-configurable discrete scale** (set in-game), independent of
screen resolution. Per-screen `k = marker_w / 29` is the only scale
input from detection.

**No cumulative rounding:** every slot position is computed from the
column anchor in float (`anchor + gap_first_f + i * stride_f`), then
rounded once. This eliminates both cumulative drift (you'd get from
`i * round(stride)`) and visual gap alternation (you'd get from
`stride_x = round(slot_w) + round(gap_slot)` rounding independently).
Visual gap then matches what STO+JPG actually renders — sometimes
2 px, sometimes 1 px depending on sub-pixel position, exactly as the
game does.

`col_anchor` is the rightmost full-bar edge across all markers in the
seat's column. "Full bar" includes the spec stripe (see
`full_bar_extent`) so plain markers and `+spec` markers in the same
column share one anchor — a `+spec` marker has a **shorter** detected
MAIN zone (HSV bands match only the dominant solid colour, not the
stripe), and using each marker's own right edge would shift
plain-marker rows leftwards relative to spec-marker rows. The chosen
anchor is the **max** over the column, so spec stripes pull the column
out to the true icon-row start.

`full_bar_extent(hsv, marker)` is gated on `classify_stripe`: if no
spec stripe is detected the function returns `marker.w` unchanged. The
gate is necessary because ability icons immediately right of the
marker can match HSV main-bands and inflate the extension on plain
markers.

### 6. Slot content classification

Per-slot ML inference is decoupled from localisation — once geometry
gives a stable bbox, classification is just `classify_patch(crop)` on
each projected region.

Production path (already implemented):

```python
matcher = SETSIconMatcher(sets_app)   # uses .config/images + cargo
for slot in projected_slots:
    crop = img[slot.y:slot.y+slot.h, slot.x:slot.x+slot.w]
    name, conf = matcher.classify_patch(crop)
```

`classify_patch` is the **ML-only** fast path (Stage 3 of `match()` —
EfficientNet-B0 on a 64×64 crop, no template/histogram fallback). The
model knows three label classes for BOFFs: real ability names,
`__empty__`, and `__inactive__` (training crops are uploaded
unfiltered by `sync.py`).

Baseline (36 GT panels, 704 GT slots matched to projections, IoU≥.30):

| GT state    | bucket-acc | exact-name | notes                       |
|-------------|------------|------------|-----------------------------|
| real        | 99.8%      | 94.9%      | localiser + classifier OK   |
| empty       | 50%        | n/a        | n=2 only                    |
| inactive    |  5.2%      | n/a        | almost always read as empty |

`__inactive__` confusion is the dominant remaining error (221/233
inactive crops classify as `__empty__`). Confidence histogram shows
nearly all such errors fall in `[0.0..0.1)` — a `conf < 0.10`
suppression rule would catch ~98% of them, but doesn't recover the
correct label.

## Visualisation

- **`tests/_diag_out/boff_markers_viz/<screen>`**: original screen
  with one coloured rectangle per detected marker (palette per band)
  + a green envelope showing the predicted ability-icon panel + a
  thin white rectangle showing real-item GT bboxes + thin cyan
  rectangles for the projected ability slots.
- **`tests/_diag_out/boff_spec_inspector/<screen>`**: per-screen
  zoomed gallery (4×) of every panel marker with the stripe sample
  region highlighted, top-3 band scores, and median sampled HSV.
  Used to tune `STRIPE_BANDS` — exposes exactly what HSV the
  classifier sees and why a band did or did not match.
- **`tests/_diag_out/boff_classify/classify.json`**: per-screen +
  rollup of `classify_patch` results, including per-state confusion
  matrix and conf histogram.

## Production integration

When wired into `warp/recognition/layout_detector.py`:

1. Call `detect_markers(img, icon_w, icon_h)` → `markers`.
2. Call `best_panel(markers, icon_w, icon_h)` → `(col_a, col_b, score)`
   or `None`.
3. If `None`: fall back to existing pixel/template/anchors strategies
   (now hits ≤3% of screens).
4. Compute the ability-icon panel envelope (visualisation only —
   `0.88` is an envelope-fit constant, NOT the per-slot geometry):
    - `env_ab_w = round(median_marker_w / 0.88)`
    - `env_ab_h = round(median_marker_h / 0.80)`
5. Project per-seat ability slots with `project_seat_slots(panel,
   hsv=hsv)`. Geometry follows section 5: bible-driven X-axis
   (`k = full_bar_w_median / 29`, `ab_w = round(k·29)`,
   `gap_slot = round(k·2)`, `stride_x = ab_w + gap_slot`,
   `gap_x = round(k·3)`) and empirical Y-axis (`ab_h = med_h / 0.63`,
   `gap_y = 0.20·med_h`). Pass `hsv` so the per-column anchor uses the
   full bar (main + spec stripe).
6. Optionally call `annotate_specs(img, panel_markers, icon_w)` to
   tag each seat with its specialization code.
7. For each projected bbox crop the patch and call
   `matcher.classify_patch(crop)` → `(name, conf)`. Treat low-conf
   inactive predictions as a known weakness (see "Slot content
   classification").

## Failure modes still in play

- All GT seats covered on the 36-screen GT set (177/177 = 100%).
  Detector still emits an extra marker on Screenshot_96 (6 vs 5 GT)
  but RANSAC anchors the correct 3+2 panel.
- Some Spec stripes still rejected at low score (e.g. very narrow or
  partially-cropped stripes). The inspector galleries surface these
  cases for further tuning.
- Tier-aware seat layout (Lt vs LtCmdr seats have different ability
  counts) is not yet handled — currently all seats assumed 4 abilities.
  See `todo_boff_tier_grid.md`.
- `__inactive__` vs `__empty__` confusion in `classify_patch` (3.4%
  bucket-acc on inactive). Likely a training-data signal problem —
  inactive icons are darker but otherwise identical in shape to empty
  slots. Possible directions: weight inactive class higher in
  retraining, add a brightness-based pre-filter, or fold the two
  states into one "no-ability" class and recover inactive via UI
  context (greyed-out tier label).
