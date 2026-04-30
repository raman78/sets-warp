# BOFF panel detection

Prototype: `tests/diag_boff_markers.py`. The detector localises the BOFF
panel on STO screenshots by finding the small profession-coloured bars
that sit on the LEFT of every seat's name strip (just below the row of
ability icons). It does NOT depend on the icons themselves, so it is
robust to empty/inactive slots and to text-rendering changes.

Latest baseline (36 GT screens, 185 seats): **98.9% seat hit, 100%
panel anchor**, 60/167 markers also enriched with a specialization
code via the post-hoc stripe classifier.

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

## Visualisation

- **`tests/_diag_out/boff_markers_viz/<screen>`**: original screen
  with one coloured rectangle per detected marker (palette per band)
  + a green envelope showing the predicted ability-icon panel + a
  thin white rectangle showing GT extent.
- **`tests/_diag_out/boff_spec_inspector/<screen>`**: per-screen
  zoomed gallery (4×) of every panel marker with the stripe sample
  region highlighted, top-3 band scores, and median sampled HSV.
  Used to tune `STRIPE_BANDS` — exposes exactly what HSV the
  classifier sees and why a band did or did not match.

## Production integration

When wired into `warp/recognition/layout_detector.py`:

1. Call `detect_markers(img, icon_w, icon_h)` → `markers`.
2. Call `best_panel(markers, icon_w, icon_h)` → `(col_a, col_b, score)`
   or `None`.
3. If `None`: fall back to existing pixel/template/anchors strategies
   (now hits ≤3% of screens).
4. Compute the ability-icon panel envelope:
    - `ab_w = round(median_marker_w / 0.88)`
    - `ab_h = round(median_marker_h / 0.80)`
    - per seat: 4 ability slots starting at
      `(marker.x + marker.w, marker.y - ab_h)` with stride `ab_w`
5. Optionally call `annotate_specs(img, panel_markers, icon_w)` to
   tag each seat with its specialization code.

## Failure modes still in play

- 2 GT seats unmatched across 36 screens (single-marker misses, do
  not break panel anchor — not high priority).
- Some Spec stripes still rejected at low score (e.g. very narrow or
  partially-cropped stripes). The inspector galleries surface these
  cases for further tuning.
- Tier-aware seat layout (Lt vs LtCmdr seats have different ability
  counts) is not yet handled — currently all seats assumed 4 abilities.
  See `todo_boff_tier_grid.md`.
