# Trait panel detection

Production module: `warp/recognition/trait_grid.py`. Prototype kept for
measurement: `tests/diag_trait_detector_v1.py`. The detector localises
trait icon panels (Personal/Starship/Reputation/Active Reputation) on
STO screenshots by finding consistent-spacing icon grids in the image,
then asking the icon classifier which section each row-group belongs to.
It does NOT rely on canonical section ordering or OCR headers.

Measured baseline (59 GT screens, 1129 trait bboxes):
**IoU≥.30 91.5%, slot OK 91.0%**, with PGT 100%, PST 95.9%, ST 88.4%,
ASR 93.3%, AGR 73.3%, GR 88.6%, SR 79.0%.

## Why structure + ML probe (not OCR / canonical order)

STO users compose trait screenshots arbitrarily — the same shot may
contain Personal Space + Reputation + Active Reputation, or just one
section, or sections in any vertical order. Section header text is
often missing, cropped or stylised beyond OCR's reach. Position-based
heuristics ("Starship is always below Personal") are wrong for
~30% of real screenshots.

What is invariant is the panel **structure**:

- 5-column canonical grid in 96% of sections (cap-driven: 11/5/7)
- Inter-icon spacing within a panel is uniform (≤ 5% jitter)
- Inter-row gap within a group is ≤ 0.4× icon_h; between groups it is
  ≥ 0.9× icon_h (the ship-name divider is unique to Starship)
- Different sections may share the same column x-coordinates (93.6%
  within ±5px)

Once a row-group is localised, its content tells us which section it
is — every personal/starship trait and reputation icon is already in
the icon classifier's training set.

## Pipeline

```
img ─► detect_icon_ccs ─► find_trait_rows ─► cluster_row_groups ─►
       (bright CCs)        (multi-chain         (Y-gap < 0.4×ih)
                            per Y line)
       │
       ▼
       lock_grids_multi  ─► resweep_rows_in_band ─► merge_starship_overflow
       (cluster rows by    (per-panel y-band         ([5] + [≤2 cols 0-1]
       (col_dx, x0))        re-extract)               separated by name divider)
       │
       ▼
       classify_group_section ─► emit_bboxes
       (icon_matcher per      (slot name → bbox list)
        sample, vote)
```

### 1. Bright CC detection

`_detect_icon_ccs(img)` thresholds the brightness channel and keeps
connected components with aspect ratio in `[ICON_AR_LO, ICON_AR_HI]`
and height in `[ICON_H_FRAC_LO, h_hi_frac] * H`. For tiny cropped
panels (`H < 250`) the upper cap is widened to 0.65 of image height,
so a 200×200 panel crop still admits its 35-px icons.

### 2. Multi-chain row extraction

`_find_trait_rows()` walks each Y-aligned set of CCs and emits **all**
non-overlapping consistent-spacing chains (≥3 icons, dx jitter ≤15%).
Two side-by-side panels at the same Y line (e.g. PGT + GR sharing
Y=595) therefore produce two separate chains instead of being merged.

### 3. Row-group clustering

`_cluster_row_groups()` clusters rows by Y-gap < 0.4×icon_h. Each
cluster represents a candidate "panel block" (one section's icons
stacked into 1-3 rows).

### 4. Multi-panel grid lock

`_lock_grids_multi()` clusters rows by their `(col_dx, x0)` signature
to ±15% / ±0.5×icon_w tolerance. Each cluster becomes a panel with its
own grid (`cols`, `col_dx`, `icon_w`, `icon_h`, `y_top`, `y_bot`).
Returns up to 4 panels per screen, sorted by row count.

### 5. Per-panel resweep

For each locked panel, `_resweep_rows_in_band()` re-extracts CC chains
constrained to the panel's grid columns and y-band. This catches icons
the initial pass missed because they sat at an unusual Y but on a
known column. `_cluster_resweep_groups()` then re-clusters the
resweep output into row-groups using the same Y-gap rule.

### 6. Starship overflow merge

`_merge_starship_overflow()` handles Starship Trait panels that wrap a
short second row (≤2 icons in cols 0-1). The signature for the merge
is "5-icon row + ≤2-icon row separated by 0.5-1.5×icon_h" — that gap
is unique to the ship-name divider in the Starship section, so it does
not collide with other section types.

### 7. ML section classification

`_classify_group_section()` calls `icon_matcher.classify_patch()` on
each icon in the group. Names are mapped to a section via:

- `cache.traits[env][trait_type][name]` → `{Space,Ground} {Personal,
  Reputation,Active Reputation} Trait`
- `cache.starship_traits[name]` → `Starship Trait`

The group's section is the majority vote over its members. Groups
that fail to vote (all `__empty__`) are dropped — a section assigned
"none" would not help the user anyway.

### 8. Emission

`_emit_bboxes()` returns `dict[slot_name → list[(x, y, w, h)]]` ready
for `LayoutDetector.detect()` to merge with the rest of the layout.

## Wiring

The detector runs as **Strategy 0** in `LayoutDetector.detect()` for
every trait-bearing build type:

| Build type | Behaviour |
|------------|-----------|
| `SPACE_TRAITS` / `GROUND_TRAITS` | Strategy 0 trait_grid (≥5 bboxes), falls back to OCR-header `_detect_traits` |
| `SPACE_MIXED` / `GROUND_MIXED` | trait_grid runs once and is merged into whichever equipment chain wins (learned / OCR-anchored / full_scan). Trait sections **overwrite** any equipment-chain trait output, since trait_grid classifies each row-group independently and is more accurate. |

Implementation:

- `warp/recognition/layout_detector.py:286-300` — TRAITS Strategy 0
- `warp/recognition/layout_detector.py:323-380` — MIXED `_merge_traits` closure
- `warp/warp_importer.py:839-851` — `_needs_matcher` includes
  `SPACE_TRAITS` / `GROUND_TRAITS` so `icon_matcher` and `app_cache`
  reach `LayoutDetector.detect()`

## Independence rule (CRITICAL)

Each row-group is classified on its own merits. The detector NEVER
walks a canonical `SPACE_ORDER → GROUND_ORDER` sequence. Source:
`feedback_trait_groups_independent.md`. A user can compose any subset
of sections in any vertical order, and the detector must handle that
without prior assumptions about section position.

## Remaining failure modes

- **Tiny `iw=14` panels** (e.g. image1/10/png): patches are too small
  for the 224×224 EfficientNet input to recognise. `__empty__` returns
  → groups vote-fail → dropped.
- **CC drops on heavily cropped panels** (e.g. some 2024-12-12
  captures with `n_ccs=14`): structure intact but coverage degrades.
- **Reputation icons not in trained classes** confidently return
  `__empty__` → group label = None → dropped. Fix would need either
  HSV frame-color analysis or matcher coverage improvement.
- **Active Rep vs Reputation distinction without ML help**: visually
  the frames differ (active = bright square traits, passive = darker
  hex frames). When the matcher cannot decide, an explicit frame-style
  classifier on the icon background could disambiguate.

## Diagnostics

`tests/diag_trait_detector_v1.py` is the prototype kept for
end-to-end measurement against the GT corpus. It mirrors the
production algorithm 1:1; on the same 59-screen corpus it returns the
same 91.5% IoU≥30 / 91.0% slot OK as the production module.
