# WARP ML Roadmap — Layout + Content Recognition

**Analysis date:** 2026-03-24
**Status:** Planning

---

## Current state — honest assessment

### Layout detection (LayoutDetector)

Four strategies, tried in order:

| Strategy | Mechanism | Problem |
|----------|-----------|---------|
| 1 — Learned layouts | Reads `anchors.json` saved from confirmed annotations | Geometric memory only — not ML. One entry per (build_type, aspect). Last-saved wins, no clustering. |
| 2 — Pixel analysis | Scans right-to-left brightness, uses ShipDB profile for count floor | Unreliable without known ship; only knows *how many* icons, not *which* slot. |
| 3 — OCR labels | EasyOCR reads slot label text ("Fore Weapons", "Deflector"…) | OCR is slow and fails on scaled/compressed screenshots. |
| 4 — Static anchors | Hardcoded relative Y-positions in `SPACE_ANCHORS_REL` | Last resort; totally wrong for different window sizes. |

**Key problem:** No dedicated ML for layout. "Learning" = storing confirmed pixel coordinates in a JSON file and replaying them. This works only when the next screenshot has the same resolution and window size as a previously confirmed one. It does NOT generalise.

### Icon recognition (SETSIconMatcher)

| Stage | Mechanism | Works? |
|-------|-----------|--------|
| 0 — pHash override | Community knowledge.json | Good when populated |
| 1 — Template matching | cv2.matchTemplate against SETS wiki-icon cache | Works well for known icons |
| 2 — Histogram | HSV correlation fallback | Weak signal alone |
| 3 — ML (EfficientNet-B0) | Local .pt fine-tuned on confirmed crops | Improves as training data grows |
| B — Session examples | Confirmed crops from annotations.json loaded at startup | Effective fallback, not a substitute for ML |

**Progress visible.** Each `Train Model` run fine-tunes the classifier. The more confirmed crops, the better.

### Cross-validation between layout and content

**Currently zero.** Layout detection and icon matching are completely independent pipelines. There is no feedback loop: if layout places a bbox in the wrong row and icon matcher returns an item whose type doesn't match the slot, nobody catches this.

### Ship Name / Type / Tier bbox — drawn manually

When user draws a bbox and selects `Ship Name`, `Ship Tier`, or `Ship Type`:
- `slot in NON_ICON_SLOTS` → icon matching is **skipped entirely**
- Ship Name field is **disabled** ("OCR only — bbox position saved")
- No OCR is run on the bbox region
- User must set Tier/Type manually from dropdown

**The bbox position is saved but the content inside is ignored.**

---

## What needs to be built

### P0 — OCR on manually drawn Ship Name / Tier / Type bbox

**Why:** User draws a bbox around a text region. We already have EasyOCR. We should run it on the crop and auto-populate the field — exactly like auto-detect does for slot labels.

**What to do:**
- In `_on_bbox_drawn`, when `slot == 'Ship Name'`: run OCR on crop, populate `name_edit` (read-only but pre-filled for display).
- When `slot == 'Ship Tier'`: run OCR on crop, extract tier token with `RE_TIER` from `text_extractor.py`, set `_tier_combo` automatically.
- When `slot == 'Ship Type'`: run OCR on crop, fuzzy-match against `ship_list.json`, pre-select best match in `_ship_type_combo`.
- Show a short spinner (or reuse the existing progress bar) if OCR takes >0.5s (run in thread).

**Files:** `trainer_window.py`, uses `TextExtractor` or inline EasyOCR call + `RE_TIER` regex.

---

### P1 — Slot inference from drawn bbox position

**Why:** When user draws a bbox at pixel (x, y), the position is a strong signal for which slot it is — especially after a few annotations have been confirmed for this screenshot. Currently the position is completely ignored; user must pick slot from dropdown manually.

**What to do:**
- In `_on_bbox_drawn`, after icon matching and before setting `slot`:
  1. Compare bbox center-Y against confirmed annotations already in `_recognition_items` — is it on the same Y-row as a known slot?
  2. Compare against learned layout from `anchors.json` for this build_type+aspect.
  3. If a slot candidate is found with confidence > threshold → pre-select it in `_slot_combo` (user can override).
- This is **suggestion**, not override. User always has final say.

**Files:** `trainer_window.py` (`_on_bbox_drawn`), helper method `_suggest_slot_from_position(bbox, stype)`.

---

### P2 — Cross-validation: layout vs content

**Why:** The two main signals (where the icon is vs what the icon is) currently never talk to each other. The cross-check is the most powerful tool for catching errors.

**What to do:**
- After icon matching returns `(name, conf)` for a given `slot`:
  - Look up `name` in `cache.equipment` → get `item['type']` (e.g., `"Engineering Console"`)
  - Check if `item['type']` is valid for `slot` using `SLOT_VALID_TYPES` from `warp_importer.py`
  - If mismatch (e.g., layout says `Tactical Consoles` but item type is `Engineering Console`) → flag in review list with a warning colour
  - Log the conflict: `cross_check: slot={slot} item_type={item_type} → CONFLICT`
- Do this in `RecognitionWorker.run()` after building the items list, and also in `_on_bbox_drawn`.
- In WARP CORE review list: show warning icon or colour for cross-check failures.

**Files:** `trainer_window.py`, `warp_importer.py` (re-export `SLOT_VALID_TYPES`).

---

### P3 — Layout learning must handle multiple ship configurations

**Why:** Current `learn_layout()` keeps only one entry per `(build_type, aspect)` — actually it keeps all entries but always picks the *latest*. A T6 escort has 4 Tactical Consoles; a T6 science vessel has 4 Science Consoles. If both were confirmed at 16:9 SPACE, the second one overwrites the first.

**What to do:**
- In `_detect_via_learned_layouts`: instead of picking `candidates[-1]` (most recent), pick the **best structural match** against the current image:
  - Score = number of slots whose `x0_rel` and `y_rel` land on a bright pixel region in the current image.
  - Pick the layout with the highest score.
- In `learn_layout()`: keep ALL entries; do not deduplicate by geometry — only deduplicate exact duplicates (same res + same slot map).
- Add a max cap (e.g., 200 entries) with LRU eviction so `anchors.json` doesn't grow unbounded.

**Files:** `layout_detector.py`.

---

### P4 — Real ML for layout detection (medium-term)

**Why:** Rule-based pixel analysis fails when STO changes UI rendering, at non-standard resolutions, or with different UI scale settings. An ML approach generalises from examples.

**Approach — lightweight regression CNN:**
- Input: screenshot resized to 256×256
- Output: for each slot in `SPACE_SLOT_ORDER_STANDARD` (11 slots), predict `(y_rel, count)` — relative Y centre and number of icons
- Architecture: MobileNetV3-Small (same as screen classifier) with a regression head instead of classification
- Training data: confirmed annotations from `annotations.json` (we already have bboxes per slot)
- Training: during `Train Model` → add a `LayoutTrainerWorker` alongside `LocalTrainWorker`
- Inference: new Strategy 0 in `LayoutDetector.detect()`, before learned layouts

**Required data volume:** ~50 confirmed screenshots (each confirming 10+ slots) = ~500 training samples. Feasible with current annotation workflow.

**Files:** new `warp/trainer/layout_trainer.py`, `layout_detector.py` (new strategy 0), `trainer_window.py` (add layout training step to `_on_train`).

---

### P5 — Auto-detect feedback loop: use icon content to refine layout

**Why:** If auto-detect places a bbox in the wrong row but the icon matcher confidently identifies a "Deflector Array" there — we know exactly which slot that is. This should anchor the rest of the layout.

**What to do:**
- After full `_process_image` pipeline, post-process `result.items`:
  - For each item where `conf >= TEMPLATE_THRESHOLD` and `slot` can be inferred from item type (unique slot items: Deflector=1, Engines=1, Warp Core=1, Shield=1):
    - Treat that (slot, bbox_Y) pair as a high-confidence anchor
  - Re-interpolate uncertain slot positions using these anchors (similar to existing `_fill_gaps`)
- This is a second pass after initial layout+matching.

**Files:** `warp_importer.py` (`_process_image`), `layout_detector.py` (expose `_fill_gaps` as public with anchor injection).

---

### P6 — Progress indicator for OCR / matching during manual bbox draw

**Why:** OCR + icon matching on a drawn crop can take 1-3 seconds. Without feedback the UI appears frozen.

**What to do:**
- In `_on_bbox_drawn` when entering the matching/OCR path:
  - Show a small `QProgressBar` (indeterminate / busy) in the bottom panel
  - Run the OCR + matching in a `QThread` (similar pattern to `RecognitionWorker`)
  - Hide progress bar when done, populate fields
- Only needed if > 500ms — add a simple timer check.

**Files:** `trainer_window.py`.

---

## Dependency order

```
P0 (OCR on text slots)    — standalone, low risk, high UX value
P1 (slot from position)   — depends on anchors.json being populated (P3 helps)
P6 (progress indicator)   — prerequisite for P0 and P1 if run in thread
P3 (layout multi-config)  — improves P1 accuracy
P2 (cross-validation)     — standalone, depends on SLOT_VALID_TYPES (already exists)
P5 (auto-detect feedback) — depends on P2 logic
P4 (layout ML)            — largest effort; P3 + P5 first to get cleaner training data
```

---

## What is NOT broken and should not be changed

- EfficientNet icon classifier — works, improving with more training data
- MobileNetV3 screen classifier — works well
- Session examples / seed from training data — effective fallback
- pHash community knowledge — works when populated
- `learn_layout()` save mechanism — correct, just needs better retrieval (P3)
- `SLOT_VALID_TYPES` enforcement — already in place in `warp_importer.py`

---

## Files involved summary

| File | Changes needed |
|------|---------------|
| `warp/trainer/trainer_window.py` | P0, P1, P2, P6 |
| `warp/recognition/layout_detector.py` | P3, P4 (new strategy 0), P5 |
| `warp/warp_importer.py` | P2 (expose SLOT_VALID_TYPES), P5 |
| `warp/trainer/layout_trainer.py` | P4 (new file) |
| `warp/recognition/text_extractor.py` | P0 (reuse RE_TIER, extract_ship_info) |
