# WARP ML Roadmap — Layout + Content Recognition

**Analysis date:** 2026-03-25
**Status:** Beta v1.5 / ML Implementation Phase

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

### 🟢 P0 — OCR on manually drawn Ship Name / Tier / Type bbox (COMPLETED)

**Mechanism:** Dedicated `OCRWorker` using EasyOCR with smart text parsing.
*   **Upscaling**: Automatically resizes text crops 2x for better recognition of small game fonts.
*   **Regex / Fuzzy Matching**: Uses `RE_TIER` regex and `difflib` to map raw OCR text to valid STO tiers (T6, T6-X2, etc.) and ship types.
*   **Correction Learning**: Learns from user corrections via `ocr_typos.json`.

**Files:** `trainer_window.py`.

---

### 🟢 P1 — Slot inference from drawn bbox position (COMPLETED)

**Implementation:**
*   Added `_suggest_slot_from_position(bbox)` in `trainer_window.py`.
*   The system compares manual bbox location with existing confirmed annotations and learned layouts in `anchors.json`.
*   Auto-selects the most likely slot in the UI dropdown during manual annotation.

**Files:** `trainer_window.py`.

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

### 🟢 P3 — Layout memory with multi-config scoring (COMPLETED)

**Mechanism:** Updated `layout_detector.py` to store multiple layouts per resolution/aspect.
*   **Scoring Mechanism**: Picks the layout whose predicted slot positions match actual bright pixels (icons) on the current image. Allows distinguishing between Escort vs Sci ship layouts.
*   **200-entry LRU cap** for performance.

**Files:** `layout_detector.py`.

---

### 🟢 P4 — CNN Layout Regression (COMPLETED)

**Mechanism:** A dedicated MobileNetV3-Small regressor trained on confirmed UI structures.
*   **Training**: Automatically happens during `Train Model` in WARP CORE.
*   **Inference**: Acts as **Strategy 0** in `LayoutDetector`. Predicts all slot coordinates at once for any UI scale.
*   **Fallback**: Seamlessly falls back to Strategy 1 (Learned) if model not trained.

**Files:** `layout_dataset_builder.py`, `layout_trainer.py`, `local_trainer.py`, `layout_detector.py`.

---

### 🟢 P5 — Icon to Layout Feedback Loop (COMPLETED)

**Mechanism:** Layout recalibration based on high-confidence icon matches.
*   **Dynamic Anchoring**: When an anchor item (Deflector, Engines, Core) is matched with high confidence (>0.85), the system calculates the delta between the predicted and actual icon position.
*   **Real-time Shift**: The entire layout grid is shifted on-the-fly for the current image, ensuring 100% accurate crops even if the game window moved or scaled slightly.

**Files:** `warp_importer.py`.

**Files:** `layout_detector.py`.

---

### 🟡 P4 — Real ML for layout detection (IN PROGRESS)

**Progress:**
*   Created `warp/trainer/layout_dataset_builder.py` for extracting training samples from `annotations.json`.
*   Defined `LayoutRegressor` architecture in `warp/trainer/layout_trainer.py` based on **MobileNetV3-Small**.
*   Model is designed for 224x224 grayscale input and outputs 56-length vector (14 slots * x,y,w,h).

**Next steps:**
*   Implement training loop in `LayoutTrainWorker`.
*   Integrate ONNX inference as Strategy 0 in `LayoutDetector`.

**Files:** `warp/trainer/layout_dataset_builder.py`, `warp/trainer/layout_trainer.py`.

---

### 🟢 P5 — Feedback Loop: Dynamic Anchoring (COMPLETED)

**Implementation:**
*   Added dynamic recalibration in `warp_importer.py` during Image Processing.
*   Once a high-confidence anchor icon (Deflector, Engines, etc.) is found, its displacement is used to shift the rest of the layout for that specific image.
*   This makes auto-detection highly resistant to small UI shifts or scaling differences.

**Files:** `warp_importer.py` (`_process_image`), `_find_anchor_recalibration` helper.

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
