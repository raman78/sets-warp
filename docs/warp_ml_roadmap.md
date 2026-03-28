# WARP ML Roadmap — Layout + Content Recognition

**Updated:** 2026-03-28
**Status:** v2.0 — P0–P5, P2 complete. Pending: P6, P7, P8, P9.

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

**Implemented (P2 complete).** After icon matching, item type is checked against `SLOT_VALID_TYPES`. Mismatches are flagged in the review list with ⚠️ warning colour and tooltip.

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

### 🟢 P2 — Cross-validation: layout vs content (COMPLETED)

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
- When an anchor item (Deflector, Engines, Core) is matched with confidence > 0.85, the delta between predicted and actual icon position is calculated.
- The entire layout grid is shifted on-the-fly for the current image — resistant to small UI shifts or scaling differences.

**Files:** `warp_importer.py` (`_process_image`, `_find_anchor_recalibration`).

---

### P6 — Progress indicator for OCR / matching during manual bbox draw

**Why:** OCR + icon matching on a drawn crop can take 1-3 seconds. Without feedback the UI appears frozen.

**What to do:**
- In `_on_bbox_drawn` when entering the matching/OCR path:
  - Show a small `QProgressBar` (indeterminate / busy) in the bottom panel
  - Run OCR + matching in a `QThread` (similar pattern to `RecognitionWorker`)
  - Hide progress bar when done, populate fields
- Only show if > 500ms — add a simple timer check before displaying.

**Testing:** Claude verifies spinner appears and disappears correctly. No user test needed.

**Files:** `trainer_window.py`.

---

### P7 — Training data augmentation (EfficientNet)

**Why:** Current EfficientNet fine-tune uses crops as-is. With small datasets (< 1000 crops per class) the model overfits. Adding augmentation during training improves generalization across different in-game UI scales, brightness settings, and display gammas without collecting more data.

**What to do:**
- In `local_trainer.py` training transform pipeline, add:
  - `transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`
  - `transforms.RandomHorizontalFlip(p=0.3)` — icons are mostly symmetric
  - `transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))` — small positional noise
- Keep validation transform clean (no augmentation) for accurate val_acc reporting

**Testing:** Claude compares val_acc before and after on a fixed held-out set. Expects ≥ same or better val_acc with less overfitting (train_acc − val_acc gap narrows).

**Files:** `warp/trainer/local_trainer.py`.

---

### P8 — Confidence fusion: template + ML combined score

**Why:** Current pipeline is strict fallback — template matching wins if it fires, ML is only used if template fails. When template score is borderline (0.5–0.7) and ML score is high (0.8+), ML should win. Combining both signals gives a more accurate final confidence.

**What to do:**
- After template match and ML inference both run, compute:
  `final_conf = max(template_conf, 0.4 * template_conf + 0.6 * ml_conf)`
  when `template_conf < 0.75` — otherwise template result stands unchanged
- Threshold for accept remains `MIN_ACCEPT_CONF = 0.40`
- Log both individual scores at DEBUG level for tuning

**Testing:** Claude runs recognition on 5–10 test crops with known labels, compares accuracy before/after. No user test needed unless recognition results look wrong.

**Files:** `warp/recognition/icon_matcher.py`.

---

### P9 — Hard negatives mining for EfficientNet

**Why:** The model confuses visually similar items (e.g., consoles of the same set). Standard random training doesn't focus on these hard cases. Mining confusing pairs and over-sampling them during training directly improves the most common failure mode.

**What to do:**
- After each training epoch, run inference on the training set
- Collect samples where `predicted != label AND conf > 0.5` (confident but wrong)
- Double-weight these samples in the next epoch's sampler (`WeightedRandomSampler`)
- Cap hard negative weight at 3× to avoid instability

**Testing:** Claude compares confusion matrix before/after on val set. Expects reduction in high-confidence errors for the top-5 most confused class pairs.

**Files:** `warp/trainer/local_trainer.py`.

---

## Dependency order (updated)

```
✅ P0 (OCR on text slots)       — DONE
✅ P1 (slot from position)      — DONE
✅ P3 (layout multi-config)     — DONE
✅ P4 (CNN layout regression)   — DONE
✅ P5 (dynamic anchoring)       — DONE
── P6 (progress indicator)      — UX, standalone, low risk
✅ P2 (cross-validation)        — DONE
── P7 (data augmentation)       — standalone, improves EfficientNet with no extra data
── P8 (confidence fusion)       — depends on both template + ML running (already do)
── P9 (hard negatives)          — depends on P7 training loop changes
```

---

## Testing policy

Each point specifies who tests and how:
- **Claude tests:** static analysis + log inspection + running app in background
- **User tests:** only when visual confirmation is required (e.g., "does the spinner look right?", "does the warning colour show for this screenshot?")
- User tests are always described with exact steps: what to launch, what to click, what to look for, what to report

---

## What is NOT broken and should not be changed

- EfficientNet icon classifier — works, improving with more training data
- MobileNetV3 screen classifier — works well
- Session examples / seed from training data — effective fallback
- pHash community knowledge — works when populated
- `learn_layout()` save mechanism — correct
- `SLOT_VALID_TYPES` enforcement — already in place in `warp_importer.py`

---

## Files involved summary

| File | Pending changes |
|------|----------------|
| `warp/trainer/trainer_window.py` | P6 |
| `warp/recognition/layout_detector.py` | — |
| `warp/warp_importer.py` | — |
| `warp/recognition/icon_matcher.py` | P8 |
| `warp/trainer/local_trainer.py` | P7, P9 |
