# WARP ML Roadmap — Layout + Content Recognition

**Updated:** 2026-03-29
**Status:** v2.3 — P0–P10 complete. P11 planned (community anchors.json).

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

### 🟢 P6 — Progress indicator for OCR / matching during manual bbox draw (COMPLETED)

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

### 🟢 P7 — Training data augmentation (EfficientNet) (COMPLETED)

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

### 🟢 P8 — Confidence fusion: template + ML combined score (COMPLETED)

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

### 🟢 P9 — Hard negatives mining for EfficientNet (COMPLETED)

**Why:** The model confuses visually similar items (e.g., consoles of the same set). Standard random training doesn't focus on these hard cases. Mining confusing pairs and over-sampling them during training directly improves the most common failure mode.

**What to do:**
- After each training epoch, run inference on the training set
- Collect samples where `predicted != label AND conf > 0.5` (confident but wrong)
- Double-weight these samples in the next epoch's sampler (`WeightedRandomSampler`)
- Cap hard negative weight at 3× to avoid instability

**Testing:** Claude compares confusion matrix before/after on val set. Expects reduction in high-confidence errors for the top-5 most confused class pairs.

**Files:** `warp/trainer/local_trainer.py`.

---

---

### 🔵 P10 — Remove layout CNN entirely

**Why:** The MobileNetV3-Small layout regressor (P4, `layout_regressor.pt`) is trained only on the local user's confirmed annotations — typically 5–20 annotated screenshots. A regressor predicting 210 output values (105 slots × 2 coordinates) needs hundreds of training examples to generalise. With < 20 examples it simply memorises the pixel coordinates of screenshots it has already seen. This is identical to what `anchors.json` (Strategy 1) already does, but slower and less debuggable.

**Critical limitation:** Layout data never reaches the central training pipeline. `SyncWorker` uploads only crops (icon images), not screenshots. So the CNN trains locally, stays locally, and produces no ecosystem benefit. Every user starts from scratch.

**What to remove:**

| File | Action |
|------|--------|
| `warp/trainer/layout_dataset_builder.py` | Delete entirely |
| `warp/trainer/layout_trainer.py` | Delete entirely |
| `warp/trainer/local_trainer.py` | Delete entirely (no icon training, no layout training) |
| `warp/recognition/layout_detector.py` | Remove Strategy 0: `_detect_via_cnn()`, `_load_cnn()`, `_cnn_model` field |
| `warp/trainer/trainer_window.py` | Remove "Train Layout Model" toolbar action + `_on_train`, `_on_train_finished`, `_on_train_cancelled`, `_TrainProgressDialog` |
| `warp/models/layout_regressor.pt` | Delete if present |

**What stays unchanged:**
- Strategy 1 (`anchors.json` learned layouts) — still fully functional
- Strategy 2 (pixel analysis)
- Strategy 3 (OCR labels)
- Strategy 4 (static fallback anchors)
- P5 (anchor recalibration on high-conf icon matches) — still useful

**Verification (Claude-side):**
1. `grep -r "_detect_via_cnn\|layout_regressor\|LayoutDatasetBuilder\|LocalTrainWorker\|layout_trainer" warp/` → 0 results
2. `python -c "from warp.recognition.layout_detector import LayoutDetector; print('OK')"` → no import error
3. `python -c "from warp.trainer.trainer_window import WarpCoreWindow; print('OK')"` → no import error
4. Verify `anchors.json` Strategy 1 still fires: add `log.debug("Strategy 1: learned layout")` if needed, confirm it appears in log on second run of same resolution screenshot

**Cleanup:** Remove `_KEY_TRAIN_REPEATS` and any remaining `warp/trainer/model_updater.py` references to `layout_regressor.pt` if any exist.

---

### 🔵 P11 — Community anchors.json

**Why:** Layout detection is currently the weakest link. Strategies 1–4 all fail in some scenario:
- Strategy 1: only works if the exact same resolution was previously confirmed
- Strategy 2: only counts icons, doesn't know slot order
- Strategy 3: OCR is slow and unreliable on compressed screenshots
- Strategy 4: static anchors are wrong for any non-standard window size

**The key insight:** Bbox grid data (normalized [0,1] slot coordinates per build type and aspect ratio) contains **no visual content** — just numbers. It can be sent to the central server without privacy concerns. If 10 different users confirm layouts for `SPACE_EQ` at 16:9, we can aggregate their grids (slot-wise median) to produce a robust community layout that works for any new user with the same screen ratio.

**Architecture:**

```
User confirms bboxes in WARP CORE
  → SyncWorker uploads crop PNGs to HF (existing P-crops pipeline)
  → SyncWorker ALSO uploads anchors_grid.json entry (new P11 addition)

admin_train.py (hourly GitHub Actions)
  → collects all anchors_grid.json entries from staging/<install_id>/
  → groups by (build_type, aspect_ratio_bucket)
  → computes slot-wise median per group (min 3 contributors)
  → writes community_anchors.json
  → uploads community_anchors.json to HF sets-sto/warp-knowledge

model_updater.py (client, every 15 min check)
  → downloads community_anchors.json alongside icon_classifier.pt
  → saves to warp/models/community_anchors.json

layout_detector.py Strategy 1 (updated)
  → first checks local anchors.json (user's own confirmed layouts)
  → if no local match → tries community_anchors.json (new)
  → key format: "<build_type>_<aspect_w>x<aspect_h>" (e.g. "SPACE_EQ_16x9")
```

**Implementation steps:**

1. **`sync.py` — upload anchors grid alongside crops**
   - After collecting confirmed annotations, build a per-screenshot grid dict:
     ```json
     {
       "build_type": "SPACE_EQ",
       "aspect": "16x9",
       "resolution": [1920, 1080],
       "slots": {
         "Fore Weapons 1": [0.123, 0.456, 0.045, 0.060],
         "Deflector": [0.210, 0.320, 0.045, 0.060]
       }
     }
     ```
     All coords normalized to [0,1] relative to screenshot size. No image data.
   - Upload as `staging/<install_id>/anchors_grid_<sha8>.json` to HF dataset
   - Only upload if ≥ 3 distinct slots confirmed on that screenshot

2. **`admin_train.py` — aggregate grids**
   - After training (or independently), scan `staging/*/anchors_grid_*.json`
   - Group by `(build_type, aspect)` — parse aspect from string `"16x9"` → bucket W:H
   - For each group with ≥ 3 distinct `install_id` contributors:
     - For each slot in the union, collect all [nx, ny, nw, nh] vectors
     - Median per component (robust to outliers)
   - Write `community_anchors.json`:
     ```json
     {
       "generated_at": "2026-03-29T12:00:00Z",
       "n_contributors": 7,
       "grids": {
         "SPACE_EQ_16x9": {
           "Fore Weapons 1": [0.123, 0.456, 0.045, 0.060],
           ...
         }
       }
     }
     ```
   - Upload to `models/community_anchors.json` in HF

3. **`model_updater.py` — add to download list**
   - Add `('models/community_anchors.json', 'community_anchors.json')` to `_MODEL_FILES`
   - File is optional (no crash if missing on HF — old installs still work)

4. **`layout_detector.py` — use community anchors as Strategy 1b**
   - In `_try_learned_layout()`, after failing local `anchors.json` lookup:
     ```python
     community = self._load_community_anchors()
     if community:
         grid = community.get(f"{build_type}_{aspect_key}")
         if grid:
             return self._grid_to_layout(grid, img_w, img_h)
     ```
   - `_load_community_anchors()`: reads `warp/models/community_anchors.json`, caches in-memory, returns None on any error

**Verification (Claude-side):**

1. After P11 `sync.py` change: `grep -n "anchors_grid" warp/trainer/sync.py` → finds upload call
2. After `admin_train.py` change: dry-run locally against test staging data → prints `community_anchors.json` contents to stdout
3. After `model_updater.py` change: `grep "community_anchors" warp/trainer/model_updater.py` → file in `_MODEL_FILES`
4. After `layout_detector.py` change: mock `community_anchors.json` with known coords, run `LayoutDetector._try_learned_layout()` on a fresh install (no local `anchors.json`) → verify it returns the community coords

**Learning speed analysis:**

| Users with confirmed layouts | Aspect ratio coverage | Community anchor quality | Expected outcome |
|-----------------------------|-----------------------|-------------------------|-----------------|
| < 3 | Any | Not generated (min threshold) | Falls through to pixel analysis |
| 3–5 | 16:9 only | Median of 3 grids — coarse but functional | ~80% correct slot placement for 16:9 users |
| 5–10 | 16:9 + 16:10 | Median stable, outliers filtered | New user gets correct layout on first screenshot |
| 10+ | Most common aspects | High-quality grid with good coverage | Near-zero layout errors for common resolutions |
| 20+ | All aspects including ultrawide | Full coverage | Strategy 1b effectively solves layout detection |

**At current scale (single developer):** 1 user = 0 community grids (need ≥ 3 distinct install_ids). Community anchors will not activate until the first beta users start confirming layouts. Local `anchors.json` (Strategy 1) remains the primary mechanism until then.

**Privacy note:** Zero visual data. Slots are referenced by name (e.g., `"Fore Weapons 1"`), coordinates are normalized floats. No screenshot content, no filenames, no metadata that could identify the game state.

**Aspect ratio buckets:**
- 16:9 (1920×1080, 2560×1440, 3840×2160)
- 16:10 (1920×1200, 2560×1600)
- 21:9 (3440×1440, 2560×1080)
- 4:3 (fallback for anything else)
- Bucket key: reduce `gcd(w, h)` → `"WxH"` string (e.g., `"16x9"`)

---

## Dependency order (updated)

```
✅ P0 (OCR on text slots)       — DONE
✅ P1 (slot from position)      — DONE
✅ P3 (layout multi-config)     — DONE
✅ P4 (CNN layout regression)   — DONE
✅ P5 (dynamic anchoring)       — DONE
✅ P6 (progress indicator)      — DONE
✅ P2 (cross-validation)        — DONE
✅ P7 (data augmentation)       — DONE
✅ P8 (confidence fusion)       — DONE
✅ P9 (hard negatives)          — DONE
✅ P10 (remove layout CNN)      — DONE
⬜ P11 (community anchors)      — PLANNED  (prerequisite: P10 cleanup)
```

---

## Testing policy

Each point specifies who tests and how:
- **Claude tests:** static analysis + log inspection + running app in background
- **User tests:** only when visual confirmation is required (e.g., "does the spinner look right?", "does the warning colour show for this screenshot?")
- User tests are always described with exact steps: what to launch, what to click, what to look for, what to report

---

## What is NOT broken and should not be changed

- EfficientNet icon classifier — central training via `admin_train.py`, improving with community crops
- MobileNetV3 screen classifier — central training only
- Session examples / seed from `annotations.json` — effective fallback
- pHash community knowledge — works when populated
- `learn_layout()` / `anchors.json` Strategy 1 — correct, stays as primary layout mechanism
- P5 anchor recalibration on high-conf icon matches — still useful
- `SLOT_VALID_TYPES` enforcement — already in place in `warp_importer.py`

---

## Files involved summary

| File | P10 action | P11 action |
|------|-----------|-----------|
| `warp/trainer/layout_dataset_builder.py` | **DELETE** | — |
| `warp/trainer/layout_trainer.py` | **DELETE** | — |
| `warp/trainer/local_trainer.py` | **DELETE** | — |
| `warp/recognition/layout_detector.py` | Remove Strategy 0 (CNN) | Add Strategy 1b (community anchors) |
| `warp/trainer/trainer_window.py` | Remove Train Layout Model button | — |
| `warp/trainer/sync.py` | — | Upload anchors_grid JSON per screenshot |
| `warp/trainer/model_updater.py` | — | Add community_anchors.json to download list |
| `sets-warp-backend/admin_train.py` | — | Aggregate grids → community_anchors.json |
