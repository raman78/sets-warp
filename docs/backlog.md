# WARP / WARP CORE — Backlog & Open Questions

**Updated:** 2026-04-03

Items here are not yet scheduled. Each has a status and open questions to resolve before implementation.

---

## 1. Ship name bbox — privacy & necessity

**Status: COMPLETE (2026-03-29)**

**Findings (2026-03-29):**
- `Ship Name` is in `NON_ICON_SLOTS` (`training_data.py` line 48).
- `update_crop_index()` returns immediately for `NON_ICON_SLOTS` — no crop PNG is saved.
- `get_confirmed_crops()` reads only from `crop_index` — Ship Name never appears there.
- `SyncWorker._upload()` calls `get_confirmed_crops()` → Ship Name content is never uploaded to HF.
- Bbox coordinates are saved locally in `annotations.json` for layout anchoring only.

**Decision:** Keep bbox (position used for P11 layout anchors). User cannot edit the name — field disabled, OCR reads it automatically. Implemented 2026-03-29.

---

## 2. User-drawn slot label bboxes (Fore Weapons, Deflector, …)

**Status: SKIPPED (2026-04-10)**

P11 community anchors cover the layout anchoring use case without requiring users to
annotate text labels separately. Adding label bbox annotation would increase UI complexity
for no practical gain — icon bbox positions already imply label positions. Decision final.

---

## 3. One-per-screen enforcement for Ship Name / Ship Tier / Ship Type

**Status: VERIFIED — already enforced**

**Findings (2026-03-29):**
- `SINGLE_INSTANCE_SLOTS` frozenset in `training_data.py` includes `Ship Name`, `Ship Type`, `Ship Tier`.
- `add_annotation()` step 3: when confirming a slot in `SINGLE_INSTANCE_SLOTS`, any existing
  confirmed annotation for the same slot is removed before inserting the new one.
- This is an overwrite policy (not a reject), which is the correct behaviour for re-annotating.

**No code change needed.**

---

## 4. Post-P10 housekeeping

**Status: COMPLETE (2026-03-29)**

### 4a. Local data cleanup
- `warp/models/layout_regressor.pt` — **deleted** (was orphaned from P4 CNN).
- `warp/training_data/anchors.json` — kept (still valid, used by Strategy 1).
- No `layout_*.png` / `layout_*.json` files found in `warp/training_data/` — nothing to clean up.

### 4b. HF dataset cleanup (`sets-sto/sto-icon-dataset`)
- Layout files were never uploaded to HF (layout training was local-only).
- No cleanup needed.

### 4c. Documentation consistency
- `docs/WARP_GUIDE.md` — updated: removed "Train Model" / "Train Layout Model" references,
  replaced Section 6 with "Community model — how it works" and Section 7 with current sync details.
- `docs/warp_core.md` — updated: removed `local_trainer.py` from file table, updated ML model
  table to show central-only training, updated training data flow diagram.
- `docs/ML_PIPELINE.md` — updated: icon classifier training section now describes `admin_train.py`
  on GitHub Actions; local training removal noted.

---

## 5. Central model pipeline — verification

**Status: COMPLETE (2026-04-10)**

**Verified in runtime log (2026-04-10):**
```
ModelUpdater: checking for remote model update...
ModelUpdater: remote model is newer (remote=2026-04-08, local=2026-04-06) — downloading...
ModelUpdater: model downloaded at 16:34 UTC — 3199 classes, val_acc=74.2%
ModelUpdater: icon matcher reloaded with new model
```
Full cycle confirmed: version check → download → matcher reload. No issues.

---

## 6. P11 — Community anchors.json

See `docs/warp_ml_roadmap.md` for full spec. Prerequisite: P10 (done).

**Status: COMPLETE (2026-04-03)**

**Implemented:**
- `sync.py` — `_upload_anchors_grid()`: uploads normalized slot grids to HF staging
- `layout_detector.py` — Strategy 1b: community anchors fallback after local miss
- `model_updater.py` — `community_anchors.json` in `_MODEL_FILES` (optional)
- `admin_train.py` (backend) — `build_community_anchors()` + `upload_community_anchors()`

**Threshold:** `min_contributors=2` (n=1 accepted as tentative truth, n≥2 consensus).

**Pending runtime activation:** Requires ≥1 user contributing confirmed layouts before
community_anchors.json is generated. Code is complete.

---

## 7. Ship Type / Ship Tier — community OCR correction data

**Status: COMPLETE (2026-04-10)**

**Previous design (opt-in) superseded.**

**Context:**
- OCR reads ship type reliably in most cases; confirmed user text is the gold standard.
- Ship Name is sensitive personal data (player name) → position only, never uploaded.
- Ship Type and Ship Tier have a fixed vocabulary and are routinely visible in public
  screenshots → low privacy concern; user confirmation adds training value.
- The comparison signal is: OCR raw output (`ml_name`) vs user-confirmed value (`name`).
  When they differ, that pair is an OCR correction example.

**Decision:** Upload Ship Type and Ship Tier crops **by default** (no opt-in toggle).
These slots are now `TEXT_LEARNING_SLOTS`, not purely position-only like Ship Name.

**Architecture — new slot split:**

| Category | Constant | Slots | Crop saved | Uploaded | ml_name included |
|----------|----------|-------|-----------|---------|-----------------|
| `POSITION_ONLY_SLOTS` | `training_data.py` | `Ship Name` | No | No | — |
| `TEXT_LEARNING_SLOTS` | `training_data.py` | `Ship Type`, `Ship Tier` | Yes | Yes | Yes |
| `NON_ICON_SLOTS` (combined) | `training_data.py` | all three | — | — | — used for UI logic |

`NON_ICON_SLOTS` remains unchanged for UI purposes (hide icon completer, show OCR
fields, suppress duplicate warning between cyan bboxes). The split is internal to
`_sync_crop_index` and `_export_crop` guards.

**Data flow:**
```
WARP CORE — user confirms Ship Type bbox:
  ann.ml_name = "F1eet Support Cruiser"   ← OCR raw (stored in Annotation)
  ann.name    = "Fleet Support Cruiser"   ← user confirmed (may differ)

sync.py._upload():
  annotations.jsonl += {crop_sha256, name, slot="Ship Type", ml_name, date}
  crops/<sha>.png   ← text region crop

admin_train.py.collect_text_corrections():
  For each (ml_name, name) pair where ml_name != name:
    votes[ml_name][install_id] = name
  winner = majority vote per ml_name key
  → ship_type_corrections.json: {"F1eet Support Cruiser": "Fleet Support Cruiser", ...}
  → uploaded to sets-sto/warp-knowledge/models/

model_updater.py:
  downloads ship_type_corrections.json (optional file)

text_extractor.py:
  after OCR → look up result in ship_type_corrections → apply if found
```

**Files to change (client — sets-warp):**
- `warp/trainer/training_data.py` — add `POSITION_ONLY_SLOTS`, `TEXT_LEARNING_SLOTS`;
  change guards in `_sync_crop_index`, `_export_crop`; add `ml_name` to crop_index
  entries for `TEXT_LEARNING_SLOTS`; restrict migration to `Ship Name` only.
- `warp/trainer/sync.py` — include `ml_name` in `annotations.jsonl` entries.
- `warp/trainer/trainer_window.py` — call `_contribute()` for Ship Type/Tier
  (was blocked by `slot not in NON_ICON_SLOTS`); skip `add_session_example()` for
  `TEXT_LEARNING_SLOTS`; ensure `ann.ml_name = ri.get('ocr_raw', '')`.
- `warp/trainer/model_updater.py` — add `ship_type_corrections.json` to `_MODEL_FILES`
  (optional download, skip if missing).
- `warp/recognition/text_extractor.py` — load corrections on init; apply after OCR
  before fuzzy ship lookup.

**Files to change (backend — sets-warp-backend):**
- `admin_train.py` — filter `TEXT_LEARNING_SLOTS` out of `collect_votes()` (icon
  training); add `collect_text_corrections()`; upload `ship_type_corrections.json`.

**Privacy boundary:**
- Upload: text crop of ship type / tier region only
- Upload: `ml_name` (OCR raw text) and `name` (user confirmed text)
- Never upload: ship name, character name, full screenshot

---

## 8. Remove local-only Ship Type annotation fallback in warp_importer.py

**Status: COMPLETE (2026-03-29)**

**Done:** Removed `ship_name_ann` / `ship_type_ann` reads and the ShipDB-via-annotations
lookup from `_load_confirmed_profile()`. Function now returns only confirmed slot counts
(which DO feed P11 community anchors). OCR handles ship type recognition autonomously.

---

## 9. Backend training logs — clearer section headers

**Status: COMPLETE (2026-04-03)**

**Context:**
GitHub Actions training output mixes PyTorch download progress, HF HTTP logs, and epoch
lines without clear separators. Hard to tell which model is training, when it started,
what data it used.

**Example of current output (confusing):**
```
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
  0%|  | 0.00/9.83M ...
Loaded backbone from previous central screen_classifier — fine-tuning
  Epoch  1/40  val_acc=71.4%  best=0.0%
Processing Files (0 / 0) ...
  Epoch  2/40  val_acc=85.7%  best=71.4%
screen_classifier saved — 7 classes, val_acc=100.0%
```

**Desired improvement (`admin_train.py`):**
- Add a clear header before each model's training block, e.g.:
  ```
  ── Training screen_classifier (MobileNetV3-Small) ──────────────────
  Dataset : 223 screenshots, 7 classes
  Backbone: fine-tuning from previous central model
  Budget  : 8 min
  ────────────────────────────────────────────────────────────────────
  ```
- Add a summary footer after saving:
  ```
  ✓ screen_classifier saved — 7 classes, val_acc=100.0%, 11 epochs
  ```
- Same pattern for icon_classifier (EfficientNet-B0).

**Scope:** `admin_train.py` only — `train()` and `train_screen_classifier()` entry points.

## 10. Bad annotation cleaning

**Status: LOCAL BUG FIXED (2026-04-03) — HF retract: accepted drift (option B)**

**Bug (fixed):** `remove_annotation()` removed entries from `_annotations` but not from
`_crop_index`. Since `get_confirmed_crops()` reads from `_crop_index`, a deleted annotation
would still appear as confirmed and be uploaded to HF by SyncWorker.

**Fix applied:** `remove_annotation()` now also removes the crop_index entry and deletes
the crop PNG on disk. Filenames embed `ann_id` as suffix — lookup is a simple scan.

**HF retract:** Once a crop reaches `sets-sto/sto-icon-dataset/staging/`, there is no
client-side delete mechanism. Two options considered:

| Option | Description | Complexity |
|--------|-------------|------------|
| A. Retraction list | Upload `retractions.json` with hashes to skip; `admin_train.py` filters | Medium |
| B. Accept drift | Central model fine-tunes regularly — 1 bad crop in 100 is ~1% noise, corrected over time | Zero |

**Decision: Option B.** Democratic voting in `collect_votes()` already mitigates single
bad labels. The local fix prevents future bad uploads; already-uploaded bad crops will be
diluted by correct crops from other users.

---

## 11. GROUND_MIXED screen classifier regression

**Status: COMPLETE (2026-04-10)**

**Symptoms (observed 2026-04-03):**
- `ScreenTypeDetector` returned UNKNOWN for 96% of 119 screenshots (threshold 0.70).
- Per-class accuracy on local training data: GROUND_MIXED 18%, TRAITS avg_conf=0.39.

**Root causes:**
1. `CONF_THRESHOLD = 0.70` was too strict for current community model.
   Focal Loss training produces lower softmax values; many correct predictions fall below it.
2. `GROUND_MIXED` class severely undertrained in community model — misclassified as TRAITS
   (30×) and SPACE_MIXED (22×). Community training data likely has very few GROUND_MIXED
   screenshots relative to other classes (class imbalance in HF staging).

**Client fix applied:**
- `screen_classifier.py`: `CONF_THRESHOLD` lowered 0.70 → 0.50, `SESSION_THRESHOLD` 0.65 → 0.55.
- `trainer_window.py`: `ScreenTypeDetectorWorker` now uses imported `CONF_THRESHOLD`
  instead of a hardcoded literal.

**Backend fix needed (`admin_train.py`):**
- Add minimum-samples-per-class guard: if a class has fewer than N samples in the
  community dataset, skip it from the label set for that training run (or oversample).
- Log per-class sample counts in training output so imbalance is visible.
- Consider replacing `_FocalLoss` with plain `CrossEntropyLoss` + class weights only;
  Focal Loss adds miscalibration without clear benefit on a 7-class balanced dataset.

---

## 12. MIXED + BOFFS layout detection — intelligent approach

**Status: COMPLETE (2026-04-10) — full scan implemented**

**Problem (resolved):**
For MIXED screenshots the `_find_panel_right_edge` heuristic failed when the image
contains multiple panels arranged arbitrarily.  For BOFFS the same pixel-column scan
was equally unreliable across different seat layouts.

**Solution implemented (2026-04-10):**

New `_detect_via_full_scan()` in `layout_detector.py`:

1. **Phase B — OCR section labels:** Full-image EasyOCR scan → match text against
   extended `SLOT_LABEL_ALIASES` (equipment + traits + boffs) → collect
   `{slot_name: (cx, cy)}` anchor positions.

2. **Phase C — Dense icon scan:** Sliding window (stride = icon_est//2), skip uniform
   patches (std < 12), run `icon_matcher.classify_patch()` (EfficientNet ML-only, no
   template matching) → NMS deduplication → list of `(x, y, w, h, name, conf, type)`.

3. **Fusion — row scoring:** Cluster detections into rows by Y proximity.
   For each (row, slot_name) pair compute score:
   - `0.65 × type_score` — fraction of icons in row whose EfficientNet type matches
     `_SCAN_SLOT_VALID_TYPES` (equipment), `_TRAIT_SLOT_MARKER` (traits), or
     `__boff_*` marker (boff abilities)
   - `0.35 × ocr_score` — proximity bonus if OCR label for that slot is nearby
   Row is assigned to highest-scoring slot (≥ 0.30 threshold).

**Screen types using full scan:**
- `SPACE_MIXED`, `GROUND_MIXED` — full scan after learned layouts fail
- `BOFFS`, `SPACE_BOFFS`, `GROUND_BOFFS` — full scan after learned layouts fail,
  before old pixel-based `_detect_boffs()` fallback

**Files changed:**
- `warp/recognition/icon_matcher.py` — added `classify_patch()` public method
- `warp/recognition/layout_detector.py` — module-level helpers + two new methods
- `warp/warp_importer.py` — passes `icon_matcher` + `app_cache` for MIXED + BOFFS types

**Detection chain (updated):**
```
MIXED / BOFFS:
  Strategy 1: Learned layouts (anchors.json) — if ≥5 slots found
  Strategy FS: Full scan (OCR + EfficientNet + fusion) — if ≥2/3 slots found
  Strategy 2+: Existing pixel / canonical / OCR / anchors fallbacks
```

---

## 13. `__inactive__` / `__empty__` slot discarded in WARP CORE add_bbox

**Status: FIXED (2026-04-10)**

`_finish_bbox_drawn` was calling `_infer_slot_from_name` for all matched names, including
virtual names (`__inactive__`, `__empty__`).  Since these are not cache items, inference
returned `None`, triggering the discard branch with "not valid for stype=..." message.

**Fix:** In the `elif name not in VIRTUAL_ITEM_NAMES:` discard branch — virtual names
bypass slot inference and keep the positional slot suggestion.  They are always valid
training labels regardless of screen type.  File: `warp/trainer/trainer_window.py`.

---

## 14. Dynamic BOFF Seat Profession Mapping Bug

**Status: BACKLOG (Discovered 2026-05-01)**

**Problem:**
With the new `LayoutDetector` strategies, BOFF seats are named dynamically (e.g., `[Boff Seat L_483]`). The current logic in `warp_dialog.py` extracts the substring after "Boff " (resulting in "Seat L_483") to determine the cluster's profession. Because this does not match valid ship professions (Tactical, Engineering, Science), the recognized BOFF abilities are forced into the first Universal fallback seat, overriding the correct build layout in the SETS UI.

**Required Fix:**
In `warp_dialog.py` (Phase 2), the system should look up the actual recognized item names in `boff_abilities.json` (the source of truth) to determine the profession of the cluster, instead of relying on the raw slot name. For example, if a row contains `Emergency Power to Engines`, it should be classified as an Engineering cluster and matched to an Engineering seat.
