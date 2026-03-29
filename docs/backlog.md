# WARP / WARP CORE — Backlog & Open Questions

**Updated:** 2026-03-29

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

**Decision needed:** Implement label bbox annotation / skip in favour of P11.

**Context:** WARP CORE lets users draw bboxes for icon slots. Currently the intent is to mark
where icons are. A related idea: let users draw bboxes around the **slot label text** (e.g.
"Fore Weapons", "Deflector") to provide ground truth for layout detection.

**Open questions:**
- Is this worth the UI complexity? Slot label positions can already be inferred from icon
  positions once icons are confirmed.
- If implemented: label bboxes would be a new slot type (e.g. `Label: Fore Weapons`) — they
  feed only into `anchors.json` / community anchors (P11), not into icon training.
- Alternative: skip label bboxes entirely; P11 community anchors cover this use case without
  requiring users to annotate text.

**Recommendation:** Skip — P11 community anchors cover this without extra UI complexity.

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

**Status: CODE REVIEW COMPLETE (2026-03-29) — runtime test pending**

**Findings from `model_updater.py` code review:**
- `_CHECK_INTERVAL_HOURS = 0.25` (15 min) — correct.
- Backend endpoint: `GET https://sets-warp-backend.onrender.com/model/version`
- Downloads 6 files from `sets-sto/warp-knowledge`: `icon_classifier.pt`, `label_map.json`,
  `icon_classifier_meta.json`, `model_version.json`, `screen_classifier.pt`, `screen_classifier_labels.json`.
- Timestamp comparison: `remote_ts > local_ts` → download; otherwise skip.
- After download: calls `SETSIconMatcher.reset_ml_session()` to reload immediately.
- `_ensure_screen_classifier()`: runs on every check, downloads screen classifier if missing
  (bypasses rate limit for this one-time bootstrap).
- Logic is correct and complete. No bugs found.

**Pending:** Actual runtime verification (open WARP CORE, wait 15 min, confirm model update
from log output). Use test script if needed:
```python
from pathlib import Path
from warp.trainer.model_updater import ModelUpdater
result = ModelUpdater().check_and_update(Path('.'))
print('update result:', result)
```

---

## 6. P11 — Community anchors.json

See `docs/warp_ml_roadmap.md` for full spec. Prerequisite: P10 (done).

**Status: IN PROGRESS (2026-03-29)**

**Implemented:**
- `sync.py` — `_upload_anchors_grid()`: uploads normalized slot grids to HF staging
- `layout_detector.py` — Strategy 1b: community anchors fallback after local miss
- `model_updater.py` — `community_anchors.json` in `_MODEL_FILES` (optional)
- `admin_train.py` (backend) — `build_community_anchors()` + `upload_community_anchors()`

**Pending:** Runtime test — requires >= 3 distinct users contributing confirmed layouts.

---

## 7. Ship type recognition — community data collection (opt-in)

**Status: Planned, design decided**

**Context:**
- OCR reads ship type reliably in most cases (text parsing bug fixed 2026-03-29).
- When OCR fails entirely → fallback is `_GENERIC_PROFILE` (generic T6 slot counts).
- Local `Ship Type` annotations in `annotations.json` help only locally — they don't feed
  any central ML training → currently useless from an ecosystem perspective.
- STO build screenshots are routinely shared publicly (Discord, Reddit, forums), so
  privacy concern for ship type text is low. Ship name is the sensitive part.

**Decision:** Add **opt-in** community upload of ship type text crops to HF.

**Design:**
- New opt-in checkbox in settings: "Contribute ship type recognition data" (default: off).
- When enabled: `Ship Type` bbox crop (text area only, not full screenshot, not ship name)
  is uploaded to HF alongside icon crops.
- `Ship Type` stays in `NON_ICON_SLOTS` when opt-in is off.
- Central `admin_train.py` uses these crops to improve OCR or train a lightweight
  ship-type text classifier.
- Local-only fallback usage of Ship Type annotations (`warp_importer.py` lines 911–928)
  should be removed — it's cheating without training value.

**Privacy boundary:**
- Upload: ship type text crop only (e.g. `"Fleet Support Cruiser"`)
- Never upload: ship name, character name, full screenshot

**Prerequisite:** P11 infrastructure (SyncWorker extension for text crops).

---

## 8. Remove local-only Ship Type annotation fallback in warp_importer.py

**Status: COMPLETE (2026-03-29)**

**Done:** Removed `ship_name_ann` / `ship_type_ann` reads and the ShipDB-via-annotations
lookup from `_load_confirmed_profile()`. Function now returns only confirmed slot counts
(which DO feed P11 community anchors). OCR handles ship type recognition autonomously.
