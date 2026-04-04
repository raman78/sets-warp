# WARP CORE — Developer Reference

> **User guide** (how to use WARP CORE, train models, review recognition):
> see **[WARP_GUIDE.md](WARP_GUIDE.md)**.

---

## Window layout

```
+------------------+------------------------------+----------------------+
|   LEFT PANEL     |       CENTER PANEL           |    RIGHT PANEL       |
|   min 400px      |       min 400px              |    min 400px         |
|                  |                              |                      |
|  Screenshots     |  +------------------------+  |  Recognition Review  |
|  (file list)     |  |   SCROLL AREA          |  |  (review list)       |
|                  |  |  +------------------+  |  |                      |
|                  |  |  |  AnnotationWidget |  |  |  [+ Add BBox] [- Rm] |
|                  |  |  |  (canvas)        |  |  |                      |
|                  |  |  +------------------+  |  |  [x] Auto >= [0.75]  |
|                  |  +------------------------+  |  [ Accept (Enter) ]  |
|  [progress bar]  |  +------------------------+  |                      |
|                  |  |  BOTTOM PANEL          |  |                      |
|                  |  |  Slot / Item / Accept  |  |                      |
|                  |  +------------------------+  |                      |
+------------------+------------------------------+----------------------+
```

Splitter initial sizes: `[400, 700, 400]`

---

## Key files

| File | Purpose |
|------|---------|
| `trainer/trainer_window.py` | Main window (`WarpCoreWindow`), toolbar, progress dialogs |
| `trainer/annotation_widget.py` | Canvas widget — zoom, bbox drawing, selection |
| `trainer/training_data.py` | `TrainingDataManager`, `AnnotationState` |
| `trainer/screen_type_trainer.py` | `ScreenTypeTrainerWorker` — MobileNetV3 (central only) |
| `trainer/sync.py` | HuggingFace crop upload (`SyncWorker`) |
| `trainer/model_updater.py` | Background community model download check |

---

## Canvas (`annotation_widget.py`)

### Zoom (Gwenview-style)

- `_fit_scale` — computed once at `load_image()` from viewport size; never changes
- `_user_scale = None` → fit-to-window mode (widget does not expand)
- `_user_scale = float` → explicit zoom; widget expands, scroll area shows scrollbars
- Ctrl+wheel intercepted at `QApplication` level (no click needed)
- `adjustSize()` after zoom informs scroll area; `sizeHint()` returns `pixmap * _scale`

### Alt+LMB draw

- Hold Alt → cursor changes to gold crosshair (`DRAW_BBOX_COLOR`)
- Alt+LMB drag → draws new bbox, triggers icon matching, auto-accepts if conf ≥ threshold
- Global `QApplication.installEventFilter` for Alt key detection without focus

### Colour constants

```python
DRAW_BBOX_COLOR = QColor(255, 200, 0)   # bbox rect + fill + crosshair cursor
```

Change this one constant to update all three visual elements simultaneously.

---

## Auto-accept

- Checkbox `[x] Auto >= [0.75]` persisted via `QSettings` (`warp_core/auto_accept_enabled`, `warp_core/auto_accept_conf`)
- `_apply_auto_accept()` called on screenshot load, after Add BBox, after Auto-Detect
- Selecting from autocomplete dropdown confirms immediately (no Enter needed)

---

## Cyan (text) bboxes — Ship Name / Ship Type / Ship Tier

Cyan bboxes are OCR text slots, not icon-matched slots. They extract raw text from the screenshot to identify the ship and tier. Key differences from normal (red/green) bboxes:

- No icon matching, no confidence score.
- Autocomplete is not filtered to the item database — text is accepted as-is.
- The duplicate overlap warning (`> 70%` overlap) is **suppressed between cyan slots** because Ship Type and Ship Tier bboxes legitimately overlap.

### Ship Type vs Ship Tier overlap

The ship type text (e.g. `"Fleet Temporal Science Vessel"`) often spans **two lines** in the screenshot, and the second line contains or is adjacent to the tier token (`T6-X`). When annotating manually:

- **Ship Type** bbox — draw over the **full type text including both lines**. This bbox will physically contain the tier token area.
- **Ship Tier** bbox — draw a **separate smaller bbox** covering **only the tier token** (e.g. just `T6-X`).

The two bboxes overlap; this is correct. The importer reads them independently: Ship Type → ship database lookup, Ship Tier → slot count adjustment.

---

## Duplicate bbox warning

When confirming, checks if bbox overlaps (> 70%) any existing confirmed bbox of a different slot → shows `QMessageBox.warning`. Exception: cyan text slots do not trigger this warning against each other.

---

## Screenshot list — colour coding

| Colour | Meaning |
|--------|---------|
| White | No annotations yet |
| Light blue `#7ec8ff` | Has annotations — in progress |
| Green `#7effc8` | Marked Done — locked |

The colour is updated whenever annotations or done state changes.

---

## Done state (`_screenshots_done`)

**Purpose:** explicitly mark a screenshot as fully annotated. Triggers a single definitive `learn_layout` write to `anchors.json` and locks the screenshot against further edits.

**Persistence:** `warp/training_data/screenshots_done.json` — a JSON list of filenames, loaded when a folder is opened.

**Button:** `✓ Mark Done` / `↩ Back to Edit` (toggle, `QPushButton` checkable) — below the progress bar in the left panel. Enabled only when a screenshot is loaded.

**Shortcut:** `Alt+D`

**Locking:** when a screenshot is Done—
- `AnnotationWidget.set_locked(True)` — `mousePressEvent` returns early, no drawing possible
- `_btn_add_bbox.setEnabled(False)` — Add BBox button disabled
- Alt+LMB draw is blocked

**Un-done (Back to Edit):** removes the screenshot from `_screenshots_done`, calls `LayoutDetector().remove_layout(path.name)` to remove its entry from `anchors.json`, unlocks drawing.

### Layout learning flow

| Moment | Action |
|--------|--------|
| Accept (Enter) on an item | Nothing — layout **not** saved per-accept |
| Switching to another screenshot (if not Done) | `_learn_layout_for(prev_path)` — saves current confirmed bboxes as one entry |
| Clicking `✓ Mark Done` | `_learn_layout_for(path)` — definitive save; screenshot locked |
| Clicking `↩ Back to Edit` | `_remove_layout_for(path)` — removes entry from `anchors.json` |
| Already-Done screenshot switched away from | Nothing — entry already saved, not duplicated |

`learn_layout` stores `source_file: path.name` in each `anchors.json` entry so `remove_layout` can find and delete it by filename.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Enter | Accept current item |
| Del / Backspace | Remove selected bbox (canvas or review list) |
| Alt+A | Toggle Add BBox mode |
| Alt+D | Toggle Mark Done / Back to Edit |
| Alt+R | Remove selected bbox |
| Alt+LMB drag | Draw new bbox directly |
| Ctrl+wheel | Zoom 1× – 6× anchored to cursor |

---

## ML model details

| Model | File | Architecture | Trained by |
|-------|------|-------------|-----------|
| Icon classifier | `models/icon_classifier.pt` | EfficientNet-B0 | Central pipeline (`admin_train.py` on GitHub Actions) |
| Screen classifier | `models/screen_classifier.pt` | MobileNetV3-Small | `screen_type_trainer.py` |

Both models use PyTorch native `.pt` state dicts (not ONNX — ONNX produced uniform-output models).

### model_version.json format

```json
{
  "version": "abc123",
  "trained_at": "2026-03-21T12:00:00Z",
  "n_classes": 42,
  "val_acc": 0.87,
  "n_samples": 1234,
  "n_users": 5
}
```

Local training adds `"source": "local"`. `ModelUpdater` compares `trained_at` timestamps; local model always takes priority after training.

---

## Training data flow

```
User confirms icon bbox in WARP CORE
  -> TrainingDataManager.add_annotation()
  -> annotations.json + crop PNG saved to warp/training_data/
  -> SyncWorker uploads crop + annotations.jsonl to HF (sets-sto/sto-icon-dataset)

User confirms Ship Type / Ship Tier bbox
  -> Same TrainingDataManager path — crop PNG + ml_name (OCR raw) also saved
  -> SyncWorker includes slot="Ship Type", ml_name in annotations.jsonl entry

User confirms Ship Name bbox
  -> annotations.json updated (position only) — NO crop, NO upload

GitHub Actions (hourly, admin_train.py)
  -> Icon entries: trains EfficientNet-B0 on community icon crops
  -> Text entries: collect_text_corrections() builds ship_type_corrections.json
  -> uploads icon_classifier.pt + ship_type_corrections.json + model_version.json
     to HF sets-sto/warp-knowledge

Background (every 15 min via WARP CORE sync timer)
  -> ModelUpdater checks remote model_version.json on HF
  -> If remote trained_at > local trained_at: downloads new icon_classifier.pt
     and ship_type_corrections.json (optional)
  -> icon_matcher.py loads new .pt on next import
  -> text_extractor.py applies corrections from ship_type_corrections.json
```

---

## NON_ICON_SLOTS — two internal categories

`NON_ICON_SLOTS = {'Ship Name', 'Ship Type', 'Ship Tier'}` — text slots, not icon slots.
Used throughout the UI to suppress icon matching, hide the item-name completer, and show
OCR widgets instead. For UI logic this set is treated uniformly.

Internally, the set is split into two categories with different data handling:

| Constant | Slots | Crop saved | Uploaded | ml_name | Purpose |
|----------|-------|-----------|---------|---------|---------|
| `POSITION_ONLY_SLOTS` | `Ship Name` | No | No | No | Layout anchor only. Ship name is personal data — never stored, never uploaded. |
| `TEXT_LEARNING_SLOTS` | `Ship Type`, `Ship Tier` | Yes | Yes | Yes | Text crop + confirmed label + OCR raw → builds community `ship_type_corrections.json`. |

`ann.ml_name` stores the raw OCR output for Ship Type / Tier. When the user confirms a
different value, the pair `(ml_name, name)` is a correction example. The backend
aggregates these democratically into `ship_type_corrections.json`; clients download it
and apply corrections in `text_extractor.py` before ShipDB lookup.

See `docs/ML_PIPELINE.md` §2, §3, §7 and `docs/backlog.md` item 7 for the full design.

---

## NON_ICON_SLOTS — known pitfalls and solutions (v1.9b)

The pitfalls below still apply to all three slots regardless of the new internal split.
Guards that were written as `slot not in NON_ICON_SLOTS` continue to use the combined
set for UI behaviour; only `_sync_crop_index` and `_export_crop` now use the finer
`POSITION_ONLY_SLOTS` guard.

---

### Bug: canvas clicks on freshly confirmed NON_ICON_SLOT bbox did nothing

**Symptom:** After drawing and confirming a Ship Name/Type bbox, clicking on it on
the canvas did not highlight it or update the review list. After switching to another
screenshot and back, everything worked.

**Root cause:** `AnnotationWidget._annotations` (used by `_hit_test`) is set at
`load_image()` and was never refreshed during a session. Clicking the bbox caused
`_hit_test` to return a stale result from disk, which did not match the freshly
added `_recognition_items` entry — so the loop in `_on_item_selected` found no
matching row.

**Fix:** Added `refresh_annotations(path)` method to `AnnotationWidget`. Called from
`_on_accept` (in `trainer_window.py`) immediately after `add_annotation`, so
`_hit_test` always reflects the current confirmed state.

---

### Bug: Ship Name bbox disappeared after app restart

**Symptom:** User confirmed Ship Name, then Ship Type. After restart, only Ship Type
was visible — both bboxes appeared at Ship Type's position.

**Root cause:** `TrainingDataManager.add_annotation` step 2 (bbox-coordinate fallback)
matched by `bbox` alone, ignoring slot. If Ship Name and Ship Type were drawn at
identical pixel coordinates (same horizontal screen line), confirming Ship Type
overwrote Ship Name's entry in `annotations.json` in-place.

**Fix:** Step 2 is now skipped for `NON_ICON_SLOTS`. These slots can legitimately
share bbox coordinates, so only step 1 (exact `ann_id` match) and step 4 (new insert)
are used for them.

```python
# training_data.py — add_annotation step 2
if slot not in NON_ICON_SLOTS:
    for i, d in enumerate(self._annotations[key]):
        if tuple(d.get('bbox', [])) == bbox_t:
            ...
```

---

### Bug: switching between NON_ICON_SLOT items was slow (fan spinning)

**Symptom:** Clicking between Ship Name / Ship Type bboxes on the canvas caused
noticeable lag and fan activity.

**Root cause:** `_on_item_selected` (called on every canvas bbox click and also from
`_on_ocr_finished`) called `_populate_name_completer(slot)` unconditionally. For
NON_ICON_SLOTS there is no entry in `_SLOT_TO_CACHE_KEY`, so `_build_search_candidates`
fell through to the `else` branch and iterated **all** equipment categories — rebuilding
a QStandardItemModel with thousands of entries on every single click.

**Fix:** Guard in `_on_item_selected`:
```python
if slot not in NON_ICON_SLOTS:
    self._populate_name_completer(slot)
```

---

### Bug: drawing second bbox reused slot of first NON_ICON_SLOT

**Symptom:** User confirmed Ship Name, then drew a new bbox — the slot combo still
showed Ship Name. OCRWorker ran as `slot='Ship Name'`, confirmed the result as Ship
Name. Step 3 (SINGLE_INSTANCE) then silently deleted the first Ship Name. After
restart only one bbox remained.

**Root cause:** After accepting a confirmed annotation, the slot combo is not reset.
P1 slot suggestion also did not change the slot if both bboxes were at similar
vertical positions.

**Fix:** In `_on_bbox_drawn`, after P1 slot suggestion, check if the current slot is
a NON_ICON_SLOT that is already confirmed for this image. If yes, auto-advance to the
next unconfirmed slot in the sequence `Ship Name → Ship Type → Ship Tier`:

```python
if _current_slot in NON_ICON_SLOTS and self._current_idx >= 0:
    _confirmed_slots = {ann.slot for ann in self._data_mgr.get_annotations(path)
                        if ann.state == AnnotationState.CONFIRMED}
    if _current_slot in _confirmed_slots:
        for _next in ('Ship Name', 'Ship Type', 'Ship Tier'):
            if _next not in _confirmed_slots:
                _current_slot = _next
                self._slot_combo.setCurrentText(_next)
                break
```

---

### Feature: confirmed NON_ICON_SLOTs hidden from slot combo dropdown

**Behaviour:** Once Ship Name/Type/Tier is confirmed for the current image, it
disappears from the slot combo. This prevents the user from accidentally adding a
second bbox for the same slot (which would trigger SINGLE_INSTANCE removal of the
first). Removing a confirmed annotation via **Remove BBox** restores the slot.

**Implementation:** `_refresh_slot_combo(stype, keep_slot='')` filters out confirmed
NON_ICON_SLOTS before rebuilding the combo. `keep_slot` is passed when a row is
selected (review list click or canvas click) so the active slot always stays visible
for editing.

Called from:
- `_on_accept` — after confirm, slot disappears
- `_on_remove_item` — after remove, slot reappears
- `_populate_review_panel` — on image load, reflects saved state
- `_on_review_row_changed` — passes `keep_slot=slot` so editing is possible
- `_on_item_selected` — passes `keep_slot=slot` so canvas clicks work

**Pitfall:** `_on_accept` must pass `keep_slot` = slot of the **current row after
`_advance_to_next_unconfirmed`**, not the slot just confirmed. Otherwise the just-
edited slot immediately disappears when we stay on the same row.
