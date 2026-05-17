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

## BOFF panel detection (profession-marker anchor)

Read-only prototype: `tests/diag_boff_markers.py`. Sampling helpers used
to characterise the marker palette: `tests/diag_boff_markers_sample.py`
and `tests/diag_boff_marker_stripe.py`. Visual reference:
`docs/images/boff_seat_marker_colors.png`
(generator: `tests/diag_boff_marker_swatch.py`).

### Why a marker-based anchor

The earlier EQ-anchored sweep + `grid_from_anchor` failed at panel
selection on more than 30 % of full-screen shots. The previous bar
detector (`tests/diag_boff_seats.py`) keyed on the dark+saturated name
bar, which also matched many decorative UI strips.

The profession marker badge (left edge of each seat's name bar, BELOW
the abilities) is a much stronger anchor:

- solid coloured, fixed-size, on every seat regardless of which
  abilities are slotted;
- combines a wide main zone (seat type) with a narrow spec stripe — a
  two-tone signature random UI rarely produces;
- placement is panel-internal, so it works regardless of where the user
  positioned the BOFF panel on screen (per the
  *no fixed-position assumptions* rule).

See `docs/sto_slots_rules.md` "Bridge Officers → Seat marker colours"
for the colour reference and HSV bands.

### Algorithm

1. **Mask per band.** Build five HSV masks
   (Tactical / Engineering / Science / Universal × the 4 spec-stripe
   colours). Tactical's hue wraps the 0/180 boundary so it uses two
   sub-bands.
2. **Connected components.** Filter by size
   (`w ∈ [0.25..1.4]·icon_w`, `h ∈ [0.4..1.6]·icon_h`) and density
   (≥ 25 % of the bbox is mask).
3. **Dedupe** overlapping detections across bands (IoU > 0.4) — keeps
   the larger component, since a seat marker may match both its main
   colour and the spec stripe colour.
4. **RANSAC pair selection.** For each candidate pair `(m_i, m_j)` at
   similar Y and `dx ∈ [3..9]·icon_w` (the L/R column gap), sweep five
   `pitch_y` candidates `[1.6, 2.0, 2.4, 3.0, 3.6]·icon_h` and count
   inliers per row index.
5. **Score.** Prefer canonical 3+2=5 layouts via
   `canon_table = {5: 1.5, 4: 1.0, 3: 0.4, 6: 0.6, 2: 0.0}`, plus
   profession diversity (≥ 2 codes → +0.6), L≥R column layout (+0.3),
   and pitch consistency.

### Baseline (2026-04-26, 35 GT screens, 175 GT seats)

| Metric                | Value         |
|-----------------------|---------------|
| Panel anchor          | 32 / 35 = 91.4 % |
| Seat hits             | 160 / 175 = 91.4 % |
| Mean markers / screen | ~33           |

Adding the **Universal** band (pale cream yellow) was a net win: it
both recovered Universal seats *and* let us tighten the Engineering
saturation band from `S > 60` to `S 120–200`, removing many false
positives from bright UI elements.

### Known failures (3 screens)

`Ambassador-broadside`, `Chronos-broadside`,
`screenshot_2026-01-23-21-27-06`: each finds ≥ 4/5 real seats but the
RANSAC search picks a competing 3+2 cluster elsewhere in the image.
Candidate next steps (not yet implemented):

- require the **two-zone marker structure** (wide main + narrow stripe
  of matching seat type) — a near-unique signature;
- post-anchor verification by counting ability icons inside the panel
  envelope.

### Specialization classification (post-hoc)

After the seat-type detector returns markers, `classify_stripe()`
samples the right edge of each marker bbox and scores it against the
five spec-stripe bands (Command, Intelligence, Temporal, Pilot,
Miracle Worker). The dominant band, if its match fraction exceeds
0.15, is recorded as the seat's specialization.

Verified on 15 user-labelled GT seats (`tests/diag_boff_spec_stripe_gt.py`):
**9/15 correctly identified.** The 6 misses are not classifier errors —
in 5 cases the seat-type detector picked a non-marker CC for that
seat (so there was no real marker for the classifier to read), and 1
case scored just below the 0.15 acceptance threshold.

The classifier deliberately runs *after* detection so it doesn't
contribute extra noise to the RANSAC panel search. An earlier
experiment used the five stripe colours as additional detection seeds:
total markers per screen rose from 33 to 40 and panel anchor dropped
from 91.4 % to 88.6 % — the reverted approach.

### Marker geometry

Per-seat sampling region used by the diagnostics:

```
y ∈ [yc + 0.6·icon_h , yc + 1.4·icon_h]
x ∈ [x_left − 0.9·icon_w , x_left − 0.05·icon_w]
```

where `yc` is the centre Y of the 4-icon ability row and `x_left` is
the leftmost ability's X.

---

## BOFF slot assignment (`warp_dialog.py`)

### Cluster → seat matching (Phase 2)

Abilities are grouped into Y-band clusters, then matched to ship seats in four passes:

1. **Named non-Universal seats** — `_find_cluster(prof, spec_prof)`; if no match, fallback `_find_cluster(spec_prof, None)`. The fallback handles dual-spec seats (e.g. Engineering-Temporal) where all detected abilities belong to the spec profession.
2. **Universal seats with spec** — `_find_cluster(spec_prof, None)`.
3. **Universal seats without spec** — remaining clusters assigned in order.

`_find_cluster` looks for a cluster whose `c_primary` matches the requested profession and (if given) the spec profession is somewhere in `c_profs`.

### Slot index mapping (Phase 3)

`_slot_indices_from_x(cluster_items, rank)` maps each ability to its correct seat slot index using X-position gaps:

- `step = min(X-gaps)` between sorted active icons
- `round(gap / step)` — a 2× gap means 1 empty slot between
- Falls back to slot 0 for a single-ability cluster

Replaces the old sequential `rank_slot = 0, 1, 2, …` fill, which misassigned abilities when a slot was empty in the middle of a seat row.

### Empty / inactive slot positions

`_fill_boff_gaps` in `LayoutDetector` adds virtual bboxes `(x, y, w, h, state='empty'/'inactive')` for positions not covered by detected active icons. These flow into the cluster as `RecognisedItem(name='__empty__'/'__inactive__', conf=1.0)`.

- Virtual items are **included** in `cluster_items` → their X coordinates anchor `_slot_indices_from_x` precisely (1 active ability in a 4-slot seat still maps to the correct slot index)
- Virtual items are **skipped** in the build-write loop via `VIRTUAL_ITEM_NAMES` check
- Virtual item crops are uploaded to HF training data (no filter in `sync.py`) → EfficientNet learns to recognise empty/inactive states directly

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
  -> Per-sha label cache (.sync_uploaded_labels.json) tracks last-sent
     (slot|name) so corrections re-emit but no-ops don't rewrite the jsonl
  -> SyncWorker uploads crop + annotations.jsonl to HF (sets-sto/sto-icon-dataset)
     at staging/<install_id>/, single commit per cycle, last-wins sha dedup

User confirms BOFF slot (incl. virtual __empty__ / __inactive__)
  -> Same path. Seat key embeds prof+spec code: Boff Seat L[T+P]_483
  -> Virtual items: crops upload to HF (training signal) but SETS write is skipped

User confirms Ship Type / Ship Tier bbox
  -> Same TrainingDataManager path — crop PNG + ml_name (OCR raw) also saved
  -> SyncWorker includes slot="Ship Type", ml_name in annotations.jsonl entry

User confirms Ship Name bbox
  -> annotations.json updated (position only) — NO crop, NO upload

User marks a screenshot Done (Alt+D)
  -> learn_layout writes a normalised slot grid to anchors.json
  -> Next sync uploads it as anchors_grid_<sha8>.json (only if not already sent)

Background (every 10 min, started 15 s after app launch)
  -> SyncManager.check_and_upload() — single cycle:
       crops → staging/<install_id>/crops/<sha>.png
       labels → staging/<install_id>/annotations.jsonl (per-sha dedup)
       screen-type captures → staging/<install_id>/screen_types/<stype>/<sha>.png
       anchor grids → staging/<install_id>/anchors_grid_<sha>.json
  -> Rate limit: 1000 file uploads / install_id / UTC day (corrections are free)

GitHub Actions (admin_train.py — hourly retrain)
  -> Icon entries: trains EfficientNet-B0 on community icon crops
  -> Embedder: admin_train_metric.py trains ArcFace embedder + builds embedding_index.npz
  -> Text entries: collect_text_corrections() builds ship_type_corrections.json
  -> Anchor grids: merged into community_anchors.json
  -> Uploads icon_classifier.pt + icon_embedder.pt + community_anchors.json +
     ship_type_corrections.json + model_version.json to HF sets-sto/warp-knowledge

Background (every 15 min via ModelUpdater)
  -> Polls backend GET /model/version (Render — has cold-start, 60s read timeout)
  -> If remote trained_at > local trained_at OR embedder is stale: downloads
     icon_classifier.pt + icon_embedder.pt + embedder_label_map.json +
     embedding_index.npz + community_anchors.json + ship_type_corrections.json
  -> Atomically swaps files, calls SETSIconMatcher.reset_ml_session()
  -> text_extractor.py reloads ship_type_corrections.json
  -> LayoutDetector.reset_community_anchors_cache()
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
