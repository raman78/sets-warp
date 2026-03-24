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
| `trainer/local_trainer.py` | `LocalTrainWorker` — EfficientNet fine-tune |
| `trainer/screen_type_trainer.py` | `ScreenTypeTrainerWorker` — MobileNetV3 |
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

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Enter | Accept current item |
| Del / Backspace | Remove selected bbox (canvas or review list) |
| Alt+A | Toggle Add BBox mode |
| Alt+R | Remove selected bbox |
| Alt+LMB drag | Draw new bbox directly |
| Ctrl+wheel | Zoom 1× – 6× anchored to cursor |

---

## ML model details

| Model | File | Architecture | Trained by |
|-------|------|-------------|-----------|
| Icon classifier | `models/icon_classifier.pt` | EfficientNet-B0 | `local_trainer.py` / community pipeline |
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
User confirms bbox in WARP CORE
  -> TrainingDataManager.add_annotation()
  -> annotations.json + crop PNG saved to warp/training_data/
  -> SyncWorker uploads crop to HF (sets-sto/sto-icon-dataset)

User clicks Train Model
  -> LocalTrainWorker fine-tunes EfficientNet on all confirmed crops
  -> icon_classifier.pt + model_version.json saved to warp/models/
  -> icon_matcher.py loads new .pt on next import

Background (8s after launch)
  -> ModelUpdater checks /model/version endpoint
  -> If remote trained_at > local trained_at: downloads from HF
  -> Skipped for 24h after bootstrap download
```
