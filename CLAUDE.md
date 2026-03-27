# SETS-WARP — Claude Code Context

## Language rules

**All code must be in English** — comments, log messages, docstrings, variable names, string literals visible in logs. No Polish in source files. When editing existing code that contains Polish log messages or comments, translate them to English.

## Rules

1. First think through the problem, read the codebase for relevant files.
2. Before you make any major changes, check in with me and I will verify the plan.
3. Please every step of the way just give me a high level explanation of what changes you made
4. Make every task and code change you do as simple as possible yet not naive. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
5. Maintain a documentation file that describes how the architecture of the app works inside and out.
6. Maintain a documentation files in the projects. Recognize which are technical and which are more human redable (manual, program description, readme)
7. Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
8. Never use workarounds. Especially never change existing code just to fix your freshly made problem. Only recent changes are supposed to be fixed. If situation requires fixing existing code it requires user one-time approval.
9. NEVER EVER USE -Force or -f (force attribute) in terminal commands. It is strictly forbidden! If there is no other way you NEED to ask the user to run the command in terminal themselves providing justification.

---

## Project overview

Star Trek Online build planning tool with ML-based screenshot recognition.

- **SETS** *(STO Equipment and Trait Selector)* — build planner (Qt GUI)
- **WARP** *(Weaponry & Armament Recognition Platform)* — screenshot recognition module
- **WARP CORE** — ML trainer UI for reviewing/correcting recognition results

**Stack:** Python 3.13.2, PySide6, OpenCV, PyTorch, EasyOCR  
**Entry point:** `sets_warp.sh` (Linux/macOS) or `sets_warp.bat` (Windows) → `bootstrap.py` → `main.py`

---

## Repository structure

```
sets-warp/
├── main.py                        # App entry
├── bootstrap.py                   # Env setup (portable Python, venv)
├── sets_warp.sh / sets_warp.bat             # Launch scripts
├── src/                           # SETS core
│   ├── app.py                     # Main window
│   ├── callbacks.py               # UI callbacks (select_ship, tier_callback, …)
│   ├── buildupdater.py            # align_space_frame, update_boff_seat, …
│   ├── constants.py               # BOFF_RANKS, SHIP_TEMPLATE, …
│   ├── datafunctions.py           # cache loading, ship selector
│   └── setsdebug.py               # logging: from src.setsdebug import log
└── warp/                          # WARP module
    ├── warp_button.py             # Injects ⚡WARP and 🧠WARP CORE buttons
    ├── warp_importer.py           # Full import pipeline
    ├── warp_dialog.py             # WARP import dialog (multi-step QDialog)
    ├── recognition/
    │   ├── text_extractor.py      # OCR: ship name/type/tier, screen type
    │   ├── screen_classifier.py   # Screen type ML (MobileNetV3-Small .pt)
    │   ├── layout_detector.py     # Bbox detection per slot
    │   └── icon_matcher.py        # Icon matching (template + histogram + EfficientNet)
    ├── trainer/
    │   ├── trainer_window.py      # WARP CORE main window (QMainWindow)
    │   ├── annotation_widget.py   # Canvas widget (screenshot + bbox overlay)
    │   ├── training_data.py       # TrainingDataManager, AnnotationState
    │   ├── local_trainer.py       # LocalTrainWorker (EfficientNet fine-tune)
    │   ├── screen_type_trainer.py # ScreenTypeTrainerWorker (MobileNetV3)
    │   └── sync.py                # HuggingFace sync
    ├── knowledge/
    │   └── sync_client.py         # Community pHash knowledge sync
    └── models/                    # Trained .pt model files
        ├── icon_classifier.pt
        ├── icon_classifier_meta.json
        ├── screen_classifier.pt
        └── screen_classifier_labels.json
```

Training data: `warp/training_data/annotations.json` + crop PNGs

---

## Key architecture decisions (from development sessions)

### Models — PyTorch native .pt (NOT ONNX)
ONNX dynamo exporter produced uniform-output models (conf=0.15 for all classes).
Replaced with `torch.save(model.state_dict(), 'model.pt')`.
- `icon_classifier.pt` — EfficientNet-B0, fine-tuned on confirmed crops
- `screen_classifier.pt` — MobileNetV3-Small, fine-tuned on confirmed screenshots

### Import pipeline (`warp_importer.py`)

`from_trainer=True` flag skips OCR (trainer always has confirmed annotations).

1. **OCR** (WARP dialog only) — `TextExtractor.extract_ship_info()`:
   - Wide top-band scan (20% height), anchored on Tier token (`T6-X2` etc.)
   - Two-stage: fast partial → full image fallback for MIXED screens
   - Slot labels used as screen type signals

2. **ShipDB lookup** (`ship_list.json`, 783 ships) — **type-first**:
   - Exact `type` field match
   - Word-subset match (OCR omits subtype words: `"Fleet Temporal Science Vessel"` → `"Fleet Nautilus Temporal Science Vessel"`); multiple candidates ranked by boff seating similarity (Jaccard) then fewest extra words
   - Fuzzy match (cutoff 0.68)
   - Keyword-based fallback profile

3. **Confirmed annotations** (from `annotations.json`):
   - Override slot counts (confirmed = authoritative)
   - Supply exact ground-truth bboxes → bypass pixel analysis
   - Provide ship name/type/tier when OCR unavailable
   - Extract boff seating for ship disambiguation

4. **Layout detection** (`layout_detector.py`):
   - Strategy 1: confirmed annotations as direct bboxes (most accurate)
   - Strategy 2: pixel analysis (counts bright cells right-to-left)
     - Single-slot rows (Deflector=1, Engines=1, etc.) always use profile count exactly
   - Strategy 3: learned layouts (anchors.json)
   - Strategy 4: default calibration anchors

5. **Icon matching** (`icon_matcher.py`):
   - Template matching (session examples) → HSV histogram k-NN → local PyTorch EfficientNet → HF ONNX fallback
   - `MIN_ACCEPT_CONF = 0.40`
   - `SLOT_VALID_TYPES` dict enforces console/weapon type constraints

6. **Write to build** via `slot_equipment_item` / `slot_trait_item`

### WARP dialog ship selection (`warp_dialog.py`)

After recognition, auto-selects ship from `cache.ships`:
- Word-subset match on `r.ship_type` against cache keys
- Calls select_ship logic: `exec_in_thread` (from `src.widgets`), `align_space_frame`, `_save_session_slots`/`_restore_session_slots`
- Tier set from OCR result (`T6-X2` etc.)

### ShipDB boff seating disambiguation

- `extract_boff_seating_from_annotations()` — groups `Boff *` annotations by y-proximity (≤10px = same row)
- `score_ship_boff_match()` — Jaccard similarity between detected profession set and ship's `boffs` field

---

## WARP CORE UI (`trainer_window.py` + `annotation_widget.py`)

### Layout

```
┌──────────────────┬──────────────────────────┬───────────────────────┐
│   LEFT PANEL     │      CENTER PANEL         │     RIGHT PANEL       │
│   min 400px      │      min 400px            │     min 400px         │
│                  │                           │                       │
│  Screenshots     │  ┌─────────────────────┐  │  Recognition Review   │
│  (file list)     │  │   SCROLL AREA       │  │  (review list)        │
│                  │  │   + CANVAS          │  │                       │
│                  │  │   (AnnotationWidget)│  │  [+ Add BBox] [- Rm]  │
│  progress bar    │  └─────────────────────┘  │                       │
│                  │  ┌─────────────────────┐  │  ☐ Auto ≥ [0.75]     │
│                  │  │   BOTTOM PANEL      │  │  [ Accept (Enter) ]   │
│                  │  │   Slot / Item / Acc │  │                       │
│                  │  └─────────────────────┘  │                       │
└──────────────────┴──────────────────────────┴───────────────────────┘
```

Splitter initial sizes: `[400, 700, 400]`

### Canvas (`annotation_widget.py`) features

**Zoom (Gwenview-style):**
- Image loads at 1:1 if it fits viewport; scales down to fit if larger (`min(1.0, min(vp_w/pw, vp_h/ph))`)
- `_fit_scale` computed in `_compute_transform()` from parent (viewport) size — updated on viewport resize
- `_user_scale = None` → fit-to-window; `_user_scale = float` → explicit zoom
- `setWidgetResizable(False)` on scroll area — widget grows beyond viewport in zoom mode → scrollbars appear
- `sizeHint()`: returns viewport size in fit mode, `image × scale` in zoom mode
- Ctrl+wheel: zoom in/out anchored to cursor; `WarpCoreWindow` global filter forwards wheel from scroll area padding to canvas (no click needed)
- Viewport resize event filter on parent: calls `_compute_transform` + `adjustSize` so fit-to-window adapts on window resize

**Modifier key cursors (IMPORTANT — known pitfall):**
- **DO NOT use `widget.setCursor()`** for Ctrl/Alt/Shift cursor changes — it only works when mouse is physically over that widget
- Use `QApplication.setOverrideCursor()` / `restoreOverrideCursor()` — applies globally regardless of mouse position
- Helpers: `_set_mod_cursor(cursor)` and `_clear_mod_cursor()` with `_mod_cursor_active` flag to avoid stacking
- `enterEvent` re-applies mod cursor if modifier is still held; clears stale override if no modifier held

**Alt+LMB draw:**
- Hold Alt over canvas → cursor changes to colored crosshair (`DRAW_BBOX_COLOR`)
- Alt+LMB drag → draws new bbox, triggers icon matching, auto-accept if conf ≥ threshold
- `enterEvent`/`leaveEvent` manage mod cursor
- Global `QApplication.installEventFilter` for Alt/Ctrl/Shift key detection

**Color constants (change one value to update all):**
```python
DRAW_BBOX_COLOR = QColor(255, 200, 0)  # bbox rect + fill + crosshair cursor
```

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Enter | Accept current item |
| Del / Backspace | Remove selected bbox (canvas or review list) |
| Alt+A | Toggle Add BBox mode |
| Alt+R | Remove selected bbox |
| Alt+LMB drag | Draw new bbox directly |
| Ctrl+wheel | Zoom 1×–6× anchored to cursor |

### Auto-accept

- Checkbox `☐ Auto ≥ [0.75]` persisted via `QSettings` (`warp_core/auto_accept_enabled`, `warp_core/auto_accept_conf`)
- `_apply_auto_accept()` called before list draw — marks high-conf items as confirmed in-place
- Also triggers after Add BBox matching and after Auto-Detect recognition
- Completer selection (picking from dropdown) auto-accepts immediately — no Enter needed

### Duplicate bbox warning

When confirming an item, checks if bbox overlaps (>70%) any existing confirmed bbox of a **different** slot → shows `QMessageBox.warning`.

---

## Screen types

| Type | Slot group |
|------|-----------|
| `SPACE_EQ` | Space equipment + Ship Name/Type/Tier |
| `GROUND_EQ` | Ground equipment |
| `TRAITS` | Space + Ground traits |
| `BOFFS` | Bridge officer abilities |
| `SPECIALIZATIONS` | Primary/Secondary specialization |
| `SPACE_MIXED` | All space slots |
| `GROUND_MIXED` | All ground slots |
| `UNKNOWN` | All slots |

---

## Console placement rules

| Item type | Allowed slots |
|-----------|--------------|
| Universal Console | Universal, Tactical, Engineering, Science |
| Tactical Console | Tactical, Universal |
| Engineering Console | Engineering, Universal |
| Science Console | Science, Universal |

`SLOT_VALID_TYPES` in `warp_importer.py` enforces this at recognition time.

---

## Slot counts per tier

- T6-X: +1 Universal Console slot
- T6-X2: +1 Device slot, +1 Starship Trait slot
- Fleet variants: +1 console vs base ship

---

## Known issues / pending features

1. **Fore/aft weapon cross-validation** — not enforced by WARP (fore-only weapons could land in aft slots)
2. **Boff rank in MIXED screens** — unknown; abilities from multiple seats share similar y-coords
3. **Direct slot-scoped autocomplete** — annotation widget shows all items for slot group, not filtered by exact slot type
4. **Cloudflare blocks stowiki** — cargo data falls back to GitHub cache (`STOCD/SETS-Data`)
5. **`cloudscraper`/`curl_cffi`** — in `pyproject.toml` but never imported

---

## Logging

```python
from src.setsdebug import log
log.info('message')   # appears in SETS log panel
log.debug('...')
log.warning('...')
```

All WARP CORE logs are prefixed with context (e.g. `WarpImporter:`, `LayoutDetector:`, `AW.zoom`).

---

## Important patterns

### Adding a new slot type
1. Add to `SLOT_GROUPS` in `trainer_window.py`
2. Add to `SLOT_VALID_TYPES` in `warp_importer.py`
3. Add to `SLOT_MAP` in `warp_dialog.py`
4. Add to `_SPACE_EQ_LABELS` or `_GROUND_EQ_LABELS` in `text_extractor.py`

### Training data flow
```
User confirms bbox in WARP CORE
→ TrainingDataManager.add_annotation() → annotations.json + crop PNG
→ Train Model → LocalTrainWorker fine-tunes EfficientNet
→ icon_classifier.pt saved to warp/models/
→ icon_matcher.py loads .pt on next match
```

### Ship selection from WARP result
```python
# In warp_dialog.py _apply_to_sets()
from src.callbacks import _save_session_slots, _restore_session_slots, align_space_frame
from src.widgets import exec_in_thread
# set button text, load image async, populate tier combo,
# call align_space_frame(sets, ship_data, clear=False)
```

---

## sets-warp-backend

Companion FastAPI service deployed on Render. Source: `/home/raman/PycharmProjects/sets-warp-backend/`.

### Purpose
Community pHash knowledge base — separate from the HF training-crop pipeline.

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `POST /contribute` | Receive crop PNG + label from WARP clients |
| `GET /knowledge` | Serve merged `knowledge.json` (phash → item_name) |
| `POST /admin/merge` | Merge contributions → knowledge.json (requires `X-Admin-Key`) |

### Storage
HF Dataset `sets-sto/warp-knowledge`:
- `contributions/YYYY-MM-DD/<uuid>.json` + `.png` — raw per-user contributions
- `knowledge.json` — merged, approved knowledge base (majority-vote per phash)

### Admin merge
```bash
# Dry-run (raport bez zapisu):
cd /home/raman/PycharmProjects/sets-warp-backend
/home/raman/PycharmProjects/sets-warp/.venv/bin/python admin_merge.py

# Apply (zapisuje knowledge.json do HF):
/home/raman/PycharmProjects/sets-warp/.venv/bin/python admin_merge.py --apply --min 1
```

Credentials in `.env`: `HF_TOKEN`, `HF_REPO_ID=sets-sto/warp-knowledge`, `ADMIN_KEY`.

### Two separate HF channels
| Channel | Repo | Token location | Purpose |
|---------|------|----------------|---------|
| pHash knowledge | `sets-sto/warp-knowledge` | Render env var only | Community overrides for icon_matcher |
| Training crops | `sets-sto/sto-icon-dataset` | `warp/hub_token.txt` | EfficientNet fine-tuning data |

`WARPSyncClient` (`warp/knowledge/sync_client.py`) talks to the Render backend.
`SyncWorker` (`warp/trainer/sync.py`) uploads directly to HF.

---

## Changes made in this development session (2026-03-20)

### Files modified
| File | Change |
|------|--------|
| `warp/trainer/trainer_window.py` | WARP CORE main window — UI fixes + sync logging |
| `warp/trainer/annotation_widget.py` | Canvas widget — zoom, cursors, bbox colors, selection |
| `warp/trainer/sync.py` | Added `_slog` logging for HF sync milestones |
| `warp/recognition/layout_detector.py` | Fixed pixel_count=1 for multi-slot rows |
| `warp/warp_importer.py` | Fixed ShipDB crash when `name`/`type` field is list |

---

### trainer_window.py changes

#### Focus management — Item Name field
- `returnPressed` → `QTimer.singleShot(0, self._review_list.setFocus)` — deferred focus return after all signals settle
- `_on_completer_activated` → calls `_on_accept()` immediately then `setFocus` on review list
- Selecting from dropdown = instant confirm, no Enter needed

#### Delete key — unified across canvas and review list
- `_ann_widget.installEventFilter(self)` added after widget creation
- `eventFilter` extended: `obj in (rl, aw)` — Delete/Backspace triggers `_on_remove_item()` from either widget
- Guarded with `getattr` to avoid `AttributeError` during UI build

#### Ctrl+wheel zoom — no click required
- `QApplication.instance().installEventFilter(self)` in `_setup_shortcuts`
- `removeEventFilter` in `WarpCoreWindow.closeEvent`
- `eventFilter` intercepts `QEvent.Type.Wheel` at app level
- Checks if mouse is globally over scroll_area rect before forwarding to `ann_widget.wheelEvent`
- Single handler — `scroll_area.installEventFilter` removed to avoid duplicate firing

#### Panel widths
- Left panel: `setMinimumWidth(400)`
- Center panel: already `setMinimumWidth(400)`
- Right panel: already `setMinimumWidth(400)`
- Splitter initial sizes: `sp.setSizes([400, 700, 400])`

#### README.md
- Removed `.exe` installer reference (doesn't exist)
- Removed `install.sh` reference (doesn't exist)
- Correct install: `sets_warp.sh` (Linux/macOS), `sets_warp.bat` (Windows)

---

### annotation_widget.py changes

#### Alt+LMB draw bbox
- `mousePressEvent`: Alt held → `_drawing=True`, `_alt_draw=True`, `setCursor(_make_draw_cursor())`
- `mouseReleaseEvent`: if `_alt_draw` → emit `annotation_added`, reset `_alt_draw`, `unsetCursor()`
- In `trainer_window.py`: `_on_bbox_drawn` routes `_alt_draw` through same path as Add BBox button

#### Alt cursor (colored crosshair)
- `DRAW_BBOX_COLOR = QColor(255, 200, 0)` — single constant controls rect color + fill + cursor
- `_make_draw_cursor()` — 12×12px pixmap, 2px pen, hotspot at center
- `enterEvent` — if Alt held on entry → show draw cursor
- `leaveEvent` — if not drawing → `unsetCursor()`
- `showEvent`/`hideEvent` → `QApplication.installEventFilter(self)` / `removeEventFilter`
- `eventFilter` — intercepts global Alt keypress/release when mouse is over canvas rect
- `mouseMoveEvent` — checks `QApplication.queryKeyboardModifiers()` before resetting cursor; Alt held → preserve draw cursor

#### Draw bbox color
- Pen, fill, and cursor all use `DRAW_BBOX_COLOR` — change one constant to update all three

#### item_deselected signal
- Added `item_deselected = Signal()` back (was missing from edited version)
- Emitted when user clicks empty area on canvas

#### Badge text removed
- `_draw_review_item` no longer draws name/slot text next to bbox
- Info shown via tooltip in review list only

#### Zoom — Gwenview-style
- Single `_scale` state (no separate `_zoom` multiplier)
- `_user_scale: float | None` — `None` = fit-to-window, `float` = explicit zoom
- `_fit_scale: float` — computed **once** at `load_image()` from viewport size, never changes
- `_compute_transform()`:
  - `user_scale=None` → `_scale = _fit_scale`, centered offsets
  - `user_scale=float` → `_scale = _user_scale`, offsets=0 (set by wheelEvent)
- `wheelEvent`:
  - Uses `_fit_scale` as base (stable, not recomputed from growing widget)
  - Min = `_fit_scale`, Max = `_fit_scale * 6.0`
  - Snaps to fit-to-window when `new_s <= fit_s * 1.001`
  - Anchor: image point under cursor stays fixed during zoom
  - Maps cursor from viewport coords to widget coords via `self.mapFrom(vp, QPoint(...))`
  - `adjustSize()` after zoom → scroll area updates scrollbars
- `sizeHint()` → returns `pixmap * _scale` so scroll area knows widget size
- `resizeEvent` → `_compute_transform()` + `update()` (fit-to-window adapts to window resize)
- `setMouseTracking(True)` + `setFocusPolicy(StrongFocus)` in `__init__`

#### Keyboard
- `keyPressEvent`: Delete removes selected annotation; Alt handling moved to `eventFilter`
- `keyReleaseEvent` removed (Alt handled globally)

---

### layout_detector.py changes

#### pixel_count floor fix
- **Bug:** `_count_icons_in_row` scans right-to-left; STO fills slots left-to-right → empty slots on the right stop the scan early → `pixel_count=1` for all multi-slot rows
- **Fix:** changed `min(max(pixel_count, 1), profile_count + 1)` to `min(max(pixel_count, profile_count), profile_count + 1)` — ShipDB profile is now the floor, pixel_count can still exceed profile by 1 (T6-X extra slots)

---

### warp_importer.py changes

#### ShipDB._load list crash fix
- **Bug:** some entries in `ship_list.json` have `name`/`type` as a list → `str.strip()` crashed
- **Fix:** `(' '.join(v) if isinstance(v, list) else str(v)).strip()` for both fields

---

### sync.py changes

#### _slog logging
- Added `from src.setsdebug import log as _slog` (SETS log panel)
- Logs: confirmed crop count, daily rate-limit counter, existing HF hash count, each upload (slot + name), final summary

### trainer_window.py (_auto_sync)

#### Logging + error handling
- Logs start of upload, per-file progress (debug), and final OK/BŁĄD
- Bare `except: pass` replaced with `except Exception as e: log.warning(...)`
- `finished` connection moved to separate `_on_sync_finished` method
