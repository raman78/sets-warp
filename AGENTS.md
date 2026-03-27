# SETS-WARP — AI Assistant Context

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

## Key architecture decisions

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
   - Word-subset match; multiple candidates ranked by boff seating similarity (Jaccard) then fewest extra words
   - Fuzzy match (cutoff 0.68)
   - Keyword-based fallback profile

3. **Confirmed annotations** (from `annotations.json`):
   - Override slot counts (confirmed = authoritative)
   - Supply exact ground-truth bboxes → bypass pixel analysis
   - Provide ship name/type/tier when OCR unavailable
   - Extract boff seating for ship disambiguation

4. **Layout detection** (`layout_detector.py`):
   - Strategy 0: CNN Layout Regressor (MobileNetV3, `layout_regressor.pt`)
   - Strategy 1: confirmed annotations as direct bboxes
   - Strategy 2: pixel analysis (counts bright cells right-to-left)
   - Strategy 3: learned layouts (anchors.json, pixel-score ranked)
   - Strategy 4: default calibration anchors

5. **Icon matching** (`icon_matcher.py`):
   - Template matching (session examples) → HSV histogram k-NN → local PyTorch EfficientNet → HF ONNX fallback
   - `MIN_ACCEPT_CONF = 0.40`
   - `SLOT_VALID_TYPES` dict enforces console/weapon type constraints

6. **Write to build** via `slot_equipment_item` / `slot_trait_item`

### WARP dialog ship selection (`warp_dialog.py`)

After recognition, always selects or clears ship:
- Word-subset match on `r.ship_type` against cache keys → `align_space_frame`
- If no ship type detected → `clear_ship()`

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
│  progress bar    │  └─────────────────────┘  │  [Layout View OFF]    │
│                  │  ┌─────────────────────┐  │                       │
│                  │  │   BOTTOM PANEL      │  │  ☐ Auto ≥ [0.75]     │
│                  │  │   Slot / Item / Acc │  │  [ Accept (Enter) ]   │
│                  │  └─────────────────────┘  │                       │
└──────────────────┴──────────────────────────┴───────────────────────┘
```

Splitter initial sizes: `[400, 700, 400]`

### Canvas (`annotation_widget.py`) features

**Zoom (Gwenview-style):**
- `_fit_scale` computed ONCE at `load_image()` from viewport size — never changes
- `_user_scale = None` → fit-to-window mode
- `_user_scale = float` → explicit zoom, widget expands, scroll area shows scrollbars
- Ctrl+wheel: zoom in/out anchored to cursor position

**Alt+LMB draw:**
- Hold Alt over canvas → cursor changes to colored crosshair (`DRAW_BBOX_COLOR`)
- Alt+LMB drag → draws new bbox, triggers icon matching, auto-accept if conf ≥ threshold

**Shift+LMB move/resize:**
- Hold Shift and drag a confirmed bbox → repositions it in-place
- Drag a corner handle → resizes it

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Enter | Accept current item |
| Del / Backspace | Remove selected bbox (canvas or review list) |
| Alt+A | Toggle Add BBox mode |
| Alt+R | Remove selected bbox |
| Alt+LMB drag | Draw new bbox directly |
| Shift+LMB drag | Move / resize existing bbox |
| Ctrl+wheel | Zoom 1×–6× anchored to cursor |

### Auto-accept

- Checkbox `☐ Auto ≥ [0.75]` persisted via `QSettings`
- `_apply_auto_accept()` called before list draw — marks high-conf items as confirmed in-place
- Also triggers after Add BBox matching and after Auto-Detect recognition
- Completer selection (picking from dropdown) auto-accepts immediately

### Layout View (CNN debug overlay)

- `Layout View ON/OFF` toggle in right panel — runs Strategy 0 (CNN regressor) on the current screenshot and renders predicted slot positions as a ghost overlay on the canvas
- Requires `layout_regressor.pt` to be trained first (via Train Model → P4 CNN tab)

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

1. **Fore/aft weapon cross-validation** — not enforced by WARP
2. **Boff rank in MIXED screens** — unknown; abilities from multiple seats share similar y-coords
3. **Direct slot-scoped autocomplete** — annotation widget shows all items for slot group, not filtered by exact slot type
4. **Cloudflare blocks stowiki** — cargo data falls back to GitHub cache (`STOCD/SETS-Data`)

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
| `GET /model/version` | Model metadata (version, trained_at, n_classes, val_acc) |
| `POST /admin/merge` | Merge contributions → knowledge.json (requires `X-Admin-Key`) |

### Two separate HF channels
| Channel | Repo | Purpose |
|---------|------|---------|
| pHash knowledge | `sets-sto/warp-knowledge` | Community overrides for icon_matcher |
| Training crops | `sets-sto/sto-icon-dataset` | EfficientNet fine-tuning data |

`WARPSyncClient` (`warp/knowledge/sync_client.py`) talks to the Render backend.
`SyncWorker` (`warp/trainer/sync.py`) uploads directly to HF.
