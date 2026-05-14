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

## CORE ARCHITECTURAL RULE

**WARP = detection. WARP CORE = trains WARP. `annotations.json` = training data ONLY.**

WARP must NEVER use `annotations.json` as direct import output. If WARP falls back to
reading user-confirmed ground truth instead of performing detection, we:
- Hide real detection bugs behind seemingly-good recognition results
- Cannot measure actual recognition quality
- Defeat the whole purpose of improving the detector

Only **WARP CORE** (the trainer) reads annotations — to display for user review and
feed back into training data for the EfficientNet / MobileNetV3 models.

**Enforcement:** `_use_confirmed = _is_trainer_call` in `warp_importer.py:~763`.
The old condition `'MIXED' in build_type or _is_trainer_call` was disabled on
2026-04-19 by user request. Old line preserved as comment — do NOT re-enable
without explicit user approval.

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
   - Strategy 2: pixel analysis — OCR-anchored EQ geometry (see below)
     - Single-slot rows (Deflector=1, Engines=1, etc.) always use profile count exactly
   - Strategy 3: learned layouts (anchors.json)
   - Strategy 4: default calibration anchors

### EQ panel geometry detector (`eq_geometry.py`)

Single source of truth for the 6-cell × N-row equipment matrix used by
SPACE/GROUND/MIXED equipment pipelines. Pure detection — no annotations.json
access, no GT look-up.

`detect_eq_geometry(img) → EQGeometry | None` returns:
- `panel_x_start`, `panel_right`, `final_dx`, `row_pitch`, `row_cys`, `mode`

Pipeline:
1. EasyOCR full image → label tokens
2. Classify tokens (fore/aft/weapons/consoles/...) + 2-line composite pairing
3. **X-cluster canonical-named hits** by x1; keep the LARGEST group as the EQ
   column. Discards off-panel labels (HUD "Shields", specialization
   "Miracle Worker", etc.)
4. `detect_stripe_start` (HSV gradient) per label → `panel_x_start` (median)
5. `row_pitch` = median of cy-gaps / canonical-step-count between EQ-column hits
6. `est_dx = row_pitch × 0.725`
7. `_detect_right_edge_adaptive_bg` (RTL adaptive-bg scan, requires 2
   consecutive bright columns above `min(col_means) + 12`) on canonical
   single-slot rows (Deflector / Engines / Warp Core / Shields) → `panel_right`
8. Math fallback when no single-slot row scan succeeds:
   `panel_right = panel_x_start + 6 × est_dx`
9. `final_dx = (panel_right − panel_x_start) / 6`
10. `row_cys` from EQ-column OCR hits + linear interpolation between
    consecutive cys (no extrapolation beyond first/last anchor)

**`DX_RATIO = 0.725`** — derived from statistical measurement across 38
GT-annotated screens (median ratio of GT dx to OCR-derived row_pitch).
Stdev ~0.03. Used only as default multiplier for `est_dx` — `final_dx`
is recomputed from the pixel-detected `panel_right` whenever possible.
Replaces an older `row_pitch / 1.5 + 3` formula which underestimated
dx by ~1.3 px/cell on big-icon panels.

Wired into `LayoutDetector`:
- `_get_eq_geometry(img)` — per-image cache keyed by `id(img)`, shares one
  OCR run across all callers
- `_find_panel_right_edge(img)` — uses `geom.panel_right` when EQ labels
  found, falls back to brightness-histogram (`_find_panel_right_edge_brightness`)
- `_detect_via_pixel_analysis(img, slot_order, profile)` — uses
  `geom.row_cys` (one per slot, top→bottom), `geom.final_dx` as cell width,
  `geom.row_pitch × 0.85` as icon height. Falls back to
  `_detect_via_pixel_analysis_legacy` (row-separator brightness scan) when
  geometry detection fails
- **SPACE_MIXED / GROUND_MIXED Strategy 1**: `_detect_via_pixel_analysis`
  (EQ grid) + `_detect_traits_via_ocr` (5-column trait grid using
  `geom.final_dx`/`geom.row_pitch`) + BOFF marker/in_mixed merge.
  Single EQ source of truth shared with SPACE_EQ/GROUND_EQ — eliminates
  the older divergent cluster-vote `panel_right` in
  `_detect_via_ocr_anchored`, which is retained as Strategy 1a fallback
  for cases where `geom` is unavailable.

BOFF and trait detection paths (Strategy 0 marker/grid detectors) do NOT
go through `_detect_via_pixel_analysis` and are unaffected.

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
| Alt+D | Toggle Mark Done / Back to Edit |
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

**"Logging" always means both:** writing to the log file **and** printing to the terminal. Never log to only one destination. Always use `src.setsdebug.log` — do NOT use `logging.getLogger(__name__)` in WARP code, as that bypasses the SETS log panel and terminal output.

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

## GitHub Actions workflows

| File | Trigger | Purpose |
|------|---------|---------|
| `release.yml` | `push: tags: v*` | Creates GitHub Release from tag |
| `build_installer.yml` | `push: tags: v*` | Builds Windows `.exe` installer and attaches to release |

**Release flow**: push tag `vX.Y` → both workflows fire simultaneously → release created + installer built and attached.

**Tag format**: `vMAJOR.MINOR` — no `b` suffix (beta phase ended at v2.0). Examples: `v2.8`, `v2.9`, `v3.0`.

**Known pitfall**: `build_installer.yml` previously triggered on `release: published`. This does NOT work when the release is created by another workflow using `GITHUB_TOKEN` — GitHub blocks cross-workflow event propagation with default tokens. Changed to `push: tags` to trigger directly.

**Creating a release**:
```bash
git tag v2.8 && git push origin v2.8
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

## Changes made in this development session (2026-05-07)

### WARP CORE re-detection — proposes ALL detected positions, not just confirmed

When the user confirms some annotations and runs autodetect again, the trainer
must re-propose freshly-detected positions on top of the user's confirmations,
not erase them. Previously, three coupled bugs caused detected-but-not-yet-
confirmed bboxes to silently disappear — fixed in `warp_importer.py`:

| Bug | Symptom | Fix |
|---|---|---|
| `layout.update(confirmed_layout)` replaced detected per-slot bboxes | Deleting a confirmed BOFF ability left the position blank instead of re-proposing | Per-slot IoU≥0.30 merge: confirmed wins on overlap (pixel-perfect), unmatched detected kept (re-proposed), unmatched confirmed appended (user drew outside grid). Helper `_bbox_iou()` added. |
| `max_count = len(confirmed_layout[slot])` capped processing to confirmed-only count | Even after merge, only N confirmed bboxes ran through the matcher | `max_count = len(layout[slot_name])` after merge — merged layout is the authoritative truth |
| `slot_defs_to_process` filtered by `confirmed_layout` only | Slots with 0 confirmations (e.g. user accepted Fore Weapons but not Aft Weapons) were never processed | Union: `set(confirmed_layout.keys()) \| set(layout.keys())` — every slot with any bbox gets processed |

Net effect: in WARP CORE, every autodetect run re-proposes the full detected
grid. Confirmed positions stay confirmed; freshly detected ones land in the
review list as pending. The trainer's `_populate_review_panel` already merges
incoming items with confirmed-by-id annotations from disk, so confirmed and
pending coexist correctly without duplicates.

### Recognition Review — user-friendly BOFF slot names

`warp/recognition/boff_keys.py` gained `pretty_slot()` — single source of
truth for converting dynamic seat keys to readable labels:

- `Boff Seat L[E]_392` → `Boff Engineering`
- `Boff Seat R[T+P]_510` → `Boff Tactical+Intelligence`
- `Boff Seat L[U]_478` → `Boff Universal`
- non-seat keys pass through unchanged

Applied in `trainer_window._add_review_row` (label + tooltip),
`_enter_manual_bbox_mode` (status hint), the rematch redraw at line ~2177,
and `annotation_widget` canvas hover tooltip.

### Canvas-click slot combo sync

`_on_item_selected` mapped raw seat keys (`Boff Seat L[E]_392`) into the slot
combo via `setCurrentText`, but the combo only contains canonical entries
(`Boff Engineering` etc.) — `setCurrentText` silently failed and the combo
stayed stale. Now uses `parse_seat_profession`/`parse_seat_spec` to derive
the canonical label (base prof preferred, spec as Universal+spec fallback).

---

## Changes made in this development session (2026-04-25)

### BOFF detection — honest baseline + 2 small wins + A1 fix

| File | Change | Commit |
|------|--------|--------|
| `warp/warp_importer.py` | **A1 fix**: gate `seed_from_training_data` in `_get_matcher` on `_from_trainer`; non-trainer path also calls `reset_ml_session()`. Closes second CORE-RULE violation flagged in 2026-04-20 memory. | `e979114` |
| `warp/recognition/layout_detector.py` | **Knob B** (`_classify_boff_profession`): `mid_blue >= 60` alone routes to temporal. HSV classifier 84.7% → 85.8%. | `e979114` |
| `warp/recognition/layout_detector.py` | **H1** (`_detect_boffs_in_mixed`): reject candidates with `> 20` bboxes (BOFF panel physical max). NOT yet e2e-verified — measure on next run, revert if regression. | `e979114` |
| `tests/diag_boff_*.py` | 4 new read-only diagnostics (HSV, sweep, e2e, coverage). Output → `tests/_diag_out/` (gitignored). | `de3c780` |
| `tests/test_*.py` | Reorganization: moved `warp/test_*.py` → `tests/`. NOTE: `test_boff_logic.py` has its own broken HSV classifier (pre-existing, flagged for removal). | `de3c780` |

### Honest baseline (34 screens, 442 GT BOFF abilities, IoU≥0.30)

- **HSV classifier accuracy**: 85.8%
- **Detector seat coverage**: 74.0%
- **Matcher ceiling (with GT bbox)**: 57.2%
- **End-to-end slot accuracy**: 33.5%

The old 2026-04-18 number (39%) was wrong — the test harness had a broken
MockApp that disabled template matching, so the matcher ran ML-only. After
fixing MockApp (`config_subfolders.images` wired), real production reaches
57.2% on perfect bboxes.

### Layout geometry on full screenshots (n=15, iw≥700)

```
panel_w_norm   17–30%, median 22%
panel_h_norm   22–32%, median 27%
icon_w_norm    1.7–3.1%, median 2.1%
structure      2 cols × (3+2) seats × 4 abilities = max 20 bboxes
```

Position varies arbitrarily, but **size and structure are stable per
resolution** — this is what makes the next-step EQ-anchored scan tractable.

### Coverage breakdown (115 uncovered GT)

| Class | GT | % of GT |
|---|---|---|
| near_detector | 58 | 13.1% |
| mirror_layout | 28 | 6.3% |
| middle_image | 16 | 3.6% |
| panel_crop | 13 | 2.9% |

`near_detector` and `mirror_layout` are the same root cause: 34% left/right
strip heuristic is too rigid when panel is shifted, mirrored, or in middle.

### Next iteration plan: EQ-anchored sliding-window scan

1. Use detected EQ as **negative anchor** (BOFFs can't overlap with EQ region)
2. Use EQ icon size to **scale** expected BOFF window (~22% × 27% of image)
3. Sweep BOFF-sized window over non-EQ region
4. Score by `(n_bboxes_in_4-icon-rows) × (grid_fit_to_2×(3+2)×4)` − penalty(>20 bboxes)
5. Window with highest score = BOFF panel

**Prototype path**: `tests/diag_boff_scan.py` (next session) using **GT EQ
bboxes** as anchor — isolates "is the concept right?" from "is EQ detector
good enough?". Production must use detected EQ (CORE RULE).

---

## Changes made in this development session (2026-04-18)

### BOFFS detection — stabilization + ModelUpdater reliability

| File | Change | Commit |
|------|--------|--------|
| `warp/trainer/model_updater.py` | `urllib` → `requests` with `(connect=5, read=15)` timeouts; network error → no timestamp save (retry next start); server "no model" → save timestamp (rate-limit). Watchdog warning if check >60s. | `e459957` |
| `warp/recognition/layout_detector.py` | **Removed rightward-preference** in `_detect_boffs` template slide — was shifting correct peaks by +32px on right column. | `e459957` |
| `warp/recognition/layout_detector.py` | Per-column band scan + merge (captures Pilot rows with icons only in one column). | `e459957` |
| `warp/recognition/layout_detector.py` | Band scoring by **peak − gap** (4 icons vs 3 gaps) + leading-gap bonus for deep dark cliff before first icon. | `e459957` |
| `warp/recognition/layout_detector.py` | Narrower sub-region crop in `_detect_boffs_in_mixed` (`panel_w = w*0.34`) — prevents dilution from adjacent content. | `e459957` |
| `warp/recognition/layout_detector.py` | **BOFF-in-MIXED tiebreak**: `sort((slot_groups, item_count))` instead of raw count. Traits panel lumped all items into one profession (16-count Boff Science) was beating real BOFFs. | `8bf6072` |

### Regression baseline (2026-04-18, 15 screenshots ≥8 BOFFs, w ≥800px)

- **106/269 (39%) IoU hits ≥ 0.3**
- 3 GOOD (≥85%): broadside.png, Chronos-broadside.png, image10.png
- 2 MOSTLY (50-85%): Nautilus.png, Screenshot_20260310
- 5 PARTIAL (<50%)
- 5 WRONG_REGION: empire, image.png, Screenshot_2026-01-19, Screenshot_96, Yeetus
- Ambassador-broadside.png: **+8** (0→8) thanks to tiebreak fix

### Research: what NOT to do next

Investigated and REJECTED (documented to avoid re-exploring):

- **OCR anchoring (Plan D)**: EasyOCR finds 0 BOFF keywords on Ambassador-broadside.png and image.png — STO rank labels too small/stylized. Would fix only 1/4 WRONG_REGION.
- **HSV badge detection (Plan G)**: `_classify_boff_profession` has **93% false-positive rate** on non-BOFF crops (53/57 Ambassador equipment/traits classify as a profession). Method is a classifier, not a detector.
- **Canonical layout per aspect bucket (Plan H)**: BOFF panel position in MIXED is not anchored to edges (normalized-y varies 0.178–0.550 across full-screen shots). Bucketing not useful.

### Remaining failure classes (each needs its own approach)

1. **BOFFs in middle of image** (Screenshot_2026-01-19 — x=337-593 on w=1065): outside 34% left/right strips. Fix: add middle strip or dynamic x-range from content.
2. **BOFFs right of equipment** (Pumwl1, empire, Screenshot_96): STO mirror UI. Detector finds them on RIGHT but IoU<0.3 — likely off-by-pixels in offset handling when using non-square MIXED dims.
3. **Traits win tiebreak** (image.png: L=16/1g, R=20/3g → RIGHT=traits wins by diversity). Need distinguishing signal: BOFF row has 2-3 rows × 4 icons max; traits has different shape.

### Key spatial insight (for future work)

- **9/13 screenshots**: BOFF left of equipment x-range
- **3/13**: BOFF right of equipment (mirror layout)
- **1/13**: 56px overlap (Yeetus edge case)
- **BOFF and equipment are always in disjoint x-bands** — could use equipment detection (reliable) as exclusion anchor, BUT only helps when our current detector already sees BOFFs on correct side with right dims

---

## Changes made in this development session (2026-04-10)

### Item 12 — Full scan for MIXED + BOFFS (Phase A + B + C)

| File | Change |
|------|--------|
| `warp/recognition/screen_classifier.py` | Added SPACE_BOFFS, GROUND_BOFFS to SCREEN_TYPES |
| `warp/recognition/text_extractor.py` | Detect 'space stations' → SPACE_BOFFS, 'standard away team' → GROUND_BOFFS |
| `warp/recognition/layout_detector.py` | Full-scan infrastructure: module-level helpers + `_detect_via_full_scan()` + `_ocr_section_labels()`; MIXED + BOFFS routed to full scan |
| `warp/recognition/icon_matcher.py` | Added `classify_patch()` public method (ML-only, used by full scan) |
| `warp/warp_importer.py` | Passes `icon_matcher` + `app_cache` for MIXED + BOFFS types; SPACE_BOFFS/GROUND_BOFFS in SLOT_ORDER + upgrade logic; `_write_boffs_to_build` ground path complete |
| `warp/trainer/trainer_window.py` | SPACE_BOFFS/GROUND_BOFFS slot groups + screen type labels |
| `warp/warp_dialog.py` | `_write_boffs_to_build` is_ground=True path — writes to ground boff seats |
| `warp/trainer/training_data.py` | TEXT_LEARNING_SLOTS added |
| `warp/trainer/sync.py` | OCR correction upload support |
| `src/datafunctions.py` | Fix None image field crash in ship_images list |
| `sets-warp-backend/admin_train.py` | SPACE_BOFFS/GROUND_BOFFS in SCREEN_TYPES; CrossEntropyLoss replaces FocalLoss |

### Full scan architecture (layout_detector.py)

```
MIXED / BOFFS detection chain:
  Strategy 1:  Learned layouts (anchors.json) — highest accuracy when present
  Strategy FS: Full scan — OCR labels + EfficientNet dense window + fusion
    Phase B: _ocr_section_labels() — full-image EasyOCR → {slot_name: (cx, cy)}
    Phase C: sliding window stride=icon_est//2, classify_patch() per patch, NMS
    Fusion:  cluster rows by Y → score(row, slot) = 0.65×type_score + 0.35×ocr_score
  Strategy 2+: Existing pixel / canonical / OCR / anchors fallbacks
```

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
