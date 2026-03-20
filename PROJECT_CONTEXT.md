# Project: SETS-WARP

## Purpose

Star Trek Online game build creator with screenshot-based build recognition.

---

## SETS — STO Equipment and Trait Selector

Main tool for build creation. A planning tool for ship and ground builds as well as space and ground skill trees for Star Trek Online, developed by the STO Community Developers.

**Features:**
- Plan space and ground builds on any ship.
- Build without being restricted to owned items.
- Share builds (JSON and PNG format); import shared builds.
- Skill tree planning with free point allocation (not possible in-game).
- Export builds in Markdown format.
- Open the wiki page of any item via context menu.

**Website:** https://stobuilds.com/apps/sets
**GitHub:** https://github.com/STOCD/SETS

---

## WARP — Weaponry & Armament Recognition Platform

Module for detecting Star Trek Online builds from screenshots using ML models. Works fully locally, with optional community knowledge synchronization via a backend server.

---

## Main Assumptions

1. **SETS-WARP is a standalone, self-contained program** — independent from system Python and system libraries. It uses an autoconfigurator (`bootstrap.py`) to manage its own isolated environment.
2. **All data is synchronized via SyncManager** — downloads item icons, ship images, and cargo data from GitHub and stowiki. Note: stowiki is currently blocked by Cloudflare anti-bot protection (active bug; no bypass implemented yet). As a workaround, cargo data falls back to the GitHub cache mirror (`STOCD/SETS-Data`).
3. **All code comments and program communication messages are in English.**

---

## Autoconfigurator — `bootstrap.py`

Entry point for the application. Manages the full lifecycle of the Python environment before launching `main.py`.

**Strategy (in order):**
1. If already running inside the project's `.venv` → launch `main.py` directly.
2. If `.venv` exists but not active → relaunch using the venv Python.
3. If `.venv` does not exist → show installer GUI, then:
   a. Download portable Python into `.python/` (~65 MB, no root, no compile).
   b. Create `.venv` using that portable Python.
   c. `pip install` all dependencies from `pyproject.toml`.
   d. Relaunch with venv Python.

**Portable Python source:** `astral-sh/python-build-standalone`
- Version: Python 3.13.2 (tag `20250317`)
- Supported platforms: Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64

**CLI flags:**
- `--reinstall` — wipe `.venv` and `.python`, force full reinstall.
- `--repair` — health-check / repair the venv without relaunching the app.

---

## SyncManager — `src/syncmanager.py`

Downloads and updates assets from GitHub. Uses the GitHub Tree API (SHA1 + size) to detect changed or missing files, then downloads only what is needed using a bounded thread pool.

**Asset groups (GitHub-backed, from `STOCD/SETS-Data`):**
- `images/` — Item icons (equipment, trait, ability icons)
- `ship_images/` — Ship images
- `cargo/` — Cargo data JSON files (equipment, traits, ships, etc.)

**Wiki-only groups (no GitHub mirror, downloaded on demand):**
- Boff ability icons — suffix `_icon_(Federation).png`
- Skill icons — suffix `.png`

**Download discipline:**
- Max 5 concurrent threads
- 404 → permanent failure, no retry
- 403 → if repeated 3× for the same source, that source is disabled for the session
- Other errors → retried up to `MAX_RETRIES=1` times with `RETRY_DELAY_S=3` pause
- Stall timeout: 10 seconds of no data = abort attempt

**Cargo data fallback chain** (in `Downloader.download_cargo_table`):
1. Try stowiki Cargo API directly (currently blocked by Cloudflare → usually fails)
2. Fall back to GitHub cache (`STOCD/SETS-Data/cargo/<filename>`)
3. If both fail → fall back to local cache (any age); if no cache → `sys.exit(1)`

Cache age: cargo data is re-downloaded after 7 days.

**Known issue:** `cloudscraper` and `curl_cffi` are listed in `pyproject.toml` dependencies but are **not imported or used anywhere in the codebase**. They were likely added in anticipation of a Cloudflare bypass implementation that was never completed. The current `Downloader` uses plain `requests.Session` with a Firefox User-Agent header as the only bot-detection mitigation.

---

## WARP Module Architecture

### Entry Points

- `warp/warp_button.py` — `inject_warp_buttons(sets_app, menu_layout)` called from `app.py` `setup_main_layout()`. Adds two buttons to the SETS top menu bar:
  - **⚡ WARP** — opens the multi-step import dialog
  - **🧠 WARP CORE** — opens the ML trainer window (singleton)

- `warp/__init__.py` — module version `0.1.0`; WARP is gracefully disabled if its dependencies are missing (try/except import guard in `app.py`).

### Main Modules

#### `warp/recognition/`

| File | Responsibility |
|---|---|
| `text_extractor.py` | OCR — extracts ship name, type, tier, and screen type from a screenshot. Two-stage scan: (1) fast partial scan (top-right + bottom) for traits/boffs/spec headers; (2) full-image scan fallback for MIXED layouts. Ship info uses wide top-band (20% height) scan anchored on Tier token (`T6-X2` etc.). Slot labels used as screen type signals: space equipment labels (`Fore Weapons`, `Warp Core`, etc.), ground labels (`Kit Modules`, `Body Armor`, etc.). |
| `screen_classifier.py` | Classifies screenshot type using a two-stage pipeline: (1) PyTorch MobileNetV3-Small (`.pt` native format, fine-tuned locally); (2) session k-NN on HSV histograms as fallback. OCR keyword fallback from `text_extractor.py` is used when both ML stages are uncertain. |
| `layout_detector.py` | Detects icon bounding boxes per slot. Pipeline: (1) confirmed annotations used directly as ground-truth bboxes when available; (2) pixel analysis — counts bright icon cells scanning right-to-left, single-slot rows (Deflector, Engines, Shield, Warp Core) always use profile count exactly; (3) learned layouts (saved anchors with slot_counts); (4) default calibration anchors. |
| `icon_matcher.py` | Matches a cropped icon against the SETS image cache. Pipeline: (1) multi-scale template matching (cv2 TM_CCOEFF_NORMED) with session examples; (2) HSV color histogram k-NN; (3) local PyTorch EfficientNet-B0 (`.pt`, fine-tuned on confirmed crops); (4) HuggingFace ONNX fallback. |

#### `warp/trainer/`

| File | Responsibility |
|---|---|
| `trainer_window.py` | WARP CORE — main trainer UI window (QMainWindow). Review, annotate, and correct WARP recognition results. Features: auto-accept checkbox (threshold-based, persisted via QSettings), keyboard shortcuts (Alt+A add bbox, Alt+R remove, Enter accept, Del remove), two-pass icon matching (slot-restricted pass 1, unrestricted fallback pass 2 when conf < 0.40), duplicate bbox overlap warning on confirm. |
| `annotation_widget.py` | Widget for annotating individual icon crops (confirm / reject / relabel). |
| `training_data.py` | `TrainingDataManager` — manages confirmed annotation crops on disk. Includes `repair_crop_index()` for data consistency, `_sync_crop_index()` for rename/re-export on update. |
| `local_trainer.py` | `LocalTrainWorker` (QThread) — fine-tunes EfficientNet-B0 on confirmed annotations. Saves native PyTorch `.pt` format (replaced ONNX dynamo exporter which produced uniform-output models). |
| `screen_type_trainer.py` | `ScreenTypeTrainerWorker` — fine-tunes MobileNetV3-Small. Saves native PyTorch `.pt` format. |
| `sync.py` | Syncs training data to/from HuggingFace Hub (`SyncWorker`, `HFTokenDialog`). |

#### `warp/knowledge/`

| File | Responsibility |
|---|---|
| `sync_client.py` | `WARPSyncClient` — non-blocking community knowledge sync. Downloads `knowledge.json` (pHash → item_name overrides) from the WARP backend. Uploads confirmed crops (rate-limited to 200/day per installation). |

#### `warp/tools/` (developer utilities, not part of the user-facing app)

| File | Responsibility |
|---|---|
| `scraper.py` | Scrapes cargo data from stowiki and `vger.stobuilds.com`; builds the JSON files used by SETS. |
| `approve_staging.py` | Approves staged community contributions on the backend. |
| `check_db.py` | Validates the knowledge database. |
| `debug_fetch.py` | Debug helper for fetching and inspecting raw wiki responses. |
| `test_pipeline.py` | End-to-end test of the full WARP recognition pipeline. |

### Import Pipeline (`warp/warp_importer.py`)

Full pipeline per screenshot:

1. `TextExtractor` reads ship name, type, and tier from screenshot (OCR runs in WARP dialog mode; skipped in WARP CORE trainer mode via `from_trainer=True` flag).
2. `ShipDB` looks up exact slot counts from `ship_list.json` (783 ships). **Lookup order:**
   a. Exact `type` field match
   b. Word-subset match — OCR words ⊆ DB type words (handles omitted subtype words like `"Fleet Temporal Science Vessel"` → `"Fleet Nautilus Temporal Science Vessel"`); when multiple candidates, ranked by boff seating similarity (Jaccard) then fewest extra words
   c. Standard fuzzy match (cutoff 0.68)
   d. Keyword-based fallback profile
3. Confirmed annotations loaded from `annotations.json` — used to:
   - Override slot counts (confirmed counts are authoritative)
   - Supply exact ground-truth bboxes (bypasses pixel analysis)
   - Provide ship name / type / tier when OCR is unavailable
   - Extract boff seating for ship disambiguation
4. Pixel-count profile refinement (only for slots not covered by confirmed annotations): `_profile_from_pixel_counts()` matches pixel counts against `_KEYWORD_PROFILES` to infer unmeasurable slots (Sec-Def, Experimental, Hangars).
5. `LayoutDetector` finds bounding boxes per slot using constrained profile.
6. `SETSIconMatcher` matches each cropped icon. Items filtered by `SLOT_VALID_TYPES` (type-to-slot compatibility) and `MIN_ACCEPT_CONF = 0.40`.
7. Results written to `sets_app.build` via `slot_equipment_item` / `slot_trait_item`.
8. `sets_app.autosave()` called.

### Dialog Flow (`warp/warp_dialog.py`)

Multi-step QDialog:
1. Select build type (SPACE / GROUND / SPACE_SKILLS / GROUND_SKILLS)
2. Select folder of screenshots
3. Background worker (`_ImportWorker` QThread) runs the pipeline with progress bar
4. Results applied automatically to current SETS build
5. Ship selection: fuzzy-matched ship type → `cache.ships` lookup → `select_ship` logic (button text, image load via `exec_in_thread` from `src.widgets`, tier combo, `align_space_frame`, `_save_session_slots` / `_restore_session_slots`)

### ShipDB Boff Seating (`warp/warp_importer.py` — `ShipDB`)

Two helper methods support ship disambiguation via boff seating:

- `extract_boff_seating_from_annotations(anns)` — groups confirmed `Boff *` annotations by y-proximity (≤10px = same row/seat), counts abilities per type, maps to ShipDB profession strings. Rank inferred from ability count (4=Commander, 3=Lt Commander, 2=Lieutenant, 1=Ensign; -1=unknown for mixed rows).
- `score_ship_boff_match(ship_entry, detected_seats)` — Jaccard similarity between detected profession set and ship's `boffs` field profession set. Returns 0.0–1.0 (0.5 = no data / neutral).

### Community Knowledge Sync (`warp/knowledge/sync_client.py`)

- Backend: `https://sets-warp-backend.onrender.com`
- `knowledge.json` — community-confirmed pHash → item_name overrides, refreshed every 24 hours
- Each installation has a random UUID (`install_id`) used for deduplication only (not for user identity)
- Contributions rate-limited to 200 per installation per day
- All network calls are non-blocking (daemon threads) and silent on failure

---

## Technologies

| Technology | Role |
|---|---|
| Python 3.13.2 | Runtime (portable, via python-build-standalone) |
| PySide6 ≥6.7, <6.10 | GUI framework (all windows, widgets, dialogs) |
| OpenCV (`opencv-python-headless`) | Template matching, image cropping, histogram comparison |
| EasyOCR | OCR for ship name / screen type extraction from screenshots |
| ONNX Runtime | Inference fallback for icon classifier (HuggingFace ONNX model) |
| PyTorch + torchvision | Local training and inference for icon classifier (EfficientNet-B0) and screen classifier (MobileNetV3-Small); native `.pt` format (replaced ONNX dynamo exporter) |
| HuggingFace Hub | Download of pre-trained ONNX models; upload/sync of training data |
| requests | HTTP client for wiki/GitHub downloads |
| cloudscraper, curl_cffi | Listed in dependencies but **not currently used** — intended for future Cloudflare bypass |
| NumPy, SciPy, scikit-image | Image processing support |
| Shapely, pyclipper | Geometric operations (layout detection) |

---

## Known Issues / Active Bugs

1. **Cloudflare blocking stowiki** — `download_cargo_table` falls back to GitHub cache, but the cache may lag behind the live wiki. `cloudscraper` and `curl_cffi` are installed but not wired up.
2. **`cloudscraper` / `curl_cffi` dead imports** — present in `pyproject.toml` but never imported. Either implement the bypass or remove the dependencies.
3. **`requests_html`, `lxml_html_clean`, `cssselect`** — also in `pyproject.toml` but not visibly used in the main codebase (may be used indirectly or are leftovers).
4. **Fore/aft weapon cross-validation gap** — WARP does not pre-filter fore-only weapons (e.g. Dual Heavy Cannons) from Aft slots. SETS handles this at the widget level; WARP relies on icon matcher confidence and type validation only.
5. **Boff rank unknown in MIXED screens** — boff ability rows in MIXED screenshots cannot be reliably split into individual seats (abilities from multiple seats share similar y-coordinates). Rank is inferred from ability count per visual row, which may be incorrect for mixed-type rows. A dedicated BOFFS screen provides accurate rank data via OCR.
6. **Direct slot-scoped filtering in WARP CORE name field** — not yet implemented. The item name autocomplete in the annotation widget shows all items for the slot group, not filtered by the exact slot type rules (e.g. Engineering Console shown when Science Console slot is selected). Pending feature.
