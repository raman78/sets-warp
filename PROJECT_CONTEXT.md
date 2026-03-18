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
| `text_extractor.py` | OCR — extracts ship name, type, tier, and screen type from a screenshot. Detects: SPACE, GROUND, SPACE_TRAITS, GROUND_TRAITS, BOFFS, SPEC. |
| `screen_classifier.py` | Classifies screenshot type. Pipeline: (1) ONNX MobileNetV3-Small → (2) session k-NN on HSV histograms → (3) OCR keyword fallback. |
| `layout_detector.py` | Detects icon bounding boxes per slot. Pipeline: (1) pixel analysis (dark separators) → (2) learned layouts → (3) OCR label positions → (4) default calibration anchors. |
| `icon_matcher.py` | Matches a cropped icon against the SETS image cache. Pipeline: (1) multi-scale template matching (cv2 TM_CCOEFF_NORMED) → (2) HSV color histogram → (3) optional ONNX EfficientNet-B0 from HF Hub. |

#### `warp/trainer/`

| File | Responsibility |
|---|---|
| `trainer_window.py` | WARP CORE — main trainer UI window (QMainWindow). Review, annotate, and correct WARP recognition results. |
| `annotation_widget.py` | Widget for annotating individual icon crops (confirm / reject / relabel). |
| `training_data.py` | `TrainingDataManager` — manages confirmed annotation crops on disk. |
| `local_trainer.py` | `LocalTrainWorker` (QThread) — fine-tunes EfficientNet-B0 (torchvision) on confirmed annotations, exports to ONNX (`warp/models/icon_classifier.onnx`). |
| `screen_type_trainer.py` | `ScreenTypeTrainerWorker` — fine-tunes MobileNetV3-Small on user-corrected screenshots, exports to ONNX (`warp/models/screen_classifier.onnx`). |
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

Full pipeline per screenshot folder:

1. `TextExtractor` reads ship name, type, and tier from screenshot.
2. `ShipDB` looks up exact slot counts from `ship_list.json` (783 ships, fields: `fore`, `aft`, `experimental`, `hangars`, `secdeflector`, `uniconsole`, `consolestac`, `consoleseng`, `consolessci`, `devices`).
3. Fallback: category-based slot profile if ship not found in DB.
4. `LayoutDetector` finds bounding boxes per slot using ship profile to constrain counts.
5. `SETSIconMatcher` matches each cropped icon against the SETS image cache.
6. Results are written to `sets_app.build` via `slot_equipment_item` / `slot_trait_item`.
7. `sets_app.autosave()` is called.

### Dialog Flow (`warp/warp_dialog.py`)

Multi-step QDialog:
1. Select build type (SPACE / GROUND / SPACE_SKILLS / GROUND_SKILLS)
2. Select folder of screenshots
3. Background worker (`_ImportWorker` QThread) runs the pipeline with progress bar
4. Review results — confirm, reject, or relabel each recognised item
5. Apply to current SETS build

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
| ONNX Runtime | Inference for screen classifier (MobileNetV3) and icon classifier (EfficientNet-B0) |
| PyTorch + torchvision | Local training of icon classifier and screen classifier (fine-tuning on user-confirmed data); exported to ONNX after training |
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
