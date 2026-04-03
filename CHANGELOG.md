# CHANGELOG

## v2.6 (2026-04-03) — TEXT_LEARNING_SLOTS; ship_type_corrections pipeline; annotation cleanup

### Feature: Ship Type / Tier OCR correction pipeline (backlog #7)
- `training_data.py`: split `NON_ICON_SLOTS` into `POSITION_ONLY_SLOTS` (Ship Name — position anchor only, no crop) and `TEXT_LEARNING_SLOTS` (Ship Type, Ship Tier — crop PNG + `ml_name` saved and uploaded).
- `sync.py`: include `ml_name` in `annotations.jsonl` entries for `TEXT_LEARNING_SLOTS`.
- `model_updater.py`: download `ship_type_corrections.json` (optional); reload `TextExtractor` corrections after model update.
- `text_extractor.py`: add `_corrections` class var + `load_corrections()` classmethod; apply community OCR corrections to `ship_type`/`ship_tier` after extraction.
- `warp_importer.py`: auto-load `ship_type_corrections.json` on `TextExtractor` init.
- `admin_train.py` (backend): `collect_text_corrections()` aggregates (ml_name, name) pairs per install_id, majority-votes per OCR key, uploads `ship_type_corrections.json` to HF.

### Bug fix: deleted annotations still uploaded (backlog #10)
- `remove_annotation()` now also removes the `crop_index` entry and deletes the crop PNG from disk. Previously, deleted annotations could still appear in `get_confirmed_crops()` and be uploaded by `SyncWorker`.

### Improvement: screen type k-NN seeded on manual confirm
- `trainer_window.py`: `ScreenTypeClassifier.add_session_example()` is called when the user manually confirms a screen type (checkbox or dropdown), improving in-session k-NN accuracy without waiting for the next image.

### Improvement: backend training log headers (backlog #9)
- `admin_train.py`: clear section banners (`── Training screen_classifier ──`) before each model's training block; summary footer after save.

## v2.5 (2026-03-29) — P11 community anchors; OCR fixes; Ship Name privacy

### Feature: P11 — Community anchors.json
- `sync.py` — `_upload_anchors_grid()`: uploads normalized slot bbox grids to HF staging
  after each sync (one JSON per install/screenshot, filtered to icon slots only).
- `admin_train.py` (backend) — `build_community_anchors()`: aggregates grids from ≥3
  distinct contributors, computes per-slot median x/y/w/h/step/count, uploads result as
  `community_anchors.json` to `sets-sto/warp-knowledge`.
- `layout_detector.py` — Strategy 1b: after local anchors miss, tries community anchors
  as fallback before falling through to pixel analysis.
- `model_updater.py` — downloads `community_anchors.json` alongside model files (optional).
- `LayoutDetector.reset_community_anchors_cache()`: invalidates in-memory cache on all live
  instances when ModelUpdater downloads a fresh file.

### Bug fixes: OCR ship type recognition
- **Section header overwrite**: ship type prefix extracted from the tier token
  (e.g. `'Fleet Support Cruiser [T6-X2]'` → `'Fleet Support Cruiser'`) was being
  overwritten by nearby screen headers like `'Personal Space Traits'`. Fixed by guarding
  the above-token block with `if not result['ship_type']`.
- **Split ship name**: when OCR breaks a long ship name across tokens (e.g.
  `'Kardashev Command Dyson Science'` above + `'Destroyer [T6-X2]'` as tier token),
  the above token is now prepended to the prefix — but only after filtering known section
  headers (`traits`, `reputation`, `abilities`, etc.) to avoid the first bug.

### Ship Name privacy (backlog items 1, 7, 8)
- **WARP CORE**: Ship Name field disabled (position-only); placeholder explains it is OCR-read.
  `_configure_name_field()` is the single source of truth for all NON_ICON_SLOT UI state.
- **Annotations**: one-time migration clears stored text from `Ship Name` and `Ship Type`
  annotation entries on first load (names were never meant to be persisted).
- **Import pipeline**: ship_name no longer written to SETS build or displayed in import
  summary — cosmetically private, only bbox position is used for layout anchors.
- **Removed Ship Type annotation fallback** (`_load_ship_info_from_annotations`):
  was a local-only shortcut that bypassed OCR and never fed central training. Removed
  along with `_load_ship_info_from_annotations()` (dead code after Ship Name migration).
  `_load_confirmed_profile()` now returns only confirmed slot counts (feed P11).

### WARP CORE fixes
- `_on_slot_changed`: `NameError: name 'is_tier' is not defined` — stale reference after
  refactoring to `_configure_name_field()`; simplified to `if slot not in NON_ICON_SLOTS`.

---

## v2.4 (2026-03-29) — Remove layout CNN; fix traits dropdown

### Architecture: P10 — layout CNN removed
- **Deleted** `layout_trainer.py`, `layout_dataset_builder.py`, `local_trainer.py` — the
  MobileNetV3-Small layout regressor trained only on local data (5–20 screenshots), never
  reached the central pipeline, and was functionally identical to `anchors.json` (Strategy 1).
- **Removed** Strategy 0 (`_detect_via_cnn`, `infer_build_type`) from `layout_detector.py`.
- **Removed** "Train Layout Model" toolbar action and `_TrainProgressDialog` from `trainer_window.py`.
- **Removed** "Layout View" debug toggle and `set_cnn_debug_items` from `annotation_widget.py`.
- Layout detection: Strategies 1–4 (anchors.json, pixel analysis, OCR, static anchors) unchanged.
- UNKNOWN screen type defaults to SPACE (was CNN-inferred, always fell back to SPACE anyway).
- Community anchors.json pipeline (P11) planned as replacement — privacy-safe bbox grids
  aggregated from all users, distributed via ModelUpdater.

### Bug fix: traits dropdown empty in WARP CORE
- `datafunctions.py` migrates `cache.traits` keys at load time:
  `'traits'→'personal'`, `'rep_traits'→'rep'`, `'active_rep_traits'→'active_rep'`.
- `trainer_window.py` was using the old (pre-migration) key names → `KeyError` → silent
  `except` → empty completer model → no dropdown for Space Reputation, Active Space Rep,
  Personal Space/Ground Traits, Ground Reputation, Active Ground Rep.
- Fixed in `_build_search_candidates` and `trait_slot_map` (6 keys each).

## v2.3 (2026-03-29) — Architecture: central-only training, 15-min model polling

### Architecture change
- **Local icon + screen-type training removed** — icon classifier and screen type classifier
  are now trained exclusively by the central pipeline (`admin_train.py` on GitHub Actions).
  Users contribute confirmed crops via HF sync; central trains on all community data and
  distributes the model back. Local training was redundant and got overwritten by the
  central model anyway.
- **Layout regressor training kept local** (`local_trainer.py`) — layout data requires full
  screenshots which are not uploaded; local-only training remains appropriate here.
- **Central model augmentation parity** (`admin_train.py`): P7 (ColorJitter 0.3/0.3/0.2,
  RandomHorizontalFlip p=0.3, RandomAffine) and P9 (WeightedRandomSampler hard negatives)
  applied to central training — previously only in local trainer.

### Model update polling
- **15-minute polling** replaces 24-hour interval (`model_updater.py`): central model
  checked every 15 min from the HF sync timer in WARP CORE.
- **Removed Point 2** (`warp_button.py`): 15s background check after WARP button click.
- **Removed Point 3** (`trainer_window.py`): ModelUpdater call on WARP CORE open.
- Both removed — model update now driven solely by the 5-min sync timer (checks every
  15 min due to internal rate limiting).
- **Fixed duplicate `_on_sync_timer`** in `trainer_window.py`: dead `_init_sync_client`
  timer and overridden method removed; knowledge refresh merged into main sync timer.

### UI
- **"Train Model" → "Train Layout Model"**: button renamed to reflect its actual scope.
- Augmentation passes dialog removed (was for icon training only).

---

## v2.2 (2026-03-28) — ML roadmap P7, P8, P9

### Improvements
- **P7 — EfficientNet augmentation** (`local_trainer.py`): training transform now includes
  `ColorJitter(0.3, 0.3, 0.2)`, `RandomHorizontalFlip(p=0.3)`, and
  `RandomAffine(degrees=5, translate=(0.05, 0.05))`; validation transform unchanged;
  reduces overfitting on small datasets without collecting more data
- **P8 — Confidence fusion** (`icon_matcher.py`): ML now runs when template score < 0.75
  (was < 0.55); fused confidence = `max(template, 0.4*template + 0.6*ml)` so high-ML /
  borderline-template cases correctly prefer the ML result; both scores logged at DEBUG;
  `FUSION_THRESHOLD = 0.75` constant added
- **P9 — Hard negatives mining** (`local_trainer.py`): after each training epoch, samples
  where `predicted != label AND conf > 0.5` get weight doubled (capped at 3×) in
  `WeightedRandomSampler`; focuses subsequent epochs on the model's hardest mistakes

---

## v2.1 (2026-03-28) — ML roadmap P6, layout inference, UX fixes

### Features
- **P6 — Async bbox matching** (`trainer_window.py`): icon matching during manual bbox
  draw now runs in a background `MatchWorker` thread; UI stays responsive; indeterminate
  `QProgressBar` in a fixed-height frame below canvas appears after 500ms if still running
- **CNN screen-type inference** (`layout_detector.py`): new `infer_build_type(img)` scores
  each build type (SPACE, GROUND, SPACE_TRAITS, GROUND_TRAITS, BOFFS, SPEC) by average
  slot presence from the CNN; used by `RecognitionWorker` for UNKNOWN screenshots and to
  distinguish SPACE_TRAITS vs GROUND_TRAITS (screen classifier returns 'TRAITS' for both)
- **CNN slot filter fix** (`layout_detector.py`): Strategy 0 now maps WarpImporter
  build_types ('SPACE', 'GROUND', …) to the correct `SLOT_GROUPS` keys ('SPACE_EQ',
  'GROUND_EQ', …); previously 'SPACE' fell through to `REGRESSOR_SLOTS` (all 42 slots)
  causing Traits/Reputation bboxes to appear on SPACE_EQ screenshots

### Improvements
- **`RecognitionWorker.run()`** refactored: image loaded once (was loaded twice), explicit
  `_STYPE_MAP` with comments, CNN inference for UNKNOWN/TRAITS before creating WarpImporter
- **Wayland popup fix** (`trainer_window.py`): general handler in `eventFilter` catches
  ALL widgets with `Qt.WindowType.Popup` flag on `QEvent.Show` (covers QComboBox dropdowns,
  QMenu, QCompleter); replaces per-combo per-view approach that caused more errors

---

## v2.0.1 (2026-03-28) — Bug fixes & sync optimization

### Bug fixes
- **updater.py** — `git fetch --tags` added before `git pull` in `_do_git_update()`;
  fixes continuous update popup loop when a tag was created after the commit was already
  on the remote (affected v2.0 release on test installs)
- **trainer_window.py** — Wayland `QCompleter` popup fix: install event filter on popup
  widget, set `windowHandle().setTransientParent()` on first `Show` event so Qt can
  create the `xdg_popup` surface (fixes "Failed to create popup" warning)

### Performance
- **sync.py** — HF upload deduplication: uploaded crop hashes now cached in
  `.sync_uploaded_hashes.json`; `list_repo_files()` called once to bootstrap, then
  skipped on all subsequent 5-minute sync ticks

### Docs
- **warp_ml_roadmap.md** — P2 cross-validation marked as complete (was already
  implemented, roadmap had stale status); duplicate P2 section removed

---

## v2.0 (2026-03-28) — Official release, beta phase complete

First non-beta release. Consolidates v2.0b + v2.1b (see entries below) and adds:

- **`scripts/upstream_sync.py`** — semi-automated upstream SETS merge helper:
  auto-applies 9 documented patches, flags manual review items with instructions
- **`docs/src_patches.md`** — complete re-application guide for future upstream merges
- **Phase 5 upstream PRs** — 3 fixes contributed back to `Shinga13/SETS`:
  item normalization, Intel Holoship uni_consoles, species sets expansion
- **Shift anchor cursors fixed** — `_set_mod_cursor()` now used in `mouseMoveEvent`
  Shift section so handle resize cursors update correctly via `changeOverrideCursor`

---

## v2.1b (2026-03-28) — WARP CORE NON_ICON_SLOT fixes + docs

### WARP CORE — NON_ICON_SLOT annotation workflow fixes (10 commits)

- **Focus protection** — canvas `enterEvent` and `eventFilter` no longer steal focus from
  `QLineEdit`/`QTextEdit`/`QAbstractSpinBox` fields; item name field stays editable while mousing over canvas
- **Canvas hit-test stale after accept** — `refresh_annotations(path)` called after `add_annotation`
  so freshly confirmed bboxes are immediately clickable on canvas
- **Ship Name/Type bbox overwrite** — `add_annotation` step 2 (bbox-coord match) now skips
  NON_ICON_SLOTS; Ship Type could silently overwrite Ship Name when drawn at same position
- **NON_ICON_SLOT completer performance** — `_populate_name_completer` skipped for Ship Name/Type/Tier
  (was iterating all equipment on every canvas click)
- **Canvas clears stale highlight** — switching to a slot with no bbox now calls `clear_highlight()`
- **Slot combo UX** — confirmed NON_ICON_SLOTS hidden from dropdown to prevent duplicates;
  restored on remove; `keep_slot` param in `_refresh_slot_combo` keeps active slot visible when editing;
  auto-advance when drawing new bbox on an already-confirmed NON_ICON_SLOT
- **Remove-bbox dialog** — default button changed to Yes (Enter confirms removal)
- **Diagnostic logs** — DEBUG-level logs added in `add_annotation` and `_on_accept` for tracing

### WARP CORE — Shift anchor cursor fix

- Shift+hover over bbox handles now correctly shows resize cursors (SizeFDiag/SizeBDiag/SizeHor/SizeVer)
- Root cause: `mouseMoveEvent` used `self.setCursor()` which was overridden by the `QApplication`
  override cursor set by `eventFilter` on Shift keypress
- Fix: use `_set_mod_cursor()` (which calls `QApplication.changeOverrideCursor()`) in Shift section
  of `mouseMoveEvent` — same approach as the rest of the modifier cursor system

### Documentation — Phase 4.2

- Created `docs/src_patches.md` — complete reference of every intentional difference between
  our `src/` and upstream SETS, with re-application instructions for future upstream merges

---

## v2.0b (2026-03-28) — Upstream SETS merge (Phase 3 + 4)

### Architecture — Phase 3 + 4: upstream merge + stabilization

Integrated 738 commits from upstream SETS into SETS-WARP. All SETS-WARP features preserved.

**New features inherited from upstream SETS:**
- **Skill tree** — full space and ground skill tree UI (`setup_space_skill_frame`, `setup_ground_skill_frame`)
- **Markdown export** — restructured `src/export.py`, `get_build_markdown`
- **Settings page** — UI Scale slider, Default Save Format, Picker Position, About sidebar with links
- **Picker improvements** — relative position mode, fix for position calculation
- **Legacy build conversion** — improved `convert_old_build` in `buildupdater.py`
- **Modifier data** — refinements throughout `datafunctions.py`
- **Linux path fixes** — `os.path.join` used consistently

**SETS-WARP additions preserved in `src/`:**
- `src/app.py`: `Downloader`/`CargoManager`/`ImageManager` init, Cloudflare cookie setup from `.env`, `cache_item_aliases`, `_set_win32_taskbar_icon`, `dual_cannons` DC icon + N/A placeholder image, `menu_layout`/`settings_scroll_layout`/`settings_scroll_frame` hooks, splash with progress bar, Preferred Backup and Show Startup Summary settings
- `src/widgets.py`: `ImageLabel(QLabel)`, `ShipImage(QLabel)` (QLabel-based with setPixmap), `TooltipLabel`, `alt_images` in Cache, `progress_bar`/`progress_detail` fields
- `src/datafunctions.py`: `SyncManager` integration, `_show_startup_dialog`, `GITHUB_CACHE_URL` fallback, full logging, `get_icon_set`/`build_cache` helpers
- `src/callbacks.py`: `_save/_restore_session_slots`, `log`
- `src/buildupdater.py`: DC ship support (`equipcannons`), item normalization (`mark`/`modifiers` defaults), boff ability alias resolution, Intel Holoship `uni_consoles` fix
- `src/constants.py`: `SEVEN_DAYS_IN_SECONDS`, `GITHUB_CACHE_URL`, expanded SPECIES (Caitian, Klingon, Talaxian, Ferasan), `TRAIT_QUERY_URL` with `icon_name` field

---

### Architecture — Phase 4: stabilization

- **`warp/app.py` overrides verified** — all 5 overrides compatible with merged `src/app.py`, no code changes needed
- **`warp_dialog.py` → `src/` dependencies verified** — all 8 functions still present with unchanged signatures

---

## v1.9b (2026-03-27)

### Architecture — Phase 2: WARP separation from `src/`

- **`src/app.py` is now WARP-free**: all WARP-specific logic moved to `warp/app.py`
- **Removed from `src/app.py`**: debug print statements, `_MODE_FILE`, `_get_install_mode`, `_save_install_mode`, `_WARP_AVAILABLE`, `inject_warp_buttons` import, WARP button injection, WARP Updates/Installation/Uninstall settings sections, `_on_uninstall`/`_run_uninstall` methods
- **`create_main_window` baseline**: app/org name set to `SETS`/`STOCD` (upstream defaults); `WarpSETS.create_main_window` override restores `sets-warp`/`SETS-WARP`
- **`WarpSETS` overrides in `warp/app.py`**: `create_main_window`, `setup_main_layout` (injects WARP buttons via `self.widgets.menu_layout`), `setup_settings_frame` (appends WARP sections via `self.widgets.settings_scroll_layout`), `_on_uninstall`/`_run_uninstall`
- **`self.widgets.menu_layout`** and **`self.widgets.settings_scroll_layout`** exposed so subclass overrides can extend the base layout
- **`adjustSize()` after appending WARP sections** — required when adding widgets to a layout that is already embedded in a `QScrollArea` with `setWidgetResizable(False)`
- **Phase 2.1 decision**: `_save_session_slots`/`_restore_session_slots` stay in `src/callbacks.py` — called from `select_ship()` and `tier_callback()` internally, moving would create wrong `src/ → warp/` dependency

---

## v1.8b (2026-03-27)

### WARP CORE — Canvas zoom & cursor fixes

- **Zoom anchor corrected (Gwenview-style)**: image loads at 1:1 if it fits the viewport; scales down to fit if it's larger. Previously `_fit_scale` was always `1.0` (never computed from viewport).
- **Scrollbars in zoom mode**: `setWidgetResizable(False)` so `adjustSize()` after zoom actually expands the widget beyond the viewport — scrollbars now appear correctly.
- **`_compute_transform` uses viewport size**: `_fit_scale` computed from parent viewport dimensions. `sizeHint` returns viewport size in fit mode, `image × scale` in zoom mode.
- **Viewport resize handled**: event filter on viewport fires `_compute_transform` + `adjustSize` on window resize so fit-to-window stays correct.
- **Ctrl+wheel from scroll area padding**: `WarpCoreWindow` global event filter intercepts wheel events when mouse is in scroll area but outside canvas, forwarding to canvas — Ctrl+wheel zoom works anywhere in the central panel.
- **Modifier key cursors fixed (`QApplication.setOverrideCursor`)**: `widget.setCursor()` only works when mouse is over that widget. Replaced with `QApplication.setOverrideCursor()` / `restoreOverrideCursor()` — cursor changes on first Ctrl/Alt/Shift press regardless of mouse position.
- **`AttributeError: QMouseEvent has no attribute key()`**: eventFilter called `event.key()` on MouseMove events — fixed by early return for `MouseMove` before key-handling branch.

### CI / GitHub Actions fixes

- **`release.yml`**: added `permissions: contents: write` — without it workflow failed with 403 when creating releases
- **`release.yml`**: removed `prerelease: true`, added `make_latest: true` — releases now correctly marked as Latest
- **`build_installer.yml`**: changed trigger from `release: published` to `push: tags: v*` — `GITHUB_TOKEN` cannot trigger other workflows via release events, so installer never built automatically; direct tag trigger fixes this

---

## v1.6b (2026-03-26)

### WARP — Integrated Autonomous Recognition Engine (P0–P5)

- **P0 — Smart OCR**: ship metadata extraction now upscales low-res regions before reading; builds a per-session typo-correction table from confirmed annotations
- **P1 — Slot inference on manual draw**: when a bbox is drawn manually, WARP infers the most likely slot type from position and surrounding confirmed boxes
- **P3 — Multi-config layout memory with pixel scoring**: `_detect_via_learned_layouts` now scores every candidate layout by sampling brightness at predicted slot positions; best pixel-score wins instead of blindly using the most recent entry; LRU cap of 200 entries (oldest evicted automatically)
- **P4 — CNN Layout Regressor (Strategy 0)**: new `layout_trainer.py` + `layout_dataset_builder.py`; trains a MobileNetV3-based regressor (`layout_regressor.pt`) that predicts slot positions directly from a screenshot; runs as Strategy 0 before all pixel-based strategies
- **P5 — Dynamic anchoring**: high-confidence matches recalibrate the anchor grid for the current screenshot resolution; reduces drift on non-standard resolutions
- **Layout View toggle** in WARP CORE right panel: `Layout View ON/OFF` button runs CNN Strategy 0 and overlays predicted slot positions as a ghost layer on the canvas (yellow style)
- **ScreenTypeTrainer log error fixed**: worker no longer crashes when emitting per-file progress after training completes
- **TrainerWindow worker orchestration fixed**: training threads are now started and joined in the correct order; no race condition when running multiple workers back-to-back

### WARP CORE

- **`setWidgetResizable(True)`** on scroll area — canvas now fills available space instead of showing a fixed-size widget with empty padding
- **`showEvent` focus**: canvas automatically receives keyboard focus when WARP CORE window opens or is raised
- **Screenshot switch**: `QTimer.singleShot(100, ...)` deferred focus to canvas after switching to a new screenshot

### Documentation

- `docs/warp_separation_roadmap.md` — plan for separating WARP into a standalone module
- `docs/warp_ml_roadmap.md` updated with P0–P5 status

---

## v1.5b (2026-03-25)

### WARP CORE

- **Shift+LMB — move / resize bboxes**: hold Shift and drag any confirmed bbox on the canvas to reposition it; drag a corner handle to resize; annotation is updated in-place without re-running icon matching
- **Slot change on confirmed bbox**: changing the slot type of an already-confirmed bbox from the review list now correctly updates the annotation and re-saves the crop; previously the change was silently ignored
- **Graceful thread cleanup**: trainer threads (`LocalTrainWorker`, `ScreenTypeTrainerWorker`) now set a stop flag and `wait()` before the window closes; prevents occasional crash on exit during training
- **Preserve item colors in screenshot list when selected**: selecting a screenshot in the left panel no longer resets row highlight colors (green = confirmed, yellow = partial, red = no match)
- **Screen type detection stats**: `ScreenTypeDetectorWorker` collects per-type accuracy stats and logs a summary after each batch detection run; per-file log lines suppressed (too noisy)
- **Remove `[index]` log noise during training**: per-item index prefix (`[0/123]`) removed from training log lines; summary line at end retained

### Documentation

- Guides moved from project root to `docs/`; all cross-links updated (`WARP_GUIDE.md`, `SETS_GUIDE.md`, `ML_PIPELINE.md`)

---

## v1.4b (2026-03-24)

### Windows fixes

- **Taskbar icon**: `SetCurrentProcessExplicitAppUserModelID` called at bootstrap level (before any Qt window) and `WM_SETICON` sent after `window.show()` via Win32 `LoadImageW` — icon now shows correctly in taskbar regardless of `python.exe` vs `pythonw.exe`
- **`charmap` crash on Polish Windows (CP1250)**: all JSON reads in `icon_matcher.py`, `screen_classifier.py`, `training_data.py`, and `sync.py` now pass `encoding='utf-8'` explicitly
- **Setup log header duplication**: `_write_log` (which writes the `====` header) was re-created on every line inside `on_line()`; moved outside the callback — one header per session
- **Torch CPU wheel**: `--extra-index-url` (not `--index-url`) passed when installing remaining packages so pip finds the CPU-only torch wheel without replacing it with a CUDA build
- **Installer directory page**: `DisableDirPage=no` added to Inno Setup script — install path is always shown and editable during setup

### ML model download

- **Startup summary dialog**: after splash closes, a small dialog summarises asset sync results, cargo data status, and ML model info (version, class count, training date); toggled via Settings → "Show Startup Summary"
- **Model download in `populate_cache`**: if `model_version.json` is absent (e.g. after SETS-only → SETS+WARP upgrade), `ModelUpdater._bg_check()` runs synchronously inside the background populate thread; startup dialog shows correct model info on first post-upgrade launch
- **`ModelUpdater` — progress callback**: `on_progress(text, current, total)` argument added; splash bar updates during download
- **`ModelUpdater` — force download when model absent**: rate-limit cache no longer prevents the download when `model_version.json` does not exist yet
- **`ModelUpdater` — write `model_version.json` post-download**: if the HF repo does not include the file, it is synthesised from backend metadata so the startup dialog always has something to display

### Updater

- **"Later" auto-snoozes current version**: clicking "Later" in the update dialog now auto-snoozes the current version (same effect as ticking "Don't remind me for vX.Y"); the dialog no longer reappears on every launch after an update

### Log rotation

- `sets_warp_early.log` and `sets_warp.log` (via `setsdebug.py`) now rotate on startup: existing log renamed to `.bak`, fresh file opened; keeps one session of history without unbounded growth
- Same rotation added to bootstrap's `sets_warp_setup.log`

### Documentation

- `docs/ML_PIPELINE.md` — full technical reference for the ML pipeline (dataset flow, model architecture, training, community sync)
- `README.md` — expanded auto-update, autoconfigure, and uninstall sections; update-check delay corrected (3 s, was 8 s); doc-links table added

---

## v1.3b (2026-03-23)

### Bug fixes

- **WARP CORE add bbox**: fixed crash "too many values to unpack" — `match()` returns 4 values, not 3
- **Local trainer — backbone loading**: `icon_classifier.pt` backbone was never actually loaded; code checked for `.onnx` (doesn't exist) and even that check was cosmetic only (no weights applied). Now loads `.pt` with classifier keys stripped so shape mismatch on head doesn't block backbone restore
- **Local trainer — stratified split**: flat random split caused classes with 1 sample to land in val only → `val_acc = 0%` → early stop at epoch 5. Fixed with per-class split: classes with ≥ 2 samples contribute 1 to val, rest to train; single-sample classes go to train only
- **Screen type trainer**: same backbone loading and stratified split fixes as local trainer
- **`strict=False` shape mismatch**: `strict=False` ignores missing/unexpected keys but still raises on shape mismatch (same key, different `n_classes`). Fixed by stripping `classifier.*` keys from state dict before loading in all four trainers (local icon, local screen, central icon, central screen)
- **Updater loop**: update dialog appeared on every launch after updating — `_repo_root()` returned `.parent.parent` (wrong directory), `git describe` failed, version fell back to stale constant `1.2b` which never matched remote `1.3b`. Fixed path + updated fallback constant
- **`.gitignore`**: `.SETS_settings.ini` and `*.ini` user settings files added to ignore list

### Community ML pipeline

- **ModelUpdater at app startup**: background model update check (once per 24 h) now fires 15 s after launch regardless of whether WARP CORE is ever opened; previously only triggered when WARP CORE window was opened
- **Central trainer — fine-tuning**: `admin_train.py` downloads previous `icon_classifier.pt` and `screen_classifier.pt` from HF before training and loads backbone weights (warm start, LR × 0.3); previously always retrained from ImageNet
- **Central trainer — `MIN_NEW_CROPS = 10`**: `--skip-if-unchanged` now also skips when fewer than 10 new crops arrived since last run
- **Central trainer — screen type cap**: per screen-type, if ≥ 30 samples exist, dataset randomly capped to 150; prevents bloat for stable UI screens
- **Central trainer — bulk download**: replaced per-file `hf_hub_download` (one HEAD+GET per crop) with `snapshot_download` per install_id; suppressed httpx INFO spam
- **Central trainer — stratified split**: same fix as local trainer applied to both `train()` and `train_screen_classifier()`

### Desktop integration

- **Per-path `.desktop` entries**: each installation path gets its own `.desktop` file keyed by an 8-char sha256 hash of `SCRIPT_DIR`; multiple installations coexist without overwriting each other
- **Legacy migration**: existing `sets-warp.desktop` is migrated automatically to the hashed name on first run if it belongs to the current install path
- **Uninstall SETS-WARP**: new button in Settings → writes a `/tmp` shell script that removes the `.desktop` entry and deletes the entire install directory after the app exits

### WARP CORE

- **Auto-accept defaults to ON**

---

## v1.2b (2026-03-22)

### Installation

- **SETS-only vs SETS+WARP choice** — first-run setup now asks which installation type to use:
  - *SETS only* (~500 MB): build planner, ship/equipment database, no screenshot recognition
  - *SETS + WARP* (~2.5 GB): full install with screenshot recognition and ML training
  - Choice shown as graphical cards in the tkinter setup window; plain-text prompt in headless mode
  - Selection saved to `.config/install_mode.txt`; re-run setup by deleting that file
- **Disk space check** — warns before install if free space is below 1.5 GB (SETS-only) or 4 GB (SETS+WARP)
- **Community ML model on first run** — fresh installs automatically download the latest centrally-trained icon classifier during setup (Step 5/5); no manual sync needed
- **`_quick_check_venv`** — health check now reads the saved install mode and only validates packages relevant to that mode

### Linux desktop integration

- **`installer/install_desktop.sh`** — installs `.desktop` entry and app icon to `~/.local/` (no root required); supports `--uninstall`
- **`sets_warp.sh`** — auto-runs `install_desktop.sh` on first Linux launch if the desktop entry is absent; silent, safe to re-run
- **`src/app.py`** — `app.setApplicationName('sets-warp')` added so Wayland `app_id` matches `StartupWMClass` in the desktop file
- **KDE Plasma 6 fix** — removed leading empty line from `sets_warp.sh` that prevented the kernel from recognising the `#!/bin/sh` shebang; KDE's `QProcess::startDetached` uses `execv()` directly without shell fallback, so the shebang must be on byte 0
- **`install_desktop.sh`** — `.desktop` file now includes `StartupNotify=true` and `Path=` (working directory)

### Windows installer

- **`installer/sets_warp.iss`** — `SetupIconFile`, `UninstallDisplayIcon`, and `IconFilename` in `[Icons]` all set to `local/icon.ico` so installer and shortcuts display the correct app icon

### UI — dark theme unification

- **`warp/style.py`** *(new)* — single source of truth for all WARP/WARP CORE colours; `apply_dark_style(widget)` sets QPalette + QSS on any window
- **`warp/warp_dialog.py`** — WARP import dialog now uses unified dark theme (was system default)
- **`warp/trainer/trainer_window.py`** — WARP CORE and all progress dialogs use unified dark theme
- **`warp/trainer/annotation_widget.py`** — canvas background aligned to theme colour `#1a1a1a`
- **`warp/warp_button.py`** — WARP and WARP CORE buttons: white text (`#eeeeee`), dark background, gold border (`#c59129`)
- **`main.py`** — checkbox `::indicator:checked` now uses solid gold fill (matching WARP CORE style) instead of SVG image; consistent across Settings and all other checkboxes
- **WARP CORE spinbox** — auto-accept confidence spinner widened to 72 px (was 58 px) to fit two-decimal values on Windows

### Documentation

- **`README.md`** — rewritten: installation options (Windows .exe, git), SETS-only vs SETS+WARP size table, Linux desktop integration, updating workflow, correct repo URL
- **`WARP_GUIDE.md`** *(new)* — full user guide: preparing screenshots, WARP import walkthrough, WARP CORE interface reference, correcting recognition, training the ML model, community sync, keyboard shortcuts, troubleshooting

---

## v1.1b (2026-03-22)

### Auto-update

- **`updater.py`** — background update checker (runs 8 s after launch, daemon thread)
- Current version read from nearest git tag (`git describe --tags`) — advances automatically after `git pull`, no code change needed
- Two update paths: git install → `git pull` + restart; zip install → download release zip + extract + restart
- Update dialog: shows release notes, "Don't remind me for vX.Y" snooze checkbox
- **Settings tab → "SETS-WARP Updates"** — Autoupdate on/off checkbox + installed version label
- Snooze persisted in QSettings (`warp_update/snoozed_version`); clears automatically on next release

### Windows Installer

- **`installer/sets_warp.iss`** — Inno Setup 6 script; installs to `%LOCALAPPDATA%\SETS-WARP` (no admin rights)
- **`launch.vbs`** — silent launcher (no cmd console window); used by Desktop / Start Menu shortcuts
- Post-install check: warns if Python 3.11+ is not found
- **`.github/workflows/build_installer.yml`** — builds `sets-warp-vX.Y-setup.exe` automatically on each GitHub release and attaches it as a release asset

---

## v1.0b (2026-03-22) — Initial Public Beta

> Tested and prepared for **Linux**. Windows support is present but not yet fully tested.

### WARP Recognition

- **Bridge officer routing rewrite** — Y/X clustering detects per-seat ability groups from bbox coordinates; profession-based matching assigns clusters to correct seats (3 passes: explicit profession → Universal+spec → Universal); Universal seat profession now set correctly before loading abilities
- **Boff routing fix** — ship_data['boffs'] order ≠ visual screen order; clustering now uses screen position instead of cargo JSON index
- **Layout detector fix** — pixel_count floor was `1` for multi-slot rows (scan stopped early at empty right-side slots); now uses ShipDB profile count as floor
- **ShipDB crash fix** — some entries in ship_list.json have `name`/`type` as a list; fixed with `join` before `strip`

### WARP CORE UI

- **Recognition stats** — after each import, logs: auto-detect %, WARP CORE fallback %, average confidence, rolling 50-session history with trend arrows (↑↓→)
- **Log tags** — `[WARP CORE]` / `[Autodetect]` moved to beginning of log line (after `[INFO]`) for easier filtering
- **Button labels** — "Clear" → "Clear Current Tab", "Clear all" → "Clear All Tabs"
- **WARP button** — replaced ⚡ emoji with `warp.jpg` icon; spacing between icon and label text
- **WARP CORE button** — height aligned to match WARP button (28px); icon size corrected
- **Menu bar alignment** — Export, Settings, WARP, WARP CORE buttons vertically centered on the menu row

### Community Knowledge Sync

- **Circuit breaker** — after any network/HTTP error, contributions are silently skipped for 5 minutes; eliminates 503 log spam when Render backend is cold-starting
- **Log level** — contribution failures downgraded from `warning` to `debug` (not actionable by user)

### Infrastructure

- `warp/ui_helpers.py` — shared Qt dialog helpers extracted
- `local/warp.jpg` — WARP button icon added
