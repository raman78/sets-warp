# CHANGELOG

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
