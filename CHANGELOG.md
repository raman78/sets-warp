# CHANGELOG

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
