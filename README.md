# SETS-WARP

A build planning and screenshot recognition tool for Star Trek Online.

> **Status:** v1.3b — tested on **Linux** and **Windows**. macOS support is present but untested.

---

## What is this?

**SETS-WARP** combines three tools in one package:

**SETS** *(STO Equipment and Trait Selector)* — build planner for Star Trek Online. Plan ship and ground loadouts, skill trees, and bridge officer assignments without owning the items in-game. Builds can be shared as PNG and JSON files.
> All credits for SETS to **Shinga** — [STOCD/SETS](https://github.com/STOCD/SETS).

**WARP** *(Weaponry & Armament Recognition Platform)* — reads your in-game screenshots and automatically fills in your SETS build. Detects equipment, traits, bridge officers, and ship information using computer vision and machine learning.

**WARP CORE** — trainer interface built into WARP. Review and correct recognition results, confirm annotations, and retrain the local ML model on your own data to improve accuracy over time.

→ See **[WARP_GUIDE.md](WARP_GUIDE.md)** for a full usage guide with step-by-step instructions for WARP and WARP CORE.
→ See **[SETS_GUIDE.md](SETS_GUIDE.md)** for SETS build planner documentation.

---

## Installation

### Windows — Installer (recommended)

Download **`sets-warp-vX.Y-setup.exe`** from the [Releases page](https://github.com/raman78/sets-warp/releases) and run it.

- Installs to `%LOCALAPPDATA%\SETS-WARP` — **no administrator rights required**
- Creates Start Menu and optional Desktop shortcut
- On first launch, chooses between **SETS only** (~3 GB) or **SETS + WARP** (~10 GB) and downloads all dependencies automatically
- Post-install wizard warns if Python 3.11+ is missing from PATH

### Windows — Git (advanced)

```bat
git clone https://github.com/raman78/sets-warp.git
cd sets-warp
sets_warp.bat
```

### Linux

```bash
git clone https://github.com/raman78/sets-warp.git
cd sets-warp
./sets_warp.sh
```

On first run a graphical setup window appears asking you to choose:

| Option | Disk space | What's included |
|--------|-----------|-----------------|
| **SETS only** | ~3 GB | Build planner, ship/equipment database |
| **SETS + WARP** | ~10 GB | Everything above + screenshot recognition, ML training |

The installer then downloads a portable Python runtime and all required packages automatically. Subsequent launches complete in under a second.

**Desktop integration** is set up automatically on first run — SETS-WARP appears in your application launcher with its own icon.

**tkinter requirement (first run only):** tkinter is needed to display the setup window. If it's missing, the script prints the exact install command for your distro (apt / dnf / pacman / zypper / apk).

### macOS

```bash
git clone https://github.com/raman78/sets-warp.git
cd sets-warp
./sets_warp.sh
```

Same automatic setup as Linux. No desktop integration step on macOS.

---

## First-run setup details

The setup window walks through these steps automatically:

1. Download portable Python 3.13 (~65 MB, one-time)
2. Create isolated virtual environment (`.venv/`)
3. Install Python packages from PyPI (~3 GB SETS-only / ~10 GB SETS+WARP, one-time)
4. *(SETS+WARP only)* Build WARP item database from STO cargo data
5. *(SETS+WARP only)* Download community ML model — so WARP works immediately without manual training

The choice between SETS-only and SETS+WARP is saved. You can switch at any time via **Settings → Install SETS + WARP** — the installer or cleanup runs automatically on restart.

---

## Updating

```bash
git pull
./sets_warp.sh     # Linux / macOS
sets_warp.bat      # Windows (git install)
```

SETS-WARP checks for updates automatically 8 seconds after launch and shows a dialog when a new version is available. The update can also be triggered manually from **Settings → SETS-WARP Updates**.

`bootstrap.py` detects and installs new dependencies on every launch — no manual `pip install` needed.

---

## How WARP works (quick summary)

1. Take screenshots of your in-game build screens (equipment, traits, bridge officers — separately or as one combined screenshot). **One build per folder.**
2. Click the **⚡ WARP** button, select the screenshot folder.
3. WARP detects your ship, reads slot counts, recognises each item, and fills your build automatically.
4. Use **WARP CORE** to review results, correct mistakes, and confirm items — building up training data that improves recognition over time.

→ Full instructions: **[WARP_GUIDE.md](WARP_GUIDE.md)**
→ ML pipeline details: **[ML_PIPELINE.md](ML_PIPELINE.md)**

---

## WARP gets smarter over time

**Local model:** Confirmed items in WARP CORE are saved as training data. After confirming enough items, click **Train Model** to retrain the icon classifier on your corrections.

**Community model:** Confirmed items are sent anonymously to a shared knowledge base. The central model is retrained hourly from community contributions and downloaded at the next startup. Fresh installs get the latest community model automatically during setup.

→ Full technical details: **[ML_PIPELINE.md](ML_PIPELINE.md)**

---

## Contributing

All credits for SETS to Shinga — [STOCD/SETS](https://github.com/STOCD/SETS).

Item data from the [official STO wiki](https://stowiki.net). If something is missing or wrong, update it there.

Bug reports and suggestions: [GitHub Issues](https://github.com/raman78/sets-warp/issues) or [STOBuilds Discord](https://discord.gg/kxwHxbsqzF) → `#sets-support`.

---

## Documentation

| File | Contents |
|------|----------|
| [WARP_GUIDE.md](WARP_GUIDE.md) | WARP and WARP CORE user guide |
| [SETS_GUIDE.md](SETS_GUIDE.md) | SETS build planner guide |
| [ML_PIPELINE.md](ML_PIPELINE.md) | ML pipeline technical reference |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
