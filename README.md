# SETS-WARP

A build planning and screenshot recognition tool for Star Trek Online.

---

## What is this?

**SETS-WARP** combines two tools:

**SETS** *(STO Equipment and Trait Selector)* is a build planner for Star Trek Online. It lets you plan ship and ground loadouts, skill trees, and bridge officer assignments — without needing to own the items in-game. Builds can be shared as PNG or JSON files.

**WARP** *(Weaponry & Armament Recognition Platform)* is a recognition module built on top of SETS. It reads your in-game screenshots and automatically fills in your build — detecting equipment, traits, bridge officers, and ship information using computer vision and machine learning.

**WARP CORE** is the trainer interface inside WARP. It lets you review and correct recognition results, confirm annotations, and train the local ML model on your own data to improve accuracy over time.

---

## How it works

1. Take screenshots of your in-game build screens (equipment, traits, bridge officers — separately or as combined screenshots).
2. Open WARP, select the build type, point it at your screenshot folder.
3. WARP detects your ship, reads slot counts, recognises each item, and fills your SETS build automatically.
4. Use WARP CORE to review results, correct mistakes, and confirm items — building up a personal training dataset that improves recognition over time.

---

## Screenshot types supported

WARP handles three ways players share their builds:

- **Separate screens** — one screenshot per game tab (equipment, traits, bridge officers, etc.), all placed in one folder.
- **Mixed screen** — a single combined screenshot assembled from multiple tabs.
- **Partial** — only some screens provided (e.g. equipment only, or traits only).

The build type selector in the WARP import dialog tells WARP what to expect.

---

## Installation

No installer — the app manages its own Python environment automatically via `bootstrap.py`. You need Python 3.11+ already installed on your system.

### Linux / macOS

```bash
git clone https://github.com/STOCD/SETS.git
cd SETS
./SETS.sh
```

On first run `SETS.sh` creates a local `.venv`, downloads a portable Python runtime, and installs all dependencies automatically. Subsequent launches are fast.

If tkinter is missing (required only for the first-run installer window), the script will print the exact install command for your Linux distribution.

### Windows

```
git clone https://github.com/STOCD/SETS.git
cd SETS
SETS.bat
```

`SETS.bat` finds your Python 3 installation and runs `bootstrap.py`, which sets up the environment on first launch.

### Images library

On first launch, SETS downloads item icons automatically. To speed this up, download the latest image archive from the [releases page](https://github.com/STOCD/SETS/releases) and extract it to `.config/images/`.

---

## Updating

```bash
git pull
./SETS.sh     # Linux / macOS
SETS.bat      # Windows
```

`bootstrap.py` automatically detects and installs any new dependencies on launch.

---

## WARP gets smarter over time

Every time you confirm an item in WARP CORE, two things happen in the background:

**Your local model improves.** Confirmed items are saved as training data. Periodically WARP retrains the icon recognition model on your corrections — so items it got wrong today, it gets right next time.

**The community knowledge base grows.** Each confirmed item is also sent (anonymously) to a shared knowledge base. When enough players confirm the same item, it's merged into the community database that all WARP users benefit from — no training required, just downloading the latest knowledge at startup.

This means WARP works reasonably well out of the box, and gets noticeably better the more you use WARP CORE to correct mistakes.

---

## Contributing

Item data comes from the [official STO wiki](https://stowiki.net). If something is missing or wrong, update it there.

For app bugs or suggestions: [STOBuilds Discord](https://discord.gg/kxwHxbsqzF) → `#sets-support`.
