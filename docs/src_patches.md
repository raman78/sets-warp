# src/ patches — intentional differences from upstream SETS

> Last updated: 2026-03-29
> Upstream ref: `ae63ef1` (STOCD/SETS main)
> Purpose: reference for future upstream syncs — every item here must be re-applied after the next merge.

---

## How to sync with a new upstream release

```bash
# 1. Dry-run — see what's new and what would be applied
python scripts/upstream_sync.py

# 2. Apply — create branch, auto-patch, get manual checklist
python scripts/upstream_sync.py --apply

# 3. Review the ⚠️ patches and manual files listed in the output
#    (use this document as reference for each manual file)

# 4. After all manual fixes are committed:
git checkout main
git merge upstream-merge-YYYY-MM-DD
git tag vX.Yb && git push origin main --tags
```

**What the script auto-applies** (no manual work):
- `src/constants.py` — SEVEN_DAYS_IN_SECONDS, GITHUB_CACHE_URL, TRAIT_QUERY_URL, SPECIES
- `src/callbacks.py` — log import, _EQUIPMENT_CATS/_TRAIT_CATS, _save/_restore_session_slots
- `src/buildupdater.py` — Intel Holoship fix, item normalization, boff alias lookup
- `src/datafunctions.py` — trait key migration

**What always requires manual review** (see sections below):
- `src/app.py` — HIGH risk, complex additions
- `src/iofunc.py` — MEDIUM, utilities and imports
- `src/widgets.py` — MEDIUM, ImageLabel/TooltipLabel
- `src/splash.py` — LOW, take our version entirely
- `src/datafunctions.py` — icon_name → cache.alt_images section

---

## Files that exist only in SETS-WARP (no merge conflict)

These files are not present in upstream. Git keeps them automatically on merge.
No re-application needed — just verify imports still work after each upstream merge.

| File | Purpose |
|------|---------|
| `src/setsdebug.py` | Logging system — `log.info/debug/warning` routed to SETS log panel |
| `src/cargomanager.py` | Cargo data manager (Cloudflare bypass, GitHub cache fallback) |
| `src/downloader.py` | HTTP icon downloader with ETA and progress reporting |
| `src/imagemanager.py` | In-memory image cache, ship image loading |
| `src/syncmanager.py` | HuggingFace sync + GitHub cache fallback for cargo data |

---

## src/app.py

### Added imports (module level)
```python
import sys
from pathlib import Path
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from .cargomanager import CargoManager
from .datafunctions import cache_skills          # standalone function, not class method
from .downloader import Downloader
from .imagemanager import ImageManager
from .iofunc import load_json
from .setsdebug import log
from .widgets import TooltipLabel
```
`cache_skills` is also removed from the class-level `from .datafunctions import (...)` block.

### Removed from class-level datafunctions import
```python
# REMOVED: cache_skills  (now standalone, not a method)
```

### __init__ additions
- `Downloader` init with Cloudflare cookie setup from `.env`
- `CargoManager(self)` init
- `ImageManager(self)` init
- `self.cache_item_aliases()` call
- `self._set_win32_taskbar_icon()` call on win32
- `log.info(...)` calls throughout for startup tracing

### Added methods
- `_set_win32_taskbar_icon()` — sets Windows taskbar app icon via ctypes WM_SETICON
- `cache_item_aliases()` — loads `aliases.json` into `self.cache.item_aliases`

### cache_icons additions
- `dual_cannons` icon loaded
- N/A placeholder image drawn via QPainter (grey rect + "N/A" text)

### init_environment addition
```python
create_folder(auto_backups)   # auto backup subfolder
```

### setup_main_layout hook (WARP injection point)
```python
self.widgets.menu_layout = menu_layout   # stored BEFORE return, after grid is built
```

### setup_ship_frame — DC label + 4-column layout
- `dc_label = TooltipLabel(...)` added for dual-carrier ship indicator
- Layout uses 4-column grid: span 1,4 / 2,1,3 / 3,0,1,4
- `self.widgets.ship['dc'] = dc_label`

### setup_build_frames — standalone cache_skills call
```python
cache_skills(self.cache.skills, self.app_dir)   # NOT self.cache_skills()
```

### setup_splash — full progress-bar version
Our version has banner (1000×600) + progress frame (1000×500) with:
- `loading_label` (phase text + ETA)
- `progress_bar` (centre column, 70% width)
- `progress_detail` (file counter "234 / 3 500")

Upstream has a simpler splash without progress.

### setup_settings_frame additions
- "Preferred Backup" setting
- "Show Startup Summary" setting
- Hook at end of method:
```python
self.widgets.settings_scroll_layout = scroll_layout   # before setWidget
self.widgets.settings_scroll_frame  = scroll_frame    # before setWidget
```

---

## src/buildupdater.py

### Added import
```python
from .setsdebug import log
```

### Federation Intel Holoship special case
```python
elif ship_data['name'] == 'Federation Intel Holoship':
    uni_consoles += 1
```
Upstream does not have this. The Intel Holoship has an extra Universal Console slot
that is not encoded in its ship data.

### Item normalization (legacy build compatibility)
In the equipment loading loop:
```python
if isinstance(item, dict):
    item.setdefault('mark', '')
    item.setdefault('modifiers', [None, None, None, None])
```
Prevents `KeyError` on old saves that predate the `mark`/`modifiers` fields.

### Boff ability alias resolution
Both space and ground boff loading blocks have this guard (replaces direct dict access):
```python
tooltip = self.cache.boff_abilities['all'].get(ability['item'], None)
if tooltip is None:
    new_name = self.cache.item_aliases['boff_abilities'].get(ability['item'], None)
    if new_name is None:
        slot.clear()
    else:
        tooltip = self.cache.boff_abilities['all'].get(new_name, None)
        if tooltip is None:
            slot.clear()
        else:
            slot.set_item_full(image(self, new_name), None, tooltip)
            # update build dict with canonical name
            ...
else:
    slot.set_item_full(image(self, ability['item']), None, tooltip)
```
Upstream uses direct `self.cache.boff_abilities['all'][ability['item']]` (crashes on renamed abilities).

---

## src/callbacks.py

### Added import
```python
from .setsdebug import log
```

### Added constants
```python
_EQUIPMENT_CATS = ('fore_weapons', 'aft_weapons', 'experimental', 'devices', 'hangars',
                   'sec_def', 'deflector', 'engines', 'core', 'shield',
                   'uni_consoles', 'eng_consoles', 'sci_consoles', 'tac_consoles')
_TRAIT_CATS = ('traits', 'starship_traits', 'rep_traits', 'active_rep_traits')
```

### Added functions
- `_save_session_slots(self)` — snapshots current space build before ship switch
- `_restore_session_slots(self)` — restores snapshot after `align_space_frame(clear=True)`

These stay in `src/callbacks.py` (not moved to `warp/`) because they are called from
`select_ship()` and `tier_callback()` — standard SETS behaviour, not WARP-exclusive.

---

## src/constants.py

### TRAIT_QUERY_URL — different fields
Our version:
```
fields=_pageName%3DPage,name,type,environment,description,icon_name=icon_name
```
Upstream:
```
fields=_pageName%3DPage,name,chartype,environment,type,isunique,description
```
We removed `chartype` (unused) and `isunique` (unused for traits).
We added `icon_name` — used in `datafunctions.py` to populate `cache.alt_images` for trait icon aliases.

### Added constants
```python
SEVEN_DAYS_IN_SECONDS = 60 * 60 * 24 * 7    # used by syncmanager.py
GITHUB_CACHE_URL = 'https://github.com/STOCD/SETS-Data/raw/refs/heads/main'
```

### Extended species sets
Federation `'Alien'` set gains: `'Caitian'`, `'Klingon'`, `'Talaxian'`
Added full Klingon species set:
```python
'Klingon': {'Klingon', 'Gorn', 'Lethean', 'Nausicaan', 'Orion', 'Alien',
            'Liberated Borg', 'Ferasan', 'Talaxian'}
```

---

## src/datafunctions.py

### Trait key names — upstream naming maintained
We use upstream's trait type keys: `'personal'`, `'rep'`, `'active_rep'`
(our old code used `'traits'`, `'rep_traits'`, `'active_rep_traits'`).

### Old cache migration (load_cargo_cache)
After loading `traits.json` from disk:
```python
_key_map = {'traits': 'personal', 'rep_traits': 'rep', 'active_rep_traits': 'active_rep'}
for env in ('space', 'ground'):
    if env in self.cache.traits and 'traits' in self.cache.traits[env]:
        self.cache.traits[env] = {
            _key_map.get(k, k): v for k, v in self.cache.traits[env].items()
        }
```
One-time migration for users upgrading from pre-merge installs with old `traits.json` on disk.

### icon_name → cache.alt_images (load_cargo_data, traits section)
```python
if trait['icon_name'] is None:
    ...
else:
    self.cache.images_set.add(trait['icon_name'])
    self.cache.alt_images[f'{name}__{trait["environment"]}__{trait_type}'] = trait['icon_name']
```
Requires `icon_name` field in `TRAIT_QUERY_URL` (see constants.py above).

### cache_item_aliases() method
Loads `aliases.json` into `self.cache.item_aliases` — used by `buildupdater.py`
for boff ability alias resolution.

---

## src/iofunc.py

### Added imports
```python
from json import load as json__load, JSONDecodeError
from pathlib import Path
from threading import Thread
from requests.cookies import create_cookie as requests__create_cookie
from lxml import html as lxml_html
from lxml.cssselect import CSSSelector
from .setsdebug import log
```
Note: upstream uses `from requests_html import HTMLSession` — we replaced this
because `requests_html` has Chromium dependency issues on Linux.

### Added class
```python
class ReturnValueThread(Thread):
    """Thread subclass that captures return value of target function."""
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args, daemon=True)
        self._return = None
    def run(self): self._return = self._target(*self._args)
    def join(self): super().join(); return self._return
```
Used by `syncmanager.py` and `downloader.py`.

### Added functions
- `auto_backup_cargo_file(self, filename)` — copies cargo file to auto_backups folder
- `_is_valid_png(filepath)` — PNG header + size validation before loading
- Constants: `_PNG_HEADER`, `_MIN_PNG_SIZE`

---

## src/splash.py

Completely replaced. Upstream has a minimal splash; ours adds:
- `_state` dict (GIL-safe, written by worker thread, polled by QTimer)
- `splash_progress(text, current, total)` — updates `_state` from any thread
- `SplashPoller` — QTimer-based poller that updates `loading_label`, `progress_bar`, `progress_detail`
- `start_splash_poller(splash_widgets)` / `stop_splash_poller()` — lifecycle management

Used in `src/app.py` `setup_splash()` to show download progress during startup.

---

## src/widgets.py

### ImageLabel — replaced implementation
Upstream `ImageLabel(QWidget)` uses `paintEvent`.
Ours: `ImageLabel(QLabel)` — uses `setPixmap` + `resizeEvent` + `_update_pixmap` with aspect-ratio scaling.

### ShipImage — replaced implementation
Upstream `ShipImage(ImageLabel)`.
Ours: `ShipImage(QLabel)` — fixed 500×280 frame, `setPixmap` directly.

### Added class: TooltipLabel
```python
class TooltipLabel(QLabel):
    """QLabel that shows a custom tooltip widget (QLabel) on hover."""
```
Used for the DC (dual-carrier) ship indicator in `setup_ship_frame`.

### Cache.reset_cache additions
```python
self.alt_images: dict = dict()    # trait icon name aliases: key → icon_name
```
Also: parameter name kept as `keep_skills=False` (upstream naming).

### WidgetStorage.ship
```python
'dc': TooltipLabel    # dual-carrier indicator label
```

---

## What to watch on next upstream merge

| Area | Risk | Action |
|------|------|--------|
| `src/app.py __init__` | HIGH — upstream actively extends it | Re-apply all manager inits + hooks after taking upstream |
| `src/datafunctions.py` | HIGH — large file, upstream modifies frequently | Full read of both sides before merge |
| `src/iofunc.py` | MEDIUM | Re-add `ReturnValueThread`, PNG helpers, `auto_backup_cargo_file` |
| `src/widgets.py ImageLabel` | MEDIUM — upstream may change their impl | Compare both impls, keep aspect-ratio scaling |
| `src/constants.py TRAIT_QUERY_URL` | LOW | Verify `icon_name` field still valid in stowiki Traits cargo table |
| `src/splash.py` | LOW | Take ours entirely — upstream version is simpler |
| `src/callbacks.py` | LOW — our additions are isolated functions | Re-add `_save/_restore_session_slots` + `_EQUIPMENT_CATS`/`_TRAIT_CATS` |
