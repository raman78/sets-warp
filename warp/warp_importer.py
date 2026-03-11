# warp/warp_importer.py
#
# Ship-first recognition strategy:
#   1. TextExtractor reads ship name + type from screenshot
#   2. ShipDB looks up exact slot counts from ship_list.json (cargo data)
#      → SETS already has this data, 783 ships, fields: fore, aft, experimental,
#        hangars, secdeflector, uniconsole, consolestac, consoleseng, consolessci, devices
#   3. Fallback: category-based profile if ship not found in DB
#   4. LayoutDetector finds bboxes using profile to constrain slot count
#   5. IconExtractor + SETSIconMatcher per slot

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

SCREENSHOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
TEMPLATE_CONF_THRESHOLD = 0.72


# ── Canonical slot order ────────────────────────────────────────────────────────
# Fixed visual top→bottom order in STO Status tab.
# This order NEVER changes regardless of ship type.
# Optional slots (mandatory=False) may simply be absent for a given ship.

SPACE_SLOT_ORDER: list[dict] = [
    {'name': 'Fore Weapons',         'key': 'fore_weapons',  'mandatory': True,  'max': 5, 'weapon': True,  'exp': False},
    {'name': 'Deflector',            'key': 'deflector',     'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Sec-Def',              'key': 'sec_def',       'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Engines',              'key': 'engines',       'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Warp Core',            'key': 'core',          'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Shield',               'key': 'shield',        'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Aft Weapons',          'key': 'aft_weapons',   'mandatory': False, 'max': 5, 'weapon': True,  'exp': False},
    {'name': 'Experimental',         'key': 'experimental',  'mandatory': False, 'max': 1, 'weapon': True,  'exp': True},
    {'name': 'Devices',              'key': 'devices',       'mandatory': True,  'max': 6, 'weapon': False, 'exp': False},
    {'name': 'Universal Consoles',   'key': 'uni_consoles',  'mandatory': False, 'max': 3, 'weapon': False, 'exp': False},
    {'name': 'Engineering Consoles', 'key': 'eng_consoles',  'mandatory': True,  'max': 5, 'weapon': False, 'exp': False},
    {'name': 'Science Consoles',     'key': 'sci_consoles',  'mandatory': True,  'max': 5, 'weapon': False, 'exp': False},
    {'name': 'Tactical Consoles',    'key': 'tac_consoles',  'mandatory': True,  'max': 5, 'weapon': False, 'exp': False},
    {'name': 'Hangars',              'key': 'hangars',       'mandatory': False, 'max': 4, 'weapon': False, 'exp': False},
]

GROUND_SLOT_ORDER: list[dict] = [
    {'name': 'Body Armor',       'key': 'armor',          'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
    {'name': 'EV Suit',          'key': 'ev_suit',        'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Personal Shield',  'key': 'personal_shield','mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Weapons',          'key': 'weapons',        'mandatory': True,  'max': 2, 'weapon': True,  'exp': False},
    {'name': 'Kit',              'key': 'kit',            'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Kit Modules',      'key': 'kit_modules',    'mandatory': True,  'max': 6, 'weapon': False, 'exp': False},
    {'name': 'Ground Devices',   'key': 'ground_devices', 'mandatory': False, 'max': 3, 'weapon': False, 'exp': False},
]

SLOT_ORDER = {'SPACE': SPACE_SLOT_ORDER, 'GROUND': GROUND_SLOT_ORDER}

SPACE_SLOTS  = [(s['name'], s['max']) for s in SPACE_SLOT_ORDER]
GROUND_SLOTS = [(s['name'], s['max']) for s in GROUND_SLOT_ORDER]
SLOT_SPECS   = {'SPACE': SPACE_SLOTS, 'GROUND': GROUND_SLOTS,
                'SPACE_SKILLS': [], 'GROUND_SKILLS': []}

# Weapon types that can only go in Experimental slot
EXPERIMENTAL_TYPES = frozenset({'Experimental Weapon'})

# OCR label → canonical slot name
SLOT_LABEL_ALIASES: dict[str, str] = {
    'Fore Weapons':         'Fore Weapons',
    'Fore Weapon':          'Fore Weapons',
    'Aft Weapons':          'Aft Weapons',
    'Aft Weapon':           'Aft Weapons',
    'Experimental Weapon':  'Experimental',
    'Experimental Weapons': 'Experimental',
    'Secondary Deflector':  'Sec-Def',
    'Sec Def':              'Sec-Def',
    'Impulse':              'Engines',
    'Impulse Engines':      'Engines',
    'Warp':                 'Warp Core',
    'Warp Engine':          'Warp Core',
    'Singularity':          'Warp Core',
    'Singularity Core':     'Warp Core',
    'Shields':              'Shield',
    'Shield':               'Shield',
    'Deflector':            'Deflector',
    'Universal Consoles':   'Universal Consoles',
    'Universal Console':    'Universal Consoles',
    'Engineering Consoles': 'Engineering Consoles',
    'Engineering Console':  'Engineering Consoles',
    'Science Consoles':     'Science Consoles',
    'Science Console':      'Science Consoles',
    'Tactical Consoles':    'Tactical Consoles',
    'Tactical Console':     'Tactical Consoles',
    'Hangar':               'Hangars',
    'Hangar Bay':           'Hangars',
    'Hangars':              'Hangars',
    'Devices':              'Devices',
    'Device':               'Devices',
}


# ── ShipDB — primary source of truth for slot counts ──────────────────────────

class ShipDB:
    """
    Wraps ship_list.json from SETS cargo.
    Provides exact slot counts per ship using the cargo data fields:
      fore, aft, experimental, hangars, secdeflector,
      uniconsole, consolestac, consoleseng, consolessci, devices

    Fields confirmed from debug_cargo output:
      ship_list.json: list[{Page, name, image, fc, tier, type, hull, ...,
                             fore, aft, consolestac, consoleseng, consolessci,
                             uniconsole, t5uconsole, experimental, secdeflector,
                             hangars, devices, ...}]
    """

    def __init__(self, cargo_dir: Path):
        self._ships: list[dict] = []
        self._index: dict[str, dict] = {}   # lowercase name → ship entry
        self._load(cargo_dir)

    def _load(self, cargo_dir: Path):
        p = cargo_dir / 'ship_list.json'
        if not p.exists():
            log.warning(f'ShipDB: ship_list.json not found at {p}')
            return
        try:
            ships = json.loads(p.read_text(encoding='utf-8'))
            self._ships = ships
            for ship in ships:
                name = (ship.get('name') or '').strip()
                if name:
                    self._index[name.lower()] = ship
            log.info(f'ShipDB: loaded {len(self._ships)} ships')
        except Exception as e:
            log.warning(f'ShipDB load error: {e}')

    def get_profile(self, ship_name: str, ship_type: str) -> dict[str, int]:
        """
        Returns exact slot counts for a ship.
        Tries: exact name match → fuzzy name match → type-keyword fallback.
        """
        # 1. Exact match
        entry = self._index.get(ship_name.lower().strip())

        # 2. Fuzzy match (handles slight OCR errors, missing prefixes)
        if entry is None and self._index:
            candidates = list(self._index.keys())
            matches = get_close_matches(
                ship_name.lower(), candidates, n=1, cutoff=0.72)
            if matches:
                entry = self._index[matches[0]]
                log.debug(f'ShipDB fuzzy: {ship_name!r} → {matches[0]!r}')

        if entry is not None:
            return self._entry_to_profile(entry)

        # 3. Type-keyword fallback
        log.debug(f'ShipDB: {ship_name!r} not found — using keyword fallback')
        return _type_keyword_profile(ship_type)

    def _entry_to_profile(self, e: dict) -> dict[str, int]:
        """Map ship_list.json fields to WARP slot profile."""
        def _int(v, default=0) -> int:
            try:    return int(v) if v is not None else default
            except: return default

        return {
            'Fore Weapons':         _int(e.get('fore'),         4),
            'Deflector':            1,
            'Sec-Def':              1 if e.get('secdeflector')  else 0,
            'Engines':              1,
            'Warp Core':            1,
            'Shield':               1,
            'Aft Weapons':          _int(e.get('aft'),          3),
            'Experimental':         1 if e.get('experimental')  else 0,
            'Devices':              _int(e.get('devices'),      4),
            'Universal Consoles':   _int(e.get('uniconsole'),   0),
            'Engineering Consoles': _int(e.get('consoleseng'),  3),
            'Science Consoles':     _int(e.get('consolessci'),  3),
            'Tactical Consoles':    _int(e.get('consolestac'),  3),
            'Hangars':              _int(e.get('hangars'),      0),
        }


# ── Keyword fallback profiles ──────────────────────────────────────────────────
# Used ONLY when ship not found in ship_list.json.
# Conservative estimates — better to miss a slot than hallucinate one.

_KEYWORD_PROFILES: list[tuple[str, dict]] = [
    # (keyword_in_type_lowercase, profile)
    # Most specific first — confirmed against actual STO ships.
    # exp=0 and hang=0 by default: these slots are RARE, only specific ships.
    # ShipDB (ship_list.json) is the primary source; this is only the fallback.
    ('carrier',        dict(fore=3, aft=3, exp=0, hang=2, sec=0, uni=0, eng=4, sci=3, tac=3, dev=5)),
    ('dreadnought',    dict(fore=5, aft=3, exp=0, hang=0, sec=0, uni=0, eng=4, sci=3, tac=3, dev=4)),
    ('miracle worker', dict(fore=5, aft=3, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=4, dev=4)),
    ('temporal',       dict(fore=5, aft=3, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=3, dev=4)),
    ('command',        dict(fore=5, aft=3, exp=0, hang=0, sec=0, uni=0, eng=4, sci=3, tac=3, dev=4)),
    ('battlecruiser',  dict(fore=5, aft=3, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=4, dev=4)),
    ('raider',         dict(fore=5, aft=2, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=5, dev=4)),
    ('destroyer',      dict(fore=4, aft=3, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=4, dev=4)),
    ('escort',         dict(fore=4, aft=3, exp=0, hang=0, sec=0, uni=0, eng=3, sci=3, tac=5, dev=4)),
    ('intel',          dict(fore=4, aft=3, exp=0, hang=0, sec=1, uni=0, eng=3, sci=4, tac=3, dev=4)),
    ('science',        dict(fore=3, aft=3, exp=0, hang=0, sec=1, uni=0, eng=3, sci=5, tac=3, dev=4)),
    ('cruiser',        dict(fore=4, aft=4, exp=0, hang=0, sec=0, uni=0, eng=5, sci=3, tac=3, dev=5)),
]

_GENERIC_PROFILE = dict(fore=4, aft=3, exp=0, hang=0, sec=0,
                         uni=0, eng=3, sci=3, tac=3, dev=4)


def _type_keyword_profile(ship_type: str) -> dict[str, int]:
    s = ship_type.lower()
    kw_dict = _GENERIC_PROFILE
    for keyword, kp in _KEYWORD_PROFILES:
        if keyword in s:
            kw_dict = kp; break

    return {
        'Fore Weapons':         kw_dict['fore'],
        'Deflector':            1,
        'Sec-Def':              kw_dict.get('sec', 0),
        'Engines':              1,
        'Warp Core':            1,
        'Shield':               1,
        'Aft Weapons':          kw_dict['aft'],
        'Experimental':         kw_dict.get('exp', 0),
        'Devices':              kw_dict['dev'],
        'Universal Consoles':   kw_dict.get('uni', 0),
        'Engineering Consoles': kw_dict['eng'],
        'Science Consoles':     kw_dict['sci'],
        'Tactical Consoles':    kw_dict['tac'],
        'Hangars':              kw_dict.get('hang', 0),
    }


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RecognisedItem:
    slot:        str
    slot_index:  int
    name:        str
    confidence:  float
    thumbnail:   Any   = None
    source_file: str   = ''
    bbox:        tuple = field(default_factory=tuple)


@dataclass
class ImportResult:
    build_type:   str
    ship_name:    str  = ''
    ship_type:    str  = ''
    ship_tier:    str  = ''
    ship_profile: dict = field(default_factory=dict)
    items:        list = field(default_factory=list)
    errors:       list = field(default_factory=list)
    warnings:     list = field(default_factory=list)


# ── WarpImporter ───────────────────────────────────────────────────────────────

class WarpImporter:
    """
    Main WARP import pipeline.

    Flow per screenshot:
      1. TextExtractor   → ship name, ship type, tier
      2. ShipDB          → exact slot profile from ship_list.json
      3. LayoutDetector  → bbox per slot (OCR labels + position + anchors)
      4. Per slot: crop → SETSIconMatcher → item name + confidence
      5. Merge across screenshots: highest confidence per (slot, index)
    """

    def __init__(self, sets_app):
        self._app     = sets_app
        self._layout  = None
        self._matcher = None
        self._text    = None
        self._shipdb  = None

    def process_folder(
        self,
        folder:      str | Path,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> ImportResult:
        folder = Path(folder)
        files  = sorted(f for f in folder.iterdir()
                        if f.suffix.lower() in SCREENSHOT_EXTENSIONS)
        if not files:
            return ImportResult(build_type='SPACE',
                                errors=[f'No images found in {folder}'])

        result = ImportResult(build_type='SPACE')
        best: dict[tuple[str, int], RecognisedItem] = {}

        for i, fpath in enumerate(files):
            if progress_cb:
                progress_cb(i, len(files), fpath.name)
            try:
                img         = self._load_image(fpath)
                file_result = self._process_image(img, str(fpath))
                if not result.ship_name and file_result.ship_name:
                    result.ship_name    = file_result.ship_name
                    result.ship_type    = file_result.ship_type
                    result.ship_tier    = file_result.ship_tier
                    result.ship_profile = file_result.ship_profile
                    result.build_type   = file_result.build_type
                for item in file_result.items:
                    key = (item.slot, item.slot_index)
                    if key not in best or item.confidence > best[key].confidence:
                        best[key] = item
                result.errors.extend(file_result.errors)
            except Exception as e:
                result.errors.append(f'{fpath.name}: {e}')
                log.exception(f'WarpImporter: {fpath}')

        result.items = list(best.values())
        return result

    def _process_image(self, img: np.ndarray, source: str) -> ImportResult:
        # Step 1 — extract ship info
        text_info  = self._get_text().extract_ship_info(img)
        ship_name  = text_info.get('ship_name', '')
        ship_type  = text_info.get('ship_type', '')
        build_type = 'GROUND' if text_info.get('build_type') == 'GROUND' else 'SPACE'

        # Step 2 — get exact slot profile from ship_list.json
        profile = self._get_shipdb().get_profile(ship_name, ship_type)

        result = ImportResult(
            build_type   = build_type,
            ship_name    = ship_name,
            ship_type    = ship_type,
            ship_tier    = text_info.get('ship_tier', ''),
            ship_profile = profile,
        )

        # Step 3 — layout detection constrained by profile
        layout  = self._get_layout().detect(img, build_type, profile)
        matcher = self._get_matcher()

        # Step 4 — match icons per slot (in canonical order)
        for slot_def in SLOT_ORDER.get(build_type, []):
            slot_name = slot_def['name']
            max_count = profile.get(slot_name, 0)
            if max_count == 0:
                continue

            bboxes = layout.get(slot_name, [])[:max_count]
            for idx, bbox in enumerate(bboxes):
                crop = self._crop(img, bbox)
                if crop is None or crop.size == 0:
                    continue
                name, conf, thumb = matcher.match(crop)
                if not name:
                    continue
                # Experimental slot: only Experimental Weapon items allowed
                if slot_def['exp'] and not self._is_experimental(name):
                    continue
                result.items.append(RecognisedItem(
                    slot        = slot_name,
                    slot_index  = idx,
                    name        = name,
                    confidence  = conf,
                    thumbnail   = thumb,
                    source_file = source,
                    bbox        = bbox,
                ))

        return result

    def _is_experimental(self, item_name: str) -> bool:
        try:
            for cat_items in self._app.cache.equipment.values():
                entry = cat_items.get(item_name, {})
                if isinstance(entry, dict) and entry.get('type') in EXPERIMENTAL_TYPES:
                    return True
        except Exception:
            pass
        return False

    def _load_image(self, path: Path) -> np.ndarray:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'Cannot read image: {path}')
        return img

    def _crop(self, img: np.ndarray, bbox: tuple) -> np.ndarray | None:
        x, y, w, h = bbox
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def _get_layout(self):
        if self._layout is None:
            from warp.recognition.layout_detector import LayoutDetector
            self._layout = LayoutDetector()
        return self._layout

    def _get_matcher(self):
        if self._matcher is None:
            from warp.recognition.icon_matcher import SETSIconMatcher
            self._matcher = SETSIconMatcher(self._app)
        return self._matcher

    def _get_text(self):
        if self._text is None:
            from warp.recognition.text_extractor import TextExtractor
            self._text = TextExtractor()
        return self._text

    def _get_shipdb(self) -> ShipDB:
        if self._shipdb is None:
            # Find cargo dir relative to SETS root
            here = Path(__file__).resolve().parent
            for _ in range(5):
                candidate = here / '.config' / 'cargo'
                if (candidate / 'ship_list.json').exists():
                    break
                here = here.parent
            else:
                candidate = Path('.config') / 'cargo'
            self._shipdb = ShipDB(candidate)
        return self._shipdb
