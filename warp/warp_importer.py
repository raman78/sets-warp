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
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

SCREENSHOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
TEMPLATE_CONF_THRESHOLD = 0.72
# Minimum confidence to include a recognition result in output
# Below this threshold the matcher is essentially guessing
MIN_ACCEPT_CONF = 0.40


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

# ── Traits (Personal / Starship / Reputation / Active) ─────────────────────────
# Separate screenshots from the Traits tab or STOCD overlay.
# Personal traits: up to 10 space + 5 ground active.
# Starship traits: up to 7 (5 base + 2 from Legendary/T6-X2).
# Reputation traits: up to 5 space + 5 ground.

SPACE_TRAITS_SLOT_ORDER: list[dict] = [
    {'name': 'Personal Space Traits',  'key': 'personal_traits',   'mandatory': True,  'max': 10, 'weapon': False, 'exp': False},
    {'name': 'Starship Traits',        'key': 'starship_traits',   'mandatory': True,  'max': 7,  'weapon': False, 'exp': False},
    {'name': 'Space Reputation',       'key': 'rep_traits',        'mandatory': True,  'max': 5,  'weapon': False, 'exp': False},
    {'name': 'Active Space Rep',       'key': 'active_rep_traits', 'mandatory': False, 'max': 5,  'weapon': False, 'exp': False},
]

GROUND_TRAITS_SLOT_ORDER: list[dict] = [
    {'name': 'Personal Ground Traits', 'key': 'personal_ground',   'mandatory': True,  'max': 10, 'weapon': False, 'exp': False},
    {'name': 'Ground Reputation',      'key': 'rep_ground_traits', 'mandatory': True,  'max': 5,  'weapon': False, 'exp': False},
    {'name': 'Active Ground Rep',      'key': 'active_ground_rep', 'mandatory': False, 'max': 5,  'weapon': False, 'exp': False},
]

# ── Bridge Officers ─────────────────────────────────────────────────────────────
# Boff abilities in the Status tab right panel.
# We treat each (profession, seat_index, rank) as a slot entry.

BOFFS_SLOT_ORDER: list[dict] = [
    {'name': 'Boff Tactical',      'key': 'boff_tac', 'mandatory': True,  'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Engineering',   'key': 'boff_eng', 'mandatory': True,  'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Science',       'key': 'boff_sci', 'mandatory': True,  'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Operations',    'key': 'boff_ops', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Intelligence',  'key': 'boff_int', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Command',       'key': 'boff_cmd', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Pilot',         'key': 'boff_plt', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Miracle Worker', 'key': 'boff_mw', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
    {'name': 'Boff Temporal',      'key': 'boff_tmp', 'mandatory': False, 'max': 20, 'weapon': False, 'exp': False},
]

# ── Captain Specializations ─────────────────────────────────────────────────────

SPEC_SLOT_ORDER: list[dict] = [
    {'name': 'Primary Specialization',   'key': 'spec_primary',   'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Secondary Specialization', 'key': 'spec_secondary', 'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
]

SLOT_ORDER = {
    'SPACE':         SPACE_SLOT_ORDER,
    'GROUND':        GROUND_SLOT_ORDER,
    'SPACE_TRAITS':  SPACE_TRAITS_SLOT_ORDER,
    'GROUND_TRAITS': GROUND_TRAITS_SLOT_ORDER,
    'BOFFS':         BOFFS_SLOT_ORDER,
    'SPEC':          SPEC_SLOT_ORDER,
}

SPACE_SLOTS        = [(s['name'], s['max']) for s in SPACE_SLOT_ORDER]
GROUND_SLOTS       = [(s['name'], s['max']) for s in GROUND_SLOT_ORDER]
SPACE_TRAITS_SLOTS = [(s['name'], s['max']) for s in SPACE_TRAITS_SLOT_ORDER]
GROUND_TRAITS_SLOTS= [(s['name'], s['max']) for s in GROUND_TRAITS_SLOT_ORDER]
BOFFS_SLOTS        = [(s['name'], s['max']) for s in BOFFS_SLOT_ORDER]
SPEC_SLOTS         = [(s['name'], s['max']) for s in SPEC_SLOT_ORDER]

SLOT_SPECS = {
    'SPACE':         SPACE_SLOTS,
    'GROUND':        GROUND_SLOTS,
    'SPACE_TRAITS':  SPACE_TRAITS_SLOTS,
    'GROUND_TRAITS': GROUND_TRAITS_SLOTS,
    'BOFFS':         BOFFS_SLOTS,
    'SPEC':          SPEC_SLOTS,
    'SPACE_SKILLS':  [],
    'GROUND_SKILLS': [],
}

# Weapon types that can only go in Experimental slot
EXPERIMENTAL_TYPES = frozenset({'Experimental Weapon'})

# Maps slot name → set of valid item 'type' values from cache
# Exact type strings come from scraper.py EQUIPMENT_TYPES keys
SLOT_VALID_TYPES: dict[str, frozenset] = {
    'Fore Weapons':          frozenset({'Ship Fore Weapon', 'Ship Weapon', 'Experimental Weapon'}),
    'Aft Weapons':           frozenset({'Ship Aft Weapon', 'Ship Weapon', 'Experimental Weapon'}),
    'Deflector':             frozenset({'Ship Deflector Dish', 'Ship Secondary Deflector'}),
    'Impulse':               frozenset({'Impulse Engine'}),
    'Engines':               frozenset({'Impulse Engine'}),
    'Warp Core':             frozenset({'Warp Engine', 'Singularity Engine'}),
    'Shield':                frozenset({'Ship Shields'}),
    'Shields':               frozenset({'Ship Shields'}),
    'Devices':               frozenset({'Ship Device'}),
    'Engineering Consoles':  frozenset({'Ship Engineering Console', 'Universal Console'}),
    'Science Consoles':      frozenset({'Ship Science Console', 'Universal Console'}),
    'Tactical Consoles':     frozenset({'Ship Tactical Console', 'Universal Console'}),
    'Hangar':                frozenset({'Hangar Bay'}),
}

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


def _profile_from_pixel_counts(pixel_counts: dict[str, int]) -> dict[str, int]:
    """
    Given slot counts measured from pixel analysis, find the closest
    matching keyword profile and use it to fill in slots that pixel
    analysis cannot measure (Sec-Def, Experimental, Hangars).
    Returns a merged profile: pixel counts + inferred unmeasurable slots.
    """
    # Measurable slots (pixel analysis can count these)
    MEASURABLE = {'Fore Weapons', 'Aft Weapons', 'Devices',
                  'Engineering Consoles', 'Science Consoles', 'Tactical Consoles'}

    # Score each keyword profile by sum of absolute differences on measurable slots
    def _score(kp: dict) -> int:
        kp_slots = {
            'Fore Weapons': kp['fore'], 'Aft Weapons': kp['aft'],
            'Devices': kp['dev'], 'Engineering Consoles': kp['eng'],
            'Science Consoles': kp['sci'], 'Tactical Consoles': kp['tac'],
        }
        return sum(abs(pixel_counts.get(slot, 0) - kp_slots.get(slot, 0))
                   for slot in MEASURABLE if pixel_counts.get(slot, 0) > 0)

    best_keyword, best_kp, best_score = '', _GENERIC_PROFILE, 999
    for keyword, kp in _KEYWORD_PROFILES:
        s = _score(kp)
        if s < best_score:
            best_score, best_keyword, best_kp = s, keyword, kp

    # Build merged profile: start from best keyword match, override with pixel counts
    merged = _type_keyword_profile(best_keyword)
    for slot, count in pixel_counts.items():
        if count > 0:
            merged[slot] = count
    try:
        from src.setsdebug import log as _sl
        _sl.info(f'WarpImporter: pixel→profile best={best_keyword!r} score={best_score}pts '
                 f'sec={merged.get("Sec-Def",0)} exp={merged.get("Experimental",0)} '
                 f'hang={merged.get("Hangars",0)}')
    except Exception:
        pass
    return merged


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
        self._index:   dict[str, dict] = {}  # lowercase name → ship entry
        self._by_type: dict[str, dict] = {}  # lowercase type → ship entry
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
                raw_name = ship.get('name') or ''
                name = (' '.join(raw_name) if isinstance(raw_name, list) else str(raw_name)).strip()
                if name:
                    self._index[name.lower()] = ship
                raw_type = ship.get('type') or ''
                stype = (' '.join(raw_type) if isinstance(raw_type, list) else str(raw_type)).strip()
                if stype:
                    self._by_type[stype.lower()] = ship
            log.info(f'ShipDB: loaded {len(self._ships)} ships, '
                     f'{len(self._by_type)} unique types')
        except Exception as e:
            log.warning(f'ShipDB load error: {e}')

    def get_profile(self, ship_name: str, ship_type: str) -> dict[str, int]:
        """
        Returns exact slot counts for a ship.
        ship_type is the primary key — it determines layout/slots.
        ship_name is cosmetic only (player-given name, irrelevant to slots).

        Priority:
          1. Exact type match
          2. Fuzzy type match (handles OCR errors, multi-line joins)
          3. Keyword fallback from type string
        """
        st = ship_type.lower().strip()

        # 1. Exact type match
        entry = self._by_type.get(st)
        if entry:
            log.debug(f'ShipDB exact type: {ship_type!r}')
            return self._entry_to_profile(entry)

        # 2. Fuzzy type match — handles OCR errors and extra/missing words
        if st and self._by_type:
            type_candidates = list(self._by_type.keys())

            # 2a. Word-subset match: OCR words are a subset of DB name words
            # e.g. 'Fleet Temporal Science Vessel' ⊆ 'Fleet Nautilus Temporal Science Vessel'
            ocr_words = set(st.split())
            subset_hits = [(c, self._by_type[c]) for c in type_candidates
                           if ocr_words.issubset(set(c.split()))]
            if len(subset_hits) == 1:
                # Unique subset match — high confidence
                log.debug(f'ShipDB subset match: {ship_type!r} → {subset_hits[0][0]!r}')
                return self._entry_to_profile(subset_hits[0][1])
            elif len(subset_hits) > 1:
                # Multiple subset matches — pick the one with fewest extra words
                best = min(subset_hits, key=lambda x: len(set(x[0].split()) - ocr_words))
                log.debug(f'ShipDB subset match (best of {len(subset_hits)}): '
                          f'{ship_type!r} → {best[0]!r}')
                return self._entry_to_profile(best[1])

            # 2b. Standard fuzzy match as fallback
            type_matches = get_close_matches(st, type_candidates, n=1, cutoff=0.68)
            if type_matches:
                entry = self._by_type[type_matches[0]]
                log.debug(f'ShipDB fuzzy type: {ship_type!r} → {type_matches[0]!r}')
                return self._entry_to_profile(entry)

        # 3. Keyword fallback from type string
        log.debug(f'ShipDB: type {ship_type!r} not found — using keyword fallback')
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
            'Universal Consoles':   _int(e.get('uniconsole'), 0) + _int(e.get('t5uconsole'), 0),
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

    def __init__(
        self,
        sets_app,
        build_type: str = 'SPACE',
        progress_callback: Callable[[int, str], None] | None = None,
        from_trainer: bool = False,
    ):
        self._app              = sets_app
        self._build_type       = build_type
        self._from_trainer     = from_trainer
        self._progress_callback = progress_callback
        self._interrupt_check = None
        self._layout  = None
        self._matcher = None
        self._text    = None
        self._shipdb  = None
        self._sync    = None   # WARPSyncClient — lazy init

    def set_interrupt_check(self, fn):
        # fn() returns True when processing should stop
        self._interrupt_check = fn

    def process_folder(
        self,
        folder:      str | Path,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> ImportResult:
        folder = Path(folder)
        files  = sorted(f for f in folder.iterdir()
                        if f.suffix.lower() in SCREENSHOT_EXTENSIONS)
        if not files:
            return ImportResult(build_type=self._build_type,
                                errors=[f'No images found in {folder}'])

        result = ImportResult(build_type=self._build_type)
        best: dict[tuple[str, int], RecognisedItem] = {}

        for i, fpath in enumerate(files):
            pct = int(i / len(files) * 90)
            if progress_cb:
                progress_cb(i, len(files), fpath.name)
            if self._interrupt_check and self._interrupt_check():
                break
            if self._progress_callback:
                self._progress_callback(pct, fpath.name)
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

    def _process_image(self, img: np.ndarray, source: str, profile_override: dict | None = None) -> ImportResult:
        _slog.info(f'WarpImporter._process_image: source={source} build_type={self._build_type}')
        # Step 1 — extract ship info via OCR.
        # build_type from caller sets the import mode but we always try OCR
        # for ship name/type — unless called from trainer (has _data_mgr attr).
        _is_trainer_call = self._from_trainer
        if _is_trainer_call:
            # Trainer always has build_type set and uses confirmed annotations
            build_type = self._build_type
            ship_name  = ''
            ship_type  = ''
            text_info  = {}
        else:
            # WARP dialog — run OCR to get ship info regardless of build_type
            text_info  = self._get_text().extract_ship_info(img)
            ship_name  = text_info.get('ship_name', '')
            ship_type  = text_info.get('ship_type', '')
            # Use caller's build_type as primary, OCR as confirmation
            if self._build_type in ('SPACE', 'GROUND', 'SPACE_TRAITS',
                                    'GROUND_TRAITS', 'BOFFS', 'SPEC',
                                    'SPACE_MIXED', 'GROUND_MIXED'):
                build_type = self._build_type
            else:
                build_type = 'GROUND' if text_info.get('build_type') == 'GROUND' else 'SPACE'
            _slog.info(f'WarpImporter: OCR result: name={ship_name!r} type={ship_type!r} '
                       f'ocr_build={text_info.get("build_type")!r} → using build_type={build_type!r}')

        # Step 2 — get exact slot profile from ship_list.json
        profile = self._get_shipdb().get_profile(ship_name, ship_type)
        _slog.info(f'WarpImporter: ShipDB profile for {ship_name!r}/{ship_type!r}: {dict((k,v) for k,v in profile.items() if v)}')
        # Apply profile override from confirmed annotations
        # Priority 1: explicitly passed override (from RecognitionWorker)
        # Priority 2: load from training_data/annotations.json on disk
        if not profile_override:
            profile_override = self._load_confirmed_profile(source)
        # If ship_name was empty (trainer mode), try from annotations on disk
        ship_tier = text_info.get('ship_tier', '')
        if not ship_name:
            _ann_ship = self._load_ship_info_from_annotations(source)
            if _ann_ship.get('ship_name'):
                ship_name = _ann_ship['ship_name']
                ship_type = _ann_ship.get('ship_type', '')
                ship_tier = _ann_ship.get('ship_tier', '')
                _slog.info(f'WarpImporter: ship info from annotations: '
                           f'{ship_name!r} / {ship_type!r} / {ship_tier!r}')
        if profile_override:
            for slot, count in profile_override.items():
                if count > profile.get(slot, 0):
                    profile[slot] = count
                    _slog.info(f'WarpImporter: profile override {slot}={count} (from confirmed annotations)')


        result = ImportResult(
            build_type   = build_type,
            ship_name    = ship_name,
            ship_type    = ship_type,
            ship_tier    = ship_tier,
            ship_profile = profile,
        )

        # Step 3a — if confirmed annotations exist for this exact file,
        # use them as ground-truth layout (exact bboxes, no pixel guessing)
        confirmed_layout = self._load_confirmed_layout(source)
        if confirmed_layout:
            _slog.info(f'WarpImporter: using confirmed layout from annotations '
                       f'({sum(len(v) for v in confirmed_layout.values())} bboxes)')
            layout = confirmed_layout
        else:
            # Step 3b — layout detection from pixel analysis
            layout = self._get_layout().detect(img, build_type, profile)

        # If ShipDB gave generic fallback (ship_name empty), refine profile
        # using actual icon counts from layout + keyword profile matching
        # Only refine slots NOT already set by confirmed annotations
        if not ship_name and layout:
            pixel_counts = {slot: len(boxes) for slot, boxes in layout.items() if boxes}
            if pixel_counts:
                refined = _profile_from_pixel_counts(pixel_counts)
                changed = False
                for slot, count in refined.items():
                    # Never override confirmed annotation counts
                    if slot in profile_override:
                        continue
                    if count > profile.get(slot, 0):
                        profile[slot] = count
                        changed = True
                if changed:
                    layout = self._get_layout().detect(img, build_type, profile)
                    _slog.info(f'WarpImporter: refined profile from pixel counts: '
                               f'{dict((k,v) for k,v in profile.items() if v)}')

        matcher = self._get_matcher()

        # Step 4 — match icons per slot (in canonical order)
        for slot_def in SLOT_ORDER.get(build_type, []):
            slot_name = slot_def['name']
            max_count = profile.get(slot_name, 0)
            if max_count == 0:
                continue

            bboxes = layout.get(slot_name, [])[:max_count]
            if not bboxes:
                _slog.info(f'  [{slot_name}] no bboxes from layout (max_count={max_count})')
            for idx, bbox in enumerate(bboxes):
                crop = self._crop(img, bbox)
                if crop is None or crop.size == 0:
                    _slog.info(f'  [{slot_name}][{idx}] bbox={bbox} — empty crop, skipped')
                    continue
                name, conf, thumb = matcher.match(crop)
                _slog.info(f'  [{slot_name}][{idx}] bbox={bbox} crop={crop.shape[1]}x{crop.shape[0]} → {name!r} conf={conf:.2f}')
                if not name:
                    continue
                # Reject low-confidence results — below threshold is a guess
                if conf < MIN_ACCEPT_CONF:
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — conf {conf:.2f} < {MIN_ACCEPT_CONF}')
                    continue
                # Validate item type matches slot category
                if not self._item_valid_for_slot(name, slot_name):
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — {name!r} wrong type for slot')
                    continue
                # Experimental slot: only Experimental Weapon items allowed
                if slot_def['exp'] and not self._is_experimental(name):
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — not experimental weapon: {name!r}')
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
                # Contribute to community knowledge (non-blocking, only high-conf)
                if conf >= TEMPLATE_CONF_THRESHOLD:
                    sync = self._get_sync_client()
                    if sync is not None:
                        sync.contribute(crop, name, confirmed=False)

        return result

    def _load_ship_info_from_annotations(self, source: str) -> dict:
        """Read Ship Name, Ship Type, Ship Tier from confirmed annotations on disk."""
        try:
            here = Path(__file__).resolve().parent
            for _ in range(6):
                ann_path = here / 'warp' / 'training_data' / 'annotations.json'
                if ann_path.exists(): break
                here = here.parent
            else:
                return {}
            import json
            data = json.loads(ann_path.read_text(encoding='utf-8'))
            fname = Path(source).name
            result = {}
            for a in data.get(fname, []):
                if a.get('state') != 'confirmed': continue
                slot = a.get('slot', '')
                name = a.get('name', '').strip()
                if not name: continue
                if slot == 'Ship Name':  result['ship_name'] = name
                elif slot == 'Ship Type': result['ship_type'] = name
                elif slot == 'Ship Tier': result['ship_tier'] = name
            return result
        except Exception as e:
            _slog.debug(f'WarpImporter: _load_ship_info_from_annotations error: {e}')
            return {}

    def _load_confirmed_layout(self, source: str) -> dict[str, list] | None:
        """
        If confirmed annotations exist for this exact source file,
        return them as a layout dict {slot_name: [bbox, ...]}.
        This gives pixel-perfect bboxes instead of estimated positions.
        Returns None if no confirmed annotations found.
        """
        _NON_ICON = frozenset({'Ship Name', 'Ship Type', 'Ship Tier',
                               'Primary Specialization', 'Secondary Specialization'})
        try:
            here = Path(__file__).resolve().parent
            for _ in range(6):
                ann_path = here / 'warp' / 'training_data' / 'annotations.json'
                if ann_path.exists(): break
                here = here.parent
            else:
                return None
            import json
            data = json.loads(ann_path.read_text(encoding='utf-8'))
            fname = Path(source).name
            ann_list = data.get(fname, [])
            layout: dict[str, list] = {}
            for a in ann_list:
                if a.get('state') != 'confirmed': continue
                slot = a.get('slot', '')
                bbox = a.get('bbox')
                if not slot or not bbox or slot in _NON_ICON: continue
                if slot not in layout:
                    layout[slot] = []
                # Convert [x,y,w,h] list to tuple
                layout[slot].append(tuple(bbox))
            if layout:
                _slog.info(f'WarpImporter: confirmed layout from disk: '
                           f'{dict((k,len(v)) for k,v in layout.items())}')
            return layout if layout else None
        except Exception as e:
            _slog.debug(f'WarpImporter: _load_confirmed_layout error: {e}')
            return None

    def _load_confirmed_profile(self, source: str) -> dict[str, int]:
        """Load confirmed annotation counts per slot from training_data on disk.
        Returns {slot_name: count} for the given source image file."""
        try:
            here = Path(__file__).resolve().parent
            for _ in range(6):
                ann_path = here / 'warp' / 'training_data' / 'annotations.json'
                if ann_path.exists():
                    break
                here = here.parent
            else:
                return {}
            import json
            data = json.loads(ann_path.read_text(encoding='utf-8'))
            fname = Path(source).name
            ann_list = data.get(fname, [])
            _NON_PROFILE = frozenset({'Ship Name', 'Ship Type', 'Ship Tier',
                                      'Primary Specialization', 'Secondary Specialization'})
            counts: dict[str, int] = {}
            ship_name_ann = ''
            ship_type_ann = ''
            for a in ann_list:
                if a.get('state') != 'confirmed': continue
                slot = a.get('slot', '')
                name = a.get('name', '')
                if slot == 'Ship Name' and name:
                    ship_name_ann = name
                elif slot == 'Ship Type' and name:
                    ship_type_ann = name
                elif slot and slot not in _NON_PROFILE:
                    counts[slot] = counts.get(slot, 0) + 1
            if counts:
                _slog.info(f'WarpImporter: confirmed profile from disk for {fname}: {counts}')
            # If ship name/type found in annotations, try ShipDB for exact profile
            if ship_name_ann or ship_type_ann:
                _slog.info(f'WarpImporter: ship from annotations: {ship_name_ann!r} / {ship_type_ann!r}')
                try:
                    db_profile = self._get_shipdb().get_profile(ship_name_ann, ship_type_ann)
                    _slog.info(f'WarpImporter: ShipDB profile from annotations: {dict((k,v) for k,v in db_profile.items() if v)}')
                    # Merge: ShipDB for exact counts, confirmed annotations as minimum
                    for slot, count in counts.items():
                        if count > db_profile.get(slot, 0):
                            db_profile[slot] = count
                    return db_profile
                except Exception as _dbe:
                    _slog.debug(f'WarpImporter: ShipDB lookup failed: {_dbe}')
            return counts
        except Exception as e:
            _slog.debug(f'WarpImporter: _load_confirmed_profile error: {e}')
            return {}

    def _item_valid_for_slot(self, item_name: str, slot_name: str) -> bool:
        """Check that item type from cache matches what the slot expects.
        Returns True if no type constraint exists for this slot (permissive)."""
        valid_types = SLOT_VALID_TYPES.get(slot_name)
        if not valid_types:
            return True  # no constraint defined — allow
        try:
            for cat_items in self._app.cache.equipment.values():
                entry = cat_items.get(item_name)
                if entry is None:
                    continue
                item_type = entry.get('type', '') if isinstance(entry, dict) else ''
                if item_type in valid_types:
                    return True
                # Item found in cache but wrong type
                _slog.info(f'  _item_valid_for_slot: {item_name!r} type={item_type!r} not valid for {slot_name!r}')
                return False
        except Exception:
            pass
        # Item not found in cache — allow (may be a new item we don't know)
        return True

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
            self._matcher = SETSIconMatcher(self._app,
                                            sync_client=self._get_sync_client())
        return self._matcher

    def _get_sync_client(self):
        if self._sync is None:
            try:
                from warp.knowledge.sync_client import WARPSyncClient
                self._sync = WARPSyncClient()
                log.info('WARP: sync client initialized')
            except Exception as e:
                log.warning(f'WARP: sync client unavailable: {e}')
                self._sync = None
        return self._sync

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
