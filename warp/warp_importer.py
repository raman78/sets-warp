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

# Virtual placeholders for empty/inactive slot positions. Mirrors
# `warp.trainer.training_data.VIRTUAL_ITEM_NAMES`; defined locally so
# warp_importer doesn't pull in the trainer package on the hot path.
VIRTUAL_ITEM_NAMES = frozenset({'__empty__', '__inactive__'})


def _bbox_iou(a, b) -> float:
    """IoU for two (x, y, w, h) bboxes."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

# Minimum confidence to include a recognition result in output
# Below this threshold the matcher is essentially guessing
MIN_ACCEPT_CONF = 0.35
# ── P5: Anchoring constants ──────────────────────────────────────────────────
# Slots used as reference points for layout recalibration
ANCHOR_SLOTS = frozenset({'Deflector', 'Engines', 'Warp Core', 'Shield'})
RECALIBRATION_MIN_CONF = 0.85


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
    {'name': 'Kit Modules',      'key': 'kit_modules',    'mandatory': True,  'max': 6, 'weapon': False, 'exp': False},
    {'name': 'Kit',              'key': 'kit',            'mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Body Armor',       'key': 'armor',          'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
    {'name': 'EV Suit',          'key': 'ev_suit',        'mandatory': False, 'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Personal Shield',  'key': 'personal_shield','mandatory': True,  'max': 1, 'weapon': False, 'exp': False},
    {'name': 'Weapons',          'key': 'weapons',        'mandatory': True,  'max': 2, 'weapon': True,  'exp': False},
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
    'SPACE_BOFFS':   BOFFS_SLOT_ORDER,   # same slot structure, different write target
    'GROUND_BOFFS':  BOFFS_SLOT_ORDER,
    'SPEC':          SPEC_SLOT_ORDER,
    # MIXED = all slot groups combined; used as fallback when no confirmed_layout exists.
    # layout_detector returns only the bboxes it actually finds, so unused slots
    # simply produce 0 bboxes and are silently skipped.
    'SPACE_MIXED':  (SPACE_SLOT_ORDER + BOFFS_SLOT_ORDER +
                     SPACE_TRAITS_SLOT_ORDER + SPEC_SLOT_ORDER),
    'GROUND_MIXED': (GROUND_SLOT_ORDER + BOFFS_SLOT_ORDER +
                     GROUND_TRAITS_SLOT_ORDER + SPEC_SLOT_ORDER),
}

# Global slot_def lookup (slot_name → slot_def) across all slot orders, in canonical sequence.
# Used when confirmed layout contains slots beyond the current build_type's order
# (e.g. a SPACE screenshot that also has traits and boff abilities annotated).
_ALL_SLOT_DEFS: dict[str, dict] = {}
for _order_list in SLOT_ORDER.values():
    for _sd in _order_list:
        _ALL_SLOT_DEFS.setdefault(_sd['name'], _sd)

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
    'Experimental':          frozenset({'Experimental Weapon'}),
    'Deflector':             frozenset({'Ship Deflector Dish'}),
    'Sec-Def':               frozenset({'Ship Secondary Deflector'}),
    'Impulse':               frozenset({'Impulse Engine'}),
    'Engines':               frozenset({'Impulse Engine'}),
    'Warp Core':             frozenset({'Warp Engine', 'Singularity Engine'}),
    'Shield':                frozenset({'Ship Shields'}),
    'Shields':               frozenset({'Ship Shields'}),
    'Devices':               frozenset({'Ship Device'}),
    'Engineering Consoles':  frozenset({'Ship Engineering Console', 'Universal Console'}),
    'Science Consoles':      frozenset({'Ship Science Console', 'Universal Console'}),
    'Tactical Consoles':     frozenset({'Ship Tactical Console', 'Universal Console'}),
    'Universal Consoles':    frozenset({'Universal Console', 'Ship Tactical Console',
                                        'Ship Engineering Console', 'Ship Science Console'}),
    'Hangar':                frozenset({'Hangar Bay'}),
    'Hangars':               frozenset({'Hangar Bay'}),
    # Ground equipment
    'Body Armor':            frozenset({'Body Armor'}),
    'EV Suit':               frozenset({'EV Suit'}),
    'Personal Shield':       frozenset({'Personal Shield'}),
    'Weapons':               frozenset({'Ground Weapon'}),
    'Kit':                   frozenset({'Kit'}),
    'Kit Modules':           frozenset({'Kit Module'}),
    'Ground Devices':        frozenset({'Ground Device'}),
}

# Slot → trait category in cache.traits[env][cat] (or cache.starship_traits)
TRAIT_SLOT_CATEGORY: dict[str, tuple[str, str]] = {
    'Personal Space Traits':  ('space',  'personal'),
    'Space Reputation':       ('space',  'rep'),
    'Active Space Rep':       ('space',  'active_rep'),
    'Personal Ground Traits': ('ground', 'personal'),
    'Ground Reputation':      ('ground', 'rep'),
    'Active Ground Rep':      ('ground', 'active_rep'),
    # Starship Trait uses its own flat dict — handled separately
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


# ── BOFF ability slot computation from ShipDB seating data ───────────────────
# Rank names cover all STO factions (English / Klingon / Romulan / Dominion).
# Dict is ordered longest-first so 'Lieutenant Commander' is matched before
# 'Lieutenant' (simple startswith matching).
_BOFF_RANK_SLOTS: dict[str, int] = {
    'Lieutenant Commander': 3,
    'Commander':            4,
    'Lieutenant':           2,
    'Ensign':               1,
    # Romulan / KDF / Dominion faction equivalents
    'Subcommander':         4,
    'Centurion':            3,
    'Fourth':               2,
    'Warrior':              1,
    'Citizen':              1,
    'Fifth':                1,
    'Third':                3,
    'Second':               4,
}

_BOFF_PROF_TO_SLOT: dict[str, str] = {
    'Tactical':           'Boff Tactical',
    'Engineering':        'Boff Engineering',
  #  'Operations':         'Boff Engineering',
    'Science':            'Boff Science',
    'Command':            'Boff Command',
    'Intelligence':       'Boff Intelligence',
    'Miracle Worker':     'Boff Miracle Worker',
    'Temporal Operative': 'Boff Temporal',
    'Pilot':              'Boff Pilot',
}

# Game-defined maximums for slots not covered by ShipDB equipment data.
# Applied when build_type is not a trainer call.
# layout_detector pixel analysis returns the actual count (≤ these caps).
_GAME_SLOT_MAXES: dict[str, int] = {
    'Personal Space Traits':  11,  # character-level cap
    'Personal Ground Traits': 11,
    'Starship Traits':         5,  # base T6 cap; T6-X/X2 adds +1/+2 via tier logic below
    'Space Reputation':        5,  # always 5 in STO
    'Ground Reputation':       5,
    'Active Space Rep':        5,
    'Active Ground Rep':       5,
    # BOFF fallback maximums — used only when ShipDB lookup fails.
    # For successful lookups _boff_profile_from_shipdb() gives exact values.
    'Boff Tactical':          12,
    'Boff Engineering':       12,
    'Boff Science':           12,
    'Boff Command':            6,
    'Boff Intelligence':       6,
    'Boff Pilot':              6,
    'Boff Miracle Worker':     6,
    'Boff Temporal':           6,
}


def _boff_profile_from_shipdb(boffs: list) -> dict[str, int]:
    """
    Compute BOFF ability-slot counts per profession from ShipDB seating list.

    Entry format: '<Rank> <Profession>[-<Specialization>]'
    e.g. 'Commander Tactical-Miracle Worker', 'Lieutenant Commander Universal'

    Universal seats can hold any profession's BOFF — their ability count is
    added to every recognized profession so the layout_detector can find the
    icons regardless of what the player placed there.
    Returns empty dict for empty/invalid input.
    """
    dedicated: dict[str, int] = {}
    universal_slots = 0

    for b in (boffs or []):
        if not b:
            continue
        rank_slots = 0
        rest = b
        for rank, slots in _BOFF_RANK_SLOTS.items():
            if b.startswith(rank + ' ') or b == rank:
                rank_slots = slots
                rest = b[len(rank):].strip()
                break
        if not rank_slots:
            continue
        primary_prof = rest.split('-')[0].strip()
        if primary_prof == 'Universal':
            universal_slots += rank_slots
        else:
            slot = _BOFF_PROF_TO_SLOT.get(primary_prof)
            if slot:
                dedicated[slot] = dedicated.get(slot, 0) + rank_slots

    if universal_slots:
        # Distribute Universal slots to every profession present (or default three)
        targets = list(dedicated) or ['Boff Tactical', 'Boff Engineering', 'Boff Science']
        for slot in targets:
            dedicated[slot] = dedicated.get(slot, 0) + universal_slots

    return dedicated


# ── ShipDB — primary source of truth for slot counts ──────────────────────────

def _parse_tier_num(tier_str: str) -> int:
    """Extract integer tier from OCR string like 'T6-X2' → 6. Returns 0 if absent."""
    if not tier_str:
        return 0
    import re
    m = re.search(r'[Tt](\d+)', str(tier_str))
    return int(m.group(1)) if m else 0


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
        # Display-name index: OCR sees in-game text built from
        # displayprefix + displayclass + displaytype + name tokens.
        # Each entry: (words_frozenset, tier_int, ship_dict).
        self._display_index: list[tuple[frozenset, int, dict]] = []
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
                # Build display-word set from displayprefix/class/type + name
                disp_parts: list[str] = []
                for key in ('displayprefix', 'displayclass', 'displaytype'):
                    v = ship.get(key)
                    if v:
                        disp_parts.append(str(v))
                if name:
                    disp_parts.append(name)
                disp_words = frozenset(' '.join(disp_parts).lower().split())
                try:
                    tier = int(ship.get('tier') or 0)
                except (TypeError, ValueError):
                    tier = 0
                if disp_words:
                    self._display_index.append((disp_words, tier, ship))
            log.info(f'ShipDB: loaded {len(self._ships)} ships, '
                     f'{len(self._by_type)} unique types, '
                     f'{len(self._display_index)} display entries')
        except Exception as e:
            log.warning(f'ShipDB load error: {e}')

    def get_profile(self, ship_name: str, ship_type: str,
                    ship_tier: str = '') -> dict[str, int]:
        """
        Returns exact slot counts for a ship.
        ship_type is the primary key — it determines layout/slots.
        ship_name is cosmetic only (player-given name, irrelevant to slots).
        ship_tier (e.g. 'T6-X2') — used to disambiguate display-name candidates.

        Priority:
          1. Exact type match
          2a. Word-subset type match
          2b. Display-name match (OCR words ⊆ display words) + tier filter
          2c. Fuzzy type match
          3. Keyword fallback
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

            # 2b. Display-name match — the `type` field in ship_list.json is
            # generic ("Cruiser", "Destroyer"; 44 unique values), but the
            # in-game text OCR sees combines displayprefix+displayclass+displaytype
            # (e.g. "Fleet Yamaguchi Support Cruiser"). Match OCR words against
            # the display-word index; tier disambiguates siblings (T5 Retrofit vs T6).
            tier_num = _parse_tier_num(ship_tier)
            disp_hits = [(dw, t, s) for (dw, t, s) in self._display_index
                         if ocr_words and ocr_words.issubset(dw)]
            if disp_hits and tier_num:
                tier_filtered = [h for h in disp_hits if h[1] == tier_num]
                if tier_filtered:
                    disp_hits = tier_filtered
            if len(disp_hits) == 1:
                _, _, ship = disp_hits[0]
                log.debug(f'ShipDB display match: {ship_type!r}+{ship_tier!r} '
                          f'→ {ship.get("name")!r}')
                return self._entry_to_profile(ship)
            elif len(disp_hits) > 1:
                # Prefer the entry with fewest extra words (closest to OCR text)
                best = min(disp_hits, key=lambda h: len(h[0] - ocr_words))
                log.debug(f'ShipDB display match (best of {len(disp_hits)}): '
                          f'{ship_type!r}+{ship_tier!r} → {best[2].get("name")!r}')
                return self._entry_to_profile(best[2])

            # 2c. Standard fuzzy match as fallback
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

        profile = {
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
        # BOFF ability counts from ship seating — derived from rank × profession
        profile.update(_boff_profile_from_shipdb(e.get('boffs') or []))
        return profile


# ── Keyword fallback profiles ──────────────────────────────────────────────────
# Used ONLY when ship not found in ship_list.json.
# Conservative estimates — better to miss a slot than hallucinate one.

_KEYWORD_PROFILES: list[tuple[str, dict]] = [
    # (keyword_in_type_lowercase, profile)
    # Most specific first — confirmed against actual STO ships.
    # exp=0 and hang=0 by default: these slots are RARE, only specific ships.
    # ShipDB (ship_list.json) is the primary source; this is only the fallback.
    ('carrier',        dict(fore=3, aft=3, exp=0, hang=2, sec=0, uni=0, eng=4, sci=3, tac=3, dev=4)),
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
    ('cruiser',        dict(fore=4, aft=4, exp=0, hang=0, sec=0, uni=0, eng=5, sci=3, tac=3, dev=4)),
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
    # Original detector slot key — preserved when `slot` gets remapped to a
    # canonical profession-named slot (e.g. seat key `Boff Seat L[T+P]_510`
    # → ability slot `Boff Tactical`). Empty string when no remap occurred.
    # Consumers that need seat-level info (warp_dialog cluster→seat matching)
    # should prefer `seat_key` when non-empty.
    seat_key:    str   = ''


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
            base_pct = int(i / len(files) * 90)
            end_pct  = int((i + 1) / len(files) * 90)
            if progress_cb:
                progress_cb(i, len(files), fpath.name)
            if self._interrupt_check and self._interrupt_check():
                break
            if self._progress_callback:
                self._progress_callback(base_pct, f'Analysing {fpath.name}…')
            try:
                img         = self._load_image(fpath)
                file_result = self._process_image(
                    img, str(fpath), _base_pct=base_pct, _end_pct=end_pct)
                if file_result.ship_name:
                    _upgrade = (
                        not result.ship_name or  # no result yet
                        (file_result.ship_tier and not result.ship_tier)  # new has tier, current doesn't
                    )
                    if _upgrade:
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
        if self._progress_callback:
            self._progress_callback(100, 'Done')
        return result

    def _process_image(self, img: np.ndarray, source: str, profile_override: dict | None = None,
                       _base_pct: int = 0, _end_pct: int = 90) -> ImportResult:
        _slog.info(f'####### WARP: {Path(source).name} | {self._build_type} #######')
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
            # Use caller's build_type as primary, OCR as confirmation.
            # Upgrade SPACE→SPACE_MIXED / GROUND→GROUND_MIXED when OCR signals
            # a richer screen type (broadside screenshots contain equipment +
            # traits + boffs simultaneously).  Never downgrade.
            _ocr_bt = text_info.get('build_type', '')
            if self._build_type in ('SPACE', 'GROUND', 'SPACE_TRAITS',
                                    'GROUND_TRAITS', 'BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS',
                                    'SPEC', 'SPACE_MIXED', 'GROUND_MIXED'):
                build_type = self._build_type
                if build_type == 'SPACE' and _ocr_bt in ('SPACE_TRAITS', 'SPACE_MIXED'):
                    build_type = 'SPACE_MIXED'
                    _slog.info('WarpImporter: upgraded SPACE → SPACE_MIXED (OCR detected richer screen)')
                elif build_type == 'SPACE' and _ocr_bt in ('BOFFS', 'SPACE_BOFFS') and text_info.get('scan_scope') == 'full':
                    build_type = _ocr_bt  # SPACE_BOFFS preferred over generic BOFFS
                    _slog.info(f'WarpImporter: upgraded SPACE → {build_type} (dedicated BOFFS screen, full scan only)')
                elif build_type == 'SPACE' and _ocr_bt == 'GROUND_BOFFS':
                    build_type = 'GROUND_BOFFS'
                    _slog.info('WarpImporter: upgraded SPACE → GROUND_BOFFS (OCR detected ground boff screen)')
                elif build_type == 'SPACE' and _ocr_bt == 'SPEC':
                    build_type = 'SPEC'
                    _slog.info('WarpImporter: upgraded SPACE → SPEC (OCR detected specialization screen)')
                elif build_type == 'GROUND' and _ocr_bt in ('GROUND_TRAITS', 'GROUND_MIXED'):
                    build_type = 'GROUND_MIXED'
                    _slog.info('WarpImporter: upgraded GROUND → GROUND_MIXED (OCR detected richer screen)')
                elif build_type == 'GROUND' and _ocr_bt == 'GROUND_BOFFS':
                    build_type = 'GROUND_BOFFS'
                    _slog.info('WarpImporter: upgraded GROUND → GROUND_BOFFS (OCR detected ground boff screen)')
            else:
                build_type = 'GROUND' if _ocr_bt == 'GROUND' else 'SPACE'
            _slog.info(f'WarpImporter: OCR result: name={ship_name!r} type={ship_type!r} '
                       f'ocr_build={text_info.get("build_type")!r} → using build_type={build_type!r}')

        # Step 2 — get exact slot profile from ship_list.json
        # Skip for GROUND/GROUND_MIXED — ShipDB contains space ship data only
        _is_ground = build_type in ('GROUND', 'GROUND_MIXED')
        _no_ship_profile = _is_ground or build_type in ('SPEC', 'BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS',
                                                         'SPACE_TRAITS', 'GROUND_TRAITS')
        ship_tier = text_info.get('ship_tier', '')
        if _no_ship_profile:
            profile = {}
            _slog.info(f'WarpImporter: {build_type} build — skipping ShipDB lookup')
        else:
            profile = self._get_shipdb().get_profile(ship_name, ship_type, ship_tier)
            _slog.info(f'WarpImporter: ShipDB profile for {ship_name!r}/{ship_type!r}/{ship_tier!r}: {dict((k,v) for k,v in profile.items() if v)}')
        if _is_trainer_call:
            # Trainer (WARP CORE): annotation counts are authoritative for equipment —
            # the user has confirmed every bbox, so the profile must match exactly.
            if not profile_override:
                profile_override = self._load_confirmed_profile(source)
            for slot, count in (profile_override or {}).items():
                if count > profile.get(slot, 0):
                    profile[slot] = count
                    _slog.info(f'WarpImporter: trainer profile {slot}={count} (confirmed)')
            # Trait/rep/boff slots: confirmed count reflects partial annotation —
            # user may not have confirmed all visible items. Apply game caps so the
            # layout detector's full detection isn't artificially capped.
            for slot, max_val in _GAME_SLOT_MAXES.items():
                if max_val > profile.get(slot, 0):
                    profile[slot] = max_val
        else:
            # WARP dialog import: coded game rules only — annotations are training data.
            # Traits / Rep / Active Rep: fixed STO game caps.
            # BOFF slots: already set by _boff_profile_from_shipdb via _entry_to_profile;
            #   _GAME_SLOT_MAXES fallbacks apply only when ShipDB lookup failed (no boffs data).
            for slot, max_val in _GAME_SLOT_MAXES.items():
                if max_val > profile.get(slot, 0):
                    profile[slot] = max_val
            # T6-X / T6-X2 tier upgrades (cumulative per level):
            #   T6-X  (level 1): +1 Universal Console, +1 Starship Trait, +1 Device
            #   T6-X2 (level 2): additional +1 each → total +2 vs base T6
            if '-X' in ship_tier:
                _x_bonus = 2 if 'X2' in ship_tier else 1
                if profile.get('Devices', 0) > 0:
                    profile['Devices'] += _x_bonus
                    _slog.info(f'WarpImporter: {ship_tier} — Devices +{_x_bonus} → {profile["Devices"]}')
                profile['Universal Consoles'] = profile.get('Universal Consoles', 0) + _x_bonus
                _slog.info(f'WarpImporter: {ship_tier} — Universal Consoles +{_x_bonus} → {profile["Universal Consoles"]}')
                profile['Starship Traits'] = profile.get('Starship Traits', 5) + _x_bonus
                _slog.info(f'WarpImporter: {ship_tier} — Starship Traits +{_x_bonus} → {profile["Starship Traits"]}')
            _slog.debug(f'WarpImporter: game-rule profile (traits/rep/boff): '
                        f'{dict((k,v) for k,v in profile.items() if "Boff" in k or "Trait" in k or "Rep" in k)}')


        result = ImportResult(
            build_type   = build_type,
            ship_name    = ship_name,
            ship_type    = ship_type,
            ship_tier    = ship_tier,
            ship_profile = profile,
        )

        # Step 3 — layout detection.
        # ARCHITECTURE RULE: annotations.json is TRAINING DATA ONLY. WARP must
        # perform clean detection via layout_detector, never fall back to user
        # annotations as output — otherwise we hide detection bugs and can't
        # measure real recognition quality.
        # Only WARP CORE (trainer) uses confirmed annotations — there every
        # bbox was explicitly confirmed by the user and represents ground truth
        # being fed back into training.
        # NOTE: The MIXED branch is DISABLED (2026-04-19) by user request to
        # force clean detection. Old behavior preserved below for reference —
        # do NOT re-enable without explicit approval.
        # _use_confirmed = 'MIXED' in build_type or _is_trainer_call
        _use_confirmed = _is_trainer_call
        confirmed_layout = self._load_confirmed_layout(source) if _use_confirmed else None
        
        _needs_matcher = build_type in (
            'SPACE_MIXED', 'GROUND_MIXED',
            'BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS',
            # Traits use the structure-driven trait_grid detector (Strategy 0)
            # which probes icons through icon_matcher.classify_patch to label
            # each row-group's section independently.
            'SPACE_TRAITS', 'GROUND_TRAITS',
        )
        layout = self._get_layout().detect(
            img, build_type, profile,
            icon_matcher=self._get_matcher() if _needs_matcher else None,
            app_cache=self._app.cache if _needs_matcher else None,
        )
        _slog.info(
            f'WarpImporter: layout → {len(layout)} slot groups, '
            f'{sum(len(v) for v in layout.values())} bboxes ({build_type})'
        )

        if confirmed_layout:
            _slog.info(f'WarpImporter: merging confirmed layout ({build_type}) — '
                       f'{sum(len(v) for v in confirmed_layout.values())} bboxes from annotations')
            # IoU-based merge: confirmed bboxes win when they overlap a detected one
            # (user drew the exact pixel-perfect rect), but unmatched detected bboxes
            # are KEPT so positions where the user deleted a confirmation get
            # re-proposed by the detector instead of vanishing from the review list.
            for slot, conf_boxes in confirmed_layout.items():
                det_boxes = list(layout.get(slot, []))
                used = [False] * len(conf_boxes)
                merged: list = []
                for d in det_boxes:
                    best_i, best_iou = -1, 0.0
                    for i, c in enumerate(conf_boxes):
                        if used[i]: continue
                        iou = _bbox_iou(d, c)
                        if iou > best_iou:
                            best_iou, best_i = iou, i
                    if best_i >= 0 and best_iou >= 0.3:
                        merged.append(conf_boxes[best_i])
                        used[best_i] = True
                    else:
                        merged.append(d)
                # Append confirmed bboxes that didn't match any detected bbox
                # (user drew outside the detector grid).
                for i, c in enumerate(conf_boxes):
                    if not used[i]:
                        merged.append(c)
                layout[slot] = merged

        # If ShipDB gave generic fallback (ship_name empty), refine profile
        # using actual icon counts from layout + keyword profile matching.
        # Only refine slots NOT already set by confirmed annotations.
        # Skip for GROUND/GROUND_MIXED — _profile_from_pixel_counts is space-only
        # (MEASURABLE set contains only space slots), so it would pick a random
        # space ship profile and corrupt the ground layout on the second run.
        if not ship_name and layout and not _is_ground:
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
                    # Keep confirmed layout — re-detection would overwrite pixel-perfect bboxes
                    if not confirmed_layout:
                        layout = self._get_layout().detect(img, build_type, profile)
                    _slog.info(f'WarpImporter: refined profile from pixel counts: '
                               f'{dict((k,v) for k,v in profile.items() if v)}')

        matcher = self._get_matcher()

        # Step 4 — match icons per slot (in canonical order)
        # When confirmed layout is available, process every slot that has
        # bboxes — confirmed by the user OR freshly detected. Filtering by
        # confirmed_layout alone would skip slots the detector found but the
        # user hasn't annotated yet (e.g. user confirms Fore Weapons but not
        # Aft Weapons; without `layout.keys()` in the union, autodetect would
        # never re-propose Aft Weapons positions).
        if confirmed_layout:
            relevant_slots = set(confirmed_layout.keys()) | set(layout.keys())
            slot_defs_to_process = [sd for sd in _ALL_SLOT_DEFS.values()
                                    if sd['name'] in relevant_slots]
        else:
            slot_defs_to_process = list(SLOT_ORDER.get(build_type, []))

        # Add dynamically detected BOFF seats to the processing list.
        # Run regardless of confirmed_layout — `Boff Seat L[T]_<y>` keys are
        # dynamic and never appear in `_ALL_SLOT_DEFS`, so the trainer-mode
        # `slot_defs_to_process` filter would always drop them. In WARP CORE
        # the user typically confirms equipment first and BOFFs later; without
        # this branch, autodetect never proposes BOFF abilities.
        seen_seat_keys = {sd['name'] for sd in slot_defs_to_process}
        for key in layout.keys():
            if key.startswith('Boff Seat') and key not in seen_seat_keys:
                slot_defs_to_process.append({
                    'name': key, 'key': '', 'mandatory': False, 'max': 4, 'weapon': False, 'exp': False
                })
                # Add them to profile so they are not skipped by max_count limit
                profile[key] = 4

        # Build per-slot candidate sets restricted by SLOT_VALID_TYPES.
        # This prevents template matching from picking items of the wrong type
        # (e.g. a shield icon matching the Warp Core slot at conf=1.00).
        slot_candidates = self._build_slot_candidates(slot_defs_to_process)

        # Count total bboxes upfront for granular progress reporting.
        total_bboxes = sum(
            len(layout.get(sd['name'], [])[:profile.get(sd['name'], 0)])
            for sd in slot_defs_to_process
            if profile.get(sd['name'], 0) > 0
        )
        processed_bboxes = 0

        # P5: Dynamic anchoring state
        current_dy = 0
        found_anchor = False
        _gear_type = build_type in ('SPACE', 'SPACE_MIXED')

        # Recognition stats counters (autodetect vs WARP CORE fallback)
        _stat_auto_n    = 0   # items recognized by ML pipeline
        _stat_core_n    = 0   # items recognized via WARP CORE session examples
        _stat_auto_conf = 0.0
        _stat_core_conf = 0.0
        _stat_skip_conf = 0   # skipped due to low confidence
        _stat_skip_type = 0   # skipped due to wrong type for slot
        _stat_per_slot: dict[str, dict] = {}  # per-slot hit/skip counters

        for slot_def in slot_defs_to_process:
            slot_name = slot_def['name']
            # The merged layout is the authoritative truth (confirmed + freshly
            # detected after IoU dedup). For BOFF seat keys especially, the
            # detector emits the full 4-ability grid — capping by confirmed-only
            # count would drop the re-proposed positions when the user deletes
            # a previously-confirmed annotation.
            if slot_name in layout:
                max_count = len(layout[slot_name])
            elif confirmed_layout and slot_name in confirmed_layout:
                max_count = len(confirmed_layout[slot_name])
            else:
                max_count = profile.get(slot_name, 0)
            if max_count == 0:
                continue

            bboxes = layout.get(slot_name, [])[:max_count]
            if not bboxes:
                _slog.info(f'  [{slot_name}] no bboxes from layout (max_count={max_count})')
            candidates = slot_candidates.get(slot_name)  # None = no type constraint
            for idx, bbox in enumerate(bboxes):
                # Emit per-slot progress so the UI stays responsive
                if self._progress_callback and total_bboxes > 0:
                    pct = _base_pct + int(processed_bboxes / total_bboxes * (_end_pct - _base_pct))
                    self._progress_callback(pct, f'{slot_name} {idx + 1}/{len(bboxes)}')
                processed_bboxes += 1

                # 5-element bboxes carry a cell state from layout detection
                # (empty/inactive positions added by _fill_boff_gaps)
                if len(bbox) == 5:
                    bx, by, bw, bh, cell_state = bbox
                    bbox4 = (bx, by, bw, bh)
                    if cell_state in ('empty', 'inactive'):
                        vname = '__empty__' if cell_state == 'empty' else '__inactive__'
                        result.items.append(RecognisedItem(
                            slot        = slot_name,
                            slot_index  = idx,
                            name        = vname,
                            confidence  = 1.0,
                            thumbnail   = None,
                            source_file = source,
                            bbox        = bbox4,
                        ))
                        continue
                    bbox = bbox4

                # Apply current dynamic Y-offset (P5)
                bx, by, bw, bh = bbox
                crop = self._crop(img, (bx, by + current_dy, bw, bh))
                
                if crop is None or crop.size == 0:
                    _slog.info(f'  [{slot_name}][{idx}] bbox={bbox} — empty crop, skipped')
                    continue
                    
                candidates = slot_candidates.get(slot_name)  # None = no type constraint
                
                # Dynamic candidate filtering for BOFF seats based on color heuristic
                if slot_name.startswith('Boff Seat'):
                    base_prof_key = self._get_layout()._classify_boff_profession(crop)
                    if base_prof_key:
                        prof_map = {
                            'tactical': 'Tactical', 'engineering': 'Engineering', 'science': 'Science',
                            'intelligence': 'Intelligence', 'command': 'Command', 'pilot': 'Pilot',
                            'miracle worker': 'Miracle Worker', 'temporal': 'Temporal Operative' # In STO it's Temporal Operative
                        }
                        base_prof = prof_map.get(base_prof_key)
                        if base_prof:
                            allowed_profs = {base_prof, 'Intelligence', 'Command', 'Pilot', 'Miracle Worker', 'Temporal Operative', 'Temporal'}
                            try:
                                boff_cache = self._app.cache.boff_abilities.get('all', {})
                                if boff_cache:
                                    candidates = {c_name for c_name, info in boff_cache.items() if info.get('profession') in allowed_profs}
                                    _slog.debug(f"  [{slot_name}][{idx}] Restricted candidates to {base_prof} + Specializations ({len(candidates)} items)")
                            except Exception:
                                pass

                name, conf, thumb, used_session = matcher.match(crop, candidate_names=candidates)
                
                # ── P5: Icon-to-Layout Feedback Loop ──────────────────────────
                # If we haven't anchored yet on this image, check if this is a good anchor
                if (not confirmed_layout and _gear_type and 
                    slot_name in ANCHOR_SLOTS and not found_anchor):
                    
                    if conf < RECALIBRATION_MIN_CONF:
                        # Initial match poor? Scan vertically for a better anchor!
                        dy_off, dy_conf, dy_name = self._find_anchor_recalibration(
                            img, slot_name, bbox, candidates)
                        if dy_conf > RECALIBRATION_MIN_CONF:
                            current_dy = dy_off
                            found_anchor = True
                            name, conf, thumb, used_session = dy_name, dy_conf, None, False
                            _slog.info(f"  [P5] Recalibrated layout Y-offset: {current_dy:+}px "
                                       f"(via {slot_name!r} conf={conf:.2f})")
                    elif conf > 0.92:
                        # Already a solid match at current_dy=0, lock it as anchor!
                        found_anchor = True
                
                _tag = '[P5 Anchored]' if found_anchor and current_dy != 0 else ('[WARP CORE]' if used_session else '[Autodetect]')
                _slog.info(f'  {_tag} [{slot_name}][{idx}] dy={current_dy:+} bbox={bbox} crop={crop.shape[1]}x{crop.shape[0]} → {name!r} conf={conf:.2f}')
                
                if not name:
                    continue
                # Reject low-confidence results — below threshold is a guess
                if conf < MIN_ACCEPT_CONF:
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — conf {conf:.2f} < {MIN_ACCEPT_CONF}')
                    _stat_skip_conf += 1
                    _stat_per_slot.setdefault(slot_name, {'ok': 0, 'skip': 0})['skip'] += 1
                    continue
                # Validate item type matches slot category
                if not self._item_valid_for_slot(name, slot_name):
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — {name!r} wrong type for slot')
                    _stat_skip_type += 1
                    _stat_per_slot.setdefault(slot_name, {'ok': 0, 'skip': 0})['skip'] += 1
                    continue
                # Experimental slot: only Experimental Weapon items allowed
                if slot_def['exp'] and not self._is_experimental(name):
                    _slog.info(f'  [{slot_name}][{idx}] SKIP — not experimental weapon: {name!r}')
                    continue
                final_slot_name = slot_name

                # Track recognition stats
                _stat_per_slot.setdefault(final_slot_name, {'ok': 0, 'skip': 0})['ok'] += 1
                if used_session:
                    _stat_core_n    += 1
                    _stat_core_conf += conf
                else:
                    _stat_auto_n    += 1
                    _stat_auto_conf += conf
                result.items.append(RecognisedItem(
                    slot        = final_slot_name,
                    slot_index  = idx,
                    name        = name,
                    confidence  = conf,
                    thumbnail   = thumb,
                    source_file = source,
                    bbox        = bbox,
                ))
                # Contribute to community knowledge (non-blocking, only high-conf, skip virtual)
                if conf >= TEMPLATE_CONF_THRESHOLD and name not in ('__empty__', '__inactive__'):
                    sync = self._get_sync_client()
                    if sync is not None:
                        sync.contribute(crop, name, confirmed=False)

        # Per-ability profession remap for BOFF seats (post-pass).
        # Non-virtuals → slot of their own ability's profession.
        # Virtuals (__empty__/__inactive__) → seat's profession (typed seats)
        # or voted dominant from sibling abilities (Universal seats).
        self._remap_boff_seat_slots(result, _stat_per_slot)

        self._log_recognition_stats(
            build_type  = build_type,
            auto_n      = _stat_auto_n,
            auto_conf   = _stat_auto_conf,
            core_n      = _stat_core_n,
            core_conf   = _stat_core_conf,
            skip_conf   = _stat_skip_conf,
            skip_type   = _stat_skip_type,
            slots_found = len(layout),
            bboxes_found= sum(len(v) for v in layout.values()),
            per_slot    = _stat_per_slot,
        )
        _slog.info(f'####### WARP: {Path(source).name} done #######')
        return result

    def _lookup_boff_profession(self, ability_name: str) -> str | None:
        """Find the profession of a BOFF ability by scanning the rank-based
        cache. Returns the SETS category name (e.g. 'Tactical', 'Command')
        or None if unknown.

        Cache shape: `boff_abilities[env][category][rank_idx][ability_name]
        = description`. There is no flat name→profession lookup, so we
        scan all (env, category, rank) buckets until a hit.
        """
        if not ability_name:
            return None
        try:
            cache = self._app.cache.boff_abilities
        except Exception:
            return None
        for env in ('space', 'ground'):
            env_dict = cache.get(env) or {}
            for category, ranks in env_dict.items():
                if not isinstance(ranks, (list, tuple)):
                    continue
                for rank_dict in ranks:
                    if isinstance(rank_dict, dict) and ability_name in rank_dict:
                        return category
        return None

    def _remap_boff_seat_slots(self, result, per_slot_stats: dict) -> None:
        """Remap items currently keyed by raw BOFF seat keys (e.g.
        `Boff Seat L[U+O]_616`) to canonical profession-named slots.

        - Non-virtual abilities → the slot of their own ability's profession.
        - Virtual items (`__empty__` / `__inactive__`):
          * typed seats (T/E/S) → seat's base profession
          * Universal seats → voted dominant profession from sibling
            non-virtuals (weight 1 + conf); fall back to spec stripe if any,
            otherwise `Boff Universal`.

        Also rebuilds `per_slot_stats` 'ok' counts so the recognition
        report shows the final, post-remap slot distribution. 'skip'
        counts are preserved as-is from the loop-time keys.
        """
        from warp.recognition.boff_keys import (
            parse_seat_profession, parse_seat_spec, is_seat_keyed,
        )
        prof_to_slot = {
            'Tactical':           'Boff Tactical',
            'Engineering':        'Boff Engineering',
            'Science':            'Boff Science',
            'Intelligence':       'Boff Intelligence',
            'Command':            'Boff Command',
            'Pilot':              'Boff Pilot',
            'Miracle Worker':     'Boff Miracle Worker',
            'Temporal Operative': 'Boff Temporal',
            'Temporal':           'Boff Temporal',
        }

        # Group seat-keyed items
        by_seat: dict[str, list] = {}
        for it in result.items:
            if is_seat_keyed(it.slot):
                by_seat.setdefault(it.slot, []).append(it)
        if not by_seat:
            return

        for seat_key, items in by_seat.items():
            seat_prof = parse_seat_profession(seat_key)  # None for U
            seat_spec = parse_seat_spec(seat_key)

            # Cache per-ability profession lookups for this seat
            own_prof: dict[str, str | None] = {}
            for it in items:
                if it.name in ('__empty__', '__inactive__'):
                    continue
                if it.name not in own_prof:
                    own_prof[it.name] = self._lookup_boff_profession(it.name)

            # Vote dominant profession from non-virtuals (count + conf)
            votes: dict[str, float] = {}
            for it in items:
                if it.name in ('__empty__', '__inactive__'):
                    continue
                prof = own_prof.get(it.name)
                if prof:
                    votes[prof] = votes.get(prof, 0.0) + 1.0 + float(it.confidence or 0.0)
            voted_prof = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None

            for it in items:
                if it.name in ('__empty__', '__inactive__'):
                    target = seat_prof or voted_prof or seat_spec
                    new_slot = prof_to_slot.get(target, 'Boff Universal') if target else 'Boff Universal'
                else:
                    p = own_prof.get(it.name)
                    new_slot = prof_to_slot.get(p, it.slot) if p else it.slot
                if new_slot != it.slot:
                    _slog.debug(f'  BOFF remap: [{it.slot}] {it.name!r} → [{new_slot}]')
                    it.seat_key = it.slot   # preserve original detector key
                    it.slot     = new_slot

        # Rebuild per-slot 'ok' counts from final result.items; preserve
        # 'skip' counts from loop-time keys so type/conf rejections aren't lost.
        rebuilt: dict[str, dict] = {}
        for it in result.items:
            rebuilt.setdefault(it.slot, {'ok': 0, 'skip': 0})['ok'] += 1
        for k, v in per_slot_stats.items():
            rebuilt.setdefault(k, {'ok': 0, 'skip': 0})['skip'] += v.get('skip', 0)
        per_slot_stats.clear()
        per_slot_stats.update(rebuilt)

    def _log_recognition_stats(
        self,
        build_type: str,
        auto_n: int,
        auto_conf: float,
        core_n: int,
        core_conf: float,
        skip_conf: int = 0,
        skip_type: int = 0,
        slots_found: int = 0,
        bboxes_found: int = 0,
        per_slot: dict | None = None,
    ) -> None:
        """Log per-session recognition stats with per-slot breakdown and trend analysis."""
        import datetime, json as _json

        total = auto_n + core_n
        attempted = total + skip_conf + skip_type

        auto_pct      = 100.0 * auto_n / total if total else 0.0
        avg_auto_conf = auto_conf / auto_n if auto_n else 0.0
        avg_core_conf = core_conf / core_n if core_n else 0.0
        hit_rate      = 100.0 * total / attempted if attempted else 0.0

        # ── Summary table ─────────────────────────────────────────────────
        _slog.info(f'┌── Recognition Report [{build_type}] ──────────────────────')
        _slog.info(f'│ Layout:    {slots_found} slot groups, {bboxes_found} bboxes')
        _slog.info(f'│ Matched:   {total}/{attempted}  hit rate {hit_rate:.0f}%')
        if total:
            _slog.info(f'│   Autodetect: {auto_n} ({auto_pct:.0f}%)  avg conf {avg_auto_conf:.2f}')
        if core_n:
            _slog.info(f'│   WARP CORE:  {core_n} ({100-auto_pct:.0f}%)  avg conf {avg_core_conf:.2f}')
        if skip_conf:
            _slog.info(f'│ Skipped (low conf): {skip_conf}')
        if skip_type:
            _slog.info(f'│ Skipped (wrong type): {skip_type}')

        # Per-slot breakdown
        if per_slot:
            _slog.info(f'│ Per-slot:')
            for slot_name in sorted(per_slot.keys()):
                s = per_slot[slot_name]
                ok, skip = s['ok'], s['skip']
                bar = '█' * ok + '░' * skip
                _slog.info(f'│   {slot_name:30s}  {ok:2d}/{ok+skip:2d}  {bar}')

        # ── Persist + trend ───────────────────────────────────────────────
        stats_path = Path(__file__).resolve().parent.parent / '.config' / 'recognition_stats.json'
        try:
            history: list[dict] = _json.loads(stats_path.read_text(encoding='utf-8'))
        except Exception:
            history = []

        entry = {
            'ts':           datetime.datetime.now().isoformat(timespec='seconds'),
            'build_type':   build_type,
            'total':        total,
            'attempted':    attempted,
            'auto_n':       auto_n,
            'core_n':       core_n,
            'skip_conf':    skip_conf,
            'skip_type':    skip_type,
            'hit_rate':     round(hit_rate, 1),
            'auto_pct':     round(auto_pct, 1),
            'avg_auto_conf': round(avg_auto_conf, 3),
            'avg_core_conf': round(avg_core_conf, 3),
            'slots_found':  slots_found,
            'bboxes_found': bboxes_found,
        }
        history.append(entry)
        history = history[-100:]

        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(_json.dumps(history, indent=2), encoding='utf-8')
        except Exception as e:
            _slog.debug(f'WarpImporter: could not save recognition stats: {e}')

        # Rolling average over previous sessions (same build_type)
        prev = [h for h in history[:-1] if h.get('build_type') == build_type]
        if prev:
            avg_hit_hist = sum(h.get('hit_rate', 0) for h in prev) / len(prev)
            avg_conf_hist = sum(h.get('avg_auto_conf', 0) for h in prev) / len(prev)
            delta_hit  = hit_rate - avg_hit_hist
            delta_conf = avg_auto_conf - avg_conf_hist
            trend_hit  = '↑' if delta_hit  > 2.0 else ('↓' if delta_hit  < -2.0 else '→')
            trend_conf = '↑' if delta_conf > 0.02 else ('↓' if delta_conf < -0.02 else '→')
            _slog.info(
                f'│ Trend (vs {len(prev)} prev):  '
                f'hit {avg_hit_hist:.0f}%{trend_hit}  conf {avg_conf_hist:.2f}{trend_conf}'
            )
        else:
            _slog.info(f'│ Trend: first session for {build_type}')
        _slog.info(f'└─────────────────────────────────────────────────────')


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
            for a in ann_list:
                if a.get('state') != 'confirmed': continue
                slot = a.get('slot', '')
                if slot and slot not in _NON_PROFILE:
                    counts[slot] = counts.get(slot, 0) + 1
            if counts:
                _slog.info(f'WarpImporter: confirmed profile from disk for {fname}: {counts}')
            return counts
        except Exception as e:
            _slog.debug(f'WarpImporter: _load_confirmed_profile error: {e}')
            return {}

    def _build_slot_candidates(self, slot_defs: list) -> dict[str, set[str]]:
        """
        For each equipment slot, build the set of valid item names from the SETS
        cache using the slot's build key (e.g. 'deflector', 'core', 'shield').

        This prevents cross-category false positives — a trait icon matching the
        Deflector slot, or a shield matching the Warp Core slot.

        Slots without an equipment cache entry (traits, boffs) get no entry here,
        so candidate_names=None is passed to match() → full index searched as before.

        Console slots include universal consoles since they are accepted everywhere.
        Boff slots are restricted to boff abilities to prevent equipment from matching.
        """
        result: dict[str, set[str]] = {}
        try:
            eq_cache = self._app.cache.equipment
        except Exception:
            return result

        # Universal consoles can go in any console slot
        uni_names: set[str] = set(eq_cache.get('uni_consoles', {}).keys())

        for sd in slot_defs:
            slot_name = sd['name']
            build_key = sd.get('key', '')
            if not build_key or build_key not in eq_cache:
                continue
            names: set[str] = set(eq_cache[build_key].keys())
            # Universal consoles are accepted in any dedicated console slot
            if 'console' in build_key:
                names |= uni_names
            if names:
                result[slot_name] = names

        # Boff slots: restrict to boff abilities only.
        # Without this, candidate_names=None → full index search → equipment items
        # (Deflectors, Consoles, etc.) can match ability slots at conf=1.00 via
        # session examples that were accidentally confirmed in the wrong slot.
        try:
            boff_names = set(self._app.cache.boff_abilities.get('all', {}).keys())
        except Exception:
            boff_names = set()
        if boff_names:
            for sd in slot_defs:
                slot_name = sd['name']
                if slot_name.startswith('Boff ') and slot_name not in result:
                    result[slot_name] = boff_names

        # Trait slots: restrict to the matching trait category.
        # Without this, candidate_names=None → full index search lets a ground
        # trait land in a space trait slot, equipment icons match trait slots,
        # or the same name repeats across every slot of a panel.
        try:
            traits_cache = self._app.cache.traits
            starship_traits_cache = getattr(
                self._app.cache, 'starship_traits', {}) or {}
        except Exception:
            traits_cache = {}
            starship_traits_cache = {}

        def _trait_names(env: str, cat: str) -> set[str]:
            try:
                return set(traits_cache.get(env, {}).get(cat, {}).keys())
            except Exception:
                return set()

        trait_slot_pools: dict[str, set[str]] = {
            'Personal Space Traits':  _trait_names('space',  'personal'),
            'Personal Ground Traits': _trait_names('ground', 'personal'),
            'Space Reputation':       _trait_names('space',  'rep'),
            'Ground Reputation':      _trait_names('ground', 'rep'),
            'Active Space Rep':       _trait_names('space',  'active_rep'),
            'Active Ground Rep':      _trait_names('ground', 'active_rep'),
            'Starship Traits':        set(starship_traits_cache.keys()),
        }
        for sd in slot_defs:
            slot_name = sd['name']
            if slot_name in result:
                continue
            pool = trait_slot_pools.get(slot_name)
            if pool:
                result[slot_name] = pool

        # Add virtual items so ML and session examples can match empty/inactive slots
        for names_set in result.values():
            names_set.update(VIRTUAL_ITEM_NAMES)

        return result

    def _item_valid_for_slot(self, item_name: str, slot_name: str) -> bool:
        """Check that the item belongs in the slot. Routes to the right
        sub-cache by slot family: equipment (cache.equipment) → trait
        (cache.traits / cache.starship_traits) → BOFF (cache.boff_abilities).
        Returns True permissively when no constraint applies or when the
        item is not in any cache (likely a new community item)."""
        # Virtual placeholders pass through — they don't represent items.
        if item_name in VIRTUAL_ITEM_NAMES:
            return True

        # ── Trait slots ──
        if slot_name == 'Starship Traits':
            try:
                if item_name in (self._app.cache.starship_traits or {}):
                    return True
            except Exception:
                return True
            _slog.info(f'  _item_valid_for_slot: {item_name!r} not a Starship Trait')
            return False
        cat_tuple = TRAIT_SLOT_CATEGORY.get(slot_name)
        if cat_tuple:
            env, cat = cat_tuple
            try:
                pool = (self._app.cache.traits or {}).get(env, {}).get(cat, {})
                if item_name in pool:
                    return True
            except Exception:
                return True
            _slog.info(f'  _item_valid_for_slot: {item_name!r} not in traits[{env}][{cat}]')
            return False

        # ── BOFF seat slots ──
        # Marker-keyed seats encode profession in the key. Universal seats
        # accept any profession (player decides); typed seats (T/E/S) and
        # legacy 'Boff <Profession>' keys must match the ability's profession.
        if slot_name.startswith('Boff'):
            from warp.recognition.boff_keys import parse_seat_profession, is_seat_keyed
            seat_prof = parse_seat_profession(slot_name)
            if not seat_prof and (is_seat_keyed(slot_name) or slot_name == 'Boff Universal'):
                return True  # Universal — any profession allowed
            if not seat_prof:
                # Legacy 'Boff Tactical' / 'Boff Engineering' / etc.
                seat_prof = slot_name.replace('Boff ', '').strip() or None
            if not seat_prof:
                return True
            try:
                for env in ('space', 'ground'):
                    rank_lists = (self._app.cache.boff_abilities
                                  .get(env, {}).get(seat_prof, []))
                    for rank_dict in rank_lists:
                        if isinstance(rank_dict, dict) and item_name in rank_dict:
                            return True
            except Exception:
                return True
            _slog.info(f'  _item_valid_for_slot: {item_name!r} not a {seat_prof} ability '
                       f'(slot {slot_name!r})')
            return False

        # ── Equipment slots ──
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
                _slog.info(f'  _item_valid_for_slot: {item_name!r} type={item_type!r} '
                           f'not valid for {slot_name!r}')
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
            # Seed session examples with confirmed crops ONLY for WARP CORE (trainer) path.
            # WARP must not read annotations.json — that would hide detection bugs behind
            # user-confirmed ground truth. Mirrors the _use_confirmed gate at line ~707.
            if self._from_trainer:
                here = Path(__file__).resolve().parent
                for _ in range(6):
                    td = here / 'warp' / 'training_data'
                    if td.exists():
                        break
                    here = here.parent
                else:
                    td = None
                if td is not None:
                    SETSIconMatcher.seed_from_training_data(td)
            else:
                # WARP path: clear any session examples a prior trainer run left in the
                # class-level state, so WARP sees pristine detection quality.
                SETSIconMatcher.reset_ml_session()
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
            corrections_path = Path(__file__).resolve().parent / 'models' / 'ship_type_corrections.json'
            if corrections_path.exists():
                TextExtractor.load_corrections(corrections_path)
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

    def _find_anchor_recalibration(
        self,
        img: np.ndarray,
        slot_name: str,
        bbox: tuple[int, int, int, int],
        candidates: set[str] | None
    ) -> tuple[int, float, str]:
        """
        P5 Helper: Scan vertically around the predicted bbox to find the best 
        structural anchor match. Returns (dy, confidence, item_name).
        """
        best_dy = 0
        best_conf = 0.0
        best_name = ''
        bx, by, bw, bh = bbox
        matcher = self._get_matcher()
        h, w = img.shape[:2]

        # Scan +/- 40px in 4px steps
        # This covers most UI shifts/scales in STO logs
        for dy in range(-40, 41, 4):
            # Safe crop region
            y1 = max(0, by + dy)
            y2 = min(h, y1 + bh)
            if y2 <= y1:
                continue
            crop = img[y1:y2, bx:bx+bw]
            name, conf, _, _ = matcher.match(crop, candidate_names=candidates)
            if conf > best_conf:
                best_conf = conf
                best_dy = dy
                best_name = name
                if conf > 0.96: # Early exit for near-perfect match
                    break
        
        return best_dy, best_conf, best_name
