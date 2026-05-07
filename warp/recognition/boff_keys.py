"""BOFF slot key schema — single source of truth for parsing the
slot-name strings that the layout detectors emit.

Producer: `layout_detector._detect_boffs_via_markers` (and legacy
band-scan fallbacks).
Consumers: `warp_dialog` (Phase 2 cluster→seat matching) and
`trainer_window._build_search_candidates` (autocomplete).

Key formats supported
---------------------
- `Boff Tactical`             — legacy profession-keyed
- `Boff Engineering`
- `Boff Science`
- `Boff Temporal` (etc.)      — legacy spec-prof seat
- `Boff Seat L[T]_483`        — new marker-keyed (side, prof code, marker_y)
- `Boff Seat L[T+P]_483`      — new with spec stripe
- `Boff Seat L[U]_483`        — Universal — caller does content-based fallback
- `Boff Seat L_483`           — legacy seat-keyed without prof code

`parse_seat_profession` returns the base profession (Tactical /
Engineering / Science) or None when the key is Universal / legacy
seat-keyed without code. `parse_seat_spec` returns the spec
profession (Command / Intelligence / Temporal / Pilot / Miracle
Worker) or None.
"""
from __future__ import annotations

import re

# Codes match boff_marker.SEAT_CODE_LABEL / SPEC_CODE_LABEL — duplicated
# here so this module stays free of CV dependencies (kept tiny so UI
# layers can import it without dragging in OpenCV / numpy).
_SEAT_CODE_TO_PROF = {
    'T': 'Tactical',
    'E': 'Engineering',
    'S': 'Science',
    # 'U' (Universal) intentionally maps to None — Universal seats have
    # no inherent profession; callers must derive it from ability content.
}
_SPEC_CODE_TO_PROF = {
    'O': 'Command',
    'P': 'Intelligence',
    'Y': 'Temporal',
    'C': 'Pilot',
    'L': 'Miracle Worker',
}

# Legacy profession-keyed names emitted by _detect_via_full_scan and
# pre-marker detectors. `Universal` is omitted because a profession-keyed
# Universal slot would itself be Unknown (caller goes content-based).
_LEGACY_PROFESSIONS = frozenset({
    'Tactical', 'Engineering', 'Science',
})
_LEGACY_SPEC_PROFESSIONS = frozenset(_SPEC_CODE_TO_PROF.values())

# Boff Seat L[T+P]_483 → groups (side, code, spec, my)
# Boff Seat L[T]_483   → groups (side, code, None, my)
_SEAT_KEY_RE = re.compile(
    r'^Boff Seat ([LR])(?:\[([TESU])(?:\+([OPYCL]))?\])?_(\d+)$'
)


def parse_seat_profession(slot_name: str) -> str | None:
    """Return the base profession for a BOFF slot key, or None if the
    key carries no profession info (Universal or legacy seat-keyed).
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff '):
        return None
    # Legacy profession-keyed: 'Boff Tactical' etc.
    rest = slot_name[5:].strip()
    if rest in _LEGACY_PROFESSIONS:
        return rest
    # Legacy spec-prof seat: 'Boff Temporal' — fall back to spec lookup.
    # Spec-prof keys carry no base profession, so still return None.
    if rest in _LEGACY_SPEC_PROFESSIONS:
        return None
    # New marker-keyed: 'Boff Seat L[T]_483' / 'Boff Seat L[T+P]_483'
    m = _SEAT_KEY_RE.match(slot_name)
    if m:
        code = m.group(2)
        if code:
            return _SEAT_CODE_TO_PROF.get(code)  # 'U' → None
    return None


def parse_seat_spec(slot_name: str) -> str | None:
    """Return the specialization profession for a BOFF slot key, or None
    if no spec is encoded.
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff '):
        return None
    # Legacy spec-prof seat: 'Boff Temporal' etc.
    rest = slot_name[5:].strip()
    if rest in _LEGACY_SPEC_PROFESSIONS:
        return rest
    # New marker-keyed: spec is the optional second bracket group.
    m = _SEAT_KEY_RE.match(slot_name)
    if m:
        spec_code = m.group(3)
        if spec_code:
            return _SPEC_CODE_TO_PROF.get(spec_code)
    return None


def is_seat_keyed(slot_name: str) -> bool:
    """True iff `slot_name` is a marker-keyed seat name (with or without
    profession code), i.e. matches `Boff Seat L_<y>` or `Boff Seat L[T]_<y>`.
    """
    return isinstance(slot_name, str) and bool(_SEAT_KEY_RE.match(slot_name))


def pretty_slot(slot_name: str) -> str:
    """Convert a dynamic BOFF seat key into a user-friendly label:
    - `Boff Seat L[E]_392`   → `Boff Engineering`
    - `Boff Seat R[T+P]_510` → `Boff Tactical+Intelligence`
    - `Boff Seat L[U]_478`   → `Boff Universal`
    - `Boff Seat L_478`      → `Boff Universal` (legacy seat-keyed without code)

    Non-seat-keyed slot names (e.g. `Boff Tactical`, `Fore Weapons`,
    `Ship Name`) are returned unchanged.
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff Seat'):
        return slot_name
    if not is_seat_keyed(slot_name):
        return slot_name
    prof = parse_seat_profession(slot_name)
    spec = parse_seat_spec(slot_name)
    base = f'Boff {prof}' if prof else 'Boff Universal'
    return f'{base}+{spec}' if spec else base
