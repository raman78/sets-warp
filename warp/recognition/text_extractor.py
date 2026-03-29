# warp/recognition/text_extractor.py
#
# Extracts ship name, type, tier and build/screen type from STO screenshots.
#
# Detected screen types:
#   SPACE         — Status tab: ship name + equipment slots
#   GROUND        — Ground equipment tab
#   SPACE_TRAITS  — Traits tab: Personal Space Traits / Starship Traits / Reputation
#   GROUND_TRAITS — Traits tab: Personal Ground Traits / Ground Reputation
#   BOFFS         — Bridge Officer abilities panel
#   SPEC          — Specialization trees (Primary / Secondary)
#   (empty)       — Unknown / not detected
#
# Specializations (Primary — 30 abilities):
#   Command Officer, Intelligence Officer, Miracle Worker,
#   Pilot, Temporal Operative
# Specializations (Secondary only — 15 abilities):
#   Constable, Commando, Strategist
# (all released between Delta Rising and Season 14)

from __future__ import annotations

import re
import logging
import numpy as np

log = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

RE_TIER      = re.compile(r'\[?(T[1-6](?:-(?:U|X|X2))?)(?:\]|$)', re.IGNORECASE)
RE_TIER_LOOSE = re.compile(r'\b(T[1-6](?:-(?:U|X|X2))?)\b', re.IGNORECASE)

# ROI for ship name/type block (top-left, fraction of image)
SHIP_INFO_ROI = (0.0, 0.0, 0.34, 0.28)

# ── Keyword sets per screen type ──────────────────────────────────────────────
# Checked against lowercase OCR text of scan regions.
# More specific keywords are listed first (longer matches win).

# ── Screen type keyword sets ─────────────────────────────────────────────────
# Each dict maps lowercase OCR substring → screen type.
# More specific (longer) strings should be listed first.

# Traits screen — labels that only appear on traits/reputation tabs
_TRAIT_KEYWORDS: dict[str, str] = {
    'personal space traits':      'SPACE_TRAITS',
    'starship traits':             'SPACE_TRAITS',
    'space reputation':            'SPACE_TRAITS',
    'active space rep':            'SPACE_TRAITS',
    'active reputation':           'SPACE_TRAITS',
    'personal ground traits':      'GROUND_TRAITS',
    'ground reputation':           'GROUND_TRAITS',
    'active ground rep':           'GROUND_TRAITS',
    'active ground reputation':    'GROUND_TRAITS',
}

# Bridge Officer screen
_BOFF_KEYWORDS: dict[str, str] = {
    'bridge officer abilities': 'BOFFS',
    'bridge officer':           'BOFFS',
    'boff abilities':           'BOFFS',
    'tactical ability':         'BOFFS',
    'engineering ability':      'BOFFS',
    'science ability':          'BOFFS',
}

# Space equipment slot labels — presence of 2+ confirms SPACE_EQ
_SPACE_EQ_LABELS: list[str] = [
    'fore weapons', 'aft weapons', 'experimental weapon',
    'deflector', 'secondary deflector',
    'impulse', 'warp core', 'singularity core',
    'shields', 'shield array',
    'engineering consoles', 'science consoles', 'tactical consoles',
    'universal consoles', 'hangar',
    'devices',
]

# Ground equipment slot labels — presence of 2+ confirms GROUND_EQ
_GROUND_EQ_LABELS: list[str] = [
    'kit modules', 'kit module',
    'body armor', 'combat armor',
    'ev suit', 'environmental suit',
    'personal shield',
    'ground weapon', 'secondary weapon',
    'ground device',
]

# Minimum label hits to confirm equipment screen type
_EQ_MIN_HITS = 2

# All STO specialization names (Primary and Secondary).
# Each appears as a section header on the Specializations screen.
_SPEC_NAMES: list[str] = [
    # Primary specializations
    'command officer',
    'intelligence officer',
    'miracle worker',
    'pilot',
    'temporal operative',
    # Secondary specializations
    'constable',
    'commando',
    'strategist',
]

# UI headers that confirm we are on the Specializations screen
_SPEC_HEADER_KEYWORDS: list[str] = [
    'primary specialization',
    'secondary specialization',
    'specialization points',
]


def _detect_type_from_text(lines_lower: list[str]) -> str:
    """
    Return detected screen type from OCR lines, or empty string if unknown.

    Priority order (most specific first):
      traits → boffs → spec → ground_eq → space_eq
    Equipment screens need 2+ matching labels to avoid false positives
    (single words like 'shields' or 'devices' can appear anywhere).
    """
    joined = ' '.join(lines_lower)

    # 1. Trait screen — very specific multi-word phrases
    for kw, btype in _TRAIT_KEYWORDS.items():
        if kw in joined:
            return btype

    # 2. Boff screen
    for kw, btype in _BOFF_KEYWORDS.items():
        if kw in joined:
            return btype

    # 3. Specialization screen — header keywords
    for kw in _SPEC_HEADER_KEYWORDS:
        if kw in joined:
            return 'SPEC'

    # 4. Specialization screen — at least one spec name present
    spec_hits = sum(1 for name in _SPEC_NAMES if name in joined)
    if spec_hits >= 1:
        return 'SPEC'

    # 5. Ground equipment — 2+ ground slot labels
    ground_hits = sum(1 for lbl in _GROUND_EQ_LABELS if lbl in joined)
    if ground_hits >= _EQ_MIN_HITS:
        return 'GROUND'

    # 6. Space equipment — 2+ space slot labels
    space_hits = sum(1 for lbl in _SPACE_EQ_LABELS if lbl in joined)
    if space_hits >= _EQ_MIN_HITS:
        return 'SPACE'

    # 7. Single strong space indicator
    strong_space = ['fore weapons', 'aft weapons', 'warp core',
                    'singularity core', 'experimental weapon']
    if any(kw in joined for kw in strong_space):
        return 'SPACE'

    # 8. Single strong ground indicator
    strong_ground = ['kit modules', 'body armor', 'ev suit', 'personal shield']
    if any(kw in joined for kw in strong_ground):
        return 'GROUND'

    return ''


class TextExtractor:
    """
    Extracts structured info from an STO screenshot.

    Returns dict:
        ship_name  : str   — e.g. "U.S.S. Genius"
        ship_type  : str   — e.g. "Typhoon Temporal Battlecruiser"
        ship_tier  : str   — e.g. "T6-X2"
        build_type : str   — SPACE | GROUND | SPACE_TRAITS | GROUND_TRAITS
                             | BOFFS | SPEC | '' (unknown)
    """

    def __init__(self):
        self._ocr = None

    def extract_ship_info(self, img: np.ndarray) -> dict:
        result = {
            'ship_name':  '',
            'ship_type':  '',
            'ship_tier':  '',
            'build_type': '',
        }
        try:
            h, w = img.shape[:2]

            # ── Pass 1: screen type detection ────────────────────────────────
            # Two-stage: fast partial scan first, full scan if needed.
            # Partial covers most cases; full scan handles MIXED layouts
            # where labels can appear anywhere on screen.
            def _ocr_region(region):
                if region.size == 0: return []
                try:
                    out = self._get_ocr().readtext(region)
                    return [t.lower() for (_, t, c) in out if c > 0.25]
                except Exception:
                    return []

            # Stage 1a: fast partial scan
            partial_lines = (
                _ocr_region(img[0:int(h * 0.55), int(w * 0.45):]) +  # top-right
                _ocr_region(img[int(h * 0.65):, :])                   # bottom
            )
            detected = _detect_type_from_text(partial_lines)
            _slog.info(f'TextExtractor: partial scan → {detected!r} '
                       f'({len(partial_lines)} tokens)')

            if not detected:
                # Stage 1b: full image scan — needed for MIXED screens
                _slog.info('TextExtractor: partial scan inconclusive — scanning full image')
                full_lines = _ocr_region(img)
                detected = _detect_type_from_text(full_lines)
                all_lines = full_lines
                _slog.info(f'TextExtractor: full scan → {detected!r} '
                           f'({len(full_lines)} tokens)')
            else:
                all_lines = partial_lines

            if detected:
                result['build_type'] = detected

            # ── Pass 2: wide-scan for ship info ───────────────────────────────
            # Scan entire top 20% of image — works regardless of where ship info
            # appears (left, centre, right, cropped screenshots)
            top_band = img[0:int(h * 0.20), :]
            try:
                ocr_out = self._get_ocr().readtext(top_band)
            except Exception as e:
                _slog.debug(f'TextExtractor OCR failed: {e}')
                return result

            if not ocr_out:
                return result

            # Sort all detections left-to-right, top-to-bottom
            ocr_out.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            items = [(bbox, t.strip(), c) for (bbox, t, c) in ocr_out
                     if c > 0.25 and t.strip()]
            _slog.info(f'TextExtractor: {len(items)} OCR tokens in top band')
            for bbox, t, c in items:
                _slog.debug(f'  OCR: {t!r} conf={c:.2f} y={bbox[0][1]:.0f}')

            # ── Find Tier token first — it's the most distinctive ─────────────
            # Tier looks like: T6-X2, T6-X, T6, T5-U etc.
            tier_idx = None
            for i, (bbox, t, c) in enumerate(items):
                m = RE_TIER_LOOSE.search(t)
                if m:
                    result['ship_tier'] = m.group(1).upper().replace(' ', '')
                    tier_idx = i
                    _slog.info(f'TextExtractor: tier={result["ship_tier"]!r} from {t!r}')
                    # Check if ship type is on the same line before tier
                    prefix = t[:m.start()].strip().rstrip(' [')
                    if len(prefix) > 4:
                        result['ship_type'] = prefix
                    break

            # ── Find ship type and name near tier ─────────────────────────────
            # Tier is the anchor. Ship type is 1-2 lines above it (same x cluster).
            # Ship name (U.S.S. ...) is typically the topmost token in the band.
            if tier_idx is not None:
                tier_y   = items[tier_idx][0][0][1]
                tier_x0  = items[tier_idx][0][0][0]  # left edge of tier token

                # Same row as tier: within 8px vertically AND left of tier token
                # Strict x-filter: must be in same horizontal cluster as tier
                same_row = [t for i, (bbox, t, c) in enumerate(items)
                            if abs(bbox[0][1] - tier_y) < 8
                            and i != tier_idx
                            and bbox[0][0] < tier_x0  # to the left of tier
                            and len(t) > 2]           # skip noise tokens

                # Lines above tier: within 40px (one or two rows max)
                above = [(bbox[0][1], t) for i, (bbox, t, c) in enumerate(items)
                         if tier_y - 40 < bbox[0][1] < tier_y - 8
                         and len(t) > 3]  # skip noise
                above.sort(key=lambda x: x[0], reverse=True)  # closest to tier first

                # Ship type = line(s) just above tier + word before tier on same row.
                # Only use above/same_row when prefix didn't already give us a type —
                # prefix is more reliable (same OCR token as tier).
                if not result['ship_type']:
                    type_parts = []
                    if above:
                        type_parts.append(above[0][1])
                    if same_row:
                        type_parts.extend(same_row)
                    if type_parts:
                        result['ship_type'] = ' '.join(type_parts).strip()

                # Ship name: look for U.S.S./I.S.S./R.R.W. pattern anywhere in band
                # or topmost token if no pattern match
                _SHIP_NAME_RE = re.compile(
                    r'^([UuIiRr]\.?[SsRr]\.?[SsWw]\.?\s+\S)', re.UNICODE)
                name_candidates = [(bbox[0][1], t)
                                   for bbox, t, c in items if c > 0.5]
                name_candidates.sort(key=lambda x: x[0])  # top to bottom
                for _, t in name_candidates:
                    if _SHIP_NAME_RE.match(t):
                        result['ship_name'] = t
                        break
                # Fallback: topmost high-conf token that isn't the type
                if not result['ship_name'] and name_candidates:
                    top = name_candidates[0][1]
                    if top != result['ship_type']:
                        result['ship_name'] = top

                _slog.info(f'TextExtractor: name={result["ship_name"]!r} '
                           f'type={result["ship_type"]!r} '
                           f'tier={result["ship_tier"]!r}')
            else:
                # No tier found — try to get ship name from first prominent text
                if items:
                    result['ship_name'] = items[0][1]
                _slog.info(f'TextExtractor: no tier found, name={result["ship_name"]!r}')

            # ── Infer build type if not already detected ──────────────────────
            if not result['build_type']:
                ship_type_lower = result['ship_type'].lower()
                if any(kw in ship_type_lower
                       for kw in ('ground', 'combat armor', 'kit')):
                    result['build_type'] = 'GROUND'
                else:
                    result['build_type'] = 'SPACE'

        except Exception as e:
            _slog.debug(f'TextExtractor: unexpected error: {e}')

        return result

    def _try_ship_name(self, img: np.ndarray, w: int, h: int, result: dict):
        """
        Attempt to extract ship name from top-left ROI
        and store in result['ship_name'] if not already set.
        """
        if result.get('ship_name'):
            return
        try:
            x2 = int(SHIP_INFO_ROI[2] * w)
            y2 = int(SHIP_INFO_ROI[3] * h)
            roi     = img[0:y2, 0:x2]
            ocr_out = self._get_ocr().readtext(roi)
            ocr_out.sort(key=lambda r: r[0][0][1])
            lines = [t.strip() for (_, t, c) in ocr_out
                     if c > 0.3 and t.strip()]
            if lines:
                result['ship_name'] = lines[0]
        except Exception:
            pass

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr
