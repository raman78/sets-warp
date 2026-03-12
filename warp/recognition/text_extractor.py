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

RE_TIER = re.compile(r'\[?(T[1-6](?:-(?:U|X|X2))?)(?:\]|$)', re.IGNORECASE)

# ROI for ship name/type block (top-left, fraction of image)
SHIP_INFO_ROI = (0.0, 0.0, 0.34, 0.28)

# ── Keyword sets per screen type ──────────────────────────────────────────────
# Checked against lowercase OCR text of scan regions.
# More specific keywords are listed first (longer matches win).

_TRAIT_KEYWORDS: dict[str, str] = {
    'personal space traits':  'SPACE_TRAITS',
    'starship traits':        'SPACE_TRAITS',
    'space reputation':       'SPACE_TRAITS',
    'active space rep':       'SPACE_TRAITS',
    'active reputation':      'SPACE_TRAITS',
    'personal ground traits': 'GROUND_TRAITS',
    'ground reputation':      'GROUND_TRAITS',
    'active ground rep':      'GROUND_TRAITS',
}

_BOFF_KEYWORDS: dict[str, str] = {
    'bridge officer abilities': 'BOFFS',
    'bridge officer':           'BOFFS',
    'boff abilities':           'BOFFS',
    'tactical ability':         'BOFFS',
    'engineering ability':      'BOFFS',
    'science ability':          'BOFFS',
}

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

    Check order: traits -> boffs -> spec headers -> spec names.
    More specific checks run first.
    """
    joined = ' '.join(lines_lower)

    # 1. Trait screen
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

            # ── Pass 1: quick scan of characteristic regions ──────────────────
            # Traits / Boffs / Spec headers appear in top-right or bottom areas.
            scan_regions = [
                img[0:int(h * 0.55), int(w * 0.45):],   # top-right panel
                img[int(h * 0.65):, :],                  # bottom strip
            ]
            all_lines: list[str] = []
            for region in scan_regions:
                if region.size == 0:
                    continue
                try:
                    out = self._get_ocr().readtext(region)
                    all_lines += [t.lower()
                                  for (_, t, c) in out if c > 0.25]
                except Exception:
                    pass

            detected = _detect_type_from_text(all_lines)
            if detected:
                result['build_type'] = detected
                # Still try to extract ship name from top-left
                self._try_ship_name(img, w, h, result)
                return result

            # ── Pass 2: standard Status tab — ship info in top-left ROI ───────
            x1 = int(SHIP_INFO_ROI[0] * w)
            y1 = int(SHIP_INFO_ROI[1] * h)
            x2 = int(SHIP_INFO_ROI[2] * w)
            y2 = int(SHIP_INFO_ROI[3] * h)
            roi     = img[y1:y2, x1:x2]
            ocr_out = self._get_ocr().readtext(roi)

        except Exception as e:
            log.debug(f'TextExtractor OCR failed: {e}')
            return result

        # Sort top-to-bottom and parse ship name / type / tier
        ocr_out.sort(key=lambda r: r[0][0][1])
        lines = [t.strip() for (_, t, c) in ocr_out if c > 0.25 and t.strip()]
        if not lines:
            return result

        result['ship_name'] = lines[0]

        for line in lines[1:]:
            m = RE_TIER.search(line)
            if m:
                result['ship_tier'] = m.group(1).upper().replace(' ', '')
                type_text = line[:m.start()].strip().rstrip(' [')
                if type_text:
                    result['ship_type'] = type_text
            elif not result['ship_type'] and len(line) > 4:
                result['ship_type'] = (result['ship_type'] + ' ' + line).strip()

        # Infer SPACE vs GROUND from ship type text
        ship_type_lower = result['ship_type'].lower()
        if any(kw in ship_type_lower for kw in ('ground', 'combat armor', 'kit')):
            result['build_type'] = 'GROUND'
        else:
            result['build_type'] = 'SPACE'

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
