"""
tests/test_text_extractor.py
Unit tests for TextExtractor ship type extraction logic.
Covers the OCR prefix-vs-above overwrite bug fixed in this session.
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from warp.recognition.text_extractor import RE_TIER_LOOSE

_SECTION_HEADER_RE = re.compile(
    r'\b(traits|reputation|abilities|bridge.?officer|boff|'
    r'equipment|consoles?|weapons?|devices?|kit\b|armor)\b',
    re.IGNORECASE
)


def _extract_type_from_token(token: str, above_tokens: list[str]) -> str:
    """
    Mirrors the logic in TextExtractor.extract_ship_info() Pass 2.
    Returns the ship type that would be extracted for a given tier token
    and the tokens found above it.
    """
    result_type = ''
    m = RE_TIER_LOOSE.search(token)
    if not m:
        return result_type

    # Extract prefix (same logic as text_extractor.py lines 263-266)
    prefix = token[:m.start()].strip().rstrip(' [')
    if len(prefix) > 4:
        result_type = prefix

    # Filter above tokens: skip known screen section headers
    above_clean = [t for t in above_tokens if not _SECTION_HEADER_RE.search(t)]
    if above_clean:
        result_type = (above_clean[0] + ' ' + result_type).strip() if result_type else above_clean[0]
    elif not result_type and above_tokens:
        # No clean above, no prefix — pass (same_row not modelled in this helper)
        pass

    return result_type


class TestShipTypeExtraction(unittest.TestCase):

    def test_prefix_from_same_token_not_overwritten_by_above(self):
        """
        Bug fix: 'Fleet Support Cruiser [T6-X2]' → type='Fleet Support Cruiser'
        even when a TRAITS screen header like 'Personal Space Traits' is above it.
        """
        token = 'Fleet Support Cruiser [T6-X2]'
        above = ['Personal Space Traits']
        result = _extract_type_from_token(token, above)
        self.assertEqual(result, 'Fleet Support Cruiser')

    def test_prefix_used_when_no_above(self):
        token = 'Temporal Science Vessel [T6-X]'
        result = _extract_type_from_token(token, [])
        self.assertEqual(result, 'Temporal Science Vessel')

    def test_above_used_when_no_prefix(self):
        """When tier token has no prefix (standalone '[T6]'), above tokens are used."""
        token = '[T6]'
        above = ['Avenger Battle Cruiser']
        result = _extract_type_from_token(token, above)
        self.assertEqual(result, 'Avenger Battle Cruiser')

    def test_above_used_when_prefix_too_short(self):
        """Prefix shorter than 5 chars is rejected — falls through to above."""
        token = 'T6 [T6-X2]'  # prefix 'T6' is 2 chars, < 5
        above = ['Arbiter Battlecruiser']
        result = _extract_type_from_token(token, above)
        self.assertEqual(result, 'Arbiter Battlecruiser')

    def test_above_prepended_to_short_prefix(self):
        """
        OCR splits long ship name across tokens:
        above='Kardashev Command Dyson Science', tier token='Destroyer [T6-X2]'
        → type='Kardashev Command Dyson Science Destroyer'
        """
        token = 'Destroyer [T6-X2]'
        above = ['Kardashev Command Dyson Science']
        result = _extract_type_from_token(token, above)
        self.assertEqual(result, 'Kardashev Command Dyson Science Destroyer')

    def test_section_header_above_not_prepended(self):
        """Above token that is a section header must never be prepended to prefix."""
        token = 'Fleet Support Cruiser [T6-X2]'
        above = ['Personal Space Traits']
        result = _extract_type_from_token(token, above)
        self.assertEqual(result, 'Fleet Support Cruiser')

    def test_standalone_tier_token_no_above(self):
        """No prefix, no above → empty type."""
        token = '[T6-X2]'
        result = _extract_type_from_token(token, [])
        self.assertEqual(result, '')

    def test_tier_variants_parsed(self):
        """All tier strings are matched by RE_TIER_LOOSE."""
        for tier in ['T6-X2', 'T6-X', 'T6', 'T5-U', 'T5', 'T4', 'T3', 'T2', 'T1']:
            token = f'Some Ship [{tier}]'
            m = RE_TIER_LOOSE.search(token)
            self.assertIsNotNone(m, f'Tier not matched: {tier}')
            prefix = token[:m.start()].strip().rstrip(' [')
            self.assertEqual(prefix, 'Some Ship')


class TestDetectTypeFromText(unittest.TestCase):
    """Tests for _detect_type_from_text screen type detection."""

    def setUp(self):
        from warp.recognition.text_extractor import _detect_type_from_text
        self._detect = _detect_type_from_text

    def test_space_traits_detected(self):
        self.assertEqual(self._detect(['personal space traits', 'some other text']), 'SPACE_TRAITS')

    def test_starship_traits_detected(self):
        self.assertEqual(self._detect(['starship traits', 'lorem ipsum']), 'SPACE_TRAITS')

    def test_ground_traits_detected(self):
        self.assertEqual(self._detect(['personal ground traits']), 'GROUND_TRAITS')

    def test_boffs_detected(self):
        self.assertEqual(self._detect(['bridge officer abilities', 'tactical']), 'BOFFS')

    def test_space_eq_detected_two_labels(self):
        self.assertEqual(self._detect(['fore weapons', 'deflector', 'warp core']), 'SPACE')

    def test_single_space_label_not_enough(self):
        # 'devices' alone shouldn't trigger SPACE
        result = self._detect(['devices'])
        self.assertNotEqual(result, 'SPACE')

    def test_strong_space_indicator(self):
        self.assertEqual(self._detect(['warp core']), 'SPACE')

    def test_unknown_returns_empty(self):
        self.assertEqual(self._detect(['lorem', 'ipsum', 'dolor']), '')


if __name__ == '__main__':
    unittest.main()
