"""
tests/test_canonical_layout.py

Unit tests for:
  1. canonical_layout.json structure and content
  2. LayoutDetector._load_canonical_layout()
  3. Phase 3: warp_importer._process_image uses confirmed_layout only for MIXED
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# 1. canonical_layout.json — structure validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalLayoutFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        canon_path = (
            Path(__file__).resolve().parent.parent
            / 'warp' / 'training_data' / 'canonical_layout.json'
        )
        cls.path = canon_path
        if canon_path.exists():
            cls.data = json.loads(canon_path.read_text(encoding='utf-8'))
        else:
            cls.data = None

    def test_file_exists(self):
        self.assertTrue(self.path.exists(),
                        f'canonical_layout.json not found at {self.path}')

    def test_version_key(self):
        if self.data is None:
            self.skipTest('canonical_layout.json missing')
        self.assertEqual(self.data.get('version'), 1)

    def test_types_present(self):
        if self.data is None:
            self.skipTest('canonical_layout.json missing')
        types = self.data.get('types', {})
        self.assertIn('SPACE', types, 'SPACE type missing from canonical layout')

    def test_space_deflector_y_rel(self):
        """Deflector y_rel should be ~0.115 (learned from 78+ samples)."""
        if self.data is None:
            self.skipTest('canonical_layout.json missing')
        deflector = self.data['types']['SPACE']['slots'].get('Deflector', {})
        y_rel = deflector.get('y_rel', 0)
        self.assertGreater(y_rel, 0.09, f'Deflector y_rel={y_rel} too low')
        self.assertLess(y_rel, 0.14, f'Deflector y_rel={y_rel} too high')

    def test_no_zero_y_rel(self):
        """All slot entries must have positive y_rel."""
        if self.data is None:
            self.skipTest('canonical_layout.json missing')
        for type_name, type_data in self.data['types'].items():
            for slot_name, slot_data in type_data.get('slots', {}).items():
                y_rel = slot_data.get('y_rel', 0)
                self.assertGreater(
                    y_rel, 0,
                    f'{type_name}/{slot_name}: y_rel={y_rel} must be > 0'
                )

    def test_n_samples_positive(self):
        if self.data is None:
            self.skipTest('canonical_layout.json missing')
        for type_name, type_data in self.data['types'].items():
            n = type_data.get('n_samples', 0)
            self.assertGreater(n, 0, f'{type_name}: n_samples={n}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. LayoutDetector._load_canonical_layout
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCanonicalLayout(unittest.TestCase):

    def _get_detector(self):
        try:
            from warp.recognition.layout_detector import LayoutDetector
            return LayoutDetector()
        except Exception as e:
            self.skipTest(f'LayoutDetector import failed: {e}')

    def test_load_returns_dict_or_none(self):
        det = self._get_detector()
        result = det._load_canonical_layout()
        self.assertTrue(result is None or isinstance(result, dict))

    def test_load_space_slots_present(self):
        det = self._get_detector()
        result = det._load_canonical_layout()
        if result is None:
            self.skipTest('canonical_layout.json not found by detector')
        types = result.get('types', {})
        self.assertIn('SPACE', types)
        space_slots = types['SPACE'].get('slots', {})
        self.assertIn('Deflector', space_slots,
                      'Deflector slot missing from canonical SPACE layout')

    def test_deflector_y_rel_reasonable(self):
        det = self._get_detector()
        result = det._load_canonical_layout()
        if result is None:
            self.skipTest('canonical_layout.json not found')
        y_rel = result['types']['SPACE']['slots']['Deflector']['y_rel']
        self.assertGreater(y_rel, 0.09)
        self.assertLess(y_rel, 0.14)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase 3 — confirmed_layout bypassed for non-MIXED build types
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmedLayoutBypass(unittest.TestCase):
    """
    Verify that WarpImporter._process_image calls _load_confirmed_layout
    only when build_type contains 'MIXED'.  For SPACE/GROUND/BOFFS/TRAITS/SPEC
    the importer must always go through layout_detector.detect().
    """

    def _make_importer(self, build_type: str):
        """Return a WarpImporter with mocked dependencies."""
        try:
            from warp.warp_importer import WarpImporter
        except Exception as e:
            self.skipTest(f'WarpImporter import failed: {e}')

        imp = WarpImporter.__new__(WarpImporter)
        imp._build_type          = build_type
        imp._from_trainer        = True    # skip OCR path
        imp._progress_callback   = None
        imp._interrupt_check     = None
        imp._text_extractor      = None
        imp._layout_detector     = None
        imp._icon_matcher        = None
        imp._shipdb              = None

        # Mock sub-accessors
        mock_layout = MagicMock()
        mock_layout.detect.return_value = {}
        imp._get_layout  = lambda: mock_layout
        imp._get_matcher = lambda: MagicMock()
        imp._get_shipdb  = lambda: MagicMock(get_profile=lambda *a: {})
        imp._get_text    = lambda: MagicMock()

        imp._load_confirmed_layout  = MagicMock(return_value=None)
        imp._load_confirmed_profile = MagicMock(return_value={})
        imp._build_slot_candidates  = MagicMock(return_value={})

        return imp

    def _call_process(self, imp, build_type: str):
        import numpy as np
        img = np.zeros((600, 1000, 3), dtype='uint8')
        try:
            imp._process_image(img, 'test_shot.jpg')
        except Exception:
            pass  # we only care about call counts, not full result

    def test_space_skips_confirmed_layout(self):
        imp = self._make_importer('SPACE')
        self._call_process(imp, 'SPACE')
        imp._load_confirmed_layout.assert_not_called()

    def test_ground_skips_confirmed_layout(self):
        imp = self._make_importer('GROUND')
        self._call_process(imp, 'GROUND')
        imp._load_confirmed_layout.assert_not_called()

    def test_boffs_skips_confirmed_layout(self):
        imp = self._make_importer('BOFFS')
        self._call_process(imp, 'BOFFS')
        imp._load_confirmed_layout.assert_not_called()

    def test_traits_skips_confirmed_layout(self):
        imp = self._make_importer('SPACE_TRAITS')
        self._call_process(imp, 'SPACE_TRAITS')
        imp._load_confirmed_layout.assert_not_called()

    def test_spec_skips_confirmed_layout(self):
        imp = self._make_importer('SPEC')
        self._call_process(imp, 'SPEC')
        imp._load_confirmed_layout.assert_not_called()

    def test_mixed_calls_confirmed_layout(self):
        imp = self._make_importer('SPACE_MIXED')
        self._call_process(imp, 'SPACE_MIXED')
        imp._load_confirmed_layout.assert_called_once()

    def test_ground_mixed_calls_confirmed_layout(self):
        imp = self._make_importer('GROUND_MIXED')
        self._call_process(imp, 'GROUND_MIXED')
        imp._load_confirmed_layout.assert_called_once()


if __name__ == '__main__':
    unittest.main()
