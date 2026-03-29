"""
tests/test_community_anchors.py
Unit tests for P11 community anchors aggregation logic.
Tests build_community_anchors() median computation without HF calls.
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'sets-warp-backend'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Pure-logic tests — extracted from build_community_anchors without HF I/O
# ---------------------------------------------------------------------------

import statistics
from collections import defaultdict


def _aggregate_entries(raw_entries: list[dict], min_contributors: int = 3) -> list[dict]:
    """
    Mirrors the aggregation core of build_community_anchors() without HF calls.
    raw_entries: list of {'install_id': str, 'build_type': str, 'aspect': float,
                           'resolution': str, 'slots': {slot: geo_dict}}
    """
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in raw_entries:
        build_type   = e.get('build_type', '')
        aspect       = e.get('aspect')
        install_id   = e['install_id']
        if not build_type or aspect is None:
            continue
        key = (build_type, round(float(aspect), 2))
        groups[key][install_id].append(e)

    results = []
    for (build_type, aspect_bucket), contributors in groups.items():
        if len(contributors) < min_contributors:
            continue
        slot_vectors: dict[str, list[dict]] = defaultdict(list)
        resolutions: list[str] = []
        for iid, entries in contributors.items():
            for e in entries:
                for slot_name, geo in e.get('slots', {}).items():
                    if isinstance(geo, dict):
                        slot_vectors[slot_name].append(geo)
                if e.get('resolution'):
                    resolutions.append(e['resolution'])

        aggregated = {}
        for slot_name, geos in slot_vectors.items():
            def _med(key):
                vals = [g[key] for g in geos if key in g]
                return round(statistics.median(vals), 5) if vals else None
            entry_out = {
                'x0_rel':   _med('x0_rel'),
                'y_rel':    _med('y_rel'),
                'w_rel':    _med('w_rel'),
                'h_rel':    _med('h_rel'),
                'step_rel': _med('step_rel'),
                'count':    round(statistics.median([g['count'] for g in geos if 'count' in g])) if geos else 1,
            }
            if None not in entry_out.values():
                aggregated[slot_name] = entry_out

        if not aggregated:
            continue
        rep_res = max(set(resolutions), key=resolutions.count) if resolutions else ''
        results.append({
            'type':           build_type,
            'aspect':         aspect_bucket,
            'res':            rep_res,
            'slots':          aggregated,
            'n_contributors': len(contributors),
        })
    return results


def _make_geo(x0=0.1, y=0.3, w=0.05, h=0.06, step=0.06, count=4):
    return {'x0_rel': x0, 'y_rel': y, 'w_rel': w, 'h_rel': h, 'step_rel': step, 'count': count}


def _make_entry(install_id, build_type='SPACE_EQ', aspect=1.78, resolution='1920x1080', slots=None):
    if slots is None:
        slots = {
            'Fore Weapons': _make_geo(),
            'Deflector':    _make_geo(x0=0.2, y=0.4, count=1),
            'Engines':      _make_geo(x0=0.25, y=0.45, count=1),
            'Shield':       _make_geo(x0=0.3, y=0.5, count=1),
        }
    return {'install_id': install_id, 'build_type': build_type,
            'aspect': aspect, 'resolution': resolution, 'slots': slots}


class TestCommunityAnchorsAggregation(unittest.TestCase):

    def test_below_min_contributors_excluded(self):
        entries = [_make_entry('user1'), _make_entry('user2')]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(result, [])

    def test_exactly_min_contributors_included(self):
        entries = [_make_entry('u1'), _make_entry('u2'), _make_entry('u3')]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['type'], 'SPACE_EQ')
        self.assertEqual(result[0]['n_contributors'], 3)

    def test_median_computed_correctly(self):
        """Median of [0.10, 0.12, 0.14] = 0.12."""
        def _slots(x0):
            return {'Fore Weapons': {'x0_rel': x0, 'y_rel': 0.3, 'w_rel': 0.05, 'h_rel': 0.06, 'step_rel': 0.06, 'count': 4},
                    'Deflector': _make_geo(x0=0.2, y=0.4, count=1),
                    'Engines':   _make_geo(x0=0.25, y=0.45, count=1),
                    'Shield':    _make_geo(x0=0.3, y=0.5, count=1)}
        entries = [
            _make_entry('u1', slots=_slots(0.10)),
            _make_entry('u2', slots=_slots(0.12)),
            _make_entry('u3', slots=_slots(0.14)),
        ]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(len(result), 1)
        fore = result[0]['slots']['Fore Weapons']
        self.assertAlmostEqual(fore['x0_rel'], 0.12, places=4)

    def test_different_aspects_not_merged(self):
        entries = [
            _make_entry('u1', aspect=1.78),
            _make_entry('u2', aspect=1.78),
            _make_entry('u3', aspect=1.78),
            _make_entry('u4', aspect=1.60),
            _make_entry('u5', aspect=1.60),
            _make_entry('u6', aspect=1.60),
        ]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(len(result), 2)
        aspects = {round(r['aspect'], 2) for r in result}
        self.assertIn(1.78, aspects)
        self.assertIn(1.6, aspects)

    def test_different_build_types_not_merged(self):
        entries = (
            [_make_entry(f'u{i}', build_type='SPACE_EQ') for i in range(3)] +
            [_make_entry(f'v{i}', build_type='GROUND') for i in range(3)]
        )
        result = _aggregate_entries(entries, min_contributors=3)
        types = {r['type'] for r in result}
        self.assertIn('SPACE_EQ', types)
        self.assertIn('GROUND', types)

    def test_empty_build_type_skipped(self):
        entries = [
            {'install_id': 'u1', 'build_type': '', 'aspect': 1.78, 'slots': {}},
            {'install_id': 'u2', 'build_type': '', 'aspect': 1.78, 'slots': {}},
            {'install_id': 'u3', 'build_type': '', 'aspect': 1.78, 'slots': {}},
        ]
        result = _aggregate_entries(entries, min_contributors=1)
        self.assertEqual(result, [])

    def test_most_common_resolution_used(self):
        entries = [
            _make_entry('u1', resolution='1920x1080'),
            _make_entry('u2', resolution='1920x1080'),
            _make_entry('u3', resolution='2560x1440'),
        ]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(result[0]['res'], '1920x1080')

    def test_same_user_multiple_entries_counts_once(self):
        """Two entries from same user should count as 1 contributor."""
        entries = [
            _make_entry('u1'),
            _make_entry('u1'),  # same install_id — counts as 1
            _make_entry('u2'),
            _make_entry('u3'),
        ]
        result = _aggregate_entries(entries, min_contributors=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['n_contributors'], 3)


class TestLayoutDetectorCacheReset(unittest.TestCase):

    def test_reset_clears_instance_cache(self):
        from warp.recognition.layout_detector import LayoutDetector
        ld = LayoutDetector()
        ld._community_anchors = [{'type': 'SPACE_EQ', 'aspect': 1.78}]  # fake cached data
        LayoutDetector.reset_community_anchors_cache()
        self.assertIsNone(ld._community_anchors)

    def test_reset_clears_multiple_instances(self):
        from warp.recognition.layout_detector import LayoutDetector
        ld1 = LayoutDetector()
        ld2 = LayoutDetector()
        ld1._community_anchors = [{'test': 1}]
        ld2._community_anchors = [{'test': 2}]
        LayoutDetector.reset_community_anchors_cache()
        self.assertIsNone(ld1._community_anchors)
        self.assertIsNone(ld2._community_anchors)


if __name__ == '__main__':
    unittest.main()
