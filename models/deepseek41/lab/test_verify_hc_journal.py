"""Keep dependency audits strict across opaque stream handle renumbering."""
import unittest
from verify_hc_journal import align_buffers, align_streams, launches

OLD = '''record 2 10
wait 3 10
launch 3 op LaunchConfigV1 {} [ScalarU32(7)]
record 3 11
wait 2 11
launch 2 next LaunchConfigV1 {} [ScalarU32(9)]
'''
NEW = '''record 2 110
wait 103 110
launch 103 op LaunchConfigV1 {} [ScalarU32(7)]
record 103 111
wait 2 111
launch 2 next LaunchConfigV1 {} [ScalarU32(9)]
'''


class DependencyIdentity(unittest.TestCase):
    def test_bijective_renumbering_keeps_launches(self):
        self.assertEqual(launches(OLD), launches(align_streams(OLD, NEW)))

    def test_changed_wait_is_rejected(self):
        with self.assertRaises(ValueError):
            align_streams(OLD, NEW.replace('wait 103 110', 'wait 103 112'))

    def test_stream_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            align_streams(OLD, NEW.replace(' 103 ', ' 2 '))

    def test_launch_moved_to_another_stream_stays_visible(self):
        moved = NEW.replace('launch 103 op', 'launch 2 op')
        with self.assertRaises(ValueError):
            align_streams(OLD, moved)

    def test_allocation_identity_preserves_extents_and_aliases(self):
        old = {'buffers': [1, 2], 'ranks': {'1': [0, 16], '2': [1, 32]}}
        new = {'buffers': [3, 4], 'ranks': {'3': [0, 16], '4': [1, 32]}}
        self.assertEqual(align_buffers(old, new, 'raw: 4294967299 offset: 7'),
                         'raw: 4294967297 offset: 7')
        new['ranks']['3'] = [0, 17]
        with self.assertRaises(ValueError):
            align_buffers(old, new, '')
        new['buffers'] = [3, 3]
        with self.assertRaises(ValueError):
            align_buffers(old, new, '')


if __name__ == '__main__':
    unittest.main()
