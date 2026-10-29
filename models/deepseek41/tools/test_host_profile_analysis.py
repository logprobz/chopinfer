import unittest

from host_profile_analysis import charge, idle, osrt_evidence, ownership


class HostProfileTest(unittest.TestCase):
    def test_no_qualifying_main_thread_call_is_a_valid_zero(self):
        rows = [dict(start=0, end=1, globalTid=1, name="futex"),
                dict(start=5, end=6, globalTid=2, name="preadv2")]
        result = osrt_evidence(rows, 1, 4, 8)
        self.assertEqual(result["capture_main_calls"], 1)
        self.assertEqual(result["step_calls"], 1)
        self.assertEqual(result["step_main_calls"], 0)
        self.assertEqual(result["main_raw_ns"], {})

    def test_missing_collector_thread_or_window_coverage_is_refused(self):
        rows = [dict(start=0, end=1, globalTid=1, name="futex"),
                dict(start=5, end=6, globalTid=2, name="preadv2")]
        for sample, thread in (([], 1), (rows, 3), (rows[:1], 1)):
            with self.subTest(sample=sample, thread=thread):
                with self.assertRaises(AssertionError):
                    osrt_evidence(sample, thread, 4, 8)

    def test_nested_diagnostics_own_their_logger_without_double_counting(self):
        ranges = [dict(start=1, end=9, name="parent"),
                  dict(start=2, end=4, name="child"),
                  dict(start=5, end=8, name="bundle.route_diagnostics"),
                  dict(start=6, end=7, name="logger")]
        spans = ownership(ranges, 0, 10)
        self.assertEqual(charge([(0, 10)], spans),
                         dict(unannotated=2, parent=3, child=2,
                              **{"bundle.route_diagnostics": 3}))
        self.assertEqual(charge([(3, 6)], spans),
                         dict(child=1, parent=1,
                              **{"bundle.route_diagnostics": 1}))
        self.assertEqual(sum(hi - lo for lo, hi, _ in spans), 10)

    def test_gpu_activity_union_clips_overlap_and_transfers(self):
        self.assertEqual(idle([(0, 3), (2, 6), (8, 12)], 0, 10), [(6, 8)])
        self.assertEqual(idle([], 0, 10), [(0, 10)])


if __name__ == "__main__":
    unittest.main()
