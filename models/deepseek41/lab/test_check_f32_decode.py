"""CPU regressions for prefixed records and the fixed paired timing limit."""

import unittest

from check_f32_decode import check, check_shards, records, speed_goals


EXPECTED = {
    "bytes_per_us": 873.8,
    "bitwise_cases": 1, "timing_cases": 1, "launch_timings": 2,
    "production": [{"baseline": "projection", "rows": [48], "timing_rows": [48],
                    "n": 17, "k": 257, "weight_bytes": 8738}],
}


def log(ratio="0.2", old="100", new="20", floor="10", floor_ratio="2"):
    return (
        "running 1 test\n"
        f"test projection_test ... F32_TIMING projection rows=48 old_us={old} new_us={new} ratio={ratio} floor_us={floor} floor_ratio={floor_ratio}\n"
        "F32_LAUNCH projection rows=48 arm=0 symbol=original_v1 us=100\n"
        "F32_LAUNCH projection rows=48 arm=1 symbol=original_v1 us=103\n"
        "F32_BITWISE projection rows=48 n=17 k=257 PASS\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )


class TimingGate(unittest.TestCase):
    def test_libtest_prefix_keeps_the_first_timing(self):
        self.assertEqual(len(records(log())["TIMING"]), 1)
        check(log(), EXPECTED)

    def test_marker_without_a_space_after_the_prefix_is_accepted(self):
        check(log().replace("test projection_test ... ", "worker[0]:"), EXPECTED)

    def test_plain_records_are_also_accepted(self):
        check(log().replace("test projection_test ... ", ""), EXPECTED)

    def test_more_than_three_percent_fails(self):
        with self.assertRaisesRegex(ValueError, "exceeds 1.03"):
            check(log(ratio="1.03001"), EXPECTED)

    def test_fivefold_is_a_goal_and_a_threefold_bucket_passes(self):
        parsed = check(log(ratio="0.32374275", new="32.374275", floor_ratio="3.2374275"), EXPECTED)
        goal = speed_goals(parsed, EXPECTED)
        self.assertEqual(goal["met"], 0)
        self.assertFalse(goal["pass_condition"])
        self.assertGreater(goal["buckets"][0]["speedup"], 3)

    def test_gate_applies_only_to_routed_buckets(self):
        expected = dict(EXPECTED, production=[dict(EXPECTED["production"][0], routed_rows=[])])
        check(log(ratio="1.2", new="120", floor_ratio="12"), expected)
        expected["production"][0]["routed_rows"] = [48]
        with self.assertRaisesRegex(ValueError, "exceeds 1.03"):
            check(log(ratio="1.2", new="120", floor_ratio="12"), expected)

    def test_floor_uses_actual_stored_weight_bytes(self):
        with self.assertRaisesRegex(ValueError, "stored weight bytes"):
            check(log(floor="20", floor_ratio="1"), EXPECTED)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            check(log(floor_ratio="3"), EXPECTED)

    def test_nonfinite_or_nonpositive_timings_fail(self):
        for kwargs in ({"ratio": "nan"}, {"ratio": "inf"}, {"old": "0"}, {"new": "-1"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "invalid paired"):
                check(log(**kwargs), EXPECTED)

    def test_missing_first_record_fails(self):
        text = "\n".join(line for line in log().splitlines() if "F32_TIMING" not in line)
        with self.assertRaisesRegex(ValueError, "TIMING: expected 1, found 0"):
            check(text, EXPECTED)

    def test_duplicate_record_fails(self):
        text = log() + "F32_TIMING projection rows=48 old_us=100 new_us=100 ratio=1\n"
        with self.assertRaisesRegex(ValueError, "TIMING: expected 1, found 2"):
            check(text, EXPECTED)

    def test_wrong_bucket_fails_even_when_counts_match(self):
        with self.assertRaisesRegex(ValueError, "rows=48: missing"):
            check(log().replace("rows=48", "rows=64"), EXPECTED)

    def test_unexecuted_test_fails(self):
        with self.assertRaisesRegex(ValueError, "did not execute"):
            check(log() + "skipping GPU test\n", EXPECTED)


class HostTimingGate(unittest.TestCase):
    def setUp(self):
        self.expected = dict(EXPECTED, host_cases=1)
        self.host = ("test prefix ... F32_HOST projection rows=48 fixture_ms=1 setup_ms=2 "
                     "poison_ms=3 comparison_ms=4 timing_ms=5 total_ms=15.1\n")

    def test_prefixed_host_time_passes(self):
        check(log() + self.host, self.expected)

    def test_poison_expansion_must_run(self):
        expected = dict(self.expected, poison_checks=1)
        with self.assertRaisesRegex(ValueError, "poison expansion"):
            check(log() + self.host, expected)
        check(log() + self.host + "F32_POISON device_expansion bytes=16777353 PASS\n", expected)

    def test_missing_host_record_fails(self):
        with self.assertRaisesRegex(ValueError, "HOST: expected"):
            check(log(), self.expected)

    def test_duplicate_or_wrong_case_fails(self):
        with self.assertRaisesRegex(ValueError, "HOST: expected"):
            check(log() + self.host * 2, self.expected)
        with self.assertRaisesRegex(ValueError, "host record does not match"):
            check(log() + self.host.replace("rows=48", "rows=64"), self.expected)

    def test_invalid_or_inconsistent_host_time_fails(self):
        for replacement in ["nan", "-1", "0"]:
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "host time"):
                check(log() + self.host.replace("total_ms=15.1", f"total_ms={replacement}"), self.expected)


class ShardedGate(unittest.TestCase):
    def setUp(self):
        import copy
        self.expected = copy.deepcopy(EXPECTED)
        self.expected.update(shards=2, bitwise_cases=2, timing_cases=2, launch_timings=4)
        self.expected["production"][0].update(rows=[48, 64], timing_rows=[48, 64])
        self.logs = [log().replace("projection rows=48", "projection case=0 rows=48"),
                     log().replace("projection rows=48", "projection case=1 rows=64")]

    def test_complete_disjoint_shards_pass(self):
        check_shards(self.logs, self.expected)

    def test_missing_or_wrong_shard_fails(self):
        with self.assertRaisesRegex(ValueError, "expected 2 shard logs"):
            check_shards(self.logs[:1], self.expected)
        with self.assertRaisesRegex(ValueError, "wrong shard"):
            check_shards(self.logs[::-1], self.expected)

    def test_duplicate_or_omitted_case_id_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            check_shards([self.logs[0] + self.logs[0], self.logs[1]], self.expected)
        with self.assertRaisesRegex(ValueError, "omit expected case IDs"):
            check_shards([self.logs[0], self.logs[1].replace("case=1", "case=2")], self.expected)

    def test_one_failed_shard_cannot_hide_behind_the_other(self):
        failed = self.logs[1].replace("1 passed; 0 failed", "0 passed; 1 failed")
        with self.assertRaisesRegex(ValueError, "shard 1: GPU test did not pass"):
            check_shards([self.logs[0], failed], self.expected)


if __name__ == "__main__":
    unittest.main()
