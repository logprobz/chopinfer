"""Check complete candidate evidence without promoting unmeasured ranges."""
import copy
import unittest
from check_f32_prefill import check

EXPECTED = {"shards": 2, "bitwise_cases": 2, "timing_cases": 2, "launch_timings": 6,
    "bytes_per_us": 100, "winning_ratio": 0.97,
    "production": [{"baseline": "projection_tile32", "reference": "projection", "candidate": "tiles32_v1", "tile": 32,
                    "n": 17, "k": 257, "weight_bytes": 1000, "rows": [512, 2048]}]}


def log(case, rows, ratio):
    new = ratio * 100
    label = f"projection_tile32 case={case} rows={rows}"
    return (f"test prefill ... F32_TIMING {label} old_us=100 new_us={new} ratio={ratio} floor_us=10 floor_ratio={new/10}\n"
        + "".join(f"F32_LAUNCH {label} arm={arm} symbol={symbol} us=10\n" for arm, symbol in [(0, "old"), (1, "prepare"), (1, "tiles")])
        + f"F32_BITWISE {label} n=17 k=257 PASS\n"
        + "test result: ok. 1 passed; 0 failed; 0 ignored\n")


class PrefillGate(unittest.TestCase):
    def setUp(self):
        self.logs = [log(0, 512, 1.1), log(1, 2048, 0.25)]

    def test_prefix_is_counted_and_only_measured_winners_are_reported(self):
        result = check(self.logs, EXPECTED)
        self.assertEqual([r["rows"] for r in result["winning_buckets"]], [2048])
        self.assertEqual(result["first_measured_win"], {"projection": 2048})
        self.assertFalse(result["serving_promotion"])

    def test_timing_margin_is_fixed_before_measurement(self):
        for ratio, wins in [(0.97, True), (0.97001, False)]:
            result = check([self.logs[0], log(1, 2048, ratio)], EXPECTED)
            self.assertEqual(bool(result["winning_buckets"]), wins)

    def test_duplicate_or_missing_shards_fail(self):
        for texts in [self.logs[:1], [self.logs[0], self.logs[0]]]:
            with self.assertRaises(ValueError):
                check(texts, EXPECTED)

    def test_missing_marker_fails(self):
        with self.assertRaisesRegex(ValueError, "TIMING"):
            check(["\n".join(x for x in self.logs[0].splitlines() if "TIMING" not in x), self.logs[1]], EXPECTED)

    def test_invalid_timing_and_wrong_bucket_fail(self):
        for changed in [self.logs[1].replace("ratio=0.25", "ratio=nan"), self.logs[1].replace("rows=2048", "rows=2049")]:
            with self.assertRaises(ValueError):
                check([self.logs[0], changed], EXPECTED)

    def test_failed_shard_is_not_a_partial_pass(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            check([self.logs[0], self.logs[1].replace("1 passed; 0 failed", "0 passed; 1 failed")], EXPECTED)

    def test_duplicate_case_ids_fail(self):
        expected = copy.deepcopy(EXPECTED)
        expected.update(bitwise_cases=3, timing_cases=3, launch_timings=9)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            check([self.logs[0] + self.logs[0], self.logs[1]], expected)

    def test_zero_winning_buckets_is_reported_without_promotion(self):
        result = check([self.logs[0], log(1, 2048, 1.2)], EXPECTED)
        self.assertEqual(result["winning_buckets"], [])
        self.assertFalse(result["serving_promotion"])


if __name__ == "__main__":
    unittest.main()
