"""CPU regressions for prefixed records and the fixed paired timing limit."""

import unittest

from check_f32_decode import check, records


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

    def test_plain_records_are_also_accepted(self):
        check(log().replace("test projection_test ... ", ""), EXPECTED)

    def test_more_than_three_percent_fails(self):
        with self.assertRaisesRegex(ValueError, "exceeds 1.03"):
            check(log(ratio="1.03001").replace("rows=48", "rows=32"), EXPECTED)

    def test_verify_requires_at_least_five_times_faster(self):
        with self.assertRaisesRegex(ValueError, "exceeds 0.2"):
            check(log(ratio="0.20001"), EXPECTED)

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


if __name__ == "__main__":
    unittest.main()
