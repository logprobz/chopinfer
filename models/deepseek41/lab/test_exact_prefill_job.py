import copy
import json
import unittest
import exact_prefill_job as job


def fixture():
    rows = []
    for e in (128, 384):
        for r in (512, 2048, 4096, 8192):
            for route in ("balanced", "skewed"):
                for variant in job.VARIANTS:
                    timing = {"ms": [2., 1.], "replays_per_sample": 1,
                              "round_samples_ms": [[[2.] * 3, [1.] * 3] for _ in range(5)]}
                    baseline = {}
                    for part in ("gateup", "down"):
                        grouped = r < 8192 and not (e == 128 and r == 2048 and part == "down")
                        baseline[part] = ["x_pack_v1", "x_r8_v1", "x_r32_v1", "x_r64_v1"] if grouped else ["column_v1"]
                    rows.append({"experts": e, "input_rows": r, "routing": route, "variant": variant,
                                 "serving_enabled": False, "bitwise": True, "clock_bitwise": True,
                                 "boundary_replays": 4, "negative_control_caught": variant == "pipe64",
                                 "baseline": baseline, "timing": {part: copy.deepcopy(timing) for part in ("gateup", "down", "projections", "full")},
                                 "clock": {"parts": {part: [{"items": 1, "cta_phase_cycles": [1] * 7} for _ in range(4)] for part in ("gateup", "down")}}})
    return rows


def log(rows):
    return "\n".join("test prefix ... exact_prefill_result " + json.dumps(row) for row in rows) + "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n"


class JobTests(unittest.TestCase):
    def test_complete_pair_and_baseline_selection(self):
        text = log(fixture())
        out = job.summarize([text, text])
        self.assertEqual(len(out["records"]), 256)
        self.assertEqual(len(out["both_device_wins"]), 128)
        self.assertTrue(out["bitwise"])

    def test_missing_controls_and_serving_winner_are_rejected(self):
        for key, value in (("bitwise", False), ("clock_bitwise", False), ("boundary_replays", 3), ("serving_enabled", True)):
            rows = fixture()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                job.summarize([log(rows), log(rows)])
        rows = fixture()
        rows[0]["baseline"]["gateup"] = ["column_v1"]
        with self.assertRaises(ValueError):
            job.summarize([log(rows), log(rows)])
        with self.assertRaises(ValueError):
            job.summarize([log(fixture()[:-1])] * 2)

    def test_both_devices_and_noise_gate(self):
        rows = fixture()
        # A tie or a single-device win never qualifies as a measured win.
        for part in rows[0]["timing"].values():
            part["ms"] = [1., 1.]
            part["round_samples_ms"] = [[[1.] * 3, [1.] * 3] for _ in range(5)]
        out = job.summarize([log(rows), log(fixture())])
        self.assertEqual(len(out["both_device_wins"]), 127)

    def test_numeric_test_verdict_and_interleaved_medians(self):
        job.passed("test result: ok. 1 passed; 0 failed; 0 ignored")
        for text in ("test result: ok. 0 passed; 0 failed; 1 ignored", "FAILED", "skipping"):
            with self.assertRaises(ValueError):
                job.passed(text)
        t = fixture()[0]["timing"]["full"]
        t["round_samples_ms"][0][0] = [2., 2., 99.]
        self.assertEqual(job.timing_ratio(t), .5)
        t["ms"][1] = 2.
        with self.assertRaises(ValueError):
            job.timing_ratio(t)


if __name__ == "__main__":
    unittest.main()
