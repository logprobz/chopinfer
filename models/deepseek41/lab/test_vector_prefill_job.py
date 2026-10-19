import copy
import json
import unittest
import vector_prefill_job as job


def fixture():
    result = []
    for e in (128, 384):
        for rows in (512, 2048, 4096, 8192):
            for route in ("balanced", "skewed"):
                for variant in job.VARIANTS:
                    baseline = {}
                    for part in ("gateup", "down"):
                        if e == 384 and rows == 8192 and part == "gateup":
                            prefix = "chop_exact_prefill_e384_before_gateup_decoded128_"
                            baseline[part] = [prefix + stage + "_v1" for stage in
                                              ("pack", "predecode", "r8", "r32", "r64", "r128")]
                        elif rows < 8192 and not (e == 128 and rows == 2048 and part == "down"):
                            baseline[part] = ["group_pack_v1", "group_r8_v1", "group_r32_v1", "group_r64_v1"]
                        else:
                            baseline[part] = ["column_v1"]
                    timing = {"ms": [2., 1.], "replays_per_sample": 1,
                              "round_samples_ms": [[[2.] * 3, [1.] * 3] for _ in range(5)]}
                    times = {part: copy.deepcopy(timing) for part in ("gateup", "down", "projections", "full")}
                    result.append({
                        "experts": e, "input_rows": rows, "routing": route, "variant": variant,
                        "serving_enabled": False, "bitwise": True, "scalar_bitwise": True,
                        "clock_bitwise": True, "boundary_replays": 4,
                        "negative_control_caught": variant == "pipe64",
                        "baseline": baseline, "timing": times, "scalar_timing": copy.deepcopy(times),
                        "clock": {"parts": {part: [{"items": 1, "cta_phase_cycles": [1] * 7}
                                                  for _ in range(4)] for part in ("gateup", "down")}},
                    })
    return result


def log(records):
    return "\n".join("test prefix ... vector_prefill_result " + json.dumps(r) for r in records) + (
        "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n")


def slow(timing):
    timing["ms"] = [1., 1.2]
    timing["round_samples_ms"] = [[[1.] * 3, [1.2] * 3] for _ in range(5)]


class VectorJobTests(unittest.TestCase):
    def test_complete_matrix_and_promoted_baseline(self):
        text = log(fixture())
        summary = job.summarize([text, text])
        self.assertEqual(len(summary["records"]), 96)
        self.assertEqual(len(summary["kernel_wins"]), 48)

    def test_gate_only_win_is_not_vetoed_by_down(self):
        records = fixture()
        for row in records:
            for part in ("down", "projections", "full"):
                slow(row["timing"][part])
        result = job.summarize([log(records), log(records)])
        self.assertEqual(len(result["kernel_wins"]), 24)
        self.assertTrue(all(w["part"] == "gateup" for w in result["kernel_wins"]))

    def test_every_routing_and_device_must_win(self):
        records = fixture()
        slow(records[0]["timing"]["gateup"])
        result = job.summarize([log(records), log(fixture())])
        self.assertEqual(len(result["kernel_wins"]), 47)

    def test_old_column_gate_is_not_a_serving_baseline(self):
        records = fixture()
        record = next(r for r in records if r["experts"] == 384 and r["input_rows"] == 8192)
        record["baseline"]["gateup"] = ["column_v1"]
        with self.assertRaises(ValueError):
            job.summarize([log(records)] * 2)

    def test_missing_scalar_parity_or_timing_is_rejected(self):
        records = fixture()
        records[0]["scalar_bitwise"] = False
        with self.assertRaises(ValueError):
            job.summarize([log(records)] * 2)
        records = fixture()
        records[0]["scalar_timing"]["down"]["round_samples_ms"].pop()
        with self.assertRaises(ValueError):
            job.summarize([log(records)] * 2)
        with self.assertRaises(ValueError):
            job.summarize([log(fixture()[:-1])] * 2)


if __name__ == "__main__":
    unittest.main()
