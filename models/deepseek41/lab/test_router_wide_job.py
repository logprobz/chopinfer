"""CPU checks of router_wide_job's data preparation, analyzer, summary and
Xid filter on synthetic inputs and logs of the format the frozen test
binary prints."""
import json
import struct
import unittest

import router_wide_job as job
from test_native_moe_job import mutate, timing


def log(worker):
    lines = ["running 1 test", "test wide::router_wide_gpu ... "]
    for case, live in job.CASES[worker]:
        arms = [{"arm": a, "bitwise": True, "timing": timing(0.40, 0.35) if live == job.ROWS else None}
                for a in job.ARMS]
        synthetic = not case.startswith("captured")
        lines.append("router_wide_record " + json.dumps({
            "case": case, "live": live, "uncertified_rows": 9 if synthetic else 0,
            "simt_weight_groups": 1 if synthetic else 0, "arms": arms, "passed": True}))
    lines.append("router_wide_done " + json.dumps({"worker": worker, "cases": 3, "failed": 0}))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out")
    return "\n".join(lines)


class Data(unittest.TestCase):
    def test_e4m3_values(self):
        self.assertEqual(job.e4m3(0x38), 1.0)
        self.assertEqual(job.e4m3(0x7e), 448.0)
        self.assertEqual(job.e4m3(0x01), 2.0 ** -9)
        self.assertEqual(job.e4m3(0xb8), -1.0)
        self.assertNotEqual(job.e4m3(0x7f), job.e4m3(0x7f))

    def test_widen_codes_is_exact_bf16(self):
        codes = bytes([0x38, 0x7e, 0x01, 0x80] + [0] * 28 + [0x38] * 32)
        scales = bytes([127, 130])
        out = job.widen_codes(codes, scales, 1, 64)
        halves = struct.unpack("<64H", out)
        self.assertEqual(halves[:4], (0x3f80, 0x43e0, 0x3b00, 0x8000))
        self.assertEqual(halves[32], 0x4100)  # 1.0 * 2^3
        with self.assertRaises(ValueError):
            job.widen_codes(codes, bytes([255, 127]), 1, 64)
        with self.assertRaises(ValueError):
            job.widen_codes(codes[:-1], scales, 1, 64)


class Analyzer(unittest.TestCase):
    def test_accepts_and_summarizes(self):
        parsed = [job.analyze(log(w), w) for w in (0, 1)]
        s = job.summarize(parsed)
        self.assertEqual(set(s), set(job.ARMS))
        self.assertEqual(len(s["f64"]["cases"]), 4)
        self.assertAlmostEqual(s["f128"]["per_chunk_saving_ms"], job.CHUNK_PROJECTION_MS * 0.125, places=4)

    def test_rejects_differences_and_missing_or_misassigned_work(self):
        good = log(0)
        arm = lambda name, change: lambda r: [change(a) for a in r["arms"] if a["arm"] == name]
        bad = [
            mutate(good, "router_wide_record", arm("f64", lambda a: a.update(bitwise=False))),
            mutate(good, "router_wide_record", lambda r: r.update(passed=False)),
            mutate(good, "router_wide_record", lambda r: r["arms"].pop()),
            mutate(good, "router_wide_record", lambda r: r.update(live=r["live"] + 1)),
            mutate(good, "router_wide_record", arm("f32", lambda a: a.update(timing=None) if a["timing"] else None)),
            mutate(good, "router_wide_record", lambda r: r.update(uncertified_rows=0)),
            mutate(good, "router_wide_done", lambda r: r.update(failed=1)),
            "\n".join(l for l in good.splitlines() if "captured-l14" not in l),
            good.replace("test result: ok. 1 passed", "test result: ok. 0 passed"),
        ]
        for text in bad:
            with self.assertRaises((ValueError, KeyError, TypeError)):
                job.analyze(text, 0)
        with self.assertRaises(ValueError):
            job.analyze(good, 1)


class Xids(unittest.TestCase):
    def test_counts_only_the_assigned_gpus(self):
        lines = ["NVRM: Xid (PCI:0000:01:00): 43, pid=7", "NVRM: Xid (PCI:0000:02:00): 13, pid=8",
                 "NVRM: Xid 79 no device", "unrelated"]
        host, own = job.own_xids(lines, {"0000:02:00"})
        self.assertEqual(len(host), 3)
        self.assertEqual(own, lines[1:3])


if __name__ == "__main__":
    unittest.main()
