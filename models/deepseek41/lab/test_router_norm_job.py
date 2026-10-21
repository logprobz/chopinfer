"""CPU tests of router_norm_job's analyzers and phase control."""
import json
import unittest
from pathlib import Path
from unittest import mock

import router_norm_job as job


def record(pattern, prior, live, **over):
    uncertified = 0 if pattern == "clean" or live < 4 else max(1, live // 37)
    r = {"pattern": pattern, "prior": prior, "live": live,
         "differing_bytes": {k: 0 for k in job.COMPARED},
         "certified_rows": live - uncertified, "uncertified_rows": uncertified,
         "repeat_bitwise": True, "graph_bitwise": True, "mutant_differs": uncertified > 0,
         "failures": [], "passed": True, "emulated": False}
    r.update(over)
    return r


def bitwise_text(phase, rows=None, emulated=False, summary=True):
    memcheck = phase == "memcheck"
    rows = rows if rows is not None else [record(p, s, l, emulated=emulated) for (p, s, l) in job.cases(memcheck)]
    lines = [f"router_norm_record {json.dumps(r)}" for r in rows]
    lines.append("router_norm_cases " + json.dumps({"executed": len(rows), "failed": 0, "memcheck": memcheck}))
    if memcheck and summary and not emulated:
        lines.append("========= ERROR SUMMARY: 0 errors")
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out")
    return "\n".join(lines)


def timing(ms=(1.0, 0.85)):
    rounds = [[[ms[0]] * 3, [ms[1]] * 3] for _ in range(5)]
    return {"arms": ["served", "certified"], "ms": list(ms), "median_round_ratio": ms[1] / ms[0],
            "round_samples_ms": rounds, "replays_per_sample": 1}


def timing_text(rows=job.TIMED_ROWS):
    lines = [f"router_norm_timing {json.dumps({'rows': r, 'timing': timing()})}" for r in rows]
    lines.append("router_norm_timed " + json.dumps({"executed": len(rows)}))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out")
    return "\n".join(lines)


class Analyzers(unittest.TestCase):
    def test_bitwise_phases_accept_complete_runs(self):
        self.assertEqual(len(job.analyze_bitwise(bitwise_text("bitwise"), "bitwise")), 24)
        self.assertEqual(len(job.analyze_bitwise(bitwise_text("memcheck"), "memcheck")), 6)
        rows = [record(p, s, l, emulated=True) for (p, s, l) in job.cases(False)]
        self.assertEqual(len(job.analyze_bitwise(bitwise_text("bitwise", rows, True), "bitwise", True)), 24)

    def test_bitwise_rejects_faults(self):
        cases = job.cases(False)
        good = [record(p, s, l) for (p, s, l) in cases]
        faults = [
            lambda r: r["differing_bytes"].update(logits=4),
            lambda r: r.update(repeat_bitwise=False),
            lambda r: r.update(mutant_differs=not r["mutant_differs"]),
            lambda r: r.update(passed=False, failures=["x"]),
            lambda r: r.update(certified_rows=r["certified_rows"] + 1),
            lambda r: r.update(emulated=True),
        ]
        for fault in faults:
            rows = [dict(r, differing_bytes=dict(r["differing_bytes"])) for r in good]
            fault(rows[-1])
            with self.assertRaises(ValueError):
                job.analyze_bitwise(bitwise_text("bitwise", rows), "bitwise")
        with self.assertRaises(ValueError):
            job.analyze_bitwise(bitwise_text("bitwise", good[:-1]), "bitwise")
        with self.assertRaises(ValueError):
            job.analyze_bitwise(bitwise_text("bitwise", good + good[:1]), "bitwise")
        with self.assertRaises(ValueError):
            job.analyze_bitwise(bitwise_text("memcheck", summary=False), "memcheck")
        clean = [record("clean", s, l) for (_, s, l) in cases]
        with self.assertRaises(ValueError):
            job.analyze_bitwise(bitwise_text("bitwise", clean), "bitwise")

    def test_timing(self):
        self.assertEqual(len(job.analyze_timing(timing_text())), 3)
        with self.assertRaises(ValueError):
            job.analyze_timing(timing_text(job.TIMED_ROWS[:2]))
        emulated = "router_norm_timed {\"executed\": 0, \"emulated\": true}\n" \
                   "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out"
        self.assertEqual(job.analyze_timing(emulated, True), [])
        with self.assertRaises(ValueError):
            job.analyze_timing(emulated)
        s = job.summarize([{"timing": job.analyze_timing(timing_text())}, {"timing": job.analyze_timing(timing_text())}])
        self.assertAlmostEqual(s["timing"]["per_chunk_full_layers_ms"], 20 * 0.15)

    def test_commands_and_environment(self):
        root = Path("/stage")
        plan = {"sanitizer": "/cuda/compute-sanitizer"}
        memcheck = job.command(root, plan, "memcheck")
        self.assertEqual(memcheck[0], "/cuda/compute-sanitizer")
        self.assertIn("router_norm_bitwise_gpu", memcheck)
        self.assertNotIn("/cuda/compute-sanitizer", job.command(root, plan, "memcheck", emulated=True))
        self.assertIn("router_norm_timing_gpu", job.command(root, plan, "timing"))
        env = job.environment(root, "memcheck", 1)
        self.assertEqual((env["CHOPINFER_KERNEL_GPU"], env["CHOPINFER_KERNEL_REQUIRE"]), ("1", "1"))
        self.assertEqual(env["CHOPINFER_DEEPSEEK41_ROUTER_NORM_MEMCHECK"], "1")
        env = job.environment(root, "bitwise", 0, emulated=True)
        self.assertEqual((env["CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE"], env["CUDA_VISIBLE_DEVICES"]), ("1", ""))
        self.assertNotIn("CHOPINFER_KERNEL_GPU", env)

    def test_idle_pair_retries_busy(self):
        gpus = job.emulated_gpus(busy=2)
        with mock.patch("time.sleep"):
            self.assertEqual(len(job.idle_pair("emulated", wait=5, poll=0, gpus=gpus)), 2)
        with self.assertRaises(ValueError):
            job.idle_pair("emulated", wait=0, poll=0, gpus=job.emulated_gpus(busy=5))


if __name__ == "__main__":
    unittest.main()
