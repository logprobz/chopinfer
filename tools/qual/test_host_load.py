"""CPU checks for the host load every receipt records and its timing label."""

from qual_test_config import setUpModule

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import common


def sample(load, idle, total, at=0.0):
    return {"at": at, "load_average": [load, load, load], "cpu_idle_ticks": idle, "cpu_ticks": total}


class HostLoadRuleTests(unittest.TestCase):
    def test_proc_is_parsed(self):
        files = {"/proc/loadavg": "1.25 0.90 0.50 2/900 1234\n",
                 "/proc/stat": "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 0 0 0 0 0\n"}
        with patch.object(common.Path, "read_text", lambda path: files[str(path)]):
            row = common.host_load_sample()
        self.assertEqual(row["load_average"], [1.25, 0.90, 0.50])
        self.assertEqual((row["cpu_idle_ticks"], row["cpu_ticks"]), (850, 1000))
        self.assertEqual(row["cpus"], 1)
        self.assertEqual(common.host_load([row, row])["cpus"], 1)

    def test_missing_proc_gives_no_sample(self):
        with patch.object(common.Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(common.host_load_sample())

    def test_an_idle_host_keeps_timing_as_evidence(self):
        # The c55 spec soak: load about 1.2 on 48 CPUs, 97% idle.
        verdict = common.host_load([sample(1.2, 0, 0), sample(1.3, 970, 1000)], cpus=48)
        self.assertEqual((verdict["idle"], verdict["timing"], verdict["reasons"]), (True, "evidence", []))
        self.assertAlmostEqual(verdict["min_cpu_idle_fraction"], 0.97)
        self.assertEqual(verdict["max_load_average_1m"], 1.3)

    def test_the_rule_bounds_are_inclusive(self):
        verdict = common.host_load([sample(4.8, 0, 0), sample(4.8, 900, 1000)], cpus=48)
        self.assertTrue(verdict["idle"])

    def test_high_load_makes_timing_informational(self):
        # The 2026-10-15 spec soak: load 61-75, 7-9% idle.
        verdict = common.host_load([sample(61, 0, 0), sample(75, 80, 1000)], cpus=48)
        self.assertEqual((verdict["idle"], verdict["timing"]), (False, "informational"))
        self.assertEqual(len(verdict["reasons"]), 2)
        verdict = common.host_load([sample(4.9, 0, 0), sample(1, 990, 1000)], cpus=48)
        self.assertFalse(verdict["idle"])
        self.assertIn("load average", verdict["reasons"][0])

    def test_one_busy_interval_is_enough(self):
        rows = [sample(1, 0, 0), sample(1, 990, 1000), sample(1, 1500, 2000), sample(1, 2490, 3000)]
        verdict = common.host_load(rows, cpus=48)
        self.assertFalse(verdict["idle"])
        self.assertAlmostEqual(verdict["min_cpu_idle_fraction"], 0.51)
        self.assertAlmostEqual(verdict["cpu_idle_fraction"], 0.83)

    def test_unknown_load_is_not_idle(self):
        for rows in ([], [sample(1, 0, 0)], [None, sample(1, 990, 1000)], [sample(1, 5, 10), sample(1, 5, 10)]):
            with self.subTest(rows=rows):
                verdict = common.host_load(rows, cpus=48)
                self.assertEqual((verdict["idle"], verdict["timing"]), (False, "informational"))
                self.assertTrue(verdict["reasons"])


    def test_the_rust_captures_apply_the_same_rule(self):
        source = (common.ROOT / "crates/runtime-eval/src/capture/host_load.rs").read_text()
        self.assertIn(f"pub const RULE: &str = {json.dumps(common.HOST_LOAD_RULE)};", source)
        self.assertIn(f"pub const IDLE_LOAD_PER_CPU: f64 = {common.IDLE_LOAD_PER_CPU};", source)
        self.assertIn(f"pub const IDLE_CPU_FRACTION: f64 = {common.IDLE_CPU_FRACTION};", source)


class ReceiptHostLoadTests(unittest.TestCase):
    def write(self, result, samples=None, end=None):
        with patch.object(common, "host_load_sample", return_value=end):
            return json.loads(Path(common.receipt("probe", {"model": "fake"}, result,
                                                  load_samples=samples)).read_text())

    def test_every_receipt_records_host_load_from_start_to_end(self):
        with patch.object(common, "HOST_LOAD_START", sample(1, 0, 0)):
            value = self.write({"passed": True, "latency_mean_seconds": 1.0}, end=sample(2, 950, 1000))
        load = value["host_load"]
        self.assertEqual(load["rule"], common.HOST_LOAD_RULE)
        self.assertEqual(load["scope"], "check_host")
        self.assertEqual(len(load["samples"]), 2)
        self.assertEqual((load["idle"], load["timing"]), (True, "evidence"))

    def test_samples_taken_during_the_check_are_judged_too(self):
        with patch.object(common, "HOST_LOAD_START", sample(1, 0, 0)):
            value = self.write({"passed": True}, samples=[sample(30, 100, 1000)], end=sample(1, 1090, 2000))
        self.assertEqual(len(value["host_load"]["samples"]), 3)
        self.assertFalse(value["host_load"]["idle"])

    def test_a_busy_host_leaves_the_verdict_alone(self):
        for passed in (True, False):
            with self.subTest(passed=passed), patch.object(common, "HOST_LOAD_START", sample(70, 0, 0)):
                value = self.write({"passed": passed, "errors": []}, end=sample(70, 80, 1000))
            self.assertEqual(value["host_load"]["timing"], "informational")
            self.assertIs(value["passed"], passed)
            self.assertEqual(value["errors"], [])


class ComparedTimingTests(unittest.TestCase):
    def test_compared_figures_count_only_when_every_run_was_idle(self):
        idle = common.host_load([sample(1, 0, 0), sample(1, 990, 1000)], cpus=48)
        busy = common.host_load([sample(70, 0, 0), sample(70, 80, 1000)], cpus=48)
        self.assertEqual(common.compared_timing({"off": idle, "on": idle})["timing"], "evidence")
        label = common.compared_timing({"off": idle, "on": busy})
        self.assertEqual(label["timing"], "informational")
        self.assertEqual(len(label["reasons"]), 1)
        self.assertTrue(label["reasons"][0].startswith("on: 1-minute load average"))
        label = common.compared_timing({"off": idle, "on": None})
        self.assertEqual((label["timing"], label["reasons"]), ("informational", ["on: no host load recorded"]))


if __name__ == "__main__":
    unittest.main()
