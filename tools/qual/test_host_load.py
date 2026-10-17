"""CPU checks for the host load every receipt records and its timing label."""

from qual_test_config import setUpModule

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import common

SOAKS = json.loads((common.ROOT / "fixtures/host-load/soak-samples.json").read_text())


def sample(idle, total, at, cpu=0, io=0, load=1.0):
    """A host load sample `at` seconds with `idle` of `total` CPU ticks and
    `cpu` and `io` cumulative pressure stall microseconds."""
    return {"at": at, "load_average": [load, load, load], "cpu_idle_ticks": idle, "cpu_ticks": total,
            "pressure_some_us": {"cpu": cpu, "io": io}}


class HostLoadRuleTests(unittest.TestCase):
    def test_proc_is_parsed(self):
        files = {"/proc/loadavg": "1.25 0.90 0.50 2/900 1234\n",
                 "/proc/stat": "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 0 0 0 0 0\n",
                 "/proc/pressure/cpu": "some avg10=0.00 avg60=0.10 avg300=0.20 total=4567\n"
                                       "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
                 "/proc/pressure/io": "some avg10=1.00 avg60=0.50 avg300=0.20 total=89012\n"
                                      "full avg10=1.00 avg60=0.50 avg300=0.20 total=88000\n"}
        with patch.object(common.Path, "read_text", lambda path: files[str(path)]):
            row = common.host_load_sample()
        self.assertEqual(row["load_average"], [1.25, 0.90, 0.50])
        self.assertEqual((row["cpu_idle_ticks"], row["cpu_ticks"]), (850, 1000))
        self.assertEqual(row["cpus"], 1)
        self.assertEqual(row["pressure_some_us"], {"cpu": 4567, "io": 89012})
        self.assertEqual(common.host_load([row, row])["cpus"], 1)

    def test_missing_proc_gives_no_sample(self):
        with patch.object(common.Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(common.host_load_sample())

    def test_missing_pressure_stall_information_is_recorded_as_none(self):
        files = {"/proc/loadavg": "1.25 0.90 0.50 2/900 1234\n",
                 "/proc/stat": "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 0 0 0 0 0\n"}

        def read(path):
            if str(path) not in files:
                raise FileNotFoundError(path)
            return files[str(path)]

        with patch.object(common.Path, "read_text", read):
            row = common.host_load_sample()
        self.assertIsNone(row["pressure_some_us"])
        verdict = common.host_load([row, {**row, "at": row["at"] + 1, "cpu_ticks": 2000, "cpu_idle_ticks": 1850}])
        self.assertEqual(verdict["reasons"], ["no pressure stall information"])

    def test_an_idle_host_keeps_timing_as_evidence(self):
        # The 2026-10-17 spec soak's steady state: 97% idle on 48 CPUs and
        # pressure shares near 0.001.
        verdict = common.host_load([sample(0, 0, 0), sample(970, 1000, 30, cpu=21000, io=6000)], cpus=48)
        self.assertEqual((verdict["idle"], verdict["timing"], verdict["reasons"]), (True, "evidence", []))
        self.assertAlmostEqual(verdict["min_cpu_idle_fraction"], 0.97)
        self.assertAlmostEqual(verdict["max_cpu_pressure_share"], 0.0007)
        self.assertAlmostEqual(verdict["max_io_pressure_share"], 0.0002)

    def test_the_rule_bounds_are_inclusive(self):
        verdict = common.host_load([sample(0, 0, 0), sample(900, 1000, 1, cpu=100000, io=100000)], cpus=48)
        self.assertEqual(verdict["reasons"], [])

    def test_the_load_average_is_recorded_not_judged(self):
        # Work before the window (a server's startup reads) lingers in the
        # 1-minute average; only what ran between the samples counts.
        verdict = common.host_load([sample(0, 0, 0, load=11.81), sample(970, 1000, 30, load=7.62)], cpus=48)
        self.assertEqual((verdict["timing"], verdict["max_load_average_1m"]), ("evidence", 11.81))

    def test_a_cpu_busy_neighbour_makes_timing_informational(self):
        # The 2026-10-15 spec soak: 7-9% idle.
        verdict = common.host_load([sample(0, 0, 0, load=61), sample(80, 1000, 30, load=75)], cpus=48)
        self.assertEqual((verdict["idle"], verdict["timing"]), (False, "informational"))
        self.assertEqual(verdict["reasons"], ["CPU idle fell to 0.080"])
        # Work queued on a pinned set of CPUs while others idle.
        verdict = common.host_load([sample(0, 0, 0), sample(920, 1000, 30, cpu=9000000)], cpus=48)
        self.assertEqual(verdict["reasons"], ["CPU pressure stall share reached 0.300"])

    def test_an_io_busy_neighbour_makes_timing_informational(self):
        # iowait counts as idle CPU, so only the I/O pressure sees it.
        verdict = common.host_load([sample(0, 0, 0), sample(990, 1000, 30, io=18000000)], cpus=48)
        self.assertEqual((verdict["idle"], verdict["timing"]), (False, "informational"))
        self.assertEqual(verdict["reasons"], ["I/O pressure stall share reached 0.600"])

    def test_one_busy_interval_is_enough(self):
        rows = [sample(0, 0, 0), sample(990, 1000, 30), sample(1500, 2000, 60), sample(2490, 3000, 90)]
        verdict = common.host_load(rows, cpus=48)
        self.assertFalse(verdict["idle"])
        self.assertAlmostEqual(verdict["min_cpu_idle_fraction"], 0.51)
        self.assertAlmostEqual(verdict["cpu_idle_fraction"], 0.83)
        rows = [sample(0, 0, 0), sample(990, 1000, 30, io=30000), sample(1980, 2000, 60, io=6030000)]
        self.assertEqual(common.host_load(rows, cpus=48)["reasons"], ["I/O pressure stall share reached 0.200"])

    def test_unknown_load_is_not_idle(self):
        for rows in ([], [sample(0, 0, 0)], [None, sample(990, 1000, 1)], [sample(5, 10, 0), sample(5, 10, 1)],
                     [sample(0, 0, 1), sample(990, 1000, 1)],
                     [sample(0, 0, 0), {**sample(990, 1000, 1), "pressure_some_us": None}]):
            with self.subTest(rows=rows):
                verdict = common.host_load(rows, cpus=48)
                self.assertEqual((verdict["idle"], verdict["timing"]), (False, "informational"))
                self.assertTrue(verdict["reasons"])

    def test_the_rust_captures_apply_the_same_rule(self):
        source = (common.ROOT / "crates/runtime-eval/src/capture/host_load.rs").read_text()
        self.assertIn(f"pub const RULE: &str = {json.dumps(common.HOST_LOAD_RULE)};", source)
        self.assertIn(f"pub const IDLE_CPU_FRACTION: f64 = {common.IDLE_CPU_FRACTION};", source)
        self.assertIn(f"pub const IDLE_PRESSURE_SHARE: f64 = {common.IDLE_PRESSURE_SHARE};", source)


class RecordedSoakTests(unittest.TestCase):
    """fixtures/host-load/soak-samples.json; host_load.rs judges the same
    samples to the same reasons."""

    def test_a_soak_after_the_servers_startup_reads_is_evidence(self):
        # The startup left the 1-minute load average at 10.35, the reason the
        # receipt was labelled informational under the load-average rule.
        verdict = common.host_load(SOAKS["qwen38-2026-10-17"])
        self.assertEqual((verdict["timing"], verdict["reasons"]), ("evidence", []))
        self.assertEqual(verdict["max_load_average_1m"], 10.35)
        self.assertGreaterEqual(verdict["min_cpu_idle_fraction"], 0.949)
        self.assertLess(verdict["max_io_pressure_share"], 0.03)
        self.assertLess(verdict["max_cpu_pressure_share"], 0.002)

    def test_older_soaks_lose_the_load_average_reason_only(self):
        # They predate pressure stall sampling, so an I/O neighbour cannot
        # be ruled out; the Qwen soak's CPU interval at 80% idle stays.
        mimo = common.host_load(SOAKS["mimo26-2026-10-16"])
        self.assertEqual(mimo["reasons"], ["no pressure stall information"])
        self.assertEqual(mimo["max_load_average_1m"], 9.43)
        qwen = common.host_load(SOAKS["qwen38-2026-10-16"])
        self.assertEqual(qwen["reasons"], ["CPU idle fell to 0.804", "no pressure stall information"])
        self.assertEqual(qwen["max_load_average_1m"], 11.81)


class ReceiptHostLoadTests(unittest.TestCase):
    def write(self, result, samples=None, end=None):
        with patch.object(common, "host_load_sample", return_value=end):
            return json.loads(Path(common.receipt("probe", {"model": "fake"}, result,
                                                  load_samples=samples)).read_text())

    def test_every_receipt_records_host_load_from_start_to_end(self):
        with patch.object(common, "HOST_LOAD_START", sample(0, 0, 0)):
            value = self.write({"passed": True, "latency_mean_seconds": 1.0}, end=sample(950, 1000, 10))
        load = value["host_load"]
        self.assertEqual(load["rule"], common.HOST_LOAD_RULE)
        self.assertEqual(load["scope"], "check_host")
        self.assertEqual(len(load["samples"]), 2)
        self.assertEqual((load["idle"], load["timing"]), (True, "evidence"))

    def test_samples_taken_during_the_check_start_its_window(self):
        # A busy start before the timed work is outside the window.
        with patch.object(common, "HOST_LOAD_START", sample(0, 0, 0, io=0)):
            value = self.write({"passed": True}, samples=[sample(100, 1000, 10, io=9000000),
                                                          sample(1090, 2000, 20, io=9000000)],
                               end=sample(2080, 3000, 30, io=9000000))
        self.assertEqual([row["at"] for row in value["host_load"]["samples"]], [10, 20, 30])
        self.assertTrue(value["host_load"]["idle"])
        with patch.object(common, "HOST_LOAD_START", sample(0, 0, 0)):
            value = self.write({"passed": True}, samples=[sample(100, 1000, 10)], end=sample(1090, 2000, 20, io=3000000))
        self.assertFalse(value["host_load"]["idle"])

    def test_a_busy_host_leaves_the_verdict_alone(self):
        for passed in (True, False):
            with self.subTest(passed=passed), patch.object(common, "HOST_LOAD_START", sample(0, 0, 0, load=70)):
                value = self.write({"passed": passed, "errors": []}, end=sample(80, 1000, 10, load=70))
            self.assertEqual(value["host_load"]["timing"], "informational")
            self.assertIs(value["passed"], passed)
            self.assertEqual(value["errors"], [])


class ComparedTimingTests(unittest.TestCase):
    def test_compared_figures_count_only_when_every_run_was_idle(self):
        idle = common.host_load([sample(0, 0, 0), sample(990, 1000, 10)], cpus=48)
        busy = common.host_load([sample(0, 0, 0), sample(80, 1000, 10)], cpus=48)
        self.assertEqual(common.compared_timing({"off": idle, "on": idle})["timing"], "evidence")
        label = common.compared_timing({"off": idle, "on": busy})
        self.assertEqual(label["timing"], "informational")
        self.assertEqual(label["reasons"], ["on: CPU idle fell to 0.080"])
        label = common.compared_timing({"off": idle, "on": None})
        self.assertEqual((label["timing"], label["reasons"]), ("informational", ["on: no host load recorded"]))


if __name__ == "__main__":
    unittest.main()
