"""CPU checks for adaptive benchmark cohorts and fixed request bodies."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


QUAL = Path(__file__).resolve().parents[1] / "tools" / "qual"
sys.path.insert(0, str(QUAL))
from qual_test_config import setUpModule

SPEC = importlib.util.spec_from_file_location("spec_bench", QUAL / "spec_bench.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
import common  # noqa: E402


class SpecBenchTests(unittest.TestCase):
    def run_benchmark(self, mode, verified, *, ignore_eos=False, short_response=False, server_log=None):
        counters = [{"available": True, "verify_items": value, "draft_tokens": value, "accepted_tokens": value} for value in verified]
        with patch.object(sys, "argv", [
            "spec_bench.py", "--arm", "chopinfer-spec", "--run-id", "cpu",
            "--speculation", mode, "--concurrencies", "1", "4", "16", "32",
            *(["--ignore-eos", "--output-tokens", "512"] if ignore_eos else []),
            *(["--server-log", str(server_log)] if server_log else []),
        ]), patch.object(BENCH, "gsm8k_questions", return_value=(
            {"instruction": "Solve.", "examples_sha256": "fixture"},
            [f"Question {index}" for index in range(32)],
        )) as questions, patch.object(BENCH, "Client") as client, patch.object(
            BENCH, "metrics_text", return_value=""
        ), patch.object(BENCH, "snapshot", return_value={}), patch.object(
            BENCH, "difference", side_effect=counters
        ), patch.object(BENCH, "chat_events", return_value=(
            0.0, {"completion_tokens": 336 if short_response else 512 if ignore_eos else 64,
                  "prompt_tokens": 10}, "stop" if short_response else "length"
        )), patch.object(BENCH, "receipt") as receipt, patch.object(
            BENCH, "trace_window", return_value={"captured": True}
        ) as trace:
            client.return_value.identity.return_value = {"model": "fixture"}
            client.return_value.request.return_value = ([], 1.0)
            with self.assertRaises(SystemExit) as result:
                BENCH.main()
            questions.assert_called_once_with(32)
            if ignore_eos:
                for call in client.return_value.request.call_args_list:
                    self.assertIs(call.args[1]["ignore_eos"], True)
                    self.assertEqual(call.args[1]["max_tokens"], 512)
            self.assertEqual(trace.call_count, 8 if server_log else 0)
            return result.exception.code, receipt.call_args.args[2]

    def test_workload_does_not_inherit_filtered_api_defaults(self):
        body = BENCH.workload("fixture", "cpu", "chat", 1, 0, "Question", "Answer", 64)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["top_p"], 1)

    def test_ignore_eos_reaches_requests_and_the_receipt(self):
        code, receipt = self.run_benchmark("fixed", [1] * 8, ignore_eos=True)
        self.assertEqual(code, 0)
        self.assertTrue(receipt["sampling"]["ignore_eos"])
        self.assertEqual(receipt["output_tokens"], 512)

    def test_ignore_eos_and_step_capture_work_together(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.touch()
            code, receipt = self.run_benchmark("adaptive", [1] * 8,
                                               ignore_eos=True, server_log=log)
        self.assertEqual(code, 0)
        self.assertTrue(receipt["sampling"]["ignore_eos"])
        for cohort in receipt["cohorts"]:
            self.assertEqual(cohort["step_trace"], {"captured": True})

    def test_ignore_eos_still_rejects_short_stopped_responses(self):
        code, receipt = self.run_benchmark("fixed", [1] * 8, ignore_eos=True,
                                           short_response=True)
        self.assertEqual(code, 1)
        self.assertFalse(receipt["passed"])
        self.assertTrue(any("336 of 512 tokens, finish_reason=stop" in e["error"]
                            for e in receipt["errors"]))

    def test_adaptive_allows_ar_fallback_at_high_concurrency(self):
        code, receipt = self.run_benchmark("adaptive", [1, 0, 0, 0] * 2)
        self.assertEqual(code, 0)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["concurrencies"], [1, 4, 16, 32])
        for cohort in receipt["cohorts"]:
            self.assertEqual(len(cohort["requests"]), cohort["concurrency"])
        self.assertEqual(receipt["speculation_mode"], "adaptive")

    def test_an_entirely_inactive_drafter_is_not_a_speculation_measurement(self):
        code, receipt = self.run_benchmark("adaptive", [0] * 8)
        self.assertEqual(code, 1)
        self.assertFalse(receipt["passed"])

    def test_fixed_mode_still_requires_verification_in_each_cohort(self):
        code, receipt = self.run_benchmark("fixed", [1, 0, 0, 0] * 2)
        self.assertEqual(code, 1)
        self.assertFalse(receipt["passed"])


class FakeHost:
    """CPU ticks and uptime that advance only with the work the benchmark
    does, sampled as common.host_load_sample reports them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.uptime, self.idle, self.total = 0.0, 0, 0

    def advance(self, seconds, idle_fraction, ticks=None):
        """`ticks` CPU ticks (default 100 per second) over `seconds` of
        uptime, which can be 0: /proc/uptime moves in 10 ms steps."""
        with self.lock:
            ticks = round(seconds * 100) if ticks is None else ticks
            self.uptime += seconds
            self.idle += round(ticks * idle_fraction)
            self.total += ticks

    def sample(self):
        with self.lock:
            return {"at": self.uptime, "uptime_seconds": self.uptime, "load_average": [1.0, 1.0, 1.0],
                    "cpu_idle_ticks": self.idle, "cpu_ticks": self.total, "cpus": 48,
                    "pressure_some_us": {"cpu": 0, "io": 0}}


class HostLoadTests(unittest.TestCase):
    """The receipt's host load judges each cohort's window alone."""

    def run_benchmark(self, request_idle, events=None, scrape=(0.1, 1.0, None)):
        """A four-cohort run (C1 and C4 per category) after 1,200 s of quiet
        start-up, each request taking 1 s at `request_idle` CPU idle (a
        number, or a list with one per cohort) and each metrics scrape
        advancing the host by `scrape` (seconds, idle fraction, ticks);
        returns the written receipt and the host's first sample."""
        host = FakeHost()
        first = host.sample()
        events = [] if events is None else events
        idle = request_idle if isinstance(request_idle, list) else [request_idle] * 4

        def identity(model):
            host.advance(1200, 0.99)
            return {"model": "fixture"}

        def request(path, body, stream):
            events.append("request")
            host.advance(1, idle[events.count("sample") // 2])
            return [], 1.0

        def metrics(client):
            events.append("metrics")
            host.advance(*scrape)
            return ""

        def sample():
            events.append("sample")
            return host.sample()

        directory = Path(os.environ["CHOPINFER_QUAL_RECEIPTS"])
        before = set(directory.iterdir())
        with patch.object(sys, "argv", ["spec_bench.py", "--arm", "chopinfer-ar", "--run-id", "cpu",
                                        "--concurrencies", "1", "4"]), \
                patch.object(BENCH, "gsm8k_questions", return_value=(
                    {"instruction": "Solve.", "examples_sha256": "fixture"}, [f"Question {i}" for i in range(4)])), \
                patch.object(BENCH, "Client") as client, patch.object(BENCH, "metrics_text", metrics), \
                patch.object(BENCH, "snapshot", return_value={}), \
                patch.object(BENCH, "difference", side_effect=lambda before, after: {"available": True}), \
                patch.object(BENCH, "chat_events", return_value=(
                    0.0, {"completion_tokens": 64, "prompt_tokens": 10}, "length")), \
                patch.object(BENCH, "host_load_sample", sample):
            client.return_value.identity.side_effect = identity
            client.return_value.request.side_effect = request
            with self.assertRaises(SystemExit) as result:
                BENCH.main()
        self.assertEqual(result.exception.code, 0)
        written, = set(directory.iterdir()) - before
        return json.loads(written.read_text()), first

    def test_samples_bound_each_cohort_window(self):
        events = []
        value, _ = self.run_benchmark(0.99, events)
        # Each cohort: the scrape before it, its window, the scrape after.
        cohort = lambda width: ["metrics", "sample", *["request"] * width, "sample", "metrics"]
        self.assertEqual(events, cohort(1) + cohort(4) + cohort(1) + cohort(4))
        load = value["host_load"]
        self.assertEqual(load["scope"], "check_host_windows")
        self.assertEqual([len(window["samples"]) for window in load["windows"]], [2] * 4)
        self.assertEqual([window["samples"][1]["at"] - window["samples"][0]["at"] for window in load["windows"]],
                         [1, 4, 1, 4])
        self.assertEqual((load["timing"], load["reasons"]), ("evidence", []))

    def test_busy_cohorts_after_a_quiet_start_read_informational(self):
        value, first = self.run_benchmark(0.7)
        load = value["host_load"]
        self.assertEqual((load["idle"], load["timing"]), (False, "informational"))
        self.assertEqual(load["reasons"], [f"window {index}: CPU idle fell to 0.700" for index in range(1, 5)])
        # A window from the start would have averaged the quiet start-up in.
        self.assertTrue(common.host_load([first, load["windows"][-1]["samples"][-1]])["idle"])
        self.assertTrue(value["passed"])

    def test_one_busy_cohort_makes_the_run_informational(self):
        value, _ = self.run_benchmark([0.99, 0.99, 0.5, 0.99])
        load = value["host_load"]
        self.assertEqual((load["timing"], load["reasons"]), ("informational", ["window 3: CPU idle fell to 0.500"]))
        self.assertEqual([window["timing"] for window in load["windows"]],
                         ["evidence", "evidence", "informational", "evidence"])

    def test_busy_scrapes_between_quiet_cohorts_are_not_judged(self):
        value, _ = self.run_benchmark(0.99, scrape=(0.1, 0.1, None))
        self.assertEqual((value["host_load"]["timing"], value["host_load"]["reasons"]), ("evidence", []))

    def test_scrapes_within_one_uptime_step_are_not_judged(self):
        # CPU ticks advance between cohorts while /proc/uptime reads the same.
        value, _ = self.run_benchmark(0.99, scrape=(0, 1.0, 10))
        self.assertEqual((value["host_load"]["timing"], value["host_load"]["reasons"]), ("evidence", []))

if __name__ == "__main__":
    unittest.main()
