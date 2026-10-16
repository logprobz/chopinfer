"""CPU checks for adaptive benchmark cohorts and fixed request bodies."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


QUAL = Path(__file__).resolve().parents[1] / "tools" / "qual"
sys.path.insert(0, str(QUAL))
from qual_test_config import setUpModule

SPEC = importlib.util.spec_from_file_location("spec_bench", QUAL / "spec_bench.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


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


if __name__ == "__main__":
    unittest.main()
