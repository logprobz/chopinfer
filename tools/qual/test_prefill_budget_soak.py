"""CPU mechanics for the assigned-slot soak runner."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dev import prefill_budget_soak as soak


class SoakTests(unittest.TestCase):
    def test_deadline_refuses_launch_without_full_slot(self):
        args = argparse.Namespace(end_utc=datetime.now(timezone.utc).isoformat())
        with patch.object(soak.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "slot needs"):
                soak.run(args)
            launch.assert_not_called()

    def test_later_cutoff_cannot_extend_twenty_minute_limit(self):
        self.assertEqual(soak.slot_seconds("2026-09-26T23:00:00Z", 0), 1200)
        self.assertEqual(soak.ARM_RESERVE_SECONDS, soak.ARM_SECONDS + soak.WARMUP_SECONDS +
                         soak.STARTUP_SECONDS + soak.WARMUP_DRAIN_SECONDS + soak.DRAIN_SECONDS)
        self.assertEqual(soak.REQUIRED_ARMS, 2)
        self.assertLessEqual(soak.REQUIRED_ARMS * soak.ARM_RESERVE_SECONDS + 30, 1200)

    def test_optional_arm_starts_only_with_its_whole_allowance(self):
        self.assertEqual([name for name, _, required in soak.ARMS if required],
                         ["fixed", "decode-aware"])
        self.assertTrue(soak.arm_fits(True, 0))
        self.assertTrue(soak.arm_fits(False, soak.ARM_RESERVE_SECONDS))
        self.assertFalse(soak.arm_fits(False, soak.ARM_RESERVE_SECONDS - 1))
        # About 5.5 minutes per warm arm leaves room for the third arm.
        self.assertTrue(soak.arm_fits(False, 1200 - 2 * 330))

    def test_owned_process_is_killed_and_collected(self):
        process = Mock()
        process.poll.return_value = None
        soak.stop(process)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with()

    def test_server_limits_and_policy_are_enforced(self):
        base = ["/data/bin/chopinfer", "serve"]
        command = soak.server_command(base, "decode-aware")
        self.assertEqual(command[-8:], ["--max-sequences", "32", "--queue-depth", "1024",
                                      "--admission", "optimistic", "--prefill-budget", "decode-aware"])
        self.assertEqual(base, ["/data/bin/chopinfer", "serve"])
        for flag in soak.PROTECTED:
            with self.assertRaises(ValueError):
                soak.server_command(base + [flag + "=32"], "fixed")
        self.assertEqual(soak.server_command(base, "fixed")[-2:], ["--prefill-budget", "fixed"])
        self.assertEqual(soak.server_command(base, "decode-aware-ratio1")[-4:],
                         ["--prefill-budget", "decode-aware", "--prefill-decode-steps", "1"])
        self.assertIn("--prefill-itl-target-ms", soak.PROTECTED)
        self.assertIn("--prefill-decode-steps", soak.PROTECTED)
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            soak.server_command(base, "adaptive")

    def test_step_summary_reads_the_measured_window_of_the_trace(self):
        def step(prefill, decode, cap, execution, predicted, trained=True):
            return json.dumps({"batch": 1, "decode_estimate_us": 16400, "decode_rows": decode,
                               "execution_us": execution, "predicted_us": predicted,
                               "prefill_cap": cap, "prefill_items": int(prefill > 0),
                               "prefill_tokens": prefill, "schema": "chopinfer.step.v1",
                               "trained": trained, "verify_items": 0}, separators=(",", ":"))
        before = step(8192, 0, None, 500000, None) + "\nstartup text\n"
        window = "\n".join([step(0, 32, 512, 16000, 16400), step(2048, 31, 2048, 176000, 160000),
                            step(2048, 31, 2048, 180000, 160000), step(1033, 31, 2048, 110000, None),
                            "not json {\"schema\":\"chopinfer.step.v1\"", "other log"]) + "\n"
        after = step(0, 1, 512, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(before + window + after)
            summary = soak.step_summary(path, len(before), len(before) + len(window))
            everything = soak.step_summary(path)
        self.assertEqual(summary["steps"], 4)
        self.assertEqual(everything["steps"], 6)
        self.assertEqual(summary["execution"]["mixed"]["steps"], 3)
        self.assertEqual(summary["execution"]["mixed"]["p99_ms"], 180)
        self.assertEqual(summary["execution"]["decode"]["p50_ms"], 16)
        self.assertEqual(summary["mixed_share"], .75)
        self.assertEqual(summary["selected_caps"], {512: 1, 2048: 3})
        self.assertEqual(summary["prefill_tokens_per_prefill_step"], (2048 * 2 + 1033) / 3)
        self.assertEqual(summary["actual_over_predicted"]["steps"], 3)
        self.assertEqual(summary["actual_over_predicted"]["p50"], 1.1)

    def test_gates_bound_itl_and_throughput_loss(self):
        fixed = {"tokens_per_second": 800, "itl_p99_ms": 428}
        self.assertEqual(soak.gates(fixed, {"tokens_per_second": 760, "itl_p99_ms": 180}),
                         {"itl_target_ms": 200, "itl_p99_within_target": True,
                          "throughput_loss_percent": 5.000000000000004,
                          "throughput_within_5_percent": False,
                          "throughput_within_10_percent": True, "qualified": True})
        self.assertFalse(soak.gates(fixed, {"tokens_per_second": 700, "itl_p99_ms": 180})["qualified"])
        self.assertFalse(soak.gates(fixed, {"tokens_per_second": 800, "itl_p99_ms": 201})["qualified"])

    def test_workload_is_reproducible_and_varied(self):
        rows = [soak.request_body("model", i, 0) for i in range(6)]
        self.assertEqual(rows, [soak.request_body("model", i, 0) for i in range(6)])
        self.assertEqual(len({row["max_tokens"] for row in rows}), 3)
        self.assertEqual(len({row["temperature"] for row in rows}), 2)

    def test_queued_request_timeout_is_independent_of_short_warmup_drain(self):
        events = [(.5, {"usage": {"completion_tokens": 1},
                         "choices": [{"text": "x", "finish_reason": "stop"}]})]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(soak, "Client") as client, \
                    patch.object(soak, "ThreadPoolExecutor") as executor, \
                    patch.object(soak.time, "monotonic", side_effect=[0, 0, 0, 2]):
                client.return_value.request.return_value = (events, .5)
                executor.return_value.__enter__.return_value.map.side_effect = (
                    lambda worker, indexes: [worker(0)])
                soak.measure("http://localhost:8347", "model", Path(directory) / "rows", duration=1)
                client.assert_called_once_with("http://localhost:8347", timeout=120, retries=0)

    def test_warmup_uses_one_short_request_stream(self):
        events = [(.5, {"usage": {"completion_tokens": 1},
                         "choices": [{"text": "x", "finish_reason": "stop"}]})]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(soak, "Client") as client, \
                    patch.object(soak, "ThreadPoolExecutor") as executor, \
                    patch.object(soak.time, "monotonic", side_effect=[0, 0, 0, 2]):
                client.return_value.request.return_value = (events, .5)
                executor.return_value.__enter__.return_value.map.side_effect = (
                    lambda worker, indexes: [worker(i) for i in indexes])
                soak.measure("http://localhost:8347", "model", Path(directory) / "rows",
                             duration=1, warmup=True)
                executor.assert_called_once_with(max_workers=1)
                body = client.return_value.request.call_args.args[1]
                self.assertEqual(body["max_tokens"], 32)
                self.assertLess(len(body["prompt"]), 100)

    def test_metrics_exclude_drain_tokens_and_include_errors(self):
        rows = [{"ttft_ms": 50, "tokens": [[1, 2], [2, 1], [11, 5]]},
                {"error": "HTTP 429"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            stats = soak.summarize(path, duration=10)
        self.assertEqual(stats["tokens_per_second"], .3)
        self.assertEqual(stats["itl_p99_ms"], 1000)
        self.assertEqual(stats["ttft_p99_ms"], 50)
        self.assertEqual(len(stats["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
