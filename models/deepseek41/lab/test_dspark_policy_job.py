#!/usr/bin/env python3
"""CPU checks of the DSpark comparison job's parsers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dspark_policy_job as job


def record(request, main_rows, ids, bits=(0, 1, 2, 3, 4), completion="committed"):
    return {"schema": "chopinfer.deepseek41.drafter-confidence.v1", "request_id": str(request),
            "main_rows": main_rows, "completion": completion, "draft_ids": list(ids),
            "confidence_f32_bits": list(bits)}


class Server:
    def __init__(self, directory):
        self.directory = Path(directory)


class Parsers(unittest.TestCase):
    def test_blocks_map_consecutive_request_ids_to_admission_order(self):
        rows = [record(41, 10, [1] * 5), record(42, 12, [2] * 5), record(41, 14, [3] * 5),
                record(42, 16, [4] * 5, completion="cancelled")]
        self.assertEqual(job.blocks(rows, 2), {(0, 10): ([1] * 5, [0, 1, 2, 3, 4]),
                                               (1, 12): ([2] * 5, [0, 1, 2, 3, 4]),
                                               (0, 14): ([3] * 5, [0, 1, 2, 3, 4])})
        # A missing or extra request makes the mapping unknown, not wrong.
        self.assertIsNone(job.blocks(rows[:1], 2))
        self.assertIsNone(job.blocks([record(41, 10, [1] * 5), record(43, 10, [1] * 5)], 2))
        with self.assertRaises(ValueError):
            job.blocks([record(41, 10, [1] * 5), record(41, 10, [2] * 5)], 1)

    def test_block_comparison_reports_common_differences_and_coverage(self):
        base = job.blocks([record(7, 10, [1] * 5), record(7, 14, [2] * 5)], 1)
        same = job.blocks([record(9, 10, [1] * 5), record(9, 12, [5] * 5)], 1)
        bits = job.blocks([record(9, 10, [1] * 5, bits=(0, 1, 2, 3, 5))], 1)
        self.assertEqual(job.compare_blocks(base, same)["differences"], [])
        self.assertEqual(job.compare_blocks(base, same)["compared"], 1)
        self.assertEqual(job.compare_blocks(base, bits)["differences"],
                         [{"request": 0, "main_rows": 10}])
        self.assertFalse(job.compare_blocks(None, same)["conclusive"])
        self.assertFalse(job.compare_blocks(base, job.blocks([record(9, 11, [1] * 5)], 1))["conclusive"])

    def test_step_costs_group_decode_steps_by_items_drafting_and_rows(self):
        rows = [{"prefill_items": 0, "prefill_tokens": 0, "decode_rows": 3, "verify_items": 5,
                 "verify_rows": 29, "execution_us": us} for us in (70000, 71000, 69000)]
        rows += [{"prefill_items": 1, "prefill_tokens": 64, "decode_rows": 8, "verify_items": 0,
                  "verify_rows": 0, "execution_us": 90000},
                 {"prefill_items": 0, "prefill_tokens": 0, "decode_rows": 8, "verify_items": 0,
                  "verify_rows": 0, "execution_us": 23000}]
        self.assertEqual(job.step_costs(rows), [
            {"items": 8, "drafting_items": 0, "rows": 8, "steps": 1, "median_us": 23000},
            {"items": 8, "drafting_items": 5, "rows": 32, "steps": 3, "median_us": 70000}])

    def test_phase_costs_take_medians_by_items_and_drafting(self):
        def row(drafting, draft):
            return {"schema": job.PHASES, "items": 8, "drafting": drafting, "catch_up_rows": 4,
                    "micros": {"catch_up": 900, "draft": draft, "target": 50}}
        costs = job.phase_costs([row(5, 3000), row(5, 5000), row(5, 4000), row(0, 0)])
        self.assertEqual([(c["drafting_items"], c["steps"]) for c in costs], [(0, 1), (5, 3)])
        self.assertEqual(costs[1]["median_us"], {"catch_up": 900, "draft": 4000, "target": 50})

    def test_server_args_replace_flags_the_runner_sets_and_append_the_rest(self):
        command = ["chopinfer", "serve", "--speculative-tokens", "5", "--speculation", "adaptive",
                   "--draft-cost", "per-block"]
        costs = "1:13500:4400:700:760,8:22300:4400:700:780"
        self.assertEqual(
            job.server_args(command, ["--speculation", "table", "--speculation-costs", costs]),
            ["chopinfer", "serve", "--speculative-tokens", "5", "--speculation", "table",
             "--draft-cost", "per-block", "--speculation-costs", costs])
        self.assertEqual(command[5], "adaptive")
        for bad in (["--speculation"], ["table", "--speculation"]):
            with self.assertRaises(ValueError):
                job.server_args(command, bad)
        with self.assertRaises(ValueError):
            job.server_args(command + ["--speculation", "fixed"], ["--speculation", "table"])

    def test_log_lines_read_only_the_phase_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            server = Server(directory)
            path = server.directory / "server.log"
            first = "bundle: " + json.dumps(record(1, 2, [3] * 5)) + "\n"
            second = "bundle: " + json.dumps(record(2, 4, [5] * 5)) + "\nnoise {\n"
            path.write_text(first + second)
            start = len(first.encode())
            rows = job.log_lines(server, start, job.log_offset(server), job.RECORD)
            self.assertEqual([row["request_id"] for row in rows], ["2"])


if __name__ == "__main__":
    unittest.main()
