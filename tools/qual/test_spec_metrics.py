"""CPU-only parser and comparison checks for speculative receipts."""

from pathlib import Path
from qual_test_config import setUpModule

import tempfile
import unittest

import common
from spec_bench import chat_events, workload
from spec_metrics import difference, snapshot, within_output
from dev.spec_summary import ARMS, compare

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "speculative receipt parsing and comparison"


CHOP = """# TYPE chop_spec_verify_items_total counter
chop_spec_verify_items_total 2
chop_spec_draft_tokens_total 4
chop_spec_accepted_tokens_total 3
chop_spec_emitted_tokens_total 5
chop_spec_rows_total 6
chop_spec_drafted_position_total{position="1"} 2
chop_spec_drafted_position_total{position="2"} 2
chop_spec_accepted_position_total{position="1"} 2
chop_spec_accepted_position_total{position="2"} 1
"""


class SpecQualificationTests(unittest.TestCase):
    def test_chopinfer_counters_start_absent_then_report_exact_deltas(self):
        before = snapshot("chop_requests_total 4\n", "chopinfer")
        after = snapshot(CHOP, "chopinfer")
        counters = difference(before, after)
        self.assertFalse(before["available"])
        self.assertEqual(counters["verify_items"], 2)
        self.assertEqual(counters["acceptance_rate"], 0.75)
        self.assertEqual(counters["tokens_per_step"], 2.5)
        self.assertEqual(counters["emitted_tokens_per_step"], 2.5)
        self.assertEqual([row["rate"] for row in counters["per_position"]], [1.0, 0.5])








    def test_resets_and_inconsistent_positions_are_refused(self):
        after = snapshot(CHOP, "chopinfer")
        with self.assertRaisesRegex(ValueError, "reset"):
            difference(after, snapshot(CHOP.replace("verify_items_total 2", "verify_items_total 1"),
                                       "chopinfer"))
        with self.assertRaisesRegex(ValueError, "disagree"):
            difference(snapshot("", "chopinfer"), snapshot(
                CHOP.replace('accepted_position_total{position="2"} 1',
                             'accepted_position_total{position="2"} 0'), "chopinfer"))

    def test_step_rows_keep_ar_and_subtract_warmup(self):
        def scrape(ar, verify, unknown):
            return CHOP + f'''chop_step_rows_dropped_total{{model_name="M"}} 0
chop_step_rows_total{{model_name="M",decode_rows="16",verify_rows="0",row_capacity="16"}} {ar}
chop_step_rows_total{{model_name="M",decode_rows="2",verify_rows="64",row_capacity="128"}} {verify}
chop_step_rows_total{{model_name="M",decode_rows="1",verify_rows="3",row_capacity="unknown"}} {unknown}
chop_step_rows_total{{model_name="other",decode_rows="0",verify_rows="48",row_capacity="64"}} 900
'''
        before = snapshot(scrape(8, 4, 0), "chopinfer", "M")
        after = snapshot(scrape(11, 6, 1), "chopinfer", "M")
        rows = difference(before, after)["step_rows"]
        self.assertEqual((rows["steps"], rows["verify_steps"], rows["unknown_capacity_steps"]), (6, 3, 1))
        eligible = sum(row["steps"] for row in rows["bins"]
                       if 65 <= row["decode_rows"] + row["verify_rows"] <= 96)
        self.assertEqual(eligible, 2)
        self.assertEqual(len(rows["bins"]), 3)
        self.assertTrue(any(row["row_capacity"] is None for row in rows["bins"]))
        self.assertFalse(snapshot(CHOP, "chopinfer")["step_rows"]["available"])

    def test_step_rows_reject_resets_loss_and_invalid_geometry(self):
        header = 'chop_step_rows_dropped_total 0\n'
        series = 'chop_step_rows_total{decode_rows="0",verify_rows="48",row_capacity="64"} 2\n'
        before = snapshot(CHOP + header + series, "chopinfer")
        for bad in [header, header + series.replace('} 2', '} 1'),
                    header.replace('total 0', 'total 1') + series, '']:
            with self.assertRaisesRegex(ValueError, "reset|disappeared|dropped"):
                difference(before, snapshot(CHOP + bad, "chopinfer"))
        for bad in [series, header + series + series,
                    header + series.replace('row_capacity="64"', 'row_capacity="32"'),
                    header + series.replace('verify_rows="48",', '')]:
            with self.assertRaises(ValueError):
                snapshot(CHOP + bad, "chopinfer")

    def test_workload_is_arm_independent_and_stream_reads_reasoning(self):
        first = workload("M", "run-1", "gsm8k", 4, 0, "How much?", "Solve.", 64)
        self.assertEqual(first, workload("M", "run-1", "gsm8k", 4, 0,
                                         "How much?", "Solve.", 64))
        self.assertNotEqual(first["messages"][0]["content"], workload(
            "M", "run-1", "gsm8k", 4, 1, "How much?", "Solve.", 64)["messages"][0]["content"])
        at, usage, finish = chat_events([
            (1.0, {"choices": [{"delta": {"reasoning_content": "step"}}]}),
            (2.0, {"choices": [{"finish_reason": "length"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 64}})])
        self.assertEqual((at, usage["completion_tokens"], finish), (1.0, 64, "length"))

    def test_comparison_requires_identical_workloads_and_writes_positions(self):
        def make(arm):
            groups = []
            for category in ("gsm8k", "chat"):
                for concurrency in (1, 4, 16):
                    groups.append({"category": category, "concurrency": concurrency,
                                   "aggregate_tokens_per_second": 100.0,
                                   "mean_ms_per_token": 10.0,
                                   "requests": [{"completion_tokens": 64,
                                                 "finish_reason": "length"} for _ in range(concurrency)],
                                   "speculation": {"available": arm != "chopinfer-ar",
                                                   "acceptance_rate": 0.75,
                                                   "tokens_per_step": 2.5,
                                                   "drafted_positions": {"1": 2},
                                                   "accepted_positions": {"1": 2}}})
            return {"passed": True, "arm": arm, "label": "exclusive", "run_id": "run-1",
                    "workload_sha256": "abc", "output_tokens": 64,
                    "concurrencies": [1, 4, 16], "server": {"url": "http://localhost"},
                    "cohorts": groups}
        receipts = {arm: (Path(f"{arm}.json"), make(arm)) for arm in ARMS}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "table.md"
            compare(receipts, output)
            text = output.read_text()
            self.assertIn("| gsm8k | 16 | accepted / drafted |", text)
            self.assertIn("| 1 | 1.000 |", text)
            self.assertIn("timing: informational (chopinfer-ar: no host load recorded; "
                          "chopinfer-spec: no host load recorded).", text)
            receipts["chopinfer-spec"][1]["workload_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "workloads differ"):
                compare(receipts, output)

    def test_comparison_labels_timing_by_both_arms_host_load(self):
        stall = {"cpu": 0, "io": 0}
        rows = lambda idle: [{"at": 0, "uptime_seconds": 0, "load_average": [1, 1, 1], "cpu_idle_ticks": 0,
                              "cpu_ticks": 0, "pressure_some_us": stall},
                             {"at": 1, "uptime_seconds": 1, "load_average": [1, 1, 1], "cpu_idle_ticks": idle,
                              "cpu_ticks": 1000, "pressure_some_us": stall}]
        idle, busy = common.host_load(rows(990), cpus=48), common.host_load(rows(80), cpus=48)

        def make(arm, load):
            groups = [{"category": category, "concurrency": 1, "aggregate_tokens_per_second": 100.0,
                       "mean_ms_per_token": 10.0, "requests": [{"completion_tokens": 64, "finish_reason": "length"}],
                       "speculation": {"available": True, "drafted_positions": {}, "accepted_positions": {}}}
                      for category in ("gsm8k", "chat")]
            return {"passed": True, "arm": arm, "label": "exclusive", "run_id": "run-1", "workload_sha256": "abc",
                    "output_tokens": 64, "concurrencies": [1], "server": {"url": "http://localhost"},
                    "cohorts": groups, "host_load": load}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "table.md"
            compare({arm: (Path(f"{arm}.json"), make(arm, idle)) for arm in ARMS}, output)
            self.assertIn("Throughput and latency timing: evidence.", output.read_text())
            compare({"chopinfer-ar": (Path("ar.json"), make("chopinfer-ar", idle)),
                     "chopinfer-spec": (Path("spec.json"), make("chopinfer-spec", busy))}, output)
            self.assertIn("timing: informational (chopinfer-spec: CPU idle fell to 0.080).", output.read_text())


if __name__ == "__main__":
    unittest.main()
