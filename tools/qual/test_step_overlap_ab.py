from qual_test_config import configure_receipts as setUpModule

import json
import unittest
import unittest.mock

import step_overlap_ab as ab


def step(rows, prefill=0, boundary=100, **host):
    fields = {field: 1 for field in ab.HOST_FIELDS}
    fields.update(boundary_us=boundary, **host)
    return {"schema": "chopinfer.step.v1", "prefill_tokens": prefill, "decode_rows": rows,
            "verify_items": 0, "host": fields}


def request(index, ids, **changes):
    row = {"index": index, "token_ids": ids, "logprobs": [-0.1] * len(ids), "text": "x" * len(ids),
           "finish_reason": "length", "usage": {"completion_tokens": len(ids)}, "done": True,
           "error": None, "complete": True}
    row.update(changes)
    return row


def arm(name, passes):
    return {"arm": name, "passes": [
        {"concurrency": concurrency, "ms_per_token": ms, "cohort": None if concurrency == 1 else
         {"tokens_per_second": 1000.0 * concurrency / ms},
         "host": {"steady": {field: 100 if field != "deferred_us" else 0 for field in ab.HOST_FIELDS}},
         "requests": [request(index, ids) for index, ids in enumerate(outputs)]}
        for concurrency, ms, outputs in passes]}


def pair():
    off = arm("off", [(1, 5.0, [[1, 2], [3]]), (16, 10.0, [[4], [5]]), (16, 10.0, [[4], [6]])])
    on = arm("on", [(1, 4.9, [[1, 2], [3]]), (16, 9.5, [[4], [7]]), (16, 9.6, [[4], [5]])])
    return off, on


class StepOverlapAbTest(unittest.TestCase):
    def test_step_lines_read_the_trace_among_other_log_lines(self):
        compact = {"separators": (",", ":")}
        text = "\n".join(["listening", json.dumps(step(16), **compact),
                          "not json {\"schema\":\"chopinfer.step.v1\"",
                          "prefix " + json.dumps(step(16, boundary=300), **compact)])
        steps = ab.step_lines(text)
        self.assertEqual([row["host"]["boundary_us"] for row in steps], [100, 300])

    def test_only_full_decode_steps_count_as_steady(self):
        steps = [step(16, boundary=100), step(16, boundary=300), step(15, boundary=900),
                 step(16, prefill=512, boundary=900), step(16, boundary=200)]
        summary = ab.summarize_steps(steps, 16)
        self.assertEqual(summary["steady"]["steps"], 3)
        self.assertEqual(summary["steady"]["boundary_us"], 200)
        self.assertEqual(summary["all"]["steps"], 5)

    def test_c1_outputs_must_match_and_batched_ones_are_reported(self):
        off, on = pair()
        result = ab.compare_arms(off, on)
        self.assertTrue(result["passed"])
        batched = result["rows"][1]
        self.assertEqual((batched["equal"], batched["compared"]), (1, 2))
        self.assertEqual(batched["off_self_equal"], (1, 2))
        self.assertFalse(batched["below_off_repeat"])
        self.assertIn("| 16 |", ab.table(result))
        on["passes"][0]["requests"][1]["token_ids"] = [9]
        self.assertFalse(ab.compare_arms(off, on)["passed"])

    def test_host_load_labels_the_timing_and_leaves_the_verdict_alone(self):
        off, on = pair()
        result = ab.compare_arms(off, on)
        # Arms without a recorded host load (older runs) are not timing evidence.
        self.assertEqual(result["host_load"]["timing"], "informational")
        self.assertIn("timing: informational", ab.table(result))
        idle = {"idle": True, "reasons": []}
        off["host_load"] = on["host_load"] = idle
        self.assertEqual(ab.compare_arms(off, on)["host_load"]["timing"], "evidence")
        on["host_load"] = {"idle": False, "reasons": ["CPU idle fell to 0.080"]}
        result = ab.compare_arms(off, on)
        self.assertEqual((result["passed"], result["host_load"]["reasons"]),
                         (True, ["on: CPU idle fell to 0.080"]))

    def test_empty_outputs_are_no_evidence(self):
        off, on = pair()
        for row in (off, on):
            for item in row["passes"][0]["requests"]:
                item["token_ids"] = []
        self.assertFalse(ab.compare_arms(off, on)["passed"])

    def test_finish_usage_and_text_must_match_at_c1(self):
        for key, value in (("finish_reason", "stop"), ("usage", {"completion_tokens": 999}),
                           ("text", "different")):
            off, on = pair()
            on["passes"][0]["requests"][0][key] = value
            self.assertFalse(ab.compare_arms(off, on)["passed"], key)

    def test_an_incomplete_stream_anywhere_fails_the_pair(self):
        for changes in ({"done": False, "complete": False}, {"error": "boom", "complete": False}):
            off, on = pair()
            on["passes"][2]["requests"][1].update(changes)
            result = ab.compare_arms(off, on)
            self.assertFalse(result["passed"])
            self.assertEqual(result["incomplete"]["on"], 1)

    def test_both_arms_must_run_the_same_nonempty_workload(self):
        cases = {
            "the on arm has no C1 pass": lambda off, on: on["passes"].pop(0),
            "an empty C1 pass": lambda off, on: on["passes"][0].update(requests=[]),
            "empty C1 in both arms": lambda off, on: [row["passes"][0].update(requests=[])
                                                      for row in (off, on)],
            "one fewer batched pass": lambda off, on: on["passes"].pop(2),
            "another request": lambda off, on: on["passes"][1]["requests"][1].update(index=9),
            "a duplicate request": lambda off, on: on["passes"][1]["requests"].append(
                dict(on["passes"][1]["requests"][0])),
            "an extra concurrency": lambda off, on: on["passes"].append(dict(on["passes"][1], concurrency=32)),
        }
        for name, change in cases.items():
            off, on = pair()
            change(off, on)
            result = ab.compare_arms(off, on)
            self.assertFalse(result["passed"], name)
            self.assertTrue(result["coverage"], name)
        # Batched-only speculative arms are a valid workload.
        off, on = pair()
        for row in (off, on):
            row["passes"].pop(0)
        self.assertTrue(ab.compare_arms(off, on)["passed"])

    def test_counts_must_be_positive(self):
        for argv in (["compare", "a", "b", "--c1-requests", "0"], ["compare", "a", "b", "--repeats", "0"],
                     ["compare", "a", "b", "--concurrencies", "1", "1"]):
            with self.assertRaises(SystemExit), unittest.mock.patch("sys.argv", ["ab"] + argv), \
                    unittest.mock.patch("sys.stderr"):
                ab.main()

    def test_collect_requires_done_finish_usage_and_no_error(self):
        chunk = {"choices": [{"text": "a", "token_ids": [5],
                              "logprobs": {"token_logprobs": [-0.5]}, "finish_reason": None}]}
        last = {"choices": [{"text": "b", "token_ids": [6], "logprobs": {"token_logprobs": [-0.25]},
                             "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"completion_tokens": 2}}
        events = [(1.0, chunk), (2.0, last), (2.1, usage)]
        row = ab.collect(0, 0.5, events, True, None)
        self.assertTrue(row["complete"])
        self.assertEqual((row["text"], row["token_ids"], row["finish_reason"]), ("ab", [5, 6], "length"))
        self.assertFalse(ab.collect(0, 0.5, events, False, None)["complete"], "no [DONE]")
        self.assertFalse(ab.collect(0, 0.5, events[:2], True, None)["complete"], "no usage")
        self.assertFalse(ab.collect(0, 0.5, [(1.0, chunk), (2.1, usage)], True, None)["complete"],
                         "no finish reason")
        errored = events + [(2.2, {"error": {"message": "late"}})]
        self.assertFalse(ab.collect(0, 0.5, errored, True, None)["complete"], "an error event")
        self.assertIsNone(ab.cohort([{"arrivals": []}])["tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
