"""CPU checks for the exact seeded preservation gate."""

from qual_test_config import setUpModule  # noqa: F401

import copy
import json
import io
from contextlib import redirect_stdout
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import seeded_equality as seeded
from seeded_equality import check_speculation, compare_records, output, request_cases

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "exact seeded preservation gate on synthetic receipts"


def capture():
    top = {str(i): -0.25 * (i + 1) for i in range(5)}
    choice = {"token_ids": [7, 8], "finish_reason": "length", "logprobs": {
        "tokens": ["a", "b"], "token_logprobs": [-0.25, -0.5], "top_logprobs": [top, top]}}
    return {"passed": True, "mechanics_only": False, "concurrency": 1, "k": 1,
            "reference_sha256": "reference", "prompt_tokens": 4, "max_tokens": 2,
            "speculation": {"available": True, "verify_items": 2, "draft_tokens": 2},
            "requests": {"a": {"request": {"seed": 17, "prompt": [1, 2]},
                                "output": output(choice, 2)}}}


class SeededEqualityTests(unittest.TestCase):
    def test_ar_record_and_compare_accept_unpublished_speculation_series(self):
        # Engine metrics publish spec series only after a verify item. An AR
        # server still returns a valid /metrics scrape with no spec series.
        choice = {"token_ids": [7, 8], "finish_reason": "length", "logprobs": {
            "tokens": ["a", "b"], "token_logprobs": [-0.25, -0.5],
            "top_logprobs": [{"a": -0.25}, {"b": -0.5}]}}
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory) / 'exact.json'
            exact.write_text(json.dumps({'texts': [{'name': 'sample', 'score_from': 1,
                                                   'prompt_token_ids': [1, 2, 3]}]}))
            args = SimpleNamespace(exact=str(exact), prompt_tokens=2, max_tokens=2,
                                   url='http://unused', model=None, timeout=1, concurrency=1, k=0, arm='ar')
            with patch.object(seeded, 'Client') as client, patch.object(seeded, 'metrics_text',
                    return_value='chop_decode_steps_total 17\n'), patch.object(seeded, 'receipt') as receipt:
                client.return_value.identity.return_value = {'model': 'qwen38'}
                client.return_value.request.return_value = ({'choices': [choice]}, 0.01)
                seeded.record(args)
                record = dict(receipt.call_args.args[2], mechanics_only=False)
                self.assertEqual(record['speculation'], {'available': False, 'step_rows': {'available': False}})
                self.assertEqual(len(record['requests']), 2)
                self.assertIsNone(compare_records(record, copy.deepcopy(record)))
                changed = copy.deepcopy(record)
                changed['requests']['sample/unfiltered']['output']['logprobs'][0] = math.nextafter(-0.25, 0)
                self.assertEqual(compare_records(record, changed)['field'], 'logprobs')
                args.k = 1
                with self.assertRaisesRegex(ValueError, 'counters are missing'):
                    seeded.record(args)

    def test_unpublished_series_exception_is_only_for_valid_ar_absence(self):
        check_speculation({'available': False}, 0)
        check_speculation({'available': False, 'step_rows': {'available': True, 'steps': 12}}, 0)
        for k in (1, 2, 3, -1):
            with self.assertRaises(ValueError):
                check_speculation({'available': False}, k)
        for counters in ({}, {'available': None}, {'available': False, 'verify_items': 1}):
            with self.assertRaises(ValueError):
                check_speculation(counters, 0)
        with self.assertRaises(ValueError):
            check_speculation({'available': True, 'verify_items': 1, 'draft_tokens': 1}, 0)

    def test_identical_outputs_pass_without_server_identity_matching(self):
        left = capture()
        right = copy.deepcopy(left)
        right["server"] = {"url": "new-server"}
        self.assertIsNone(compare_records(left, right))

    def test_every_output_field_is_exact_including_top_scores(self):
        changes = {"token_ids": [7, 9], "tokens": ["a", "c"],
                   "logprobs": [-0.25, math.nextafter(-0.5, 0)], "finish_reason": "stop"}
        for field, value in changes.items():
            left = capture()
            right = copy.deepcopy(left)
            right["requests"]["a"]["output"][field] = value
            self.assertEqual(compare_records(left, right), {"request": "a", "field": field})
        left = capture()
        right = copy.deepcopy(left)
        right["requests"]["a"]["output"]["top_logprobs"][0]["4"] = math.nextafter(-1.25, 0)
        self.assertEqual(compare_records(left, right)["field"], "top_logprobs")

    def test_completion_top_map_allows_selected_token_and_duplicate_labels(self):
        left = capture()
        tops = left["requests"]["a"]["output"]["top_logprobs"]
        tops[0]["selected-outside-top-five"] = -2.0
        self.assertIsNone(compare_records(left, copy.deepcopy(left)))
        # The completion API keys alternatives by decoded label, so two
        # token IDs with the same label collapse to one map entry.
        tops[0].pop("4")
        tops[0].pop("3")
        self.assertIsNone(compare_records(left, copy.deepcopy(left)))

    def test_concurrent_records_require_explicit_diagnostic_mode(self):
        left = capture()
        left['concurrency'] = 16
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            compare_records(left, copy.deepcopy(left))
        self.assertIsNone(compare_records(left, copy.deepcopy(left), diagnostic=True))
        changed = copy.deepcopy(left)
        changed['requests']['a']['output']['logprobs'][0] = math.nextafter(-0.25, 0)
        self.assertEqual(compare_records(left, changed, diagnostic=True)['field'], 'logprobs')
        for value in (0, -1, None, True):
            changed = copy.deepcopy(left)
            changed['concurrency'] = value
            with self.assertRaises(ValueError):
                compare_records(changed, copy.deepcopy(changed))

    def test_diagnostic_cli_never_reports_a_passing_exact_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            record = capture()
            record['concurrency'] = 16
            path.write_text(json.dumps(record))
            args = SimpleNamespace(mechanics_only=False, command='compare', diagnostic=True,
                                   receipts=[str(path), str(path)])
            with patch.object(seeded, 'arguments') as parser, redirect_stdout(io.StringIO()) as out:
                parser.return_value.parse_args.return_value = args
                self.assertEqual(seeded.main(), 0)
                result = json.loads(out.getvalue())
                self.assertIsNone(result['passed'])
                self.assertTrue(result['diagnostic'])
                self.assertTrue(result['equal'])

    def test_signed_zero_is_not_silently_equal(self):
        left = capture()
        right = copy.deepcopy(left)
        left["requests"]["a"]["output"]["logprobs"][0] = 0.0
        right["requests"]["a"]["output"]["logprobs"][0] = -0.0
        self.assertIsNotNone(compare_records(left, right))

    def test_missing_or_invalid_evidence_is_refused(self):
        for change in (lambda r: r.update(requests={}),
                       lambda r: r.update(passed=False),
                       lambda r: r.update(mechanics_only=True),
                       lambda r: r.update(speculation={}),
                       lambda r: r["speculation"].update(verify_items=0),
                       lambda r: r["speculation"].update(draft_tokens=4),
                       lambda r: r["requests"]["a"]["output"].update(token_ids=[]),
                       lambda r: r["requests"]["a"]["output"].update(logprobs=[float("nan"), -0.5]),
                       lambda r: r["requests"]["a"]["output"].update(top_logprobs=[{}, {}])):
            left = capture()
            right = copy.deepcopy(left)
            change(right)
            with self.assertRaises(ValueError):
                compare_records(left, right)

    def test_changed_workload_or_draft_configuration_is_refused(self):
        for key, value in (("k", 0), ("max_tokens", 3), ("concurrency", 16),
                           ("reference_sha256", "different")):
            left = capture()
            right = copy.deepcopy(left)
            right[key] = value
            with self.assertRaises(ValueError):
                compare_records(left, right)
        right = capture()
        right["requests"]["a"]["request"]["seed"] = 18
        with self.assertRaises(ValueError):
            compare_records(capture(), right)

    def test_cases_exercise_filtered_history_and_unfiltered_rng(self):
        cases = request_cases({"texts": [{"name": "t", "prompt_token_ids": [1, 2, 3]}]}, 256, 128)
        self.assertEqual(set(cases), {"t/unfiltered", "t/filtered-history"})
        self.assertEqual(cases["t/unfiltered"]["prompt"], [1, 2])
        self.assertGreater(cases["t/filtered-history"]["frequency_penalty"], 0)
        self.assertGreater(cases["t/filtered-history"]["min_p"], 0)
        with self.assertRaises(ValueError):
            request_cases({"texts": []}, 256, 128)


if __name__ == "__main__":
    unittest.main()
