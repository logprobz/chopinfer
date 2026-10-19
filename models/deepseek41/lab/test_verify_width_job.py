"""Check strict served comparison and capacity accounting without a GPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import verify_width_job as job


class VerifyWidthJobTests(unittest.TestCase):
    def response(self):
        n = job.OUTPUT_TOKENS
        return {"usage": {"prompt_tokens": 1, "completion_tokens": n}, "choices": [{
            "finish_reason": "length", "token_ids": [7] * n,
            "logprobs": {"tokens": ["x"] * n, "token_logprobs": [-0.5] * n,
                         "top_logprobs": [{"x": -0.5}] * n}}]}

    def test_collapsed_keys_are_counted_without_losing_score_bits(self):
        response = self.response()
        scores = response["choices"][0]["logprobs"]
        scores["top_logprobs"] = [dict((str(i), -float(i)) for i in range(5))
                                 for _ in range(job.OUTPUT_TOKENS)]
        for position, size in enumerate(range(1, 6)):
            # A token label may overwrite another ID's score during JSON
            # serialization. Keep that map value and the selected score.
            labels = ["x", "a", "b", "c", "d"]
            scores["top_logprobs"][position] = {labels[i % size]: -float(i) for i in range(5)}
        actual = job.canonical(response)
        stats = actual["top_logprob_stats"]
        self.assertEqual(stats["collapsed_positions"], 4)
        self.assertEqual(stats["collapsed_entries"], 10)
        self.assertEqual(stats["sixth_key_positions"], 0)
        self.assertEqual(actual["top_bits"][0], {"x": job.bits(-4.0)})
        self.assertEqual(actual["selected_bits"][0], job.bits(-0.5))

    def sampled_response(self):
        response = self.response()
        scores = response["choices"][0]["logprobs"]
        scores["tokens"] = ["sampled"] * job.OUTPUT_TOKENS
        scores["token_logprobs"] = [-6.0] * job.OUTPUT_TOKENS
        row = {str(i): -float(i) for i in range(5)}
        row["sampled"] = -6.0
        scores["top_logprobs"] = [row.copy() for _ in range(job.OUTPUT_TOKENS)]
        return response

    def test_sixth_key_requires_the_sampled_token_and_exact_selected_bits(self):
        response = self.sampled_response()
        actual = job.canonical(response, sampled=True)
        self.assertEqual(actual["top_logprob_stats"]["sixth_key_positions"], job.OUTPUT_TOKENS)
        self.assertEqual(actual["top_logprob_stats"]["collapsed_entries"], 0)
        self.assertEqual(len(actual["top_bits"][0]), 6)
        with self.assertRaisesRegex(ValueError, "position 0"):
            job.canonical(response)
        for label, score in [("not-emitted", -6.0), ("sampled", -6.000000000000001)]:
            broken = copy.deepcopy(response)
            row = broken["choices"][0]["logprobs"]["top_logprobs"][0]
            del row["sampled"]
            row[label] = score
            with self.assertRaisesRegex(ValueError, "position 0"):
                job.canonical(broken, sampled=True)

    def test_top_key_or_one_score_bit_difference_still_fails(self):
        left = {"case": [{"request": {"temperature": 1},
                          "canonical": job.canonical(self.sampled_response(), sampled=True)}]}
        for mutate in (lambda row: row.pop("0"),
                       lambda row: row.__setitem__("sampled", job.bits(-6.000000000000001))):
            right = copy.deepcopy(left)
            mutate(right["case"][0]["canonical"]["top_bits"][0])
            self.assertFalse(job.compare(left, right)["bitwise"])

    def test_rejected_payload_is_saved_before_validation(self):
        response = self.response()
        response["choices"][0]["logprobs"]["top_logprobs"][0] = {}
        raw = json.dumps(response, indent=2) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            server = SimpleNamespace(directory=Path(directory), url="http://unused")

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            p = SimpleNamespace(MODEL="fixture", write=write, http=lambda *a, **kw: raw)
            with patch.object(job, "CONCURRENCIES", (1,)):
                with self.assertRaisesRegex(ValueError, "position 0.*raw response"):
                    job.capture(p, server, [[1]], "first")
            saved = server.directory / "responses/first/c1-t0/000.response.json"
            self.assertEqual(saved.read_text(), raw)
            request = json.loads(saved.with_name("000.request.json").read_text())
            self.assertEqual(request["temperature"], 0)

    def test_one_logprob_bit_fails_even_with_identical_tokens(self):
        left = {"case": [{"request": {"seed": 1}, "canonical": job.canonical(self.response())}]}
        right = copy.deepcopy(left)
        right["case"][0]["canonical"]["selected_bits"][0] = job.bits(-0.5000000000000001)
        self.assertTrue(job.compare(left, left)["bitwise"])
        self.assertFalse(job.compare(left, right)["bitwise"])
        with self.assertRaises(ValueError):
            job.compare(left, {})

    def test_incomplete_and_nonfinite_outputs_are_refused(self):
        for mutate in (lambda row: row["choices"][0]["token_ids"].pop(),
                       lambda row: row["choices"][0]["logprobs"]["token_logprobs"].__setitem__(0, float("nan")),
                       lambda row: row["choices"][0]["logprobs"]["top_logprobs"].__setitem__(0, {})):
            row = self.response()
            mutate(row)
            with self.assertRaises(ValueError):
                job.canonical(row)

    def test_exact_capacity_is_derived_from_live_rows_not_old_host_estimate(self):
        row = {"prefill_tokens": 0, "decode_rows": 1, "verify_items": 1,
               "verify_rows": 6, "row_capacity": 8, "execution_us": 123.0}
        exact = job.width_costs([row], True)[0]
        bucket = job.width_costs([row], False)[0]
        self.assertEqual((exact["items"], exact["live_rows"], exact["graph_rows"]), (2, 7, 7))
        self.assertEqual(bucket["graph_rows"], 8)
        self.assertEqual(job.width_costs([{**row, "prefill_tokens": 1}], True), [])


if __name__ == "__main__":
    unittest.main()
