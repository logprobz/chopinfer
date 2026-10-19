"""Check strict served comparison and capacity accounting without a GPU."""
import copy
import unittest

import verify_width_job as job


class VerifyWidthJobTests(unittest.TestCase):
    def response(self):
        n = job.OUTPUT_TOKENS
        return {"usage": {"completion_tokens": n}, "choices": [{
            "finish_reason": "length", "token_ids": [7] * n,
            "logprobs": {"tokens": ["x"] * n, "token_logprobs": [-0.5] * n,
                         "top_logprobs": [{"x": -0.5}] * n}}]}

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
