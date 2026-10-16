"""Logit gaps between a scored row's tokens; needs no torch."""

import unittest

from tools.refrun import margins

# One row: forced token 7 ranked outside the top 3; logits in nats.
SCORES = {"token": [7], "logit": [-2.25], "top1": [4], "margin": [0.5],
          "topk_ids": [[4, 9, 2]], "topk_logits": [[3.0, 2.5, 1.0]], "topk_logprobs": [[-0.4, -0.9, -2.4]]}


class GapTests(unittest.TestCase):
    def test_gaps_between_ranked_and_forced_tokens(self):
        self.assertEqual(margins.gap(SCORES, 0, 4, 9), SCORES["margin"][0])
        self.assertEqual(margins.gap(SCORES, 0, 4, 2), 2.0)
        self.assertEqual(margins.gap(SCORES, 0, 2, 4), -2.0)
        self.assertEqual(margins.gap(SCORES, 0, 4, 7), 5.25)
        self.assertEqual(margins.logits(SCORES, 0), {4: 3.0, 9: 2.5, 2: 1.0, 7: -2.25})

    def test_unranked_tokens_have_no_gap(self):
        self.assertIsNone(margins.gap(SCORES, 0, 4, 5))
        self.assertIsNone(margins.gap(SCORES, 0, 5, 4))


if __name__ == "__main__":
    unittest.main()
