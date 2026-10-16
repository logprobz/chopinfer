"""Vectors of the limit rules in limits.py, including today's gate numbers."""

from qual_test_config import setUpModule  # noqa: F401

from decimal import Decimal
import json
import math
import struct
import unittest

from common import ROOT
from limits import (NEAR_TIE_NATS, bit_equal, first_difference, flip, near_tie, paired, rounded_up,
                    spread_limits, top1_summary, within)
import multimodal
import teacher

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "limit rule vectors and the shipped gates' numbers, read by name"


def spread(mean, top1=0.0, near=0.08, total=0.10):
    return {"mean_abs_dlogprob": mean, "top1_decisive": top1, "top1_near_tie": near, "top1_disagreement": total}


class SpreadRuleTests(unittest.TestCase):
    def test_rounding_up_to_the_step_gives_the_decimal(self):
        for value, expected in ((0.0, 0.0), (0.01, 0.05), (0.068489, 0.1), (0.1, 0.1), (0.1001, 0.15),
                                (0.12, 0.15), (0.2194418431591794, 0.25), (0.31, 0.35)):
            self.assertEqual(rounded_up(value, 0.05), expected, value)

    def test_values_at_a_step_keep_it_and_any_value_above_takes_the_next(self):
        for k in range(1, 21):
            step = float(Decimal(k) * Decimal("0.05"))
            following = float(Decimal(k + 1) * Decimal("0.05"))
            self.assertEqual(rounded_up(step, 0.05), step, step)
            for below in (math.nextafter(step, 0.0), step - 1e-10, step - 1e-6):
                self.assertEqual(rounded_up(below, 0.05), step, below)
            for above in (math.nextafter(step, math.inf), step + 1e-10, step + 1e-6):
                self.assertEqual(rounded_up(above, 0.05), following, above)
        self.assertEqual(rounded_up(0.1500000001, 0.05), 0.2)
        self.assertEqual(rounded_up(0.10000000001, 0.05), 0.15)
        self.assertEqual(rounded_up(0.1499999999, 0.05), 0.15)

    def test_binary_float_artifacts_are_read_as_their_shortest_decimal(self):
        # Float quotients near a step: 0.15 / 0.05 is 2.9999999999999996 and
        # 0.45 / 0.05 is 9.000000000000002 in binary; both are multiples.
        self.assertEqual(rounded_up(0.15, 0.05), 0.15)
        self.assertEqual(rounded_up(0.45, 0.05), 0.45)
        self.assertEqual(rounded_up(0.7, 0.05), 0.7)
        # Binary sums that land one ulp above a step lie above it.
        self.assertEqual(0.1 + 0.05, 0.15000000000000002)
        self.assertEqual(rounded_up(0.1 + 0.05, 0.05), 0.2)
        self.assertEqual(rounded_up(3 * 0.05, 0.05), 0.2)
        self.assertEqual(rounded_up(0.1 + 0.2, 0.05), 0.35)
        # Limits are exact decimals, so rounding them again keeps them.
        for value in (0.01, 0.12, 0.2194418431591794, 0.31, 0.1500000001):
            self.assertEqual(rounded_up(rounded_up(value, 0.05), 0.05), rounded_up(value, 0.05))

    def test_a_mean_just_above_the_large_set_floor_raises_the_limit(self):
        self.assertEqual(spread_limits([spread(0.10000000001)], 1000), spread(0.15, 0.08))
        self.assertEqual(spread_limits([spread(0.1)], 1000), spread(0.1, 0.08))
        self.assertEqual(spread_limits([spread(0.1500000001)], 999), spread(0.2, 0.08))
        self.assertEqual(spread_limits([spread(0.15)], 999), spread(0.15, 0.08))

    def test_floors_until_the_worst_variant_exceeds_them(self):
        self.assertEqual(spread_limits([], 10), spread(0.15, 0.08))
        self.assertEqual(spread_limits([spread(0.02, 0.01), spread(0.14, 0.05)], 10), spread(0.15, 0.08))
        self.assertEqual(spread_limits([spread(0.02, 0.09), spread(0.16, 0.05)], 10), spread(0.2, 0.09))

    def test_sets_of_a_thousand_positions_have_the_tighter_mean_floor(self):
        self.assertEqual(spread_limits([spread(0.0)], 999), spread(0.15, 0.08))
        self.assertEqual(spread_limits([spread(0.0)], 1000), spread(0.1, 0.08))
        self.assertEqual(spread_limits([spread(0.12)], 1000), spread(0.15, 0.08))
        # The tighter floor is declared per gate.
        self.assertEqual(spread_limits([], 1000, large_set_mean_floor=0.15), spread(0.15, 0.08))

    def test_an_optional_cap_bounds_the_mean_limit(self):
        self.assertEqual(spread_limits([spread(0.31)], 40), spread(0.35, 0.08))
        self.assertEqual(spread_limits([spread(0.31)], 40, mean_cap=0.3), spread(0.3, 0.08))
        self.assertEqual(spread_limits([spread(0.21)], 40, mean_cap=0.3), spread(0.25, 0.08))

    def test_within_includes_the_limits(self):
        self.assertTrue(within(spread(0.15, 0.08), spread(0.15, 0.08)))
        self.assertFalse(within(spread(0.1500001, 0.0), spread(0.15, 0.08)))
        self.assertFalse(within(spread(0.0, 0.0800001), spread(0.15, 0.08)))


class NearTieRuleTests(unittest.TestCase):
    # The reference's stored top 5 at one position: its top-1 7, then 8 at
    # 0.1 nats, 9 at 0.3 and 4 at 2.5.
    TOP = [[7, -0.6], [8, -0.7], [9, -0.9], [4, -3.1], [5, -3.2]]

    def test_agreement_is_no_flip(self):
        self.assertIsNone(flip(7, 7, self.TOP))
        self.assertIsNone(flip(7, 7))

    def test_a_flip_below_the_limit_is_a_near_tie(self):
        near = flip(7, 8, self.TOP)
        self.assertAlmostEqual(near["gap"], 0.1)
        self.assertFalse(near["decisive"])

    def test_a_flip_beyond_the_limit_is_decisive(self):
        self.assertEqual(NEAR_TIE_NATS, 0.3)
        self.assertTrue(flip(7, 4, self.TOP)["decisive"])
        self.assertTrue(flip(7, 9, [[7, 0.0], [9, -0.3000001]])["decisive"])

    def test_the_limit_itself_is_a_near_tie_in_every_gate(self):
        # The image-probe gate accepts a margin of exactly 0.3 nats; so do the
        # top-1 rule, the media greedy walk and the GSM8K branch rule.
        self.assertEqual(flip(7, 9, [[7, 0.0], [9, -0.3]]), {"gap": 0.3, "decisive": False})
        self.assertTrue(near_tie(-0.3))
        self.assertFalse(near_tie(0.3000001))
        reference = {"answer": {"tokens": [[65], [66]]}, "steps": [[1, 0.0, [[1, 0.0, [65]], [2, -0.3, [67]]]]]}
        observed = {"finish_reason": "length", "logprobs": {"content": [{"token": "C", "bytes": [67]}]}}
        self.assertTrue(multimodal.reference_greedy_agreement(reference, observed, NEAR_TIE_NATS)["passed"])
        self.assertEqual(multimodal.NEAR_TIE_LIMIT, NEAR_TIE_NATS)

    def test_a_token_outside_the_stored_top_k_is_decisive(self):
        self.assertEqual(flip(7, 3, self.TOP), {"gap": None, "decisive": True})
        # Without a stored top-k every flip is decisive.
        self.assertEqual(flip(7, 8), {"gap": None, "decisive": True})

    def test_raw_near_tie_and_decisive_counts(self):
        flips = [None, flip(7, 8, self.TOP), flip(7, 4, self.TOP), flip(7, 3, self.TOP), None]
        self.assertEqual(top1_summary(flips, 5), {
            "top1_disagreements": 3, "top1_disagreement": 0.6, "top1_near_ties": 1, "top1_near_tie": 0.2,
            "top1_decisive_flips": 2, "top1_decisive": 0.4})
        self.assertIsNone(top1_summary([], 0)["top1_decisive"])

    def test_the_top1_limit_applies_to_the_decisive_rate(self):
        limits = spread_limits([], 100)
        # 2 near ties and 8 decisive flips in 100 positions: 10% raw, 8% decisive.
        flips = [flip(7, 8, self.TOP)] * 2 + [flip(7, 4, self.TOP)] * 8 + [None] * 90
        measured = {"mean_abs_dlogprob": 0.05, **top1_summary(flips, 100)}
        self.assertEqual(measured["top1_disagreement"], 0.1)
        self.assertTrue(within(measured, limits))
        flips[-1] = flip(7, 3, self.TOP)
        self.assertFalse(within({"mean_abs_dlogprob": 0.05, **top1_summary(flips, 100)}, limits))

    def test_near_tie_flips_have_their_own_limit(self):
        # Every position a near-tie flip with the teacher token's logprob
        # unchanged: mean error and decisive rate are zero, yet it fails.
        limits = spread_limits([], 100)
        swapped = {"mean_abs_dlogprob": 0.0, **top1_summary([flip(7, 8, self.TOP)] * 100, 100)}
        self.assertEqual((swapped["top1_decisive"], swapped["top1_near_tie"]), (0.0, 1.0))
        self.assertFalse(within(swapped, limits))
        self.assertEqual(limits["top1_near_tie"], 0.08)
        at_limit = {"mean_abs_dlogprob": 0.0, **top1_summary([flip(7, 8, self.TOP)] * 8 + [None] * 92, 100)}
        self.assertTrue(within(at_limit, limits))
        self.assertFalse(within({**at_limit, "top1_near_tie": 0.0800001}, limits))

    def test_a_variant_raises_the_near_tie_limit_by_its_rate_rounded_up(self):
        variant = {"mean_abs_dlogprob": 0.02, **top1_summary([flip(7, 8, self.TOP)] * 9 + [None] * 91, 100)}
        self.assertEqual(spread_limits([variant], 100)["top1_near_tie"], 0.09)
        variant = {"mean_abs_dlogprob": 0.02, **top1_summary([flip(7, 8, self.TOP)] * 91 + [None] * 909, 1000)}
        self.assertEqual(spread_limits([variant], 1000)["top1_near_tie"], 0.1)
        # The decisive limit is untouched by near ties, and the mean limits by both.
        self.assertEqual(spread_limits([variant], 1000)["top1_decisive"], 0.08)
        self.assertEqual(spread_limits([variant], 1000)["mean_abs_dlogprob"], 0.1)

    def test_all_flips_together_have_a_cap(self):
        # 4 near ties and 4 decisive flips in 50 positions: 8% each, within
        # both component limits, but 16% in all, above the 10% total cap.
        limits = spread_limits([], 50)
        self.assertEqual(limits["top1_disagreement"], 0.10)
        mixed = {"mean_abs_dlogprob": 0.0, **top1_summary(
            [flip(7, 8, self.TOP)] * 4 + [flip(7, 4, self.TOP)] * 4 + [None] * 42, 50)}
        self.assertEqual((mixed["top1_near_tie"], mixed["top1_decisive"], mixed["top1_disagreement"]),
                         (0.08, 0.08, 0.16))
        self.assertFalse(within(mixed, limits))
        five = {"mean_abs_dlogprob": 0.0, **top1_summary(
            [flip(7, 8, self.TOP)] * 3 + [flip(7, 4, self.TOP)] * 2 + [None] * 45, 50)}
        self.assertTrue(within(five, limits))

    def test_a_variant_raises_the_total_cap_to_its_exact_rate(self):
        # MiMo chat's BF16 variant: 65 flips of 486 (13.37%), 20 of them near ties.
        variant = {"mean_abs_dlogprob": 0.4, **top1_summary(
            [flip(7, 8, self.TOP)] * 20 + [flip(7, 4, self.TOP)] * 45 + [None] * 421, 486)}
        limits = spread_limits([variant], 486)
        self.assertEqual((limits["top1_decisive"], limits["top1_near_tie"], limits["top1_disagreement"]),
                         (45 / 486, 0.08, 65 / 486))
        # 45 decisive and 38 near-tie flips fit each component limit, not the cap.
        capture = {"mean_abs_dlogprob": 0.0, **top1_summary(
            [flip(7, 8, self.TOP)] * 38 + [flip(7, 4, self.TOP)] * 45 + [None] * 403, 486)}
        self.assertFalse(within(capture, limits))
        # The cap is the variant's own rate, unrounded: 65 flips pass, 66 do not.
        self.assertTrue(within({**capture, "top1_disagreement": 65 / 486}, limits))
        self.assertFalse(within({**capture, "top1_disagreement": 66 / 486}, limits))

    def test_a_variant_raises_the_limit_only_by_its_decisive_rate(self):
        # A variant with 30% raw and 10% decisive flips raises the limit to 10%.
        variant = {"mean_abs_dlogprob": 0.02, **top1_summary(
            [flip(7, 8, self.TOP)] * 20 + [flip(7, 4, self.TOP)] * 10 + [None] * 70, 100)}
        self.assertEqual(spread_limits([variant], 100), spread(0.15, 0.1, 0.2, 0.3))
        # A total rate between whole percents is kept as it is.
        odd = {"mean_abs_dlogprob": 0.02, **top1_summary([flip(7, 4, self.TOP)] * 11 + [None] * 89, 101)}
        self.assertEqual(spread_limits([odd], 101)["top1_disagreement"], 11 / 101)
        near_only = {"mean_abs_dlogprob": 0.02, **top1_summary([flip(7, 8, self.TOP)] * 30 + [None] * 70, 100)}
        self.assertEqual(spread_limits([near_only], 100), spread(0.15, 0.08, 0.3, 0.3))


class TodaysLimitTests(unittest.TestCase):
    """The limits the gates on main derive from the shipped fixtures."""

    def test_mimo_teacher_pooled_and_per_category(self):
        texts = json.loads((ROOT / "fixtures/mimo26/teacher-exact-reference.json").read_text())["texts"]
        # The F32 gate variant's BF16 spread: 0.1086 pooled, rounded up to 0.15.
        self.assertEqual(teacher.reference_limits(texts)[0], spread(0.15, 0.08))
        categories = {category: teacher.reference_limits([text for text in texts if text["category"] == category])[0]
                      for category in {text["category"] for text in texts}}
        # Chat's BF16 variant flips at 65 of 486 positions, 45 of them decisive.
        self.assertEqual(categories, {"chat": spread(0.45, 45 / 486, total=65 / 486), "code": spread(0.15, 0.08),
                                      "long-context": spread(0.15, 0.08), "math": spread(0.15, 0.08),
                                      "multilingual": spread(0.15, 0.08), "prose": spread(0.15, 0.08),
                                      "prose-frozen": spread(0.15, 0.08)})

    def test_mimo_media_cases(self):
        reference, _, _ = multimodal.load_reference(ROOT / "fixtures/mimo26/multimodal/reference.json",
                                                  ROOT / "fixtures/mimo26/multimodal/exact-reference.json")
        self.assertEqual(len(reference["cases"]), 11)
        for case in reference["cases"]:
            self.assertEqual(multimodal.case_limits(case), spread(0.15, 0.08), case["id"])

    def test_qwen_image_probes(self):
        path = ROOT / "fixtures/qwen38/image-probe-references.json"
        capture = json.loads(path.read_text())["provenance"]["capture_sha256"]
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(ROOT / "models/qwen38/qual.json")}):
            probes, rule, _ = multimodal.probe_references(capture, path)
        self.assertNotIn("mean_cap", rule)
        # The runner's BF16 run raises gradient's near-tie and total limits (4
        # near ties in 25 positions).
        self.assertEqual({name: probes[name]["limits"] for name, item in probes.items() if item["kind"] == "free-text"},
                         {"gradient-333x517": spread(0.15, 0.08, 0.16, 0.16), "small-100x60": spread(0.15, 0.08),
                          "wide-1000x40": spread(0.15, 0.08), "large-4500x4000": spread(0.15, 0.08)})


class PairedRuleTests(unittest.TestCase):
    def test_mean_gap_and_discordant_flips(self):
        test = [(0.10, False), (0.30, True), (0.20, True), (0.0, False)]
        baseline = [(0.10, True), (0.10, False), (0.20, True), (0.0, False)]
        gate = paired(test, baseline, mean_margin=0.05, sigmas=2.0)
        self.assertAlmostEqual(gate["mean_gap"], 0.05)
        self.assertEqual((gate["top1_test_only"], gate["top1_baseline_only"]), (1, 1))
        self.assertEqual(gate["top1_margin"], 2.0 * 2 ** 0.5)
        self.assertTrue(gate["passed"])
        self.assertFalse(paired(test, baseline, mean_margin=0.04, sigmas=2.0)["passed"])

    def test_excess_flips_beyond_the_sigmas_fail(self):
        test = [(0.0, True)] * 5
        baseline = [(0.0, False)] * 5
        # Five test-only flips against a margin of 2 * sqrt(5).
        self.assertFalse(paired(test, baseline, mean_margin=0.0, sigmas=2.0)["passed"])
        self.assertTrue(paired(test, baseline, mean_margin=0.0, sigmas=3.0)["passed"])

    def test_the_arms_must_score_the_same_positions(self):
        with self.assertRaises(ValueError):
            paired([(0.0, False)], [], mean_margin=0.0, sigmas=2.0)


class ExactRuleTests(unittest.TestCase):
    def answer(self, **changes):
        return {"text": "ab", "reasoning": None, "finish_reason": "stop", "token_ids": [1, 2],
                "tokens": ["a", "b"], "logprobs": [-0.5, 0.0],
                "top_logprobs": [[["a", -0.5], ["c", -1.0]], [["b", 0.0]]], **changes}

    def test_equal_answers(self):
        self.assertIsNone(first_difference(self.answer(), self.answer()))
        self.assertIsNone(first_difference(self.answer(), self.answer(logprobs=[-0.5, 0])))

    def test_the_binary_refuses_what_it_cannot_judge(self):
        """Through `chopinfer eval rescore` itself: a missing operand, an
        empty field selection, a boolean decimal and a non-finite result are
        refused, never judged."""
        import kinds
        calls = [{"op": "limit", "rule": "bit_equal", "args": {"left": None}},
                 {"op": "limit", "rule": "flips", "args": {"positions": [{}]}},
                 {"op": "limit", "rule": "first_difference",
                  "args": {"left": {"text": "a"}, "right": {"text": "b"}, "positions": [], "answer": []}},
                 {"op": "limit", "rule": "rounded_up", "args": {"value": True, "step": 0.05}},
                 {"op": "limit", "rule": "rounded_up", "args": {"value": 1.0000000000000002e308, "step": 1e308}},
                 {"op": "limit", "rule": "paired", "args": {"test": [[0.0, True]] * 4, "baseline": [[0.0, False]] * 4,
                                                            "mean_margin": 0.0, "sigmas": 1e308}}]
        for call, result in zip(calls, kinds.evaluated(calls), strict=True):
            self.assertEqual(result.get("error", {}).get("type"), "invalid", call)
        with self.assertRaises(ArithmeticError):
            rounded_up(True, 0.05)
        with self.assertRaises(ValueError):
            paired([(0.0, True)] * 4, [(0.0, False)] * 4, mean_margin=0.0, sigmas=1e308)
        with self.assertRaises(ValueError):
            first_difference(self.answer(), self.answer(text="b"), positions=(), answer=())

    def test_the_binary_matches_the_previous_rules_or_refuses(self):
        """The limit op over crates/runtime-eval/tests/data/limit-op-cases.json
        through `chopinfer eval rescore`: Python's previous result, bit for
        bit, or a refusal where it raised; a narrowed case (a boolean as a
        number, an int beyond the rule's exact doubles) is refused although
        Python judged it."""
        import kinds

        def same(a, b):
            if isinstance(a, bool) or isinstance(b, bool):
                return type(a) is type(b) and a == b
            if isinstance(a, float) and isinstance(b, float):
                return struct.pack(">d", a) == struct.pack(">d", b)
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(map(same, a, b))
            return a == b

        cases = json.loads((ROOT / "crates/runtime-eval/tests/data/limit-op-cases.json").read_text())["cases"]
        results = kinds.evaluated([{"op": "limit", "rule": case["rule"], "args": case["args"]} for case in cases])
        for case, result in zip(cases, results, strict=True):
            expected = case["expected"]
            if "value" in expected:
                self.assertTrue("value" in result and same(expected["value"], result["value"]), (case, result))
            else:
                self.assertEqual(result.get("error", {}).get("type"), expected["error"], (case, result))
        with self.assertRaises(ArithmeticError):
            spread_limits([spread(0.1, near=True)], 1000)
        with self.assertRaises(ValueError):
            flip([1], [2], [[[1], -0.5], [[2], -0.6]])
        self.assertEqual(rounded_up(9007199254740993, 2), 9007199254740994.0)

    def test_floats_differ_in_their_bits(self):
        self.assertFalse(bit_equal(0.0, -0.0))
        self.assertFalse(bit_equal(-0.5, -0.5000000000000001))
        self.assertFalse(bit_equal(None, 0.0))
        self.assertFalse(bit_equal(True, 1.0))
        # JSON, and so any stored receipt, holds no NaN: the rule refuses one.
        with self.assertRaises(ValueError):
            bit_equal(float("nan"), float("nan"))
        self.assertEqual(first_difference(self.answer(), self.answer(logprobs=[-0.5, -0.0])),
                         ("logprobs", 1, 0.0, -0.0))
        top = [[["a", -0.5], ["c", -1.0000000000000002]], [["b", 0.0]]]
        self.assertEqual(first_difference(self.answer(), self.answer(top_logprobs=top)),
                         ("top_logprobs", 0, [["a", -0.5], ["c", -1.0]], top[0]))

    def test_the_first_position_comes_first_then_the_answer_fields(self):
        other = self.answer(token_ids=[1, 3], tokens=["a", "c"], text="ac", finish_reason="length")
        self.assertEqual(first_difference(self.answer(), other), ("token_ids", 1, 2, 3))
        self.assertEqual(first_difference(self.answer(), self.answer(finish_reason="length")),
                         ("finish_reason", None, "stop", "length"))

    def test_a_longer_answer_differs_where_the_shorter_ends(self):
        longer = self.answer(token_ids=[1, 2, 4])
        self.assertEqual(first_difference(self.answer(), longer), ("token_ids", 2, None, 4))
        self.assertEqual(first_difference(self.answer(token_ids=None), self.answer(token_ids=[])), None)


if __name__ == "__main__":
    unittest.main()
