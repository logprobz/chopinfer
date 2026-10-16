from types import SimpleNamespace
import unittest

import preempt_compare
from preempt_compare import burst, compare_records, completion
from unittest.mock import Mock, patch


def run(tokens, preemptions, gaps=None):
    """A record of one request whose position p has top-2 gap gaps[p]."""
    gaps = gaps or {}
    top = [{"a": -0.1, "b": -0.1 - gaps.get(position, 5.0)} for position in range(len(tokens))]
    return {"prompts_sha256": "x", "output_tokens": 8, "preemptions": preemptions,
            "requests": [{"name": "r", "tokens": tokens, "logprobs": [-0.1] * len(tokens),
                          "top": top, "finish_reason": "length"}]}


class CompareTest(unittest.TestCase):
    def test_completion_retains_client_usage_for_exactness_comparison(self):
        client = Mock()
        client.request.return_value = ({'choices': [{'logprobs': {'tokens': ['a'], 'token_logprobs': [-0.1], 'top_logprobs': [{'a': -0.1}]}, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}}, 0.1)
        self.assertEqual(completion(client, 'fixture', 'prompt', 1)['usage'], {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4})

    def test_completion_sends_min_tokens_only_when_set(self):
        client = Mock()
        client.request.return_value = ({'choices': [{'logprobs': {'tokens': ['a'], 'token_logprobs': [-0.1], 'top_logprobs': [{'a': -0.1}]}, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}}, 0.1)
        completion(client, 'fixture', 'prompt', 8)
        self.assertNotIn('min_tokens', client.request.call_args[0][1])
        completion(client, 'fixture', 'prompt', 8, min_tokens=8)
        self.assertEqual(client.request.call_args[0][1]['min_tokens'], 8)

    def test_burst_records_each_request_and_fails_short_outputs(self):
        client = Mock()
        client.identity.return_value = {"model": "fixture"}
        lengths = iter([4, 4, 2])

        def request(path, body):
            count = next(lengths)
            self.assertEqual(body["min_tokens"], 4)
            return ({"choices": [{"logprobs": {"tokens": ["a"] * count, "token_logprobs": [-0.1] * count,
                                               "top_logprobs": [{"a": -0.1}] * count},
                                  "finish_reason": "length"}],
                     "usage": {"prompt_tokens": 10 + count, "completion_tokens": count}}, 0.1)

        client.request.side_effect = request
        args = SimpleNamespace(url="http://fixture", timeout=5, model=None, requests=3, concurrency=1,
                               output_tokens=4, min_tokens=4, require_preemptions=False, arm="pressure")
        prompts = [(f"natural-{index}", "question") for index in range(2)]
        with patch.object(preempt_compare, "Client", return_value=client), \
                patch.object(preempt_compare, "prompts", return_value=prompts), \
                patch.object(preempt_compare, "metrics_text", return_value="vllm:num_preemptions_total 0\n"), \
                patch.object(preempt_compare, "receipt") as receipt:
            self.assertEqual(burst(args), 1)
        value = receipt.call_args[0][2]
        self.assertEqual([(row["prompt_tokens"], row["completion_tokens"]) for row in value["request_tokens"]],
                         [(14, 4), (14, 4), (12, 2)])
        self.assertNotIn("prompt_tokens_min", value)
        self.assertEqual(value["errors"], ["burst-2"])

    def test_identical_runs_pass(self):
        report = compare_records(run(list("abcdefgh"), 0), run(list("abcdefgh"), 2), 4, 0.5)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["identical_requests"], 1)
        self.assertEqual(report["positions_after_first_preemption"], 4)

    def test_a_divergence_before_the_first_preemption_fails(self):
        report = compare_records(run(list("abcdefgh"), 0, {2: 0.01}), run(list("abXdefgh"), 2), 4, 0.5)
        self.assertFalse(report["passed"])

    def test_a_later_near_tie_passes_and_a_clear_choice_fails(self):
        tie = compare_records(run(list("abcdefgh"), 0, {5: 0.2}), run(list("abcdeXgh"), 1), 4, 0.5)
        self.assertTrue(tie["passed"], tie)
        self.assertEqual(tie["divergences"][0]["position"], 5)
        clear = compare_records(run(list("abcdefgh"), 0, {5: 0.9}), run(list("abcdeXgh"), 1), 4, 0.5)
        self.assertFalse(clear["passed"])

    def test_batched_pairs_judge_early_divergences_by_the_near_tie_rule(self):
        # C16 companions differ from the first step, so a first divergence
        # before the preemption point is judged like a later one.
        tie = (run(list("abcdefgh"), 0, {2: 0.2}), run(list("abXdefgh"), 2))
        self.assertFalse(compare_records(*tie, 4, 0.5)["passed"])
        report = compare_records(*tie, 4, 0.5, ties_before_preemption=True)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["ties_before_preemption"])
        self.assertEqual([(row["name"], row["position"]) for row in report["divergences"]], [("r", 2)])
        self.assertAlmostEqual(report["divergences"][0]["top2_gap"], 0.2)
        self.assertEqual((report["divergences"][0]["before_first_preemption"], report["divergences"][0]["tokens"]),
                         (True, ["c", "X"]))
        stop = compare_records(run(list("abcdefgh"), 0, {6: 0.1}), run(list("abcdef"), 1), 4, 0.5)
        self.assertEqual((stop["divergences"][0]["before_first_preemption"], stop["divergences"][0]["tokens"]),
                         (False, ["g", None]))
        # The near-tie limit is the same: a decisive early divergence fails.
        clear = (run(list("abcdefgh"), 0, {2: 0.5}), run(list("abXdefgh"), 2))
        self.assertFalse(compare_records(*clear, 4, 0.5, ties_before_preemption=True)["passed"])
        # Later divergences and the preemption counts keep their rules.
        late = (run(list("abcdefgh"), 0, {5: 0.9}), run(list("abcdeXgh"), 1))
        self.assertFalse(compare_records(*late, 4, 0.5, ties_before_preemption=True)["passed"])
        self.assertFalse(compare_records(tie[0], run(list("abXdefgh"), 0), 4, 0.5,
                                         ties_before_preemption=True)["passed"])
        self.assertFalse(compare_records(*tie, 4, 0.5)["ties_before_preemption"])

    def test_the_batched_rule_refuses_exact_and_baseline_pairs(self):
        tie = (run(list("abcdefgh"), 0, {2: 0.2}), run(list("abXdefgh"), 2))
        for flags in ({"exact": True}, {"baseline": True}):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "ties before"):
                compare_records(*tie, 4, 0.5, ties_before_preemption=True, **flags)

    def test_an_early_stop_is_a_divergence(self):
        report = compare_records(run(list("abcdefgh"), 0, {6: 0.1}), run(list("abcdef"), 1), 4, 0.5)
        self.assertTrue(report["passed"], report)
        report = compare_records(run(list("abcdefgh"), 0), run(list("abcdef"), 1), 4, 0.5)
        self.assertFalse(report["passed"])

    def test_an_exact_pair_fails_on_any_bit_but_passes_when_identical(self):
        reserved, preempted = run(list("abcdefgh"), 0, {5: 0.2}), run(list("abcdefgh"), 1, {5: 0.2})
        self.assertTrue(compare_records(reserved, preempted, 4, 0.5, exact=True)["passed"])
        preempted["requests"][0]["logprobs"][6] = -0.1000001
        self.assertTrue(compare_records(reserved, preempted, 4, 0.5)["passed"])
        report = compare_records(reserved, preempted, 4, 0.5, exact=True)
        self.assertEqual(report["failures"], ["r: not bit-identical from position 6"])
        tie = compare_records(reserved, run(list("abcdeXgh"), 1), 4, 0.5, exact=True)
        self.assertEqual(tie["failures"], ["r: not bit-identical from position 5"])

    def test_an_exact_pair_compares_the_sign_of_zero(self):
        for field in ("logprobs", "top"):
            reserved, preempted = run(list("abcdef"), 0), run(list("abcdef"), 1)
            if field == "logprobs":
                reserved["requests"][0]["logprobs"][5] = 0.0
                preempted["requests"][0]["logprobs"][5] = -0.0
            else:
                reserved["requests"][0]["top"][5]["a"] = 0.0
                preempted["requests"][0]["top"][5]["a"] = -0.0
            self.assertTrue(compare_records(reserved, preempted, 4, 0.5)["passed"])
            report = compare_records(reserved, preempted, 4, 0.5, exact=True)
            self.assertEqual(report["failures"], ["r: not bit-identical from position 5"], field)

    def test_a_missing_renamed_or_duplicate_request_is_refused(self):
        reserved = run(list("abcd"), 0)
        reserved["requests"].append(dict(reserved["requests"][0], name="s"))
        missing = run(list("abcd"), 1)
        renamed = run(list("abcd"), 1)
        renamed["requests"].append(dict(renamed["requests"][0], name="t"))
        duplicate = run(list("abcd"), 1)
        duplicate["requests"].append(dict(duplicate["requests"][0]))
        for preempted in (missing, renamed, duplicate):
            with self.assertRaises(ValueError):
                compare_records(reserved, preempted, 2, 0.5, exact=True)
        with self.assertRaises(ValueError):
            compare_records(duplicate, duplicate, 2, 0.5, baseline=True, exact=True)

    def test_preemption_counts_are_checked(self):
        self.assertFalse(compare_records(run(list("ab"), 1), run(list("ab"), 1), 4, 0.5)["passed"])
        self.assertFalse(compare_records(run(list("ab"), 0), run(list("ab"), 0), 4, 0.5)["passed"])

    def test_a_baseline_pair_skips_the_preemption_counts(self):
        report = compare_records(run(list("abcdef"), 0), run(list("abcdef"), 0), 0, 0.5, baseline=True)
        self.assertTrue(report["passed"], report)

    def test_failed_requests_fail(self):
        failed = run(list("ab"), 1)
        failed["requests"][0] = {"name": "r", "error": "HTTP 500"}
        self.assertFalse(compare_records(run(list("ab"), 0), failed, 4, 0.5)["passed"])


if __name__ == "__main__":
    unittest.main()


class DeclarationTest(unittest.TestCase):
    """The declared preemption checks: a C16 pair judged by near ties needs an exact C1 pair."""

    def checks(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for path in sorted(root.glob("models/*/qual.json")):
            value = json.loads(path.read_text())
            yield path.parent.name, value, {name: check for name, check in value["checks"].items()
                                            if check.get("module") == "preempt_compare"}

    def test_a_batched_c16_pair_comes_with_an_exact_c1_pair(self):
        import release_preview as runner
        batched = exact_c16 = 0
        for model, value, checks in self.checks():
            for name, check in checks.items():
                args = check["args"]
                inputs = {args[args.index(flag) + 1].strip("{}") for flag in ("--reserved", "--preempted")}
                concurrency = {runner.RECORDS[key][2][runner.RECORDS[key][2].index("--concurrency") + 1]
                               for key in inputs}
                paths = {tuple(rule["path"]) for rule in check["rules"]}
                with self.subTest(model=model, check=name):
                    self.assertEqual(len(concurrency), 1)
                    if "--ties-before-preemption" in args:
                        batched += 1
                        self.assertNotIn("--exact", args)
                        self.assertEqual(concurrency, {"16"})
                        self.assertIn(("ties_before_preemption",), paths)
                        self.assertNotIn(("exact",), paths)
                        c1 = [other for other in checks.values() if "--exact" in other["args"]
                              and {runner.RECORDS[key.strip("{}")][2][-1] for key in
                                   (other["args"][other["args"].index("--reserved") + 1],
                                    other["args"][other["args"].index("--preempted") + 1])} == {"1"}]
                        self.assertEqual(len(c1), 1)
                        full = value["cohorts"]["full"]["checks"]
                        self.assertIn(name, full)
                        self.assertTrue(any(other in full for other, check in checks.items() if check is c1[0]))
                    else:
                        self.assertIn("--exact", args)
                        self.assertIn(("exact",), paths)
                        exact_c16 += concurrency == {"16"}
        # Some models judge C16 by near ties; the others keep C16 exact.
        self.assertGreaterEqual(batched, 2)
        self.assertGreaterEqual(exact_c16, 1)

    def test_c1_records_run_on_the_preemption_arms_alone(self):
        import release_preview as runner
        for suffix, arm in (("reserved", "reserved"), ("candidate", "optimistic")):
            c16, c1 = runner.RECORDS[f"preemption_{suffix}"], runner.RECORDS[f"preemption_{suffix}_c1"]
            self.assertEqual((c1[0], c1[1]), (arm, "preempt_compare.py"))
            self.assertEqual(c1[2][:-1], c16[2][:-1])
            self.assertEqual((c16[2][-2:], c1[2][-2:]), (["--concurrency", "16"], ["--concurrency", "1"]))
