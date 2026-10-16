"""Offline row labelling and scoring checks for verify_exact.py."""

from qual_test_config import setUpModule

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

EXACT = Path(__file__).resolve().parents[2] / "fixtures/mimo26/teacher-exact-reference.json"
from verify_exact import (agreed_continuations, compare, disagrees, draft_count, exact_path,
                          expected_counters, report, row_labels, same_top1, variants)

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "verify_exact row labelling and scoring"


def row(logprob, top, **label):
    return {"logprob": logprob, "top": top, **label}


class BoundTopologyTests(unittest.TestCase):
    def compare_bound(self, change=None):
        from common import checkpoint_pin
        pin = checkpoint_pin()
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory) / 'exact.json'
            exact.write_text(json.dumps(ONE_TEXT))
            captured = records(hashlib.sha256(exact.read_bytes()).hexdigest())
            paths = []
            for index, value in enumerate(captured):
                identity = {'checkpoint_identity': pin['checkpoint_identity'],
                            'startup_nonce': str(index + 1) * 32,
                            'package_identity': 'a' * 64, 'metadata_identity': 'b' * 64,
                            'profile': {'tensor_parallel_size': 2, 'speculation': {'tokens': index}}}
                value['server'] = {'model': pin['served_model_name'],
                                   'health': {'serving': identity},
                                   'checkpoint': {k: identity[k] for k in (
                                       'checkpoint_identity', 'startup_nonce', 'package_identity', 'metadata_identity')}}
                if change:
                    change(index, value['server'])
                path = Path(directory) / f'{index}.json'
                path.write_text(json.dumps(value)); paths.append(str(path))
            args = argparse.Namespace(exact=str(exact), ar=paths[0], verify=[paths[1]])
            output = io.StringIO()
            with mock.patch('verify_exact.pins_exact', return_value=True), \
                    mock.patch('verify_exact.load_exact', return_value=ONE_TEXT), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit):
                    report(args)
            return json.loads(output.getvalue())

    def test_report_keeps_independent_authenticated_arm_profiles(self):
        value = self.compare_bound()
        self.assertEqual(value['bound']['ar']['startup_nonce'], '1' * 32)
        candidate, = value['bound']['candidates'].values()
        self.assertEqual(candidate['startup_nonce'], '2' * 32)
        self.assertEqual(candidate['profile']['tensor_parallel_size'], 2)
        self.assertNotIn('server', value)

    def test_report_refuses_health_or_alias_substitution_in_either_arm(self):
        for arm in (0, 1):
            for key in ('startup_nonce', 'checkpoint_identity', 'package_identity', 'metadata_identity', 'model'):
                def change(index, server):
                    if index == arm:
                        target = server if key == 'model' else server['health']['serving']
                        target[key] = 'foreign'
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    self.compare_bound(change)


class RowLabelTests(unittest.TestCase):
    def test_draft_counts_round_down_to_verify_buckets(self):
        self.assertEqual([draft_count(7, remaining) for remaining in range(1, 10)],
                         [0, 1, 1, 3, 3, 3, 3, 7, 7])
        self.assertEqual(draft_count(3, 20), 3)
        self.assertEqual(draft_count(0, 20), 0)

    def test_labels_cover_prefill_then_verify_blocks(self):
        labels = row_labels(3, 7)
        self.assertEqual([label["row"] for label in labels],
                         ["prefill"] + ["verify"] * 4 + ["verify"] * 2)
        self.assertEqual([label.get("depth") for label in labels], [None, 0, 1, 2, 3, 0, 1])
        self.assertEqual({label["row"] for label in row_labels(0, 5)[1:]}, {"decode"})
        self.assertEqual(expected_counters(labels, 7, 7),
                         {"verify_items": 2, "draft_tokens": 4, "rows": 6})
        # A mismatch on the last row of the first block ends drafting there.
        self.assertEqual(expected_counters(labels, 7, 4),
                         {"verify_items": 1, "draft_tokens": 3, "rows": 4})
        # A mismatch on the prefill row leaves the request undrafted.
        self.assertEqual(expected_counters(labels, 7, 0),
                         {"verify_items": 0, "draft_tokens": 0, "rows": 0})

    def test_shared_reference_prefixes_shorten_drafts(self):
        texts = [{"prompt_token_ids": [1, 2, 3, 4, 5, 6]}, {"prompt_token_ids": [1, 2, 3, 9]}]
        agreed = agreed_continuations(texts, texts[0]["prompt_token_ids"])
        self.assertEqual(agreed, [3, 2, 1, 0, 2, 1])
        labels = row_labels(3, 5, lambda emitted: agreed[1 + emitted])
        self.assertEqual([(label["row"], label.get("drafts")) for label in labels],
                         [("prefill", None), ("verify", 1), ("verify", 1),
                          ("verify", 1), ("verify", 1)])


class TopOneTests(unittest.TestCase):
    def test_reference_top1_is_read_from_its_logprob(self):
        top = {"a": -0.1, "b": -2.0}
        self.assertFalse(disagrees(row(-0.1, top), 5, 5, {}))
        self.assertTrue(disagrees(row(-2.0, top), 5, 5, {}))
        # Exact prefers another token: its string must be the server's top-1.
        self.assertFalse(disagrees(row(-2.0, top), 5, 9, {"9": "a"}))
        self.assertTrue(disagrees(row(-2.0, top), 5, 9, {"9": "c"}))
        self.assertTrue(disagrees(row(-0.1, top), 5, 9, {"9": "a"}))
        self.assertTrue(same_top1(row(-2.0, top), row(-1.5, {"a": -0.2})))
        self.assertFalse(same_top1(row(-0.1, top), row(-1.5, {"a": -0.2})))


MIMO_EXACT = {"texts": [{"name": "t", "category": "c", "score_from": 1,
                         "prompt_token_ids": [0, 5, 6, 7],
                         "exact": {"logprob": [-1.0, -2.0, -3.0], "top1": [5, 6, 8]}}]}
# chopinfer.qwen38.teacher-reference.v2 (the GPU runner's layout): no
# TP4-partials variant; bf16 and f64 are spread variants, not exact ones.
QWEN_EXACT = {"schema": "chopinfer.qwen38.teacher-reference.v2", "texts": [{
    "name": "t", "category": "c", "prompt_token_ids": [0, 5, 6, 7], "score_from": 1,
    "exact": {"logprob": [-1.0, -2.0, -3.0], "top1": [5, 6, 8], "margin": [1.0, 1.0, 1.0]},
    "bf16": {"logprob": [-1.1, -2.0, -3.1], "top1": [5, 6, 8], "margin": [0.9, 1.0, 1.1]}}]}
# One forced text, as the gate references score it.
ONE_TEXT = {"texts": [{"name": "t", "category": "c", "score_from": 1, "prompt_token_ids": [0, 5, 6, 7],
                       "exact": {"logprob": [-1.0, -2.0, -3.0], "top1": [5, 6, 8]}}]}
PREFILL = {"t": {"1": {"logprob": -1.0, "top1": 5}, "2": {"logprob": -2.1, "top1": 6},
                 "3": {"logprob": -3.0, "top1": 8}}}


def records(digest=None):
    """An AR and a K=1 record of text t, each from one request at score_from."""
    ar = {"arm": "ar", "k": 0, "concurrency": 1, "detokenized": {"8": "x"}, "prefill": PREFILL,
          "texts": [{"name": "t", "passes": [{"positions": {
              "1": row(-1.0, {"a": -1.0}, row="prefill"),
              "2": row(-2.0, {"b": -2.0}, row="decode"),
              "3": row(-3.0, {"x": -0.5}, row="decode")}}]}]}
    verify = {"arm": "k1", "k": 1, "concurrency": 1, "detokenized": {"8": "x"}, "prefill": PREFILL,
              "texts": [{"name": "t", "passes": [{"positions": {
                  "1": row(-1.0, {"a": -1.0}, row="prefill"),
                  "2": row(-2.5, {"b": -2.5}, row="verify", drafts=1, depth=0),
                  "3": row(-3.0, {"y": -0.5}, row="verify", drafts=1, depth=1)}}]}]}
    for recorded in (ar, verify):
        recorded["reference_sha256"] = digest
        recorded["texts"][0]["passes"][0]["requests"] = [
            {"start": 1, "maximum": 3, "emitted": 3, "matched": 3}]
    return ar, verify


def run_report(reference, bucket=None, bucket_evidence=None, ar_bucket_evidence=None,
               capacities=None, ar_capacities=None, observed=None, ar_observed=None):
    """report's JSON and exit code for records written against `reference`."""
    with tempfile.TemporaryDirectory() as directory:
        exact = Path(directory, "exact.json")
        exact.write_text(json.dumps(reference))
        ar, verify = records(hashlib.sha256(exact.read_bytes()).hexdigest())
        if bucket_evidence is not None:
            verify["decode_bucket"] = bucket_evidence
        if ar_bucket_evidence is not None:
            ar["decode_bucket"] = ar_bucket_evidence
        if observed is not None:
            verify["speculation"] = {"observed": observed}
        if ar_observed is not None:
            ar["speculation"] = {"observed": ar_observed}
        Path(directory, "ar.json").write_text(json.dumps(ar))
        Path(directory, "k1.json").write_text(json.dumps(verify))
        args = argparse.Namespace(exact=str(exact), ar=str(Path(directory, "ar.json")),
                                  verify=[str(Path(directory, "k1.json"))], decode_bucket=bucket,
                                  row_capacity=capacities, ar_row_capacity=ar_capacities)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                report(args)
            except SystemExit as exited:
                code = exited.code
        return json.loads(output.getvalue())["records"][args.verify[0]], code


def ar_draw(arm, logprobs, top=None):
    """A further AR record of text t with decode rows of these logprobs at 2 and 3."""
    ar, _ = records()
    ar["arm"] = arm
    positions = ar["texts"][0]["passes"][0]["positions"]
    for position, logprob in zip(("2", "3"), logprobs):
        positions[position]["logprob"] = logprob
        if top is not None:
            positions[position]["top"] = top
    return ar


def run_controls(reference, controls, candidate="verify", bucket=None, ar_evidence=None):
    """report's output and exit code for the AR record, `controls` and a
    candidate: the K=1 record, or with "decode" an AR record at C64."""
    with tempfile.TemporaryDirectory() as directory:
        exact = Path(directory, "exact.json")
        exact.write_text(json.dumps(reference))
        digest = hashlib.sha256(exact.read_bytes()).hexdigest()
        ar, verify = records(digest)
        if candidate == "decode":
            verify = {**copy.deepcopy(ar), "arm": "c64", "concurrency": 64}
        if ar_evidence is not None:
            ar["decode_bucket"] = ar_evidence
        paths = []
        for index, recorded in enumerate([ar, verify, *controls]):
            recorded = copy.deepcopy(recorded)
            if recorded["reference_sha256"] is None:
                recorded["reference_sha256"] = digest
            for text in recorded["texts"]:
                for scored in text["passes"]:
                    scored.setdefault("phase", 0)
            paths.append(str(Path(directory, f"{index}.json")))
            Path(paths[-1]).write_text(json.dumps(recorded))
        args = argparse.Namespace(exact=str(exact), ar=paths[0], verify=[paths[1]], control=paths[2:],
                                  decode_bucket=bucket)
        output = io.StringIO()
        code = None
        with contextlib.redirect_stdout(output), mock.patch("verify_exact.exact_pinned", return_value=False):
            try:
                report(args)
            except SystemExit as exited:
                if not isinstance(exited.code, int):
                    raise
                code = exited.code
        return json.loads(output.getvalue()), paths, code


class ForcingServer:
    """Forces each token on which every reference sharing the history agrees,
    as --debug-reference-tokens does; elsewhere it emits `sampled(position)`."""

    def __init__(self, references, sampled):
        self.references, self.sampled, self.prompts = references, sampled, []

    def request(self, path, body):
        history = list(body["prompt"])
        self.prompts.append(len(history))
        emitted = []
        for _ in range(body["max_tokens"]):
            matches = [ids for ids in self.references
                       if len(ids) > len(history) and ids[:len(history)] == history]
            following = {ids[len(history)] for ids in matches}
            token = following.pop() if len(following) == 1 else self.sampled(len(history))
            emitted.append(token)
            history.append(token)
        return {"choices": [{"token_ids": emitted, "finish_reason": "length", "logprobs": {
            "token_logprobs": [-0.5] * len(emitted), "top_logprobs": [{"x": -0.5}] * len(emitted)}}]}, 0.01


class RecordTests(unittest.TestCase):
    references = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 9, 9, 9], [0, 8, 8]]

    def record(self, sampled, phase=0):
        from verify_exact import record_pass
        text = {"name": "a", "prompt_token_ids": self.references[0], "score_from": 1}
        texts = [{"prompt_token_ids": ids} for ids in self.references]
        server = ForcingServer(self.references, sampled)
        recorded = record_pass(server, "M", text, 0, phase,
                               agreed_continuations(texts, text["prompt_token_ids"]))
        return recorded, server

    def test_requests_stop_before_positions_the_references_disagree_on(self):
        # The references diverge after [0] (position 1) and after [0, 1, 2] (position 3).
        recorded, server = self.record(lambda position: 99)
        self.assertEqual([(r["start"], r["maximum"]) for r in recorded["requests"]], [(2, 1), (4, 4)])
        self.assertEqual(server.prompts, [2, 4])
        self.assertEqual(list(recorded["positions"]), ["2", "4", "5", "6", "7"])
        self.assertEqual([row["row"] for row in recorded["positions"].values()],
                         ["prefill", "prefill", "decode", "decode", "decode"])

    def test_a_chance_match_at_an_ambiguous_position_scores_nothing_more(self):
        # A server that happens to emit the reference token where nothing is
        # forced must not change which positions the record scores.
        chance, _ = self.record(lambda position: self.references[0][position])
        unforced, _ = self.record(lambda position: 99)
        self.assertEqual(list(chance["positions"]), list(unforced["positions"]))

    def test_a_server_that_does_not_force_is_refused(self):
        from verify_exact import record_pass
        text = {"name": "a", "prompt_token_ids": [0, 1, 2, 3], "score_from": 1}

        class Free:
            def request(self, path, body):
                return {"choices": [{"token_ids": [7] * body["max_tokens"], "finish_reason": "length",
                                     "logprobs": {"token_logprobs": [-1.0] * body["max_tokens"],
                                                  "top_logprobs": [{}] * body["max_tokens"]}}]}, 0.01

        with self.assertRaisesRegex(RuntimeError, "left the reference"):
            record_pass(Free(), "M", text, 0, 0, agreed_continuations([text], text["prompt_token_ids"]))


class CoverageTests(unittest.TestCase):
    references = RecordTests.references

    def records(self, phases):
        from verify_exact import record_pass
        texts = [{"name": f"t{index}", "prompt_token_ids": ids, "score_from": 1}
                 for index, ids in enumerate(self.references)]
        server = ForcingServer(self.references, lambda position: 99)

        def recorded(arm, k, phases, pooled):
            passes = {text["name"]: [record_pass(server, "M", text, 0, phase,
                                                 agreed_continuations(texts, text["prompt_token_ids"]))
                                     for phase in phases] for text in texts}
            return {"arm": arm, "k": k, "pooled_decode": pooled,
                    "texts": [{"name": name, "passes": value} for name, value in passes.items()]}

        return {"texts": texts}, recorded("ar", 0, [0], False), recorded("k7", 7, phases, True)

    def test_decode_records_must_score_exactly_the_forced_positions(self):
        from verify_exact import coverage
        exact, ar, pooled = self.records([0, 1, 2])
        result = coverage(ar, pooled, exact, "decode", None)
        # t0 runs 2 and 4-7, t1 2 and 4-5, t2 2: decode rows at 5, 6, 7 and 5.
        self.assertEqual((result["expected"], result["scored"], result["passed"]), (4, 4, True))
        dropped = copy.deepcopy(pooled)
        dropped["texts"][0]["passes"][0]["positions"].pop("6")
        for scored in dropped["texts"][0]["passes"][1:]:
            scored["positions"].pop("6", None)
        result = coverage(ar, dropped, exact, "decode", None)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing"], [["t0", 6]])
        # A row at the unforced position 3 in both records is not one forcing reaches.
        extra_ar, extra = copy.deepcopy(ar), copy.deepcopy(pooled)
        for recorded in (extra_ar, extra):
            recorded["texts"][0]["passes"][0]["positions"]["3"] = {"logprob": -1.0, "top": {}, "row": "decode"}
        result = coverage(extra_ar, extra, exact, "decode", None)
        self.assertEqual(result["unexpected"], [["t0", 3]])
        self.assertFalse(result["passed"])

    def test_host_drafted_verify_rows_keep_the_minimum(self):
        from verify_exact import HOST_DRAFTED_MINIMUM_POSITIONS, coverage
        self.assertEqual(HOST_DRAFTED_MINIMUM_POSITIONS, 4108)
        for positions, passed in ((4107, False), (4108, True)):
            result = coverage(None, None, None, "verify", {"all": {"positions": positions}})
            self.assertEqual((result["rule"], result["passed"]), ("host-drafted", passed))


class CompareTests(unittest.TestCase):
    def test_pooled_mtp_rows_have_no_invented_verify_depths(self):
        from verify_exact import relabel
        ar, mixed = records()
        mixed.update(k=3, pooled_decode=True)
        exact = copy.deepcopy(QWEN_EXACT)
        relabel(mixed, exact)
        positions = mixed["texts"][0]["passes"][0]["positions"]
        self.assertEqual([r["row"] for r in positions.values()], ["prefill", "decode", "decode"])
        self.assertTrue(all("depth" not in r and "drafts" not in r for r in positions.values()))
        result = compare(ar, mixed, exact, kind="decode")
        self.assertEqual(result["all"]["positions"], 2)
        self.assertFalse(result["all"]["gate_against_ar"]["passed"])


    def test_depths_pair_verify_with_ar_decode_at_the_same_position(self):
        exact = MIMO_EXACT
        ar, verify = records()
        report = compare(ar, verify, exact)
        depth0 = report["by_depth"]["0"]
        self.assertEqual(depth0["positions"], 1)
        self.assertAlmostEqual(depth0["mean_abs_dlogprob"]["row"], 0.5)
        self.assertAlmostEqual(depth0["mean_abs_dlogprob"]["ar"], 0.0)
        self.assertAlmostEqual(depth0["mean_abs_dlogprob"]["prefill"], 0.1)
        self.assertEqual(report["by_depth"]["1"]["top1_counts"]["row_flip"], 1)
        self.assertEqual(report["by_depth"]["1"]["top1_counts"]["ar_flip"], 0)
        self.assertEqual(report["by_depth"]["1"]["top1_counts"]["row_vs_ar_flip"], 1)
        self.assertEqual(report["all"]["positions"], 2)
        self.assertEqual(report["all"]["gate_against_ar"]["top1_server_only"], 1)
        self.assertEqual(report["prefill_last_row"]["positions"], 1)
        # An AR record compared with itself pairs every decode row exactly.
        same = compare(ar, ar, exact, "decode")
        self.assertEqual(same["all"]["positions"], 2)
        self.assertEqual(same["all"]["mean_abs_dlogprob"]["row_vs_ar"], 0)


class ReferenceTests(unittest.TestCase):
    def test_qwen_reference_scores_like_the_mimo_reference(self):
        qwen = copy.deepcopy(QWEN_EXACT)
        self.assertEqual(variants(qwen), ["exact"])
        self.assertEqual(variants(MIMO_EXACT), ["exact"])
        ar, verify = records()
        self.assertEqual(compare(ar, verify, qwen), compare(ar, verify, MIMO_EXACT))

    def test_report_compares_only_the_variants_the_reference_has(self):
        mimo, mimo_code = run_report(MIMO_EXACT)
        qwen, qwen_code = run_report(QWEN_EXACT)
        self.assertEqual(list(mimo), ["arm", "k", "concurrency", "kind", "pooled_decode", "speculation", "exact",
                                      "coverage", "equal_top1_to_ar"])
        self.assertEqual(list(qwen), ["arm", "k", "concurrency", "kind", "pooled_decode", "speculation", "exact",
                                      "coverage", "equal_top1_to_ar"])
        # The records' last verify row picks y where AR picks x.
        self.assertFalse(qwen["equal_top1_to_ar"])
        self.assertEqual(qwen["exact"], mimo["exact"])
        self.assertEqual(qwen["exact"]["by_depth"]["0"]["positions"], 1)
        self.assertEqual(qwen_code, mimo_code)

    def test_report_refuses_records_of_another_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory, "exact.json")
            exact.write_text(json.dumps(QWEN_EXACT))
            ar, verify = records(hashlib.sha256(EXACT.read_bytes()).hexdigest())
            Path(directory, "ar.json").write_text(json.dumps(ar))
            Path(directory, "k1.json").write_text(json.dumps(verify))
            args = argparse.Namespace(exact=str(exact), ar=str(Path(directory, "ar.json")),
                                      verify=[str(Path(directory, "k1.json"))])
            with self.assertRaisesRegex(SystemExit, "different exact reference"):
                report(args)

    def test_exact_option_resolves_paths_and_config(self):
        self.assertEqual(exact_path(None), EXACT)
        self.assertEqual(exact_path("some/file.json"), Path("some/file.json"))
        with mock.patch.dict(os.environ, {"CHOPINFER_DATA": "/data/example"}):
            self.assertEqual(exact_path("qwen38"),
                             Path(__file__).resolve().parents[2] / "fixtures/qwen38/teacher-reference.json")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CHOPINFER_QUAL_CONFIG"):
                exact_path(None)





class EqualityTests(unittest.TestCase):
    """The relative gate can pass verify rows whose
    greedy choice differs from AR's; the strict checks must not."""

    def test_the_strict_check_refuses_a_differing_top1_the_relative_gate_passes(self):
        from verify_exact import require_equal_top1
        exact = copy.deepcopy(QWEN_EXACT)
        exact["texts"][0]["exact"]["top1"][2] = 9
        ar, verify = records()
        for record in (ar, verify):
            record["detokenized"]["9"] = "z"
        verify["texts"][0]["passes"][0]["positions"]["2"]["logprob"] = -2.0
        verify["texts"][0]["passes"][0]["positions"]["2"]["top"] = {"b": -2.0}
        result = compare(ar, verify, exact)
        self.assertTrue(result["all"]["gate_against_ar"]["passed"])
        self.assertFalse(require_equal_top1(result))
        _, same = records()
        same["texts"][0]["passes"][0]["positions"]["3"]["top"] = {"x": -0.5}
        self.assertTrue(require_equal_top1(compare(ar, same, exact)))

    def test_greedy_sequences_must_match_token_for_token(self):
        import kinds
        ar = {"requests": {"a": [1, 2, 3], "b": [4, 5]},
              "reference_sha256": "reference", "prompt_tokens": 3, "max_tokens": 3, "concurrency": 1,
              "finish_reasons": {"a": "length", "b": "stop"}}
        def differences(other):
            return kinds.KINDS["greedy_pin"].score(other, ar)["first_differences"]
        self.assertEqual(differences(copy.deepcopy(ar)), [])
        other = {**ar, "requests": {"a": [1, 2, 7], "b": [4, 5]}}
        self.assertEqual(differences(other), [{"request": "a", "position": 2, "ar_token": 3, "other_token": 7}])
        # A shorter sequence differs where it ends.
        short = {**ar, "requests": {"a": [1, 2, 3], "b": [4]}}
        self.assertEqual(differences(short), [{"request": "b", "position": 1, "ar_token": 5, "other_token": None}])
        with self.assertRaises(ValueError):
            differences({**ar, "requests": {"a": [1, 2, 3]}})


class ControlTests(unittest.TestCase):
    """--control: the gate compares against the mean of several AR draws."""

    def test_one_control_equal_to_ar_reproduces_the_single_control_report(self):
        for candidate in ("verify", "decode"):
            today, _, today_code = run_controls(ONE_TEXT, [], candidate)
            ar, _ = records()
            output, _, code = run_controls(ONE_TEXT, [ar], candidate)
            record, single = output["records"], today["records"]
            [record], [single] = record.values(), single.values()
            self.assertEqual(code, today_code)
            self.assertEqual(record["exact"], single["exact"])
            self.assertEqual(record["single_control"], single["exact"]["all"])
            self.assertEqual({key: value for key, value in record["coverage"].items() if key != "controls"},
                             single["coverage"])
            self.assertEqual({key: value for key, value in record.items()
                              if key not in ("controls", "single_control", "coverage")},
                             {key: value for key, value in single.items() if key != "coverage"})
            self.assertNotIn("controls", single)
            self.assertNotIn("single_control", single)
            self.assertNotIn("controls", single["coverage"])

    def test_controls_average_the_ar_error_and_top1_keeps_ar(self):
        # AR errors at positions 2 and 3: 0 and 0; the controls' 0.2, 0.4 and
        # 0.4, 0. The second control also picks another top-1 at both.
        controls = [ar_draw("b8", [-2.2, -3.4]), ar_draw("b16", [-2.4, -3.0], {"z": -0.1})]
        for count in (2, 3):
            output, paths, _ = run_controls(ONE_TEXT, controls[:count - 1])
            [record] = output["records"].values()
            errors = [[0.0, 0.0], [0.2, 0.4], [0.4, 0.0]][:count]
            # Position 2 is a depth-0 verify row and position 3 a depth-1 one.
            for depth in (0, 1):
                self.assertAlmostEqual(record["exact"]["by_depth"][str(depth)]["mean_abs_dlogprob"]["ar"],
                                       sum(draw[depth] for draw in errors) / count)
            pooled = record["exact"]["all"]
            self.assertAlmostEqual(pooled["mean_abs_dlogprob"]["ar"], sum(map(sum, errors)) / (2 * count))
            self.assertAlmostEqual(pooled["gate_against_ar"]["mean_abs_dlogprob_gap"],
                                   pooled["mean_abs_dlogprob"]["row"] - pooled["mean_abs_dlogprob"]["ar"])
            # The top-1 counts still read --ar's rows.
            self.assertEqual(pooled["top1_counts"], record["single_control"]["top1_counts"])
            self.assertEqual(pooled["top1_counts"]["ar_flip"], 0)
            self.assertAlmostEqual(record["single_control"]["mean_abs_dlogprob"]["ar"], 0.0)
            draws = record["controls"]["draws"]
            self.assertEqual([draw["path"] for draw in draws], paths[:1] + paths[2:])
            self.assertEqual([draw["role"] for draw in draws], ["ar"] + ["control"] * (count - 1))
            self.assertEqual([draw["arm"] for draw in draws], ["ar", "b8", "b16"][:count])
            for draw, expected in zip(draws, errors):
                self.assertAlmostEqual(draw["mean_abs_dlogprob"], sum(expected) / 2)
                self.assertIsNone(draw["decode_bucket"])
            self.assertAlmostEqual(record["controls"]["mean_abs_dlogprob"], pooled["mean_abs_dlogprob"]["ar"])
            self.assertEqual(record["controls"]["positions"], 2)

    def test_controls_that_are_not_clean_ar_draws_are_refused(self):
        drafted = ar_draw("b8", [-2.0, -3.0])
        drafted["speculation"] = {"observed": {"draft_tokens": 3, "verify_items": 1}}
        speculative = {**ar_draw("b8", [-2.0, -3.0]), "k": 1}
        pooled = {**ar_draw("b8", [-2.0, -3.0]), "pooled_decode": True}
        for control in (drafted, speculative, pooled):
            with self.assertRaisesRegex(SystemExit, "not an AR record"):
                run_controls(ONE_TEXT, [control])
        other = {**ar_draw("b8", [-2.0, -3.0]), "reference_sha256": "another"}
        with self.assertRaisesRegex(SystemExit, "different exact reference"):
            run_controls(ONE_TEXT, [other])

    def test_an_unbound_control_is_refused_for_the_pinned_reference(self):
        control = {**ar_draw("b8", [-2.0, -3.0]), "server": {"live": False}}
        with mock.patch("verify_exact.pins_exact", return_value=True), \
                mock.patch("verify_exact.load_exact", side_effect=lambda path: ONE_TEXT), \
                mock.patch("verify_exact.bound_live", side_effect=lambda server: server["live"]):
            with self.assertRaisesRegex(SystemExit, "bound live"):
                run_controls(ONE_TEXT, [control])

    def test_a_control_missing_a_forced_position_fails_coverage(self):
        missing = ar_draw("b8", [-2.0, -3.0])
        missing["texts"][0]["passes"][0]["positions"].pop("3")
        missing["texts"][0]["passes"][0]["requests"] = [{"start": 1, "maximum": 2, "emitted": 2, "matched": 2}]
        for candidate in ("decode", "verify"):
            output, paths, code = run_controls(ONE_TEXT, [missing], candidate)
            [record] = output["records"].values()
            self.assertFalse(record["coverage"]["passed"])
            self.assertEqual(record["coverage"]["controls"],
                             [{"path": paths[2], "scored": 1, "missing": [["t", 3]]}])
            self.assertEqual(code, 1)
            # Only the position every draw scored pairs.
            self.assertEqual(record["exact"]["all"]["positions"], 1)
        output, _, code = run_controls(ONE_TEXT, [ar_draw("b8", [-2.0, -3.0])], "decode")
        [record] = output["records"].values()
        self.assertTrue(record["coverage"]["passed"])
        self.assertEqual(code, 0)

    def test_a_control_without_shared_positions_fails_cleanly(self):
        empty = ar_draw("b8", [-2.0, -3.0])
        for position in ("2", "3"):
            empty["texts"][0]["passes"][0]["positions"].pop(position)
        empty["texts"][0]["passes"][0]["requests"] = [{"start": 1, "maximum": 1, "emitted": 1, "matched": 1}]
        for candidate in ("decode", "verify"):
            output, _, code = run_controls(ONE_TEXT, [empty], candidate)
            [record] = output["records"].values()
            self.assertEqual(code, 1)
            self.assertEqual(record["exact"]["all"], {"positions": 0})
            self.assertFalse(record["coverage"]["passed"])
            self.assertEqual(record["controls"]["positions"], 0)
            self.assertIsNone(record["controls"]["mean_abs_dlogprob"])

    def test_each_control_bucket_is_its_own_and_gates_when_buckets_are_required(self):
        evidence = lambda bucket: {"bucket": bucket, "passed": True, "steps": 1, "invalid_batches": []}
        good = {**ar_draw("b8", [-2.0, -3.0]), "decode_bucket": evidence(8)}
        failed = {**ar_draw("b16", [-2.0, -3.0]), "decode_bucket": {**evidence(16), "passed": False}}
        absent = ar_draw("c1", [-2.0, -3.0])
        comparison = {"all": {"positions": 2, "gate_against_ar": {"passed": True},
                              "top1_counts": {"row_vs_ar_flip": 0}}}
        with mock.patch("verify_exact.coverage", return_value={"passed": True}):
            output, _, code = run_controls(ONE_TEXT, [good, failed, absent], "decode", 64)
            [record] = output["records"].values()
            self.assertEqual([(draw["decode_bucket"], draw.get("bucket_covered"))
                              for draw in record["controls"]["draws"]],
                             [(None, None), (8, True), (16, False), (None, False)])
            self.assertEqual(code, 1)
            with mock.patch("verify_exact.compare", return_value=comparison), \
                    mock.patch("verify_exact.bucket_matches",
                               side_effect=lambda recorded, bucket: recorded.get("arm") != "b16"):
                _, _, code = run_controls(ONE_TEXT, [good, failed], "decode", 64)
                self.assertEqual(code, 1)
                _, _, code = run_controls(ONE_TEXT, [good], "decode", 64)
                self.assertEqual(code, 0)
        # Without --decode-bucket a control's bucket is reported, not gated.
        output, _, code = run_controls(ONE_TEXT, [good, failed], "decode")
        [record] = output["records"].values()
        self.assertEqual([draw["decode_bucket"] for draw in record["controls"]["draws"]], [None, 8, 16])
        self.assertTrue(all("bucket_covered" not in draw for draw in record["controls"]["draws"]))
        self.assertEqual(code, 0)


def pooled_text(count):
    """One text whose `count` positions after the prefill row are each exact's top-1."""
    ids = [0] + [5 + index for index in range(count + 1)]
    return {"texts": [{"name": "t", "category": "c", "score_from": 1, "prompt_token_ids": ids,
                       "exact": {"logprob": [-1.0] * (count + 1), "top1": ids[1:]}}]}


def pooled_record(arm, errors, flips=0, k=7):
    """A pooled-decode record (an AR one with k=0) of pooled_text(len(errors)):
    decode rows these errors from exact, the first `flips` with another top-1."""
    ids = pooled_text(len(errors))["texts"][0]["prompt_token_ids"]
    positions = {"1": row(-1.0, {"a": -2.0}, row="prefill")}
    for index, error in enumerate(errors):
        logprob = -1.0 - error
        positions[str(index + 2)] = row(logprob, {"z": logprob + 1} if index < flips else {"a": logprob - 1},
                                        row="decode")
    count = len(positions)
    return {"arm": arm, "k": k, "pooled_decode": bool(k), "concurrency": 8, "detokenized": {},
            "prefill": {"t": {str(position): {"logprob": -1.0, "top1": ids[position]}
                              for position in range(1, count + 1)}},
            "texts": [{"name": "t", "passes": [{"phase": 0, "positions": positions, "requests": [
                {"start": 1, "maximum": count, "emitted": count, "matched": count}]}]}]}


def run_pool(candidates, controls=(), pool=True):
    """report's output and exit code for an AR record with zero error, `controls`
    and the `candidates`, pooled with --pool-verify unless `pool` is false."""
    with tempfile.TemporaryDirectory() as directory:
        count = max(int(position) for position in candidates[0]["texts"][0]["passes"][0]["positions"]) - 1
        exact = Path(directory, "exact.json")
        exact.write_text(json.dumps(pooled_text(count)))
        digest = hashlib.sha256(exact.read_bytes()).hexdigest()
        paths = []
        for index, recorded in enumerate([pooled_record("ar", [0.0] * count, k=0), *candidates, *controls]):
            paths.append(str(Path(directory, f"{index}.json")))
            Path(paths[-1]).write_text(json.dumps({**recorded, "reference_sha256": digest}))
        verify = paths[1:1 + len(candidates)]
        args = argparse.Namespace(exact=str(exact), ar=paths[0], verify=verify,
                                  control=paths[1 + len(candidates):], pool_verify=pool)
        output = io.StringIO()
        code = None
        with contextlib.redirect_stdout(output), mock.patch("verify_exact.exact_pinned", return_value=False):
            try:
                report(args)
            except SystemExit as exited:
                if not isinstance(exited.code, int):
                    raise
                code = exited.code
        # Name records apart from the temporary directory, so runs compare.
        return (json.loads(output.getvalue().replace(directory, "records")),
                [path.replace(directory, "records") for path in verify], code)


class PoolTests(unittest.TestCase):
    """--pool-verify: the mean rule reads the mean over several candidate draws."""

    def test_one_pooled_record_reproduces_the_single_record_report(self):
        control = pooled_record("b8", [0.002, 0.0], k=0)
        for errors in ([0.004, 0.0], [0.012, 0.0]):
            single, _, single_code = run_pool([pooled_record("c8", errors)], [control], pool=False)
            output, _, code = run_pool([pooled_record("c8", errors)], [control])
            self.assertEqual(code, single_code)
            self.assertEqual(output["records"], single["records"])
            self.assertNotIn("pooled_candidate", single)
            [record] = output["records"].values()
            pooled = output["pooled_candidate"]
            self.assertEqual(pooled["row_minus_ar"], record["exact"]["all"]["row_minus_ar"])
            self.assertEqual(pooled["row_minus_ar"]["mean"],
                             record["exact"]["all"]["gate_against_ar"]["mean_abs_dlogprob_gap"])
            self.assertEqual(pooled["passed"], code == 0)
        self.assertEqual(single_code, 1)

    def test_pooling_averages_row_errors_per_position(self):
        # AR errors 0 and 0, the control's 0.002 and 0: the AR mean is 0.001 and 0.
        control = pooled_record("b8", [0.002, 0.0], k=0)
        draws = [pooled_record("c8", [0.004, 0.0]), pooled_record("c16", [0.0, 0.008])]
        output, paths, code = run_pool(draws, [control])
        pooled = output["pooled_candidate"]
        self.assertEqual(code, 0)
        self.assertEqual(pooled["positions"], 2)
        self.assertEqual([(draw["path"], draw["arm"]) for draw in pooled["draws"]],
                         list(zip(paths, ["c8", "c16"])))
        for draw, expected in zip(pooled["draws"], (0.002, 0.004)):
            self.assertAlmostEqual(draw["mean_abs_dlogprob"], expected)
        self.assertAlmostEqual(pooled["mean_abs_dlogprob"]["candidate"], 0.003)
        self.assertAlmostEqual(pooled["mean_abs_dlogprob"]["ar"], 0.0005)
        self.assertAlmostEqual(pooled["row_minus_ar"]["mean"], ((0.002 - 0.001) + (0.004 - 0.0)) / 2)
        self.assertIsNone(pooled["row_minus_ar"]["ci95"])
        # Only the positions every draw pairs are pooled; the short draw fails coverage.
        short = pooled_record("c32", [0.0, 0.0])
        short["texts"][0]["passes"][0]["positions"].pop("3")
        short["texts"][0]["passes"][0]["requests"] = [{"start": 1, "maximum": 2, "emitted": 2, "matched": 2}]
        output, _, code = run_pool([*draws, short], [control])
        self.assertEqual(output["pooled_candidate"]["positions"], 1)
        self.assertAlmostEqual(output["pooled_candidate"]["row_minus_ar"]["mean"], 0.004 / 3 - 0.001)
        self.assertEqual(code, 1)

    def test_a_draw_above_the_margin_passes_when_the_mean_is_within_it(self):
        above, clean = pooled_record("c32", [0.016, 0.0]), pooled_record("c8", [0.0, 0.0])
        output, _, code = run_pool([above, clean])
        gaps = [record["exact"]["all"]["gate_against_ar"] for record in output["records"].values()]
        self.assertEqual([gate["passed"] for gate in gaps], [False, True])
        self.assertAlmostEqual(output["pooled_candidate"]["row_minus_ar"]["mean"], 0.004)
        self.assertTrue(output["pooled_candidate"]["passed"])
        self.assertEqual(code, 0)
        self.assertEqual(run_pool([above, clean], pool=False)[2], 1)
        output, _, code = run_pool([above, pooled_record("c8", [0.012, 0.0])])
        self.assertAlmostEqual(output["pooled_candidate"]["row_minus_ar"]["mean"], 0.007)
        self.assertFalse(output["pooled_candidate"]["passed"])
        self.assertEqual(code, 1)

    def test_a_draw_failing_its_top_one_bound_fails_the_pooled_report(self):
        # Six flips against none: above 2 * sqrt(6).
        flipped, clean = pooled_record("c32", [0.0] * 6, flips=6), pooled_record("c8", [0.0] * 6)
        output, _, code = run_pool([flipped, clean])
        self.assertTrue(output["pooled_candidate"]["passed"])
        gate = output["records"][next(iter(output["records"]))]["exact"]["all"]["gate_against_ar"]
        self.assertEqual((gate["top1_server_only"], gate["top1_baseline_only"]), (6, 0))
        self.assertEqual(code, 1)
        self.assertEqual(run_pool([pooled_record("c32", [0.0] * 6, flips=4), clean])[2], 0)

    def test_records_of_different_row_kinds_are_refused(self):
        verify = {**pooled_record("k1", [0.0, 0.0], k=1), "pooled_decode": False}
        with self.assertRaisesRegex(SystemExit, "one row kind"):
            run_pool([pooled_record("c8", [0.0, 0.0]), verify])


class TracingServer:
    """Forces the reference continuation of one text and appends a decode
    step of row capacity `capacity` to the server log for each request."""

    def __init__(self, ids, log, capacity):
        self.ids, self.log, self.capacity = ids, log, capacity

    def identity(self, model):
        return {"model": "M"}

    def request(self, path, body):
        if path == "/detokenize":
            return {"prompt": "x"}, 0.01
        if body.get("echo"):
            rows = [None] + [{str(token): {"logprob": -1.0, "rank": 1}} for token in self.ids[1:]]
            return {"choices": [{"prompt_logprobs": rows}]}, 0.01
        start = len(body["prompt"])
        emitted = self.ids[start:start + body["max_tokens"]]
        with self.log.open("a") as log:
            log.write(json.dumps({"schema": "chopinfer.step.v1", "decode_rows": 5, "verify_items": 0,
                                  "row_capacity": self.capacity}) + "\n")
        return {"choices": [{"token_ids": emitted, "finish_reason": "length", "logprobs": {
            "token_logprobs": [-1.0] * len(emitted), "top_logprobs": [{"x": -1.0}] * len(emitted)}}]}, 0.01


class SmallBucketTests(unittest.TestCase):
    """B8 and B16 AR draws carry bucket evidence through the command line."""

    def main(self, argv):
        import verify_exact
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["verify_exact.py", *argv]), contextlib.redirect_stdout(output):
            try:
                verify_exact.main()
            except SystemExit as exited:
                return exited.code, output.getvalue()
        return None, output.getvalue()

    def record(self, directory, exact, arm, bucket, capacity):
        log = Path(directory, f"{arm}.log")
        log.write_text("startup\n")
        server = TracingServer(ONE_TEXT["texts"][0]["prompt_token_ids"], log, capacity)
        with mock.patch("verify_exact.Client", lambda url, timeout: server), \
                mock.patch("verify_exact.metrics_text", return_value=""), \
                mock.patch("verify_exact.snapshot", return_value={}), \
                mock.patch("verify_exact.difference", return_value={}):
            code, output = self.main(["record", "--exact", str(exact), "--arm", arm, "--k", "0",
                                      "--decode-bucket", str(bucket), "--server-log", str(log)])
        receipt = json.loads(output.splitlines()[-1])["receipt"]
        return code, receipt

    def test_record_and_report_with_eight_and_sixteen_row_controls(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("verify_exact.exact_pinned", return_value=False):
            exact = Path(directory, "exact.json")
            exact.write_text(json.dumps(ONE_TEXT))
            recorded = {arm: self.record(directory, exact, arm, bucket, capacity)
                        for arm, bucket, capacity in (("c32", 32, 32), ("b64", 64, 64), ("b8", 8, 8),
                                                      ("b16", 16, 16), ("b16-wrong", 16, 8))}
            self.assertEqual({arm: code for arm, (code, _) in recorded.items()},
                             {"c32": 0, "b64": 0, "b8": 0, "b16": 0, "b16-wrong": 1})
            evidence = json.loads(Path(recorded["b8"][1]).read_text())["decode_bucket"]
            self.assertEqual((evidence["bucket"], evidence["steps"], evidence["passed"]), (8, 1, True))

            def report(*controls):
                argv = ["report", "--exact", str(exact), "--ar", recorded["c32"][1],
                        "--verify", recorded["b64"][1], "--decode-bucket", "64"]
                for control in controls:
                    argv += ["--control", recorded[control][1]]
                code, output = self.main(argv)
                [record] = json.loads(output)["records"].values()
                return code, [(draw["decode_bucket"], draw.get("bucket_covered"))
                              for draw in record["controls"]["draws"]]

            self.assertEqual(report("b8", "b16"), (0, [(32, None), (8, True), (16, True)]))
            self.assertEqual(report("b8", "b16-wrong"), (1, [(32, None), (8, True), (16, False)]))
            # An 8-row AR record can itself be the --ar control of an 8-row bucket gate.
            code, _ = self.main(["report", "--exact", str(exact), "--ar", recorded["b8"][1],
                                 "--verify", recorded["b16"][1], "--decode-bucket", "16",
                                 "--ar-decode-bucket", "8"])
            self.assertEqual(code, 0)


class DecodeBucketTests(unittest.TestCase):
    def coverage(self, steps, bucket=64):
        from verify_exact import decode_bucket_coverage
        return decode_bucket_coverage(steps, bucket)

    def test_every_decode_step_must_use_the_requested_bucket(self):
        steps = [{"decode_rows": 40, "verify_items": 0, "row_capacity": 64},
                 {"decode_rows": 64, "verify_items": 0, "row_capacity": 64}]
        self.assertTrue(self.coverage(steps)["passed"])
        for capacity in (None, 32, 128):
            self.assertFalse(self.coverage(steps + [{"decode_rows": 40, "row_capacity": capacity}])["passed"])

    def test_executed_capacity_is_authoritative_when_fewer_rows_are_active(self):
        self.assertTrue(self.coverage([{"decode_rows": 1, "row_capacity": 64}])["passed"])

    def test_invalid_utf8_does_not_discard_valid_trace_events(self):
        from verify_exact import decode_bucket_trace
        trace = b"invalid log byte: \xff\n" + json.dumps({"schema": "chopinfer.step.v1",
                "decode_rows": 40, "verify_items": 0, "row_capacity": 64}).encode() + b"\n"
        result = decode_bucket_trace(trace, 64)
        self.assertTrue(result["passed"])
        self.assertEqual(result["sha256"], hashlib.sha256(trace).hexdigest())

    def test_empty_prefill_only_and_mixed_verify_traces_do_not_prove_decode(self):
        for steps in ([], [{"decode_rows": 0, "prefill_tokens": 2048}],
                      [{"decode_rows": 40, "verify_items": 1}]):
            self.assertFalse(self.coverage(steps)["passed"])

    def test_prefill_steps_can_coexist_with_a_held_decode_bucket(self):
        self.assertTrue(self.coverage([{"decode_rows": 0, "prefill_tokens": 64},
                                       {"decode_rows": 72, "prefill_tokens": 2048, "row_capacity": None},
                                       {"decode_rows": 72, "prefill_tokens": 0, "row_capacity": 128}], 128)["passed"])

    def test_mixed_prefill_cannot_establish_a_single_executed_decode_capacity(self):
        mixed = {"decode_rows": 40, "prefill_tokens": 2048, "row_capacity": None}
        result = self.coverage([mixed])
        self.assertFalse(result["passed"])
        self.assertEqual(result["mixed_prefill_steps"], 1)
        self.assertFalse(self.coverage([mixed, {"decode_rows": 40, "row_capacity": 32}])["passed"])

    def test_report_requires_the_control_bucket_as_well_as_the_candidate(self):
        evidence = lambda bucket: {"bucket": bucket, "passed": True, "steps": 1, "invalid_batches": []}
        reference = {"texts": [{"name": "t", "category": "c", "score_from": 1,
                                "prompt_token_ids": [0, 5, 6, 7],
                                "exact": {"logprob": [-1.0, -2.0, -3.0], "top1": [5, 6, 8]}}]}
        comparison = {"all": {"positions": 2, "gate_against_ar": {"passed": True},
                              "top1_counts": {"row_vs_ar_flip": 0}}}
        with mock.patch("verify_exact.compare", return_value=comparison), \
                mock.patch("verify_exact.coverage", return_value={"passed": True}), \
                mock.patch("verify_exact.exact_pinned", return_value=False):
            for control in (None, evidence(64), {**evidence(32), "steps": 0},
                            {**evidence(32), "passed": False}):
                _, code = run_report(reference, 64, evidence(64), control)
                self.assertEqual(code, 1)
            result, code = run_report(reference, 64, evidence(64), evidence(32))
        self.assertEqual(code, 0)
        self.assertTrue(result["ar_bucket_covered"])

    def test_report_refuses_missing_or_wrong_bucket_evidence(self):
        from verify_exact import bucket_matches
        for recorded in ({}, {"decode_bucket": {"bucket": 128, "passed": True}},
                         {"decode_bucket": {"bucket": 64, "passed": False}},
                         {"decode_bucket": {"bucket": 64, "passed": True, "steps": 0,
                                            "invalid_batches": []}}):
            self.assertFalse(bucket_matches(recorded, 64))
        self.assertTrue(bucket_matches({"decode_bucket": {"bucket": 64, "passed": True,
                                                          "steps": 1, "invalid_batches": []}}, 64))
        self.assertTrue(bucket_matches({}, None))

    def test_report_exposes_missing_trace_as_failed_coverage(self):
        reference = {"texts": [{"name": "t", "category": "c", "score_from": 1,
                                "prompt_token_ids": [0, 5, 6, 7],
                                "exact": {"logprob": [-1.0, -2.0, -3.0], "top1": [5, 6, 8]}}]}
        comparison = {"all": {"positions": 2, "gate_against_ar": {"passed": True},
                              "top1_counts": {"row_vs_ar_flip": 0}}}
        with mock.patch("verify_exact.compare", return_value=comparison), \
                mock.patch("verify_exact.coverage", return_value={"passed": True}), \
                mock.patch("verify_exact.exact_pinned", return_value=False):
            _, ungated = run_report(reference)
            result, code = run_report(reference, bucket=64)
        self.assertEqual(ungated, 0)
        self.assertEqual(code, 1)
        self.assertFalse(result["bucket_covered"])
        self.assertIsNone(result["decode_bucket"])


class RowCapacityTests(unittest.TestCase):
    @staticmethod
    def observed(verify=0, capacity=8):
        return {"verify_items": 1 if verify else 0, "draft_tokens": verify - 1 if verify else 0,
                "step_rows": {"available": True, "dropped": 0,
                              "bins": [{"decode_rows": 0 if verify else 1, "verify_rows": verify,
                                        "row_capacity": capacity, "steps": 3}]}}

    def test_actual_capacity_is_required_for_each_arm(self):
        from verify_exact import row_capacity_coverage
        self.assertTrue(row_capacity_coverage(self.observed(), [8], False)["passed"])
        self.assertTrue(row_capacity_coverage(self.observed(4), [8], True)["passed"])
        self.assertFalse(row_capacity_coverage(self.observed(), [8], True)["passed"])
        self.assertFalse(row_capacity_coverage(self.observed(4), [8], False)["passed"])
        self.assertEqual(row_capacity_coverage(self.observed(), [8, 16], False)["missing"], [16])

    def test_incomplete_or_malformed_histograms_cannot_prove_capacity(self):
        from verify_exact import row_capacity_coverage
        for observed in ({}, {"step_rows": {"available": False}}, self.observed(capacity=None),
                         self.observed(capacity=True), self.observed(capacity=0)):
            self.assertFalse(row_capacity_coverage(observed, [8], False)["passed"])
        for key, value in (("dropped", 1), ("bins", [])):
            observed = self.observed()
            observed["step_rows"][key] = value
            self.assertFalse(row_capacity_coverage(observed, [8], False)["passed"])
        for key, value in (("steps", -1), ("steps", True), ("decode_rows", 9)):
            observed = self.observed()
            observed["step_rows"]["bins"][0][key] = value
            self.assertFalse(row_capacity_coverage(observed, [8], False)["passed"])

    def test_report_recomputes_required_capacity_from_recorded_observations(self):
        comparison = {"all": {"positions": 2, "gate_against_ar": {"passed": True},
                              "top1_counts": {"row_vs_ar_flip": 0}}}
        with mock.patch("verify_exact.compare", return_value=comparison), \
                mock.patch("verify_exact.coverage", return_value={"passed": True}), \
                mock.patch("verify_exact.exact_pinned", return_value=False):
            for observed, ar_observed, expected in ((None, None, 1),
                    (self.observed(4), self.observed(capacity=16), 1),
                    (self.observed(4), self.observed(), 0)):
                row, code = run_report(ONE_TEXT, capacities=[8], ar_capacities=[8],
                                       observed=observed, ar_observed=ar_observed)
                self.assertEqual(code, expected)
                self.assertIn("row_capacity_coverage", row)


if __name__ == "__main__":
    unittest.main()
