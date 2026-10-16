"""Offline capture and multimodal qualification checks with synthetic API data."""

from qual_test_config import setUpModule
from vectors import top_columns

import hashlib
import json
import os
from pathlib import Path
import tempfile
from test_authenticated_reads import locked_inputs
import unittest
from unittest.mock import patch

from common import ROOT, read_config, qualification_config
from multimodal import (MEAN_ABS_LOGPROB_LIMIT, case_limits,
                        evaluate_case, evaluate_probe, greedy_agreement, load_probes,
                        load_reference, media_variants, ordered_clauses, probe_forced_agreement, reference_probe,
                        probe_images, probe_references,
                        reference_greedy_agreement,
                        exit_status, run, run_probes, score_forced)


REFERENCE = ROOT / "fixtures/mimo26/multimodal/reference.json"
EXACT_REFERENCE = ROOT / "fixtures/mimo26/multimodal/exact-reference.json"
EXACT_SCHEMA = "chopinfer.mimo26.multimodal-exact-reference.v2"
QWEN38_SHORT = ("blue-square", "frame-00", "four-images")
QWEN38_FREE_TEXT = ("gradient-333x517", "small-100x60", "wide-1000x40", "large-4500x4000")
CAPTURE = ROOT / "fixtures/mimo26/capture"
import sys
sys.path.insert(0, str(CAPTURE))
from multimodal import cases  # noqa: E402
import multimodal  # noqa: E402
import multimodal_cases  # noqa: E402
import importlib.util

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "multimodal capture checks on synthetic API data; other gates are selected explicitly"


spec = importlib.util.spec_from_file_location("multimodal_capture", CAPTURE / "media_answers.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class FakeClient:
    def request(self, path, body):
        if "prompt_logprobs" in body:
            raise RuntimeError("prompt_logprobs is unsupported")
        return {"choices": [{"message": {"content": "blue"}}],
                "usage": {"prompt_tokens": 2}}, 0.2


def complete_rows(tokens):
    """Consistent exact columns in which each token is the top-1 at -0.3."""
    top5 = [[[token, -0.3], *[[token + other, -1.5 - other] for other in range(1, 5)]] for token in tokens]
    return {"logprob": [-0.3] * len(tokens), "top1": list(tokens), "margin": [2.2] * len(tokens),
            **top_columns(top5)}


class MultimodalTests(unittest.TestCase):
    def test_fixture_parts_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "blue-square.png").write_bytes(b"png")
            digest = hashlib.sha256(b"png").hexdigest()
            (directory / "manifest.json").write_text(json.dumps({"files": {"blue-square.png": digest}}))
            case = {"id": "image", "kind": "image", "file": "blue-square.png", "prompt": "blue?"}
            body = multimodal_cases.messages(case, directory)
            self.assertEqual(body[0]["content"][1]["image_url"]["url"],
                             "data:image/png;base64,cG5n")
            audio = {"id": "audio", "kind": "audio", "transport": "input_audio",
                     "file": "blue-square.png", "prompt": "transcribe"}
            self.assertEqual(multimodal_cases.messages(audio, directory)[0]["content"][1],
                             {"type": "input_audio", "input_audio": {"data": "cG5n", "format": "wav"}})
            (directory / "blue-square.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                multimodal_cases.messages(case, directory)


    def test_reference_hash_and_case_set(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            reference = {"schema": "mimo26-multimodal-reference/v1",
                         "cases": [{"id": case["id"], "kind": case["kind"]} for case in cases()]}
            capture.write_capture(directory, reference, {})
            loaded, digest, _ = load_reference(directory / "reference.json")
            self.assertEqual(loaded, reference)
            self.assertEqual(digest, hashlib.sha256((directory / "reference.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                capture.write_capture(directory, reference, {})
            (directory / "reference.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_reference(directory / "reference.json")

    def test_exact_reference_joins_the_capture_it_scores(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            reference = {"schema": "mimo26-multimodal-reference/v1",
                         "cases": [{"id": case["id"], "kind": case["kind"], "forced_prompt_token_ids": [1, 2, 3],
                                    "answer_token_start": 2} for case in cases()]}
            capture.write_capture(directory, reference, {})
            digest = hashlib.sha256((directory / "reference.json").read_bytes()).hexdigest()
            rows = complete_rows([3])

            def write_exact(capture_sha256=digest, answer=(3,), rows=rows, drop=None):
                exact = {"schema": EXACT_SCHEMA, "capture_sha256": capture_sha256,
                         "cases": [{"id": case["id"], "kind": case["kind"], "answer_token_ids": list(answer),
                                    **{name: rows for name in media_variants() if name != drop}}
                                   for case in cases()]}
                data = json.dumps(exact).encode()
                (directory / "exact-reference.json").write_bytes(data)
                manifest = json.loads((directory / "manifest.json").read_text())
                manifest["files"]["exact-reference.json"] = hashlib.sha256(data).hexdigest()
                (directory / "manifest.json").write_text(json.dumps(manifest))

            write_exact()
            loaded, _, _ = load_reference(directory / "reference.json", directory / "exact-reference.json")
            self.assertEqual(loaded["cases"][0]["exact"], {name: rows for name in media_variants()})
            self.assertEqual(loaded["cases"][-1]["exact"], {name: rows for name in media_variants()})
            changes = [{"capture_sha256": "0" * 64}, {"answer": (4,)}, {"answer": (3, 4)},
                       *({"drop": name} for name in media_variants())]
            for change in changes:
                with self.subTest(change=change):
                    write_exact(**change)
                    with self.assertRaises(ValueError):
                        load_reference(directory / "reference.json", directory / "exact-reference.json")
            write_exact()
            (directory / "exact-reference.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_reference(directory / "reference.json", directory / "exact-reference.json")
            # A publisher's incomplete or inconsistent columns, with a matching manifest.
            for column, value in (("top1", []), ("topk_ids", []), ("margin", []), ("logprob", [float("nan")]),
                                  ("top1", [9]), ("topk_ids", [[3]]), ("margin", [0.5]),
                                  ("logprob", [-0.1])):
                with self.subTest(column=column, value=value):
                    write_exact(rows={**rows, column: value})
                    with self.assertRaises(ValueError):
                        load_reference(directory / "reference.json", directory / "exact-reference.json")

    def test_committed_exact_reference_scores_the_committed_capture(self):
        reference, _, _ = load_reference(REFERENCE, EXACT_REFERENCE)
        for case in reference["cases"]:
            limits = case_limits(case)
            self.assertGreaterEqual(limits["mean_abs_dlogprob"], MEAN_ABS_LOGPROB_LIMIT)
            self.assertEqual(len(case["exact"]["exact"]["topk_ids"][0]), 5)

    def test_logprob_limits_and_token_mismatch(self):
        reference = {"forced_prompt_token_ids": [1, 2, 3], "answer_token_start": 2,
                     "forced_prompt_logprobs": [None, {}, {"3": {"rank": 1, "logprob": -0.9}}],
                     "exact": {"exact": {"logprob": [-0.3], "top1": [3], **top_columns([[[3, -0.3]]])}}}
        observed = {"prompt_token_ids": [1, 2, 3],
                    "prompt_logprobs": [None, {}, {"3": {"rank": 1, "logprob": -0.32}}]}
        result = score_forced(reference, observed)
        self.assertTrue(result["passed"])
        self.assertEqual(result["limits"], {"mean_abs_dlogprob": 0.15, "top1_decisive": 0.08, "top1_near_tie": 0.08,
                                            "top1_disagreement": 0.10})
        observed["prompt_logprobs"][2]["3"]["logprob"] = -0.5
        self.assertFalse(score_forced(reference, observed)["passed"])
        # A reference variant 0.21 away raises the mean limit to 0.25.
        reference["exact"]["bf16"] = {"logprob": [-0.51], "top1": [3], **top_columns([[[3, -0.51]]])}
        result = score_forced(reference, observed)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["limits"]["mean_abs_dlogprob"], 0.25)
        observed["prompt_token_ids"] = [1, 9, 3]
        with self.assertRaisesRegex(ValueError, "differ"):
            score_forced(reference, observed)

    def test_served_flips_count_against_the_decisive_or_the_near_tie_limit(self):
        reference = {"forced_prompt_token_ids": [1, 2, 3], "answer_token_start": 2,
                     "forced_prompt_logprobs": [None, {}, {"3": {"rank": 1, "logprob": -0.9}}],
                     "exact": {"exact": {"logprob": [-0.6], "top1": [3], **top_columns([[[3, -0.6], [5, -0.8]]])}}}

        def observed(other):
            return {"prompt_token_ids": [1, 2, 3], "prompt_logprobs": [
                None, {}, {"3": {"rank": 2, "logprob": -0.7}, str(other): {"rank": 1, "logprob": -0.6}}]}
        near = score_forced(reference, observed(5))
        self.assertEqual((near["top1_disagreement"], near["top1_near_ties"], near["top1_decisive"]), (1.0, 1, 0.0))
        # One near tie in one position exceeds the 8% near-tie limit.
        self.assertEqual((near["top1_near_tie"], near["limits"]["top1_near_tie"]), (1.0, 0.08))
        self.assertFalse(near["passed"])
        outside = score_forced(reference, observed(6))
        self.assertEqual((outside["top1_near_ties"], outside["top1_decisive"]), (0, 1.0))
        self.assertFalse(outside["passed"])

    def test_truncated_columns_fail_the_case_instead_of_scoring_fewer_positions(self):
        case = {"id": "image-frames-00.png", "kind": "image", "file": "frames/00.png", "prompt": "?"}
        expected = {"prompt_token_ids": [1, 2], "answer": "red blue",
                    "forced_prompt_token_ids": [1, 2, 3, 4], "answer_token_start": 2,
                    "forced_prompt_logprobs": [None, {}, {"3": {"logprob": -0.3}}, {"4": {"logprob": -0.3}}],
                    "exact": {"exact": {**complete_rows([3, 4]), "top1": [], **top_columns([])}}}

        class Wrong:
            def request(self, path, body):
                if body.get("prompt_logprobs"):
                    return {"prompt_token_ids": [1, 2, 3, 4], "prompt_logprobs": [
                        None, {}, {"3": {"logprob": -0.3}, "8": {"logprob": -0.1}},
                        {"4": {"logprob": -0.3}, "8": {"logprob": -0.1}}]}, 0.1
                return {"choices": [{"message": {"content": "unrelated answer"}, "token_ids": [42]}],
                        "usage": {"prompt_tokens": 2}}, 0.1

        with patch("multimodal.messages", return_value=[]):
            row = evaluate_case(Wrong(), "M", case, expected, Path("/unused"), {})
        self.assertFalse(row["passed"])
        self.assertFalse(row["logprobs"]["passed"])
        self.assertFalse(row["greedy"]["passed"])

    def test_cases_of_a_thousand_positions_have_the_tighter_mean_floor(self):
        def case(count):
            rows = {"logprob": [-0.3] * count, "top1": [3] * count, **top_columns([[[3, -0.3]]] * count)}
            return {"exact": {"exact": rows, "bf16": rows}}
        floors = {"top1_decisive": 0.08, "top1_near_tie": 0.08, "top1_disagreement": 0.10}
        self.assertEqual(case_limits(case(999)), {"mean_abs_dlogprob": 0.15, **floors})
        self.assertEqual(case_limits(case(1000)), {"mean_abs_dlogprob": 0.10, **floors})

    def test_limits_follow_the_worst_reference_variant(self):
        gate = {"logprob": [-0.1, -0.2, -0.3, -0.4], "top1": [1, 2, 3, 4],
                **top_columns([[[1, -0.1]], [[2, -0.2]], [[3, -0.3], [8, -0.5]], [[4, -0.4], [9, -0.8]]])}
        close = {"logprob": [-0.11, -0.2, -0.3, -0.4], "top1": [1, 2, 3, 4]}
        flipped = {"logprob": [-0.1, -0.2, -0.3, -0.4], "top1": [1, 2, 3, 9]}
        limits = case_limits({"exact": {"exact": gate, "close": close, "flipped": flipped}})
        self.assertEqual(limits, {"mean_abs_dlogprob": 0.15, "top1_decisive": 0.25, "top1_near_tie": 0.08,
                                  "top1_disagreement": 0.25})
        # A flip the gate variant ranks 0.2 nats below its top-1 is a near tie:
        # it raises the near-tie limit, not the decisive one.
        near = {"logprob": [-0.1, -0.2, -0.3, -0.4], "top1": [1, 2, 8, 4]}
        limits = case_limits({"exact": {"exact": gate, "close": close, "near": near}})
        self.assertEqual(limits, {"mean_abs_dlogprob": 0.15, "top1_decisive": 0.08, "top1_near_tie": 0.25,
                                  "top1_disagreement": 0.25})

    def test_full_case_passes_with_matching_answer_count_and_logprobs(self):
        case = {"id": "image-blue-square.png", "kind": "image", "file": "blue-square.png",
                "prompt": "What color?"}
        expected = {"prompt_token_ids": [1, 2], "answer": "blue",
                    "forced_prompt_token_ids": [1, 2, 3], "answer_token_start": 2,
                    "forced_prompt_logprobs": [None, {}, {"3": {"rank": 1, "logprob": -0.3}}],
                    "exact": {"exact": {"logprob": [-0.3], "top1": [3], **top_columns([[[3, -0.3]]])}}}
        class HealthyClient:
            def request(self, path, body):
                if body.get("prompt_logprobs"):
                    return {"prompt_token_ids": [1, 2, 3], "prompt_logprobs": [
                        None, {}, {"3": {"rank": 1, "logprob": -0.31}}]}, 0.1
                return {"choices": [{"message": {"content": "blue"}, "token_ids": [3]}],
                        "usage": {"prompt_tokens": 2}}, 0.2
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "blue-square.png").write_bytes(b"png")
            (directory / "manifest.json").write_text(json.dumps({"files": {
                "blue-square.png": hashlib.sha256(b"png").hexdigest()}}))
            row = evaluate_case(HealthyClient(), "M", case, expected, directory,
                                {"image_expected": "blue"})
        self.assertTrue(row["passed"])
        self.assertEqual(row["logprobs"]["scored"], 1)
        self.assertTrue(row["greedy"]["exact"])

    def test_greedy_answer_may_part_only_at_an_exact_near_tie(self):
        expected = {"answer": "red.", "answer_token_start": 1, "forced_prompt_token_ids": [0, 5, 6],
                    "exact": {"exact": {**top_columns([[[5, -0.1], [7, -3.0]],
                                                 [[8, -1.2], [6, -1.4], [9, -1.6]]])}}}
        # Equal text walks the reference tokens: token 6 is 0.2 below the top-1.
        exact = greedy_agreement(expected, "red.")
        self.assertTrue(exact["passed"] and exact["exact"])
        self.assertAlmostEqual(exact["reference_margin"], 0.2)
        near = greedy_agreement(expected, "red and more", generated_ids=[5, 8, 10])
        self.assertTrue(near["passed"])
        self.assertEqual((near["divergence"], near["served_token_id"], near["reference_token_id"]), (1, 8, 6))
        far = greedy_agreement(expected, "red?", generated_ids=[5, 9])
        self.assertFalse(far["passed"])
        self.assertIn("near-tie", far["reason"])
        outside = greedy_agreement(expected, "blue", generated_ids=[4])
        self.assertIn("top 5", outside["reason"])
        self.assertFalse(greedy_agreement(expected, "red", generated_ids=[5])["passed"])
        self.assertFalse(greedy_agreement(expected, "red and more")["passed"])
        # The reference answer itself can sit beyond the limit of the exact top-1.
        expected["exact"]["exact"]["topk_ids"][1] = [8, 6]
        expected["exact"]["exact"]["topk_logits"][1] = [-0.5, -1.4]
        expected["exact"]["exact"]["topk_logprobs"][1] = [-0.5, -1.4]
        self.assertFalse(greedy_agreement(expected, "red.")["passed"])

    def test_audio_uses_its_own_near_tie_limit(self):
        expected = {"prompt_token_ids": [0], "answer": "The tone",
                    "answer_token_start": 1, "forced_prompt_token_ids": [0, 5, 6],
                    "forced_prompt_logprobs": [None, {"5": {"logprob": 0.0}}, {"6": {"logprob": -0.1}}],
                    "exact": {"exact": {"logprob": [-0.1, -0.1], "top1": [5, 6], "margin": [2.2, 3.0],
                                        **top_columns([[[5, -0.1], [7, -2.3]], [[6, -0.1], [8, -3.1]]])}}}
        class AudioClient:
            text = "Here tone"
            count = 1
            ids = [7, 6]
            forced_error = 0.0

            def request(self, path, body):
                if body.get("return_token_ids") is not True:
                    raise AssertionError("decode must request generated token IDs")
                if body.get("prompt_logprobs"):
                    return {"prompt_token_ids": [0, 5, 6], "prompt_logprobs": [
                        None, {"5": {"logprob": -0.1}}, {"6": {"logprob": -0.1 - self.forced_error}}]}, 0.1
                return {"choices": [{"message": {"content": self.text}, "token_ids": self.ids}],
                        "usage": {"prompt_tokens": self.count}}, 0.1

        client = AudioClient()
        case = {"id": "audio", "kind": "audio"}
        # A declaration that qualifies audio states its transcript clauses;
        # one that serves audio unqualified declares none.
        policy = multimodal.media_policy()
        policy = {**policy, "expectations": {**policy["expectations"], "kinds": {
            **policy["expectations"]["kinds"], "audio": {"field": "audio_expected_clauses", "single": False}}}}
        with patch("multimodal.messages", return_value=[]), \
                patch.object(multimodal, "media_policy", return_value=policy):
            def evaluate():
                return evaluate_case(client, "M", case, expected, Path("/unused"),
                                     {"audio_expected_clauses": ["tone"]})
            result = evaluate()
            self.assertTrue(result["passed"])
            self.assertEqual(result["logprobs"]["top1_disagreement"], 0)
            self.assertAlmostEqual(result["greedy"]["reference_margin"], 2.2)
            self.assertEqual(result["greedy"]["near_tie_limit"], 2.3)
            self.assertEqual(result["greedy"]["divergence"], 0)
            self.assertEqual(result["greedy"]["served_token_id"], 7)
            self.assertEqual(result["greedy"]["reference_token_id"], 5)
            for attribute, value in (("text", "Here wrong"), ("count", 2),
                                     ("forced_error", 0.4), ("ids", None), ("ids", [99, 6])):
                with self.subTest(attribute=attribute, value=value):
                    previous = getattr(client, attribute)
                    setattr(client, attribute, value)
                    self.assertFalse(evaluate()["passed"])
                    setattr(client, attribute, previous)
        for kind in ("image", "video", "text"):
            with self.subTest(kind=kind):
                result = greedy_agreement(expected, "Here tone", kind=kind, generated_ids=[7, 6])
                self.assertFalse(result["passed"])
                self.assertEqual(result["near_tie_limit"], 0.3)

    def test_reference_that_contradicts_the_fixture_is_refused(self):
        case = next(case for case in cases() if case["file"] == "blue-square.png")
        reference = {"cases": [{"id": case["id"], "answer": "red"}]}
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "manifest.json").write_text(json.dumps({"image_expected": "blue"}))
            with locked_inputs({"media_fixtures": str(directory)},
                               qualification={"media": read_config(str(ROOT / "models/mimo26/qual.json"))["qualification"]["media"]}), patch("multimodal.cases", return_value=[case]):
                with self.assertRaisesRegex(ValueError, "contradicts"):
                    run(FakeClient(), {"model": "M"}, reference, directory, {})

    def test_serving_rows_fail_when_logprob_api_is_unavailable(self):
        answers = {"image": "blue", "audio": "tone", "video": "red blue", "text": "red blue"}
        reference = {"cases": []}
        for case in cases():
            reference["cases"].append({"id": case["id"], "prompt_token_ids": [1, 2],
                                       "answer": answers[case["kind"]],
                                       "forced_prompt_token_ids": [1, 2, 3],
                                       "answer_token_start": 2,
                                       "forced_prompt_logprobs": [None, {}, {"3": {"rank": 1, "logprob": -0.3}}]})
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            files = {case["file"]: hashlib.sha256(b"x").hexdigest()
                     for case in cases() if case.get("file")}
            for file in files:
                path = directory / file
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x")
            (directory / "manifest.json").write_text(json.dumps({
                "files": files, "image_expected": "blue", "audio_expected_clauses": ["tone"],
                "video_expected": ["red", "blue"]}))
            with locked_inputs({"media_fixtures": str(directory),
                                "frontend": str(ROOT / "models/mimo26/compiler/src/frontend.json"),
                                "media_cases": str(CAPTURE / "multimodal_cases.py")},
                               qualification={"media": read_config(str(ROOT / "models/mimo26/qual.json"))["qualification"]["media"]}):
                result = run(FakeClient(), {"model": "M"}, reference, directory, files)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["rows"]), len(list(cases())))
        self.assertTrue(result["rows"][0]["token_count_match"])
        self.assertFalse(result["rows"][0]["logprobs"]["passed"])
        self.assertTrue(ordered_clauses("first red, then blue", ["red", "blue"]))
        self.assertFalse(ordered_clauses("blue then red", ["red", "blue"]))
        # A server that serves text and images alone: video is not
        # applicable, never sent and never passed; the rest run as before.
        server = {"model": "M", "health": {"status": "ready", "modalities": ["text", "image"]}}
        with tempfile.TemporaryDirectory() as root, \
                patch.object(FakeClient, "request", autospec=True,
                             side_effect=FakeClient.request) as sent:
            directory = Path(root)
            for file in files:
                path = directory / file
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x")
            (directory / "manifest.json").write_text(json.dumps({
                "files": files, "image_expected": "blue", "audio_expected_clauses": ["tone"],
                "video_expected": ["red", "blue"]}))
            with locked_inputs({"media_fixtures": str(directory),
                                "frontend": str(ROOT / "models/mimo26/compiler/src/frontend.json"),
                                "media_cases": str(CAPTURE / "multimodal_cases.py")},
                               qualification={"media": read_config(str(ROOT / "models/mimo26/qual.json"))["qualification"]["media"]}):
                partial = run(FakeClient(), server, reference, directory, files)
        skipped = [row for row in partial["rows"] if row.get("status") == "not_applicable"]
        self.assertEqual({row["kind"] for row in skipped}, {"video"})
        self.assertTrue(all(row["passed"] is None for row in skipped))
        self.assertEqual(partial["not_applicable"], [
            "the server does not serve video input (--modalities text,image)"])
        self.assertEqual(len(partial["rows"]), len(list(cases())))
        sent_kinds = {content.get("type") for call in sent.call_args_list
                      for message in call.args[2]["messages"] if isinstance(message["content"], list)
                      for content in message["content"]}
        self.assertNotIn("input_audio", sent_kinds)
        self.assertNotIn("video_url", sent_kinds)
        self.assertNotIn("audio_url", sent_kinds)
        self.assertFalse(partial["passed"])
        self.assertEqual(exit_status(partial), 1)
        self.assertEqual(exit_status({"passed": True, "errors": [], "not_applicable": ["x"]}), 0)
        self.assertEqual(exit_status({"passed": False, "errors": [], "not_applicable": ["x"]}), 0)


def entry(token, logprob, alternatives=()):
    """One chat logprob entry; `alternatives` are (token, logprob) pairs."""
    top = [(token, logprob), *alternatives]
    return {"token": token, "logprob": logprob, "bytes": list(token.encode()),
            "top_logprobs": [{"token": t, "logprob": v, "bytes": list(t.encode())} for t, v in top]}


def choice(tokens, finish_reason="stop"):
    return {"message": {"content": "".join(entry["token"] for entry in tokens
                                           if entry["token"] != "<|im_end|>")},
            "logprobs": {"content": tokens}, "finish_reason": finish_reason}


RULE = {"variants": ["bf16", "f64"], "mean_step": 0.05, "mean_floor": 0.15, "mean_cap": 0.3, "top1_floor": 0.08,
        "near_tie": 0.3}


def runner_rows(tokens, logprobs, top1s, margins, other):
    """Runner score columns (top 5 ranked by logit, logits the logprobs plus
    10) for forced `tokens` with these logprobs, top-1 IDs and margins;
    `other(position, rank)` names the remaining top-5 IDs."""
    columns = {"logprob": [], "top1": [], "margin": [], "topk_ids": [], "topk_logits": [], "topk_logprobs": []}
    for position, (token, logprob, top1, margin) in enumerate(zip(tokens, logprobs, top1s, margins)):
        best = logprob if top1 == token else logprob + margin
        ids = [top1] + ([] if top1 == token else [token])
        values = [best, best - margin]
        ids += [other(position, rank) for rank in range(len(ids), 5)]
        values += [values[-1] - rank for rank in range(1, 6 - len(values))]
        for column, value in (("logprob", logprob), ("top1", top1), ("margin", margin), ("topk_ids", ids),
                              ("topk_logits", [v + 10 for v in values]), ("topk_logprobs", values)):
            columns[column].append(value)
    return columns


def probe_answer(probe):
    """Make a mock API response from the declared HF greedy answer columns."""
    if "response" in probe:
        return probe["response"]["body"]
    config = qualification_config()
    key = config["qualification"]["media"]["reference"]
    path = ROOT / config["qualification"]["references"][key]["path"]
    fixture = json.loads(path.read_text())
    item = fixture["probes"][probe["name"]]
    scores = item.get("greedy") or item["generated"]
    def word(token, score):
        data = fixture["token_bytes"][str(token)]
        return {"token": bytes(data).decode("utf-8", errors="replace"), "bytes": data, "logprob": score}
    rows = []
    for index, token in enumerate(scores["answer"]):
        row = word(token, scores["logprob"][index])
        row["top_logprobs"] = [word(token, score) for token, score in
                               zip(scores["topk_ids"][index], scores["topk_logprobs"][index], strict=True)]
        rows.append(row)
    return {"choices": [{"message": {"content": item["answer"]["text"]}, "logprobs": {"content": rows}}],
            "usage": {"prompt_tokens": len(probe["token_ids"])}}


def reference_entry(probe, kind, table):
    """A reference probe with HF answer tokens and synthetic spread scores. `table` maps token bytes to the
    IDs it assigns (from 1000), the fixture's token_bytes inverted."""
    def token(data):
        return table.setdefault(bytes(data), 1000 + len(table))
    content = probe_answer(probe)["choices"][0]["logprobs"]["content"]
    ids = [token(row["bytes"]) for row in content]
    steps = {"answer": ids, "logprob": [row["logprob"] for row in content], "top1": ids,
             "margin": [0.5] * len(content),
             "topk_ids": [[token(top["bytes"]) for top in row["top_logprobs"]] for row in content],
             "topk_logprobs": [[top["logprob"] for top in row["top_logprobs"]] for row in content]}
    steps["topk_logits"] = [[value + 10 for value in row] for row in steps["topk_logprobs"]]
    stopped = kind == "short"
    forced = ids if stopped else ids[:-1]
    logprobs = [row["logprob"] for row in content][:len(forced)]

    def other(position, rank):
        return token(f"other-{rank}".encode())
    item = {"kind": kind, "inputs": {"images": probe.get("images", [])},
            "answer": {"token_ids": ids, "stopped": True,
                       "text": probe_answer(probe)["choices"][0]["message"]["content"]},
            "answer_token_ids": forced,
            "exact": runner_rows(forced, logprobs, forced, [0.5] * len(forced), other)}
    if kind == "short":
        item["greedy"] = steps
        for variant in RULE["variants"]:
            item[variant] = item["exact"]
        return item
    item["generated"] = {"key_sha256": "0" * 64, **steps}
    # bf16 sits 0.12 from exact and flips position 2; f64 sits 0.01 away and
    # its margin moves by more than the reference's 0.5 at position 3.
    flipped = list(forced)
    flipped[2] = token(b"flip")
    item["bf16"] = runner_rows(forced, [v - 0.12 for v in logprobs], flipped, [0.5] * len(forced), other)
    margins = [0.5] * len(forced)
    margins[3] = 1.2
    item["f64"] = runner_rows(forced, [v - 0.01 for v in logprobs], forced, margins, other)
    return item


def reference_file(root, probes, digest, change=lambda probes: None):
    table = {}
    references = {probe["name"]: reference_entry(
        probe, "short" if probe["name"] in QWEN38_SHORT else "free-text", table) for probe in probes}
    change(references)
    path = Path(root) / "references.json"
    path.write_text(json.dumps({"schema": "chopinfer.qwen38.image-probe-references.v5",
                                "provenance": {"capture_sha256": digest}, "rule": RULE,
                                "token_bytes": {str(i): list(data) for data, i in table.items()},
                                "probes": references}))
    return path


class Qwen38ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        selected = patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(ROOT / "models/qwen38/qual.json")})
        selected.start()
        cls.addClassCleanup(selected.stop)

    def setUp(self):
        self.probes, self.hashes, self.digest = load_probes()
        self.by_name = {probe["name"]: probe for probe in self.probes}
        with tempfile.TemporaryDirectory() as root:
            self.references, self.rule, _ = probe_references(
                self.digest, reference_file(root, self.probes, self.digest))

    def test_capture_loads_with_its_image_digests(self):
        names = [probe["name"] for probe in self.probes]
        self.assertEqual(names, ["blue-square", "frame-00", "four-images", "gradient-333x517",
                                 "small-100x60", "wide-1000x40", "large-4500x4000"])
        self.assertEqual(set(self.hashes), {name for probe in self.probes for name in probe["images"]})

    def test_images_must_match_the_capture(self):
        with tempfile.TemporaryDirectory() as root:
            functional, generated = Path(root) / "functional", Path(root) / "generated"
            (functional / "frames").mkdir(parents=True)
            generated.mkdir()
            files = {"blue-square": functional / "blue-square.png",
                     "frame-00": functional / "frames/00.png"}
            files |= {name: generated / f"{name}.png" for name in self.hashes if name not in files}
            for name, path in files.items():
                path.write_bytes(name.encode())
            hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in files}
            with locked_inputs({"media_fixtures": str(functional),
                                "media_images": str(ROOT / "fixtures/qwen38/capture/images.py")},
                               qualification={"media": read_config(str(ROOT / "models/qwen38/qual.json"))["qualification"]["media"]}):
                images = probe_images(generated, functional, hashes)
                self.assertEqual(images["frame-00"], b"frame-00")
                files["small-100x60"].write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "small-100x60"):
                    probe_images(generated, functional, hashes)
                files["small-100x60"].write_bytes(b"small-100x60")
                files["blue-square"].write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "lock entry media_fixtures.*sha256 differs"):
                    probe_images(generated, functional, hashes)


    def test_the_frozen_references_cover_the_capture(self):
        references, rule, provenance = probe_references(self.digest)
        self.assertEqual(provenance["capture_sha256"], self.digest)
        self.assertEqual((provenance["checkpoint"]["revision"], provenance["gate_variant"]),
                         ("236dfdf285828023ca3bcd3f37366c58a3469b13", "exact"))
        self.assertEqual(rule["variants"], ["bf16", "f64"])
        self.assertEqual({name: reference["limits"]["mean_abs_dlogprob"]
                          for name, reference in references.items() if "limits" in reference},
                         {"gradient-333x517": 0.15, "small-100x60": 0.15, "wide-1000x40": 0.15,
                          "large-4500x4000": 0.15})
        # Top-1 (decisive, near tie, all flips): gradient's BF16 run flips 4 of
        # its 25 positions, all near ties; the others stay at the floors.
        self.assertEqual({name: tuple(reference["limits"][key] for key in ("top1_decisive", "top1_near_tie",
                                                                           "top1_disagreement"))
                          for name, reference in references.items() if "limits" in reference},
                         {"gradient-333x517": (0.08, 0.16, 0.16), "small-100x60": (0.08, 0.08, 0.10),
                          "wide-1000x40": (0.08, 0.08, 0.10), "large-4500x4000": (0.08, 0.08, 0.10)})
        for name, reference in references.items():
            self.assertEqual(reference["inputs"]["images"], self.by_name[name]["images"])
            self.assertTrue(reference["answer"]["stopped"], name)
            self.assertEqual(len(reference["steps"]), len(reference["answer"]["token_ids"]))
            # A short answer forces its generated tokens; free text forces them as a chat renders
            # them, without the end token (no F32 answer starts with whitespace).
            forced = [row[0] for row in reference["forced"]]
            ids = reference["answer"]["token_ids"]
            self.assertEqual(forced, ids if reference["kind"] == "short" else ids[:-1], name)
        self.assertEqual(references["blue-square"]["answer"]["text"], " blue")
        self.assertEqual(references["four-images"]["answer"]["text"], "4")

    def test_image_spread_has_no_cap(self):
        import copy
        vectors = json.loads((ROOT / "tests/qual-image-spread-vectors.json").read_text())
        for vector in vectors["cases"]:
            with self.subTest(vector["name"]):
                table = {}
                item = reference_entry(self.by_name["gradient-333x517"], "free-text", table)
                for variant in RULE["variants"]:
                    rows = copy.deepcopy(item["exact"])
                    rows["logprob"] = [x - vector["spread"] for x in rows["logprob"]]
                    for field in ("topk_logprobs", "topk_logits"):
                        rows[field] = [[x - vector["spread"] for x in row] for row in rows[field]]
                    item[variant] = rows
                reference = reference_probe("probe", item, RULE,
                                            {str(i): list(data) for data, i in table.items()})
                self.assertAlmostEqual(reference["limits"]["mean_abs_dlogprob"], vector["mean_limit"])

    def test_greedy_answer_may_part_only_at_a_reference_near_tie(self):
        stop = entry("<|im_end|>", -0.01)
        vllm = {"logprobs": {"content": [entry("A", -0.1),
                                         entry(" red", -0.6, [(" blue", -0.8), (" green", -2.0)]),
                                         entry(" square", -0.2), stop]},
                "message": {"content": "A red square"}}
        table = {}
        item = reference_entry({"response": {"body": {"choices": [vllm]}}}, "short", table)
        reference = reference_probe("short", item, RULE, {str(i): list(data) for data, i in table.items()})
        exact = reference_greedy_agreement(reference, choice(vllm["logprobs"]["content"]), 0.3)
        self.assertTrue(exact["exact"])
        near = reference_greedy_agreement(
            reference, choice([entry("A", -0.1), entry(" blue", -0.5), stop]), 0.3)
        self.assertEqual((near["passed"], near["divergence"]), (True, 1))
        self.assertAlmostEqual(near["reference_margin"], 0.2)
        far = reference_greedy_agreement(reference, choice([entry("A", -0.1), entry(" green", -0.5)]), 0.3)
        self.assertFalse(far["passed"])
        outside = reference_greedy_agreement(reference, choice([entry("A", -0.1), entry(" pink", -0.5)]), 0.3)
        self.assertIn("outside", outside["reason"])
        early = reference_greedy_agreement(reference, choice([entry("A", -0.1)]), 0.3)
        self.assertFalse(early["passed"])

    def test_references_follow_the_rule_and_must_come_from_the_loaded_capture(self):
        # bf16 is the worst variant: 0.12 rounds up to 0.15, one flip of `scored`.
        small = self.references["small-100x60"]
        self.assertEqual(small["limits"]["mean_abs_dlogprob"], 0.15)
        self.assertEqual(small["limits"]["top1_disagreement"], max(0.10, 1 / small["scored"]))
        self.assertAlmostEqual(small["spread"]["f64"]["mean_abs_dlogprob"], 0.01)
        self.assertNotIn("limits", self.references["blue-square"])
        with tempfile.TemporaryDirectory() as root:
            def shifted(variant, by):
                def change(references):
                    rows = references["wide-1000x40"][variant]
                    rows["logprob"] = [value - by for value in rows["logprob"]]
                    rows["topk_logprobs"] = [[value - by for value in row] for row in rows["topk_logprobs"]]
                return change
            # The mean limit follows the spread without a cap.
            limits = probe_references(self.digest, reference_file(root, self.probes, self.digest,
                                                                   shifted("f64", 0.31)))
            self.assertEqual(limits[0]["wide-1000x40"]["limits"]["mean_abs_dlogprob"], 0.35)

            # The top-1 limit follows the worst variant above the floor.
            def flips(references):
                wide = references["wide-1000x40"]
                wide["f64"] = runner_rows(wide["answer_token_ids"], wide["f64"]["logprob"],
                                          [7] * 9 + wide["answer_token_ids"][9:], wide["f64"]["margin"],
                                          lambda position, rank: 100 + rank)
            limits = probe_references(self.digest, reference_file(root, self.probes, self.digest, flips))
            wide = limits[0]["wide-1000x40"]
            self.assertEqual((wide["limits"]["top1_decisive"], wide["limits"]["top1_disagreement"]),
                             (9 / wide["scored"], 9 / wide["scored"]))
            with self.assertRaisesRegex(ValueError, "small-100x60 f64: no scores"):
                probe_references(self.digest, reference_file(
                    root, self.probes, self.digest, lambda refs: refs["small-100x60"].pop("f64")))
            with self.assertRaisesRegex(ValueError, "wide-1000x40 exact position 0: margin differs"):
                probe_references(self.digest, reference_file(
                    root, self.probes, self.digest,
                    lambda refs: refs["wide-1000x40"]["exact"]["margin"].__setitem__(0, 0.25)))
            with self.assertRaisesRegex(ValueError, "measured on capture"):
                probe_references("0" * 64, reference_file(root, self.probes, self.digest))
            for change in (lambda refs: refs.pop("large-4500x4000"),
                           lambda refs: refs["blue-square"].update(kind="free-text"),
                           lambda refs: refs.update(extra=refs["frame-00"])):
                with self.assertRaisesRegex(ValueError, "must be exactly"):
                    probe_references(self.digest, reference_file(root, self.probes, self.digest, change))

    def test_a_short_answer_passes_with_the_reference_answer(self):
        probe = self.by_name["blue-square"]
        captured = probe_answer(probe)
        images = {name: name.encode() for name in self.hashes}
        requests = []

        class Server:
            def request(self, path, body):
                requests.append(body)
                return {**captured, "usage": {"prompt_tokens": len(probe["token_ids"])}}, 0.1

        row = evaluate_probe(Server(), "M", probe, images, self.references["blue-square"], 0.3)
        self.assertTrue(row["passed"], row)
        self.assertTrue(row["greedy"]["exact"])
        self.assertNotIn("teacher_forced", row)
        self.assertEqual(len(requests), 1)
        body = requests[0]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((body["temperature"], body["max_tokens"], body["top_logprobs"]), (0, 64, 5))
        parts = body["messages"][0]["content"]
        self.assertEqual(parts[-1], {"type": "text", "text": probe["text"]})
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/png;base64,Ymx1ZS1zcXVhcmU=")

        class Miscounted(Server):
            def request(self, path, body):
                return {**captured, "usage": {"prompt_tokens": 1}}, 0.1

        self.assertFalse(evaluate_probe(Miscounted(), "M", probe, images,
                                         self.references["blue-square"], 0.3)["passed"])

    def forced_response(self, probe, reference, shift=0.0, flips=(), trim=False):
        """The server's teacher-forced response to the reference answer: each
        token's logprob moved by `shift`, another top-1 at `flips`."""
        rows = [None] + [{} for _ in probe["token_ids"][1:]]
        ids = list(probe["token_ids"])
        for offset, (token, logprob, _) in enumerate(reference["forced"]):
            served = 999 if trim and offset == 0 else token
            row = {str(served): {"logprob": logprob + shift, "decoded_token": "t"}}
            if offset in flips:
                row["7"] = {"logprob": logprob + shift + 0.5, "decoded_token": "x"}
            ids.append(served)
            rows.append(row)
        return {"prompt_token_ids": ids, "prompt_logprobs": rows}

    def test_a_probe_uses_only_the_reference_answer(self):
        """A capture of requests only evaluates the
        reference alone, and the policy's thinking switch is what the
        requests send."""
        probe = {key: value for key, value in self.by_name["small-100x60"].items() if key != "response"}
        reference = self.references["small-100x60"]
        captured = probe_answer(self.by_name["small-100x60"])
        forced = self.forced_response(probe, reference)
        images = {name: name.encode() for name in self.hashes}
        requests = []

        class Server:
            def request(self, path, body):
                requests.append(body)
                if body.get("continue_final_message"):
                    return forced, 0.1
                return {**captured, "usage": {"prompt_tokens": len(probe["token_ids"])}}, 0.1

        policy = {**multimodal.media_policy(), "thinking": True}
        with patch.object(multimodal, "media_policy", return_value=policy):
            row = evaluate_probe(Server(), "M", probe, images, reference, 0.3)
        self.assertTrue(row["passed"], row)
        self.assertNotIn("vllm_answer", row)
        self.assertEqual(requests[0]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertTrue(row["teacher_forced"]["passed"])

    def test_a_reasoning_answer_has_the_reference_answer_budget(self):
        """A model that reasons first (its reference answer holds the
        reasoning) gets the reference answer's length as its budget, not 64
        tokens that end inside the reasoning, and the row keeps the served
        reasoning beside the content."""
        probe = {key: value for key, value in self.by_name["small-100x60"].items() if key != "response"}
        reference = self.references["small-100x60"]
        reasoning = {**reference, "answer": {**reference["answer"], "token_ids": list(range(266))}}
        captured = probe_answer(self.by_name["small-100x60"])
        forced = self.forced_response(probe, reference)
        images = {name: name.encode() for name in self.hashes}
        requests = []

        class Server:
            def request(self, path, body):
                requests.append(body)
                if body.get("continue_final_message"):
                    return forced, 0.1
                choice = captured["choices"][0]
                thought = {**choice, "message": {**choice["message"], "reasoning_content": "Look first."}}
                return {"choices": [thought], "usage": {"prompt_tokens": len(probe["token_ids"])}}, 0.1

        policy = {**multimodal.media_policy(), "thinking": True}
        with patch.object(multimodal, "media_policy", return_value=policy):
            row = evaluate_probe(Server(), "M", probe, images, reasoning, 0.3)
            self.assertEqual(requests[0]["max_tokens"], 266)
            self.assertEqual(row["reasoning"], "Look first.")
            self.assertEqual(row["answer"], captured["choices"][0]["message"]["content"])
            self.assertTrue(row["teacher_forced"]["passed"], row)
            requests.clear()
            # A reference answer within 64 tokens keeps the 64-token budget.
            evaluate_probe(Server(), "M", probe, images, reference, 0.3)
            self.assertEqual((len(reference["answer"]["token_ids"]) < 64, requests[0]["max_tokens"]), (True, 64))

    def test_a_run_without_the_scorer_is_refused_before_any_request(self):
        """Every row scores in chopinfer eval; without it the probe refuses
        to start instead of serving rows it cannot score."""
        clients = []

        class Client:
            def __init__(self, *args):
                clients.append(args)

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CHOPINFER_EVAL": str(Path(directory) / "chopinfer")}), \
                patch.object(multimodal, "Client", Client), \
                patch.object(sys, "argv", ["multimodal.py", "--url", "http://127.0.0.1:9"]):
            with self.assertRaisesRegex(SystemExit, "refused: the eval binary .* does not exist"):
                multimodal.main()
            # A file that exists but cannot run is refused the same way.
            scorer = Path(directory) / "chopinfer"
            scorer.write_bytes(b"")
            scorer.chmod(0o600)
            with self.assertRaisesRegex(SystemExit, "refused: the eval binary .* is not executable"):
                multimodal.main()
        self.assertEqual(clients, [])

    def test_free_text_answers_gate_on_the_reference_answer_teacher_forced(self):
        probe = self.by_name["wide-1000x40"]
        reference = self.references["wide-1000x40"]
        exact = probe_forced_agreement(probe, reference, self.forced_response(probe, reference))
        self.assertEqual((exact["scored"], exact["mean_abs_dlogprob"], exact["passed"]), (len(reference["forced"]), 0.0, True))
        shifted = probe_forced_agreement(probe, reference, self.forced_response(probe, reference, 0.2))
        self.assertFalse(shifted["passed"])
        bad_positions = tuple(range(1, int(0.08 * len(reference["forced"])) + 2))
        flipped = probe_forced_agreement(probe, reference,
                                          self.forced_response(probe, reference, flips=bad_positions))
        self.assertAlmostEqual(flipped["top1_disagreement"], len(bad_positions) / len(reference["forced"]))
        self.assertFalse(flipped["passed"])
        # The non-gating report: position 2 flips in a variant and position
        # 3's margin moves by more than itself; 1 and 4 are confident.
        report = flipped["near_tie_report"]
        self.assertEqual((report["gates"], report["near_tie"], report["confident"]), (False, 2, len(bad_positions) - 2))
        by_position = {row["position"]: row for row in report["flips"]}
        self.assertEqual(by_position[2]["variants_flipping"], ["bf16"])
        self.assertEqual(by_position[3]["margin_moves"], ["f64"])
        self.assertFalse(by_position[1]["near_tie"])
        self.assertAlmostEqual(by_position[1]["served_margin"], -0.5)
        self.assertEqual((by_position[1]["served_top1"], by_position[1]["reference_margin"]), (7, 0.5))
        self.assertEqual(exact["near_tie_report"]["flips"], [])
        wrong = self.forced_response(probe, reference)
        wrong["prompt_token_ids"][-1] = 5
        wrong["prompt_logprobs"][-1] = {"5": {"logprob": -1.0}}
        with self.assertRaisesRegex(ValueError, f"forced answer token {len(reference['forced']) - 1}"):
            probe_forced_agreement(probe, reference, wrong)
        # The reference forced the answer as the template renders it, so a
        # served first token that differs is an error, not a skipped row.
        with self.assertRaisesRegex(ValueError, "forced answer token 0"):
            probe_forced_agreement(probe, reference, self.forced_response(probe, reference, trim=True))

    def test_a_free_text_probe_may_part_from_the_reference_greedy_answer(self):
        probe = self.by_name["small-100x60"]
        reference = self.references["small-100x60"]
        captured = probe_answer(probe)
        forced = self.forced_response(probe, reference, 0.05)
        images = {name: name.encode() for name in self.hashes}
        requests = []
        other = [{"token": " Other", "logprob": -0.1, "bytes": list(b" Other"), "top_logprobs": []}]

        class Server:
            def request(self, path, body):
                requests.append(body)
                if body.get("continue_final_message"):
                    return forced, 0.1
                parted = {**captured["choices"][0], "message": {"content": " Other"},
                          "logprobs": {"content": other}}
                return {"choices": [parted], "usage": {"prompt_tokens": len(probe["token_ids"])}}, 0.1

        row = evaluate_probe(Server(), "M", probe, images, reference, 0.3)
        self.assertTrue(row["passed"], row)
        self.assertEqual((row["greedy"]["passed"], row["greedy"]["gates"]), (False, False))
        body = requests[1]
        self.assertEqual(body["messages"][-1], {"role": "assistant", "content": reference["answer"]["text"]})
        self.assertEqual((body["max_tokens"], body["prompt_logprobs"], body["return_token_ids"]),
                         (1, 5, True))
        self.assertNotIn("top_logprobs", body)
        forced.update(prompt_token_ids=[1])
        self.assertIn("captured prompt", evaluate_probe(Server(), "M", probe, images, reference, 0.3)
                      ["teacher_forced"]["error"])

    def test_run_needs_the_pinned_probes_and_a_consistent_reference(self):
        images = {name: b"x" for name in self.hashes}
        requests = []

        class Silent:
            def request(self, path, body):
                requests.append(body)
                raise RuntimeError("no server")

        result = run_probes(Silent(), {"model": "M"}, self.probes, images, self.references, self.rule,
                            ["frame-00"])
        self.assertEqual([row["id"] for row in result["rows"]], ["frame-00"])
        self.assertFalse(result["passed"])
        self.assertEqual(set(result["limits"]["teacher_forced"]), set(QWEN38_FREE_TEXT))
        with self.assertRaisesRegex(ValueError, "unknown probes"):
            run_probes(Silent(), {"model": "M"}, self.probes, images, self.references, self.rule,
                       ["frame-99"])
        # A server without image input sends no probe; none applies.
        text_only = {"model": "M", "health": {"modalities": ["text"]}}
        requests.clear()
        result = run_probes(Silent(), text_only, self.probes, images, self.references, self.rule)
        self.assertEqual(requests, [])
        self.assertTrue(all(row["status"] == "not_applicable" for row in result["rows"]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["not_applicable"],
                         ["the server does not serve image input (--modalities text)"])
        requests.clear()
        with self.assertRaisesRegex(ValueError, "pinned image-probe set"):
            run_probes(Silent(), {"model": "M"}, self.probes[1:], images, self.references, self.rule)
        wrong = json.loads(json.dumps(self.references))
        wrong["blue-square"]["answer"]["text"] = " Red"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            run_probes(Silent(), {"model": "M"}, self.probes, images, wrong, self.rule)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
