"""Offline checks for the evals.py scorers and statistics."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evals
from evals import divergence, extract_choice, flags, mcnemar, mmlu_pro_subset, paired, score


class EvalsTests(unittest.TestCase):
    def test_choice_extraction_order(self):
        self.assertEqual(extract_choice("B looks wrong. The answer is (C)."), "C")
        self.assertEqual(extract_choice("Answer: D"), "D")
        self.assertEqual(extract_choice("Between A and B I pick B"), "B")
        self.assertIsNone(extract_choice("no letter here"))

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar(0, 6), 2 / 64)
        self.assertAlmostEqual(mcnemar(3, 3), 1.0)

    def test_long_context_scoring(self):
        item = {"suite": "longctx", "expected": {"all": ["1234567"], "none": ["7654321"]}}
        self.assertTrue(score(item, "The number is 1,234,567.")[1])
        self.assertFalse(score(item, "1234567 or 7654321")[1])
        tracking = {"suite": "longctx", "expected": {"all": ["ABCDE", "FGHIJ"], "none": ["KLMNO"]}}
        self.assertTrue(score(tracking, "ABCDE FGHIJ")[1])
        self.assertFalse(score(tracking, "ABCDE")[1])

    def test_gsm8k_scoring(self):
        item = {"suite": "gsm8k", "expected": "18"}
        self.assertTrue(score(item, "so\n#### 18")[1])
        self.assertFalse(score(item, "the answer is 18")[1])

    def test_mmlu_pro_subset_is_proportional_and_stable(self):
        rows = [{"question_id": i, "category": "a" if i % 4 else "b"} for i in range(4000)]
        first = mmlu_pro_subset(rows)
        self.assertEqual(len(first), 1200)
        self.assertEqual(sum(row["category"] == "b" for row in first), 300)
        self.assertEqual(first, mmlu_pro_subset(list(reversed(rows))))

    def test_flags_one_sided_and_length_differences(self):
        def row(correct, finish="stop", tokens=10):
            return {"suite": "gsm8k", "correct": correct, "finish_reason": finish,
                    "extracted": "1" if correct else None, "content": "", "prompt_tokens": tokens}
        first = {f"i{k}": row(k >= 6, "length" if k < 6 else "stop") for k in range(20)}
        second = {f"i{k}": row(True, tokens=11 if k == 0 else 10) for k in range(20)}
        value = paired(first, second)
        self.assertEqual(value["only_length"], [6, 0])
        self.assertEqual(value["prompt_token_mismatch_ids"], ["i0"])
        found = flags({"suites": {"gsm8k": value}, "groups": {"gsm8k/x": value}}, "a", "b")
        self.assertTrue(any("gsm8k: a loses significantly" in flag for flag in found))
        self.assertTrue(any("length stops differ" in flag for flag in found))
        self.assertTrue(any("different prompt token counts" in flag for flag in found))

    def test_divergence_aligns_reasoning_and_content(self):
        def token(text, *top):
            return {"token": text, "top_logprobs": [{"logprob": value} for value in top]}
        full = {"reasoning": "ab", "content": "xy",
                "logprobs": [token("<think>", 0, -9), token("a", -0.1, -3), token("b", -0.60, -0.62),
                             token("</think>", 0, -9), token("x", -1, -2), token("y", -0.1, -0.4)]}
        content_only = {"reasoning": "ab", "content": "xz",
                        "logprobs": [token("x", -1, -2), token("z", -0.2, -0.3)]}
        found = divergence(full, content_only)
        self.assertEqual((found["part"], found["offset"]), ("content", 1))
        self.assertAlmostEqual(found["margins"][0], 0.3)  # min of "x" (1.0) and "y" (0.3)
        self.assertAlmostEqual(found["margins"][1], 0.1)
        other = divergence(full, {**content_only, "reasoning": "ac"})
        self.assertEqual(other["part"], "reasoning")
        self.assertAlmostEqual(other["margins"][0], 0.02)
        self.assertIsNone(other["margins"][1])
        self.assertIsNone(divergence(full, dict(full)))

    def test_declared_sampling_replaces_the_greedy_settings_of_every_body(self):
        rows = [{"question": "What is 1 + 1?", "answer": "1 + 1 = 2\n#### 2"}]
        with patch.object(evals, "thinking_kwargs", lambda thinking: {"enable_thinking": thinking}):
            greedy = evals.gsm8k_items(rows)[0]
            # Without a declaration the body is the greedy one, key for key.
            self.assertEqual(greedy["body"], {
                "messages": [{"role": "system", "content": evals.GSM8K_INSTRUCTION},
                             {"role": "user", "content": "What is 1 + 1?"}],
                "temperature": 0, "max_tokens": evals.REASONING_TOKENS,
                "chat_template_kwargs": {"enable_thinking": True}})
            recipe = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 262144}
            with patch.object(evals, "SAMPLING", recipe):
                sampled = evals.gsm8k_items(rows)[0]
                # Every suite's body takes it, the long-context suite's included.
                self.assertEqual(evals.chat([], evals.LONG_TOKENS, False)["max_tokens"], 262144)
            self.assertEqual(sampled["body"], {**greedy["body"], **recipe})
            self.assertNotEqual(evals.body_digest(sampled), evals.body_digest(greedy))
            self.assertNotEqual(evals.workload_digest([sampled]), evals.workload_digest([greedy]))
            with patch.object(evals, "SAMPLING", {"temperature": 1.0, "seed": 7}):
                with self.assertRaisesRegex(ValueError, r"SAMPLING sets \['seed'\]"):
                    evals.gsm8k_items(rows)

class ArmFiles:
    def arm(self, root, run, arm, correct, receipts, commit="c"):
        """An arm's results file in `run` and its receipt in `receipts`."""
        results = root / "cache/runs" / run / f"{arm}.jsonl"
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text("".join(json.dumps({"id": f"gsm8k-{index}", "suite": "gsm8k", "group": "gsm8k",
                                               "body_sha256": "b", "correct": index < correct, "extracted": "1",
                                               "content": "", "finish_reason": "stop"}) + "\n"
                                   for index in range(4)))
        receipts.mkdir(parents=True, exist_ok=True)
        path = receipts / f"20261007T000000Z-{run}-evals-{arm}.json"
        path.write_text(json.dumps({
            "schema": "chopinfer-qualification/v1", "check": f"evals-{arm}",
            "server": {"model": arm, "health": {"serving": {"version": "1.0", "source_commit": commit * 40,
                                                            "package_identity": arm[0] * 64}}},
            "run_id": run, "arm": arm, "datasets": {}, "workload_sha256": "w", "concurrency": 1,
            "long_concurrency": 1, "top_logprobs": 0, "limit": None, "at": "t",
            "results": str(results), "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest()}))
        return path


class CompareBindingTests(ArmFiles, unittest.TestCase):
    """compare reads each arm's results only as its receipt recorded them,
    and takes a named receipt of an arm recorded elsewhere."""

    def compare(self, root, run, arms, named=(), bound=False, stored=()):
        receipts = root / "gate"
        args = SimpleNamespace(run_id=run, arms=arms, receipt=list(named), bound=bound, stored=list(stored))
        with patch.object(evals, "CACHE", root / "cache"), \
                patch.dict(os.environ, {"CHOPINFER_QUAL_RECEIPTS": str(receipts)}), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            evals.compare(args)
        written = sorted(receipts.glob("*-evals-compare.json"))
        return exit.exception.code, json.loads(written[-1].read_text()) if written else None

    def test_a_named_receipt_supplies_an_arm_recorded_elsewhere(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = self.arm(root, "run", "reference", 3, root / "earlier")
            self.arm(root, "run", "candidate", 3, root / "gate")
            code, value = self.compare(root, "run", ["candidate", "reference"], [f"reference={recorded}"])
            self.assertEqual(code, 0)
            self.assertEqual(value["named_receipts"], {"reference": str(recorded)})
            self.assertEqual(value["unpaired_items"], {"candidate": 0, "reference": 0})
            # Without it the arm is not among the gate's receipts.
            self.assertIn("no evals receipt", str(self.compare(root, "run", ["candidate", "reference"])[0]))

    def test_a_bound_comparison_requires_every_arms_results_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = self.arm(root, "run", "reference", 3, root / "earlier")
            found = self.arm(root, "run", "candidate", 3, root / "gate")
            value = json.loads(found.read_text())
            del value["results_sha256"]
            found.write_text(json.dumps(value))
            with (root / "cache/runs/run/candidate.jsonl").open("a") as results:
                results.write(json.dumps({"id": "gsm8k-9", "suite": "gsm8k", "group": "gsm8k", "body_sha256": "b",
                                          "correct": True, "extracted": "1", "content": "",
                                          "finish_reason": "stop"}) + "\n")
            named = [f"reference={recorded}"]
            # Without --bound an older receipt without a digest is compared;
            # a qualification comparison refuses it.
            self.assertEqual(self.compare(root, "run", ["candidate", "reference"], named)[0], 0)
            code, value = self.compare(root, "run", ["candidate", "reference"], named, bound=True)
            self.assertIn("records no results digest", str(code))

    def test_a_bound_comparison_requires_one_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = self.arm(root, "run", "reference", 3, root / "earlier", commit="d")
            self.arm(root, "run", "candidate", 3, root / "gate")
            named = [f"reference={recorded}"]
            code, _ = self.compare(root, "run", ["candidate", "reference"], named, bound=True)
            self.assertIn("arms of one build", str(code))
            # Another package of the same code (another checkpoint's) compares.
            recorded = self.arm(root, "run", "reference", 3, root / "earlier")
            code, value = self.compare(root, "run", ["candidate", "reference"], named, bound=True)
            self.assertEqual((code, value["bound"]), (0, True))

    def test_an_arm_whose_results_changed_or_another_arms_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = self.arm(root, "run", "reference", 3, root / "earlier")
            other = self.arm(root, "run", "fp8", 3, root / "elsewhere")
            self.arm(root, "run", "candidate", 3, root / "gate")
            code, _ = self.compare(root, "run", ["candidate", "reference"], [f"reference={other}"])
            self.assertIn("is not the reference arm's receipt with its results and digest", str(code))
            with (root / "cache/runs/run/reference.jsonl").open("a") as results:
                results.write(json.dumps({"id": "gsm8k-9", "suite": "gsm8k", "group": "gsm8k", "body_sha256": "b",
                                          "correct": True, "extracted": "1", "content": "",
                                          "finish_reason": "stop"}) + "\n")
            code, value = self.compare(root, "run", ["candidate", "reference"], [f"reference={recorded}"])
            self.assertIn("is not the results the reference arm's receipt recorded", str(code))
            self.assertIsNone(value)


    def test_a_named_receipt_of_another_run_is_compared_by_its_own_results(self):
        """A gate's candidate arm, recorded by an earlier check run under that
        run's own ID: its named receipt binds the run it records and the
        results file it names, whatever the comparison's --run-id."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self.arm(root, "candidate-run", "candidate", 3, root / "candidate-check")
            stored = self.arm(root, "stored-run", "candidate", 4, root / "store")
            for run in ("compare-run", None):
                with self.subTest(run=run):
                    code, value = self.compare(root, run, ["candidate", "reference"], [f"candidate={candidate}"],
                                               stored=[f"reference={stored}"])
                    self.assertEqual(code, 0)
                    self.assertEqual(value["named_receipts"], {"candidate": str(candidate)})
                    self.assertEqual(value["arm_runs"], {"candidate": "candidate-run", "reference": "stored-run"})
                    self.assertEqual(value["suites"]["gsm8k"]["only_second_correct"], 1)
            self.assertFalse((root / "cache/runs/compare-run").exists())

    def test_a_named_receipt_whose_results_changed_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self.arm(root, "candidate-run", "candidate", 3, root / "candidate-check")
            stored = self.arm(root, "stored-run", "candidate", 4, root / "store")
            with (root / "cache/runs/candidate-run/candidate.jsonl").open("a") as results:
                results.write(json.dumps({"id": "gsm8k-9", "suite": "gsm8k", "group": "gsm8k", "body_sha256": "b",
                                          "correct": True, "extracted": "1", "content": "",
                                          "finish_reason": "stop"}) + "\n")
            code, value = self.compare(root, "compare-run", ["candidate", "reference"], [f"candidate={candidate}"],
                                       stored=[f"reference={stored}"])
            self.assertIn("is not the results the candidate arm's receipt recorded", str(code))
            self.assertIsNone(value)
            # A named receipt that names no results file, or no run, is not bound.
            for field in ("results", "run_id"):
                with self.subTest(field=field):
                    unbound = root / f"no-{field}.json"
                    unbound.write_text(json.dumps({key: item for key, item in json.loads(candidate.read_text()).items()
                                                   if key != field}))
                    code, _ = self.compare(root, "compare-run", ["candidate", "reference"], [f"candidate={unbound}"],
                                           stored=[f"reference={stored}"])
                    self.assertIn("is not the candidate arm's receipt with its results and digest", str(code))

    def test_named_arms_of_two_runs_compare_bound_when_one_build_served_them(self):
        """A qualification comparison of a candidate arm and a recorded arm of
        another gate run: both named, each by its own run and digest."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self.arm(root, "candidate-run", "candidate", 3, root / "candidate-check")
            fp8 = self.arm(root, "fp8-run", "fp8", 3, root / "fp8-gate")
            named = [f"candidate={candidate}", f"fp8={fp8}"]
            code, value = self.compare(root, "fp8-run", ["candidate", "fp8"], named, bound=True)
            self.assertEqual((code, value["bound"]), (0, True))
            self.assertEqual(value["arm_runs"], {"candidate": "candidate-run", "fp8": "fp8-run"})
            fp8 = self.arm(root, "fp8-run", "fp8", 3, root / "fp8-gate", commit="d")
            code, _ = self.compare(root, "fp8-run", ["candidate", "fp8"], named, bound=True)
            self.assertIn("arms of one build", str(code))

    def test_both_arms_of_one_run_are_found_by_its_run_id(self):
        """The manual comparison: `compare --run-id RUN --arms A B`, both arms
        found among the receipts; an arm to be found needs the run ID."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.arm(root, "run", "candidate", 3, root / "gate")
            self.arm(root, "run", "reference", 4, root / "gate")
            self.arm(root, "other", "reference", 0, root / "gate")
            code, value = self.compare(root, "run", ["candidate", "reference"])
            self.assertEqual(code, 0)
            self.assertEqual((value["named_receipts"], value["stored_receipts"]), ({}, {}))
            self.assertEqual(value["arm_runs"], {"candidate": "run", "reference": "run"})
            self.assertEqual(value["suites"]["gsm8k"]["correct"], [3, 4])
            self.assertEqual(value["items"], 4)
            # Its item coverage counts the items both arms answered.
            candidate = self.arm(root, "run", "candidate", 3, root / "gate")
            results = root / "cache/runs/run/candidate.jsonl"
            with results.open("a") as output:
                output.write(json.dumps({"id": "gsm8k-9", "suite": "gsm8k", "group": "gsm8k", "body_sha256": "b",
                                         "correct": True, "extracted": "1", "content": "",
                                         "finish_reason": "stop"}) + "\n")
            candidate.write_text(json.dumps({**json.loads(candidate.read_text()),
                                             "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest()}))
            code, value = self.compare(root, "run", ["candidate", "reference"])
            self.assertEqual((code, value["items"], value["unpaired_items"]),
                             (0, 4, {"candidate": 1, "reference": 0}))
            code, _ = self.compare(root, None, ["candidate", "reference"])
            self.assertIn("--run-id is needed to find the candidate arm's receipt", str(code))

    def test_a_stored_arm_of_an_earlier_run_is_read_by_its_own_receipt(self):
        """The runner's stored last good: another run's receipt (its own arm
        name and run ID) with the results file it names."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = self.arm(root, "earlier-run", "candidate", 4, root / "store")
            self.arm(root, "run", "candidate", 3, root / "gate")
            code, value = self.compare(root, "run", ["candidate", "stored"], stored=[f"stored={stored}"])
            self.assertEqual(code, 0)
            self.assertEqual(value["stored_receipts"], {"stored": str(stored)})
            self.assertEqual(value["suites"]["gsm8k"]["only_second_correct"], 1)
            # A stored arm is not of this run's build: never a bound comparison.
            code, _ = self.compare(root, "run", ["candidate", "stored"], stored=[f"stored={stored}"], bound=True)
            self.assertIn("not a stored arm", str(code))
            with (root / "cache/runs/earlier-run/candidate.jsonl").open("a") as results:
                results.write("\n")
            code, _ = self.compare(root, "run", ["candidate", "stored"], stored=[f"stored={stored}"])
            self.assertIn("is not the results the stored arm's receipt recorded", str(code))


class DeclaredCompareTests(ArmFiles, unittest.TestCase):
    """Each model's declared evals_compare, its arguments filled as checks.py
    fills a check run's (a run ID of its own, an earlier check's receipt by
    that check's id), compares the candidate arm its evals_candidate check
    recorded in another check run, under that run's ID."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_each_declared_compare_binds_its_evals_candidate_check_receipt(self):
        declared = [(path.parent.name, json.loads(path.read_text())["checks"])
                    for path in sorted(self.ROOT.glob("models/*/qual.json"))]
        declared = [(model, checks) for model, checks in declared if "evals_compare" in checks]
        self.assertTrue(declared)
        for model, checks in declared:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = {"url": "http://unused", "model": "m", "server_log": str(root / "server.log"),
                           "fp8_gsm8k_run": "fp8-run"}
                ran = [arg.format_map({**context, "run_id": "candidate-check-run"})
                       for arg in checks["evals_candidate"]["args"]]
                arm, run = (ran[ran.index(flag) + 1] for flag in ("--arm", "--run-id"))
                candidate = self.arm(root, run, arm, 3, root / "candidate-check")
                context.update(
                    run_id="compare-check-run", evals_candidate=str(candidate),
                    stored_evals_candidate=str(self.arm(root, "stored-run", "candidate", 3, root / "store")),
                    fp8_gsm8k=str(self.arm(root, "fp8-run", "fp8", 3, root / "fp8-gate")))
                arguments = [arg.format_map(context) for arg in checks["evals_compare"]["args"]]
                receipts = root / "compare-check"
                with patch.object(evals, "CACHE", root / "cache"), \
                        patch.dict(os.environ, {"CHOPINFER_QUAL_RECEIPTS": str(receipts)}), \
                        patch("sys.argv", ["evals.py", *arguments]), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                    os.environ.pop("CHOPINFER_QUAL_CONFIG", None)
                    evals.main()
                self.assertEqual(exit.exception.code, 0)
                value = json.loads(next(receipts.glob("*-evals-compare.json")).read_text())
                self.assertIn(str(candidate), value["named_receipts"].values())
                self.assertEqual(value["unpaired_items"], {name: 0 for name in value["arms"]})


class ResumeTests(unittest.TestCase):
    """A run reuses a cached answer only from the same producer: checkpoint,
    build, served model and request settings, for the same request body."""

    ITEMS = [{"id": f"gsm8k-{index}", "suite": "gsm8k", "group": "gsm8k", "expected": "1",
              "body": {"messages": [{"role": "user", "content": f"q{index}"}], "max_tokens": 8}}
             for index in range(3)]

    PROFILE = {"name": "spec", "tensor_parallel_size": 4, "context_parallel_size": 1,
               "speculation": {"mode": "fixed", "tokens": 3, "drafter": "mtp"}, "kv_cache_dtype": "fp8",
               "prefix_cache": True, "max_sequences": 64}

    def client(self, checkpoint, serving=None):
        calls = []
        serving = {"profile": self.PROFILE, "model_options": {"topk": 8}, "model_options_qualified": True,
                   "modalities": ["text", "image"], **(serving or {})}

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def identity(self, model=None):
                health = {key: value for key, value in serving.items() if key != "modalities"}
                return {"model": "m", "health": {"serving": {"checkpoint_identity": checkpoint * 64,
                                                             "source_commit": "c" * 40, **health}}}

            def request(self, path, body=None):
                if path == "/v1/models":
                    return {"data": [{"id": "m", "modalities": serving["modalities"]}]}, 0.01
                calls.append(body)
                return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}],
                        "usage": {}}, 0.01
        return Client, calls

    def arm_run(self, root, checkpoint, top_logprobs=0, refusal=None, serving=None):
        """Run the arm; `refusal`, when given, is what confirming the live
        server raises. Returns the completion requests made and the receipt,
        or the refusal when none was written."""
        client, calls = self.client(checkpoint, serving)
        written = []
        args = SimpleNamespace(url="http://unused", timeout=10, model=None, suites=["gsm8k"], long_lengths=None,
                               limit=None, arm="arm", run_id="run", mechanics_only=False, concurrency=2,
                               long_concurrency=1, top_logprobs=top_logprobs, server_log=None)
        with contextlib.ExitStack() as stack:
            if refusal is not None:
                stack.enter_context(patch.object(evals, "confirm_live", side_effect=SystemExit(refusal)))
            with patch.object(evals, "CACHE", root), patch.object(evals, "fetch", return_value={}), \
                    patch.object(evals, "build_items", return_value=self.ITEMS), \
                    patch.object(evals, "Client", client), \
                    patch.object(evals, "receipt", side_effect=lambda check, server, value, *rest: written.append(value)), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                evals.run(args)
        return len(calls), written[-1] if written else stopped.exception.code

    def test_a_run_with_a_server_log_refuses_a_server_off_the_pin(self):
        import declaration
        config = sorted(declaration.ROOT.glob("models/*/qual.json"))[0]
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
            client, calls = self.client("e")
            args = SimpleNamespace(url="http://unused", timeout=10, model=None, suites=["gsm8k"], long_lengths=None,
                                   limit=None, arm="arm", run_id="run", mechanics_only=False, concurrency=2,
                                   long_concurrency=1, top_logprobs=0, server_log=Path(directory) / "server.log")
            with patch.object(evals, "CACHE", Path(directory)), patch.object(evals, "fetch", return_value={}), \
                    patch.object(evals, "build_items", return_value=self.ITEMS), \
                    patch.object(evals, "Client", client), patch.object(evals, "receipt") as written, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                evals.run(args)
            written.assert_not_called()
            self.assertEqual(calls, [])

    def test_cached_answers_of_another_producer_are_asked_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.arm_run(root, "a")[0], 3)
            calls, value = self.arm_run(root, "a")
            self.assertEqual((calls, value["resumed"], value["passed"]), (0, 3, True))
            # Another checkpoint, or other request settings, reuse nothing.
            calls, value = self.arm_run(root, "b")
            self.assertEqual((calls, value["resumed"], value["passed"]), (3, 0, True))
            self.assertEqual(value["producer"]["checkpoint_identity"], "b" * 64)
            self.assertEqual(self.arm_run(root, "b", top_logprobs=2)[0], 3)
            # Rows no validated capture sealed (an older run's) are asked again too.
            (root / "runs/run/arm.seal.json").unlink()
            self.assertEqual(self.arm_run(root, "b", top_logprobs=2)[0], 3)

    def test_answers_from_another_serving_configuration_are_asked_again(self):
        changes = {"profile": {"name": "ar", "speculation": {"mode": "off", "tokens": 0, "drafter": None}},
                   "kv_cache_dtype": {"kv_cache_dtype": "bf16"}, "context_parallel": {"context_parallel_size": 4}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.arm_run(root, "a")[0], 3)
            for name, change in changes.items():
                with self.subTest(name):
                    serving = {"profile": {**self.PROFILE, **change}}
                    calls, value = self.arm_run(root, "a", serving=serving)
                    self.assertEqual((calls, value["resumed"]), (3, 0))
                    self.assertEqual(value["producer"]["profile"], serving["profile"])
            for name, serving in {"model options": {"model_options": {"topk": 4}},
                                  "modalities": {"modalities": ["text"]}}.items():
                with self.subTest(name):
                    calls, value = self.arm_run(root, "a", serving=serving)
                    self.assertEqual((calls, value["resumed"]), (3, 0))
            # An unchanged configuration reuses them.
            self.assertEqual(self.arm_run(root, "a", serving={"modalities": ["text"]})[0], 0)

    def test_an_unbound_capture_is_not_reused_by_a_bound_run(self):
        # R1-14 round 3: an unbound run validates no response identity, so
        # answers a foreign process wrote are sealed; a bound retry must ask
        # every item again under the real binding and confirmation.
        import common
        import declaration
        from test_determinism import startup_log
        config = sorted(declaration.ROOT.glob("models/*/qual.json"))[0]
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
            root = Path(directory)
            pin = common.checkpoint_pin()
            calls = []

            class Client:
                foreign = False

                def __init__(self, *args, **kwargs):
                    self.responders = set()

                def identity(self, model=None):
                    return {"url": "http://127.0.0.1:39001", "model": pin["served_model_name"],
                            "health": {"serving": {"checkpoint_identity": pin["checkpoint_identity"],
                                                   "startup_nonce": "a" * 32, "source_commit": "c" * 40,
                                                   "profile": self.PROFILE}}}

                def request(self, path, body=None):
                    self.responders.add(("1" * 64, "b" * 32) if Client.foreign and path != "/v1/models"
                                        else (pin["checkpoint_identity"], "a" * 32))
                    if path == "/v1/models":
                        return {"data": [{"id": pin["served_model_name"], "modalities": ["text"]}]}, 0.01
                    calls.append(body)
                    return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}],
                            "usage": {}}, 0.01
            Client.PROFILE = self.PROFILE

            def run(server_log):
                calls.clear()
                written = []
                args = SimpleNamespace(url="http://127.0.0.1:39001", timeout=10, model=None, suites=["gsm8k"],
                                       long_lengths=None, limit=None, arm="arm", run_id="run",
                                       mechanics_only=False, concurrency=2, long_concurrency=1, top_logprobs=0,
                                       server_log=server_log)
                with patch.object(evals, "CACHE", root), patch.object(evals, "fetch", return_value={}), \
                        patch.object(evals, "build_items", return_value=self.ITEMS), \
                        patch.object(evals, "Client", Client), \
                        patch.object(evals, "receipt", side_effect=lambda check, server, value, *rest:
                                     written.append({"server": server, **value})), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    evals.run(args)
                return len(calls), written[-1] if written else stopped.exception.code

            log = startup_log(root, pin["checkpoint_identity"])
            # A bound capture of foreign answers is refused: the control.
            Client.foreign = True
            calls_made, refusal = run(log)
            self.assertEqual(calls_made, 3)
            self.assertIn("named another process", refusal)
            # An unbound capture of the same foreign answers seals them.
            calls_made, unbound = run(None)
            self.assertEqual((calls_made, unbound["resumed"]), (3, 0))
            Client.foreign = False
            calls_made, value = run(log)
            self.assertEqual((calls_made, value["resumed"], value["passed"]), (3, 0, True))
            self.assertIsNone(unbound["producer"]["binding"])
            self.assertEqual(value["producer"]["binding"]["checkpoint_identity"], pin["checkpoint_identity"])
            self.assertTrue(value["server"]["checkpoint"]["startup_nonce"])
            # Bound answers serve a bound retry, and unbound use keeps its own.
            self.assertEqual(run(log)[0], 0)
            calls_made, value = run(None)
            self.assertEqual((calls_made, value["resumed"]), (3, 0))
            self.assertEqual(run(None)[0], 0)

    def test_a_refused_capture_leaves_no_answer_to_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refusal = "refused: responses from the server named another process"
            self.assertEqual(self.arm_run(root, "a", refusal=refusal), (3, refusal))
            # The retry asks every item again and sets the refused rows aside.
            calls, value = self.arm_run(root, "a")
            self.assertEqual((calls, value["resumed"], value["passed"]), (3, 0, True))
            results = root / "runs/run/arm.jsonl"
            self.assertEqual(len(results.read_text().splitlines()), 3)
            self.assertEqual(len((root / "runs/run/arm.unsealed.jsonl").read_text().splitlines()), 3)
            self.assertEqual(value["results_sha256"], hashlib.sha256(results.read_bytes()).hexdigest())

    def test_an_edited_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.arm_run(root, "a")[0], 3)
            results = root / "runs/run/arm.jsonl"
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            for row in rows:
                row.update(content="#### 2", extracted="2", correct=False)
            results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            calls, refusal = self.arm_run(root, "a")
            self.assertEqual(calls, 0)
            self.assertIn("refused", refusal)
            self.assertIn("not the results", refusal)


if __name__ == "__main__":
    unittest.main()
