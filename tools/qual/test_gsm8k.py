"""Offline checks for the GSM8K answer parser and the receipt rescore."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import declaration
import evals
import gsm8k
from gsm8k import extract, parse


class ParseTests(unittest.TestCase):
    def test_marked_answer(self):
        self.assertEqual(parse("so\n#### 18"), ("18", "####"))
        self.assertEqual(parse("#### 1\nthen\n#### 2,500"), ("2.5E+3", "####"))
        self.assertEqual(parse("#### -3.50"), ("-3.5", "####"))
        self.assertEqual(parse("#### $75.00"), ("75", "####"))
        self.assertEqual(parse("#### \\$1,200"), ("1.2E+3", "####"))

    def test_boxed_answer(self):
        self.assertEqual(parse(r"$$\boxed{9360}$$"), ("9.36E+3", "boxed"))
        self.assertEqual(parse(r"$$\boxed{15} \text{ liters}$$"), ("15", "boxed"))
        self.assertEqual(parse(r"\boxed{3} first, finally \boxed {4}"), ("4", "boxed"))

    def test_marked_answer_wins(self):
        self.assertEqual(parse(r"#### 7" "\n" r"\boxed{8}"), ("7", "####"))
        self.assertEqual(parse(r"\boxed{8}" "\n" r"#### 7"), ("7", "####"))

    def test_commas_and_dollars(self):
        self.assertEqual(parse(r"\boxed{9{,}360}"), ("9.36E+3", "boxed"))
        self.assertEqual(parse(r"\boxed{\$25{,}000}"), ("2.5E+4", "boxed"))
        self.assertEqual(parse(r"\boxed{$18.00}"), ("18", "boxed"))
        self.assertEqual(parse(r"\boxed{ -7 }"), ("-7", "boxed"))
        self.assertEqual(parse(r"\boxed{1,2}"), (None, None))
        self.assertEqual(parse(r"\boxed{12,34.5}"), (None, None))
        self.assertEqual(parse(r"\boxed{1.}"), (None, None))

    def test_nested_braces(self):
        self.assertEqual(parse(r"\boxed{\boxed{6}}"), ("6", "boxed"))
        self.assertEqual(parse(r"\boxed{{9}}"), (None, None))

    def test_units_percent_and_fractions_are_unparsed(self):
        for text in (r"\boxed{50\%}", r"\boxed{50%}", r"\boxed{12 \text{ dollars}}", r"\boxed{5 \text{ million}}",
                     r"\boxed{5\text{ apples}}", r"\boxed{\frac{1}{4}}", r"\boxed{\dfrac{72}{2}}", r"\boxed{-3/4}",
                     r"\boxed{\frac{400}{11}}"):
            self.assertEqual(parse(text), (None, None), text)

    def test_non_numeric_boxed_is_unparsed(self):
        for text in (r"\boxed{x = 5}", r"\boxed{5 \text{ or } 6}", r"\boxed{\text{18}}", r"\boxed{\text{about } 5}",
                     r"\boxed{2\frac{1}{2}}", r"\boxed{9", r"\boxed{9} and \boxed{10"):
            self.assertEqual(parse(text), (None, None), text)

    def test_empty(self):
        for text in (None, "", "the answer is 18", r"\boxed{}"):
            self.assertEqual(parse(text), (None, None))
            self.assertIsNone(extract(text))

    def test_evals_scores_with_the_same_parser(self):
        self.assertEqual(evals.score({"suite": "gsm8k", "expected": "18"}, r"\boxed{18}"), ("18", True))


class FloorTests(unittest.TestCase):
    def test_each_model_passes_at_its_declared_floor(self):
        root = Path(__file__).resolve().parents[2]
        rows = lambda correct: [{"correct": index < correct} for index in range(200)]
        for model, value in declaration.previewed().items():
            floor = value["preview"]["gsm8k_floor"]
            config = str(root / "models" / model / "qual.json")
            with self.subTest(model=model), patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": config}):
                self.assertEqual(gsm8k.screen_floor(), floor)
                self.assertTrue(gsm8k.screen_passed(rows(floor), [], False))
                self.assertFalse(gsm8k.screen_passed(rows(floor - 1), [], False))
                self.assertFalse(gsm8k.screen_passed(rows(200), [{"error": "x"}], False))
                self.assertFalse(gsm8k.screen_passed(rows(floor)[:-1], [], False))

    def test_an_explicit_floor_screens_only_a_model_without_a_declared_one(self):
        root = Path(__file__).resolve().parents[2]
        previewed = declaration.previewed()
        # A shipped model without a preview section, and a copy of a previewed
        # declaration without its section, as every shipped model may preview.
        with tempfile.TemporaryDirectory() as directory:
            value = json.loads(json.dumps(next(iter(previewed.values()))))
            del value["preview"]
            value["model"] = "unpreviewed"
            folder = Path(directory) / "models" / "unpreviewed"
            folder.mkdir(parents=True)
            (folder / "qual.json").write_text(json.dumps(value))
            undeclared = [(path.parent.name, {"CHOPINFER_QUAL_CONFIG": str(path)})
                          for path in sorted(root.glob("models/*/qual.json")) if path.parent.name not in previewed]
            undeclared.append(("unpreviewed", {"CHOPINFER_QUAL_CONFIG": str(folder / "qual.json"),
                                               "CHOPINFER_QUAL_MODEL_ROOT": directory}))
            for model, environment in undeclared:
                with self.subTest(model=model), patch.dict(os.environ, environment):
                    with self.assertRaisesRegex(ValueError, "--floor"):
                        gsm8k.screen_floor()
                    self.assertEqual(gsm8k.screen_floor(193), 193)
                    for bad in (0, 201):
                        with self.assertRaises(ValueError):
                            gsm8k.screen_floor(bad)
        for model, value in previewed.items():
            config = str(root / "models" / model / "qual.json")
            with self.subTest(model=model), patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": config}):
                with self.assertRaisesRegex(ValueError, "declares its GSM8K floor"):
                    gsm8k.screen_floor(value["preview"]["gsm8k_floor"] - 1)


    def test_a_variant_uses_its_own_declared_floor(self):
        """A checkpoint variant (qual.VARIANT.json) is screened at its own
        floor, not its model's, and one that declares none names it with
        --floor even when its model declares one."""
        model, value = next(iter(declaration.previewed().items()))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "models" / model
            folder.mkdir(parents=True)
            (folder / "qual.json").write_text(json.dumps(value))
            variant = json.loads(json.dumps(value))
            variant.pop("configurations", None)
            variant["variant"] = "demo"
            variant["preview"]["gsm8k_floor"] = 150
            (folder / "qual.demo.json").write_text(json.dumps(variant))
            del variant["preview"]
            variant["variant"] = "bare"
            (folder / "qual.bare.json").write_text(json.dumps(variant))
            select = lambda name: patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(folder / name),
                                                         "CHOPINFER_QUAL_MODEL_ROOT": directory})
            with select("qual.demo.json"):
                self.assertEqual(gsm8k.screen_floor(), 150)
                # Its declared floor is not replaced by an explicit one.
                with self.assertRaisesRegex(ValueError, f"{model}.demo declares its GSM8K floor \\(150\\)"):
                    gsm8k.screen_floor(140)
            with select("qual.bare.json"):
                with self.assertRaisesRegex(ValueError, f"{model}.bare declares no preview"):
                    gsm8k.screen_floor()
                self.assertEqual(gsm8k.screen_floor(140), 140)
            with select("qual.json"):
                self.assertEqual(gsm8k.screen_floor(), value["preview"]["gsm8k_floor"])


def examples(directory):
    """A frozen-shape GSM8K input, so the screen runs without the reference root."""
    root = directory / "gsm8k"
    root.mkdir()
    raw = json.dumps([{"index": i, "question": f"Question {i}", "expected": "1"} for i in range(200)]).encode()
    (root / "examples.json").write_bytes(raw)
    (root / "manifest.json").write_text(json.dumps({
        "examples_sha256": hashlib.sha256(raw).hexdigest(), "expected_count": 200, "instruction": "Solve.",
        "temperature": 0, "seed": 17, "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True}}))
    return root


class BindingTests(unittest.TestCase):
    """A scored screen counts only from the pinned checkpoint's live server."""

    def setUp(self):
        self.gate = declaration.path(next(iter(declaration.previewed())))
        self.pin = declaration.load(self.gate)["qualification"]["checkpoint"]

    def run_screen(self, identity):
        server = {"url": "http://127.0.0.1:1", "model": self.pin["served_model_name"],
                  "health": {"serving": {"checkpoint_identity": identity, "startup_nonce": "c" * 32,
                                         "profile": {"tensor_parallel_size": 2, "speculation": {"tokens": 3}}}}}
        response = {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}]}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(self.gate),
                                        "CHOPINFER_QUAL_RECEIPTS": directory}), \
                patch.object(gsm8k, "FIXTURE", examples(Path(directory))), \
                patch.object(gsm8k, "Client") as client, patch.object(gsm8k, "receipt") as receipt, \
                patch.object(sys, "argv", ["gsm8k.py", "--limit", "2"]):
            client.return_value.identity.return_value = server
            client.return_value.request.return_value = (response, 0.1)
            client.return_value.responders = {(identity, "c" * 32)}
            with self.assertRaises(SystemExit) as stopped:
                gsm8k.main()
            return stopped.exception.code, client, receipt

    def test_the_pinned_server_is_bound_and_its_serving_recorded(self):
        _, client, receipt = self.run_screen(self.pin["checkpoint_identity"])
        server, result = receipt.call_args.args[1], receipt.call_args.args[2]
        self.assertEqual(server["checkpoint"]["startup_nonce"], "c" * 32)
        self.assertEqual(result["serving"], {"tensor_parallel_size": 2, "speculation": {"tokens": 3}})
        self.assertEqual(client.return_value.request.call_count, 2)

    def test_another_checkpoint_is_refused_before_any_request(self):
        code, client, receipt = self.run_screen("b" * 64)
        self.assertRegex(code, "refused: the server loaded checkpoint b{64}, not the pinned")
        client.return_value.request.assert_not_called()
        receipt.assert_not_called()


class RescoreTests(unittest.TestCase):
    def screen(self, root, contents, server="m", floor=194):
        """A 200-row receipt read by the `####`-only rule; the first rows answer with `contents`,
        served as `server`; `floor` None leaves it out, as receipts before the recorded floor did."""
        responses = root / "run" / "responses"
        responses.mkdir(parents=True)
        rows = []
        for index in range(200):
            content = contents[index] if index < len(contents) else f"#### {index}"
            (responses / f"{index}.json").write_text(json.dumps({"response": {"choices": [
                {"message": {"content": content}, "finish_reason": "stop"}]}}))
            actual, expected = extract(content) if "####" in content else None, gsm8k.number(str(index))
            rows.append({"index": index, "expected": expected, "actual": actual, "finish_reason": "stop",
                         "correct": actual == expected})
        path = root / "run" / "20260930T000000Z-gsm8k.json"
        value = {"check": "gsm8k", "mechanics_only": False, "server": {"model": server},
                 "passed": False, "correct": sum(r["correct"] for r in rows), "errors": [], "rows": rows}
        if floor is not None:
            value["floor"] = floor
        path.write_text(json.dumps(value))
        return path

    def run_rescore(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as code:
            gsm8k.rescore(argv)
        return code.exception.code, output.getvalue()

    def test_rescore_reads_boxed_answers_and_keeps_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.screen(root, [r"\boxed{0}", r"\boxed{x}"])
            before = path.read_bytes()
            code, table = self.run_rescore(["--output", str(root / "out"), str(path)])
            self.assertEqual(code, 0, table)
            self.assertEqual(path.read_bytes(), before)
            value = json.loads((root / "out" / path.name).read_text())
            self.assertEqual((value["correct"], value["rescore"]["previous_correct"]), (199, 198))
            self.assertEqual(value["answer_rules"], {"####": 198, "boxed": 1, "unparsed": 1})
            self.assertEqual(value["rescore"]["gained"], [0])
            self.assertEqual([value["rows"][i]["rule"] for i in (0, 1, 2)], ["boxed", None, "####"])
            self.assertTrue(value["passed"])
            self.assertIn("| 198 | 199 | True |", table)

    def test_a_receipt_without_a_floor_is_judged_at_its_served_model_floor(self):
        root = Path(__file__).resolve().parents[2]
        models = {model: (json.loads((root / "models" / model / "profiles.json").read_text())["served_model_name"],
                          value["preview"]["gsm8k_floor"])
                  for model, value in declaration.previewed().items()}
        self.assertGreater(len(set(floor for _, floor in models.values())), 1)
        for model, (name, floor) in models.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                # Every item from the floor on answers wrong, so exactly `floor` are correct.
                path = self.screen(Path(tmp), [], server=name, floor=None)
                value = json.loads(path.read_text())
                for row in value["rows"][floor:]:
                    row.update(actual="x", correct=False)
                path.write_text(json.dumps(value))
                shutil.rmtree(Path(tmp) / "run" / "responses")
                with patch.dict(os.environ):
                    os.environ.pop("CHOPINFER_QUAL_CONFIG", None)
                    copy = gsm8k.rescore_receipt(path, None)
                self.assertEqual((copy["floor"], copy["correct"], copy["passed"]), (floor, floor, True))
                for row in value["rows"][floor - 1:floor]:
                    row.update(actual="x", correct=False)
                path.write_text(json.dumps(value))
                with patch.dict(os.environ):
                    os.environ.pop("CHOPINFER_QUAL_CONFIG", None)
                    self.assertFalse(gsm8k.rescore_receipt(path, None)["passed"])
                # Another model's configuration, or an unknown served name, is refused.
                for other in models:
                    config = str(root / "models" / other / "qual.json")
                    with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": config}):
                        if other == model:
                            self.assertEqual(gsm8k.rescore_receipt(path, None)["floor"], floor)
                        else:
                            with self.assertRaisesRegex(ValueError, "CHOPINFER_QUAL_CONFIG selects"):
                                gsm8k.rescore_receipt(path, None)
                value["server"]["model"] = "alias"
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "names no single previewed model"):
                    gsm8k.rescore_receipt(path, None)

    def test_rescore_without_responses_lists_unread_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.screen(root, [r"\boxed{0}"])
            for file in (root / "run" / "responses").iterdir():
                file.unlink()
            (root / "run" / "responses").rmdir()
            code, _ = self.run_rescore(["--output", str(root / "out"), str(path)])
            self.assertEqual(code, 0)
            value = json.loads((root / "out" / path.name).read_text())
            self.assertEqual((value["correct"], value["rescore"]["unread_indices"]), (199, [0]))
            self.assertEqual(value["answer_rules"], {"####": 199, "boxed": 0, "unparsed": 1})

    def test_rescore_without_responses_keeps_a_recorded_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.screen(root, [])
            shutil.rmtree(root / "run" / "responses")
            value = json.loads(path.read_text())
            for row in value["rows"]:
                row["rule"] = "####"
            value["rows"][5]["rule"] = "boxed"
            path.write_text(json.dumps(value))
            code, _ = self.run_rescore(["--output", str(root / "out"), str(path)])
            self.assertEqual(code, 0)
            value = json.loads((root / "out" / path.name).read_text())
            self.assertEqual(value["rows"][5]["rule"], "boxed")
            self.assertEqual(value["answer_rules"], {"####": 199, "boxed": 1, "unparsed": 0})

    def test_rescore_requires_a_named_response_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.screen(root, [])
            with contextlib.redirect_stderr(io.StringIO()) as error:
                code, _ = self.run_rescore(["--output", str(root / "out"), f"{path}={root / 'missing'}"])
            self.assertEqual(code, 2)
            self.assertIn("no such response directory", error.getvalue())
            self.assertFalse((root / "out").exists())

    def test_rescore_rejects_responses_that_do_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.screen(root, [])
            (root / "run" / "responses" / "5.json").write_text(json.dumps({"response": {"choices": [
                {"message": {"content": "#### 6"}, "finish_reason": "stop"}]}}))
            code, table = self.run_rescore(["--output", str(root / "out"), f"{path}={root / 'run' / 'responses'}"])
            self.assertEqual(code, 1)
            self.assertIn("does not match", table)
            self.assertFalse((root / "out" / path.name).exists())


if __name__ == "__main__":
    unittest.main()
