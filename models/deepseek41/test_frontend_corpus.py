"""DeepSeek V4.1 Flash's committed frontend corpus against the pinned encoder.

`fixtures/deepseek41/frontend/corpus.json` must still be what the checkpoint's
`encoding/encoding.py` renders, and what its tokenizer gives, at revision
1bae0e10 (`frontend_cases.py` pins it), and it must pass the coverage and
divergence gate with `models/deepseek41/frontend-divergences.json`. Needs
transformers and the pinned encoder and tokenizer files, from
`CHOPINFER_DEEPSEEK_TOKENIZER_DIR` or `CHOPINFER_DEEPSEEK_CHECKPOINT`; skipped
without them. The recipe oracle's own helpers (`recipe_reference.py`) are
checked without the recipe package.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/deepseek41/test_frontend_corpus.py
"""

from pathlib import Path
import base64
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import Model, corpus_test_case, reference_variables  # noqa: E402
from models.deepseek41 import recipe_reference  # noqa: E402

FrontendCorpus = corpus_test_case("deepseek41", ROOT)


def reference_directory(test):
    """The directory of the pinned encoder and tokenizer files, or a skip."""
    model = Model("deepseek41", root=ROOT)
    variables = reference_variables(model.pin)
    if not any(os.environ.get(name) for name in variables):
        test.skipTest(f"{' and '.join(variables)} are unset")
    return model, model.checkpoint()


class ChatRenders(unittest.TestCase):
    def test_chat_fixture_uses_served_requests(self):
        from tools.refrun.frontend_corpus import model_reference, template_source

        model, _ = reference_directory(self)
        sys.path.insert(0, str(ROOT / "tools/qual"))
        from tools.qual.frontend_goldens import capture_bodies

        reference = model_reference(model, template_source(model)[0])
        rows = json.loads((ROOT / "fixtures/deepseek41/frontend/chat-renders.json").read_bytes())
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(model.directory / "qual.json")}):
            bodies, _ = capture_bodies(model.declaration["models"][0]["id"], rows, [])
        self.assertEqual(len(rows), 61)
        for row, body in zip(rows, bodies):
            with self.subTest(id=row["id"]):
                result = reference.outcome(model.hf_input(body))
                self.assertEqual({key: row[key] for key in ("rendered_text", "token_ids")},
                                 {key: result[key] for key in ("rendered_text", "token_ids")})


class ToolImageOrder(unittest.TestCase):
    def test_vendor_collects_tool_images_after_sorting(self):
        from models.deepseek41.frontend_cases import PIN, tool_image_cases

        _, checkpoint = reference_directory(self)
        source = checkpoint / PIN["source"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), PIN["source_sha256"])
        encoder = runpy.run_path(str(source))
        for name, request, order in tool_image_cases():
            with self.subTest(name=name):
                prompt, media = encoder["encode_messages"](
                    request["messages"], thinking_mode="chat", return_multi_modal_data=True)
                self.assertEqual([image["url"] for image in media["images"]], order)
                self.assertEqual(prompt.count("<｜deepseek_image｜>"), len(order))


class RecipeOracleHelpers(unittest.TestCase):
    """recipe_reference.py's own logic, with stand-ins for the recipe package."""

    def test_the_installed_digest_covers_the_recorded_files_and_refuses_a_changed_one(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "pkg").mkdir()
        (directory / "pkg/__init__.py").write_bytes(b"x = 1\n")
        info = directory / "pkg-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n")

        def line(path, data):
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            return f"{path},sha256={digest},{len(data)}"

        (info / "RECORD").write_text("\n".join([
            line("pkg/__init__.py", b"x = 1\n"), line("pkg-1.0.dist-info/METADATA", b"other"),
            "pkg/__pycache__/__init__.cpython-312.pyc,,", "pkg-1.0.dist-info/RECORD,,"]) + "\n")
        distribution = importlib.metadata.PathDistribution(info)
        expected = hashlib.sha256((line("pkg/__init__.py", b"x = 1\n").rsplit(",", 1)[0] + "\n").encode())
        self.assertEqual(recipe_reference.installed_sha256(distribution), expected.hexdigest())
        (directory / "pkg/__init__.py").write_bytes(b"x = 2\n")
        with self.assertRaisesRegex(SystemExit, "pkg/__init__.py no longer has the hash its RECORD lists"):
            recipe_reference.installed_sha256(distribution)

    def test_the_parser_reads_the_completion_without_its_stop_token(self):
        self.assertEqual(recipe_reference.completion_ids([5, 6, 1], [1]), [5, 6])
        self.assertEqual(recipe_reference.completion_ids([5, 6], [1]), [5, 6])
        self.assertEqual(recipe_reference.completion_ids([1, 5], [1]), [1, 5])

    def test_a_choice_reads_as_a_turn(self):
        choice = {"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "r",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}
        self.assertEqual(recipe_reference.turn(choice), {
            "reasoning": "r", "content": "", "tool_calls": [{"name": "f", "arguments": "{}"}],
            "finish_reason": "tool_calls"})

    def test_conversion_refusals_carry_their_status_and_message(self):
        class ConversionError(Exception):
            def __init__(self, status_code, body):
                self.status_code, self.body = status_code, body

        def request(body):
            if body["messages"] == "unreadable":
                raise ValueError("invalid request body: unknown variant `developer`")
            assert body["model"] == recipe_reference.MODEL
            return SimpleNamespace(convert=lambda options: (_ for _ in ()).throw(ConversionError(
                400, json.dumps({"error": {"message": "Prompt must contain the word 'json'"}}))))

        recipe = SimpleNamespace(ChatCompletionRequest=request, ConversionError=ConversionError)
        self.assertEqual(recipe_reference.convert(recipe, None, {"messages": "unreadable"}),
                         (None, {"status": 400, "message": "invalid request body: unknown variant `developer`"}))
        self.assertEqual(recipe_reference.convert(recipe, None, {"messages": []}),
                         (None, {"status": 400, "message": "Prompt must contain the word 'json'"}))

    def test_an_internal_conversion_error_stops_the_oracle(self):
        class ConversionError(Exception):
            def __init__(self, status_code, body):
                self.status_code, self.body = status_code, body

        for status in (500, 503, 302):
            def request(body, status=status):
                return SimpleNamespace(convert=lambda options: (_ for _ in ()).throw(ConversionError(
                    status, json.dumps({"error": {"message": "internal"}}))))

            recipe = SimpleNamespace(ChatCompletionRequest=request, ConversionError=ConversionError)
            with self.subTest(status=status), self.assertRaisesRegex(
                    SystemExit, f"deepseek-recipe failed with status {status}"):
                recipe_reference.convert(recipe, None, {"messages": []})


class StrictSchemaReference(unittest.TestCase):
    """The reference's strict schema normalization (`frontend_cases.py`)
    against the same expectations as strict_schema.rs's tests."""

    def refusal(self, node):
        from models.deepseek41.frontend_cases import strict_tool_parameters

        with self.assertRaises(ValueError) as refused:
            strict_tool_parameters({"type": "object", "properties": {"x": node}})
        return str(refused.exception)

    def test_refusal_floats_are_spelled_as_the_recipe_spells_them(self):
        # Each pair is what the deepseek-recipe 0.2.0 wheel writes for the
        # value as an integer's minimum.
        for sent, written in [
                ("1e-5", "0.00001"), ("1.5e-5", "0.000015"), ("1e-6", "1e-6"), ("-2.5e-6", "-2.5e-6"),
                ("0.0001", "0.0001"), ("0.1", "0.1"), ("1e15", "1000000000000000.0"),
                ("9.999e15", "9999000000000000.0"), ("1e16", "1e+16"), ("1.5e16", "1.5e+16"),
                ("1.2345678901234568e17", "1.2345678901234568e+17"), ("100.0", "100.0"), ("-0.0", "-0.0"),
                ("123.456", "123.456"), ("1e300", "1e+300"), ("5e-324", "5e-324"),
                ("18446744073709551616", "1.8446744073709552e+19"),
                # Halfway between two shortest spellings: the even one.
                ("1000000000000000.25", "1000000000000000.2"), ("807175938442761.25", "807175938442761.2"),
                ("-932080889275546.25", "-932080889275546.2"), ("253424307849306.625", "253424307849306.62"),
                ("72723312945259.125", "72723312945259.12")]:
            with self.subTest(sent=sent):
                self.assertEqual(self.refusal({"type": "integer", "minimum": json.loads(sent)}),
                                 f"Invalid tool parameters schema : invalid type: floating point `{written}`, "
                                 "expected i64")

    def test_refusal_strings_are_quoted_as_json(self):
        # Only quotes, backslashes and control characters are escaped, so
        # U+1C89, new in Unicode 16, which Python's Unicode database does not
        # hold, stays as it is, as strict_schema.rs keeps it.
        for sent, written in [("\u1c89", '"\u1c89"'), ("\U0001f600", '"\U0001f600"'),
                              ("x'y\"\\\n\t\x01", r'''"x'y\"\\\n\t\u0001"'''), ("\x7f", '"\x7f"'),
                              ("a\u0301", '"a\u0301"'), ("\u0378", '"\u0378"'),
                              ("\u200b\ufeff", '"\u200b\ufeff"')]:
            with self.subTest(sent=sent):
                self.assertEqual(self.refusal({"type": "number", "minimum": sent}),
                                 f"Invalid tool parameters schema : invalid type: string {written}, "
                                 "expected a JSON number")

    def test_numbers_keep_the_digits_sent(self):
        # The recipe rounds a constant beyond u64 to a float; the reference
        # keeps it whole, as the served format does (a deliberate divergence).
        from models.deepseek41.frontend_cases import compact_json, strict_tool_parameters

        schema = json.loads('{"type": "object", "properties": {"x": {"type": "number", '
                            '"const": 18446744073709551617}}}')
        self.assertEqual(compact_json(strict_tool_parameters(schema)["properties"]["x"]),
                         '{"type":"number","const":18446744073709551617}')

    def test_positional_nodes_are_refused(self):
        # A sequence where the recipe reads one single-type node, which serde
        # would read by position, is refused (a deliberate divergence).
        for node, refusal in [
                (["string", None, None, None, None],
                 "invalid type: sequence, expected internally tagged enum JsonSchemaType"),
                (["object", {"a": {"type": "string"}}, False, ["a"], None, None],
                 "invalid type: sequence, expected internally tagged enum JsonSchemaType"),
                (["date"], "unknown variant `date`, expected one of `object`, `string`, `number`, `integer`, "
                           "`boolean`, `array`, `null`"),
                ([], "missing field `type`")]:
            with self.subTest(node=node):
                self.assertEqual(self.refusal({"anyOf": [node]}),
                                 f"Invalid tool parameters schema : field `anyOf`: {refusal}")


if __name__ == "__main__":
    unittest.main()
