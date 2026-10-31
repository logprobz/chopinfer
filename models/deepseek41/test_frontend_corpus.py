"""DeepSeek V4.1 Flash's committed frontend corpus against the pinned encoder.

`fixtures/deepseek41/frontend/corpus.json` must still be what the checkpoint's
`encoding/encoding.py` renders, and what its tokenizer gives, at revision
1bae0e10 (`frontend_cases.py` pins it), and it must pass the coverage and
divergence gate with `models/deepseek41/frontend-divergences.json`. Needs
transformers and the pinned encoder and tokenizer files, from
`CHOPINFER_DEEPSEEK_CHECKPOINT`; skipped without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/deepseek41/test_frontend_corpus.py
"""

from pathlib import Path
import hashlib
import json
import os
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import corpus_test_case  # noqa: E402

FrontendCorpus = corpus_test_case("deepseek41", ROOT)


class ChatRenders(unittest.TestCase):
    def test_chat_fixture_uses_served_requests(self):
        from tools.refrun.frontend_corpus import Model, model_reference, template_source

        sys.path.insert(0, str(ROOT / "tools/qual"))
        from tools.qual.frontend_goldens import capture_bodies

        model = Model("deepseek41", root=ROOT)
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

        checkpoint = os.environ.get(PIN["checkpoint_env"])
        if not checkpoint:
            self.skipTest(f"{PIN['checkpoint_env']} is unset")
        source = Path(checkpoint) / PIN["source"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), PIN["source_sha256"])
        encoder = runpy.run_path(str(source))
        for name, request, order in tool_image_cases():
            with self.subTest(name=name):
                prompt, media = encoder["encode_messages"](
                    request["messages"], thinking_mode="chat", return_multi_modal_data=True)
                self.assertEqual([image["url"] for image in media["images"]], order)
                self.assertEqual(prompt.count("<｜deepseek_image｜>"), len(order))

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
                ("18446744073709551616", "1.8446744073709552e+19")]:
            with self.subTest(sent=sent):
                self.assertEqual(self.refusal({"type": "integer", "minimum": json.loads(sent)}),
                                 f"Invalid tool parameters schema : invalid type: floating point `{written}`, "
                                 "expected i64")

    def test_refusal_strings_are_quoted_as_the_recipe_quotes_them(self):
        # U+1C89, new in Unicode 16, which Python's Unicode database does not
        # hold, stays as it is, as Rust's Debug and the wheel keep it.
        for sent, written in [("\u1c89", '"\u1c89"'), ("\U0001f600", '"\U0001f600"'), ("x'y\"\n", '"x\'y\\"\\n"'),
                              ("a\u0301", '"a\\u{301}"'), ("\u0378", '"\\u{378}"'),
                              ("\u200b\ufeff", '"\\u{200b}\\u{feff}"')]:
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
