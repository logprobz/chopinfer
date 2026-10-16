"""Offline needle scoring checks, including the FP8 KV 128K answer and the
gate-declared prompt sources and answer rules."""

from qual_test_config import setUpModule

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import declaration
import needle

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "needle scoring on synthetic answers"


EXPECTED = "8D71A93C,46B2F8E0,C5920AD7"


class NeedleTests(unittest.TestCase):
    def run_answer(self, parts, flags=(), frozen_expected=EXPECTED):
        manifest = {"contexts": [{"total_context": 131072, "expected": EXPECTED}]}
        frozen = {"prompt": [1], "max_tokens": 131071, "total_context": 131072,
                  "expected": frozen_expected}
        events = [(float(i), {"choices": [{"text": part}]})
                  for i, part in enumerate(parts)]
        with tempfile.TemporaryDirectory() as temporary, \
                patch("sys.argv", ["needle.py", "--lengths", "131072", *flags]), \
                patch.object(needle, "Client") as client, \
                patch.object(needle, "FIXTURE", Path(temporary)), \
                patch.object(needle, "receipt") as receipt:
            (Path(temporary) / "manifest.json").write_text(json.dumps(manifest))
            (Path(temporary) / "context-131072.json").write_text(json.dumps(frozen))
            client.return_value.identity.return_value = {"model": "MiMo-V2.6-Flash-MOPD"}
            client.return_value.request.return_value = events, 0.1
            with self.assertRaises(SystemExit) as stopped:
                needle.main()
        result = receipt.call_args.args[2]
        self.assertEqual(stopped.exception.code, 0 if result["passed"] else 1)
        return result

    def test_recorded_fp8_answer(self):
        # The answer a MiMo FP8 KV run produced, with a space after each comma.
        output = "8D71A93C, 46B2F8E0, C5920AD7"
        result = self.run_answer([output])
        self.assertTrue(result["passed"])
        self.assertEqual(result["rows"][0]["output"], output)

    def test_whitespace_and_stream_token_boundaries(self):
        for parts in (["The codes are: 8D71A93C,  46B2F8E0,\nC5920AD7."],
                      ["8D", "71A93C", ",", "\n\t", "46B2", "F8E0,", "\u00a0C5920AD7"],
                      ["8D71 A93C,46B2F8E0,C5920AD7"]):
            with self.subTest(parts=parts):
                self.assertTrue(self.run_answer(parts)["passed"])

    def test_non_whitespace_differences_fail(self):
        for output in (EXPECTED.lower(), EXPECTED.replace(",", ";"),
                       EXPECTED.replace("C5920AD7", "C5920AD8"),
                       "46B2F8E0,8D71A93C,C5920AD7", "8D71A93C,46B2F8E0", ""):
            with self.subTest(output=output):
                self.assertFalse(self.run_answer([output])["passed"])

    def test_strict_preserves_literal_substring_matching(self):
        self.assertTrue(self.run_answer([f'Answer: "{EXPECTED}".'], ["--strict"])["passed"])
        self.assertFalse(self.run_answer([EXPECTED.replace(",", ", ")], ["--strict"])["passed"])

    def test_mechanics_still_requires_events(self):
        self.assertTrue(self.run_answer(["unrelated"], ["--mechanics-only"])["passed"])
        self.assertFalse(self.run_answer([], ["--mechanics-only"])["passed"])

    def test_fixture_consistency_stays_literal(self):
        result = self.run_answer([EXPECTED], frozen_expected=EXPECTED.replace(",", ", "))
        self.assertFalse(result["passed"])
        self.assertIn("manifest expected string differs", result["rows"][0]["error"])


    def test_alphanumeric_rule(self):
        codes = EXPECTED
        self.assertTrue(needle.answer_matches(codes, "Codes: 8d71a93c; 46B2F8E0, C5920AD7.", match="alphanumeric"))
        self.assertFalse(needle.answer_matches(codes, ", 2025-06-15 12:00:00+00:00", match="alphanumeric"))
        with self.assertRaises(ValueError):
            needle.answer_matches(codes, codes, match="other")

    def test_each_gate_declares_its_own_needles(self):
        for name, fmt, match, lengths in (
                ("mimo26", "context-files", "whitespace", [8192, 131072, 524288, 1048576]),
                ("qwen38", "prompt-set", "alphanumeric", [8192, 131072, 262144])):
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.path(name))}):
                declared = needle.spec()
            self.assertEqual((declared["format"], declared["match"], declared["lengths"]), (fmt, match, lengths))

    def test_prompt_sets_are_authenticated(self):
        rows = [{"name": "needle_8k", "token_ids": [5, 6, 7], "max_tokens": 1, "total_context": 4,
                 "expected": "A,B"}]
        blob = json.dumps({"needles": rows}).encode()
        with tempfile.TemporaryDirectory() as directory:
            path, frozen = Path(directory) / "prompts.json", Path(directory) / "frozen.json"
            path.write_bytes(blob)
            frozen.write_text(json.dumps({"full_file": {"sha256": digest(blob)}, "needles": [
                {"name": "needle_8k", "token_ids_sha256": digest(json.dumps([5, 6, 7]).encode())}]}))
            self.assertEqual(needle.prompt_set(path, frozen), {4: ([5, 6, 7], 1, "A,B")})
            path.write_bytes(blob + b" ")
            with self.assertRaisesRegex(ValueError, "differs"):
                needle.prompt_set(path, frozen)

    @unittest.skipUnless(os.environ.get("CHOPINFER_DATA") and (
        Path(os.environ["CHOPINFER_DATA"]) / "qwen38-refs/prompts/prompts.json").is_file(),
        "needs CHOPINFER_DATA with qwen38-refs/prompts/prompts.json")
    def test_qwen_prompt_set_matches_its_frozen_digests(self):
        data = os.environ.get("CHOPINFER_DATA")
        path = Path(data) / "qwen38-refs/prompts/prompts.json"
        rows = needle.prompt_set(path, common.ROOT / "fixtures/qwen38/prompts.json")
        self.assertEqual(sorted(rows), [8192, 131072, 262144])
        self.assertTrue(all(expected == "8D71A93C,46B2F8E0,C5920AD7" for _, _, expected in rows.values()))



def digest(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
