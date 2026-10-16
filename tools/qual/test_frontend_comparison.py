"""CPU checks for the frontend capture comparison."""

import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

# The comparison needs a gate config that declares a frontend_goldens check.
from qual_test_config import capture_fixture, setUpModule  # noqa: F401

import frontend_goldens as compare

from frontend_goldens import REFUSED_MEDIA, identical_ids

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "capture comparison against the pinned gate's frontend_goldens check"


def refused():
    return {"status": 400, "error": {"error": {"message": REFUSED_MEDIA}}}


class IdenticalIdsTest(unittest.TestCase):
    def test_media_refusals_gate(self):
        theirs = [{"token_ids": [1]}, {"token_ids": [2, 3]}]
        ours = [{"token_ids": [1]}, refused()]
        strict = identical_ids(ours, theirs, "chat-renders.json")
        self.assertEqual((strict["mismatches"], strict["media_refused"], strict["agrees"]), ([1], [1], False))

    def test_other_mismatches_gate(self):
        theirs = [{"token_ids": [1]}, {"token_ids": [2]}]
        ours = [{"token_ids": [9]}, refused()]
        report = identical_ids(ours, theirs, "chat-renders.json")
        self.assertEqual((report["mismatches"], report["media_refused"], report["agrees"]), ([0, 1], [1], False))


class DiagnosticCoverageTests(unittest.TestCase):
    def test_media_refusal_is_a_failed_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ours, theirs = root / 'ours', root / 'reference'
            ours.mkdir()
            theirs.mkdir()
            (ours / 'chat-renders.json').write_text(json.dumps([refused()]))
            (theirs / 'chat-renders.json').write_text(json.dumps([{'token_ids': [1]}]))
            with capture_fixture(theirs) as reference, \
                 patch('sys.argv', ['compare', str(ours), '--reference', str(reference), '--required-file', 'chat-renders.json']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(), 1)

    def test_partial_frontend_scope_and_explicit_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'chat-renders.json').write_text(json.dumps([{'token_ids': [1, 2]} for _ in range(75)]))
            for required, expected in [('chat-renders.json', 0), ('tokenizer-golden.json', 3)]:
                with capture_fixture(root) as reference, \
                     patch('sys.argv', ['compare', temporary, '--reference', str(reference),
                                        '--required-file', required, '--json', str(root / 'report.json')]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(compare.main(), expected)
                value = json.loads((root / 'report.json').read_text())
                self.assertEqual(value['coverage']['gated'], 75 if expected == 0 else 0)
                self.assertEqual(value['coverage']['missing'], [] if expected == 0 else [required])
                self.assertNotIn('skipped', value)

    def test_empty_capture_is_insufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.json'
            with patch('sys.argv', ['compare', temporary, '--reference', temporary,
                                    '--json', str(report)]), contextlib.redirect_stdout(io.StringIO()):
                status = compare.main()
            self.assertEqual(status, 3)
            value = json.loads(report.read_text())
            self.assertEqual(value['status'], 'insufficient')
            self.assertEqual(value['coverage']['gated'], 0)


if __name__ == "__main__":
    unittest.main()
