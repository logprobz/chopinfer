"""CPU coverage for declared package concurrency limits."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from concurrency import add_arguments, validate
import spec_bench


class ConcurrencyTests(unittest.TestCase):
    def parse(self, words, defaults=(1, 16, 32)):
        parser = argparse.ArgumentParser()
        add_arguments(parser, defaults)
        args = parser.parse_args(words)
        validate(parser, args)
        return args

    def test_defaults_and_declared_capacity(self):
        self.assertEqual(self.parse([]).concurrencies, [1, 16, 32])
        self.assertEqual(self.parse([], spec_bench.CONCURRENCIES).concurrencies, [1, 4, 16])
        self.assertEqual(self.parse(['--max-sequences', '256', '--concurrencies', '32', '64', '128', '256'])
                         .concurrencies, [32, 64, 128, 256])
        for words in [['--concurrencies', '64'], ['--max-sequences', '0'],
                      ['--concurrencies', '0'], ['--concurrencies', '16', '1'],
                      ['--concurrencies', '1', '1'],
                      ['--max-sequences', '256', '--concurrencies', '257']]:
            with self.subTest(words=words), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(words)

    def test_large_spec_cohort_reuses_authenticated_examples(self):
        import hashlib
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            raw = json.dumps([{'question': 'one'}, {'question': 'two'}]).encode()
            (fixture / 'examples.json').write_bytes(raw)
            (fixture / 'manifest.json').write_text(json.dumps({
                'expected_count': 2, 'examples_sha256': hashlib.sha256(raw).hexdigest()}))
            with patch.object(spec_bench, 'FIXTURE', fixture):
                _, questions = spec_bench.gsm8k_questions(256)
                self.assertEqual(questions, ['one', 'two'] * 128)
                (fixture / 'examples.json').write_bytes(b'[]')
                with self.assertRaises(ValueError):
                    spec_bench.gsm8k_questions(256)

    def test_redeploy_dry_run_passes_an_explicit_sequence_limit(self):
        # The profile supplies the limit unless one is given; the package
        # declares none, so --package-max-sequences is refused.
        root = Path(__file__).resolve().parents[2]
        command = ['bash', str(root / 'tools/dev/qwen38-redeploy.sh'), '--dry-run',
                   '--checkpoint', '/data/checkpoint', '--work-dir', '/data/dev-test',
                   '--logs-dir', '/data/logs-test']
        env = dict(os.environ, CHOPINFER_LOCAL_ENV='/data/nonexistent-conc-sched-env')
        for words, status in [([], 0), (['--max-sequences', '256'], 0), (['--max-sequences', '0'], 2),
                              (['--max-sequences', '256', '--prefix-cache', 'on',
                                '--release-timeout', '42'], 0),
                              (['--package-max-sequences', '256'], 2)]:
            with self.subTest(words=words):
                result = subprocess.run(command + words, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, status, result.stderr)
                if status == 0:
                    serve = [line for line in result.stdout.splitlines() if ' serve --model ' in line]
                    self.assertEqual(len(serve), 1, result.stdout)
                    if words:
                        self.assertIn('--max-sequences 256', serve[0])
                    else:
                        self.assertNotIn('--max-sequences', serve[0])
                if '--prefix-cache' in words:
                    self.assertIn('--prefix-cache on', result.stdout)
                    self.assertIn('42s for PID exit and GPU release', result.stdout)
                if '--package-max-sequences' in words:
                    self.assertIn('pass --max-sequences N', result.stderr)
