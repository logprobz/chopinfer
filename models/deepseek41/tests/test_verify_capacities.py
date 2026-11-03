"""Every verify check requires capacities its pair can form.

Since d27fe227bd a package with exact verify graphs reports each verify
step's live row count as its capacity, while AR decode keeps its power-of-two
buckets. A pair at concurrency C (a power of two) with draft budget K forms
at most C * (K + 1) verify rows and C AR decode rows. A larger required
capacity can never be observed."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/qual'))

import release_preview

# The named capacities the configurations cover: the tuned verify rows and
# the shapes balanced serves at C1, C2, C4 and C8.
CAPACITIES = [2, 4, 6, 8, 12, 16, 24, 32]


def values(words, flag):
    """The value after each `flag` in `words`."""
    return [words[index + 1] for index, word in enumerate(words) if word == flag]


def ruled(rules, path):
    """The value of the one `eq` rule on `path`."""
    found = [rule['value'] for rule in rules if rule['path'] == path and rule['op'] == 'eq']
    if len(found) != 1:
        raise AssertionError(f'expected one eq rule on {path}, found {len(found)}')
    return found[0]


class VerifyCapacityTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / 'models/deepseek41/qual.json').read_text())
        self.verify = sorted(name for name, check in self.policy['checks'].items()
                             if check['module'] == 'verify_exact')

    def runs(self):
        """(configuration, check, draft budget) for every verify check a
        configuration runs; `default` serves the preview's speculation."""
        runs = [('default', name, self.policy['preview']['speculation']['tokens'])
                for name in self.policy['cohorts']['full']['checks'] if name in self.verify]
        for name, config in self.policy['configurations'].items():
            runs += [(name, check, config['serving'].get('speculation_tokens')) for check in config['checks']
                     if check in self.verify]
        return runs

    def test_each_verify_check_runs_in_one_configuration(self):
        checks = [check for _, check, _ in self.runs()]
        self.assertEqual(sorted(checks), self.verify)

    def test_required_capacities_are_formable(self):
        for config, name, budget in self.runs():
            with self.subTest(config=config, check=name):
                check = self.policy['checks'][name]
                candidate, ar = check['input_rules']['verify_candidate'], check['input_rules']['verify_ar']
                recorded = check['constants']['release_preview']['RECORD_ARGUMENTS']['value']
                self.assertEqual(ruled(candidate, ['k']), budget)
                self.assertEqual(ruled(ar, ['k']), 0)
                concurrency = ruled(check['rules'], ['records', '*', 'concurrency'])
                self.assertEqual(ruled(candidate, ['concurrency']), concurrency)
                self.assertEqual(ruled(ar, ['concurrency']), concurrency)
                # AR decode reports its power-of-two bucket, so only there is it C rows.
                self.assertEqual(concurrency & (concurrency - 1), 0)
                rows = concurrency * (budget + 1)
                self.assertEqual(ruled(check['rules'], ['records', '*', 'row_capacity_coverage', 'required']),
                                 [rows])
                self.assertEqual(ruled(check['rules'], ['records', '*', 'ar_row_capacity_coverage', 'required']),
                                 [concurrency])
                self.assertEqual(values(check['args'], '--row-capacity'), [str(rows)])
                self.assertEqual(values(check['args'], '--ar-row-capacity'), [str(concurrency)])
                self.assertIn('--require-row-shapes', check['args'])
                # The records the release preview runs for this check.
                for key, capacity in (('verify_candidate', rows), ('verify_ar', concurrency)):
                    _, _, words = release_preview.record_for(key, check)
                    self.assertEqual(values(recorded[key], '--concurrency'), [str(concurrency)])
                    self.assertEqual(values(words, '--concurrency'), [str(concurrency)])
                    self.assertEqual(values(words, '--row-capacity'), [str(capacity)])
                    self.assertIn('--require-row-shapes', words)
                if config != 'default':
                    serve = self.policy['configurations'][config]['serve']
                    self.assertEqual(values(serve, '--speculative-tokens'), [str(budget)])
                    self.assertEqual(values(serve, '--speculation'), ['fixed'])
                    self.assertEqual(config, f'verify-b{rows}')

    def test_configurations_cover_the_named_capacities(self):
        rows = [ruled(self.policy['checks'][name]['rules'], ['records', '*', 'row_capacity_coverage', 'required'])
                for _, name, _ in self.runs()]
        self.assertEqual(sorted(capacity for [capacity] in rows), CAPACITIES)


if __name__ == '__main__':
    unittest.main()
