"""The long-context declaration keeps the explicit F32-only floor policy."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/qual'))

import kinds
import teacher
import test_teacher


class LongContextQualificationTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / 'models/deepseek41/qual.json').read_text())
        self.check = self.policy['checks']['long_context']

    def score(self, reference, shift=0):
        capture = copy.deepcopy(teacher.variant_scores(reference['texts'], 'exact'))
        for row in capture:
            row['logprob'] = [value - shift for value in row['logprob']]
        return kinds.call('score_capture', kinds.KINDS['teacher_forced'], capture,
                          reference, self.check)['metrics']

    def test_full_suite_uses_the_floor_only_reference(self):
        self.assertIn('long_context', self.policy['cohorts']['full']['checks'])
        self.assertIn('long_context', self.policy['change_classes']['quality'])
        self.assertEqual(self.policy['references']['long_context']['variants'], ['exact'])
        self.assertNotIn('long_context', self.policy['configurations']['kv-bf16']['checks'])
        reference = test_teacher.FloorOnlyReferenceTests.fixture()
        metrics = self.score(reference)
        self.assertTrue(metrics['passed'])
        self.assertEqual(metrics['limits'], {'mean_abs_dlogprob': 0.10, 'top1_decisive': 0.08,
                                            'top1_near_tie': 0.08, 'top1_disagreement': 0.10})
        self.assertFalse(self.score(reference, shift=0.125)['passed'])

    def test_missing_opt_in_or_incomplete_reference_is_refused(self):
        with self.assertRaises(ValueError):
            self.score(test_teacher.FloorOnlyReferenceTests.fixture(999))
        del self.check['constants']['teacher']['ALLOW_FLOOR_ONLY']
        with self.assertRaises(ValueError):
            self.score(test_teacher.FloorOnlyReferenceTests.fixture())


if __name__ == '__main__':
    unittest.main()
