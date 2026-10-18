"""Direct-gate evidence requirements for the DSpark declaration."""
import copy
import json
from pathlib import Path
import unittest

import kinds
import spec_verify
from test_spec_verify import bound, copy_records


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'models/deepseek41/qual.json'


class DsparkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.declaration = json.loads(PATH.read_text())
        self.check = self.declaration['checks']['spec_verify']
        self.model = self.declaration['qualification']['checkpoint']['served_model_name']

    def captures(self, k=5, drafter='dspark', positions=64):
        tokens = ['token'] * positions
        spec, ar = copy_records([(tokens, tokens)] * 16)
        records = []
        for record, depth in [(spec, k), (ar, 0)]:
            record = bound(record, identity=self.declaration['qualification']['checkpoint']['checkpoint_identity'],
                           model=self.model, tokens=depth, tensor_parallel_size=4, kv='mixed', package='c' * 64)
            record['passed'] = True
            record['sampling'] = {'temperature': 0, 'top_p': 1, 'logprobs': 5}
            record['sets']['natural'] = copy.deepcopy(record['sets']['copy'])
            if depth:
                record['serving']['speculation']['drafter'] = drafter
            records.append(record)
        return records

    def permits(self, value, rules):
        rules = copy.deepcopy(rules)
        for rule in rules:
            if rule['value'] == '{model}':
                rule['value'] = self.model
        try:
            return all(kinds.predicate(value, rule) for rule in rules)
        except (KeyError, TypeError, ValueError):
            return False

    def generic_score(self, spec, ar):
        with kinds.policy(PATH, self.check):
            report = spec_verify.compare_records(spec, ar)
            self.assertTrue(report['passed'])
            scored = kinds.SpecVerify().score(report, check=self.check)
            self.assertTrue(scored['passed'])
        return scored

    def test_only_the_declared_drafter_and_full_budget_supply_candidate_evidence(self):
        for k, drafter in [(5, 'dspark'), (1, 'dspark'), (4, 'dspark'), (5, 'mtp')]:
            with self.subTest(k=k, drafter=drafter):
                spec, ar = self.captures(k, drafter)
                scored = self.generic_score(spec, ar)
                self.assertTrue(self.permits(scored, self.check['rules']))
                self.assertTrue(self.permits(ar, self.check['input_rules']['spec_ar']))
                self.assertEqual(self.permits(spec, self.check['input_rules']['spec_candidate']),
                                 k == 5 and drafter == 'dspark')
        for field in ['drafter', 'tokens']:
            spec, _ = self.captures()
            del spec['serving']['speculation'][field]
            self.assertFalse(self.permits(spec, self.check['input_rules']['spec_candidate']))


if __name__ == '__main__':
    unittest.main()
