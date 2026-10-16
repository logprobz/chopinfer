"""Both media formats rescore from a complete source-revision fixture pack."""

import unittest
from unittest.mock import patch

import common
import declaration
import kinds
import vectors


class MediaRevisionTests(unittest.TestCase):
    def test_both_models_rescore_at_head(self):
        kind = kinds.of('multimodal')
        check = {'rules': [{'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'docs/check-kinds.md'}]}
        for model in ('mimo26', 'qwen38'):
            with self.subTest(model=model), kinds.policy(declaration.path(model), check), \
                 patch.object(common, 'pinned_binding', return_value=None):
                # Runner workload binding has its own tests; this test needs no external pixels.
                config = declaration.load(declaration.path(model))
                current = kind.reference(check, config)
                reference = kind.reference(check, config, revision='HEAD')
                self.assertEqual(reference, current)
                rows = []
                if reference.get('format') == 'image-probes':
                    for name in reference['order']:
                        item = reference['probes'][name]
                        key = 'greedy' if item['kind'] == 'short' else 'teacher_forced'
                        scores = {'passed': True, 'exact': True} if key == 'greedy' else {
                            'passed': True, 'scored': item['scored'], 'mean_abs_dlogprob': 0.0,
                            'top1_disagreement': 0.0, 'near_tie_report': {'available': True, 'flips': []}}
                        rows.append({'id': name, key: scores, 'fixture_expectation_match': True,
                                     'token_count_match': True, 'passed': True})
                    receipt = vectors.receipt('multimodal', reference={'sha256': reference['sha256']},
                                              rows=rows, passed=True)
                else:
                    for item in reference['cases']:
                        gate = item['exact'][reference['gate']]
                        scores = {'passed': True, 'scored': len(item['answer_token_ids']),
                                  'mean_abs_dlogprob': 0.0, 'max_abs_dlogprob': 0.0,
                                  'top1_disagreement': 0.0, 'top1_flips': [],
                                  **vectors.served_rows(item['answer_token_ids'], gate['logprob'], gate['top1'])}
                        rows.append({'id': item['id'], 'kind': item['kind'], 'logprobs': scores,
                                     'greedy': {'exact': True, 'passed': True},
                                     'fixture_expectation_match': True, 'token_count_match': True, 'passed': True})
                    receipt = vectors.receipt('multimodal', exact_reference_sha256=reference['sha256'],
                                              rows=rows, passed=True)
                result = kinds.rescore(kind, receipt, reference, check)
                self.assertEqual(result['status'], 'pass', result.get('error'))
