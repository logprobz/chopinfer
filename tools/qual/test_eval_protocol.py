"""The scorer bridge refuses incompatible or malformed results."""

import copy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import kinds


class ProtocolTests(unittest.TestCase):
    def evaluate(self, response, op='rescore'):
        done = subprocess.CompletedProcess([], 0, json.dumps(response), '')
        with patch.object(kinds, 'eval_binary', return_value=Path('scorer')):
            return kinds.evaluated([{'op': op}], run=lambda *args, **kwargs: done)

    def test_invalid_responses_are_refused(self):
        result = {'status': 'pass', 'metrics': {'passed': True}, 'predicates': [], 'recorded_fields': []}
        valid = {'schema': 'chopinfer-eval-rescore-result/v1', 'results': [result]}
        self.assertEqual(self.evaluate(valid), [result])
        invalid = [None, [], {}, {**valid, 'schema': 'chopinfer-eval-rescore-result/v0'},
                   {**valid, 'results': []}, {**valid, 'results': [result, result]},
                   {**valid, 'results': [None]}]
        for key, value in [('status', 'unknown'), ('metrics', []), ('metrics', {'score': float('nan')}),
                           ('metrics', {'score': float('inf')}), ('predicates', {}),
                           ('predicates', [{'passed': 'true'}]), ('recorded_fields', [1]), ('error', {})]:
            invalid.append({**valid, 'results': [{**result, key: value}]})
        for key in result:
            missing = copy.deepcopy(valid)
            del missing['results'][0][key]
            invalid.append(missing)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.evaluate(response)

    def test_operation_fields_are_required(self):
        for op, result in [('verify', {'recorded_fields': []}), ('score', {'metrics': {}}),
                           ('judge', {'status': 'fail', 'predicates': []})]:
            valid = {'schema': 'chopinfer-eval-rescore-result/v1', 'results': [result]}
            self.assertEqual(self.evaluate(valid, op), [result])
            for invalid in ({}, {'error': {'type': 'unknown', 'message': 'error'}},
                            {'error': {'type': 'invalid', 'message': 1}}):
                with self.subTest(op=op, invalid=invalid), self.assertRaises(RuntimeError):
                    self.evaluate({**valid, 'results': [invalid]}, op)

    def test_missing_crashed_and_malformed_scorers_fail_closed(self):
        for error in (FileNotFoundError('missing'), OSError('cannot execute')):
            with patch.object(kinds, 'eval_binary', return_value=Path('scorer')), self.assertRaises(OSError):
                kinds.evaluated([{'op': 'score'}], run=lambda *a, **kw: self.raise_error(error))
        for code, stdout in [(1, ''), (-9, ''), (0, 'bad json')]:
            done = subprocess.CompletedProcess([], code, stdout, 'failed')
            with patch.object(kinds, 'eval_binary', return_value=Path('scorer')), self.assertRaises(RuntimeError):
                kinds.evaluated([{'op': 'score'}], run=lambda *a, **kw: done)

    @staticmethod
    def raise_error(error):
        raise error

class StandaloneComparisonTests(unittest.TestCase):
    def test_greedy_finish_difference_uses_the_eval_verdict(self):
        import argparse
        import contextlib
        import io
        import tempfile
        import greedy_equality
        root = Path(__file__).resolve().parents[2]
        vector = json.loads((root / 'tests/qual-vectors/greedy_pin.json').read_text())['vectors'][0]
        left, right = copy.deepcopy(vector['reference']), copy.deepcopy(vector['receipt'])
        name = next(iter(right['finish_reasons']))
        right['finish_reasons'][name] = 'length' if left['finish_reasons'][name] == 'stop' else 'stop'
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ('left.json', 'right.json')]
            for path, value in zip(paths, (left, right)):
                path.write_text(json.dumps(value))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as exit_:
                greedy_equality.compare(argparse.Namespace(ar=str(paths[0]), other=str(paths[1])))
            self.assertEqual(exit_.exception.code, 1)
            result = json.loads(stdout.getvalue())
            self.assertFalse(result['equal'])
            self.assertTrue(result['finish_differences'])

class ProbeCaptureTests(unittest.TestCase):
    def test_image_probe_judgments_require_the_scorer(self):
        import multimodal
        reference = {'answer': {'tokens': [[97]]},
                     'steps': [[1, -0.1, [[1, -0.1, [97]], [2, -0.2, [98]]]]]}
        observed = {'finish_reason': 'length', 'logprobs': {'content': [{'token': 'b'}]}}
        forced_reference = {'forced': [[2, -0.2, 2]], 'top': [[[2, -0.2], [3, -1.0]]],
                            'scored': 1, 'limits': {'mean_abs_dlogprob': 0.15, 'top1_decisive': 0.08,
                                                   'top1_near_tie': 0.08, 'top1_disagreement': 0.1}}
        forced = {'prompt_token_ids': [1, 2],
                  'prompt_logprobs': [None, {'2': {'logprob': -0.2}, '3': {'logprob': -1.0}}]}
        with patch.object(kinds, 'call', side_effect=RuntimeError('scorer unavailable')) as bridge:
            with self.assertRaisesRegex(RuntimeError, 'scorer unavailable'):
                multimodal.reference_greedy_agreement(reference, observed, 0.3)
            with self.assertRaisesRegex(RuntimeError, 'scorer unavailable'):
                multimodal.probe_forced_agreement({'token_ids': [1]}, forced_reference, forced)
            self.assertEqual(bridge.call_count, 2)

class GenericApplicabilityTests(unittest.TestCase):
    def test_unserved_grounding_retains_identity_validation(self):
        import common
        pin = 'a' * 64
        server = {'health': {'modalities': ['text'], 'serving': {
            'checkpoint_identity': pin, 'startup_nonce': 'b' * 32}}}
        reason = common.not_applicable(server, 'image')
        value = {'schema': 'chopinfer-qualification/v1', 'server': server,
                 'rows': [], 'passed': False, 'not_applicable': [reason]}
        check = {'module': 'grounding', 'rules': [
            {'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'docs/check-kinds.md'}]}
        with patch.object(common, 'checkpoint_pin', return_value={'checkpoint_identity': pin}):
            self.assertEqual(kinds.rescore(kinds.RECEIPT, value, None, check)['status'], 'not_applicable')
            value['not_applicable'] = ['invented reason']
            self.assertEqual(kinds.rescore(kinds.RECEIPT, value, None, check)['status'], 'insufficient')
