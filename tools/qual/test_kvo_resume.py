import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from kvo_resume import RESUME_COUNTERS, assess, bitwise_c1, package_drafter, resume_metric_deltas, serve_arguments, stop_child


class ResumeRunnerTest(unittest.TestCase):
    def test_arms_enable_cache_and_fixed_drafts_and_isolate_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mimo.args'
            path.write_bytes('\0'.join(['serve', '--topology', 'fixture-tp4', '--prefix-cache', 'off',
                                        '--admission', 'optimistic', '--debug-preempt-every', '4',
                                        '--max-preemptions', '5', '--speculation', 'adaptive', '--speculative-tokens', '7']).encode())
            for optimistic in (False, True):
                args = serve_arguments(path, 8439, optimistic)
                self.assertEqual(args[args.index('--prefix-cache') + 1], 'on')
                self.assertEqual(args[args.index('--speculation') + 1], 'fixed')
                self.assertEqual(args[args.index('--speculative-tokens') + 1], '7')
                self.assertEqual(args[args.index('--devices') + 1], '0,1,2,3')
                self.assertEqual('--debug-preempt-every' in args, optimistic)
                self.assertEqual('--max-preemptions' in args, optimistic)
                if optimistic:
                    self.assertEqual(args[args.index('--max-preemptions') + 1], '3')

    def test_autoregressive_arms_do_not_enable_speculation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'autoregressive.args'
            for depth in ([], ['--speculative-tokens', '0']):
                for policy in ([], ['--speculation', 'adaptive']):
                    path.write_bytes('\0'.join(['serve', '--topology', 'fixture-tp4'] + depth + policy).encode())
                    for optimistic in (False, True):
                        with self.subTest(depth=depth, policy=policy, optimistic=optimistic):
                            args = serve_arguments(path, 8439, optimistic)
                            self.assertNotIn('--speculation', args)
                            if depth:
                                self.assertEqual(args[args.index('--speculative-tokens') + 1], '0')
                            else:
                                self.assertNotIn('--speculative-tokens', args)

    def test_cleanup_only_terminates_the_owned_child_and_collects_it(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('server', 10), 0]
        stop_child(child)
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)
        child.reset_mock()
        child.poll.return_value = 0
        stop_child(child)
        child.terminate.assert_not_called()

    def test_assessment_rejects_missing_rows_and_unexercised_preemption(self):
        row = {'tokens': ['a'] * 96, 'logprobs': [-0.1] * 96,
               'top': [{'a': -0.1, 'b': -0.2}] * 96, 'finish_reason': 'length', 'usage': {'completion_tokens': 96}}
        records = {label: {'prompts_sha256': 'same', 'output_tokens': 96,
                          'preemptions': 36 if label.startswith('optimistic') else 0,
                          'resume_metrics': {'exact_resumes': 20 if label.startswith('optimistic') else 0, 'capture_failures': 0, 'restore_failures': 0},
                          'requests': [dict(row, name=str(i)) for i in range(36)]}
                   for label in ('reserved-c1', 'reserved-c16', 'repeat-c16', 'optimistic-c1', 'optimistic-c16')}
        report = assess(records)
        self.assertTrue(report['c1']['passed'])
        self.assertTrue(report['c1_bitwise']['identical'])
        for metric, value in [('exact_resumes', 0), ('capture_failures', 1), ('restore_failures', 1)]:
            invalid = copy.deepcopy(records)
            invalid['optimistic-c1']['resume_metrics'][metric] = value
            with self.assertRaises(ValueError):
                assess(invalid)
        invalid = copy.deepcopy(records)
        del invalid['optimistic-c16']['resume_metrics']
        with self.assertRaises(ValueError):
            assess(invalid)
        self.assertIn('requires comparison', report['c16_assessment'])
        invalid = copy.deepcopy(records)
        invalid['optimistic-c16']['preemptions'] = 0
        with self.assertRaises(ValueError):
            assess(invalid)
        invalid = copy.deepcopy(records)
        invalid['optimistic-c1']['requests'].pop()
        with self.assertRaises(ValueError):
            assess(invalid)

    def test_metric_deltas_refuse_missing_reset_and_invalid_counters(self):
        def scrape(value):
            return ''.join(f'{name} {value}\n' for name in RESUME_COUNTERS.values())
        self.assertEqual(resume_metric_deltas(scrape(2), scrape(5)), dict.fromkeys(RESUME_COUNTERS, 3))
        for before, after in [(scrape(0), ''), (scrape(2), scrape(1)), (scrape(0), scrape('NaN')), (scrape(0), scrape('0.5'))]:
            with self.assertRaises(ValueError):
                resume_metric_deltas(before, after)

    def test_bitwise_report_checks_logprobs_finish_and_usage(self):
        row = {'name': 'one', 'tokens': ['a'], 'logprobs': [0.0], 'top': [{'a': 0.0}], 'finish_reason': 'length', 'usage': {'completion_tokens': 1}}
        before = {'requests': [row]}
        for field, value in [('logprobs', [-0.0]), ('top', [{'a': -0.0}]), ('finish_reason', 'stop'), ('usage', {'completion_tokens': 2})]:
            after = copy.deepcopy(before)
            after['requests'][0][field] = value
            self.assertFalse(bitwise_c1(before, after)['identical'])

    def test_provenance_reads_the_declared_drafter_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {'speculation': {'bundle_drafter': True, 'drafter_component': 'draft', 'drafter_kind': 'dflash'}}
            (path / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(package_drafter(['serve', '--bundle', directory])['kind'], 'dflash')
            del manifest['speculation']['drafter_kind']
            (path / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'drafter_kind'):
                package_drafter(['serve', '--bundle', directory])
            manifest['speculation']['bundle_drafter'] = False
            (path / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(package_drafter(['serve', '--bundle', directory])['kind'], 'none')

    def test_removed_package_selection_is_refused_with_the_replacement(self):
        for arguments in (['serve', '--bundle', 'mimo.v26.flash.v2', '--install-root', '/absent'],
                          ['serve', '--bundle', 'mimo.v26.flash.v2']):
            with self.assertRaisesRegex(ValueError, 'serve --model CHECKPOINT'):
                package_drafter(arguments)

    def test_model_arguments_read_the_package_the_helper_prepares(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'cache/p-abc'
            package.mkdir(parents=True)
            (package / 'manifest.json').write_text(json.dumps(
                {'speculation': {'bundle_drafter': True, 'drafter_component': 'draft', 'drafter_kind': 'mtp'}}))
            calls = root / 'calls'
            helper = root / 'chopinfer-model-qwen38'
            record = {'result': 'prepared-direct', 'package': {
                'bundle': str(package), 'package_identity': 'abc', 'checkpoint_identity': 'c', 'metadata_identity': 'm',
                'profile': {'name': 'tp4-mtp', 'profile': {'speculation': {'mode': 'adaptive', 'tokens': 3}}}}}
            helper.write_text(f'#!/bin/sh\necho "$@" > {calls}\necho progress >&2\necho \'{json.dumps(record)}\'\n')
            helper.chmod(0o755)
            arguments = ['serve', '--model', str(root / 'ckpt'), '--model-cache-dir', str(root / 'cache'),
                         '--listen', '127.0.0.1:1']
            result = package_drafter(arguments, helper)
            self.assertEqual(result['kind'], 'mtp')
            self.assertEqual(result['profile_tokens'], 3)
            self.assertEqual(result['package']['package_identity'], 'abc')
            self.assertEqual(result['manifest'], str((package / 'manifest.json').resolve()))
            self.assertEqual(calls.read_text().split(),
                             ['prepare', '--direct', '--model', str(root / 'ckpt'), '--cache-dir', str(root / 'cache')])
            with self.assertRaisesRegex(ValueError, 'helper'):
                package_drafter(arguments)

    def test_topology_comes_from_the_model_profile_or_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qwen38.args'
            profiles = 'models/qwen38/profiles.json'
            accepted = [['--model', 'ckpt'], ['--model', 'ckpt', '--tensor-parallel-size', '4'],
                        ['--profile', f'{profiles}#tp4-mtp', '--bundle', 'p', '--checkpoint', 'c']]
            refused = [['--model', 'ckpt', '--tensor-parallel-size', '2'],
                       ['--profile', f'{profiles}#tp2-ar', '--bundle', 'p', '--checkpoint', 'c'],
                       ['--bundle', 'p']]
            for extra in accepted:
                path.write_bytes('\0'.join(['serve'] + extra).encode())
                serve_arguments(path, 8439, False)
            for extra in refused:
                path.write_bytes('\0'.join(['serve'] + extra).encode())
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    serve_arguments(path, 8439, False)

    def test_profile_depth_is_pinned_fixed_when_the_arguments_carry_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mimo.args'
            path.write_bytes('\0'.join(['serve', '--model', 'ckpt']).encode())
            args = serve_arguments(path, 8439, True, default_tokens=7)
            self.assertEqual(args[args.index('--speculation') + 1], 'fixed')
            self.assertNotIn('--speculation', serve_arguments(path, 8439, True))

if __name__ == '__main__':
    unittest.main()
