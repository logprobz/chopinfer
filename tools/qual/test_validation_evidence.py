"""Applicability requires observed identity and complete coverage, not a similar run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validation_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def test_reader_bounds_and_json_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.json'
            path.write_text('{"row": "' + 'x' * (21 << 20) + '"}')
            with self.assertRaisesRegex(ValueError, 'bounded'):
                evidence.read_json(path)
            self.assertEqual(len(evidence.read_json(path, max_bytes=64 << 20)['row']), 21 << 20)
            for value in ('{"row":1,"row":2}', '{"row":NaN}', '{"row":1e309}'):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    evidence.read_json(path)

    def complete(self, kind='model_quality', scope=None):
        obligation = {'id': 'model:claim', 'kind': kind}
        if scope:
            obligation['scope'] = scope
        dimensions = evidence.KERNEL if scope == 'kernel_operation' else evidence.REQUIRED[kind]
        record = evidence.Record(kind, obligation['id'], 'passed', coverage={'comparisons': 8, 'complete': True})
        expected = {'candidate': {}, 'baseline': {}, 'coverage': {'comparisons': 8}}
        for arm in ('candidate', 'baseline') if kind == 'fixed_batch_exactness' else ('candidate',):
            for name in dimensions:
                value = True if name == 'exclusive_device' else {'sha256': 'a' * 64}
                record.bind(name, value, '/' + arm + '/' + name, arm)
                expected[arm][name] = value
        return obligation, record, expected

    def test_complete_synthetic_observations_cover_only_their_claim(self):
        obligation, record, context = self.complete()
        status = evidence.assess(record, obligation, context)
        self.assertEqual((status['execution'], status['result'], status['applicability']), ('executed', 'passed', 'current'))
        self.assertTrue(status['satisfied'])
        self.assertIn('candidate.checkpoint', status['observed_fields'])
        obligation['id'] = 'another:model'
        self.assertFalse(evidence.assess(record, obligation, context)['satisfied'])

    def test_changed_required_dimensions_are_stale_without_erasing_historical_pass(self):
        for dimension in evidence.MODEL | {'policy'}:
            with self.subTest(dimension=dimension):
                obligation, record, context = self.complete()
                context['candidate'][dimension] = {'sha256': 'b' * 64}
                result = evidence.assess(record, obligation, context)
                self.assertEqual(result['result'], 'passed')
                self.assertEqual(result['applicability'], 'stale')
                self.assertFalse(result['satisfied'])

    def test_new_checkpoint_cannot_inherit_accuracy_from_compatible_execution(self):
        obligation, record, context = self.complete()
        context['candidate']['checkpoint'] = {'sha256': 'b' * 64}
        self.assertEqual(record.observed['candidate']['native_module']['value'], context['candidate']['native_module'])
        self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'stale')

    def test_missing_expected_or_observed_binding_remains_missing(self):
        for arm in ['observed', 'expected']:
            obligation, record, context = self.complete()
            if arm == 'observed':
                record.observed['candidate'].pop('host')
            else:
                context['candidate'].pop('host')
            status = evidence.assess(record, obligation, context)
            self.assertEqual(status['applicability'], 'missing')
            self.assertFalse(status['satisfied'])

    def test_fixed_batch_requires_separate_baseline_and_candidate(self):
        obligation, record, context = self.complete('fixed_batch_exactness')
        self.assertTrue(evidence.assess(record, obligation, context)['satisfied'])
        context['baseline']['workload'] = {'sha256': 'b' * 64}
        self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'stale')
        record.observed['baseline'].clear()
        self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'missing')

    def test_checkpoint_independent_kernel_claim_does_not_prove_model_quality(self):
        obligation, record, context = self.complete('gpu_correctness', 'kernel_operation')
        self.assertNotIn('checkpoint', context['candidate'])
        self.assertTrue(evidence.assess(record, obligation, context)['satisfied'])
        obligation['kind'] = 'model_quality'
        obligation.pop('scope')
        self.assertFalse(evidence.assess(record, obligation, context)['satisfied'])

    def test_cpu_fake_host_or_fixed_batch_cannot_satisfy_device_or_performance_claim(self):
        for offered in ['cpu_contract', 'fake_host', 'fixed_batch_exactness']:
            obligation, record, context = self.complete('gpu_correctness')
            record.kind = offered
            result = evidence.assess(record, obligation, context)
            self.assertEqual(result['applicability'], 'missing')
            self.assertTrue(any('kind mismatch' in why for why in result['reasons']))
        obligation, record, context = self.complete('service_performance')
        record.kind = 'gpu_correctness'
        self.assertFalse(evidence.assess(record, obligation, context)['satisfied'])

    def test_current_failed_execution_is_distinct_from_missing_or_stale(self):
        obligation, record, context = self.complete()
        record.result = 'failed'
        status = evidence.assess(record, obligation, context)
        self.assertEqual(status['applicability'], 'current')
        self.assertEqual(status['result'], 'failed')
        self.assertFalse(status['satisfied'])

    def test_zero_partial_skipped_truncated_and_failed_subprocesses_cannot_pass(self):
        for coverage in [{'comparisons': 0, 'complete': True}, {'comparisons': True, 'complete': True},
                         {'comparisons': 8, 'complete': False},
                         *[{'comparisons': 8, 'complete': True, key: True} for key in
                           ('skipped', 'partial_goldens', 'truncated', 'subprocess_failed')]]:
            obligation, record, context = self.complete('frontend_cargo_goldens')
            record.coverage = coverage
            self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'missing')
        for execution, result in [('not_run', None), ('executed', 'skipped')]:
            obligation, record, context = self.complete()
            record.execution, record.result = execution, result
            self.assertFalse(evidence.assess(record, obligation, context)['satisfied'])

    def test_nonzero_partial_coverage_does_not_cover_a_larger_required_workload(self):
        obligation, record, context = self.complete('frontend_cargo_goldens')
        context['coverage'] = {'comparisons': 10, 'golden_cases': ['tokenizer', 'chat', 'decode', 'streaming']}
        status = evidence.assess(record, obligation, context)
        self.assertEqual(status['applicability'], 'missing')
        self.assertTrue(any('required workload' in why for why in status['reasons']))

    def test_shared_device_timing_never_becomes_performance_evidence(self):
        obligation, record, context = self.complete('service_performance')
        record.bind('exclusive_device', False, '/measurement_label')
        context['candidate']['exclusive_device'] = False
        self.assertFalse(evidence.assess(record, obligation, context)['satisfied'])

    def test_feature_scope_and_malformed_hashes_cannot_pass(self):
        obligation, record, context = self.complete('cpu_contract')
        record.bind('test_scope', {'features': ['default']}, '/features')
        context['candidate']['test_scope'] = {'features': ['cuda-fake-driver-tests']}
        self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'stale')
        obligation, record, context = self.complete()
        context['candidate']['host'] = {'sha256': 'not-a-digest'}
        self.assertEqual(evidence.assess(record, obligation, context)['applicability'], 'stale')

    def attached(self, kind, profiles, *, scope=None, result='passed', stale=False):
        obligation, record, context = self.complete(kind, scope)
        obligation.update(profiles=[row['profile'] for row in profiles if row['model'] == 'model'],
                          evidence={'selection': 'selected'}, targets=['named comparisons'],
                          observation={'features': ['default'], 'platform': 'test platform'})
        source = {'revision': 'current', 'change_identity_sha256': 'a' * 64}
        record.bind('source', source, '/source')
        if kind in ('cpu_contract', 'fake_host'):
            record.bind('test_scope', {'targets': obligation['targets'], **obligation['observation']}, '/test_scope')
        if stale:
            record.bind('source', {'revision': 'previous'}, '/source')
        record.result = result
        # Caller assertions about profile coverage must not bypass the guard.
        context['profiles'] = obligation['profiles']
        record.coverage['profiles'] = obligation['profiles']
        report = {'obligations': [obligation], 'profiles': profiles, 'gaps': [],
                  'source_identity': {'revision': 'current'}, 'change_identity_sha256': 'a' * 64}
        with patch.object(evidence, 'load', return_value=[record]):
            evidence.attach(report, context, ['synthetic record'])
        return obligation['evidence']

    def test_multi_current_profile_deployment_claims_require_aggregation(self):
        profiles = [{'model': 'model', 'profile': name, 'removed': False} for name in ('tp1', 'tp4')]
        for kind in ('model_quality', 'service_performance', 'fixed_batch_exactness', 'gpu_correctness'):
            with self.subTest(kind=kind):
                status = self.attached(kind, profiles)
                self.assertEqual((status['execution'], status['result'], status['applicability']),
                                 ('executed', 'passed', 'missing'))
                self.assertFalse(status['satisfied'])
                self.assertTrue(any('per-profile' in why for why in status['reasons']))

    def test_single_current_profile_and_independent_claims_stay_applicable(self):
        profiles = [{'model': 'model', 'profile': 'tp4', 'removed': False},
                    {'model': 'model', 'profile': 'removed-tp1', 'removed': True},
                    {'model': 'another', 'profile': 'tp8', 'removed': False}]
        for kind in ('model_quality', 'service_performance', 'fixed_batch_exactness', 'gpu_correctness'):
            with self.subTest(kind=kind):
                self.assertTrue(self.attached(kind, profiles)['satisfied'])
        profiles.append({'model': 'model', 'profile': 'tp8', 'removed': False})
        for kind, scope in [('cpu_contract', None), ('fake_host', None), ('frontend_cargo_goldens', None),
                            ('artifact_identity', None), ('gpu_correctness', 'kernel_operation')]:
            with self.subTest(kind=kind, scope=scope):
                self.assertTrue(self.attached(kind, profiles, scope=scope)['satisfied'])

    def test_profile_guard_preserves_failed_skipped_and_stale_diagnostics(self):
        profiles = [{'model': 'model', 'profile': name, 'removed': False} for name in ('tp1', 'tp4')]
        for result in ('failed', 'skipped', 'passed'):
            for stale in (True, False):
                with self.subTest(result=result, stale=stale):
                    status = self.attached('model_quality', profiles, result=result, stale=stale)
                    self.assertEqual(status['result'], result)
                    self.assertFalse(status['satisfied'])
                    self.assertEqual(status['applicability'], 'stale' if stale else 'missing')
                    self.assertTrue(any('per-profile' in why for why in status['reasons']))
                    if stale:
                        self.assertIn('candidate: changed source', status['reasons'])

    def release_fixture(self, root, *, bad_hash=False, wrapper=False, variant=None):
        raw = root / 'raw.json'; raw.write_text(json.dumps({'score': 1.0}))
        rules = [{'path': ['score'], 'op': 'le', 'value': 1.0}]
        config = {'checks': [{'id': 'check', 'module': 'teacher', 'rules': rules}]}
        if variant:
            config['variant'] = variant
        value = {'schema': 'chopinfer-release-receipt/v1', 'model': 'model', 'config': config,
                 'config_sha256': hashlib.sha256((json.dumps(config, indent=2) + '\n').encode()).hexdigest(),
                 'checks': [{'id': 'check', 'passed': True, 'receipt': str(raw),
                             'sha256': 'b' * 64 if bad_hash else hashlib.sha256(raw.read_bytes()).hexdigest()}],
                 'not_run': ['later']}
        if wrapper:
            value['observed'] = self.complete()[1].observed
            value['coverage'] = {'comparisons': 100, 'complete': True}
        path = root / 'summary.json'; path.write_text(json.dumps(value))
        return path, rules

    def test_release_adapter_rechecks_hashes_predicates_and_preserves_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, rules = self.release_fixture(Path(temporary))
            records = evidence.load(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1].execution, 'not_run')
            obligation = {'id': 'model:gate:check', 'kind': 'model_quality', 'gate': {'rules': rules}}
            status = evidence.assess(records[0], obligation, {'candidate': {'policy': records[0].observed['candidate']['policy']['value']}})
            self.assertEqual(status['result'], 'passed')
            self.assertEqual(status['predicate_result'], 'passed')
            self.assertEqual(status['applicability'], 'missing')
            obligation['gate']['rules'][0]['value'] = 0.5
            self.assertEqual(evidence.assess(records[0], obligation, {})['predicate_result'], 'failed')
            path, _ = self.release_fixture(Path(temporary), bad_hash=True)
            self.assertEqual(evidence.assess(evidence.load(path)[0], obligation, {})['applicability'], 'stale')

    def test_a_variant_run_claims_its_own_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.release_fixture(Path(temporary), variant='export')
            self.assertEqual([record.claim for record in evidence.load(path)],
                             ['model.export:gate:check', 'model.export:gate:later'])

    def test_asserted_wrapper_cannot_turn_legacy_evidence_into_observed_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.release_fixture(Path(temporary), wrapper=True)
            record = evidence.load(path)[0]
            self.assertNotIn('checkpoint', record.observed['candidate'])
            self.assertNotIn('host', record.observed['candidate'])
            self.assertTrue(record.issues)

    def test_existing_fixed_batch_receipt_is_not_a_complete_new_identity(self):
        value = {'passed': True, 'steps': 13, 'rows': 32, 'bytes_recorded_or_compared': 128,
                 'reference': 'baseline', 'identity': {'format': 'fixed-batch-packages-v4',
                 'source': {'binary_sha256': 'a' * 64}, 'topology': 'tp4', 'trace': {'steps': []}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'; path.write_text(json.dumps(value))
            record = evidence.load(path)[0]
        self.assertEqual(record.kind, 'fixed_batch_exactness')
        self.assertEqual(record.result, 'passed')
        self.assertFalse(evidence.assess(record, {'id': 'model:fixed_batch', 'kind': record.kind}, {})['satisfied'])

    def test_kernel_adapter_retains_available_identity_and_requires_coverage(self):
        value = {'correct': True, 'measurement_label': 'exclusive-device',
                 'cubin_sha256': 'a' * 64, 'source_closure_sha256': 'b' * 64}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'kernel.json'; path.write_text(json.dumps(value))
            record = evidence.load(path)[0]
        self.assertEqual(record.result, 'passed')
        self.assertIn('kernel', record.observed['candidate'])
        self.assertFalse(record.coverage['complete'])

    def test_diagnostic_field_paths_preserve_unknown_checkpoint_and_observations(self):
        value = {'format': 'chopinfer.diagnostic.v1', 'mode': 'fixed-schedule', 'status': 'match',
                 'completion': 'full', 'completed_steps': 13, 'rows': 32, 'artifacts': {},
                 'description': {'expected_steps': 13, 'source_checkpoint': None,
                   'invariants': {'host': {'binary_sha256': 'a' * 64}, 'trace': 'b' * 64, 'topology': 'tp4'},
                   'context': {'observed_graph_execution': None, 'configured_graph_switches': {'graph': True}}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'diagnostic.json'; path.write_text(json.dumps(value))
            record = evidence.load(path)[0]
        self.assertEqual(record.result, 'passed')
        self.assertEqual(record.observed['candidate']['host']['field'], '/description/invariants/host')
        self.assertNotIn('checkpoint', record.observed['candidate'])
        self.assertNotIn('settings', record.observed['candidate'])
        self.assertFalse(evidence.assess(record, {'id': 'model:fixed_batch', 'kind': record.kind}, {})['satisfied'])

    def test_invalid_and_unknown_receipts_are_missing_without_following_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'bad.json'
            for raw in ['{', '[]', '{}', '{"schema":"chopinfer-release-receipt/v1","checks":"invalid"}',
                        '{"schema":"future"}', '{"schema":"chopinfer.diagnostic.v1"}']:
                path.write_text(raw)
                records = evidence.load(path)
                self.assertTrue(records[0].issues)
            link = root / 'link.json'; link.symlink_to(path)
            self.assertEqual(evidence.load(link)[0].execution, 'not_run')
            sub = root / 'dir'; sub.mkdir(); (sub / 'data.json').write_text('{}')
            alias = root / 'alias'; alias.symlink_to(sub, target_is_directory=True)
            self.assertEqual(evidence.load(alias / 'data.json')[0].execution, 'not_run')
            with patch.object(evidence, 'MAX_BYTES', 2):
                self.assertTrue(evidence.load(path)[0].issues)

    def test_lifecycle_origins_partial_and_stale_are_separate_from_result(self):
        value = {'format': 'chopinfer.diagnostic.v1', 'mode': 'journal',
                 'extension': 'qwen38.lifecycle.v1', 'backend': 'qwen38-fake-host',
                 'execution_result': 'passed', 'completion': 'full', 'completed_actions': 9,
                 'expected_actions': 9, 'artifacts': {}, 'identity': {'input_sha256': 'a' * 64},
                 'producer': {'binary_sha256': 'b' * 64},
                 'device': {'execution': 'not_run', 'reason': 'idle-device evidence missing'}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'diagnostic.json'
            path.write_text(json.dumps(value))
            record = evidence.load(path)[0]
            self.assertEqual(record.result, 'passed')
            self.assertEqual(record.observed['candidate']['implementation']['field'], '/producer')
            self.assertEqual(record.observed['candidate']['workload']['field'], '/identity')
            self.assertEqual(record.raw['device']['execution'], 'not_run')
            self.assertNotIn('devices', record.observed['candidate'])
            obligation = {'id': 'qwen:lifecycle', 'kind': 'fake_host'}
            expected = {'candidate': {'implementation': {'binary_sha256': 'c' * 64}},
                        'coverage': {'comparisons': 9}}
            status = evidence.assess(record, obligation, expected)
            self.assertEqual((status['result'], status['applicability']), ('passed', 'stale'))
            value['completion'] = 'closed-failure'
            value['completed_actions'] = 7
            value['execution_result'] = 'failed'
            path.write_text(json.dumps(value))
            record = evidence.load(path)[0]
            self.assertFalse(record.coverage['complete'])
            status = evidence.assess(record, obligation, {})
            self.assertEqual((status['result'], status['applicability']), ('failed', 'missing'))

    def test_cpu_lifecycle_device_status_does_not_attach_to_gpu_obligations(self):
        value = {'format': 'chopinfer.diagnostic.v1', 'mode': 'journal',
                 'extension': 'qwen38.lifecycle.v1', 'backend': 'qwen38-fake-host',
                 'execution_result': 'passed', 'completion': 'full', 'completed_actions': 9,
                 'expected_actions': 9, 'artifacts': {}, 'identity': {},
                 'device': {'execution': 'not_run'}}
        obligation = {'id': 'unrelated:gpu', 'kind': 'gpu_correctness',
                      'evidence': {'selection': 'selected', 'execution': 'not_run'}}
        report = {'gaps': [], 'obligations': [obligation]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'diagnostic.json'
            path.write_text(json.dumps(value))
            evidence.attach(report, {}, [path])
        self.assertEqual(len(report['supplied_receipts']), 1)
        self.assertEqual(report['supplied_receipts'][0]['kind'], 'fake_host')
        self.assertNotIn('evidence_runs', obligation)


if __name__ == '__main__':
    unittest.main()
