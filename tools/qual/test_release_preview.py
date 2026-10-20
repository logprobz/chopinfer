"""CPU tests of the correctness runner: its plan, stage rules, verify markers,
stored baselines and a whole run with fake servers and children."""
from qual_test_config import setUpModule  # noqa: F401

import contextlib
import ctypes
import io
import json
import os
import re
from pathlib import Path
import shutil
import signal
import time
import tempfile
import sys
import threading
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import declaration
import release_preview as runner

QUALIFICATION_RUNS_ONCE = "runner orchestration with fake servers and children over the shipped declarations"
# Every model whose qual.json declares a preview section.
MODELS = sorted(runner.declaration.previewed(runner.ROOT))
# A value input's value, as another run's check run ID looks.
VALUE = '20261014T000000000000Z-0123abcd'


def model(predicate=lambda facts: True):
    """The first previewed model whose facts satisfy `predicate`."""
    return next(name for name in MODELS if predicate(runner.model_facts(name)))


def sharded(facts):
    return len(facts['checks'].get('prefix_exact', {}).get('files', [])) > 1


def warmed(facts):
    return bool(facts['soak_warmup'])


def determinism_receipt(name, arm, run_id, image):
    from determinism import (COLD_REPEAT, HIT_ANCHORS, HITS, PREFIX_HIT, PRIMERS, SAMPLING, hit_requests,
                             ids_digest, request_digest, request_names, request_set)
    names = request_names(image)
    ids = list(range(2800))

    def row(request):
        tokens = 1 if request in PRIMERS else 2
        return {'name': request, 'cache_salt': f"simulated-long-primer-{HITS[request]}" if request in HITS
                else f'simulated-{request}', 'text': ' ab'[:tokens + 1], 'reasoning': None, 'finish_reason': 'length',
                'token_ids': [1, 2][:tokens], 'tokens': [' a', 'b'][:tokens], 'logprobs': [-0.125, -0.25][:tokens],
                'top_logprobs': [[[' a', -0.125]], [['b', -0.25]]][:tokens],
                'usage': {'prompt_tokens': PRIMERS.get(request) or (2800 if request.startswith('long') else 8),
                          'completion_tokens': tokens,
                          'prompt_tokens_details': {'cached_tokens': HITS[request] // 64 * 64 if request in HITS else 0}}}
    return {'passed': True, 'mechanics_only': False, 'errors': [], 'server': {'model': name}, 'arm': arm,
            'run_id': run_id, 'concurrency': 1, 'sampling': SAMPLING,
            'request_set': {'names': names, 'sha256': request_digest(request_set(image) + hit_requests(ids)),
                            'long_prompt_tokens': len(ids), 'long_prompt_token_ids_sha256': ids_digest(ids)},
            'image': {'included': image, 'reason': 'simulated'}, 'cold_repeat': dict(COLD_REPEAT, equal=True),
            'prefix_hit': dict(PREFIX_HIT, anchors=list(HIT_ANCHORS), equal=True),
            'requests': [row(request) for request in names]}


def gsm8k_receipt(correct=200):
    return {'passed': True, 'mechanics_only': False, 'errors': [], 'count': 200, 'correct': correct,
            'examples_sha256': 'fake', 'sampling': {'seed': 1},
            'rows': [{'index': i, 'expected': '1', 'actual': '1', 'rule': '####', 'correct': i < correct,
                      'finish_reason': 'stop', 'response_sha256': f'{i:064x}'} for i in range(200)]}


def spec_receipt(name, arm, run_id):
    return {'passed': True, 'mechanics_only': False, 'errors': [], 'server': {'model': name}, 'output_tokens': 64,
            'arm': f'chopinfer-{arm}', 'run_id': run_id, 'sampling': {'temperature': 0, 'top_p': 1, 'logprobs': 5},
            'sets': {set_name: {'prompts_sha256': 'fake', 'tokens_per_engine_step': 1,
                                'speculation': ({'available': True, 'verify_items': 1, 'draft_tokens': 1}
                                                if arm == 'spec' else
                                                {'available': False, 'step_rows': {
                                                    'available': True, 'verify_steps': 0, 'steps': 1, 'bins': [
                                                        {'decode_rows': 1, 'verify_rows': 0, 'steps': 1,
                                                         'row_capacity': 1}]}}),
                                'requests': [{'tokens': ['a', 'b'], 'logprobs': [-0.1, -0.2],
                                              'top': [{'a': -0.1}, {'b': -0.2}], 'finish_reason': 'length'}
                                             for _ in range(16)]} for set_name in ('copy', 'natural')}}


def soak_receipt(facts, duration):
    value = {'passed': True, 'mechanics_only': False, 'errors': [], 'duration_seconds': duration,
             'measured_duration_seconds': duration, 'completed': 10, 'concurrency': facts['max_sequences'],
             'memory_samples': [{'elapsed_seconds': float(at), 'mib': 100, 'host_rss_mib': 100}
                                for at in range(0, duration + 1, 30)],
             'memory_scope': 'serving_process_tree', 'memory_growth_mib': 0,
             'mixed_requests': {'structured': 1, 'structured_truncated': 0}}
    if runner.soak_rule_arguments(facts):
        value.update(warmup_seconds=facts['soak_warmup'], cache_allowance_mib=facts['soak_allowance'])
    return value


def check_rows(facts, arms, checks):
    """The row IDs of `checks` on `arms`: a regression check's once per store."""
    return {f'default/{arm}/{check}/{store}' if runner.stored_inputs(facts['checks'][check])
            else f'default/{arm}/{check}' for arm in arms for check in checks for store in runner.STORES}


class RecordArgumentTests(unittest.TestCase):
    def check(self, arguments=None):
        check = {'module': 'verify_exact', 'files': ['verify_ar', 'verify_candidate']}
        if arguments is not None:
            check['constants'] = {'release_preview': {'RECORD_ARGUMENTS': {
                'value': arguments, 'source': 'tools/qual/README.md#correctness-runner'}}}
        return check

    def test_absent_arguments_preserve_both_default_records(self):
        for name in ('verify_ar', 'verify_candidate'):
            self.assertEqual(runner.record_for(name, self.check()), runner.RECORDS[name])

    def test_declared_arguments_select_concurrency_and_require_observed_capacity(self):
        args = ['--concurrency', '4', '--row-capacity', '32']
        base = list(runner.RECORDS['verify_candidate'][2])
        check = self.check({'verify_candidate': args})
        arm, script, actual = runner.record_for('verify_candidate', check)
        self.assertEqual((arm, script), runner.RECORDS['verify_candidate'][:2])
        self.assertEqual(actual, base + args)
        self.assertEqual(runner.RECORDS['verify_candidate'][2], base)
        self.assertEqual(runner.record_for('verify_ar', check), runner.RECORDS['verify_ar'])

    def test_refuse_sampling_changes_invalid_values_and_unknown_inputs(self):
        for args in (['--temperature', '1'], ['--concurrency', '0'], ['--concurrency'],
                     ['--concurrency', '1', '--concurrency', '2'], ['--k', '3'],
                     ['--row-capacity', '-1'], '4'):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.record_for('verify_candidate', self.check({'verify_candidate': args}))
        with self.assertRaises(ValueError):
            runner.record_for('verify_ar', self.check({'missing_input': ['--concurrency', '4']}))
        check = self.check({'spec_candidate': ['--concurrency', '4']})
        check['files'].append('spec_candidate')
        with self.assertRaises(ValueError):
            runner.record_for('spec_candidate', check)

    def test_plan_keeps_the_declared_values_and_refuses_conflicting_inputs(self):
        facts = dict(runner.model_facts(model()))
        check = self.check({'verify_ar': ['--concurrency', '4'],
                            'verify_candidate': ['--concurrency', '4', '--row-capacity', '32']})
        facts.update(full=['verify_exact'], checks={'verify_exact': check})
        config = {'screens': False, 'features': [], 'serve': []}
        arms = runner.plan(facts, 'full', 'default', config)
        records = {step['input']: step for arm in arms for step in arm['steps'] if step['kind'] == 'record'}
        self.assertEqual(records['verify_ar']['arguments'][-2:], ['--concurrency', '4'])
        self.assertEqual(records['verify_candidate']['arguments'][-4:],
                         ['--concurrency', '4', '--row-capacity', '32'])
        facts['full'].append('other_bucket')
        facts['checks']['other_bucket'] = self.check({'verify_candidate': ['--concurrency', '2']})
        with self.assertRaisesRegex(ValueError, 'conflicting record arguments'):
            runner.plan(facts, 'full', 'default', config)

    def test_ar_override_drops_the_configurations_speculation_mode(self):
        base = ['--speculative-tokens', '5', '--speculation', 'fixed']
        self.assertEqual(runner.serve_overrides(base, runner.ARMS['verify-ar']),
                         runner.ARMS['verify-ar'])
        self.assertEqual(runner.serve_overrides(base, runner.ARMS['verify-spec']),
                         base + runner.ARMS['verify-spec'])


class PlanTests(unittest.TestCase):
    def test_quick_starts_the_default_arms_with_determinism_first_and_checks_on_the_quick_arms(self):
        for model in MODELS:
            with self.subTest(model=model):
                facts = runner.model_facts(model)
                arms = runner.plan(facts, 'quick', 'default', runner.configurations(model)['default'])
                self.assertEqual([arm['arm'] for arm in arms], ['spec-repeat', 'ar', 'spec'])
                for arm in arms:
                    self.assertEqual(arm['steps'][0]['id'], f"determinism-{arm['arm']}")
                    self.assertEqual(arm['serve'], runner.ARMS[arm['arm']])
                spec = {step['id'] for step in arms[-1]['steps']}
                self.assertTrue({'gsm8k-spec', 'greedy-spec', 'spec_candidate', 'needle', 'prefix_slice', 'queue',
                                 'gsm8k_branches'} <= spec)
                for arm in arms:
                    checks = [step['check'] for step in arm['steps'] if step['kind'] == 'check']
                    self.assertEqual(checks, facts['quick'] if arm['arm'] in facts['quick_arms'] else [])

    def test_full_records_each_declared_input_once_on_its_own_server(self):
        name = model(lambda facts: 'prefix_exact' in facts['full'] and 'verify_exact' in facts['full'])
        facts = runner.model_facts(name)
        arms = runner.plan(facts, 'full', 'default', runner.configurations(name)['default'])
        by_arm = {arm['arm']: arm for arm in arms}
        self.assertEqual(arms[-1]['arm'], 'spec')
        expected = {'verify-ar', 'verify-spec', 'prefix', 'reserved', 'optimistic', 'spec-repeat', 'ar', 'spec'}
        if facts.get('kv_offload'):
            expected.update({'kv-offload-plain-off', 'kv-offload-plain-swap',
                             'kv-offload-pressure-off', 'kv-offload-pressure-swap'})
        self.assertEqual(set(by_arm), expected)
        records = [step for arm in arms for step in arm['steps'] if step['kind'] == 'record']
        inputs = [step['input'] for step in records]
        self.assertEqual(len(inputs), len(set(inputs)))
        # The determinism stages record determinism_ar and determinism_spec;
        # the declared spec_verify inputs are the screen's records.
        self.assertNotIn('determinism_ar', inputs)
        self.assertEqual(sorted(name for name in inputs if name.startswith('spec_')), ['spec_ar', 'spec_candidate'])
        shards = [step for step in by_arm['prefix']['steps'] if step['kind'] == 'record']
        files = facts['checks']['prefix_exact']['files']
        self.assertEqual([step['input'] for step in shards], files)
        self.assertEqual([step['arguments'][-1] for step in shards], [f'{i}/{len(files)}' for i in range(1, len(files) + 1)])
        self.assertEqual(by_arm['verify-spec']['serve'], runner.FORCED)
        self.assertIn('soak-ar', [step['id'] for step in by_arm['ar']['steps']])
        checks = [step['check'] for step in by_arm['spec']['steps'] if step['kind'] == 'check']
        self.assertEqual(checks, facts['full'])

    def test_full_frontend_check_has_a_bound_producer_on_its_own_arm(self):
        found = 0
        for model in MODELS:
            facts = runner.model_facts(model)
            for name, config in runner.configurations(model).items():
                selected = runner.configuration_facts(facts, name)
                checks = [c for c in selected['full'] if selected['checks'][c]['module'] == 'frontend_goldens'
                          and 'capture' in selected['checks'][c].get('files', [])]
                if not checks:
                    continue
                arms = runner.plan(facts, 'full', name, config)
                for check in checks:
                    arm = next(a for a in arms if any(s.get('check') == check for s in a['steps']))
                    records = [s for s in arm['steps'] if s.get('input') == 'capture']
                    self.assertEqual(len(records), 1)
                    record, = records
                    self.assertTrue(record['produces'])
                    self.assertEqual(record['script'], 'frontend_goldens.py')
                    self.assertIn('--server-log', record['arguments'])
                    self.assertIn('--capture-out', record['arguments'])
                    self.assertLess(arm['steps'].index(record), next(i for i, s in enumerate(arm['steps']) if s.get('check') == check))
                    found += 1
        self.assertGreater(found, 0)

    def test_layer_checks_capture_on_their_diagnostic_arm_without_rebuilding_references(self):
        found = 0
        for name in MODELS:
            facts = runner.model_facts(name)
            for config_name, config in runner.configurations(name).items():
                selected = runner.configuration_facts(facts, config_name)
                layers = [check for check in selected['full']
                          if 'layer_dump' in selected['checks'][check].get('files', [])]
                if not layers:
                    continue
                found += len(layers)
                arms = runner.plan(facts, 'full', config_name, config, {'layer_reference': 'frozen-reference'})
                records = [(arm, step) for arm in arms for step in arm['steps'] if step['kind'] == 'record']
                self.assertNotIn('layer_reference', [step['input'] for _, step in records])
                for check in layers:
                    arm = next(arm for arm in arms if any(step.get('check') == check for step in arm['steps']))
                    record = next(step for step in arm['steps'] if step.get('input') == 'layer_dump')
                    self.assertEqual(record['script'], selected['checks'][check]['module'] + '.py')
                    self.assertTrue(record['produces'])
                    self.assertIn('diagnostic-taps', arm['features'])
                    self.assertEqual(arm['environment']['CHOPINFER_TAPS_DIR'], '{taps_dir}')
                    self.assertIn('--reference', record['arguments'])
                    self.assertIn(tuple(sorted(arm['features'])),
                                  runner.configuration_features(facts, config_name, config, 'full'))
        self.assertGreater(found, 0)

    def test_pinned_encoders_are_captured_and_judged_on_the_same_diagnostic_arm(self):
        manifests = sorted(runner.ROOT.glob('models/*/encoder-parity-inputs.json'))
        self.assertGreaterEqual(len(manifests), 2)
        for manifest in manifests:
            name = manifest.parent.name
            with self.subTest(model=name):
                facts = runner.model_facts(name)
                config = runner.configurations(name)['default']
                arms = runner.plan(facts, 'full', 'default', config)
                [arm] = [arm for arm in arms if any(step.get('check') == 'encoder_parity' for step in arm['steps'])]
                self.assertEqual(arm['arm'], 'encoders')
                self.assertIn('diagnostic-taps', arm['features'])
                self.assertEqual(arm['environment']['CHOPINFER_REQUEST_TRACE'], '1')
                self.assertEqual(arm['serve'][arm['serve'].index('--modalities') + 1], 'all')
                [record] = [step for step in arm['steps'] if step.get('input') == 'encoder_parity_run']
                self.assertEqual(record['script'], 'encoder_capture.py')
                self.assertTrue(record['produces'])
                self.assertLess(arm['steps'].index(record), next(i for i, step in enumerate(arm['steps'])
                                                               if step.get('check') == 'encoder_parity'))
                self.assertIn(tuple(sorted(arm['features'])),
                              runner.configuration_features(facts, 'default', config, 'full'))

    def test_a_check_module_can_bring_its_own_arm_and_record(self):
        facts = runner.model_facts(model())
        facts = dict(facts, full=['offload'], checks={'offload': {'module': 'fake_offload', 'files': ['offload_swap']}})
        module = SimpleNamespace(RUNNER_ARMS={'swap': ['--kv-offload', 'on']},
                                 RUNNER_RECORDS={'offload_swap': ('swap', 'fake_offload.py', ['record', '--url', '{url}'])})
        with patch.object(runner, 'module_records', lambda name, model=None: (module.RUNNER_ARMS, module.RUNNER_RECORDS)):
            arms = runner.plan(facts, 'full', 'default', {'serve': ['--x'], 'environment': {}, 'features': []})
        swap = next(arm for arm in arms if arm['arm'] == 'swap')
        self.assertEqual(swap['serve'], ['--x', '--kv-offload', 'on'])
        self.assertEqual(swap['steps'], [{'kind': 'record', 'id': 'offload_swap', 'input': 'offload_swap',
                                          'script': 'fake_offload.py', 'arguments': ['record', '--url', '{url}']}])
        self.assertEqual(arms[-1]['arm'], 'spec')

    def test_a_continuation_check_records_on_its_servers_and_is_judged_after_them(self):
        name = model(lambda facts: 'spec_continuation' in facts['full'])
        facts = runner.model_facts(name)
        config = runner.configurations(name)['default']
        arms = runner.plan(facts, 'full', 'default', config)
        self.assertEqual([arm['arm'] for arm in arms[-2:]], ['spec', runner.OFFLINE])
        offline = arms[-1]
        self.assertIs(offline['server'], False)
        records = {step['input']: arm['arm'] for arm in arms for step in arm['steps'] if step['kind'] == 'record'}
        self.assertEqual((records['continuation_spec'], records['continuation_ar']), ('spec', 'ar'))
        self.assertEqual([(step['kind'], step['id'], step.get('produces')) for step in offline['steps']],
                         [('record', 'continuation_spec_reference', True), ('record', 'continuation_ar_reference', True),
                          ('check', 'check-spec_continuation', None)])
        self.assertIn('{continuation_spec}', offline['steps'][0]['arguments'])
        self.assertIn('{checkpoint}', offline['steps'][0]['arguments'])
        served = [step['check'] for arm in arms[:-1] for step in arm['steps'] if step['kind'] == 'check']
        self.assertNotIn('spec_continuation', served)
        # Records given with --input are not taken again; the references and
        # the check still run once the servers are down.
        given = runner.plan(facts, 'full', 'default', config, {'continuation_spec': 's', 'continuation_ar': 'a'})
        inputs = [step['input'] for arm in given for step in arm['steps'] if step['kind'] == 'record']
        self.assertNotIn('continuation_spec', inputs)
        self.assertEqual([step['id'] for step in given[-1]['steps']], [step['id'] for step in offline['steps']])
        facts, configs = runner.select_checks(facts, {'default': config}, 'full', ['spec_continuation'])
        selected = runner.plan(facts, 'full', 'default', configs['default'])
        self.assertEqual([arm['arm'] for arm in selected], ['ar', 'spec', runner.OFFLINE])

    def test_a_module_s_offline_records_and_check_follow_every_server(self):
        facts = runner.model_facts(model())
        facts = dict(facts, full=['late'], checks={'late': {'module': 'fake_late',
                                                            'files': ['late_record', 'late_reference']}})
        options = {'RUNNER_RECORDS': {'late_record': ('ar', 'fake_late.py', ['record', '--url', '{url}'])},
                   'RUNNER_OFFLINE_RECORDS': {'late_reference': ('fake_late.py', ['reference', '--record',
                                                                                  '{late_record}', '--out', '{run_dir}'])},
                   'RUNNER_CHECK_ARM': runner.OFFLINE}

        def option(module, name, default, model=None):
            return options.get(name, default) if module == 'fake_late' else default

        with patch.object(runner, 'module_option', option):
            arms = runner.plan(facts, 'full', 'default', {'serve': ['--x'], 'environment': {}, 'features': [],
                                                          'screens': False})
        self.assertEqual([(arm['arm'], arm.get('server', True)) for arm in arms],
                         [('ar', True), (runner.OFFLINE, False)])
        self.assertEqual(arms[-1]['serve'], [])
        self.assertEqual([(step['kind'], step['id'], step.get('produces')) for step in arms[-1]['steps']],
                         [('record', 'late_reference', True), ('check', 'check-late', None)])

    def test_a_configuration_without_the_screens_runs_its_declared_checks_and_records(self):
        name = model(lambda facts: 'verify_exact' in facts['full'])
        facts = runner.model_facts(name)
        config = dict(runner.configurations(name)['default'], serve=['--x'], screens=False, tensor_parallel_size=2)
        arms = runner.plan(facts, 'full', 'default', config)
        steps = [step for arm in arms for step in arm['steps']]
        self.assertFalse([step for step in steps if step['kind'] == 'stage' and step['stage'] != 'determinism'])
        self.assertIn('verify_ar', [step['input'] for step in steps if step['kind'] == 'record'])
        self.assertEqual([step['check'] for step in arms[-1]['steps'] if step['kind'] == 'check'], facts['full'])
        self.assertTrue(all(arm['serve'][0] == '--x' for arm in arms))
        quick = runner.plan(facts, 'quick', 'default', config)
        self.assertFalse([step for arm in quick for step in arm['steps']
                          if step['kind'] == 'stage' and step['stage'] != 'determinism'])
        self.assertEqual([step['check'] for arm in quick for step in arm['steps'] if step['kind'] == 'check'],
                         facts['quick'])

    def test_declared_configurations_bring_their_own_checks_serving_and_package(self):
        for name in MODELS:
            facts = runner.model_facts(name)
            declared = runner.configurations(name)
            self.assertEqual(declared['default']['profile'], facts['profile'])
            self.assertEqual(list(runner.selected_configurations(name, [], 'quick')), ['default'])
            self.assertEqual(list(runner.selected_configurations(name, [], 'full')), list(declared))
            for config, spec in declaration.configurations(name).items():
                with self.subTest(model=name, config=config):
                    served = declared[config]
                    self.assertEqual(served['serve'], spec.get('serve', []))
                    self.assertEqual(served['tensor_parallel_size'], spec.get('tensor_parallel_size', 4))
                    self.assertFalse(served['screens'])
                    expanded = runner.read(declaration.named(f'{name}.{config}'))
                    self.assertEqual(expanded['variant'], config)
                    arms = runner.plan(facts, 'full', config, served)
                    checks = [step['check'] for arm in arms for step in arm['steps'] if step['kind'] == 'check']
                    selected = expanded['cohorts']['full']['checks']
                    self.assertCountEqual(checks, selected)
                    for arm in arms:
                        ordered = [step['check'] for step in arm['steps'] if step['kind'] == 'check']
                        self.assertEqual(ordered, [check for check in selected if check in ordered])
                    # The configuration's own server starts last; a module's
                    # offline steps run after it, on no server.
                    last = [arm for arm in arms if arm.get('server', True)][-1]
                    self.assertEqual(last['serve'][:len(served['serve'])], served['serve'])

    def test_two_gpu_configurations_run_together_on_separate_pairs(self):
        configs = {'default': {'tensor_parallel_size': 4}, 'tp2-a': {'tensor_parallel_size': 2},
                   'tp2-b': {'tensor_parallel_size': 2}}
        both = threading.Barrier(2, timeout=10)
        ran = []

        def one(run, name, config, devices=None):
            ran.append((name, devices))
            if devices:
                # Each pair waits for the other: they only pass together.
                both.wait()

        with patch.object(runner, 'run_configuration', one), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4,5,6,7'}):
            runner.run_configurations(None, configs)
        self.assertEqual(ran[0], ('default', None))
        self.assertEqual(sorted(devices for _, devices in ran[1:]), ['4,5', '6,7'])

    def test_a_configuration_producer_records_the_run_its_checks_read(self):
        facts = runner.model_facts(model())
        facts = dict(facts, full=['parity'], checks={'parity': {'module': 'encoder_parity', 'files': ['parity_run']}})
        config = dict(runner.configurations(model())['default'], screens=False, producer='models/x/producer.py')
        [arm] = runner.plan(facts, 'full', 'default', config)
        self.assertEqual(arm['steps'][0], {'kind': 'record', 'id': 'parity_run', 'input': 'parity_run',
                                           'script': 'models/x/producer.py', 'arguments': runner.PRODUCER,
                                           'produces': True})
        with tempfile.TemporaryDirectory() as temporary:
            arm_dir = Path(temporary)
            context = {'fill': {'url': 'u', 'server_log': 'l', 'taps_dir': 't'}, 'records': {}, 'declaration': 'd',
                       'env': {}}
            calls = []

            def child(run, context, directory, script, arguments, seconds, config_path):
                calls.append((script, runner.fill_arguments(arguments, context['fill'])))
                (directory / 'run').mkdir(parents=True)
                return 0

            with patch.object(runner, 'run_child', child):
                result = runner.run_record(None, context, arm['steps'][0], arm_dir)
            self.assertEqual((result['exit'], context['records']['parity_run']), (0, str(arm_dir / 'parity_run/run')))
            self.assertEqual(calls[0][1][-2:], ['--run', str(arm_dir / 'parity_run/run')])
            # A producer that leaves no run directory fails its record.
            with patch.object(runner, 'run_child', lambda *a: 0):
                self.assertEqual(runner.run_record(None, context, dict(arm['steps'][0], id='none'), arm_dir)['exit'], 1)

    def test_given_inputs_are_not_recorded_and_unknown_configurations_are_refused(self):
        name = model(lambda facts: 'prefix_exact' in facts['full'] and not sharded(facts))
        facts = runner.model_facts(name)
        arms = runner.plan(facts, 'full', 'default', runner.configurations(name)['default'],
                           inputs={'prefix_record': '/given'})
        self.assertNotIn('prefix', [arm['arm'] for arm in arms])
        with self.assertRaisesRegex(ValueError, 'declares no configuration'):
            runner.selected_configurations(name, ['tp9'])


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.facts = runner.model_facts(model())

    def test_the_image_request_needs_media_and_image_among_the_served_modalities(self):
        for media, words, image in ((True, [], True), (True, ['--modalities', 'auto'], True),
                                    (True, ['--modalities', 'all'], True),
                                    (True, ['--speculative-tokens', '5', '--modalities', 'text,image'], True),
                                    (True, ['--modalities', 'text'], False),
                                    (True, ['--modalities', 'text,video'], False),
                                    (False, [], False), (False, ['--modalities', 'all'], False)):
            with self.subTest(media=media, words=words):
                request = runner.image_request({'media': media}, words)
                self.assertEqual(request == ['--image'], image)
                if not image:
                    self.assertEqual(request[0], '--image-reason')
                    self.assertTrue(request[1])
        name = self.facts['name']
        runner.judge_determinism_capture(determinism_receipt(name, 'spec', 'r', False), self.facts, 'spec', 'r', False)
        with self.assertRaisesRegex(ValueError, 'image request differs'):
            runner.judge_determinism_capture(determinism_receipt(name, 'spec', 'r', True), self.facts, 'spec', 'r',
                                             False)

    def test_gsm8k_holds_the_floor_and_counts_length_limited_answers_incorrect(self):
        values, extra = runner.judge_gsm8k(gsm8k_receipt(), self.facts)
        self.assertEqual(values, [runner.metric('correct', 200, 'ge', self.facts['gsm8k_floor'])])
        value = gsm8k_receipt()
        value['rows'][7].update(finish_reason='length', actual='', correct=False)
        value['correct'] = 199
        self.assertEqual(runner.judge_gsm8k(value, self.facts)[1]['length_limited_indices'], [7])
        with self.assertRaisesRegex(ValueError, 'below'):
            runner.judge_gsm8k(gsm8k_receipt(self.facts['gsm8k_floor'] - 1), self.facts)
        value = gsm8k_receipt()
        value['rows'][3]['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'length-limited'):
            runner.judge_gsm8k(value, self.facts)

    def test_soak_is_judged_by_the_soak_kind_with_the_stage_arguments(self):
        plain = runner.model_facts(model(lambda facts: not warmed(facts)))
        rule = runner.soak_arguments(plain, 1200)
        values, _ = runner.judge_soak(soak_receipt(plain, 1200), plain, rule)
        self.assertEqual({value['metric']: value['value'] for value in values},
                         {'memory_growth_mib': 0, 'host_memory_growth_mib': 0, 'steady_memory_growth_mib': 0,
                          'steady_host_memory_growth_mib': 0})
        with self.assertRaisesRegex(ValueError, 'shorter than the declared duration'):
            runner.judge_soak(soak_receipt(plain, 600), plain, rule)
        # The measured interval, not only the requested duration, must cover the rule.
        short = dict(soak_receipt(plain, 1200), measured_duration_seconds=0)
        with self.assertRaisesRegex(ValueError, 'shorter than the declared duration'):
            runner.judge_soak(short, plain, rule)
        gap = soak_receipt(plain, 1200)
        gap['memory_samples'] = [row for row in gap['memory_samples'] if not 300 < row['elapsed_seconds'] < 600]
        with self.assertRaisesRegex(ValueError, 'do not span the measurement interval'):
            runner.judge_soak(gap, plain, rule)
        incomplete = soak_receipt(plain, 1200)
        del incomplete['memory_samples'][3]['host_rss_mib']
        with self.assertRaisesRegex(ValueError, 'incomplete device or host memory samples'):
            runner.judge_soak(incomplete, plain, rule)
        # Host growth fails like device growth, whatever the receipt's own field says.
        leak = soak_receipt(plain, 1200)
        for row in leak['memory_samples']:
            row['host_rss_mib'] += 5 * row['elapsed_seconds']
        with self.assertRaisesRegex(ValueError, 'grew more than'):
            runner.judge_soak(leak, plain, rule)
        device = soak_receipt(plain, 1200)
        device['memory_samples'][-1]['mib'] += 513
        with self.assertRaisesRegex(ValueError, 'grew more than'):
            runner.judge_soak(device, plain, rule)
        with self.assertRaisesRegex(ValueError, 'below the declared concurrency'):
            runner.judge_soak(dict(soak_receipt(plain, 1200), concurrency=1), plain, rule)
        with self.assertRaisesRegex(ValueError, 'prefix-cache'):
            runner.judge_soak(soak_receipt(plain, 1200), plain, runner.soak_arguments(plain, 1200, prefix=True))
        # A declared warm-up and cache allowance: a bounded growth in the
        # warm-up passes, a steady leak after it fails.
        if not any(warmed(runner.model_facts(name)) for name in MODELS):
            return
        # Each growth prints the limit it is judged against: over the whole
        # soak the steady limit plus the memory's declared allowance.
        allowed = runner.model_facts(model(lambda facts: warmed(facts) and facts['soak_allowance']['host'] > 0))
        value = soak_receipt(allowed, 1200)
        for row in value['memory_samples']:
            row['host_rss_mib'] += 600 if row['elapsed_seconds'] >= 30 else 0
        values, _ = runner.judge_soak(value, allowed, runner.soak_arguments(allowed, 1200))
        limits = {item['metric']: (item['value'], item['limit']) for item in values}
        self.assertEqual(limits['host_memory_growth_mib'], (600, 512 + allowed['soak_allowance']['host']))
        self.assertEqual(limits['memory_growth_mib'], (0, 512 + allowed['soak_allowance']['device']))
        self.assertEqual(limits['steady_host_memory_growth_mib'][1], 512)
        self.assertEqual(limits['steady_memory_growth_mib'][1], 512)
        warm = runner.model_facts(model(warmed))
        rule = runner.soak_arguments(warm, 1200)
        value = soak_receipt(warm, 1200)
        for row in value['memory_samples']:
            row['host_rss_mib'] += min(row['elapsed_seconds'], warm['soak_warmup'] * 2 // 3) * 8
        runner.judge_soak(value, warm, rule)
        for row in value['memory_samples']:
            row['host_rss_mib'] += max(row['elapsed_seconds'] - warm['soak_warmup'], 0)
        with self.assertRaisesRegex(ValueError, 'grew more than'):
            runner.judge_soak(value, warm, rule)

    def test_queue_needle_and_greedy_rules(self):
        clients = runner.queue_clients(self.facts)
        runner.judge_queue({'passed': True, 'mechanics_only': False, 'requests': clients, 'concurrency': clients},
                           self.facts)
        with self.assertRaises(ValueError):
            runner.judge_queue({'passed': True, 'mechanics_only': False, 'requests': clients - 1,
                                'concurrency': clients}, self.facts)
        runner.judge_needle({'passed': True, 'mechanics_only': False,
                             'rows': [{'context': self.facts['context'], 'passed': True}]}, self.facts)
        with self.assertRaises(ValueError):
            runner.judge_needle({'passed': True, 'mechanics_only': False,
                                 'rows': [{'context': 8192, 'passed': True}]}, self.facts)
        greedy = {'passed': True, 'mechanics_only': False, 'server': {'model': self.facts['name']}, 'concurrency': 1,
                  'k': self.facts['depth'], 'prompt_tokens': 256, 'max_tokens': 256, 'reference_sha256': 'r',
                  'requests': {'text': [1, 2]}, 'finish_reasons': {'text': 'length'}}
        runner.judge_greedy(greedy, self.facts, 'spec', 'r')
        for change in ({'k': 0}, {'reference_sha256': 'other'}, {'max_tokens': 128}, {'requests': {'text': []}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.judge_greedy(dict(greedy, **change), self.facts, 'spec', 'r')

    def test_spec_verify_captures_must_exercise_their_paths(self):
        runner.judge_spec_capture(spec_receipt(self.facts['name'], 'spec', 'run'), self.facts, 'spec', 'run')
        runner.judge_spec_capture(spec_receipt(self.facts['name'], 'ar', 'run'), self.facts, 'ar', 'run')
        lazy = spec_receipt(self.facts['name'], 'spec', 'run')
        lazy['sets']['copy']['speculation']['verify_items'] = 0
        with self.assertRaisesRegex(ValueError, 'execution path'):
            runner.judge_spec_capture(lazy, self.facts, 'spec', 'run')
        with self.assertRaisesRegex(ValueError, 'wrong spec_verify arm or run'):
            runner.judge_spec_capture(spec_receipt(self.facts['name'], 'spec', 'other'), self.facts, 'spec', 'run')

    def test_printed_lines_show_values_limits_and_stored_values(self):
        row = {'id': 'default/spec/gsm8k', 'status': 'pass',
               'values': [dict(runner.metric('correct', 196, 'ge', 193), **{'last-good': 197, 'seed': 196})]}
        self.assertEqual(runner.line(row), 'default/spec/gsm8k PASS; correct 196 >= 193 (last-good 197, seed 196)')
        values = runner.predicate_values({'mean_abs_dlogprob': 0.06, 'limits': {'mean_abs_dlogprob': 0.1}},
                                         [{'path': ['mean_abs_dlogprob'], 'op': 'le', 'value': 0.1},
                                          {'path': ['passed'], 'op': 'eq', 'value': True}])
        self.assertEqual(values, [runner.metric('mean_abs_dlogprob', 0.06, 'le', 0.1)] * 2)

    def test_values_skip_a_gates_object_that_is_not_a_gate_list(self):
        # spec_verify's receipt holds its copy bounds as a `gates` object.
        receipt = {'passed': True, 'sets': {'copy': {'scored_positions': 64}},
                   'gates': {'set': 'copy', 'maximum_abs_dlogprob': 0.15, 'maximum_top1_disagreement': 0.08}}
        values = runner.predicate_values(receipt, [{'path': ['passed'], 'op': 'eq', 'value': True},
                                                   {'path': ['sets', 'copy', 'scored_positions'], 'op': 'ge',
                                                    'value': 1}])
        self.assertEqual(values, [runner.metric('sets.copy.scored_positions', 64, 'ge', 1)])


class ManifestSelectionTests(unittest.TestCase):
    def test_all_declared_plain_and_keyed_manifests_reach_verification(self):
        plain, keyed = [], []
        for name in MODELS:
            raw = runner.read(runner.ROOT / 'models' / name / 'qual.json')
            selected = runner.configurations(name)
            self.assertEqual(set(selected), {'default', *raw.get('configurations', {})})
            for config, value in selected.items():
                pin = (raw.get('configurations', {}).get(config, {}).get('qualification', {})
                       .get('checkpoint', raw['qualification']['checkpoint']))
                expected = pin['lock'] + ('#' + pin['pin'] if 'pin' in pin else '')
                with self.subTest(model=name, config=config):
                    self.assertEqual(value['manifest'], expected)
                (keyed if 'pin' in pin else plain).append((name, config))
            pin = raw['qualification']['checkpoint']
            with self.subTest(model=name, config='facts'):
                self.assertEqual(runner.model_facts(name)['manifest'],
                                 pin['lock'] + ('#' + pin['pin'] if 'pin' in pin else ''))
        self.assertGreaterEqual(len(plain), 2)
        self.assertGreaterEqual(len(keyed), 2)


class MarkerTests(unittest.TestCase):
    def test_a_marker_holds_until_a_checkpoint_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'VALIDATION', Path(temporary) / 'v'):
            checkpoint = Path(temporary) / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'model.safetensors').write_bytes(b'x' * 8)
            manifest = str(Path(temporary) / 'manifest.json')
            runner.write(manifest, {})
            self.assertIsNone(runner.verify_marker('model', checkpoint, manifest))
            runner.write(runner.marker_path('model', checkpoint, manifest),
                         {'state_sha256': runner.state_digest(runner.checkpoint_state(checkpoint)),
                          'manifest': runner.manifest_identity(manifest)})
            self.assertIsNotNone(runner.verify_marker('model', checkpoint, manifest))
            os.utime(checkpoint / 'model.safetensors', ns=(1, 1))
            self.assertIsNone(runner.verify_marker('model', checkpoint, manifest))

    def test_cached_proof_requires_the_same_manifest_file_entry_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'VALIDATION', Path(temporary) / 'v'):
            base = Path(temporary)
            checkpoint = base / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'model.safetensors').write_bytes(b'x' * 8)
            manifest = base / 'manifest.json'
            runner.write(manifest, {'first': {}, 'second': {}})
            selection = str(manifest) + '#first'
            proof = {'state_sha256': runner.state_digest(runner.checkpoint_state(checkpoint)),
                     'manifest': runner.manifest_identity(selection)}
            marker = runner.marker_path('model', checkpoint, selection)
            runner.write(marker, proof)
            self.assertEqual(runner.verify_marker('model', checkpoint, selection), proof)
            other_file = base / 'other.json'
            other_file.write_bytes(manifest.read_bytes())
            for other in (str(manifest) + '#second', str(manifest), str(other_file) + '#first'):
                runner.write(runner.marker_path('model', checkpoint, other), proof)
                self.assertIsNone(runner.verify_marker('model', checkpoint, other))
            runner.write(marker, {**proof, 'manifest': {**proof['manifest'], 'sha256': '0' * 64}})
            self.assertIsNone(runner.verify_marker('model', checkpoint, selection))
            for invalid in ({'state_sha256': proof['state_sha256']}, [], None):
                runner.write(marker, invalid)
                self.assertIsNone(runner.verify_marker('model', checkpoint, selection))
            runner.write(marker, proof)
            manifest.unlink()
            self.assertIsNone(runner.verify_marker('model', checkpoint, selection))
            manifest.symlink_to(other_file)
            self.assertIsNone(runner.verify_marker('model', checkpoint, selection))
            with self.assertRaises(OSError):
                runner.manifest_identity(selection)



class PrepareTests(unittest.TestCase):
    def test_prepare_builds_once_verifies_once_and_records_the_package(self):
        name = model()
        facts = runner.model_facts(name)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            base = Path(temporary)
            checkpoint = base / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'model.safetensors').write_bytes(b'x' * 8)
            manifest = base / 'manifest.json'
            manifest.write_text('{}\n')
            facts = dict(facts, manifest=str(manifest))
            target = base / 'target'
            calls = []
            change_during_verify = False

            def call(command, log, seconds, env):
                command = list(map(str, command))
                calls.append(command)
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                with Path(log).open('w') as output:
                    if command[0] == 'cargo':
                        (target / 'release').mkdir(parents=True, exist_ok=True)
                        for binary in ('chopinfer', runner.host_name(), facts['helper'], *facts['modules']):
                            (target / 'release' / binary).write_text('fake')
                    elif command[1] == 'verify':
                        self.assertEqual(command[command.index('--manifest') + 1], str(manifest))
                        if change_during_verify:
                            manifest.write_text('{"changed": true}\n')
                        output.write('verify: 1 verified, 0 failed, 0 missing, 0 unrecorded; 8 bytes\n')
                    else:
                        bundle = Path(command[command.index('--cache-dir') + 1]) / 'package'
                        runner.write(bundle / 'manifest.json', {'resource_requirements': {
                            'required_capabilities': [runner.MEDIA_CAPABILITY]}})
                        output.write(json.dumps({'package': {
                            'bundle_id': facts['bundle'], 'tensor_parallel_size': 4, 'bundle': str(bundle),
                            'package_identity': 'p', 'checkpoint_identity': 'c', 'metadata_identity': 'm',
                            'profile': {'name': facts['profile'], 'profile': {'speculation': {
                                'tokens': facts['depth'], 'drafter': facts['drafter']}}}}}) + '\n')
                return 0

            arguments = ['prepare', '--model', name, '--target-dir', str(target)]
            with patch.object(runner, 'call', call), patch.object(runner, 'source', FakeRun.identity), \
                    patch.object(runner, 'model_facts', return_value=facts), \
                    patch.dict(os.environ, {facts['checkpoint']: str(checkpoint)}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(arguments), 0)
                first = len(calls)
                self.assertEqual(runner.main(arguments), 0)
                reused = len(calls)
                first_proof = runner.verify_marker(name, checkpoint, facts['manifest'])
                self.assertIsNotNone(first_proof)
                manifest.write_text('{}  \n')
                self.assertEqual(runner.main(arguments), 0)
                fresh = runner.verify_marker(name, checkpoint, facts['manifest'])
                self.assertNotEqual(first_proof['manifest']['sha256'], fresh['manifest']['sha256'])
                self.assertEqual(sum(c[1] == 'verify' for c in calls), 2)
                manifest.unlink()
                self.assertEqual(runner.main(arguments), 2)
                self.assertEqual(sum(c[1] == 'verify' for c in calls), 2)
                manifest.write_text('{}\n')
                change_during_verify = True
                marker = runner.marker_path(name, checkpoint, facts['manifest'])
                previous = marker.read_bytes()
                self.assertEqual(runner.main(arguments), 2)
                self.assertEqual(marker.read_bytes(), previous)
                self.assertIsNone(runner.verify_marker(name, checkpoint, facts['manifest']))
            kinds_of = [command[0] if command[0] == 'cargo' else command[1] for command in calls[:reused]]
            # The second prepare reuses the install and the verify marker and
            # only asks the helper for its (reused) package again.
            self.assertEqual(kinds_of[:first].count('cargo'), 3)
            self.assertEqual(kinds_of.count('verify'), 1)
            self.assertNotIn('cargo', kinds_of[first:])
            install = runner.install_root(FakeRun.identity())
            state = runner.read(install / 'install.json')
            self.assertEqual(state['packages'][f'{name}/default']['media'], True)
            self.assertIn('bin/chopinfer', state['files'])


class CheckoutBuildTests(unittest.TestCase):
    """prepare builds an install only from the checkout it runs in, in a
    cargo target that checkout owns, and reuses only such an install."""

    def setUp(self):
        self.name = model()
        facts = runner.model_facts(self.name)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        checkpoint = self.base / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'model.safetensors').write_bytes(b'x' * 8)
        manifest = self.base / 'manifest.json'
        manifest.write_text('{}\n')
        self.facts = dict(facts, manifest=str(manifest))
        self.checkpoint = checkpoint
        self.builds = []
        # prepare reads policies the scorer judges after these tests replace
        # or remove CARGO_TARGET_DIR, so pin the scorer it resolves to now.
        self.scorer = str(runner.kinds.eval_binary())
        for patcher in (patch.object(runner, 'VALIDATION', self.base / 'validation'),
                        patch.object(runner, 'TARGET_ROOT', self.base / 'targets')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, command, log, seconds, env):
        """The build writes into the target cargo would use (CARGO_TARGET_DIR);
        verify and the direct package answer as prepare expects."""
        command = list(map(str, command))
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open('w') as output:
            if command[0] == 'cargo':
                target = Path(env['CARGO_TARGET_DIR'])
                self.builds.append(str(target))
                (target / 'release').mkdir(parents=True, exist_ok=True)
                for binary in ('chopinfer', runner.host_name(), self.facts['helper'], *self.facts['modules']):
                    (target / 'release' / binary).write_text(f'built in {target}')
            elif command[1] == 'verify':
                output.write('verify: 1 verified, 0 failed, 0 missing, 0 unrecorded; 8 bytes\n')
            else:
                bundle = Path(command[command.index('--cache-dir') + 1]) / 'package'
                runner.write(bundle / 'manifest.json', {'resource_requirements': {
                    'required_capabilities': [runner.MEDIA_CAPABILITY]}})
                output.write(json.dumps({'package': {
                    'bundle_id': self.facts['bundle'], 'tensor_parallel_size': 4, 'bundle': str(bundle),
                    'package_identity': 'p', 'checkpoint_identity': 'c', 'metadata_identity': 'm',
                    'profile': {'name': self.facts['profile'], 'profile': {'speculation': {
                        'tokens': self.facts['depth'], 'drafter': self.facts['drafter']}}}}}) + '\n')
        return 0

    def prepare(self, checkout, *extra, environ=None):
        """prepare as run from `checkout`; its exit status and stderr."""
        errors = io.StringIO()
        variables = {self.facts['checkpoint']: str(self.checkpoint), 'CHOPINFER_EVAL': self.scorer,
                     **(environ or {})}
        with patch.object(runner, 'call', self.call), patch.object(runner, 'source', FakeRun.identity), \
                patch.object(runner, 'model_facts', return_value=self.facts), \
                patch.object(runner, 'checkout', lambda: Path(checkout)), \
                patch.dict(os.environ, variables), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            if environ is None:
                os.environ.pop('CARGO_TARGET_DIR', None)
            code = runner.main(['prepare', '--model', self.name, *extra])
        return code, errors.getvalue()

    def test_without_a_target_each_checkout_builds_in_its_own(self):
        first, second = self.base / 'one' / 'chopinfer', self.base / 'two' / 'chopinfer'
        self.assertEqual(self.prepare(first)[0], 0)
        install = runner.install_root(FakeRun.identity())
        shutil.move(install, self.base / 'aside')
        self.assertEqual(self.prepare(second)[0], 0)
        self.assertEqual(len(self.builds), 6)
        targets = sorted(set(self.builds))
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertEqual(Path(target).parent, self.base / 'targets')
            self.assertNotEqual(Path(target), self.base / 'validation' / 'target')

    def test_a_caller_s_cargo_target_is_kept(self):
        own = self.base / 'own-target'
        self.assertEqual(self.prepare(self.base / 'one', environ={'CARGO_TARGET_DIR': str(own)})[0], 0)
        self.assertEqual(set(self.builds), {str(own)})

    def test_a_second_checkout_cannot_use_the_first_checkout_s_target(self):
        target = self.base / 'shared-target'
        self.assertEqual(self.prepare(self.base / 'one', '--target-dir', str(target))[0], 0)
        install = runner.install_root(FakeRun.identity())
        shutil.move(install, self.base / 'aside')
        code, errors = self.prepare(self.base / 'two', '--target-dir', str(target))
        self.assertEqual(code, 2)
        self.assertIn('belongs to the checkout', errors)
        self.assertEqual(len(self.builds), 3)

    def test_a_used_target_without_an_owner_is_claimed_only_when_assigned_here(self):
        target = self.base / 'used-target'
        (target / 'release').mkdir(parents=True)
        code, errors = self.prepare(self.base / 'one', '--target-dir', str(target))
        self.assertEqual(code, 2)
        self.assertIn('names no owning checkout', errors)
        target.with_name('used-target.worktree').write_text(f"{self.base / 'one'}\n")
        self.assertEqual(self.prepare(self.base / 'one', '--target-dir', str(target))[0], 0)
        self.assertEqual((target / runner.TARGET_OWNER).read_text().strip(), str(self.base / 'one'))

    def test_two_checkouts_claiming_one_empty_target_at_once_leave_one_owner(self):
        # Regression: both passed the empty-target check
        # before either claimed it, and both returned the target.
        target = self.base / 'contested'
        first, second = self.base / 'one', self.base / 'two'
        mkdir, results, interleaved_once = Path.mkdir, {}, []

        def claim(root):
            with patch.object(runner, 'checkout', lambda: root):
                try:
                    results[root] = runner.build_target(str(target))
                except ValueError as error:
                    results[root] = error

        def interleaved(path, *args, **kwargs):
            # The second checkout runs its whole claim once the first has
            # found the target empty and is about to claim it.
            if path == target and not interleaved_once:
                interleaved_once.append(path)
                claim(second)
            return mkdir(path, *args, **kwargs)

        with patch.object(Path, 'mkdir', interleaved):
            claim(first)
        self.assertEqual(results[second], target)
        self.assertIsInstance(results[first], ValueError)
        self.assertIn(f'belongs to the checkout {second}', str(results[first]))
        self.assertEqual((target / runner.TARGET_OWNER).read_text(), f'{second}\n')
        self.assertEqual(sorted(path.name for path in target.iterdir()), [runner.TARGET_OWNER])

    def extension(self, checkout, retained=lambda install: None):
        """`checkout` adds a second model's helper to the install the first
        checkout built, after `retained` changed it; prepare's exit status
        and stderr. Nothing is built and the helper is not added."""
        self.assertEqual(self.prepare(self.base / 'one')[0], 0)
        install = runner.install_root(FakeRun.identity())
        retained(install)
        builds = len(self.builds)
        self.facts = dict(self.facts, helper='second-model-helper')
        code, errors = self.prepare(checkout)
        self.assertEqual(len(self.builds), builds)
        state = runner.read(install / 'install.json')
        self.assertNotIn('bin/second-model-helper', state['files'])
        return code, errors

    def test_the_checkout_that_built_an_install_extends_it(self):
        self.assertEqual(self.prepare(self.base / 'one')[0], 0)
        self.facts = dict(self.facts, helper='second-model-helper')
        self.assertEqual(self.prepare(self.base / 'one')[0], 0)
        self.assertEqual(len(self.builds), 6)
        install = runner.install_root(FakeRun.identity())
        state = runner.read(install / 'install.json')
        self.assertIn('bin/second-model-helper', state['files'])
        self.assertEqual(state['build']['files'], state['files'])
        with patch.object(runner, 'checkout', lambda: self.base / 'one'):
            self.assertTrue(runner.built_here(install, state, FakeRun.identity(), Path(self.builds[-1])))

    def test_a_second_checkout_cannot_extend_the_first_checkout_s_install(self):
        # Regression: adding another model's helper kept
        # the first checkout's files and recorded them as built by the second.
        code, errors = self.extension(self.base / 'two')
        self.assertEqual(code, 2)
        self.assertIn('was not built from this checkout', errors)

    def test_an_install_without_build_provenance_cannot_be_extended(self):
        def legacy(install):
            state = runner.read(install / 'install.json')
            runner.write(install / 'install.json', {key: value for key, value in state.items() if key != 'build'})
        # Not even by the checkout that built it: its files have no record.
        code, errors = self.extension(self.base / 'one', legacy)
        self.assertEqual(code, 2)
        self.assertIn('was not built from this checkout', errors)

    def test_a_second_checkout_cannot_reuse_the_first_checkout_s_install(self):
        self.assertEqual(self.prepare(self.base / 'one')[0], 0)
        install = runner.install_root(FakeRun.identity())
        state = runner.read(install / 'install.json')
        self.assertEqual({key: state['build'][key] for key in ('commit', 'checkout')},
                         {'commit': FakeRun.identity()['commit'], 'checkout': str(self.base / 'one')})
        self.assertEqual(state['build']['files'], state['files'])
        # The same checkout and target reuse it without building again.
        self.assertEqual(self.prepare(self.base / 'one')[0], 0)
        self.assertEqual(len(self.builds), 3)
        code, errors = self.prepare(self.base / 'two')
        self.assertEqual(code, 2)
        self.assertIn('was not built from this checkout', errors)
        self.assertIn('move', errors)
        self.assertEqual(len(self.builds), 3)
        # Neither may an install without build provenance, or one whose files
        # changed since they were built, be reused.
        for change in (lambda: runner.write(install / 'install.json', {key: value for key, value in state.items()
                                                                       if key != 'build'}),
                       lambda: (runner.write(install / 'install.json', state),
                                (install / 'bin/chopinfer').write_text('replaced'))):
            change()
            with self.subTest(change=change):
                code, errors = self.prepare(self.base / 'one')
                self.assertEqual(code, 2)
                self.assertIn('was not built from this checkout', errors)


class TerminationTests(unittest.TestCase):
    def test_stop_records_actual_clean_and_forced_child_exits(self):
        for ignore in (False, True):
            script = ('import signal,sys,time; '
                      'signal.signal(signal.SIGTERM, signal.SIG_IGN if ' + repr(ignore) +
                      ' else lambda *_: sys.exit(0)); print("ready",flush=True); time.sleep(30)')
            process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE,
                                       text=True, start_new_session=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                stopped = runner.stop(process)
                self.assertEqual(stopped['returncode'], -9 if ignore else 0)
                self.assertEqual(stopped['forced'], ignore)
                self.assertTrue(stopped['confirmed'])
                self.assertEqual([stage['signal'] for stage in stopped['stages']],
                                 ['SIGTERM', 'SIGKILL'] if ignore else ['SIGTERM'])
            finally:
                if process.poll() is None:
                    process.kill(); process.wait(timeout=10)
                process.stdout.close()


def ready(root):
    """The surrogate is ready and its wrapper waits for it: GNU timeout 9.4
    loses a TERM that lands before its parent has recorded the forked child,
    which a server's startup outlasts."""
    pid = root / 'server.pid'
    try:
        return ('READY' in (root / 'server.log').read_text()
                and Path(f'/proc/{pid.read_text()}/wchan').read_text() == 'sigsuspend')
    except (FileNotFoundError, ValueError):
        return False


def reap(pid, seconds):
    """Reap `pid` once it is this process's child (a subreaper adopts
    orphans as they lose their parent); None when it is not reaped in time."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                return os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass
        time.sleep(.02)
    return None


class ServerSignalTests(unittest.TestCase):
    """Use GNU timeout and real signal handlers, including a second-TERM exit."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        source = cls.base / 'server.c'
        source.write_text(r'''#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
static volatile sig_atomic_t stopping = 0;
static void term(int sig) {
    (void)sig;
    if (stopping++) {
        const char msg[] = "SECOND_TERM_EARLY_EXIT\n";
        write(1, msg, sizeof(msg)-1);
        _exit(0);
    }
    const char msg[] = "FIRST_TERM\n";
    write(1, msg, sizeof(msg)-1);
}
static volatile sig_atomic_t witnessed = 0;
static void witness_term(int sig) { (void)sig; witnessed++; }
static void witness_report(int sig) { (void)sig; _exit(witnessed); }
static void *spin(void *unused) {
    volatile unsigned long count = 0;
    for (;;) count++;
    return unused;
}
static pid_t ignoring_child(void) {
    int ready[2]; if (pipe(ready)) exit(4);
    pid_t child = fork(); if (child < 0) exit(5);
    if (!child) {
        close(ready[0]); signal(SIGTERM, SIG_IGN);
        write(ready[1], "x", 1); close(ready[1]);
        return 0;
    }
    close(ready[1]); char byte; if (read(ready[0], &byte, 1) != 1) exit(6);
    close(ready[0]); printf("DESCENDANT %ld\n", (long)child); fflush(stdout);
    return child;
}
int main(void) {
    /* The surrogate takes its mode and cleanup delay from the environment
       and ignores the serve words it is started with. */
    const char *mode = getenv("PROBE_MODE"), *delay_ms = getenv("PROBE_DELAY_MS");
    if (!mode || !delay_ms) return 2;
    struct sigaction action = {0};
    action.sa_handler = term;
    action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, 0)) return 3;
    printf("SERVER %ld\n", (long)getpid()); fflush(stdout);
    alarm(25);
    if (!strcmp(mode, "hung") || !strcmp(mode, "orphan")) {
        if (!ignoring_child()) for (;;) pause();
    }
    /* A TERM-ignoring child and grandchild. */
    if (!strcmp(mode, "tree")) {
        if (!ignoring_child()) {
            if (!ignoring_child()) for (;;) pause();
            for (;;) pause();
        }
    }
    /* Busy worker threads take a signal at once, as a loaded server does.
       A witness in the server's group counts the TERMs a group forward
       sends it; the server reaps it and reports the count. */
    pid_t witness = 0;
    if (!strcmp(mode, "busy")) {
        int ready[2]; if (pipe(ready)) return 8;
        witness = fork(); if (witness < 0) return 9;
        if (!witness) {
            struct sigaction counting = {0};
            sigemptyset(&counting.sa_mask);
            counting.sa_handler = witness_term; sigaction(SIGTERM, &counting, 0);
            counting.sa_handler = witness_report; sigaction(SIGUSR1, &counting, 0);
            close(ready[0]); write(ready[1], "x", 1); close(ready[1]);
            for (;;) pause();
        }
        close(ready[1]); char byte; if (read(ready[0], &byte, 1) != 1) return 10;
        close(ready[0]);
        for (int i = 0; i < 3; i++) { pthread_t thread; if (pthread_create(&thread, 0, spin, 0)) return 7; }
    }
    puts("READY"); fflush(stdout);
    struct timespec tick = {0, 10000000};
    while (!stopping) nanosleep(&tick, 0);
    puts("DRAIN_ENTERED"); fflush(stdout);
    if (!strcmp(mode, "hung")) for (;;) pause();
    long ms = atol(delay_ms);
    struct timespec delay = {ms / 1000, ms % 1000 * 1000000};
    while (nanosleep(&delay, &delay)) {}
    if (witness) {
        int status;
        if (kill(witness, SIGUSR1) || waitpid(witness, &status, 0) != witness) return 11;
        printf("WITNESS_TERMS %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    puts("CLEANUP_COMPLETE"); fflush(stdout);
    return 0;
}
''')
        cls.probe = cls.base / 'server'
        subprocess.run(['gcc', '-std=c11', '-pthread', '-Wall', '-Wextra', '-Werror', str(source),
                        '-o', str(cls.probe)], check=True, capture_output=True, timeout=30)

    def run_server(self, mode='clean', delay_ms=250, vanish_before_kill=False, serve=()):
        # Adopt the surrogate's orphaned descendants so the test reaps them.
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            root = Path(temporary)
            context = self.context(root, mode, delay_ms)
            killpg = os.killpg
            reaped = set()
            def disappear(group, sig):
                if vanish_before_kill and sig == signal.SIGKILL:
                    # Remove and reap the real residual descendant after the
                    # successful group probe, before stop's KILL syscall.
                    killpg(group, sig)
                    for line in (root / 'server.log').read_text().splitlines():
                        if line.startswith('DESCENDANT '):
                            child, status = os.waitpid(int(line.split()[1]), 0)
                            self.assertEqual(os.waitstatus_to_exitcode(status), -signal.SIGKILL)
                            reaped.add(child)
                return killpg(group, sig)
            def health(url):
                deadline = time.monotonic() + 5
                while not ready(root):
                    self.assertLess(time.monotonic(), deadline, 'surrogate did not become ready')
                    time.sleep(.01)
                return {'data': [{'id': 'surrogate'}]} if url.endswith('/models') else {}
            try:
                with patch.object(runner, 'http', health), \
                        patch.object(runner, 'listener_owned', return_value=True), patch.object(os, 'killpg', disappear):
                    if mode in ('clean', 'busy'):
                        with runner.server(context, {'serve': list(serve)}, root):
                            started = time.monotonic()
                    else:
                        with self.assertRaisesRegex(ValueError, 'nonclean|forced'):
                            with runner.server(context, {'serve': list(serve)}, root):
                                started = time.monotonic()
                elapsed = time.monotonic() - started
                events = (root / 'server.log').read_text().splitlines()
                stopped = runner.read(root / 'shutdown.json')
                for line in events:
                    if line.startswith('DESCENDANT '):
                        pid = int(line.split()[1])
                        if pid in reaped:
                            continue
                        # This process adopts the descendant (it is a
                        # subreaper) only once its parent has exited.
                        self.assertEqual(reap(pid, 2), -signal.SIGKILL, 'owned descendant survived KILL')
                return stopped, events, elapsed
            finally:
                # Cleanup must also work on the red implementation.
                pid_path = root / 'server.pid'
                if pid_path.exists():
                    try:
                        os.killpg(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                for line in (root / 'server.log').read_text().splitlines():
                    if line.startswith('DESCENDANT '):
                        try:
                            os.waitpid(int(line.split()[1]), 0)
                        except ChildProcessError:
                            pass
                # The backstop, adopted when its wrapper exited, exits once the
                # group is empty; reap it after the test.
                if (root / 'backstop.pid').exists():
                    self.addCleanup(reap, int((root / 'backstop.pid').read_text()), 5)
                self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)

    def context(self, root, mode, delay_ms):
        (root / 'bin').mkdir()
        (root / 'bin/chopinfer').symlink_to(self.probe)
        return {'fill': {}, 'install': root, 'checkpoint': root, 'env': dict(os.environ),
                'facts': {'name': 'surrogate'},
                'config': {'environment': {'PROBE_MODE': mode, 'PROBE_DELAY_MS': str(delay_ms)}}}

    def test_server_sends_one_term_and_waits_for_cleanup(self):
        for _ in range(3):
            with self.subTest(trial=_):
                stopped, events, elapsed = self.run_server()
                self.assertEqual(events.count('FIRST_TERM'), 1)
                self.assertNotIn('SECOND_TERM_EARLY_EXIT', events)
                self.assertIn('CLEANUP_COMPLETE', events)
                self.assertGreaterEqual(elapsed, .25)
                self.assertEqual(stopped['returncode'], 0)
                self.assertFalse(stopped['termination']['forced'])

    def test_server_waits_for_slow_cleanup_without_early_exit(self):
        stopped, events, elapsed = self.run_server(delay_ms=750)
        self.assertIn('CLEANUP_COMPLETE', events)
        self.assertNotIn('SECOND_TERM_EARLY_EXIT', events)
        self.assertGreaterEqual(elapsed, .75)
        self.assertLess(elapsed, 5)
        self.assertTrue(stopped['termination']['confirmed'])

    def test_hung_server_and_descendant_get_group_kill_after_its_drain_and_margin(self):
        # A zero drain keeps this test at the 10 s cleanup margin.
        stopped, events, elapsed = self.run_server('hung', serve=['--drain-timeout-ms', '0'])
        self.assertGreaterEqual(elapsed, 10)
        self.assertLess(elapsed, 20)
        self.assertEqual(stopped['returncode'], -signal.SIGKILL)
        self.assertTrue(stopped['termination']['forced'])
        self.assertEqual([s['wait_seconds'] for s in stopped['termination']['stages']], [10, 10])
        self.assertNotIn('CLEANUP_COMPLETE', events)

    def test_wrapper_exit_does_not_hide_surviving_descendant(self):
        stopped, events, elapsed = self.run_server('orphan')
        self.assertIn('CLEANUP_COMPLETE', events)
        self.assertEqual(stopped['returncode'], 0)
        self.assertTrue(stopped['termination']['forced'])
        self.assertLess(elapsed, 15)

    def test_residual_group_failure_survives_esrch_before_kill(self):
        stopped, events, elapsed = self.run_server('orphan', vanish_before_kill=True)
        self.assertEqual(stopped['returncode'], 0)
        self.assertTrue(stopped['termination']['confirmed'])
        self.assertTrue(stopped['termination']['forced'])
        term, kill = stopped['termination']['stages']
        self.assertTrue(term['group_remaining'])
        self.assertTrue(term['sent'])
        self.assertFalse(kill['sent'])
        # The default 30 s drain plus the 10 s margin, then the 10 s KILL wait.
        self.assertEqual([term['wait_seconds'], kill['wait_seconds']], [40, 10])
        self.assertIn('CLEANUP_COMPLETE', events)
        self.assertLess(elapsed, 15)

    def test_recorded_residual_group_cannot_be_hidden_by_forced_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner.write(directory / 'health.json', {'process_group': 4242})
            runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': 0,
                'termination': {'returncode': 0, 'confirmed': True, 'forced': False,
                                'stages': [{'signal': 'SIGTERM', 'group_remaining': True},
                                           {'signal': 'SIGKILL', 'sent': False}]}})
            with self.assertRaisesRegex(ValueError, 'residual group'):
                runner.require_shutdown(directory)

    def test_utility_keeps_group_term_route(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(runner.timeout_command([self.probe], 30), stdout=log, stderr=log,
                                       env={**os.environ, 'PROBE_MODE': 'clean', 'PROBE_DELAY_MS': '250'},
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while True:
                    log.seek(0)
                    if 'READY' in log.read():
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                with patch.object(os, 'killpg', wraps=os.killpg) as kill:
                    stopped = runner.stop(process)
                    self.assertEqual(kill.call_args_list[0].args, (process.pid, signal.SIGTERM))
                # Utilities and workers keep the 5 s TERM wait and 10 s wrapper KILL.
                self.assertEqual([stage['wait_seconds'] for stage in stopped['stages']], [5])
                self.assertIn('--kill-after=10s', process.args)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)

    def test_server_takes_one_term_under_busy_worker_threads(self):
        # Without --foreground, GNU timeout forwards the stop's TERM to the
        # server and again to its group: the witness takes one every time,
        # and busy threads took both in 1 to 12 trials in 100, the second
        # exiting 0 without its drain.
        doubled = 0
        for _ in range(100):
            stopped, events, elapsed = self.run_server('busy', delay_ms=20)
            doubled += 'SECOND_TERM_EARLY_EXIT' in events
            self.assertIn('WITNESS_TERMS 0', events)
            self.assertEqual(events.count('FIRST_TERM'), 1)
            self.assertEqual(stopped['returncode'], 0)
            self.assertFalse(stopped['termination']['forced'])
        self.assertEqual(doubled, 0)

    # An owner process that runs server() with the surrogate, so a test can
    # kill, hang or pause it: argv is the tools/qual directory, the arm
    # directory and a JSON object with the surrogate's mode and cleanup delay,
    # the wrapper's time limit in seconds (0 keeps the runner's), the backstop's
    # KILL delay (shortened here) and what the owner does: `stop` stops at
    # once, `hold` hangs while serving, `pause` stops itself (SIGSTOP) as its
    # stop starts.
    OWNER = r"""
import json, os, signal, sys, time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import release_preview as runner
root, options = Path(sys.argv[2]), json.loads(sys.argv[3])
make, stop = runner.timeout_command, runner.stop
def command(words, seconds, *args, **keywords):
    return make(words, options['expire'] or seconds, *args, **keywords)
def paused(*args, **keywords):
    print('STOPPING', flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
    return stop(*args, **keywords)
def health(url):
    while not ('READY' in (root / 'server.log').read_text()
               and Path(f'/proc/{(root / "server.pid").read_text()}/wchan').read_text() == 'sigsuspend'):
        time.sleep(.01)
    return {'data': [{'id': 'surrogate'}]} if url.endswith('/models') else {}
context = {'fill': {}, 'install': root, 'checkpoint': root, 'env': dict(os.environ), 'facts': {'name': 'surrogate'},
           'config': {'environment': {'PROBE_MODE': options['mode'], 'PROBE_DELAY_MS': str(options['delay_ms'])}}}
with patch.object(runner, 'timeout_command', command), patch.object(runner, 'http', health), \
        patch.object(runner, 'listener_owned', return_value=True), \
        patch.object(runner, 'stop', paused if options['behavior'] == 'pause' else stop), \
        patch.object(runner, 'server_backstop', lambda drain: {'kill_after_seconds': options['bound']}, create=True):
    with runner.server(context, {'serve': ['--drain-timeout-ms', '0']}, root):
        print('SERVING', flush=True)
        if options['behavior'] == 'hold':
            time.sleep(600)
"""

    def run_owner(self, mode, act, expire=0, delay_ms=250, bound=2, behavior='stop'):
        """Run OWNER as a child, adopting the server's processes as they are
        orphaned; `act(root, owner)` drives it and returns the descendants'
        exit codes. Then kills the owner if it still runs and reaps the
        wrapper and backstop. Returns the descendant, wrapper, backstop and
        owner exit codes, whether the group is gone and the server log."""
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            root = Path(temporary)
            self.context(root, mode, delay_ms)
            group = None
            options = {'mode': mode, 'expire': expire, 'delay_ms': delay_ms, 'bound': bound, 'behavior': behavior}
            with (root / 'owner.log').open('w') as log:
                owner = subprocess.Popen([sys.executable, '-I', '-B', '-c', self.OWNER, str(Path(runner.__file__).parent),
                                          str(root), json.dumps(options)], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                while not (root / 'server.pid').exists():
                    self.assertLess(time.monotonic(), deadline, 'the owner did not start its server')
                    time.sleep(.01)
                time.sleep(.05)
                group = int((root / 'server.pid').read_text())
                codes = act(root, owner)
                if owner.poll() is None:
                    owner.kill()
                owner.wait(timeout=10)
                # A living owner reaped its wrapper; a killed one left it here.
                wrapper = reap(group, 5 if owner.returncode == -signal.SIGKILL else .5)
                # A server the wrapper did not reap is adopted here.
                reap(self.pids(root, 'SERVER')[0], .5)
                watchdog = reap(int((root / 'backstop.pid').read_text()), 5) \
                    if (root / 'backstop.pid').exists() else None
                try:
                    os.killpg(group, 0)
                    gone = False
                except ProcessLookupError:
                    gone = True
                return SimpleNamespace(codes=codes, wrapper=wrapper, watchdog=watchdog, owner=owner.returncode,
                                       gone=gone, events=(root / 'server.log').read_text().splitlines())
            finally:
                # Cleanup must also work on the red implementation.
                if owner.poll() is None:
                    owner.send_signal(signal.SIGCONT)
                    owner.kill()
                    owner.wait(timeout=10)
                if group is not None:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    reap(group, .5)
                for pid in self.pids(root, 'SERVER') + self.pids(root):
                    reap(pid, .5)
                self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)

    def pids(self, root, event='DESCENDANT'):
        return [int(line.split()[1]) for line in (root / 'server.log').read_text().splitlines()
                if line.startswith(event + ' ')]

    def wait_for(self, root, event, seconds):
        deadline = time.monotonic() + seconds
        while event not in (root / 'server.log').read_text().splitlines():
            self.assertLess(time.monotonic(), deadline, f'no {event}')
            time.sleep(.01)

    def test_backstop_kills_the_group_when_the_owner_dies_mid_stop(self):
        def act(root, owner):
            # The stop's TERM put the server in its drain, which hangs, with
            # a TERM-ignoring descendant; its owner dies before any KILL.
            self.wait_for(root, 'DRAIN_ENTERED', 10)
            owner.kill()
            owner.wait(timeout=10)
            return [reap(pid, 6) for pid in self.pids(root)]
        run = self.run_owner('hung', act)
        self.assertEqual(run.codes, [-signal.SIGKILL])
        self.assertTrue(any(line.startswith('backstop: KILL process group') for line in run.events))
        self.assertNotIn('CLEANUP_COMPLETE', run.events)
        self.assertEqual(run.wrapper, -signal.SIGKILL)
        self.assertEqual(run.watchdog, 0)
        self.assertTrue(run.gone)

    def test_backstop_kills_survivors_of_a_foreground_expiry(self):
        def act(root, owner):
            # The wrapper's 2 s expiry TERMs the server alone, which exits 0
            # and leaves a TERM-ignoring child and grandchild; the owner hangs.
            self.wait_for(root, 'FIRST_TERM', 10)
            codes = [reap(pid, 8) for pid in self.pids(root)]
            self.assertIsNone(owner.poll())
            return codes
        run = self.run_owner('tree', act, expire=2, behavior='hold')
        self.assertEqual(run.codes, [-signal.SIGKILL, -signal.SIGKILL])
        self.assertIn('CLEANUP_COMPLETE', run.events)
        self.assertNotIn('SECOND_TERM_EARLY_EXIT', run.events)
        self.assertTrue(any(line.startswith('backstop: KILL process group') for line in run.events))
        self.assertEqual(run.wrapper, 124)
        self.assertEqual(run.watchdog, 0)
        self.assertTrue(run.gone)

    def test_backstop_clock_starts_at_the_term_after_a_paused_stop(self):
        # The owner pauses as its stop starts, longer than most of the
        # backstop's delay (shortened to 4 s), then resumes. The 2 s drain
        # still completes: the delay runs from the one TERM.
        def act(root, owner):
            deadline = time.monotonic() + 10
            while 'STOPPING' not in (root / 'owner.log').read_text():
                self.assertLess(time.monotonic(), deadline, 'the owner did not start its stop')
                time.sleep(.01)
            time.sleep(3)
            owner.send_signal(signal.SIGCONT)
            owner.wait(timeout=20)
            return []
        run = self.run_owner('clean', act, delay_ms=2000, bound=4, behavior='pause')
        self.assertEqual(run.events.count('FIRST_TERM'), 1)
        self.assertIn('CLEANUP_COMPLETE', run.events)
        self.assertFalse(any(line.startswith('backstop: KILL') for line in run.events))
        self.assertEqual(run.owner, 0)
        self.assertEqual(run.watchdog, 0)
        self.assertTrue(run.gone)

    def test_backstop_stops_the_server_with_one_term_when_its_owner_dies(self):
        # The owner dies while serving, before any stop: the backstop sends the
        # one TERM itself and the server drains and exits cleanly.
        def act(root, owner):
            deadline = time.monotonic() + 10
            while 'SERVING' not in (root / 'owner.log').read_text():
                self.assertLess(time.monotonic(), deadline, 'the owner did not serve')
                time.sleep(.01)
            owner.kill()
            owner.wait(timeout=10)
            return []
        run = self.run_owner('clean', act, behavior='hold')
        self.assertEqual(run.events.count('FIRST_TERM'), 1)
        self.assertIn('CLEANUP_COMPLETE', run.events)
        self.assertNotIn('SECOND_TERM_EARLY_EXIT', run.events)
        self.assertFalse(any(line.startswith('backstop: KILL') for line in run.events))
        self.assertEqual(run.wrapper, 0)
        self.assertEqual(run.watchdog, 0)
        self.assertTrue(run.gone)

    def test_backstop_kills_even_when_its_log_write_fails(self):
        # The watchdog's stderr is full (ENOSPC); its owner is gone, the server
        # hangs in its drain and a TERM-ignoring descendant remains.
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        with tempfile.TemporaryDirectory(dir=self.base) as temporary, open('/dev/full', 'w') as full:
            root = Path(temporary)
            wrapper = runner.timeout_command([self.probe], 600, 45, foreground=True)
            with (root / 'server.log').open('w') as log:
                process, link = runner.backstopped(
                    wrapper, 1, stdout=log, stderr=full,
                    env={**os.environ, 'PROBE_MODE': 'hung', 'PROBE_DELAY_MS': '0'})
            (root / 'server.pid').write_text(str(process.pid))
            try:
                watchdog = runner.backstop_pid(link)
                deadline = time.monotonic() + 5
                while not ready(root):
                    self.assertLess(time.monotonic(), deadline, 'surrogate did not become ready')
                    time.sleep(.01)
                link.close()
                codes = [reap(pid, 6) for pid in self.pids(root)]
                self.assertEqual(codes, [-signal.SIGKILL])
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                # The server, KILLed with its wrapper, is adopted here.
                self.assertEqual(reap(self.pids(root, 'SERVER')[0], 2), -signal.SIGKILL)
                self.assertEqual(reap(watchdog, 5), 0)
            finally:
                link.close()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
                for pid in self.pids(root, 'SERVER') + self.pids(root):
                    reap(pid, .5)
                self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)


class ServerDrainTests(unittest.TestCase):
    """The server stop honors the drain its serve words declare: a real
    listener under GNU timeout, started and stopped through server()."""

    SURROGATE = r"""
import http.server, json, os, signal, sys, threading, time
words = sys.argv[1:]
def value(flag, default=None):
    found = [index for index, word in enumerate(words) if word == flag]
    if len(found) > 1:
        sys.exit(2)
    return words[found[0] + 1] if found else default
host, port = value('--listen').rsplit(':', 1)
name = value('--served-model-name')
drain_ms = int(value('--drain-timeout-ms', '30000'))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'data': [{'id': name}]} if self.path == '/v1/models' else {}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_):
        pass
listener = http.server.ThreadingHTTPServer((host, int(port)), Handler)
threading.Thread(target=listener.serve_forever, daemon=True).start()
terms = []
def term(*_):
    terms.append(1)
    if len(terms) > 1:
        print('SECOND_TERM_EARLY_EXIT', flush=True)
        os._exit(0)
signal.signal(signal.SIGTERM, term)
print('READY', flush=True)
while not terms:
    time.sleep(.01)
print(f'DRAIN {drain_ms}', flush=True)
# In-flight work keeps the listener up until it finishes or the drain ends.
time.sleep(min(int(os.environ['SURROGATE_BUSY_MS']), drain_ms) / 1000)
listener.shutdown()
print('LISTENER_CLOSED', flush=True)
# Then teardown after the drain: runtime shutdown, worker join, engine drop.
time.sleep(int(os.environ['SURROGATE_CLEANUP_MS']) / 1000)
print('CLEANUP_COMPLETE', flush=True)
"""

    def run_server(self, serve, busy_ms, cleanup_ms=0):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'install/bin/chopinfer'
            executable.parent.mkdir(parents=True)
            executable.write_text(f'#!{sys.executable} -I\n{self.SURROGATE}')
            executable.chmod(0o755)
            context = {'fill': {}, 'install': root / 'install', 'checkpoint': root, 'env': dict(os.environ),
                       'facts': {'name': 'surrogate'}, 'config': {'environment': {'SURROGATE_BUSY_MS': str(busy_ms),
                                                                 'SURROGATE_CLEANUP_MS': str(cleanup_ms)}}}
            directory = root / 'arm'
            directory.mkdir()
            try:
                with patch.dict(os.environ, {'no_proxy': '127.0.0.1', 'NO_PROXY': '127.0.0.1'}):
                    try:
                        with runner.server(context, {'serve': serve}, directory) as (_url, pid):
                            wrapper = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
                            started = time.monotonic()
                        error = None
                    except ValueError as raised:
                        error = raised
                elapsed = time.monotonic() - started
                return (runner.read(directory / 'launch.json'), runner.read(directory / 'shutdown.json'),
                        (directory / 'server.log').read_text().splitlines(), wrapper, elapsed, error)
            finally:
                # Cleanup must also work on the red implementation.
                try:
                    os.killpg(int((directory / 'server.pid').read_text()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass

    def test_server_waits_its_declared_drain_before_group_kill(self):
        # 6 s of work under a declared 7 s drain outlasts the utilities' 5 s TERM wait.
        launch, stopped, events, wrapper, elapsed, error = self.run_server(['--drain-timeout-ms', '7000'], 6000)
        self.assertIsNone(error)
        drain = {'source': 'serve', 'milliseconds': 7000, 'term_wait_seconds': 17, 'wrapper_kill_after_seconds': 22}
        self.assertEqual(launch['drain'], drain)
        self.assertEqual(stopped['termination']['drain'], drain)
        self.assertEqual(stopped['returncode'], 0)
        self.assertFalse(stopped['termination']['forced'])
        self.assertEqual(stopped['termination']['stages'], [{'signal': 'SIGTERM', 'wait_seconds': 17, 'sent': True,
                                                             'timed_out': False, 'target': 'timeout_process',
                                                             'via': 'backstop'}])
        # The surrogate read the drain the runner derived, from the same argv.
        self.assertIn('DRAIN 7000', events)
        self.assertIn('CLEANUP_COMPLETE', events)
        self.assertNotIn('SECOND_TERM_EARLY_EXIT', events)
        self.assertGreaterEqual(elapsed, 6)
        self.assertLess(elapsed, 17)
        # The wrapper's own KILL comes after the graceful wait.
        self.assertEqual(wrapper[1:5], ['--foreground', '--signal=TERM', '--kill-after=22s', '0s'])
        self.assertEqual(wrapper[wrapper.index('--drain-timeout-ms') + 1], '7000')

    def test_server_without_drain_flag_waits_the_native_default(self):
        launch, stopped, events, wrapper, elapsed, error = self.run_server([], 250)
        self.assertIsNone(error)
        drain = {'source': 'default', 'milliseconds': 30000, 'term_wait_seconds': 40,
                 'wrapper_kill_after_seconds': 45}
        self.assertEqual(stopped['termination']['drain'], drain)
        self.assertEqual([stage['wait_seconds'] for stage in stopped['termination']['stages']], [40])
        self.assertIn('DRAIN 30000', events)
        self.assertIn('--kill-after=45s', wrapper)
        self.assertNotIn('--drain-timeout-ms', wrapper)
        self.assertLess(elapsed, 5)

    def test_cleanup_after_a_full_drain_fits_the_margin(self):
        # A 1 s drain used in full, then 7 s of teardown: past the old 5 s
        # margin, within the 10 s one.
        launch, stopped, events, wrapper, elapsed, error = self.run_server(['--drain-timeout-ms', '1000'], 1000,
                                                                           7000)
        self.assertIsNone(error)
        self.assertEqual(stopped['termination']['drain']['term_wait_seconds'], 11)
        self.assertEqual(stopped['returncode'], 0)
        self.assertFalse(stopped['termination']['forced'])
        self.assertEqual([stage['wait_seconds'] for stage in stopped['termination']['stages']], [11])
        self.assertEqual(events[-3:], ['DRAIN 1000', 'LISTENER_CLOSED', 'CLEANUP_COMPLETE'])
        self.assertGreaterEqual(elapsed, 8)
        self.assertLess(elapsed, 11)
        self.assertIn('--kill-after=16s', wrapper)

    def test_cleanup_past_the_margin_is_a_forced_failure(self):
        launch, stopped, events, wrapper, elapsed, error = self.run_server(['--drain-timeout-ms', '0'], 0, 30000)
        self.assertRegex(str(error), 'nonclean')
        self.assertEqual(stopped['returncode'], -signal.SIGKILL)
        self.assertTrue(stopped['termination']['forced'])
        term, kill = stopped['termination']['stages']
        self.assertEqual([term['wait_seconds'], term['timed_out'], kill['sent']], [10, True, True])
        self.assertIn('LISTENER_CLOSED', events)
        self.assertNotIn('CLEANUP_COMPLETE', events)
        self.assertGreaterEqual(elapsed, 10)
        self.assertLess(elapsed, 15)

    def test_only_the_server_wrapper_runs_in_the_foreground(self):
        launch, stopped, events, wrapper, elapsed, error = self.run_server([], 250)
        self.assertIsNone(error)
        # The recorded wrapper is the argv the server ran under, with its armed backstop.
        self.assertIn('wrapper', launch)
        self.assertEqual(launch['wrapper'], wrapper)
        self.assertEqual(wrapper[1:4], ['--foreground', '--signal=TERM', '--kill-after=45s'])
        self.assertEqual(wrapper[5:], launch['command'])
        self.assertEqual(launch['backstop'], {'kill_after_seconds': 55})
        backstop = stopped['termination']['backstop']
        self.assertEqual(backstop, {'kill_after_seconds': 55, 'pid': backstop['pid'], 'armed': True})
        # A utility's expiry keeps signaling its whole group.
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(subprocess, 'Popen', wraps=subprocess.Popen) as popen:
            self.assertEqual(runner.call(['true'], Path(temporary) / 'log', 30, dict(os.environ)), 0)
        utility = popen.call_args.args[0]
        self.assertEqual(utility[1:4], ['--signal=TERM', '--kill-after=10s', '30s'])
        self.assertNotIn('--foreground', utility)

    def test_server_wrapper_has_no_time_limit_of_its_own(self):
        # GNU timeout's own expiry would TERM serve independently of the
        # backstop, which could then send a second TERM. Duration 0 disables
        # it; the KILL delay still follows a forwarded TERM.
        launch, stopped, events, wrapper, elapsed, error = self.run_server([], 250)
        self.assertIsNone(error)
        self.assertEqual(wrapper[4], '0s')
        ignoring = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("R", flush=True); ' \
                   'time.sleep(30)'
        process = subprocess.Popen([*wrapper[:3], '--kill-after=1s', wrapper[4], sys.executable, '-c', ignoring],
                                   stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'R')
            deadline = time.monotonic() + 5
            while Path(f'/proc/{process.pid}/wchan').read_text() != 'sigsuspend':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            started = time.monotonic()
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=10), 128 + signal.SIGKILL)
            self.assertLess(time.monotonic() - started, 5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            process.stdout.close()

    def test_drain_is_read_at_serve_option_positions(self):
        drain = '--drain-timeout-ms'
        default = {'source': 'default', 'milliseconds': 30000, 'term_wait_seconds': 40,
                   'wrapper_kill_after_seconds': 45}
        self.assertEqual(runner.server_drain(['--speculative-tokens', '0']), default)
        # An option's value is never read as an option, even when it looks like one.
        self.assertEqual(runner.server_drain(['--api-key', '--profile']), default)
        self.assertEqual(runner.server_drain(['--api-key', drain]), default)
        self.assertEqual(runner.server_drain(['--api-key', drain, drain, '1000'])['milliseconds'], 1000)
        # --served-model-name takes every word up to the next option; valueless flags take none.
        self.assertEqual(runner.server_drain(['--served-model-name', 'a', 'b', drain, '7'])['milliseconds'], 7)
        self.assertEqual(runner.server_drain(['--allow-credentials', drain, '7'])['milliseconds'], 7)
        self.assertEqual(runner.server_drain(['--model', 'm', '--served-model-name', 'a', '7'])['source'],
                         'default')
        for arguments in ([drain, '1000', drain, '1000'], [drain], ['--api-key', 'k', drain],
                          [drain, '--max-sequences', '4'], ['--drain-timeout-ms=1000']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                runner.server_drain(arguments)

    def test_drain_value_uses_native_u64_syntax_without_a_runner_cap(self):
        drain = '--drain-timeout-ms'
        for value, milliseconds, wait in (('0', 0, 10), ('1', 1, 11), ('1000', 1000, 11), ('1001', 1001, 12),
                                          ('+1000', 1000, 11), ('0600001', 600001, 611),
                                          (str(2 ** 64 - 1), 2 ** 64 - 1, 18446744073709552 + 10)):
            with self.subTest(value=value):
                self.assertEqual(runner.server_drain([drain, value]),
                                 {'source': 'serve', 'milliseconds': milliseconds, 'term_wait_seconds': wait,
                                  'wrapper_kill_after_seconds': wait + 5})
        for value in (str(2 ** 64), '-1', '-0', '+', '', '1e3', '1.5', ' 1000', '1000 ', '1_000', '0x10',
                      '\u0661\u0662'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'unsigned 64-bit'):
                runner.server_drain([drain, value])

    def test_serve_option_facts_match_runtime_host(self):
        source = (runner.ROOT / 'crates/runtime-host/src/command.rs').read_text()
        valueless = re.search(r'VALUELESS_FLAGS_V1: &\[&str\] = &\[(.*?)\];', source, re.S).group(1)
        self.assertEqual(tuple(re.findall(r'"(--[a-z-]+)"', valueless)), runner.VALUELESS_SERVE_FLAGS)
        self.assertIn('drain_timeout_ms: parsed.drain_timeout_ms.unwrap_or(30_000)', source)
        self.assertEqual(runner.DRAIN_DEFAULT_MS, 30000)

    def test_refused_drain_starts_no_server(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(subprocess, 'Popen', side_effect=AssertionError('started')):
            root = Path(temporary)
            context = {'fill': {}, 'install': root, 'checkpoint': root, 'env': {},
                       'facts': {'name': 'surrogate'}, 'config': {'environment': {}}}
            with self.assertRaisesRegex(ValueError, 'unsigned 64-bit'):
                with runner.server(context, {'serve': ['--drain-timeout-ms', str(2 ** 64)]}, root):
                    pass
            self.assertFalse((root / 'launch.json').exists())
            self.assertFalse((root / 'server.pid').exists())


class LateExitTests(unittest.TestCase):
    """An exit that lands after the TERM wait expired, before a KILL can
    reach the group, is not a clean stop."""

    class LateZero:
        pid = 4242
        returncode = None

        def __init__(self):
            self.waits = []

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            pass

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired('server', timeout)
            self.returncode = 0
            return 0

    COMMAND = ['/install/bin/chopinfer', 'serve', '--model', '/checkpoint']

    def write_record(self, directory, termination, launch=None):
        # termination None writes a legacy exit record; launch None a legacy launch.
        runner.write(directory / 'health.json', {'process_group': 4242})
        runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': 0,
                                                   **({} if termination is None else {'termination': termination})})
        runner.write(directory / 'launch.json', {'command': self.COMMAND, 'url': 'http://127.0.0.1:1'}
                     if launch is None else launch)

    def test_stop_marks_late_zero_after_expired_term_forced(self):
        process = self.LateZero()
        def gone(_group, sig):
            raise ProcessLookupError(sig)
        with patch.object(os, 'killpg', gone):
            stopped = runner.stop(process, timeout_forwarder=True)
        self.assertEqual(process.waits, [5, 10])
        self.assertEqual(stopped['returncode'], 0)
        term, kill = stopped['stages']
        self.assertTrue(term['timed_out'])
        self.assertFalse(kill['sent'])
        self.assertTrue(stopped['forced'])

    def test_recorded_late_zero_is_refused_even_when_marked_unforced(self):
        # The record the earlier stop wrote for this race: zero, unforced.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_record(directory, {'returncode': 0, 'confirmed': True, 'forced': False, 'stages': [
                {'signal': 'SIGTERM', 'wait_seconds': 5, 'sent': True, 'timed_out': True, 'target': 'timeout_process'},
                {'signal': 'SIGKILL', 'wait_seconds': 10, 'sent': False, 'timed_out': False}]})
            with self.assertRaisesRegex(ValueError, 'expired TERM wait'):
                runner.require_shutdown(directory)

    def records(self):
        """A clean drain stop as server() records it: (launch, termination)."""
        drain = runner.server_drain([])
        command = self.COMMAND
        backstop = {'kill_after_seconds': 55}
        launch = {'command': command, 'url': 'http://127.0.0.1:1', 'drain': drain, 'backstop': backstop,
                  'wrapper': ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=45s', '0s',
                              *command]}
        stage = {'signal': 'SIGTERM', 'wait_seconds': drain['term_wait_seconds'], 'sent': True, 'timed_out': False,
                 'target': 'timeout_process', 'via': 'backstop'}
        clean = {'returncode': 0, 'confirmed': True, 'forced': False, 'stages': [stage], 'drain': drain,
                 'backstop': {**backstop, 'pid': 4243, 'armed': True}}
        return launch, clean

    def test_drain_records_bind_their_wait_to_the_recorded_serve_words(self):
        drain = runner.server_drain([])
        command = self.COMMAND
        launch, clean = self.records()
        stage = clean['stages'][0]
        unmarked = {key: value for key, value in clean.items() if key != 'drain'}
        legacy_launch = {key: value for key, value in launch.items() if key != 'drain'}
        cases = {
            'admitted': (clean, launch, None),
            # Either record's drain marker makes it a drain record, needing both.
            'termination marker removed': ({**unmarked, 'stages': [{**stage, 'wait_seconds': 5}]}, launch,
                                           'without its launch or termination'),
            'termination marker removed, wait kept': (unmarked, launch, 'without its launch or termination'),
            'termination removed': (None, launch, 'without its launch or termination'),
            'launch marker removed': (clean, legacy_launch, 'without its launch or termination'),
            'short wait': ({**clean, 'stages': [{**stage, 'wait_seconds': 5}]}, launch, 'wait its declared drain'),
            'unsent': ({**clean, 'stages': [{**stage, 'sent': False}]}, launch, 'wait its declared drain'),
            'no stages': ({**clean, 'stages': []}, launch, 'wait its declared drain'),
            'other target': ({**clean, 'stages': [{**stage, 'target': 'group'}]}, launch,
                             'wait its declared drain'),
            'other argv': (clean, {**launch, 'command': [*command, '--drain-timeout-ms', '0']},
                           'differs from its recorded serve'),
            'other launch drain': (clean, {**launch, 'drain': runner.server_drain(['--drain-timeout-ms', '0'])},
                                   'differs from its recorded serve'),
            'other termination drain': ({**clean, 'drain': runner.server_drain(['--drain-timeout-ms', '0'])},
                                        launch, 'differs from its recorded serve'),
            'not serve': (clean, {**launch, 'command': [command[0], 'run', *command[2:]]},
                          'differs from its recorded serve'),
        }
        for name, (termination, recorded, refusal) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.write_record(directory, termination, recorded)
                if refusal is None:
                    runner.require_shutdown(directory)
                else:
                    with self.assertRaisesRegex(ValueError, refusal):
                        runner.require_shutdown(directory)

    def test_server_stop_needs_a_foreground_wrapper_and_an_armed_backstop(self):
        launch, clean = self.records()
        wrapper, command = launch['wrapper'], launch['command']
        def without(record, *keys):
            return {key: value for key, value in record.items() if key not in keys}
        cases = {
            'admitted': (clean, launch, None),
            # The record the reviewed server stop of 2026-10-13 wrote.
            'no wrapper or backstop': (without(clean, 'backstop'), without(launch, 'wrapper', 'backstop'),
                                       'server wrapper'),
            'no foreground': (clean, {**launch, 'wrapper': [wrapper[0], *wrapper[2:]]}, 'server wrapper'),
            'no wrapper': (clean, without(launch, 'wrapper'), 'server wrapper'),
            'foreground elsewhere': (clean, {**launch, 'wrapper': [wrapper[0], *wrapper[2:5], '--foreground',
                                                                   *command]}, 'server wrapper'),
            'other KILL delay': (clean, {**launch, 'wrapper': [*wrapper[:3], '--kill-after=10s', *wrapper[4:]]},
                                 'server wrapper'),
            'other wrapped command': (clean, {**launch, 'wrapper': [*wrapper[:5], *command, '--drain-timeout-ms',
                                                                    '0']}, 'server wrapper'),
            'no duration word': (clean, {**launch, 'wrapper': [*wrapper[:4], *command]}, 'server wrapper'),
            # The wrapper's own expiry would be a second TERM source.
            'old 604800 s limit': (clean, {**launch, 'wrapper': [*wrapper[:4], '604800s', *command]},
                                   'server wrapper'),
            'not timeout': (clean, {**launch, 'wrapper': ['/usr/bin/env', *wrapper[1:]]}, 'server wrapper'),
            'no launch backstop': (clean, without(launch, 'backstop'), 'backstop'),
            'no stop backstop': (without(clean, 'backstop'), launch, 'backstop'),
            'backstop never started': ({**clean, 'backstop': {**clean['backstop'], 'pid': None}}, launch, 'backstop'),
            'unarmed backstop': ({**clean, 'backstop': {**clean['backstop'], 'armed': False}}, launch, 'backstop'),
            # JSON 1 and 1.0 equal True in Python; only true is armed.
            'armed as 1': ({**clean, 'backstop': {**clean['backstop'], 'armed': 1}}, launch, 'backstop'),
            'armed as 1.0': ({**clean, 'backstop': {**clean['backstop'], 'armed': 1.0}}, launch, 'backstop'),
            # The watchdog is forked; the wrapper and group leader is not it.
            'watchdog is the wrapper': ({**clean, 'backstop': {**clean['backstop'], 'pid': 4242}}, launch,
                                        'backstop'),
            'TERM not sent by the backstop': ({**clean, 'stages': [{key: value for key, value in
                                                                     clean['stages'][0].items() if key != 'via'}]},
                                              launch, 'send one TERM'),
            # Durations are JSON integers.
            'launch delay 55.0': (clean, {**launch, 'backstop': {'kill_after_seconds': 55.0}}, 'backstop'),
            'stop delay 55.0': ({**clean, 'backstop': {**clean['backstop'], 'kill_after_seconds': 55.0}}, launch,
                                'backstop'),
            'TERM wait 40.0': ({**clean, 'stages': [{**clean['stages'][0], 'wait_seconds': 40.0}]}, launch,
                               'send one TERM'),
            'shorter backstop': ({**clean, 'backstop': {**clean['backstop'], 'kill_after_seconds': 5}},
                                 {**launch, 'backstop': {'kill_after_seconds': 5}}, 'backstop'),
            # A wrapper or backstop marker alone still makes a drain record.
            'drain markers removed': (without(clean, 'drain'), without(launch, 'drain'),
                                      'without its launch or termination'),
        }
        for name, (termination, recorded, refusal) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.write_record(directory, termination, recorded)
                if refusal is None:
                    runner.require_shutdown(directory)
                else:
                    with self.assertRaisesRegex(ValueError, refusal):
                        runner.require_shutdown(directory)

    def test_legacy_records_keep_their_verdicts(self):
        # Neither the launch nor the stop records a drain.
        legacy = {'returncode': 0, 'confirmed': True, 'forced': False, 'stages': [
            {'signal': 'SIGTERM', 'wait_seconds': 5, 'sent': True, 'timed_out': False, 'target': 'timeout_process'}]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_record(directory, legacy)
            runner.require_shutdown(directory)
            self.write_record(directory, None)
            runner.require_shutdown(directory)
            runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': -9, 'termination': {
                **legacy, 'returncode': -9, 'forced': True}})
            with self.assertRaisesRegex(ValueError, 'nonclean'):
                runner.require_shutdown(directory)
            # A record without its launch is missing evidence, never legacy.
            self.write_record(directory, legacy)
            (directory / 'launch.json').unlink()
            with self.assertRaises(OSError):
                runner.require_shutdown(directory)


@contextlib.contextmanager
def inert_gpu_query():
    """nvidia-smi on PATH, answering one fixture GPU, so a CPU fixture's run
    passes the runner's GPU query instead of bypassing it."""
    real, which = subprocess.run, shutil.which

    def run(command, *args, **kwargs):
        if command[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(command, 0, 'index, name\n0, Fixture\n', '')
        return real(command, *args, **kwargs)

    with patch.object(runner.shutil, 'which', lambda word: '/stub/nvidia-smi' if word == 'nvidia-smi' else which(word)), \
            patch.object(runner.subprocess, 'run', run):
        yield


class FakeRun:
    """A run of one model with every server and child faked: the children
    write receipts the stage rules accept, unless `failing` names a stage."""

    def __init__(self, base, model, failing=(), install_source=None, checkout=None):
        self.base, self.model, self.failing = base, model, set(failing)
        self.install_source = self.identity() if install_source is None else install_source
        self.checkout = self.identity() if checkout is None else checkout
        self.facts = runner.model_facts(model)
        self.checkpoint = base / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'model.safetensors').write_bytes(b'x' * 8)
        self.install = runner.install_root(self.identity())
        self.commands = []

    @staticmethod
    def identity():
        return {'commit': 'c' * 40, 'dirty': False, 'dirty_sha256': None}

    def prepare(self):
        (self.install / 'bin').mkdir(parents=True)
        bundle = self.base / 'bundle'
        runner.write(bundle / 'manifest.json', {'kv_layout': {'page_tokens': 64}})
        runner.write(self.install / 'install.json', {
            'source': self.install_source, 'files': {'bin/chopinfer': 'f'},
            'packages': {f'{self.model}/default': {'checkpoint': str(self.checkpoint), 'media': True,
                                                   'bundle': str(bundle), 'profile': self.facts['profile']}}})
        for config in runner.configurations(self.model).values():
            manifest = config['manifest']
            runner.write(runner.marker_path(self.model, self.checkpoint, manifest),
                         {'state_sha256': runner.state_digest(runner.checkpoint_state(self.checkpoint)),
                          'manifest': runner.manifest_identity(manifest)})

    @contextlib.contextmanager
    def server(self, context, arm, directory):
        (directory / 'server.log').write_text('fake server\n')
        command = [str(context['install'] / 'bin/chopinfer'), 'serve', '--model', str(context['checkpoint']),
                   *(word.format(**context['fill']) for word in arm['serve'])]
        runner.write(directory / 'launch.json', {'command': command, 'url': 'http://127.0.0.1:1'})
        runner.write(directory / 'health.json', {'arm': arm['arm'], 'process_group': 4242})
        yield 'http://127.0.0.1:1', 4242
        runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': 0})

    def call(self, command, log, seconds, env):
        command = list(map(str, command))
        self.commands.append(command)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text('fake child\n')
        script = Path(command[2]).name if len(command) > 2 else command[0]
        if script == 'verify_exact.py' and command[3] == 'references':
            Path(command[command.index('--out') + 1]).write_text('{}\n')
            return 0
        if script == 'checks.py':
            check = command[4]
            directory = Path(command[command.index('--receipts') + 1]) / f'run-{check}'
            directory.mkdir(parents=True)
            receipt = directory / f'{check}.json'
            runner.write(receipt, {'passed': True, 'mean_abs_dlogprob': 0.05})
            status = 'fail' if check in self.failing else 'pass'
            runner.write(directory / 'check.json', {'check': check, 'status': status, 'row': {
                'id': check, 'passed': status == 'pass', 'receipt': str(receipt),
                'predicates': [{'path': ['mean_abs_dlogprob'], 'op': 'le', 'value': 0.1, 'passed': status == 'pass'}],
                **({} if status == 'pass' else {'error': 'threshold predicate failed'})}})
            return 0 if status == 'pass' else 1
        if script == 'prefix_exact.py' and command[3] == 'report':
            with Path(log).open('a') as stream:
                stream.write(json.dumps({'passed': 'prefix_slice' not in self.failing,
                                         'slice': {'index': 1, 'count': 6}}, indent=1) + '\n')
            return 0
        if command[1:3] == ['eval', 'capture']:
            # eval capture writes its record into --report.
            receipts = Path(command[command.index('--report') + 1])
            receipts.mkdir(parents=True, exist_ok=True)
            script = command[command.index('--kind') + 1]
        else:
            receipts = Path(env['CHOPINFER_QUAL_RECEIPTS'])
        for flag in ('--run', '--out', '--capture-out'):
            if flag in command and Path(command[command.index(flag) + 1]) == receipts / 'run':
                (receipts / 'run').mkdir()
        arm = receipts.parent.name
        run_id = command[command.index('--run-id') + 1] if '--run-id' in command else None
        name = self.facts['name']
        if script == 'determinism':
            value = determinism_receipt(name, arm, run_id, '--image' in command)
        elif script == 'gsm8k.py':
            value = gsm8k_receipt(150 if f'gsm8k-{arm}' in self.failing else 200)
        elif script == 'greedy_pin':
            k = int(command[command.index('--k') + 1])
            value = {'passed': True, 'mechanics_only': False, 'server': {'model': name}, 'concurrency': 1, 'k': k,
                     'prompt_tokens': 256, 'max_tokens': 256, 'reference_sha256': self.exact_sha256,
                     'requests': {'text': [1, 2] if arm == 'ar' or 'greedy' not in self.failing else [1, 3]},
                     'finish_reasons': {'text': 'length'}}
        elif script == 'spec_verify.py':
            value = spec_receipt(name, command[command.index('--arm') + 1].split('-', 1)[1], run_id)
        elif script == 'needle.py':
            value = {'passed': True, 'mechanics_only': False, 'rows': [{'context': self.facts['context'], 'passed': True}]}
        elif script == 'soak.py':
            value = soak_receipt(self.facts, runner.AR_SOAK_SECONDS)
        elif script == 'gsm8k_branches.py':
            from gsm8k_branches import TOP_LOGPROBS
            from limits import NEAR_TIE_NATS
            ar, spec = (Path(command[command.index(flag) + 1]) for flag in ('--ar', '--spec'))
            value = {'passed': True, 'mechanics_only': False, 'errors': [], 'check': 'gsm8k-branches',
                     'ar': {'receipt': str(ar), 'sha256': runner.sha(ar)},
                     'spec': {'receipt': str(spec), 'sha256': runner.sha(spec)}, 'near_tie_nats': NEAR_TIE_NATS,
                     'top_logprobs': TOP_LOGPROBS, 'lost': [], 'gained': [], 'items': []}
        else:
            value = {'passed': True, 'mechanics_only': False, 'script': script}
        value.setdefault('errors', [])
        runner.write(receipts / f"{script.removesuffix('.py')}.json", value)
        return 0

    def run(self, *extra):
        self.prepare()
        from common import input_path
        import declaration
        self.exact_sha256 = runner.sha(input_path('exact', config=declaration.load(declaration.path(self.model))))
        clients = runner.queue_clients(self.facts)
        with patch.object(runner, 'server', self.server), patch.object(runner, 'call', self.call), \
                patch.object(runner, 'source', lambda: self.checkout), \
                inert_gpu_query(), \
                patch.object(runner, 'queue', lambda url, facts: {
                    'passed': True, 'mechanics_only': False, 'requests': clients, 'concurrency': clients}), \
                patch.dict(os.environ, {self.facts['checkpoint']: str(self.checkpoint)}), \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(['run', '--model', self.model, '--out', str(self.base / 'run'), *extra])
        return code, runner.read(self.base / 'run/results.json')


class FakeRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qual_test_config import preview_references
        super().setUpClass()
        cls.reference_root = cls.enterClassContext(preview_references(MODELS))


class RunTests(FakeRunTests):
    @contextlib.contextmanager
    def validation(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            yield Path(temporary)

    def test_fake_run_references_remain_private_and_authenticated(self):
        from common import input_path
        for name in MODELS:
            with self.subTest(model=name):
                config = declaration.load(declaration.path(name))
                path = input_path('exact', config=config)
                self.assertTrue(path.is_relative_to(self.reference_root))
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b' ')
                    with self.assertRaisesRegex(ValueError, 'lock entry exact: sha256 differs'):
                        input_path('exact', config=config)
                finally:
                    path.write_bytes(original)

    def test_a_failed_gpu_query_refuses_before_any_server_starts(self):
        # The run records the GPUs it ran on. A query that cannot run, fails,
        # times out or answers nothing refuses the run before any server
        # starts and before its directory exists.
        real, which, name = subprocess.run, shutil.which, model()
        answers = {
            # nvidia-smi is not on PATH: the run refuses without a query.
            'missing': None,
            # Found but cannot run: the real command on a PATH without it.
            'unrunnable': lambda command, kwargs, base: real(command, **kwargs, env={'PATH': str(base / 'none')}),
            'exit 18': lambda command, kwargs, base: subprocess.CompletedProcess(command, 18, '', 'fixture failure'),
            'empty': lambda command, kwargs, base: subprocess.CompletedProcess(command, 0, '', ''),
            'timeout': lambda command, kwargs, base: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 30)),
            'answered': lambda command, kwargs, base: subprocess.CompletedProcess(
                command, 0, 'index, name\n0, Fixture\n', '')}
        for case, answer in answers.items():
            with self.subTest(case=case), self.validation() as base:
                fake = FakeRun(base, name)
                fake.prepare()
                queries, launched = [], []

                def child(command, *args, **kwargs):
                    if command[0] == 'nvidia-smi':
                        queries.append(command)
                        return answer(command, kwargs, base)
                    return real(command, *args, **kwargs)

                errors = io.StringIO()
                with patch.object(runner, 'source', lambda: fake.checkout), \
                        patch.object(runner.shutil, 'which', lambda word: (answer and '/stub/nvidia-smi')
                                     if word == 'nvidia-smi' else which(word)), \
                        patch.object(runner.subprocess, 'run', child), \
                        patch.object(runner, 'run_configurations', lambda run, configs: launched.append(sorted(configs))), \
                        patch.object(runner, 'finish', lambda run: 0), \
                        patch.dict(os.environ, {fake.facts['checkpoint']: str(fake.checkpoint)}), \
                        contextlib.redirect_stderr(errors):
                    code = runner.main(['run', '--model', name, '--out', str(base / 'run')])
                self.assertEqual(len(queries), 0 if case == 'missing' else 1)
                if case == 'answered':
                    self.assertEqual(code, 0, errors.getvalue())
                    self.assertEqual(len(launched), 1)
                    self.assertEqual(runner.read(base / 'run/run.json')['gpus'], 'index, name\n0, Fixture\n')
                else:
                    self.assertEqual(code, 2)
                    self.assertEqual(launched, [])
                    self.assertIn('GPU query', errors.getvalue())
                    self.assertIn('no server was started', errors.getvalue())
                    self.assertFalse((base / 'run').exists())

    def test_passing_workers_cannot_hide_nonclean_shutdown(self):
        class Killed(FakeRun):
            @contextlib.contextmanager
            def server(self, context, arm, directory):
                with super().server(context, arm, directory) as endpoint:
                    yield endpoint
                if arm['arm'] == 'spec':
                    runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': -9})
        with self.validation() as base:
            code, result = Killed(base, model()).run('--suite', 'quick')
            self.assertEqual(code, 1)
            failures = [r for r in result['rows'] if r['status'] not in runner.PASSING]
            self.assertEqual([r['id'] for r in failures], ['default/spec/server'])
            self.assertIn('nonclean', failures[0]['error'])

    def test_configuration_alias_binds_launch_and_all_capture_arguments(self):
        """A configuration served under another name than the model's (a
        declared one, or an external variant of another checkpoint, which
        names its own) binds that name to the launch and every capture."""
        from test_variant import VARIANT, rehomed_model, selected, synthetic_variant
        tested = 0
        for name in MODELS:
            for config in runner.configurations(name):
                tested += self.served_alias(name, config)
        model = rehomed_model()
        self.assertIn(model, MODELS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'variants'
            synthetic_variant(root, model, checkpoint={'served_model_name': 'someone/requant'})
            with selected(root):
                tested += self.served_alias(model, VARIANT)
        self.assertGreaterEqual(tested, 1)

    def served_alias(self, name, config):
        """Whether `config` of model `name` is served under another name; if
        so, a full fake run of it binds that name everywhere."""
        import common
        facts = runner.model_facts(name)
        value = declaration.load(declaration.named(runner.declaration_name(name, config)))
        pin = value['qualification']['checkpoint']
        alias = pin['served_model_name']
        if alias == facts['name']:
            return 0
        selected = runner.configuration_facts(facts, config)
        self.assertEqual(selected['name'], alias)
        server = {'model': facts['name'], 'health': {'serving': {
            'checkpoint_identity': pin['checkpoint_identity'], 'startup_nonce': '1' * 32}}}
        with self.assertRaisesRegex(ValueError, 'the server serves'):
            common.bind_checkpoint(server, None, 'http://unused', value)
        server['model'] = selected['name']
        common.bind_checkpoint(server, None, 'http://unused', value)
        server['health']['serving']['checkpoint_identity'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'the server loaded checkpoint'):
            common.bind_checkpoint(server, None, 'http://unused', value)
        spec = runner.selected_configurations(name, [config], 'full')[config]
        with self.subTest(model=name, config=config), self.validation() as base:
            fake = FakeRun(base, name)
            original_prepare = fake.prepare
            original_server = fake.server

            def prepare():
                original_prepare()
                state = runner.read(fake.install / 'install.json')
                state['packages'][f'{name}/{config}'] = state['packages'][f'{name}/default']
                runner.write(fake.install / 'install.json', state)
                # An external variant's checkpoint is verified against its own lock.
                runner.write(runner.marker_path(name, fake.checkpoint, spec['manifest']),
                             {'state_sha256': runner.state_digest(runner.checkpoint_state(fake.checkpoint)),
                              'manifest': runner.manifest_identity(spec['manifest'])})

            @contextlib.contextmanager
            def bound_server(context, arm, directory):
                self.assertEqual(context['facts']['name'], alias)
                self.assertEqual(context['fill']['name'], alias)
                with original_server(context, arm, directory) as endpoint:
                    yield endpoint

            fake.prepare, fake.server = prepare, bound_server
            with patch.dict(os.environ, {spec['checkpoint']: str(fake.checkpoint)}):
                code, result = fake.run('--suite', 'full', '--config', config, '--no-baseline-update')
            self.assertEqual(code, 0, result['rows'])
            self.assertEqual(result['external_variants'], [config] if spec.get('external') else [])
            commands = [c for c in fake.commands if '--served-model' in c or '--model' in c]
            self.assertTrue(commands)
            for command in commands:
                option = '--served-model' if '--served-model' in command else '--model'
                self.assertEqual(command[command.index(option) + 1], alias)
        return 1

    def test_run_refuses_changed_or_missing_verification_manifest_before_children(self):
        for change in ('unchanged', 'changed', 'missing'):
            with self.subTest(change=change), self.validation() as base:
                fake = FakeRun(base, model())
                manifest = base / 'manifest.json'
                runner.write(manifest, {'model.safetensors': {'sha256': 'a' * 64}})
                fake.facts['manifest'] = str(manifest)
                prepare = fake.prepare

                def prepared():
                    prepare()
                    if change == 'changed':
                        runner.write(manifest, {'model.safetensors': {'sha256': 'b' * 64}})
                    elif change == 'missing':
                        manifest.unlink()

                fake.prepare = prepared
                with patch.object(runner, 'model_facts', return_value=fake.facts):
                    code, result = fake.run('--suite', 'quick')
                self.assertEqual((code, result['passed']), (0, True) if change == 'unchanged' else (1, False))
                if change != 'unchanged':
                    self.assertFalse(fake.commands)

    def test_a_quick_run_passes_every_row_and_reports_the_stored_values(self):
        for model in MODELS:
            with self.subTest(model=model), self.validation() as base:
                fake = FakeRun(base, model)
                code, result = fake.run('--suite', 'quick')
                failed = [row for row in result['rows'] if row['status'] not in runner.PASSING]
                self.assertEqual((code, failed), (0, []))
                ids = {row['id'] for row in result['rows']}
                self.assertTrue({'default/spec/gsm8k', 'default/ar/gsm8k', 'default/spec/prefix_slice',
                                 'default/determinism-spec-vs-spec-repeat', 'default/determinism-spec-vs-ar',
                                 'default/greedy-ar-vs-spec', 'default/spec-verify', 'default/gsm8k-paired',
                                 *check_rows(fake.facts, fake.facts['quick_arms'], fake.facts['quick'])} <= ids)
                # With nothing stored yet, each regression row says so, as information.
                regressions = [row for row in result['rows'] if row['id'].endswith(runner.STORES)]
                self.assertTrue(regressions)
                self.assertEqual({row['status'] for row in regressions}, {'informational'})
                self.assertTrue(all(row['reason'].startswith('no baseline yet') for row in regressions))
                greedy = next(row for row in result['rows'] if row['id'] == 'default/greedy-ar-vs-spec')
                self.assertEqual(greedy['status'], 'pass' if fake.facts['spec_greedy'] == 'exact' else 'informational')
                self.assertTrue((base / 'run/report.md').is_file())
                self.assertEqual((result['source'], result['install_source']), (fake.identity(), fake.identity()))
                # A quick run never moves the stored baselines.
                self.assertFalse((runner.VALIDATION / 'last-good').exists())

    def test_determinism_and_greedy_records_are_eval_captures(self):
        """Each arm's determinism and greedy record is taken by the install's
        `chopinfer eval capture` under the model's declaration, bound to the
        arm's startup log; no Python record runs."""
        with self.validation() as base:
            fake = FakeRun(base, model())
            code, result = fake.run('--suite', 'quick')
            self.assertEqual(code, 0, result['rows'])
            captures = [command for command in fake.commands if command[1:3] == ['eval', 'capture']]
            self.assertEqual(sorted(command[command.index('--kind') + 1] for command in captures),
                             ['determinism'] * 3 + ['greedy_pin'] * 2)
            for command in captures:
                self.assertEqual(command[0], str(fake.install / 'bin/chopinfer'))
                self.assertEqual(command[command.index('--model') + 1], str(runner.ROOT / 'models' / fake.model))
                self.assertNotIn('--variant', command)
                arm = Path(command[command.index('--report') + 1]).parent.name
                self.assertEqual(command[command.index('--server-log') + 1],
                                 str(base / 'run' / 'default' / arm / 'server.log'))
                self.assertEqual(command[command.index('--served-model') + 1], fake.facts['name'])
            scripts = {Path(command[2]).name for command in fake.commands if len(command) > 2}
            self.assertFalse({'determinism.py', 'greedy_equality.py'} & scripts)

    def test_the_determinism_image_request_follows_the_served_modalities(self):
        """A media package serves image unless the configuration's serve
        words select text alone: then each record sends no image (a server
        serving text refuses one) and its judge, at the run and on rescore,
        expects none; the default modalities still send it."""
        class Served(FakeRun):
            serving = []

            @contextlib.contextmanager
            def server(self, context, arm, directory):
                self.serving = arm['serve']
                with super().server(context, arm, directory) as endpoint:
                    yield endpoint

            def call(self, command, log, seconds, env):
                # A server serving text alone refuses an image request.
                words = self.serving
                if '--image' in map(str, command) and '--modalities' in words \
                        and 'image' not in words[words.index('--modalities') + 1].split(','):
                    self.commands.append(list(map(str, command)))
                    return 1
                return super().call(command, log, seconds, env)

        declared = runner.configurations
        for words, image in (([], True), (['--modalities', 'text'], False)):
            def configurations(name):
                value = declared(name)
                value['default'] = dict(value['default'], serve=[*value['default']['serve'], *words])
                return value
            with self.subTest(words=words), self.validation() as base, \
                    patch.object(runner, 'configurations', configurations):
                fake = Served(base, model())
                code, result = fake.run('--suite', 'quick')
                records = [row for row in result['rows'] if row['id'].endswith('/determinism-record')]
                self.assertEqual(len(records), 3)
                self.assertTrue(all(row['status'] == 'pass' for row in records), records)
                self.assertEqual(code, 0, [row for row in result['rows'] if row['status'] not in runner.PASSING])
                captures = [command for command in fake.commands if command[1:3] == ['eval', 'capture']
                            and command[command.index('--kind') + 1] == 'determinism']
                self.assertEqual(len(captures), 3)
                for command in captures:
                    self.assertEqual('--image' in command, image)
                    self.assertEqual('--image-reason' in command, not image)
                for receipt in (runner.read(row['receipt']) for row in records):
                    self.assertEqual(receipt['image']['included'], image)
                code, rows = RescoreTests().replay(fake)
                self.assertEqual(code, 0, [row for row in rows if row['status'] not in runner.PASSING])
                self.assertEqual({row['status'] for row in rows if row['id'].endswith('/determinism-record')},
                                 {'pass'})

    def test_a_regression_row_compares_with_each_store_s_arm(self):
        """paired_teacher reads the store's teacher receipt of the same
        configuration and arm against this run's: a seeded store gives a
        judged row, an empty one says it holds no baseline yet."""
        with self.validation() as base:
            name = model(lambda facts: 'paired_teacher' in facts['quick'])
            stored = base / 'seeded/teacher.json'
            runner.write(stored, {'passed': True})
            clean = FakeRun.identity()
            runner.Store('seed', name).write(base / 'seeded', {
                'model': name, 'suite': 'full', 'source': clean, 'install_source': clean,
                'rows': [{'id': 'default/spec/teacher', 'status': 'pass', 'receipt': str(stored)}]}, 'test seed')
            fake = FakeRun(base, name)
            code, result = fake.run('--suite', 'quick')
            status = {row['id']: row for row in result['rows']}
            self.assertEqual(status['default/spec/paired_teacher/seed']['status'], 'pass')
            self.assertEqual(status['default/spec/paired_teacher/last-good']['status'], 'informational')
            inputs = runner.read(base / 'run/default/spec/checks/paired_teacher-seed/inputs.json')
            self.assertEqual(Path(inputs['stored_teacher']).read_text(), stored.read_text())
            command = next(command for command in fake.commands
                           if 'paired_teacher' in command and 'paired_teacher-seed' in ' '.join(command))
            self.assertIn(f"teacher={status['default/spec/teacher']['receipt']}", command)

    def test_full_answers_the_regression_checks_on_every_quick_arm(self):
        name = model(lambda facts: 'ar' in facts['quick_arms'] and 'paired_teacher' in facts['quick'])
        facts = runner.model_facts(name)
        arms = {arm['arm']: arm for arm in runner.plan(facts, 'full', 'default', runner.configurations(name)['default'])}
        checks = [step['check'] for step in arms['ar']['steps'] if step['kind'] == 'check']
        self.assertEqual(checks, ['teacher', 'paired_teacher'])

    def test_failures_fail_the_run_and_keep_every_other_verdict(self):
        with self.validation() as base:
            fake = FakeRun(base, model(lambda facts: 'teacher' in facts['quick']),
                           failing={'teacher', 'gsm8k-ar', 'prefix_slice'})
            code, result = fake.run('--suite', 'quick')
            status = {row['id']: row['status'] for row in result['rows']}
            self.assertEqual(code, 1)
            self.assertEqual((status['default/spec/teacher'], status['default/ar/gsm8k'],
                              status['default/spec/prefix_slice'], status['default/gsm8k-paired']),
                             ('fail', 'fail', 'fail', 'not_run'))
            others = [check for check in fake.facts['quick'] if check != 'teacher'
                      and not runner.stored_inputs(fake.facts['checks'][check])]
            self.assertEqual({status[f'default/spec/{check}'] for check in others}, {'pass'})

    def test_a_speculative_greedy_difference_fails_an_exact_model(self):
        with self.validation() as base:
            name = model(lambda facts: facts['spec_greedy'] == 'exact')
            code, result = FakeRun(base, name, failing={'greedy'}).run('--suite', 'quick')
            row = next(row for row in result['rows'] if row['id'] == 'default/greedy-ar-vs-spec')
            self.assertEqual((code, row['status']), (1, 'fail'))

    def test_a_run_refuses_a_stale_verify_or_missing_install_before_any_server(self):
        with self.validation() as base:
            name = model()
            fake = FakeRun(base, name)
            with patch.object(runner, 'source', fake.identity), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(['run', '--model', name, '--out', str(base / 'none')]), 2)
            fake.prepare()
            os.utime(fake.checkpoint / 'model.safetensors', ns=(1, 1))
            servers = []
            with patch.object(runner, 'server', lambda *a: servers.append(a)), \
                    patch.object(runner, 'source', fake.identity), \
                    inert_gpu_query(), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(['run', '--model', name, '--out', str(base / 'stale')]), 1)
            result = runner.read(base / 'stale/results.json')
            self.assertEqual([row['id'] for row in result['rows']], ['default/configuration'])
            self.assertIn('not verified in its current state', result['rows'][0]['error'])
            self.assertEqual(servers, [])

    def test_the_continuation_check_runs_after_its_servers_without_one(self):
        name = model(lambda facts: 'spec_continuation' in facts['full'])
        with self.validation() as base:
            fake = FakeRun(base, name)
            code, result = fake.run('--suite', 'full', '--config', 'default', '--check', 'spec_continuation')
            launched = (base / 'run/default/offline/launch.json').exists()
        statuses = {row['id']: row['status'] for row in result['rows']}
        self.assertEqual(code, 0, statuses)
        self.assertEqual(statuses['default/offline/spec_continuation'], 'pass')
        self.assertNotIn('default/offline/health', statuses)
        self.assertFalse(launched)
        steps = [(Path(command[2]).name, command[3]) for command in fake.commands if len(command) > 3]
        order = [step for step in steps if step[0] in ('spec_continuation.py', 'checks.py')]
        self.assertEqual(order, [('spec_continuation.py', 'record'), ('spec_continuation.py', 'record'),
                                 ('spec_continuation.py', 'reference'), ('spec_continuation.py', 'reference'),
                                 ('checks.py', 'check')])
        references = [command for command in fake.commands if command[2:4] and command[3] == 'reference']
        for command, arm in zip(references, ('spec', 'ar')):
            self.assertEqual(command[command.index('--checkpoint') + 1], str(fake.checkpoint))
            self.assertEqual(Path(command[command.index('--record') + 1]).parent.parent.name, arm)
        check = next(command for command in fake.commands if Path(command[2]).name == 'checks.py')
        self.assertIn('--offline', check)
        self.assertNotIn('--url', check)

    def test_a_full_run_records_its_inputs_and_passes_them_to_the_checks(self):
        with self.validation() as base:
            fake = FakeRun(base, model(lambda facts: 'verify_exact' in facts['full']))
            code, result = fake.run('--suite', 'full', '--config', 'default', '--no-baseline-update')
            checks = [command for command in fake.commands if Path(command[2]).name == 'checks.py']
            verify = next(command for command in checks if command[4] == 'verify_exact')
            inputs = runner.read(verify[verify.index('--inputs') + 1])
            self.assertTrue({'verify_ar', 'verify_candidate', 'server_log'} <= set(inputs))
            self.assertIn('/verify-ar/', inputs['verify_ar'])
            status = {row['id']: row['status'] for row in result['rows']}
            # Inputs no record produces (an operator capture, a reference arm)
            # leave their checks not run, which fails the run.
            unproduced = [row for row in result['rows'] if row['status'] == 'not_run']
            self.assertTrue(all(row['error'].startswith('no ') for row in unproduced))
            self.assertEqual(code, 1 if unproduced else 0)
            self.assertEqual({row['status'] for row in result['rows']} - {'not_run'} - set(runner.PASSING), set())
            self.assertFalse((runner.VALIDATION / 'last-good').exists())


def producer_and_reader():
    """A two-check cohort of the mechanics fixture: `api` writes a capture
    under its raw directory, and `structured` reads that capture by the
    producer's check-scoped name, as a teacher_arms check reads its
    teacher's capture."""
    config = declaration.load(declaration.MECHANICS)
    checks = {check['id']: check for check in config['checks']}
    checks['api']['args'] = [*checks['api']['args'], '--capture-out', '{raw}/capture.json']
    checks['structured']['args'] = [*checks['structured']['args'], '--capture', '{check_raw[api]}/capture.json',
                                    '--earlier', '{api}']
    return config


def capture_worker(read, failing=(), fail_again=()):
    """subprocess.run for gate workers: `api` writes its capture and a
    receipt; `structured` passes only when the capture it is given is the
    one `api` wrote, and records the path it read in `read`. A check in
    `fail_again` fails from its second run on. Any other child (git, the
    scorer) runs as it would."""
    real = subprocess.run
    runs = {}

    def execute(command, *args, **kwargs):
        if '--worker' not in command:
            return real(command, *args, **kwargs)
        name, arguments = command[5], command[6:]
        runs[name] = runs.get(name, 0) + 1
        passed = name not in failing and not (name in fail_again and runs[name] > 1)
        if name == 'api':
            Path(arguments[arguments.index('--capture-out') + 1]).write_text('{"capture": "api"}\n')
        elif name == 'structured':
            path = Path(arguments[arguments.index('--capture') + 1])
            read.append(path)
            passed = passed and path.is_file() and path.read_text() == '{"capture": "api"}\n'
        value = {'schema': 'chopinfer-qualification/v1', 'check': name, 'passed': passed,
                 'skipped': [], 'errors': [], 'mechanics_only': True}
        (Path(kwargs['env']['CHOPINFER_QUAL_RECEIPTS']) / f'fresh-{name}.json').write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, 0)
    return execute


class CheckRawTests(unittest.TestCase):
    """A check reads a file an earlier check wrote under the producer's
    check-scoped raw directory, `{check_raw[ID]}`. release_preview runs each
    check as its own checks.py run, each with its own `{raw}`; it passes a
    passing check's raw directory on to the later checks."""

    def judged(self, context, base, check, read, failing=(), arm='arm'):
        """`check` through judged_check's checks.py run, in process, in the
        arm directory `arm`."""
        import checks
        import release_gate
        config = producer_and_reader()
        declared = next(item for item in config['checks'] if item['id'] == check)
        run = SimpleNamespace(replay=False, facts={'model': 'mechanics'}, used_features=set(),
                              install=base / 'install', rows=[])
        run.row = lambda config, arm, name, status, **fields: {'id': name, 'status': status, **fields}

        def call(command, log, seconds, env):
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with patch.object(sys, 'argv', ['checks.py', *command[3:]]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return checks.main()

        with patch.object(runner, 'call', call), \
                patch.object(checks, 'load', return_value=(config, declaration.MECHANICS)), \
                patch.object(checks.Client, 'identity', return_value={'model': 'fixture/tp4'}), \
                patch.object(release_gate.subprocess, 'run', side_effect=capture_worker(read, failing)):
            return runner.judged_check(run, context, check, declared, base / arm, 'spec', check, {})

    def context(self):
        return {'config_name': 'default', 'suite': 'quick', 'fill': {'url': 'http://unused'},
                'facts': {'name': 'fixture/tp4'}, 'env': dict(os.environ), 'checks': {}, 'check_raw': {}}

    def test_a_later_check_reads_the_capture_its_producer_wrote(self):
        # GPU pass 1 (C02): each checks.py run has its own {raw}, so a
        # reader of {raw}/<file> looked in its own empty directory.
        with tempfile.TemporaryDirectory() as temporary:
            base, read = Path(temporary), []
            context = self.context()
            produced = self.judged(context, base, 'api', read)
            self.assertEqual(produced['status'], 'pass', produced.get('error'))
            row = self.judged(context, base, 'structured', read)
            self.assertEqual(row['status'], 'pass', row.get('error'))
            (path,) = read
            run = Path(produced['check_run']).parent
            self.assertEqual(path, run / 'checks' / 'capture.json')
            self.assertEqual(context['check_raw']['api'], str(run / 'checks'))

    def test_a_producer_that_fails_or_is_refused_on_a_retry_leaves_no_capture_to_read(self):
        # A producer's receipt and raw directory once stayed published after
        # a later attempt of it failed (another serving arm sharing the
        # context) or was refused (a retry in the same directory, two runs
        # there), and the reader passed on the old capture.
        for retry, arm, failing in (('failed', 'second-arm', ('api',)), ('refused', 'arm', ())):
            with self.subTest(retry=retry), tempfile.TemporaryDirectory() as temporary:
                base, read = Path(temporary), []
                context = self.context()
                self.assertEqual(self.judged(context, base, 'api', read)['status'], 'pass')
                again = self.judged(context, base, 'api', read, failing, arm=arm)
                self.assertEqual(again['status'], 'fail')
                if retry == 'refused':
                    self.assertIn('missing or ambiguous receipt', again['error'])
                self.assertNotIn('api', context['checks'])
                self.assertNotIn('api', context['check_raw'])
                row = self.judged(context, base, 'structured', read, arm=arm)
                self.assertEqual(row['status'], 'fail')
                self.assertIn('no raw directory of check api', row['error'])
                # The reader's worker never ran.
                self.assertEqual(read, [])

    def test_a_reader_without_its_passing_producer_is_refused(self):
        for failing in (('api',), ()):
            with self.subTest(producer='failed' if failing else 'absent'), \
                    tempfile.TemporaryDirectory() as temporary:
                base, read = Path(temporary), []
                context = self.context()
                if failing:
                    self.assertEqual(self.judged(context, base, 'api', read, failing)['status'], 'fail')
                row = self.judged(context, base, 'structured', read)
                self.assertEqual(row['status'], 'fail')
                self.assertIn('no raw directory of check api', row['error'])
                self.assertEqual(read, [])


class FailedChild(FakeRun):
    """A fake run whose `check` child (checks.py check) misbehaves: exits 17
    or times out (124) after writing a passing check.json, or exits 0 with a
    passing status but no row or receipt."""

    def __init__(self, base, model, check, mode):
        super().__init__(base, model)
        self.check, self.mode = check, mode

    def call(self, command, log, seconds, env):
        code = super().call(command, log, seconds, env)
        command = list(map(str, command))
        if Path(command[2]).name == 'checks.py' and command[4] == self.check:
            directory = Path(command[command.index('--receipts') + 1]) / f'run-{self.check}'
            if self.mode == 'missing-receipt':
                (directory / f'{self.check}.json').unlink()
                runner.write(directory / 'check.json', dict(runner.read(directory / 'check.json'), row=None))
                return 0
            if self.mode == 'missing-file':
                (directory / f'{self.check}.json').unlink()
                return 0
            return 124 if self.mode == 'timeout' else 17
        return code


class CheckFailureTests(FakeRunTests):
    def test_a_failed_timed_out_or_receiptless_check_child_fails_its_row_and_the_run(self):
        name = model(lambda facts: 'teacher' in facts['quick'])
        for mode, error in (('error', 'checks.py exited 17'), ('timeout', 'checks.py exited 124'),
                            ('missing-receipt', 'holds no row for teacher'), ('missing-file', 'No such file')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
                code, result = FailedChild(Path(temporary), name, 'teacher', mode).run('--suite', 'quick')
                rows = [row for row in result['rows'] if row['id'].endswith('/teacher')]
                self.assertEqual((code, result['passed']), (1, False))
                self.assertTrue(rows and all(row['status'] == 'fail' for row in rows))
                self.assertTrue(all(error in row['error'] for row in rows), [row['error'] for row in rows])

    def test_an_input_its_input_rules_name_is_required_before_the_check_runs(self):
        """A check whose input rules alone name an input (the run ID a
        recorded arm's receipt must hold) is not run without it."""
        declared = {'module': 'evals', 'args': ['compare', '--receipt', 'fp8={fp8_gsm8k}'], 'files': ['fp8_gsm8k'],
                    'input_rules': {'fp8_gsm8k': [{'path': ['run_id'], 'op': 'eq', 'value': '{fp8_gsm8k_run}',
                                                   'source': 'tools/qual/README.md#release-gates'}]},
                    'rules': [], 'timeout_seconds': 1}
        facts = {'checks': {'compare': declared}}
        self.assertIn('fp8_gsm8k_run', runner.placeholders(declared))
        rows, judged = [], []
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'judged_check', lambda *args: judged.append(args[-1])):
            receipt = Path(temporary) / 'fp8.json'
            receipt.write_text('{}\n')
            for inputs in ({'fp8_gsm8k': str(receipt)}, {'fp8_gsm8k': str(receipt), 'fp8_gsm8k_run': 'fp8-run'}):
                run = SimpleNamespace(inputs=inputs, row=lambda *args, **fields: rows.append((args[3], fields)))
                context = {'facts': facts, 'config_name': 'c', 'records': {}, 'suite': 'full', 'checks': {}}
                runner.run_check(run, context, {'check': 'compare'}, Path(temporary), 'spec')
        self.assertEqual([status for status, _ in rows], ['not_run'])
        self.assertIn('no fp8_gsm8k_run', rows[0][1]['error'])
        self.assertEqual([inputs['fp8_gsm8k_run'] for inputs in judged], ['fp8-run'])

    def test_the_check_verdict_needs_agreeing_exit_status_row_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / 'receipt.json'
            runner.write(receipt, {'passed': True})
            row = {'id': 'x', 'passed': True, 'receipt': str(receipt)}
            result = {'check': 'x', 'status': 'pass', 'row': row}
            self.assertEqual(runner.check_verdict(0, result, 'x', {})[0], 'pass')
            for code, value, declared, message in (
                    (1, result, {}, 'exited 1 with status'), (None, result, {}, 'exited None'),
                    (0, dict(result, check='y'), {}, 'not x'),
                    (0, dict(result, row=dict(row, receipt=None)), {}, 'no receipt'),
                    (0, dict(result, row=dict(row, passed=False)), {}, 'did not pass'),
                    (0, dict(result, status='informational', row=dict(row, status='informational')), {},
                     'not declared informational'),
                    (0, dict(result, status='not_applicable', row=dict(row, status='not_applicable')), {},
                     'without its reason'),
                    (0, dict(result, status='fail'), {}, 'exited 0 with status')):
                with self.subTest(code=code, message=message), self.assertRaisesRegex(ValueError, message):
                    runner.check_verdict(code, value, 'x', declared)
            informational = {'check': 'x', 'status': 'informational', 'row': {'id': 'x', 'status': 'informational'}}
            self.assertEqual(runner.check_verdict(0, informational, 'x', {'informational_reason': 'r'})[0],
                             'informational')


class RescoreTests(FakeRunTests):
    @staticmethod
    def rescore(root):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(['rescore', str(root), '--json', str(root / 'rescored.json')])
        return code, runner.read(root / 'rescored.json')['rows'] if (root / 'rescored.json').is_file() else None

    def recorded(self, base, name, failing=()):
        fake = FakeRun(base, name, failing)
        code, result = fake.run('--suite', 'quick')
        return fake, code, result

    def replay(self, fake, judged=lambda receipt, model, config, check: {'status': 'pass'}):
        with patch.object(runner, 'call', fake.call), patch.object(runner, 'rescore_check', judged), \
                patch.dict(os.environ, {fake.facts['checkpoint']: str(fake.checkpoint)}):
            return self.rescore(fake.base / 'run')

    def test_a_check_subset_replays_only_its_own_plan(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            name = model(lambda facts: len(facts['quick']) > 1)
            fake = FakeRun(Path(temporary), name)
            check = fake.facts['quick'][-1]
            code, result = fake.run('--suite', 'quick', '--check', check)
            self.assertEqual(code, 0, [row for row in result['rows'] if row['status'] not in runner.PASSING])
            ids = [row['id'] for row in result['rows']]
            ran = {row.split('/')[2] for row in ids if row.count('/') >= 2} - {'health'}
            self.assertEqual(ran, {check}, ids)
            code, rows = self.replay(fake)
            self.assertEqual(code, 0)
            self.assertEqual([row['id'] for row in rows], ids)

    def test_the_rescore_entry_points_judge_shutdown_evidence_without_pythonpath(self):
        # The test process has the repository on sys.path; the entry points
        # run without it, from outside the checkout and as the package.
        # A clean copy also excludes the checkout's local.env, which has
        # precedence over the fixture's selected environment file.
        from qual_test_config import clean_checkout
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'), \
                clean_checkout() as checkout:
            # A subset whose checks import nothing else from the checkout, as
            # the offload checks do: only the shutdown evidence needs it.
            fake = FakeRun(Path(temporary), model(lambda facts: 'api' in facts['quick']))
            code, _ = fake.run('--suite', 'quick', '--check', 'api')
            self.assertEqual(code, 0)
            env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            env[fake.facts['checkpoint']] = str(fake.checkpoint)
            run = fake.base / 'run'
            arm = sorted(path.name for path in (run / 'default').iterdir() if (path / 'shutdown.json').is_file())[0]
            runner.write(run / 'default' / arm / 'shutdown.json', {'pid': runner.read(
                run / 'default' / arm / 'health.json')['process_group'], 'returncode': -9})
            for command, cwd in (
                    (['bash', str(checkout / 'tools/qual/release_preview.sh')], Path(temporary)),
                    ([sys.executable, '-B', '-m', 'tools.qual', 'preview'], checkout)):
                with self.subTest(command=command[-1]):
                    out = Path(temporary) / f'rescored-{len(command)}.json'
                    done = subprocess.run([*command, 'rescore', str(run), '--json', str(out)], cwd=cwd, env=env,
                                          capture_output=True, text=True, timeout=600, check=False)
                    self.assertNotIn('Traceback', done.stderr, done.stderr[-2000:])
                    self.assertEqual(done.returncode, 1, done.stderr[-2000:])
                    self.assertTrue(out.is_file(), done.stdout + done.stderr)
                    rows = {row['id']: row for row in runner.read(out)['rows']}
                    self.assertIn(f'default/{arm}/server', rows, rows)
                    server = rows[f'default/{arm}/server']
                    self.assertEqual(server['status'], 'fail')
                    self.assertIn('nonclean or unconfirmed server shutdown: -9', server['error'])

    @staticmethod
    def text_only():
        """The declared configurations with `default` serving text alone."""
        declared = runner.configurations

        def configurations(name):
            value = declared(name)
            value['default'] = dict(value['default'], serve=runner.serve_overrides(
                value['default']['serve'], ['--modalities', 'text']))
            return value
        return patch.object(runner, 'configurations', configurations)

    def determinism_rows(self, rows):
        return [row for row in rows if row['id'].endswith('/determinism-record')]

    def test_a_replay_keeps_a_text_capture_s_image_expectation_when_the_declaration_changes(self):
        """A record captured serving text alone, rescored after the
        declaration returns to the default modalities, is judged by the
        server its arm launched: it still expects no image."""
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            fake = FakeRun(Path(temporary), model())
            with self.text_only():
                code, result = fake.run('--suite', 'quick')
            self.assertEqual(code, 0, [row for row in result['rows'] if row['status'] not in runner.PASSING])
            code, rows = self.replay(fake)
            self.assertEqual(code, 0, [row for row in rows if row['status'] not in runner.PASSING])
            self.assertEqual([row['status'] for row in self.determinism_rows(rows)], ['pass'] * 3)

    def test_a_replay_under_a_text_declaration_cannot_admit_an_image_capture_missing_its_image(self):
        """A record captured under the default modalities without its image
        request fails on rescore, also after the declaration selects text."""
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            fake = FakeRun(Path(temporary), model())
            code, result = fake.run('--suite', 'quick')
            self.assertEqual(code, 0)
            for row in self.determinism_rows(result['rows']):
                receipt = runner.read(row['receipt'])
                runner.write(Path(row['receipt']), determinism_receipt(fake.facts['name'], receipt['arm'],
                                                                       receipt['run_id'], False))
            for text_only, patched in ((False, contextlib.nullcontext()), (True, self.text_only())):
                with self.subTest(text_only=text_only), patched:
                    code, rows = self.replay(fake)
                    self.assertEqual(code, 1)
                    self.assertEqual([(row['status'], row['error']) for row in self.determinism_rows(rows)],
                                     [('fail', 'determinism image request differs from what the server serves')] * 3)

    def test_a_launch_without_its_command_is_judged_by_the_package_as_before(self):
        """A stored launch that records no serve command (the default
        modalities, judged by the package's media capability before served
        modalities were read) rejudges as it did, whatever the declaration
        now serves."""
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            fake, code, _ = self.recorded(Path(temporary), model())
            self.assertEqual(code, 0)
            for launch in (fake.base / 'run').glob('default/*/launch.json'):
                runner.write(launch, {key: value for key, value in runner.read(launch).items() if key != 'command'})
            with self.text_only():
                code, rows = self.replay(fake)
            self.assertEqual(code, 0, [row for row in rows if row['status'] not in runner.PASSING])
            self.assertEqual([row['status'] for row in self.determinism_rows(rows)], ['pass'] * 3)

    def test_a_replay_rejudges_every_row_the_plan_expects(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            name = model(lambda facts: 'teacher' in facts['quick'])
            fake, code, result = self.recorded(Path(temporary), name)
            self.assertEqual(code, 0)
            code, rows = self.replay(fake)
            self.assertEqual(code, 0)
            self.assertEqual([row['id'] for row in rows], [row['id'] for row in result['rows']])
            self.assertEqual([row['recorded'] for row in rows], [row['status'] for row in result['rows']])
            # A check judged again under rules that now fail it fails.
            code, rows = self.replay(fake, lambda receipt, model, config, check: {'status': 'fail', 'error': 'rule'})
            self.assertEqual(code, 1)
            self.assertEqual({row['status'] for row in rows if row['id'].endswith('/teacher')}, {'fail'})

    def test_replay_refuses_hidden_nonclean_or_unconfirmed_shutdowns(self):
        for termination in ({'pid': 4242, 'returncode': 0}, {'pid': 4242, 'returncode': -9},
                            {'pid': 4242, 'returncode': None}, {'pid': 4242, 'returncode': False},
                            {'pid': 4242, 'returncode': 0, 'termination': {'confirmed': True, 'forced': True}},
                            None):
            with self.subTest(termination=termination), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
                fake, code, result = self.recorded(Path(temporary), model())
                self.assertEqual(code, 0)
                original = (fake.base / 'run/results.json').read_bytes()
                path = fake.base / 'run/default/spec/shutdown.json'
                if termination is None:
                    path.unlink()
                else:
                    runner.write(path, termination)
                code, rows = self.replay(fake)
                good = (termination == {'pid': 4242, 'returncode': 0}
                        and type(termination.get('returncode')) is int)
                self.assertEqual(code, 0 if good else 1)
                if not good:
                    failure = next(row for row in rows if row['id'] == 'default/spec/server')
                    self.assertEqual(failure['status'], 'fail')
                self.assertEqual((fake.base / 'run/results.json').read_bytes(), original)

    def test_a_replay_fails_missing_receipts_and_recomputes_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            name = model(lambda facts: 'teacher' in facts['quick'])
            fake, code, result = self.recorded(Path(temporary), name)
            run = fake.base / 'run'
            teacher = next(row for row in result['rows'] if row['id'] == 'default/spec/teacher')
            Path(teacher['receipt']).unlink()
            greedy = next(row for row in result['rows'] if row['id'] == 'default/ar/greedy')
            Path(greedy['receipt']).unlink()
            # The recorded comparison file is not read: the comparison is recomputed.
            Path(next(row for row in result['rows'] if row['id'] == 'default/greedy-ar-vs-spec')['receipt']).unlink()
            code, rows = self.replay(fake)
            status = {row['id']: row for row in rows}
            self.assertEqual(code, 1)
            self.assertEqual(status['default/spec/teacher']['status'], 'fail')
            self.assertIn('No such file', status['default/spec/teacher']['error'])
            self.assertEqual(status['default/ar/greedy']['status'], 'fail')
            self.assertEqual(status['default/greedy-ar-vs-spec']['status'], 'not_run')
            self.assertEqual(status['default/spec/gsm8k']['status'], 'pass')
            # A child whose exit was never kept fails, whatever its receipt says.
            (run / 'default/spec/gsm8k/check.log.exit').unlink()
            code, rows = self.replay(fake)
            self.assertIn('exited None', next(row for row in rows if row['id'] == 'default/spec/gsm8k')['error'])

    def test_a_replay_keeps_a_server_that_failed_after_its_checks(self):
        class CleanupFailed(FakeRun):
            @contextlib.contextmanager
            def server(self, context, arm, directory):
                with super().server(context, arm, directory) as endpoint:
                    yield endpoint
                raise OSError('shutdown.json write failed')

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            fake = CleanupFailed(Path(temporary), model(lambda facts: 'teacher' in facts['quick']))
            code, result = fake.run('--suite', 'quick')
            failed = sorted(row['id'] for row in result['rows'] if row['status'] not in runner.PASSING)
            self.assertEqual(code, 1)
            self.assertTrue(failed and all(name.endswith('/server') for name in failed))
            code, rows = self.replay(fake)
            self.assertEqual(code, 1)
            self.assertEqual(sorted(row['id'] for row in rows if row['status'] not in runner.PASSING), failed)
            self.assertEqual({row.get('kept') for row in rows if row['id'] in failed}, {'a recorded execution failure'})

    def test_an_empty_or_unplanned_run_does_not_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = model()
            runner.write(root / 'results.json', {'model': name, 'rows': []})
            runner.write(root / 'run.json', {'model': name, 'suite': 'quick', 'configurations': [], 'inputs': {}})
            self.assertEqual(self.rescore(root), (2, None))
            runner.write(root / 'run.json', {'packages': {}})
            self.assertEqual(self.rescore(root)[0], 2)
            runner.write(root / 'results.json', {'model': name, 'rows': [
                {'id': 'default/spec/teacher', 'status': 'pass', 'check_run': str(root / 'missing.json')}]})
            self.assertEqual(self.rescore(root)[0], 2)


class PromotionTests(FakeRunTests):
    def full_pass(self, base, checkout, install, main, configs=None, name=None, feature_source=..., observed_envs=None,
                  checks=(), values=()):
        """A full run of the install with every input no record produces
        given, so every row passes; git calls a commit on main when it is in
        `main`."""
        name = name or model(lambda facts: 'verify_exact' in facts['full'])
        fake = FakeRun(base, name, install_source=install, checkout=checkout)
        if observed_envs is not None:
            original_call = fake.call

            def observed(command, log, seconds, env):
                observed_envs.append((list(map(str, command)), dict(env)))
                return original_call(command, log, seconds, env)

            fake.call = observed
        declared_configs = runner.configurations(name)
        prepare = fake.prepare

        def prepared():
            prepare()
            state = runner.read(fake.install / 'install.json')
            package = state['packages'][f'{name}/default']
            state['packages'] = {f'{name}/{config}': dict(package) for config in declared_configs}
            runner.write(fake.install / 'install.json', state)
            needed = set().union(*(runner.configuration_features(fake.facts, config, spec, 'full')
                                   for config, spec in declared_configs.items()))
            for features in needed - {()}:
                featured = fake.install.with_name(fake.install.name + ''.join(f'+{f}' for f in features))
                featured_state = dict(state, source=install if feature_source is ... else feature_source)
                if feature_source is None:
                    featured_state.pop('source')
                runner.write(featured / 'install.json', featured_state)

        fake.prepare = prepared
        given = set()
        for config, spec in declared_configs.items():
            facts = runner.configuration_facts(fake.facts, config)
            planned = runner.plan(fake.facts, 'full', config, spec)
            produced = {step['input'] for arm in planned for step in arm['steps'] if step['kind'] == 'record'}
            names = set()
            for check in facts['full']:
                declared = facts['checks'][check]
                names.update(declared.get('files', []))
                names.update(runner.placeholders(declared))
                names.update(runner.module_option(declared['module'], 'RUNNER_EXTERNAL_INPUTS', []))
            given.update(names - produced - set(runner.STAGE_INPUTS) - set(facts['checks'])
                         - {runner.LOG_INPUT, 'url', 'model', 'raw', 'run', 'run_id'})
        path = base / 'external-input.json'
        path.write_text('{}\n')
        if not set(values) <= given:
            raise AssertionError(f'{sorted(set(values) - given)} are not inputs the run must be given')
        given -= set(values)
        git = runner.subprocess.run

        def answer(command, *args, **kwargs):
            if command[:2] == ['git', '-C'] and 'merge-base' in command:
                return SimpleNamespace(returncode=0 if command[command.index('--is-ancestor') + 1] in main else 1)
            return git(command, *args, **kwargs)

        with patch.object(runner.subprocess, 'run', answer), \
                patch.object(runner, 'device_pairs', lambda: ['0,1', '2,3']):
            code, result = fake.run('--suite', 'full', '--install', str(fake.install),
                                    *(word for config in configs or [] for word in ('--config', config)),
                                    *(word for check in checks for word in ('--check', check)),
                                    *(word for key in sorted(given) for word in ('--input', f'{key}={path}')),
                                    *(word for key in sorted(values) for word in ('--input-value', f'{key}={VALUE}')))
        return name, code, result

    def test_a_value_input_reaches_the_checks_as_written_and_replays(self):
        # A placeholder a check's arguments fill with an input no record
        # produces and no file names: an input that is a value (a run ID).
        found = None
        for name in MODELS:
            facts = runner.model_facts(name)
            for config, spec in runner.configurations(name).items():
                selected = runner.configuration_facts(facts, config)
                produced = {step['input'] for arm in runner.plan(facts, 'full', config, spec)
                            for step in arm['steps'] if step['kind'] == 'record'}
                for check in selected['full']:
                    declared = selected['checks'][check]
                    keys = sorted(key for key in runner.placeholders(declared) - set(declared.get('files', []))
                                  - produced - set(selected['checks']) - {'url', 'model', 'raw', 'run', 'run_id'}
                                  if not key.startswith(runner.STORED))
                    if keys:
                        found = found or (name, config, check, keys[0])
        if found is None:
            self.skipTest('no declared check fills an argument with a value input')
        name, config, check, key = found
        clean = FakeRun.identity()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            _, code, result = self.full_pass(Path(temporary), clean, clean, {clean['commit']}, [config], name=name,
                                             values=[key])
            self.assertEqual((code, result['passed']), (0, True),
                             [row for row in result['rows'] if row['status'] not in runner.PASSING])
            run = Path(temporary) / 'run'
            state = runner.read(run / 'run.json')
            self.assertEqual(state['input_values'], {key: VALUE})
            self.assertNotIn(key, state['inputs'])
            given = [runner.read(path) for path in sorted(run.glob(f'{config}/*/checks/{check}*/inputs.json'))]
            self.assertTrue(given)
            self.assertTrue(all(value[key] == VALUE for value in given), given)
            # A replay gives the recorded value again, not a path.
            replayed = []
            with patch.object(runner, 'run_configuration', lambda run_, *args: replayed.append(dict(run_.inputs))), \
                    patch.object(runner, 'finish', lambda run_: None):
                runner.rescore_run(run)
            self.assertEqual([inputs[key] for inputs in replayed], [VALUE])

    def test_a_value_input_is_named_once_and_never_empty(self):
        path = Path(__file__)
        facts = runner.model_facts(MODELS[0])
        for words, message in ((['run-a'], 'KEY=VALUE'), (['key='], 'KEY=VALUE'), (['bad key=run-a'], 'KEY=VALUE'),
                               (['key=run-a', 'key=run-b'], 'twice')):
            args = SimpleNamespace(input=[], input_value=words)
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, message):
                runner.Run(args, facts, {}, FakeRun.identity(), path, path)
        args = SimpleNamespace(input=[f'key={path}'], input_value=['key=run-a'])
        with self.assertRaisesRegex(ValueError, 'twice'):
            runner.Run(args, facts, {}, FakeRun.identity(), path, path)
        args = SimpleNamespace(input=['key=/no/such/path'], input_value=[])
        with self.assertRaisesRegex(ValueError, 'existing path'):
            runner.Run(args, facts, {}, FakeRun.identity(), path, path)
        run = runner.Run(SimpleNamespace(input=[f'file={path}'], input_value=['key=run-a']), facts, {},
                         FakeRun.identity(), path, path)
        self.assertEqual((run.inputs, run.input_values), ({'file': str(path.resolve()), 'key': 'run-a'},
                                                          {'key': 'run-a'}))

    def test_focused_full_run_passes_without_promoting(self):
        clean = FakeRun.identity()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            name, code, result = self.full_pass(Path(temporary), clean, clean, {clean['commit']}, ['default'])
            self.assertGreater(len(runner.configurations(name)), 1)
            self.assertEqual((code, result['passed']), (0, True))
            self.assertEqual({row['id'].split('/')[0] for row in result['rows']}, {'default'})
            self.assertFalse(runner.Store('last-good', name).root.exists())
            self.assertEqual(result['configurations'], ['default'])

    def test_a_check_subset_runs_on_every_selected_configuration_and_never_promotes(self):
        clean = FakeRun.identity()
        for name in MODELS:
            facts, configs = runner.model_facts(name), runner.configurations(name)
            cohorts = {config: runner.configuration_facts(facts, config)['full'] for config in configs}
            shared = [check for check in cohorts['default'] if all(check in full for full in cohorts.values())]
            partial = [check for check in cohorts['default'] if not all(check in full for full in cohorts.values())]
            if shared and partial:
                break
        else:
            self.fail('no model has both a check every configuration declares and one it does not')
        # Every selected configuration must declare each selected check.
        with self.assertRaisesRegex(ValueError, 'checks outside'):
            runner.select_checks(facts, configs, 'full', [partial[0]])
        with self.assertRaisesRegex(ValueError, 'checks outside'):
            runner.select_checks(facts, configs, 'full', ['no_such_check'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['prepare', '--model', name, '--check', shared[0]])
        chosen, _ = runner.select_checks(facts, {'default': configs['default']}, 'full', [partial[0]])
        self.assertEqual(runner.configuration_facts(chosen, 'default')['full'], [partial[0]])
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            _, code, result = self.full_pass(Path(temporary), clean, clean, {clean['commit']}, name=name,
                                             checks=[shared[0]])
            self.assertEqual((code, result['passed']), (0, True),
                             [row for row in result['rows'] if row['status'] not in runner.PASSING])
            self.assertEqual((result['configurations'], result['selected_checks']), (sorted(configs), [shared[0]]))
            ran = {row['id'].split('/', 2)[2] for row in result['rows'] if row['id'].count('/') == 2}
            self.assertIn(shared[0], ran)
            self.assertFalse(ran & (set(cohorts['default']) - {shared[0]}), ran)
            self.assertEqual({row['id'].split('/')[0] for row in result['rows']}, set(configs))
            run = Path(temporary) / 'run'
            self.assertEqual(runner.read(run / 'run.json')['selected_checks'], [shared[0]])
            # A selected check subset is not a whole-model pass: last good
            # stays, and it cannot become the seed.
            self.assertFalse(runner.whole_model_full_pass(result, name))
            self.assertFalse(runner.Store('last-good', name).root.exists())
            self.assertIn('subset', runner.Store('last-good', name).update(run, result))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'subset'):
                runner.baseline_command(SimpleNamespace(action='reseed', model=name, source=str(run),
                                                        reason='operator: test', config=[]))
            self.assertIsNone(runner.Store('seed', name).result('default'))
            self.assertTrue(runner.whole_model_full_pass(dict(result, selected_checks=[]), name))

    def test_only_a_clean_main_checkout_and_install_move_last_good(self):
        clean = FakeRun.identity()
        dirty = dict(clean, dirty=True, dirty_sha256='a' * 64)
        other = dict(clean, commit='e' * 40)
        main = {clean['commit']}
        for checkout, install, moves in ((clean, clean, True), (dirty, clean, False), (clean, dirty, False),
                                         (other, clean, False), (clean, other, False), (clean, {}, False),
                                         (dirty, dirty, False)):
            with self.subTest(checkout=checkout, install=install), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
                name, code, result = self.full_pass(Path(temporary), checkout, install, main)
                # The correctness result stands either way.
                self.assertEqual((code, result['passed']), (0, True), [row for row in result['rows']
                                                                       if row['status'] not in runner.PASSING])
                self.assertEqual((result['source'], result['install_source']), (checkout, install))
                self.assertEqual(runner.Store('last-good', name).result('default') is not None, moves)

    def test_full_capture_children_keep_declared_reference_paths_and_tool_python(self):
        name = model(lambda facts: 'long_context_128k' in facts['full'])
        config = declaration.load(declaration.path(name))
        key = config['qualification']['references']['long_context_128k']['path']['env']
        clean = FakeRun.identity()
        observed = []
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            base = Path(temporary)
            reference = base / 'external'
            reference.mkdir()
            with patch.dict(os.environ, {key: str(reference), 'CHOPINFER_TEST_DEBUG_OVERRIDE': '1',
                                         'CHOPINFER_REFRUN_PYTHON': sys.executable}):
                _, code, result = self.full_pass(base, clean, clean, {clean['commit']}, ['default'], name=name,
                                                 observed_envs=observed)
            self.assertEqual((code, result['passed']), (0, True))
            captures = [env for command, env in observed if 'long_context_128k' in command]
            self.assertTrue(captures)
            for env in captures:
                self.assertEqual(env.get(key), str(reference))
                self.assertEqual(env.get('CHOPINFER_REFRUN_PYTHON'), sys.executable)
                self.assertNotIn('CHOPINFER_TEST_DEBUG_OVERRIDE', env)

    def test_every_used_feature_install_is_recorded_and_admitted(self):
        clean = FakeRun.identity()
        dirty = dict(clean, dirty=True, dirty_sha256='a' * 64)
        other = dict(clean, commit='e' * 40)
        for name in MODELS:
            facts = runner.model_facts(name)
            features = set().union(*(runner.configuration_features(facts, config, spec, 'full')
                                     for config, spec in runner.configurations(name).items())) - {()}
            if not features:
                continue
            for featured, moves in ((clean, True), (dirty, False), (other, False), (None, False)):
                with self.subTest(model=name, source=featured), tempfile.TemporaryDirectory() as temporary, \
                        patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
                    base = Path(temporary)
                    _, code, result = self.full_pass(base, clean, clean, {clean['commit']}, name=name,
                                                     feature_source=featured)
                    self.assertEqual((code, result['passed']), (0, True))
                    self.assertEqual(runner.Store('last-good', name).root.exists(), moves)
                    used = result['used_installs']
                    self.assertEqual(set(used), {'', *('+'.join(f) for f in features)})
                    self.assertEqual(used['']['source'], clean)
                    self.assertEqual(runner.read(base / 'run/run.json')['used_installs'], used)
                    for key in used.keys() - {''}:
                        self.assertEqual(used[key]['source'], featured)
                        self.assertTrue(Path(used[key]['path']).name.endswith('+' + key))
                        self.assertEqual(used[key]['files'], {'bin/chopinfer': 'f'})
                    if moves:
                        store = runner.Store('last-good', name)
                        self.assertEqual(store.result('default')['used_installs'], used)
                        # Store.update independently refuses a missing feature identity.
                        for key in used:
                            incomplete = {k: v for k, v in used.items() if k != key}
                            self.assertIn('records no used install',
                                          store.update(base / 'run', dict(result, used_installs=incomplete)))
                    else:
                        self.assertFalse((runner.VALIDATION / 'history.jsonl').exists())


class StoreTests(unittest.TestCase):
    def test_an_evals_arm_is_stored_with_its_results(self):
        """A stored evals receipt names the stored copy of its results file,
        whose digest it records, so `evals.py compare --stored` reads it."""
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            run = Path(temporary) / 'run'
            results = run / 'candidate.jsonl'
            results.parent.mkdir(parents=True)
            results.write_text('{"id": "gsm8k-0"}\n')
            receipt = run / 'evals.json'
            runner.write(receipt, {'check': 'evals-candidate', 'results': str(results),
                                   'results_sha256': runner.sha(results)})
            clean = FakeRun.identity()
            store = runner.Store('seed', model())
            store.write(run, {'model': model(), 'suite': 'full', 'source': clean, 'install_source': clean,
                              'rows': [{'id': 'default/spec/evals_candidate', 'status': 'pass',
                                        'receipt': str(receipt)}]}, 'test')
            stored = runner.read(store.receipt('default', 'default/spec/evals_candidate'))
            self.assertEqual(Path(stored['results']).read_text(), results.read_text())
            self.assertEqual(runner.sha(stored['results']), stored['results_sha256'])
            self.assertTrue(Path(stored['results']).is_relative_to(store.root))

    def test_last_good_moves_only_on_a_clean_main_pass_and_the_seed_only_on_a_reseed(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, 'VALIDATION', Path(temporary) / 'validation'):
            run = Path(temporary) / 'run'
            receipt = run / 'gsm8k.json'
            runner.write(receipt, gsm8k_receipt())
            name = model()
            clean = {'commit': 'c' * 40, 'dirty': False}
            result = {'model': name, 'suite': 'full', 'passed': True, 'source': clean, 'install_source': clean,
                      'configurations': sorted(runner.configurations(name)),
                      'rows': [{'id': 'default/spec/gsm8k', 'status': 'pass', 'receipt': str(receipt),
                                'values': [runner.metric('correct', 200, 'ge', 193)]}]}
            features = {(), *(f for config, spec in runner.configurations(name).items()
                               for f in runner.configuration_features(runner.model_facts(name), config, spec, 'full'))}
            result['used_installs'] = {'+'.join(f): {'source': clean} for f in features}
            result['rows'].extend({'id': f'{config}/spec/health', 'status': 'pass'}
                                  for config in result['configurations'] if config != 'default')
            for config, spec in runner.configurations(name).items():
                for arm in runner.plan(runner.model_facts(name), 'full', config, spec):
                    directory = run / config / arm['arm']
                    runner.write(directory / 'launch.json', {'url': 'http://127.0.0.1:1'})
                    runner.write(directory / 'health.json', {'process_group': 4242})
                    runner.write(directory / 'shutdown.json', {'pid': 4242, 'returncode': 0})
            runner.write(run / 'results.json', result)
            store = runner.Store('last-good', name)
            shutdown = run / 'default/spec/shutdown.json'
            runner.write(shutdown, {'pid': 4242, 'returncode': -9})
            self.assertIn('shutdown evidence refused', store.update(run, result))
            runner.write(shutdown, {'pid': 4242, 'returncode': 0})

            original_run = runner.subprocess.run

            def answer(command, *args, **kwargs):
                if '--is-ancestor' not in command:
                    return original_run(command, *args, **kwargs)
                on_main = command[command.index('--is-ancestor') + 1] == clean['commit']
                return SimpleNamespace(returncode=0 if on_main else 1)

            with patch.object(runner.subprocess, 'run', answer):
                for incomplete in (dict(result, configurations=['default']),
                                   dict(result, rows=result['rows'][:1]),
                                   {k: v for k, v in result.items() if k != 'configurations'},
                                   dict(result, suite='quick'), dict(result, passed=False)):
                    self.assertIn('whole-model full pass', store.update(run, incomplete))
                # Both the validation checkout and the tested install must be
                # clean and on main; a result without either source never moves.
                for key in ('source', 'install_source'):
                    for source_, message in ((dict(clean, dirty=True), 'uncommitted changes'),
                                             (dict(clean, commit='f' * 40), 'is not on main'),
                                             (None, 'records no source')):
                        self.assertIn(message, store.update(run, dict(result, **{key: source_})))
                    self.assertIn('records no source', store.update(run, {k: v for k, v in result.items() if k != key}))
                self.assertIsNone(store.value('default', 'default/spec/gsm8k', 'correct'))
                self.assertFalse((runner.VALIDATION / 'history.jsonl').exists())
                self.assertIn('stored', store.update(run, result))
            self.assertEqual(store.value('default', 'default/spec/gsm8k', 'correct'), 200)
            self.assertTrue((store.root / 'default/baseline/default__spec__gsm8k.json').is_file())
            stored = runner.read(store.root / 'default/result.json')['rows'][0]
            self.assertEqual(stored['stored'], str(store.root / 'default/baseline/default__spec__gsm8k.json'))
            self.assertIsNone(runner.Store('seed', name).result('default'))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'reason'):
                    runner.baseline_command(SimpleNamespace(action='reseed', model=name, source=str(run),
                                                            reason=' ', config=[]))
                runner.baseline_command(SimpleNamespace(action='reseed', model=name, source=str(run),
                                                        reason='operator: test', config=[]))
            self.assertEqual(runner.Store('seed', name).value('default', 'default/spec/gsm8k', 'correct'), 200)
            history = [json.loads(line) for line in (runner.VALIDATION / 'history.jsonl').read_text().splitlines()]
            self.assertEqual([(row['store'], row['reason']) for row in history],
                             [('last-good', 'full pass'), ('seed', 'operator: test')])


if __name__ == '__main__':
    unittest.main()
