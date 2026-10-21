"""CPU runner policy and failure tests; the TP4 fixture exercises real HTTP."""

from qual_test_config import configure_receipts as setUpModule

import contextlib
from copy import deepcopy
import io
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qual_test_config import capture_fixture, inventory, released

import checks
import common
import declaration
import kinds
import release_gate as gate
import soak


class GateTests(unittest.TestCase):
    def test_plan_preserves_cohort_order_predicates_and_prerequisites(self):
        config = declaration.load(declaration.path('mimo26'))
        original = deepcopy(config)
        description = gate.describe(config, 'full')
        self.assertEqual(config, original)
        self.assertEqual(description['selected_checks'], config['profiles']['full']['checks'])
        self.assertNotIn('budget_seconds', description)
        for row in description['checks']:
            source = next(c for c in config['checks'] if c['id'] == row['id'])
            self.assertEqual(row['rules'], source['rules'])
            self.assertEqual(row['arguments'], source['args'] + source.get('profile_args', {}).get('full', []))
        verify = next(c for c in description['checks'] if c['id'] == 'verify_exact')
        self.assertIn('file: verify_ar', verify['prerequisites'])
        self.assertFalse(description['qualified'])
        self.assertEqual(description['execution'], 'not_run')

    def test_plan_cli_requires_no_server_inputs_or_receipt_directory(self):
        with patch('sys.argv', ['release_gate.py', '--model', 'qwen38', '--profile', 'full', '--plan']), \
             patch.object(gate, 'run', side_effect=AssertionError('execution')), \
             patch.object(gate, 'input_path', side_effect=AssertionError('prerequisite resolution')), \
             patch.object(gate.Client, 'identity', side_effect=AssertionError('HTTP')), \
             patch.object(gate.subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(gate.importlib, 'import_module', side_effect=AssertionError('worker import')), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(gate.main(), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result['cohort'], 'full')
        self.assertEqual(result['selected_checks'], [c['id'] for c in result['checks']])

    def test_model_selector_accepts_model_and_variant_names_and_refuses_paths(self):
        with patch.object(gate.declaration, 'named', return_value=gate.declaration.MECHANICS) as named:
            for name in ('demo-lab-fp4', 'demo_2', 'demo.variant'):
                with self.subTest(name), patch('sys.argv', ['release_gate.py', f'--model={name}', '--plan']), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(gate.main(), 0)
                    named.assert_called_with(name)
            for name in ('../demo', 'demo/exports', 'demo.json.x', '-demo', ''):
                with self.subTest(name), patch('sys.argv', ['release_gate.py', f'--model={name}', '--plan']), \
                        contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                    gate.main()
                self.assertIn('--model must name a model declaration', error.getvalue())
            self.assertEqual(named.call_count, 3)

    def test_qwen_diagnostic_coverage_preserves_receipt_admission_and_predicates(self):
        import qwen38_layers
        import frontend_goldens
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {key: str(root) for key in ('layer_dump', 'layer_reference', 'capture')}
            layer_reference = root / 'unbound-reference'
            layer_reference.mkdir()
            inputs['layer_reference'] = str(layer_reference)
            fixtures = {'needles.json': [{'name': str(i), 'expected': '7',
                                         'response': {'choices': [{'text': '7'}]}} for i in range(3)],
                        'api-samples.json': {str(i): {'status': 200, 'body': {'choices': [{'text': 'answer'}]}}
                                             for i in range(15)},
                        'chat-renders.json': [{'token_ids': [1]} for _ in range(75)],
                        'tokenizer-golden.json': [{'token_ids': [1]} for _ in range(81)]}
            for name, value in fixtures.items():
                (root / name).write_text(json.dumps(value))
            capture_path = root / 'compared.json'
            with capture_fixture(root) as reference, \
                 patch('sys.argv', ['compare', str(root), '--reference', str(reference), '--json', str(capture_path)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(frontend_goldens.main(), 0)
            capture = json.loads(capture_path.read_text())
            # The actual worker against a run without taps must not admit.
            from test_qwen38_layers import write_dump
            write_dump(root,'layer00.attn.in',[[1.]*2560],2560)
            layers = None
            config = declaration.load(declaration.path('qwen38'))
            config['inputs']['layer_reference'] = str(layer_reference)
            config['checks'] = [c for c in config['checks'] if c['id'] in ('qwen38_layers', 'qwen38_capture')]
            for profile in config['profiles'].values():
                profile['checks'] = ['qwen38_layers', 'qwen38_capture']
            path = root / 'policy.json'
            path.write_text(json.dumps(config))
            for insufficient in (False, True):
                reports = {'qwen38_layers': layers, 'qwen38_capture': dict(capture)}
                if insufficient:
                    reports['qwen38_capture'].pop('tokenizer-golden.json')
                run_process = gate.subprocess.run
                def execute(command, **kwargs):
                    if '--worker' not in command:
                        return run_process(command, **kwargs)
                    if command[5] == 'qwen38_layers':
                        worker=[sys.executable,'-B',str(gate.ROOT/'tools/qual/qwen38_layers.py'),
                                '--dump',str(root),'--reference',str(layer_reference),
                                '--json',command[command.index('--json')+1]]
                        return run_process(worker,**kwargs)
                    value = reports[command[5]]
                    Path(command[command.index('--json') + 1]).write_text(json.dumps(value))
                    return subprocess.CompletedProcess(command, 0)
                from qual_test_config import fixture_lock
                with fixture_lock(config), \
                     patch.object(gate.Client, 'identity', return_value={'model': 'fixture'}), \
                     patch.object(gate.subprocess, 'run', side_effect=execute), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = gate.run(config, path, 'full', 'http://unused', None, inputs, root)
                summary = json.loads(sorted(root.glob('*/summary.json'))[-1].read_text())
                self.assertEqual(result,1,summary)
                row=next(row for row in summary['checks'] if row['id']=='qwen38_layers')
                self.assertFalse(row['passed'],row)

    def test_shipped_configs_have_sources_and_coverage(self):
        # Every declaration is checked, so a new model's gate needs no entry
        # here. Every released model (its model.toml) has its own declaration;
        # each model's test_inventory.py pins whether it is released.
        paths = declaration.shipped()
        self.assertEqual(paths[0], declaration.MECHANICS)
        self.assertEqual([path.parent.name for path in paths[1:] if path.name == declaration.NAME and released(path)],
                         list(inventory.released(declaration.ROOT)))
        for path in paths:
            with self.subTest(gate=str(path)):
                config = declaration.load(path)
                gate.validate(config)
                self.assertTrue(set(config['profiles']['quick']['checks']) <= set(config['profiles']['full']['checks']))
                for check in config['checks']:
                    for rule in check['rules']:
                        self.assertTrue((gate.ROOT / rule['source'].split('#')[0]).is_file(), rule['source'])

    @staticmethod
    def teacher_declaration(configured=False):
        """The first shipped model declaration with a teacher check (and,
        when `configured`, with a configuration that runs it)."""
        return next(path for path in declaration.shipped()[1:]
                    if path.name == declaration.NAME and 'teacher' in json.loads(path.read_text())['checks']
                    and (not configured or any('teacher' in spec.get('checks', []) for spec in
                                               declaration.configurations(path.parent.name).values())))

    def eval_run(self, change=None, configuration=None):
        """Runs the teacher check of the first shipped model declaration with
        one (or of its `configuration`), full cohort, through run_check with
        a fake `chopinfer eval run`; the row, the eval commands it ran, the
        run directory and the model's directory."""
        path = self.teacher_declaration(configured=configuration is not None)
        config = declaration.load(path if configuration is None else
                                  declaration.named(f'{path.parent.name}.{configuration}'))
        check = next(check for check in config['checks'] if check['id'] == 'teacher')
        if change:
            check = change(deepcopy(check))
        commands = []
        run_process = gate.subprocess.run

        def execute(command, **kwargs):
            if command[1:3] != ['eval', 'run']:
                return run_process(command, **kwargs)
            commands.append(command)
            report = Path(command[command.index('--report') + 1])
            (report / 'records').mkdir(parents=True)
            (report / 'records/teacher.json').write_text(json.dumps({'passed': True}))
            (report / 'captures').mkdir()
            (report / 'captures/teacher.json').write_bytes(b'{"capture": 1}\n')
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'checks').mkdir()
            policy = root / 'policy.json'
            policy.write_text(json.dumps(config))
            context = {'url': 'http://127.0.0.1:1', 'model': 'served', 'server_log': str(root / 'server.log'),
                       'raw': str(root / 'checks'), 'run': str(root), 'run_id': 'run'}
            with patch.object(gate.subprocess, 'run', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                row = gate.run_check(check, [], 'full', context, root, policy, float('inf'), False, None)
            return row, commands, root, path.parent

    def test_a_model_declaration_s_eval_captured_check_runs_through_eval(self):
        row, commands, root, model = self.eval_run()
        report = root / 'checks/eval-teacher'
        self.assertEqual(commands, [[str(gate.kinds.eval_binary()), 'eval', 'run', '--model', str(model),
                                     '--served-model', 'served', '--check', 'teacher', '--cohort', 'full',
                                     '--url', 'http://127.0.0.1:1', '--server-log', str(root / 'server.log'),
                                     '--report', str(report)]])
        self.assertEqual(row['receipt'], str(report / 'records/teacher.json'))
        self.assertEqual(row['eval_report'], str(report / 'report.json'))

    def test_a_configuration_s_check_runs_under_its_variant(self):
        model = self.teacher_declaration(configured=True).parent.name
        configuration = next(name for name, spec in declaration.configurations(model).items()
                             if 'teacher' in spec.get('checks', []))
        row, commands, root, directory = self.eval_run(configuration=configuration)
        self.assertEqual(len(commands), 1, row)
        command = commands[0]
        self.assertEqual(command[command.index('--model') + 1:command.index('--model') + 4],
                         [str(directory), '--variant', configuration])
        self.assertEqual(row['receipt'], str(root / 'checks/eval-teacher/records/teacher.json'))

    def test_eval_s_capture_goes_to_the_declared_capture_out_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, target = Path(temporary) / 'report', Path(temporary) / 'teacher-capture.json'
            (report / 'records').mkdir(parents=True)
            (report / 'records/teacher.json').write_text('{}')
            check = {'id': 'teacher'}
            with self.assertRaisesRegex(ValueError, 'wrote no capture of teacher'):
                gate.eval_record(check, ['--capture-out', str(target)], report)
            (report / 'captures').mkdir()
            (report / 'captures/teacher.json').write_bytes(b'{"capture": 1}\n')
            self.assertEqual(gate.eval_record(check, ['--capture-out', str(target)], report),
                             report / 'records/teacher.json')
            self.assertEqual(target.read_bytes(), b'{"capture": 1}\n')
            (report / 'records/teacher.json').unlink()
            with self.assertRaisesRegex(ValueError, 'wrote no record of teacher'):
                gate.eval_record(check, [], report)

    def test_eval_refuses_a_policy_check_the_declaration_does_not_state(self):
        def widened(check):
            check['args'] = [*check['args'], '--texts', 'a']
            return check
        row, commands, _, _ = self.eval_run(widened)
        self.assertEqual(commands, [])
        self.assertFalse(row['passed'])
        self.assertIn('declares another teacher check than this run', row['error'])

    def test_structured_length_finish_still_fails_the_gate(self):
        import structured
        client = unittest.mock.Mock()
        client.request.return_value = ({'choices': [{'message': {'content': '{}'},
                                                   'finish_reason': 'length'}]}, 1.0)
        row = structured.run_case(client, 'chat_so_json_object', '/v1/chat/completions',
                                  {}, structured.check_json(None), False)
        self.assertEqual(row['status'], 'truncated')
        config = declaration.load(declaration.path('mimo26'))
        check = next(check for check in config['checks'] if check['id'] == 'structured')
        rule = next(rule for rule in check['rules'] if rule['path'][-1] == 'truncated')
        self.assertFalse(gate.judge({'summary': {'overall': structured.summarize([row])}}, rule))

    def test_empty_missing_and_nonfinite_metrics_cannot_pass(self):
        rule = dict(path=['rows', '*', 'score'], op='le', value=0.15)
        self.assertFalse(gate.judge({'rows': []}, rule))
        for value in (None, float('nan'), float('inf'), True, .151):
            self.assertFalse(gate.judge({'rows': [{'score': value}]}, rule))
        self.assertTrue(gate.judge({'rows': [{'score': .15}]}, rule))
        with self.assertRaises(KeyError):
            gate.judge({'rows': [{}]}, rule)

    def exercise(self, mode='pass', mechanics=False, declared=False, model='fixture', external=False, profile='full'):
        config = json.loads(declaration.MECHANICS.read_text())
        config['qualification']['mechanics_only'] = declared
        config['model'] = model
        config['checks'][0]['args'] = []
        config['checks'][1]['args'] = []
        calls = []
        if mode in ('informational', 'informational_failure'):
            config['checks'][0]['informational_reason'] = 'RL capture, MOPD alpha'
        if mode == 'missing_input':
            config['checks'][0]['args'] = ['{required_capture}']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            def execute(command, **kwargs):
                if external:
                    self.assertEqual(kwargs['env']['CHOPINFER_QUAL_MODEL_ROOT'], str(root))
                name = command[5]
                calls.append(name)
                if mode == 'missing_receipt':
                    return subprocess.CompletedProcess(command, 0)
                if mode == 'timeout':
                    raise subprocess.TimeoutExpired(command, 1)
                if mode == 'exit':
                    return subprocess.CompletedProcess(command, 2)
                if mode == 'wrong_shape':
                    value = '[]'
                elif mode == 'malformed':
                    value = '{broken'
                else:
                    failed = mode in ('failed', 'informational_failure') or (mode == 'api_fails' and name == 'api')
                    receipt = {'passed': not failed,
                               'skipped': ['quality'] if mode == 'skipped' else [],
                               'mechanics_only': mechanics}
                    if mode == 'not_applicable' and name == 'api':
                        receipt['not_applicable'] = ['the server does not serve image input (--modalities text)']
                    value = json.dumps(receipt)
                receipts = Path(kwargs['env']['CHOPINFER_QUAL_RECEIPTS'])
                (receipts / ('fresh-' + name + '.json')).write_text(value)
                return subprocess.CompletedProcess(command, 0)
            with patch.object(gate.Client, 'identity', return_value={'model': 'fixture/tp4'}), \
                 patch.object(gate.subprocess, 'check_output', return_value='revision\n'), \
                 patch.object(gate.subprocess, 'run', side_effect=execute), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = gate.run(config, path, profile, 'http://unused', None, {}, root, mechanics,
                                  model_root=root if external else None)
            summary = json.loads(next(root.glob('*/summary.json')).read_text())
            self.assertTrue(next(root.glob('*/summary.md')).is_file())
            return status, summary, calls

    def test_scorer_failures_write_failed_summaries(self):
        import kinds
        for message in ('scorer is missing', 'scorer crashed'):
            with self.subTest(message=message), patch.object(kinds, 'evaluated', side_effect=RuntimeError(message)):
                status, summary, _ = self.exercise()
                self.assertEqual(status, 1)
                self.assertFalse(summary['passed'])
                self.assertFalse(summary['qualified'])
                self.assertEqual(summary['checks'][0]['error'], message)
                self.assertIn('seconds', summary['checks'][0])

    def test_all_checks_pass_and_full_qualifies(self):
        status, summary, calls = self.exercise()
        self.assertEqual(status, 0)
        self.assertTrue(summary['qualified'])
        self.assertEqual(calls, ['api', 'structured'])
        self.assertEqual(summary['not_run'], [])

    def test_informational_reference_is_not_a_pass_or_full_qualification(self):
        status, summary, calls = self.exercise('informational')
        self.assertEqual(status, 0)
        self.assertEqual(calls, ['structured'])
        self.assertFalse(summary['qualified'])
        self.assertIsNone(summary['checks'][0]['passed'])
        self.assertEqual(summary['checks'][0]['status'], 'informational')
        self.assertFalse(summary['checks'][0]['evaluated'])
        self.assertEqual(self.exercise('informational_failure')[0], 1)
    def test_an_unbound_modality_claim_cannot_bypass_the_gate(self):
        status, summary, calls = self.exercise('not_applicable')
        self.assertEqual(calls, ['api'])
        self.assertEqual(status, 1)
        self.assertFalse(summary['qualified'])
        self.assertIs(summary['checks'][0]['passed'], False)
        self.assertIn('not_applicable needs a qualification receipt', summary['checks'][0]['error'])

    def test_served_modalities_come_from_health(self):
        server = lambda health: {'model': 'M', 'health': health}
        self.assertIsNone(common.served_modalities(server(None)))
        self.assertIsNone(common.served_modalities(server({'status': 'ready'})))
        self.assertIsNone(common.not_applicable(server({'status': 'ready'}), 'video'))
        served = server({'status': 'ready', 'modalities': ['text', 'audio']})
        self.assertEqual(common.served_modalities(served), {'text', 'audio'})
        self.assertIsNone(common.not_applicable(served, 'audio'))
        self.assertEqual(common.not_applicable(served, 'image'),
                         'the server does not serve image input (--modalities text,audio)')

    def test_mechanics_declaration_cannot_qualify_without_flag(self):
        for model in ('fixture', 'independent-example'):
            with self.subTest(model=model):
                status, summary, calls = self.exercise(declared=True, model=model)
                self.assertEqual(status, 0)
                self.assertTrue(summary['passed'])
                self.assertFalse(summary['qualified'])
                self.assertTrue(summary['mechanics_only'])
                self.assertEqual(calls, ['api', 'structured'])

    def test_external_mechanics_profiles_keep_their_input_root(self):
        for profile in ('quick', 'full'):
            for flag in (False, True):
                with self.subTest(profile=profile, flag=flag):
                    status, summary, _ = self.exercise(declared=True, mechanics=flag,
                                                       external=True, profile=profile)
                    self.assertEqual(status, 0)
                    self.assertFalse(summary['qualified'])
                    self.assertTrue(summary['mechanics_only'])
                    self.assertIn('model_revision', summary['provenance'])
                    self.assertIn('sdk_revision', summary['provenance'])

    def test_external_marker_and_module_validation(self):
        config = json.loads(declaration.MECHANICS.read_text())
        for value in ('true', 1, None, []):
            config['qualification']['mechanics_only'] = value
            with self.assertRaisesRegex(ValueError, 'must be boolean'):
                gate.validate(config, external=True)
        del config['qualification']['mechanics_only']
        with self.assertRaisesRegex(ValueError, 'must declare'):
            gate.validate(config, external=True)
        gate.validate(config)
        config['qualification']['mechanics_only'] = True
        for module in ('../common', 'os'):
            config['checks'][0]['module'] = module
            with self.assertRaisesRegex(ValueError, 'existing qualification module'):
                gate.validate(config, external=True)

    def test_external_cli_keeps_explicit_declaration_with_or_without_flag(self):
        config = json.loads(declaration.MECHANICS.read_text())
        config['model'] = 'independent-example'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'renamed.json'
            path.write_text(json.dumps(config))
            for flag in ([], ['--mechanics-only']):
                args = ['release_gate.py', '--config', str(path), '--model-root', str(root),
                        '--url', 'http://unused', '--receipts', str(root / 'receipts'), *flag]
                with patch.object(sys, 'argv', args), patch.object(gate, 'run', return_value=0) as run:
                    self.assertEqual(gate.main(), 0)
                self.assertEqual(run.call_args.args[0]['model'], 'independent-example')
                self.assertEqual(run.call_args.kwargs['model_root'], root)

    def test_external_input_paths_survive_copied_policy(self):
        config = {'inputs': {'reference': 'inputs/reference.json'}, 'receipts': 'receipts'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / 'logs/policy.json'
            frozen.parent.mkdir()
            frozen.write_text(json.dumps(config))
            self.assertEqual(common.input_path('reference', config=config, root=root),
                             root / 'inputs/reference.json')
            with patch.dict(os.environ, {'CHOPINFER_QUAL_MODEL_ROOT': str(root),
                                         'CHOPINFER_QUAL_CONFIG': str(frozen)}):
                self.assertEqual(common.input_path('reference'), root / 'inputs/reference.json')

    def test_mechanics_never_qualifies(self):
        status, summary, _ = self.exercise(mechanics=True)
        self.assertEqual(status, 0)
        self.assertFalse(summary['qualified'])

    def test_fail_closed_and_write_summary(self):
        for mode in ('failed', 'exit', 'timeout', 'malformed', 'skipped', 'missing_receipt', 'wrong_shape'):
            with self.subTest(mode=mode):
                status, summary, calls = self.exercise(mode)
                self.assertEqual(status, 1)
                self.assertFalse(summary['qualified'])
                self.assertEqual(calls, ['api'])
                self.assertEqual(summary['not_run'], ['structured'])

    def test_quick_runs_every_check_after_a_failure(self):
        status, summary, calls = self.exercise('api_fails', profile='quick')
        self.assertEqual(status, 1)
        self.assertFalse(summary['passed'])
        self.assertEqual(calls, ['api', 'structured'])
        self.assertEqual(summary['not_run'], [])
        self.assertEqual([(row['id'], row['passed']) for row in summary['checks']],
                         [('api', False), ('structured', True)])
        # Full still stops at the first failure.
        status, summary, calls = self.exercise('api_fails')
        self.assertEqual((status, calls, summary['not_run']), (1, ['api'], ['structured']))

    def test_missing_preflight_input_writes_failed_summary(self):
        status, summary, calls = self.exercise('missing_input')
        self.assertEqual(status, 1)
        self.assertEqual(calls, [])
        self.assertIn('required_capture', summary['error'])
        self.assertEqual(summary['not_run'], ['api', 'structured'])

    def test_real_subprocess_worker_reuses_structured_validator(self):
        config = json.loads(declaration.MECHANICS.read_text())
        check = config['checks'][1]
        check['args'] = ['--self-test']
        check.pop('receipt')
        check['stdout_json'] = True
        check['rules'] = [dict(path=['self_test'], op='eq', value='passed',
                               source='docs/serving-request-surface.md')]
        config['checks'] = [check]
        for profile in config['profiles'].values():
            profile['checks'] = ['structured']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            with patch.object(gate.Client, 'identity', return_value={'model': 'fixture/tp4'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = gate.run(config, path, 'quick', 'http://unused', None, {},
                                  Path(temporary), mechanics=True)
            summary = json.loads(next(Path(temporary).glob('*/summary.json')).read_text())
        self.assertEqual(status, 0)
        self.assertEqual(summary['checks'][0]['exit_code'], 0)
        self.assertFalse(summary['qualified'])

    def test_remote_soak_never_calls_local_probes(self):
        with patch('sys.argv', ['soak.py', '--remote-only', '--duration', '1', '--concurrency', '1']), \
             patch.object(soak, 'serving_pid', side_effect=AssertionError('local PID probe')), \
             patch.object(soak, 'device_memory', side_effect=AssertionError('GPU probe')), \
             patch.object(soak.Client, 'identity', return_value={'model': 'fixture'}), \
             patch.object(soak.Client, 'request', return_value=({'choices': [{}]}, .001)), \
             patch.object(soak, 'receipt') as receipt:
            with self.assertRaises(SystemExit) as result:
                soak.main()
        self.assertEqual(result.exception.code, 0)
        value = receipt.call_args.args[2]
        self.assertGreater(value['completed'], 0)
        self.assertEqual(value['memory_scope'], 'not_sampled_remote')
        self.assertIsNone(value['memory_growth_mib'])

    def test_the_soak_mixes_only_served_media(self):
        requested = []
        def media(case_ids):
            requested.append(list(case_ids))
            return [(case_id, [{'role': 'user', 'content': 'x'}]) for case_id in case_ids]
        health = {'status': 'ready', 'modalities': ['text', 'image']}
        answer = {'choices': [{'message': {'content': 'Blue.'}}]}
        with patch('sys.argv', ['soak.py', '--remote-only', '--media-mix', '--duration', '1',
                                '--concurrency', '1']), \
             patch.object(common, 'startup'), \
             patch.object(soak, 'answer_control', return_value={'chat_template_kwargs': {}}), \
             patch.object(soak, 'thinking_can_be_disabled', return_value=True), \
             patch.object(soak, 'media_requests', side_effect=media), \
             patch.object(soak.Client, 'identity', return_value={'model': 'fixture', 'health': health}), \
             patch.object(soak.Client, 'request', return_value=(answer, .001)), \
             patch.object(soak, 'receipt') as receipt:
            with self.assertRaises(SystemExit) as result:
                soak.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(requested, [['image-blue-square.png']])
        value = receipt.call_args.args[2]
        self.assertEqual(value['media_cases'], ['image-blue-square.png'])
        self.assertEqual(value['not_applicable'], [
            'the server does not serve video input (--modalities text,image)'])

    def test_a_not_applicable_receipt_rescores_as_not_applicable(self):
        import checks
        import kinds
        identity = 'a' * 64
        reason = 'the server does not serve image input (--modalities text)'
        value = {'schema': 'chopinfer-qualification/v1', 'passed': False, 'not_applicable': [reason],
                 'server': {'health': {'modalities': ['text'], 'serving': {
                     'checkpoint_identity': identity, 'startup_nonce': 'a' * 32}}}}
        with patch('common.checkpoint_pin', return_value={'checkpoint_identity': identity}):
            result = kinds.rescore(kinds.RECEIPT, value, None,
                                   {'module': 'grounding', 'rules': [{'path': ['passed'], 'op': 'eq', 'value': True}]})
        self.assertEqual((result['status'], result['reason']), ('not_applicable', reason))
        self.assertIn('not_applicable', kinds.STATUSES)
        self.assertEqual(checks.recorded_status({'status': 'not_applicable', 'passed': None}), 'not_applicable')


# A synthetic model whose teacher checks `chopinfer eval run` captures from a
# fake server, as crates/runtime-eval/tests/teacher.rs builds them: 23 scored
# positions with one near-tie flip, a top-1 disagreement of 1/23.
CHECKPOINT = '4' * 64
NONCE = '0123456789abcdef0123456789abcdef'


class CheckRawGateTests(unittest.TestCase):
    """The whole-cohort gate gives every check one raw directory; a later
    check names an earlier one's files by the producer's check-scoped name,
    `{check_raw[ID]}`, which resolves there too, and only once that check
    passed. The per-check path is in test_release_preview's CheckRawTests."""

    def cohort(self, failing=(), fail_again=(), order=None):
        from test_release_preview import capture_worker, producer_and_reader
        read = []
        config = producer_and_reader()
        if order is not None:
            config['profiles']['quick']['checks'] = order
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(gate.Client, 'identity', return_value={'model': 'fixture/tp4'}), \
                    patch.object(gate.subprocess, 'check_output', return_value='revision\n'), \
                    patch.object(gate.subprocess, 'run', side_effect=capture_worker(read, failing, fail_again)), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = gate.run(config, declaration.MECHANICS, 'quick', 'http://unused', None, {}, root, True)
            summary = json.loads(next(root.glob('*/summary.json')).read_text())
            raw = next(root.glob('*/checks'))
            self.rows = summary['checks']
            return status, {row['id']: row for row in summary['checks']}, read, raw

    def test_a_later_check_reads_the_capture_its_producer_wrote(self):
        status, rows, read, raw = self.cohort()
        self.assertEqual(status, 0, rows)
        self.assertEqual(read, [raw / 'capture.json'])
        self.assertEqual(rows['structured']['arguments'][-3], str(raw / 'capture.json'))

    def test_a_reader_after_a_failed_producer_is_refused(self):
        status, rows, read, _ = self.cohort(failing=('api',))
        self.assertEqual(status, 1)
        self.assertFalse(rows['api']['passed'])
        self.assertIn('no raw directory of check api', rows['structured']['error'])
        self.assertEqual(read, [])

    def test_a_producer_run_again_that_fails_withdraws_its_capture(self):
        # Only a passing run publishes; a second run of the producer
        # withdraws the first's receipt and raw directory.
        status, _, read, _ = self.cohort(fail_again=('api',), order=['api', 'api', 'structured'])
        self.assertEqual(status, 1)
        self.assertEqual([row['passed'] for row in self.rows], [True, False, False])
        self.assertIn('no raw directory of check api', self.rows[2]['error'])
        self.assertEqual(read, [])


def declared_configs():
    """Every shipped declaration and each configuration a model's qual.json
    declares, materialized as release_preview and checks.py read them."""
    for path in declaration.shipped():
        yield path
        if path.name == declaration.NAME:
            for name in declaration.configurations(path.parent.name):
                yield declaration.materialized(path.parent.name, name)


class RawFileTests(unittest.TestCase):
    def test_no_check_reads_another_check_s_raw_file_through_its_own_raw(self):
        # GPU pass 1 (C02): a configuration's teacher_arms read its teacher's
        # capture as {raw}/teacher-capture.json, which is the teacher's file
        # only when one gate run shares {raw}; release_preview's per-check
        # runs each have their own. A file another check writes is named by
        # that check's {check_raw[ID]}.
        import re
        checked = 0
        for path in declared_configs():
            config = declaration.load(path)
            named = {}
            for check in config['checks']:
                text = json.dumps([check.get('args', []), check.get('profile_args', {}), check.get('cohort_args', {})])
                for name in set(re.findall(r'\{raw\}/([\w.-]+)', text)):
                    named.setdefault(name, []).append(check['id'])
            for name, users in named.items():
                checked += 1
                with self.subTest(config=str(path), file=name):
                    self.assertEqual(len(users), 1, f'{{raw}}/{name} is named by {users}: a check reads a file '
                                                    f'another wrote as {{check_raw[ID]}}/{name}')
        self.assertGreater(checked, 0)


def ranked(token, first, gap):
    return [[token, first], [token + 100, first - gap]] + [
        [token + other, first - gap - 1.0 - index] for index, other in enumerate(range(101, 104))]


def variant(ids, start, first, gap):
    positions = ids[start:]
    rows = [ranked(token, first, gap) for token in positions]
    column = lambda index: [[pair[index] for pair in row] for row in rows]
    return {'logprob': [first] * len(positions), 'top1': positions, 'margin': [gap] * len(positions),
            'topk_ids': column(0), 'topk_logits': column(1), 'topk_logprobs': column(1)}


def served_row(token, position):
    shift = 0.015625 * (position % 4)
    lead, second = (token + 100, token) if token == 5 else (token, token + 100)
    return {str(lead): {'logprob': -0.1875 - shift, 'rank': 1},
            str(second): {'logprob': -0.4375 - shift, 'rank': 2},
            str(token + 101): {'logprob': -2.5, 'rank': 3}}


@contextlib.contextmanager
def teacher_server(answered=NONCE):
    """A fake server's URL. Its answers to requests with a body name the
    startup nonce `answered`, while /health names its own."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def answer(self, body):
            if self.path == '/v1/models':
                value, nonce = {'data': [{'id': 'Example'}]}, NONCE
            elif self.path == '/health':
                value, nonce = {'status': 'ready', 'serving': {'checkpoint_identity': CHECKPOINT,
                                                               'startup_nonce': NONCE}}, NONCE
            else:
                ids = body['prompt']
                rows = [None] + [served_row(token, position) for position, token in enumerate(ids) if position]
                value, nonce = {'choices': [{'index': 0, 'text': ' x', 'finish_reason': 'length',
                                             'prompt_token_ids': ids, 'prompt_logprobs': rows,
                                             'logprobs': {'token_logprobs': [None], 'tokens': [' x']}}],
                                'usage': {'prompt_tokens': len(ids), 'completion_tokens': 1}}, answered
            text = json.dumps(value).encode()
            self.send_response(200)
            for key, header in (('content-type', 'application/json'), ('content-length', str(len(text))),
                                ('x-chopinfer-checkpoint-identity', CHECKPOINT),
                                ('x-chopinfer-startup-nonce', nonce), ('connection', 'close')):
                self.send_header(key, header)
            self.end_headers()
            self.wfile.write(text)

        def do_GET(self):
            self.answer(None)

        def do_POST(self):
            self.answer(json.loads(self.rfile.read(int(self.headers['content-length']))))

        def log_message(self, *arguments):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def teacher_root(root, top1_limit=0.1):
    """A model root with the example model's declaration, its teacher
    reference and the adopted reference lock; the declaration's path. Its
    teacher and long_context checks hold top1_disagreement to
    `top1_limit`."""
    import hashlib
    import refs
    model, fixtures = root / 'models/example', root / 'fixtures/example'
    model.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    (root / 'tools/qual').mkdir(parents=True)
    (root / 'tools/qual/modules.json').write_bytes((gate.ROOT / 'tools/qual/modules.json').read_bytes())
    lock = {'model.safetensors': {'sha256': 'ab', 'size': 1}}
    (model / 'checkpoint-files.json').write_text(json.dumps(lock))
    checkpoint = {'repository': 'org/example', 'revision': 'abc', 'served_model_name': 'Example',
                  'source_lock_sha256': hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(',', ':'))
                                                       .encode()).hexdigest(),
                  'checkpoint_identity': CHECKPOINT}
    texts = [{'name': name, 'category': category, 'prompt_token_ids': ids, 'score_from': start,
              'exact': variant(ids, start, -0.25, 0.25), 'bf16': variant(ids, start, -0.375, 0.125)}
             for name, category, ids, start in (('t0', 'prose', list(range(1, 16)), 1),
                                                ('t1', 'code', [1, *range(30, 40)], 2))]
    reference = json.dumps({'schema': 'example.teacher.v1', 'texts': texts, 'provenance': {
        'checkpoint': checkpoint, 'variants': {'exact': 'f64', 'bf16': 'bf16'}}}).encode()
    (fixtures / 'teacher.json').write_bytes(reference)
    (fixtures / 'teacher-texts.json').write_text(json.dumps({'texts': [{'name': 't0'}, {'name': 't1'}]}))
    (model / 'frontend.json').write_text(json.dumps({'reasoning': {'thinking_keywords': ['enable_thinking']}}))
    rules = [{'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'docs/example.md'},
             {'path': ['top1_disagreement'], 'op': 'le', 'value': top1_limit, 'source': 'docs/example.md'}]
    arguments = ['--url', '{url}', '--model', '{model}']
    value = {
        'schema': 'chopinfer-qual-model/v1', 'model': 'example',
        'cohorts': {'quick': {'checks': ['teacher', 'long_context']},
                    'full': {'checks': ['teacher', 'long_context']}},
        'change_classes': {'frontend': [], 'quality': [], 'service': [], 'speculation': [], 'prefix': []},
        'checks': {
            'teacher': {'module': 'teacher', 'args': arguments,
                        'cohort_args': {'full': ['--capture-out', '{raw}/teacher-capture.json']},
                        'timeout_seconds': 60, 'receipt': '*-teacher.json', 'rules': rules},
            'long_context': {'module': 'teacher', 'args': [*arguments, '--receipt', 'long-context'],
                             'timeout_seconds': 60, 'receipt': '*-long-context.json', 'rules': rules}},
        'references': {'teacher': {'path': 'fixtures/example/teacher.json',
                                   'sha256': hashlib.sha256(reference).hexdigest(),
                                   'schema': 'example.teacher.v1', 'variants': ['exact', 'bf16']}},
        'inputs': {'fixture_root': 'fixtures/example', 'teacher_reference': 'fixtures/example/teacher.json',
                   'exact': 'fixtures/example/teacher.json', 'frontend': 'models/example/frontend.json'},
        'qualification': {'teacher_reference_kind': 'exact', 'teacher_receipt': 'teacher',
                          'teacher_capture_schema': 'example.capture.v1', 'verify_buckets': [2],
                          'chars_per_token': 4.0,
                          'checkpoint': {**checkpoint, 'lock': str(model / 'checkpoint-files.json')}},
        'receipts': {'env': 'RUNTIME_EVAL_TEST_UNSET', 'path': 'qual/example'}}
    source = model / 'qual.json'
    source.write_text(json.dumps(value))
    with patch.dict(os.environ, CHOPINFER_QUAL_MODEL_ROOT=str(root)):
        refs.lock_path('example', root).write_text(json.dumps(refs.snapshot(declaration.load(source), root)))
    return source


class EvalVerdictTests(unittest.TestCase):
    """Declared checks run through the real `chopinfer eval run` (the
    CHOPINFER_EVAL binary) against a fake server, then scored and rescored
    as the runner scores them."""

    def run_declared(self, name, top1_limit=0.1, answered=NONCE):
        """The row of check `name` of the example model, full cohort, run
        by run_check; the row, the declaration and its path, the policy and
        the run directory, under a temporary root removed at cleanup."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = teacher_root(root, top1_limit)
        environment = patch.dict(os.environ, CHOPINFER_QUAL_MODEL_ROOT=str(root),
                                 CHOPINFER_QUAL_CONFIG=str(source))
        environment.start()
        self.addCleanup(environment.stop)
        config = declaration.load(source)
        check = next(check for check in config['checks'] if check['id'] == name)
        directory = root / 'runner'
        (directory / 'checks').mkdir(parents=True)
        policy = directory / 'policy.json'
        policy.write_text(json.dumps(config))
        with teacher_server(answered) as url:
            log = root / 'server.log'
            compact = lambda value: json.dumps(value, separators=(',', ':'))
            log.write_text(f"startup serving_identity: {compact({'checkpoint_identity': CHECKPOINT, 'startup_nonce': NONCE})}\n"
                           f"{compact({'status': 'listening', 'address': url.removeprefix('http://')})}\n")
            context = {'url': url, 'model': 'Example', 'server_log': str(log), 'raw': str(directory / 'checks'),
                       'run': str(directory), 'run_id': 'run'}
            with contextlib.redirect_stdout(io.StringIO()):
                row = gate.run_check(check, [], 'full', context, directory, policy, float('inf'), False, root)
        return row, config, source, policy, directory

    def rescore_receipt(self, path, config, policy):
        """checks.py rescore of the single receipt `path`, as the runner's
        replay runs it, against the example declaration."""
        import checks
        with patch.object(checks, 'load', return_value=(config, policy)), \
                patch.object(declaration, 'renamed', return_value={}):
            return checks.rescore_receipt(Path(path), json.loads(Path(path).read_text()), 'example', 'current')

    def test_a_passing_long_context_record_rescores_as_its_declared_check(self):
        row, config, _, policy, directory = self.run_declared('long_context')
        self.assertTrue(row['passed'], row)
        receipt = directory / 'checks/eval-long_context/records/long_context.json'
        self.assertEqual(row['receipt'], str(receipt))
        value = json.loads(receipt.read_text())
        # The record keeps the receipt name the capture declares as its check.
        self.assertEqual((value['schema'], value['check']), ('chopinfer-qualification/v1', 'long-context'))
        rescored = self.rescore_receipt(receipt, config, policy)
        self.assertEqual((rescored['id'], rescored['recorded_id'], rescored['status']),
                         ('long_context', 'long-context', 'pass'), rescored)
        self.assertEqual(rescored['metrics']['top1_disagreements'], 1)

    def rescore_row(self, row, config, policy, directory):
        """checks.py's rescore of the gate row `row`, as a gate run's rescore
        judges it against the example declaration."""
        import checks
        check = next(check for check in config['checks'] if check['id'] == row['id'])
        return checks.rescore_row(directory / 'summary.json', row, {**check, '_cohort': 'full'}, config, policy,
                                  False, None)

    def test_a_failed_declared_predicate_keeps_its_receipt_capture_and_classification(self):
        # The top-1 disagreement of 1/23 fails a declared limit of 0: eval
        # completes its verdict and exits 1.
        row, config, _, policy, directory = self.run_declared('teacher', top1_limit=0)
        report = directory / 'checks/eval-teacher'
        self.assertEqual(json.loads((report / 'report.json').read_text())['exit'], 1)
        self.assertEqual((row['passed'], row['exit_code'], row.get('error')),
                         (False, 1, 'threshold predicate failed'), row)
        self.assertEqual(row['receipt'], str(report / 'records/teacher.json'))
        self.assertEqual(row['sha256'], gate.digest(report / 'records/teacher.json'))
        self.assertEqual([predicate['passed'] for predicate in row['predicates']], [True, False])
        self.assertEqual((directory / 'checks/teacher-capture.json').read_bytes(),
                         (report / 'captures/teacher.json').read_bytes())
        # Rescored as the gate judged it, with every metric; the replay's
        # single-receipt rescore judges the same record.
        rescored = self.rescore_row(row, config, policy, directory)
        self.assertEqual((rescored['status'], rescored['receipt']), ('fail', row['receipt']), rescored)
        self.assertNotIn('receipt_status', rescored)
        self.assertEqual(rescored['metrics']['top1_disagreements'], 1)
        self.assertEqual(rescored['metrics'], {key: rescored['recorded_metrics'][key] for key in rescored['metrics']})
        self.assertEqual(self.rescore_receipt(row['receipt'], config, policy)['status'], 'fail')
        # A judged verdict is judged again by the current limits.
        self.assertEqual(self.rescore_row(row, self.relaxed(config), policy, directory)['status'], 'pass')
        # The gate's own judge cannot pass what eval failed.
        with patch.object(gate.kinds.of('teacher'), 'judge', return_value=('pass', [])):
            row, *_ = self.run_declared('teacher', top1_limit=0)
        self.assertEqual((row['passed'], row['exit_code']), (False, 1), row)
        self.assertIn("failed teacher's declared predicates", row['error'])

    def relaxed(self, config):
        relaxed = deepcopy(config)
        for check in relaxed['checks']:
            check['rules'][1]['value'] = 0.1
        return relaxed

    def assert_refused(self, mutate):
        """A report of eval's exit 1 that `mutate` leaves partial or
        inconsistent is no judged verdict: the live row is a failed run,
        and a stored judged row whose report it is cannot rescore to pass
        under limits its record passes."""
        teacher = gate.kinds.of('teacher')
        capture = teacher.capture

        def partial(command, environment, log, timeout):
            returncode = capture(command, environment, log, timeout)
            report = Path(command[command.index('--report') + 1]) / 'report.json'
            value = json.loads(report.read_text())
            mutate(value, value['checks'][0])
            report.write_text(json.dumps(value))
            return returncode

        with patch.object(teacher, 'capture', partial):
            row, *_, directory = self.run_declared('teacher', top1_limit=0)
        self.assertEqual((row['passed'], row['exit_code'], row['error']), (False, 1, 'check exited 1'), row)
        self.assertNotIn('receipt', row)
        self.assertFalse((directory / 'checks/teacher-capture.json').exists())
        row, config, _, policy, directory = self.run_declared('teacher', top1_limit=0)
        self.assertEqual(row['error'], 'threshold predicate failed', row)
        self.assertEqual(self.rescore_row(row, self.relaxed(config), policy, directory)['status'], 'pass')
        report = directory / 'checks/eval-teacher/report.json'
        value = json.loads(report.read_text())
        mutate(value, value['checks'][0])
        report.write_text(json.dumps(value))
        rescored = self.rescore_row(row, self.relaxed(config), policy, directory)
        self.assertEqual((rescored['status'], rescored.get('error'), rescored.get('receipt_status')),
                         ('fail', 'check exited 1', 'pass'), rescored)

    def test_a_bare_failed_predicate_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: entry.update(predicates=[{'passed': False}]))

    def test_a_null_after_the_predicates_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: entry['predicates'].append(None))

    def test_a_predicate_without_its_source_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: entry['predicates'][1].pop('source'))

    def test_a_check_without_metrics_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: entry.pop('metrics'))

    def test_metrics_other_than_the_record_s_are_no_judged_verdict(self):
        # Still a consistent failure of top1_disagreement le 0, but not the
        # record's 1/23.
        self.assert_refused(lambda value, entry: entry['metrics'].update(top1_disagreement=0.5))

    def test_a_check_error_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: entry.update(error='capture failed'))

    def test_a_boolean_exit_is_no_judged_verdict(self):
        self.assert_refused(lambda value, entry: value.update(exit=True))

    def test_a_row_the_gate_s_judge_passed_is_no_judged_verdict(self):
        # Eval's verdict is complete, but the gate's own judge passes the
        # record: the row is a failed run whatever the limits.
        with patch.object(gate.kinds.of('teacher'), 'judge', return_value=('pass', [])):
            row, config, _, policy, directory = self.run_declared('teacher', top1_limit=0)
        self.assertIn("failed teacher's declared predicates", row['error'])
        self.assertEqual(row['receipt'], str(directory / 'checks/eval-teacher/records/teacher.json'))
        rescored = self.rescore_row(row, self.relaxed(config), policy, directory)
        self.assertEqual((rescored['status'], rescored.get('error'), rescored.get('receipt_status')),
                         ('fail', 'check exited 1', 'pass'), rescored)

    def test_a_refused_eval_run_fails_and_keeps_its_record_only_as_evidence(self):
        # Passing numbers, but each scored response names another process:
        # eval refuses the run (exit 1, a run failure) after writing the record.
        row, config, _, policy, directory = self.run_declared('teacher', answered='f' * 32)
        report = directory / 'checks/eval-teacher'
        value = json.loads((report / 'report.json').read_text())
        self.assertEqual((value['exit'], value['checks'][0]['status']), (1, 'insufficient'))
        self.assertTrue(value['failure']['message'].startswith('refused: '), value['failure'])
        self.assertEqual((row['passed'], row['exit_code'], row['error']), (False, 1, 'check exited 1'), row)
        self.assertNotIn('receipt', row)
        self.assertFalse((directory / 'checks/teacher-capture.json').exists())
        # The rescore finds the record eval left, and the run still fails.
        rescored = self.rescore_row(row, config, policy, directory)
        self.assertEqual((rescored['status'], rescored['error'], rescored.get('receipt'), rescored.get('unrecorded')),
                         ('fail', 'check exited 1', str(report / 'records/teacher.json'), True), rescored)
        self.assertEqual(rescored['receipt_status'], 'pass')
        self.assertEqual(rescored['recorded_metrics']['top1_disagreements'], 1)
        # A linked record does not erase the failed run either.
        linked = {**row, 'receipt': str(report / 'records/teacher.json'),
                  'sha256': gate.digest(report / 'records/teacher.json')}
        self.assertEqual(self.rescore_row(linked, config, policy, directory)['status'], 'fail')


class WaiverGateTests(unittest.TestCase):
    """Declared waivers on a release-gates/v1 config (docs/check-kinds.md#waivers):
    one validator, eval's, refuses a waiver without its reason, decision or
    a looser ceiling before judgment on every path, and a gate whose only
    check is waived passes when it is rescored as recorded."""

    RULE = {'path': ['mean_gap'], 'op': 'le', 'value': 0.005, 'source': 'test'}

    def config(self, **fields):
        waiver = {'predicate': {'path': ['mean_gap'], 'op': 'le'}, 'ceiling': 0.0057,
                  'reason': 'sampling-level evidence', 'decision': 'operator, 2026-10-15', **fields}
        check = {'id': 'tapdiff', 'module': 'tapdiff', 'args': [], 'timeout_seconds': 60,
                 'receipt': '*-tapdiff.json', 'rules': [self.RULE], 'waivers': [waiver]}
        return {'schema': 'chopinfer-release-gates/v1', 'model': 'fixture',
                'profiles': {'full': {'checks': ['tapdiff']}}, 'checks': [check],
                'inputs': {}, 'qualification': {'mechanics_only': False}}

    def run_gate(self, config, root):
        """The full gate on `config`, its one worker writing a receipt whose
        mean_gap fails the rule and is within the waiver's ceiling."""
        path = root / 'config.json'
        path.write_text(json.dumps(config))
        real, workers = subprocess.run, []

        def capture(command, **kwargs):
            if '--worker' not in command:
                return real(command, **kwargs)
            workers.append(command)
            raw = Path(kwargs['env']['CHOPINFER_QUAL_RECEIPTS'])
            (raw / 'fixture-tapdiff.json').write_text(json.dumps(
                {'schema': 'chopinfer-qualification/v1', 'check': 'tapdiff', 'scored': 1, 'mean_gap': 0.0057,
                 'errors': [], 'skipped': [], 'mechanics_only': False}))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(gate.Client, 'identity', return_value={'model': 'fixture'}), \
                patch.object(gate.subprocess, 'run', side_effect=capture):
            code = gate.run(config, path, 'full', 'http://unused', None, {}, root / 'live', False)
        return code, path, workers

    def test_an_invalid_waiver_is_refused_before_judgment(self):
        cases = [('reason', {'reason': ''}), ('reason', {'reason': '   '}), ('decision', {'decision': ''}),
                 ('decision', {'decision': ' \t'}), ('ceiling', {'ceiling': 0.005}),
                 ('predicate', {'predicate': {'path': ['other'], 'op': 'le'}}),
                 ('category', {'predicate': None, 'category': 'prose', 'metric': 'top1_disagreement'})]
        for field, fields in cases:
            config = self.config(**fields)
            if fields.get('predicate', 0) is None:
                del config['checks'][0]['waivers'][0]['predicate']
            with self.subTest(field=field, fields=fields):
                with self.assertRaisesRegex(ValueError, rf'waivers\.0\.{field}'):
                    gate.validate(config)
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(ValueError, rf'waivers\.0\.{field}'):
                        self.run_gate(config, Path(temporary))
                # The judge of a kind Python scores refuses them too.
                with self.assertRaisesRegex(ValueError, rf'waivers\.0\.{field}'):
                    kinds.of('tapdiff').judge({'mean_gap': 0.0057}, config['checks'][0])
        gate.validate(self.config())

    def test_an_all_waived_gate_qualifies_and_rescores_as_passing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config()
            code, path, workers = self.run_gate(config, root)
            self.assertEqual((code, len(workers)), (0, 1))
            summary_path = next((root / 'live').glob('*/summary.json'))
            summary = json.loads(summary_path.read_text())
            self.assertEqual((summary['passed'], summary['qualified']), (True, True))
            self.assertEqual(summary['checks'][0]['status'], 'waived')
            args = SimpleNamespace(model='fixture', paths=[summary_path], limits='recorded', check=None,
                                   json=root / 'rescore.json', require_recorded=True)
            with patch.object(checks, 'load', return_value=(config, path)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checks.rescore_command(args), 0)
            report = json.loads(args.json.read_text())
            self.assertEqual(report['checks'][0]['status'], 'waived')
            self.assertEqual(report['gates'][0]['passed'], True)

    def test_a_gate_passes_with_a_pass_or_a_waiver_and_nothing_failing(self):
        for statuses, passed in ((['waived'], True), (['pass'], True), (['informational', 'waived'], True),
                                 (['informational'], False), (['not_applicable'], False),
                                 (['informational', 'not_applicable'], False), (['waived', 'fail'], False),
                                 ([], False)):
            with self.subTest(statuses=statuses):
                self.assertEqual(checks.gate_passed(statuses), passed)


if __name__ == '__main__':
    unittest.main()
