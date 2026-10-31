"""CPU self-test of the depth tuner's planning and selection."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate  # noqa: E402
import prompts  # noqa: E402
import recorded  # noqa: E402
import report  # noqa: E402
import prefill_report  # noqa: E402
import stage  # noqa: E402
import tune  # noqa: E402

# A synthetic model: depths 0-5 over 32 verify rows, two repeats each.
MODEL = {'max_depth': 5, 'verify_rows': 32, 'concurrencies': [1, 2, 4, 8, 16, 32], 'repeats': 2}
CONFIG = {'name': 'synthetic', 'model': MODEL, 'kinds': {'greedy': 0.5, 'sampled': 0.5},
          'selection': {'margin': 0.015, 'spread': 0.03}}


def fixture_content(root, name):
    import copy
    binary = root / f'install-{name}/bin/chopinfer'
    recipe = json.loads(binary.with_name('identity.json').read_text())
    proof = {'schema': 'chopinfer.weight-content.v1', 'producer_sha256': stage.digest(binary),
             'checkpoint': recipe['checkpoint'], 'images': copy.deepcopy(recipe['images'])}
    for image in proof['images']:
        image['identity']['content_sha256'] = 'e' * 64
    path = root / f'logs/weight-content-{name}.json'
    path.write_text(json.dumps(proof))
    path.chmod(0o600)
    return path


def frozen_fixture(root):
    """Two frozen installs with different recipes and equal proven weight bytes."""
    import shutil
    logs = root / 'logs'
    logs.mkdir(parents=True)
    identity = {'schema': 'chopinfer.weight-recipes.v1', 'content_verified': False,
                'checkpoint': {'identity': 'a' * 64, 'stamps': [{'size': 4}]},
                'images': [{'image_role': 'weights', 'rank': 0, 'recipe_identity': 'b' * 64,
                            'byte_length': 4, 'identity': {'content_sha256': '', 'layout_sha256': 'c' * 64}}]}
    for name, revision in (('base', '1' * 40), ('head', '2' * 40)):
        binary = root / f'install-{name}' / 'bin/chopinfer'
        binary.parent.mkdir(parents=True)
        binary.write_text('#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n'
                          f'# Synthetic revision {revision}\n'
                          'if "fixture-ar" in sys.argv and any(x.startswith(("--speculation", "--speculative-", "--draft-")) for x in sys.argv):\n'
                          '    sys.exit("--speculation requires positive --speculative-tokens")\n'
                          'if sys.argv[1:3] == ["weights", "identity"]:\n'
                          '    out = Path(sys.argv[sys.argv.index("--output") + 1])\n'
                          '    out.write_text(Path(__file__).with_name("identity.json").read_text())\n'
                          'elif "--list-model-options" in sys.argv:\n'
                          '    print("{}")\nelse:\n    sys.exit(2)\n')
        binary.chmod(0o755)
        identity['images'][0]['recipe_identity'] = ('b' if name == 'base' else 'd') * 64
        (binary.parent / 'identity.json').write_text(json.dumps(identity))
        fixture_content(root, name)
        source = root / f'source-{name}'
        shutil.copytree(HERE.parents[1] / 'tools/qual', source / 'tools/qual',
                        ignore=shutil.ignore_patterns('__pycache__'))
        (source / '.commit').write_text(revision + '\n')
        bundle = root / f'cache-{name}' / 'fixture'
        bundle.mkdir(parents=True)
        (bundle / 'manifest.json').write_text('{}\n')
        (logs / f'version-{name}.txt').write_text(f'fixture ({revision})\n')
        (logs / f'install-{name}.sha256').write_text(f'{stage.digest(binary)}  {binary}\n')
        (logs / f'prepare-{name}.json').write_text(json.dumps({'result': 'prepared-direct', 'bundle': str(bundle)}))
        stage.protect(binary.parent.parent)
        stage.protect(bundle)
    workload = root / 'workload.json'
    workload.write_text(json.dumps({'name': 'fixture', 'prompts': 128, 'mix': [{'kind': 'chat', 'weight': 1}],
                                   'sources': {'globs': [], 'min_bytes': 0, 'max_bytes': 0}}))
    return {'model': 'fixture', 'workload': str(workload), 'mode': 'ab', 'repeats': 2,
            'concurrencies': [1], 'speculation_mode': 'balanced',
            'arms': [{'label': 'a', 'depth': 1}, {'label': 'b', 'depth': 1, 'install': 'head'}],
            'pairs': [['a', 'b']], 'served_model_name': 'fixture',
            'checkpoint_env': 'CHOPINFER_TEST_CHECKPOINT', 'tensor_parallel_size': 1,
            'kv_pages': 64, 'kv_pages_line': 'fixture: {pages} KV pages', 'serve_arguments': [],
            'max_depth': 1, 'verify_rows': 32, 'work_seconds': 900, 'queue_timeout_seconds': 1000,
            'tokens': {'default': 64}, 'warm': [], 'warm_tokens': 8, 'kinds': {'sampled': 1},
            'sampling': {'seed': 7, 'temperature': 1.0, 'top_p': 1.0}, 'selection': CONFIG['selection'],
            'planning': {'startup_seconds': 1, 'process_overhead_seconds': 1, 'settle_seconds': 1,
                         'process_reserve_seconds': 1, 'tokens_per_second': {'1': 100},
                         'ar_tokens_per_second': {'1': 100}}}


def stage_fixture(root, model, names=('base', 'head'), destination='stage', proofs=()):
    import os
    import subprocess
    config = root / 'model.json'
    config.write_text(json.dumps(model))
    argv = [sys.executable, '-B', str(HERE / 'stage.py'), '--model', str(config),
            '--install-dir', str(root), str(root / destination)]
    for name in names:
        argv += ['--install-name', name]
    for name, path in proofs:
        argv += ['--content-proof', f'{name}={path}']
    env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
    env['CHOPINFER_TEST_CHECKPOINT'] = str(root / 'checkpoint')
    return subprocess.run(argv, env=env, text=True, capture_output=True, timeout=30)


def window(run, label, kind, c, depth, rate, histogram):
    record = {'name': f'{kind}-c{c}', 'kind': kind, 'concurrency': c, 'depth': depth, 'tokens': 256,
              'request_seconds': c * 256 / rate, 'request_tokens_per_second': rate,
              'full': {'tokens_per_second': rate * 1.2, 'histogram': histogram},
              'streams': {'p10': rate / c}, 'energy': {'joules_per_token': 1000 / rate},
              'acceptance': {'draft_tokens': 10 * depth, 'accepted_tokens': 5 * depth, 'verify_items': 10,
                             'per_position': [{'position': p, 'drafted': 10, 'accepted': 10 - 2 * p}
                                              for p in range(1, depth + 1)]}}
    path = run / label / 'windows' / f'{kind}-c{c}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def priming_fixture(c):
    return {'completed_requests': c, 'per_slot_completed': {str(i): 1 for i in range(c)},
            'last_completed': 1, 'window_started': 2}


def warmup_fixture(config):
    return {'completed': True, 'windows': [{'concurrency': c, 'repeat': repeat,
        'seed_set': tune.seed_set(config, repeat, c), 'cached_prompt_tokens': 0,
        'requests': c, 'ended': 1} for c, repeat in tune.warmup_windows(config)]}


def ab_fixture(root, **overrides):
    model = {**MODEL, 'mode': 'ab', 'concurrencies': [8], 'tensor_parallel_size': 1,
             'arms': [{'label': label, 'depth': 3} for label in ('a', 'b')],
             'pairs': [['a', 'b']], **overrides}
    config = {**CONFIG, 'model': model, 'kinds': {'sampled': 1},
              'sampling': {'seed': 41, 'temperature': 1.0, 'top_p': 1.0}}
    power = {'enforced_power_limits_w': [250], 'sm_clock_mhz': [1800]}
    summary = {'status': 'passed', 'exactness': {'passed': True}, 'processes': {},
               'power': {'before': power, 'after': power}, 'page_cache_warmup': warmup_fixture(config)}
    for visit in tune.visits(config):
        repeat = visit['passes'][0][0]
        for _, c in visit['passes'][0][1]:
            rate = 100 if visit['arm'] == model['arms'][0]['label'] else 110
            row = {'arm': visit['arm'], 'visit': visit['visit'], 'repeat': repeat, 'kind': 'sampled',
                   'concurrency': c, 'depth': 3, 'request_tokens_per_second': rate,
                   'priming': priming_fixture(c), 'replacement_submissions': 1, 'prefill_share': .01,
                   'window_seconds': 15, 'request_seconds': 15, 'temperature': 1.0,
                   'cached_prompt_tokens': 0, 'seed_set': tune.seed_set(config, repeat, c),
                   'faults': {'complete': True, 'major_faults': 0},
                   'full': {'tokens_per_second': rate, 'histogram': {}, 'steps': 20, 'tokens': 100,
                            'allocated_steps': 0},
                   'energy': {'joules_per_token': 1, 'sm_mhz_range': [[1800, 1800]],
                              'enforced_power_limits_w': [250]},
                   'cells': {f'i{c}-r32': {'cycle_ms': [100 / rate] * 20, 'execution_ms': [1] * 20}}}
            tune.r.write(root / visit['label'] / 'windows' / f'c{c}.json', row)
        summary['processes'][visit['label']] = {'process_group': visit['visit'] + 1,
            'weights': {'source': 'borrowed'}, 'faults': {'complete': True, 'major_faults': 0},
            'power_before': power, 'power_after': power}
    tune.r.write(root / 'config.json', config)
    tune.r.write(root / 'summary.json', summary)
    return config, summary


class Tuner(unittest.TestCase):
    def test_counters_sum_eviction_classes_and_keep_legacy_recordings(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        for values in [(0, 0, 0), (2, 3, 5)]:
            text = 'vllm:num_preemptions_total{model_name="fixture",engine="0"} 7\n'
            text += ''.join(f'chop_prefix_cache_evictions_total{{class="{name}"}} {value}\n'
                            for name, value in zip(('superseded', 'retained', 'maximal'), values))
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                server = SimpleNamespace(url='http://fixture', directory=Path(directory))
                with patch.object(tune.r, 'http', return_value=text):
                    self.assertEqual(tune.counters(server, 'final'),
                                     {'preemptions': 7, 'prefix_cache_evictions': sum(values)})
                self.assertEqual((server.directory / 'final.prom').read_text(), text)
        legacy = 'vllm:num_preemptions_total 7\nchop_prefix_cache_evictions_total 11\n'
        with tempfile.TemporaryDirectory() as directory, patch.object(tune.r, 'http', return_value=legacy):
            server = SimpleNamespace(url='http://fixture', directory=Path(directory))
            self.assertEqual(tune.counters(server, 'final'), {'preemptions': 7, 'prefix_cache_evictions': 11})

    def test_external_prompt_file_seals_literals_and_checks_tokenized_lengths(self):
        import os
        import subprocess
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'CHOPINFER_DATA': directory}):
            root = Path(directory)
            model = frozen_fixture(root)
            entries = [{'prompt': f'Code {i}:'.ljust(256, 'x'), 'kind': 'coding', 'target_length': 12} for i in range(4)]
            entries += [{'messages': [{'role': 'user', 'content': f'Chat {i}:'.ljust(512, 'y')}],
                         'kind': 'chat', 'target_length': 16} for i in range(4)]
            entries[0] = entries[0]['prompt']
            source = root / 'prompts.json'
            source.write_text(json.dumps(entries))
            model['prompt_file'] = str(source)
            model['kv_pages'] = 'default'
            backing = root / 'rank.bin'
            backing.write_bytes(b'\0' * 4096)
            model['cold_engram'] = {'matched': {'lengths': [12, 16], 'tolerance': 0},
                'images': [{'path': str(backing), 'bytes': 4096,
                            'views': [{'name': 'fixture', 'offset': 0, 'bytes': 4096}]}]}
            result = stage_fixture(root, model)
            self.assertEqual(result.returncode, 0, result.stderr)
            seal = json.loads((root / 'stage/runtime-seal.json').read_text())
            self.assertIn('prompt_file', list(seal))
            self.assertEqual(seal['prompt_file'], {'path': str(source), 'size': source.stat().st_size,
                                                   'sha256': stage.digest(source)})
            self.assertEqual(seal['sha256'][str(source)], stage.digest(source))
            self.assertEqual(seal['prompts'][0]['prompt'], entries[0])
            self.assertEqual(seal['prompts'][4]['messages'], entries[4]['messages'])
            source = root / 'prompts.jsonl'
            source.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\n')
            model['prompt_file'] = str(source)
            model.pop('workload')
            result = stage_fixture(root, model, destination='jsonl-stage')
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = root / 'jsonl-stage'
            env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
            env['CHOPINFER_WEIGHT_KEEPER_DIR'] = str(root / 'synthetic-keeper')
            def run(stage_dir, out):
                return subprocess.run([sys.executable, '-B', str(stage_dir / 'harness/tune.py'), '--dry-run',
                    '--output', str(root / out), '--payload-sha', stage.digest(stage_dir / 'runtime-seal.json')],
                    env=env, text=True, capture_output=True, timeout=90)
            result = run(staged, 'valid')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((root / 'valid/summary.json').read_text())
            self.assertEqual(summary['windows'], {'planned': 8, 'executed': 8})
            entries[1]['target_length'] = 16
            source.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\n')
            result = run(staged, 'tampered')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('sealed file changed', result.stderr)
            result = stage_fixture(root, model, destination='wrong-length-stage')
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run(root / 'wrong-length-stage', 'wrong-length')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('prompt length', result.stdout + result.stderr)

    def test_default_kv_pool_omits_fixed_flags_and_cold_requires_it(self):
        from unittest.mock import patch
        with patch.object(tune.r, 'KV_PAGES', 'default'):
            self.assertNotIn('--kv-pages', tune.r.serve_options('cache', [], 'checkpoint'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model['cold_engram'] = {'warm_fraction': .1}
            result = stage_fixture(root, model)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cold Engram requires the full default KV pool', result.stderr)

    def test_fixed_and_default_pool_stub_checks_counts_and_pressure(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
            env['CHOPINFER_WEIGHT_KEEPER_DIR'] = str(root / 'synthetic-keeper')
            for case, pool, controls, expected in [('fixed', 64, {}, 'passed'),
                ('default', 'default', {}, 'passed'),
                ('changed', 'default', {'TUNE_STUB_KV_CHANGE_LABEL': 'b-r0'}, 'failed')]:
                model['kv_pages'] = pool
                result = stage_fixture(root, model, destination=f'stage-{case}')
                self.assertEqual(result.returncode, 0, result.stderr)
                staged = root / f'stage-{case}'
                spec = json.loads((staged / 'keeper-spec.json').read_text())
                self.assertEqual('--kv-pages' in spec['argv'], pool != 'default')
                out = root / case
                result = subprocess.run([sys.executable, '-B', str(staged / 'harness/tune.py'), '--dry-run',
                    '--output', str(out), '--payload-sha', stage.digest(staged / 'runtime-seal.json')],
                    env={**env, **controls}, text=True, capture_output=True, timeout=90)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['status'], expected, result.stdout + result.stderr)
                self.assertEqual(result.returncode, 0 if expected == 'passed' else 1)
                if expected == 'passed':
                    for row in summary['processes'].values():
                        self.assertEqual(row['kv_pages_per_rank'], [4096 if pool == 'default' else 64])
                        self.assertEqual(row['metrics']['preemptions'], 0)
                        self.assertEqual(row['metrics']['prefix_cache_evictions'], 0)
                    for path in out.glob('*/server/launch.json'):
                        argv = json.loads(path.read_text())['serve_argv']
                        self.assertEqual('--kv-pages' in argv, pool != 'default')
                else:
                    self.assertIn('default KV pages', ' '.join(summary['errors']))

    def test_matched_cold_reports_latency_cells_and_voids_warm_visits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, summary = ab_fixture(root, concurrencies=[1])
            config['model']['kv_pages'] = 'default'
            summary['kv_pages_per_rank'] = [4096]
            for process in summary['processes'].values():
                process['kv_pages_per_rank'] = [4096]
            tune.r.write(root / 'summary.json', summary)
            config['cold_engram'] = {'warm_fraction': .1, 'matched': {'lengths': [8, 32], 'tolerance': 0}}
            tune.r.write(root / 'config.json', config)
            for path in list(root.glob('*/windows/*.json')):
                row = json.loads(path.read_text())
                path.unlink()
                for length in (8, 32):
                    latency = 1 if row['arm'] == 'a' else .8
                    prompt = [length, row['repeat']] + [0] * (length - 2)
                    request = {'prompt': prompt, 'seed': row['seed_set'][0] + 1, 'token_ids': [1],
                               'cache_salt': f"{row['label'] if 'label' in row else path.parent.parent.name}-{length}",
                               'cached_prompt_tokens': 0}
                    value = {**row, 'target_length': length, 'length_tolerance': 0,
                             'ttft_seconds': [latency], 'prefill_execution_us': [latency * 1000],
                             'mapped_gather_us': [1], 'prompt_lengths': [length], 'matched_requests': [request],
                             'cold_engram': {'state': 'cold', 'after_advice': [{'resident_fraction': 0}],
                                 'acknowledgement': {'reset_cache': True, 'released_mapping_bytes': 4096}}}
                    tune.r.write(path.with_name(f'l{length}.json'), value)
            result = report.ab(root)
            self.assertEqual(result['weighted']['a/b']['verdict'], 'resolved gain')
            self.assertTrue(result['contract_a_admissible'])
            self.assertAlmostEqual(result['pairs']['a/b']['c1/l8']['ttft_seconds']['mean'], .25)
            path = root / 'b-r0/windows/l8.json'
            value = json.loads(path.read_text())
            value['cold_engram']['after_advice'][0]['resident_fraction'] = .1
            tune.r.write(path, value)
            result = report.ab(root)
            self.assertEqual(result['status'], 'void')
            self.assertEqual(result['pairs'], {})

    def test_matched_cold_assigns_disjoint_pairs_at_declared_lengths(self):
        config = {**CONFIG, 'sampling': {'temperature': 1, 'seed': 1}, 'kinds': {'sampled': 1},
                  'model': {**MODEL, 'mode': 'ab', 'concurrencies': [1], 'pairs': [['a', 'b']],
                            'arms': [{'label': label, 'depth': 0} for label in ('a', 'b')]}}
        tokens = [[i] + [0] * (length - 1) for length in (8192, 32768) for i in range(8)]
        pools = prompts.matched_visit_pools(tokens, tune.visits(config), [8192, 32768], 0, 2)
        for length in (8192, 32768):
            self.assertEqual(pools[f'a-r0/l{length}'].remaining, pools[f'b-r0/l{length}'].remaining)
            self.assertEqual(pools[f'b-r1/l{length}'].remaining, pools[f'a-r1/l{length}'].remaining)
            self.assertFalse(set(pools[f'a-r0/l{length}'].assignment['indices'])
                             & set(pools[f'a-r1/l{length}'].assignment['indices']))
            self.assertEqual(pools[f'a-r0/l{length}'].assignment['lengths'], [length] * 2)
        with self.assertRaisesRegex(ValueError, 'within tolerance'):
            prompts.matched_visit_pools(tokens, tune.visits(config), [16384], 10, 2)

    def test_matched_cold_stub_voids_one_warm_visit_and_refuses_cached_tokens(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            backing = root / 'rank.bin'
            backing.write_bytes(b'\0' * 4096)
            model['cold_engram'] = {'warm_fraction': .1, 'matched': {'lengths': [64, 128], 'tolerance': 0},
                'images': [{'path': str(backing), 'bytes': 4096,
                            'views': [{'name': 'fixture', 'offset': 0, 'bytes': 4096}]}]}
            model['kv_pages'] = 'default'
            result = stage_fixture(root, model)
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = root / 'stage'
            env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
            env.update(CHOPINFER_WEIGHT_KEEPER_DIR=str(root / 'synthetic-keeper'),
                       TUNE_STUB_TOKEN_LENGTHS='64,128')
            for case, control, expected in [('valid', {}, 'passed'),
                ('warm', {'TUNE_STUB_RESIDENCY': '.1', 'TUNE_STUB_RESIDENCY_LABEL': 'b-r0'}, 'void'),
                ('cached', {'TUNE_STUB_CACHED': 'b-r0'}, 'failed')]:
                out = root / case
                result = subprocess.run([sys.executable, '-B', str(staged / 'harness/tune.py'), '--dry-run',
                    '--output', str(out), '--payload-sha', stage.digest(staged / 'runtime-seal.json')],
                    env={**env, **control}, text=True, capture_output=True, timeout=90)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['status'], expected, result.stdout + result.stderr)
                self.assertEqual(result.returncode, 0 if expected == 'passed' else 1)
                if case == 'valid':
                    self.assertEqual(summary['windows'], {'planned': 8, 'executed': 8})
                    self.assertTrue(summary['exactness']['passed'])
                    result = json.loads((out / 'ab.json').read_text())
                    self.assertEqual(set(result['pairs']['a/b']), {'c1/l64', 'c1/l128'})
                    self.assertFalse(result['contract_a_admissible'])
                    for row in report.load(out)[1]:
                        self.assertEqual(row['cold_engram']['state'], 'cold')
                        self.assertEqual(row['cached_prompt_tokens'], 0)
                        self.assertEqual(row['prompt_lengths'], [row['target_length']])
                if case == 'warm':
                    self.assertIn('partly warm', summary['void_reason'])
                    self.assertEqual(json.loads((out / 'ab.json').read_text())['pairs'], {})
                    with self.assertRaisesRegex(ValueError, 'void'):
                        report.load(out)
                if case == 'cached':
                    self.assertIn('cached', ' '.join(summary['errors']))

    def test_ar_has_no_draft_arguments_and_refuses_depth_controls(self):
        config = {'model': {'mode': 'ab', 'speculation_mode': 'off', 'serve_arguments': [],
                            'arms': [{'label': 'a'}, {'label': 'b'}]}}
        self.assertEqual(tune.arm_arguments(config, {'label': 'a', 'depth': 0, 'arguments': []}), [])
        import copy
        for override in ({'depths': [0]}, {'max_depth': 1}, {'mode': 'sweep'},
                         {'arms': [{'label': 'a', 'depth': 0}]}, {'arms': [{'label': 'a', 'depth': 1}]},
                         {'extra_arguments': ['--speculation', 'balanced']},
                         {'serve_arguments': ['--draft-cost', '1']},
                         {'arms': [{'label': 'a', 'arguments': ['--speculative-tokens', '0']}]}):
            with self.subTest(override=override):
                invalid = copy.deepcopy(config)
                invalid['model'].update(override)
                with self.assertRaisesRegex(ValueError, 'AR'):
                    tune.arms(invalid)
        import stub
        self.assertEqual(stub.options([], ar=True), ('off', [(1, 0)], 32))
        with self.assertRaisesRegex(ValueError, 'requires positive'):
            stub.options(['--speculation', 'balanced'], ar=True)

    def test_ar_environment_arms_run_ab_cold_and_exact_with_frozen_parsers(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model.update(speculation_mode='off', max_depth=0, serve_arguments=['--profile', 'fixture-ar'],
                         arms=[{'label': 'a', 'env': {'CHOPINFER_TEST_PREFILL': None}},
                               {'label': 'b', 'install': 'head', 'env': {'CHOPINFER_TEST_PREFILL': '1'}}])
            for case in ('ab', 'cold', 'exact'):
                if case == 'cold':
                    model['kv_pages'] = 'default'
                    backing = root / 'rank.bin'
                    backing.write_bytes(b'\0' * 4096)
                    model['cold_engram'] = {'images': [{'path': str(backing), 'bytes': 4096,
                        'views': [{'name': 'fixture', 'offset': 0, 'bytes': 4096}]}]}
                    model['planning'].update(tokens_per_second={'1': 10}, ar_tokens_per_second={'1': 10})
                if case == 'exact':
                    model.pop('cold_engram')
                    model.update(mode='exact', kinds={'greedy': .5, 'sampled': .5})
                names = ('head',) if case == 'exact' else ('base', 'head')
                result = stage_fixture(root, model, names=names, destination=f'stage-{case}')
                self.assertEqual(result.returncode, 0, result.stderr)
                staged = root / f'stage-{case}'
                spec = json.loads((staged / 'keeper-spec.json').read_text())
                self.assertFalse(any(tune.draft_option(word) for word in spec['argv']))
                env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
                env.update(CHOPINFER_WEIGHT_KEEPER_DIR=str(root / 'synthetic-keeper'),
                           CHOPINFER_TEST_CHECKPOINT=str(root / 'checkpoint'), CHOPINFER_TEST_PREFILL='inherited')
                preflight = subprocess.run([sys.executable, '-B', str(staged / 'harness/preflight.py'), str(staged)],
                    env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
                out = root / f'run-{case}'
                result = subprocess.run([sys.executable, '-B', str(staged / 'harness/tune.py'), '--dry-run',
                    '--output', str(out), '--payload-sha', stage.digest(staged / 'runtime-seal.json')],
                    env=env, text=True, capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['windows']['executed'], 8 if case == 'exact' else 4)
                for row in summary['processes'].values():
                    self.assertEqual(row['speculation_mode'], 'off')
                launches = list(out.glob('*/server/launch.json')) + list(out.glob('references/*/launch.json'))
                self.assertGreaterEqual(len(launches), 2 if case == 'exact' else 5)
                for path in launches:
                    launch = json.loads(path.read_text())
                    self.assertFalse(any(tune.draft_option(word) for word in launch['serve_argv']))
                    self.assertEqual(json.loads(path.with_name('argv-preflight.log.exit.json').read_text())['exit_code'], 0)
                    controls = json.loads(path.with_name('stub-install.json').read_text())['model_controls']
                    self.assertEqual(controls.get('CHOPINFER_TEST_PREFILL'), launch['env']['CHOPINFER_TEST_PREFILL'])
                if case == 'exact':
                    self.assertEqual(len(list((out / 'a/reference').glob('*.json'))), 2)
                else:
                    self.assertTrue(summary['exactness']['passed'])


    def test_content_proofs_admit_recipe_changes_and_refuse_invalid_proofs(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            result = stage_fixture(root, model)
            self.assertEqual(result.returncode, 0, result.stderr)
            seal = json.loads((root / 'stage/runtime-seal.json').read_text())
            installs = seal['installs']
            self.assertEqual(installs['base']['content_key'], installs['head']['content_key'])
            proofs = [json.loads(Path(installs[name]['weight_keeper_identity']).read_text()) for name in ('base', 'head')]
            self.assertNotEqual(proofs[0]['producer_sha256'], proofs[1]['producer_sha256'])
            self.assertNotEqual(proofs[0]['images'][0]['recipe_identity'], proofs[1]['images'][0]['recipe_identity'])
            spec = json.loads((root / 'stage/keeper-spec.json').read_text())
            self.assertEqual(spec['identity_sha256'], stage.digest(installs['base']['weight_keeper_identity']))
            self.assertIn(installs['base']['weight_keeper_identity'], spec['argv'])
            path = root / 'logs/weight-content-head.json'
            proof = json.loads(path.read_text())
            path.unlink()
            result = stage_fixture(root, model, destination='missing-proof')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('content proof', result.stderr)
            mutations = [lambda v: v.update(producer_sha256=proofs[0]['producer_sha256']),
                         lambda v: v['images'][0]['identity'].update(content_sha256='f' * 64),
                         lambda v: v['images'][0]['identity'].update(layout_sha256='f' * 64),
                         lambda v: v['images'][0].update(recipe_identity='f' * 64),
                         lambda v: v['images'][0].update(byte_length=8),
                         lambda v: v['checkpoint']['stamps'][0].update(size=8),
                         lambda v: v.update(images=[])]
            for i, mutate in enumerate(mutations):
                with self.subTest(proof=i):
                    value = copy.deepcopy(proof)
                    mutate(value)
                    path.write_text(json.dumps(value))
                    path.chmod(0o600)
                    result = stage_fixture(root, model, destination=f'bad-proof-{i}')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(result.stderr, 'content proof|content_key')
            path.write_text(json.dumps(proof))
            alternate = root / 'content.json'
            path.rename(alternate)
            result = stage_fixture(root, model, destination='lane-proof', proofs=[('head', alternate)])
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_install_uses_an_explicit_content_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model['arms'] = [{'label': 'a', 'depth': 1}, {'label': 'b', 'depth': 1}]
            path = root / 'logs/weight-content-head.json'
            result = stage_fixture(root, model, names=('head',), proofs=[('head', path)])
            self.assertEqual(result.returncode, 0, result.stderr)
            seal = json.loads((root / 'stage/runtime-seal.json').read_text())
            self.assertNotIn('installs', seal)
            install = seal['install']
            self.assertRegex(install['content_key'], '^[0-9a-f]{64}$')
            self.assertIn(install['weight_keeper_identity'], seal['sha256'])
            spec = json.loads((root / 'stage/keeper-spec.json').read_text())
            self.assertEqual(spec['identity_sha256'], stage.digest(install['weight_keeper_identity']))
            self.assertIn(install['weight_keeper_identity'], spec['argv'])
            proof = json.loads(path.read_text())
            proof['images'][0]['recipe_identity'] = 'f' * 64
            bad = root / 'bad-content.json'
            bad.write_text(json.dumps(proof))
            bad.chmod(0o600)
            result = stage_fixture(root, model, names=('head',), destination='bad', proofs=[('head', bad)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('content proof', result.stderr)
            result = stage_fixture(root, model, names=('head',), destination='unknown', proofs=[('base', path)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('content-proof needs one NAME=FILE per selected build', result.stderr)
            result = stage_fixture(root, model, names=('head',), destination='plain')
            self.assertEqual(result.returncode, 0, result.stderr)
            seal = json.loads((root / 'plain/runtime-seal.json').read_text())
            self.assertNotIn('weight_keeper_identity', seal['install'])

    def test_multi_install_requires_ab_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            for mode in ('sweep', 'costs', 'exact'):
                with self.subTest(mode=mode):
                    model['mode'] = mode
                    result = stage_fixture(root, model, destination=mode)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('multiple installs require mode ab', result.stderr)

    def test_metadata_jobs_require_complete_consistent_provenance(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, summary = ab_fixture(root)
            identities = {'a': {'install': 'base', 'revision': '1' * 40},
                          'b': {'install': 'head', 'revision': '2' * 40}}
            for arm in config['model']['arms']:
                arm['install'] = identities[arm['label']]['install']
            tune.r.write(root / 'config.json', config)
            for path in root.glob('*/windows/*.json'):
                row = json.loads(path.read_text())
                tune.r.write(path, {**row, **identities[row['arm']]})
            for label, process in summary['processes'].items():
                process.update(identities[label.split('-')[0]])
            for value in (None, {}, {'a': identities['a']}, {**identities, 'b': {'install': 'head'}}):
                with self.subTest(map=value):
                    modified = copy.deepcopy(summary)
                    if value is not None:
                        modified['arm_installs'] = value
                    tune.r.write(root / 'summary.json', modified)
                    with self.assertRaisesRegex(ValueError, 'provenance'):
                        report.ab(root)
            summary['arm_installs'] = identities
            tune.r.write(root / 'summary.json', summary)
            self.assertEqual(report.ab(root)['status'], 'valid')
            path = root / 'b-r1/windows/c8.json'
            row = json.loads(path.read_text())
            tune.r.write(path, {**row, **identities['a']})
            for present in (True, False):
                with self.subTest(mixed=True, present=present):
                    modified = copy.deepcopy(summary)
                    if not present:
                        modified.pop('arm_installs')
                    tune.r.write(root / 'summary.json', modified)
                    with self.assertRaisesRegex(ValueError, 'provenance|revision differs'):
                        report.ab(root)
            ab_fixture(root)
            self.assertEqual(report.ab(root)['status'], 'valid')

    def test_stage_two_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            result = stage_fixture(root, model)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            seal = json.loads((root / 'stage/runtime-seal.json').read_text())
            self.assertIn('installs', list(seal))
            self.assertEqual(set(seal['installs']), {'base', 'head'})
            self.assertEqual(seal['default_install'], 'base')
            self.assertEqual(seal['install'], seal['installs']['base'])
            self.assertTrue(all(str(root / f'install-{name}/bin/chopinfer') in seal['sha256']
                                for name in ('base', 'head')))

    def test_stage_refuses_unknown_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model['arms'][1]['install'] = 'missing'
            result = stage_fixture(root, model)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unsealed install', result.stderr)

    def test_stage_refuses_unused_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model['arms'][1].pop('install')
            result = stage_fixture(root, model)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unused sealed install: head', result.stderr)

    def test_single_install_seal_keeps_legacy_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            result = stage_fixture(root, model, names=())
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / 'stage/runtime-seal.json'
            original = path.read_bytes()
            seal = json.loads(original)
            self.assertEqual(set(seal), {'schema', 'ready', 'config', 'install', 'harness_source', 'prompts',
                                         'prompt_sources', 'cold_files', 'sha256'})
            self.assertNotIn('name', seal['install'])
            result = stage_fixture(root, model, names=('head',))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(tune.arm_install(seal, {'label': 'legacy'})[0], 'head')

    def test_keeper_refuses_different_recipes_and_serve_options(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            path = root / 'install-head/bin/identity.json'
            identity = json.loads(path.read_text())
            mutations = [lambda v: v['checkpoint'].update(identity='d' * 64),
                         lambda v: v['checkpoint']['stamps'][0].update(size=5),
                         lambda v: v['images'][0].update(recipe_identity='f' * 64),
                         lambda v: v['images'][0]['identity'].update(layout_sha256='d' * 64),
                         lambda v: v['images'][0].update(byte_length=5),
                         lambda v: v['images'][0].update(rank=1)]
            for i, mutate in enumerate(mutations):
                with self.subTest(identity=i):
                    value = copy.deepcopy(identity)
                    mutate(value)
                    path.write_text(json.dumps(value))
                    result = stage_fixture(root, model, destination=f'identity-{i}')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('keeper identity differs', result.stderr)
            path.write_text(json.dumps(identity))
            model['arms'][1]['arguments'] = ['--kv-pages', '65']
            result = stage_fixture(root, model, destination='options')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('identical serve options', result.stderr)
            model['arms'][1]['arguments'] = ['--speculation-widths', '8,16']
            result = stage_fixture(root, model, destination='speculation')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_keeper_uses_first_arm_install_and_preserves_cli_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            model['arms'][0]['install'] = 'head'
            model['arms'][1]['install'] = 'base'
            result = stage_fixture(root, model, names=('head', 'base'))
            self.assertEqual(result.returncode, 0, result.stderr)
            seal = json.loads((root / 'stage/runtime-seal.json').read_text())
            self.assertEqual(tune.arm_install(seal, {'label': 'default'})[0], 'head')
            result = stage_fixture(root, model, destination='first-base')
            self.assertEqual(result.returncode, 0, result.stderr)
            spec = json.loads((root / 'first-base/keeper-spec.json').read_text())
            self.assertEqual(spec['install'], str(root / 'install-head'))

    def test_two_install_ab_stub_records_revisions_and_compares_outputs(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = frozen_fixture(root)
            result = stage_fixture(root, model)
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = root / 'stage'
            env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
            env.update(CHOPINFER_WEIGHT_KEEPER_DIR=str(root / 'synthetic-keeper'),
                       CHOPINFER_TEST_CHECKPOINT=str(root / 'checkpoint'))
            preflight = subprocess.run([sys.executable, '-B', str(staged / 'harness/preflight.py'), str(staged)],
                                       env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            cases = json.loads(next(staged.glob('validation/argv-*/preflight.json')).read_text())['cases']
            identities = {'a': {'install': 'base', 'revision': '1' * 40},
                          'b': {'install': 'head', 'revision': '2' * 40}}
            for row in cases:
                self.assertEqual(row['revision'], identities[row['arm']]['revision'])
                self.assertEqual(row['argv'][0], str(root / f"install-{row['install']}/bin/chopinfer"))
            for case, control in (('standard', {}), ('different', {'TUNE_STUB_PERTURB_INSTALL': 'head'}), ('cold', {})):
                if case == 'cold':
                    model['kv_pages'] = 'default'
                    # Replay must follow the baseline arm even when it names
                    # the second sealed install.
                    model['arms'][0]['install'] = 'head'
                    model['arms'][1]['install'] = 'base'
                    identities = {'a': {'install': 'head', 'revision': '2' * 40},
                                  'b': {'install': 'base', 'revision': '1' * 40}}
                    backing = root / 'rank.bin'
                    backing.write_bytes(b'\0' * 4096)
                    model['cold_engram'] = {'images': [{'path': str(backing), 'bytes': 4096,
                        'views': [{'name': 'fixture', 'offset': 0, 'bytes': 4096}]}]}
                    model['planning'].update(tokens_per_second={'1': 10}, ar_tokens_per_second={'1': 10})
                    result = stage_fixture(root, model, destination='cold-stage')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    staged = root / 'cold-stage'
                out = root / case
                result = subprocess.run([sys.executable, '-B', str(staged / 'harness/tune.py'), '--dry-run',
                                         '--output', str(out), '--payload-sha', stage.digest(staged / 'runtime-seal.json')],
                                        env={**env, **control}, text=True, capture_output=True, timeout=90)
                summary = json.loads((out / 'summary.json').read_text())
                with self.subTest(case=case):
                    if case == 'different':
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn('outputs differ', ' '.join(summary['errors']))
                        continue
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(summary['windows'], {'executed': 4, 'planned': 4})
                    self.assertTrue(summary['exactness']['passed'])
                    self.assertEqual(summary['arm_installs'], identities)
                    for row in report.load(out)[1]:
                        self.assertEqual({k: row[k] for k in ('install', 'revision')}, identities[row['arm']])
                    for label, row in summary['processes'].items():
                        self.assertEqual({k: row[k] for k in ('install', 'revision')}, identities[label.split('-')[0]])
                    launches = list(out.glob('*/server/launch.json')) + list(out.glob('references/*/launch.json'))
                    self.assertTrue(launches)
                    for path in launches:
                        launch = json.loads(path.read_text())
                        name = launch['install']
                        self.assertEqual(launch['serve_argv'][0], str(root / f'install-{name}/bin/chopinfer'))
                        self.assertIn(str(root / f'cache-{name}'), launch['serve_argv'])
                        stub = json.loads(path.with_name('stub-install.json').read_text())
                        self.assertEqual((stub['install'], stub['revision']), (name, launch['revision']))
                        proof = Path(launch['serve_argv'][launch['serve_argv'].index('--weight-keeper-identity') + 1])
                        self.assertEqual(proof.name, f'content-{name}.json')
                        self.assertEqual(stub['identity_sha256'], stage.digest(proof))
                    result = json.loads((out / 'ab.json').read_text())
                    self.assertEqual(result['arm_installs'], identities)
                    for arm in identities:
                        self.assertEqual(result['pairs']['a/b']['1']['arms'][arm]['revision'], identities[arm]['revision'])
                        self.assertIn(identities[arm]['revision'], (out / 'ab.md').read_text())
                    if case == 'standard':
                        self.assertGreater(summary['exactness']['matched_keys'], 0)
                        self.assertEqual(summary['page_cache_warmup']['install'], 'base')
                    else:
                        self.assertGreater(summary['exactness']['replayed_keys'], 0)
                        self.assertEqual({k: summary['reference_processes']['a'][k] for k in ('install', 'revision')}, identities['a'])
                        self.assertEqual(summary['reference_processes']['a']['kv_pages_per_rank'], [4096])
                        reference = json.loads((out / 'references/a/launch.json').read_text())
                        self.assertEqual(reference['serve_argv'][0], str(root / 'install-head/bin/chopinfer'))
                    path = next(out.glob('*/windows/*.json'))
                    row = json.loads(path.read_text())
                    row['revision'] = 'wrong'
                    path.write_text(json.dumps(row))
                    with self.assertRaisesRegex(ValueError, 'revision differs'):
                        report.ab(out)

    def test_weighted_targets_step_support_and_coverage(self):
        def mutate(root, function):
            for path in root.glob('*/windows/*.json'):
                row = json.loads(path.read_text())
                function(row)
                tune.r.write(path, row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = ab_fixture(root, target_cells=['decode:c8'])
            def incidental(row):
                row['cells']['i8-r16'] = {'cycle_ms': [20 if row['arm'] == 'a' else 21], 'execution_ms': [1]}
            mutate(root, incidental)
            out = report.ab(root)
            self.assertTrue(out['weighted']['a/b']['timing_accepted'])
            self.assertEqual(out['pairs']['a/b']['8']['cells']['i8-r16']['verdict'], 'informational')
            config['model']['target_cells'] = ['c8/i8-r16']
            tune.r.write(root / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'target'):
                report.ab(root)
            config['model']['target_cells'] = ['c8/i8-r32']
            config['model']['cell_weights'] = {'verify:r32': 0}
            tune.r.write(root / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'target'):
                report.ab(root)
            config['model'].update(target_cells=['decode:c8'], cell_weights={}, change='simplification')
            mutate(root, lambda row: row['cells'].pop('i8-r16', None) if row['arm'] == 'b' and row['repeat'] == 1 else None)
            tune.r.write(root / 'config.json', config)
            out = report.ab(root)['weighted']['a/b']
            self.assertTrue(out['timing_accepted'])
            self.assertEqual(out['coverage_gaps'], [])
            config['model']['required_shapes'] = {'8': ['i8-r16']}
            tune.r.write(root / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'required shape'):
                report.ab(root)
            config['model'].update(required_shapes={}, target_cells=['decode_step:c8'])
            def decode(row):
                row['cells']['i8-r8'] = {'cycle_ms': [10 if row['arm'] == 'a' else 9] * 20, 'execution_ms': []}
            mutate(root, decode)
            tune.r.write(root / 'config.json', config)
            out = report.ab(root)
            self.assertEqual(out['pairs']['a/b']['8']['cells']['i8-r8']['weight'], 1)
            self.assertTrue(out['weighted']['a/b']['timing_accepted'])
            self.assertIn('Coverage gaps:', (root / 'ab.md').read_text())

    def test_speedup_ignores_incidental_and_arm_specific_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ab_fixture(root, target_cells=['decode:c8', 'c8/i8-r32'])
            for path in root.glob('*/windows/*.json'):
                row = json.loads(path.read_text())
                if row['arm'] == 'a' or row['repeat'] == 0:
                    row['cells']['i8-r16'] = {'cycle_ms': [20], 'execution_ms': []}
                    row['cells']['i8-r64'] = {'cycle_ms': [20] * 20, 'execution_ms': []}
                tune.r.write(path, row)
            result = report.ab(root)
            self.assertTrue(result['weighted']['a/b']['timing_accepted'])
            self.assertEqual(result['weighted']['a/b']['coverage_gaps'], ['c8/i8-r64'])
            self.assertEqual(result['pairs']['a/b']['8']['cells']['i8-r16']['verdict'], 'informational')

            # The model that declares the repository job, found without naming it.
            configs = [json.loads(path.read_text()) for path in sorted(HERE.parents[1].glob('models/*/depth-tune.json'))]
            base = next(c for c in configs if 'confidence-ab-decode' in c.get('jobs', {}))
            model = {**base, **base['jobs']['confidence-ab-decode'], 'tensor_parallel_size': 1}
            # Recreate the repository job with resolved throughput gains and
            # the disjoint decode/verify shapes its two arms serve.
            import shutil
            for path in root.iterdir():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            config, _ = ab_fixture(root, **model)
            for path in root.glob('*/windows/*.json'):
                row = json.loads(path.read_text())
                c = row['concurrency']
                width = c if row['arm'] == 'd0' else 2 * c
                row['cells'] = {f'i{c}-r{width}': {'cycle_ms': [20] * 50, 'execution_ms': []}}
                tune.r.write(path, row)
            result = report.ab(root)
            self.assertTrue(result['weighted']['d0/d1-conf0']['timing_accepted'])
            cell = result['pairs']['d0/d1-conf0']['8']['cells']['i8-r8']
            self.assertEqual((cell['verdict'], cell['reason']), ('informational', 'arm-specific shape'))
            for requirement in ({'target_cells': ['c8/i8-r8']}, {'required_shapes': {'8': ['i8-r8']}},
                                {'required_shapes': {'1': ['i1-r1']}}):
                with self.subTest(requirement=requirement):
                    tune.r.write(root / 'config.json', {**config, 'model': {**model, **requirement}})
                    with self.assertRaisesRegex(ValueError, 'target|required shape'):
                        report.ab(root)

    def test_cell_weight_keys_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in ('verify:32', 'decode:c7', 'decode_step:c0', 'c8/i7-r32', 'c7/i7-r32', 'verify:r0'):
                with self.subTest(key=key):
                    ab_fixture(root, cell_weights={key: 0})
                    with self.assertRaisesRegex(ValueError, 'cell weight'):
                        report.ab(root)
            ab_fixture(root, cell_weights={'decode:c8': 2, 'decode_step:c8': 0, 'verify:r32': 3, 'c8/i8-r32': 4})
            result = report.ab(root)['weighted']['a/b']
            self.assertEqual(result['weight'], 6)

    def test_time_share_can_admit_a_rare_but_expensive_cell(self):
        rows = [{'request_seconds': 15, 'cells': {'i8-r16': {'cycle_ms': [1000]}}} for _ in range(4)]
        self.assertEqual(report.decision_cells({}, rows, 8, 'speedup')['i8-r16']['weight'], 0)
        self.assertEqual(report.decision_cells({'min_cell_time_share': .05}, rows, 8, 'speedup')['i8-r16']['weight'], 1)

    def test_energy_is_a_separate_provisional_veto(self):
        self.assertTrue(report.energy_check({'8': [1, 1.04, 1.04, 1], '16': [1, 1.04, 1.04, 1]})['veto'])
        self.assertFalse(report.energy_check({'8': [1, 1.03, 1.03, 1]})['veto'])
        self.assertFalse(report.energy_check({'8': [1, 1.10, 1.10, 1], '16': [1, .99, .99, 1]})['veto'])
        self.assertFalse(report.energy_check({'8': [None] * 4})['comparable'])

    def test_exactness_keys_are_scoped_to_concurrency_and_only_extras_replay(self):
        config = {'model': {'pairs': [['a', 'b']], 'concurrencies': [1, 2]}}
        deferred = []
        for c in (1, 2):
            for repeat in (0, 1):
                for arm in ('a', 'b'):
                    requests = [{'prompt': [i], 'seed': 100 * repeat + i, 'token_ids': [c],
                                 'selected_bits': [str(c)]} for i in range(c + 1)]
                    deferred.append({'arm': arm, 'concurrency': c, 'repeat': repeat, 'requests': requests})
        pending, matched = tune.exactness_plan(config, deferred)
        self.assertEqual(pending, {})
        self.assertEqual(matched, 10)
        deferred[-1]['requests'].pop()
        pending, _ = tune.exactness_plan(config, deferred)
        self.assertEqual(set(pending['a']), {2})
        self.assertEqual(len(pending['a'][2]), 1)
        deferred[-1]['requests'][0]['token_ids'] = [99]
        with self.assertRaisesRegex(RuntimeError, 'matched sampled outputs differ'):
            tune.exactness_plan(config, deferred)

    def test_unmatched_replay_fills_its_original_concurrency(self):
        import threading
        import time
        first_wave = threading.Barrier(4)
        lock = threading.Lock()
        seen = []
        def replay(item):
            with lock:
                index = len(seen)
                seen.append(item)
            if index < 4:
                first_wave.wait(timeout=5)
            time.sleep(.001)
        padding = tune.replay_cohort({'target': 'expected'}, 4, ['baseline'], replay)
        self.assertEqual(sum(value[1][1] is not None for value in seen), 1)
        self.assertGreaterEqual(padding, 3)
        self.assertEqual(len(seen), padding + 1)

    def test_cold_visit_assignment_is_deterministic_and_balances_lengths(self):
        tokens = [[i] * length for i, length in enumerate([8, 16, 32] * 4)]
        visits = [{'label': str(i)} for i in range(4)]
        pools = prompts.visit_pools(tokens, visits)
        assigned = [pools[str(i)].reserve(3) for i in range(4)]
        self.assertEqual([[len(p) for p in values] for values in assigned], [[8, 16, 32]] * 4)
        self.assertEqual(len({tuple(p) for values in assigned for p in values}), 12)
        again = prompts.visit_pools(tokens, visits)
        self.assertEqual(assigned, [again[str(i)].reserve(3) for i in range(4)])

    def test_cold_nearest_lengths_preserve_source_order_and_pool_capacity(self):
        tokens = [[i] * (32 + (i * 73) % 257) for i in range(256)]
        visits = [{'label': label, 'arm': label[0]} for label in ('a0', 'b0', 'b1', 'a1')]
        pools = prompts.visit_pools(tokens, visits)
        self.assertEqual([len(pool.remaining) for pool in pools.values()], [64] * 4)
        used = []
        for pool in pools.values():
            indices = [p[0] for p in pool.remaining]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(pool.assignment['indices'], indices)
            self.assertLessEqual(max(pool.assignment['neighbor_span_tokens']), 2)
            self.assertNotEqual([len(p) for p in pool.remaining], sorted(len(p) for p in pool.remaining))
            used.extend(indices)
        self.assertEqual(sorted(used), list(range(256)))
        again = prompts.visit_pools(tokens, visits)
        self.assertEqual({k: p.remaining for k, p in pools.items()}, {k: p.remaining for k, p in again.items()})

    def test_page_cache_warmup_is_required_before_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, summary = ab_fixture(root)
            summary.pop('page_cache_warmup')
            tune.r.write(root / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'warm-up'):
                report.ab(root)
            summary['page_cache_warmup'] = warmup_fixture(config)
            summary['page_cache_warmup']['windows'][0]['ended'] = 3
            tune.r.write(root / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'warm-up'):
                report.ab(root)

    def test_single_repeat_concurrency_warms_only_its_used_seed_set(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, summary = ab_fixture(root, concurrencies=[8, 16], concurrency_repeats={'8': 1})
            summary['page_cache_warmup']['windows'] = [w for w in summary['page_cache_warmup']['windows']
                                                       if (w['concurrency'], w['repeat']) != (8, 1)]
            tune.r.write(root / 'summary.json', summary)
            self.assertEqual(report.ab(root)['status'], 'valid')
            config['tokens'] = {'default': 64}
            state = {'tokens': [[1]]}
            result = {'started': 0, 'ended': 1, 'requests': [None] * 16,
                      'cached_prompt_tokens': 0, 'priming': {}}
            seal = {'config': config, 'install': {'install': 'install-head', 'revision': 'synthetic'}}
            with patch.object(tune, 'SEAL', seal, create=True), \
                    patch.object(tune, 'closed_loop', return_value=result) as controller:
                tune.page_cache_warmup(None, {'label': 'w0'}, state, None)
            self.assertEqual([call.args[1] for call in controller.call_args_list], ['w0-c8-r0', 'w0-c16-r0', 'w0-c16-r1'])

    def test_dry_run_control_cycles_ignore_wall_clock_skew(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = ab_fixture(root, concurrencies=[1], speculation_mode='off', max_depth=0,
                                   arms=[{'label': label} for label in ('a', 'b')])
            config['tokens'] = {'default': 64}
            seal = {'config': config, 'install': {'install': 'install-head', 'revision': 'synthetic'}}
            for dry in (True, False):
                with self.subTest(dry=dry), patch.object(tune, 'SEAL', seal, create=True):
                    state = {'tokens': [[1]], 'processes': {}, 'exact': {}}
                    rows = []
                    for arm, cycle_ms in zip(tune.visits(config), (11, 10, 10, 10)):
                        # Inject an A1 wall-clock offset into otherwise identical steps.
                        steps = [{'batch': i, 'prefill_items': 0, 'decode_rows': 1,
                                  'verify_items': 0, 'verify_rows': 0,
                                  'completed_unix_ns': 1_000_000_000 + i * cycle_ms * 1_000_000,
                                  'execution_us': 1000, 'route': {'fallback_launches': 0, 'families': {}}}
                                 for i in range(21)]
                        window = {'steps': steps, 'cold_steps': steps, 'started': 2, 'ended': 17,
                                  'window_seconds': 15, 'cached_prompt_tokens': 0,
                                  'priming': priming_fixture(1), 'replacement_submissions': 1,
                                  'prefill_share': 0, 'tokens': 20,
                                  'faults': {'complete': True, 'major_faults': 0},
                                  'acceptance': None, 'energy': {}, 'requests': []}
                        record = {'windows': {}, 'confidence': {'enabled': False}}
                        state['processes'][arm['label']] = record
                        with patch.object(tune, 'closed_loop', return_value=window):
                            tune.run_ab_windows(SimpleNamespace(dry=dry), arm, state, None, root / arm['label'])
                        row, = record['windows'].values()
                        self.assertEqual(row['steps'], steps)
                        self.assertEqual(row['full']['mean_step_ms'], cycle_ms)
                        self.assertEqual(row['cells']['i1-r1']['cycle_ms'], [1 if dry else cycle_ms] * 20)
                        rows.append(row)
                    control = report.control_offsets(config, rows)
                    self.assertEqual(control['status'], 'clear' if dry else 'offset')

    def test_a1_control_offset_invalidates_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, summary = ab_fixture(root, control_shapes={'8': ['i8-r16']})
            def controls(values):
                for visit, value in zip(tune.visits(config), values):
                    path = root / visit['label'] / 'windows/c8.json'
                    row = json.loads(path.read_text())
                    row['cells']['i8-r16'] = {'cycle_ms': [value] * 20, 'execution_ms': []}
                    tune.r.write(path, row)
            for values in ([11, 10, 10.01, 10], [9, 10, 10.01, 10]):
                with self.subTest(values=values):
                    controls(values)
                    rows = report.load(root)[1]
                    validity = report.ab_validity(config, rows, summary)
                    self.assertEqual(validity['status'], 'invalid')
                    self.assertIn('A1 offset', validity['reason'])
                    self.assertFalse(report.ab(root)['contract_a_admissible'])
                    self.assertEqual(report.ab(root)['pairs'], {})
            controls([10.05, 10, 10, 10])
            self.assertEqual(report.ab(root)['status'], 'valid')
            controls([11, 10, 10, 11])
            self.assertEqual(report.ab(root)['status'], 'valid')
            # Undrafted controls are discovered without an explicit list.
            config['model'].pop('control_shapes')
            tune.r.write(root / 'config.json', config)
            for visit, value in zip(tune.visits(config), [11, 10, 10, 10]):
                path = root / visit['label'] / 'windows/c8.json'
                row = json.loads(path.read_text())
                row['cells']['i8-r8'] = {'cycle_ms': [value] * 20, 'execution_ms': []}
                tune.r.write(path, row)
            self.assertEqual(report.ab(root)['status'], 'invalid')

    def test_cold_pause_timeout_does_not_resume_requests(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch
        original_wait = threading.Event.wait
        def wait(event, timeout=None):
            return False if timeout == 120 else original_wait(event, timeout)
        event = {'usage': {'completion_tokens': 1, 'prompt_tokens': 1,
                          'prompt_tokens_details': {'cached_tokens': 0}},
                 'choices': [{'token_ids': [1], 'logprobs': {'token_logprobs': [-.1]}, 'finish_reason': 'length'}]}
        with tempfile.TemporaryDirectory() as directory:
            server = SimpleNamespace(directory=Path(directory), dry=True, url='stub')
            (server.directory / 'server.log').touch()
            with patch.object(threading.Event, 'wait', wait), \
                    patch.object(tune.r, 'remaining', return_value=120), \
                    patch.object(tune.r, 'stream', return_value=[event]) as stream:
                with self.assertRaisesRegex(RuntimeError, 'cold priming pause timed out'):
                    tune.closed_loop(server, 'timeout', lambda i: [1], 1,
                                     {'temperature': 1, 'top_p': 1}, [1], None,
                                     before_measure=lambda: self.fail('timed-out priming reached preparation'))
            self.assertEqual(stream.call_count, 1)

    def test_stage_refuses_work_and_cold_prompt_overruns(self):
        model = {**MODEL, 'mode': 'ab', 'concurrencies': [1],
                 'arms': [{'label': 'a', 'depth': 1}, {'label': 'b', 'depth': 1}], 'pairs': [['a', 'b']],
                 'planning': {'startup_seconds': 1, 'process_overhead_seconds': 1, 'settle_seconds': 1,
                              'tokens_per_second': {'1': 20}, 'ar_tokens_per_second': {'1': 20}}}
        config = {'model': model, 'tokens': {'default': 256}, 'warm': [], 'warm_tokens': 8,
                  'work_seconds': 1, 'kinds': {'sampled': 1}}
        with self.assertRaisesRegex(ValueError, 'work budget'):
            stage.check_budget(config, 256)
        config.update(work_seconds=10000, cold_engram={'warm_fraction': .1})
        count = stage.cold_budget(config)['prompts_required']
        with self.assertRaisesRegex(ValueError, 'cold prompt pool'):
            stage.check_budget(config, count - 1)
        self.assertLess(stage.check_budget(config, count), config['work_seconds'])
        self.assertEqual(config['cold_engram']['prompts_required'], count)
        sizes = config['cold_engram']['balanced_pool_sizes']
        actual = prompts.visit_pools([[i] * (i + 1) for i in range(count)], tune.visits(config))
        self.assertEqual(sizes, {label: len(pool.remaining) for label, pool in actual.items()})
        self.assertTrue(all(sizes[label] >= need for label, need in config['cold_engram']['prompt_needs'].items()))

    def test_cold_file_seal_detects_changed_metadata_without_reading_pages(self):
        import faults
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image'
            path.write_bytes(bytes(4096))
            identity = faults.file_identity(path)
            self.assertEqual(identity['size'], 4096)
            self.assertEqual(identity, faults.file_identity(path))
            os.utime(path, ns=(identity['mtime_ns'], identity['mtime_ns'] + 1))
            self.assertNotEqual(identity, faults.file_identity(path))

    def test_declared_target_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ab_fixture(root, target_cells=['decode:c8', 'c8/i8-r64'])
            with self.assertRaisesRegex(ValueError, 'target'):
                report.ab(root)

    def test_recorded_adapter_refuses_void_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tune.r.write(root / 'source/config.json', {'model': {'tensor_parallel_size': 1,
                                                                  'kv_pages_line': 'fixture {pages}'}})
            tune.r.write(root / 'source/summary.json', {'status': 'void', 'errors': ['power changed']})
            with self.assertRaisesRegex(ValueError, 'void'):
                recorded.main(root / 'source', root / 'output')

    def test_void_run_cannot_be_pooled_by_other_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(CONFIG))
            (root / 'summary.json').write_text(json.dumps({'status': 'void'}))
            for reader in (report.costs, report.evaluate):
                with self.subTest(reader=reader.__name__), self.assertRaisesRegex(ValueError, 'void'):
                    reader(root)

    def test_cold_ab_cpu_stub_and_negative_controls(self):
        import hashlib
        import os
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = root / 'stage' / 'harness'
            harness.mkdir(parents=True)
            for name in stage.HARNESS:
                shutil.copy2(HERE / name, harness / name)
            config = {'name': 'CPU cold A/B', 'work_seconds': 90, 'kinds': {'sampled': 1},
                      'tokens': {'default': 64}, 'warm': [], 'warm_tokens': 8,
                      'sampling': {'seed': 7, 'temperature': 1.0, 'top_p': 1.0},
                      'cold_engram': {'warm_fraction': .1},
                      'model': {'mode': 'ab', 'repeats': 2, 'concurrencies': [1], 'speculation_mode': 'balanced',
                                'arms': [{'label': 'a', 'depth': 1}, {'label': 'b', 'depth': 1}], 'pairs': [['a', 'b']],
                                'served_model_name': 'fixture', 'checkpoint_env': 'CHOPINFER_TEST_CHECKPOINT',
                                'tensor_parallel_size': 1, 'kv_pages': 'default', 'kv_pages_line': 'fixture: {pages} KV pages',
                                'serve_arguments': [], 'max_depth': 1, 'verify_rows': 32,
                                'planning': {'process_reserve_seconds': 1}}}
            seal = {'ready': False, 'config': config, 'install': {'install': str(root / 'install'),
                    'cache': str(root / 'cache'), 'revision': 'synthetic'}, 'harness_source': str(HERE.parents[1]),
                    'prompts': [{'prompt': f'Case {i}: ' + 'x' * (128 + (i * 73) % 257 * 8),
                                 'sha256': str(i)} for i in range(128)],
                    'sha256': {}}
            body = json.dumps(seal).encode()
            (harness.parent / 'runtime-seal.json').write_bytes(body)
            payload = hashlib.sha256(body).hexdigest()
            env = {k: v for k, v in os.environ.items() if not k.startswith('TUNE_')}
            env.update(CHOPINFER_WEIGHT_KEEPER_DIR=str(root / 'synthetic-keeper'),
                       TUNE_STUB_RESIDENCY='0.8', TUNE_STUB_TOKEN_BYTES='4')
            for case, changes, expected in [('valid', {}, 'passed'),
                                            ('keeper', {'CHOPINFER_WEIGHT_KEEPER_DIR': ''}, 'failed'),
                                            ('cached', {'TUNE_STUB_CACHED': 'a-r0'}, 'failed'),
                                            ('power', {'TUNE_DRY_POWER_CHANGE_LABEL': 'after-a-r0'}, 'void'),
                                            ('slowcold', {'TUNE_STUB_PREFILL_DELAY_SECONDS': '1.5'}, 'passed'),
                                            ('standard', {}, 'passed')]:
                out = root / case
                if case == 'standard':
                    config.pop('cold_engram')
                    config['model']['concurrencies'] = [8, 16]
                    body = json.dumps(seal).encode()
                    (harness.parent / 'runtime-seal.json').write_bytes(body)
                    payload = hashlib.sha256(body).hexdigest()
                result = subprocess.run([sys.executable, '-B', str(harness / 'tune.py'), '--dry-run',
                                         '--output', str(out), '--payload-sha', payload],
                                        env={**env, **changes}, text=True, capture_output=True, timeout=90)
                summary = json.loads((out / 'summary.json').read_text())
                with self.subTest(case=case):
                    self.assertEqual(summary['status'], expected, result.stdout + result.stderr)
                    self.assertEqual(result.returncode, 0 if expected == 'passed' else 1)
                if case in ('valid', 'standard'):
                    rows = report.load(out)[1]
                    self.assertEqual(len(rows), 4 if case == 'valid' else 8)
                    self.assertTrue(all(r['priming']['window_started'] >= r['priming']['last_completed'] for r in rows))
                    self.assertTrue(all(r['full']['steps'] > 0 for r in rows))
                    self.assertTrue(all(.15 <= r['window_seconds'] < 1 for r in rows))
                    self.assertTrue(all(r['window_seconds'] == r['request_seconds'] for r in rows))
                    if case == 'standard':
                        self.assertEqual(len(summary['page_cache_warmup']['windows']), 4)
                        continue
                    cold_report = json.loads((out / 'ab.json').read_text())
                    self.assertFalse(cold_report['contract_a_admissible'])
                    self.assertFalse(cold_report['weighted']['a/b']['timing_accepted'])
                    self.assertEqual(cold_report['weighted']['a/b']['verdict'], 'cold, unmatched prompts')
                    self.assertTrue(all(r['cold_engram']['state'] == 'partly warm' for r in rows))
                    prompts_used = [tuple(q['prompt']) for r in rows for q in r['requests']]
                    self.assertEqual(len(prompts_used), len(set(prompts_used)))
                    self.assertTrue(all(r['mapped_gather_us'] for r in rows))
                    lengths = [len(q['prompt']) for r in rows for q in r['requests']]
                    self.assertGreater(max(lengths) - min(lengths), 100)
                    self.assertEqual(sum(len(p['indices']) for p in summary['cold_prompt_assignment'].values()), 128)
                if case == 'power':
                    self.assertEqual(json.loads((out / 'ab.json').read_text())['pairs'], {})
                if case == 'slowcold':
                    rows = report.load(out)[1]
                    self.assertTrue(all(row['request_tokens_per_second'] == 0 for row in rows))
                    self.assertTrue(all(min(row['ttft_seconds']) >= .15 for row in rows))
                    self.assertTrue(all(row['prefill_execution_us'] for row in rows))

    def test_ab_seed_sets_match_pairs_and_differ_between_pairs(self):
        config = {'sampling': {'seed': 41, 'temperature': 1.0, 'top_p': 1.0}}
        first, second = (tune.seed_set(config, repeat, 8) for repeat in (0, 1))
        self.assertEqual(first, tune.seed_set(config, 0, 8))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(len(set(first)), 8)

    def test_ab_requires_keeper_for_speculative_visits(self):
        model = {'mode': 'ab', 'speculation_mode': 'balanced'}
        with self.assertRaisesRegex(RuntimeError, 'keeper'):
            tune.require_keeper(model, None)
        tune.require_keeper(model, {'directory': 'synthetic'})

    def test_visit_fault_evidence_is_complete_and_bounded(self):
        import faults
        faults.validate({'complete': True, 'major_faults': 1000})
        for evidence in ({}, {'complete': False, 'major_faults': 0},
                         {'complete': True, 'major_faults': 1001}):
            with self.assertRaisesRegex(ValueError, 'fault'):
                faults.validate(evidence)

    def test_missing_optional_route_counts_are_unknown_in_reports(self):
        row = {'request_tokens_per_second': 1,
               'full': {'tokens_per_second': 1, 'histogram': {}}}
        self.assertIsNone(report.measure([row])['fallback_launches'])
        self.assertEqual(report.measure([{**row, 'fallback_launches': 0}])['fallback_launches'], 0)
        self.assertIsNone(report.measure([row, {**row, 'fallback_launches': 2}])['fallback_launches'])
        self.assertEqual(report.measure([{**row, 'fallback_launches': 2}] * 2)['fallback_launches'], 4)

    def test_staged_job_preserves_fallback_allowances(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / 'workload.json'
            workload.write_text(json.dumps({'name': 'fixture', 'prompts': 1}))
            model = {**MODEL, 'model': 'fixture', 'workload': str(workload),
                     'concurrencies': [1], 'tokens': {'default': 32}, 'warm': [], 'warm_tokens': 8,
                     'kinds': {'sampled': 1}, 'sampling': {'seed': 1}, 'selection': {},
                     'work_seconds': 60, 'queue_timeout_seconds': 90,
                     'checkpoint_env': 'CHOPINFER_TEST_CHECKPOINT', 'tensor_parallel_size': 1,
                     'allow_fallbacks': {'mapped_rows': 'Prepared prefill pending'},
                     'jobs': {'small': {'concurrencies': [1]}}}
            source = root / 'model.json'
            source.write_text(json.dumps(model))
            destination = root / 'staged'
            argv = ['stage.py', str(destination), '--model', str(source), '--job', 'small', '--template']
            with patch.object(sys, 'argv', argv), patch.object(stage, 'check_budget', return_value=1), \
                    patch.object(stage.builder, 'build', return_value={'prompts': ['fixture'], 'sources': []}):
                stage.main()
            config = json.loads((destination / 'runtime-seal.json').read_text())['config']
            allow = config.get('allow_fallbacks', {})
            steps = [{'route': {'fallback_launches': 4, 'families': {'mapped_rows': 4}}}]
            self.assertEqual(tune.check_fallbacks('window', steps, allow)['fallback_launches'], 4)
            with self.assertRaisesRegex(RuntimeError, 'unallowed fallback'):
                tune.check_fallbacks('window', [{'route': {'fallback_launches': 1,
                                                         'families': {'unrelated': 1}}}], allow)

    def test_fallback_gate_requires_complete_counts_and_justified_families(self):
        steps = [{'route': {'fallback_launches': 3, 'families': {'projection': 3}}}]
        with self.assertRaisesRegex(RuntimeError, 'unallowed fallback'):
            tune.check_fallbacks('window', steps, {})
        self.assertEqual(tune.check_fallbacks('window', steps, {'projection': 'Range qualification pending'})['fallback_launches'], 3)
        for incomplete in ([], [{}]):
            with self.assertRaisesRegex(RuntimeError, 'missing route counters'):
                tune.check_fallbacks('window', incomplete, {})
        with self.assertRaisesRegex(RuntimeError, 'reason'):
            tune.check_fallbacks('window', steps, {'projection': ''})
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            tune.check_fallbacks('window', [{'route': {'fallback_launches': 0, 'families': {'projection': 3}}}], {'projection': 'Pending'})


    def test_confidence_startup_must_match_the_requested_arm(self):
        flag = ['--speculation-confidence', '{"floor":0}']
        on = 'startup speculation_confidence: on, floor 0, 5 positions\n'
        off = 'startup speculation_confidence: off, the package returns host values\n'
        self.assertTrue(tune.confidence_startup(on, flag)['enabled'])
        self.assertFalse(tune.confidence_startup(off, [])['enabled'])
        self.assertFalse(tune.confidence_startup('', [])['enabled'])
        for log, args in [(off, flag), ('', flag), (on, []), (on, ['--speculation-confidence', 'off']),
                          (on + off, flag)]:
            with self.assertRaises(RuntimeError):
                tune.confidence_startup(log, args)

    def test_allocated_steps_keep_width_and_reserved_shape_and_assert_execution(self):
        steps = [{'batch': i, 'completed_unix_ns': i * 1_000_000, 'prefill_items': 0,
                  'decode_rows': 0, 'verify_items': 8, 'verify_rows': 32, 'reserved_rows': 48,
                  'verify_emitted_tokens': 12, 'accepted_tokens': 8, 'execution_us': 800} for i in range(3)]
        self.assertEqual(tune.cells(steps),
                         {'i8-r32-v8-R48': {'cycle_ms': [1.0, 1.0], 'execution_ms': [0.8, 0.8]}})
        full = tune.full_concurrency(steps, 8)
        self.assertEqual(full['histogram'], {'i8-r32-v8-R48': 2})
        self.assertEqual((full['allocated_steps'], full['allocated_share']), (2, 1.0))
        # Older traces omit the mandatory token of items allocated zero
        # drafts. Benchmark requests ignore EOS and reserve their bonus.
        self.assertEqual(full['tokens'], 32)
        tune.check_depth('allocated', steps, 3)
        self.assertEqual(tune.check_allocations('allocated', steps, True, True), 3)
        with self.assertRaisesRegex(RuntimeError, 'no allocated steps'):
            tune.check_allocations('ignored', [], True, True)
        with self.assertRaisesRegex(RuntimeError, 'without confidence'):
            tune.check_allocations('unrequested', steps, False, False)
        self.assertEqual(tune.check_allocations('full width', [], True, False), 0)

    def test_arms_may_run_a_production_depth_schedule(self):
        config = {'model': {'speculation_mode': 'balanced', 'mode': 'ab', 'extra_arguments': [],
                            'arms': [{'label': 'f', 'depths': '1:5,8:3,16:3,32:0', 'install': 'base'},
                                     {'label': 'h', 'depths': '1:5,8:3,16:3,32:0', 'install': 'head'}]}}
        declared = tune.arms(config)
        self.assertEqual([arm['depth'] for arm in declared], [5, 5])
        arguments = tune.arm_arguments(config, declared[0])
        self.assertEqual(arguments.count('--speculation-depths'), 1)
        self.assertEqual(arguments[arguments.index('--speculation-depths') + 1], '1:5,8:3,16:3,32:0')
        schedule = declared[0]['depths']
        self.assertEqual([tune.step_depth(schedule, i) for i in (1, 7, 8, 16, 31, 32, 64)], [5, 5, 3, 3, 3, 0, 0])
        step = {'batch': 1, 'verify_items': 16, 'verify_rows': 64}
        tune.check_depth('schedule', [step], schedule)
        with self.assertRaisesRegex(RuntimeError, 'above depth 0'):
            tune.check_depth('schedule', [{**step, 'verify_items': 32, 'verify_rows': 33}], schedule)
        for bad in ('8:3', '1:5,1:3', '1:5,8:x', ''):
            with self.subTest(schedule=bad), self.assertRaisesRegex(ValueError, 'depth schedule'):
                tune.arms({'model': {**config['model'], 'arms': [{'label': 'f', 'depths': bad}]}})
        with self.assertRaisesRegex(ValueError, 'replaces depth'):
            tune.arms({'model': {**config['model'], 'arms': [{'label': 'f', 'depth': 3, 'depths': '1:3'}]}})
        import stub
        mode, schedule, rows = stub.options(arguments)
        self.assertEqual((mode, schedule, rows), ('balanced', [(1, 5), (8, 3), (16, 3), (32, 0)], 32))

    def test_stage_files_are_protected_from_other_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'stage'
            root.mkdir()
            keeper = root / 'keeper-spec.json'
            keeper.write_text('{}\n')
            keeper.chmod(0o664)
            job = root / 'tune.job'
            job.write_text('# timeout 60\n')
            job.chmod(0o775)
            root.chmod(0o775)
            stage.protect(root)
            self.assertEqual(keeper.stat().st_mode & 0o777, 0o644)
            self.assertEqual(job.stat().st_mode & 0o777, 0o755)
            self.assertEqual(root.stat().st_mode & 0o777, 0o755)

    def test_allocation_expectation_uses_capacity_width_and_common_reservation(self):
        config = {'model': {**MODEL, 'verify_widths': [6, 8, 16, 32]}}
        for c, capacity, depth, expected in [(8, 48, 3, True), (16, 64, 1, True),
                                              (8, 32, 3, False), (32, 64, 3, False)]:
            arm = {'depth': depth, 'arguments': ['--speculative-rows', str(capacity)]}
            self.assertEqual(tune.allocation_possible(config, arm, c), expected)
        # Decided in advance, so never allocated: one item takes a prefix,
        # and a floor of one fills 16 items' 32 rows.
        for c, floor, expected in [(1, 1, False), (16, 1, False), (16, 0, True), (8, 1, True)]:
            arm = {'depth': 3, 'arguments': ['--speculative-rows', '48', '--speculation-confidence',
                                             json.dumps({'floor': floor})]}
            self.assertEqual(tune.allocation_possible(config, arm, c), expected, (c, floor))

    def test_ab_visits_share_overlapping_pairs_and_keep_short_controls(self):
        model = {**MODEL, 'mode': 'ab', 'concurrencies': [1, 8, 16, 32],
                 'concurrency_repeats': {'1': 1, '32': 1},
                 'arms': [{'label': label, 'depth': 3} for label in ('a', 'cap', 'b')],
                 'pairs': [['a', 'b'], ['cap', 'b']]}
        config = {'model': model, 'kinds': {'sampled': 1.0}}
        visits = tune.visits(config)
        self.assertEqual([v['arm'] for v in visits], ['a', 'cap', 'b', 'b', 'cap', 'a'])
        self.assertEqual([len(v['passes'][0][1]) for v in visits], [4, 4, 4, 2, 2, 2])
        self.assertEqual(tune.planned_windows(config), 18)
        with self.assertRaisesRegex(ValueError, 'sampled only'):
            tune.visits({**config, 'kinds': {'greedy': 1.0}})
        with self.assertRaisesRegex(ValueError, 'baseline before'):
            tune.visits({**config, 'model': {**model, 'pairs': [['b', 'a']]}})

    def test_ab_report_keeps_arm_identity_order_and_refuses_missing_repeats(self):
        model = {**MODEL, 'mode': 'ab', 'concurrencies': [8],
                 'arms': [{'label': label, 'depth': 3} for label in ('a', 'b')], 'pairs': [['a', 'b']]}
        config = {**CONFIG, 'model': model, 'kinds': {'sampled': 1.0},
                  'sampling': {'seed': 41, 'temperature': 1.0, 'top_p': 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(config))
            power = {'enforced_power_limits_w': [250], 'sm_clock_mhz': [1800]}
            summary = {'power': {'before': power, 'after': power}, 'processes': {},
                       'exactness': {'passed': True}, 'page_cache_warmup': warmup_fixture(config)}
            for visit in tune.visits(config):
                label, repeat = visit['arm'], visit['passes'][0][0]
                window(root, visit['label'], 'sampled', 8, 3, (400 if label == 'a' else 440) + repeat,
                       {'i8-r32-v8-R48' if label == 'b' else 'i8-r32-v8': 2})
                path = root / visit['label'] / 'windows' / 'sampled-c8.json'
                row = json.loads(path.read_text())
                row.update(arm=label, visit=visit['visit'], repeat=repeat,
                           priming=priming_fixture(8), replacement_submissions=1, prefill_share=.01,
                           seed_set=tune.seed_set(config, repeat, 8), window_seconds=15,
                           temperature=1.0, cached_prompt_tokens=0, request_seconds=15,
                           faults={'complete': True, 'major_faults': 0},
                           allocated_steps=2 if label == 'b' else 0,
                           cells={'i8-r32-v8-R48' if label == 'b' else 'i8-r32':
                                  {'cycle_ms': [20, 22], 'execution_ms': [19, 21]}})
                row['full'].update(steps=2, tokens=40, allocated_steps=row['allocated_steps'])
                row['energy'].update(sm_mhz_range=[[1700, 1900]], enforced_power_limits_w=[250])
                summary['processes'][visit['label']] = {'weights': {'source': 'borrowed'},
                    'process_group': visit['visit'] + 1,
                    'faults': row['faults'], 'power_before': power, 'power_after': power}
                path.write_text(json.dumps(row))
            (root / 'summary.json').write_text(json.dumps(summary))
            out = report.ab(root)['pairs']['a/b']['8']
            self.assertEqual(out['order'], ['a', 'b', 'b', 'a'])
            self.assertEqual(out['arms']['b']['request'], [440, 441])
            self.assertEqual(out['arms']['b']['allocated_share'], 1.0)
            self.assertEqual(out['arms']['b']['tokens_per_step'], 20)
            self.assertEqual(out['verdict'], 'resolved gain')
            costs = report.costs(root)['cells']
            self.assertEqual(costs['i8-r32-v8-R48']['reserved'], 48)
            self.assertIn('i8-r32', costs)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'visit order'):
                report.ab(root)

    def test_paired_verdict_thresholds_and_regression_floor(self):
        self.assertEqual(report.paired([100, 102, 102, 100])['verdict'], 'resolved gain')
        self.assertEqual(report.paired([100, 101, 99, 100])['verdict'], 'neutral')
        self.assertEqual(report.paired([100, 99.6, 99.6, 100])['verdict'], 'neutral')
        self.assertEqual(report.paired([100, 99, 99, 100])['verdict'], 'resolved regression')
        self.assertFalse(report.paired([100, 101, 101, 100], 'variant')['accepted'])
        self.assertTrue(report.paired([100, 102, 102, 100], 'table_row')['accepted'])
        self.assertTrue(report.paired([100, 101.5, 101.5, 100], 'variant')['accepted'])
        self.assertFalse(report.paired([100, 99, 99, 100], 'simplification')['accepted'])

    def test_ab_validity_refuses_corrupted_evidence_and_voids_changed_power(self):
        import copy
        model = {**MODEL, 'mode': 'ab', 'concurrencies': [1],
                 'arms': [{'label': label, 'depth': 0} for label in ('a', 'b')], 'pairs': [['a', 'b']]}
        config = {'model': model, 'kinds': {'sampled': 1}, 'sampling': {'seed': 3}}
        power = {'enforced_power_limits_w': [250], 'sm_clock_mhz': [1800]}
        summary = {'power': {'before': power, 'after': power}, 'processes': {}, 'exactness': {'passed': True}, 'page_cache_warmup': warmup_fixture(config)}
        rows = []
        for visit in tune.visits(config):
            repeat = visit['passes'][0][0]
            row = {'label': visit['label'], 'arm': visit['arm'], 'visit': visit['visit'],
                   'kind': 'sampled', 'concurrency': 1, 'repeat': repeat,
                   'priming': priming_fixture(1), 'replacement_submissions': 1, 'prefill_share': .01,
                   'seed_set': tune.seed_set(config, repeat, 1), 'window_seconds': 15, 'temperature': 1.0,
                   'cached_prompt_tokens': 0, 'request_seconds': 15, 'faults': {'complete': True, 'major_faults': 0},
                   'energy': {'sm_mhz_range': [[1800, 1800]], 'enforced_power_limits_w': [250]}}
            rows.append(row)
            summary['processes'][visit['label']] = {'weights': {'source': 'borrowed'}, 'faults': row['faults'],
                                                   'process_group': visit['visit'] + 1,
                                                   'power_before': power, 'power_after': power}
        self.assertEqual(report.ab_validity(config, rows, summary)['status'], 'valid')
        for field, value in [('seed_set', [9]), ('cached_prompt_tokens', 1),
                             ('faults', {'complete': False, 'major_faults': 0}),
                             ('faults', {'complete': True, 'major_faults': 1001}), ('visit', 9),
                             ('window_seconds', 20), ('request_seconds', 14.9),
                             ('priming', {}), ('priming', {**priming_fixture(1), 'window_started': .5}),
                             ('priming', {**priming_fixture(1), 'per_slot_completed': {}}),
                             ('replacement_submissions', None), ('replacement_submissions', -1),
                             ('replacement_submissions', 1.5)]:
            altered = copy.deepcopy(rows)
            altered[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                report.ab_validity(config, altered, summary)
        for altered in (rows[:-1], rows + rows[:1]):
            with self.assertRaisesRegex(ValueError, 'counts'):
                report.ab_validity(config, altered, summary)
        altered = copy.deepcopy(summary)
        altered['processes']['b-r0']['weights']['source'] = 'loaded'
        with self.assertRaisesRegex(ValueError, 'keeper'):
            report.ab_validity(config, rows, altered)
        altered = copy.deepcopy(summary)
        altered['power']['after'] = {**power, 'enforced_power_limits_w': [249]}
        self.assertEqual(report.ab_validity(config, rows, altered)['status'], 'void')

    def test_cold_prompts_never_repeat_and_exhaust_explicitly(self):
        pool = prompts.Unseen([[1, 2], [3, 4]])
        self.assertEqual(pool.take(), [1, 2])
        self.assertEqual(pool.take(), [3, 4])
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            pool.take()
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            prompts.Unseen([[1, 2], [1, 2]])

    def test_cold_residency_checks_bounds_and_refuses_symlinks(self):
        import faults
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rank.img'
            path.write_bytes(bytes(8192))
            fd = faults.open_regular(path)
            try:
                measured = faults.residency(fd, 1, 4096)
                self.assertEqual(measured['pages'], 2)
                self.assertTrue(0 <= measured['resident_fraction'] <= 1)
                with self.assertRaises(ValueError):
                    faults.residency(fd, 8192, 1)
            finally:
                os.close(fd)
            link = path.with_suffix('.link')
            link.symlink_to(path)
            with self.assertRaises(OSError):
                faults.open_regular(link)

    def test_cold_control_requires_ack_before_file_advice(self):
        import faults
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'server.log').write_text('')
            image = root / 'rank.img'
            image.write_bytes(bytes(4096))
            server = SimpleNamespace(directory=root, env={}, dry=False)
            control = faults.ColdEngram(server, {'images': [{'path': str(image), 'bytes': 4096,
                'views': [{'name': 'rows', 'offset': 0, 'bytes': 4096}]}]})
            self.assertEqual(server.env['CHOPINFER_MAPPED_ROW_CONTROL'], str(control.path))

            def acknowledge(unused):
                command = json.loads(control.path.read_text())
                self.assertTrue(command['reset_cache'] and command['release_page_tables'])
                with (root / 'server.log').open('a') as log:
                    log.write(json.dumps({'schema': 'chopinfer.mapped-row-policy.v1', 'status': 'applied',
                                          'epoch': command['epoch'], 'reset_cache': True,
                                          'released_mapping_bytes': 4096}) + '\n')

            try:
                with patch('faults.time.sleep', side_effect=acknowledge), patch('os.posix_fadvise') as advice:
                    self.assertEqual(control.prepare()['epoch'], 1)
                    self.assertEqual(control.prepare()['epoch'], 2)
                    self.assertEqual(advice.call_count, 2)
                # A dry run applies the control protocol but never advises real image pages.
                server.dry = True
                with patch('faults.time.sleep', side_effect=acknowledge), patch('os.posix_fadvise') as advice:
                    self.assertTrue(control.prepare()['synthetic'])
                    advice.assert_not_called()
                server.dry = False
                with patch('runner.remaining', return_value=.01), patch('os.posix_fadvise') as advice:
                    with self.assertRaisesRegex(RuntimeError, 'acknowledged'):
                        control.prepare()
                    advice.assert_not_called()
            finally:
                control.close()

    def test_every_prompt_prefix_keeps_the_mix(self):
        mix = [{'kind': 'completion', 'weight': 5}, {'kind': 'edit', 'weight': 3}, {'kind': 'chat', 'weight': 2}]
        kinds = prompts.order(mix, 32)
        self.assertEqual(kinds[:4], ['completion', 'edit', 'chat', 'completion'])
        for n in range(1, 33):
            for entry in mix:
                share = kinds[:n].count(entry['kind'])
                self.assertLessEqual(abs(share - n * entry['weight'] / 10), 1, (n, entry))

    def test_a_depth_measures_only_the_concurrencies_its_rows_can_change(self):
        config = {'model': MODEL}
        cs = lambda d, every=False: sorted({c for _, c in tune.windows_for(config, d, every)})
        self.assertEqual(cs(0), [1, 2, 4, 8, 16, 32])
        self.assertEqual(cs(1), [1, 2, 4, 8, 16])
        self.assertEqual(cs(3), [1, 2, 4, 8])
        self.assertEqual(cs(5), [1, 2, 4])
        self.assertEqual(cs(5, True), [1, 2, 4, 8, 16, 32])

    def test_one_server_per_depth_starts_with_the_deepest(self):
        self.assertEqual(tune.order({'model': MODEL}), ['d5', 'd0', 'd1', 'd2', 'd3', 'd4'])
        named = {**MODEL, 'arms': [{'label': 'g64', 'depth': 5, 'arguments': ['--speculation-graph-rows', '64']},
                                   {'label': 'g32', 'depth': 5}]}
        self.assertEqual(tune.arms({'model': named})[1], {'label': 'g32', 'depth': 5, 'arguments': []})

    def test_the_job_plans_every_arms_windows_per_repeat(self):
        self.assertEqual(tune.planned_windows({'model': MODEL}), 2 * 2 * (6 + 5 + 4 + 4 + 3 + 3))
        costs = {**MODEL, 'concurrencies': [8, 16], 'verify_rows': 128,
                 'arms': [{'label': 'd5', 'depth': 5}, {'label': 'd0', 'depth': 0}]}
        self.assertEqual(tune.planned_windows({'model': costs, 'kinds': {'greedy': 1.0}}), 2 * (2 + 2))

    def test_kinds_select_the_windows(self):
        config = {'model': MODEL, 'kinds': {'greedy': 1.0}}
        self.assertEqual(tune.windows_for(config, 5), [('greedy', 1), ('greedy', 2), ('greedy', 4)])

    def test_the_shallowest_depth_within_the_margin_wins(self):
        rates = {  # C: {depth: request tok/s}, the same in both kinds
            1: {0: 100, 1: 150, 2: 180, 3: 200, 4: 210, 5: 212},
            8: {0: 400, 1: 430, 2: 470, 3: 520},
            16: {0: 600, 1: 560},
            32: {0: 850},
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps(CONFIG))
            for c, row in rates.items():
                for depth, rate in row.items():
                    for p in (1, 2, 3):
                        for kind in ('greedy', 'sampled'):
                            window(run, f'd{depth}p{p}', kind, c, depth, rate * (1 + 0.002 * p),
                                   {f'i{c}-r{c * (depth + 1)}': 10})
            # C4: depth 5 lands on depth 4's plans, which beat depth 3 clearly.
            for depth, rate, rows in ((0, 250, 4), (3, 330, 16), (4, 360, 20), (5, 361, 20)):
                for p in (1, 2, 3):
                    for kind in ('greedy', 'sampled'):
                        window(run, f'd{depth}p{p}', kind, 4, depth, rate, {f'i4-r{rows}': 10})
            out = report.evaluate(run)
        chosen = {c: v['depth'] for c, v in out['selection'].items()}
        # C1: depth 5 beats depth 4 by 1%, within the margin; C4: depth 5
        # serves depth 4's shapes; C16: drafting loses.
        self.assertEqual(chosen, {'1': 4, '4': 4, '8': 3, '16': 0, '32': 0})
        self.assertEqual(out['selection']['4']['depths'][5]['equivalent_to'], 4)
        self.assertEqual(out['depths'], '1:4,8:3,16:0')

    def test_prompt_groups_merge_into_one_window(self):
        full = tune.merge_full([
            {'steps': 10, 'tokens': 100, 'seconds': 0.5, 'histogram': {'i1-r6-v1': 10}, 'drafted_steps': 10},
            {'steps': 30, 'tokens': 100, 'seconds': 1.5, 'histogram': {'i1-r6-v1': 28, 'i1-r1-v0': 2},
             'drafted_steps': 28}])
        self.assertEqual((full['steps'], full['tokens_per_second'], full['histogram']),
                         (40, 100.0, {'i1-r1-v0': 2, 'i1-r6-v1': 38}))
        accept = tune.merge_acceptance([
            {'draft_tokens': 10, 'accepted_tokens': 6, 'verify_items': 2,
             'per_position': [{'position': 1, 'drafted': 2, 'accepted': 2}]},
            {'draft_tokens': 10, 'accepted_tokens': 4, 'verify_items': 2,
             'per_position': [{'position': 1, 'drafted': 2, 'accepted': 1}]}])
        self.assertEqual((accept['acceptance_rate'], accept['per_position'][0]['rate']), (0.5, 0.75))
        energy = tune.merge_energy([
            {'samples': 3, 'seconds': 1.0, 'mean_board_w': 600.0, 'sm_mhz_range': [[1700, 1790]], 'synthetic': False},
            {'samples': 3, 'seconds': 3.0, 'mean_board_w': 800.0, 'sm_mhz_range': [[1690, 1792]], 'synthetic': False}],
            tokens=1000)
        self.assertEqual((energy['joules_per_token'], energy['sm_mhz_range']), (3.0, [[1690, 1792]]))
        energy = tune.merge_energy([
            {'samples': 1, 'seconds': 1.0, 'mean_board_w': 600.0, 'sm_mhz_range': [[1700, 1700]],
             'sm_mhz_mean': [1700.0], 'power_cap_share': [1.0], 'synthetic': False},
            {'samples': 3, 'seconds': 3.0, 'mean_board_w': 800.0, 'sm_mhz_range': [[1800, 1800]],
             'sm_mhz_mean': [1800.0], 'power_cap_share': [0.0], 'synthetic': False}],
            tokens=1000)
        self.assertEqual((energy['sm_mhz_mean'], energy['power_cap_share']), ([1775.0], [0.25]))

    def test_power_windows_record_clocks_and_the_power_cap_share(self):
        gpus, tune.r.GPUS = tune.r.GPUS, 2
        try:
            power = tune.Power(Path(tempfile.mkdtemp()), dry=False)
            lines = ['0, 240.0, 1800, 250.00, Active', '1, 200.0, 1900, 250.00, Not Active',
                     '0, 245.0, 1700, 250.00, Active', '1, 210.0, 1950, 250.00, Active',
                     '0, 230.0, 1850, 250.00, Not Active', '1, 205.0, 1900, 250.00, Not Active',
                     '0, 248.0, 1750, 250.00, Active', '1, 215.0, 1880, 250.00, Not Active']
            power.process = type('Process', (), {'stdout': (line + '\n' for line in lines)})()
            power.read()
            energy = power.window(float('-inf'), float('inf'), 100)
        finally:
            tune.r.GPUS = gpus
        self.assertEqual(energy['samples'], 4)
        self.assertEqual(energy['sm_mhz_range'], [[1700.0, 1850.0], [1880.0, 1950.0]])
        self.assertEqual(energy['sm_mhz_mean'], [1775.0, 1907.5])
        self.assertEqual(energy['power_cap_share'], [0.75, 0.25])

    def test_a_job_runs_only_its_depths(self):
        model = {**MODEL, 'depths': [5, 3]}
        self.assertEqual(tune.order({'model': model}), ['d5', 'd3'])

    def test_steps_give_cycle_and_execution_times_by_shape(self):
        def step(batch, ns, prefill=0, decode=0, items=0, rows=0, us=None):
            return {'batch': batch, 'completed_unix_ns': ns, 'prefill_items': prefill, 'decode_rows': decode,
                    'verify_items': items, 'verify_rows': rows, 'execution_us': us}
        steps = [step(1, 0, prefill=8), step(2, 40_000_000, items=8, rows=48, us=39_000),
                 step(3, 80_500_000, items=8, rows=48, us=39_500), step(4, 100_000_000, decode=8, us=19_000)]
        self.assertEqual(tune.cells(steps), {'i8-r48': {'cycle_ms': [40.5], 'execution_ms': [39.5]},
                                             'i8-r8': {'cycle_ms': [19.5], 'execution_ms': [19.0]}})

    def test_costs_tabulate_every_width_over_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps({**CONFIG, 'model': {**MODEL, 'concurrencies': [8]}}))
            for depth, ms in ((3, 38.0), (5, 52.0)):
                for repeat in (0, 1):
                    path = run / f'd{depth}' / 'windows' / f'greedy-c8-r{repeat}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows = 8 * (depth + 1)
                    path.write_text(json.dumps({
                        'name': 'greedy-c8', 'kind': 'greedy', 'concurrency': 8, 'depth': depth, 'repeat': repeat,
                        'request_tokens_per_second': 300 + repeat, 'full': {'tokens_per_second': 400},
                        'energy': {'joules_per_token': 2.0},
                        'cells': {f'i8-r{rows}': {'cycle_ms': [ms + repeat, ms + 2], 'execution_ms': [ms - 1]},
                                  'i7-r28': {'cycle_ms': [30.0], 'execution_ms': []}}}))
            (run / 'summary.json').write_text(json.dumps({'processes': {'d5': {
                'args': ['--speculation-depths', '1:5', '--speculative-rows', '128', '--speculation-graph-rows', '64'],
                'gpu_memory_mib': {'ready': [1000]},
                'memory_after_warmup': {'ranks': [{'rank': 0, 'kv_mib': 9, 'taken_mib': 700, 'free_mib': 1200}]}}}}))
            out = report.costs(run)
            text = (run / 'costs.md').read_text()
        self.assertEqual(out['memory_after_warmup']['d5']['graph_rows'], '64')
        self.assertIsNone(out['memory_after_warmup']['d5']['widths'])
        self.assertIn('| d5 | 128 | 64 | - | 700 / 1200 |', text)
        self.assertEqual(out['summary'], {'i8-r32': 39.5, 'i8-r48': 53.5})
        self.assertEqual(out['cells']['i8-r48']['steps'], 4)
        self.assertEqual(out['cells']['i7-r28']['arms'], ['d3', 'd5'])
        self.assertAlmostEqual(out['windows']['c8/greedy/d5']['request_spread'], 1 / 300)

    def test_costs_arms_verify_their_exact_rows(self):
        # Each drafting arm of a model's costs job lists a width for every
        # row count its depth reaches at the job's concurrencies, so the
        # profile's own widths cannot land a plan on another width.
        for path in sorted((HERE.parents[1] / 'models').glob('*/depth-tune.json')):
            model = json.loads(path.read_text())
            for name, job in model.get('jobs', {}).items():
                if job.get('mode') != 'costs':
                    continue
                config = {'model': {**model, **job}, 'kinds': job.get('kinds', model['kinds'])}
                for arm in tune.arms(config):
                    if arm['depth'] == 0:
                        continue
                    args = arm['arguments']
                    self.assertIn('--speculation-widths', args, (path, name, arm['label']))
                    widths = {int(x) for x in args[args.index('--speculation-widths') + 1].split(',')}
                    for _, c in tune.windows_for(config, arm['depth']):
                        self.assertIn(c * (arm['depth'] + 1), widths, (path, name, arm['label'], c))

    def test_recorded_runs_parse_into_a_cost_table(self):
        def step(batch, ns, items=8, rows=32, prefill=0):
            return {'schema': 'chopinfer.step.v1', 'batch': batch, 'completed_unix_ns': ns, 'prefill_items': prefill,
                    'decode_rows': 0, 'verify_items': items, 'verify_rows': rows, 'verify_emitted_tokens': 20,
                    'execution_us': 37_000}
        steps = [step(1, 0, prefill=8), *(step(b, b * 38_000_000) for b in range(2, 6))]
        config = {'name': 'recorded', 'model': {**MODEL, 'concurrencies': [8], 'tensor_parallel_size': 2,
                                                'kv_pages_line': 'model: {pages} KV pages per rank'}}
        with tempfile.TemporaryDirectory() as directory:
            run, out = Path(directory) / 'run', Path(directory) / 'out'
            server = run / 'd3' / 'server'
            server.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps(config))
            log = ['model: 64 KV pages per rank', 'startup weight_upload: {"uploaded_bytes": 10, "lent_bytes": 0}']
            log += [f'device memory after warmup rank {i}: KV 9 MiB, 280 MiB taken since sizing (graphs), '
                    f'1200 MiB free' for i in range(2)]
            for x in steps:
                log += ['step ' + json.dumps(x), 'bundle: ' + json.dumps({
                    'schema': 'chopinfer.fixture.commit-shape.v1',
                    'instance_id': '1', 'step_id': str(x['batch']), 'micros': {'total': 100}})]
            (server / 'server.log').write_text('\n'.join(log) + '\n')
            (server / 'final.prom').write_text('vllm:num_preemptions_total 0\n' + ''.join(
                f'chop_prefix_cache_evictions_total{{class="{name}"}} {value}\n'
                for name, value in [('superseded', 2), ('retained', 3), ('maximal', 5)]))
            window = run / 'd3' / 'windows' / 'greedy-c8.json'
            window.parent.mkdir()
            window.write_text(json.dumps({'name': 'greedy-c8', 'kind': 'greedy', 'concurrency': 8, 'depth': 3,
                                          'request_tokens_per_second': 400.0, 'steps': steps}))
            recorded.main(run, out)
            result = json.loads((out / 'recorded-check.json').read_text())
            costs = json.loads((out / 'costs.json').read_text())
            self.assertEqual((result['servers'], result['windows']), (1, 1))
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['processes']['d3']['metrics']['prefix_cache_evictions'], 10)
            self.assertEqual(recorded.r.trace_rows(server / 'server.log'), steps)
            self.assertEqual(costs['cells']['i8-r32']['steps'], 3)
            (server / 'server.log').write_text('\n'.join(log[:3]) + '\n')
            with self.assertRaises(SystemExit):
                recorded.main(run, Path(directory) / 'short')

    def test_closed_loop_adapter_rebuilds_rates_and_rejects_bad_seeds(self):
        import faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'recorded'
            root.mkdir()
            power = {'enforced_power_limits_w': [250]}
            summary = {'status': 'passed', 'telemetry': {k: {'power': power} for k in ('before', 'after')},
                       'servers': {}, 'abba': {'AR-vs-DSpark': {}}}
            original = {'passed': True, 'mechanics_only': False, 'rows': {'1': {
                'arms': {}, 'window_tok_s': {'verdict': 'decisive gain'}}}}
            for label, seed, count in [('A1', 1, 100), ('B1', 1, 110), ('B2', 2, 110), ('A2', 2, 100)]:
                request = {'slot': 0, 'sequence': 0, 'request': {'seed': seed, 'prompt': 'sample', 'temperature': 1.0},
                           'error': None, 'finish_reason': 'length', 'usage': {'completion_tokens': 120,
                           'prompt_tokens_details': {'cached_tokens': 0}}, 'events': [[1, count], [16, 120-count]]}
                path = root / (label + '.json')
                path.write_text(json.dumps([request]))
                snapshot = {'process_group': 1, 'processes': {}}
                fault = faults.difference(snapshot, snapshot)
                window = {'window_monotonic_seconds': [0, 15], 'window_seconds': 15,
                          'faults': fault, 'fault_counters_before': snapshot, 'fault_counters_after': snapshot,
                          'requests_path': str(path), 'requests': {'total': 1, 'completed': 1}, 'output_tokens': 120,
                          'tokens': count, 'energy': {'j_per_token': 1}}
                summary['servers'][label] = {'weights': {'source': 'borrowed'}}
                summary['abba']['AR-vs-DSpark'][label] = {'server': label, 'decode': {'1': window}}
                original['rows']['1']['arms'][label] = {'window_tok_s': count / 15}
            (root / 'summary.json').write_text(json.dumps(summary))
            (root / 'comparison.json').write_text(json.dumps(original))
            (root / 'gpu-samples.csv').write_text('time,0,100,100,1800,250\n')
            result = recorded.closed_loop(root, Path(directory) / 'out')
            self.assertEqual(result['pairs']['AR/DSpark']['1']['verdict'], 'resolved gain')
            self.assertEqual(result['windows'], 4)
            self.assertFalse(result['contract_a_admissible'])
            for index, label in enumerate(('no change (both head arms inside the base range)', 'read by hand')):
                original['rows']['1']['window_tok_s']['verdict'] = label
                (root / 'comparison.json').write_text(json.dumps(original))
                disagreement = recorded.closed_loop(root, Path(directory) / f'disagreement-{index}')
                self.assertFalse(disagreement['harness_agreement'])
                with self.assertRaisesRegex(ValueError, 'differs'):
                    recorded.closed_loop(root, Path(directory) / f'required-{index}', require_agreement=True)
            path = root / 'B1.json'
            row = json.loads(path.read_text())
            row[0]['request']['seed'] = 9
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                recorded.closed_loop(root, Path(directory) / 'bad')

    def test_model_configs_measure_sampled_windows_only(self):
        # Temperature 1.0 is the serving workload: tuner jobs measure and
        # select on sampled windows only; greedy is an exactness check.
        for path in sorted((HERE.parents[1] / 'models').glob('*/depth-tune.json')):
            model = json.loads(path.read_text())
            self.assertEqual(model['sampling']['temperature'], 1.0, path)
            for name, job in [('', model), *model.get('jobs', {}).items()]:
                if job.get('mode') == 'exact':
                    self.assertEqual(set(job['kinds']), {'greedy', 'sampled'}, (path, name))
                    continue
                self.assertEqual(set(job.get('kinds', model['kinds'])), {'sampled'}, (path, name))

    def test_runs_merge_into_one_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            low, high = Path(directory) / 'low', Path(directory) / 'high'
            for run, rates in ((low, {1: {0: 100, 5: 200}, 4: {0: 300, 3: 400}}),
                               (high, {8: {0: 400, 3: 500}, 16: {0: 600, 1: 590}})):
                run.mkdir()
                (run / 'config.json').write_text(json.dumps(
                    {**CONFIG, 'model': {**MODEL, 'concurrencies': sorted(rates)}}))
                for c, row in rates.items():
                    for depth, rate in row.items():
                        for kind in ('greedy', 'sampled'):
                            window(run, f'd{depth}p1', kind, c, depth, rate, {f'i{c}-r{c * (depth + 1)}': 10})
            self.assertEqual(report.merged([low, high]), '1:5,4:3,16:0')

    def test_unstable_depths_are_not_chosen(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps(CONFIG))
            for depth, rate in ((0, 100), (1, 150), (2, 190)):
                for p in (1, 2):
                    for kind in ('greedy', 'sampled'):
                        shapes = {f'i1-r{depth + 1}': 10 + (p if depth == 2 else 0)}
                        window(run, f'd{depth}p{p}', kind, 1, depth, rate, shapes)
            out = report.evaluate(run)
        self.assertEqual(out['selection']['1']['depth'], 1)
        self.assertFalse(out['selection']['1']['depths'][2]['stable'])



def confidence_log(path, truth, records, seed=7, contract='fixture.drafter-confidence.v1', requests=97):
    """A server log of drafter records drawn from `truth`, per position
    `(scale, offset)`, over `requests` requests drafting in turn, with the
    samples each position should yield and the stops left unlabelled: a
    request's last record cannot prove its first unknown a rejection."""
    import math
    import random
    import struct
    rng = random.Random(seed)
    committed = [1] * requests
    blocks = []
    for index in range(records):
        request = index % requests
        emitted = rng.choice((3, 4, 5))
        scores = [struct.unpack('<f', struct.pack('<f', rng.uniform(-3, 3)))[0] for _ in range(6)]
        outcomes = ['not_emitted'] * 6
        for position in range(emitted):
            outcomes[position] = 'unknown'
        for position in range(emitted):
            scale, offset = truth[position]
            if rng.random() >= 1 / (1 + math.exp(-(scale * scores[position] + offset))):
                break
            outcomes[position] = 'accepted'
        record = {'schema': 'chopinfer.fixture.drafter-confidence.v1', 'confidence_contract': contract,
                  'instance_id': '1', 'step_id': str(index), 'request_id': str(request),
                  'main_rows': committed[request], 'emitted_positions': emitted,
                  'completion': 'committed', 'draft_ids': [1] * 6,
                  'confidence_f32_bits': [struct.unpack('<I', struct.pack('<f', x))[0] for x in scores],
                  'outcomes': outcomes}
        committed[request] += outcomes.count('accepted') + 1
        blocks.append(record)
    last = {record['request_id']: record['step_id'] for record in blocks}
    expected, unproven = {}, 0
    for record in blocks:
        for position, outcome in enumerate(record['outcomes']):
            if outcome == 'unknown' and last[record['request_id']] == record['step_id']:
                unproven += 1
                break
            if outcome in ('accepted', 'unknown'):
                expected[position] = expected.get(position, 0) + 1
            if outcome != 'accepted':
                break
    lines = ['startup readiness: 1.0s', *('bundle: ' + json.dumps(record) for record in blocks)]
    cancelled = dict(blocks[-1], step_id=str(records), completion='cancelled',
                     outcomes=['cancelled'] * 3 + ['not_emitted'] * 3)
    lines.append('bundle: ' + json.dumps(cancelled))
    lines.append('diagnostic writer: dropped 3 lines (total 3)')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return expected, unproven


class Calibration(unittest.TestCase):
    TRUTH = [(1.5, 0.5), (1.2, 0.2), (1.0, 0.0), (0.8, -0.2), (0.6, -0.4)]

    def test_constant_scores_fit_the_observed_rate(self):
        # Ridge keeps a constant feature's Hessian invertible. It must not
        # leave an imbalanced population at the initial probability of 0.5.
        for accepted in (20, 80):
            with self.subTest(accepted=accepted):
                points = [(1.0, int(index < accepted), index % 2) for index in range(100)]
                scale, offset = calibrate.fit(points)
                self.assertAlmostEqual(calibrate.sigmoid(scale + offset), accepted / 100,
                                       delta=0.002)
                self.assertLess(calibrate.log_loss(points, scale, offset),
                                calibrate.log_loss(points, 0.0, 0.0))

    def test_commit_shape_records_do_not_enter_confidence_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.log'
            confidence_log(path, self.TRUTH, records=250)
            original, dropped = calibrate.records(path)
            lines = path.read_text().splitlines()
            commit = json.dumps({'schema': 'chopinfer.fixture.commit-shape.v1',
                                 'instance_id': '1', 'step_id': '1', 'micros': {'total': 100}})
            path.write_text('\n'.join(line + '\nbundle: ' + commit for line in lines) + '\n')
            after, after_dropped = calibrate.records(path)
            self.assertEqual((after, after_dropped), (original, dropped))

    def test_startup_warmup_records_are_excluded_before_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            row = {'schema': 'fixture.drafter-confidence.v1', 'request_id': 'warmup'}
            served = {**row, 'request_id': 'served'}
            log.write_text(json.dumps(row) + '\nstartup readiness: 12.3s\n' +
                           json.dumps(served) + '\n')
            rows, dropped = calibrate.records(log)
            self.assertEqual([r['request_id'] for r in rows], ['served'])
            self.assertEqual(dropped, 0)
            # Startup refusal still applies even when it precedes readiness.
            log.write_text('startup speculation_confidence: on, floor 0, 5 positions\n' + log.read_text())
            with self.assertRaisesRegex(SystemExit, 'served with confidence allocation'):
                calibrate.records(log)
            # Hand-written and extracted record-only logs remain supported.
            log.write_text(json.dumps(served) + '\n')
            self.assertEqual(len(calibrate.records(log)[0]), 1)

    def test_logs_without_records_name_the_level_they_served_at(self):
        with tempfile.TemporaryDirectory() as directory:
            info = Path(directory) / 'info.log'
            info.write_text('startup logging: level=info\nstartup readiness: 1.0s\n')
            silent = Path(directory) / 'silent.log'
            silent.write_text('startup readiness: 1.0s\n')
            with self.assertRaises(SystemExit) as refused:
                calibrate.calibrate([info, silent], None, 1)
            message = str(refused.exception)
            self.assertIn(f'{info} served at level=info', message)
            self.assertIn(f'{silent} has no "startup logging: level=" line', message)
            self.assertIn('--log-level debug', message)

    def test_allocating_server_logs_are_refused_before_labelling(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            # Even a log without records must be refused: an unknown draft
            # from an allocated step cannot establish a rejection.
            log.write_text('startup speculation_confidence: on, floor 0, 5 positions\n')
            with self.assertRaisesRegex(SystemExit, 'served with confidence allocation'):
                calibrate.records(log)
            log.write_text('startup speculation_confidence: off, the package returns host values\n')
            self.assertEqual(calibrate.records(log), ([], 0))

    def test_records_fit_each_positions_acceptance_given_earlier_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            expected, unproven = confidence_log(log, self.TRUTH, 6000)
            out = Path(directory) / 'out' / 'calibration.json'
            self.assertEqual(calibrate.main(['--sampled', str(log), '--positions', '3',
                                             '--out', str(out)]), 0)
            calibration = json.loads(out.read_text())
            table = json.loads(out.with_suffix('.report.json').read_text())
        self.assertEqual(calibration['contract'], 'fixture.drafter-confidence.v1')
        self.assertEqual(calibration['floor'], 1)
        self.assertEqual(calibration['greedy'], calibration['sampled'])
        self.assertEqual(len(calibration['sampled']), 3)
        report = table['sampled']
        self.assertEqual((report['records'], report['dropped_log_lines'], report['complete']),
                         (6001, 3, False))
        self.assertEqual(report['unproven_stops'], unproven)
        self.assertGreater(unproven, 0)
        self.assertEqual(len(report['logs'][0]['sha256']), 64)
        for row, (scale, offset) in zip(report['positions'], self.TRUTH):
            position = row['position'] - 1
            # A position's samples are the records that reached it: the
            # cancelled block, positions after the first unknown, and an
            # unknown no later record of its request proves rejected are out.
            self.assertEqual(row['samples'], expected[position])
            self.assertAlmostEqual(row['scale'], scale, delta=0.2)
            self.assertAlmostEqual(row['offset'], offset, delta=0.2)
            self.assertLess(row['log_loss_held_out'], row['log_loss_constant'])
            self.assertGreater(row['auc'], 0.7)
            self.assertLess(row['ece'], 0.05)
            self.assertAlmostEqual(row['mean_predicted'], row['accepted_given_earlier'], delta=0.01)
            self.assertLessEqual(row['accepted_over_drafted'], 1.0)
        for pair in calibration['sampled']:
            for value in pair:
                self.assertEqual(value, calibrate.f32(value))

    def test_auc_ranks_accepted_samples_above_rejected_ones(self):
        self.assertEqual(calibrate.auc([(0.0, 0, 0), (1.0, 1, 0)]), 1.0)
        self.assertEqual(calibrate.auc([(1.0, 0, 0), (0.0, 1, 0)]), 0.0)
        self.assertEqual(calibrate.auc([(0.5, 0, 0), (0.5, 1, 0)]), 0.5)
        self.assertIsNone(calibrate.auc([(0.5, 1, 0)]))

    def test_thin_positions_mixed_contracts_and_other_kinds_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'run' / 'd5' / 'server' / 'server.log'
            confidence_log(log, self.TRUTH, 400)
            with self.assertRaises(SystemExit) as thin:
                calibrate.main(['--sampled', str(log), '--positions', '5', '--out', str(root / 'a.json')])
            self.assertRegex(str(thin.exception), r'position [2-5] has \d+ samples, fewer than 200')
            other = root / 'other.log'
            confidence_log(other, self.TRUTH, 10, contract='other.drafter-confidence.v1')
            with self.assertRaises(SystemExit) as mixed:
                calibrate.main(['--sampled', str(log), str(other), '--min-samples', '1',
                                '--out', str(root / 'b.json')])
            self.assertIn('fit one at a time', str(mixed.exception))
            repeated = root / 'repeated.log'
            text = log.read_text().splitlines()
            repeated.write_text('\n'.join(text + [text[1]]) + '\n')
            with self.assertRaises(SystemExit) as repeats:
                calibrate.main(['--sampled', str(repeated), '--min-samples', '1', '--out', str(root / 'e.json')])
            self.assertIn('repeats', str(repeats.exception))
            (root / 'run' / 'config.json').write_text(json.dumps({'kinds': {'greedy': 1, 'sampled': 1}}))
            with self.assertRaises(SystemExit) as kinds:
                calibrate.main(['--sampled', str(root / 'run'), '--out', str(root / 'c.json')])
            self.assertIn("not only sampled", str(kinds.exception))
            (root / 'run' / 'config.json').write_text(json.dumps({'kinds': {'sampled': 1.0}}))
            self.assertEqual(calibrate.main(['--sampled', str(root / 'run'), '--positions', '2',
                                             '--min-samples', '50', '--out', str(root / 'd.json')]), 0)


def prefill_fixture(root, gain=.1, control_regression=False):
    ab_fixture(root, concurrencies=[1], change='table_row',
               prefill_rows={'1': [192, 256]}, target_cells=['prefill:c1:r192'])
    for path in root.glob('*/windows/*.json'):
        row = json.loads(path.read_text())
        row['steps'] = []
        for width in (192, 256):
            micros = 100000
            if row['arm'] == 'b':
                micros *= 1 - gain if width == 192 else 1.1 if control_regression else 1
            row['steps'].extend(dict(prefill_items=1, prefill_tokens=width,
                                     decode_rows=0, verify_items=0, execution_us=micros)
                                for _ in range(20))
        path.write_text(json.dumps(row))


class PrefillReport(unittest.TestCase):
    def test_synthetic_prefill_cells_never_qualify_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefill_fixture(root)
            config, rows = report.load(root)
            rows.sort(key=lambda row: row['visit'])
            for row in rows:
                row['synthetic'] = True
            cells = prefill_report.decision_cells(config['model'], rows, 1, 'table_row', report.paired)
            self.assertTrue(all(cell['mechanics_only'] for cell in cells.values()))
            self.assertFalse(any(cell['accepted'] for cell in cells.values()))

    def test_affected_prefill_bucket_can_be_the_declared_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefill_fixture(root)
            result = report.ab(root)
            self.assertTrue(result['weighted']['a/b']['timing_accepted'])
            cell = result['pairs']['a/b']['1']['prefill_cells']['prefill:c1:r192']
            self.assertEqual(cell['steps'], [20] * 4)
            self.assertEqual(cell['execution_ms_median'], [100, 90, 90, 100])

    def test_control_regression_and_subthreshold_gain_prevent_adoption(self):
        for gain, regression in ((.1, True), (.01, False)):
            with self.subTest(gain=gain, regression=regression), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prefill_fixture(root, gain, regression)
                self.assertFalse(report.ab(root)['weighted']['a/b']['timing_accepted'])

    def test_missing_or_mixed_prefill_cannot_supply_required_coverage(self):
        for mutation in ('missing', 'mixed', 'invalid'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prefill_fixture(root)
                path = next(root.glob('*/windows/*.json'))
                row = json.loads(path.read_text())
                if mutation == 'missing':
                    row['steps'] = [s for s in row['steps'] if s['prefill_tokens'] != 256]
                elif mutation == 'mixed':
                    for step in row['steps']:
                        step['verify_items'] = 1
                else:
                    row['steps'][0]['execution_us'] = 0
                path.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    report.ab(root)



if __name__ == '__main__':
    unittest.main()
