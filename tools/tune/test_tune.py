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
import stage  # noqa: E402
import tune  # noqa: E402

# A synthetic model: depths 0-5 over 32 verify rows, two repeats each.
MODEL = {'max_depth': 5, 'verify_rows': 32, 'concurrencies': [1, 2, 4, 8, 16, 32], 'repeats': 2}
CONFIG = {'name': 'synthetic', 'model': MODEL, 'kinds': {'greedy': 0.5, 'sampled': 0.5},
          'selection': {'margin': 0.015, 'spread': 0.03}}


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

            base = json.loads((HERE.parents[1] / 'models/deepseek41/depth-tune.json').read_text())
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
            with patch.object(tune, 'SEAL', {'config': config}, create=True), \
                    patch.object(tune, 'closed_loop', return_value=result) as controller:
                tune.page_cache_warmup(None, {'label': 'w0'}, state, None)
            self.assertEqual([call.args[1] for call in controller.call_args_list], ['w0-c8-r0', 'w0-c16-r0', 'w0-c16-r1'])

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
                                'tensor_parallel_size': 1, 'kv_pages': 64, 'kv_pages_line': 'fixture: {pages} KV pages',
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
                                            ('slowcold', {'TUNE_STUB_PREFILL_DELAY_SECONDS': '0.25'}, 'passed'),
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
            server = SimpleNamespace(directory=root, env={}, dry=True)
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
            (server / 'final.prom').write_text('vllm:num_preemptions_total 0\nchop_prefix_cache_evictions_total 0\n')
            window = run / 'd3' / 'windows' / 'greedy-c8.json'
            window.parent.mkdir()
            window.write_text(json.dumps({'name': 'greedy-c8', 'kind': 'greedy', 'concurrency': 8, 'depth': 3,
                                          'request_tokens_per_second': 400.0, 'steps': steps}))
            recorded.main(run, out)
            result = json.loads((out / 'recorded-check.json').read_text())
            costs = json.loads((out / 'costs.json').read_text())
            self.assertEqual((result['servers'], result['windows']), (1, 1))
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


if __name__ == '__main__':
    unittest.main()
