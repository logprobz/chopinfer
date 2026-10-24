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


class Tuner(unittest.TestCase):
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
            log += ['step ' + json.dumps(x) for x in steps]
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
            self.assertEqual(costs['cells']['i8-r32']['steps'], 3)
            (server / 'server.log').write_text('\n'.join(log[:3]) + '\n')
            with self.assertRaises(SystemExit):
                recorded.main(run, Path(directory) / 'short')

    def test_model_configs_measure_sampled_windows_only(self):
        # Temperature 1.0 is the serving workload: tuner jobs measure and
        # select on sampled windows only; greedy is an exactness check.
        for path in sorted((HERE.parents[1] / 'models').glob('*/depth-tune.json')):
            model = json.loads(path.read_text())
            self.assertEqual(model['sampling']['temperature'], 1.0, path)
            for name, job in [('', model), *model.get('jobs', {}).items()]:
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



def confidence_log(path, truth, records, seed=7, contract='fixture.drafter-confidence.v1'):
    """A server log of drafter records drawn from `truth`, per position
    `(scale, offset)`, and the samples each position should yield."""
    import math
    import random
    import struct
    rng = random.Random(seed)
    expected = {}
    lines = ['startup readiness: 1.0s']
    for index in range(records):
        emitted = rng.choice((3, 4, 5))
        scores = [struct.unpack('<f', struct.pack('<f', rng.uniform(-3, 3)))[0] for _ in range(6)]
        outcomes = ['not_emitted'] * 6
        for position in range(emitted):
            outcomes[position] = 'unknown'
        for position in range(emitted):
            expected[position] = expected.get(position, 0) + 1
            scale, offset = truth[position]
            if rng.random() >= 1 / (1 + math.exp(-(scale * scores[position] + offset))):
                break
            outcomes[position] = 'accepted'
        record = {'schema': 'chopinfer.fixture.drafter-confidence.v1', 'confidence_contract': contract,
                  'instance_id': '1', 'step_id': str(index), 'request_id': str(index % 97),
                  'emitted_positions': emitted, 'completion': 'committed', 'draft_ids': [1] * 6,
                  'confidence_f32_bits': [struct.unpack('<I', struct.pack('<f', x))[0] for x in scores],
                  'outcomes': outcomes}
        lines.append('bundle: ' + json.dumps(record))
    cancelled = dict(record, completion='cancelled', outcomes=['cancelled'] * 3 + ['not_emitted'] * 3)
    lines.append('bundle: ' + json.dumps(cancelled))
    lines.append('diagnostic writer: dropped 3 lines (total 3)')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return expected


class Calibration(unittest.TestCase):
    TRUTH = [(1.5, 0.5), (1.2, 0.2), (1.0, 0.0), (0.8, -0.2), (0.6, -0.4)]

    def test_records_fit_each_positions_acceptance_given_earlier_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            expected = confidence_log(log, self.TRUTH, 6000)
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
        self.assertEqual((report['records'], report['dropped_log_lines']), (6001, 3))
        for row, (scale, offset) in zip(report['positions'], self.TRUTH):
            position = row['position'] - 1
            # A position's samples are the records that reached it: the
            # cancelled block and positions after the first unknown are out.
            self.assertEqual(row['samples'], expected[position])
            self.assertAlmostEqual(row['scale'], scale, delta=0.2)
            self.assertAlmostEqual(row['offset'], offset, delta=0.2)
            self.assertLess(row['log_loss_held_out'], row['log_loss_constant'])
            self.assertAlmostEqual(row['mean_predicted'], row['accepted_given_earlier'], delta=0.01)
            self.assertLessEqual(row['accepted_over_drafted'], 1.0)
        for pair in calibration['sampled']:
            for value in pair:
                self.assertEqual(value, calibrate.f32(value))

    def test_thin_positions_mixed_contracts_and_other_kinds_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'run' / 'd5' / 'server' / 'server.log'
            confidence_log(log, self.TRUTH, 400)
            with self.assertRaises(SystemExit) as thin:
                calibrate.main(['--sampled', str(log), '--positions', '5', '--out', str(root / 'a.json')])
            self.assertIn('position 3 has', str(thin.exception))
            other = root / 'other.log'
            confidence_log(other, self.TRUTH, 10, contract='other.drafter-confidence.v1')
            with self.assertRaises(SystemExit) as mixed:
                calibrate.main(['--sampled', str(log), str(other), '--min-samples', '1',
                                '--out', str(root / 'b.json')])
            self.assertIn('fit one at a time', str(mixed.exception))
            (root / 'run' / 'config.json').write_text(json.dumps({'kinds': {'greedy': 1, 'sampled': 1}}))
            with self.assertRaises(SystemExit) as kinds:
                calibrate.main(['--sampled', str(root / 'run'), '--out', str(root / 'c.json')])
            self.assertIn("not only sampled", str(kinds.exception))
            (root / 'run' / 'config.json').write_text(json.dumps({'kinds': {'sampled': 1.0}}))
            self.assertEqual(calibrate.main(['--sampled', str(root / 'run'), '--positions', '2',
                                             '--min-samples', '50', '--out', str(root / 'd.json')]), 0)


if __name__ == '__main__':
    unittest.main()
