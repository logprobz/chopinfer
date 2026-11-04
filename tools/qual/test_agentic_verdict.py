"""CPU checks of the stateful replay verdict over A1, B1, B2, A2 receipts."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from agentic_metrics import summarize
import agentic_verdict as verdict

METRICS_BEFORE = ('vllm:num_preemptions_total{model_name="m"} 1\n'
                  'chop_prefix_cache_evictions_total{class="maximal"} 2\n'
                  'chop_prefix_ram_missed_evictions_total{reason="headroom"} 0\n')


def level(concurrency=8, ttft=100.0, cached=50, warm=.5, gap=10.0, stalls=0, generated=10, samples=40,
          pid=1, sessions_per_minute=True):
    """A replay level as agentic_trace builds it, summarized by agentic_metrics."""
    turns = []
    for index in range(samples):
        gaps = [dict(ms=150.0 if index < stalls else gap, holdback_possible=False)] + \
               [dict(ms=gap, holdback_possible=False)] * 3
        turns.append(dict(started=1 + index * .1, ended=1.05 + index * .1, output_deltas=generated,
                          output_tokens=generated, ttft_ms=ttft + index % 3, inter_delta_ms=[gap],
                          elapsed_seconds=.05, cached_tokens=cached, prompt_tokens=100, step_gaps=gaps,
                          warm=index < warm * samples, warm_threshold_tokens=60, failure=None, timed_out=False))
    sessions = [dict(phase='measured', turns=turns, ended=9, valid=sessions_per_minute, elapsed_seconds=8,
                     tool_errors=0, failure=None, runner_error=None)]
    summary = summarize(sessions, 0, 10, concurrency)
    after = METRICS_BEFORE.replace('} 1\n', '} 4\n').replace('} 2\n', '} 7\n').replace('} 0\n', '} 3\n')
    startup = dict(pid=pid, start_ticks='100')
    summary.update(replica=0, sessions=sessions,
                   server_metrics=dict(before=dict(text=METRICS_BEFORE), after=dict(text=after)),
                   prefill_budget=dict(startup_before=startup, startup_after=startup))
    return summary


def receipt(arm, seed, levels):
    return dict(schema='chopinfer-qualification/v1', check='agentic-mixed', receipt_kind='sweep', aborted=None,
                passed=True, mechanics_only=False, arm=arm, workload_schema='chopinfer.agentic-mixed.v3',
                source_revision='chopinfer.agentic-trace.v1', corpus_digest='trace', workload_digest='trace',
                profile=dict(seed=seed, levels=[row['concurrency'] for row in levels], replicas=1),
                server=dict(model='m'), run_id=f'{arm}-{seed}', levels=levels,
                host_load=dict(idle=True, reasons=[]))


def abba(cells=(8,), **changes):
    """Visits whose levels take level() defaults, changed per visit by
    CHANGES[visit] (a dict of level() arguments)."""
    pids = dict(A1=1, B1=2, B2=3, A2=4)
    arms = dict(A1='fixed-tier-off', B1='fixed-tier-on', B2='fixed-tier-on', A2='fixed-tier-off')
    seeds = dict(A1='first', B1='first', B2='second', A2='second')
    return {visit: receipt(arms[visit], seeds[visit],
                           [level(concurrency, pid=pids[visit], **changes.get(visit, {})) for concurrency in cells])
            for visit in verdict.VISITS}


POWER = [dict(at=at, watts={'0': 300.0}, limits={'0': 300.0}, sm_clocks={'0': 2000}) for at in (0, 10)]


def judged(visits, gains=('cache_hit_rate',), cells=None, power=True, faults=True):
    return verdict.verdict(visits, list(gains), cells,
                           {visit: copy.deepcopy(POWER) for visit in verdict.VISITS} if power else None,
                           dict.fromkeys(verdict.VISITS, 10) if faults else None)


def both(**values):
    """The same level() change in B1 and B2."""
    return dict(B1=values, B2=values)


class VerdictTests(unittest.TestCase):
    def test_a_declared_gain_without_regression_replaces_the_old_behavior(self):
        result = judged(abba(**both(cached=80, warm=.9)), gains=('cache_hit_rate', 'sessions_kept_warm'))
        metrics = result['cells'][0]['metrics']
        hit = metrics['cache_hit_rate']
        self.assertEqual(hit['unit'], 'percentage points')
        self.assertAlmostEqual(hit['d1'], 30)
        self.assertAlmostEqual(hit['d2'], 30)
        self.assertEqual((hit['mean'], hit['spread'], hit['verdict']), (hit['d1'], 0, 'resolved gain'))
        self.assertAlmostEqual(metrics['sessions_kept_warm']['mean'], 40)
        self.assertEqual(metrics['ttft_p95_ms']['verdict'], 'unresolved')
        self.assertEqual(result['resolved_gains'], ['cache_hit_rate@C8', 'sessions_kept_warm@C8'])
        self.assertEqual((result['decision'], result['problems'], result['accepted']), ('replace', [], True))
        self.assertTrue(result['fresh_processes'])
        evidence = result['cells'][0]['evidence']['B1']
        self.assertEqual((evidence['cached_prompt_tokens'], evidence['warm_returns'], evidence['returns']), (3200, 36, 40))
        self.assertEqual(evidence['server']['preemptions'], dict(delta=3, labels={}))
        self.assertEqual(evidence['server']['ram_missed_evictions'], dict(delta=3, labels={'reason=headroom': 3}))
        self.assertIsNone(evidence['server']['ram_hits'])

    def test_lower_is_better_metrics_are_inverted(self):
        result = judged(abba(**both(ttft=80.0)), gains=('return_ttft_p95_ms', 'ttft_p50_ms'))
        row = result['cells'][0]['metrics']['return_ttft_p95_ms']
        # A1 / B1 - 1 over p95 TTFTs of 102 and 82.
        self.assertAlmostEqual(row['d1'], 102 / 82 - 1)
        self.assertEqual(row['verdict'], 'resolved gain')
        self.assertEqual(result['decision'], 'replace')

    def test_a_regression_needs_both_pairs_below_the_spread_and_the_floor(self):
        # 10% slower first tokens in both pairs.
        slower = judged(abba(**both(ttft=110.0)))['cells'][0]['metrics']['ttft_p95_ms']
        self.assertEqual(slower['verdict'], 'resolved regression')
        # 0.3% slower in both pairs stays within the 0.5% floor of ratio metrics.
        close = judged(abba(**both(ttft=100.3)))['cells'][0]['metrics']['ttft_p50_ms']
        self.assertLess(close['mean'], 0)
        self.assertEqual(close['verdict'], 'unresolved')
        # Differences have no floor: one stall in each B visit is resolved.
        stalls = judged(abba(**both(stalls=1)))['cells'][0]['metrics']['stalls_per_1000_gaps']
        self.assertEqual((stalls['kind'], stalls['d1'], stalls['d2']), ('difference', -6.25, -6.25))
        self.assertEqual(stalls['verdict'], 'resolved regression')
        # Opposite signs resolve nothing.
        mixed = judged(abba(B1=dict(ttft=110.0), B2=dict(ttft=90.0)))['cells'][0]['metrics']['ttft_p95_ms']
        self.assertEqual(mixed['verdict'], 'unresolved')

    def test_gains_and_regressions_in_different_metrics_are_a_tradeoff(self):
        result = judged(abba(**both(cached=80, ttft=120.0)))
        self.assertEqual(result['resolved_gains'], ['cache_hit_rate@C8'])
        self.assertIn('ttft_p95_ms@C8', result['resolved_regressions'])
        self.assertEqual(result['decision'], 'tradeoff')
        self.assertFalse(result['accepted'])

    def test_gains_count_only_in_declared_metrics_and_cells(self):
        visits = abba(cells=(8, 16), **both(generated=12, cached=80))
        result = judged(visits, gains=('cache_hit_rate',), cells=[16])
        cells = {cell['concurrency']: cell['metrics'] for cell in result['cells']}
        self.assertEqual(cells[8]['cache_hit_rate']['verdict'], 'undeclared gain')
        self.assertEqual(cells[16]['cache_hit_rate']['verdict'], 'resolved gain')
        self.assertEqual(cells[16]['output_tokens_per_second']['verdict'], 'undeclared gain')
        self.assertEqual(result['resolved_gains'], ['cache_hit_rate@C16'])
        with self.assertRaisesRegex(ValueError, 'not among the measured'):
            judged(visits, cells=[32])

    def test_zero_baseline_rates_use_differences(self):
        result = judged(abba(A1=dict(cached=0, warm=0, stalls=4), A2=dict(cached=0, warm=0, stalls=4),
                             **both(cached=40)), gains=('cache_hit_rate', 'sessions_kept_warm'))
        metrics = result['cells'][0]['metrics']
        self.assertEqual(metrics['cache_hit_rate']['values']['A1'], 0)
        self.assertAlmostEqual(metrics['cache_hit_rate']['d1'], 40)
        self.assertEqual(metrics['sessions_kept_warm']['verdict'], 'resolved gain')
        # Fewer stalls in B is positive: A - B.
        self.assertEqual(metrics['stalls_per_1000_gaps']['d1'], 25)
        self.assertEqual(metrics['stall_ms_per_1000_gaps']['verdict'], 'undeclared gain')

    def test_missing_samples_are_unavailable_never_a_tie(self):
        visits = abba(**both(cached=80))
        for visit in ('A1', 'A2'):
            visits[visit]['levels'] = [level(samples=10, pid=dict(A1=1, A2=4)[visit])]
            del visits[visit]['levels'][0]['sessions_kept_warm']
        result = judged(visits, gains=('cache_hit_rate', 'sessions_kept_warm', 'ttft_p95_ms'))
        metrics = result['cells'][0]['metrics']
        self.assertEqual(metrics['ttft_p95_ms']['verdict'], 'unavailable')
        self.assertEqual(metrics['ttft_p95_ms']['unavailable'], {'A1': '10 samples; p95 needs 20',
                                                                 'A2': '10 samples; p95 needs 20'})
        self.assertEqual(metrics['ttft_p50_ms']['verdict'], 'unresolved')
        self.assertEqual(metrics['sessions_kept_warm']['unavailable'], {'A1': 'not recorded', 'A2': 'not recorded'})
        self.assertEqual(result['unavailable_declared'], ['sessions_kept_warm@C8', 'ttft_p95_ms@C8'])
        # A zero baseline rate gives no ratio.
        idle = judged(abba(A1=dict(sessions_per_minute=False), A2=dict(sessions_per_minute=False)))
        row = idle['cells'][0]['metrics']['valid_sessions_per_minute']
        self.assertEqual((row['verdict'], row['unavailable']), ('unavailable', 'zero baseline rate'))

    def test_energy_compares_joules_per_token_and_vetoes_a_consistent_increase(self):
        visits = abba(cells=(8, 16), **both(cached=80))
        power = {visit: copy.deepcopy(POWER) for visit in verdict.VISITS}
        faults = dict.fromkeys(verdict.VISITS, 0)
        for visit in ('B1', 'B2'):
            for sample in power[visit]:
                sample['watts'] = {'0': 330.0}
        result = verdict.verdict(visits, ['cache_hit_rate'], None, power, faults)
        energy = result['cells'][0]['energy']
        self.assertEqual(energy['values']['A1'], 300 * 10 / 400)
        self.assertAlmostEqual(energy['mean'], .1)
        self.assertEqual(energy['verdict'], 'increase')
        self.assertTrue(result['energy_veto'])
        self.assertFalse(result['accepted'])
        # 2% is informational.
        for visit in ('B1', 'B2'):
            for sample in power[visit]:
                sample['watts'] = {'0': 306.0}
        self.assertFalse(verdict.verdict(visits, ['cache_hit_rate'], None, power, faults)['energy_veto'])
        # Different limits void the job and leave energy unavailable.
        for sample in power['B2']:
            sample['limits'] = {'0': 250.0}
        result = verdict.verdict(visits, ['cache_hit_rate'], None, power, faults)
        self.assertEqual(result['cells'][0]['energy']['unavailable'], 'power limits differ between visits')
        self.assertIn('power limits changed between visits: the job is void', result['problems'])
        self.assertFalse(result['accepted'])

    def test_missing_evidence_and_busy_hosts_block_acceptance(self):
        visits = abba(**both(cached=80))
        result = judged(visits, power=False, faults=False)
        self.assertEqual(result['decision'], 'replace')
        self.assertFalse(result['accepted'])
        self.assertIn('A1: major faults not recorded', result['problems'])
        self.assertIn('power limits not recorded for every visit', result['problems'])
        self.assertEqual(result['cells'][0]['energy']['verdict'], 'unavailable')
        faults = dict.fromkeys(verdict.VISITS, 10) | {'B2': 1001}
        result = verdict.verdict(visits, ['cache_hit_rate'], None,
                                 {visit: POWER for visit in verdict.VISITS}, faults)
        self.assertEqual(result['problems'], ['B2: 1001 major faults exceed the budget of 1000; rerun the visit'])
        busy = copy.deepcopy(visits)
        busy['B1']['host_load'] = dict(idle=False, reasons=['load'])
        self.assertTrue(any(problem.startswith('timing is informational') for problem in judged(busy)['problems']))
        unidentified = copy.deepcopy(visits)
        del unidentified['A2']['levels'][0]['prefill_budget']['startup_after']
        result = judged(unidentified)
        self.assertIsNone(result['fresh_processes'])
        self.assertIn('server processes not identified', result['problems'])

    def test_receipts_that_are_not_one_abba_are_refused(self):
        def changed(edit):
            visits = abba()
            edit(visits)
            return visits
        cases = {
            'aborted': lambda v: v['B1'].update(passed=False),
            'not an agentic replay': lambda v: v['A2'].update(check='prefix-exact'),
            'workload_digest': lambda v: v['B2'].update(workload_digest='other'),
            'other than the seed': lambda v: v['B2']['profile'].update(levels=[16]),
            'share one seed': lambda v: v['B1']['profile'].update(seed='second'),
            'A1 and A2 must be one arm': lambda v: v['A2'].update(arm='fixed-tier-on'),
            'one server process': lambda v: v['B2']['levels'][0]['prefill_budget'].update(
                startup_before=dict(pid=1, start_ticks='100'), startup_after=dict(pid=1, start_ticks='100')),
            'exactly one replica': lambda v: v['A1']['levels'][0].update(replica=1),
            'different models': lambda v: v['B1']['server'].update(model='other'),
        }
        for message, edit in cases.items():
            with self.subTest(case=message), self.assertRaisesRegex(ValueError, message):
                judged(changed(edit))
        # The two pairs need different seeds.
        same = changed(lambda v: [v[visit]['profile'].update(seed='first') for visit in ('B2', 'A2')])
        with self.assertRaisesRegex(ValueError, 'share one seed'):
            judged(same)
        with self.assertRaisesRegex(ValueError, 'declare at least one gain'):
            judged(abba(), gains=())
        with self.assertRaisesRegex(ValueError, 'unknown'):
            judged(abba(), gains=('hit_rate',))

    def test_command_line_prints_the_verdict_and_refuses_bad_input(self):
        visits = abba(**both(cached=80))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for visit, value in visits.items():
                paths.append(root / f'{visit}.json')
                paths[-1].write_text(json.dumps(value))
                (root / f'{visit}-power.json').write_text(json.dumps(POWER))
            options = ['--gain', 'cache_hit_rate', '--cell', '8']
            for visit in verdict.VISITS:
                options += ['--power', f'{visit}={root / f"{visit}-power.json"}', '--major-faults', f'{visit}=3']
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(verdict.main([*map(str, paths), *options]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result['schema'], verdict.SCHEMA)
            self.assertTrue(result['accepted'], result['problems'])
            self.assertEqual(result['visits']['B2']['major_faults'], 3)
            errors = io.StringIO()
            with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
                verdict.main([str(paths[0]), str(paths[0]), str(paths[2]), str(paths[3]), '--gain', 'cache_hit_rate'])
            self.assertEqual(caught.exception.code, 1)
            self.assertIn('Verdict refused', errors.getvalue())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                verdict.main([*map(str, paths), '--gain', 'cache_hit_rate', '--power', 'C1=x'])
            self.assertEqual(caught.exception.code, 2)

    def test_dry_run_trace_receipts_form_an_abba(self):
        from unittest.mock import patch
        import agentic_mixed
        import agentic_trace
        workload = agentic_trace.generate(2, 2, 'abba', [32], 0, returns='cyclic')
        for session in workload['sessions']:
            for turn in session['turns']:
                turn.update(think_seconds=0, append_tokens=4, output_tokens=2)
        receipts, processes = {}, iter(range(10, 14))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps(workload))

            def receipt(check, identity, result, dry, **_):
                receipts[result['arm']] = dict(result, schema='chopinfer-qualification/v1', check=check,
                                               server=identity, mechanics_only=dry, host_load=None)
                return Path(directory) / f'{len(receipts)}.json'
            for visit in verdict.VISITS:
                process = dict(pid=next(processes), start_ticks='1', policy='fixed', unavailable=None)
                with patch('common.startup'), patch('common.receipt', side_effect=receipt), \
                        patch.object(agentic_trace, 'observe_startup', return_value=process), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(agentic_mixed.main([
                        '--workload', 'trace', '--trace', str(path), '--dry-run', '--arm', f'fixed-{visit}',
                        '--seed', 'first' if visit in ('A1', 'B1') else 'second', '--levels', '1',
                        '--replicas', '1', '--min-samples', '1', '--seconds', '5']), 0)
        visits = {visit: receipts[f'fixed-{visit}'] for visit in verdict.VISITS}
        for visit in ('A2', 'B2'):
            visits[visit]['arm'] = visits[visit[0] + '1']['arm']
        result = judged(visits, gains=('sessions_kept_warm',))
        self.assertTrue(result['fresh_processes'])
        self.assertTrue(result['mechanics_only'])
        self.assertIn('mechanics-only visits', result['problems'])
        cell = result['cells'][0]
        self.assertEqual(cell['evidence']['A1']['returns'], 4)
        self.assertNotEqual(cell['metrics']['sessions_kept_warm']['verdict'], 'unavailable')
        self.assertEqual(cell['metrics']['return_ttft_p95_ms']['verdict'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
