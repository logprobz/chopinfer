import json
import tempfile
import unittest
from pathlib import Path

import hc_cols_job as job

SERVED = {16: [['chop_deepseek41_hc_round_chain_w4_v1', [160, 16, 1]],
               ['chop_deepseek41_hc_finish_norm_w32_v1', [1, 16, 1]]],
          32: [['chop_deepseek41_hc_round_rows_r2_v1', [160, 16, 1]],
               ['chop_deepseek41_hc_finish_norm_w32_v1', [1, 32, 1]]]}
VARIANTS = ['twin32', 'cols_r2_w8']


def plans(served=SERVED):
    lines = []
    for rows, launches in served.items():
        arms = [dict(arm='served', launches=[dict(symbol=s, grid=g, block=[256, 1, 1], dynamic_shared_bytes=0)
                                              for s, g in launches])]
        lines.append('HC_COLS_PLANS ' + json.dumps(dict(rows=rows, widen=False, arms=arms)))
    return '\n'.join(lines)


def kernels(served=SERVED):
    value = {}
    for rows, launches in served.items():
        value[str(rows)] = [dict(kernel=s, launches_per_cycle=77, grids=[dict(grid=g, launches=77 * 20)])
                            for s, g in launches]
        value[str(rows)].append(dict(kernel='chop_sinkhorn_hc_deepseek41_expand_f32_v1', launches_per_cycle=3,
                                     grids=[dict(grid=[10, rows, 1], launches=60)]))
    return value


def coverage():
    cases = [dict(id=f'{k}:{n}:[{s}]', kind=k, rows=r) for k, r in job.KINDS.items() for s in range(2)
             for n in ([0] if k == 'decode8x2' else [0, 1])]
    ids = [c['id'] for c in cases]
    return dict(cases=cases, variants=VARIANTS, sanitizer_cases=[next(i for i in ids if i.startswith(k + ':'))
                                                                   for k in job.KINDS],
                timing_cases=[i for i in ids if i.startswith(('decode8x2:', 'verify32:0:'))],
                timing_records=4 * (len(VARIANTS) + 1) * 2, profile_arms=2 * (len(VARIANTS) + 1))


def matrix(cov, sanitizer=False, kinds=tuple(job.KINDS)):
    if sanitizer:
        cases = [c for c in cov['sanitizer_cases'] if c.split(':', 1)[0] in kinds]
        configurations = ['served']
    else:
        cases = [c['id'] for c in cov['cases']]
        configurations = ['served', 'other']
    reports = [dict(case=c, variant=v, configuration=g, passed=True) for c in cases for v in VARIANTS
               for g in configurations]
    served = [dict(case=c, configuration=g, output_sha256=f'{c}{g}') for c in cases for g in configurations]
    firsts = {}
    for c in cov['cases']:
        firsts.setdefault(c['kind'], c['id'])
    checks = [dict(case=f'{firsts[k]}:{v}:served', name='graph_replay1', passed=True) for k in job.KINDS
              for v in VARIANTS]
    return dict(passed=True, sanitizer=sanitizer, variants=VARIANTS, reports=reports, served=served,
                checks=checks, negatives=0 if sanitizer else 30,
                captured=[] if sanitizer else [dict(case=c, passed=True) for c in cases])


def raw_csv(order):
    head = ['ID', 'Kernel Name', 'launch__registers_per_thread', *job.NCU_METRICS]
    rows = [head, [''] * len(head)]
    n = 0
    for o in order:
        for s in o['symbols']:
            rows.append([str(n), s, '64', '2,000', '1,000,000', '100,000', '3,000,000'] +
                        ['1'] * (len(job.NCU_METRICS) - 4))
            n += 1
    return '\n'.join(','.join(f'"{x}"' for x in r) for r in rows)


class Layout(unittest.TestCase):
    def test_served_plans_must_be_the_profiled_pair(self):
        self.assertEqual(job.profiled(plans(), kernels())[32], sorted(SERVED[32]))
        moved = {r: [[s, [g[0], g[1] + 1, 1]] for s, g in v] for r, v in SERVED.items()}
        with self.assertRaises(ValueError):
            job.profiled(plans(), kernels(moved))
        swapped = {**SERVED, 16: SERVED[32]}
        with self.assertRaises(ValueError):
            job.profiled(plans(swapped), kernels())
        with self.assertRaises(ValueError):
            job.profiled(plans({32: SERVED[32]}), kernels())


class Analysis(unittest.TestCase):
    def test_step_totals_scale_the_mean_boundary_saving(self):
        records = [dict(rows=32, variant='cols_r2_w8', hot=False, site=s, reference_us=30.0, candidate_us=c,
                        paired_ratio=c / 30.0) for s, c in ((0, 10.0), (7, 20.0))]
        [row] = job.step_totals(records)
        self.assertEqual((row['sites'], row['saved_us_per_step'], row['faster']), (2, 77 * 15.0, 2))
        with self.assertRaises(ValueError):
            job.step_totals(records + records[:1])

    def test_matrix_requires_every_report_and_control(self):
        cov = coverage()
        self.assertEqual(len(job.check_matrix(matrix(cov), False, cov)), len(cov['cases']) * 2)
        self.assertEqual(len(job.check_matrix(matrix(cov, True, job.RACECHECK_KINDS), True, cov,
                                              job.RACECHECK_KINDS)), 2)
        good = matrix(cov)
        for change in (dict(negatives=20), dict(passed=False), dict(checks=good['checks'][:-1]),
                       dict(reports=good['reports'][:-1]), dict(captured=good['captured'][1:]),
                       dict(served=good['served'][1:]), dict(variants=VARIANTS[:1])):
            with self.assertRaises(ValueError):
                job.check_matrix(dict(good, **change), False, cov)
        bad = matrix(cov)
        bad['reports'][3]['passed'] = False
        with self.assertRaises(ValueError):
            job.check_matrix(bad, False, cov)

    def test_ncu_join_maps_launch_order_and_sums_arms(self):
        order = [dict(case='c', rows=16, arm='served', symbols=[s for s, _ in SERVED[16]]),
                 dict(case='c', rows=16, arm='cols_r2_w8', symbols=['chop_deepseek41_hc_round_cols_r2_v1',
                                                                    'chop_deepseek41_hc_finish_split_w8_v1'])]
        joined = job.ncu_join(raw_csv(order), order)
        self.assertEqual(len(joined), 4)
        table = {t['arm']: t for t in job.ncu_table(joined)}
        self.assertEqual((table['served']['us'], table['served']['dram_read_mb']), (4.0, 2.0))
        with self.assertRaises(ValueError):
            job.ncu_join(raw_csv(order[:1]), order)
        with self.assertRaises(ValueError):
            job.ncu_join(raw_csv(order[::-1]), order)

    def test_analyze_accepts_only_complete_agreeing_gpus(self):
        cov = coverage()
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d) / 'root', Path(d) / 'out'
            (root / 'snapshot/cubins').mkdir(parents=True)
            out.mkdir()
            (root / 'snapshot/cubins/coverage.json').write_text(json.dumps(cov))
            (root / 'staged.json').write_text(json.dumps(dict(environment={})))
            order = []
            for case, rows in (('decode8x2:0:[0]', 16), ('verify32:0:[0]', 32)):
                for arm in ['served'] + VARIANTS:
                    order.append(dict(case=case, rows=rows, arm=arm, symbols=[f'chop_deepseek41_hc_{arm}_a',
                                                                               f'chop_deepseek41_hc_{arm}_b']))
            for o in (0, 1):
                for kind in job.STAGES:
                    (out / f'{kind}-{o}.exit.json').write_text(json.dumps(dict(exit_code=0)))
                (out / f'memcheck-{o}.log').write_text('ERROR SUMMARY: 0 errors')
                (out / f'racecheck-{o}.log').write_text(
                    'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')
                (out / f'memcheck-{o}.json').write_text(json.dumps(matrix(cov, True)))
                (out / f'racecheck-{o}.json').write_text(json.dumps(matrix(cov, True, job.RACECHECK_KINDS)))
                (out / f'matrix-{o}.json').write_text(json.dumps(matrix(cov)))
                records = [dict(case=c, rows=16 if c.startswith('decode') else 32, site=n, variant=v, hot=h,
                                reference_us=30.0, candidate_us=20.0, paired_ratio=0.67)
                           for n, c in enumerate(cov['timing_cases']) for v in ['served'] + VARIANTS
                           for h in (False, True)]
                (out / f'timing-{o}.json').write_text(json.dumps(dict(records=records)))
                (out / f'ncu-order-{o}.json').write_text(json.dumps(dict(order=order)))
                (out / f'ncu-{o}-raw.csv').write_text(raw_csv(order))
            self.assertEqual(job.analyze(root, out), 0)
            report = (out / 'REPORT.md').read_text()
            self.assertIn('| 0 | 32 | cols_r2_w8 | False |', report)
            changed = matrix(cov)
            changed['served'][0]['output_sha256'] = 'other'
            (out / 'matrix-1.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                job.analyze(root, out)
            (out / 'matrix-1.json').write_text(json.dumps(matrix(cov)))
            (out / 'racecheck-1.log').write_text('RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)')
            with self.assertRaises(ValueError):
                job.analyze(root, out)

    def test_listing_and_result_markers(self):
        self.assertTrue(job.passed('test result: ok. 1 passed; 0 failed; 0 ignored'))
        self.assertFalse(job.passed('test result: ok. 1 passed; 0 failed; 0 ignored skipped'))
        self.assertFalse(job.passed('test result: ok. 0 passed; 0 failed; 1 ignored'))
        for text, ok in (('ERROR SUMMARY: 0 errors', True), ('ERROR SUMMARY: 2 errors', False)):
            if ok:
                job.clean(text, 'memcheck')
            else:
                with self.assertRaises(ValueError):
                    job.clean(text, 'memcheck')


if __name__ == '__main__':
    unittest.main()
