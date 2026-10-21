import json
import tempfile
import unittest
from pathlib import Path

import hc_tail16_job as job


def launch(symbol, grid, block):
    return (f'launch 2 chop_deepseek41_{symbol} LaunchConfigV1 {{ flags: 1, grid_x: {grid[0]}, grid_y: {grid[1]}, '
            f'grid_z: 1, block_x: {block}, block_y: 1, block_z: 1, shared_memory_bytes: 0, cluster: '
            f'LaunchClusterV1 {{ x: 1, y: 1, z: 1 }} }} [NullDevicePointer]')


SERVED = [('hc_round_expand_mix_partials_v1', (20, 16), 256), ('hc_round_expand_mix_finish_v1', (10, 16), 256)]
TAIL = [('hc_round_chain_w4_v1', (160, 16), 128), ('hc_finish_norm_w32_v1', (16, 1), 1024)]
NORM = ('norm_quantize_deepseek41_input_v1', (16, 1), 256)


def journal(groups, n=4 * 77, standalone=12):
    lines = []
    for _ in range(n):
        lines += [launch(*g) for g in groups]
        lines.append('launch 2 chop_deepseek41_other_v1 LaunchConfigV1 { flags: 1, grid_x: 1')
    lines += [launch(*NORM)] * standalone
    return '\n'.join(lines).replace('chop_deepseek41_norm_quantize', 'chop_norm_quantize')


def plans(served, tail):
    arms = [dict(arm='Served', launches=[dict(symbol='chop_deepseek41_' + s, grid=[g[0], g[1], 1], block=[b, 1, 1],
                                              dynamic_shared_bytes=0) for s, g, b in served]),
            dict(arm='Tail', launches=[dict(symbol='chop_deepseek41_' + s, grid=[g[0], g[1], 1], block=[b, 1, 1],
                                            dynamic_shared_bytes=0) for s, g, b in tail])]
    return 'HC_TAIL16_PLANS ' + json.dumps(dict(widen=False, arms=arms))


class Layouts(unittest.TestCase):
    def test_plans_match_the_journals_and_count_every_boundary(self):
        served = SERVED + [NORM]
        text = plans(served, TAIL).replace('chop_deepseek41_norm_quantize', 'chop_norm_quantize')
        result = job.layouts(text, journal(served), journal(TAIL))
        self.assertEqual((result['boundaries_per_rank'], result['standalone_norms']), (77, 12))
        with self.assertRaises(ValueError):
            job.layouts(text, journal(served), journal(TAIL, standalone=11))
        with self.assertRaises(ValueError):
            job.layouts(text, journal(served), journal(TAIL, 4 * 76))
        with self.assertRaises(ValueError):
            job.layouts(text, journal(TAIL), journal(served))
        bad = [(s, (g[0] + 1, g[1]), b) for s, g, b in TAIL]
        with self.assertRaises(ValueError):
            job.layouts(text, journal(served), journal(bad))


class Analysis(unittest.TestCase):
    def test_step_totals_average_cases_per_site(self):
        records = []
        for site in ('[0,0]', '[0,1]'):
            records.append(dict(case=f'decode8x2:{site}', kind='decode8x2', pair='tail', hot=False,
                                reference_us=30.0, candidate_us=20.0))
            for n, cand in enumerate((18.0, 22.0)):
                records.append(dict(case=f'verify32:{n}:{site}', kind='verify32', pair='tail', hot=False,
                                    reference_us=30.0, candidate_us=cand))
        rows = {r['kind']: r for r in job.step_totals(records)}
        self.assertEqual(rows['decode8x2']['saved_us_per_step'], 20.0)
        self.assertEqual(rows['verify32']['saved_us_per_step'], 20.0)
        self.assertEqual(rows['verify32']['sites'], 2)

    def test_matrix_requires_every_report_and_control(self):
        reports = [dict(case=f'c{i}', configuration=c, passed=True, output_sha256='x') for i in range(3)
                   for c in ('served', 'other')]
        value = dict(passed=True, memcheck=False, reports=reports, negatives=10,
                     checks=[dict(passed=True)])
        self.assertEqual(len(job.check_matrix(value, False, 3)), 6)
        for change in (dict(negatives=8), dict(passed=False), dict(checks=[dict(passed=False)]),
                       dict(reports=reports[:-1])):
            with self.assertRaises(ValueError):
                job.check_matrix(dict(value, **change), False, 3)

    def test_listing_and_result_markers(self):
        self.assertTrue(job.passed('test result: ok. 1 passed; 0 failed; 0 ignored'))
        self.assertFalse(job.passed('test result: ok. 1 passed; 0 failed; 0 ignored skipped'))
        self.assertFalse(job.passed('test result: ok. 0 passed; 0 failed; 1 ignored'))


if __name__ == '__main__':
    unittest.main()
