"""CPU parser checks, including replay of a recorded earlier projection run."""
import copy
import json
import os
from pathlib import Path
import unittest
import projection_zero_job as job


def sample_time():
    return dict(ms=[1.0, 1.0], round_samples_ms=[[[1.0]*3, [1.0]*3] for _ in range(5)],
                replays_per_sample=1)


def fixture(worker, mem):
    layer = (0, 39)[worker]
    data = {m: [] for m in job.MARKERS}
    data['BITS'] = [dict(rows=r, label=p, tile=m, comparisons=8, bitwise=True)
                    for r, p in job.cases(mem) for m in job.tiles(r)]
    data['NEGATIVE'] = [dict(mutation=m, changed=True) for m in
                        ('omit_chain', 'force_certificate', 'force_decode')]
    data['ISOLATION'] = [dict(tile=m, bitwise=True) for m in (32, 64)]
    data['OCCUPANCY'] = [dict(tile=m, resident_ctas=1, threads=128,
                              shared_bytes={32: 32768, 64: 57344}[m]) for m in (32, 64)]
    data['DONE'] = [job.expected_done(worker, mem)]
    if not mem:
        data['CAPTURE'] = [dict(layer=layer, rows=8192, tile=m, comparisons=8, bitwise=True)
                           for m in (32, 64)]
        data['CAPTURE_PREFIX'] = [dict(layer=layer, rows=r, comparisons=8, bitwise=True)
                                  for r in (1, 8, 32)]
        data['PARTITION'] = [dict(layer=layer, tile=m, first=f, rows=r, bitwise=True)
                             for m in (32, 64) for f, r in
                             ((0, 129), (129, 320), (449, 2048), (2497, 5695))]
        data['TIME'] = [dict(rows=r, label=p, tile=m,
                             arms=['base_prefill_dispatch', 'zero_components'],
                             includes_prepare=True, timing=sample_time(), cold=sample_time(), prepare=None)
                        for r, p, m in ([(r, 'bf16', m) for r in job.ROWS for m in job.tiles(r)] +
                                        [(8192, 'capture', m) for m in (32, 64)] +
                                        [(r, 'capture_prefix', 32) for r in (1, 8, 32)])]
    return data


def render(data, mem=False):
    text = '\n'.join('test prefix ... PROJECTION_ZERO_' + k + ' ' + json.dumps(row)
                     for k, rows in data.items() for row in rows)
    text += '\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out;\n'
    if mem:
        text += '========= ERROR SUMMARY: 0 errors\n'
    return text


class ParserTests(unittest.TestCase):
    def test_full_and_memcheck_matrices(self):
        for worker in (0, 1):
            for mem in (False, True):
                job.analyze(render(fixture(worker, mem), mem), worker, mem)

    def test_missing_and_duplicate_records_fail(self):
        for marker in job.MARKERS:
            for duplicate in (False, True):
                data = fixture(0, False)
                if duplicate:
                    data[marker].append(copy.deepcopy(data[marker][0]))
                else:
                    data[marker].pop()
                with self.assertRaises(ValueError, msg=marker):
                    job.analyze(render(data), 0, False)

    def test_failed_control_and_missing_cold_sample_fail(self):
        with self.assertRaises(ValueError):
            job.analyze(render(fixture(0, False)), 0, False, 32)
        data = fixture(0, False)
        data['NEGATIVE'][2]['changed'] = False
        with self.assertRaises(ValueError):
            job.analyze(render(data), 0, False)
        data = fixture(0, False)
        data['TIME'][0]['cold']['round_samples_ms'][0][0].pop()
        with self.assertRaises(ValueError):
            job.analyze(render(data), 0, False)

    def test_nonfinite_and_forged_medians_fail(self):
        for value in (float('nan'), float('inf'), 0.0, -1.0, 2.0):
            sample = sample_time()
            sample['ms'][0] = value
            with self.assertRaises(ValueError):
                job.timing(sample)

    def test_zero_tests_skips_and_sanitizer_failures_fail(self):
        for summary in ('ok. 0 passed; 0 failed; 0 ignored;',
                        'ok. 1 passed; 0 failed; 1 ignored;',
                        'FAILED. 0 passed; 1 failed; 0 ignored;'):
            with self.assertRaises(ValueError):
                job.libtest('test result: ' + summary, 1)
        text = render(fixture(0, True), True).replace('ERROR SUMMARY: 0 errors', 'ERROR SUMMARY: 1 errors')
        with self.assertRaises(ValueError):
            job.analyze(text, 0, True)

    def test_recorded_earlier_output(self):
        name = os.environ.get('CHOPINFER_PROJECTION_ZERO_RECORDED')
        if name is None:
            # The standalone job preflight requires this input explicitly.
            return
        text = Path(name).read_text()
        job.libtest(text, 1)
        rows = job.records(text, 'ROUTER_TIME')
        self.assertEqual(len(rows), 56)
        for row in rows:
            job.timing(row['timing'])
            if row['prepare'] is not None:
                job.timing(row['prepare'])
        self.assertEqual(len(job.records(text, 'ROUTER_DONE')), 1)


if __name__ == '__main__':
    unittest.main()
