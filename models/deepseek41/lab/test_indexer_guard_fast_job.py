"""Require each shard, arithmetic control and timing sample without a GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import indexer_guard_fast_job as job


def fixture():
    logs, receipts = [], []
    for shard in (0, 1):
        lines = [f'test prefix ... GUARD_MATRIX_COMPLETE shard={shard} shapes=51 total=102',
                 f'GUARD_REAL_COMPLETE shard={shard} captures=105',
                 f'SERVING_MMA_COMPLETE shard={shard} comparisons=150']
        cases = [dict(matrix=i, layer=layer, rows=rows, offset=offset, case=case)
                 for i, (layer,rows,offset,case) in enumerate(job.shapes()) if i % 2 == shard]
        cases.extend(dict(capture=i, context=job.CONTEXTS[i], layer=(2,8,14,20,24)[i%5], rows=rows)
                     for i in range(shard,30,2) for rows in job.ROWS)
        for shape in cases:
            n = shape.get('case', 0)
            control = n == 1
            admitted = 100 if shape['rows'] <= 32 else 0
            coarse, fine, fallback = (admitted,0,0)
            if n == 1:
                coarse, fine, fallback = (0,admitted,admitted)
            if n == 8:
                coarse, fine, fallback = (0,admitted,0)
            if n == 9:
                coarse, fine, fallback = (90,40,10)
            if n == 10:
                coarse, fine, fallback = (0,admitted,admitted)
            case = dict(shape=shape, guard_mismatches=0, fast=True, admitted=admitted,
                        coarse_slots=coarse, fine_slots=fine, fallback_slots=fallback, fallback_rows=0,
                        counts_measured=shape['rows'] <= 32, negative_control_rejected=control,
                        unguarded_score_mismatches=int(control))
            lines.append('test prefix ... GUARD_CASE ' + json.dumps(case))
            rounds = []
            if n == 0:
                for i, pair in enumerate(job.PAIRS):
                    lines.append('prefix GUARD_TIMING ' + json.dumps(dict(shape=shape,arms=pair,
                                 median_ms=[1.0,0.8],paired_ratio=0.8,exclusive=True)))
                    rounds.append(dict(kind='paired_summary',rounds=5,measurement=i))
                    rounds.extend(dict(measurement=i,round=r,arm=a,elapsed_ms=1.0)
                                  for r in range(5) for a in range(2))
            receipts.append(dict(shape=shape,fast=True,mismatches=0,measurement_label='exclusive-device',timing_rounds=rounds))
        lines.append('test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s')
        logs.append('\n'.join(lines))
    return logs, receipts


def check_receipts(records, report):
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for i, record in enumerate(records):
            path = Path(directory) / f'{i}.json'
            path.write_text(json.dumps(record))
            paths.append(path)
        job.receipts(paths, report)


class Checks(unittest.TestCase):
    def test_all_markers_and_raw_samples(self):
        logs, records = fixture()
        report = job.results(logs)
        self.assertEqual(report['timing_pairs'], 846)
        self.assertEqual(len(report['frequency']), 21)
        check_receipts(records, report)

    def test_missing_duplicate_failing_and_control_records_rejected(self):
        logs, _ = fixture()
        for before, after in [('shapes=51','shapes=50'),('0 failed','1 failed'),
                              ('"matrix": 0','"matrix": 2'),('"rows": 1','"rows": 2'),
                              ('"guard_mismatches": 0','"guard_mismatches": 1'),
                              ('"negative_control_rejected": true','"negative_control_rejected": false'),
                              ('"fine_slots": 100','"fine_slots": 101'),
                              ('"coarse_slots": 90','"coarse_slots": 100'),
                              ('"exclusive": true','"exclusive": false'),
                              ('"paired_ratio": 0.8','"paired_ratio": NaN')]:
            changed = [logs[0].replace(before,after,1), logs[1]]
            self.assertNotEqual(changed, logs)
            with self.assertRaises(AssertionError, msg=before):
                job.results(changed)

    def test_missing_or_duplicate_raw_samples_rejected(self):
        logs, records = fixture()
        report = job.results(logs)
        for mutate in ('drop_sample', 'duplicate_sample', 'drop_case', 'shared_measurement'):
            changed = copy.deepcopy(records)
            if mutate == 'drop_sample':
                changed[0]['timing_rounds'].pop()
            elif mutate == 'duplicate_sample':
                changed[0]['timing_rounds'].append(changed[0]['timing_rounds'][-1])
            elif mutate == 'drop_case':
                changed.pop()
            else:
                changed[0]['measurement_label'] = 'shared-device'
            with self.assertRaises(AssertionError, msg=mutate):
                check_receipts(changed, report)


if __name__ == '__main__':
    unittest.main()
