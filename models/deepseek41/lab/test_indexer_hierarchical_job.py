"""Fail closed on missing hierarchy cases, controls and raw timing samples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import indexer_hierarchical_job as job


def fixture():
    logs, receipts = [], []
    for shard in (0,1):
        lines = [f'prefix GUARD_MATRIX_COMPLETE shard={shard} shapes={74-shard} total=147',
                 f'prefix GUARD_REAL_COMPLETE shard={shard} captures=105',
                 f'prefix SERVING_MMA_COMPLETE shard={shard} comparisons=150']
        shapes = [dict(matrix=i,layer=l,rows=r,offset=o,case=c,top_k={21:128,22:1,23:511}.get(c,512))
                  for i,(l,r,o,c) in enumerate(job.shapes()) if i%2 == shard]
        shapes.extend(dict(capture=i,context=job.guard.CONTEXTS[i],layer=(2,8,14,20,24)[i%5],rows=r)
                      for i in range(shard,30,2) for r in job.guard.ROWS)
        for shape in shapes:
            control = shape.get('case') == 20
            case = dict(shape=shape,mismatches=0,old_launches=4,new_launches=5,
                        intermediate_bytes=shape['rows']*4*513*8,
                        negative_control_rejected=control,negative_mismatches=int(control))
            lines.append('prefix HIER_CASE '+json.dumps(case))
            rounds = []
            if shape.get('case',0) == 0:
                for stage in job.STAGES:
                    lines.append('prefix HIER_TIMING '+json.dumps(dict(shape=shape,stage=stage,
                                 median_ms=[1.,.8],paired_ratio=.8,exclusive=True)))
                    rounds.append(dict(kind='paired_summary',rounds=5,measurement=stage))
                    rounds.extend(dict(round=r,arm=a,measurement=stage,elapsed_ms=1.) for r in range(5) for a in range(2))
            receipts.append(dict(shape=shape,mismatches=0,measurement_label='exclusive-device',timing_rounds=rounds))
        lines.append('test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1s')
        logs.append('\n'.join(lines))
    return logs,receipts


def check_receipts(records,report):
    with tempfile.TemporaryDirectory() as temp:
        paths = []
        for i,record in enumerate(records):
            path = Path(temp)/f'{i}.json';path.write_text(json.dumps(record));paths.append(path)
        job.receipts(paths,report)


class Checks(unittest.TestCase):
    def test_all_coverage_and_raw_timings(self):
        logs,records = fixture()
        report = job.results(logs)
        self.assertEqual(report['timing_pairs'],864)
        check_receipts(records,report)

    def test_missing_duplicate_wrong_and_failing_markers(self):
        logs,_ = fixture()
        for before,after in [('shapes=74','shapes=73'),('0 failed','1 failed'),('"matrix": 0','"matrix": 2'),
                             ('"mismatches": 0','"mismatches": 1'),('"new_launches": 5','"new_launches": 4'),
                             ('"top_k": 1}','"top_k": 2}'),('"negative_mismatches": 1','"negative_mismatches": 0'),
                             ('"stage": "merge"','"stage": "select"'),('"exclusive": true','"exclusive": false')]:
            changed=[logs[0].replace(before,after,1),logs[1]]
            self.assertNotEqual(changed,logs)
            with self.assertRaises(AssertionError,msg=before):job.results(changed)

    def test_missing_case_sample_or_exclusive_receipt(self):
        logs,records = fixture();report=job.results(logs)
        for case in ('missing','sample','shared'):
            changed=copy.deepcopy(records)
            if case=='missing':changed.pop()
            elif case=='sample':changed[0]['timing_rounds'].pop()
            else:changed[0]['measurement_label']='shared-device'
            with self.assertRaises(AssertionError,msg=case):check_receipts(changed,report)


if __name__ == '__main__':unittest.main()
