"""Reject incomplete core qualification receipts before consuming a GPU slot."""
import copy
import json
import os
from pathlib import Path
import unittest
import fold_core_job as job


def catalog(phase):
    return dict(phase=phase, rows=[1, 8, 32] if phase=='decode' else [64, 2496, 8192],
                memcheck_rows=[8] if phase=='decode' else [65],
                variants=[dict(n=1280,k=5120,groups=1,variant=job.VALID[phase][0])])


def inputs():
    return dict(captures=[dict(id='verify-32-0-q-a',phase='verify-32',layer=0,
                              n=1280,k=5120,groups=1,rows=32)])


def sample_time():
    return dict(ms=[1.0,1.0],round_samples_ms=[[[1.0]*3,[1.0]*3] for _ in range(5)],replays_per_sample=1)


def records(phase, mem):
    cat=catalog(phase);variant=job.VALID[phase][0]
    result=[]
    for n,k,g,rows,v,source in job.expected(cat,inputs(),variant,mem):
        result.append(dict(phase=phase,n=n,k=k,groups=g,rows=rows,variant=v,source=source,
                           comparisons=8,bitwise=True,negative=True,delayed_admission=phase=='decode',
                           memcheck=mem,warm=sample_time(),cold=sample_time()))
    return result


def render(phase, mem, rows):
    text='\n'.join('test prefix ... FOLD_CORE_CASE '+json.dumps(r) for r in rows)
    text+='\nFOLD_CORE_DONE '+json.dumps(dict(phase=phase,memcheck=mem,cases=len(rows)))
    text+='\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out;\n'
    if mem:text+='========= ERROR SUMMARY: 0 errors\n'
    return text


class Tests(unittest.TestCase):
    def test_complete_matrices(self):
        for phase in job.VALID:
            for mem in (False,True):
                job.analyze(render(phase,mem,records(phase,mem)),catalog(phase),inputs(),job.VALID[phase][0],mem)

    def test_missing_duplicate_and_wrong_variant_fail(self):
        for mode in ('missing','duplicate','wrong'):
            rows=records('decode',False)
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows.append(copy.deepcopy(rows[0]))
            else:rows[0]['variant']='wrong'
            with self.assertRaises(ValueError):
                job.analyze(render('decode',False,rows),catalog('decode'),inputs(),job.VALID['decode'][0],False)

    def test_failed_controls_and_incomplete_samples_fail(self):
        for key in ('negative','bitwise','delayed_admission','comparisons','samples'):
            rows=records('decode',False)
            if key=='samples':rows[0]['cold']['round_samples_ms'][0][0].pop()
            elif key=='comparisons':rows[0][key]=0
            else:rows[0][key]=False
            with self.assertRaises(ValueError):
                job.analyze(render('decode',False,rows),catalog('decode'),inputs(),job.VALID['decode'][0],False)

    def test_zero_tests_and_dirty_memcheck_fail(self):
        text=render('prefill',True,records('prefill',True))
        for bad in (text.replace('1 passed','0 passed'),text.replace('0 errors','1 errors')):
            with self.assertRaises(ValueError):
                job.analyze(bad,catalog('prefill'),inputs(),job.VALID['prefill'][0],True)

    def test_recorded_event_parser(self):
        path=os.environ.get('CHOPINFER_FOLD_CORE_RECORDED')
        if path is None:return
        text=Path(path).read_text();job.shared.libtest(text,1)
        events=job.shared.records(text,'PROJECTION_ZERO_TIME')
        self.assertEqual(len(events),24)
        for event in events:
            job.shared.timing(event['timing']);job.shared.timing(event['cold'])


if __name__=='__main__':unittest.main()
