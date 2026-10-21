"""Reject incomplete core qualification receipts before consuming a GPU slot."""
import copy
import json
import os
from pathlib import Path
import unittest
import tempfile
from unittest import mock
from types import SimpleNamespace
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

    def test_local_rows_require_real_memcheck_and_served_counts(self):
        cat=dict(phase='decode',rows=[8,32],memcheck_rows=[8,32],captured_memcheck=True,
                 variants=[dict(n=8192,k=1280,groups=1,variant='local32n32')])
        matrix=dict(captures=[dict(id=f'{phase}-{layer}-q-b',phase=phase,layer=layer,
            rows=rows,n=8192,k=1280,groups=1) for phase,rows in [('decode-8',8),('verify-32',32)]
            for layer in [0,39]])
        for mem,wanted in [(True,4),(False,8)]:
            keys=job.expected(cat,matrix,'local32n32',mem)
            self.assertEqual(len(keys),wanted)
            cases=[dict(phase='decode',n=n,k=k,groups=g,rows=r,variant=v,source=source,
                comparisons=8,bitwise=True,negative=True,delayed_admission=True,memcheck=mem,
                warm=sample_time(),cold=sample_time()) for n,k,g,r,v,source in keys]
            job.analyze(render('decode',mem,cases),cat,matrix,'local32n32',mem)
            missing=[c for c in cases if c['source']=='synthetic']
            with self.assertRaises(ValueError):
                job.analyze(render('decode',mem,missing),cat,matrix,'local32n32',mem)
            wrong=copy.deepcopy(cases);wrong[-1]['memcheck']=not mem
            with self.assertRaises(ValueError):
                job.analyze(render('decode',mem,wrong),cat,matrix,'local32n32',mem)

    def test_execute_runs_both_workers_and_stops_after_bad_memcheck(self):
        cat=dict(phase='decode',rows=[8,32],memcheck_rows=[8,32],captured_memcheck=True,
                 variants=[dict(n=8192,k=1280,groups=1,variant='local32n16')])
        matrix=dict(captures=[dict(id=f'{phase}-0-q-b',phase=phase,layer=0,
            rows=rows,n=8192,k=1280,groups=1) for phase,rows in [('decode-8',8),('verify-32',32)]])
        for broken in [False,True]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'inputs').mkdir();out=root/'out';out.mkdir()
                (root/'inputs/manifest.json').write_text(json.dumps(matrix))
                calls=[]
                def spawn(cmd,env,stdout,**_):
                    mem=env['CHOPINFER_FOLD_CORE_MEMCHECK']=='1'
                    self.assertEqual('--tool' in cmd,mem)
                    self.assertIn('gpu_pair',cmd)
                    calls.append((mem,env['CHOPINFER_KERNEL_GPU']))
                    keys=job.expected(cat,matrix,'local32n16',mem)
                    cases=[dict(phase='decode',n=n,k=k,groups=g,rows=r,variant=v,source=source,
                        comparisons=8,bitwise=True,negative=True,delayed_admission=True,memcheck=mem,
                        warm=sample_time(),cold=sample_time()) for n,k,g,r,v,source in keys]
                    if broken and mem: cases[-1]['negative']=False
                    stdout.write(render('decode',mem,cases));stdout.flush()
                    return SimpleNamespace(wait=lambda **_:0,poll=lambda:0,pid=1)
                plan=dict(sanitizer='recorded-sanitizer',catalogs=dict(decode=cat),
                          budgets=dict(decode=dict(memcheck=60,timing=210,overhead=30)))
                with mock.patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'), \
                     mock.patch.object(job,'seal'), \
                     mock.patch.object(job.shared,'power',return_value=[250,250]), \
                     mock.patch.object(job.subprocess,'Popen',side_effect=spawn), \
                     mock.patch.object(job.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')):
                    if broken:
                        with self.assertRaises(ValueError): job.execute(root,plan,'decode','local32n16',out)
                    else: job.execute(root,plan,'decode','local32n16',out)
                self.assertEqual(calls,[(True,'0'),(True,'1')]+([] if broken else [(False,'0'),(False,'1')]))
                verdict=json.loads((out/'verdict.json').read_text())
                self.assertEqual(verdict['exit_code'],1 if broken else 0)
                if not broken:self.assertEqual(len(json.loads((out/'results.json').read_text())),2)

    def test_recorded_fold_parser(self):
        root=os.environ.get('CHOPINFER_FOLD_CORE_RECORDED_ROOT')
        if root is None:
            self.skipTest('requires a recorded fold pair')
        root=Path(root)
        meta=json.loads((root/'recorded-fold.json').read_text())
        for mem in [False,True]:
            name='recorded-memcheck.log' if mem else 'recorded-fold.log'
            cases=job.analyze((root/name).read_text(),meta['catalog'],meta['inputs'],meta['variant'],mem)
            self.assertEqual(len(cases),16 if mem else 104)

    def test_recorded_event_parser(self):
        path=os.environ.get('CHOPINFER_FOLD_CORE_RECORDED')
        if path is None:return
        text=Path(path).read_text();job.shared.libtest(text,1)
        events=job.shared.records(text,'PROJECTION_ZERO_TIME')
        self.assertEqual(len(events),24)
        for event in events:
            job.shared.timing(event['timing']);job.shared.timing(event['cold'])


if __name__=='__main__':unittest.main()
