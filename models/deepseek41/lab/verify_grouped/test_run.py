"""Exercise each parser, phase, shard and timeout without a GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import run


def index():
    frames=[]
    for rows in [4,32]:
        for i in range(4):
            frames.append(dict(file=f'rows{rows}-{i}/capture.json',rows=rows,records=[dict(layer=layer,rows=rows,
                partial_sha256='a'*64,output_sha256='b'*64) for layer in [0,2,20]]))
    return dict(captures=frames,complete=True,records=24)


def output(rows,phase):
    corpus=index();lines=[]
    for key,record in run.cases(corpus,rows).items():
        if phase=='time':
            for group in run.GROUPS:
                for mode in run.TIMINGS:
                    value=dict(id=key,rows=rows,group=group,mode=mode,old_us=12.5,new_us=5.0,paired_ratio=0.4)
                    lines.append('VERIFY_GROUPED_TIMING '+json.dumps(value))
        else:
            value=dict(id=key,rows=rows,groups=run.GROUPS,modes=run.MODES,poison_replays=64,partitions=4,
                       graph_replays=0 if phase=='memcheck' else 8,negative=True,
                       partial_sha256=record['partial_sha256'],output_sha256=record['output_sha256'])
            lines.append('VERIFY_GROUPED_OK '+json.dumps(value))
            for group in run.GROUPS:
                heads=group or (1 if rows<=4 else 8);ctas=rows*16//heads*8
                units=2 if key.endswith('-L00') else 10
                value=dict(id=key,rows=rows,group=group,ctas=ctas,cycles=[ctas]*6+[ctas*100,rows*16//heads*units],bitwise=True)
                lines.append('VERIFY_GROUPED_CLOCK '+json.dumps(value))
    lines[0]='test verify_grouped::test ... '+lines[0]
    lines+=['VERIFY_GROUPED_COMPLETE '+json.dumps(dict(cases=12,timed=phase=='time',memcheck=phase=='memcheck')),
            'test result: ok. 1 passed; 0 failed; 0 ignored; 17 filtered out; finished in 0.01s']
    if phase=='memcheck':lines.append('========= ERROR SUMMARY: 0 errors')
    return '\n'.join(lines)+'\n'


class Tests(unittest.TestCase):
    def test_complete_parsers_and_numeric_results(self):
        for rows in [4,32]:
            for phase,_ in run.PHASES:
                result=run.parse(output(rows,phase),index(),rows,phase)
                self.assertEqual(result['cases'],12)
                if phase=='time':
                    self.assertEqual(len(result['timings']),144)
                    self.assertTrue(all(r['new_us']/r['old_us']==0.4 for r in result['timings']))
                else:self.assertEqual(result['poison_replays'],768)
    def test_parsers_reject_incomplete_or_wrong_controls(self):
        for phase,_ in run.PHASES:
            text=output(4,phase)
            changes=[text.replace('1 passed','0 passed'),text.replace('0 ignored','1 ignored'),text+'GPU work skipped\n',
                     text.replace('"cases": 12','"cases": 11'),text+'VERIFY_GROUPED_COMPLETE {}\n',
                     text.replace('"rows": 4','"rows": 32',1),'\n'.join(text.splitlines()[1:]),text+'\n'+text.splitlines()[0]]
            if phase=='time':changes += [text.replace('"new_us": 5.0','"new_us": NaN',1),text.replace('"group": 2','"group": 3',1)]
            else:changes += [text.replace('"partitions": 4','"partitions": 3',1),text.replace('"negative": true','"negative": false',1),text.replace('a'*64,'c'*64,1),text.replace('VERIFY_GROUPED_CLOCK','WRONG_CLOCK',1),text.replace('"bitwise": true','"bitwise": false',1)]
            if phase=='memcheck':changes += [text.replace('0 errors','1 errors'),text.replace('ERROR SUMMARY: 0 errors','none')]
            for bad in changes:
                with self.assertRaises(ValueError):run.parse(bad,index(),4,phase)
    def test_all_workflow_failures_stop_later_phases(self):
        class Fake:
            def __init__(self,fail=None):self.events=[];self.fail=fail;self.result=None
            def event(self,name):
                self.events.append(name)
                if name==self.fail:raise TimeoutError(name)
            def preflight(self):self.event('preflight')
            def power(self,edge):self.event('power-'+edge)
            def journal_preflight(self):self.event('journal-preflight')
            def resources(self):self.event('resources');return dict(ranks=2,symbols=33)
            def phase(self,phase,limit):self.event(phase);return dict(limit=limit)
            def journal(self):self.event('journal')
            def verdict(self,result):self.events.append('verdict');self.result=result
        phases=['preflight','power-before','journal-preflight','resources','memcheck','check','time','power-after','journal']
        good=Fake();self.assertEqual(run.workflow(good)['exit_code'],0);self.assertEqual(good.events,phases+['verdict'])
        for failed in phases:
            bad=Fake(failed)
            with self.assertRaises(TimeoutError):run.workflow(bad)
            self.assertEqual(bad.events[-1],'verdict');self.assertEqual(bad.result['exit_code'],1)
            if failed in phases[:7]:
                for later in phases[phases.index(failed)+1:7]:self.assertNotIn(later,bad.events)
    def test_actual_two_shard_processes_and_parsers(self):
        with tempfile.TemporaryDirectory(prefix='verify-phase-flow-') as folder:
            root=Path(folder);script=root/'emit.py'
            script.write_text('import os,sys\nsys.path.insert(0,'+repr(str(Path(__file__).parent))+')\nimport test_run\nassert "CHOPINFER_KERNEL_GPUS" not in os.environ\nrows=int(os.environ["CHOPINFER_VERIFY_ROWS"])\nassert os.environ["CHOPINFER_KERNEL_GPU"]==("0" if rows==4 else "1")\nprint(test_run.output(rows,sys.argv[1]))\n')
            ops=object.__new__(run.Operations);ops.root=root;ops.output=root;ops.capture=root;ops.index=index();ops.deadline=time.monotonic()+30
            ops.command=lambda phase:[sys.executable,str(script),phase]
            with mock.patch.dict(os.environ,CHOPINFER_KERNEL_GPUS='9'):
                for phase,limit in run.PHASES:
                    result=ops.phase(phase,limit);self.assertEqual(set(result),{'4','32'})
            # A failed first shard must still collect the second future.
            ops.command=lambda phase:[sys.executable,'-c','import os,time; time.sleep(0.1); raise SystemExit(3)']
            with self.assertRaises(ValueError):ops.phase('check',5)
            for rows in [4,32]:self.assertTrue((root/f'check-{rows}.log').exists())
    def test_timeout_collects_child_process(self):
        with tempfile.TemporaryDirectory(prefix='verify-timeout-') as folder:
            root=Path(folder);ops=object.__new__(run.Operations);ops.deadline=time.monotonic()+30
            with self.assertRaises(subprocess.TimeoutExpired):
                ops.call([sys.executable,'-c','import time;time.sleep(30)'],os.environ,root/'timeout.log',0.1)
            with self.assertRaises(ValueError):
                ops.call([sys.executable,'-c','raise SystemExit(7)'],os.environ,root/'failed.log',2)
            timer=threading.Timer(0.1,lambda:run.cancel(15,None));timer.start()
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    ops.call([sys.executable,'-c','import time;time.sleep(30)'],os.environ,root/'cancelled.log',20)
            finally:timer.join();run.CANCELLED=False
    def test_commands_keep_selector_and_sanitizer_error_exit(self):
        ops=object.__new__(run.Operations);ops.root=Path('/stage');ops.config=dict(sanitizer='/sanitizer')
        for phase,_ in run.PHASES:
            argv=ops.command(phase)
            self.assertIn(f'verify_grouped::gpu_verify_grouped_{phase}',argv)
            self.assertEqual(argv[-3:],['--exact','--ignored','--nocapture'])
            if phase=='memcheck':self.assertEqual(argv[:5],['/sanitizer','--tool','memcheck','--error-exitcode','86'])

if __name__=='__main__':unittest.main()
