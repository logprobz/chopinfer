import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import bf16_partial_job as job


def capture(token=7):
    return dict(prompt_sha256='same-input', canonical=dict(token_ids=[token], tokens=['x'],
                selected_bits=[job.bits(-0.0)], top_bits=[{'x':job.bits(-0.0)}]))


class Parser(unittest.TestCase):
    def fixture(self):
        choice=dict(index=0, token_ids=[7], finish_reason='length', logprobs=dict(
            tokens=['x'], token_logprobs=[-0.0], top_logprobs=[{'x':-0.0}]))
        return [(1.0, dict(choices=[choice])), (1.1, dict(choices=[], usage=dict(
            prompt_tokens=8192, completion_tokens=1, prompt_tokens_details=dict(cached_tokens=0))))]

    def test_stream_covers_ids_scores_and_cold_input(self):
        value=job.events(self.fixture(),8192,1,0,True)
        self.assertEqual(value['ttft_seconds'],1)
        self.assertEqual(value['canonical']['selected_bits'],[job.bits(-0.0)])
        for field in ['token_ids','logprobs']:
            rows=self.fixture();del rows[0][1]['choices'][0][field]
            with self.assertRaises(ValueError):job.events(rows,8192,1,0,True)
        rows=self.fixture();rows[-1][1]['usage']['prompt_tokens_details']['cached_tokens']=1
        with self.assertRaisesRegex(ValueError,'not cold'):job.events(rows,8192,1,0,True)

    def test_report_only_differences_are_counted(self):
        old={'prefill':[capture()]};new=copy.deepcopy(old)
        new['prefill'][0]['canonical']['token_ids'][0]=8
        new['prefill'][0]['canonical']['selected_bits'][0]=job.bits(0.0)
        new['prefill'][0]['canonical']['top_bits'][0]={'x':job.bits(-1),'y':job.bits(-2)}
        result=job.compare(old,new)
        self.assertFalse(result['bitwise'])
        self.assertEqual([result[k] for k in ['requests','positions','token_ids','selected_logprobs','top_key_changes','top_logprobs']],[1,1,1,1,1,1])
        self.assertTrue(job.compare(old,old)['bitwise'])

    def test_missing_cases_and_nonfinite_scores_refuse(self):
        for a,b in [({},{}),({'x':[]},{'x':[]}),({'x':[capture()]},{'y':[capture()]})]:
            with self.assertRaises(ValueError):job.compare(a,b)
        with self.assertRaises(ValueError):job.bits(float('nan'))

    def test_faults_are_per_process_and_detect_counter_reset(self):
        samples=[{'3:20':dict(minor=4,major=1)},{'3:20':dict(minor=8,major=2),'4:21':dict(minor=1,major=0)},
                 {'3:20':dict(minor=9,major=2),'4:21':dict(minor=3,major=0)}]
        self.assertEqual(job.fault_delta(samples),{'3:20':dict(minor=5,major=1),'4:21':dict(minor=2,major=0)})
        samples[-1]['3:20']['minor']=0
        with self.assertRaises(ValueError):job.fault_delta(samples)


class Workflow(unittest.TestCase):
    def exercise(self, bad_repeat=False, failed_accuracy=False, zero_lab=False):
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary);assets=stage/'assets';(assets/'logs').mkdir(parents=True)
            (assets/'logs/prepare.json').write_text(json.dumps(dict(bundle='cpu-only')))
            (assets/'tokens32k.json').write_text(json.dumps([1]*32768))
            baseline=stage/'qual/deepseek41/baselines/default-20261020';baseline.mkdir(parents=True)
            (baseline/'SHA256SUMS').write_text('')
            instances=[]
            def write(path,value):
                path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            class Server:
                def __init__(self,folder,profile,**kwargs):
                    self.directory=folder/profile;self.directory.mkdir();self.profile=profile
                    self.url='http://127.0.0.1:19001';self.pidfile=self.directory/'pid.json';self.stopped=False
                    instances.append(self)
                def start(self):
                    self.readiness=120
                    write(self.directory/'ready.json',dict(health=dict(serving=dict(source_commit='frozen',
                        profile=dict(name=self.profile,tensor_parallel_size=4),
                        model_options=dict(prefill_chunk_tokens=8192,prefill_upper_skip=True)))))
                def stop(self):self.stopped=True
            def command(argv,log,**kwargs):
                Path(log).write_text(f'test result: ok. {0 if zero_lab else 1} passed; 0 failed; 0 ignored;')
                write(stage/'run/gpu-oracle.json',dict(status='passed',real_cases=249,chunk_cases=13,real_executions=1992,chunk_executions=52))
            def benchmark(p,common,server,ids,prompts,label):
                mode=int(server.env['CHOPINFER_DEEPSEEK41_BF16_PARTIALS'])
                token=8 if mode else 7
                if bad_repeat and label.endswith('-3-0'):token=9
                return ({name:[capture(token)] for name in ['prefill-8192','prefill-32768','decode-c1','decode-c8']},
                        {name:{} for name in ['prefill-8192','prefill-32768','decode-c1','decode-c8']})
            def accuracy(*args):
                if failed_accuracy:raise ValueError('synthetic gate rejection')
                return dict(paired_passed=True,absolute={k:dict(texts=t,positions=n) for k,(t,n) in job.EXPECTED.items()})
            p=SimpleNamespace(Server=Server,write=write,command=command,MODEL='fixture',SOURCE=assets/'source-integrated',
                INSTALL=assets/'install-integrated',telemetry=lambda *args:dict(power_limits_w=[250]*4,xids=[]),
                interrupted=lambda *args:None,http=lambda *args,**kwargs:'{"tokens":[1,2]}',
                bench=SimpleNamespace(stream_prompt=lambda *args:'prompt'))
            saved={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM)}
            try:
                with (patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='0,1,2,3',CHOPINFER_DATA=str(stage),CHOPINFER_DEEPSEEK_CHECKPOINT='fixture')),
                      patch.dict(sys.modules,dict(perf_runner=p,common=SimpleNamespace(),release_preview=SimpleNamespace(environment=lambda **kwargs:{}))),
                      patch.object(job,'verify_seal',return_value=dict(revision='frozen')),
                      patch.object(job,'benchmark',benchmark),patch.object(job,'accuracy',accuracy)):
                    if bad_repeat or failed_accuracy or zero_lab:
                        with self.assertRaises(ValueError):job.run(stage,stage/'run')
                    else:job.run(stage,stage/'run')
                result=json.loads((stage/'run/summary.json').read_text())
                self.assertTrue(all(s.stopped for s in instances))
                return result
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for sig,handler in saved.items():signal.signal(sig,handler)

    def test_abba_reports_cross_mode_differences(self):
        result=self.exercise()
        self.assertEqual(result['status'],'passed')
        self.assertEqual([a['partials'] for a in result['arms']],list(job.ORDER)*2)
        self.assertEqual(result['counts'],dict(servers=8,cold_ttft=16,decode_cohorts=16,scored_positions=18488,paired_profiles=2))
        self.assertFalse(result['comparisons']['tp4-ar-off-on']['bitwise'])
        self.assertFalse(result['comparisons']['tp4-ar-off-on']['gated'])

    def test_repeat_difference_is_a_gate(self):
        self.assertEqual(self.exercise(bad_repeat=True)['status'],'failed')

    def test_failed_accuracy_preserves_both_abba_reports(self):
        result=self.exercise(failed_accuracy=True)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(len(result['arms']),8)
        self.assertEqual(len(result['comparisons']),12)

    def test_zero_lab_tests_are_not_evidence(self):
        self.assertEqual(self.exercise(zero_lab=True)['arms'],[])


if __name__=='__main__':unittest.main(verbosity=2)
