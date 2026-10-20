"""CPU checks for coverage, exactness, configuration and job lifecycle."""
import collections
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import time
import sys
import tempfile
import unittest

import run

SOURCE = Path(__file__).resolve().parents[4]


def fixture(config="balanced", concurrency=8, count=12):
    cap, mode = run.CONFIGS[config]
    if mode == "fixed":
        left = 32-concurrency
        shape=[]
        for _ in range(concurrency):
            k=min(cap,left);shape.append(k);left-=k
    else:
        shape=[min(cap,(32-concurrency)//concurrency)]*concurrency
    v=sum(k>0 for k in shape);d=concurrency-v;r=sum(k+1 for k in shape if k)
    capacity=d+r if v else 1<<(concurrency-1).bit_length()
    lines=[]
    for step in range(1,count+1):
        if cap:
            lines.append(dict(schema="chopinfer.deepseek41.step-shape.v1",instance_id="1",step_id=str(step),
                              items=concurrency,micros={"target":1000},bulk_passes=0))
        if v:lines.append(dict(schema="chopinfer.device-results.v1",cycle_epoch=step,kind="acceptance_packet"))
        for i,k in enumerate(shape):
            if k:lines.append(dict(schema="chopinfer.deepseek41.drafter-confidence.v1",instance_id="1",step_id=str(step),
                                  request_id=str(i+1),emitted_positions=k,verify_rows=k+1,completion="committed",
                                  outcomes=["unknown"]*k+["not_emitted"]*(5-k)))
        lines.append(dict(schema="chopinfer.step.v1",batch=step,prefill_items=0,decode_rows=d,verify_rows=r,
                          verify_items=v,row_capacity=capacity,live_rows=d+r,execution_us=1000,
                          speculation=None if not cap else dict(mode=mode,reason="drafting" if mode=="adaptive" else mode,planned_drafts=sum(shape))))
    positions={str(j):sum(k>=j for k in shape)*count for j in range(1,6)}
    metrics=dict(verify_items=v*count,draft_tokens=sum(shape)*count,emitted_tokens=v*count,
                 accepted_tokens=0,rows=r*count,drafted_positions=positions,
                 accepted_positions={str(j):0 for j in range(1,6)},
                 step_rows={"dropped":0,"bins":[dict(decode_rows=d,verify_rows=r,row_capacity=capacity,steps=count)]})
    if config == 'k0':
        metrics = dict(available=False, step_rows=metrics['step_rows'])
    return lines,metrics


def encode(lines):return '\n'.join(json.dumps(r) for r in lines)+'\n'


def settings(stage):
    run.write(stage/'config.json',dict(source=str(SOURCE),binary=str(stage/'bin/chopinfer'),
        package=str(stage/'package'),checkpoint=str(stage/'checkpoint'),native_revision='0'*40))


class JobTests(unittest.TestCase):
    def test_all_fourteen_cases_have_positive_full_batch_coverage(self):
        for config in run.CONFIGS:
            for c in (8,16):
                records,metrics=fixture(config,c)
                audit,_=run.audit_window(encode(records),metrics,config,c)
                self.assertEqual(audit['full_steps'],12)
                self.assertEqual(sum(g['steps'] for g in audit['full']),8)
    def test_absent_counters_require_trace_and_histogram_proof(self):
        for c in (8,16):
            rows,metrics=fixture('k0',c)
            self.assertNotIn('accepted_positions',metrics)
            audit,_=run.audit_window(encode(rows),metrics,'k0',c)
            self.assertTrue(audit['coverage']['per_item_exact'])
            self.assertFalse(metrics['available'])
            self.assertEqual(metrics['draft_tokens'],0)
            self.assertEqual(metrics['accepted_positions'],dict.fromkeys(map(str,range(1,17)),0))
            rows,metrics=fixture('balanced',c)
            unavailable=dict(available=False,step_rows=metrics['step_rows'])
            with self.assertRaisesRegex(AssertionError,'missing speculation counters'):
                run.audit_window(encode(rows),unavailable,'balanced',c)
        rows,metrics=fixture('k0',8)
        metrics['step_rows']['available']=False
        with self.assertRaisesRegex(AssertionError,'step-row counters unavailable'):
            run.audit_window(encode(rows),metrics,'k0',8)

    def test_empty_position_maps_are_zero_only_when_totals_agree(self):
        rows,metrics=fixture('uniform1',8)
        metrics['accepted_positions']={}
        run.audit_window(encode(rows),metrics,'uniform1',8)
        rows,metrics=fixture('uniform1',8)
        metrics['drafted_positions']={}
        with self.assertRaises(AssertionError):run.audit_window(encode(rows),metrics,'uniform1',8)

    def test_zero_draft_adaptive_without_counters_is_reportable(self):
        rows,metrics=fixture('k0',8)
        steps=run.analysis.parse(encode(rows),require_phase=False)
        for step in steps:
            step['speculation']=dict(mode='adaptive',reason='await_ar_baseline',planned_drafts=0)
        self.assertTrue(run.analysis.coverage(steps,metrics)['per_item_exact'])
        self.assertEqual(run.fallback_report(steps,'adaptive',8)['actual_zero_verify_rate'],1)

    def test_missing_packet_or_duplicate_diagnostic_fails(self):
        records,metrics=fixture()
        broken=[r for r in records if r['schema']!='chopinfer.device-results.v1']
        with self.assertRaises(AssertionError):run.audit_window(encode(broken),metrics,'balanced',8)
        duplicate=copy.deepcopy(records)
        index=next(i for i,r in enumerate(duplicate) if r['schema'].endswith('drafter-confidence.v1'))
        duplicate.insert(index,copy.deepcopy(duplicate[index]))
        with self.assertRaises(AssertionError):run.audit_window(encode(duplicate),metrics,'balanced',8)
    def test_zero_or_short_coverage_fails(self):
        for n in (0,1,11):
            records,metrics=fixture(count=n)
            with self.assertRaises(AssertionError):run.audit_window(encode(records),metrics,'balanced',8)
    def test_exact_geometry_and_accepted_counts_cannot_drift(self):
        records,metrics=fixture('uniform1',8)
        records[-1]['row_capacity']=32
        with self.assertRaises(AssertionError):run.audit_window(encode(records),metrics,'uniform1',8)
        records,metrics=fixture()
        metrics['accepted_positions']['1']=1
        with self.assertRaises(AssertionError):run.audit_window(encode(records),metrics,'balanced',8)
    def test_zero_draft_adaptive_window_reports_each_reason_without_requiring_activation(self):
        records,metrics=fixture('k0',8)
        steps=[r for r in records if r['schema']=='chopinfer.step.v1']
        reasons=['estimated_slower','realized_slower','cooldown']*4
        for step,reason in zip(steps,reasons):
            step['speculation']=dict(mode='adaptive',reason=reason,planned_drafts=0)
        report=run.fallback_report(steps,'adaptive',8)
        self.assertEqual(report['actual_zero_verify_rate'],1)
        self.assertEqual(report['backoff_rate'],1)
        self.assertEqual(report['reason_counts'],dict.fromkeys(set(reasons),4))
        self.assertEqual(report['reason_rates_per_plan'],dict.fromkeys(set(reasons),1/3))
        for step in steps:step.pop('speculation')
        with self.assertRaises(AssertionError):run.fallback_report(steps,'adaptive',8)

    def test_expired_readiness_deadline_cleans_up_the_owned_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings(root)
            runner=run.Runner(root,root/'out','matrix',dry=True)
            runner.deadline=time.monotonic()-1
            with self.assertRaises(TimeoutError):
                with runner.serving('k0',0):self.fail('expired deadline passed')
            self.assertIsNone(runner.server)
            pid=json.loads((root/'out/0-k0/process.json').read_text())['pid']
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            self.assertTrue((root/'out/0-k0/cleanup.json').exists())

    def test_readiness_never_retries_an_interruption_as_an_http_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings(root)
            runner=run.Runner(root,root/'out','matrix',dry=True)
            runner.deadline=time.monotonic()+.5
            def interrupted(*_args,**_kwargs):raise InterruptedError('requested stop')
            runner.query=interrupted
            with self.assertRaises(InterruptedError):
                with runner.serving('k0',0):self.fail('interruption swallowed')
            self.assertIsNone(runner.server)

    def test_signal_stops_the_owned_server_and_does_not_publish_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings(root)
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('run.py')),
                str(root),str(root/'out'),'--mode','matrix','--dry-run'],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+15
                pidfile=root/'out/0-k0/process.json'
                while not pidfile.exists():
                    self.assertIsNone(process.poll())
                    self.assertLess(time.monotonic(),deadline)
                    time.sleep(.02)
                process.send_signal(signal.SIGTERM)
                output,_=process.communicate(timeout=15)
                self.assertNotEqual(process.returncode,0,output)
                self.assertFalse((root/'out/summary.json').exists())
                self.assertFalse((root/'matrix-result.json').exists())
                pid=json.loads(pidfile.read_text())['pid']
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            finally:
                if process.poll() is None:process.kill();process.communicate(timeout=10)

    def test_cache_requires_both_budgeted_sets(self):
        rows=[dict(schema='chopinfer.mapped-row-cache.v1',phase='ready',cache_id=i,
                   charged_bytes=1<<30,budget_bytes=2<<30,locked_bytes=1024,process_charged_bytes=i*(1<<30)) for i in (1,2)]
        self.assertEqual(run.cache_ready('',required=False),[])
        with self.assertRaises(AssertionError):run.cache_ready('')
        self.assertEqual(len(run.cache_ready(encode(rows))),2)
        with self.assertRaises(AssertionError):run.cache_ready(encode(rows[:1]))
        rows.append(dict(schema='chopinfer.mapped-row-cache.v1',phase='budget-skip'))
        with self.assertRaises(AssertionError):run.cache_ready(encode(rows))
    def test_native_arguments_keep_k0_free_of_speculation_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings(root)
            runner=run.Runner(root,root/'out','matrix')
            for config in run.CONFIGS:
                argv=runner.argv(config,12345)
                self.assertEqual(argv[argv.index('--max-sequences')+1],'16')
                self.assertEqual(argv[argv.index('--verify-warmup')+1],'off')
                if config=='k0':
                    self.assertNotIn('--speculation',argv);self.assertNotIn('--draft-cost',argv)
                else:self.assertEqual(argv[argv.index('--speculative-rows')+1],'32')
    def test_winner_requires_complete_real_exact_matrix(self):
        receipt=dict(schema=run.SCHEMA,status='passed',mode='matrix',dry_run=False,bitwise=True,
                     server_count=7,case_count=14,servers=[])
        for config in run.CONFIGS:
            receipt['servers'].append(dict(config=config,cohorts=[dict(concurrency=c,
                steady={'aggregate_tokens_per_second':200 if config=='balanced' else 100}) for c in (8,16)]))
        self.assertEqual(run.choose_winner(receipt)[0],'balanced')
        for key,value in [('dry_run',True),('bitwise',False),('case_count',0),('status','failed')]:
            broken=copy.deepcopy(receipt);broken[key]=value
            with self.assertRaises(AssertionError):run.choose_winner(broken)
    def test_unsupported_true_uniform_widths_are_not_mislabeled(self):
        self.assertIn('budgeted',run.case_label('uniform2',16))
        self.assertIn('budgeted',run.case_label('uniform3',16))
        self.assertEqual(run.case_label('uniform3',8),'uniform3-C8')


def replay(root):
    summary=json.loads((root/'C8.json').read_text());window=summary['engram']
    text=(root/'server.log').read_bytes()[window['log_start_bytes']:window['log_end_bytes']].decode()
    audit,_=run.audit_window(text,summary['acceptance'],'fixed5',8,exact_geometry=False,diagnostics=False)
    assert audit['full_steps']==42 and audit['coverage']['verify_items']==452
    print('RECORDED_REPLAY_OK',audit['full_steps'],audit['coverage']['verify_items'])


def replay_failed_k0(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arm=root/'0-k0'
    startup=json.loads((arm/'startup.json').read_text())
    assert startup['health']['serving']['profile']['speculation']['tokens']==0
    argv=json.loads((arm/'command.json').read_text())
    assert argv[argv.index('--speculative-tokens')+1]=='0'
    text=(arm/'server.log').read_text()
    marker='startup readiness:'
    assert text.count(marker)==1
    # Startup resets engine batch IDs. The retained readiness line separates
    # startup steps from the only completed workload, C8 warmup.
    warm=text.split(marker,1)[1].split('\n',1)[1]
    steps=run.analysis.parse(warm,require_phase=False)
    hist=collections.Counter((s['decode_rows'],s['verify_rows'],s['row_capacity'])
                             for s in steps if not s['prefill_items'])
    metrics=dict(available=False,step_rows=dict(available=True,dropped=0,bins=[
        dict(decode_rows=d,verify_rows=v,row_capacity=c,steps=n) for (d,v,c),n in sorted(hist.items())]))
    # The original run saved no scrapes before its assertion. This is a
    # reconstructed metric input, not an independent recorded histogram.
    run.write(output/'reconstructed-delta.json',metrics)
    audit,parsed=run.audit_window(warm,metrics,'k0',8)
    assert len(parsed)==65 and audit['full_steps']==62
    assert sum(s['decode_rows']+s['prompt_completions'] for s in steps)==512
    assert not run.cache_ready(text,required=False)
    (output/'warmup.log').write_text(warm)
    report=dict(passed=True,arm='k0-C8 warmup',startup_seconds=startup['seconds'],
                completed_steps=len(steps),full_C8_steps=audit['full_steps'],output_tokens_from_steps=512,
                verify_items=0,resident_ready_sets=0,coverage=audit,
                metric_input='Reconstructed from the native absent-family contract and recorded steps. Original Prometheus scrapes and response bodies were not saved before failure.',
                saved_receipts=str(root),timed_cohort_started=False)
    run.write(output/'replay.json',report)
    print('FAILED_K0_REPLAY_OK',len(steps),audit['full_steps'],512)


def native_preflight(stage):
    stage=Path(stage)
    config=json.loads((stage/'config.json').read_text())
    expected=json.loads((Path(config['package'])/'manifest.json').read_text())['model_options']
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    for k in ('CHOPINFER_KERNEL_GPU','CHOPINFER_KERNEL_GPUS','CHOPINFER_KERNEL_REQUIRE'):
        environment.pop(k,None)
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        runner=run.Runner(stage,Path(tmp)/'out','matrix')
        for config in run.CONFIGS:
            argv=runner.argv(config,19351)+['--list-model-options']
            result=subprocess.run(argv,env=environment,capture_output=True,text=True,timeout=90)
            (stage/f'argv-{config}.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,(config,result.returncode,result.stderr)
            assert json.loads(result.stdout)==expected
            results.append(dict(config=config,argv=argv,exit_code=result.returncode))
    run.write(stage/'native-preflight.json',dict(passed=True,configurations=results,
        scope='Frozen binary parses every complete serve argv, applies the profile and verifies the package. Option listing returns before CUDA initialization.'))
    print('NATIVE_ARGV_OK',len(results))


def dry_run(output, faults=()):
    output=Path(output);output.mkdir(parents=True)
    stages=[]
    for fault in ('',*faults):
        stage=output/(fault or 'success');stage.mkdir();settings(stage);(stage/'runs').mkdir()
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        env['X7_STUB_FAULT']=fault
        modes=('matrix','confirm') if not fault else ('matrix',)
        for mode in modes:
            path=stage/'runs'/mode
            argv=[sys.executable,'-B',str(Path(__file__).with_name('run.py')),str(stage),str(path),'--mode',mode,'--dry-run']
            result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=100)
            (stage/(mode+'.log')).write_text(result.stdout+result.stderr)
            if fault:
                assert result.returncode != 0,(fault,result.stdout)
                assert not (path/'summary.json').exists(),fault
                assert (path/'6-balanced/C8/warm-after.prom').exists(),fault
                assert (path/'6-balanced/C8/warm-outputs.json').exists(),fault
            else:
                assert result.returncode==0,(mode,result.stdout,result.stderr)
                receipt=json.loads((path/'summary.json').read_text())
                assert receipt['status']=='passed' and receipt['dry_run']
                assert receipt['case_count']==(14 if mode=='matrix' else 8)
                if mode=='matrix':
                    assert receipt['servers'][0]['resident_cache_status']=='enabled-unused-no-verification'
                    assert all(not c['positional_acceptance']['available'] for c in receipt['servers'][0]['cohorts'])
            for pidfile in path.glob('*/process.json'):
                pid=json.loads(pidfile.read_text())['pid']
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise AssertionError(f'owned server survived: {pid}')
        stages.append(fault or 'success')
    print('DRY_RUN_OK',stages)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--replay':replay(Path(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1]=='--replay-k0':replay_failed_k0(Path(sys.argv[2]),Path(sys.argv[3]))
    elif len(sys.argv)>1 and sys.argv[1]=='--native':native_preflight(Path(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1]=='--dry-run':dry_run(sys.argv[2],sys.argv[3:])
    else:unittest.main(verbosity=2)
