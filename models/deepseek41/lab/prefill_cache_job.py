#!/usr/bin/env python3
"""Bounded capture and pair replay jobs with strict real-list admission."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import prefill_attention_job as logs
import prefill_selected_capture as capture
ROOT=Path(__file__).resolve().parent
BINARY=ROOT/'snapshot/bin/prefill-attention'
CUBINS=ROOT/'snapshot/cubins'

def gate(text,variant,offset,rows,timing):
    logs.clean_gpu_output(text)
    if timing:
        records=[]
        for m in re.finditer(r'TIMING prefill_cache ([^\n]+)',text):
            r=dict(re.findall(r'(\w+)=([^\s]+)',m[1]));records.append(r)
            assert r['variant']==variant and int(r['offset'])==offset and int(r['rows'])==rows and r['actual_selected']=='true' and r['samples']=='3'
            for name in ['paired_total_us','cached_total_us','producer_us','ratio']:
                assert 0<float(r[name])<1e9
        assert sorted(r['rank'] for r in records)==['0','1']
        return records
    records=re.findall(r'PASS prefill_cache rank=(\d+) variant=(\w+) offset=(\d+) rows=(\d+) workspace=true output=true replay=true f64=true cache_guards=true guard_control=true disjoint=true refresh=true actual_selected=true page_cases=(\d+) selected_sha256=([0-9a-f]{64}) cache_bytes=(\d+)',text)
    assert len(records)==2 and sorted(r[0] for r in records)==['0','1']
    assert all(r[1]==variant and int(r[2])==offset and int(r[3])==rows for r in records)
    assert all(int(r[4]) == (0 if variant=='window' else 3) for r in records)
    return records

def main():
    flags=set(sys.argv[1:]);assert flags<= {'--capture','--dry-run','--samples-only'}
    whole='--capture' in flags;dry='--dry-run' in flags;samples_only='--samples-only' in flags
    assert not (whole and samples_only)
    start=time.time();deadline=time.monotonic()+500
    out=Path(os.environ['L11_CACHE_RUN_DIR']);out.mkdir(exist_ok=True)
    verdict=dict(exit_code=1,capture=whole,dry_run=dry,samples_only=samples_only)
    def interrupted(sig,frame):raise InterruptedError(f'job signal {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def step(name,args,timeout=30,**kw):
        text,_=logs.run_step(out,name,args,min(timeout,max(.001,deadline-time.monotonic())),**kw)
        return text+'\n'+(out/f'{name}.stderr.log').read_text() if name.startswith(('sample-','check-','time-')) else text
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    def power(name):
        text=step('power-'+name,['nvidia-smi','-i',visible,'--query-gpu=index,power.limit','--format=csv,noheader,nounits'])
        values=[l.split(',') for l in text.splitlines() if l.strip()]
        assert len(values)==(4 if whole else 2) and all(float(r[1])==250 for r in values)
    try:
        step('hashes',['sha256sum','--check','--quiet',str(ROOT/'SHA256SUMS')],cwd=ROOT)
        capture.check_space(ROOT)
        assert len(json.loads((ROOT/'prompt.json').read_text()))==32768
        if dry:
            step('snapshot',[str(BINARY),'cache-check',str(CUBINS)])
            for variant in ['mixed32','mixed64']:
                for offset in [0,24576]:
                    step(f'input-{variant}-{offset}',[str(BINARY),'cache-sample-check',str(CUBINS),str(ROOT/'samples/lists.json'),variant,str(offset)])
            verdict['exit_code']=0;print('PASS cache_job_dry_run no_gpu_access=true actual_sample_inputs=true');return
        assert len(visible.split(','))==(4 if whole else 2)
        power('before')
        assert not step('idle',['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits']).strip()
        if whole:
            step('capture',[sys.executable,'-B',str(ROOT/'prefill_selected_capture.py'),'--root',str(ROOT),'--output',str(out/'served')],495)
        else:
            env=dict(os.environ,CHOPINFER_KERNEL_GPUS='0,1',CHOPINFER_KERNEL_REQUIRE='1',CHOPINFER_KERNEL_RECEIPTS=str(out/'receipts'))
            for name in ['CHOPINFER_KERNEL_GPU','CHOPINFER_KERNEL_SHARED_DEVICE','KERNELS_LOCK_UPDATE']:env.pop(name,None)
            step('resources',[sys.executable,'-B',str(ROOT/'prefill_attention_resources.py')],30,env=env)
            evidence=[]
            def run(mode,variant,offset,manifest,rows,seconds):
                name=f'{mode}-{variant}-{offset}'
                text=step(name,[str(BINARY),'cache-'+{'sample':'sample','check':'bitwise','time':'time'}[mode],str(CUBINS),str(manifest),variant,str(offset)],seconds+5,env=dict(env,CHOPINFER_KERNEL_TIMEOUT_SECS=str(seconds)))
                evidence.append(dict(phase=mode,variant=variant,offset=offset,result=gate(text,variant,offset,rows,mode=='time')))
                (out/'progress.json').write_text(json.dumps(evidence,indent=2)+'\n')
            for variant in ['mixed32','mixed64']:
                for offset in [0,24576]:run('sample',variant,offset,ROOT/'samples/lists.json',32,20)
            if not samples_only:
                ready=json.loads((ROOT/'capture-ready.json').read_text());manifest=Path(ready['path'])
                assert manifest.is_file() and not manifest.is_symlink() and manifest.resolve().is_relative_to(ROOT) and capture.digest(manifest)==ready['sha256']
                data=json.loads(manifest.read_text());assert data['samples'] is False and len(data['records'])==156
                for variant in ['window','mixed32','mixed64']:
                    for offset in capture.OFFSETS:run('check',variant,offset,manifest,8192,20)
                for variant in ['window','mixed32','mixed64']:
                    for offset in capture.OFFSETS:run('time',variant,offset,manifest,8192,10)
            verdict['evidence']=evidence
        verdict['exit_code']=0
    except BaseException as error:
        verdict['error']=repr(error)
        with (out/'wrapper.stderr.log').open('a') as f:traceback.print_exc(file=f)
    finally:
        if not dry:
            try:power('after')
            except BaseException as error:verdict.update(exit_code=1,power_error=repr(error))
            try:
                text,code=logs.run_step(out,'journal',['journalctl','-k','--since',f'@{int(start)}','--no-pager'],30,check=False)
                xids=re.findall(r'^.*NVRM:.*Xid.*$',text,re.MULTILINE)
                (out/'xids.txt').write_text('\n'.join(xids));verdict['journal_exit_code']=code
                if code or xids:verdict['exit_code']=1
            except BaseException as error:verdict.update(exit_code=1,journal_error=repr(error))
        if whole and not dry and verdict['exit_code']==0:
            manifest=out/'served/capture-set.json'
            tmp=ROOT/'capture-ready.json.tmp';tmp.write_text(json.dumps(dict(path=str(manifest),sha256=capture.digest(manifest)))+'\n');tmp.replace(ROOT/'capture-ready.json')
        verdict['elapsed_seconds']=time.time()-start
        (out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict));raise SystemExit(verdict['exit_code'])

if __name__=='__main__':main()
