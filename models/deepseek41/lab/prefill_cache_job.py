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

def capture_file(root, relative):
    path=Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'capture path must be relative: {relative}')
    current=root
    for part in path.parts:
        current=current/part
        if current.is_symlink():
            raise ValueError(f'capture symlink: {current}')
    if not current.is_file():
        raise ValueError(f'capture file missing: {current}')
    return current

def full_capture_manifest(root):
    """Validate published capture data independently of mutable serving caches."""
    root=root.resolve()
    ready=json.loads(capture_file(root,'capture-ready.json').read_text())
    manifest=capture_file(root,Path(ready['path']).relative_to(root))
    if capture.digest(manifest)!=ready['sha256']:
        raise ValueError('capture manifest digest mismatch')
    data=json.loads(manifest.read_text())
    if data['schema']!='chopinfer.prefill-cache-lists.v1' or data['samples'] is not False:
        raise ValueError('capture must contain full selected lists')
    checked={}
    def reference(ref, size=None):
        path=capture_file(manifest.parent,ref['path'])
        if path not in checked:
            checked[path]=capture.digest(path)
        if checked[path]!=ref['sha256']:
            raise ValueError(f'capture data digest mismatch: {path}')
        if size is not None and path.stat().st_size!=size:
            raise ValueError(f'capture data size mismatch: {path}')
        return path
    prompt=json.loads(reference(data['prompt']).read_text())
    if len(prompt)!=capture.ROWS*len(capture.OFFSETS):
        raise ValueError('capture prompt length mismatch')
    expected={(off,layer) for off in capture.OFFSETS for layer in (0,*range(2,40))}
    seen=set()
    for record in data['records']:
        offset=record['offset'];layer=record['layer'];key=(offset,layer)
        if key not in expected or key in seen:
            raise ValueError('capture coverage mismatch')
        seen.add(key)
        positions=list(range(offset,offset+capture.ROWS))
        if record['rows']!=capture.ROWS or record['positions']!=positions:
            raise ValueError('capture row extent mismatch')
        reference(record['window'],capture.ROWS*128*4)
        if layer==0:
            if record['variant']!='window':
                raise ValueError('capture window variant mismatch')
            continue
        if record['variant'] not in ('mixed32','mixed64'):
            raise ValueError('capture mixed variant mismatch')
        reference(record['selected'],capture.ROWS*512*4)
        meta=json.loads(reference(record['metadata']).read_text())
        required=dict(schema='chopinfer.prefill-selected.v1',layer=layer,offset=offset,
            rows=capture.ROWS,top_k=512,positions=positions,token_ids=prompt[offset:offset+capture.ROWS],
            rank=0,replicated=True,ratio=2 if record['variant']=='mixed32' else 1)
        if any(meta.get(k)!=v for k,v in required.items()):
            raise ValueError('capture selected metadata mismatch')
    if seen!=expected:
        raise ValueError('capture coverage mismatch')
    return manifest

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
    if sys.argv[1:]==['--validate-capture']:
        manifest=full_capture_manifest(ROOT)
        print(json.dumps(dict(path=str(manifest),sha256=capture.digest(manifest),records=156)))
        return
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
        hashes=ROOT/('SHA256SUMS' if whole else 'PAIR_SHA256SUMS')
        step('hashes',['sha256sum','--check','--quiet',str(hashes)],cwd=ROOT)
        manifest=None
        if not whole and not samples_only:
            ready=json.loads(step('capture-inputs',[sys.executable,'-B',str(Path(__file__).resolve()),'--validate-capture'],60))
            manifest=Path(ready['path'])
        capture.check_space(ROOT)
        assert len(json.loads((ROOT/'prompt.json').read_text()))==32768
        if dry:
            step('snapshot',[str(BINARY),'cache-check',str(CUBINS)])
            for variant in ['mixed32','mixed64']:
                for offset in [0,24576]:
                    step(f'input-{variant}-{offset}',[str(BINARY),'cache-sample-check',str(CUBINS),str(ROOT/'samples/lists.json'),variant,str(offset)])
            if manifest is not None:
                for variant in ['window','mixed32','mixed64']:
                    for offset in capture.OFFSETS:
                        step(f'full-input-{variant}-{offset}',[str(BINARY),'cache-list-check',str(CUBINS),str(manifest),variant,str(offset)])
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
                assert manifest is not None
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
