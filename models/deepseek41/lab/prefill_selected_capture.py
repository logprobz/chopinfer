#!/usr/bin/env python3
"""Capture full selected lists using the transactional TP4 tap path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

OFFSETS=(0,8192,16384,24576)
ROWS=8192
RESERVE=150*1024**3
BUDGET=4*1024**3

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4<<20),b''): h.update(data)
    return h.hexdigest()

def check_space(path):
    free=shutil.disk_usage(path).free
    if free-BUDGET<RESERVE: raise RuntimeError('capture needs 4 GiB plus the 150 GiB reserve')
    return free

def reference(path,root):
    return dict(path=str(path.relative_to(root)),sha256=digest(path))

def inventory(output,prompt,producer):
    records=[];seen=set();total=0
    for step in sorted((output/'taps').glob('step*')):
        files=sorted(step.glob('selected-L*/selected.json'))
        if not files: continue
        origin=json.loads((step/'capture.json').read_text())
        done=json.loads((step/'completion.json').read_text())
        assert done['completion']=='full' and done['expected']==done['observed'] and done['expected']>0
        assert origin['producer']['startup_nonce']==producer and origin['program']=='Prefill'
        meta=[json.loads(p.read_text()) for p in files]
        offset=meta[0]['offset']
        assert offset in OFFSETS and offset not in seen;seen.add(offset)
        assert [m['layer'] for m in meta]==list(range(2,40))
        positions=list(range(offset,offset+ROWS))
        window=step/'window-keys.i32';assert window.stat().st_size==ROWS*128*4
        records.append(dict(layer=0,variant='window',offset=offset,rows=ROWS,positions=positions,window=reference(window,output)))
        for path,m in zip(files,meta):
            assert m['schema']=='chopinfer.prefill-selected.v1' and m['rows']==ROWS and m['top_k']==512
            assert m['offset']==offset and m['positions']==positions and m['token_ids']==prompt[offset:offset+ROWS]
            assert m['rank']==0 and m['replicated'] is True and m['ratio'] in (1,2)
            selected=path.parent/'selected.i32'
            assert selected.stat().st_size==ROWS*512*4
            records.append(dict(layer=m['layer'],variant='mixed32' if m['ratio']==2 else 'mixed64',offset=offset,rows=ROWS,positions=positions,
                selected=reference(selected,output),metadata=reference(path,output),window=reference(window,output)))
    assert seen==set(OFFSETS) and len(records)==156
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert total<BUDGET and shutil.disk_usage(output).free>=RESERVE
    result=dict(schema='chopinfer.prefill-cache-lists.v1',samples=False,producer=producer,records=records,bytes=total,
                prompt=reference(output/'prompt.json',output),source='committed TP4 prefill selected lists')
    (output/'capture-set.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def http(url,path,body=None,timeout=5):
    request=urllib.request.Request(url+path,None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)

def capture(root,output):
    check_space(root);output.mkdir(exist_ok=False)
    prompt=json.loads((root/'prompt.json').read_text());assert len(prompt)==32768
    (output/'prompt.json').write_text(json.dumps(prompt)+'\n')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    env=dict(os.environ,CHOPINFER_TAPS_DIR=str(output/'taps'),CHOPINFER_TAPS_PREFILL_SELECTED='1',
             CHOPINFER_TAPS_SITES='L02.x.hc.attn.pre',CHOPINFER_TAPS_FROM='0')
    for key in ['CHOPINFER_TAPS_BYTES','CHOPINFER_TAPS_LAYERS','CHOPINFER_TAPS_SCOPE_JSON','CHOPINFER_TAPS_INDEXER','CHOPINFER_KERNEL_GPU','CHOPINFER_KERNEL_GPUS']:
        env.pop(key,None)
    command=[str(root/'install/bin/chopinfer'),'serve','--model',env['CHOPINFER_DEEPSEEK_CHECKPOINT'],
        '--tensor-parallel-size','4','--model-cache-dir',str(root/'model-cache'),'--listen',f'127.0.0.1:{port}',
        '--served-model-name','DeepSeek-V4.1-Flash','--prefix-cache','off','--max-sequences','1',
        '--prefill-budget','fixed','--max-batch-tokens','8192','--prefill-chunk-tokens','8192',
        '--drain-timeout-ms','5000','--speculative-tokens','0']
    started=time.time();deadline=time.monotonic()+480;process=None
    record=dict(command=command,started_unix_seconds=started,exit_code=None)
    def interrupted(signum,frame):raise InterruptedError(f'capture signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(490)
    try:
        with (output/'server.stdout.log').open('w') as stdout,(output/'server.stderr.log').open('w') as stderr:
            process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            ready=min(deadline,time.monotonic()+240)
            while True:
                if process.poll() is not None:raise RuntimeError('capture server exited before health')
                try:health=http(url,'/health');break
                except OSError:
                    if time.monotonic()>ready:raise TimeoutError('capture readiness deadline')
                    time.sleep(.2)
            (output/'health.json').write_text(json.dumps(health,indent=2)+'\n')
            response=http(url,'/v1/completions',dict(model='DeepSeek-V4.1-Flash',prompt=prompt,temperature=0,max_tokens=1,ignore_eos=True,stream=False),max(1,deadline-time.monotonic()))
            assert response['usage']['prompt_tokens']==32768 and response['usage']['completion_tokens']==1
            (output/'response.json').write_text(json.dumps(response,indent=2)+'\n')
            result=inventory(output,prompt,health['serving']['startup_nonce'])
            print(f"PASS prefill_selected_capture offsets=4 layers=38 rows=8192 records={len(result['records'])} committed=true",flush=True)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM): signal.signal(sig,signal.SIG_IGN)
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            record['exit_code']=process.returncode
        record['ended_unix_seconds']=time.time()
        (output/'server.step.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();capture(a.root,a.output)
