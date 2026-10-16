"""Run sealed later-layer controls on one assigned GPU, capped at 20 minutes."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

SELECTOR = 'later_context::prefill_later_layer_context'
MODES = ('captured', 'seeded', 'reordered', 'relocated', 'warm')

def digest(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('deadline hashing prepared input')
            h.update(block)
    return h.hexdigest()

def stop(child, grace=1):
    def send(sig):
        try: os.killpg(child.pid, sig)
        except ProcessLookupError: pass
    try:
        if child.poll() is None:
            send(signal.SIGTERM)
            try: child.wait(timeout=grace)
            except subprocess.TimeoutExpired: pass
    finally:
        if child.poll() is None: send(signal.SIGKILL)
        child.wait()

def run(command, env, log, deadline):
    remaining = deadline-time.monotonic()
    if remaining < 1: raise TimeoutError('operator deadline reached')
    with log.open('x') as output:
        child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
        except BaseException:
            stop(child, min(3, max(0.1, deadline-time.monotonic())))
            raise
    if code: raise RuntimeError(f'exit {code}: {log}')

def trace_summary(path, device):
    con = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    rows = con.execute('SELECT k.start,k.end,k.streamId,k.graphId,k.deviceId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start,k.end').fetchall()
    con.close()
    if not rows or {row[4] for row in rows} != {device}:
        raise RuntimeError('missing trace kernels or kernels on an unassigned GPU')
    gates = [r for r in rows if r[5].startswith('chop_moe_mimo_gateup_prefill_')]
    if not gates or not gates[-1][3]: raise RuntimeError('node-level CUDA graph IDs are required')
    target = gates[-1][3]
    stream = gates[-1][2]
    latest = 0
    selected = {'gate_up': [], 'down': []}
    for start, end, sid, graph, _, name in rows:
        if sid != stream: continue
        charge = max(0, end-max(start, latest))/1000
        latest = max(latest, end)
        if graph != target: continue
        for stage, prefix in [('gate_up','chop_moe_mimo_gateup_prefill_'),('down','chop_moe_mimo_down_prefill_')]:
            if name.startswith(prefix): selected[stage].append({'raw_us':(end-start)/1000, 'charge_us':charge})
    if any(len(values)<5 for values in selected.values()):
        raise RuntimeError('need at least five complete gate/down graph replays')
    return {'graph_id':target, 'selection':'last five replays of the final captured layer graph',
            'stages':{stage:{'samples':values[-5:], **{key:statistics.median(v[key] for v in values[-5:]) for key in ('raw_us','charge_us')}} for stage,values in selected.items()}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('device',type=int,choices=range(4));parser.add_argument('end_utc')
    args=parser.parse_args()
    # The test plans its rank's weights from the checkpoint.
    if not os.environ.get('CHOPINFER_MIMO_CHECKPOINT'): parser.error('set CHOPINFER_MIMO_CHECKPOINT to the MiMo checkpoint')
    root = Path(os.environ['CHOPINFER_DATA']) / 'kopt-moe-prefill'
    end=datetime.datetime.fromisoformat(args.end_utc.replace('Z','+00:00'))
    if end.tzinfo is None: parser.error('UTC offset required')
    seconds=min(1190,end.timestamp()-time.time()-5)
    if seconds<30: parser.error('run window expired')
    deadline=time.monotonic()+seconds
    def interrupted(signum, _frame):
        raise TimeoutError(f'run interrupted by signal {signum}')
    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    ready=json.loads((root/'later-layer-ready.json').read_text())
    for path,want in ready['sha256'].items():
        if time.monotonic() >= deadline: raise TimeoutError('deadline during preflight')
        if digest(path, deadline)!=want: raise RuntimeError('prepared file changed: '+path)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    logs=root/'logs'/('later-'+stamp);logs.mkdir(exist_ok=False)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('CHOPINFER_KERNEL_', 'CHOPINFER_MOE_PREFILL_', 'CHOPINFER_PREFILL_LEGACY_')):
            env.pop(key)
    env.update(CHOPINFER_KERNEL_GPU=str(args.device),CHOPINFER_MIMO_LATER_REPLAY=str(root/'later-layer-data'))
    results=[]
    for mode in MODES:
        for tokens in (1024,2048):
            for layer in (1,5,21,30,47):
                name=f'l{layer:02}-t{tokens}-{mode}';directory=logs/name;directory.mkdir()
                case_env=env|{'TMPDIR':str(root/'tmp'),'CHOPINFER_MIMO_PREFILL_LAYER':str(layer),'CHOPINFER_MIMO_PREFILL_TOKENS':str(tokens),'CHOPINFER_MIMO_LATER_MODE':mode,'CHOPINFER_KERNEL_RECEIPTS':str(directory/'receipts')}
                report=directory/'profile'
                command=[ready['nsys'],'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--output',str(report),ready['binary'],'--exact',SELECTOR,'--nocapture','--test-threads=1']
                with (directory/'gpu-telemetry.csv').open('x') as telemetry:
                    sampler=subprocess.Popen(['nvidia-smi','--id',str(args.device),
                        '--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                        '--format=csv','--loop-ms=200'],stdout=telemetry,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        run(command,case_env,directory/'run.log',min(deadline,time.monotonic()+90))
                    finally:
                        stop(sampler)
                text=(directory/'run.log').read_text()
                if '1 passed; 0 failed' not in text or 'skipping' in text.lower(): raise RuntimeError('test did not execute: '+name)
                receipts=[json.loads(p.read_text()) for p in (directory/'receipts').rglob('*.json')]
                receipts=[r for r in receipts if 'layer_graph_median_us' in r]
                if len(receipts)!=1 or receipts[0].get('ordinal')!=args.device or receipts[0].get('measurement_label')!='exclusive-device' or receipts[0].get('bitwise_graph_and_control') is not True:
                    raise RuntimeError('missing exact/exclusive receipt: '+name)
                database=directory/'profile.sqlite'
                run([ready['nsys'],'export','--type=sqlite','--output',str(database),str(report)+'.nsys-rep'],case_env,directory/'export.log',min(deadline,time.monotonic()+30))
                summary=trace_summary(database,args.device)
                record={'name':name,'receipt':receipts[0],'trace':summary,'telemetry_exit':sampler.returncode,'telemetry_log':str(directory/'gpu-telemetry.csv')}
                results.append(record)
                (logs/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(name,'gate charge us',summary['stages']['gate_up']['charge_us'],flush=True)
    if len(results)!=50: raise RuntimeError('incomplete later-layer control matrix')
    (logs/'success.json').write_text(json.dumps({'device':args.device,'cases':50,'binary_sha256':digest(ready['binary']),'results':str(logs/'results.json')},indent=2)+'\n')
    print(logs/'success.json',flush=True)

if __name__=='__main__': main()
