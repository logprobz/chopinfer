"""Run one bounded TP4 DSpark capture, then replay its attention boundaries."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import corpus

MODEL='DeepSeek-V4.1-Flash'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def command(root,config,port):
    return [str(root/'install/bin/chopinfer'),'serve','--bundle',config['package'],'--checkpoint',config['checkpoint'],
            '--profile','tp4-dspark','--listen',f'127.0.0.1:{port}','--served-model-name',MODEL,
            '--prefix-cache','off','--max-sequences','32','--max-batch-tokens','8192','--prefill-chunk-tokens','8192',
            '--drain-timeout-ms','5000']


def response(value,body):
    count=len(body['prompt']);length=len(body['prompt'][0]);tokens=body['max_tokens']
    corpus.require(value['usage']['prompt_tokens']==count*length and value['usage']['completion_tokens']==count*tokens,'response token counts')
    corpus.require(value['usage']['prompt_tokens_details']['cached_tokens']==0,'request restored a prefix')
    choices=value['choices'];corpus.require(len(choices)==count and {c['index'] for c in choices}==set(range(count)),'response choice coverage')
    for choice in choices:
        corpus.require(choice['finish_reason']=='length' and len(choice['token_ids'])==tokens,'incomplete generation')
        if 'prompt_token_ids' in choice:corpus.require(choice['prompt_token_ids']==body['prompt'][choice['index']],'response prompt IDs differ')
    return dict(prompt_tokens=count*length,completion_tokens=count*tokens,cached_tokens=0)


def preflight(root,config,output):
    corpus.require(root.is_absolute() and not root.is_symlink(),'stage root must be absolute and regular')
    for name,sha in config['files'].items():
        path=root/name
        corpus.require(path.is_file() and not path.is_symlink() and not path.stat().st_mode&0o022 and corpus.digest(path)==sha,'frozen input differs: '+name)
    manifest=json.loads(Path(config['package']).joinpath('manifest.json').read_text())
    corpus.require(corpus.digest(Path(config['package'])/'manifest.json')==config['package_identity'],'package identity differs')
    corpus.require(manifest['batch_limits']['maximum_prefill_tokens']>=8192,'package cannot admit an 8192-row chunk')
    argv=command(root,config,8080)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PATH=str(root/'install/bin')+os.pathsep+os.environ['PATH'])
    result=subprocess.run(argv+['--list-model-options'],env=env,capture_output=True,text=True,timeout=90)
    write(output/'argv.json',dict(argv=argv,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    corpus.require(result.returncode==0 and json.loads(result.stdout)['schema']=='chop.model-options.v1','frozen server argv rejected')
    env.update(CHOPINFER_ATTENTION_CUBINS=str(root/'cubins'))
    for selector,count in [('frozen_modules',1),('verify_cache::cpu_verify_cache_preserves_partitions_and_all_eight_splits',1)]:
        args=[str(root/'snapshot/attention-pair-tests'),selector,'--exact','--nocapture']
        if selector=='frozen_modules':args.append('--ignored')
        result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=60)
        label=selector.split('::')[-1]
        (output/f'{label}.log').write_text(result.stdout+result.stderr)
        corpus.require(result.returncode==0,'frozen CPU check failed')
        corpus.test_result(result.stdout+result.stderr,count)
    listed=subprocess.run([str(root/'snapshot/attention-pair-tests'),'--list'],capture_output=True,text=True,check=True,timeout=30).stdout
    for test in ['captured_replay::frozen_captured_attention','captured_replay::gpu_captured_attention_replay']:
        corpus.require(f'{test}: test' in listed,'missing frozen replay selector')
    return dict(argv=argv,cpu_only=True,checks=3)


def workflow(ops):
    result={'exit_code':1}
    error=None
    try:
        ops.preflight()
        ops.power('before')
        ops.journal_preflight()
        try:
            ops.start()
            ops.capture()
        finally:
            ops.stop()
        complete=ops.seal()
        result['replay']=ops.replay(complete)
        result['exit_code']=0
    except Exception as exc:
        error=exc;result['error']=str(exc)
    finally:
        try:
            ops.power('after')
            ops.journal()
        except Exception as exc:
            result['exit_code']=1;result['cleanup_error']=str(exc)
            if error is None:error=exc
        ops.verdict(result)
    if error is not None:raise error
    return result


class Operations:
    def __init__(self,root,config,output,seconds=780):
        self.root,self.config,self.output=root,config,output
        self.started=time.monotonic();self.deadline=self.started+seconds;self.epoch=time.time()
        self.process=None;self.nonce=secrets.token_hex(16);self.snapshot=corpus.digest(root/'staged.json')
        self.captures=Path(tempfile.mkdtemp(prefix='chopinfer-o2-verify-capture-',dir='/tmp'))
        self.server_log=output/'server.log';self.url=''
        write(output/'capture-location.json',dict(root=str(self.captures),nonce=self.nonce,snapshot_sha256=self.snapshot))
    def remaining(self,limit):
        left=self.deadline-time.monotonic()
        if left<=0:raise TimeoutError('capture work deadline reached')
        return min(left,limit)
    def preflight(self):
        import shutil
        corpus.require(shutil.disk_usage('/tmp').free>100*(1<<30),'capture requires 100 GiB free on root volume')
        corpus.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3','whole-machine allocation required')
        preflight(self.root,self.config,self.output)
    def power(self,edge):
        result=subprocess.run(['nvidia-smi','-i','0,1,2,3','--query-gpu=power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        (self.output/f'power-{edge}.log').write_text(result.stdout+result.stderr)
        corpus.require(result.returncode==0 and [float(x.strip()) for x in result.stdout.splitlines()]==[250.0]*4,'250 W cap required on all four GPUs')
    def journal_preflight(self):
        result=subprocess.run(['journalctl','-k','-n','1','--no-pager'],capture_output=True,text=True,timeout=15)
        (self.output/'journal-preflight.log').write_text(result.stdout+result.stderr)
        corpus.require(result.returncode==0 and result.stdout.strip() and not any(x in result.stdout+result.stderr for x in ['No entries','No journal files','not seeing messages']),'kernel journal unavailable')
    def request(self,route,body=None,timeout=150):
        data=None if body is None else json.dumps(body).encode()
        request=urllib.request.Request(self.url+route,data,{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=self.remaining(timeout)) as stream:return json.load(stream)
    def start(self):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        env=dict(os.environ,CHOPINFER_ATTENTION_CAPTURE_DIR=str(self.captures),CHOPINFER_ATTENTION_CAPTURE_NONCE=self.nonce,
                 CHOPINFER_ATTENTION_CAPTURE_SNAPSHOT_SHA256=self.snapshot,CHOPINFER_STEP_TRACE='1',PYTHONDONTWRITEBYTECODE='1')
        for name in list(env):
            if name.startswith('CHOPINFER_TAPS_') or name in ('CHOPINFER_MODEL_OPTIONS','CHOPINFER_KERNEL_GPU','CHOPINFER_KERNEL_GPUS','KERNELS_LOCK_UPDATE'):
                env.pop(name,None)
        argv=command(self.root,self.config,port);write(self.output/'command.json',argv)
        with self.server_log.open('w') as log:
            self.process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        until=time.monotonic()+self.remaining(240)
        while True:
            if self.process.poll() is not None:raise RuntimeError('capture server exited before readiness')
            try:
                health=self.request('/health',timeout=2);write(self.output/'health.json',health);break
            except OSError:
                if time.monotonic()>until:raise TimeoutError('capture server readiness timeout')
                time.sleep(.2)
    def capture(self):
        seeds=json.loads((self.root/'prompts.json').read_text())
        corpus.require(len(seeds)==8 and len({tuple(s) for s in seeds})==8 and all(s for s in seeds),'eight diverse prompt seeds required')
        for ordinal,(group,items,context,quota) in enumerate(corpus.GROUPS):
            arm=dict(group=group,items=items,context=context,quota=quota,nonce=self.nonce)
            temporary=self.captures/'armed.next';write(temporary,arm);temporary.replace(self.captures/'armed.json')
            inputs=[]
            for index in range(items):
                seed=seeds[(index+ordinal)%len(seeds)]
                inputs.append((seed*((context+len(seed)-1)//len(seed)))[:context])
            budget=256 if items==8 else 32
            body=dict(model=MODEL,prompt=inputs,n=1,max_tokens=budget,min_tokens=budget,ignore_eos=True,temperature=0,seed=0,
                      return_token_ids=True,stream=False,cache_salt=self.nonce+'-'+group)
            write(self.output/f'{group}-request.json',body)
            value=self.request('/v1/completions',body)
            write(self.output/f'{group}-response.json',dict(request=body,response=value,counts=response(value,body)))
            frames=list(self.captures.glob(group+'-*/capture.json'))
            corpus.require(len(frames)==quota,'capture did not reach requested group quota: '+group)
        (self.captures/'armed.json').unlink()
    def stop(self):
        if self.process is None:return
        with contextlib.suppress(ProcessLookupError):os.killpg(self.process.pid,signal.SIGTERM)
        try:self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):os.killpg(self.process.pid,signal.SIGKILL)
            self.process.wait(timeout=15)
        self.process=None
    def seal(self):
        result=corpus.seal(self.captures,self.nonce,self.snapshot,corpus.origins(self.server_log.read_text()))
        write(self.output/'corpus-summary.json',dict(path=str(self.captures/'corpus.json'),sha256=corpus.digest(self.captures/'corpus.json'),records=result['records']))
        return result
    def replay(self,complete):
        env=dict(os.environ,CHOPINFER_ATTENTION_CUBINS=str(self.root/'cubins'),CHOPINFER_ATTENTION_CAPTURE_ROOT=str(self.captures),
                 CHOPINFER_KERNEL_GPU='0',CHOPINFER_KERNEL_REQUIRE='1',CHOPINFER_KERNEL_RECEIPTS=str(self.captures/'receipts'))
        binary=str(self.root/'snapshot/attention-pair-tests')
        for selector,name,gpu in [('captured_replay::frozen_captured_attention','operand-cpu',False),('captured_replay::gpu_captured_attention_replay','replay',True)]:
            result=subprocess.run([binary,selector,'--ignored','--exact','--nocapture','--test-threads=1'],env=env,capture_output=True,text=True,timeout=self.remaining(120))
            text=result.stdout+result.stderr;(self.output/f'{name}.log').write_text(text)
            corpus.require(result.returncode==0,'captured operand replay failed')
            if gpu:proof=corpus.replay_result(text,complete)
            else:corpus.test_result(text,1)
        return proof
    def journal(self):
        import re
        result=subprocess.run(['journalctl','-k','--since',f'@{int(self.epoch)}','--no-pager'],capture_output=True,text=True,timeout=15)
        (self.output/'kernel.log').write_text(result.stdout+result.stderr)
        xids=[line for line in result.stdout.splitlines() if re.search(r'NVRM:.*\bXid\b',line,re.I)]
        write(self.output/'xids.json',xids)
        corpus.require(result.returncode==0 and not xids,'kernel journal failed or contains Xids')
    def verdict(self,result):
        result.update(elapsed_seconds=time.monotonic()-self.started,capture_root=str(self.captures),snapshot_sha256=self.snapshot)
        write(self.output/'verdict.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);config=json.loads((args.root/'staged.json').read_text())
    if args.dry_run:
        result=preflight(args.root,config,args.output);write(args.output/'verdict.json',dict(exit_code=0,gpu_launched=False,**result));return
    def interrupted(number,_frame):raise InterruptedError(f'capture interrupted by signal {number}')
    for number in (signal.SIGINT,signal.SIGTERM):signal.signal(number,interrupted)
    workflow(Operations(args.root,config,args.output))


if __name__=='__main__':main()
