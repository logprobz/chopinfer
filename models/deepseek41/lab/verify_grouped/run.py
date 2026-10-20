"""Run authenticated verify attention controls before paired timing."""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

GROUPS=[0,2,4,8]
MODES=['original','changed_query','one_inactive','all_inactive','invalid_batch','invalid_page','holes','all_holes']
TIMINGS=['cold_sources','cold_attention','warm_attention']
PHASES=[('memcheck',360),('check',180),('time',360)]
CANCELLED=False

def cancel(_signum,_frame):
    global CANCELLED
    CANCELLED=True


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def test_result(text):
    require(re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)==[('ok','1','0','0')], 'one executed test required')
    require('GPU work skipped' not in text,'GPU execution skipped')


def markers(text,tag):return [json.loads(line.split(tag+' ',1)[1]) for line in text.splitlines() if tag+' ' in line]


def cases(index,rows):
    result={Path(f['file']).parent.name+f'-L{r["layer"]:02}':r for f in index['captures'] if f['rows']==rows for r in f['records']}
    require(len(result)==12,'each row shard needs twelve captured records')
    return result


def parse(text,index,rows,phase):
    test_result(text);wanted=cases(index,rows)
    completion=markers(text,'VERIFY_GROUPED_COMPLETE')
    require(completion==[dict(cases=12,timed=phase=='time',memcheck=phase=='memcheck')],'phase completion mismatch')
    if phase=='time':
        entries=markers(text,'VERIFY_GROUPED_TIMING')
        expected={(key,group,mode) for key in wanted for group in GROUPS for mode in TIMINGS}
        require(len(entries)==len(expected) and {(r['id'],r['group'],r['mode']) for r in entries}==expected,'paired timing coverage')
        for r in entries:
            require(r['rows']==rows and all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) and r[k]>0 for k in ('old_us','new_us','paired_ratio')),'invalid timing')
        return dict(cases=12,timings=entries)
    entries=markers(text,'VERIFY_GROUPED_OK')
    require(len(entries)==12 and {r['id'] for r in entries}==wanted.keys(),'correctness identity coverage')
    for r in entries:
        source=wanted[r['id']]
        require(r['rows']==rows and r['groups']==GROUPS and r['modes']==MODES,'control matrix differs')
        require(r['poison_replays']==64 and r['partitions']==4 and r['graph_replays']==(0 if phase=='memcheck' else 8) and r['negative'] is True,'control execution count differs')
        require(all(r[k]==source[k] for k in ('partial_sha256','output_sha256')),'captured expected bits differ')
    clocks=markers(text,'VERIFY_GROUPED_CLOCK')
    require(len(clocks)==48 and {(r['id'],r['group']) for r in clocks}=={(key,group) for key in wanted for group in GROUPS},'clock coverage')
    for r in clocks:
        heads=r['group'] or (1 if rows<=4 else 8)
        cycles=r['cycles'];units=2 if r['id'].endswith('-L00') else 10
        require(r['rows']==rows and r['bitwise'] is True and r['ctas']==rows*16//heads*8,'clock CTA count')
        require(len(cycles)==8 and all(isinstance(x,int) and x>=0 for x in cycles) and cycles[6]>0 and sum(cycles[:6])<=cycles[6] and cycles[7]==rows*16//heads*units,'clock stamp totals')
    if phase=='memcheck':
        require(len(re.findall(r'ERROR SUMMARY: 0 errors',text))==1 and not re.search(r'ERROR SUMMARY: [1-9]',text),'memcheck summary missing or nonzero')
    return dict(cases=12,poison_replays=768,partitions=48,graph_replays=0 if phase=='memcheck' else 96,negative_controls=12)


def workflow(ops):
    result=dict(exit_code=1);error=None
    try:
        ops.preflight();ops.power('before');ops.journal_preflight()
        result['resources']=ops.resources()
        result['phases']={}
        for phase,limit in PHASES:result['phases'][phase]=ops.phase(phase,limit)
        result['exit_code']=0
    except Exception as exc:error=exc;result['error']=str(exc)
    finally:
        try:ops.power('after');ops.journal()
        except Exception as exc:
            result['exit_code']=1;result['cleanup_error']=str(exc)
            if error is None:error=exc
        ops.verdict(result)
    if error is not None:raise error
    return result


class Operations:
    def __init__(self,root,output):
        self.root,self.output=root,output
        self.config=json.loads((root/'staged.json').read_text())
        self.capture=Path(self.config['capture_root']);self.index=json.loads((self.capture/'corpus.json').read_text())
        self.started=time.monotonic();self.epoch=time.time();self.deadline=self.started+1080
    def remaining(self,limit):
        left=self.deadline-time.monotonic();require(not CANCELLED and left>0,'work deadline exhausted or cancelled');return min(limit,left)
    def env(self,rows,device=None):
        env=dict(os.environ,CHOPINFER_ATTENTION_CAPTURE_ROOT=str(self.capture),CHOPINFER_ATTENTION_CUBINS=str(self.root/'snapshot/cubins'),CHOPINFER_VERIFY_ROWS=str(rows))
        env.pop('CHOPINFER_KERNEL_GPUS',None);env.pop('CHOPINFER_KERNEL_GPU',None)
        if device is not None:env['CHOPINFER_KERNEL_GPU']=str(device)
        return env
    def command(self,phase):
        binary=str(self.root/'snapshot/attention-pair-tests')
        selector=f'verify_grouped::gpu_verify_grouped_{phase}'
        command=[binary,selector,'--exact','--ignored','--nocapture']
        if phase=='memcheck':command=[self.config['sanitizer'],'--tool','memcheck','--error-exitcode','86']+command
        return command
    def call(self,argv,env,log,limit):
        # Each child owns its output file and is synchronously collected, even
        # when the other shard fails. No GPU process survives the phase timeout.
        with log.open('w') as output:
            process=subprocess.Popen(argv,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+self.remaining(limit)
                while True:
                    left=deadline-time.monotonic()
                    if CANCELLED or left<=0:raise subprocess.TimeoutExpired(argv,limit)
                    try:code=process.wait(timeout=min(1,left));break
                    except subprocess.TimeoutExpired:pass
            except BaseException:
                with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                finally:
                    with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=5)
                raise
        require(code==0,f'command exited {code}: {log.name}')
        return log.read_text()
    def preflight(self):
        require(digest(Path(self.config['sanitizer']))==self.config['sanitizer_sha256'],'sanitizer binary changed')
        for file,sha in self.config['files'].items():
            path=self.root/file
            require(path.is_file() and not path.is_symlink() and not path.stat().st_mode&0o022 and digest(path)==sha,'frozen input differs: '+file)
        for file,sha in self.config['captures'].items():
            path=self.capture/file
            require(path.is_file() and not path.is_symlink() and digest(path)==sha,'capture input differs: '+file)
        require(self.index['complete'] is True and self.index['records']==24,'incomplete operand corpus')
        binary=str(self.root/'snapshot/attention-pair-tests')
        for rows in [4,32]:
            log=self.output/f'cpu-{rows}.log'
            text=self.call([binary,'verify_grouped::frozen_verify_grouped','--exact','--ignored','--nocapture'],self.env(rows),log,90)
            test_result(text);require(text.count('VERIFY_GROUPED_CPU_OK records=12 modes=8 groups=4')==1,'CPU shard matrix incomplete')
        text=self.call([binary,'frozen_modules','--exact','--ignored','--nocapture'],self.env(4),self.output/'modules.log',90)
        test_result(text)
        listed=subprocess.run([binary,'--list'],capture_output=True,text=True,check=True,timeout=15).stdout
        for phase,_ in PHASES:require(f'verify_grouped::gpu_verify_grouped_{phase}: test' in listed,'missing GPU selector')
    def resources(self):
        code="import json,sys;from pathlib import Path;import resources;print(json.dumps(resources.collect(Path(sys.argv[1]))))"
        text=self.call([sys.executable,'-c',code,str(self.root)],dict(os.environ,PYTHONPATH=str(self.root/'scripts')),self.output/'resources.log',60)
        # CUDA diagnostics are mixed with the final JSON in this log.
        result=json.loads(text.splitlines()[-1]);wanted=json.loads((self.root/'resource-requests.json').read_text())
        require(len(result['ranks'])==2 and [r['rank'] for r in result['ranks']]==[0,1],'resource rank coverage')
        for rank in result['ranks']:
            require(len(rank['functions'])==len(wanted) and {r['symbol'] for r in rank['functions']}=={r['symbol'] for r in wanted},'resource symbol coverage')
        write(self.output/'resources.json',result);return dict(ranks=2,symbols=len(wanted))
    def phase(self,phase,limit):
        def worker(device,rows):
            text=self.call(self.command(phase),self.env(rows,device),self.output/f'{phase}-{rows}.log',limit)
            return str(rows),parse(text,self.index,rows,phase)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,*pair) for pair in enumerate([4,32])]
            results=[];errors=[]
            for future in futures:
                try:results.append(future.result())
                except Exception as exc:errors.append(exc)
            if errors:raise errors[0]
        result=dict(results);write(self.output/f'{phase}.json',result);return result
    def power(self,edge):
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        require(len(devices)==2 and all(re.fullmatch(r'\d+',d) for d in devices),'runner must allocate a GPU pair')
        text=self.call(['nvidia-smi','-i',','.join(devices),'--query-gpu=power.limit','--format=csv,noheader,nounits'],os.environ,self.output/f'power-{edge}.log',15)
        require([float(line.strip()) for line in text.splitlines()]==[250.0]*2,'250 W cap required')
    def journal_preflight(self):
        text=self.call(['journalctl','-k','-n','1','--no-pager'],os.environ,self.output/'journal-preflight.log',15)
        require(text.strip() and not any(s in text for s in ['No entries','No journal files','not seeing messages']),'kernel journal unavailable')
    def journal(self):
        text=self.call(['journalctl','-k','--since',f'@{self.epoch:.0f}','--no-pager'],os.environ,self.output/'journal.log',15)
        require(not re.search(r'NVRM: Xid|Xid \(',text),'Xid during pair')
    def verdict(self,result):
        result['elapsed_seconds']=time.monotonic()-self.started;write(self.output/'verdict.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    args.output.mkdir(parents=True,exist_ok=False);ops=Operations(args.root,args.output)
    if args.dry_run:
        ops.preflight();ops.verdict(dict(exit_code=0,cpu_only=True,gpu_launched=False,checks=3))
    else:workflow(ops)

if __name__=='__main__':main()
