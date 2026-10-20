#!/usr/bin/env python3
"""Freeze the tested prefill scorer candidate without submitting a GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import indexer_prefill_tile_job as job


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','build','previous','captures','profiles','binary'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];lab=repo/'models/deepseek41/lab'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo),'commit tested sources first'
    assert not (a.root/'pair.job').exists(),'refuse to overwrite a staged job'
    assert shutil.disk_usage(a.root.parent).free>150*(1<<30)
    assert (a.root/'cpu/final.exit').read_text().strip()=='0'
    a.root.mkdir(exist_ok=True);snap=a.root/'snapshot';snap.mkdir()
    for name in ('indexer_prefill_tile_job.py','test_indexer_prefill_tile_job.py'):
        shutil.copy2(lab/name,snap/name)
    shutil.copy2(lab/'indexer_prefill_tile_pair.sh',snap/'run-pair.sh')
    shutil.copy2(lab/'indexer_prefill_tile_worker.sh',snap/'worker.sh')
    shutil.copy2(a.binary,snap/'prefill-tile-tests')
    old=a.previous/'snapshot'
    subprocess.run(['sha256sum','--check','--quiet',str(a.previous/'stage.sha256')],check=True)
    for name in ('indexer_prefill_probe.cubin','probe_sm120.cubin','position_mma_tile_sm120.cubin'):
        shutil.copy2(old/name,snap/name)
    shutil.copy2(a.build/'position_mma_prefill_tile_sm120.cubin',snap/'position_mma_prefill_tile_sm120.cubin')
    declarations=json.loads((a.build/'candidate-modules.json').read_text())
    assert len(declarations)==1 and len(declarations[0]['symbols'])==8
    assert digest(snap/'position_mma_prefill_tile_sm120.cubin')==declarations[0]['sha256']
    shutil.copy2(a.build/'candidate-modules.json',snap/'candidate-modules.json')
    captures=[]
    for step,context in (('step0001',8192),('step0009',32768)):
        for layer in (2,8,14,20,24):
            directory=a.captures/step/f'indexer-L{layer:02}'
            meta=json.loads((directory/'operands.json').read_text())
            assert meta['context']==context and meta['layer']==layer and len(meta['positions'])==32
            files=['operands.json','query.bf16','weights.f32','keys.e2m1','scales.ue8m0','row_tags.u64','selected.i32','status.u32',
                   '../capture.json','../completion.json','../items.json']
            if layer==24:files+=['candidates.i32','candidate_tags.u64','candidate_visible.i32','candidate_status.u32']
            captures.append(dict(path=str(directory.resolve()),context=context,layer=layer,rows=32,
                                 sha256={name:digest(directory/name) for name in files}))
    (snap/'captures.json').write_text(json.dumps(dict(captures=captures),indent=2)+'\n')
    profiles=json.loads(a.profiles.read_text())['records']
    profiles=[{k:profiles[key][k] for k in ('path','sha256')} for key in ('prefill/0/0','prefill/3/0')]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    source=a.root/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(source),head],cwd=repo,check=True)
    stage=dict(head=head,profiles=profiles,job_timeout_seconds=880,worker_timeout_seconds=820,
               source_sha256=digest(source),cases=94,memcheck_cases=20,timing_pairs=176,raw_samples=1056,
               input_scope='10 sealed captures of 32 sampled prefill rows; full 8192-row matrices use seeded synthetic operands',
               serving_option='text_config.indexer_prefill_tile=true (unqualified; default false)',
               offsets=[0,8192,16384,24576],sha256={})
    for path in snap.rglob('*'):
        path.chmod(path.stat().st_mode&~0o022)
        if path.is_file():stage['sha256'][str(path.relative_to(a.root))]=digest(path)
    (a.root/'stage.json').write_text(json.dumps(stage,indent=2)+'\n')
    (a.root/'pair.job').write_text(f'# timeout 880\n# gpus 2\nexec /usr/bin/bash {snap.resolve()}/run-pair.sh "$@"\n')
    static=[*sorted(p for p in snap.rglob('*') if p.is_file()),a.root/'stage.json',a.root/'pair.job',source]
    (a.root/'stage.sha256').write_text(''.join(f'{digest(path)}  {path.resolve()}\n' for path in static))
    (a.root/'pair.job.sha256').write_text(f'{digest(a.root/"pair.job")}  {a.root.resolve()}/pair.job\n')
    for path in [snap,*static,a.root/'stage.sha256',a.root/'pair.job.sha256']:
        path.chmod(path.stat().st_mode&~0o022)
    validated=job.validate(a.root/'stage.json')
    (a.root/'cpu/sealed-preflight.json').write_text(json.dumps(validated,indent=2)+'\n')
    # The outer build shell is admitted by heavy.sh; the frozen job itself
    # deliberately runs on four of that slot's CPUs and never invokes heavy.
    cpus=sorted(os.sched_getaffinity(0))[:4]
    assert len(cpus)==4
    command=['taskset','-c',','.join(map(str,cpus)),'bash',str(a.root/'pair.job'),'--dry-run']
    with (a.root/'cpu/dry-run.log').open('w') as log:
        dry=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    (a.root/'cpu/dry-run.exit').write_text(str(dry.returncode)+'\n')
    assert dry.returncode==0
    print(json.dumps(dict(path=str(a.root/'pair.job'),sha256=digest(a.root/'pair.job'),staged=True,queued=False),indent=2))


if __name__=='__main__':main()
