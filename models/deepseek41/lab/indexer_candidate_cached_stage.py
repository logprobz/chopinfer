#!/usr/bin/env python3
"""Seal the fused-publication qualification pair without submitting it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import indexer_candidate_cached_job as job
from indexer_prefill_tile_job import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','build','previous','baseline-run','binary'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];lab=repo/'models/deepseek41/lab'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo),'commit tested sources first'
    assert not (a.root/'pair.job').exists(),'refuse to overwrite a staged job'
    assert shutil.disk_usage(a.root.parent).free>150*(1<<30)
    assert (a.root/'cpu/final.exit').read_text().strip()=='0'
    subprocess.run(['sha256sum','--check','--quiet',str(a.previous/'stage.sha256')],check=True)
    a.root.mkdir(exist_ok=True);snap=a.root/'snapshot';snap.mkdir()
    for name in ('indexer_candidate_cached_job.py','test_indexer_candidate_cached_job.py','indexer_prefill_tile_job.py'):
        shutil.copy2(lab/name,snap/name)
    shutil.copy2(lab/'indexer_candidate_cached_pair.sh',snap/'run-pair.sh')
    shutil.copy2(lab/'indexer_candidate_cached_worker.sh',snap/'worker.sh')
    shutil.copy2(a.binary,snap/'prefill-tile-tests')
    old=a.previous/'snapshot'
    for name in ('indexer_prefill_probe.cubin','probe_sm120.cubin','position_mma_tile_sm120.cubin',
                 'position_mma_prefill_tile_sm120.cubin','captures.json'):
        shutil.copy2(old/name,snap/name)
    shutil.copy2(old/'candidate-modules.json',snap/'tile-modules.json')
    new='position_candidates_cached_sm120.cubin'
    shutil.copy2(a.build/new,snap/new)
    declarations=json.loads((a.build/'candidate-modules.json').read_text())
    assert len(declarations)==1 and len(declarations[0]['symbols'])==1
    assert digest(snap/new)==declarations[0]['sha256']
    shutil.copy2(a.build/'candidate-modules.json',snap/'candidate-modules.json')
    previous=json.loads((a.previous/'stage.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    source=a.root/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(source),head],cwd=repo,check=True)
    baseline_paths=[p for pattern in ('gpu-*.log','memcheck-*.log','sanitizer-*.log') for p in a.baseline_run.glob(pattern)]
    baseline_paths.extend(p for i in (0,1) for p in (a.baseline_run/f'receipts-{i}').rglob('*.json'))
    stage=dict(head=head,profiles=previous['profiles'],job_timeout_seconds=880,worker_timeout_seconds=820,
               source_sha256=digest(source),cases=198,memcheck_cases=36,timing_pairs=60,raw_samples=360,
               input_scope='Two sealed layer-20 captures, four synthetic chunk offsets and direct poisoned score fixtures',
               baseline_run=str(a.baseline_run.resolve()),
               baseline_sha256={str(p.relative_to(a.baseline_run)):digest(p) for p in baseline_paths},
               serving_option='lab only; fused candidate publication is not routed',sha256={})
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
    cpus=sorted(os.sched_getaffinity(0))[:4]
    assert len(cpus)==4
    with (a.root/'cpu/dry-run.log').open('w') as log:
        dry=subprocess.run(['taskset','-c',','.join(map(str,cpus)),'bash',str(a.root/'pair.job'),'--dry-run'],
                           stdout=log,stderr=subprocess.STDOUT)
    (a.root/'cpu/dry-run.exit').write_text(str(dry.returncode)+'\n')
    assert dry.returncode==0
    print(json.dumps(dict(path=str(a.root/'pair.job'),sha256=digest(a.root/'pair.job'),staged=True,queued=False),indent=2))


if __name__=='__main__':main()
