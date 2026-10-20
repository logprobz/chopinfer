"""Freeze a diagnostic install and replay executable without queuing GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_proof(repo,source,revision,hook):
    patch=json.loads((source/'attention-capture-patch.json').read_text())
    changed={r['file']:r for r in patch['changes']}
    if set(changed)!={'models/deepseek41/bundle/src/instance/execute.rs','models/deepseek41/bundle/src/sm120/mod.rs'}:raise ValueError('wrong patched modules')
    result={}
    process=subprocess.Popen(['git','-C',str(repo),'archive','--format=tar',revision],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for entry in archive:
                if entry.isdir():continue
                path=source/entry.name
                if entry.issym():
                    if not path.is_symlink() or os.readlink(path)!=entry.linkname:raise ValueError('source symlink differs')
                    continue
                if not entry.isfile() or path.is_symlink():raise ValueError('unexpected source file')
                with archive.extractfile(entry) as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
                expected=sha
                if entry.name in changed:
                    if changed[entry.name]['before_sha256']!=sha:raise ValueError('patch does not start at the exact base')
                    expected=changed[entry.name]['after_sha256']
                if digest(path)!=expected:raise ValueError('undeclared source change: '+entry.name)
                result[entry.name]=sha
        if process.wait():raise ValueError('git archive failed')
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate()
        process.wait()
    actual=source/'models/deepseek41/bundle/src/instance/execute/attention_capture.rs'
    if digest(actual)!=digest(hook):raise ValueError('compiled observer differs from frozen observer')
    helper=source/'models/deepseek41/bundle/src/sm120/attention_capture.rs'
    if digest(helper)!=digest(hook.with_name('sm120.rs')):raise ValueError('compiled SM120 metadata differs')
    return dict(base_revision=revision,patch=patch,observer_sha256=digest(hook),sm120_sha256=digest(helper),base_files=result)


def freeze(args):
    for value in vars(args).values():
        if isinstance(value,Path) and not value.is_absolute():raise ValueError('staging paths must be absolute')
    if args.root.is_relative_to(args.repo):raise ValueError('capture artifacts belong outside the repository')
    proof=source_proof(args.repo,args.source,args.base,Path(__file__).with_name('hook.rs'))
    args.root.mkdir(exist_ok=False)
    for directory in ('install/bin','install/lib','scripts','snapshot','cubins'):(args.root/directory).mkdir(parents=True)
    for target,source in {
        'install/bin/chopinfer':args.target/'release/chopinfer',
        'install/bin/chopinfer-model-deepseek41':args.target/'release/chopinfer-model-deepseek41',
        'install/lib/libbundle_deepseek41.so':args.target/'release/libbundle_deepseek41.so',
        'snapshot/attention-pair-tests':args.test_binary,
    }.items():
        if not source.is_file() or source.is_symlink():raise ValueError('build artifact absent')
        shutil.copy2(source,args.root/target)
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ('.py','.rs','.md'):shutil.copy2(path,args.root/'scripts'/path.name)
    for path in args.cubins.iterdir():
        if path.is_file() and path.suffix in ('.cubin','.json'):shutil.copy2(path,args.root/'cubins'/path.name)
    manifest=json.loads((args.root/'cubins/attention-modules.json').read_text())
    for record in manifest['modules']:
        if digest(args.root/'cubins'/(record['name']+'.cubin'))!=record['sha256']:raise ValueError('replay cubin differs')
    for family in ('latent_sources_scheduled','latent_decode_cache'):
        records=json.loads((args.source/f'models/deepseek41/bundle/kernels.lock.d/{family}.lock').read_text())
        for record in records:
            if digest(args.root/'cubins'/(record['name']+'.cubin'))!=record['cubin_sha256']:raise ValueError('replay cubin differs from served lock')
    prompts=json.loads(args.prompts.read_text())['decode-c8']
    if len(prompts)!=8 or len({tuple(p) for p in prompts})!=8:raise ValueError('recorded diverse prompt coverage')
    (args.root/'prompts.json').write_text(json.dumps(prompts)+'\n')
    (args.root/'source.json').write_text(json.dumps(proof,indent=2)+'\n')
    argv=[str(args.root/'install/bin/chopinfer-model-deepseek41'),'prepare','--direct','--model',str(args.checkpoint),
          '--tensor-parallel-size','4','--cache-dir',str(args.root/'cache')]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    (args.root/'prepare.stdout.log').write_text(result.stdout);(args.root/'prepare.stderr.log').write_text(result.stderr)
    result.check_returncode();prepared=json.loads(result.stdout)
    package=Path(prepared['bundle'])
    if not package.is_relative_to(args.root/'cache') or prepared['profile']['name']!='tp4-dspark':raise ValueError('unexpected prepared profile/package')
    files=[path for path in args.root.rglob('*') if path.is_file()]
    if any(path.is_symlink() for path in files):raise ValueError('snapshot contains a symlink')
    config=dict(base_revision=args.base,package=str(package),package_identity=digest(package/'manifest.json'),checkpoint=str(args.checkpoint),
                prepared=prepared,prompt_provenance=dict(file=str(args.prompts),sha256=digest(args.prompts)),
                files={str(path.relative_to(args.root)):digest(path) for path in sorted(files)})
    (args.root/'staged.json').write_text(json.dumps(config,indent=2)+'\n')
    files.append(args.root/'staged.json')
    sums=args.root/'SHA256SUMS';sums.write_text(''.join(f'{digest(path)}  {path.relative_to(args.root)}\n' for path in sorted(files)))
    template='''# timeout 840
set -euo pipefail
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
capture_root=@ROOT@
(
    cd "$capture_root"
    printf '%s  SHA256SUMS\\n' '@SHA@' | sha256sum --check --quiet -
    sha256sum --check --quiet SHA256SUMS
)
mode=${1:-run}
case "$mode" in run|--dry-run) ;; *) exit 2;; esac
output="$capture_root/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'output=%s\\n' "$output"
if [ "$mode" = --dry-run ]; then
    exec python3 -B "$capture_root/scripts/run.py" --root "$capture_root" --output "$output" --dry-run
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec timeout --signal=TERM --kill-after=25s 805s python3 -B "$capture_root/scripts/run.py" --root "$capture_root" --output "$output"
'''
    job=args.root/'capture.job';job.write_text(template.replace('@ROOT@',shlex.quote(str(args.root))).replace('@SHA@',digest(sums)))
    job.with_suffix('.job.sha256').write_text(digest(job)+'  capture.job\n')
    for path in (args.root,*args.root.rglob('*')):path.chmod(path.stat().st_mode&~0o022)
    print(json.dumps(dict(job=str(job),sha256=digest(job),queued=False,dry_run_pending=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','source','target','root','checkpoint','test-binary','cubins','prompts'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--base',required=True)
    freeze(parser.parse_args())
