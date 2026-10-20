"""Seal a pair job only after the real captured operands pass exact replay."""
import argparse
import json
from pathlib import Path
import shutil
import shlex
import sys
import run


def freeze(args):
    require=run.require
    require(args.root.is_absolute() and not args.root.exists(),'stage must be a new absolute directory')
    verdict=json.loads((args.capture_run/'verdict.json').read_text())
    require(verdict.get('exit_code')==0 and verdict.get('replay')==dict(records=24,poison_replays=48,bitwise=True),'capture has not passed exact GPU replay')
    location=json.loads((args.capture_run/'capture-location.json').read_text())
    capture=Path(location['root']);index=json.loads((capture/'corpus.json').read_text())
    require(index['complete'] and index['records']==24 and index['nonce']==location['nonce'] and index['snapshot_sha256']==location['snapshot_sha256'],'capture identity differs')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'attention_capture'))
    import corpus
    for frame in index['captures']:
        require(corpus.digest(capture/frame['file'])==frame['sha256'],'capture frame changed')
        value=json.loads((capture/frame['file']).read_text())
        checked=[corpus.validate_record(capture,record,frame['rows']) for record in value['records']]
        require(checked==frame['records'],'captured record metadata changed')
    for rows in [4,32]:run.cases(index,rows)
    args.root.mkdir(parents=True)
    for name in ['scripts','snapshot','snapshot/cubins']:(args.root/name).mkdir()
    for file in ['run.py','test_run.py','README.md']:
        shutil.copyfile(Path(__file__).parent/file,args.root/'scripts'/file)
    shutil.copyfile(args.binary,args.root/'snapshot/attention-pair-tests')
    (args.root/'snapshot/attention-pair-tests').chmod(0o755)
    manifest=json.loads((args.cubins/'attention-modules.json').read_text())
    shutil.copyfile(args.cubins/'attention-modules.json',args.root/'snapshot/cubins/attention-modules.json')
    for module in manifest['modules']:
        name=module['name']+'.cubin';source=args.cubins/name
        require(run.digest(source)==module['sha256'],'cubin differs from audit')
        shutil.copyfile(source,args.root/'snapshot/cubins'/name)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'prefill_attention_resources.py',args.root/'scripts/resources.py')
    requests=[]
    for module in manifest['modules']:
        for resource in module.get('resources',[]):
            symbol=resource['symbol']
            if '_verify' in symbol and ('_cached_' in symbol or '_cache_decode_control0_' in symbol):
                requests.append(dict(resource,module=module['name'],dynamic_shared_bytes=0 if '_cache_decode_' in symbol else 73984))
    require(len(requests)==33,'expected 24 grouped consumers, six default consumers and three producers')
    run.write(args.root/'resource-requests.json',requests)
    files={str(p.relative_to(args.root)):run.digest(p) for p in sorted(args.root.rglob('*')) if p.is_file()}
    inputs={str(p.relative_to(capture)):run.digest(p) for p in sorted(capture.rglob('*')) if p.is_file() and (p.suffix=='.bin' or p.name in ['corpus.json','capture.json'])}
    require(all(not p.is_symlink() for p in capture.rglob('*')),'capture contains symlinks')
    config=dict(schema=1,capture_root=str(capture),capture_run=str(args.capture_run),files=files,captures=inputs,
                sanitizer=str(args.sanitizer),sanitizer_sha256=run.digest(args.sanitizer))
    run.write(args.root/'staged.json',config)
    sealed=[*files,'staged.json']
    (args.root/'SHA256SUMS').write_text(''.join(f'{run.digest(args.root/name)}  {name}\n' for name in sorted(sealed)))
    for path in args.root.rglob('*'):path.chmod(path.stat().st_mode&~0o022)
    args.root.chmod(args.root.stat().st_mode&~0o022)
    digest=run.digest(args.root/'SHA256SUMS')
    job=f'''# timeout 1140
# gpus 2
set -euo pipefail
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
pair_root={shlex.quote(str(args.root))}
(
    cd "$pair_root"
    printf '%s  SHA256SUMS\\n' '{digest}' | sha256sum --check --quiet -
    sha256sum --check --quiet SHA256SUMS
)
mode=${{1:-run}}
case "$mode" in run|--dry-run) ;; *) exit 2;; esac
output="$pair_root/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'output=%s\\n' "$output"
if [ "$mode" = --dry-run ]; then
    exec python3 -B "$pair_root/scripts/run.py" --root "$pair_root" --output "$output" --dry-run
fi
exec timeout --signal=TERM --kill-after=25s 1105s python3 -B "$pair_root/scripts/run.py" --root "$pair_root" --output "$output"
'''
    (args.root/'pair.job').write_text(job)
    print(json.dumps(dict(job=str(args.root/'pair.job'),sha256=run.digest(args.root/'pair.job'))))


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','capture-run','binary','cubins','sanitizer']:parser.add_argument('--'+name,type=Path,required=True)
    freeze(parser.parse_args())

if __name__=='__main__':main()
