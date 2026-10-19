"""Seal the direct-fold pair job from this checkout's verified build artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def stage(source, target, binary, compiled, output):
    source, target, binary, compiled, output = [p.resolve() for p in (source, target, binary, compiled, output)]
    assert not output.is_relative_to(source)
    assert binary.is_relative_to(target) and binary.is_file()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    snapshot = output/'snapshot'
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot/'bin').mkdir()
    (snapshot/'cubins').mkdir()
    shutil.copy2(binary, snapshot/'bin/fold-tests')
    family = source/'models/deepseek41/bundle/kernels.lock.d'
    locks = list(family.glob('mxfp8_fold*.lock')) + [source/'crates/kernel-lab/kernels.lock',
        source/'models/deepseek41/lab/kernels.lock.d/mxfp8_fold_direct.lock']
    # The dependency module may live in another named bundle family.
    locks += [p for p in family.glob('*.lock') if p not in locks]
    required = {}
    for lock in locks:
        for record in json.loads(lock.read_text()):
            name = record['name']
            if name.startswith('mxfp8_fold') or name in ('probe_sm120', 'dependency_probes_sm120'):
                assert name not in required or required[name] == record['cubin_sha256']
                required[name] = record['cubin_sha256']
    assert 'probe_sm120' in required and 'mxfp8_fold_direct_sm120' in required
    for name, expected in required.items():
        candidates = [compiled/f'{name}.cubin', *target.glob(f'*/build/*/out/{name}.cubin')]
        candidates = [p for p in candidates if p.is_file() and digest(p) == expected]
        assert candidates, f'missing locked build artifact {name}'
        shutil.copy2(candidates[0], snapshot/'cubins'/f'{name}.cubin')
    tools = source/'models/deepseek41/lab/tools'
    for name in ('fold_bandwidth_report.py', 'fold_stream_report.py', 'fold_bandwidth_job.py',
                 'test_fold_bandwidth_report.py'):
        shutil.copy2(tools/name, snapshot/name)
    shutil.copy2(compiled/'mxfp8_fold_direct_sm120.identity.json', snapshot/'resources.json')
    (snapshot/'source-revision.txt').write_text(revision+'\n')
    (snapshot/'module-digests.json').write_text(json.dumps(required, indent=2)+'\n')
    manifest = ''.join(f'{digest(p)}  {p.relative_to(snapshot)}\n' for p in sorted(snapshot.rglob('*')) if p.is_file())
    (snapshot/'SHA256SUMS').write_text(manifest)
    for path in [snapshot, *snapshot.rglob('*')]:
        path.chmod(path.stat().st_mode & ~0o022)
    template = (tools/'fold_bandwidth.job.in').read_text()
    for name, value in dict(SNAPSHOT=str(snapshot), ROOT=str(output),
                            MANIFEST_SHA=digest(snapshot/'SHA256SUMS')).items():
        assert "'" not in value and '\n' not in value
        template = template.replace('@'+name+'@', value)
    assert template.splitlines()[:2] == ['# timeout 880', '# gpus 2']
    job = output/'fold-bandwidth.job'
    with job.open('x') as stream:
        stream.write(template)
    job.chmod(job.stat().st_mode & ~0o022)
    print(json.dumps(dict(job=str(job), sha256=digest(job), revision=revision, modules=len(required))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ('source', 'target', 'binary', 'compiled', 'output'):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    stage(args.source, args.target, args.binary, args.compiled, args.output)
