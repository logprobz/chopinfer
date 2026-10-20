"""Freeze a prepared installation and write two whole-machine jobs, without queueing."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time

from run import digest, write

HERE = Path(__file__).resolve().parent


def configure(root):
    root = root.resolve()
    revision = (root / 'NATIVE_REVISION').read_text().strip()
    assert (root / 'install/src/.commit').read_text().strip() == revision
    checkpoint = Path((root / 'install/CHECKPOINT').read_text().strip()).resolve()
    package = Path((root / 'install/PACKAGE').read_text().strip()).resolve()
    assert package.is_relative_to(root / 'install')
    write(root / 'config.json', dict(source=str(root / 'install/src'),
          binary=str(root / 'install/bin/chopinfer'), checkpoint=str(checkpoint),
          package=str(package), native_revision=revision))
    (root / 'harness').mkdir(exist_ok=True)
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.md'):
            shutil.copyfile(path, root / 'harness' / path.name)


def files(root):
    paths = [root / 'config.json', root / 'cpu-validation.json', root / 'timeout-audit.json']
    for directory in ('install', 'harness'):
        for path in (root / directory).rglob('*'):
            assert not path.is_symlink(), path
            if path.is_file():
                paths.append(path)
    return sorted(paths)


def seal(root):
    assert json.loads((root / 'cpu-validation.json').read_text())['passed']
    assert json.loads((root / 'native-preflight.json').read_text())['passed']
    assert json.loads((root / 'timeout-audit.json').read_text())['passed']
    configure(root)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o022)
    selected = files(root)
    for path in selected:
        assert path.is_file() and not path.is_symlink(), path
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o022)
    for directory in ('install', 'harness'):
        for path in (root / directory, *(root / directory).rglob('*')):
            if path.is_dir():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o022)
    manifest = root / 'SHA256SUMS'
    manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n' for p in selected))
    manifest.chmod(0o644)
    pin = digest(manifest)
    report = dict(manifest_sha256=pin, files=len(selected), queued=False, jobs=[])
    for mode in ('matrix', 'confirm'):
        job = root / f'x7-width-{mode}.job'
        job.write_text(f'''# timeout 1200
# Whole-machine sampled verify widths, TP4, 250 W per GPU.
set -euo pipefail
source "$HOME/.config/chopinfer/local.env"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_REQUIRE
exec timeout --signal=TERM --kill-after=15s 1170s python3 -B {shlex.quote(str(root / 'harness/stage.py'))} {shlex.quote(str(root))} --execute {mode} --manifest {pin}
''')
        job.chmod(0o644)
        subprocess.run(['bash', '-n', str(job)], check=True)
        report['jobs'].append(dict(mode=mode, path=str(job), sha256=digest(job)))
    verify(root, pin)
    write(root / 'staged-jobs.json', report)
    print(json.dumps(report, indent=2))


def verify(root, pin):
    manifest = root / 'SHA256SUMS'
    assert digest(manifest) == pin, 'seal changed'
    expected = set()
    for line in manifest.read_text().splitlines():
        sha, name = line.split('  ', 1)
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert not path.is_symlink() and path.is_file()
        assert not path.stat().st_mode & 0o022, path
        assert digest(path) == sha, f'sealed file changed: {name}'
        expected.add(path)
    assert expected == set(files(root)), 'frozen file inventory changed'


def execute(root, mode, pin):
    os.environ['X7_JOB_STARTED_MONOTONIC'] = str(time.monotonic())
    verify(root, pin)
    settings = json.loads((root / 'config.json').read_text())
    source = Path(settings['source'])
    sys.path.insert(0, str(source))
    output = root / 'runs' / (mode + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(os.getpid()))
    output.parent.mkdir(exist_ok=True)
    # Use the standard content-verified stamp cache. CPU staging primes it;
    # no model or CUDA library is loaded here.
    result = subprocess.run([sys.executable, '-B', '-m', 'tools.refrun.stamps',
               '--checkpoint', settings['checkpoint'], '--lock', 'models/deepseek41/checkpoint-files.json'],
               cwd=source, capture_output=True, text=True, timeout=30)
    output.with_suffix('.checkpoint.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, 'checkpoint lock verification failed'
    print(f'Output: {output}', flush=True)
    os.execv(sys.executable, [sys.executable, '-B', str(root / 'harness/run.py'), str(root), str(output), '--mode', mode])


def audit_timeout(root, previous):
    readiness = json.loads((previous / 'startup.json').read_text())['seconds']
    text = (previous / 'server.log').read_text()
    warmups = re.findall(r'startup generation warmup: ([0-9.]+)s', text)
    assert len(warmups) == 1
    warmup = float(warmups[0])
    row = json.loads((previous / 'C8.json').read_text())
    # Host samples bracket the full request cohort, including the prompt.
    samples = row['host_load']['samples']
    seconds = samples[-1]['at'] - samples[0]['at']
    assert seconds >= row['cohort_decode']['seconds']
    startup_estimate = readiness - warmup + 10
    cohort_allowance = max(12, seconds * 2)
    forecast = {mode: servers * startup_estimate + servers * 2 * cohort_allowance + 60
                for mode, servers in [('matrix', 7), ('confirm', 4)]}
    assert all(v < 1140 for v in forecast.values()), forecast
    assert readiness < 170 and seconds < 30
    write(root / 'timeout-audit.json', dict(passed=True, previous=str(previous),
          observed_readiness_seconds=readiness, observed_generation_warmup_seconds=warmup,
          observed_C8_cohort_seconds=seconds, estimated_startup_with_verify_warmup_off=startup_estimate,
          cohort_allowance_seconds=cohort_allowance, forecasts_seconds=forecast,
          work_deadline=1140, outer_timeout=1170, queue_timeout=1200,
          limitation='Forecast replaces the measured full verify warmup with a ten-second allowance. Startup with the new matrix arguments and C16 duration remain GPU measurements. A short or timed-out matrix fails; it never skips cases.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', type=Path)
    p.add_argument('--configure', action='store_true')
    p.add_argument('--seal', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--execute', choices=('matrix', 'confirm'))
    p.add_argument('--manifest')
    p.add_argument('--timings', type=Path)
    args = p.parse_args()
    assert sum(bool(v) for v in (args.configure, args.seal, args.verify, args.execute, args.timings)) == 1
    root = args.stage.resolve()
    if args.configure: configure(root)
    elif args.seal: seal(root)
    elif args.verify: verify(root, args.manifest)
    elif args.execute: execute(root, args.execute, args.manifest)
    else: audit_timeout(root, args.timings)


if __name__ == '__main__':
    main()
