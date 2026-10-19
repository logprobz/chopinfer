"""Run frozen direct-fold workers; the queue owns GPU admission and power caps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import fold_bandwidth_report as report

NSYS = '/usr/local/cuda/bin/nsys'
GPU_TEST = 'mxfp8_fold_bandwidth::direct_fold_gpu'


def test(binary, name, log, env):
    with log.open('w') as output:
        subprocess.run([str(binary), '--exact', name, '--test-threads=1', '--include-ignored',
                        '--nocapture'], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    report.check_log(log.read_text())


def worker_environment(device, root):
    assert device in (0, 1)
    env = os.environ.copy()
    env.pop('CHOPINFER_KERNEL_GPUS', None)
    env['CHOPINFER_KERNEL_GPU'] = str(device)
    env['CHOPINFER_KERNEL_REQUIRE'] = '1'
    env['CHOPINFER_KERNEL_RECEIPTS'] = str(root/f'receipts{device}')
    return env


def dry(snapshot, root):
    binary = snapshot/'bin/fold-tests'
    env = os.environ.copy()
    for variable in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS', 'CHOPINFER_KERNEL_REQUIRE'):
        env.pop(variable, None)
    env['CHOPINFER_DEEPSEEK41_FOLD_CUBINS'] = str(snapshot/'cubins')
    tests = {
        'cpu': 'mxfp8_fold_bandwidth::direct_fold_cpu_matrix',
        'modules': 'mxfp8_fold_bandwidth::direct_fold_dry_run',
        'module-controls': 'mxfp8_fold_decode::modules::dry_run_refuses_missing_or_wrong_timing_probe',
        'ordinal-controls': 'mxfp8_fold_bandwidth::direct_fold_rejects_missing_or_multiple_ordinals',
    }
    for label, name in tests.items():
        test(binary, name, root/f'{label}.log', env)
    for device in (0, 1):
        test(binary, 'mxfp8_fold_bandwidth::direct_fold_worker_dry_run',
             root/f'worker{device}.log', worker_environment(device, root))
    declared = subprocess.check_output([str(binary), '--list'], text=True)
    assert GPU_TEST + ': test' in declared
    (root/'tests.txt').write_text(declared)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(snapshot),
                    '-p', 'test_fold_bandwidth_report.py'], check=True)
    assert os.access(NSYS, os.X_OK)
    assert all(not p.stat().st_mode & 0o022 for p in [snapshot, *snapshot.rglob('*')])
    result = report.inventory(root)
    print(f'CPU inventory complete: {result["cases"]} cases, {len(result["modules"])} modules, one ordinal per worker')


def pair(snapshot, root):
    children = []
    for device in (0, 1):
        env = worker_environment(device, root)
        env['CHOPINFER_DEEPSEEK41_FOLD_CUBINS'] = str(snapshot/'cubins')
        with (root/f'gpu{device}.log').open('w') as output:
            child = subprocess.Popen([NSYS, 'profile', '--trace=cuda', '--cuda-graph-trace=node',
                '--sample=none', '--cpuctxsw=none', '--export=sqlite', '--force-overwrite=true',
                '--output='+str(root/f'fold{device}'), str(snapshot/'bin/fold-tests'),
                '--exact', GPU_TEST, '--include-ignored', '--test-threads=1', '--nocapture'],
                env=env, stdout=output, stderr=subprocess.STDOUT)
        children.append(child)
    exits = [child.wait() for child in children]
    (root/'process-exits.json').write_text(json.dumps(dict(zip(('gpu0', 'gpu1'), exits)))+'\n')
    assert exits == [0, 0], exits
    report.summarize(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('dry', 'pair'))
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    assert args.snapshot.is_absolute() and args.root.is_absolute()
    globals()[args.mode](args.snapshot, args.root)
