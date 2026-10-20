"""Frozen projection qualification with exact case accounting and bounded phases."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

ROWS = [1, 8, 32, 448, 512, 960, 2048, 2496, 4096, 8191, 8192]
SHAPES = [32, 384, 512]
MARKERS = ['BITS', 'CAPTURE', 'CAPTURE_PREFIX', 'PARTITION', 'NEGATIVE',
           'ISOLATION', 'TIME', 'OCCUPANCY', 'DONE']


def cases(mem):
    if mem:
        return [(r, p) for p in ('bf16', 'residual', 'mixed', 'weight', 'edges')
                for r in (8, 129)]
    return [(r, 'bf16') for r in ROWS] + [
        (r, p) for r in (8, 32, 129, 512)
        for p in ('residual', 'mixed', 'weight', 'edges')]


def tiles(rows):
    return [32] if rows <= 32 else [32, 64]


def records(text, marker):
    return [json.loads(line.split(marker + ' ', 1)[1])
            for line in text.splitlines() if marker + ' ' in line]


def libtest(text, count):
    results = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    if results != [('ok', str(count), '0', '0')]:
        raise ValueError(f'expected {count} executed passing tests, got {results}')


def exact_keys(rows, keys, expected, label):
    actual = [tuple(row[k] for k in keys) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(label + ': missing, extra or duplicate cases')


def timing(t):
    if len(t['ms']) != 2 or len(t['round_samples_ms']) != 5 or t['replays_per_sample'] != 1:
        raise ValueError('incomplete timing')
    medians = [[], []]
    for round_ in t['round_samples_ms']:
        if len(round_) != 2 or any(len(a) != 3 for a in round_):
            raise ValueError('incomplete paired rounds')
        for arm, samples in enumerate(round_):
            if any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError('invalid time')
            medians[arm].append(statistics.median(samples))
    for got, samples in zip(t['ms'], medians):
        if not math.isfinite(got) or got <= 0 or not math.isclose(got, statistics.median(samples), rel_tol=1e-6):
            raise ValueError('timing median differs from samples')


def expected_done(worker, mem, n=512):
    count = sum(len(tiles(r)) for r, _ in cases(mem))
    return dict(n=n, worker=worker, memcheck=mem, cases=count, comparisons=count * 8,
                capture=0 if mem else 2, negative=3, isolation=2, partitions=0 if mem else 8)


def analyze(text, worker, mem, n=512):
    libtest(text, 1)
    result = {m: records(text, 'PROJECTION_ZERO_' + m) for m in MARKERS}
    exact_keys(result['BITS'], ('rows', 'label', 'tile'),
               [(r, p, m) for r, p in cases(mem) for m in tiles(r)], 'bits')
    layer = (0, 39)[worker]
    exact_keys(result['CAPTURE'], ('layer', 'rows', 'tile'),
               [] if mem else [(layer, 8192, m) for m in (32, 64)], 'capture')
    exact_keys(result['CAPTURE_PREFIX'], ('layer', 'rows'),
               [] if mem else [(layer, r) for r in (1, 8, 32)], 'capture prefix')
    for key in ('BITS', 'CAPTURE', 'CAPTURE_PREFIX'):
        if any(r['comparisons'] != 8 or r['bitwise'] is not True for r in result[key]):
            raise ValueError('incomplete bitwise observations')
    exact_keys(result['PARTITION'], ('layer', 'tile', 'first', 'rows'), [] if mem else [
        (layer, m, f, r) for m in (32, 64)
        for f, r in ((0, 129), (129, 320), (449, 2048), (2497, 5695))], 'partition')
    exact_keys(result['NEGATIVE'], ('mutation',), [(v,) for v in (
        'omit_chain', 'force_certificate', 'force_decode')], 'controls')
    if any(r['changed'] is not True for r in result['NEGATIVE']):
        raise ValueError('ineffective control')
    exact_keys(result['ISOLATION'], ('tile',), [(32,), (64,)], 'isolation')
    if any(r['bitwise'] is not True for k in ('PARTITION', 'ISOLATION') for r in result[k]):
        raise ValueError('failed independence')
    expected = [] if mem else ([(r, 'bf16', m) for r in ROWS for m in tiles(r)] +
                               [(8192, 'capture', m) for m in (32, 64)] +
                               [(r, 'capture_prefix', 32) for r in (1, 8, 32)])
    exact_keys(result['TIME'], ('rows', 'label', 'tile'), expected, 'timing')
    for r in result['TIME']:
        if r['arms'] != ['base_prefill_dispatch', 'zero_components'] or r['includes_prepare'] is not True:
            raise ValueError('wrong paired arms')
        timing(r['timing'])
        timing(r['cold'])
        if r['prepare'] is not None:
            timing(r['prepare'])
    exact_keys(result['OCCUPANCY'], ('tile',), [(32,), (64,)], 'occupancy')
    if any(r['resident_ctas'] <= 0 or r['threads'] != 128 or
           r['shared_bytes'] != {32: 32768, 64: 57344}[r['tile']] for r in result['OCCUPANCY']):
        raise ValueError('invalid occupancy')
    if result['DONE'] != [expected_done(worker, mem, n)]:
        raise ValueError('missing completion')
    if mem and len(re.findall(r'ERROR SUMMARY: 0 errors', text)) != 1:
        raise ValueError('missing clean memcheck')
    return result


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def environment(root, n, worker=None, mem=False):
    env = dict(os.environ)
    for key in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS'):
        env.pop(key, None)
    env.update(CHOPINFER_PROJECTION_ZERO_ROOT=str(root), CHOPINFER_PROJECTION_ZERO_N=str(n),
               CHOPINFER_KERNEL_TIMEOUT_SECS='720')
    if worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE='1',
                   CHOPINFER_PROJECTION_ZERO_WORKER=str(worker),
                   CHOPINFER_PROJECTION_ZERO_MEMCHECK='1' if mem else '0')
    return env


def seal(root, plan):
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and not p.is_relative_to(root / 'runs')}
    if actual != set(plan['files']) | {'PLAN.json', 'pair.job', 'SHA256SUMS'}:
        raise ValueError('unsealed file set')
    for name, meta in plan['files'].items():
        p = root / name
        stat = p.lstat()
        if p.is_symlink() or stat.st_mode & 0o022 or stat.st_size != meta['bytes'] or sha(p) != meta['sha256']:
            raise ValueError('changed or writable file: ' + name)
    if sha(Path(plan['sanitizer'])) != plan['sanitizer_sha256']:
        raise ValueError('sanitizer changed')


def dry_run(root, plan, out):
    seal(root, plan)
    names = ['cpu::launch_matrix', 'cpu::certificate_exhausts_bf16_and_keeps_zero_merges',
             'cpu::row_certificate_is_independent_and_workspace_is_disjoint', 'cpu::capture_admission']
    for n in SHAPES:
        for i, name in enumerate(names):
            r = subprocess.run([str(root / 'bin/projection-test'), name, '--exact',
                                '--include-ignored', '--nocapture'], env=environment(root, n),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               check=True, timeout=180)
            (out / f'cpu-{n}-{i}.log').write_text(r.stdout)
            libtest(r.stdout, 1)
        [dry] = records((out / f'cpu-{n}-0.log').read_text(), 'PROJECTION_ZERO_DRY')
        if dry['rows'] != ROWS or dry['cases'] != [list(x) for x in cases(False)] or dry['memcheck_cases'] != [list(x) for x in cases(True)]:
            raise ValueError('CPU/GPU matrix disagreement')
        resources = {}
        seen = set()
        paths = [root / 'cubins/projection_zero_sm120.cubin'] + [
            root / 'reference' / (name + '.cubin') for name in dry['embedded']]
        for path in paths:
            meta = json.loads(path.with_suffix('.identity.json').read_text())
            if sha(path) != meta['cubin_sha256']:
                raise ValueError('cubin identity mismatch')
            elf = subprocess.check_output([plan['cuobjdump'], '--dump-elf-symbols', str(path)], text=True)
            symbols = dry['symbols'] if path.parent.name == 'cubins' else dry['embedded'][path.stem]['symbols']
            for symbol in symbols:
                if symbol not in elf:
                    raise ValueError('missing export ' + symbol)
                seen.add(symbol)
            resources.update((r['symbol'], r) for r in meta['resources'])
        for launch in dry['launches']:
            s = launch['symbol']
            r = resources[s]
            if (s not in seen or r['stack_bytes'] or r['local_bytes'] or
                    r['max_threads'] != launch['block'][0] or launch['cluster'] != [1, 1, 1] or
                    r['shared_static_bytes'] + launch['shared_bytes'] > 101376):
                raise ValueError('invalid launch resources ' + s)
    probe = subprocess.check_output([plan['cuobjdump'], '--dump-elf-symbols',
                                     str(root / 'cubins/probe_sm120.cubin')], text=True)
    if any(s not in probe for s in ('chop_probe_flush_v1', 'chop_probe_read_v1')):
        raise ValueError('missing timing probe')
    dump = subprocess.check_output([plan['cuobjdump'], '--dump-sass',
                                   str(root / 'cubins/projection_zero_sm120.cubin')], text=True)
    if re.search(r'\bHMMA[^\n]*TF32', dump):
        raise ValueError('TF32 instruction')
    (out / 'candidate.sass').write_text(dump)
    b = plan['budget']
    if not (b['memcheck_seconds'] + b['replay_seconds'] + b['overhead_seconds'] <= b['alarm_seconds'] < 1200):
        raise ValueError('invalid timeout budget')
    (out / 'dry-run.json').write_text(json.dumps(dict(cpu_tests=12, gpu_launched=False, budget=b), indent=2) + '\n')


def phase(root, plan, out, mem):
    deadline = time.monotonic() + plan['budget']['memcheck_seconds' if mem else 'replay_seconds']
    results = []
    for n in SHAPES:
        children = []
        try:
            for worker in (0, 1):
                cmd = [str(root / 'bin/projection-test'), 'projection_gpu', '--exact',
                       '--include-ignored', '--nocapture', '--test-threads=1']
                if mem:
                    cmd = [plan['sanitizer'], '--tool', 'memcheck', '--error-exitcode', '86',
                           '--target-processes', 'all', '--require-cuda-init', 'yes',
                           '--destroy-on-device-error', 'context', '--launch-timeout', '60', *cmd]
                path = out / f'{"memcheck" if mem else "gpu"}-{n}-{worker}.log'
                file = path.open('w')
                child = subprocess.Popen(cmd, env=environment(root, n, worker, mem), stdout=file,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children.append((child, file, path))
            for child, _, path in children:
                if child.wait(timeout=max(1, deadline-time.monotonic())):
                    raise ValueError('failed ' + str(path))
            for worker, (_, _, path) in enumerate(children):
                results.append(dict(n=n, worker=worker, observations=analyze(path.read_text(), worker, mem, n)))
        finally:
            for child, file, _ in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                file.close()
    return results


def power(visible):
    text = subprocess.check_output(['nvidia-smi', '-i', visible,
                                   '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True)
    values = [float(v.strip()) for v in text.splitlines()]
    if len(values) != 2 or any(abs(v-250) > 0.1 for v in values):
        raise ValueError('expected 250 W on both assigned GPUs')
    return values


def execute(root, plan, out):
    seal(root, plan)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 2 or len(set(visible.split(','))) != 2:
        raise ValueError('runner must assign two GPUs')
    started = time.time()
    verdict = dict(exit_code=1, gpu_launched=False, serving_enabled=False)
    try:
        verdict['power_before'] = power(visible)
        verdict['gpu_launched'] = True
        phase(root, plan, out, True)
        results = phase(root, plan, out, False)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        verdict['exit_code'] = 0
    except Exception as error:
        verdict['error'] = str(error)
    finally:
        try:
            verdict['power_after'] = power(visible)
        except Exception as error:
            verdict['power_error'] = str(error)
            verdict['exit_code'] = 1
        j = subprocess.run(['journalctl', '-k', '--no-pager', '--since', '@'+str(int(started))],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15)
        (out / 'kernel.log').write_text(j.stdout)
        verdict['journal_exit'] = j.returncode
        verdict['xids'] = [l for l in j.stdout.splitlines() if re.search(r'NVRM: Xid|Xid \(', l)]
        if j.returncode or verdict['xids']:
            verdict['exit_code'] = 1
        verdict['elapsed_seconds'] = time.time()-started
        (out / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    if verdict['exit_code']:
        raise ValueError(str(verdict))
    print('Passed all three projection shapes on both GPU workers')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, _):
        raise TimeoutError(f'interrupted {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.root / 'PLAN.json').read_text())
    (dry_run if args.dry_run else execute)(args.root, plan, args.out)


if __name__ == '__main__':
    main()
