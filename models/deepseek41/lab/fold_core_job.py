"""Run one frozen fold-core candidate per short GPU pair."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import projection_zero_job as shared

VALID = {'decode': ['m8n16k4', 'm8n32k2', 'm32n8k4'],
         'prefill': ['m32s3', 'm64s2', 'm64s3', 'm128s2']}


def expected(catalog, inputs, variant, mem):
    shapes = [v for v in catalog['variants'] if v['variant'] == variant]
    if not shapes:
        raise ValueError('undeclared variant')
    keys = []
    for shape in shapes:
        dims = tuple(shape[k] for k in ('n', 'k', 'groups'))
        keys += [(*dims, rows, variant, 'synthetic') for rows in catalog['memcheck_rows' if mem else 'rows']]
        if mem:
            continue
        for capture in inputs['captures']:
            if tuple(capture[k] for k in ('n', 'k', 'groups')) != dims:
                continue
            if catalog['phase'] == 'decode':
                rows = [1, 8] if capture['rows'] == 8 else [32]
            else:
                rows = [64, 2496, 8192] if capture['phase'] == 'verify-32' and capture['layer'] == 0 else []
            keys += [(*dims, r, variant, capture['id'] + ('-repeated' if r > capture['rows'] else '-prefix')) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate expected cases')
    return keys


def analyze(text, catalog, inputs, variant, mem):
    shared.libtest(text, 1)
    cases = shared.records(text, 'FOLD_CORE_CASE')
    keys = expected(catalog, inputs, variant, mem)
    shared.exact_keys(cases, ('n', 'k', 'groups', 'rows', 'variant', 'source'), keys, 'fold core')
    for case in cases:
        if (case['phase'] != catalog['phase'] or case['memcheck'] is not mem or
                case['comparisons'] != 8 or case['bitwise'] is not True or case['negative'] is not True or
                case['delayed_admission'] is not (catalog['phase'] == 'decode')):
            raise ValueError('incomplete correctness evidence')
        if not mem:
            shared.timing(case['warm'])
            shared.timing(case['cold'])
    if shared.records(text, 'FOLD_CORE_DONE') != [dict(phase=catalog['phase'], memcheck=mem, cases=len(keys))]:
        raise ValueError('missing completion')
    if mem and text.count('ERROR SUMMARY: 0 errors') != 1:
        raise ValueError('missing clean memcheck')
    return cases


def environment(root, phase, variant, worker=None, mem=False):
    env = dict(os.environ)
    for key in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS'):
        env.pop(key, None)
    env.update(CHOPINFER_FOLD_CORE_ROOT=str(root), CHOPINFER_FOLD_CORE_PHASE=phase,
               CHOPINFER_FOLD_CORE_VARIANT=variant, CHOPINFER_KERNEL_TIMEOUT_SECS='800')
    if worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE='1',
                   CHOPINFER_FOLD_CORE_MEMCHECK='1' if mem else '0')
    return env


def seal(root, plan):
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and not p.is_relative_to(root / 'runs')}
    if actual != set(plan['files']) | {'PLAN.json', 'SHA256SUMS'}:
        raise ValueError('unsealed files')
    for name, meta in plan['files'].items():
        p = root / name
        if p.is_symlink() or p.stat().st_mode & 0o022 or p.stat().st_size != meta['bytes'] or shared.sha(p) != meta['sha256']:
            raise ValueError('changed snapshot: ' + name)
    if shared.sha(Path(plan['sanitizer'])) != plan['sanitizer_sha256']:
        raise ValueError('sanitizer changed')


def dry_run(root, plan, phase, variant, out):
    seal(root, plan)
    env = environment(root, phase, variant)
    tests = ['cpu_catalog', 'data::row_replication_keeps_complete_rows', 'data::captured_inputs_match_shapes',
             'data::synthetic_rows_are_distinct_across_tile_boundaries']
    for index, test in enumerate(tests):
        result = subprocess.run([str(root/'bin/fold-core-tests'), test, '--exact', '--include-ignored', '--nocapture'],
                                env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        (out/f'cpu-{index}.log').write_text(result.stdout)
        shared.libtest(result.stdout, 1)
    parser_env = dict(env, CHOPINFER_FOLD_CORE_RECORDED=str(root/'source/recorded-event.log'))
    result = subprocess.run(['python3', '-B', str(root/'source/test_fold_core_job.py')],
                            env=parser_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    (out/'parser.log').write_text(result.stdout)
    if 'Ran 5 tests' not in result.stdout or '\nOK' not in result.stdout:
        raise ValueError('parser tests did not execute')
    [catalog] = shared.records((out/'cpu-0.log').read_text(), 'FOLD_CORE_CATALOG')
    wanted = [v for v in plan['catalogs'][phase]['variants'] if v['variant'] == variant]
    if catalog['variants'] != wanted or catalog['phase'] != phase or catalog['rows'] != plan['catalogs'][phase]['rows']:
        raise ValueError('CPU matrix differs')
    resources = {}
    seen = set()
    for name in [*catalog['references'], 'fold_core_decode_sm120', 'fold_core_prefill_sm120']:
        path = root / ('reference' if name in catalog['references'] else 'cubins') / (name+'.cubin')
        identity = json.loads(path.with_suffix('.identity.json').read_text())
        if identity['cubin_sha256'] != shared.sha(path):
            raise ValueError('identity differs')
        symbols = catalog['references'][name]['symbols'] if name in catalog['references'] else [r['symbol'] for r in identity['resources']]
        elf = subprocess.check_output([plan['cuobjdump'], '--dump-elf-symbols', str(path)], text=True)
        for symbol in symbols:
            if symbol not in elf:
                raise ValueError('missing symbol ' + symbol)
            seen.add(symbol)
        for r in identity['resources']:
            if r['local_bytes'] or r['stack_bytes']:
                raise ValueError('spill or stack: '+r['symbol'])
            resources[r['symbol']] = r
    for call in catalog['plans']:
        r = resources[call['symbol']]
        if (call['symbol'] not in seen or call['block'][0] != r['max_threads'] or
                call['shared'] + r['shared_static_bytes'] > 101376):
            raise ValueError('invalid launch resource: ' + call['symbol'])
    probe = subprocess.check_output([plan['cuobjdump'], '--dump-elf-symbols', str(root/'cubins/probe_sm120.cubin')], text=True)
    if any(s not in probe for s in ('chop_probe_read_v1', 'chop_probe_flush_v1')):
        raise ValueError('missing timing probe')
    inputs = json.loads((root/'inputs/manifest.json').read_text())
    budget = plan['budgets'][phase]
    if sum(budget[k] for k in ('memcheck', 'timing', 'overhead')) > 1150:
        raise ValueError('invalid timeout budget')
    (out/'dry-run.json').write_text(json.dumps(dict(phase=phase, variant=variant, cpu_tests=4,
        cases_per_device=len(expected(catalog, inputs, variant, False)),
        memcheck_per_device=len(expected(catalog, inputs, variant, True)),
        gpu_launched=False, budget=budget), indent=2)+'\n')


def run_phase(root, plan, phase, variant, out, mem):
    deadline = time.monotonic() + plan['budgets'][phase]['memcheck' if mem else 'timing']
    children = []
    try:
        for worker in (0, 1):
            cmd = [str(root/'bin/fold-core-tests'), 'gpu_pair', '--exact', '--include-ignored', '--nocapture', '--test-threads=1']
            if mem:
                cmd = [plan['sanitizer'], '--tool', 'memcheck', '--error-exitcode', '86', '--target-processes', 'all',
                       '--require-cuda-init', 'yes', '--destroy-on-device-error', 'context', '--launch-timeout', '60', *cmd]
            path = out / f'{"memcheck" if mem else "gpu"}-{worker}.log'
            file = path.open('w')
            child = subprocess.Popen(cmd, env=environment(root, phase, variant, worker, mem), stdout=file,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append((child, file, path))
        for child, _, path in children:
            if child.wait(timeout=max(1, deadline-time.monotonic())):
                raise ValueError('failed '+str(path))
        inputs = json.loads((root/'inputs/manifest.json').read_text())
        return [analyze(path.read_text(), plan['catalogs'][phase], inputs, variant, mem) for _, _, path in children]
    finally:
        for child, file, _ in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            file.close()


def execute(root, plan, phase, variant, out):
    seal(root, plan)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 2 or len(set(visible.split(','))) != 2:
        raise ValueError('runner must assign two GPUs')
    started = time.time()
    verdict = dict(exit_code=1, gpu_launched=False, phase=phase, variant=variant, serving_enabled=False)
    try:
        verdict['power_before'] = shared.power(visible)
        verdict['gpu_launched'] = True
        run_phase(root, plan, phase, variant, out, True)
        result = run_phase(root, plan, phase, variant, out, False)
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        verdict['exit_code'] = 0
    except Exception as error:
        verdict['error'] = str(error)
    finally:
        try:
            verdict['power_after'] = shared.power(visible)
        except Exception as error:
            verdict['power_error'] = str(error)
            verdict['exit_code'] = 1
        j = subprocess.run(['journalctl', '-k', '--no-pager', '--since', '@'+str(int(started))],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15)
        (out/'kernel.log').write_text(j.stdout)
        verdict['journal_exit'] = j.returncode
        verdict['xids'] = [line for line in j.stdout.splitlines() if re.search(r'NVRM: Xid|Xid \(', line)]
        if j.returncode or verdict['xids']:
            verdict['exit_code'] = 1
        verdict['elapsed_seconds'] = time.time()-started
        (out/'verdict.json').write_text(json.dumps(verdict, indent=2)+'\n')
    if verdict['exit_code']:
        raise ValueError(str(verdict))
    print('Both GPU workers passed', phase, variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--phase', choices=VALID, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.variant not in VALID[args.phase]:
        p.error('variant does not belong to phase')
    def interrupted(signum, _):
        raise TimeoutError(f'interrupted {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.root/'PLAN.json').read_text())
    (dry_run if args.dry_run else execute)(args.root, plan, args.phase, args.variant, args.out)


if __name__ == '__main__':
    main()
