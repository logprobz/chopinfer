"""Run and analyze the frozen Decode-16 HC tail pair (tests/sm120/hc_tail16).

`preflight` checks the snapshot, the modules' exports, the CPU tests, the
lab plans against the bundle's exported Decode-16 journals (the served
boundary before the binding, the tail after it) and this file's tests.
`run` runs memcheck, the bitwise matrix and the paired timing on one GPU.
`analyze` accepts both GPUs' evidence only when complete and reports the
saving per Decode-16 step (77 boundaries per rank). Never accepts zero
tests or partial evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

SELECTORS = dict(memcheck='replay::memcheck_real_rows16_on_gpu',
                 matrix='replay::real_rows16_bitwise_invariances_and_controls_on_gpu',
                 timing='timing::paired_cold_hot_tail_and_aa_on_gpu',
                 artifacts='artifacts::frozen_modules_and_cases_on_cpu')
CPU = ('device::tests::served_and_tail_plans_bind_every_role',
       'replay::comparison_rejects_changed_values_and_row_positions',
       'corpus::absent_or_null_widen_is_disabled', SELECTORS['artifacts'])
SERVED = ('hc_round_expand_mix_partials_v1', 'hc_round_expand_mix_finish_v1', 'norm_quantize_deepseek41_input_v1')
TAIL = ('hc_round_chain_w4_v1', 'hc_finish_norm_w32_v1')
BOUNDARIES = 77
LAUNCH = re.compile(r'launch \d+ (\S+) LaunchConfigV1 \{ flags: \d+, grid_x: (\d+), grid_y: (\d+), grid_z: (\d+), '
                    r'block_x: (\d+), block_y: (\d+), block_z: (\d+), shared_memory_bytes: (\d+),')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def passed(text):
    return text.count('1 passed; 0 failed; 0 ignored') == 1 and not re.search(r'\bskipp(?:ed|ing)\b', text, re.I)


def journal_shapes(text, suffixes):
    """(symbol, grid, block, shared) of the journal's launches whose symbol
    ends with one of `suffixes`, with counts."""
    shapes = {}
    for line in text.splitlines():
        m = LAUNCH.match(line)
        if m and m[1].endswith(suffixes):
            key = (m[1], [int(x) for x in m.group(2, 3, 4)], [int(x) for x in m.group(5, 6, 7)], int(m[8]))
            shapes[json.dumps(key)] = shapes.get(json.dumps(key), 0) + 1
    return shapes


def plan_shapes(text):
    """Per arm, the lab plans' (symbol, grid, block, shared) at each widen."""
    result = {}
    for line in text.splitlines():
        if 'HC_TAIL16_PLANS ' in line:
            value = json.loads(line.split('HC_TAIL16_PLANS ', 1)[1])
            for arm in value['arms']:
                result.setdefault(arm['arm'], set()).update(
                    json.dumps([l['symbol'], l['grid'], l['block'], l['dynamic_shared_bytes']]) for l in arm['launches'])
    return result


def layouts(plans_text, before, after):
    """The lab's arms launch exactly the bundle's Decode-16 boundary
    shapes: the served group before the binding and the tail after. The
    input norm also runs at the standalone sites, which stay."""
    plans = plan_shapes(plans_text)
    old, new = journal_shapes(before, SERVED), journal_shapes(after, TAIL)
    if set(old) != plans.get('Served') or set(new) != plans.get('Tail'):
        raise ValueError(f'lab plans differ from the Decode-16 journals: {plans} {sorted(old)} {sorted(new)}')
    fused = 4 * BOUNDARIES
    mixer = {k: n for k, n in old.items() if SERVED[2] not in k}
    norm = sum(n for k, n in old.items() if SERVED[2] in k)
    kept = sum(journal_shapes(after, SERVED[2:]).values())
    if journal_shapes(after, SERVED[:2]) or journal_shapes(before, TAIL) or norm - kept != fused:
        raise ValueError('the binding does not replace every served boundary')
    if sorted(set(mixer.values()) | set(new.values())) != [fused]:
        raise ValueError(f'expected {BOUNDARIES} boundaries per rank')
    return dict(served=sorted(old), tail=sorted(new), boundaries_per_rank=BOUNDARIES, standalone_norms=kept)


def command(root, selector):
    return [str(root / 'snapshot/hc-tail16-tests'), '--exact', selector, '--include-ignored', '--nocapture',
            '--test-threads=1']


def environment(root, ordinal=None):
    import os
    c = json.loads((root / 'staged.json').read_text())
    env = dict(os.environ, **c['environment'])
    for name in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS', 'CHOPINFER_KERNEL_SHARED_DEVICE',
                 'CHOPINFER_HC_TAIL16_EXPORT', 'KERNELS_LOCK_UPDATE'):
        env.pop(name, None)
    if ordinal is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(ordinal), CHOPINFER_KERNEL_REQUIRE='1')
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return c, env


def preflight(root, out):
    c, env = environment(root)
    for name, sha in c['files'].items():
        path = root / name
        if path.is_symlink() or path.stat().st_mode & 0o022 or digest(path) != sha:
            raise ValueError('snapshot differs: ' + name)
    modules = json.loads((root / 'snapshot/cubins/modules.json').read_text())
    for m in modules:
        path = root / 'snapshot/cubins' / (m['name'] + '.cubin')
        if digest(path) != m['sha256']:
            raise ValueError('module digest differs')
        text = subprocess.run([c['cuobjdump'], '--dump-elf-symbols', str(path)], text=True, capture_output=True,
                              check=True).stdout
        symbols = {line.split()[-1] for line in text.splitlines()
                   if all(v in line for v in ('STT_FUNC', 'STB_GLOBAL', 'STO_ENTRY'))}
        if symbols != set(m['symbols']):
            raise ValueError('module exports differ: ' + m['name'])
    listing = subprocess.run([str(root / 'snapshot/hc-tail16-tests'), '--list'], env=env, text=True,
                             capture_output=True, check=True).stdout
    for selector in SELECTORS.values():
        if f'{selector}: test' not in listing:
            raise ValueError('missing test ' + selector)
    texts = {}
    for index, selector in enumerate(CPU):
        result = subprocess.run(command(root, selector), env=env, text=True, capture_output=True, timeout=300)
        text = result.stdout + result.stderr
        (out / f'cpu-{index}.log').write_text(text)
        if result.returncode or not passed(text):
            raise ValueError('CPU test failed: ' + selector)
        texts[selector] = text
    layout = layouts(texts[CPU[0]], (root / c['journals']['before']).read_text(),
                     (root / c['journals']['after']).read_text())
    coverage = json.loads((root / 'snapshot/cubins/coverage.json').read_text())
    if len(coverage['cases']) != 3 * BOUNDARIES or coverage['matrix_reports'] != 6 * BOUNDARIES:
        raise ValueError('coverage differs')
    if sum(c['stage_seconds'].values()) + c['preflight_and_cleanup_seconds'] > c['outer_seconds']:
        raise ValueError('stage timeouts exceed the job')
    value = dict(cpu_tests=len(CPU), modules=len(modules), symbols=sum(len(m['symbols']) for m in modules),
                 cases=len(coverage['cases']), layout=layout, gpu_launched=False)
    (out / 'preflight.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def run(root, out, ordinal):
    c, env = environment(root, ordinal)
    env.update(CHOPINFER_HC_TAIL16_REPORT=str(out / f'matrix-{ordinal}.json'),
               CHOPINFER_HC_TAIL16_MEMCHECK_REPORT=str(out / f'memcheck-{ordinal}.json'),
               CHOPINFER_HC_TAIL16_TIMING_REPORT=str(out / f'timing-{ordinal}.json'),
               CHOPINFER_KERNEL_RECEIPTS=str(out / f'receipts-{ordinal}'))
    failed = False
    for kind in ('memcheck', 'matrix', 'timing'):
        label = f'{kind}-{ordinal}'
        argv = command(root, SELECTORS[kind])
        if kind == 'memcheck':
            argv = [c['sanitizer'], '--tool', 'memcheck', '--error-exitcode', '77'] + argv
        start = time.monotonic()
        with (out / (label + '.log')).open('w') as f:
            try:
                rc = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=c['stage_seconds'][kind], check=False).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        (out / (label + '.exit.json')).write_text(json.dumps(dict(exit_code=rc, elapsed_seconds=time.monotonic() - start,
                                                                  command=argv)) + '\n')
        failed |= rc != 0
    return int(failed)


def step_totals(records):
    """Per (kind, pair, hot): the summed per-site saving of one Decode-16
    step (each site's cases averaged first), in microseconds."""
    sites = {}
    for r in records:
        site = r['case'].split(':', 1)[1] if r['kind'] == 'decode8x2' else r['case'].split(':', 2)[2]
        sites.setdefault((r['kind'], r['pair'], r['hot'], site), []).append(r)
    rows = {}
    for (kind, pair, hot, site), rs in sites.items():
        entry = rows.setdefault((kind, pair, hot), dict(kind=kind, pair=pair, hot=hot, sites=0, reference=0.0,
                                                        candidate=0.0, faster=0))
        ref = statistics.fmean(r['reference_us'] for r in rs)
        cand = statistics.fmean(r['candidate_us'] for r in rs)
        entry['sites'] += 1
        entry['reference'] += ref
        entry['candidate'] += cand
        entry['faster'] += cand < ref
    result = []
    for key in sorted(rows):
        e = rows[key]
        result.append(dict(e, reference_us_per_step=round(e.pop('reference'), 3),
                           candidate_us_per_step=round(e.pop('candidate'), 3)))
        result[-1]['saved_us_per_step'] = round(result[-1]['reference_us_per_step']
                                                - result[-1]['candidate_us_per_step'], 3)
    return result


def check_matrix(value, memcheck, cases):
    if value['passed'] is not True or value['memcheck'] is not memcheck:
        raise ValueError('matrix failed')
    reports = value['reports']
    expected = 2 * 1 if memcheck else 2 * cases
    if len(reports) != expected or not all(r['passed'] for r in reports):
        raise ValueError('incomplete or failing reports')
    if not memcheck:
        if value['negatives'] != 10 or not value['checks'] or not all(c['passed'] for c in value['checks']):
            raise ValueError('controls incomplete')
    return {(r['case'], r['configuration']): r['output_sha256'] for r in reports}


def analyze(root, out):
    c, _ = environment(root)
    cases = len(json.loads((root / 'snapshot/cubins/coverage.json').read_text())['cases'])
    results, digests = [], []
    for ordinal in (0, 1):
        for kind in ('memcheck', 'matrix', 'timing'):
            if json.loads((out / f'{kind}-{ordinal}.exit.json').read_text())['exit_code'] != 0:
                raise ValueError(f'{kind}-{ordinal} failed')
        log = (out / f'memcheck-{ordinal}.log').read_text()
        if 'ERROR SUMMARY: 0 errors' not in log:
            raise ValueError('memcheck reported errors')
        check_matrix(json.loads((out / f'memcheck-{ordinal}.json').read_text()), True, cases)
        matrix = json.loads((out / f'matrix-{ordinal}.json').read_text())
        digests.append(check_matrix(matrix, False, cases))
        timing = json.loads((out / f'timing-{ordinal}.json').read_text())
        if len(timing['records']) != 4 * cases:
            raise ValueError('timing incomplete')
        results.append(dict(ordinal=ordinal, checks=len(matrix['checks']), negatives=matrix['negatives'],
                            captured=len(matrix['captured']), captured_passed=matrix['captured_passed'],
                            timing=step_totals(timing['records'])))
    if digests[0] != digests[1]:
        raise ValueError('the GPUs disagree on outputs')
    value = dict(complete=True, cases=cases, results=results)
    (out / 'analysis.json').write_text(json.dumps(value, indent=2) + '\n')
    lines = ['# Decode-16 HC tail against the served boundary on real captured rows', '',
             f'{cases} cases per GPU, both configurations bitwise, outputs equal across GPUs.', '',
             '| GPU | Kind | Pair | Hot | Sites | Served us/step | Candidate us/step | Saved us/step | Faster sites |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        for t in r['timing']:
            lines.append(f"| {r['ordinal']} | {t['kind']} | {t['pair']} | {t['hot']} | {t['sites']} | "
                         f"{t['reference_us_per_step']} | {t['candidate_us_per_step']} | {t['saved_us_per_step']} | "
                         f"{t['faster']} |")
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'run', 'analyze'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ordinal', type=int)
    args = p.parse_args()
    if args.action == 'preflight':
        print(json.dumps(preflight(args.root, args.output)))
    elif args.action == 'run':
        raise SystemExit(run(args.root, args.output, args.ordinal))
    else:
        raise SystemExit(analyze(args.root, args.output))
