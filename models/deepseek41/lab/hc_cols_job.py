"""Run and analyze the frozen wide-bucket HC pair (tests/sm120/hc_cols).

`preflight` checks the snapshot, the modules' exports, that the served
modules are the profiled install's, the CPU tests, the lab's served plans
against the profiled Decode-16 and Decode-32 kernels, the Nsight Compute
metrics and argument vector, and this file's tests. `run` runs memcheck,
the bitwise matrix, the paired timing, Nsight Compute and racecheck on one
GPU. `analyze` accepts both GPUs' evidence only when complete and reports
the saving per step (77 boundaries per rank). Never accepts zero tests or
partial evidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

SELECTORS = dict(sanitizer='replay::sanitized_real_rows_on_gpu',
                 matrix='replay::real_rows_bitwise_invariances_and_controls_on_gpu',
                 timing='timing::paired_cold_hot_served_and_candidates_on_gpu',
                 profile='profile::eager_arms_for_nsight_compute_on_gpu',
                 artifacts='artifacts::frozen_modules_and_cases_on_cpu')
CPU = ('device::tests::served_plans_and_candidates_bind_every_role',
       'replay::comparison_rejects_changed_values_row_positions_and_written_lanes',
       'replay::every_control_runs_under_a_prior_that_shows_it',
       'corpus::absent_or_null_widen_is_disabled', SELECTORS['artifacts'])
STAGES = ('memcheck', 'matrix', 'timing', 'ncu', 'racecheck')
BOUNDARIES = 77
KINDS = dict(decode8x2=16, verify32h=16, verify32=32)
RACECHECK_KINDS = ('decode8x2', 'verify32')
# Input mutations and the column pair's five kernel controls, per kind.
NEGATIVES_PER_KIND = 10
NCU_REGEX = '^chop_deepseek41_hc_'
NCU_SECTIONS = ('LaunchStats', 'Occupancy')
NCU_METRICS = ('gpu__time_duration.sum',
               'dram__bytes_op_read.sum',
               'dram__bytes_op_write.sum',
               'lts__t_bytes.sum',
               'dram__throughput.avg.pct_of_peak_sustained_elapsed',
               'sm__throughput.avg.pct_of_peak_sustained_elapsed',
               'sm__warps_active.avg.pct_of_peak_sustained_active',
               'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
               'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
               'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
               'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def passed(text):
    return text.count('1 passed; 0 failed; 0 ignored') == 1 and not re.search(r'\bskipp(?:ed|ing)\b', text, re.I)


def plan_shapes(text):
    """Per (rows, arm), the lab plans' (symbol, grid, block) at each widen."""
    result = {}
    for line in text.splitlines():
        if 'HC_COLS_PLANS ' in line:
            value = json.loads(line.split('HC_COLS_PLANS ', 1)[1])
            for arm in value['arms']:
                result.setdefault((value['rows'], arm['arm']), set()).update(
                    json.dumps([l['symbol'], l['grid'], l['block'], l['dynamic_shared_bytes']]) for l in arm['launches'])
    return result


def profiled(plans_text, kernels):
    """The lab's served arm launches, at 16 and 32 rows, exactly the HC
    pair the 300 W profile measured at C16 and C32: the same symbols and
    grids, 77 launches each per cycle. `kernels` maps rows to the profile's
    per-kernel rows (kernel, launches_per_cycle, grids)."""
    plans = plan_shapes(plans_text)
    result = {}
    for rows in (16, 32):
        served = sorted(json.loads(x)[:2] for x in plans.get((rows, 'served'), ()))
        if len(served) != 2:
            raise ValueError(f'missing served plans at {rows} rows')
        seen = []
        for k in kernels[str(rows)]:
            if k['launches_per_cycle'] == BOUNDARIES and k['kernel'].startswith('chop_deepseek41_hc_'):
                if len(k['grids']) != 1:
                    raise ValueError('a profiled HC kernel ran at more than one grid')
                seen.append([k['kernel'], k['grids'][0]['grid']])
        if sorted(seen) != served:
            raise ValueError(f'lab served plans differ from the profile at {rows} rows: {served} {seen}')
        result[rows] = served
    return result


def command(root, selector):
    return [str(root / 'snapshot/hc-cols-tests'), '--exact', selector, '--include-ignored', '--nocapture',
            '--test-threads=1']


def environment(root, ordinal=None):
    c = json.loads((root / 'staged.json').read_text())
    env = dict(os.environ, **c['environment'])
    for name in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS', 'CHOPINFER_KERNEL_SHARED_DEVICE',
                 'CHOPINFER_HC_COLS_EXPORT', 'CHOPINFER_HC_COLS_SANITIZER_KINDS', 'KERNELS_LOCK_UPDATE'):
        env.pop(name, None)
    if ordinal is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(ordinal), CHOPINFER_KERNEL_REQUIRE='1')
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return c, env


def sanitizer(c, tool):
    argv = [c['sanitizer'], '--tool', tool, '--error-exitcode', '86', '--target-processes', 'all',
            '--require-cuda-init', 'yes', '--destroy-on-device-error', 'context', '--launch-timeout', '120']
    if tool == 'racecheck':
        argv += ['--racecheck-report', 'all']
    return argv


def clean(text, tool):
    if tool == 'memcheck' and text.count('ERROR SUMMARY: 0 errors') != 1:
        raise ValueError('missing clean memcheck')
    if tool == 'racecheck':
        found = re.findall(r'RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)', text)
        if found != [('0', '0', '0')] and not (not found and text.count('ERROR SUMMARY: 0 errors') == 1):
            raise ValueError('missing clean racecheck')


def ncu_argv(c, ordinal, report):
    argv = [c['ncu'], '--target-processes', 'all', '--devices', str(ordinal), '-k', 'regex:' + NCU_REGEX,
            '--cache-control', 'all', '--clock-control', 'none']
    for s in NCU_SECTIONS:
        argv += ['--section', s]
    return [*argv, '--metrics', ','.join(NCU_METRICS), '-f', '-o', str(report)]


def check_metrics(c):
    """Every requested metric, suffix included, exists on this chip (CPU only)."""
    bases = sorted({m.split('.')[0] for m in NCU_METRICS})
    listed = subprocess.run([c['ncu'], '--query-metrics-mode', 'suffix', '--metrics', ','.join(bases), '--chip',
                             'gb202'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    names = {line.split()[0] for line in listed.stdout.splitlines() if line.strip()}
    missing = [m for m in NCU_METRICS if m not in names]
    if listed.returncode or missing:
        raise ValueError(f'unknown ncu metrics: {missing}')


def ncu_join(raw_text, order):
    """Map each profiled launch to its case, rows and arm by launch order."""
    rows = [r for r in csv.reader(raw_text.splitlines()) if r]
    head, data = rows[0], rows[2:]
    col = {n: i for i, n in enumerate(head)}
    for metric in NCU_METRICS:
        if metric not in col:
            raise ValueError('ncu raw page lacks ' + metric)
    data.sort(key=lambda r: int(r[col['ID']]))
    regex = re.compile(NCU_REGEX)
    want = [(o, symbol) for o in order for symbol in o['symbols'] if regex.match(symbol)]
    if len(want) != len(data) or not want:
        raise ValueError(f'ncu profiled {len(data)} launches, expected {len(want)}')
    joined = []
    for (o, symbol), r in zip(want, data):
        if r[col['Kernel Name']] != symbol:
            raise ValueError(f'ncu launch order differs: {r[col["Kernel Name"]]} for {symbol}')
        values = {}
        for m in NCU_METRICS + ('launch__registers_per_thread',):
            try:
                values[m] = float(r[col[m]].replace(',', '')) if m in col else None
            except ValueError:
                values[m] = None
        joined.append(dict(case=o['case'], rows=o['rows'], arm=o['arm'], symbol=symbol, metrics=values))
    return joined


def ncu_table(joined):
    """Per (rows, arm): the boundary's summed time and traffic."""
    arms = {}
    for j in joined:
        a = arms.setdefault((j['rows'], j['arm']), dict(rows=j['rows'], arm=j['arm'], launches=0, us=0.0,
                                                         dram_read_mb=0.0, dram_write_mb=0.0, l2_mb=0.0,
                                                         kernels=[]))
        m = j['metrics']
        a['launches'] += 1
        a['us'] += m['gpu__time_duration.sum'] / 1e3
        a['dram_read_mb'] += m['dram__bytes_op_read.sum'] / 1e6
        a['dram_write_mb'] += m['dram__bytes_op_write.sum'] / 1e6
        a['l2_mb'] += m['lts__t_bytes.sum'] / 1e6
        a['kernels'].append(dict(symbol=j['symbol'], us=round(m['gpu__time_duration.sum'] / 1e3, 2),
                                 registers=m['launch__registers_per_thread'],
                                 dram_pct=m['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
                                 warps_active_pct=m['sm__warps_active.avg.pct_of_peak_sustained_active'],
                                 long_scoreboard=m['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'],
                                 barrier=m['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio']))
    return [dict(v, us=round(v['us'], 2), dram_read_mb=round(v['dram_read_mb'], 3),
                 dram_write_mb=round(v['dram_write_mb'], 3), l2_mb=round(v['l2_mb'], 3))
            for _, v in sorted(arms.items())]


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
        if m['name'] in c['served_modules'] and c['served_modules'][m['name']] != m['sha256']:
            raise ValueError('served module differs from the profiled install: ' + m['name'])
    if not set(c['served_modules']) <= {m['name'] for m in modules}:
        raise ValueError('a served module is not staged')
    listing = subprocess.run([str(root / 'snapshot/hc-cols-tests'), '--list'], env=env, text=True,
                             capture_output=True, check=True).stdout
    for selector in SELECTORS.values():
        if f'{selector}: test' not in listing:
            raise ValueError('missing test ' + selector)
    texts = {}
    for index, selector in enumerate(CPU):
        result = subprocess.run(command(root, selector), env=env, text=True, capture_output=True, timeout=600)
        text = result.stdout + result.stderr
        (out / f'cpu-{index}.log').write_text(text)
        if result.returncode or not passed(text):
            raise ValueError('CPU test failed: ' + selector)
        texts[selector] = text
    kernels = json.loads((root / 'snapshot/profiled-hc.json').read_text())
    layout = profiled(texts[CPU[0]], kernels)
    coverage = json.loads((root / 'snapshot/cubins/coverage.json').read_text())
    if (len(coverage['cases']) != 5 * BOUNDARIES or len(coverage['variants']) != 21
            or coverage['matrix_reports'] != 5 * BOUNDARIES * 21 * 2):
        raise ValueError('coverage differs')
    check_metrics(c)
    # The ncu argument vector parses (ncu runs /bin/true, which launches nothing).
    probe = subprocess.run([*ncu_argv(c, 0, out / 'argv-probe'), '/bin/true'], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)
    (out / 'ncu-argv.log').write_text(probe.stdout)
    if probe.returncode or 'ERROR' in probe.stdout:
        raise ValueError('ncu argument vector refused')
    if sum(c['stage_seconds'].values()) + c['preflight_and_cleanup_seconds'] > c['outer_seconds']:
        raise ValueError('stage timeouts exceed the job')
    value = dict(cpu_tests=len(CPU), modules=len(modules), symbols=sum(len(m['symbols']) for m in modules),
                 cases=len(coverage['cases']), profiled_layout={str(k): v for k, v in layout.items()},
                 gpu_launched=False)
    (out / 'preflight.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def stage(c, env, root, out, ordinal, kind):
    """The command and environment of one GPU stage."""
    env = dict(env)
    if kind in ('memcheck', 'racecheck'):
        env['CHOPINFER_HC_COLS_SANITIZER_REPORT'] = str(out / f'{kind}-{ordinal}.json')
        if kind == 'racecheck':
            env['CHOPINFER_HC_COLS_SANITIZER_KINDS'] = ','.join(RACECHECK_KINDS)
        return sanitizer(c, kind) + command(root, SELECTORS['sanitizer']), env
    if kind == 'matrix':
        env['CHOPINFER_HC_COLS_REPORT'] = str(out / f'matrix-{ordinal}.json')
        return command(root, SELECTORS['matrix']), env
    if kind == 'timing':
        env['CHOPINFER_HC_COLS_TIMING_REPORT'] = str(out / f'timing-{ordinal}.json')
        return command(root, SELECTORS['timing']), env
    env['CHOPINFER_HC_COLS_PROFILE_REPORT'] = str(out / f'ncu-order-{ordinal}.json')
    return ncu_argv(c, ordinal, out / f'ncu-{ordinal}') + command(root, SELECTORS['profile']), env


def run(root, out, ordinal):
    c, env = environment(root, ordinal)
    env['CHOPINFER_KERNEL_RECEIPTS'] = str(out / f'receipts-{ordinal}')
    failed = False
    # Cheap stages first: a memcheck fault ends the GPU in its first minute.
    for kind in STAGES:
        label = f'{kind}-{ordinal}'
        argv, stage_env = stage(c, env, root, out, ordinal, kind)
        start = time.monotonic()
        with (out / (label + '.log')).open('w') as f:
            try:
                rc = subprocess.run(argv, env=stage_env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=c['stage_seconds'][kind], check=False).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        if kind == 'ncu' and rc == 0:
            result = subprocess.run([c['ncu'], '--import', str(out / f'ncu-{ordinal}.ncu-rep'), '--page', 'raw',
                                     '--csv', '--print-units', 'base'], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=300)
            (out / f'ncu-{ordinal}-raw.csv').write_text(result.stdout)
            rc = result.returncode
        (out / (label + '.exit.json')).write_text(json.dumps(dict(exit_code=rc, elapsed_seconds=time.monotonic() - start,
                                                                  command=argv)) + '\n')
        failed |= rc != 0
        if rc != 0 and kind == 'memcheck':
            break
    return int(failed)


def step_totals(records):
    """Per (rows, variant, hot): the per-boundary mean over sites and the
    saving of one step (77 boundaries per rank), in microseconds."""
    groups = {}
    for r in records:
        groups.setdefault((r['rows'], r['variant'], r['hot']), []).append(r)
    result = []
    for (rows, variant, hot), rs in sorted(groups.items()):
        sites = {r['site'] for r in rs}
        if len(sites) != len(rs):
            raise ValueError('a site was timed twice')
        ref = statistics.fmean(r['reference_us'] for r in rs)
        cand = statistics.fmean(r['candidate_us'] for r in rs)
        result.append(dict(rows=rows, variant=variant, hot=hot, sites=len(rs), reference_us=round(ref, 3),
                           candidate_us=round(cand, 3), saved_us_per_step=round(BOUNDARIES * (ref - cand), 1),
                           faster=sum(r['candidate_us'] < r['reference_us'] for r in rs),
                           ratio_median=round(statistics.median(r['paired_ratio'] for r in rs), 4)))
    return result


def check_matrix(value, sanitizer, coverage, kinds=None):
    """Every report and control passed, complete for the run's cases.
    Returns the served outputs' digests by (case, configuration)."""
    variants = coverage['variants']
    if value['passed'] is not True or value['sanitizer'] is not sanitizer or value['variants'] != variants:
        raise ValueError('matrix failed')
    if sanitizer:
        cases = [x for x in coverage['sanitizer_cases'] if x.split(':', 1)[0] in kinds]
        configurations = ('served',)
    else:
        cases = [x['id'] for x in coverage['cases']]
        configurations = ('served', 'other')
    reports = value['reports']
    expected = {(c, v, g) for c in cases for v in variants for g in configurations}
    if len(reports) != len(expected) or {(r['case'], r['variant'], r['configuration']) for r in reports} != expected \
            or not all(r['passed'] for r in reports):
        raise ValueError('incomplete or failing reports')
    served = {(r['case'], r['configuration']): r['output_sha256'] for r in value['served']}
    if set(served) != {(c, g) for c in cases for g in configurations}:
        raise ValueError('incomplete served digests')
    if not sanitizer:
        kinds_count = len(KINDS)
        if value['negatives'] != NEGATIVES_PER_KIND * kinds_count or not value['checks'] \
                or not all(c['passed'] for c in value['checks']):
            raise ValueError('controls incomplete')
        controls = {c['case'].rsplit(':', 1)[0] for c in value['checks'] if c['name'] == 'graph_replay1'}
        if len(controls) != kinds_count * len(variants):
            raise ValueError('a candidate missed its controls')
        if len(value['captured']) != len(cases) or not all(c['passed'] for c in value['captured']):
            raise ValueError('captured outputs differ')
    return served


def analyze(root, out):
    c, _ = environment(root)
    coverage = json.loads((root / 'snapshot/cubins/coverage.json').read_text())
    results, digests = [], []
    for ordinal in (0, 1):
        for kind in STAGES:
            if json.loads((out / f'{kind}-{ordinal}.exit.json').read_text())['exit_code'] != 0:
                raise ValueError(f'{kind}-{ordinal} failed')
        for tool in ('memcheck', 'racecheck'):
            clean((out / f'{tool}-{ordinal}.log').read_text(), tool)
        check_matrix(json.loads((out / f'memcheck-{ordinal}.json').read_text()), True, coverage, tuple(KINDS))
        check_matrix(json.loads((out / f'racecheck-{ordinal}.json').read_text()), True, coverage, RACECHECK_KINDS)
        matrix = json.loads((out / f'matrix-{ordinal}.json').read_text())
        digests.append(check_matrix(matrix, False, coverage))
        timing = json.loads((out / f'timing-{ordinal}.json').read_text())
        if len(timing['records']) != coverage['timing_records'] or \
                {r['case'] for r in timing['records']} != set(coverage['timing_cases']):
            raise ValueError('timing incomplete')
        order = json.loads((out / f'ncu-order-{ordinal}.json').read_text())['order']
        if len(order) != coverage['profile_arms']:
            raise ValueError('profile incomplete')
        joined = ncu_join((out / f'ncu-{ordinal}-raw.csv').read_text(), order)
        results.append(dict(ordinal=ordinal, checks=len(matrix['checks']), negatives=matrix['negatives'],
                            reports=len(matrix['reports']), captured=len(matrix['captured']),
                            timing=step_totals(timing['records']), ncu=ncu_table(joined)))
    if digests[0] != digests[1]:
        raise ValueError('the GPUs disagree on outputs')
    value = dict(complete=True, cases=len(coverage['cases']), results=results)
    (out / 'analysis.json').write_text(json.dumps(value, indent=2) + '\n')
    lines = ['# Wide-bucket HC candidates against the served boundary on real captured rows', '',
             f"{len(coverage['cases'])} cases per GPU, every candidate bitwise in both configurations, served "
             'outputs equal across GPUs, memcheck and racecheck clean.', '',
             '| GPU | Rows | Variant | Hot | Sites | Served us | Candidate us | Saved us/step | Faster | Ratio |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for t in r['timing']:
            lines.append(f"| {r['ordinal']} | {t['rows']} | {t['variant']} | {t['hot']} | {t['sites']} | "
                         f"{t['reference_us']} | {t['candidate_us']} | {t['saved_us_per_step']} | {t['faster']} | "
                         f"{t['ratio_median']} |")
    lines += ['', '| GPU | Rows | Arm | Launches | ncu us | DRAM read MB | DRAM write MB | L2 MB |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        for n in r['ncu']:
            lines.append(f"| {r['ordinal']} | {n['rows']} | {n['arm']} | {n['launches']} | {n['us']} | "
                         f"{n['dram_read_mb']} | {n['dram_write_mb']} | {n['l2_mb']} |")
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
