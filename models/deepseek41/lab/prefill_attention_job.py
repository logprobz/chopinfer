#!/usr/bin/env python3
"""Run the allocated software profiling pair and reject incomplete evidence."""
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / 'snapshot/bin/prefill-attention'
CUBINS = ROOT / 'snapshot/cubins'
SOURCES = ('mixed', 'window', 'selected')
MODES = ('full', 'loads', 'resident')
STAGES = ('query_setup', 'gather_decode_stage', 'qk', 'softmax', 'pv', 'epilogue', 'total')

def parse(text):
    assert not re.search(r'\bFAILED\b|panicked at|SKIP|skipping|did not execute|ERR_NVGPUCTRPERM', text)
    for failed, ignored in re.findall(r'test result:.*?; (\d+) failed; (\d+) ignored', text):
        assert int(failed) == int(ignored) == 0
    expected_gates = set(itertools.product(range(2), ('window', 'mixed64', 'mixed32')))
    gates = []
    for name, suffix in [('prefill_attention_bitwise', ' f64=true'), ('prefill_ablation_full', '')]:
        matches = list(re.finditer(rf'PASS {name} rank=(\d+) variant=(\w+) rows=8192 workspace=true output=true replay=true{suffix}', text))
        assert len(matches) == 6 and {(int(m[1]), m[2]) for m in matches} == expected_gates, name
        gates.extend(m.end() for m in matches)
    timings = {}
    matches = list(re.finditer(r'TIMING prefill_ablation rank=(\d+) source=(\w+) mode=(\w+) rows=8192 median_us=(\S+)', text))
    assert matches and min(m.start() for m in matches) > max(gates), 'timing preceded a bitwise gate'
    for m in matches:
        key = (int(m[1]), m[2], m[3])
        value = float(m[4])
        assert key not in timings and math.isfinite(value) and value > 0, m[0]
        timings[key] = value
    assert set(timings) == set(itertools.product(range(2), SOURCES, ('reference',) + MODES)), 'incomplete timing matrix'
    stages = {}
    for m in re.finditer(r'STAGES prefill_ablation rank=(\d+) source=(\w+) mode=(\w+) ctas=(\d+) cycles=(\{[^\n]+\})', text):
        key = (int(m[1]), m[2], m[3])
        assert key not in stages and int(m[4]) == 16384, m[0]
        data = json.loads(m[5])
        assert set(data) == set(STAGES) | {'scope'}
        assert 'do not sum across CTAs' in data['scope']
        for field in STAGES:
            median, p95 = data[field]['median'], data[field]['p95']
            assert isinstance(median, int) and isinstance(p95, int) and 0 <= median <= p95, data
            if field in ('total', 'gather_decode_stage'): assert median > 0
            if field in ('qk', 'softmax', 'pv'):
                assert median == p95 == 0 if key[2] == 'loads' else median > 0
        stages[key] = data
    assert set(stages) == set(itertools.product(range(2), SOURCES, MODES)), 'incomplete stage matrix'
    return dict(timings=[dict(rank=k[0], source=k[1], mode=k[2], median_us=v) for k,v in timings.items()],
                stages=[dict(rank=k[0], source=k[1], mode=k[2], cycles=v) for k,v in stages.items()])

def controls():
    lines = []
    for rank, variant in itertools.product(range(2), ('window', 'mixed64', 'mixed32')):
        for name, suffix in [('prefill_attention_bitwise',' f64=true'),('prefill_ablation_full','')]:
            lines.append(f'libtest prefix PASS {name} rank={rank} variant={variant} rows=8192 workspace=true output=true replay=true{suffix}')
    for rank, source in itertools.product(range(2), SOURCES):
        for mode in ('reference',) + MODES:
            lines.append(f'prefix TIMING prefill_ablation rank={rank} source={source} mode={mode} rows=8192 median_us=1.5')
            if mode == 'reference': continue
            data = {name:dict(median=0 if mode == 'loads' and name in ('qk','softmax','pv') else 1,
                              p95=0 if mode == 'loads' and name in ('qk','softmax','pv') else 2) for name in STAGES}
            data['scope'] = 'do not sum across CTAs'
            lines.append(f'prefix STAGES prefill_ablation rank={rank} source={source} mode={mode} ctas=16384 cycles={json.dumps(data)}')
    text = '\n'.join(lines) + '\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n'
    assert len(parse(text)['timings']) == 24
    for bad in [text.replace(lines[0], ''), text + lines[-1], text.replace('median_us=1.5','median_us=nan'),
                text.replace('ctas=16384','ctas=0'), text + '\nFAILED\n', text.replace('0 ignored','1 ignored'),
                lines[12] + '\n' + text]:
        try: parse(bad)
        except (AssertionError, ValueError, KeyError): pass
        else: raise AssertionError('parser accepted corrupt control')
    print('PASS ablation_controls prefixed_markers=true incomplete_duplicate_nan_failed_ignored=true')


def parse_reuse(text, layers):
    assert len(layers) == 40 and sorted(layers).count(0) == 2
    assert not re.search(r'\bFAILED\b|panicked at|SKIP|skipping|did not execute', text)
    modes = ('vector', 'pair')
    variants = ('window', 'mixed64', 'mixed32')
    gates = list(re.finditer(r'PASS prefill_attention_bitwise rank=(\d+) variant=(\w+) rows=8192 workspace=true output=true replay=true f64=true candidate=(\w+) reuse_cases=true', text))
    assert len(gates) == 12
    assert {(int(m[1]),m[2],m[3]) for m in gates} == set(itertools.product(range(2),variants,modes))
    records = {}
    fields = ('old_attention_us','candidate_attention_us','old_merge_us','candidate_merge_us',
              'old_total_us','candidate_total_us','gflop','paired_ratio')
    for match in re.finditer(r'TIMING prefill_attention ([^\n]+)', text):
        assert match.start() > max(g.end() for g in gates), 'timing preceded a bitwise gate'
        row = dict(re.findall(r'(\w+)=([^\s]+)', match[1]))
        key = (int(row['rank']),int(row['layer']),row['candidate'])
        assert key not in records and key[0] in (0,1) and 0 <= key[1] < 40 and key[2] in modes
        assert row['rows'] == '8192' and row['variant'] == variants[layers[key[1]]]
        assert row['warmup'] == '1' and row['samples'] == '3'
        for field in fields:
            row[field] = float(row[field])
            assert math.isfinite(row[field]) and row[field] > 0, (field,row)
        expected = 4 * 8192 * 16 * 512 * (128 if layers[key[1]] == 0 else 640) / 1e9
        assert math.isclose(row['gflop'], expected, rel_tol=1e-9)
        records[key] = row
    assert set(records) == set(itertools.product(range(2),range(40),modes)), 'incomplete all-layer timing matrix'
    totals = []
    for rank, mode in itertools.product(range(2),modes):
        selected = [r for k,r in records.items() if k[0] == rank and k[2] == mode]
        old = sum(r['old_total_us'] for r in selected)
        candidate = sum(r['candidate_total_us'] for r in selected)
        totals.append(dict(rank=rank,candidate=mode,old_total_us=old,candidate_total_us=candidate,
                           ratio=candidate/old,faster=candidate<old))
    return dict(timings=list(records.values()),totals=totals)


def reuse_controls():
    layers = [0]*2 + [2]*18 + [1]*20
    lines = []
    for rank,variant,mode in itertools.product(range(2),('window','mixed64','mixed32'),('vector','pair')):
        lines.append(f'prefix PASS prefill_attention_bitwise rank={rank} variant={variant} rows=8192 workspace=true output=true replay=true f64=true candidate={mode} reuse_cases=true')
    for rank,layer,mode in itertools.product(range(2),range(40),('vector','pair')):
        variant = ('window','mixed64','mixed32')[layers[layer]]
        gflop = 4*8192*16*512*(128 if layers[layer]==0 else 640)/1e9
        lines.append(f'prefix TIMING prefill_attention rank={rank} layer={layer} variant={variant} rows=8192 old_attention_us=10 candidate_attention_us=5 old_merge_us=1 candidate_merge_us=1 old_total_us=11 candidate_total_us=6 gflop={gflop} paired_ratio=0.55 candidate={mode} warmup=1 samples=3')
    text = '\n'.join(lines)
    assert len(parse_reuse(text,layers)['timings']) == 160
    for bad in [text.replace(lines[-1],''),text+'\n'+lines[-1],text.replace('candidate_attention_us=5','candidate_attention_us=nan'),
                text.replace('reuse_cases=true','reuse_cases=false'),text.replace('f64=true','f64=false'),
                lines[12]+'\n'+text,text+'\nFAILED\n',text.replace('samples=3','samples=7')]:
        try: parse_reuse(bad,layers)
        except (AssertionError,ValueError,KeyError): pass
        else: raise AssertionError('all-layer control accepted corrupt output')
    print('PASS reuse_controls all_40_layers=true both_candidates=true bitwise_before_timing=true')

class StepError(RuntimeError):
    pass


def run_step(out, name, args, timeout=30, *, env=None, cwd=None, check=True):
    """Keep diagnostics on disk before checking status, including timeouts."""
    if not re.fullmatch(r'[a-z0-9-]+', name):
        raise ValueError(f'invalid step name: {name}')
    stdout_path, stderr_path = out / f'{name}.stdout.log', out / f'{name}.stderr.log'
    record = dict(command=[str(arg) for arg in args], cwd=str(cwd or Path.cwd()),
                  timeout_seconds=timeout, exit_code=None,
                  stdout=str(stdout_path), stderr=str(stderr_path))
    child, error = None, None
    started = time.monotonic()
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        try:
            child = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=env,
                                     cwd=cwd, start_new_session=True)
            record['pid'] = child.pid
            child.wait(timeout=timeout)
        except BaseException as caught:
            error = caught
            record['error'] = repr(caught)
            traceback.print_exc(file=stderr)
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            if child is not None:
                record['exit_code'] = child.returncode
            record['elapsed_seconds'] = time.monotonic() - started
            (out / f'{name}.step.json').write_text(json.dumps(record, indent=2) + '\n')
    if error is not None or (check and record['exit_code'] != 0):
        raise StepError(f'{name}: exit={record["exit_code"]}; stdout={stdout_path}; stderr={stderr_path}') from error
    return stdout_path.read_text(), record['exit_code']

def main():
    flags = set(sys.argv[1:])
    assert flags <= {'--dry-run','--resources-only','--reuse'} and len(flags) == len(sys.argv[1:])
    dry = '--dry-run' in flags
    resources_only = '--resources-only' in flags
    reuse = '--reuse' in flags
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','')
    start = int(time.time())
    out = Path(os.environ.get('L11_ABLATION_RUN_DIR', ROOT / f'{"dry" if dry else "run"}-{start}-{os.getpid()}'))
    out.mkdir(exist_ok=True)
    print(f'RUN prefill_ablation output={out}', flush=True)
    result = {'exit_code':1, 'dry_run':dry, 'resources_only':resources_only,
              'visible_devices':visible, 'device_order':os.environ.get('CUDA_DEVICE_ORDER')}
    def step(name, args, timeout=30, **kwargs):
        return run_step(out, name, args, timeout, **kwargs)[0]
    def interrupted(signum, frame): raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    def power(stage):
        data = step(f'power-{stage}', ['nvidia-smi','-i',visible,'--query-gpu=index,power.limit','--format=csv,noheader,nounits'])
        (out/f'power-{stage}.csv').write_text(data)
        rows = list(csv.reader(data.splitlines()))
        assert len(rows) == 2 and all(float(row[1]) == 250 for row in rows), rows
    try:
        step('hashes', ['sha256sum','--check','--quiet',str(ROOT/'SHA256SUMS')], cwd=ROOT)
        controls()
        if reuse: reuse_controls()
        manifest = json.loads((CUBINS/'snapshot.json').read_text())
        requests = json.loads((ROOT/'resource-requests.json').read_text())
        checked = step('snapshot', [str(BINARY),'check',str(CUBINS)])
        assert f'PASS prefill_attention_snapshot modules={len(manifest["modules"])} entries={len(requests)+4} probe_sm120=true' in checked
        if dry:
            result['exit_code'] = 0
            print('PASS ablation_job_dry_run no_gpu_access=true')
            return
        assert visible and len(visible.split(',')) == 2, 'runner must allocate an idle GPU pair'
        power('before')
        processes = step('processes-before', ['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'])
        (out/'processes-before.txt').write_text(processes)
        assert not processes.strip(), 'allocated GPUs are not idle'
        env = dict(os.environ)
        for key in ('CHOPINFER_KERNEL_GPU','CHOPINFER_KERNEL_GPUS','CHOPINFER_KERNEL_SHARED_DEVICE','KERNELS_LOCK_UPDATE'):
            env.pop(key,None)
        for name in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING'):
            assert not env.get(name), f'profile refuses {name}'
        gpu_timeout = 700 if reuse else 1050
        env.update(CHOPINFER_KERNEL_GPUS='0,1',CHOPINFER_KERNEL_REQUIRE='1',CHOPINFER_KERNEL_TIMEOUT_SECS=str(gpu_timeout),
                   CHOPINFER_KERNEL_RECEIPTS=str(out/'receipts'))
        resources = step('resources', [sys.executable,'-B',str(ROOT/'prefill_attention_resources.py')],90,env=env)
        (out/'resources.json').write_text(resources)
        data = json.loads(resources)
        assert [r['rank'] for r in data['ranks']] == [0,1] and all(len(r['functions']) == len(requests) for r in data['ranks'])
        if not resources_only:
            log = step('gpu', [str(BINARY),'run-reuse' if reuse else 'ablation',str(CUBINS)],gpu_timeout,env=env)
            log += '\n' + (out/'gpu.stderr.log').read_text()
            result.update(parse_reuse(log,manifest['layers']) if reuse else parse(log))
        result['exit_code'] = 0
    except BaseException as error:
        result['error'] = repr(error)
        with (out/'wrapper.stderr.log').open('a') as log:
            traceback.print_exc(file=log)
    finally:
        if not dry:
            try: power('after')
            except Exception as error:
                result['power_error'],result['exit_code'] = repr(error),1
            try:
                journal, code = run_step(out, 'journal', ['journalctl','-k','--since',f'@{start}','--no-pager'],check=False)
                xids = re.findall(r'^.*NVRM:.*Xid.*$',journal,re.MULTILINE)
                (out/'xids.txt').write_text('\n'.join(xids))
                result['journal_exit_code'] = code
                if code or xids: result['exit_code'] = 1
            except Exception as error:
                result['journal_error'],result['exit_code'] = repr(error),1
        (out/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(out),'exit_code':result['exit_code'],'error':result.get('error')}))
    raise SystemExit(result['exit_code'])

if __name__ == '__main__': main()
