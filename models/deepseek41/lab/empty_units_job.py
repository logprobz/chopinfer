#!/usr/bin/env python3
"""Bounded pair checks and timing for ordered empty-unit prefill scheduling."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback

import prefill_attention_job as logs
import prefill_cache_job as capture

ROOT = Path(__file__).resolve().parent
OFFSETS = (0, 8192, 16384, 24576)
KEYS = ('rank', 'layer', 'case', 'first', 'rows', 'decoded', 'served_fused')


def fields(line):
    pairs = re.findall(r'(\w+)=([^\s]+)', line)
    if len(dict(pairs)) != len(pairs):
        raise ValueError('duplicate fields')
    return dict(pairs)


def cases(offset, small=False):
    result = []
    if small:
        shapes = [(layer, 'control', 4095, 65, decoded)
                  for layer in (2, 20) for decoded in (False, True)]
    else:
        shapes = [(layer, 'full', 0, 8192, False) for layer in (2, 20)]
        if offset > 0:
            shapes += [(layer, 'cached', 0, 8192, True) for layer in (2, 20)]
        shapes += [(layer, 'retained', 8192-(64+128*(39-layer)), 64+128*(39-layer), False)
                   for layer in range(20, 40)]
        shapes += [(layer, 'part', first, 4096, False)
                   for first in (0, 4096) for layer in (2, 20)]
    for rank in range(2):
        for layer, name, first, rows, decoded in shapes:
            served = (name == 'full' and offset == 0) or name == 'cached' or (name in ('retained', 'part') and offset in (0, 24576)
                                       and (rows == 4096 or rows >= 320))
            result.append(dict(rank=str(rank), offset=str(offset), layer=str(layer), case=name,
                               first=str(first), rows=str(rows), decoded=str(decoded).lower(),
                               served_fused=str(served).lower()))
    return result


def parse(text, offset, timing=False, small=False, hashes=None):
    logs.clean_gpu_output(text)
    if not timing and 'TIMING empty_units ' in text:
        raise ValueError('timing appeared before the exactness gate')
    expected = {tuple(r[k] for k in KEYS): r for r in cases(offset, small)}
    marker = 'TIMING' if timing else 'PASS'
    seen = {}
    for match in re.finditer(rf'{marker} empty_units ([^\n]+)', text):
        row = fields(match[1])
        key = tuple(row[k] for k in KEYS)
        if key not in expected or key in seen or any(row[k] != v for k, v in expected[key].items()):
            raise ValueError('unexpected, duplicate or mislabelled case')
        if timing:
            if any(row[k] != v for k, v in dict(samples='3', warmup='1', producer_included='true').items()):
                raise ValueError('timing configuration differs')
            for key_name in ('served_us', 'fused_us', 'skip_reference_us', 'candidate_us'):
                row[key_name] = float(row[key_name])
                if not math.isfinite(row[key_name]) or not 0 < row[key_name] < 1e9:
                    raise ValueError('invalid timing')
        else:
            for flag in ('partial', 'output', 'poison', 'cache_guards', 'actual_selected'):
                if row[flag] != 'true':
                    raise ValueError('failed exactness flag')
            if (row['audit_replays'] != '6' or row['timed_replays'] != '6'
                    or row['route_replays'] != '2' or row['partitions'] != '2'):
                raise ValueError('incomplete replay coverage')
            if row['adversarial_replays'] != ('2' if small else '0'):
                raise ValueError('incomplete invalid-input coverage')
            if not re.fullmatch('[0-9a-f]{64}', row['selected_sha256']):
                raise ValueError('missing selection identity')
            if hashes is not None and row['selected_sha256'] != hashes[(int(row['layer']), int(row['first']), int(row['rows']))]:
                raise ValueError('selection identity differs')
        seen[key] = row
    if set(seen) != set(expected):
        raise ValueError(f'incomplete cases: {len(seen)} of {len(expected)}')
    return list(seen.values())


def input_hashes(manifest, offset, small=False):
    path = Path(manifest)
    data = json.loads(path.read_text())
    files = {}
    result = {}
    for row in cases(offset, small):
        layer, first, rows = (int(row[k]) for k in ('layer', 'first', 'rows'))
        if layer not in files:
            ref = next(r for r in data['records'] if r['offset'] == offset and r['layer'] == layer)['selected']
            raw = capture.capture_file(path.parent, ref['path']).read_bytes()
            if len(raw) != 8192*512*4 or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('captured file differs')
            files[layer] = raw
        result[(layer, first, rows)] = hashlib.sha256(files[layer][first*2048:(first+rows)*2048]).hexdigest()
    return result


def selftest(recorded):
    """Exercise the decoder on recorded output, then strict new-schema fixtures."""
    text = Path(recorded).read_text()
    old = [fields(m[1]) for m in re.finditer(r'TIMING head_finish ([^\n]+)', text)]
    if len(old) != 52:
        raise ValueError('recorded endpoint must contain 52 timing cases')
    for row in old:
        if row['samples'] != '3' or row['warmup'] != '1':
            raise ValueError('recorded timing configuration differs')
        for name in ('current_total_us', 'head_total_us', 'fused_reference_us', 'fused_total_us'):
            if not math.isfinite(float(row[name])) or float(row[name]) <= 0:
                raise ValueError('recorded time invalid')
    for offset in OFFSETS:
        for small in (False, True):
            for timing in (False, True):
                lines = []
                for i, row in enumerate(cases(offset, small)):
                    label = ' '.join(f'{k}={v}' for k, v in row.items())
                    if timing:
                        history = old[i % len(old)]
                        tail = (f'served_us={history["current_total_us"]} fused_us={history["head_total_us"]} '
                                f'skip_reference_us={history["fused_reference_us"]} candidate_us={history["fused_total_us"]} '
                                'samples=3 warmup=1 producer_included=true')
                    else:
                        tail = ('partial=true output=true poison=true audit_replays=6 timed_replays=6 route_replays=2 '
                                f'adversarial_replays={2 if small else 0} '
                                'partitions=2 cache_guards=true actual_selected=true selected_sha256='+'0'*64)
                    lines.append(f'{"TIMING" if timing else "PASS"} empty_units {label} {tail}')
                fixture = '\n'.join(lines)
                assert len(parse('libtest prefix '+fixture, offset, timing, small)) == len(lines)
                for bad in ('', '\n'.join(lines[:-1]), fixture+'\n'+lines[0], fixture.replace('rank=0', 'rank=8'),
                            fixture.replace('rows=', 'rows=1', 1), fixture+'\nFAILED explicit control'):
                    try:
                        parse(bad, offset, timing, small)
                    except (ValueError, AssertionError, KeyError):
                        pass
                    else:
                        raise AssertionError('bad record accepted')
    return {'recorded_cases': len(old), 'schema_cases': 16, 'rejected_mutations': 96}


def main():
    args = sys.argv[1:]
    dry = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']
    if len(args) != 2 or args[0] not in ('pair', 'memcheck') or int(args[1]) not in OFFSETS:
        raise ValueError('usage: empty_units_job.py pair|memcheck OFFSET [--dry-run]')
    mode, offset = args[0], int(args[1])
    small = mode == 'memcheck'
    config = json.loads((ROOT/'job-config.json').read_text())
    out = Path(os.environ['O2_RUN_DIR'])
    started = time.time()
    deadline = time.monotonic()+config['work_seconds']
    verdict = dict(exit_code=1, mode=mode, offset=offset, dry_run=dry)
    def step(name, command, timeout, **kwargs):
        text, _ = logs.run_step(out, name, command, min(timeout, max(.001, deadline-time.monotonic())), **kwargs)
        return text+'\n'+(out/f'{name}.stderr.log').read_text()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    def power(name):
        text = step('power-'+name, ['nvidia-smi', '-i', visible, '--query-gpu=index,power.limit', '--format=csv,noheader,nounits'], 15)
        rows = [r.split(',') for r in text.splitlines() if r.strip()]
        if len(rows) != 2 or any(float(r[1]) != 250 for r in rows):
            raise ValueError('expected assigned pair at 250 W')
    def interrupted(sig, frame):
        raise InterruptedError(f'job signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if hashlib.sha256(Path(config['recorded_timing']).read_bytes()).hexdigest() != config['recorded_sha256']:
            raise ValueError('recorded parser input changed')
        verdict['parser'] = selftest(config['recorded_timing'])
        manifest = capture.full_capture_manifest(Path(config['capture_root']))
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != config['capture_sha256']:
            raise ValueError('captured manifest identity changed')
        hashes = input_hashes(manifest, offset, small)
        binary, cubins = ROOT/'snapshot/bin/prefill-attention', ROOT/'snapshot/cubins'
        step('snapshot', [str(binary), 'empty-snapshot', str(cubins)], 30)
        if dry:
            verdict['exit_code'] = 0
        else:
            if len(visible.split(',')) != 2:
                raise ValueError('assigned GPU pair required')
            power('before')
            idle = step('idle', ['nvidia-smi', '-i', visible, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], 15)
            if idle.strip():
                raise ValueError('assigned pair is occupied')
            env = dict(os.environ, CHOPINFER_KERNEL_GPUS='0,1', CHOPINFER_KERNEL_REQUIRE='1',
                       CHOPINFER_KERNEL_RECEIPTS=str(out/'receipts'))
            for name in ('CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_SHARED_DEVICE', 'KERNELS_LOCK_UPDATE'):
                env.pop(name, None)
            step('resources', [sys.executable, '-B', str(ROOT/'prefill_attention_resources.py')], 30, env=env)
            resource_rows = json.loads((out/'resources.stdout.log').read_text())['ranks']
            if len(resource_rows) != 2 or any(len(r['functions']) != 8 for r in resource_rows):
                raise ValueError('incomplete occupancy inventory')
            verdict['resources'] = resource_rows
            phases = ['memcheck'] if small else ['check', 'time']
            for phase in phases:
                seconds = config['phase_seconds'][phase]
                command = [str(binary), 'empty-'+phase, str(cubins), str(manifest), str(offset)]
                if small:
                    command = [config['sanitizer'], '--tool', 'memcheck', '--error-exitcode', '99', '--target-processes', 'all', *command]
                text = step(phase, command, seconds+10, env=dict(env, CHOPINFER_KERNEL_TIMEOUT_SECS=str(seconds)))
                records = parse(text, offset, phase == 'time', small, hashes)
                if small and not re.search(r'ERROR SUMMARY: 0 errors', text):
                    raise ValueError('missing clean memcheck summary')
                verdict[phase] = records
                (out/'progress.json').write_text(json.dumps(verdict, indent=2)+'\n')
            verdict['exit_code'] = 0
    except BaseException as error:
        verdict['error'] = repr(error)
        traceback.print_exc()
    finally:
        if not dry:
            # Cleanup has its own reserve after the bounded work phase.
            deadline = time.monotonic()+45
            try:
                power('after')
                text = step('journal', ['journalctl', '-k', '--since', f'@{int(started)}', '--no-pager'], 25)
                xids = re.findall(r'^.*NVRM:.*Xid.*$', text, re.MULTILINE)
                (out/'xids.txt').write_text('\n'.join(xids))
                if xids:
                    raise ValueError('kernel Xid observed')
            except BaseException as error:
                verdict.update(exit_code=1, cleanup_error=repr(error))
        verdict['elapsed_seconds'] = time.time()-started
        (out/'verdict.json').write_text(json.dumps(verdict, indent=2)+'\n')
    print(json.dumps(verdict))
    raise SystemExit(verdict['exit_code'])


if __name__ == '__main__':
    main()
