#!/usr/bin/env python3
"""Qualify a frozen prefill cache binding against its serving base."""
import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.request

from prefill_attention_job import run_step

ROOT = Path(__file__).resolve().parent
MODEL = 'DeepSeek-V4.1-Flash'
OFFSETS = (0, 8192, 16384, 24576)
EMBED = 'chop_row_embedding_deepseek41_tp4_v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def canonical(response):
    assert response['usage']['prompt_tokens'] == 32768
    assert response['usage']['completion_tokens'] == 1
    assert (response['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) == 0
    assert len(response['choices']) == 1
    row = response['choices'][0]
    assert len(row['token_ids']) == 1 and type(row['token_ids'][0]) is int
    lp = row['logprobs']
    assert all(len(lp[key]) == 1 for key in ('tokens', 'token_logprobs', 'top_logprobs'))
    assert 1 <= len(lp['top_logprobs'][0]) <= 5
    def bits(value):
        assert type(value) in (float, int) and math.isfinite(value)
        return struct.pack('>d', value).hex()
    return dict(ids=row['token_ids'], tokens=lp['tokens'],
                selected=[bits(v) for v in lp['token_logprobs']],
                top=[{k: bits(v) for k, v in sorted(lp['top_logprobs'][0].items())}])


def trace_rows(text):
    rows = []
    for line in text.splitlines():
        if '"chopinfer.step.v1"' in line:
            rows.append(json.loads(line[line.index('{'):]))
    return rows


def symbol(name):
    return name.removeprefix('deepseek41_')


def union_ms(intervals):
    end = 0
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def analyze(path, cached):
    chunks = []
    with sqlite3.connect(path) as db:
        devices = [r[0] for r in db.execute('SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId')]
        assert devices == [0, 1, 2, 3], devices
        for rank in devices:
            raw = db.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=? ORDER BY k.start,k.end', (rank,)).fetchall()
            rows = [(a, b, symbol(n)) for a, b, n in raw]
            starts = [i for i, r in enumerate(rows) if r[2] == EMBED]
            assert len(starts) == 4, (rank, len(starts))
            starts.append(len(rows))
            for offset, first, last in zip(OFFSETS, starts, starts[1:]):
                block = rows[first:last]
                frontier = block[0][0]
                exclusive = collections.Counter()
                pending, layers = [], []
                for start, end, name in block:
                    exclusive[name] += max(0, end - max(start, frontier)) / 1e6
                    frontier = max(frontier, end)
                    if name.startswith('chop_latent_prefill_'):
                        pending.append((start, end, name))
                    elif name.startswith('chop_latent_sources_') and '_merge_' not in name:
                        layer = len(layers)
                        variant = 'window' if '_window_' in name else 'mixed32' if '_p32_' in name else 'mixed64'
                        decoded = cached and (variant == 'window' or offset > 0)
                        assert ('_decoded_' in name) == decoded, (rank, offset, layer, name)
                        expected = []
                        if decoded:
                            expected = ['chop_latent_prefill_clear_pages_v1']
                            if variant != 'window':
                                expected.append(f'chop_latent_prefill_mark_pages_p{32 if variant == "mixed32" else 64}_v1')
                            expected.append('chop_latent_prefill_decode_window_d512_v1')
                            if variant != 'window':
                                expected.append(f'chop_latent_prefill_decode_pages_d512_p{32 if variant == "mixed32" else 64}_v1')
                        assert [r[2] for r in pending] == expected, (rank, offset, layer)
                        layers.append(dict(layer=layer, variant=variant, decoded=decoded,
                                           attention_ms=(end-start)/1e6,
                                           cache_ms=union_ms([(a, b) for a, b, _ in pending]),
                                           spans=[(a, b) for a, b, _ in pending] + [(start, end)]))
                        pending = []
                    elif name.startswith('chop_latent_sources_') and '_merge_' in name:
                        assert layers and 'merge_ms' not in layers[-1]
                        layers[-1]['merge_ms'] = (end-start)/1e6
                        layers[-1]['spans'].append((start, end))
                assert not pending and len(layers) == 40
                assert collections.Counter(l['variant'] for l in layers) == dict(window=2, mixed32=18, mixed64=20)
                for layer in layers:
                    assert 'merge_ms' in layer
                    layer['combined_union_ms'] = union_ms(layer.pop('spans'))
                chunks.append(dict(rank=rank, offset=offset, layers=layers,
                                   gpu_envelope_ms=(frontier-block[0][0])/1e6,
                                   exclusive_ms=dict(exclusive)))
    return chunks


def http(url, path, body=None, timeout=5):
    request = urllib.request.Request(url+path, None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def kill_group(pid, process=None):
    assert pid != os.getpgrp()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    if process is not None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
    if process is not None:
        process.wait(timeout=5)


def interrupted(number, frame):
    raise InterruptedError(f'signal {number}')


def arm(out, name, config):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(900)
    request_ids = json.loads((ROOT/'prompt.json').read_text())
    assert len(request_ids) == 32768
    install, cache = Path(config['install']), Path(config['cache'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('CHOPINFER_TAPS_', 'CHOPINFER_DEEPSEEK41_', 'CHOPINFER_SAMPLING_CAPTURE')):
            env.pop(key)
    env.update(CHOPINFER_STEP_TRACE='1', CHOPINFER_STEP_PROFILE_EVERY='1',
               CHOPINFER_SAMPLING_CAPTURE=str(out/'logits.f32'), CHOPINFER_SAMPLING_CAPTURE_ROWS='2')
    serve = [str(install/'bin/chopinfer'), 'serve', '--model', env['CHOPINFER_DEEPSEEK_CHECKPOINT'],
             '--tensor-parallel-size', '4', '--model-cache-dir', str(cache), '--listen', f'127.0.0.1:{port}',
             '--served-model-name', MODEL, '--prefix-cache', 'off', '--max-sequences', '1',
             '--prefill-budget', 'fixed', '--max-batch-tokens', '8192', '--prefill-chunk-tokens', '8192',
             '--drain-timeout-ms', '5000', '--speculative-tokens', '0']
    session = f'l11-cache-{os.getpid()}-{time.time_ns()}'
    nsys = shutil.which('nsys')
    assert nsys, 'nsys is required for serving route validation'
    child = [sys.executable, '-B', str(ROOT/'prefill_cache_serving.py'), '--child', str(out/'server-group.json'), *serve]
    launch = [nsys, 'launch', '--session-new', session, '--trace=cuda', '--cuda-graph-trace=node', '--wait=all', *child]
    write(out/'launch.json', dict(command=launch, environment={k: v for k, v in env.items() if k.startswith('CHOPINFER_STEP_')}))
    process = None
    def utility(label, args, seconds=20, check=True):
        return run_step(out, label, args, seconds, check=check)[0]
    try:
        with (out/'server.stdout.log').open('w') as stdout, (out/'server.stderr.log').open('w') as stderr:
            process = subprocess.Popen(launch, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            deadline = time.monotonic()+600
            while True:
                assert process.poll() is None, 'server exited before readiness'
                try:
                    health = http(url, '/health')
                    models = http(url, '/v1/models')
                    assert MODEL in [v['id'] for v in models['data']]
                    write(out/'health.json', health)
                    break
                except OSError:
                    assert time.monotonic() < deadline, 'server readiness timeout'
                    time.sleep(.2)
            samples = []
            for repeat in range(2):
                response = http(url, '/v1/completions', dict(model=MODEL, prompt=request_ids, temperature=0.7, seed=17,
                                max_tokens=1, ignore_eos=True, logprobs=5, return_token_ids=True, stream=False), 60)
                write(out/f'exact-{repeat}.json', response)
                samples.append(canonical(response))
            assert samples[0] == samples[1], 'cold replay changed reported bits'
            data = (out/'logits.f32').read_bytes()
            assert len(data) == 2*129280*4, ('expected two full F32 vocabulary rows', len(data))
            assert data[:len(data)//2] == data[len(data)//2:], 'cold replay changed F32 logits'
            utility('profile-start', [nsys, 'start', '--session', session, '--sample=none', '--cpuctxsw=none', '--output', str(out/'prefill')])
            try:
                response = http(url, '/v1/completions', dict(model=MODEL, prompt=request_ids, temperature=0, max_tokens=1,
                                ignore_eos=True, logprobs=5, return_token_ids=True, stream=False), 60)
                write(out/'timed-response.json', response)
            finally:
                utility('profile-stop', [nsys, 'stop', '--session', session], 30)
            reports = list(out.glob('prefill*.nsys-rep'))
            assert len(reports) == 1
            utility('profile-export', [nsys, 'export', '--type=sqlite', '--output', str(out/'prefill.sqlite'), str(reports[0])], 40)
            chunks = analyze(out/'prefill.sqlite', name == 'candidate')
            text = (out/'server.stdout.log').read_text() + '\n' + (out/'server.stderr.log').read_text()
            steps = [r for r in trace_rows(text) if r.get('prefill_tokens', 0)]
            assert [r['prefill_tokens'] for r in steps] == [8192]*12
            for chunk in chunks:
                chunk['wall_ms'] = steps[8+chunk['offset']//8192]['execution_us']/1000
            write(out/'result.json', dict(exact=samples, timed=canonical(response), chunks=chunks,
                                         full_logits_sha256=digest(out/'logits.f32')))
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        try:
            pidfile = out/'server-group.json'
            if pidfile.exists():
                kill_group(json.loads(pidfile.read_text())['pgid'])
            utility('profile-shutdown', [nsys, 'shutdown', '--session', session, '--kill=sigkill'], 15, False)
        finally:
            if process is not None:
                kill_group(process.pid, process)
                write(out/'server.exit.json', dict(exit_code=process.returncode))


def verify(root):
    manifest = json.loads((root/'artifacts.json').read_text())
    assert re.fullmatch('[0-9a-f]{40}', manifest['base_revision']), manifest['base_revision']
    for filename, wanted in manifest['sha256'].items():
        path = Path(filename)
        assert path.is_absolute() and path.is_file() and not path.stat().st_mode & 0o022
        assert digest(path) == wanted, filename
    assert len(json.loads((root/'prompt.json').read_text())) == 32768
    return manifest


def job(out, dry):
    out.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', dry_run=dry)
    since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    deadline = time.monotonic()+2000
    def step(name, args, seconds=25, check=True):
        return run_step(out, name, args, min(seconds, max(.001, deadline-time.monotonic())), check=check, cwd=ROOT)[0]
    def telemetry(when):
        power = step('power-'+when, ['nvidia-smi', '--query-gpu=index,power.limit', '--format=csv,noheader,nounits'])
        rows = [r.split(',') for r in power.splitlines() if r.strip()]
        assert len(rows) == 4 and all(float(r[1]) == 250 for r in rows)
        idle = step('compute-'+when, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
        assert not idle.strip(), 'GPU processes remain'
        journal = step('journal-'+when, ['journalctl', '-k', '--since', since, '--no-pager', '-o', 'short-iso'])
        errors = (out/f'journal-{when}.stderr.log').read_text()
        assert not re.search(r'permission|not seeing|No journal files', journal+errors, re.I)
        assert not re.search(r'\bNVRM:.*\bXid\b', journal)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        step('hashes', [sys.executable, '-B', str(ROOT/'prefill_cache_serving.py'), '--verify'])
        config = json.loads((ROOT/'artifacts.json').read_text())
        if dry:
            step('controls', [sys.executable, '-B', '-m', 'unittest', 'test_prefill_cache_serving'], 30)
            for name in ('base', 'candidate'):
                binary = str(Path(config['arms'][name]['install'])/'bin/chopinfer')
                step(name+'-version', [binary, '--version'])
                step(name+'-help', [binary, 'serve', '--help'])
            result.update(status='passed', scope='CPU manifest, parser and negative controls only; no GPU access')
        else:
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
            assert shutil.disk_usage(ROOT).free >= 154*1024**3
            telemetry('before')
            for name in ('base', 'candidate'):
                target = out/name
                target.mkdir()
                step(name, [sys.executable, '-B', str(ROOT/'prefill_cache_serving.py'), '--arm', name, '--output', str(target)], 930)
            a, b = [json.loads((out/name/'result.json').read_text()) for name in ('base', 'candidate')]
            assert a['exact'] == b['exact'] and a['timed'] == b['timed'], 'served token/logprob bits differ'
            assert (out/'base/logits.f32').read_bytes() == (out/'candidate/logits.f32').read_bytes(), 'full F32 logits differ'
            result.update(status='passed', full_logits_rows=2, full_logits_values=2*129280,
                          replay=True, offsets=list(OFFSETS), ranks=4, arms=dict(base=a, candidate=b),
                          scope='one profiled A/B; full F32 vocabulary plus reported binary64 logprobs exact')
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if not dry:
            deadline = time.monotonic()+55
            try:
                telemetry('after')
            except BaseException as error:
                result.update(status='failed', telemetry_error=repr(error))
                write(out/'summary.json', result)
                raise
        write(out/'summary.json', result)
    print('PASS prefill_cache_serving ' + str(out), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        if os.getpid() != os.getpgrp():
            os.setsid()
        write(Path(sys.argv[2]), dict(pgid=os.getpgrp()))
        os.execv('/usr/bin/timeout', ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '925s', *sys.argv[3:]])
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--arm', choices=('base', 'candidate'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(ROOT)
    elif args.arm:
        arm(args.output, args.arm, json.loads((ROOT/'artifacts.json').read_text())['arms'][args.arm])
    else:
        job(args.output, args.dry_run)


if __name__ == '__main__':
    main()
