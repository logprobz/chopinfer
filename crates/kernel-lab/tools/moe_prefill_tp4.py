#!/usr/bin/env python3
"""Run four prepared TP4 servers sequentially and compare cold prefill.

CONFIG is a JSON list of four arms in order: MiMo main, MiMo candidate,
Qwen main, Qwen candidate. Each arm supplies family (mimo or qwen), revision
(main or candidate), argv (direct server command), url, and optional env.
Build and package both revisions before the GPU slot. No lock is acquired.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'qual'))
from common import Client


class SlotExpired(Exception):
    pass


def stop(process):
    try:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Also reap the child if the slot alarm interrupts graceful shutdown.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def validate(arms):
    expected = [('mimo', 'main'), ('mimo', 'candidate'), ('qwen', 'main'), ('qwen', 'candidate')]
    if [(a['family'], a['revision']) for a in arms] != expected:
        raise ValueError('need MiMo main/candidate followed by Qwen main/candidate')
    for arm in arms:
        argv = arm['argv']
        if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
            raise ValueError('argv must be a list of strings, not a shell command')
        for flag, value in [('--prefix-cache', 'off'), ('--speculative-tokens', '0'),
                            ('--prefill-chunk-tokens', '2048')]:
            if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
                raise ValueError(f'require {flag} {value}')
        if argv.count('--devices') != 1:
            raise ValueError('require --devices with four distinct ordinals')
        devices = argv[argv.index('--devices') + 1].split(',')
        if len(devices) != 4 or len(set(devices)) != 4:
            raise ValueError('require four distinct devices')


def measure(client, model, prompt):
    started = time.monotonic()
    events, latency = client.request('/v1/completions', {
        'model': model, 'prompt': prompt, 'max_tokens': 16, 'temperature': 0,
        'return_token_ids': True, 'stream': True,
        'stream_options': {'include_usage': True}}, stream=True)
    ids, text, first, usage = [], [], None, None
    for at, event in events:
        usage = event.get('usage') or usage
        for choice in event.get('choices', []):
            tokens = choice.get('token_ids') or []
            if tokens and first is None:
                first = at
            ids.extend(tokens)
            text.append(choice.get('text') or '')
    if not ids or first is None or usage is None or usage['completion_tokens'] != len(ids):
        raise ValueError('missing token IDs, timing or matching usage')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    if cached:
        raise ValueError(f'expected cold prefill, reused {cached} tokens')
    return dict(token_ids=ids, text=''.join(text), prompt_tokens=usage['prompt_tokens'],
                ttft_seconds=first-started, latency_seconds=latency,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('end_utc')
    args = parser.parse_args()
    arms = json.loads(args.config.read_text())
    validate(arms)
    end = datetime.datetime.fromisoformat(args.end_utc.replace('Z', '+00:00'))
    if end.tzinfo is None:
        parser.error('end time must include its UTC offset')
    budget = min(1195, end.timestamp() - time.time() - 5)
    if budget <= 0:
        parser.error('end time has passed')
    root = Path(os.environ['CHOPINFER_DATA']) / 'kopt-moe-prefill'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    logs = root / 'logs' / f'tp4-{stamp}'
    logs.mkdir(parents=True, exist_ok=False)
    (logs / 'config.json').write_text(json.dumps(arms, indent=2) + '\n')
    deadline = time.monotonic() + budget

    def expired(*unused):
        raise SlotExpired('TP4 run window ended')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    prompts, results = {}, {}
    try:
        for arm in arms:
            model = {'mimo': 'mimo26', 'qwen': 'qwen38'}[arm['family']]
            os.environ['CHOPINFER_QUAL_CONFIG'] = str(
                Path(__file__).resolve().parents[3] / 'models' / model / 'qual.json')
            name = arm['family'] + '-' + arm['revision']
            # Refuse an occupied endpoint instead of mistaking an old server for ours.
            try:
                urllib.request.urlopen(arm['url'] + '/health', timeout=1).close()
            except (urllib.error.URLError, TimeoutError):
                pass
            else:
                raise RuntimeError(f'endpoint already occupied: {arm["url"]}')
            with (logs / f'{name}.log').open('x') as log:
                process = subprocess.Popen(arm['argv'], env=os.environ | arm.get('env', {}),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    client = Client(arm['url'], timeout=10, retries=0)
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'{name} exited before readiness: {process.returncode}')
                        try:
                            urllib.request.urlopen(arm['url'] + '/health', timeout=1).close()
                            identity = client.identity()
                            break
                        except (urllib.error.URLError, TimeoutError, RuntimeError):
                            time.sleep(0.5)
                    client.timeout = max(1, min(120, deadline-time.monotonic()))
                    unit = 'A queue records each request and then validates the result. '
                    probe, _ = client.request('/v1/completions', {
                        'model': identity['model'], 'prompt': unit * 64,
                        'max_tokens': 1, 'temperature': 0})
                    family_prompts = prompts.setdefault(arm['family'], {})
                    cases = {}
                    for tokens in (8192, 32768):
                        prompt = family_prompts.setdefault(tokens, unit * max(
                            1, round(tokens * 64 / probe['usage']['prompt_tokens'])))
                        samples = [measure(client, identity['model'], prompt) for _ in range(3)]
                        if any(abs(s['prompt_tokens']-tokens) > 128 for s in samples):
                            raise ValueError('prompt length differs from target by more than 128')
                        if any(s['token_ids'] != samples[0]['token_ids'] for s in samples):
                            raise ValueError('greedy output differs between repeats')
                        cases[str(tokens)] = dict(samples=samples, median_ttft_seconds=statistics.median(
                            s['ttft_seconds'] for s in samples))
                    results[name] = dict(identity=identity, cases=cases)
                    (logs / f'{name}.json').write_text(json.dumps(results[name], indent=2) + '\n')
                    if arm['revision'] == 'candidate':
                        for tokens, case in cases.items():
                            baseline = results[arm['family'] + '-main']['cases'][tokens]
                            for sample in case['samples']:
                                for field in ('token_ids', 'text', 'prompt_tokens', 'prompt_sha256'):
                                    if sample[field] != baseline['samples'][0][field]:
                                        raise ValueError(f'{name} {tokens}: {field} differs from main')
                            case['ttft_ratio_to_main'] = case['median_ttft_seconds'] / baseline['median_ttft_seconds']
                    print(f'passed {name}', flush=True)
                finally:
                    stop(process)
        (logs / 'success.json').write_text(json.dumps(results, indent=2) + '\n')
        print(logs / 'success.json')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
