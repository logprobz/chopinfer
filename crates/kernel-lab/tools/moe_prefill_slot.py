#!/usr/bin/env python3
"""Run prebuilt prefill tests on one idle GPU within a bounded run window.

Usage: moe_prefill_slot.py DEVICE END_UTC --binary OPS_BINARY --mode record
END_UTC is an ISO timestamp such as 2026-09-26T15:20:00Z. No lock is acquired.
The whole run is capped at 20 minutes and the earlier operator deadline.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

MIMO = 'moe::prefill::grouped_moe_mimo_prefill_skewed'
QWEN_CHECK = 'moe::qwen38::qwen38_moe_prefill_matches_oracles'
QWEN_BENCH = 'moe::qwen38::qwen38_moe_benchmarks'


def run(command, env, log, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('run window has ended')
    with log.open('x') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            # This process group was created here and contains only this test.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    text = log.read_text()
    if code != 0 or '1 passed; 0 failed' not in text or 'skipping' in text.lower():
        raise RuntimeError(f'test did not run successfully: exit {code}, {log}')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('device', type=int, choices=range(4))
    parser.add_argument('end_utc')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--mode', choices=('record', 'compare'), required=True)
    parser.add_argument('--candidate-out', type=Path,
                        help='audited gate/up cubins; requires compare mode')
    parser.add_argument('--test-timeout', type=float, default=180,
                        help='maximum seconds per test, bounded by the slot deadline')
    args = parser.parse_args()
    if not 1 <= args.test_timeout <= 1195:
        parser.error('test timeout must be between 1 and 1195 seconds')
    if args.candidate_out is not None and args.mode != 'compare':
        parser.error('candidate cubins require compare mode and an existing baseline')
    end = datetime.datetime.fromisoformat(args.end_utc.replace('Z', '+00:00'))
    if end.tzinfo is None:
        parser.error('end time must include its UTC offset')
    budget = min(1195, end.timestamp() - time.time() - 5)
    if budget <= 0:
        parser.error('end time has passed')
    deadline = time.monotonic() + budget
    root = Path(os.environ['CHOPINFER_DATA']) / 'kopt-moe-prefill'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    logs = root / 'logs' / stamp
    logs.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('CHOPINFER_KERNEL_GPUS', 'CHOPINFER_KERNEL_SHARED_DEVICE',
                'CHOPINFER_MOE_PREFILL_LOGITS', 'CHOPINFER_MOE_PREFILL_ROUTING',
                'CHOPINFER_MOE_PREFILL_CANDIDATE_OUT'):
        env.pop(key, None)
    env.update(CHOPINFER_KERNEL_GPU=str(args.device), CHOPINFER_KERNEL_BENCH='1',
               CHOPINFER_MOE_TILED_ROUTE='1', CHOPINFER_QWEN38_RANKS='4',
               CHOPINFER_QWEN_MOE_RANK='1', CHOPINFER_MOE_PREFILL_CONTEXT='1',
               CHOPINFER_KERNEL_RECEIPTS=str(root / 'receipts' / stamp),
               CHOPINFER_MOE_PREFILL_BASELINE_MODE=args.mode)
    if args.candidate_out is not None:
        env['CHOPINFER_MOE_PREFILL_CANDIDATE_OUT'] = str(args.candidate_out.resolve(strict=True))
    binary = str(args.binary.resolve(strict=True))
    cases = [(name, MIMO, {'CHOPINFER_MOE_PREFILL_ROUTING': name})
             for name in ('skewed', 'balanced', 'hot')]
    for layer in (1, 24, 47):
        replay = root / 'replays' / f'layer{layer:02}'
        if not (replay / 'manifest.json').is_file():
            parser.error(f'missing replay fixture: {replay}')
        cases.append((f'replay-layer{layer:02}', MIMO,
                      {'CHOPINFER_MOE_PREFILL_LOGITS': str(replay)}))
    completed = []
    for name, selector, extra in cases:
        case_env = env | extra | {'CHOPINFER_MOE_PREFILL_TOKENS': '1024,2048',
                                 'CHOPINFER_MOE_PREFILL_BASELINE': str(root / 'baseline' / name)}
        log = logs / f'{name}.log'
        run([binary, '--exact', selector, '--nocapture', '--test-threads=1'], case_env, log,
            min(deadline, time.monotonic() + args.test_timeout))
        completed.append(str(log))
        print(f'passed {name}: {log}', flush=True)
    for tokens in (1024, 2048):
        case_env = env | {'CHOPINFER_QWEN_MOE_TOKENS': str(tokens),
                         'CHOPINFER_MOE_PREFILL_BASELINE': str(root / 'baseline' / 'qwen')}
        for selector in (QWEN_CHECK, QWEN_BENCH):
            log = logs / f'qwen-{tokens}-{selector.rsplit("::", 1)[1]}.log'
            run([binary, '--exact', selector, '--nocapture', '--test-threads=1'], case_env, log,
                min(deadline, time.monotonic() + args.test_timeout))
            completed.append(str(log))
            print(f'passed Qwen {tokens}: {log}', flush=True)
    receipts = [json.loads(p.read_text()) for p in (root / 'receipts' / stamp).rglob('*.json')]
    if not receipts or any(r.get('measurement_label') != 'exclusive-device' for r in receipts):
        raise RuntimeError('missing or shared-device receipts are not performance evidence')
    if sum('gate_up_context_us' in r for r in receipts) != 14:
        raise RuntimeError('expected 12 MiMo and two Qwen context benchmark receipts')
    (logs / 'success.json').write_text(json.dumps(dict(
        binary=binary, binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        candidate_out=env.get('CHOPINFER_MOE_PREFILL_CANDIDATE_OUT'),
        mode=args.mode, device=args.device, end_utc=args.end_utc, completed=completed), indent=2) + '\n')
    print(logs / 'success.json')


if __name__ == '__main__':
    main()
