"""Run the sealed prefill review qualification on one GPU, capped at eight minutes."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

TOKENS = (2048, 2047, 1025, 1024, 1021, 129, 128, 127, 17, 16, 15, 1)
CASES = {
    'mimo': ('moe::prefill::boundaries::mimo_prefill_boundaries_bitwise', ('spread', 'hot-local')),
    'qwen': ('moe::qwen38::boundaries::qwen_prefill_boundaries_bitwise', ('last-expert', 'hot-local', 'empty')),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(command, env, log, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('operator deadline reached')
    with log.open('x') as output:
        child = subprocess.Popen(command, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
        finally:
            if child.poll() is None:
                # Only this runner's own process group is stopped and reaped.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait()
    if code:
        raise RuntimeError(f'exit {code}: {log}')


def validate(model, text, receipts):
    fixtures = CASES[model][1]
    expected = {(model + '-' + fixture, tokens) for fixture in fixtures for tokens in TOKENS}
    observed = [(r.get('case'), r.get('live_tokens')) for r in receipts]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError(f'{model}: missing or duplicate executed cases')
    if ('1 passed; 0 failed' not in text or 'skipping' in text.lower()
            or f'boundary executed model={model} cases={len(expected)}' not in text):
        raise RuntimeError(f'{model}: test skipped or did not finish')
    for r in receipts:
        if (not r['bitwise_and_capacity_sentinels_passed'] or r['capacity_tokens'] != 8192
                or len(r['arms']) != 5 or len(r['live_sha256']) != 3):
            raise RuntimeError(f'{model}: incomplete output checks')
        counts = r['expert_counts']
        if model == 'qwen' and r['case'] == 'qwen-last-expert':
            if len(counts) != 128 or counts[-1] != r['live_tokens'] or sum(counts[:-1]):
                raise RuntimeError('last local expert did not receive every live token')
        if model == 'qwen' and r['case'] == 'qwen-empty' and any(counts):
            raise RuntimeError('empty rank unexpectedly has local work')
        if r['case'].endswith('hot-local') and r['live_tokens'] in (2047, 1025):
            if not r['full_then_partial_cta_reuse']:
                raise RuntimeError('full-to-partial persistent CTA reuse was not exercised')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('device', type=int, choices=range(4))
    parser.add_argument('end_utc')
    args = parser.parse_args()
    root = Path(os.environ['CHOPINFER_DATA']) / 'kopt-moe-prefill'
    end = datetime.datetime.fromisoformat(args.end_utc.replace('Z', '+00:00'))
    if end.tzinfo is None:
        parser.error('UTC offset required')
    seconds = min(470, end.timestamp() - time.time() - 5)
    if seconds < 30:
        parser.error('run window expired')
    deadline = time.monotonic() + seconds

    def interrupted(signum, _frame):
        raise TimeoutError(f'operator deadline or signal {signum}')

    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    ready_path = Path(os.environ.get('CHOPINFER_MOE_BOUNDARY_READY', root / 'boundary-ready.json'))
    ready = json.loads(ready_path.read_text())
    for path, want in ready['sha256'].items():
        if digest(path) != want:
            raise RuntimeError('prepared artifact changed: ' + path)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    logs = root / 'logs' / ('boundary-' + stamp)
    logs.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('CHOPINFER_KERNEL_', 'CHOPINFER_MOE_', 'CHOPINFER_QWEN_')) or key in (
                'CUDA_VISIBLE_DEVICES', 'RUNTIME_KERNELS_CUBIN_DIR', 'CHOPINFER_QWEN38_LEAN'):
            env.pop(key)
    env.update(CHOPINFER_KERNEL_GPU=str(args.device), CHOPINFER_QWEN38_RANKS='4',
               CHOPINFER_QWEN_MOE_RANK='1', CHOPINFER_MOE_TILED_ROUTE='1',
               CHOPINFER_MOE_BOUNDARY_BASELINE=ready.get('baseline_dir', str(root / 'boundary' / 'baseline')),
               CHOPINFER_MOE_CUBIN_DIR=ready.get('candidate_dir', str(root / 'boundary' / 'candidate')),
               TMPDIR=str(root / 'tmp'))
    results = []
    try:
        for model, (selector, _) in CASES.items():
            receipts_dir = logs / model
            model_env = env | {'CHOPINFER_KERNEL_RECEIPTS': str(receipts_dir)}
            log = logs / (model + '.log')
            run([ready['binary'], '--exact', selector, '--nocapture', '--test-threads=1'],
                model_env, log, deadline)
            receipts = [json.loads(p.read_text()) for p in receipts_dir.rglob('*.json')]
            validate(model, log.read_text(), receipts)
            results.append({'model': model, 'executed_cases': len(receipts), 'exit_code': 0})
        (logs / 'success.json').write_text(json.dumps({
            'device': args.device, 'ready': ready, 'results': results,
            'executed_cases': sum(r['executed_cases'] for r in results),
            'qualification': 'all live stage bits and all capacity/nonlocal sentinels',
        }, indent=2) + '\n')
        print(logs / 'success.json', flush=True)
    except BaseException as error:
        (logs / 'failure.json').write_text(json.dumps({'error': str(error), 'completed': results}, indent=2) + '\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
