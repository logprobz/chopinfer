#!/usr/bin/env python3
"""Frozen same-install BF16-partial ABBA experiment, with unchanged accuracy gates."""
import argparse
import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import sys
import threading
import time

PROFILES = ('tp4-ar', 'tp4-dspark')
ORDER = (0, 1, 1, 0)
OUTPUT_TOKENS = 128
EXPECTED = {'teacher': (35, 3974), 'chat': (17, 5270)}
WORK_SECONDS = 3540
STARTUP_SECONDS = 180
QUAL_SECONDS = 300
PAIRED_SECONDS = 300
LAB_SECONDS = 240


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bits(value):
    require(type(value) in (float, int) and math.isfinite(value), 'nonfinite logprob')
    return struct.pack('!d', value).hex()


def events(records, rows, count, started, cold):
    ids, selected, top, tokens, arrivals = [], [], [], [], []
    usage = None
    finish = None
    for at, value in records:
        require('error' not in value, 'stream error')
        if value.get('usage') is not None:
            usage = value['usage']
        for choice in value.get('choices', []):
            require(choice.get('index', 0) == 0, 'unexpected choice index')
            finish = choice.get('finish_reason') or finish
            lp = choice.get('logprobs') or {}
            part = choice.get('token_ids') or []
            if part:
                require(all(type(i) is int and 0 <= i < 2**32 for i in part), 'invalid token IDs')
                require(all(len(lp.get(k, [])) == len(part) for k in ('tokens', 'token_logprobs', 'top_logprobs')), 'incomplete logprobs')
                arrivals.append((at, len(part)))
                ids.extend(part)
                tokens.extend(lp['tokens'])
                selected.extend(bits(v) for v in lp['token_logprobs'])
                for row in lp['top_logprobs']:
                    require(isinstance(row, dict) and 1 <= len(row) <= 5, 'invalid top-logprob map')
                    top.append({key: bits(value) for key, value in sorted(row.items())})
            else:
                require(not lp.get('tokens'), 'token IDs absent from a scored event')
    require(usage and usage.get('prompt_tokens') == rows and usage.get('completion_tokens') == count, 'wrong executed token count')
    if cold:
        require((usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0, 'prefill was not cold')
    require(len(ids) == count and finish == 'length' and arrivals and arrivals[0][0] > started, 'incomplete stream')
    return dict(canonical=dict(token_ids=ids, tokens=tokens, selected_bits=selected, top_bits=top),
                ttft_seconds=arrivals[0][0]-started, arrivals=arrivals, usage=usage)


def compare(left, right):
    require(left and left.keys() == right.keys(), 'comparison case coverage differs')
    counts = dict(requests=0, positions=0, token_ids=0, token_labels=0, selected_logprobs=0,
                  top_key_changes=0, top_logprobs=0)
    for name in left:
        require(left[name] and len(left[name]) == len(right[name]), 'comparison request coverage differs')
        for a, b in zip(left[name], right[name]):
            require(a['prompt_sha256'] == b['prompt_sha256'], 'comparison prompts differ')
            a, b = a['canonical'], b['canonical']
            require(len(a['token_ids']) == len(b['token_ids']) > 0, 'comparison output coverage differs')
            counts['requests'] += 1
            counts['positions'] += len(a['token_ids'])
            for field, key in [('token_ids', 'token_ids'), ('tokens', 'token_labels'), ('selected_bits', 'selected_logprobs')]:
                counts[key] += sum(x != y for x, y in zip(a[field], b[field]))
            for x, y in zip(a['top_bits'], b['top_bits']):
                counts['top_key_changes'] += len(x.keys() ^ y.keys())
                counts['top_logprobs'] += sum(x[k] != y[k] for k in x.keys() & y.keys())
    counts['bitwise'] = not any(counts[k] for k in counts if k not in ('requests', 'positions'))
    return counts


def fault_sample(pgid, proc=Path('/proc')):
    result = {}
    for path in proc.glob('[0-9]*/stat'):
        try:
            value = path.read_text()
            fields = value[value.rfind(')')+2:].split()
            if int(fields[2]) == pgid:
                result[path.parent.name+':'+fields[19]] = dict(minor=int(fields[7]), major=int(fields[9]))
        except (OSError, ValueError, IndexError):
            continue
    require(result, 'server process fault counters unavailable')
    return result


def fault_delta(samples):
    require(len(samples) >= 2 and all(samples), 'missing process fault samples')
    values = {}
    for sample in samples:
        for key, current in sample.items():
            item = values.setdefault(key, dict(first=current, last=current))
            require(all(current[k] >= item['last'][k] for k in ('minor', 'major')), 'process fault counters decreased')
            item['last'] = current
    return {key: {kind: value['last'][kind]-value['first'][kind] for kind in ('minor', 'major')}
            for key, value in values.items()}


class Budget:
    def __init__(self):
        self.start = time.monotonic()
        self.limit = WORK_SECONDS

    @contextlib.contextmanager
    def phase(self, seconds):
        left = min(seconds, self.limit-(time.monotonic()-self.start))
        require(left > 0, 'global deadline exhausted')
        signal.setitimer(signal.ITIMER_REAL, left)
        try:
            yield left
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def measured(common, server, result):
    pgid = read(server.pidfile)['pgid']
    loads = [common.host_load_sample()]
    faults = [fault_sample(pgid)]
    errors = []
    stop = threading.Event()

    def sample():
        while not stop.wait(.5):
            try:
                loads.append(common.host_load_sample())
                faults.append(fault_sample(pgid))
            except Exception as error:
                errors.append(str(error))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        loads.append(common.host_load_sample())
        faults.append(fault_sample(pgid))
        require(not errors, 'measurement sampler failed: '+str(errors))
        result.update(host_load=common.host_load(loads), process_faults=fault_delta(faults),
                      process_fault_samples=faults)


def serve_argv(stage, profile, port):
    return [str(stage/'assets/install-integrated/bin/chopinfer'), 'serve', '--bundle',
            read(stage/'assets/logs/prepare.json')['bundle'], '--checkpoint',
            os.environ['CHOPINFER_DEEPSEEK_CHECKPOINT'], '--profile',
            str(stage/'assets/source-integrated/models/deepseek41/profiles.json')+'#'+profile,
            '--max-context-tokens', '1048576', '--listen', f'127.0.0.1:{port}',
            '--served-model-name', 'DeepSeek-V4.1-Flash', '--drain-timeout-ms', '5000']


def request(p, server, ids, count, salt, timeout, output):
    client = p.bench.Client(server.url, timeout, retries=0)
    body = dict(model=p.MODEL, prompt=ids, max_tokens=count, min_tokens=count,
                ignore_eos=True, temperature=0, seed=19, logprobs=5, return_token_ids=True,
                stream=True, stream_options=dict(include_usage=True), cache_salt=salt)
    started = time.monotonic()
    records, latency = client.request('/v1/completions', body, stream=True)
    p.write(output.with_suffix('.request.json'), body)
    p.write(output.with_suffix('.events.json'), records)
    result = events(records, len(ids), count, started, cold=True)
    result.update(latency_seconds=latency, prompt_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
    p.write(output.with_suffix('.json'), result)
    return result


def speculation(delta):
    """An adaptive window can contain only ordinary decode steps."""
    require(delta.get('available') is True, 'DSpark speculation metrics unavailable')
    for name in ('verify_items', 'draft_tokens', 'accepted_tokens'):
        require(type(delta.get(name)) is int and delta[name] >= 0, 'invalid speculation counter: '+name)
    return dict(verify_items=delta['verify_items'], draft_tokens=delta['draft_tokens'],
                accepted_tokens=delta['accepted_tokens'], zero_draft_window=delta['draft_tokens'] == 0,
                verification_observed=delta['verify_items'] > 0)


def benchmark(p, common, server, ids, prompts, label):
    captures, timings = {}, {}
    for rows, timeout in [(8192, 25), (32768, 50)]:
        timed = timings[f'prefill-{rows}'] = {}
        with measured(common, server, timed):
            row = request(p, server, ids[:rows], 1, f'{label}-cold-{rows}', timeout,
                          server.directory/f'prefill-{rows}')
        captures[f'prefill-{rows}'] = [row]
        timed['ttft_seconds'] = row['ttft_seconds']
    for concurrency in (1, 8):
        # Populate each graph before timing. Fresh cache salts retain cold
        # request inputs without changing prompt IDs across arms.
        def batch(count, suffix):
            barrier = threading.Barrier(concurrency)
            def one(index):
                barrier.wait(timeout=10)
                return request(p, server, prompts[index], count, f'{label}-{suffix}-{index}', 90,
                               server.directory/f'{suffix}-{index}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                return list(pool.map(one, range(concurrency)))
        batch(32, f'warm-c{concurrency}')
        timed = timings[f'decode-c{concurrency}'] = {}
        before = p.metrics(server, f'c{concurrency}-before')
        with measured(common, server, timed):
            rows = batch(OUTPUT_TOKENS, f'decode-c{concurrency}')
        timed.update(steady=p.bench.steady([r['arrivals'] for r in rows], trim=16),
                     cohort=p.bench.cohort([r['arrivals'] for r in rows]),
                     metrics=p.spec_metrics.difference(before, p.metrics(server, f'c{concurrency}-after')))
        if server.profile == 'tp4-dspark':
            timed['speculation'] = speculation(timed['metrics'])
        captures[f'decode-c{concurrency}'] = rows
        p.write(server.directory/'benchmark.json', dict(captures=captures, timings=timings))
    return captures, timings


def validate_accuracy(directory, baseline, profile, common):
    report = read(directory/'eval/report.json')
    require(report['passed'] and report['exit'] == 0 and report['failure'] is None, 'absolute accuracy failed')
    require([(c['id'], c['status']) for c in report['checks']] == [('teacher', 'pass'), ('chat', 'pass')], 'absolute checks missing')
    counts = {}
    for name, (texts, positions) in EXPECTED.items():
        value = read(directory/f'eval/records/{name}.json')
        accepted = read(baseline/profile/f'{name}.json')
        require(value['passed'] is True and value['mechanics_only'] is False and not value.get('errors') and not value.get('skipped'), 'incomplete absolute receipt')
        require(value['reference_sha256'] == accepted['reference_sha256'], 'reference moved')
        require(value['scored'] == positions and len(value['rows']) == texts, 'wrong absolute coverage')
        require([r['name'] for r in value['rows']] == [r['name'] for r in accepted['rows']], 'absolute text set differs')
        for row, old in zip(value['rows'], accepted['rows']):
            require(row['scored'] == len(row['served_rows']) == old['scored'] and row['prompt_tokens'] == old['prompt_tokens'], 'scored row coverage differs')
            require(row['served_rows_sha256'] == common.rows_digest(row['served_rows']), 'served row digest differs')
        require(digest(directory/f'eval/captures/{name}.json') == value['capture_sha256'], 'capture digest differs')
        counts[name] = dict(texts=texts, positions=positions)
    return counts


def accuracy(p, common, budget, stage, server, baseline):
    directory = server.directory
    env = p.env_for(directory)
    env['CHOPINFER_QUAL_MODEL_ROOT'] = str(p.SOURCE)
    argv = [p.INSTALL/'bin/chopinfer', 'eval', 'run', '--model', p.SOURCE/'models/deepseek41',
            '--check', 'teacher', '--check', 'chat', '--cohort', 'full', '--served-model', p.MODEL,
            '--url', server.url, '--server-log', directory/'server.log', '--timeout', '1100',
            '--report', directory/'eval']
    with budget.phase(QUAL_SECONDS) as seconds:
        p.command(argv, directory/'eval.log', seconds=seconds, env=env)
    counts = validate_accuracy(directory, baseline, server.profile, common)
    env['CHOPINFER_QUAL_RECEIPTS'] = str(directory/'paired')
    env['PYTHONPATH'] = str(p.SOURCE)
    argv = [sys.executable, '-B', '-m', 'tools.qual', 'paired-teacher', '--accepted',
            baseline/server.profile/'teacher.json', '--candidate', directory/'eval/records/teacher.json',
            '--url', server.url, '--model', p.MODEL]
    with budget.phase(PAIRED_SECONDS) as seconds:
        p.command(argv, directory/'paired.log', seconds=seconds, env=env)
    receipts = list((directory/'paired').glob('*-paired_teacher.json'))
    require(len(receipts) == 1, 'paired receipt missing or duplicated')
    paired = read(receipts[0])
    require(paired.get('passed') is True and not paired.get('mechanics_only') and not paired.get('skipped'), 'paired gate failed')
    return dict(absolute=counts, paired_receipt=str(receipts[0]), paired_passed=True)


def verify_seal(stage):
    seal = read(stage/'manifest.json')
    for name, expected in seal['files'].items():
        path = stage/name
        require(path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o022, 'unsealed file: '+name)
        require(digest(path) == expected, 'changed staged artifact: '+name)
    return seal


def oracle_coverage(value):
    require(value['status'] == 'passed' and value['real_cases'] == 249 and value['chunk_cases'] == 13
            and value['real_executions'] == 1992 and value['chunk_executions'] == 52, 'GPU fixture coverage differs')


def reuse_ar(stage, seal, baseline, common):
    """Reuse complete AR arms only with authenticated, identical executables.

    The prior enclosing job may have failed later in DSpark. Its AR repeat
    gates, absolute and paired accuracy, and GPU oracle must all have passed.
    Every reused receipt is also part of this stage's immutable file seal.
    """
    if not seal.get('reuse_ar'):
        return None
    root = stage/'reuse'
    source_seal = read(root/'source-manifest.json')
    require(digest(root/'source-manifest.json') == seal['reuse_ar']['manifest_sha256'], 'reuse source seal differs')
    prior = read(root/'summary.json')
    require(prior['seal'] == source_seal and source_seal['revision'] == seal['revision'], 'reuse binary revision differs')
    require(source_seal['baseline_manifest_sha256'] == digest(baseline/'SHA256SUMS'), 'reuse baseline differs')
    require(prior['controls'] == dict(DMA=0, overlap=0, partials=list(ORDER)), 'reuse controls differ')
    binaries = {name: sha for name, sha in source_seal['files'].items()
                if name.startswith('assets/install-integrated/') or name == 'assets/partial-tests'}
    require(set(binaries) == {'assets/install-integrated/bin/'+name for name in
            ('chopinfer', 'chopinfer-host', 'chopinfer-model-deepseek41')} |
            {'assets/install-integrated/lib/libbundle_deepseek41.so', 'assets/partial-tests'}, 'reuse binary inventory differs')
    for name, sha in binaries.items():
        require(seal['files'].get(name) == sha and digest(stage/name) == sha, 'reuse requires byte-identical binaries: '+name)
    for name in ('models/deepseek41/profiles.json', 'models/deepseek41/qual.json',
                 'models/deepseek41/qual-references.lock.json', 'tools/qual/paired_teacher.py'):
        path = 'assets/source-integrated/'+name
        require(seal['files'][path] == source_seal['files'][path], 'reuse policy differs: '+name)
    require(seal['files']['assets/tokens32k.json'] == source_seal['files']['assets/tokens32k.json'], 'reuse prompt differs')
    for when in ('before', 'after'):
        telemetry = prior[when]
        require(telemetry['journal_checked'] and not telemetry['xids'] and
                telemetry['power_limits_w'] == [250.0]*4, 'reuse telemetry failed')
    oracle = read(root/'gpu-oracle.json')
    oracle_coverage(oracle)
    require(re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored;',
                     (root/'gpu-oracle.log').read_text()), 'reused GPU test did not execute')
    arms = [a for a in prior['arms'] if a['profile'] == 'tp4-ar']
    require([(a['index'], a['partials'], a['status']) for a in arms] ==
            [(i, arm, 'passed') for i, arm in enumerate(ORDER)], 'reuse needs four passed AR arms in ABBA order')
    captures = {}
    for arm in arms:
        index = arm['index']
        directory = root/f"tp4-ar-{index}-{arm['partials']}"/'tp4-ar'
        rows = read(directory/'captures.json')
        expected = {'prefill-8192':(1,1), 'prefill-32768':(1,1),
                    'decode-c1':(1,OUTPUT_TOKENS), 'decode-c8':(8,OUTPUT_TOKENS)}
        require(rows.keys() == expected.keys(), 'reuse capture cases differ')
        for name, (requests, tokens) in expected.items():
            require(len(rows[name]) == requests and all(len(r['canonical']['token_ids']) == tokens for r in rows[name]), 'reuse capture counts differ')
        require(read(directory/'ready.json')['health']['serving']['source_commit'] == seal['revision'], 'reused server revision differs')
        captures['tp4-ar', index] = rows
        arm['reused_from'] = seal['reuse_ar']['source_run']
    for first, second in ((0,3), (1,2)):
        require(compare(captures['tp4-ar', first], captures['tp4-ar', second])['bitwise'], 'reused AR repeats differ')
    directory = root/'tp4-ar-1-1/tp4-ar'
    absolute = validate_accuracy(directory, baseline, 'tp4-ar', common)
    receipts = list((directory/'paired').glob('*-paired_teacher.json'))
    require(len(receipts) == 1, 'reused paired receipt missing or duplicated')
    paired = read(receipts[0])
    require(paired.get('passed') is True and not paired.get('mechanics_only') and not paired.get('skipped'), 'reused paired gate failed')
    for name, path in (('accepted', baseline/'tp4-ar/teacher.json'), ('candidate', directory/'eval/records/teacher.json')):
        require(json.dumps(paired['arms'][name], sort_keys=True, allow_nan=False) ==
                json.dumps(read(path), sort_keys=True, allow_nan=False), 'reused paired inputs differ')
    accuracy = dict(absolute=absolute, paired_passed=True, paired_receipt=str(receipts[0]),
                    reused_from=seal['reuse_ar']['source_run'])
    prompts = read(root/'decode-prompts.json')
    require(len(prompts) == 8 and all(ids and all(type(i) is int for i in ids) for ids in prompts), 'reused decode prompts differ')
    return dict(arms=arms, captures=captures, oracle=oracle, accuracy=accuracy, prompts=prompts)


def run(stage, output):
    os.environ['CHOPINFER_EVAL'] = str(stage/'assets/install-integrated/bin/chopinfer')
    source = stage/'assets/source-integrated'
    sys.path[:0] = [str(stage/'assets'), str(source/'tools/qual')]
    import perf_runner as p
    import common
    import release_preview as preview
    budget = Budget()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(os.environ['CHOPINFER_DATA'])/'qual/deepseek41/baselines/default-20261020'
    result = dict(status='running', arms=[], comparisons={}, accuracy={}, controls=dict(DMA=0, overlap=0, partials=list(ORDER)),
                  limits=dict(work_seconds=WORK_SECONDS, startup=STARTUP_SECONDS, absolute=QUAL_SECONDS, paired=PAIRED_SECONDS))
    def save(): p.write(output/'summary.json', result)
    def expired(signum, frame): raise TimeoutError('BF16 partial phase deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(signum, p.interrupted)
    since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    failure = None
    try:
        result['seal'] = verify_seal(stage)
        limit = result['seal'].get('work_seconds', WORK_SECONDS)
        require(type(limit) is int and 0 < limit <= WORK_SECONDS, 'invalid staged work deadline')
        budget.limit = limit
        result['limits']['work_seconds'] = limit
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3', 'whole-machine GPU selection missing')
        result['before'] = p.telemetry(output, 'before', False, since)
        require(not result['before']['xids'], 'Xid before work')
        # Immutable baseline validation is also performed by the CPU dry run.
        for line in (baseline/'SHA256SUMS').read_text().splitlines():
            sha, name = line.split(maxsplit=1)
            require(digest(baseline/name.lstrip('*')) == sha, 'baseline checksum differs')
        reused = reuse_ar(stage, result['seal'], baseline, common)
        result['execution_counts'] = dict(new_servers=0, reused_servers=4 if reused else 0,
                                         new_gpu_oracle_cases=0 if reused else 262, reused_gpu_oracle_cases=262 if reused else 0)
        if reused:
            result['gpu_oracle'] = reused['oracle']
            result['accuracy']['tp4-ar'] = reused['accuracy']
            result['arms'].extend(reused['arms'])
            result['reuse_ar'] = result['seal']['reuse_ar']
        else:
            env = preview.environment(gpu=True)
            env.update(CHOPINFER_KERNEL_GPUS='0,1,2,3', CHOPINFER_KERNEL_REQUIRE='1',
                       CHOPINFER_BF16_PARTIAL_INPUTS=str(stage/'real-inputs'),
                       CHOPINFER_BF16_PARTIAL_REPORT=str(output/'gpu-oracle.json'), CHOPINFER_KERNEL_TIMEOUT_SECS='230')
            with budget.phase(LAB_SECONDS) as seconds:
                p.command([stage/'assets/partial-tests', '--exact', 'real_partials_replays_and_chunks_on_gpu',
                           '--include-ignored', '--nocapture'], output/'gpu-oracle.log', seconds=seconds, env=env)
            text = (output/'gpu-oracle.log').read_text()
            require(re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored;', text), 'GPU test did not execute')
            result['gpu_oracle'] = read(output/'gpu-oracle.json')
        oracle_coverage(result['gpu_oracle'])
        ids = read(stage/'assets/tokens32k.json')
        require(len(ids) == 32768 and all(type(i) is int for i in ids), 'wrong frozen long prompt')
        all_captures = reused['captures'] if reused else {}
        prompts = reused['prompts'] if reused else None
        if prompts is not None:
            p.write(output/'decode-prompts.json', prompts)
        for profile in PROFILES:
            for index, arm in enumerate(ORDER):
                if (profile, index) in all_captures:
                    continue
                label = f'{profile}-{index}-{arm}'
                folder = output/label
                folder.mkdir()
                server = p.Server(folder, profile, use_nsys=False, prefill_chunk=8192)
                port = int(server.url.rsplit(':', 1)[1])
                server.server_command = serve_argv(stage, profile, port)
                server.launch = [sys.executable, '-B', str(stage/'assets/perf_runner.py'), 'server-child', str(server.pidfile), *server.server_command]
                server.env = preview.environment(gpu=True)
                server.env.update(CHOPINFER_DEEPSEEK41_BF16_PARTIALS=str(arm), CHOPINFER_DEEPSEEK41_PREFILL_DMA='0',
                                  CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP='0', CHOPINFER_DEEPSEEK41_PREFILL_BF16_GATHER='1',
                                  CHOPINFER_QUAL_MODEL_ROOT=str(source), CHOPINFER_EVAL=str(p.INSTALL/'bin/chopinfer'))
                record = dict(profile=profile, index=index, partials=arm, status='running')
                result['arms'].append(record)
                save()
                try:
                    with budget.phase(STARTUP_SECONDS):
                        server.start()
                    result['execution_counts']['new_servers'] += 1
                    record['startup_seconds'] = server.readiness
                    health = read(server.directory/'ready.json')['health']['serving']
                    require(health['source_commit'] == result['seal']['revision'], 'live source differs')
                    require(health['profile']['name'] == profile and health['profile']['tensor_parallel_size'] == 4
                            and health['model_options']['prefill_chunk_tokens'] == 8192 and health['model_options']['prefill_upper_skip'] is True,
                            'live profile or prefill options differ')
                    if prompts is None:
                        prompts = [json.loads(p.http(server.url, '/tokenize', dict(model=p.MODEL,
                            prompt=p.bench.stream_prompt('diverse', i), add_special_tokens=False)))['tokens'] for i in range(8)]
                        p.write(output/'decode-prompts.json', prompts)
                    before_arm = p.metrics(server, 'arm-before') if profile == 'tp4-dspark' else None
                    with budget.phase(300):
                        captures, record['timings'] = benchmark(p, common, server, ids, prompts, label)
                    all_captures[profile, index] = captures
                    p.write(server.directory/'captures.json', captures)
                    save()
                    if index == 1:
                        try:
                            result['accuracy'][profile] = accuracy(p, common, budget, stage, server, baseline)
                        except Exception as error:
                            # Complete the report-only comparison even when an
                            # accuracy gate rejects the candidate. Never turn
                            # that rejection into a successful job status.
                            result['accuracy'][profile] = dict(error=repr(error), paired_passed=False)
                    record['status'] = 'passed'
                    if before_arm is not None:
                        record['arm_metrics'] = p.spec_metrics.difference(before_arm, p.metrics(server, 'arm-after'))
                        record['speculation'] = speculation(record['arm_metrics'])
                        record['speculation']['scope'] = 'whole arm after startup, including warmups and accuracy'
                        if not record['speculation']['verification_observed']:
                            record.update(status='failed', error='DSpark arm did not execute verification')
                finally:
                    server.stop()
                save()
            for first, second, name, gated in [(0,3,'off-repeat',True), (1,2,'on-repeat',True), (0,1,'off-on',False), (3,2,'off-on-reverse',False)]:
                comparison = compare(all_captures[profile, first], all_captures[profile, second])
                result['comparisons'][profile+'-'+name] = dict(gated=gated, **comparison)
                require(not gated or comparison['bitwise'], profile+' '+name+' changed bits')
        for index in range(4):
            comparison = compare(all_captures['tp4-ar', index], all_captures['tp4-dspark', index])
            result['comparisons'][f'ar-dspark-{index}'] = dict(gated=True, **comparison)
            require(comparison['bitwise'], 'AR and DSpark differ within one arithmetic mode')
        counts = dict(servers=len(result['arms']), cold_ttft=sum(len([k for k in a['timings'] if k.startswith('prefill')]) for a in result['arms']),
                      decode_cohorts=sum(len([k for k in a['timings'] if k.startswith('decode')]) for a in result['arms']),
                      scored_positions=sum(v['positions'] for a in result['accuracy'].values() for v in a.get('absolute', {}).values()),
                      paired_profiles=len(result['accuracy']))
        result['counts'] = counts
        require(all(a['status'] == 'passed' for a in result['arms']), 'one or more DSpark arms did not execute verification')
        require(all(a.get('paired_passed') for a in result['accuracy'].values()), 'one or more accuracy gates failed')
        require(counts == dict(servers=8, cold_ttft=16, decode_cohorts=16, scored_positions=18488, paired_profiles=2), 'job coverage differs')
        result.update(status='passed', counts=counts)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        failure = error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 30)
        try:
            result['after'] = p.telemetry(output, 'after', False, since)
            require(not result['after']['xids'], 'Xid during job')
        except BaseException as error:
            result.update(status='failed', telemetry_error=repr(error))
            failure = failure or error
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result['elapsed_seconds'] = time.monotonic()-budget.start
            save()
    if failure:
        raise failure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.stage.resolve(), args.output.resolve())
