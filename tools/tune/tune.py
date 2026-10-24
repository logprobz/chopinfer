#!/usr/bin/env python3
"""Speculative depth tuner job: measure every draft depth at every concurrency
and pick a per-concurrency depth schedule.

Usage: python3 -B STAGE/harness/tune.py --output RUN --payload-sha SHA [--dry-run]

stage.py seals the stage: the model config (models/<model>/depth-tune.json),
the workload's prompts (prompts.py), the install and its prepared package.
The job:
1. checks the machine: no GPU compute process, power limits recorded and
   constant (and equal to the config's expected watts), SM clocks recorded and
   never asserted, no kernel Xids;
2. runs an untimed warm-up process W0 at the deepest depth through every
   window, so every timed process finds the same warm page cache, and keeps
   W0's outputs as the reference;
3. runs `processes_per_depth` fresh servers per depth d in 0..max_depth (or
   the job's `depths`), in alternating order, each with `--speculation MODE
   --speculation-depths 1:d` and the job's `extra_arguments` (the profile's
   other settings, verify widths included, stay); a process starts only
   while `planning.process_reserve_seconds` of the work budget remain: test KV
   mode, zero preemptions and prefix-cache evictions, speculation mode on
   /health, memory after warmup recorded, no step drafting past d;
4. per server, for each concurrency C whose rows a depth d can still change
   (C x (d + 1) <= verify_rows, and every C at depth 0), one greedy and one
   fixed-seed sampled window: one batched request per prompt group, the
   window's `groups[C]` groups of C prompts in turn (token IDs from the
   server's /tokenize, chat prompts through its template), `tokens` tokens
   each. Every greedy window must equal W0's bits
   at once and every sampled window at the end: draft depth never changes an
   emitted token.
Per window: request tok/s (C x tokens over send-to-last-token), the step
trace, full-concurrency steps (count, tok/s, shape histogram), per-stream
tok/s, acceptance per draft position (/metrics), energy (100 ms board power of
every GPU) and SM clocks. report.py then selects the schedule.
"""
import argparse
import hashlib
import json
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import faults
import report
import runner as r

S = Path(__file__).resolve().parent
STAGE = S.parent
DATA_RESERVE = 50 * 1024 ** 3
MEMORY = re.compile(r'device memory after warmup rank (\d+): KV (\d+) MiB, (\d+) MiB taken since sizing '
                    r'\(graphs\), (\d+) MiB free')
MEMORY_WARNING = re.compile(r'WARNING: rank (\d+) has (\d+) MiB free after warmup')
PREEMPTIONS = re.compile(r'^vllm:num_preemptions_total(?:\{[^}]*\})? (\S+)$', re.M)
EVICTIONS = re.compile(r'^chop_prefix_cache_evictions_total(?:\{[^}]*\})? (\S+)$', re.M)
WEIGHT_UPLOAD = re.compile(r'^startup weight_upload: (\{.*\})$', re.M)
SEAL = None
spec_metrics = None


class Power:
    """Board power and SM clocks of every GPU every 100 ms for a server's
    lifetime, stamped on arrival. Clocks are recorded, never asserted."""

    QUERY = ['/usr/bin/nvidia-smi', '--query-gpu=index,power.draw.instant,clocks.sm', '--format=csv,noheader,nounits',
             '-lms', '100']

    def __init__(self, directory, dry):
        self.directory, self.dry = directory, dry
        self.samples, self.process, self.thread = [], None, None

    def start(self):
        if self.dry:
            return
        self.process = subprocess.Popen(self.QUERY, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                        start_new_session=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        pending = {}
        for line in self.process.stdout:
            now = time.monotonic()
            try:
                index, watts, mhz = (x.strip() for x in line.split(','))
                pending[int(index)] = (float(watts), float(mhz))
            except ValueError:
                continue
            if len(pending) == r.GPUS:
                self.samples.append((now, [pending[i] for i in range(r.GPUS)]))
                pending = {}

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.thread.join(timeout=10)
        r.write(self.directory / 'power-samples.json', {'query': self.QUERY, 'samples': self.samples})

    def window(self, started, ended, tokens):
        rows = ([(started, [(250.0, 1800.0)] * r.GPUS)] if self.dry
                else [x for x in self.samples if started <= x[0] <= ended])
        if len(rows) < (1 if self.dry else 3):
            raise RuntimeError(f'{len(rows)} power samples in a {ended - started:.1f} s window')
        watts = statistics.fmean(sum(w for w, _ in gpus) for _, gpus in rows)
        seconds = ended - started
        return {'samples': len(rows), 'seconds': seconds, 'mean_board_w': watts,
                'sm_mhz_range': [[min(g[i][1] for _, g in rows), max(g[i][1] for _, g in rows)] for i in range(r.GPUS)],
                'joules_per_token': watts * seconds / tokens, 'synthetic': self.dry}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def validate(payload, dry):
    global SEAL, spec_metrics
    body = (STAGE / 'runtime-seal.json').read_bytes()
    if hashlib.sha256(body).hexdigest() != payload:
        raise RuntimeError('runtime seal differs from the job wrapper')
    m = json.loads(body)
    if not dry and not m['ready']:
        raise RuntimeError('template only')
    for path, wanted in m['sha256'].items():
        actual = Path(path)
        if not actual.is_absolute() or actual.is_symlink() or digest(actual) != wanted:
            raise RuntimeError('sealed file changed: ' + path)
    if m['ready']:
        install = m['install']
        for root in (Path(install['install']), Path(install['prepared']['bundle'])):
            for path in (root, *root.rglob('*')):
                if path.is_symlink() or path.stat().st_mode & 0o022:
                    raise RuntimeError('unsafe frozen install path: ' + str(path))
    SEAL = m
    sys.path.insert(0, str(Path(m['harness_source']) / 'tools/qual'))
    import spec_metrics as imported
    spec_metrics = imported
    model = m['config']['model']
    r.MODEL, r.CHECKPOINT_ENV, r.GPUS = model['served_model_name'], model['checkpoint_env'], model['tensor_parallel_size']
    r.KV_PAGES, r.KV_LINE, r.SERVE = model['kv_pages'], model['kv_pages_line'], model['serve_arguments']
    return m


def idle(directory, label, dry):
    if dry:
        r.write(directory / f'idle-{label}.json', {'synthetic': True, 'passed': True})
        return
    deadline = time.monotonic() + 20
    while True:
        r.command(['/usr/bin/nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                  directory / f'compute-{label}.csv', 15)
        r.command(['/usr/bin/nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                  directory / f'memory-{label}.csv', 15)
        rows = [list(map(int, x.split(','))) for x in (directory / f'memory-{label}.csv').read_text().splitlines()]
        if not (directory / f'compute-{label}.csv').read_text().strip() and [x[0] for x in rows] == list(range(r.GPUS)) \
                and all(x[1] <= 32 for x in rows):
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'{label}: residual GPU process or memory')
        time.sleep(.5)
    r.write(directory / f'idle-{label}.json', {'passed': True, 'memory_mib': rows})


def bits(x):
    return struct.pack('>d', float(x)).hex()


def settle(server, offset):
    """Wait until no step record arrives for a second, so each window holds
    its own steps (a response can precede its last step's record)."""
    log = server.directory / 'server.log'
    pause = .02 if server.dry else .25
    deadline = time.monotonic() + 30
    seen, quiet = -1, 0
    while quiet < 4:
        count = len(r.trace_rows(log, offset))
        quiet = quiet + 1 if count == seen else 0
        seen = count
        if time.monotonic() > deadline:
            raise RuntimeError('engine step trace did not settle')
        time.sleep(pause)
    return r.trace_rows(log, offset)


def full_concurrency(steps, c):
    """Steps with C live items and no prefill, each timed from the previous
    step's completion: count, emitted tokens per second, the step-shape
    histogram (items, rows, verify items) and the drafted share."""
    if any(b['batch'] != a['batch'] + 1 for a, b in zip(steps, steps[1:])):
        raise ValueError('incomplete engine trace')
    rows, micros, tokens, drafted = Counter(), 0.0, 0, 0
    for a, b in zip(steps, steps[1:]):
        items = b['decode_rows'] + b['verify_items']
        if b['prefill_items'] or a['prefill_items'] or items != c:
            continue
        rows[f"i{items}-r{b['decode_rows'] + b['verify_rows']}-v{b['verify_items']}"] += 1
        micros += (b['completed_unix_ns'] - a['completed_unix_ns']) / 1000
        tokens += b['decode_rows'] + b.get('verify_emitted_tokens', 0)
        drafted += bool(b['verify_items'])
    count = sum(rows.values())
    return {'steps': count, 'tokens': tokens, 'seconds': micros / 1e6,
            'tokens_per_second': tokens / micros * 1e6 if micros else 0.0,
            'mean_step_ms': micros / count / 1000 if count else None,
            'histogram': dict(sorted(rows.items())), 'drafted_steps': drafted,
            'drafted_share': drafted / count if count else 0.0}


def batched_finishes(steps, c, sent_ns):
    """Seconds from sending a batched window to each stream's last token,
    from its engine steps; None when they do not account for C streams."""
    last = max((i for i, s in enumerate(steps) if s['prefill_items']), default=-1)
    decode = steps[last + 1:]
    live = [s['decode_rows'] + s['verify_items'] for s in decode]
    if not live or live[0] != c:
        return None
    finishes = []
    for step, here, after in zip(decode, live, live[1:] + [0]):
        finishes += [(step['completed_unix_ns'] - sent_ns) / 1e9] * max(here - after, 0)
    return finishes if len(finishes) == c else None


def streams(finishes, tokens):
    """Per-stream tok/s (tokens over send-to-last-token), with its minimum,
    p10, median and maximum."""
    if not finishes:
        return None
    rates = sorted(tokens / x for x in finishes)
    return {'tokens_per_second': rates, 'min': rates[0], 'p10': rates[int(0.1 * (len(rates) - 1))],
            'median': statistics.median(rates), 'max': rates[-1]}


def check_depth(name, steps, depth):
    for s in steps:
        if s.get('prefill_items') or not s.get('verify_items'):
            continue
        if s['verify_rows'] > s['verify_items'] * (depth + 1):
            raise RuntimeError(f"{name}: batch {s['batch']} verified {s['verify_rows']} rows for "
                               f"{s['verify_items']} items above depth {depth}")


def tokenize(server, prompts):
    """Token IDs of every prompt from the server: raw text as is, chat
    messages through the server's template."""
    out = []
    for p in prompts:
        body = {'model': r.MODEL, **({'messages': p['messages']} if 'messages' in p else
                                     {'prompt': p['prompt'], 'add_special_tokens': True})}
        out.append(json.loads(r.http(server.url, '/tokenize', body, timeout=30))['tokens'])
    if len({tuple(x) for x in out}) != len(out) or min(map(len, out)) < 2:
        raise RuntimeError('workload prompts are not distinct')
    return out


def scrape(server):
    return spec_metrics.snapshot(r.http(server.url, '/metrics'), 'chopinfer', r.MODEL)


def acceptance(before, after):
    value = spec_metrics.difference(before, after)
    if not value.get('available'):
        return None
    return {k: value.get(k) for k in ('draft_tokens', 'accepted_tokens', 'verify_items', 'acceptance_rate',
                                      'mean_acceptance_length', 'per_position')}


def window(server, name, kind, inputs, tokens, salt, sampling):
    """One batched request of `inputs`; greedy as one response, sampled as a
    stream at the workload's fixed seed."""
    body = {'model': r.MODEL, 'prompt': inputs, 'n': 1, 'cache_salt': salt, 'max_tokens': tokens,
            'min_tokens': tokens, 'ignore_eos': True, 'logprobs': 1, 'return_token_ids': True}
    if kind == 'greedy':
        body.update(temperature=0, stream=False)
    else:
        body.update(temperature=sampling['temperature'], top_p=sampling['top_p'], seed=sampling['seed'],
                    stream=True, stream_options={'include_usage': True})
    offset = (server.directory / 'server.log').stat().st_size
    before = scrape(server)
    counters = faults.snapshot(server)
    sent_ns = time.time_ns()
    started = time.monotonic()
    c = len(inputs)
    ids, selected, finish, usage = [[] for _ in range(c)], [[] for _ in range(c)], [None] * c, None
    if kind == 'greedy':
        response = json.loads(r.http(server.url, '/v1/completions', body, timeout=r.remaining(600)))
        usage = response['usage']
        for x in response['choices']:
            ids[x['index']], finish[x['index']] = x['token_ids'], x['finish_reason']
            selected[x['index']] = x['logprobs']['token_logprobs']
    else:
        for event in r.stream(server.url, '/v1/completions', body, timeout=r.remaining(600)):
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                i = choice['index']
                ids[i].extend(choice.get('token_ids') or [])
                selected[i].extend((choice.get('logprobs') or {}).get('token_logprobs') or [])
                finish[i] = choice.get('finish_reason') or finish[i]
    ended = time.monotonic()
    if (not usage or usage['completion_tokens'] != c * tokens or usage['prompt_tokens'] != sum(map(len, inputs))
            or usage['prompt_tokens_details']['cached_tokens'] != 0):
        raise RuntimeError(f'{name}: incomplete or cached workload: {usage}')
    if any(f != 'length' or len(x) != tokens or len(y) != tokens for f, x, y in zip(finish, ids, selected)):
        raise RuntimeError(f'{name}: token count differs')
    steps = settle(server, offset)
    exact = [{'token_ids': x, 'selected_bits': [bits(v) for v in y]} for x, y in zip(ids, selected)]
    return {'started': started, 'ended': ended, 'steps': steps, 'exact': exact,
            'finishes': batched_finishes(steps, c, sent_ns), 'acceptance': acceptance(before, scrape(server)),
            'faults': faults.difference(counters, faults.snapshot(server))}


def merge_full(parts):
    """One window's full-concurrency steps over its prompt groups."""
    histogram = Counter()
    for x in parts:
        histogram.update(x['histogram'])
    steps, tokens, seconds = (sum(x[k] for x in parts) for k in ('steps', 'tokens', 'seconds'))
    drafted = sum(x['drafted_steps'] for x in parts)
    return {'steps': steps, 'tokens': tokens, 'seconds': seconds,
            'tokens_per_second': tokens / seconds if seconds else 0.0,
            'mean_step_ms': seconds * 1000 / steps if steps else None, 'histogram': dict(sorted(histogram.items())),
            'drafted_steps': drafted, 'drafted_share': drafted / steps if steps else 0.0}


def merge_acceptance(parts):
    if any(x is None for x in parts):
        return None
    out = {k: sum(x[k] or 0 for x in parts) for k in ('draft_tokens', 'accepted_tokens', 'verify_items')}
    positions = {}
    for x in parts:
        for row in x['per_position']:
            cell = positions.setdefault(row['position'], [0, 0])
            cell[0] += row['drafted']
            cell[1] += row['accepted']
    out['acceptance_rate'] = out['accepted_tokens'] / out['draft_tokens'] if out['draft_tokens'] else None
    out['mean_acceptance_length'] = 1 + out['accepted_tokens'] / out['verify_items'] if out['verify_items'] else None
    out['per_position'] = [{'position': j, 'drafted': d, 'accepted': a, 'rate': a / d if d else None}
                           for j, (d, a) in sorted(positions.items())]
    return out


def merge_energy(parts, tokens):
    """Energy of the groups' request spans only (not the pauses between)."""
    joules = sum(x['mean_board_w'] * x['seconds'] for x in parts)
    seconds = sum(x['seconds'] for x in parts)
    ranges = [x['sm_mhz_range'] for x in parts]
    return {'samples': sum(x['samples'] for x in parts), 'seconds': seconds, 'mean_board_w': joules / seconds,
            'sm_mhz_range': [[min(r[i][0] for r in ranges), max(r[i][1] for r in ranges)] for i in range(len(ranges[0]))],
            'joules_per_token': joules / tokens, 'synthetic': parts[0]['synthetic']}


def first_difference(reference, exact):
    for i, (a, b) in enumerate(zip(reference, exact)):
        if a != b:
            field = 'token_ids' if a['token_ids'] != b['token_ids'] else 'selected_bits'
            at = next(j for j, (x, y) in enumerate(zip(a[field], b[field])) if x != y)
            return f'stream {i} {field}[{at}]'
    return 'stream count'


def counters(server, name):
    text = r.http(server.url, '/metrics')
    (server.directory / f'{name}.prom').write_text(text)
    found = [PREEMPTIONS.findall(text), EVICTIONS.findall(text)]
    if any(len(x) != 1 for x in found):
        raise RuntimeError(f'{name}: metrics lack one preemption and one eviction counter')
    preemptions, evictions = (float(x[0]) for x in found)
    return {'preemptions': preemptions, 'prefix_cache_evictions': evictions}


def weight_source(log):
    rows = [json.loads(x) for x in WEIGHT_UPLOAD.findall(log)]
    uploaded = sum(int(x.get('uploaded_bytes') or 0) for x in rows)
    lent = sum(int(x.get('lent_bytes') or 0) for x in rows)
    source = 'borrowed' if lent and not uploaded else 'loaded' if uploaded and not lent else 'mixed' if lent else 'none'
    return {'source': source, 'images': len(rows), 'uploaded_bytes': uploaded, 'lent_bytes': lent}


def memory_lines(log):
    ranks = [{'rank': int(a), 'kv_mib': int(kv), 'taken_mib': int(taken), 'free_mib': int(free)}
             for a, kv, taken, free in MEMORY.findall(log)]
    warnings = [{'rank': int(a), 'free_mib': int(free)} for a, free in MEMORY_WARNING.findall(log)]
    return {'ranks': ranks, 'warnings': warnings}


def windows_for(config, depth, every=False):
    """The (kind, C) windows a depth measures: every C at depth 0 (and for
    W0), otherwise the C whose rows depth `depth` can still change."""
    model = config['model']
    cs = [c for c in model['concurrencies'] if every or depth == 0 or c * (depth + 1) <= model['verify_rows']]
    return [(kind, c) for kind in ('greedy', 'sampled') for c in cs]


def run_process(run, label, depth, state, dry, every=False):
    config = SEAL['config']
    model = config['model']
    install = SEAL['install']
    root = run / label
    arguments = ['--speculation', model['speculation_mode'], '--speculation-depths', f'1:{depth}',
                 *model.get('extra_arguments', [])]
    server = r.Server(root / 'server', install['install'], install['cache'], arguments, dry, S / 'stub.py')
    record = state['processes'][label] = {'depth': depth, 'args': arguments, 'windows': {}}
    power = Power(root, dry)
    kv_line = re.compile(re.escape(model['kv_pages_line']).replace(re.escape('{pages}'), r'(\d+)'))
    try:
        power.start()
        server.start()
        record['readiness_seconds'] = server.readiness
        mode = ((server.health.get('serving') or {}).get('profile') or {}).get('speculation', {}).get('mode')
        record['speculation_mode'] = mode
        if mode != model['speculation_mode']:
            raise RuntimeError(f"{label}: serves speculation {mode}, expected {model['speculation_mode']}")
        log = (server.directory / 'server.log').read_text(errors='replace')
        record['weights'] = weight_source(log)
        if not dry and record['weights']['source'] != 'loaded':
            raise RuntimeError(f"{label}: weights {record['weights']}; without a weight keeper every server loads")
        found = kv_line.findall(log)
        record['kv_pages_per_rank'] = [int(x) for x in found]
        if not found or any(int(x) != r.KV_PAGES for x in found):
            raise RuntimeError(f'{label}: KV pages {found}, expected {r.KV_PAGES} per rank')
        if not state['tokens']:
            state['tokens'] = tokenize(server, SEAL['prompts'])
            r.write(run / 'prompt-tokens.json', state['tokens'])
        inputs = state['tokens']
        for c in config['warm']:
            warm = [list(x) for x in inputs[:c]]
            for x in warm:
                x[1] = x[0]
            window(server, f'warm-c{c}', 'greedy', warm, config['warm_tokens'], f'tune-{label}-warm-c{c}', None)
        record['memory_after_warmup'] = memory_lines((server.directory / 'server.log').read_text(errors='replace'))
        record['metrics_after_warmup'] = counters(server, 'after-warmup')
        for kind, c in windows_for(config, depth, every):
            name = f'{kind}-c{c}'
            tokens = config['tokens'].get(str(c), config['tokens']['default'])
            # Low concurrencies serve several prompt groups in turn, so a
            # window covers more than its first C prompts.
            groups = model.get('groups', {}).get(str(c), 1)
            parts = []
            for g in range(groups):
                w = window(server, f'{name}-g{g}', kind, [list(x) for x in inputs[g * c:(g + 1) * c]], tokens,
                           f'tune-{label}-{name}-g{g}', config['sampling'])
                check_depth(f'{label} {name} group {g}', w['steps'], depth)
                w['energy'] = power.window(w['started'], w['ended'], c * tokens)
                parts.append(w)
            seconds = sum(w['ended'] - w['started'] for w in parts)
            finishes = [f for w in parts for f in w['finishes']] if all(w['finishes'] for w in parts) else None
            value = {'name': name, 'kind': kind, 'concurrency': c, 'depth': depth, 'tokens': tokens, 'groups': groups,
                     'request_seconds': seconds, 'request_tokens_per_second': groups * c * tokens / seconds,
                     'full': merge_full([full_concurrency(w['steps'], c) for w in parts]),
                     'streams': streams(finishes, tokens), 'acceptance': merge_acceptance([w['acceptance'] for w in parts]),
                     'energy': merge_energy([w['energy'] for w in parts], groups * c * tokens),
                     'faults': [w['faults'] for w in parts],
                     'prompts': [p['sha256'] for p in SEAL['prompts'][:groups * c]]}
            exact = [stream for w in parts for stream in w['exact']]
            r.write(root / 'windows' / f'{name}.json', {**value, 'steps': [w['steps'] for w in parts], 'exact': exact})
            record['windows'][name] = {k: value[k] for k in ('request_tokens_per_second', 'full')}
            reference = state['reference'].setdefault(name, exact)
            if exact is not reference:
                if exact != reference:
                    where = first_difference(reference, exact)
                    if kind == 'greedy':
                        raise RuntimeError(f'{label} {name}: greedy output differs from W0 at {where}')
                    state['sampled_differences'].append(f'{label} {name}: differs from W0 at {where}')
                state['exact'][name] = state['exact'].get(name, 0) + 1
            save(run, state)
        record['metrics'] = counters(server, 'final')
        if record['metrics']['preemptions'] or record['metrics']['prefix_cache_evictions']:
            raise RuntimeError(f"{label}: {record['metrics']} (the test KV mode requires zero)")
    finally:
        try:
            server.stop()
        finally:
            power.stop()
            save(run, state)


def order(config):
    """Process labels: passes alternate ascending and descending depth, and
    later passes start from the middle."""
    depths = config['model'].get('depths') or list(range(config['model']['max_depth'] + 1))
    out = []
    for p in range(config['model']['processes_per_depth']):
        seq = depths if p % 2 == 0 else depths[::-1]
        shift = (p // 2) * (len(depths) // 2)
        out += [f'd{d}p{p + 1}' for d in seq[shift:] + seq[:shift]]
    return out


def save(run, state):
    out = {k: v for k, v in state.items() if k not in ('reference', 'tokens')}
    r.write(run / 'summary.json', out)
    lines = [f"{SEAL['config']['name']}: {state['status']}", '']
    for label, row in state['processes'].items():
        memory = row.get('memory_after_warmup') or {}
        lines.append(f"{label}: depth {row['depth']}, ready {row.get('readiness_seconds', 0):.1f} s, weights "
                     f"{row.get('weights', {}).get('source')}, KV pages {row.get('kv_pages_per_rank')}, metrics "
                     f"{row.get('metrics')}; memory after warmup " + '; '.join(
                         f"rank {m['rank']}: {m['free_mib']} MiB free" for m in memory.get('ranks', []))
                     + (f"; WARNINGS {memory['warnings']}" if memory.get('warnings') else ''))
        for name, value in row['windows'].items():
            full = value['full']
            lines.append(f"  {name}: {value['request_tokens_per_second']:.1f} tok/s, full {full['steps']} steps "
                         f"{full['tokens_per_second']:.1f} tok/s, {dict(Counter(full['histogram']).most_common(3))}")
    lines += ['', *(f'Exact against W0: {k} in {n} timed windows' for k, n in sorted(state['exact'].items()))]
    lines += ['Sampled difference: ' + x for x in state['sampled_differences']]
    lines += [f"Power {k}: limits {v.get('power_limits_w')} W, SM MHz {v.get('sm_clock_mhz')}"
              for k, v in state['power'].items()]
    lines += ['Error: ' + x for x in state['errors']]
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(args):
    m = validate(args.payload_sha, args.dry_run)
    config = m['config']
    r.EXPECTED_POWER_W = args.expected_power_w if args.expected_power_w is not None else config.get('expected_power_w')
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    r.write(run / 'config.json', config)
    r.write(run / 'prompts.json', m['prompts'])
    state = {'status': 'running', 'dry_run': args.dry_run, 'revision': m['install'].get('revision'),
             'order': ['W0', *order(config)], 'processes': {}, 'exact': {}, 'reference': {}, 'tokens': [],
             'sampled_differences': [], 'power': {}, 'telemetry': {}, 'errors': []}
    r.WORK_DEADLINE = time.monotonic() + config['work_seconds']
    since = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    began = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, r.interrupted)
    reference = None
    try:
        if shutil.disk_usage(run).free < DATA_RESERVE:
            raise RuntimeError('the output volume is under its free-space reserve')
        before = state['telemetry']['before'] = r.telemetry(run, 'before', args.dry_run, since)
        reference = before['power_limits_w']
        state['power']['before'] = before['power']
        deepest = max(config['model'].get('depths') or [config['model']['max_depth']])
        reserve = config['model']['planning']['process_reserve_seconds']
        for label in state['order']:
            depth = deepest if label == 'W0' else int(label[1:].split('p')[0])
            left = r.WORK_DEADLINE - time.monotonic()
            if left < reserve:
                raise RuntimeError(f'work budget: {left:.0f} s left, under the {reserve} s a process needs; '
                                   f'stopped before {label}')
            idle(run, f'before-{label}', args.dry_run)
            run_process(run, label, depth, state, args.dry_run, every=label == 'W0')
            state['power'][label] = r.power_check(run, f'after-{label}', args.dry_run, reference)
            save(run, state)
        if state['sampled_differences']:
            raise RuntimeError(f"fixed-seed sampled outputs differ from W0: {state['sampled_differences']}")
        state['status'] = 'passed'
    except BaseException as error:
        state['status'] = 'failed'
        state['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            after = state['telemetry']['after'] = r.telemetry(run, 'after', args.dry_run, since, reference)
            state['power']['after'] = after['power']
            if after['xids']:
                raise RuntimeError('kernel Xids during the job')
        except BaseException as error:
            state['status'] = 'failed'
            state['errors'].append('cleanup: ' + str(error))
        state['elapsed_seconds'] = time.monotonic() - began
        try:
            state['depths'] = report.evaluate(run)['depths']
        except Exception as error:
            state['errors'].append(f'report: {type(error).__name__}: {error}')
        save(run, state)
    print(json.dumps({'output': str(run), 'status': state['status'], 'depths': state.get('depths'),
                      'errors': state['errors'], 'elapsed_seconds': round(state['elapsed_seconds'], 1)}), flush=True)
    return int(state['status'] != 'passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--payload-sha', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--expected-power-w', type=float)
    raise SystemExit(main(ap.parse_args()))
