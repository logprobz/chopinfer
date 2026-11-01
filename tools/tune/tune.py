#!/usr/bin/env python3
"""Sealed depth-tuning and served A/B jobs.

Sweep, costs and exact modes use fixed-token batches and an initial warm
reference pass. A/B uses matched-seed ABBA visits, each in a fresh process,
primed 15-second windows, and concurrency-scoped exactness with unmatched replay.
Speculative A/B visits require the keeper. Cold Engram windows acknowledge
cache reset and page-table release, advise backing files and record residency.

Usage: python3 -B STAGE/harness/tune.py --output RUN --payload-sha SHA [--dry-run]

All modes check zero cached prompt tokens, depth and route evidence, KV
preemptions and evictions. Enforced power limits must remain constant within
the job; clocks are recorded and never set. CPU stub results test mechanics.
"""
import argparse
import hashlib
import json
import os
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faults
import prompts as prompt_builder
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
SEAL_RUN = None
spec_metrics = None


class Power:
    """Board power, SM clocks and the software power-cap clock event of
    every GPU every 100 ms for a server's lifetime, stamped on arrival.
    Clocks and power capping are recorded, never asserted."""

    QUERY = ['/usr/bin/nvidia-smi', '--query-gpu=index,power.draw.instant,clocks.sm,enforced.power.limit,'
             'clocks_event_reasons.sw_power_cap', '--format=csv,noheader,nounits', '-lms', '100']

    def __init__(self, directory, dry):
        self.directory, self.dry = directory, dry
        self.samples, self.process, self.thread = [], None, None
        self.limits = []
        self.reference = None

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
                index, watts, mhz, limit, capped = (x.strip() for x in line.split(','))
                pending[int(index)] = (float(watts), float(mhz), capped == 'Active')
                self.limits.append((now, int(index), float(limit)))
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
        r.write(self.directory / 'power-limits.json', self.limits)
        if self.limits:
            by_gpu = {i: {v for _, index, v in self.limits if index == i} for i in range(r.GPUS)}
            if any(len(values) != 1 or (self.reference is not None and values != {self.reference[i]})
                   for i, values in by_gpu.items()):
                raise r.PowerChanged('enforced power limits changed within a visit')

    def window(self, started, ended, tokens):
        rows = ([(started, [(250.0, 1800.0, False)] * r.GPUS)] if self.dry
                else [x for x in self.samples if started <= x[0] <= ended])
        if len(rows) < (1 if self.dry else 3):
            raise RuntimeError(f'{len(rows)} power samples in a {ended - started:.1f} s window')
        watts = statistics.fmean(sum(g[0] for g in gpus) for _, gpus in rows)
        seconds = ended - started
        return {'samples': len(rows), 'seconds': seconds, 'mean_board_w': watts,
                'enforced_power_limits_w': self.reference,
                'sm_mhz_range': [[min(g[i][1] for _, g in rows), max(g[i][1] for _, g in rows)] for i in range(r.GPUS)],
                'sm_mhz_mean': [statistics.fmean(g[i][1] for _, g in rows) for i in range(r.GPUS)],
                'power_cap_share': [sum(g[i][2] for _, g in rows) / len(rows) for i in range(r.GPUS)],
                'joules_per_token': watts * seconds / tokens, 'synthetic': self.dry}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def sealed_installs(seal):
    if 'installs' in seal:
        return seal['installs'], seal['default_install']
    install = seal['install']
    leaf = Path(install['install']).name
    name = leaf[len('install-'):] if leaf.startswith('install-') else 'head'
    return {name: install}, name


def arm_install(seal, arm):
    installs, default = sealed_installs(seal)
    name = arm.get('install', default)
    if name not in installs:
        raise ValueError(f"{arm['label']}: unsealed install {name}")
    return name, installs[name]


def arm_identity(seal, arm):
    name, install = arm_install(seal, arm)
    return {'install': name, 'revision': install.get('revision')}


def check_installs(seal, require_proofs=False):
    installs, default = sealed_installs(seal)
    if len(installs) > 1 and seal['config']['model'].get('mode') != 'ab':
        raise ValueError('multiple installs require mode ab')
    if default not in installs or installs[default] != seal['install']:
        raise ValueError('default install differs from the first sealed install')
    used = {arm_install(seal, arm)[0] for arm in arms(seal['config'])}
    if len(installs) > 1 and (unused := set(installs) - used):
        raise ValueError('unused sealed install: ' + ', '.join(sorted(unused)))
    if len(installs) > 1 and require_proofs:
        keys = {install.get('content_key') for install in installs.values()}
        if len(keys) != 1 or not all(isinstance(key, str) and re.fullmatch('[0-9a-f]{64}', key) for key in keys):
            raise ValueError('multi-install seal needs matching keeper content keys')
        for install in installs.values():
            proof = install.get('weight_keeper_identity')
            if not proof or not Path(proof).is_absolute() or proof not in seal['sha256']:
                raise ValueError('multi-install seal is missing a sealed content proof')


def draft_option(word):
    return word.startswith(('--speculation', '--speculative-', '--draft-'))


def arm_environment(config, arm):
    environment = arm.get('env', {})
    owned = {'CHOPINFER_DATA', config['model'].get('checkpoint_env'), 'CHOPINFER_WEIGHT_KEEPER_DIR',
             'CHOPINFER_STEP_TRACE', 'CHOPINFER_ROUTE_COUNTERS', 'CHOPINFER_MAPPED_ROW_CONTROL'}
    if (not isinstance(environment, dict)
            or any(not re.fullmatch('CHOPINFER_[A-Z0-9_]+', key) or key in owned
                   or (value is not None and (not isinstance(value, str) or '\0' in value))
                   for key, value in environment.items())):
        raise ValueError('arm env must name model controls with string or null values, without launcher overrides')
    return environment


def arm_arguments(config, arm, install=None):
    model = config['model']
    speculation = [] if model['speculation_mode'] == 'off' else [
        '--speculation', model['speculation_mode'], '--speculation-depths', arm.get('depths', f"1:{arm['depth']}")]
    return [*speculation,
            *model.get('extra_arguments', []), *arm['arguments'],
            *(['--weight-keeper-identity', install['weight_keeper_identity']]
              if install and 'weight_keeper_identity' in install else [])]


def arm_bundle(config, install):
    return install['prepared']['bundle'] if config['model']['speculation_mode'] == 'off' else None


def check_kv_config(config):
    model = config['model']
    pool = model.get('kv_pages')
    if config.get('cold_engram') and pool != 'default':
        raise ValueError('cold Engram requires the full default KV pool')
    if pool == 'default':
        for arm in arms(config):
            if any(word.split('=', 1)[0] in ('--kv-pages', '--kv-memory-fraction', '--gpu-memory-utilization')
                   for word in [*model.get('serve_arguments', []), *arm_arguments(config, arm)]):
                raise ValueError('default KV pool cannot have an explicit size or memory fraction override')


def kv_pages(log, config, state, label):
    model = config['model']
    pattern = re.escape(model['kv_pages_line']).replace(re.escape('{pages}'), r'(\d+)')
    pages = [int(value) for value in re.findall(pattern, log)]
    if not pages or any(value <= 0 for value in pages):
        raise RuntimeError(f'{label}: missing or invalid KV page count')
    if model['kv_pages'] == 'default':
        if state.setdefault('kv_pages_per_rank', pages) != pages:
            raise RuntimeError(f"{label}: default KV pages {pages} differ from job baseline {state['kv_pages_per_rank']}")
    elif any(value != model['kv_pages'] for value in pages):
        raise RuntimeError(f"{label}: KV pages {pages}, expected {model['kv_pages']} per rank")
    return pages


def validate(payload, dry):
    global SEAL, spec_metrics
    body = (STAGE / 'runtime-seal.json').read_bytes()
    if hashlib.sha256(body).hexdigest() != payload:
        raise RuntimeError('runtime seal differs from the job wrapper')
    m = json.loads(body)
    check_kv_config(m['config'])
    if not dry and not m['ready']:
        raise RuntimeError('template only')
    for path, wanted in m['sha256'].items():
        actual = Path(path)
        if not actual.is_absolute() or actual.is_symlink() or digest(actual) != wanted:
            raise RuntimeError('sealed file changed: ' + path)
    if 'prompt_file' in m:
        source = m['prompt_file']
        if (m['sha256'].get(source['path']) != source['sha256']
                or Path(source['path']).stat().st_size != source['size']):
            raise RuntimeError('sealed prompt file identity differs')
    for identity in m.get('cold_files', []):
        if faults.file_identity(identity['path']) != identity:
            raise RuntimeError('sealed cold image metadata changed')
    check_installs(m, require_proofs=True)
    if m['ready']:
        for install in sealed_installs(m)[0].values():
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


def gpu_memory(directory, label):
    """Each GPU's index, bus ID and used MiB."""
    r.command(['/usr/bin/nvidia-smi', '--query-gpu=index,pci.bus_id,memory.used', '--format=csv,noheader,nounits'],
              directory / f'memory-{label}.csv', 15)
    rows = [[x.strip() for x in line.split(',')] for line in (directory / f'memory-{label}.csv').read_text().splitlines()
            if line.strip()]
    return [(int(index), bus.lower(), int(used)) for index, bus, used in rows]


def idle(directory, label, dry, lent=None):
    """No GPU compute process, and no GPU memory beyond what the weight
    keeper lends (`lent`: MiB by bus ID)."""
    if dry:
        r.write(directory / f'idle-{label}.json', {'synthetic': True, 'passed': True})
        return
    lent = lent or {}
    deadline = time.monotonic() + 20
    while True:
        r.command(['/usr/bin/nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                  directory / f'compute-{label}.csv', 15)
        rows = gpu_memory(directory, label)
        held = [lent.get(bus[-12:], 0) for _, bus, _ in rows]
        if not (directory / f'compute-{label}.csv').read_text().strip() and [x[0] for x in rows] == list(range(r.GPUS)) \
                and all(used <= keep + 32 for (_, _, used), keep in zip(rows, held)):
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'{label}: residual GPU process or memory')
        time.sleep(.5)
    r.write(directory / f'idle-{label}.json', {'passed': True, 'memory_mib': [x[2] for x in rows],
                                               'keeper_mib': held})


def keeper(run, dry):
    """The weight keeper runner6 started for this job (`# keeper use`), if
    it answers: its directory and the MiB it lends on each GPU by bus ID."""
    directory = os.environ.get('CHOPINFER_WEIGHT_KEEPER_DIR')
    if not directory:
        return None
    if dry:
        return {'directory': directory, 'lent_mib': {}, 'synthetic': True}
    _, install = arm_install(SEAL, arms(SEAL['config'])[0])
    binary = Path(install['install']) / 'bin/chopinfer'
    log = run / 'keeper-status.txt'
    code = r.command([binary, 'weights', 'status', '--keeper-dir', directory], log, 60, check=False)
    text = log.read_text(errors='replace')
    try:
        registry = json.loads(text[text.index('{'):])
    except ValueError:
        registry = None
    if code or not registry:
        r.write(run / 'keeper.json', {'available': False, 'exit_code': code})
        return None
    lent = {}
    for device in registry.get('devices', []):
        key = str(device.get('device', '')).lower()[-12:]
        lent[key] = lent.get(key, 0) + int(device.get('bytes', 0)) // 2 ** 20
    value = {'directory': directory, 'lent_mib': lent, 'pid': registry.get('pid')}
    r.write(run / 'keeper.json', {'available': True, **value})
    return value


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


def shape_key(step, histogram=False):
    items = step['decode_rows'] + step['verify_items']
    key = f"i{items}-r{step['decode_rows'] + step['verify_rows']}"
    if histogram or 'reserved_rows' in step:
        key += f"-v{step['verify_items']}"
    if 'reserved_rows' in step:
        key += f"-R{step['decode_rows'] + step['reserved_rows']}"
    return key


def full_concurrency(steps, c):
    """Steps with C live items and no prefill, each timed from the previous
    step's completion: count, emitted tokens per second, the step-shape
    histogram (items, rows, verify items) and the drafted share."""
    if any(b['batch'] != a['batch'] + 1 for a, b in zip(steps, steps[1:])):
        raise ValueError('incomplete engine trace')
    rows, micros, tokens, drafted, allocated = Counter(), 0.0, 0, 0, 0
    for a, b in zip(steps, steps[1:]):
        items = b['decode_rows'] + b['verify_items']
        if b['prefill_items'] or a['prefill_items'] or items != c:
            continue
        rows[shape_key(b, histogram=True)] += 1
        micros += (b['completed_unix_ns'] - a['completed_unix_ns']) / 1000
        # Fixed-length benchmark requests reserve the mandatory target
        # token even when allocation assigns an item no draft. Older trace
        # folds leave those items out of verify_emitted_tokens.
        tokens += b['decode_rows'] + (b['verify_items'] + b['accepted_tokens'] if 'reserved_rows' in b
                                     else b.get('verify_emitted_tokens', 0))
        drafted += bool(b['verify_items'])
        allocated += 'reserved_rows' in b
    count = sum(rows.values())
    return {'steps': count, 'tokens': tokens, 'seconds': micros / 1e6,
            'tokens_per_second': tokens / micros * 1e6 if micros else 0.0,
            'mean_step_ms': micros / count / 1000 if count else None,
            'histogram': dict(sorted(rows.items())), 'drafted_steps': drafted,
            'drafted_share': drafted / count if count else 0.0,
            'allocated_steps': allocated, 'allocated_share': allocated / count if count else 0.0}


def cells(steps):
    """Each decode or verify step's cycle time (from the previous step's
    completion) and execution time in milliseconds, by `i{items}-r{rows}`,
    over steps without prefill in either."""
    out = {}
    for a, b in zip(steps, steps[1:]):
        if b['prefill_items'] or a['prefill_items'] or b['batch'] != a['batch'] + 1:
            continue
        cell = out.setdefault(shape_key(b), {'cycle_ms': [], 'execution_ms': []})
        cell['cycle_ms'].append((b['completed_unix_ns'] - a['completed_unix_ns']) / 1e6)
        if b.get('execution_us') is not None:
            cell['execution_ms'].append(b['execution_us'] / 1000)
    return out


def merge_cells(parts):
    out = {}
    for part in parts:
        for key, cell in part.items():
            into = out.setdefault(key, {'cycle_ms': [], 'execution_ms': []})
            into['cycle_ms'] += cell['cycle_ms']
            into['execution_ms'] += cell['execution_ms']
    return out


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


def schedule_depths(text):
    """An arm's `depths` in `--speculation-depths` form (ITEMS:DEPTH,...,
    items strictly increasing from 1), as (items, depth) pairs."""
    if not isinstance(text, str) or not re.fullmatch(r'[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*', text):
        raise ValueError(f'invalid depth schedule: {text!r}')
    pairs = [tuple(int(x) for x in part.split(':')) for part in text.split(',')]
    items = [i for i, _ in pairs]
    if items[0] != 1 or any(b <= a for a, b in zip(items, items[1:])):
        raise ValueError(f'depth schedule items must start at 1 and increase: {text!r}')
    return pairs


def step_depth(depth, items):
    """The depth bound for a batch of `items`: an int, or a schedule whose
    last entry at or below `items` applies."""
    if isinstance(depth, int):
        return depth
    return [d for i, d in schedule_depths(depth) if i <= items][-1]


def check_depth(name, steps, depth):
    for s in steps:
        if s.get('prefill_items') or not s.get('verify_items'):
            continue
        bound = step_depth(depth, s['verify_items'])
        if s['verify_rows'] > s['verify_items'] * (bound + 1):
            raise RuntimeError(f"{name}: batch {s['batch']} verified {s['verify_rows']} rows for "
                               f"{s['verify_items']} items above depth {bound}")


def argument_value(arguments, flag, default=None):
    values = [arguments[i + 1] for i, x in enumerate(arguments[:-1]) if x == flag]
    return values[-1] if values else default


def confidence_startup(log, arguments):
    requested = argument_value(arguments, '--speculation-confidence', 'off') != 'off'
    lines = re.findall(r'^startup speculation_confidence: (on|off)(?:, (.*))?$', log, re.M)
    if len(lines) > 1:
        raise RuntimeError('multiple confidence startup lines')
    enabled = bool(lines and lines[0][0] == 'on')
    if requested != enabled:
        raise RuntimeError(f'confidence requested {requested}, server reported {lines or "no startup line"}')
    return {'enabled': enabled, 'reason': lines[0][1] if lines else 'legacy server without confidence'}


def confidence_floor(arguments):
    """The floor of the --speculation-confidence calibration, None when off."""
    value = argument_value(arguments, '--speculation-confidence', 'off')
    return None if value == 'off' else json.loads(value)['floor']


def confidence_chooses(floor, items, width, drafts):
    """Whether confidence chooses the verified drafts of `items` items that
    each reserve `drafts` within `width` rows (the engine's
    `VerifyAllocation::chooses`): rows remain after the floor, and at least
    two items reserve drafts past it. Otherwise the step is not allocated."""
    floor = next((f for f in range(floor, -1, -1) if items * (1 + min(f, drafts)) <= width), 0)
    return items >= 2 and drafts > floor and items * (1 + min(floor, drafts)) < width


def allocation_possible(config, arm, items):
    """A full cohort with executable drafts and output room can reserve
    beyond its balanced width, and confidence chooses among the reserved
    drafts. Tail limits may reduce individual steps."""
    model = config['model']
    args = [*model.get('serve_arguments', []), *model.get('extra_arguments', []), *arm['arguments']]
    capacity = int(argument_value(args, '--speculative-rows', model['verify_rows']))
    rows = min(items * (step_depth(arm.get('depths', arm['depth']), items) + 1), capacity)
    widths = argument_value(args, '--speculation-widths')
    widths = [int(x) for x in widths.split(',')] if widths else model.get('verify_widths', [])
    width = max((w for w in widths if items <= w <= rows), default=rows)
    drafts = min(model['max_depth'], max(0, capacity // items - 1))
    reserved = items * (1 + drafts)
    return items < width < reserved and confidence_chooses(confidence_floor(args) or 0, items, width, drafts)


def check_allocations(name, steps, enabled, required):
    count = sum('reserved_rows' in step for step in steps)
    if count and not enabled:
        raise RuntimeError(f'{name}: {count} allocated steps without confidence')
    if enabled and required and not count:
        raise RuntimeError(f'{name}: no allocated steps where reservations can exceed width')
    return count


STEP_PHASES = ('prefill', 'verify', 'decode')


def step_phases(step):
    """The kinds of rows a step record carries: prompt tokens, verify rows
    and plain decode rows. A mixed step carries several."""
    return {phase for phase, key in zip(STEP_PHASES, ('prefill_tokens', 'verify_rows', 'decode_rows'))
            if step.get(key)}


def fallback_scopes(name, allow):
    """Each allowed family's step phases, None for every step. A family's
    value is its reason, or {"reason": ..., "phases": [...]} admitting it
    only in steps that carry one of those phases."""
    if not isinstance(allow, dict):
        raise RuntimeError(f'{name}: allow_fallbacks needs a reason per family')
    scopes = {}
    for family, value in allow.items():
        phases = None
        if isinstance(value, dict):
            listed = value.get('phases')
            if set(value) != {'reason', 'phases'} or not isinstance(listed, list) or not listed \
                    or len(set(listed)) != len(listed) or not set(listed) <= set(STEP_PHASES):
                raise RuntimeError(f'{name}: allow_fallbacks {family} needs a reason and phases from {STEP_PHASES}')
            value, phases = value['reason'], set(listed)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f'{name}: allow_fallbacks needs a reason per family')
        scopes[family] = phases
    return scopes


def arm_fallbacks(config, arm):
    """The allowances an arm's windows check: the job's, with the arm's own
    `allow_fallbacks` adding a family or replacing its job allowance for
    that arm only, so a candidate's expected route stays a failure in its
    baseline's windows."""
    own = arm.get('allow_fallbacks', {})
    fallback_scopes(f"{arm['label']} arm", own)
    return {**config.get('allow_fallbacks', {}), **own}


def check_fallbacks(name, steps, allow):
    """A served window needs complete route evidence and explained fallbacks,
    each in a step of a phase its allowance admits."""
    scopes = fallback_scopes(name, allow)
    if not steps or any(not isinstance(step.get('route'), dict) for step in steps):
        raise RuntimeError(f'{name}: missing route counters')
    families = {}
    total = 0
    for step in steps:
        route = step['route']
        counts = route.get('families')
        if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise RuntimeError(f'{name}: invalid route family counters')
        count = route.get('fallback_launches')
        if type(count) is not int or count != sum(counts.values()):
            raise RuntimeError(f'{name}: inconsistent route counters')
        total += count
        for family, n in counts.items():
            if n and family not in scopes:
                raise RuntimeError(f'{name}: unallowed fallback family {family}: {n} launches')
            phases = scopes.get(family)
            if n and phases is not None and not phases & step_phases(step):
                kinds = '/'.join(p for p in STEP_PHASES if p in step_phases(step)) or 'empty'
                raise RuntimeError(f'{name}: fallback family {family} outside its allowed phases '
                                   f'({", ".join(p for p in STEP_PHASES if p in phases)}): {n} launches in a {kinds} step')
            families[family] = families.get(family, 0) + n
    return {'fallback_launches': total, 'families': families}


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
    allocated = sum(x.get('allocated_steps', 0) for x in parts)
    return {'steps': steps, 'tokens': tokens, 'seconds': seconds,
            'tokens_per_second': tokens / seconds if seconds else 0.0,
            'mean_step_ms': seconds * 1000 / steps if steps else None, 'histogram': dict(sorted(histogram.items())),
            'drafted_steps': drafted, 'drafted_share': drafted / steps if steps else 0.0,
            'allocated_steps': allocated, 'allocated_share': allocated / steps if steps else 0.0}


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
    samples = sum(x['samples'] for x in parts)
    merged = {'samples': samples, 'seconds': seconds, 'mean_board_w': joules / seconds,
              'sm_mhz_range': [[min(r[i][0] for r in ranges), max(r[i][1] for r in ranges)] for i in range(len(ranges[0]))],
              'joules_per_token': joules / tokens, 'synthetic': parts[0]['synthetic']}
    # Per-GPU means over every sample of the groups, where each part has them.
    for key in ('sm_mhz_mean', 'power_cap_share'):
        if all(key in x for x in parts):
            merged[key] = [sum(x[key][i] * x['samples'] for x in parts) / samples for i in range(len(ranges[0]))]
    return merged


def first_difference(reference, exact):
    for i, (a, b) in enumerate(zip(reference, exact)):
        if a != b:
            field = 'token_ids' if a['token_ids'] != b['token_ids'] else 'selected_bits'
            at = next(j for j, (x, y) in enumerate(zip(a[field], b[field])) if x != y)
            return f'stream {i} {field}[{at}]'
    return 'stream count'


def parse_counters(text, name):
    """Sum eviction classes, retaining support for older unlabelled recordings."""
    preemptions = PREEMPTIONS.findall(text)
    evictions = EVICTIONS.findall(text)
    if len(preemptions) != 1 or not evictions:
        raise RuntimeError(f'{name}: metrics need one preemption counter and at least one eviction series')
    return {'preemptions': float(preemptions[0]), 'prefix_cache_evictions': sum(map(float, evictions))}


def counters(server, name):
    text = r.http(server.url, '/metrics')
    (server.directory / f'{name}.prom').write_text(text)
    return parse_counters(text, name)


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
    the reference pass), otherwise the C whose rows depth `depth` can still
    change. Kinds follow the config's `kinds`."""
    model = config['model']
    cs = [c for c in model['concurrencies'] if every or model.get('mode') in ('ab', 'exact')
          or depth == 0 or c * (depth + 1) <= model['verify_rows']]
    kinds = [kind for kind in ('greedy', 'sampled') if kind in config.get('kinds', {'greedy': 1, 'sampled': 1})]
    return [(kind, c) for kind in kinds for c in cs]


def arms(config):
    """Server arms in run order, one server each: the model config's `arms`
    ({label, depth, arguments}), or one per depth of `depths` (or
    0..max_depth), the deepest first (it also runs the reference pass), then
    the rest ascending."""
    model = config['model']
    if model.get('speculation_mode') == 'off':
        if ('depths' in model or model.get('max_depth', 0) != 0 or model.get('mode', 'sweep') == 'sweep'
                or any('depth' in arm for arm in model.get('arms', []))):
            raise ValueError('AR jobs cannot declare depth arms or depth sweeps')
        declared = [{'arguments': [], **arm, 'depth': 0} for arm in model.get('arms') or [{'label': 'ar'}]]
        for arm in declared:
            if any(draft_option(word) for word in [*model.get('serve_arguments', []),
                                                  *arm_arguments(config, arm)]):
                raise ValueError('AR jobs cannot pass speculation or draft flags')
            arm_environment(config, arm)
            arm_fallbacks(config, arm)
        return declared
    if model.get('arms'):
        declared = []
        for arm in model['arms']:
            arm_environment(config, arm)
            arm_fallbacks(config, arm)
            if 'depths' in arm:
                # A production schedule (for example a profile's default)
                # instead of one uniform depth; `depth` is its deepest entry.
                if 'depth' in arm or model.get('speculation_mode') != 'balanced':
                    raise ValueError(f"{arm['label']}: a depth schedule replaces depth and needs balanced mode")
                arm = {**arm, 'depth': max(d for _, d in schedule_depths(arm['depths']))}
            declared.append({'arguments': [], **arm})
        return declared
    depths = model.get('depths') or list(range(model['max_depth'] + 1))
    deepest = max(depths)
    return [{'label': f'd{d}', 'depth': d, 'arguments': []}
            for d in [deepest, *sorted(d for d in depths if d != deepest)]]


def planned_windows(config):
    """Timed windows the job runs: every arm's windows, `repeats` times."""
    matched = matched_cold(config)
    return sum(len(windows) for visit in visits(config) for _, windows in visit['passes']) * (
        len(matched['lengths']) if matched else 1)


def matched_cold(config):
    cold = config.get('cold_engram', {})
    if 'matched' not in cold:
        return None
    spec = cold['matched']
    if not isinstance(spec, dict) or set(spec) != {'lengths', 'tolerance'}:
        raise ValueError('matched cold needs lengths and tolerance')
    lengths, tolerance = spec['lengths'], spec['tolerance']
    model = config['model']
    if (not isinstance(lengths, list) or not lengths or any(type(n) is not int or n <= 0 for n in lengths)
            or type(tolerance) is not int or not 0 <= tolerance < min(lengths)
            or sorted(set(lengths)) != lengths or any(b - a <= 2 * tolerance for a, b in zip(lengths, lengths[1:]))
            or model.get('mode') != 'ab' or model['repeats'] != 2
            or any(model.get('concurrency_repeats', {}).get(str(c), 2) != 2 for c in model['concurrencies'])):
        raise ValueError('matched cold needs disjoint positive length ranges and four visits per cell')
    return spec


def seed_set(config, repeat, count):
    base = config['sampling']['seed']
    return [base + repeat * 1_000_000 + slot * 10_000 for slot in range(count)]


def require_keeper(model, keep):
    if (model.get('require_keeper') or (model.get('mode') == 'ab'
            and model.get('speculation_mode') != 'off')) and not keep:
        raise RuntimeError('A/B speculative visits require weight keeper borrowing')


def closed_loop(server, name, inputs, tokens, sampling, seeds, power, seconds=15, before_measure=None, one_cohort=False):
    """Prime every slot before opening a fixed-duration replacement window.

    Cold measurements pause after priming so the control can run while idle.
    Standard measurements keep replacing requests throughout priming.
    """
    count = len(seeds)
    condition = threading.Condition()
    stop, resume = threading.Event(), threading.Event()
    gate = threading.Barrier(count + 1)
    requests, events, primed, errors = [], [], {}, []
    timing = {'deadline': None}
    offset = (server.directory / 'server.log').stat().st_size
    duration = .15 if server.dry else seconds

    def slot(number):
        sequence = 0
        try:
            gate.wait(timeout=10)
            while not stop.is_set():
                if timing['deadline'] is not None and time.monotonic() >= timing['deadline']:
                    break
                index = sequence * count + number
                prompt = inputs(index)
                seed = seeds[number] + sequence
                if sequence >= 10_000:
                    raise RuntimeError('request seed sequence exhausted')
                body = {'model': r.MODEL, 'prompt': [prompt], 'n': 1, 'cache_salt': f'{name}-{index}',
                        'max_tokens': tokens, 'min_tokens': tokens, 'ignore_eos': True,
                        'logprobs': 1, 'return_token_ids': True, 'temperature': sampling['temperature'],
                        'top_p': sampling['top_p'], 'seed': seed, 'stream': True,
                        'stream_options': {'include_usage': True}}
                row = {'index': index, 'slot': number, 'sequence': sequence, 'seed': seed,
                       'prompt': prompt, 'cache_salt': body['cache_salt'],
                       'submitted': time.monotonic(), 'token_ids': [], 'selected_bits': []}
                usage, finish = None, None
                for event in r.stream(server.url, '/v1/completions', body, timeout=r.remaining(120)):
                    now = time.monotonic()
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        ids = choice.get('token_ids') or []
                        if ids:
                            row.setdefault('first_token', now)
                        row['token_ids'].extend(ids)
                        row['selected_bits'].extend(bits(v) for v in
                                                   (choice.get('logprobs') or {}).get('token_logprobs', []))
                        finish = choice.get('finish_reason') or finish
                        with condition:
                            events.append((now, len(ids)))
                row['completed'] = time.monotonic()
                if (not usage or usage['completion_tokens'] != tokens or usage['prompt_tokens'] != len(prompt)
                        or usage['prompt_tokens_details']['cached_tokens'] != 0):
                    raise RuntimeError('incomplete or cached closed-loop workload')
                if finish != 'length' or len(row['token_ids']) != tokens or len(row['selected_bits']) != tokens:
                    raise RuntimeError('incomplete closed-loop output')
                row['cached_prompt_tokens'] = usage['prompt_tokens_details']['cached_tokens']
                with condition:
                    requests.append(row)
                    if sequence == 0:
                        primed[number] = row['completed']
                    condition.notify_all()
                if sequence == 0 and before_measure is not None:
                    if not resume.wait(timeout=r.remaining(120)):
                        raise RuntimeError('cold priming pause timed out before measurement')
                if one_cohort and sequence == 1:
                    break
                sequence += 1
        except BaseException as error:
            with condition:
                errors.append(error)
                condition.notify_all()
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(slot, i) for i in range(count)]
        began = time.monotonic()
        gate.wait(timeout=10)
        try:
            prime_deadline = began + r.remaining(120)
            with condition:
                while len(primed) != count or time.monotonic() - began < (.02 if server.dry else 4):
                    if errors:
                        raise errors[0]
                    if time.monotonic() >= prime_deadline:
                        raise RuntimeError('priming wave exceeded its deadline')
                    condition.wait(.01)
            cold_state = before_measure() if before_measure else None
            if errors:
                raise errors[0]
            before = scrape(server)
            initial = faults.snapshot(server)
            started, start_ns = time.monotonic(), time.time_ns()
            deadline = timing['deadline'] = started + duration
            resume.set()
            while time.monotonic() < deadline:
                if errors:
                    raise errors[0]
                stop.wait(min(.01, max(0, deadline - time.monotonic())))
            final = faults.snapshot(server)
            ended, end_ns = final['monotonic_seconds'], time.time_ns()
            after = scrape(server)
        finally:
            stop.set()
            resume.set()
        for future in futures:
            future.result()
    steps = settle(server, offset)
    duration = ended - started
    measured = [s for s in steps if start_ns <= s['completed_unix_ns'] < end_ns]
    emitted = sum(n for at, n in events if started <= at < ended)
    if (not emitted and before_measure is None) or not requests:
        raise RuntimeError('empty closed-loop window')
    requests.sort(key=lambda x: x['index'])
    in_flight = sum(max(0, min(row['completed'], ended) - max(row['submitted'], started))
                    for row in requests) / duration
    if not server.dry and not one_cohort and in_flight < .95 * count:
        raise RuntimeError('underfilled closed-loop window')
    energy = power.window(started, ended, max(emitted, 1))
    if not emitted:
        energy['joules_per_token'] = None
    return {'started': started, 'ended': ended, 'steps': measured, 'requests': requests,
            'cold_steps': [s for s in steps if s['completed_unix_ns'] >= start_ns],
            'faults': faults.difference(initial, final), 'tokens': emitted,
            'energy': energy, 'window_seconds': duration,
            'cached_prompt_tokens': sum(row['cached_prompt_tokens'] for row in requests),
            'priming': {'completed_requests': len(primed), 'per_slot_completed': primed,
                        'last_completed': max(primed.values()), 'window_started': started},
            'replacement_submissions': sum(row['sequence'] > 0 and started <= row['submitted'] < ended for row in requests),
            'prefill_share': sum(s['execution_us'] for s in measured if s['prefill_items']) / (duration * 1e6),
            'cold_preparation': cold_state, 'synthetic': server.dry, 'acceptance': acceptance(before, after)}


def warmup_windows(config):
    model = config['model']
    return [(c, repeat) for repeat in (0, 1) for c in model['concurrencies']
            if repeat < model.get('concurrency_repeats', {}).get(str(c), model['repeats'])]


def page_cache_warmup(server, arm, state, power):
    config = SEAL['config']
    rows = []
    for c, repeat in warmup_windows(config):
        seeds = seed_set(config, repeat, c)
        tokens = config['tokens'].get(str(c), config['tokens']['default'])
        w = closed_loop(server, f'w0-c{c}-r{repeat}', lambda i: state['tokens'][i % len(state['tokens'])],
                        tokens, config['sampling'], seeds, power)
        rows.append({'concurrency': c, 'repeat': repeat, 'seed_set': seeds, **arm_identity(SEAL, arm),
                     'started': w['started'], 'ended': w['ended'], 'requests': len(w['requests']),
                     'cached_prompt_tokens': w['cached_prompt_tokens'], 'priming': w['priming']})
    state['page_cache_warmup'] = {'completed': True, 'windows': rows, 'process': arm['label'],
                                **arm_identity(SEAL, arm)}


def run_ab_windows(server, arm, state, power, root, cold=None):
    config, model = SEAL['config'], SEAL['config']['model']
    record = state['processes'][arm['label']]
    matched = matched_cold(config)
    for repeat, windows in arm['passes']:
        for kind, c, length in ((kind, c, length) for kind, c in windows
                                for length in (matched['lengths'] if matched else [None])):
            name = f'{kind}-c{c}-r{repeat}'
            if length is not None:
                name += f'-l{length}'
            seeds = seed_set(config, repeat, c)
            tokens = config['tokens'].get(str(c), config['tokens']['default'])
            if cold:
                pool = state['cold_pools'][f"{arm['label']}/l{length}" if matched else arm['label']]
                remaining_windows = sum(len(part) for _, part in arm['passes'])
                budget = 2 * c if matched else config['cold_engram'].get('request_budgets', {}).get(
                    str(c), len(pool.remaining) // remaining_windows)
                allocated = pool.reserve(budget)
                inputs = lambda i: allocated[i] if i < len(allocated) else prompt_builder.exhausted()
            else:
                inputs = lambda i: state['tokens'][i % len(state['tokens'])]
            w = closed_loop(server, f"{arm['label']}-{name}",
                            inputs, tokens, config['sampling'], seeds, power,
                            before_measure=cold.prepare if cold else None, one_cohort=bool(matched))
            check_depth(name, w['steps'], arm.get('depths', arm['depth']))
            route = check_fallbacks(name, w['cold_steps'] if cold else w['steps'], arm_fallbacks(config, arm))
            full = full_concurrency(w['steps'], c)
            if not full['steps'] and not cold:
                raise RuntimeError(f'{name}: no full-concurrency steps')
            step_cells = cells(w['steps'])
            if server.dry:
                # The stub uses wall time to place steps in the window. Its
                # fixed execution times keep control cycles independent of CPU load.
                for cell in step_cells.values():
                    cell['cycle_ms'] = list(cell['execution_ms'])
            value = {'name': name, 'kind': kind, 'concurrency': c, 'depth': arm['depth'], **arm_identity(SEAL, arm),
                     'arm': arm['arm'], 'visit': arm['visit'], 'repeat': repeat, 'seed_set': seeds,
                     'temperature': config['sampling']['temperature'], 'window_seconds': w['window_seconds'],
                     'window_ended': w['ended'],
                     'cached_prompt_tokens': w['cached_prompt_tokens'],
                     'priming': w['priming'], 'replacement_submissions': w['replacement_submissions'],
                     'prefill_share': w['prefill_share'], 'request_tokens_per_second': w['tokens'] / (w['ended'] - w['started']),
                     'request_seconds': w['ended'] - w['started'], 'synthetic': server.dry,
                     'full': full, 'cells': step_cells, 'faults': w['faults'],
                     'fallback_launches': route['fallback_launches'], 'acceptance': w['acceptance'],
                     'allocated_steps': check_allocations(name, w['cold_steps'] if cold else w['steps'], record['confidence']['enabled'],
                                                         allocation_possible(config, arm, c)),
                     'energy': w['energy'], 'requests': w['requests'], 'steps': w['steps']}
            if cold:
                cold_state = w['cold_preparation']
                cold_state['after_window'] = cold.sample()
                value['cold_engram'] = cold_state
                value['mapped_gather_us'] = [s['host']['mapped_gather_us'] for s in w['cold_steps']
                                            if 'mapped_gather_us' in s.get('host', {})]
                value['prefill_execution_us'] = [s['execution_us'] for s in w['cold_steps'] if s['prefill_items']]
                value['ttft_seconds'] = [row['first_token'] - row['submitted'] for row in w['requests']
                                         if row['submitted'] >= w['started']]
                value['prompt_lengths'] = [len(row['prompt']) for row in w['requests'] if row['submitted'] >= w['started']]
                if matched:
                    timed = [row for row in w['requests'] if row['sequence'] == 1]
                    if len(timed) != c or any(abs(len(row['prompt']) - length) > matched['tolerance'] for row in timed):
                        raise RuntimeError('matched cold cohort is incomplete or outside its length tolerance')
                    value.update(target_length=length, length_tolerance=matched['tolerance'], matched_requests=timed)
            r.write(root / 'windows' / f'{name}.json', value)
            state.setdefault('deferred', []).append({'arm': arm['arm'], 'concurrency': c, 'repeat': repeat, 'requests': w['requests']})
            record['windows'][name] = value
            state['exact'][name] = state['exact'].get(name, 0) + 1
    record['faults'] = faults.combine([v['faults'] for v in record['windows'].values()])
    faults.validate(record['faults'])


def exactness_plan(config, deferred):
    """Compare paired keys within C; replay only keys without a counterpart."""
    pending, matched = {}, 0
    for baseline, treatment in config['model']['pairs']:
        for c in config['model']['concurrencies']:
            for repeat in range(config['model'].get('concurrency_repeats', {}).get(str(c), 2)):
                arms_by_key = {}
                for arm in (baseline, treatment):
                    values = {}
                    for visit in deferred:
                        if (visit['arm'], visit['concurrency'], visit['repeat']) != (arm, c, repeat):
                            continue
                        for row in visit['requests']:
                            key = (tuple(row['prompt']), row['seed'], len(row['token_ids']))
                            exact = {k: row[k] for k in ('token_ids', 'selected_bits')}
                            if key in values and values[key] != exact:
                                raise RuntimeError('duplicate exactness key differs within one concurrency')
                            values[key] = exact
                    arms_by_key[arm] = values
                base, head = arms_by_key[baseline], arms_by_key[treatment]
                common = base.keys() & head.keys()
                if not config.get('cold_engram') and len(common) < c:
                    raise RuntimeError('missing matched exactness keys within concurrency')
                for key in common:
                    if base[key] != head[key]:
                        raise RuntimeError(f'C{c}: matched sampled outputs differ')
                matched += len(common)
                for key in base.keys() ^ head.keys():
                    pending.setdefault(baseline, {}).setdefault(c, {})[key] = base.get(key, head.get(key))
    return pending, matched


def replay_cohort(expected, count, donors, replay):
    """Keep C slots occupied until all unmatched targets finish."""
    if not expected or not donors:
        raise ValueError('replay requires targets and same-concurrency baseline inputs')
    targets = list(expected.items())
    lock, stop = threading.Lock(), threading.Event()
    gate = threading.Barrier(count)
    state = {'next': 0, 'remaining': len(targets), 'padding_requests': 0}

    def slot(number):
        sequence = 0
        try:
            while not stop.is_set():
                with lock:
                    if state['remaining'] == 0:
                        return
                    target = state['next'] < len(targets)
                    if target:
                        item = targets[state['next']]
                        state['next'] += 1
                    else:
                        item = (donors[(number + sequence) % len(donors)], None)
                if sequence == 0:
                    gate.wait(timeout=10)
                replay((f'{number}-{sequence}', item))
                with lock:
                    if target:
                        state['remaining'] -= 1
                    else:
                        state['padding_requests'] += 1
                sequence += 1
        except BaseException:
            stop.set()
            raise
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(slot, i) for i in range(count)]
        for future in futures:
            future.result()
    return state['padding_requests']


def reference_ab(run, state, dry, keep):
    """Replay unmatched requests at their original concurrency after timing."""
    config = SEAL['config']
    model = config['model']
    declared = {a['label']: a for a in arms(config)}
    pending, matched = exactness_plan(config, state['deferred'])
    replayed, padding = 0, 0
    for baseline, by_concurrency in pending.items():
        arm = declared[baseline]
        _, install = arm_install(SEAL, arm)
        identity = arm_identity(SEAL, arm)
        arguments = arm_arguments(config, arm, install)
        if keep:
            arguments += ['--weight-keeper', keep['directory'], '--weight-keeper-policy', 'require']
        idle(run, f'reference-{baseline}', dry, (keep or {}).get('lent_mib'))
        server = r.Server(run / 'references' / baseline, install['install'], install['cache'],
                          arguments, dry, S / 'stub.py', identity=identity,
                          environment=arm_environment(config, arm), ar=model['speculation_mode'] == 'off',
                          bundle=arm_bundle(config, install))
        state.setdefault('reference_processes', {})[baseline] = identity
        if dry:
            server.env['TUNE_STUB_CLOSED_LOOP'] = '1'
        power = Power(server.directory, dry)
        power.reference = state['power']['before']['enforced_power_limits_w']
        def replay(item):
            index, (key, wanted) = item
            prompt, seed, tokens = key
            body = {'model': r.MODEL, 'prompt': [list(prompt)], 'n': 1,
                    'cache_salt': f'reference-{baseline}-c{c}-{index}', 'max_tokens': tokens, 'min_tokens': tokens,
                    'ignore_eos': True, 'logprobs': 1, 'return_token_ids': True, 'stream': True,
                    'stream_options': {'include_usage': True}, **config['sampling'], 'seed': seed}
            exact, usage, finish = {'token_ids': [], 'selected_bits': []}, None, None
            for event in r.stream(server.url, '/v1/completions', body, timeout=r.remaining(120)):
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    exact['token_ids'].extend(choice.get('token_ids') or [])
                    exact['selected_bits'].extend(bits(v) for v in (choice.get('logprobs') or {}).get('token_logprobs', []))
                    finish = choice.get('finish_reason') or finish
            if (not usage or usage['completion_tokens'] != tokens or finish != 'length'
                    or (wanted is not None and exact != wanted)):
                raise RuntimeError(f'{baseline}: sampled reference differs for seed {seed}')


        try:
            reference_power = r.power_check(server.directory, 'before', dry,
                                            state['power']['before']['enforced_power_limits_w'])
            state['power'][f'reference-{baseline}-before'] = reference_power
            power.start()
            server.start()
            log = (server.directory / 'server.log').read_text()
            reference_pages = kv_pages(log, config, state, baseline)
            if model['kv_pages'] == 'default':
                state['reference_processes'][baseline]['kv_pages_per_rank'] = reference_pages
            if keep and weight_source(log)['source'] != 'borrowed':
                raise RuntimeError('reference process did not borrow keeper weights')
            for c, expected in sorted(by_concurrency.items()):
                donors = [(tuple(row['prompt']), row['seed'], len(row['token_ids']))
                          for visit in state['deferred'] if visit['arm'] == baseline and visit['concurrency'] == c
                          for row in visit['requests']]
                padding += replay_cohort(expected, c, donors, replay)
                replayed += len(expected)
            end_power = r.power_check(server.directory, 'after', dry,
                                      state['power']['before']['enforced_power_limits_w'])
            state['power'][f'reference-{baseline}-after'] = end_power
            r.write(server.directory / 'exactness.json', {'passed': True, **identity,
                    'kv_pages_per_rank': reference_pages,
                    'requests_per_concurrency': {str(c): len(v) for c, v in by_concurrency.items()},
                    'power_before': reference_power, 'power_after': end_power})
        finally:
            try:
                server.stop()
            finally:
                power.stop()
    state['exactness'] = {'passed': True, 'matched_keys': matched, 'replayed_keys': replayed,
                          'arm_installs': {arm['label']: arm_identity(SEAL, arm) for arm in arms(config)},
                          'padding_requests': padding,
                          'scope': 'Keys compared only within their original concurrency.'}


def visits(config):
    """For A/B, one mirrored block gives every declared baseline/treatment
    pair chronological ABBA visits, sharing visits across overlapping pairs."""
    model = config['model']
    declared = arms(config)
    if model.get('mode') != 'ab':
        return [{**arm, 'arm': arm['label'], 'visit': i,
                 'passes': [(p, windows_for(config, arm['depth'])[::1 if p % 2 == 0 else -1])
                            for p in range(model['repeats'])]} for i, arm in enumerate(declared)]
    if model['repeats'] != 2 or set(config['kinds']) != {'sampled'}:
        raise ValueError('ab needs one ABBA block (two repeats per arm), sampled only')
    if config.get('sampling', {}).get('temperature', 1.0) != 1.0:
        raise ValueError('ab needs temperature 1.0')
    labels = [arm['label'] for arm in declared]
    if len(labels) != len(set(labels)) or not model.get('pairs'):
        raise ValueError('ab needs unique arms and declared pairs')
    for baseline, treatment in model['pairs']:
        if baseline not in labels or treatment not in labels or labels.index(baseline) >= labels.index(treatment):
            raise ValueError('declare every baseline before its treatments for ABBA')
    lead = model.get('lead_visit', False)
    if type(lead) is not bool or lead and config.get('cold_engram'):
        raise ValueError('lead_visit is a boolean of standard A/B jobs')
    out = []
    if lead:
        # A leading visit of the first declared arm on repeat 0's windows and
        # seeds, measured as A1 is and held to every runtime check, that no
        # pair, control check or verdict reads (report.ab drops its rows after
        # validation): it takes a first-visit offset off A1.
        arm = declared[0]
        out.append({**arm, 'arm': arm['label'], 'label': f"{arm['label']}-lead", 'visit': 0, 'lead': True,
                    'passes': [(0, list(windows_for(config, arm['depth'])))]})
    for repeat, group in enumerate((declared, declared[::-1])):
        for arm in group:
            windows = [(kind, c) for kind, c in windows_for(config, arm['depth'])
                       if repeat < model.get('concurrency_repeats', {}).get(str(c), 2)]
            out.append({**arm, 'arm': arm['label'], 'label': f"{arm['label']}-r{repeat}",
                        'visit': len(out), 'passes': [(repeat, windows[::1 if repeat == 0 else -1])]})
    return out


def order(config):
    """The arms' labels in run order."""
    return [visit['label'] for visit in visits(config)]


def run_windows(server, arm, state, power, root, passes, reference):
    """Every window of `passes` ((repeat, [(kind, C), ...]) each), compared
    with the reference; with `reference` set the first output of each window
    becomes it and nothing is recorded as timed."""
    config = SEAL['config']
    model = config['model']
    label, depth = arm['label'], arm['depth']
    inputs = state['tokens']
    record = state['processes'][label]
    for repeat, windows in passes:
        for kind, c in windows:
            name = f'{kind}-c{c}'
            tokens = config['tokens'].get(str(c), config['tokens']['default'])
            # Low concurrencies serve several prompt groups in turn, so a
            # window covers more than its first C prompts.
            groups = model.get('groups', {}).get(str(c), 1)
            tag = 'reference' if reference else f'r{repeat}'
            parts = []
            for g in range(groups):
                w = window(server, f'{name}-{tag}-g{g}', kind, [list(x) for x in inputs[g * c:(g + 1) * c]], tokens,
                           f'tune-{label}-{name}-{tag}-g{g}', config['sampling'])
                check_depth(f'{label} {name} {tag} group {g}', w['steps'], arm.get('depths', depth))
                w['route'] = check_fallbacks(name, w['steps'], arm_fallbacks(config, arm))
                w['allocated_steps'] = check_allocations(
                    f'{label} {name} {tag} group {g}', w['steps'], record['confidence']['enabled'],
                    allocation_possible(config, arm, c))
                w['energy'] = power.window(w['started'], w['ended'], c * tokens)
                parts.append(w)
            exact = [stream for w in parts for stream in w['exact']]
            if reference:
                state['reference'][name] = exact
                r.write(root / 'reference' / f'{name}.json', {'name': name, 'exact': exact, **arm_identity(SEAL, arm)})
                continue
            seconds = sum(w['ended'] - w['started'] for w in parts)
            finishes = [f for w in parts for f in w['finishes']] if all(w['finishes'] for w in parts) else None
            value = {'name': name, 'kind': kind, 'concurrency': c, 'depth': depth, 'repeat': repeat, 'tokens': tokens,
                     **arm_identity(SEAL, arm),
                     'arm': arm['arm'], 'visit': arm['visit'],
                     'allocated_steps': sum(w['allocated_steps'] for w in parts),
                     'fallback_launches': sum(w['route']['fallback_launches'] for w in parts),
                     'groups': groups, 'request_seconds': seconds,
                     'request_tokens_per_second': groups * c * tokens / seconds,
                     'full': merge_full([full_concurrency(w['steps'], c) for w in parts]),
                     'cells': merge_cells([cells(w['steps']) for w in parts]),
                     'streams': streams(finishes, tokens), 'acceptance': merge_acceptance([w['acceptance'] for w in parts]),
                     'energy': merge_energy([w['energy'] for w in parts], groups * c * tokens),
                     'faults': [w['faults'] for w in parts],
                     'prompts': [p['sha256'] for p in SEAL['prompts'][:groups * c]]}
            if not value['full']['steps']:
                raise RuntimeError(f'{label} {name} {tag}: no full-concurrency steps')
            r.write(root / 'windows' / f'{name}-r{repeat}.json',
                    {**value, 'steps': [w['steps'] for w in parts], 'exact': exact})
            record['windows'][f'{name}-r{repeat}'] = {k: value[k] for k in
                                                    ('request_tokens_per_second', 'full', 'install', 'revision')}
            wanted = state['reference'].get(name)
            if wanted is None:
                raise RuntimeError(f'{label} {name}: no reference output')
            if exact != wanted:
                where = first_difference(wanted, exact)
                if kind == 'greedy':
                    raise RuntimeError(f'{label} {name} r{repeat}: greedy output differs from W0 at {where}')
                state['sampled_differences'].append(f'{label} {name} r{repeat}: differs from W0 at {where}')
            state['exact'][name] = state['exact'].get(name, 0) + 1
            save(SEAL_RUN, state)


def run_arm(run, arm, state, dry, keep, reference_pass, warmup_only=False):
    config = SEAL['config']
    model = config['model']
    _, install = arm_install(SEAL, arm)
    identity = arm_identity(SEAL, arm)
    label, depth = arm['label'], arm['depth']
    root = run / label
    require_keeper(model, keep)
    arguments = arm_arguments(config, arm, install)
    if keep:
        arguments += ['--weight-keeper', keep['directory'], '--weight-keeper-policy', 'require']
    server = r.Server(root / 'server', install['install'], install['cache'], arguments, dry, S / 'stub.py',
                      identity=identity, environment=arm_environment(config, arm), ar=model['speculation_mode'] == 'off',
                      bundle=arm_bundle(config, install))
    cold = faults.ColdEngram(server, config['cold_engram']) if config.get('cold_engram') else None
    if model.get('mode') == 'ab' and dry:
        server.env['TUNE_STUB_CLOSED_LOOP'] = '1'
    records = state.setdefault('warmup_processes', {}) if warmup_only else state['processes']
    record = records[label] = {'depth': depth, 'args': arguments, 'windows': {}, 'gpu_memory_mib': {}, **identity,
                                          'reference_pass': reference_pass}
    if arm.get('env'):
        record['env'] = arm['env']
    power = Power(root, dry)

    def memory(when):
        if not dry:
            record['gpu_memory_mib'][when] = [used for _, _, used in gpu_memory(root, when)]
    try:
        record['power_before'] = r.power_check(root, 'before', dry, state['power']['before']['enforced_power_limits_w'])
        power.reference = record['power_before']['enforced_power_limits_w']
        power.start()
        server.start()
        record['process_group'] = json.loads(server.pidfile.read_text())['pgid']
        record['readiness_seconds'] = server.readiness
        memory('ready')
        mode = ((server.health.get('serving') or {}).get('profile') or {}).get('speculation', {}).get('mode')
        record['speculation_mode'] = mode
        if mode != model['speculation_mode']:
            raise RuntimeError(f"{label}: serves speculation {mode}, expected {model['speculation_mode']}")
        if mode == 'off' and server.health['serving']['profile']['speculation'].get('tokens') != 0:
            raise RuntimeError(f'{label}: AR profile reports draft tokens')
        log = (server.directory / 'server.log').read_text(errors='replace')
        record['confidence'] = confidence_startup(log, [*model['serve_arguments'], *arguments])
        record['weights'] = weight_source(log)
        wanted = 'borrowed' if keep else 'loaded'
        if (not dry or model.get('mode') == 'ab') and record['weights']['source'] != wanted:
            raise RuntimeError(f"{label}: weights {record['weights']}, expected {wanted}")
        record['kv_pages_per_rank'] = kv_pages(log, config, state, label)
        if not state['tokens']:
            state['tokens'] = tokenize(server, SEAL['prompts'])
            if 'prompt_file' in SEAL:
                prompt_builder.validate_external_lengths(SEAL['prompts'], state['tokens'], matched_cold(config))
            if cold:
                matched = matched_cold(config)
                state['cold_pools'] = (prompt_builder.matched_visit_pools(state['tokens'], visits(config),
                    matched['lengths'], matched['tolerance'], 2 * sum(model['concurrencies'])) if matched else
                    prompt_builder.visit_pools(state['tokens'], visits(config)))
                state['cold_prompt_assignment'] = {label: pool.assignment for label, pool in state['cold_pools'].items()}
                budgets = config['cold_engram'].get('request_budgets', {})
                for visit in ([] if matched else visits(config)):
                    needed = sum(config['warm']) + sum(budgets.get(str(c), 2 * c)
                             for _, part in visit['passes'] for _, c in part)
                    if len(state['cold_pools'][visit['label']].remaining) < needed:
                        raise RuntimeError('cold prompt pool is too small after balancing token lengths')
            r.write(run / 'prompt-tokens.json', state['tokens'])
        pool = state['cold_pools'][label] if cold and not matched_cold(config) else None
        # Warm windows run the measured kind (sampled when the job measures
        # it) on altered prompts, so they share no cached prefix.
        warm_kind = windows_for(config, depth, True)[0][0]
        for c in config['warm']:
            warm = [list(p) for p in pool.reserve(c)] if pool else [list(x) for x in state['tokens'][:c]]
            if not pool:
                for x in warm:
                    x[1] = x[0]
            window(server, f'warm-c{c}', warm_kind, warm, config['warm_tokens'], f'tune-{label}-warm-c{c}',
                   config['sampling'])
            memory(f'warm-c{c}')
        record['memory_after_warmup'] = memory_lines((server.directory / 'server.log').read_text(errors='replace'))
        record['metrics_after_warmup'] = counters(server, 'after-warmup')
        if warmup_only:
            page_cache_warmup(server, arm, state, power)
        elif model.get('mode') == 'ab':
            run_ab_windows(server, arm, state, power, root, cold)
        elif reference_pass:
            run_windows(server, arm, state, power, root, [(0, windows_for(config, depth, True))], True)
        if model.get('mode') != 'ab':
            run_windows(server, arm, state, power, root, arm['passes'], False)
        memory('end')
        record['metrics'] = counters(server, 'final')
        if record['metrics']['preemptions'] or record['metrics']['prefix_cache_evictions']:
            raise RuntimeError(f"{label}: {record['metrics']} (the test KV mode requires zero)")
    finally:
        try:
            record['power_after'] = r.power_check(root, 'after', dry,
                                                  state['power']['before']['enforced_power_limits_w'])
        finally:
            try:
                server.stop()
            finally:
                if cold:
                    cold.close()
                power.stop()
                save(run, state)


def save(run, state):
    out = {k: v for k, v in state.items() if k not in ('reference', 'tokens', 'deferred', 'unseen', 'cold_pools')}
    r.write(run / 'summary.json', out)
    lines = [f"{SEAL['config']['name']}: {state['status']}", '']
    for label, row in state['processes'].items():
        memory = row.get('memory_after_warmup') or {}
        lines.append(f"{label}: install {row['install']}, revision {row['revision']}, "
                     f"depth {row['depth']}, ready {row.get('readiness_seconds', 0):.1f} s, weights "
                     f"{row.get('weights', {}).get('source')}, KV pages {row.get('kv_pages_per_rank')}, metrics "
                     f"{row.get('metrics')}; memory after warmup " + '; '.join(
                         f"rank {m['rank']}: {m['free_mib']} MiB free" for m in memory.get('ranks', []))
                     + (f"; WARNINGS {memory['warnings']}" if memory.get('warnings') else '')
                     + (f"; GPU MiB {row['gpu_memory_mib']}" if row.get('gpu_memory_mib') else ''))
        for name, value in row['windows'].items():
            full = value['full']
            lines.append(f"  {name}: {value['request_tokens_per_second']:.1f} tok/s, full {full['steps']} steps "
                         f"{full['tokens_per_second']:.1f} tok/s, {dict(Counter(full['histogram']).most_common(3))}")
    exact_label = 'Paired exactness' if state['mode'] == 'ab' else 'Exact against W0'
    lines += ['', *(f'{exact_label}: {k} in {n} timed windows' for k, n in sorted(state['exact'].items()))]
    if state.get('windows'):
        lines.append(f"Timed windows: {state['windows']['executed']} executed of {state['windows']['planned']} planned")
    lines += ['Sampled difference: ' + x for x in state['sampled_differences']]
    lines += [f"Power {k}: limits {v.get('power_limits_w')} W, SM MHz {v.get('sm_clock_mhz')}"
              for k, v in state['power'].items()]
    if state.get('keeper') is not None:
        lines.append(f"Weight keeper: {state['keeper']}")
    lines += ['Error: ' + x for x in state['errors']]
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(args):
    m = validate(args.payload_sha, args.dry_run)
    config = m['config']
    r.EXPECTED_POWER_W = args.expected_power_w
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    r.write(run / 'config.json', config)
    r.write(run / 'prompts.json', m['prompts'])
    global SEAL_RUN
    SEAL_RUN = run
    state = {'status': 'running', 'dry_run': args.dry_run, 'revision': m['install'].get('revision'),
             'arm_installs': {arm['label']: arm_identity(m, arm) for arm in arms(config)},
             'mode': config['model'].get('mode', 'sweep'), 'order': order(config), 'processes': {}, 'exact': {},
             'reference': {}, 'tokens': [], 'sampled_differences': [], 'power': {}, 'telemetry': {}, 'keeper': None,
             'errors': []}
    r.WORK_DEADLINE = time.monotonic() + config['work_seconds']
    since = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    began = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, r.interrupted)
    reference = None
    try:
        if shutil.disk_usage(run).free < DATA_RESERVE:
            raise RuntimeError('the output volume is under its free-space reserve')
        keep = keeper(run, args.dry_run)
        require_keeper(config['model'], keep)
        if len(sealed_installs(m)[0]) > 1 and not keep:
            raise RuntimeError('multi-install jobs require weight keeper borrowing')
        state['keeper'] = {'used': bool(keep), **(keep or {})}
        before = state['telemetry']['before'] = r.telemetry(run, 'before', args.dry_run, since)
        reference = before['power_limits_w']
        state['power']['before'] = before['power']
        reserve = config['model']['planning']['process_reserve_seconds']
        if state['mode'] == 'ab' and not config.get('cold_engram'):
            warmup = {**visits(config)[0], 'label': 'page-cache-warmup'}
            idle(run, 'before-page-cache-warmup', args.dry_run, (keep or {}).get('lent_mib'))
            run_arm(run, warmup, state, args.dry_run, keep, reference_pass=False, warmup_only=True)
        for index, arm in enumerate(visits(config)):
            left = r.WORK_DEADLINE - time.monotonic()
            if left < reserve:
                raise RuntimeError(f'work budget: {left:.0f} s left, under the {reserve} s an arm needs; '
                                   f"stopped before {arm['label']}")
            idle(run, f"before-{arm['label']}", args.dry_run, (keep or {}).get('lent_mib'))
            run_arm(run, arm, state, args.dry_run, keep, reference_pass=index == 0 and state['mode'] != 'ab')
            state['power'][arm['label']] = r.power_check(run, f"after-{arm['label']}", args.dry_run, reference)
            save(run, state)
        if state['mode'] == 'ab':
            reference_ab(run, state, args.dry_run, keep)
        if state['sampled_differences']:
            raise RuntimeError(f"fixed-seed sampled outputs differ from W0: {state['sampled_differences']}")
        planned = planned_windows(config)
        executed = sum(state['exact'].values())
        state['windows'] = {'planned': planned, 'executed': executed}
        if executed != planned:
            raise RuntimeError(f'{executed} timed windows executed, {planned} planned')
        state['status'] = 'passed'
    except BaseException as error:
        state['status'] = 'void' if isinstance(error, (r.PowerChanged, faults.PartlyWarm)) else 'failed'
        if state['status'] == 'void':
            state['void_reason'] = str(error)
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
            state['status'] = 'void' if isinstance(error, r.PowerChanged) or state['status'] == 'void' else 'failed'
            state['errors'].append('cleanup: ' + str(error))
        state['elapsed_seconds'] = time.monotonic() - began
        save(run, state)
        try:
            if state['mode'] == 'costs':
                state['costs'] = report.costs(run)['summary']
            elif state['mode'] == 'ab':
                result = report.ab(run)
                state['ab'] = result['pairs']
                if result['status'] != 'valid':
                    state['status'] = 'void' if result['status'] == 'void' else 'failed'
                    state['errors'].append('report: ' + result['reason'])
            elif state['mode'] == 'exact':
                state['costs'] = report.costs(run)['summary']
            else:
                state['depths'] = report.evaluate(run)['depths']
        except Exception as error:
            if state['status'] != 'void':
                state['status'] = 'failed'
            state['errors'].append(f'report: {type(error).__name__}: {error}')
        save(run, state)
    print(json.dumps({'output': str(run), 'status': state['status'], 'depths': state.get('depths'),
                      'costs': state.get('costs'),
                      'errors': state['errors'], 'elapsed_seconds': round(state['elapsed_seconds'], 1)}), flush=True)
    return int(state['status'] != 'passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--payload-sha', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--expected-power-w', type=float)
    raise SystemExit(main(ap.parse_args()))
