#!/usr/bin/env python3
"""Counterbalanced serving A/B of two installs of one model (see tools/ab/serving/README.md).

Usage: run.py RUN.json (--dry-run | --out NEW_DIR)
Exit 0: every configured evidence gate passed; 1: a gate or the run failed
(result.json says which, and whether ownership is retained); 2: refused
before any arm started."""
import argparse
import csv
import datetime
import io
import hashlib
import shutil
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

from common import Refusal, need, read, loads, write, digest, regular, finite, keys, subset
from evidence import (SMOKE, Comparator, phases_from, comparison_spec, acceptance, timing_summary, fixed_k_exposure,
                      SPECULATIVE_KINDS)
from model import MODELS, facts, serves, serving
from processes import Owned, Commands, TOKEN, capabilities

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent.parent / 'qual'
SCHEMA = 'chopinfer.ab-serving.v1'
VARIANTS = ('ar', 'spec')


def location(value, root):
    need(type(value) is str and value, 'path must be a nonempty string')
    text = os.path.expandvars(os.path.expanduser(value))
    need('$' not in text, f'unexpanded path variable: {value}')
    path = Path(text)
    return (path if path.is_absolute() else root / path).absolute()


def string_list(value, where):
    need(type(value) is list and all(type(x) is str and x and '\0' not in x for x in value),
         f'{where}: expected argv strings')
    return value


RESERVED = {'--model', '--model-cache-dir', '--listen', '--speculative-tokens', '--speculation',
            '--bundle', '--frontend', '--profile'}


def serve_args(value):
    value = string_list(value, 'serve_args')
    need(not any(x.split('=', 1)[0] in RESERVED or x == '--' for x in value),
         'serve_args cannot override model/cache/listen/speculation/profile/explicit bundle ownership')
    return value


def merge(a, b):
    """`b` over `a`, object by object."""
    result = dict(a)
    for key, value in b.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def verify_install(root, model):
    """The install's INSTALL.json receipt, checked against every bin/ and lib/
    file; `model` (model.facts) names the files it must hold."""
    root = Path(root)
    record = read(regular(root / 'INSTALL.json', True))
    need(type(record.get('commit')) is str and re.fullmatch('[0-9a-f]{40}', record['commit']),
         'INSTALL.json needs a full hexadecimal source commit')
    files = record.get('files')
    required = {model['server'], model['compiler'], model['bundle']}
    need(type(files) is dict and required <= files.keys(), 'incomplete INSTALL.json file inventory')
    expected = set()
    for name, sha in files.items():
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in ('bin', 'lib'),
             f'unsafe install path: {name}')
        need(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha), f'invalid digest: {name}')
        expected.add(name)
        expected.update(str(p) for p in relative.parents if str(p) != '.')
    actual = {str(p.relative_to(root)) for top in ('bin', 'lib') for p in [root / top, *(root / top).rglob('*')]}
    need(expected == actual, 'install bin/lib inventory differs from its receipt')
    for p in [root, *(root / name for name in actual)]:
        mode = p.lstat().st_mode
        need(not stat.S_ISLNK(mode) and not mode & 0o022 and (stat.S_ISREG(mode) or stat.S_ISDIR(mode)),
             f'install path is symlink, special, or group/other-writable: {p}')
    for name, sha in files.items():
        need(digest(regular(root / name)) == sha, f'install digest mismatch: {name}')
    need(all(os.access(root / name, os.X_OK) for name in (model['server'], model['compiler'])),
         'installed executables are not executable')
    return dict(path=str(root), receipt_sha256=digest(root / 'INSTALL.json'), **record)


def cache_record(path):
    record = read(regular(path))
    keys(record, {'version', 'checksum', 'value'}, set(), 'cache record')
    need(type(record['version']) is int and record['version'] == 1, 'unknown cache record version')
    value = record['value']
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    need(hashlib.sha256(encoded).hexdigest() == record['checksum'], 'cache record checksum mismatch')
    keys(value, {'kind', 'state', 'dependencies', 'files', 'last_selected', 'payload'}, set(), 'cache entry')
    need(type(value['last_selected']) is int and 0 <= value['last_selected'] <= 2**64-1, 'invalid cache selection time')
    return {k: v for k, v in value.items() if k != 'last_selected'}


def snapshot_install(source, destination, receipt, model):
    destination.mkdir(parents=True)
    for relative, sha in receipt['files'].items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(Path(source) / relative, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        need(digest(target) == sha, 'install changed while being snapshotted')
        target.chmod(0o555 if relative.startswith('bin/') else 0o444)
    write(destination / 'INSTALL.json', {k: v for k, v in receipt.items() if k not in ('path', 'receipt_sha256')}, 0o444)
    for directory in sorted((p for p in destination.rglob('*') if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    destination.chmod(0o555)
    return verify_install(destination, model)


def cache_packages(cache, checkpoint_identity, bundle_id, module_sha):
    need(cache.is_dir(), f'model cache missing: {cache}')
    packages = {}
    for p in (cache / 'records').glob('p-*.json'):
        entry = cache_record(p)
        if entry.get('state') != 'complete' or entry.get('payload', {}).get('checkpoint') != checkpoint_identity:
            continue
        manifest = cache / p.stem / 'manifest.json'
        body = read(regular(manifest))
        if body.get('id') != bundle_id:
            continue
        sha = digest(manifest)
        need(p.stem == 'p-' + sha, 'prepared package manifest digest mismatch')
        native = body.get('native_module', {})
        if native.get('sha256') != module_sha:
            continue
        rel = Path(native.get('path', ''))
        need(not rel.is_absolute() and '..' not in rel.parts and rel.parts, 'unsafe native module path')
        module_path = regular(manifest.parent / rel)
        need(digest(module_path) == module_sha and module_path.stat().st_size == native.get('byte_length'), 'cached native module differs from the installed candidate')
        packages[sha] = dict(manifest=str(manifest), record=str(p), record_sha256=digest(p), immutable_record=entry, native_sha256=module_sha)
    need(packages, f'no prepared package for the requested checkpoint in {cache}')
    return packages


def config(path):
    path = Path(path).absolute()
    c = read(regular(path))
    keys(c, {'schema', 'model', 'checkpoint', 'checkpoint_identity', 'installs', 'port', 'gpus'},
         {'models', 'serve_profile', 'serve_args', 'variants', 'phases', 'order', 'smoke', 'qual_tools',
          'client', 'client_assets',
          'env', 'timeouts', 'budget_seconds', 'acceptance', 'profile', 'query_command',
          'telemetry_interval', 'phase_timeout', 'expect', 'base_new', 'seed'}, 'run file')
    need(c['schema'] == SCHEMA, 'unknown run-file schema')
    root = path.parent
    c['models'] = str(location(c['models'], root)) if 'models' in c else str(MODELS)
    model = facts(c['model'], c['models'])
    c['checkpoint'] = str(location(c['checkpoint'], root))
    need(Path(c['checkpoint']).is_dir(), 'checkpoint directory missing')
    need(type(c['checkpoint_identity']) is str and c['checkpoint_identity'], 'checkpoint identity missing')
    need(type(c['port']) is int and 1024 <= c['port'] <= 65535, 'invalid unprivileged port')
    need(type(c['gpus']) is list and c['gpus'] and all(type(x) in (str, int) for x in c['gpus']), 'GPU IDs required')
    c['gpus'] = [str(x) for x in c['gpus']]
    need(len(c['gpus']) == len(set(c['gpus'])) and all(re.fullmatch(r'[0-9]+|GPU-[0-9a-fA-F-]+', x) for x in c['gpus']),
         'invalid or duplicate GPU IDs')
    keys(c['installs'], {'base', 'new'}, set(), 'installs')
    receipts, packages = {}, {}
    for arm, installation in c['installs'].items():
        keys(installation, {'path', 'cache'}, {'serve_args', 'env', 'expect', 'commit'}, f'install {arm}')
        installation['path'] = str(location(installation['path'], root))
        installation['cache'] = str(location(installation['cache'], root))
        installation['serve_args'] = serve_args(installation.get('serve_args', []))
        receipts[arm] = verify_install(installation['path'], model)
        # /health must name the receipt's commit; a requested commit binds the receipt.
        need(installation.get('commit', receipts[arm]['commit']) == receipts[arm]['commit'],
             f'install {arm} is built from {receipts[arm]["commit"]}, not the requested {installation.get("commit")}')
        packages[arm] = cache_packages(Path(installation['cache']), c['checkpoint_identity'], model['bundle_id'],
                                       receipts[arm]['files'][model['bundle']])
    if 'serve_profile' in c:
        # The profile's options, passed explicitly, and what /health must
        # report of them; only a prepared package that serves the profile
        # may be selected.
        name = c['serve_profile']
        need(name in model['profile_names'], f'{c["model"]} declares no serve profile {name!r}')
        profile = read(model['profiles'])['profiles'][name]
        need(profile['tensor_parallel'] == len(c['gpus']),
             f'profile {name} serves {profile["tensor_parallel"]} ranks; the run has {len(c["gpus"])} GPUs')
        options, expectation = serving(name, profile)
        extra = serve_args(c.get('serve_args', []))
        flags = {x for x in options if x.startswith('--') and x != '--model-option'}
        need(not flags & {x.split('=', 1)[0] for x in extra} and '--model-option' not in extra,
             f'serve_args repeat options of profile {name}')
        c['serve_args'] = options + extra
        c['expect'] = merge(expectation, c.get('expect', {}))
        for arm in packages:
            packages[arm] = {sha: p for sha, p in packages[arm].items() if serves(read(p['manifest']), profile)}
            need(packages[arm], f'no prepared package of install {arm} serves profile {name}')
    c.setdefault('expect', {'profile': {'tensor_parallel_size': len(c['gpus'])}})
    c['serve_args'] = serve_args(c.get('serve_args', ['--tensor-parallel-size', str(len(c['gpus']))]))
    c.setdefault('variants', {'ar': {'mode': 'off', 'tokens': 0}})
    need(type(c['variants']) is dict and c['variants'] and c['variants'].keys() <= set(VARIANTS), 'variants must be ar and/or spec')
    c.setdefault('smoke', False)
    need(type(c['smoke']) is bool, 'smoke must be true or false')
    if c['smoke']:
        # One AB pair of one variant: base against new, nothing more.
        need(len(c['variants']) == 1, 'a smoke run serves one variant')
    else:
        need('spec' not in c['variants'] or 'ar' in c['variants'], 'a speculative variant requires an AR control variant')
    for name, variant in c['variants'].items():
        keys(variant, {'mode', 'tokens'}, {'drafter', 'serve_args', 'env', 'expect'}, f'variant {name}')
        need(type(variant['tokens']) is int and 0 <= variant['tokens'] <= 16, 'invalid speculative depth')
        need((name == 'ar' and variant['mode'] == 'off' and variant['tokens'] == 0 and 'drafter' not in variant) or
             (name == 'spec' and variant['mode'] in ('fixed', 'adaptive') and variant['tokens'] > 0),
             'variant name/mode/token depth mismatch')
        if name == 'spec':
            if 'drafter' not in variant:
                need(len(model['drafters']) == 1, f'{c["model"]} declares drafters {model["drafters"]}; name one')
                variant['drafter'] = model['drafters'][0]
            need(variant['drafter'] in model['drafters'], f'{c["model"]} declares no drafter {variant["drafter"]!r}')
        variant['serve_args'] = serve_args(variant.get('serve_args', []))
    if c['smoke']:
        c.setdefault('order', ['base', 'new'])
        need(c['order'] in (['base', 'new'], ['new', 'base']), 'a smoke order is AB or BA')
    else:
        c.setdefault('order', ['base', 'new', 'new', 'base'])
        need(c['order'] in (['base', 'new', 'new', 'base'], ['new', 'base', 'base', 'new']),
             'order must be ABBA or BAAB, providing both orders and two repeats')
    phases = phases_from(c.get('phases', 'standard'))
    if 'spec' in c['variants']:
        need(any(p['kind'] in SPECULATIVE_KINDS for p in phases), 'speculation needs a phase with draft-token coverage')
    c['phases'] = [p['name'] for p in phases]
    if c['smoke']:
        need(set(SMOKE) <= set(c['phases']), f'a smoke run needs the phases {SMOKE}')
    c['qual_tools'] = str(location(c['qual_tools'], root)) if 'qual_tools' in c else str(QUAL)
    if any(p['kind'] == 'determinism' for p in phases):
        regular(Path(c['qual_tools']) / 'determinism.py')
    c['client'] = str(location(c.get('client', str(HERE / 'client.py')), root))
    regular(c['client'])
    assets = c.get('client_assets', {})
    need(type(assets) is dict, 'client_assets must be a name/path object')
    c['client_assets'] = {}
    for name, value in assets.items():
        need(name in ('gsm8k-examples.json', 'gsm8k-manifest.json'), 'unexpected client asset name')
        c['client_assets'][name] = str(regular(location(value, root)))
    if any(p['kind'] == 'gsm8k' for p in phases):
        need(set(c['client_assets']) == {'gsm8k-examples.json', 'gsm8k-manifest.json'}, 'GSM8K fixtures missing')
        manifest = read(c['client_assets']['gsm8k-manifest.json'])
        need(digest(c['client_assets']['gsm8k-examples.json']) == manifest['examples_sha256'], 'GSM8K fixture digest mismatch')
    for mapping in [c, *c['installs'].values(), *c['variants'].values()]:
        environment = mapping.setdefault('env', {})
        need(type(environment) is dict and all(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', k)
             and type(v) is str and '\0' not in v for k, v in environment.items()), 'invalid environment mapping')
        need(not environment.keys() & {TOKEN, 'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE'},
             'reserved harness environment override')
        need(type(mapping.setdefault('expect', {})) is dict, 'expect must be a health-serving subset')
    c.setdefault('budget_seconds', 1740)
    finite(c['budget_seconds'], 'budget_seconds', 1)
    need(c['budget_seconds'] <= 1740, 'budget exceeds queue allowance')
    defaults = dict(query=5, admission=120, startup=360, http=5, term=15, kill=5, release=30, cleanup_reserve=90)
    custom = c.get('timeouts', {})
    need(type(custom) is dict and custom.keys() <= defaults.keys(), 'unknown timeout')
    defaults.update(custom)
    for key, value in defaults.items():
        finite(value, f'timeouts.{key}', 0.01)
    need(defaults['query'] <= 30 and defaults['cleanup_reserve'] < c['budget_seconds'], 'invalid query/cleanup deadline')
    need(defaults['cleanup_reserve'] >= defaults['term'] + defaults['kill'] + defaults['release'] + 5,
         'cleanup reserve must cover TERM, KILL, release and utility cleanup')
    c['timeouts'] = defaults
    c.setdefault('telemetry_interval', 1)
    finite(c['telemetry_interval'], 'telemetry_interval', 0.05)
    c.setdefault('phase_timeout', 240)
    finite(c['phase_timeout'], 'phase_timeout', 0.05)
    c['query_command'] = string_list(c.get('query_command', ['nvidia-smi']), 'query_command')
    need(c['query_command'], 'query command missing')
    c.setdefault('base_new', 'bitwise')
    need(c['base_new'] in ('bitwise', 'report-only'), 'base_new must be bitwise or report-only')
    # A smoke's only gate is base against new: report-only would leave none.
    need(not c['smoke'] or c['base_new'] == 'bitwise', 'a smoke run gates base against new bitwise')
    c.setdefault('acceptance', {'max_rate_drop': 0.0})
    keys(c['acceptance'], {'max_rate_drop'}, {'reason'}, 'acceptance')
    finite(c['acceptance']['max_rate_drop'], 'max_rate_drop', 0)
    need(c['acceptance']['max_rate_drop'] <= 1, 'max_rate_drop exceeds 1')
    if c['acceptance']['max_rate_drop']:
        need(type(c['acceptance'].get('reason')) is str and c['acceptance']['reason'].strip(), 'nonzero acceptance allowance needs a recorded reason')
    c.setdefault('seed', 'ab-serving')
    need(type(c['seed']) is str and c['seed'], 'seed must be a nonempty string')
    if 'profile' in c:
        keys(c['profile'], {'command'}, {'args'}, 'profile')
        c['profile']['command'] = string_list(c['profile']['command'], 'profile command')
        need(c['profile']['command'], 'nsys command missing')
        args = string_list(c['profile'].get('args', ['--trace=cuda', '--cuda-graph-trace=node', '--sample=none']), 'profile args')
        need(not any(x.split('=', 1)[0] in ('--session-new', '--session', '--output', '-o', '--start-later', '--') for x in args),
             'profile args override harness ownership/output')
        c['profile']['args'] = args
    return c, phases, dict(run_file=str(path), run_file_sha256=digest(path), model=model, installs=receipts,
                           packages=packages)


def plan(c):
    arms = []
    for variant in VARIANTS:
        if variant not in c['variants']:
            continue
        count = {'base': 0, 'new': 0}
        for ordinal, install in enumerate(c['order'], 1):
            count[install] += 1
            arms.append(dict(name=f'{variant}-{ordinal}-{install}-{count[install]}', install=install,
                             variant=variant, repeat=count[install], ordinal=ordinal))
    return arms


def port_free(port):
    with socket.socket() as sock:
        # TIME_WAIT from the released arm is not an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


class GPU:
    def __init__(self, c, commands, env):
        self.c, self.commands, self.env = c, commands, env

    def query(self, fields, deadline, compute=False):
        remaining = deadline - time.monotonic()
        need(remaining > 0, 'GPU query deadline exhausted before spawn')
        args = self.c['query_command'] + ['--id=' + ','.join(self.c['gpus']),
            '--query-' + ('compute-apps=' if compute else 'gpu=') + fields, '--format=csv,noheader,nounits']
        result = self.commands.run(args, min(self.c['timeouts']['query'], remaining), env=self.env, deadline=deadline)
        need(time.monotonic() <= deadline, 'GPU query completed after its deadline')
        need(result.returncode == 0 and not result.stderr.strip(), 'GPU query failed or wrote diagnostics')
        try:
            return result.stdout.decode().strip()
        except UnicodeError as error:
            raise Refusal('GPU query returned invalid text') from error

    def pids(self, deadline):
        text = self.query('pid', deadline, True)
        lines = text.splitlines()
        need(all(re.fullmatch(r'[1-9][0-9]*', x.strip()) for x in lines), 'GPU process query is malformed')
        return set(map(int, lines))

    def idle(self, deadline, check=None):
        while True:
            if check:
                check()
            if not self.pids(deadline):
                return
            need(time.monotonic() < deadline, 'GPUs stayed busy through the release/admission deadline')
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def telemetry(self, deadline):
        rows = list(csv.reader(io.StringIO(self.query('index,clocks.sm,clocks.mem,power.draw', deadline))))
        need(len(rows) == len(self.c['gpus']), 'telemetry does not cover all selected GPUs')
        values = []
        for row in rows:
            need(len(row) == 4 and re.fullmatch(r'[0-9]+', row[0].strip()), 'malformed GPU telemetry')
            record = dict(index=int(row[0]))
            for key, value in zip(('sm_mhz', 'memory_mhz', 'power_w'), row[1:]):
                try:
                    record[key] = finite(float(value), f'GPU {key}', 0)
                except ValueError as error:
                    raise Refusal('unparsable GPU telemetry') from error
            values.append(record)
        need(len({v['index'] for v in values}) == len(values), 'duplicate GPU telemetry row')
        return values


class Runner:
    def __init__(self, c, phases, preflight, output):
        self.c, self.phases, self.preflight = c, phases, preflight
        self.output = Path(output).resolve()
        need(not any((p / '.git').exists() for p in (self.output, *self.output.parents)),
             'result directory must be outside a repository')
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.hard = self.started + c['budget_seconds']
        self.work = self.hard - c['timeouts']['cleanup_reserve']
        self.commands = Commands(records=self.output / 'utility-owners')
        self.cancelled = False
        self.owner = None
        self.owner_dir = None
        self.arms = []
        self.nonces = set()
        self.metadata_identity = None
        self.install_snapshots = {}
        for name, receipt in preflight['installs'].items():
            self.install_snapshots[name] = snapshot_install(receipt['path'], self.output / 'installs' / name, receipt,
                                                            preflight['model'])
            self.check()
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=','.join(c['gpus']), **c['env'])
        self.gpu = GPU(c, self.commands, self.env)
        self.tools = self.output / 'tools'
        self.tools.mkdir()
        # The helpers run from this hashed snapshot: the client, the
        # comparator with its IEEE-bit core, and the HTTP and target wrappers.
        files = {p.name: p for p in HERE.glob('*.py') if not p.name.startswith(('test_', 'stubs'))}
        files['client.py'] = Path(c['client'])
        files.update({name: Path(p) for name, p in c['client_assets'].items()})
        self.tool_hashes = {}
        for name, source in files.items():
            before = digest(regular(source))
            data = source.read_bytes()
            (self.tools / name).write_bytes(data)
            (self.tools / name).chmod(0o644)
            need(digest(source) == before == digest(self.tools / name), 'tool changed while being snapshotted')
            self.tool_hashes[name] = before
        # The determinism phase runs tools/qual in place; its files are pinned
        # here and checked like the snapshot.
        self.qual_hashes = {}
        if any(p['kind'] == 'determinism' for p in phases):
            self.qual_hashes = {str(p): digest(regular(p)) for p in sorted(Path(c['qual_tools']).glob('*.py'))}
        self.comparator = Comparator(self.tools / 'compare.py', c['qual_tools'])
        write(self.output / 'plan.json', dict(schema='chopinfer.ab-serving.plan.v1', config=c, preflight=preflight,
             arms=plan(c), tools_sha256=self.tool_hashes, qual_tools_sha256=self.qual_hashes,
             install_snapshots=self.install_snapshots))

    def check(self):
        need(not self.cancelled, 'run interrupted')
        need(time.monotonic() < self.work, 'run work budget exhausted')

    def remaining(self, limit=None):
        self.check()
        return min(limit if limit is not None else float('inf'), self.work - time.monotonic())

    def log(self, text):
        with (self.output / 'status.log').open('a') as f:
            f.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + text + '\n')
        print(text, flush=True)

    def tool_check(self):
        need(all(digest(self.tools / n) == sha for n, sha in self.tool_hashes.items()), 'snapshotted tool changed')
        need(all(digest(p) == sha for p, sha in self.qual_hashes.items()), 'a tools/qual file changed')

    def scorer(self, install):
        return Path(self.install_snapshots[install]['path']) / self.preflight['model']['server']

    def owned_gpu(self, deadline):
        live = {p.pid for p in self.owner.members() if p.live}
        need(self.gpu.pids(deadline) <= live, 'GPU process is not owned by this arm')

    def request(self, url, deadline, json_value=False):
        self.check()
        timeout = min(self.c['timeouts']['http'], deadline - time.monotonic())
        need(timeout > 0, 'HTTP deadline exhausted')
        result = self.commands.run([sys.executable, '-B', str(self.tools / 'http_request.py'), url, str(timeout)],
            timeout, env=self.env, check=self.check, deadline=self.hard)
        self.check()
        need(time.monotonic() <= deadline, 'HTTP completed after its deadline')
        if result.returncode == 3:
            raise urllib.error.URLError(result.stderr.decode(errors='replace'))
        need(result.returncode == 0, 'HTTP helper failed: ' + result.stderr.decode(errors='replace'))
        text = result.stdout.decode()
        return loads(text) if json_value else text

    def verify_server(self, arm, process):
        server = self.preflight['model']['server']
        path = Path(self.install_snapshots[arm['install']]['path']) / server
        wanted = self.preflight['installs'][arm['install']]['files'][server]
        with path.open('rb') as stream:
            elf = stream.read(4) == b'\x7fELF'
        if elf:
            with open(f'/proc/{process.pid}/exe', 'rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            need(actual == wanted, 'running executable differs from the pinned install')
        else:
            # CPU fixtures and explicitly pinned script installs expose their
            # script path in cmdline, since /proc/exe identifies the interpreter.
            argv = (Path('/proc') / str(process.pid) / 'cmdline').read_bytes().split(b'\0')
            need(str(path).encode() in argv and digest(path) == wanted, 'running script differs from the pinned install')
        current = self.owner.proc.get(process.pid)
        need(current is not None and current.identity == process.identity and current.live, 'server identity changed during executable verification')

    def serve_command(self, arm):
        installation = self.c['installs'][arm['install']]
        variant = self.c['variants'][arm['variant']]
        argv = [str(Path(self.install_snapshots[arm['install']]['path']) / self.preflight['model']['server']), 'serve',
            '--model', self.c['checkpoint'], '--model-cache-dir', installation['cache'], '--listen', f'127.0.0.1:{self.c["port"]}']
        argv += [*self.c['serve_args'], *installation['serve_args'], *variant['serve_args'],
            '--speculative-tokens', str(variant['tokens'])]
        if variant['mode'] != 'off':
            argv += ['--speculation', variant['mode']]
        return argv

    def health(self, arm, directory, deadline, startup=False):
        need(time.monotonic() < deadline, 'readiness deadline exhausted')
        value = self.request(f'http://127.0.0.1:{self.c["port"]}/health', deadline, True)
        need(value.get('status') == 'ready', 'server is not ready')
        live = value['serving']
        expected = dict(bundle_id=self.preflight['model']['bundle_id'], checkpoint_identity=self.c['checkpoint_identity'],
            source_commit=self.preflight['installs'][arm['install']]['commit'])
        subset(expected, live)
        need(live.get('package_identity') in self.preflight['packages'][arm['install']], 'server selected an unprepared package')
        package = self.preflight['packages'][arm['install']][live['package_identity']]
        need(digest(regular(package['manifest'])) == live['package_identity'] and cache_record(package['record']) == package['immutable_record'], 'selected prepared package changed')
        need(type(live.get('metadata_identity')) is str and re.fullmatch('[0-9a-f]{64}', live['metadata_identity']), 'metadata identity missing or malformed')
        if self.metadata_identity is None:
            self.metadata_identity = live['metadata_identity']
        need(live['metadata_identity'] == self.metadata_identity, 'checkpoint metadata identity changed between arms')
        manifest = read(package['manifest'])
        options = manifest.get('model_options', {}).get('options', [])
        layout = next((x for x in options if x.get('name') == 'checkpoint.quantization_layout'), None)
        # Only a package that declares a checkpoint layout is held to it.
        need(layout is None or live.get('model_options', {}).get('checkpoint.quantization_layout')
             in layout['definition']['allowed_values'], 'served checkpoint layout differs from its package')
        variant = self.c['variants'][arm['variant']]
        subset(dict(speculation=dict(tokens=variant['tokens'], mode=variant['mode'])), live['profile'])
        expected = {}
        for mapping in (self.c, self.c['installs'][arm['install']], variant):
            expected = merge(expected, mapping['expect'])
        subset(expected, live)
        need(live['profile']['speculation'].get('drafter') == variant.get('drafter'), 'wrong served drafter')
        # Bind /health to the identity emitted by this owned launch, rather
        # than accepting another server using the same install and port.
        identities = [loads(line.split(': ', 1)[1]) for line in (directory / 'server.log').read_text(errors='replace').splitlines()
                      if line.startswith('startup serving_identity: ')]
        logged_live = {k: v for k, v in live.items() if k != 'served_model_names'}
        need(len(identities) == 1 and identities[0] == logged_live, 'health is not this launch\'s logged startup identity')
        nonce = live.get('startup_nonce')
        need(type(nonce) is str and nonce, 'startup nonce missing')
        if startup:
            need(nonce not in self.nonces, 'server startup nonce was reused')
            self.nonces.add(nonce)
            write(directory / 'health.json', value)
        else:
            need(live == read(directory / 'health.json')['serving'], 'serving identity changed during capture')
        need(time.monotonic() <= deadline, 'readiness completed after its deadline')
        return value

    def handshake(self, directory, deadline):
        path = directory / 'target.json'
        while True:
            self.check()
            need(time.monotonic() < deadline, 'target handshake deadline expired')
            need(not self.owner.exited(), 'launcher exited before target identity handshake')
            if path.exists():
                target = read(path)
                need(target.get('token') == self.owner.token, 'target handshake has wrong owner token')
                process = self.owner.proc.get(target['pid'])
                need(process is not None and process.live and process.identity == (target['pid'], target['start'], target['uid'])
                     and process.uid == os.getuid(), 'target handshake identity is stale or invalid')
                argv = (Path('/proc') / str(process.pid) / 'cmdline').read_bytes().split(b'\0')
                need(str(self.tools / 'target.py').encode() in argv and self.owner.token.encode() in argv
                     and process.session == process.pid == process.group, 'target handshake is not this launched session anchor')
                self.owner.known[process.identity] = process
                self.owner.anchors[process.identity] = process
                self.owner.save()
                need(time.monotonic() < deadline, 'target handshake completed after its deadline')
                write(directory / 'target.go', dict(token=self.owner.token), 0o600)
                while time.monotonic() < deadline:
                    self.check()
                    server_path = directory / 'server.json'
                    if server_path.exists():
                        server = read(server_path)
                        child = self.owner.proc.get(server['pid'])
                        need(child is not None and child.identity == (server['pid'], server['start'], server['uid'])
                             and server['token'] == self.owner.token and child.session == process.session
                             and child.parent == process.pid, 'server handshake is not the anchored child')
                        self.owner.known[child.identity] = child
                        self.owner.save()
                        need(time.monotonic() < deadline, 'server handshake completed after its deadline')
                        return child
                    time.sleep(.02)
                raise Refusal('server identity handshake deadline expired')
            need(time.monotonic() < deadline, 'target handshake deadline expired')
            self.owner.members()
            time.sleep(0.02)

    def stop(self):
        if self.owner is None:
            return
        owner = self.owner
        profile_error = None
        if self.c.get('profile') and owner.process is not None:
            try:
                result = self.commands.run(self.c['profile']['command'] + ['shutdown',
                    '--session=' + owner.nsys_session, '--kill=sigterm'],
                    min(self.c['timeouts']['query'], max(0, self.hard - time.monotonic())),
                    env=self.env, stdout_path=self.owner_dir / 'nsys-shutdown.log', deadline=self.hard)
                if result.returncode:
                    profile_error = f'nsys shutdown exit {result.returncode}'
            except Exception as error:
                profile_error = str(error)
        owner.terminate(self.c['timeouts']['term'], self.c['timeouts']['kill'], deadline=self.hard)
        # Preserve ownership and the unreaped leader until release is proven.
        self.gpu.idle(min(self.hard, time.monotonic() + self.c['timeouts']['release']))
        owner.reap(timeout=min(1.0, max(0.001, self.hard - time.monotonic())))
        write(self.owner_dir / 'released.json', dict(released=True, profile_shutdown_diagnostic=profile_error))
        self.owner = None
        self.log('released ' + self.owner_dir.name)
        if profile_error:
            raise Refusal(profile_error + '; owned processes were nevertheless cleaned up')

    def run_arm(self, arm):
        self.check()
        need(self.owner is None, 'previous arm is still owned; refusing next arm')
        self.tool_check()
        need(verify_install(self.c['installs'][arm['install']]['path'], self.preflight['model']) == self.preflight['installs'][arm['install']],
             'install changed before arm')
        need(verify_install(self.install_snapshots[arm['install']]['path'], self.preflight['model']) == self.install_snapshots[arm['install']],
             'snapshotted install changed before arm')
        self.gpu.idle(min(self.work, time.monotonic() + self.c['timeouts']['admission']), check=self.check)
        port_free(self.c['port'])
        directory = self.output / arm['name']
        directory.mkdir()
        self.owner_dir = directory
        nsys_name = 'ab-serving-' + uuid.uuid4().hex if self.c.get('profile') else None
        self.owner = Owned(directory / 'ownership.json', nsys_session=nsys_name)
        variant = self.c['variants'][arm['variant']]
        env = dict(self.env, **self.c['installs'][arm['install']]['env'])
        env.update(variant['env'])
        command = self.serve_command(arm)
        wrapped = [sys.executable, '-B', str(self.tools / 'target.py'), '--record', str(directory / 'target.json'),
            '--release', str(directory / 'target.go'), '--token', self.owner.token,
            '--timeout', str(self.c['timeouts']['startup']), '--', *command]
        if self.c.get('profile'):
            wrapped = self.c['profile']['command'] + ['profile', '--session-new=' + nsys_name,
                '--output=' + str(directory / 'profile'), *self.c['profile']['args'], *wrapped]
        write(directory / 'command.json', dict(command=wrapped, environment={k: env[k] for k in
            set(self.c['env']) | set(variant['env']) | set(self.c['installs'][arm['install']]['env'])}))
        self.log('starting ' + arm['name'])
        with (directory / 'server.log').open('wb') as log:
            self.owner.spawn(wrapped, stdout=log, env=env)
        deadline = min(self.work, time.monotonic() + self.c['timeouts']['startup'])
        target = self.handshake(directory, deadline)
        while True:
            self.check()
            now = self.owner.proc.get(target.pid)
            need(now is not None and now.identity == target.identity and now.live, 'server exited before readiness')
            need(not self.owner.exited(), 'launcher exited before readiness')
            try:
                self.health(arm, directory, deadline, startup=True)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            self.owner.members()
            need(time.monotonic() < deadline, 'server readiness deadline expired')
            time.sleep(0.05)
        self.verify_server(arm, target)
        self.owner.state = 'running'; self.owner.save()
        self.owned_gpu(min(self.work, time.monotonic() + self.c['timeouts']['query']))
        samples, counters = [], []
        last_sample = 0.0
        current_phase = None
        def heartbeat():
            nonlocal last_sample
            self.check()
            need(not self.owner.exited(), 'server launcher exited during capture')
            now = self.owner.proc.get(target.pid)
            need(now is not None and now.identity == target.identity and now.live, 'server exited during capture')
            if time.monotonic() - last_sample >= self.c['telemetry_interval']:
                deadline = min(self.work, time.monotonic() + self.c['timeouts']['query'])
                self.owned_gpu(deadline)
                samples.append(dict(monotonic=time.monotonic(), phase=current_phase,
                                    gpus=self.gpu.telemetry(min(self.work, time.monotonic() + self.c['timeouts']['query']))))
                last_sample = time.monotonic()
        try:
            for phase in self.phases:
                current_phase = phase['name']; heartbeat()
                self.health(arm, directory, self.work)
                before = self.request(f'http://127.0.0.1:{self.c["port"]}/metrics', self.work)
                (directory / ('metrics-before-' + current_phase + '.txt')).write_text(before)
                command = [sys.executable, '-B', str(self.tools / 'client.py'), f'http://127.0.0.1:{self.c["port"]}',
                    str(directory / 'capture.json'), phase['kind'], phase['name'], *phase['args'], '--seed', self.c['seed']]
                if phase['kind'] == 'gsm8k':
                    command += ['--gsm8k-examples', str(self.tools / 'gsm8k-examples.json'),
                                '--gsm8k-manifest', str(self.tools / 'gsm8k-manifest.json')]
                if phase['kind'] == 'determinism':
                    # Both arms use the shared tip tools and their matching scorer.
                    command += ['--qual', self.c['qual_tools'], '--scorer', str(self.scorer('new'))]
                result = self.commands.run(command, self.remaining(min(phase['timeout'], self.c['phase_timeout'])),
                    env=env, stdout_path=directory / (current_phase + '.log'), check=heartbeat, deadline=self.hard)
                need(result.returncode == 0, f'capture phase {current_phase} failed with {result.returncode}')
                done = self.phases[:self.phases.index(phase) + 1]
                self.comparator.capture(directory, done)
                after = self.request(f'http://127.0.0.1:{self.c["port"]}/metrics', self.work)
                (directory / ('metrics-after-' + current_phase + '.txt')).write_text(after)
                counters.append(dict(phase=current_phase, **acceptance(before, after,
                    speculative=variant['tokens'] > 0, maximum=variant['tokens'],
                    fixed_k=variant['tokens'] if variant['mode'] == 'fixed' else None,
                    require_drafts=variant['tokens'] > 0 and phase['kind'] in SPECULATIVE_KINDS)))
                self.tool_check()
                self.log(arm['name'] + ': ' + current_phase + ' validated')
            heartbeat()
            self.health(arm, directory, self.work)
            self.verify_server(arm, target)
            capture = self.comparator.capture(directory, self.phases)
            write(directory / 'acceptance.json', counters)
            write(directory / 'telemetry.json', samples)
            write(directory / 'complete.json', dict(capture_sha256=digest(directory / 'capture.json'),
                acceptance_sha256=digest(directory / 'acceptance.json'), telemetry_sha256=digest(directory / 'telemetry.json')))
        finally:
            self.stop()
        need(verify_install(self.c['installs'][arm['install']]['path'], self.preflight['model']) == self.preflight['installs'][arm['install']],
             'install changed during arm')
        need(verify_install(self.install_snapshots[arm['install']]['path'], self.preflight['model']) == self.install_snapshots[arm['install']],
             'snapshotted install changed during arm')
        self.arms.append(dict(**arm, directory=str(directory), capture=capture,
                              acceptance=counters, telemetry=samples))

    def gates(self):
        lookup = {(a['install'], a['variant'], a['repeat']): a for a in self.arms}
        reports = []
        pairs = []
        repeats = (1,) if self.c.get('smoke') else (1, 2)
        for variant in self.c['variants']:
            if self.c.get('smoke'):
                pairs.append((f'base-new-{variant}-1', ('base', variant, 1), ('new', variant, 1),
                              self.c['base_new'] == 'bitwise'))
                continue
            for repeat in repeats:
                pairs.append((f'base-new-{variant}-{repeat}', ('base', variant, repeat), ('new', variant, repeat),
                              self.c['base_new'] == 'bitwise'))
            for install in ('base', 'new'):
                pairs.append((f'determinism-{install}-{variant}', (install, variant, 1), (install, variant, 2), True))
        if 'spec' in self.c['variants'] and 'ar' in self.c['variants']:
            for repeat in repeats:
                pairs.append((f'new-ar-spec-{repeat}', ('new', 'ar', repeat), ('new', 'spec', repeat), True))
        for name, left, right, required in pairs:
            report = self.comparator.compare(lookup[left]['directory'], lookup[right]['directory'], self.phases,
                self.output / (name + '.json'), self.commands, self.remaining(60), required=required,
                scorer=self.scorer('new') if any(p['kind'] == 'determinism' for p in self.phases) else None)
            reports.append(dict(name=name, required=required, **report))
        rates = []
        if 'spec' in self.c['variants']:
            for phase in self.phases:
                if phase['kind'] not in SPECULATIVE_KINDS:
                    continue
                row = dict(phase=phase['name'])
                for install in ('base', 'new'):
                    records, exposures = [], []
                    variant = self.c['variants']['spec']
                    for repeat in repeats:
                        arm = lookup[(install, 'spec', repeat)]
                        matches = [x for x in arm['acceptance'] if x['phase'] == phase['name']]
                        need(len(matches) == 1, 'missing or duplicate phase acceptance record')
                        record = matches[0]
                        records.append(record)
                        if variant['mode'] == 'fixed':
                            exposures.append(fixed_k_exposure(record, variant['tokens'],
                                f'{install} spec repeat {repeat} {phase["name"]}'))
                    drafted = sum(x['drafted'] for x in records); accepted = sum(x['accepted'] for x in records)
                    need(drafted > 0, 'acceptance comparison has zero drafts')
                    row[install] = dict(drafted=drafted, accepted=accepted, rate=accepted / drafted)
                    if variant['mode'] == 'fixed':
                        row[install].update(fixed_k=variant['tokens'], fixed_k_verifies_by_repeat=exposures)
                row['rate_delta'] = row['new']['rate'] - row['base']['rate']
                row['passed'] = row['rate_delta'] >= -self.c['acceptance']['max_rate_drop']
                rates.append(row)
            write(self.output / 'acceptance-gate.json', dict(phases=rates, policy=self.c['acceptance']))
            need(rates and all(r['passed'] for r in rates), 'measured acceptance fell beyond the declared allowance')
        return reports, rates

    def run(self):
        reports = []
        try:
            for arm in plan(self.c):
                self.run_arm(arm)
            reports, rates = self.gates()
            telemetry = []
            for arm in self.arms:
                devices = {g['index'] for sample in arm['telemetry'] for g in sample['gpus']}
                telemetry.append(dict(arm=arm['name'], devices={str(index): {key: statistics.mean(
                    g[key] for s in arm['telemetry'] for g in s['gpus'] if g['index'] == index)
                    for key in ('sm_mhz', 'memory_mhz', 'power_w')} for index in sorted(devices)}))
            summary = dict(schema='chopinfer.ab-serving.result.v1', passed=True,
                scope='configured workload, not a release/accuracy qualification',
                comparator_sha256=self.comparator.sha256, order=self.c['order'], comparisons=reports, acceptance=rates,
                timing=timing_summary(self.arms, self.phases), gpu_means=telemetry,
                profiled=bool(self.c.get('profile')))
            write(self.output / 'result.json', summary)
            self.log('PASS: configured evidence gates; timing is report-only')
            return 0
        except BaseException as error:
            failures = [f'{type(error).__name__}: {error}']
            # Retry cleanup with the same retained owner, never a new arm.
            try:
                self.stop()
            except BaseException as cleanup:
                failures.append('cleanup: ' + str(cleanup))
            try:
                self.commands.cleanup(deadline=self.hard)
            except BaseException as cleanup:
                failures.append('utility cleanup: ' + str(cleanup))
            write(self.output / 'result.json', dict(schema='chopinfer.ab-serving.result.v1', passed=False,
                errors=failures, completed_arms=[a['name'] for a in self.arms],
                ownership_retained=self.owner is not None,
                ownership_record=str(self.owner.record) if self.owner is not None else None))
            self.log('FAIL: ' + '; '.join(failures))
            return 1


def main(argv=None):
    os.umask(0o022)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_file', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    runner = None
    cancelled = False
    def interrupted(signum, frame):
        nonlocal cancelled
        cancelled = True
        if runner is not None:
            runner.cancelled = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        capabilities()
        c, phases, preflight = config(args.run_file)
        comparator = Comparator()
        comparator.module.validate_spec(comparison_spec(phases))
        need(not cancelled, 'interrupted during preflight')
        if args.dry_run:
            print(json.dumps(dict(dry_run=True, gpu_queries=0, config=c, arms=plan(c), preflight=preflight,
                comparator=str(comparator.path), comparator_sha256=comparator.sha256), indent=2))
            return 0
        need(args.out is not None, '--out must name a new external result directory')
        runner = Runner(c, phases, preflight, args.out)
        runner.cancelled = cancelled
        return runner.run()
    except (Exception, KeyboardInterrupt) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
