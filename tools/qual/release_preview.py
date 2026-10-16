#!/usr/bin/env python3
"""The direct correctness run: one model's configurations, served from one
release install, checked against the HF F32 references and the stored
baselines. See tools/qual/README.md#correctness-runner.

    release_preview.sh prepare --model M [--config C ...] [--target-dir DIR]
    release_preview.sh run --model M [--config C ...] [--suite quick|full]
                           [--input KEY=PATH ...] [--input-value KEY=VALUE ...] [--out DIR]
                           [--no-baseline-update]
    release_preview.sh plan --model M [--config C ...] [--suite quick|full]
    release_preview.sh rescore RUN [--legacy-preview MODEL] [--json FILE]
    release_preview.sh baseline show --model M [--config C]
    release_preview.sh baseline reseed --model M --from RUN --reason TEXT

`prepare` (CPU) builds the install of the checked-out commit if it is
missing, in a cargo target this checkout owns, and reuses an existing install
only when it records a build from this checkout and target with unchanged
files. It prepares each checkpoint's direct package and runs `chopinfer verify`
once per checkpoint, caching a marker keyed by the files' identity, size and
times. `run` refuses a missing install or a stale marker (exit 2) rather than
hash inside a GPU run. It starts each arm's server once, records the inputs
the checks read, runs the declared checks and the runner's screens, prints
each check's values, limits and stored values, and exits 1 when any check
failed or did not run. Records are information: run.json says what was
tested, and nothing is refused for missing provenance; only last good's
promotion needs the checkout and every used install source clean and on
main. `rescore`
replays a run with the current code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request

import declaration
import kinds

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
DATA = Path(os.environ.get('CHOPINFER_DATA', '/data'))
VALIDATION = DATA / 'validation'
# Cargo targets prepare builds installs in, one per checkout (target_for), so
# a checkout never builds in, or reuses an install from, another one's target.
TARGET_ROOT = Path(os.environ.get('CHOPINFER_PREVIEW_TARGETS') or VALIDATION / 'targets')
# The file inside a target that names the checkout owning it.
TARGET_OWNER = 'chopinfer-checkout'
# The queue grants these CPUs to GPU jobs; a run keeps to them.
CPUS = set(range(8)) | set(range(24, 32))
MEDIA_CAPABILITY = 'chop.media-inputs.v1'
SUITES = ('quick', 'full')
PASSING = ('pass', 'informational', 'not_applicable', 'waived')
STORES = ('last-good', 'seed')
# Hang guards in seconds, not budgets: a server start, its shutdown, and the
# runner's own stages (a declared check uses its timeout_seconds).
STARTUP_TIMEOUT = 900
SHUTDOWN_TIMEOUT = 30
# A server's single TERM starts its native drain (`--drain-timeout-ms`, 30 s by
# default). Its stop waits that window plus a cleanup margin before the group
# KILL; its timeout wrapper's own KILL comes later still. After the drain,
# native serve delivers final responses, stops its async runtime and joins
# the worker, which cancels requests, drains KV transfers and drops the
# engine; a quiescent TP2 teardown measured 5.2 s. The margin is bounded grace
# for that sequence, not a guarantee: a longer cleanup is a forced stop.
# Utilities keep the 5 s TERM and 10 s KILL waits.
# The server's wrapper runs with --foreground: without it GNU timeout
# forwards a TERM to serve and again to its group, and native serve exits 0 at
# once on a second TERM, skipping its drain. Foreground expiry signals serve
# alone, so a backstop owns the group's cleanup (BACKSTOP below).
DRAIN_DEFAULT_MS = 30000
DRAIN_CLEANUP_SECONDS = 10
STOP_KILL_SECONDS = 10
# The serve flags that take no value (runtime-host VALUELESS_FLAGS_V1).
VALUELESS_SERVE_FLAGS = ('--enable-request-id-headers', '--allow-credentials', '--fake-host-services',
                         '--list-model-options')
WRAPPER_KILL_SECONDS = 5
STAGE_TIMEOUTS = {'record': 3600, 'prefix': 10800, 'gsm8k': 1800, 'gsm8k_branches': 300, 'greedy': 900,
                  'needle': 1800, 'queue': 300, 'prefix_slice': 3600, 'references': 300}
# The AR soak's duration and memory limit (the declared soak runs on the
# speculative server with the declaration's own arguments).
AR_SOAK_SECONDS = 1200
SOAK_GROWTH_MIB = 512
GSM8K_ITEMS = 200
GREEDY_TOKENS = 256
# quick scores one contiguous slice of prefix_exact's position set
# (prefix_exact.py record --shard 1/N; report --slice).
QUICK_PREFIX_SHARDS = 6

# The serve words each arm adds to its configuration's serve command.
FORCED = ['--debug-reference-tokens', '{reference_tokens}']
ARMS = {
    'verify-ar': ['--speculative-tokens', '0', *FORCED],
    'verify-spec': FORCED,
    'prefix': FORCED,
    'reserved': ['--admission', 'reserved'],
    'optimistic': ['--admission', 'optimistic', '--debug-preempt-every', '32'],
    'prefix-baseline': ['--prefix-cache', 'off', '--prefill-chunk-tokens', '2048'],
    'prefix-floor': ['--prefix-cache', 'off', '--prefill-chunk-tokens', '4096'],
    'prefix-candidate': ['--prefix-cache', 'on', '--prefill-chunk-tokens', '2048'],
    'spec-repeat': ['--prefill-chunk-tokens', '1024'],
    'ar': ['--speculative-tokens', '0'],
    'spec': [],
}
# Start order: forced-token and admission arms first, the default (speculative)
# server last, so its declared checks find every record.
ARM_ORDER = tuple(ARMS)
# The configuration's steps that need no server: a module's offline records
# (RUNNER_OFFLINE_RECORDS) and checks (RUNNER_CHECK_ARM "offline"). They run
# after every server of the configuration has stopped, on its GPUs.
OFFLINE = 'offline'
SHARED = ['--url', '{url}', '--model', '{name}', '--timeout', '120']
# A configuration's producer (a model's script, relative to the root) makes
# the run directory a check reads: `record`, against the live server, its taps
# in the arm's directory.
PRODUCER = ['record', '--url', '{url}', '--server-log', '{server_log}', '--taps-dir', '{taps_dir}', '--run',
            '{run_dir}']
# The record each declared input names: (arm, script, arguments). A check
# module may add its own arms and records in the same form, as module-level
# RUNNER_ARMS and RUNNER_RECORDS, for inputs this table does not name
# (README#correctness-runner).
RECORDS = {
    'spec_ar': ('ar', 'spec_verify.py', ['record', *SHARED, '--arm', 'chopinfer-ar', '--run-id', '{run_id}',
                                         '--output-tokens', '64', '--server-log', '{server_log}']),
    'spec_candidate': ('spec', 'spec_verify.py', ['record', *SHARED, '--arm', 'chopinfer-spec', '--run-id',
                                                  '{run_id}', '--output-tokens', '64', '--server-log', '{server_log}']),
    'verify_ar': ('verify-ar', 'verify_exact.py', ['record', '--url', '{url}', '--model', '{name}', '--arm', 'ar',
                                                   '--k', '0', '--require-row-shapes', '--server-log', '{server_log}']),
    'verify_candidate': ('verify-spec', 'verify_exact.py', ['record', '--url', '{url}', '--model', '{name}', '--arm',
                                                            'k{depth}', '--k', '{depth}', '--pooled-decode', '--require-row-shapes',
                                                            '--server-log', '{server_log}']),
    'prefix_record': ('prefix', 'prefix_exact.py', ['record', '--url', '{url}', '--model', '{name}', '--server-log',
                                                    '{server_log}', '--run-id', '{run_id}', '--concurrency', '8']),
    'preemption_reserved': ('reserved', 'preempt_compare.py', ['record', '--arm', 'reserved', '--url', '{url}',
                                                               '--model', '{name}', '--concurrency', '16']),
    'preemption_candidate': ('optimistic', 'preempt_compare.py', ['record', '--arm', 'optimistic', '--url', '{url}',
                                                                  '--model', '{name}', '--concurrency', '16']),
    'preemption_reserved_c1': ('reserved', 'preempt_compare.py', ['record', '--arm', 'reserved', '--url', '{url}',
                                                                  '--model', '{name}', '--concurrency', '1']),
    'preemption_candidate_c1': ('optimistic', 'preempt_compare.py', ['record', '--arm', 'optimistic', '--url',
                                                                     '{url}', '--model', '{name}', '--concurrency',
                                                                     '1']),
    'prefix_baseline': ('prefix-baseline', 'prefix_cache.py', ['record', '--arm', 'baseline', '--url', '{url}',
                                                               '--model', '{name}', '--run-id', '{run_id}']),
    'prefix_floor': ('prefix-floor', 'prefix_cache.py', ['record', '--arm', 'floor', '--url', '{url}', '--model',
                                                         '{name}', '--run-id', '{run_id}']),
    'prefix_candidate': ('prefix-candidate', 'prefix_cache.py', ['record', '--arm', 'candidate', '--url', '{url}',
                                                                 '--model', '{name}', '--run-id', '{run_id}']),
}
# The declared inputs a screen stage records (each arm's determinism record,
# taken by `chopinfer eval capture`).
STAGE_INPUTS = {'determinism_ar': 'ar', 'determinism_spec': 'spec'}
# A sharded prefix record input (prefix_record_I of N) records shard I/N.
SHARD_INPUT = re.compile(r'prefix_record_(\d+)$')
# Inputs that come from elsewhere than a record of this run: an arm's log,
# or the operator's --input.
LOG_INPUT = 'server_log'
# A check input named stored_<check> is the stored receipt of <check> in the
# same configuration and arm: such a check is a regression row, run once per
# store (README#correctness-runner).
STORED = 'stored_'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=30).strip()


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


# Facts ---------------------------------------------------------------------

def declaration_name(model, config):
    """The declaration `checks.py --model` reads for a configuration."""
    return model if config == 'default' else f'{model}.{config}'


def checkpoint_manifest(checkpoint):
    """The native verifier accepts a plain file or a selected entry in it."""
    return checkpoint['lock'] + ('#' + checkpoint['pin'] if 'pin' in checkpoint else '')


def configurations(model):
    """The configurations a model declares (qual.json `configurations`, after
    `default`), each with how it is served: serve words, server environment,
    install features, tensor-parallel size (a configuration at 2 or fewer runs
    beside another on its own GPU pair), the checkpoint's environment
    variable and registry manifest, the served profile and speculation depth,
    and whether the runner's screens run there (only on `default`; another
    configuration runs its declared checks)."""
    facts = model_facts(model)
    result = {'default': {'serve': [], 'environment': {}, 'features': [], 'tensor_parallel_size': 4,
                          'screens': True, 'checkpoint': facts['checkpoint'], 'manifest': facts['manifest'],
                          'profile': facts['profile'], 'depth': facts['depth']}}
    for name, spec in declaration.configurations(model).items():
        serving = spec.get('serving', {})
        pin = (spec.get('qualification') or {}).get('checkpoint')
        result[name] = {
            'serve': list(spec.get('serve', [])), 'environment': dict(spec.get('environment', {})),
            'features': list(spec.get('install_features', [])),
            'tensor_parallel_size': spec.get('tensor_parallel_size', 4), 'screens': spec.get('screens', False),
            'checkpoint': f"CHOPINFER_{spec['checkpoint']}_CHECKPOINT" if 'checkpoint' in spec else facts['checkpoint'],
            'manifest': checkpoint_manifest(pin) if pin else facts['manifest'],
            'profile': serving.get('profile_name', facts['profile']),
            'depth': serving.get('speculation_tokens', facts['depth']), 'producer': spec.get('producer')}
    return result


def configuration_facts(facts, name):
    """The model's facts with a configuration's own checks and cohorts, read
    from its expanded declaration (`chopinfer eval declaration`)."""
    if name != 'default':
        value = read(declaration.named(declaration_name(facts['model'], name)))
        facts = dict(facts, name=value['qualification']['checkpoint']['served_model_name'],
                     checks=value['checks'], quick=value['cohorts']['quick']['checks'],
                     full=value['cohorts']['full']['checks'])
    # A development subset (select_checks) replaces the suite's checks.
    return dict(facts, **facts.get('selected_checks', {}))


def model_facts(model):
    """What the runner serves and checks for a model, from its qual.json
    preview section, profiles.json and model.toml."""
    directory = ROOT / 'models' / model
    path = declaration.path(model)
    declaration.load(path)
    value = read(path)
    preview = value.get('preview')
    require(preview, f'{model}: qual.json declares no preview section')
    profiles = read(directory / 'profiles.json')
    profile = profiles['profiles'].get(preview['profile'])
    require(profile, f"{model}: preview profile {preview['profile']} is not in profiles.json")
    speculation = preview['speculation']
    require(all(profile['speculation'].get(key) == speculation[key] for key in speculation),
            f"{model}: profiles.json {preview['profile']} speculation differs from the declared preview speculation")
    inventory = tomllib.loads((directory / 'model.toml').read_text())
    helpers = [name for name in inventory['compiler']['bin'] if name.startswith('chopinfer-model-')]
    require(len(helpers) == 1, f'{model}: model.toml must list one chopinfer-model-* helper')
    checkpoint = value['qualification']['checkpoint']
    declared = (value['checks'].get('soak') or {}).get('constants', {}).get('soak', {})
    return {'model': model, 'name': profiles['served_model_name'], 'bundle': profile['bundle_id'],
            'helper': helpers[0], 'profile': preview['profile'], 'max_sequences': profile['max_sequences'],
            'topology': profile['topology'],
            'manifest': checkpoint_manifest(checkpoint),
            'checkpoint': f"CHOPINFER_{inventory['checkpoint_env']}_CHECKPOINT",
            'context': max(value['qualification']['needle']['lengths']),
            'drafter': speculation['drafter'], 'depth': speculation['tokens'],
            'packages': [inventory['compiler']['package'], inventory['bundle']['package']],
            'modules': [f'lib{name}.so' for name in inventory['bundle']['cdylib']],
            'quick': value['cohorts']['quick']['checks'], 'full': value['cohorts']['full']['checks'],
            'quick_arms': tuple(preview.get('quick_arms', ('ar', 'spec'))),
            'spec_greedy': preview['spec_greedy']['rule'], 'gsm8k_floor': preview['gsm8k_floor'],
            'soak_prefix_mix': preview.get('soak_prefix_mix', True),
            'numerical_reference': preview['numerical_reference'],
            'soak_constants': declared,
            'soak_warmup': declared.get('WARMUP_SECONDS', {}).get('value', 0),
            'soak_allowance': declared.get('CACHE_ALLOWANCE_MIB', {}).get('value', {'host': 0, 'device': 0}),
            'checks': value['checks'], 'kv_offload': value.get('kv_offload')}


def queue_clients(facts):
    return 2 * facts['max_sequences'] + 1


# Processes -----------------------------------------------------------------

def timeout_command(command, seconds, kill_after=10, foreground=False):
    # Only the server's wrapper runs in the foreground; a utility's expiry
    # keeps signaling its whole group.
    command = list(map(str, command))
    command[0] = shutil.which(command[0]) or command[0]
    require(Path(command[0]).is_absolute(), f'executable must resolve to an absolute path: {command[0]}')
    return [shutil.which('timeout') or '/usr/bin/timeout', *(['--foreground'] if foreground else []),
            '--signal=TERM', f'--kill-after={int(kill_after)}s', f'{int(seconds)}s', *command]


# The server's cleanup backstop. The launcher runs as the server session's
# leader, forks the watchdog and execs the timeout wrapper under the same pid,
# so the session and group id is the wrapper's pid. The watchdog leaves that
# group for its own, so the stop's group probe never sees it, but stays in the
# session, which keeps the id from being reused while it lives: it can only
# signal the wrapper and the server's group. It sends the server's one TERM,
# to the wrapper, when the runner's stop asks for it over the socket, or on its
# own when the socket closes without a request (the runner died). From that
# TERM, or from the wrapper's exit, it waits kill_after_seconds, KILLs the
# group, and exits once the group is empty or 10 s after its first KILL. Its
# log line is best effort and never delays the KILL.
BACKSTOP = r'''
import os, select, signal, socket, sys, time
link, bound, wrapper = socket.socket(fileno=int(sys.argv[1])), float(sys.argv[2]), sys.argv[3:]
group = os.getpid()
if os.fork():
    link.close()
    os.execv(wrapper[0], wrapper)
for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(number, signal.SIG_IGN)
os.setpgid(0, 0)

def reply(data):
    try:
        link.sendall(data)
    except OSError:
        pass

def term():
    try:
        os.kill(group, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False

reply(b'%d\n' % os.getpid())
deadline, killed = None, False
try:
    while True:
        os.killpg(group, 0)
        if deadline is None:
            try:
                request = link.recv(64) if select.select([link], [], [], .1)[0] else None
            except OSError:
                request = b''
            if request is not None:
                # A stop request, or the runner is gone: one TERM, armed from it.
                sent = term()
                if request:
                    reply(b'TERM\n' if sent else b'GONE\n')
                deadline = time.monotonic() + bound
            elif os.getppid() != group:
                deadline = time.monotonic() + bound
        elif time.monotonic() < deadline:
            time.sleep(.1)
        elif time.monotonic() < deadline + 10:
            os.killpg(group, signal.SIGKILL)
            if not killed:
                killed = True
                try:
                    os.write(2, b'backstop: KILL process group %d\n' % group)
                except Exception:
                    pass
            time.sleep(.1)
        else:
            os._exit(1)
except ProcessLookupError:
    os._exit(0)
'''


def server_backstop(drain):
    """The backstop's KILL delay from its TERM: past the stop's TERM and KILL
    waits and the wrapper's own KILL."""
    return {'kill_after_seconds': drain['term_wait_seconds'] + STOP_KILL_SECONDS + WRAPPER_KILL_SECONDS}


def backstopped(wrapper, kill_after_seconds, **options):
    """Start `wrapper` in its own session under BACKSTOP. Returns the
    wrapper's process and the runner's end of the watchdog's socket: closing
    it without a stop request makes the watchdog stop the server."""
    link, theirs = socket.socketpair()
    with theirs:
        try:
            process = subprocess.Popen([sys.executable, '-I', '-c', BACKSTOP, str(theirs.fileno()),
                                        str(kill_after_seconds), *wrapper], pass_fds=(theirs.fileno(),),
                                       start_new_session=True, **options)
        except BaseException:
            link.close()
            raise
    return process, link


def backstop_line(link, seconds=10):
    link.settimeout(seconds)
    data = b''
    while not data.endswith(b'\n'):
        chunk = link.recv(64)
        require(chunk, 'the server backstop is gone')
        data += chunk
    return data


def backstop_pid(link, seconds=10):
    return int(backstop_line(link, seconds))


def backstop_term(link, seconds=10):
    """Have the backstop send the server's one TERM and arm its KILL from it;
    True once it confirms the TERM reached the wrapper."""
    try:
        link.sendall(b'T')
        return backstop_line(link, seconds) == b'TERM\n'
    except (OSError, ValueError):
        return False


def serve_options(arguments):
    """The (flag, values) options of `serve` arguments, as runtime-host
    profiles::options and the serve parser split them: a valueless flag takes
    none, --served-model-name every following word up to the next option, and
    every other flag the next word even when it looks like an option."""
    options, words = [], list(map(str, arguments))
    while words:
        flag, values = words.pop(0), []
        if flag not in VALUELESS_SERVE_FLAGS and words:
            values.append(words.pop(0))
            while flag == '--served-model-name' and words and not words[0].startswith('--'):
                values.append(words.pop(0))
        options.append((flag, values))
    return options


def server_drain(arguments):
    """The drain window `serve` arguments select, read at the option
    positions and with the u64 syntax `chopinfer serve` uses (30000 when
    absent; serve --model applies a profile, which carries no drain), with
    the stop's TERM wait and the wrapper's later KILL. Refuses what serve
    would refuse, never rewrites it."""
    flag = '--drain-timeout-ms'
    given = []
    for option, values in serve_options(arguments):
        require(not option.startswith(flag + '='), f'serve reads no {flag}= option: {option}')
        if option == flag:
            given.append(values)
    require(len(given) <= 1, f'{flag} is given more than once')
    if given:
        value = given[0][0] if given[0] else None
        require(value is not None and re.fullmatch(r'\+?[0-9]+', value) and int(value) < 2 ** 64,
                f'{flag} must be an unsigned 64-bit integer: {value}')
        milliseconds, source = int(value), 'serve'
    else:
        milliseconds, source = DRAIN_DEFAULT_MS, 'default'
    wait = -(-milliseconds // 1000) + DRAIN_CLEANUP_SECONDS
    return {'source': source, 'milliseconds': milliseconds, 'term_wait_seconds': wait,
            'wrapper_kill_after_seconds': wait + WRAPPER_KILL_SECONDS}


def stop(process, *, timeout_forwarder=False, term_wait=5, forward=None):
    # Only server() passes its owned timeout Popen as a TERM forwarder, its
    # drain's TERM wait, and `forward`, which has the backstop send the TERM
    # and returns whether it was sent. Utilities keep group TERM; hard KILL
    # always covers the owned group.
    stages = []
    for signal_number, wait in ((signal.SIGTERM, term_wait), (signal.SIGKILL, STOP_KILL_SECONDS)):
        stage = {'signal': signal_number.name, 'wait_seconds': wait, 'sent': False, 'timed_out': False}
        stages.append(stage)
        try:
            if timeout_forwarder and signal_number == signal.SIGTERM:
                stage['target'] = 'timeout_process'
                # Signaling both timeout and its group delivers TERM twice.
                # Popen checks/reaps its own child before sending the signal.
                if process.poll() is None:
                    if forward is None:
                        process.send_signal(signal_number)
                        stage['sent'] = True
                    else:
                        stage['via'] = 'backstop'
                        stage['sent'] = forward()
            else:
                os.killpg(process.pid, signal_number)
                stage['sent'] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=wait)
            if timeout_forwarder and signal_number == signal.SIGTERM:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                # A reaped wrapper alone cannot prove its descendants exited.
                # Force the surviving group and retain a failed termination.
                stage['group_remaining'] = True
            else:
                break
        except subprocess.TimeoutExpired:
            stage['timed_out'] = True
    # An expired TERM wait is forced even when the exit lands before the KILL.
    return {'returncode': process.returncode, 'confirmed': type(process.returncode) is int,
            'forced': any(stage.get('group_remaining') or stage['timed_out']
                          or (stage['signal'] == 'SIGKILL' and stage['sent']) for stage in stages),
            'stages': stages}


def environment(gpu=False, target=None, config=None):
    """This process's environment for children: shipping defaults only (no
    CHOPINFER_* debug overrides), the declared input locations kept."""
    env = dict(os.environ)
    keep = {'CHOPINFER_DATA', 'CHOPINFER_LOCAL_ENV', 'CHOPINFER_EVAL', 'CHOPINFER_REFRUN_PYTHON'}

    def declared(value):
        if isinstance(value, dict):
            if isinstance(value.get('env'), str):
                keep.add(value['env'])
            for child in value.values():
                declared(child)
        elif isinstance(value, list):
            for child in value:
                declared(child)

    if config is not None:
        declared(config.get('inputs', {}))
        declared(config.get('qualification', {}).get('references', {}))
    for key in list(env):
        if key.startswith('CHOPINFER_') and key not in keep and not key.endswith(('_CHECKPOINT', '_FIXTURES')):
            del env[key]
    env.update(CARGO_NET_OFFLINE='true', CARGO_INCREMENTAL='0', PYTHONDONTWRITEBYTECODE='1',
               CHOPINFER_DATA=str(DATA), CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if target:
        env['CARGO_TARGET_DIR'] = str(target)
    if gpu:
        env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def call(command, log, seconds, env):
    """Run `command` with its output in `log`; returns its exit status."""
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('w') as output:
        output.write(json.dumps({'command': list(map(str, command)), 'timeout_seconds': seconds}) + '\n')
        output.flush()
        process = subprocess.Popen(timeout_command(command, seconds), cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=seconds + 15)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            stop(process)


def checked(command, log, seconds, env):
    code = call(command, log, seconds, env)
    require(code == 0, f'exit {code}: {log}')


# Install and checkpoint verify ------------------------------------------------

def source():
    """The checked-out commit and whether the tree has changes (recorded,
    never refused)."""
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain', '--untracked-files=no')
    return {'commit': commit, 'dirty': bool(status),
            'dirty_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest() if status else None}


def checkout():
    """The checkout this runner builds and runs: the repository it lives in."""
    return ROOT.resolve()


def target_for(root):
    """The default cargo target of checkout `root`: wt-<name>-<12 hex of
    the sha256 of its path> under TARGET_ROOT."""
    name = re.sub(r'[^A-Za-z0-9._-]', '_', Path(root).name)
    return TARGET_ROOT / f"wt-{name}-{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"


def build_target(argument):
    """The cargo target prepare builds in: --target-dir, else the caller's
    CARGO_TARGET_DIR, else this checkout's own (target_for). The target must
    belong to this checkout: one another checkout claimed is refused, and an
    unclaimed one is claimed only when it is empty or a `<target>.worktree`
    file beside it names this checkout. The claim links a complete owner file
    into place, so of two checkouts claiming at once one wins and the other
    is refused. Then cargo's freshness never rests on another checkout's
    sources."""
    root = checkout()
    target = Path(argument or os.environ.get('CARGO_TARGET_DIR') or target_for(root)).resolve()
    owner = target / TARGET_OWNER
    if not owner.is_file():
        assigned = target.with_name(target.name + '.worktree')
        require(not target.exists() or not any(target.iterdir())
                or (assigned.is_file() and assigned.read_text().strip() == str(root)),
                f'the cargo target {target} is in use and names no owning checkout: give this checkout '
                'its own, empty --target-dir')
        target.mkdir(parents=True, exist_ok=True)
        descriptor, claim = tempfile.mkstemp(prefix=f'{TARGET_OWNER}.', dir=target)
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, 'w') as handle:
                handle.write(f'{root}\n')
            os.link(claim, owner)
        except FileExistsError:
            pass
        finally:
            os.unlink(claim)
    require(owner.read_text().strip() == str(root),
            f'the cargo target {target} belongs to the checkout {owner.read_text().strip()}, not {root}: '
            'give this checkout its own --target-dir')
    return target


def built_here(root, state, identity, target):
    """Whether the install at `root` was built from this checkout at
    `identity` in `target`, with every file it lists unchanged since."""
    build = state.get('build')
    if not isinstance(build, dict) or not state.get('files') or build.get('files') != state['files']:
        return False
    if (build.get('commit'), build.get('dirty_sha256'), build.get('checkout'), build.get('target')) != (
            identity['commit'], identity['dirty_sha256'], str(checkout()), str(target)):
        return False
    return all((root / name).is_file() and sha(root / name) == digest for name, digest in state['files'].items())


def install_root(identity, features=()):
    name = identity['commit'][:12] + (f"-dirty-{identity['dirty_sha256'][:12]}" if identity['dirty'] else '')
    return VALIDATION / 'installs' / (name + ''.join(f'+{feature}' for feature in sorted(features)))


def host_name():
    manifest = tomllib.loads((ROOT / 'crates/runtime-host/Cargo.toml').read_text())
    names = [row['name'] for row in manifest['bin'] if row['name'].endswith('-host')]
    require(len(names) == 1, 'runtime-host must declare one worker host binary')
    return names[0]


def build_commands(facts, features=()):
    """The release build (tools/release/build.sh) without packaging: the CUDA
    host, the model's helper and its native module."""
    cargo = ['cargo', 'build', '--release', '--locked', '--offline']
    bundle = cargo + ['-p', facts['packages'][1], '--lib']
    if features:
        bundle += ['--features', ','.join(sorted(features))]
    return [cargo + ['-p', 'runtime-host', '--no-default-features', '--features', 'linux-cuda-host',
                     '--bin', 'chopinfer', '--bin', host_name()],
            cargo + ['-p', facts['packages'][0], '--bin', facts['helper']], bundle]


def install(root, target, facts):
    """Copy the build into the release archive layout (helpers beside the
    host in bin/, native modules in lib/), nothing group or world writable."""
    files = {}
    for directory, names in (('bin', ['chopinfer', host_name(), facts['helper']]), ('lib', facts['modules'])):
        (root / directory).mkdir(parents=True, exist_ok=True)
        for name in names:
            destination = root / directory / name
            shutil.copyfile(Path(target) / 'release' / name, destination)
            destination.chmod(0o755)
            files[f'{directory}/{name}'] = sha(destination)
    for path in [root, *root.rglob('*')]:
        if path.is_dir():
            path.chmod(0o755)
    return files


def checkpoint_state(root):
    """Device, inode, size, mtime and ctime of every checkpoint file (through
    hub snapshot links). An equal state detects rewrites, replacement and
    truncation through ordinary writes; the checkpoint must stay immutable for
    the bytes `verify` hashed to be the bytes a server reads."""
    root = Path(root)
    state = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            info = path.stat()
            if stat.S_ISREG(info.st_mode):
                state[str(path.relative_to(root))] = [info.st_dev, info.st_ino, info.st_size,
                                                      info.st_mtime_ns, info.st_ctime_ns]
    require(state, f'checkpoint has no files: {root}')
    return state


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def manifest_identity(manifest):
    """Read the selected manifest without following a file link. Identity and
    content must stay stable across both this read and the native verify."""
    require(isinstance(manifest, str) and manifest, 'checkpoint verification needs a manifest')
    file, separator, entry = manifest.rpartition('#')
    if not separator:
        file, entry = manifest, None
    require(file and (not separator or entry), 'manifest must be FILE or FILE#ENTRY')
    path = (ROOT / file).absolute()

    def identity(info):
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode), f'verification manifest is not a regular file: {path}')
        before = identity(info)
        # Match the native verifier's document bound.
        data = stream.read((64 << 20) + 1)
        require(len(data) <= 64 << 20, f'verification manifest is too large: {path}')
        require(before == identity(os.fstat(stream.fileno())) == identity(path.lstat()),
                f'verification manifest changed while being read: {path}')
    return {'path': str(path), 'entry': entry, 'sha256': hashlib.sha256(data).hexdigest(), 'file': before}


def marker_path(model, checkpoint, manifest):
    selection = json.dumps([str(checkpoint), manifest]).encode()
    return VALIDATION / 'verified' / f'{model}-{hashlib.sha256(selection).hexdigest()[:16]}.json'


def verify_marker(model, checkpoint, manifest):
    """The cached verify only while checkpoint state and the selected
    manifest's identity and content match. Legacy markers cannot qualify."""
    path = marker_path(model, checkpoint, manifest)
    if not path.is_file():
        return None
    try:
        value = read(path)
        current = manifest_identity(manifest)
        return value if (isinstance(value, dict) and value.get('manifest') == current
                         and value.get('state_sha256') == state_digest(checkpoint_state(checkpoint))) else None
    except (OSError, ValueError):
        return None


def verify_summary(log):
    lines = [line for line in Path(log).read_text().splitlines() if line.startswith('verify: ')]
    require(len(lines) == 1 and ' 0 failed, 0 missing' in lines[0], f'checkpoint verify summary: {log}')
    return lines[0]


def last_json(path):
    """The JSON document a command printed last, after any progress lines."""
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    for start in reversed([0] + [i + 1 for i, c in enumerate(text) if c == '\n']):
        if text.startswith('{', start):
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                continue
            if not text[end:].strip():
                return value
    raise ValueError(f'{path}: no JSON result')


def direct_package(facts, value, config):
    """A direct package's identities and media capability; it must be the
    configuration's bundle at its tensor-parallel size and serve its profile
    through its defaults (the declared speculation where that is the
    preview's profile)."""
    package = value.get('package', value) if isinstance(value, dict) else {}
    size = config['tensor_parallel_size']
    require(package.get('bundle_id') == facts['bundle'] and package.get('tensor_parallel_size') == size,
            f"{facts['model']}: direct package is not the TP{size} bundle")
    profile = package.get('profile') or {}
    settings = profile.get('profile') or {}
    speculation = settings.get('speculation') or {}
    require(profile.get('name') == config['profile'], f"{facts['model']}: helper selected {profile.get('name')}")
    if config['profile'] == facts['profile']:
        require(speculation.get('tokens') == facts['depth'] and speculation.get('drafter') == facts['drafter'],
                f"{facts['model']}: profile speculation changed: {speculation}")
    bundle = Path(package['bundle'])
    manifest = read(bundle / 'manifest.json')
    media = MEDIA_CAPABILITY in ((manifest.get('resource_requirements') or {}).get('required_capabilities') or [])
    return {'bundle': str(bundle), 'profile': profile['name'], 'media': media,
            **{key: package.get(key) for key in ('package_identity', 'checkpoint_identity', 'metadata_identity')}}


def image_request(package, words):
    """The determinism record's image words for a server of `package`
    started with serve `words`: `--image` when the package provides media and
    `--modalities` selects image (`auto`, the default, and `all` do; a list
    must name it), else `--image-reason` saying why not."""
    if not package['media']:
        return ['--image-reason', 'the configuration\'s package serves no image inputs']
    selected = words[words.index('--modalities') + 1] if '--modalities' in words else 'auto'
    if selected in ('auto', 'all') or 'image' in selected.split(','):
        return ['--image']
    return ['--image-reason', f'the configuration serves no image inputs (--modalities {selected})']


def launched_image(package, directory):
    """The image words for the server an arm launched, from its recorded
    serve command, so a capture and every later rescore of it read the same
    server; a launch that records no command is judged by the package alone,
    as before served modalities were read."""
    command = read(directory / 'launch.json').get('command')
    require(command is None or (isinstance(command, list) and all(isinstance(word, str) for word in command)
                                and command[1:2] == ['serve']),
            f'{directory}: invalid server launch command')
    return image_request(package, command[2:] if command else [])


def prepare(args):
    facts = model_facts(args.model)
    configs = selected_configurations(args.model, args.config, args.suite)
    identity = source()
    target = build_target(args.target_dir)
    env = environment(target=target)
    needed = {name: configuration_features(facts, name, config, args.suite) for name, config in configs.items()}
    for features in sorted(set().union(*needed.values())):
        root = install_root(identity, features)
        record = root / 'install.json'
        state = read(record) if record.is_file() else {'source': identity, 'files': {}, 'packages': {}}
        # Evidence must come from binaries built from this checkout: an
        # install is reused, or extended with another model's helper, only
        # when every file it holds was.
        require(not state['files'] or built_here(root, state, identity, target),
                f'{root} was not built from this checkout ({checkout()}) in its target ({target}), or its '
                f'files changed since: move {root} and {root.parent / (root.name + "-build")} aside, then '
                'prepare again to rebuild it here')
        if not all(f'bin/{name}' in state['files'] for name in ('chopinfer', facts['helper'])):
            logs = root.parent / f'{root.name}-build'
            for index, command in enumerate(build_commands(facts, features)):
                checked(command, logs / f'build-{index + 1}.log', 14400, env)
            state['files'].update(install(root, target, facts))
            state['build'] = {'commit': identity['commit'], 'dirty_sha256': identity['dirty_sha256'],
                              'checkout': str(checkout()), 'target': str(target), 'built': utc(),
                              'files': dict(state['files'])}
            write(record, state)
        for name, config in configs.items():
            if features not in needed[name]:
                continue
            checkpoint = Path(os.environ.get(config['checkpoint'], '/missing'))
            require(checkpoint.is_absolute() and checkpoint.is_dir(), f"set {config['checkpoint']}")
            logs = root / 'logs' / args.model / name
            if verify_marker(args.model, checkpoint, config['manifest']) is None:
                before = checkpoint_state(checkpoint)
                manifest = manifest_identity(config['manifest'])
                command = [root / 'bin/chopinfer', 'verify', '--model', checkpoint,
                           '--manifest', ROOT / config['manifest'], '--allow-unrecorded']
                checked(command, logs / 'verify.log', 14400, env)
                require(checkpoint_state(checkpoint) == before, f'{args.model}: checkpoint changed while it was verified')
                require(manifest_identity(config['manifest']) == manifest,
                        f'{args.model}: verification manifest changed while checkpoint was verified')
                write(marker_path(args.model, checkpoint, config['manifest']), {
                    'manifest': manifest,
                    'model': args.model, 'path': str(checkpoint), 'summary': verify_summary(logs / 'verify.log'),
                    'state_sha256': state_digest(before), 'files': len(before), 'at': utc(),
                    'verified_by': str(root / 'bin/chopinfer')})
            packages = []
            # The second request must reuse the first package, as each serve does.
            for attempt in ('emit', 'reuse'):
                command = [root / 'bin' / facts['helper'], 'prepare', '--direct', '--model', checkpoint,
                           '--tensor-parallel-size', str(config['tensor_parallel_size']),
                           '--cache-dir', root / 'model-cache',
                           *config.get('prepare', [])]
                checked(command, logs / f'direct-{attempt}.log', 3600, env)
                packages.append(direct_package(facts, last_json(logs / f'direct-{attempt}.log'), config))
            require(packages[0] == packages[1], f'{args.model}: a repeated direct preparation selected another package')
            state['packages'][f'{args.model}/{name}'] = {'checkpoint': str(checkpoint), **packages[0]}
            write(record, state)
        print(f'Prepared: {root}')
    return 0


# Planning ------------------------------------------------------------------

def selected_configurations(model, names, suite='quick'):
    """The named configurations; by default `default` for quick and every
    declared one for full."""
    declared = configurations(model)
    names = names or (['default'] if suite == 'quick' else list(declared))
    unknown = [name for name in names if name not in declared]
    require(not unknown, f'{model} declares no configuration {", ".join(unknown)}; it declares {sorted(declared)}')
    return {name: declared[name] for name in names}


def module_option(module, option, default, model=None):
    """Optional runner facts owned by a check's module."""
    if not (TOOLS / f'{module}.py').is_file():
        return default
    import importlib
    value = getattr(importlib.import_module(module), option, default)
    return value(model) if callable(value) else value


def configuration_features(facts, name, config, suite):
    """The installs a configuration and its diagnostic arms need."""
    return {tuple(sorted(config['features'])),
            *(tuple(sorted(arm.get('features', config['features'])))
              for arm in plan(facts, suite, name, config))}


def module_records(module, model=None):
    """A module's arms and records; callable options select its declared model inputs."""
    return (module_option(module, 'RUNNER_ARMS', {}, model),
            module_option(module, 'RUNNER_RECORDS', {}, model))


def placeholders(check, suite=None):
    """The input names a declared check's arguments, cohort arguments (of
    `suite`, else every cohort's), files and input rules (such as a recorded
    arm's run ID) read."""
    cohort = check.get('cohort_args', {})
    text = json.dumps([check.get('args', []), cohort if suite is None else cohort.get(suite, []),
                       check.get('input_rules', {})])
    # {check_raw[ID]} reads check ID's run, as {ID} reads its receipt.
    return {*check.get('files', []), *re.findall(r'\{(\w+)\}', text), *re.findall(r'\{check_raw\[(\w+)\]\}', text)}


def stored_inputs(check):
    """The stored_<check> inputs of a regression check, sorted."""
    return sorted(name for name in placeholders(check) if name.startswith(STORED))


def record_for(name, check, model=None):
    """(arm, script, arguments) of the record a check's input names, or None."""
    shard = SHARD_INPUT.match(name)
    if shard:
        arm, script, arguments = RECORDS['prefix_record']
        count = len([key for key in check.get('files', []) if SHARD_INPUT.match(key)])
        return arm, script, [*arguments, '--shard', f'{shard.group(1)}/{count}']
    if name in RECORDS:
        return RECORDS[name]
    return module_records(check['module'], model)[1].get(name)


def serve_overrides(base, overrides):
    """Replace configuration options that an arm pins; the CLI refuses duplicates."""
    pinned = {word for word in overrides if word.startswith('--')}
    words = []
    index = 0
    while index < len(base):
        word = base[index]
        count = 2 if index + 1 < len(base) and not base[index + 1].startswith('--') else 1
        if word not in pinned:
            words.extend(base[index:index + count])
        index += count
    return [*words, *overrides]


def plan(facts, suite, config_name, config, inputs=()):
    """The arms of one configuration, in start order, each with its steps:
    records (an input a check reads), stages (the runner's own screens) and
    checks (declared checks, by id). Inputs given with --input are not
    recorded."""
    facts = configuration_facts(facts, config_name)
    checks = facts['checks']
    selected = facts['quick'] if suite == 'quick' else facts['full']
    arms = {}

    def extra_arms(module):
        extra = module_records(module, facts['model'])[0]
        return extra(facts, config) if callable(extra) else extra

    def arm(name):
        if name == OFFLINE and name not in arms:
            arms[name] = {'arm': OFFLINE, 'serve': [], 'server': False, 'steps': []}
        if name not in arms:
            words = ARMS.get(name)
            if words is None:
                words = next((extra[name] for check in selected
                              for extra in [extra_arms(checks[check]['module'])] if name in extra), None)
            require(words is not None, f'no serve words for arm {name}')
            arms[name] = {'arm': name, 'serve': list(words), 'steps': []}
            environment, features = {}, set(config['features'])
            for check in selected:
                module = checks[check]['module']
                environment.update(module_option(module, 'RUNNER_ENVIRONMENT', {}, facts['model']).get(name, {}))
                features.update(module_option(module, 'RUNNER_FEATURES', {}, facts['model']).get(name, []))
            if environment:
                arms[name]['environment'] = environment
            if environment or features != set(config['features']):
                arms[name]['features'] = sorted(features)
        return arms[name]

    def step(name, kind, **fields):
        steps = arm(name)['steps']
        if not any(row['id'] == fields['id'] for row in steps):
            steps.append({'kind': kind, **fields})

    screens = config.get('screens', True)
    # Every fresh server records determinism first, so each arm's capture
    # starts from the same state; a configuration without the screens records
    # it only where a declared check reads it.
    declared = {name for check in selected for name in checks[check].get('files', [])}
    for name in ('spec', 'ar', 'spec-repeat'):
        if screens or f'determinism_{name}' in declared:
            step(name, 'stage', id=f'determinism-{name}', stage='determinism')
    if suite == 'full':
        for check in selected:
            for name in checks[check].get('files', []):
                if name in (LOG_INPUT, *STAGE_INPUTS) or name in inputs:
                    continue
                offline = module_option(checks[check]['module'], 'RUNNER_OFFLINE_RECORDS', {},
                                        facts['model']).get(name)
                if offline is not None:
                    # Recorded from this run's other records once the servers
                    # are down: its output directory is the input.
                    step(OFFLINE, 'record', id=name, input=name, script=offline[0], arguments=offline[1],
                         produces=True)
                    continue
                record = record_for(name, checks[check], facts['model'])
                if record is None and config.get('producer') and name not in module_option(
                        checks[check]['module'], 'RUNNER_EXTERNAL_INPUTS', [], facts['model']):
                    # The configuration's producer makes the run directory
                    # its checks read, against the speculative server.
                    step('spec', 'record', id=name, input=name, script=config['producer'], arguments=PRODUCER,
                         produces=True)
                    continue
                if record is None:
                    continue
                where, script, arguments = record
                output = ({'produces': True} if name in module_option(
                    checks[check]['module'], 'RUNNER_DIRECTORY_INPUTS', [], facts['model']) else {})
                step(where, 'record', id=name, input=name, script=script, arguments=arguments, **output)
    if screens:
        for name in ('ar', 'spec'):
            step(name, 'stage', id=f'gsm8k-{name}', stage='gsm8k')
            step(name, 'stage', id=f'greedy-{name}', stage='greedy')
            key = 'spec_ar' if name == 'ar' else 'spec_candidate'
            step(name, 'record', id=key, input=key, script=RECORDS[key][1], arguments=RECORDS[key][2])
        if suite == 'quick':
            step('spec', 'stage', id='needle', stage='needle')
            step('spec', 'stage', id='prefix_slice', stage='prefix_slice')
        else:
            step('ar', 'stage', id='soak-ar', stage='soak')
        step('spec', 'stage', id='queue', stage='queue')
        step('spec', 'stage', id='gsm8k_branches', stage='gsm8k_branches')
    for check in selected:
        check_arm = module_option(checks[check]['module'], 'RUNNER_CHECK_ARM', 'spec', facts['model'])
        for name in (facts['quick_arms'] if suite == 'quick' and screens and check_arm == 'spec' else (check_arm,)):
            step(name, 'check', id=f'check-{check}', check=check)
    if suite == 'full' and screens:
        # Full also answers quick's regression checks on the other quick arms
        # (with the checks they compare), so a stored full pass holds every
        # arm quick compares against.
        for name in facts['quick_arms']:
            for check in (check for check in selected if name != 'spec' and check in facts['quick']
                          and stored_inputs(checks[check])):
                for source in stored_inputs(checks[check]):
                    if source[len(STORED):] in checks:
                        step(name, 'check', id=f'check-{source[len(STORED):]}', check=source[len(STORED):])
                step(name, 'check', id=f'check-{check}', check=check)
    ordered = [arms[name] for name in ARM_ORDER if name in arms]
    ordered += [value for name, value in arms.items() if name not in ARM_ORDER]
    # The speculative server answers the declared checks: it starts last, and
    # the offline steps follow once it has stopped.
    ordered.sort(key=lambda value: 2 if value['arm'] == OFFLINE else value['arm'] == 'spec')
    for value in ordered:
        if value.get('server', True):
            value['serve'] = serve_overrides(config['serve'], value['serve'])
    return ordered


def select_checks(facts, configs, suite, selected):
    """A development subset has no model screens and cannot update last good.
    Each selected configuration must declare every selected check in the suite."""
    if not selected:
        return facts, configs
    for name in configs:
        unknown = set(selected) - set(configuration_facts(facts, name)[suite])
        require(not unknown, f"checks outside {name} {suite}: {sorted(unknown)}")
    return ({**facts, 'selected_checks': {suite: list(dict.fromkeys(selected))}},
            {name: {**config, 'screens': False} for name, config in configs.items()})


def plan_command(args):
    facts = model_facts(args.model)
    configs = selected_configurations(args.model, args.config, args.suite)
    facts, configs = select_checks(facts, configs, args.suite, getattr(args, 'check', []))
    print(json.dumps({name: plan(facts, args.suite, name, config) for name, config in configs.items()}, indent=2))
    return 0


# Servers -------------------------------------------------------------------

def fresh_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def listener_owned(port, group):
    output = subprocess.check_output([shutil.which('ss') or '/usr/bin/ss', '-H', '-ltnp', f'sport = :{port}'],
                                     text=True, timeout=5)
    pids = {int(pid) for pid in re.findall(r'\bpid=(\d+)\b', output)}
    return len(pids) == 1 and all(os.getpgid(pid) == group for pid in pids)


def http(url, body=None, seconds=5):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers)
    # No retries: a 429/503 must not be hidden by the queue check.
    with urllib.request.urlopen(request, timeout=seconds) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


@contextmanager
def server(context, arm, directory):
    """One arm's server: `chopinfer serve --model DIR` from the install, as a
    user starts it, with the arm's serve words. Yields (url, pid)."""
    port = fresh_port()
    url = f'http://127.0.0.1:{port}'
    words = [word.format(**context['fill']) for word in arm['serve']]
    command = [context['install'] / 'bin/chopinfer', 'serve', '--model', context['checkpoint'],
               '--tensor-parallel-size', str(context['config'].get('tensor_parallel_size', 4)),
               '--model-cache-dir', context['install'] / 'model-cache',
               '--listen', f'127.0.0.1:{port}', '--served-model-name', context['facts']['name'], *words]
    drain = server_drain(command[2:])
    backstop = server_backstop(drain)
    # Duration 0 disables timeout's own expiry, which would be a second TERM
    # source; its KILL delay still follows the backstop's forwarded TERM.
    wrapper = timeout_command(command, 0, drain['wrapper_kill_after_seconds'], foreground=True)
    write(directory / 'launch.json', {'command': list(map(str, command)), 'wrapper': wrapper, 'url': url,
                                      'drain': drain, 'backstop': backstop,
                                      'environment': {**context['config']['environment'], **arm.get('environment', {})},
                                      'devices': context['env'].get('CUDA_VISIBLE_DEVICES')})
    env = dict(context['env'], **{key: value.format(**context['fill'])
                                  for key, value in {**context['config']['environment'], **arm.get('environment', {})}.items()})
    with (directory / 'server.log').open('w') as log:
        process, link = backstopped(wrapper, backstop['kill_after_seconds'], cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        (directory / 'server.pid').write_text(str(process.pid))
        watchdog = None
        try:
            watchdog = backstop_pid(link)
            (directory / 'backstop.pid').write_text(str(watchdog))
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                require(process.poll() is None, 'server exited before health')
                require(time.monotonic() < deadline, 'server startup timed out')
                try:
                    health = http(url + '/health')
                    models = http(url + '/v1/models')
                    require(context['facts']['name'] in [row['id'] for row in models['data']], 'wrong served model')
                    require(listener_owned(port, process.pid), 'health listener is not owned by this run')
                except (OSError, ValueError, KeyError):
                    time.sleep(1)
                    continue
                write(directory / 'health.json', {'health': health, 'models': models, 'process_group': process.pid})
                break
            yield url, process.pid
        finally:
            # The backstop sends the one TERM and arms its KILL from it, so a
            # pause before the request cannot shorten the drain; if this
            # process dies first, the closed socket makes it stop the server.
            with link:
                termination = stop(process, timeout_forwarder=True, term_wait=drain['term_wait_seconds'],
                                   forward=lambda: backstop_term(link))
            armed = termination['stages'][0].get('via') == 'backstop' and termination['stages'][0]['sent']
            write(directory / 'shutdown.json', {'returncode': process.returncode, 'pid': process.pid,
                                                 'termination': {**termination, 'drain': drain,
                                                                 'backstop': {**backstop, 'pid': watchdog,
                                                                              'armed': armed}}})
            require_shutdown(directory)


def require_shutdown(directory):
    """Admit an explicit clean exit of this arm, including legacy exit records."""
    # The script runs from tools/qual; tools.model_sdk resolves from the
    # checkout root, as in common.refcache.
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from tools.model_sdk.diagnostic import read_json
    value = read_json(directory / 'shutdown.json')
    require(type(value.get('returncode')) is int and value['returncode'] == 0,
            f'{directory}: nonclean or unconfirmed server shutdown: {value.get("returncode")}')
    if 'termination' in value:
        stopped = value['termination']
        require(isinstance(stopped, dict) and stopped.get('confirmed') is True
                and stopped.get('forced') is False and type(stopped.get('returncode')) is int
                and stopped['returncode'] == value['returncode'],
                f'{directory}: forced or unconfirmed server termination')
        stages = stopped.get('stages', [])
        require(isinstance(stages, list) and all(isinstance(stage, dict) and not stage.get('group_remaining')
                                                for stage in stages),
                f'{directory}: residual group or invalid server termination stages')
        # An exit after an expired TERM wait is late even when no KILL landed.
        require(all(stage.get('signal') == 'SIGTERM' and stage.get('timed_out') is False for stage in stages),
                f'{directory}: server exit after an expired TERM wait or a KILL stage')
    # A launch or stop that records a drain, a wrapper or a backstop is a
    # drain record and needs all of its evidence. Only when neither does is it
    # legacy (an exit record or a 5 s stop), judged as above.
    launch = read_json(directory / 'launch.json')
    require(isinstance(launch, dict), f'{directory}: invalid server launch record')
    stopped = value.get('termination')
    markers = ('drain', 'wrapper', 'backstop')
    if any(key in launch for key in markers) or (isinstance(stopped, dict) and any(key in stopped for key in markers)):
        require(isinstance(stopped, dict) and 'drain' in stopped and 'drain' in launch,
                f'{directory}: server drain recorded without its launch or termination evidence')
        drain = stopped['drain']
        command = launch.get('command')
        require(isinstance(command, list) and all(isinstance(word, str) for word in command)
                and command[1:2] == ['serve'] and launch['drain'] == drain
                and drain == server_drain(command[2:]),
                f'{directory}: server drain differs from its recorded serve arguments')
        # A wrapper without --foreground, or with a time limit of its own, can
        # deliver a second TERM, which skips the drain and still exits 0.
        wrapper = launch.get('wrapper')
        require(isinstance(wrapper, list) and all(isinstance(word, str) for word in wrapper)
                and Path(wrapper[0] if wrapper else '').name == 'timeout'
                and wrapper[1:4] == ['--foreground', '--signal=TERM',
                                     f'--kill-after={drain["wrapper_kill_after_seconds"]}s']
                and wrapper[4:5] == ['0s'] and wrapper[5:] == command,
                f'{directory}: server wrapper does not forward one TERM in the foreground with its drain KILL delay '
                f'and no time limit')
        # The watchdog is a forked process, never the wrapper (the group
        # leader); these are record checks, valid when rescore replays them.
        backstop = server_backstop(drain)
        armed = stopped.get('backstop')
        health = read_json(directory / 'health.json')
        require(isinstance(launch.get('backstop'), dict) and launch['backstop'] == backstop
                and type(launch['backstop'].get('kill_after_seconds')) is int and isinstance(armed, dict)
                and type(armed.get('kill_after_seconds')) is int
                and type(armed.get('pid')) is int and armed['pid'] > 0
                and armed['pid'] not in (value.get('pid'), health.get('process_group'))
                and armed.get('armed') is True
                and armed == {**backstop, 'pid': armed['pid'], 'armed': True},
                f'{directory}: server stop has no armed cleanup backstop')
        stages = stopped.get('stages')
        require(isinstance(stages, list) and len(stages) == 1 and stages[0].get('signal') == 'SIGTERM'
                and stages[0].get('sent') is True and stages[0].get('timed_out') is False
                and stages[0].get('target') == 'timeout_process' and stages[0].get('via') == 'backstop'
                and type(stages[0].get('wait_seconds')) is int
                and stages[0].get('wait_seconds') == drain['term_wait_seconds'],
                f'{directory}: server stop did not send one TERM and wait its declared drain')
    health = read_json(directory / 'health.json')
    require(type(value.get('pid')) is int and value['pid'] == health.get('process_group'),
            f'{directory}: shutdown does not identify this arm process')
    return value


def shutdown_failures(directory, model, suite, names, inputs, selected=()):
    """Check every planned arm independently of the run's claimed verdict;
    a selected check subset plans only its own arms."""
    facts, configs = select_checks(model_facts(model), selected_configurations(model, names, suite), suite, selected)
    failures = {}
    for name, config in configs.items():
        for arm in plan(facts, suite, name, config, inputs):
            if not arm.get('server', True):
                continue
            label = f'{name}/{arm["arm"]}/server'
            try:
                require_shutdown(directory / name / arm['arm'])
            except (OSError, ValueError, KeyError) as error:
                failures[label] = str(error)
    return failures


@contextmanager
def replayed(context, arm, directory):
    """A recorded arm's server, for rescore: its health as recorded."""
    health = read(directory / 'health.json')
    launch = read(directory / 'launch.json')
    yield launch['url'], health['process_group']
    require_shutdown(directory)


@contextmanager
def serverless(context, arm, directory):
    """The offline steps' place in a configuration: no server, the
    configuration's GPUs free, every record of its servers available."""
    context['fill'].update({**context['records'], 'url': '', 'pid': None})
    yield None, None


# Stage rules ---------------------------------------------------------------

def clean_receipt(value):
    require(isinstance(value, dict), 'receipt must be an object')
    require(value.get('passed') is True, 'receipt did not pass')
    require(value.get('mechanics_only') is False, 'receipt is mechanics-only or lacks mode')
    require(not value.get('skipped') and not value.get('errors') and not value.get('not_run'),
            'receipt has errors, skips or omitted checks')


def judge_gsm8k(value, facts):
    clean_receipt(value)
    rows = value.get('rows', [])
    require(value.get('count') == len(rows) == GSM8K_ITEMS, 'lost GSM8K items')
    require(len({r['index'] for r in rows}) == GSM8K_ITEMS, 'duplicate GSM8K items')
    require(all('error' not in r and r.get('finish_reason') in ('stop', 'length') for r in rows),
            'incomplete GSM8K answer')
    require(all(r.get('actual') is None or isinstance(r['actual'], str) for r in rows)
            and all(r.get('rule') in (None, '####', 'boxed') for r in rows), 'invalid GSM8K answer')
    limited = [r for r in rows if r['finish_reason'] == 'length']
    unparsed = [r for r in rows if r['finish_reason'] == 'stop' and not (r.get('actual') or '').strip()]
    require(all(r.get('correct') is False for r in limited), 'length-limited GSM8K answer counted as correct')
    require(all(r.get('correct') is False for r in unparsed), 'unparsed GSM8K answer counted as correct')
    floor = facts['gsm8k_floor']
    require(value.get('correct') == sum(r['correct'] for r in rows), 'GSM8K count differs from its rows')
    values = [metric('correct', value['correct'], 'ge', floor)]
    require(value['correct'] >= floor, f'GSM8K screen below {floor}/{GSM8K_ITEMS}')
    return values, {'length_limited_indices': sorted(r['index'] for r in limited),
                    'unparsed_answer_indices': sorted(r['index'] for r in unparsed),
                    'boxed_answer_indices': sorted(r['index'] for r in rows if r.get('rule') == 'boxed')}


def judge_needle(value, facts):
    clean_receipt(value)
    rows = value.get('rows', [])
    require(len(rows) == 1 and rows[0].get('context') == facts['context'] and rows[0].get('passed') is True,
            'missing maximum-context needle')
    return [metric('context', rows[0]['context'], 'eq', facts['context'])], {}


def judge_queue(value, facts):
    clean_receipt(value)
    count = queue_clients(facts)
    require(value.get('requests') == count and value.get('concurrency') == count,
            f'queue check must complete all {count} simultaneous clients')
    return [metric('requests', value['requests'], 'eq', count)], {}


def soak_arguments(facts, duration, prefix=False):
    """A soak stage's soak.py rule arguments: its duration, the profile's
    concurrency, the structured mix and the memory limit (with `prefix`, the
    previous preview's speculative soak, also the prefix-cache mix)."""
    return ['--duration', str(duration), '--concurrency', str(facts['max_sequences']), '--structured-mix',
            '--memory-growth-mib', str(SOAK_GROWTH_MIB), *(['--prefix-mix'] if prefix else []),
            *soak_rule_arguments(facts)]


def judge_soak(value, facts, arguments):
    """A soak stage, judged by the soak kind with the stage's own arguments
    and the model's declared warm-up and cache allowance: the measured
    interval and its samples, the concurrency, device and host memory growth
    and the mixes."""
    clean_receipt(value)
    check = {'module': 'soak', 'args': list(map(str, arguments)), 'constants': {'soak': facts['soak_constants']}}
    metrics = kinds.KINDS['soak'].score(value, None, check)
    # The limit each growth is judged against (soak.memory_rule): growth
    # after the warm-up the steady limit, growth over the whole soak that
    # limit plus the memory's declared cache allowance.
    allowance = facts['soak_allowance']
    limits = {'memory_growth_mib': SOAK_GROWTH_MIB + allowance['device'],
              'host_memory_growth_mib': SOAK_GROWTH_MIB + allowance['host'],
              'steady_memory_growth_mib': SOAK_GROWTH_MIB, 'steady_host_memory_growth_mib': SOAK_GROWTH_MIB}
    values = [metric(name, metrics[name], 'le', limit) for name, limit in limits.items()
              if metrics.get(name) is not None]
    if metrics['passed'] is not True:
        prefix = value.get('prefix_mix') or {}
        if '--prefix-mix' in arguments and not (prefix.get('canary_passed') is True and prefix.get('hits_total', 0) > 0
                                                and prefix.get('idle_blocks_used') == 0
                                                and prefix.get('idle_active_sequences') == 0):
            raise ValueError('missing or failed prefix-cache soak coverage')
        require((value.get('mixed_requests') or {}).get('structured_truncated') == 0, 'truncated structured requests')
        raise ValueError(f'memory grew more than {SOAK_GROWTH_MIB} MiB after the declared warm-up, or beyond the '
                         'declared cache allowance over the soak')
    return values, {}


def judge_greedy(value, facts, arm, exact_sha256):
    require(value.get('k') == (0 if arm == 'ar' else facts['depth']), 'wrong greedy speculation depth')
    require(value.get('passed') is True and value.get('mechanics_only') is False, 'greedy capture did not pass')
    require(value.get('concurrency') == 1 and value.get('requests'), 'missing C1 greedy requests')
    require(all(ids and all(type(t) is int and t >= 0 for t in ids) for ids in value['requests'].values()),
            'empty or invalid greedy tokens')
    require(set(value['requests']) == set(value.get('finish_reasons', {})), 'missing greedy finishes')
    require(all(v in ('stop', 'length') for v in value['finish_reasons'].values()), 'bad greedy finish')
    require(value.get('server', {}).get('model') == facts['name'], 'greedy capture model mismatch')
    require(value.get('reference_sha256') == exact_sha256, 'greedy exact inputs changed')
    require(value.get('prompt_tokens') == GREEDY_TOKENS and value.get('max_tokens') == GREEDY_TOKENS,
            'greedy capture changed its workload')
    return [metric('requests', len(value['requests']), None, None)], {}


def judge_spec_capture(value, facts, arm, run_id):
    clean_receipt(value)
    require(value.get('arm') == f'chopinfer-{arm}' and value.get('run_id') == run_id, 'wrong spec_verify arm or run')
    require(value.get('server', {}).get('model') == facts['name'], 'wrong spec_verify model')
    require(value.get('output_tokens') == 64 and value.get('sampling') == {'temperature': 0, 'top_p': 1, 'logprobs': 5},
            'wrong spec_verify workload')
    require(set(value.get('sets', {})) == {'copy', 'natural'}, 'missing spec_verify prompt sets')
    for group in value['sets'].values():
        require(group.get('prompts_sha256') and len(group.get('requests', [])) == 16, 'incomplete spec_verify prompt set')
        for row in group['requests']:
            require(row.get('finish_reason') in ('stop', 'length') and 0 < len(row.get('tokens', [])) <= 64
                    and len(row['tokens']) == len(row.get('logprobs', [])) == len(row.get('top', [])),
                    'incomplete spec_verify row')
            require(all(isinstance(x, (float, int)) and math.isfinite(x) for x in row['logprobs'])
                    and all(top and all(isinstance(x, (float, int)) and math.isfinite(x) for x in top.values())
                            for top in row['top']), 'invalid spec_verify logprobs')
        counters = group.get('speculation') or {}
        if arm == 'spec':
            require(counters.get('available') is True, 'missing spec_verify speculation counters')
            require(counters.get('verify_items', 0) > 0, 'spec_verify arm did not exercise the expected execution path')
        else:
            # AR may omit speculation counters; when present they must agree
            # with the step histogram's proof of decode-only work.
            if counters.get('available') is True:
                require(counters.get('verify_items') == 0, 'spec_verify AR counters must report zero verify items')
            step_rows = counters.get('step_rows') or {}
            require(step_rows.get('available') is True, 'missing spec_verify AR step-row counters')
            bins = step_rows.get('bins') or []
            require(step_rows.get('verify_steps') == 0 and all(row.get('verify_rows') == 0 for row in bins)
                    and any(row.get('decode_rows', 0) > 0 and row.get('steps', 0) > 0 for row in bins),
                    'spec_verify AR arm did not exercise the expected execution path')
    return [], {}


def judge_determinism_capture(value, facts, arm, run_id, image):
    clean_receipt(value)
    require(value.get('arm') == arm and value.get('run_id') == run_id, 'wrong determinism arm or run')
    require(value.get('server', {}).get('model') == facts['name'], 'wrong determinism model')
    # The determinism kind's coverage of a record, against itself.
    kinds.KINDS['determinism'].verify(value, value)
    require(value['image']['included'] == image, 'determinism image request differs from what the server serves')
    return [metric('requests', len(value['requests']), None, None)], {}


def compare_determinism(left, right):
    """Bit-identical C1 answers of two arms, of the two cold long-prompt sends
    within each, and of each prefix-hit send and its record's cold token-ID
    send."""
    from determinism import describe
    kind = kinds.KINDS['determinism']
    comparison = kind.score(left, right)
    status, _ = kind.judge(comparison, {})
    return comparison, None if status == 'pass' else describe(comparison)


def compare_greedy(reference, value, rule):
    """The greedy_pin kind's comparison; `rule` exact refuses any difference."""
    kind = kinds.KINDS['greedy_pin']
    comparison = kind.score(value, reference)
    if rule == 'exact':
        status, _ = kind.judge(comparison, {'rule': 'exact'})
        require(status == 'pass',
                f"greedy token difference in {len(comparison['first_differences'])} of {len(reference['requests'])} "
                f"requests; finish reasons changed in {len(comparison['finish_differences'])}")
    return comparison


def compare_spec(ar, spec):
    from spec_verify import compare_records
    try:
        return compare_records(spec, ar)
    except SystemExit as error:
        raise ValueError(str(error)) from error


def metric(name, value, op, limit):
    return {'metric': name, 'value': value, 'op': op, 'limit': limit}


# Running -------------------------------------------------------------------

class Run:
    """One model's run: its directory, install, results and stored baselines."""

    def __init__(self, args, facts, configs, identity, install, directory):
        self.args, self.facts, self.configs, self.identity = args, facts, configs, identity
        self.install, self.directory = install, directory
        # A rescore replays a recorded run: its children's exits and receipts
        # are read, not produced, and what it judges anew goes to `out`.
        self.replay, self.out = False, directory
        self.install_source = None
        self.installs = {(): install}
        self.install_metadata = {}
        self.used_features = set()
        self.rows = []
        self.started = time.monotonic()
        self.inputs = {}
        for entry in args.input:
            key, separator, path = entry.partition('=')
            require(separator and key and Path(path).exists(), f'--input takes KEY=PATH of an existing path: {entry}')
            self.inputs[key] = str(Path(path).resolve())
        # An input that is a value, not a file (such as the run ID of another
        # run's check): given to the checks as written, recorded apart from
        # the paths.
        self.input_values = {}
        for entry in getattr(args, 'input_value', []):
            key, separator, value = entry.partition('=')
            require(separator and re.fullmatch(r'\w+', key) and value,
                    f'--input-value takes KEY=VALUE with a non-empty value: {entry}')
            require(key not in self.inputs and key not in self.input_values,
                    f'--input-value {key} is given twice')
            self.input_values[key] = value
        self.inputs.update(self.input_values)
        self.stores = {name: Store(name, facts['model']) for name in STORES}

    def row(self, config, arm, name, status, **fields):
        row = {'id': f'{config}/{arm}/{name}' if arm else f'{config}/{name}', 'status': status, **fields}
        for value in row.get('values', []):
            for store in STORES:
                value[store] = self.stores[store].value(config, row['id'], value['metric'])
        self.rows.append(row)
        print(line(row), flush=True)
        return row


def line(row):
    """One printed result: the status, then each value against its limit and
    the stored last good and seed values."""
    parts = [f"{row['id']} {row['status'].upper()}"]
    for value in row.get('values', []):
        text = f"{value['metric']} {value['value']!r}"
        if value.get('op'):
            text += f" {({'le': '<=', 'ge': '>=', 'eq': '=='}).get(value['op'], value['op'])} {value['limit']!r}"
        stored = [f"{store} {value.get(store)!r}" for store in STORES if value.get(store) is not None]
        if stored:
            text += f" ({', '.join(stored)})"
        parts.append(text)
    if row.get('error'):
        parts.append(f"error: {row['error']}")
    if row.get('reason'):
        parts.append(row['reason'])
    return '; '.join(parts)


def predicate_values(receipt, predicates):
    """A judged check's values: each predicate on a number, with the
    receipt's value at its path, and a kind's computed limits. A gate list
    (kv_offload's) adds its gates; other kinds use `gates` for other shapes,
    such as spec_verify's copy bounds object, which are not listed."""
    gates = (receipt or {}).get('gates')
    values = [metric(gate['name'], gate['value'], gate['op'], gate['limit'])
              for gate in gates] if isinstance(gates, list) else []
    for rule in predicates or []:
        if isinstance(rule.get('value'), (int, float)) and not isinstance(rule['value'], bool):
            measured = kinds.select(receipt, rule['path']) if receipt is not None else None
            if isinstance(measured, list) and len(measured) == 1:
                measured = measured[0]
            values.append(metric('.'.join(map(str, rule['path'])), measured, rule['op'], rule['value']))
    limits = (receipt or {}).get('limits')
    if isinstance(limits, dict):
        for key, limit in limits.items():
            if isinstance(limit, (int, float)) and key in receipt:
                values.append(metric(key, receipt[key], 'le', limit))
    return values


def fill_arguments(arguments, fill):
    result = []
    for word in arguments:
        if word == '{image}':
            result += fill['image']
        else:
            result.append(word.format(**fill))
    return result


def run_child(run, context, directory, script, arguments, seconds, config_path):
    """One tools/qual child against the live server: its receipts in
    `directory`, scored by the install's chopinfer eval."""
    directory.mkdir(parents=True, exist_ok=True)
    run.used_features.add(())
    env = dict(context['env'], CHOPINFER_EVAL=str(run.install / 'bin/chopinfer'),
               CHOPINFER_QUAL_CONFIG=str(config_path), CHOPINFER_QUAL_RECEIPTS=str(directory))
    command = [sys.executable, '-B', str(ROOT / script if '/' in script else TOOLS / script),
               *fill_arguments(arguments, context['fill'])]
    return run_call(run, command, directory / 'check.log', seconds, env)


def eval_capture(run, context, directory, kind, arguments, seconds):
    """One record of the shared kind `kind` against the live server, taken by
    the install's `chopinfer eval capture` under the configuration's
    declaration, bound to the arm's startup log: its receipt in
    `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    run.used_features.add(())
    fill, config = context['fill'], context['config_name']
    command = [run.install / 'bin/chopinfer', 'eval', 'capture', '--kind', kind, '--model',
               ROOT / 'models' / run.facts['model'], *([] if config == 'default' else ['--variant', config]),
               '--served-model', fill['name'], '--url', fill['url'], '--server-log', fill['server_log'],
               '--timeout', '120', '--report', directory, *fill_arguments(arguments, fill)]
    return run_call(run, command, directory / 'check.log', seconds, context['env'])


def run_call(run, command, log, seconds, env):
    """A child of the run: its exit status, also kept beside its log. A
    rescore reads the kept status instead (None when none was kept)."""
    status = Path(f'{log}.exit')
    if run.replay:
        return int(status.read_text()) if status.is_file() else None
    code = call(command, log, seconds, env)
    status.write_text(f'{code}\n')
    return code


def one_receipt(directory, pattern='*.json'):
    candidates = [path for path in directory.glob(pattern) if pattern != '*.json' or path.name != 'check.json']
    require(len(candidates) == 1, f'missing or ambiguous receipt in {directory}')
    return candidates[0]


def run_record(run, context, step, arm_dir):
    """One record: its receipt (a producer's: its run directory) is the
    input its checks read."""
    directory = arm_dir / step['id']
    started = time.monotonic()
    seconds = (600 if step['script'] == 'kv_offload.py' else
               STAGE_TIMEOUTS['prefix' if step['script'] == 'prefix_exact.py' else 'record'])
    produced = step.get('produces', False)
    context['fill']['run_dir'] = str(directory / 'run')
    try:
        code = run_child(run, context, directory, step['script'], step['arguments'], seconds, context['declaration'])
    except KeyError as error:
        # An offline record names a record of this run that was not taken.
        code = f'not run: no {error.args[0]}'
    if code != 0:
        receipt = None
    elif produced:
        receipt = directory / 'run' if (directory / 'run').is_dir() else None
        code = code if receipt else 1
    else:
        receipt = one_receipt(directory)
    if receipt is not None:
        context['records'][step['input']] = str(receipt)
    return {'id': step['id'], 'exit': code, 'receipt': str(receipt) if receipt else None,
            'seconds': round(time.monotonic() - started, 1)}


def run_stage(run, context, step, arm_dir, arm):
    """One of the runner's screens on the live server, judged at once (the
    comparisons between arms wait for the end of the configuration)."""
    facts, fill = run.facts, context['fill']
    stage = step['stage']
    directory = arm_dir / stage
    started = time.monotonic()
    config = context['config_name']
    try:
        if stage == 'determinism':
            code = eval_capture(run, context, directory, 'determinism',
                                ['--arm', arm, '--run-id', '{run_id}', '{image}'], STAGE_TIMEOUTS['record'])
            require(code == 0, f'determinism record exited {code}')
            path = one_receipt(directory)
            values, extra = judge_determinism_capture(read(path), facts, arm, fill['run_id'],
                                                      fill['image'] == ['--image'])
            context['screens'][f'determinism-{arm}'] = str(path)
            if f'determinism_{arm}' in STAGE_INPUTS:
                context['records'][f'determinism_{arm}'] = str(path)
        elif stage == 'gsm8k':
            arguments = [*SHARED, '--limit', str(GSM8K_ITEMS), '--concurrency', '8', '--response-dir',
                         str(directory / 'responses')]
            code = run_child(run, context, directory, 'gsm8k.py', arguments, STAGE_TIMEOUTS['gsm8k'],
                             context['declaration'])
            require(code == 0, f'gsm8k exited {code}')
            path = one_receipt(directory)
            values, extra = judge_gsm8k(read(path), facts)
            context['screens'][f'gsm8k-{arm}'] = str(path)
        elif stage == 'greedy':
            depth = 0 if arm == 'ar' else facts['depth']
            arguments = ['--arm', arm, '--cold-cache', '--concurrency', '1', '--k', str(depth), '--prompt-tokens',
                         str(GREEDY_TOKENS), '--max-tokens', str(GREEDY_TOKENS)]
            code = eval_capture(run, context, directory, 'greedy_pin', arguments, STAGE_TIMEOUTS['greedy'])
            require(code == 0, f'greedy record exited {code}')
            path = one_receipt(directory)
            values, extra = judge_greedy(read(path), facts, arm, context['exact_sha256'])
            context['screens'][f'greedy-{arm}'] = str(path)
        elif stage == 'needle':
            arguments = ['--url', '{url}', '--model', '{name}', '--timeout', str(STAGE_TIMEOUTS['needle']),
                         '--lengths', str(facts['context'])]
            code = run_child(run, context, directory, 'needle.py', arguments, STAGE_TIMEOUTS['needle'],
                             context['declaration'])
            require(code == 0, f'needle exited {code}')
            path = one_receipt(directory)
            values, extra = judge_needle(read(path), facts)
        elif stage == 'queue':
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'queue.json'
            if not run.replay:
                write(path, queue(fill['url'], facts))
            values, extra = judge_queue(read(path), facts)
        elif stage == 'soak':
            rule = soak_arguments(facts, AR_SOAK_SECONDS)
            code = run_child(run, context, directory, 'soak.py', [*SHARED, *rule, '--server-pid', '{pid}'],
                             AR_SOAK_SECONDS + 600, context['declaration'])
            require(code == 0, f'soak exited {code}')
            path = one_receipt(directory)
            values, extra = judge_soak(read(path), facts, rule)
        elif stage == 'prefix_slice':
            arguments = [*RECORDS['prefix_record'][2], '--shard', f'1/{QUICK_PREFIX_SHARDS}']
            code = run_child(run, context, directory, 'prefix_exact.py', arguments, STAGE_TIMEOUTS['prefix_slice'],
                             context['declaration'])
            require(code == 0, f'prefix_exact slice record exited {code}')
            record = one_receipt(directory)
            path, values, extra = prefix_slice_report(run, context, record, directory)
        elif stage == 'gsm8k_branches':
            ar, spec = (context['screens'].get(f'gsm8k-{name}') for name in ('ar', 'spec'))
            if not (ar and spec):
                return run.row(config, arm, stage, 'not_run', error='a GSM8K arm did not pass')
            arguments = [*SHARED, '--ar', ar, '--spec', spec]
            code = run_child(run, context, directory, 'gsm8k_branches.py', arguments, STAGE_TIMEOUTS['gsm8k_branches'],
                             context['declaration'])
            require(code == 0, f'gsm8k_branches exited {code}')
            path = one_receipt(directory)
            probe = read(path)
            clean_receipt(probe)
            require(probe['ar']['sha256'] == sha(ar) and probe['spec']['sha256'] == sha(spec),
                    'the branch probe read other GSM8K receipts than the arms recorded')
            context['screens']['gsm8k_branches'] = str(path)
            values, extra = [], {}
        else:
            raise ValueError(f'unknown stage {stage}')
        return run.row(config, arm, stage_row(stage), 'pass', values=values, receipt=str(path), sha256=sha(path),
                       seconds=round(time.monotonic() - started, 1), **extra)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        return run.row(config, arm, stage_row(stage), 'fail', error=str(error),
                       seconds=round(time.monotonic() - started, 1))


def stage_row(stage):
    """A stage's row name; the determinism record is named apart from the
    declared determinism check, which runs on the same server."""
    return 'determinism-record' if stage == 'determinism' else stage


def prefix_slice_report(run, context, record, directory):
    """quick's prefix exactness: one slice's report (prefix_exact report
    --slice), every position of the slice exact."""
    run.used_features.add(())
    env = dict(context['env'], CHOPINFER_EVAL=str(run.install / 'bin/chopinfer'),
               CHOPINFER_QUAL_CONFIG=str(context['declaration']))
    directory = context['out'] / directory.relative_to(context['directory']) / 'report'
    log = directory / 'report.log'
    code = call([sys.executable, '-B', str(TOOLS / 'prefix_exact.py'), 'report', '--slice', '--record', str(record)],
                log, STAGE_TIMEOUTS['prefix_slice'], env)
    value = last_json(log)
    output = directory / 'report.json'
    write(output, value)
    require(code == 0 and value.get('passed') is True, 'prefix_exact slice is not exact')
    return output, [metric('passed', value['passed'], 'eq', True)], {'slice': value.get('slice')}


def queue(url, facts):
    """2 x cap + 1 simultaneous clients, every one answered, none refused."""
    import threading
    count = queue_clients(facts)
    barrier = threading.Barrier(count)

    def one(index):
        barrier.wait(timeout=15)
        response = http(url + '/v1/completions', {
            'model': facts['name'], 'prompt': f'Request {index}. List the integers from 1 to 500:',
            'temperature': 0, 'max_tokens': 128}, 110)
        require(response['choices'][0]['finish_reason'] in ('stop', 'length'), 'invalid queue finish')
        require(response['usage']['completion_tokens'] > 0, 'empty queue answer')
        return index

    with ThreadPoolExecutor(max_workers=count) as pool:
        rows = list(pool.map(one, range(count)))
    return {'passed': rows == list(range(count)), 'mechanics_only': False, 'requests': len(rows),
            'concurrency': count, 'retries': 0, 'errors': []}


def soak_rule_arguments(facts):
    """The model's declared soak warm-up and cache allowance, passed only when
    declared."""
    if not facts['soak_warmup'] and not any(facts['soak_allowance'].values()):
        return []
    return ['--warmup-seconds', str(facts['soak_warmup']), '--cache-allowance-mib',
            f"{facts['soak_allowance']['host']},{facts['soak_allowance']['device']}"]


def run_check(run, context, step, arm_dir, arm):
    """One declared check through checks.py, as the 2026-10-11 final release gate ran each: its inputs
    are this configuration's records, the arm's server log and --input
    paths, and earlier passing checks' receipts. A regression check (one
    reading stored_<check> inputs) runs once per store, against that store's
    receipts, as `<check>/<store>`; a store without them reports no baseline
    yet, as information."""
    check = step['check']
    declared = context['facts']['checks'][check]
    config = context['config_name']
    # A new attempt of a check withdraws what an earlier one published (in
    # another serving arm, or an earlier try): later checks read only what
    # this attempt publishes once it passes.
    withdraw(context, check)
    inputs = {**run.inputs, **context['records'], LOG_INPUT: str(arm_dir / 'server.log')}
    named = placeholders(declared, context['suite'])
    stored = sorted(name for name in named if name.startswith(STORED))
    known = {*inputs, *context['checks'], *context['facts']['checks'], *stored, 'url', 'model', 'raw', 'run', 'run_id'}
    missing = sorted(name for name in named if name not in known)
    if missing and not declared.get('informational_reason'):
        return run.row(config, arm, check, 'not_run', error=f"no {', '.join(missing)}: give it with --input "
                       'or --input-value, or run the full suite')
    if not stored:
        return judged_check(run, context, check, declared, arm_dir, arm, check, inputs)
    for store in STORES:
        label = f'{check}/{store}'
        receipts = {name: run.stores[store].receipt(config, f'{config}/{arm}/{name[len(STORED):]}')
                    for name in stored}
        recorded = (arm_dir / 'checks' / label.replace('/', '-') / 'inputs.json').is_file()
        if None in receipts.values() and not (run.replay and recorded):
            run.row(config, arm, label, 'informational',
                    reason=f"no baseline yet: the {store} store holds no "
                           f"{', '.join(name[len(STORED):] for name in stored)} receipt of this arm")
            continue
        judged_check(run, context, check, declared, arm_dir, arm, label,
                     {**inputs, **{name: str(path) for name, path in receipts.items() if path}})


def withdraw(context, check):
    """Forget the receipt and raw directory an earlier attempt of `check`
    published, so a later check cannot read them after this attempt fails
    or is refused."""
    context['checks'].pop(check, None)
    context.setdefault('check_raw', {}).pop(check, None)


def judged_check(run, context, check, declared, arm_dir, arm, label, inputs):
    """`check` run once through checks.py with `inputs`, as the row `label`;
    only a passing run of `check` itself publishes its receipt and raw
    directory to later checks."""
    config = context['config_name']
    if label == check:
        withdraw(context, check)
    directory = arm_dir / 'checks' / label.replace('/', '-')
    directory.mkdir(parents=True, exist_ok=True)
    if not run.replay:
        write(directory / 'inputs.json', inputs)
    where = ['--offline'] if arm == OFFLINE else ['--url', context['fill']['url']]
    command = [sys.executable, '-B', str(TOOLS / 'checks.py'), 'check', check, '--model',
               declaration_name(run.facts['model'], config), '--cohort', context['suite'], *where,
               '--served-model', context['facts']['name'], '--inputs', str(directory / 'inputs.json'),
               '--receipts', str(directory)]
    for name, path in context['checks'].items():
        command += ['--with', f'{name}={path}']
    for name, path in context.setdefault('check_raw', {}).items():
        command += ['--with-raw', f'{name}={path}']
    run.used_features.add(())
    env = dict(context['env'], CHOPINFER_EVAL=str(run.install / 'bin/chopinfer'))
    started = time.monotonic()
    code = run_call(run, command, directory / 'checks.log', declared['timeout_seconds'] + 300, env)
    try:
        check_run = one_receipt(directory, '*/check.json')
        status, row, receipt = check_verdict(code, read(check_run), check, declared)
        rescored = {}
        if run.replay and (status in ('pass', 'waived', 'not_applicable')
                           or (status == 'fail' and row.get('error') == 'threshold predicate failed')):
            # Only a judged verdict is judged again; a failed run stays failed.
            rescored = rescore_check(row['receipt'], run.facts['model'], config, check)
            status = rescored['status']
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        return run.row(config, arm, label, 'fail', error=str(error), exit=code,
                       seconds=round(time.monotonic() - started, 1))
    if status in ('pass', 'waived') and label == check:
        context['checks'][check] = row['receipt']
        # Each check is its own checks.py run with its own {raw}: a later
        # check names this one's files by {check_raw[<check>]}.
        context['check_raw'][check] = str(check_run.parent / 'checks')
    return run.row(config, arm, label, status, values=predicate_values(receipt, row.get('predicates')),
                   receipt=row.get('receipt'), check_run=str(check_run), exit=code,
                   error=rescored.get('error') or row.get('error') or (
                       None if status in PASSING else 'the check did not pass'),
                   differing_metrics=rescored.get('differing_metrics'), seconds=round(time.monotonic() - started, 1))


def check_verdict(code, result, check, declared):
    """A checks.py check run's status, row and receipt, trusted only when the
    process and its check.json agree: exit 0 with a passing status, exit 1
    with a failing one (a timeout or any other exit fails), the row of this
    check, and for a pass or a not-applicable verdict a readable receipt. An
    informational row holds only for a check declared informational."""
    require(code in (0, 1), f'checks.py exited {code}')
    status = result.get('status')
    require(result.get('check') == check, f'check.json is for {result.get("check")!r}, not {check}')
    require((code == 0) == (status in PASSING), f'checks.py exited {code} with status {status!r}')
    row = result.get('row')
    # A check refused before it ran (such as a missing earlier receipt)
    # leaves no row; its error is why.
    require(isinstance(row, dict) and row.get('id') == check,
            f'check.json holds no row for {check}' + (f": {result['error']}" if result.get('error') else ''))
    receipt = None
    if status == 'informational':
        require(row.get('status') == 'informational' and declared.get('informational_reason'),
                f'{check} is not declared informational')
    elif status in ('pass', 'waived', 'not_applicable'):
        require(row.get('receipt'), f'{check} has no receipt')
        receipt = read(row['receipt'])
        require(isinstance(receipt, dict), f'{check}: the receipt is not a JSON object')
        if status in ('pass', 'waived'):
            require(row.get('passed') is True, f'{check}: the row did not pass')
        else:
            require(row.get('status') == 'not_applicable' and row.get('reason'),
                    f'{check}: a not-applicable verdict without its reason')
    elif row.get('receipt') and Path(row['receipt']).is_file():
        receipt = read(row['receipt'])
    return status, row, receipt


def compare_screens(run, context):
    """The screens between arms: determinism spec against a fresh start with
    1024-token chunks and against AR (unless the declared determinism check
    ran), C1 greedy AR against speculative under the declared rule, the
    spec_verify copy bounds, and paired GSM8K."""
    config, screens = context['config_name'], context['screens']
    comparisons = [('spec', 'spec-repeat')]
    if 'determinism' not in context['ran_checks']:
        comparisons.append(('spec', 'ar'))
    for left, right in comparisons:
        name = f'determinism-{left}-vs-{right}'
        try:
            paths = [screens[f'determinism-{arm}'] for arm in (left, right)]
            comparison, failure = compare_determinism(*map(read, paths))
            path = context['out'] / f'{name}.json'
            write(path, comparison)
            require(failure is None, f'served output is not deterministic: {failure}')
            run.row(config, None, name, 'pass', receipt=str(path), values=[
                metric('requests', comparison.get('requests'), None, None)])
        except KeyError:
            run.row(config, None, name, 'not_run', error='a determinism record did not pass')
        except (ValueError, TypeError) as error:
            run.row(config, None, name, 'fail', error=str(error))
    rule = run.facts['spec_greedy']
    try:
        ar, spec = (read(screens[f'greedy-{arm}']) for arm in ('ar', 'spec'))
        comparison = compare_greedy(ar, spec, rule)
        path = context['out'] / 'greedy-ar-vs-spec.json'
        write(path, comparison)
        status = 'pass' if rule == 'exact' else 'informational'
        run.row(config, None, 'greedy-ar-vs-spec', status, receipt=str(path), rule=rule,
                first_differences=comparison['first_differences'], finish_differences=comparison['finish_differences'],
                values=[metric('requests_differing', len({d['request'] for d in comparison['first_differences']
                                                         + comparison['finish_differences']}),
                               'eq' if rule == 'exact' else None, 0 if rule == 'exact' else None)])
    except KeyError:
        run.row(config, None, 'greedy-ar-vs-spec', 'not_run', error='a greedy capture did not pass')
    except (ValueError, TypeError) as error:
        run.row(config, None, 'greedy-ar-vs-spec', 'fail', error=str(error))
    if 'spec_verify' not in context['ran_checks']:
        # Without the declared spec_verify check, its copy bounds run here.
        try:
            ar, spec = (read(context['records'][key]) for key in ('spec_ar', 'spec_candidate'))
            judge_spec_capture(ar, run.facts, 'ar', context['fill']['run_id'])
            judge_spec_capture(spec, run.facts, 'spec', context['fill']['run_id'])
            comparison = compare_spec(ar, spec)
            path = context['out'] / 'spec-verify.json'
            write(path, comparison)
            require(comparison['passed'], 'spec_verify numerical bounds exceeded')
            run.row(config, None, 'spec-verify', 'pass', receipt=str(path), gates=comparison.get('gates'))
        except KeyError:
            run.row(config, None, 'spec-verify', 'not_run', error='a spec_verify record did not pass')
        except (ValueError, TypeError) as error:
            run.row(config, None, 'spec-verify', 'fail', error=str(error))
    try:
        from gsm8k_branches import pair, verdict
        ar, spec = (read(screens[f'gsm8k-{arm}']) for arm in ('ar', 'spec'))
        probe = read(screens['gsm8k_branches']) if 'gsm8k_branches' in screens else None
        comparison = pair(ar, spec, probe, run.facts['gsm8k_floor'])
        path = context['out'] / 'gsm8k-paired.json'
        write(path, comparison)
        run.row(config, None, 'gsm8k-paired', 'pass', receipt=str(path), lost=comparison['lost'],
                gained=comparison['gained'], branches=[dict(item, verdict=verdict(item))
                                                       for item in (probe or {}).get('items', [])])
    except KeyError:
        run.row(config, None, 'gsm8k-paired', 'not_run', error='a GSM8K arm did not pass')
    except (ValueError, TypeError) as error:
        run.row(config, None, 'gsm8k-paired', 'fail', error=str(error))


def run_configuration(run, name, config, devices=None):
    facts = run.facts
    directory = run.directory / name
    package = run.packages[f"{facts['model']}/{name}"]
    checkpoint = Path(package['checkpoint'])
    marker = None if run.replay else verify_marker(facts['model'], checkpoint, config['manifest'])
    require(run.replay or marker is not None, f'{checkpoint} is not verified in its current state: run '
            f"`release_preview.sh prepare --model {facts['model']} --config {name}` first")
    declaration_path = declaration.named(declaration_name(facts['model'], name))
    from common import input_path
    loaded = declaration.load(declaration_path)
    exact_path = input_path('exact', config=loaded)
    selected_facts = configuration_facts(facts, name)
    external = {}
    for check in selected_facts['quick' if run.args.suite == 'quick' else 'full']:
        for key in module_option(selected_facts['checks'][check]['module'], 'RUNNER_EXTERNAL_INPUTS', [], facts['model']):
            if key not in run.inputs:
                path = input_path(key, optional=True, config=loaded)
                if path is not None:
                    external[key] = str(path)
    env = environment(gpu=True, config=loaded)
    if devices:
        env['CUDA_VISIBLE_DEVICES'] = devices
    reference_tokens = directory / 'reference-tokens.json'
    out = run.out / name
    out.mkdir(parents=True, exist_ok=True)
    if not run.replay:
        checked([sys.executable, '-B', str(TOOLS / 'verify_exact.py'), 'references', '--out', reference_tokens],
                directory / 'references.log', STAGE_TIMEOUTS['references'],
                dict(env, CHOPINFER_QUAL_CONFIG=str(declaration_path)))
    context = {'config_name': name, 'config': config, 'declaration': declaration_path, 'directory': directory,
               'out': out,
               'install': run.installs.get(tuple(sorted(config['features'])), run.install), 'checkpoint': checkpoint,
               'facts': selected_facts, 'env': env,
               'suite': run.args.suite,
               'records': external, 'screens': {}, 'checks': {}, 'check_raw': {}, 'ran_checks': [],
               'exact_sha256': sha(exact_path), 'marker': marker,
               'fill': {**external, **run.inputs, 'run_id': f'{run.directory.name}-{name}', 'reference_tokens': str(reference_tokens),
                        'name': selected_facts['name'], 'depth': config['depth'], 'checkpoint': str(checkpoint)}}
    if selected_facts.get('kv_offload') and any(selected_facts['checks'][check]['module'] == 'kv_offload'
                                                for check in selected_facts[run.args.suite]):
        from kv_offload import configuration_option, pressure_pages
        layout = read(Path(package['bundle']) / 'manifest.json')['kv_layout']
        policy = selected_facts['kv_offload'] | config.get('kv_offload', {})
        context['fill']['kv_offload_pressure_pages'] = pressure_pages(
            policy, layout['page_tokens'], configuration_option(config, '--context-parallel-size',
                                                              config.get('context_parallel_size', 1)))
    for arm in plan(facts, run.args.suite, name, config, {**external, **run.inputs}):
        features = tuple(sorted(arm.get('features', config['features'])))
        context['install'] = run.installs[features]
        run.used_features.add(features)
        arm_dir = directory / arm['arm']
        arm_dir.mkdir(parents=True, exist_ok=True)
        try:
            require(run.replay or verify_marker(facts['model'], checkpoint, config['manifest']) == marker,
                    'checkpoint files or verification manifest changed since their verify')
            context['fill']['taps_dir'] = str(arm_dir / 'taps')
            served = arm.get('server', True)
            with (serverless if not served else replayed if run.replay else server)(context, arm, arm_dir) as (url, pid):
                if served:
                    context['fill'].update(url=url, pid=pid, server_log=str(arm_dir / 'server.log'),
                                           image=launched_image(package, arm_dir))
                    run.row(name, arm['arm'], 'health', 'pass', receipt=str(arm_dir / 'health.json'))
                for step in arm['steps']:
                    if step['kind'] == 'record':
                        result = run_record(run, context, step, arm_dir)
                        if result['exit'] != 0:
                            run.row(name, arm['arm'], f"record-{step['id']}", 'fail',
                                    error=f"record exited {result['exit']}", log=str(arm_dir / step['id'] / 'check.log'))
                    elif step['kind'] == 'stage':
                        run_stage(run, context, step, arm_dir, arm['arm'])
                    else:
                        context['ran_checks'].append(step['check'])
                        run_check(run, context, step, arm_dir, arm['arm'])
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            run.row(name, arm['arm'], 'server', 'fail', error=str(error))
    if config.get('screens', True):
        compare_screens(run, context)


def device_pairs():
    """The two GPU pairs of this job's four GPUs (CUDA_VISIBLE_DEVICES order)."""
    devices = [device for device in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',') if device]
    require(len(devices) >= 4, f'a run needs four GPUs, not {devices}')
    return [','.join(devices[:2]), ','.join(devices[2:4])]


def run_configurations(run, configs):
    """Each configuration once: those that need all four GPUs in turn, then
    those at two or fewer GPUs two at a time, each on its own GPU pair."""
    def one(name, config, devices=None):
        try:
            run_configuration(run, name, config, devices)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            run.row(name, None, 'configuration', 'fail', error=str(error))

    wide = [(name, config) for name, config in configs.items() if config.get('tensor_parallel_size', 4) > 2]
    narrow = [(name, config) for name, config in configs.items() if config.get('tensor_parallel_size', 4) <= 2]
    for name, config in wide:
        one(name, config)
    if narrow:
        import queue as queues
        pairs = queues.Queue()
        for pair in device_pairs():
            pairs.put(pair)

        def paired(item):
            pair = pairs.get()
            try:
                one(*item, devices=pair)
            finally:
                pairs.put(pair)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(paired, narrow))


def run_command(args):
    facts = model_facts(args.model)
    configs = selected_configurations(args.model, args.config, args.suite)
    facts, configs = select_checks(facts, configs, args.suite, getattr(args, 'check', []))
    identity = source()
    install = Path(args.install).resolve() if args.install else install_root(identity)
    require((install / 'install.json').is_file(), f'no install at {install}: run `release_preview.sh prepare '
            f'--model {args.model}` first')
    state = read(install / 'install.json')
    directory = Path(args.out).resolve() if args.out else (
        VALIDATION / 'runs' / args.model / f"{utc()}-{identity['commit'][:12]}-{args.suite}")
    require(not directory.exists(), f'{directory} exists')
    # The GPUs this run measures on are part of its record: a query that
    # cannot run (no nvidia-smi), fails, times out or answers nothing refuses
    # before any server starts.
    require(shutil.which('nvidia-smi'), 'the GPU query cannot run: nvidia-smi is not on PATH; no server was started')
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                '--format=csv'], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError('the GPU query (nvidia-smi) timed out after 30 s; no server was started') from None
    except OSError as error:
        raise ValueError(f'the GPU query (nvidia-smi) could not run ({error}); no server was started') from None
    require(query.returncode == 0 and query.stdout.strip(),
            f'the GPU query (nvidia-smi) failed with exit {query.returncode} '
            f'({query.stderr.strip() or "no output"}); no server was started')
    gpus = query.stdout
    directory.mkdir(parents=True)
    cpus = os.sched_getaffinity(0) & CPUS
    if cpus:
        os.sched_setaffinity(0, cpus)
    run = Run(args, facts, configs, identity, install, directory)
    run.packages = dict(state['packages'])
    run.install_source = state.get('source')
    run.install_metadata[()] = {'path': str(install), 'source': state.get('source'), 'files': state.get('files')}
    # A configuration with install features is served from that install
    # (prepare builds it beside the plain one).
    needed = set().union(*(configuration_features(facts, name, config, args.suite)
                          for name, config in configs.items()))
    for features in sorted(needed - {()}):
        featured = install.with_name(install.name + ''.join(f'+{feature}' for feature in features))
        require((featured / 'install.json').is_file(), f'no install at {featured}: run `release_preview.sh '
                f"prepare --model {args.model} --suite {args.suite}` first")
        run.installs[features] = featured
        featured_state = read(featured / 'install.json')
        run.install_metadata[features] = {'path': str(featured), 'source': featured_state.get('source'),
                                          'files': featured_state.get('files')}
        run.packages.update({key: value for key, value in featured_state['packages'].items()
                             if configs.get(key.split('/', 1)[-1], {}).get('features')})
    missing = [name for name in configs if f'{args.model}/{name}' not in run.packages]
    require(not missing, f"configurations {', '.join(missing)} are not prepared: run `release_preview.sh prepare "
            f"--model {args.model}{''.join(' --config ' + name for name in missing)}`")
    write(directory / 'run.json', {
        'model': args.model, 'suite': args.suite, 'configurations': sorted(configs), 'source': identity,
        'install': str(install), 'install_source': state.get('source'), 'install_files': state['files'],
        'installs': {'+'.join(features): str(path) for features, path in run.installs.items() if features},
        'install_metadata': {'+'.join(features): value for features, value in run.install_metadata.items()},
        'used_installs': {}, 'packages': run.packages,
        'inputs': {key: value for key, value in run.inputs.items() if key not in run.input_values},
        'input_values': run.input_values, 'selected_checks': getattr(args, 'check', []), 'argv': sys.argv, 'started': utc(),
        'gpus': gpus})
    run_configurations(run, configs)
    return finish(run)


def whole_model_full_pass(result, model):
    """Admission requires every declared configuration selected and observed.
    A focused full run or a selected check subset can pass its checks without
    qualifying the model."""
    declared = set(configurations(model))
    selected = result.get('configurations')
    rows = result.get('rows')
    return (result.get('model') == model and result.get('suite') == 'full' and result.get('passed') is True
            and not result.get('selected_checks')
            and isinstance(selected, list) and all(isinstance(name, str) for name in selected)
            and len(selected) == len(declared) and set(selected) == declared
            and isinstance(rows, list) and bool(rows)
            and all(isinstance(row, dict) and isinstance(row.get('id'), str)
                    and row.get('status') in PASSING for row in rows)
            and {row['id'].split('/', 1)[0] for row in rows} == declared)


def finish(run):
    for label, error in shutdown_failures(run.directory, run.facts['model'], run.args.suite,
                                           list(run.configs), run.inputs, getattr(run.args, 'check', [])).items():
        if any(row['id'] == label.split('/')[0] + '/configuration' and row['status'] == 'fail'
               for row in run.rows):
            continue
        if not any(row['id'] == label and row['status'] == 'fail' for row in run.rows):
            config, arm, _ = label.split('/')
            run.row(config, arm, 'server', 'fail', error=error)
    used = {'+'.join(features): run.install_metadata.get(features, {
        'path': str(run.installs[features]), 'source': None, 'files': None})
        for features in sorted(run.used_features)}
    if not run.replay:
        state = read(run.directory / 'run.json')
        write(run.directory / 'run.json', dict(state, used_installs=used))
    passed = bool(run.rows) and all(row['status'] in PASSING for row in run.rows)
    result = {'model': run.facts['model'], 'suite': run.args.suite, 'passed': passed, 'rows': run.rows,
              'configurations': sorted(run.configs),
              'source': run.identity, 'install_source': run.install_source, 'used_installs': used,
              'seconds': round(time.monotonic() - run.started, 1),
              'numerical_reference': run.facts['numerical_reference'],
              'selected_checks': getattr(run.args, 'check', [])}
    write(run.out / 'results.json', result)
    report(run.out, result)
    if (whole_model_full_pass(result, run.facts['model'])
            and not run.args.no_baseline_update and not run.replay):
        update = run.stores['last-good'].update(run.directory, result)
        print(update, flush=True)
    print(f"{run.facts['model']} {run.args.suite}: {'PASS' if passed else 'FAIL'}; {run.out / 'results.json'}",
          flush=True)
    return 0 if passed else 1


def report(directory, result):
    tested = result.get('install_source') or {}
    lines = [f"# {result['model']} {result['suite']} correctness run", '',
             f"Passed: {result['passed']}. Checkout: {result['source']['commit']}"
             f"{' (with uncommitted changes)' if result['source']['dirty'] else ''}. Tested install: "
             f"{tested.get('commit', 'unknown')}{' (with uncommitted changes)' if tested.get('dirty') else ''}.", '',
             result['numerical_reference'], '']
    for features, value in result.get('used_installs', {}).items():
        identity = value.get('source') or {}
        lines.append(f"Used install {features or 'plain'}: {identity.get('commit', 'unknown')}"
                     f"{' (with uncommitted changes)' if identity.get('dirty') else ''}.")
    lines.extend(['', '| Check | Result | Values | Error |', '|---|---|---|---|'])
    for row in result['rows']:
        values = '; '.join(line({'id': '', 'status': '', 'values': [value]}).split('; ', 1)[1]
                           for value in row.get('values', []))
        lines.append(f"| {row['id']} | {row['status']} | {values} | {row.get('error', '')} |")
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')


# Stored baselines ------------------------------------------------------------

def used_install_sources(result):
    """Require a recorded identity for each install the selected plans use.
    Missing historical provenance remains runnable but cannot enter a store."""
    model = result['model']
    facts = model_facts(model)
    configs = selected_configurations(model, result['configurations'], result['suite'])
    needed = {(), *(features for name, config in configs.items()
                   for features in configuration_features(facts, name, config, result['suite']))}
    used = result.get('used_installs')
    require(isinstance(used, dict), 'records no used installs')
    for features in sorted(needed):
        key = '+'.join(features)
        require(isinstance(used.get(key), dict), f'records no used install for {key or "plain"}')
    return [(f'tested install {key or "plain"}', value.get('source') if isinstance(value, dict) else None)
            for key, value in used.items()]


class Store:
    """The last passing full run (last-good) or the operator's seed of one
    model: per configuration, its result rows and the baseline files the
    regression rows compare against."""

    def __init__(self, name, model):
        self.name, self.model = name, model
        self.root = VALIDATION / name / model

    def result(self, config):
        path = self.root / config / 'result.json'
        return read(path) if path.is_file() else None

    def value(self, config, row_id, name):
        result = self.result(config)
        if result is None:
            return None
        row = next((row for row in result['rows'] if row['id'] == row_id), None)
        value = next((value for value in (row or {}).get('values', []) if value['metric'] == name), None)
        return None if value is None else value['value']

    def receipt(self, config, row_id):
        """The stored receipt of a row of `config`, or None."""
        path = self.root / config / 'baseline' / (row_id.replace('/', '__') + '.json')
        return path if path.is_file() else None

    def write(self, directory, result, reason):
        """Store a run's rows and its baseline receipts, per configuration."""
        for config in sorted({row['id'].split('/', 1)[0] for row in result['rows']}):
            target = self.root / config
            staging = target.with_name(target.name + '.new')
            if staging.exists():
                shutil.rmtree(staging)
            (staging / 'baseline').mkdir(parents=True)
            rows = [row for row in result['rows'] if row['id'].split('/', 1)[0] == config]
            for index, row in enumerate(rows):
                if row.get('receipt') and row['status'] in PASSING and Path(row['receipt']).is_file():
                    destination = staging / 'baseline' / (row['id'].replace('/', '__') + '.json')
                    try:
                        receipt = read(row['receipt'])
                    except (ValueError, UnicodeDecodeError):
                        receipt = None
                    results = receipt.get('results') if isinstance(receipt, dict) else None
                    if isinstance(results, str) and Path(results).is_file():
                        # An evals arm's rows: kept beside its receipt, which
                        # names the stored copy (its digest unchanged).
                        shutil.copyfile(results, destination.with_suffix('.jsonl'))
                        receipt['results'] = str(target / 'baseline' / destination.with_suffix('.jsonl').name)
                        write(destination, receipt)
                    else:
                        shutil.copyfile(row['receipt'], destination)
                    rows[index] = dict(row, stored=str(target / 'baseline' / destination.name))
            write(staging / 'result.json', {'run': str(directory), 'source': result['source'],
                                            'install_source': result.get('install_source'),
                                            'used_installs': result.get('used_installs'),
                                            'suite': result['suite'], 'rows': rows, 'at': utc()})
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
        line = {'store': self.name, 'model': self.model, 'run': str(directory),
                'commit': (result.get('install_source') or {}).get('commit'), 'checkout': result['source']['commit'],
                'reason': reason, 'at': utc()}
        with (VALIDATION / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(line, sort_keys=True) + '\n')
        return f'{self.name} {self.model}: stored {directory}'

    def update(self, directory, result):
        """Last good moves only on a whole-model full pass where both the
        validation checkout (its declarations and check code) and every used
        install were built from clean commits that main contains; otherwise
        the result stands and nothing moves."""
        if result.get('selected_checks'):
            return f'{self.name} {self.model}: not updated (a selected check subset)'
        if not whole_model_full_pass(result, self.model):
            return f'{self.name} {self.model}: not updated (requires a whole-model full pass)'
        failures = shutdown_failures(directory, self.model, result['suite'], result['configurations'], {})
        if failures:
            return f'{self.name} {self.model}: not updated (server shutdown evidence refused: {failures})'
        try:
            installs = used_install_sources(result)
        except ValueError as error:
            return f'{self.name} {self.model}: not updated ({error})'
        sources = [('validation checkout', result.get('source')), ('tested install', result.get('install_source'))]
        for what, source_ in [*sources, *installs]:
            if not isinstance(source_, dict) or not isinstance(source_.get('commit'), str) or 'dirty' not in source_:
                return f'{self.name} {self.model}: not updated (the {what} records no source)'
            if source_['dirty'] is not False:
                return f'{self.name} {self.model}: not updated (the {what} has uncommitted changes)'
            code = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', source_['commit'], 'main'],
                                  capture_output=True, timeout=30).returncode
            if code != 0:
                return f'{self.name} {self.model}: not updated ({what} {source_["commit"][:12]} is not on main)'
        return self.write(directory, result, 'full pass')


def baseline_command(args):
    if args.action == 'show':
        for name in STORES:
            store = Store(name, args.model)
            for config in args.config or sorted(path.name for path in store.root.glob('*') if path.is_dir()):
                result = store.result(config)
                print(json.dumps({'store': name, 'config': config, 'run': (result or {}).get('run'),
                                  'commit': ((result or {}).get('source') or {}).get('commit')}))
        return 0
    require(args.reason and args.reason.strip(), 'a re-seed needs --reason')
    directory = Path(args.source).resolve()
    result = read(directory / 'results.json')
    require(result['model'] == args.model and result['suite'] == 'full' and result['passed'],
            'a seed is a passing full run of the model')
    require(not result.get('selected_checks'), 'a seed is not a selected check subset')
    require(not shutdown_failures(directory, args.model, result['suite'], result['configurations'], {}),
            'server shutdown evidence refused')
    print(Store('seed', args.model).write(directory, result, args.reason))
    return 0


# Rescoring -------------------------------------------------------------------

# The rows of a run's execution, not of a judged result.
EXECUTION = ('server', 'configuration')
LEGACY_STAGES = ('determinism', 'gsm8k', 'greedy', 'spec_verify', 'needle', 'queue', 'soak', 'spec_soak')


def rescore_command(args):
    """Re-judge a stored run's receipts with this code: a runner run, or with
    --legacy-preview MODEL a previous preview run (preview_legacy.py layout).
    Prints and writes rows; exit 1 when any fails."""
    root = Path(args.run).resolve()
    if args.legacy_preview:
        rows = rescore_legacy(root, args.legacy_preview)
        for row in rows:
            print(line(row))
    else:
        rows = rescore_run(root)
    if args.json:
        write(args.json, {'run': str(root), 'rows': rows})
    return 0 if rows and all(row['status'] in PASSING for row in rows) else 1


def judge_receipt(stage, value, facts, arm, run_id, image, exact_sha256, soak_seconds=AR_SOAK_SECONDS):
    """A screen stage's verdict on a stored receipt, as run_stage judges it."""
    if stage == 'gsm8k':
        return judge_gsm8k(value, facts)
    if stage == 'greedy':
        return judge_greedy(value, facts, arm, exact_sha256)
    if stage == 'needle':
        return judge_needle(value, facts)
    if stage == 'queue':
        return judge_queue(value, facts)
    if stage in ('soak', 'spec_soak'):
        return judge_soak(value, facts, soak_arguments(facts, soak_seconds,
                                                       stage == 'spec_soak' and facts['soak_prefix_mix']))
    if stage == 'determinism':
        return judge_determinism_capture(value, facts, arm, run_id, image)
    if stage == 'spec_verify':
        return judge_spec_capture(value, facts, arm, run_id)
    raise ValueError(f'unknown stage {stage}')


def rescore_run(root):
    """A runner run replayed with this code: the same plan, each recorded
    child's exit and receipt judged again by the stage rules and checks.py
    rescore, and the comparisons between arms recomputed from the recorded
    arms. A row the plan expects and the run lacks fails. Output goes to
    RUN/rescore/<UTC>/; the stored baselines never move."""
    state = read(root / 'run.json')
    recorded = read(root / 'results.json')
    lacking = [key for key in ('model', 'suite', 'configurations', 'source', 'install', 'packages', 'inputs')
               if key not in state]
    require(not lacking, f"{root}: run.json lacks {', '.join(lacking)}: not a runner run")
    model = state['model']
    require(recorded.get('model') == model, f'{root}: results.json and run.json name different models')
    names = state['configurations']
    require(isinstance(names, list) and names, f'{root}: the run names no configuration')
    facts = model_facts(model)
    configs = selected_configurations(model, names)
    args = argparse.Namespace(suite=state['suite'], input=[f'{key}={path}' for key, path in state['inputs'].items()],
                              input_value=[f'{key}={value}' for key, value in state.get('input_values', {}).items()],
                              no_baseline_update=True, check=state.get('selected_checks', []))
    facts, configs = select_checks(facts, configs, args.suite, args.check)
    run = Run(args, facts, configs, state['source'], Path(state['install']), root)
    run.replay, run.out = True, root / 'rescore' / utc()
    run.packages, run.install_source = state['packages'], state.get('install_source')
    run.installs.update({tuple(key.split('+')): Path(path) for key, path in state.get('installs', {}).items()})
    run.install_metadata = {tuple(key.split('+')) if key else (): value
                            for key, value in state.get('used_installs', {}).items()}
    for name, config in configs.items():
        try:
            run_configuration(run, name, config)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            run.row(name, None, 'configuration', 'fail', error=str(error))
    statuses = {row['id']: row['status'] for row in recorded.get('rows', [])}
    for row in run.rows:
        row['recorded'] = statuses.get(row['id'])
    # A failed server, configuration or record is not judged again: one the
    # replay does not meet again (a server that failed on exit after its
    # checks) is kept as recorded.
    replayed = {row['id'] for row in run.rows}
    for old in recorded.get('rows', []):
        name = old['id'].rsplit('/', 1)[-1]
        if (old['status'] not in PASSING and old['id'] not in replayed
                and (name in EXECUTION or name.startswith('record-'))):
            run.rows.append(dict(old, recorded=old['status'], kept='a recorded execution failure'))
            print(line(run.rows[-1]), flush=True)
    finish(run)
    return run.rows


def rescore_check(receipt, model, config, check):
    """A declared check's stored receipt, re-judged by checks.py rescore
    against the current declaration as `check`, the check its row ran (two
    checks' receipt patterns may match one file); a passing status holds
    only with exit 0, and any exit but 0, 1 or 3 (insufficient) refuses."""
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'rescore.json'
        done = subprocess.run([sys.executable, '-B', str(TOOLS / 'checks.py'), 'rescore', '--model',
                               declaration_name(model, config), '--check', check, '--json', str(output), receipt],
                              cwd=ROOT, capture_output=True, text=True, timeout=3600)
        require(done.returncode in (0, 1, 3) and output.is_file(),
                f'checks.py rescore exited {done.returncode}: {done.stderr.strip()[-300:]}')
        rows = read(output)['checks']
    require(len(rows) == 1, f'checks.py rescore judged {len(rows)} checks in one receipt')
    value = rows[0]
    require((done.returncode == 0) == (value['status'] in PASSING),
            f"checks.py rescore exited {done.returncode} with status {value['status']!r}")
    return {'status': value['status'], 'differing_metrics': value.get('differing_metrics'),
            'error': value.get('error')}


def rescore_legacy(root, model):
    """A previous preview run's rows, re-judged: each stage receipt by the
    stage rules above (the preview's soak durations), the quick gate's checks
    by checks.py rescore, and the comparisons between arms."""
    facts = model_facts(model)
    attempt = root / 'attempt'
    prepared = read(attempt / 'prepared.json') if (attempt / 'prepared.json').is_file() else read(root / 'prepared.json')
    run_id = root.name
    image = None
    package = (prepared.get('checkpoints') or {}).get(model, {}).get('bundle')
    if package and (Path(package) / 'manifest.json').is_file():
        manifest = read(Path(package) / 'manifest.json')
        image = MEDIA_CAPABILITY in ((manifest.get('resource_requirements') or {}).get('required_capabilities') or [])
    budgets = prepared.get('budgets', {})
    exact_sha256 = prepared['greedy_policy'][model]['reference_sha256']
    rows, records, ran = [], {}, set()
    for group in sorted((attempt / model).glob('*/group.json')):
        for old in read(group)['checks']:
            parts = old['id'].split('/')
            if len(parts) != 3 or parts[2] not in LEGACY_STAGES + ('quick',):
                continue
            _, arm, stage = parts
            ran.add(stage)
            new = {'id': old['id'], 'status': 'fail', 'recorded': old.get('status')}
            if not old.get('receipt'):
                rows.append(dict(new, status='not_run', error='the run recorded no receipt'))
                continue
            try:
                value = read(old['receipt'])
                if stage == 'quick':
                    new.update(rescore_quick(old['receipt']))
                else:
                    seconds = budgets.get('spec_soak_duration' if stage == 'spec_soak' else 'soak_duration',
                                          AR_SOAK_SECONDS)
                    values, extra = judge_receipt(stage, value, facts, arm, run_id,
                                                  image if image is not None else value.get('image', {}).get('included'),
                                                  exact_sha256, seconds)
                    new.update(status='pass', values=values, **extra)
                records[(arm, stage)] = value
            except (ValueError, KeyError, TypeError, OSError) as error:
                new['error'] = str(error)
            rows.append(new)
    require(rows, f'{root}: the run holds no stage or quick gate row of {model}')
    rows += legacy_comparisons(facts, records, run_id, attempt / model, ran)
    return rows


def rescore_quick(summary):
    """The quick gate's checks, re-judged by checks.py rescore under the
    policy the gate run recorded."""
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'rescore.json'
        done = subprocess.run([sys.executable, '-B', str(TOOLS / 'checks.py'), 'rescore', '--model',
                               read(summary)['model'], '--limits', 'recorded', '--json', str(output),
                               str(Path(summary).parent)], cwd=ROOT, capture_output=True, text=True, timeout=3600)
        require(done.returncode in (0, 1, 3) and output.is_file(),
                f'checks.py rescore exited {done.returncode}: {done.stderr.strip()[-300:]}')
        value = read(output)
    subchecks = [{'id': row['id'], 'status': row['status'], 'recorded': row.get('recorded'),
                  'differing_metrics': row.get('differing_metrics')} for row in value['checks']]
    require(len(value['gates']) == 1 and subchecks, 'checks.py rescore found no quick gate run')
    passed = value['gates'][0]['passed'] is True and done.returncode == 0
    return {'status': 'pass' if passed else 'fail', 'subchecks': subchecks}


def legacy_comparisons(facts, records, run_id, directory, ran):
    """The previous preview's comparisons between arms, recomputed from the
    arms' re-judged records; a comparison of stages the run had fails as not
    run when a record is missing, and one of stages it never ran is absent."""
    rows = []
    model = facts['model']
    if 'spec_verify' in ran:
        try:
            comparison = compare_spec(records[('ar', 'spec_verify')], records[('spec', 'spec_verify')])
            require(comparison['passed'], 'spec_verify numerical bounds exceeded')
            rows.append({'id': f'{model}/spec-verify', 'status': 'pass', **comparison})
        except KeyError:
            rows.append({'id': f'{model}/spec-verify', 'status': 'not_run',
                         'error': 'a spec_verify record did not pass'})
        except ValueError as error:
            rows.append({'id': f'{model}/spec-verify', 'status': 'fail', 'error': str(error)})
    for left, right in (('spec', 'spec-repeat'), ('spec', 'ar')):
        name = f'{model}/determinism-{left}-vs-{right}'
        try:
            comparison, failure = compare_determinism(records[(left, 'determinism')], records[(right, 'determinism')])
            rows.append({'id': name, 'status': 'pass' if failure is None else 'fail', 'determinism': comparison,
                         **({'error': failure} if failure else {})})
        except KeyError:
            rows.append({'id': name, 'status': 'not_run'})
    try:
        comparison = compare_greedy(records[('ar', 'greedy')], records[('spec', 'greedy')], facts['spec_greedy'])
        rows.append({'id': f'{model}/spec/greedy-vs-ar',
                     'status': 'pass' if facts['spec_greedy'] == 'exact' else 'informational',
                     'first_differences': comparison['first_differences'],
                     'finish_differences': comparison['finish_differences']})
    except KeyError:
        rows.append({'id': f'{model}/spec/greedy-vs-ar', 'status': 'not_run', 'error': 'a greedy capture did not pass'})
    except ValueError as error:
        rows.append({'id': f'{model}/spec/greedy-vs-ar', 'status': 'fail', 'error': str(error)})
    try:
        from gsm8k_branches import pair
        branches = sorted(directory.glob('*/*/gsm8k_branches/*.json'))
        probe = read(branches[0]) if branches else None
        from gsm8k_branches import verdict
        comparison = pair(records[('ar', 'gsm8k')], records[('spec', 'gsm8k')], probe, facts['gsm8k_floor'])
        rows.append({'id': f'{model}/gsm8k-paired', 'status': 'pass', 'lost': comparison['lost'],
                     'gained': comparison['gained'],
                     'branches': [dict(item, verdict=verdict(item)) for item in (probe or {}).get('items', [])]})
    except KeyError:
        rows.append({'id': f'{model}/gsm8k-paired', 'status': 'not_run'})
    except ValueError as error:
        rows.append({'id': f'{model}/gsm8k-paired', 'status': 'fail', 'error': str(error)})
    return rows


# Command line ----------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'plan'):
        command = commands.add_parser(name)
        command.add_argument('--model', required=True)
        command.add_argument('--config', action='append', default=[])
        command.add_argument('--suite', choices=SUITES, default='quick',
                             help='without --config: default for quick, every configuration for full')
        if name != 'prepare':
            command.add_argument('--check', action='append', default=[],
                                 help='development subset, without screens or last-good promotion')
        if name == 'prepare':
            command.add_argument('--target-dir', help='cargo target directory, owned by this checkout (default: '
                                 'CARGO_TARGET_DIR, else wt-<name>-<hash> under $CHOPINFER_PREVIEW_TARGETS or '
                                 '$CHOPINFER_DATA/validation/targets)')
        if name == 'run':
            command.add_argument('--install', help='an install other than the checked-out commit\'s')
            command.add_argument('--input', action='append', default=[], metavar='KEY=PATH',
                                 help='an input no record of this run produces')
            command.add_argument('--input-value', action='append', default=[], metavar='KEY=VALUE',
                                 help='such an input that is a value, not a path (another run\'s check run ID)')
            command.add_argument('--out', help='the run directory (default: under $CHOPINFER_DATA/validation/runs)')
            command.add_argument('--no-baseline-update', action='store_true',
                                 help='never store this run as the last good one')
    rescore = commands.add_parser('rescore')
    rescore.add_argument('run')
    rescore.add_argument('--legacy-preview', metavar='MODEL', help='a previous preview run (preview_legacy.py)')
    rescore.add_argument('--json', type=Path)
    baseline = commands.add_parser('baseline')
    baseline.add_argument('action', choices=('show', 'reseed'))
    baseline.add_argument('--model', required=True)
    baseline.add_argument('--config', action='append', default=[])
    baseline.add_argument('--from', dest='source', help='reseed: the passing full run')
    baseline.add_argument('--reason', help='reseed: why the seed moves (recorded)')
    args = parser.parse_args(argv)
    handlers = {'prepare': prepare, 'run': run_command, 'plan': plan_command, 'rescore': rescore_command,
                'baseline': baseline_command}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'release_preview: {error!r}' if isinstance(error, KeyError) else f'release_preview: {error}',
              file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
