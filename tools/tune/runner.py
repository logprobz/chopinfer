#!/usr/bin/env python3
"""Server control for the depth tuner: one server per process, launched
through the direct path of a sealed install with its prepared package, each
in its own session with a lifetime backstop. The sealed model config names
the checkpoint variable, served model name, tensor parallel size, KV pool and
serve options.

Only `server-child` runs this file directly; tune.py imports it."""
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
# Set by the job from the sealed model config.
MODEL = None
CHECKPOINT_ENV = None
GPUS = 4
KV_PAGES = None
# The model's KV log line, `{pages}` standing for the pages per rank.
KV_LINE = None
SERVE = []
# Environment the servers keep: every other CHOPINFER_ variable is a control
# the job must not inherit (each arm states its own options).
KEPT = {'CHOPINFER_DATA'}
WORK_DEADLINE = None
# Power limits are recorded at the start and must not change; a given value
# is also required on every GPU.
EXPECTED_POWER_W = None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def http(url, path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    with urllib.request.urlopen(urllib.request.Request(url + path, data, headers), timeout=timeout) as response:
        return response.read().decode()


def stream(url, path, body, timeout=10):
    """The JSON events of a server-sent event stream, until `[DONE]`."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            if line == 'data: [DONE]':
                return
            yield json.loads(line[6:])
    raise RuntimeError(f'{path}: stream ended without [DONE]')


def remaining(limit):
    if WORK_DEADLINE is None:
        return limit
    budget = WORK_DEADLINE - time.monotonic()
    if budget <= 0:
        raise TimeoutError('job work deadline reached')
    return min(limit, budget)


def command(argv, log, seconds=30, env=None, check=True):
    """Each utility has a timeout and a separate process group."""
    argv = list(map(str, argv))
    argv[0] = str(Path(shutil.which(argv[0]) or argv[0]).resolve())
    write(str(log) + '.command.json', {'argv': argv, 'timeout_seconds': seconds})
    with Path(log).open('w') as output:
        process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except BaseException:
            kill_group(process.pid, process, grace=1)
            raise
    write(str(log) + '.exit.json', {'exit_code': code})
    if check and code:
        raise RuntimeError(f'{argv[0]} exited {code}: {log}')
    return code


def group_alive(pgid):
    """Unreapable zombies hold no GPU allocations."""
    for status in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = status.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return False


def kill_group(pgid, process=None, grace=8):
    if pgid == os.getpgrp():
        raise RuntimeError('refusing to signal the runner process group')
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process is not None:
            process.poll()
        if not group_alive(pgid):
            break
        time.sleep(.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    if process is not None:
        process.wait(timeout=5)
    if group_alive(pgid):
        raise RuntimeError(f'residual live process group {pgid}')


class JobInterrupted(BaseException):
    """A stop signal must bypass retry and optional-metrics error handlers."""


_interrupting = False


def interrupted(number, frame):
    global _interrupting
    # A timeout wrapper may forward a signal already delivered to this group.
    # Only the first signal unwinds work; later ones must not cut cleanup.
    if _interrupting:
        return
    _interrupting = True
    raise JobInterrupted(f'received signal {number}')


def trace_rows(path, offset=0):
    with path.open(errors='replace') as stream:
        stream.seek(offset)
        lines = stream.readlines()
    rows = []
    for line in lines:
        start = line.find('{')
        if '"chopinfer.step.v1"' in line and start >= 0:
            try:
                rows.append(json.loads(line[start:]))
            except ValueError:
                pass
    return rows


class Server:
    """One direct-path server of `install` and its prepared package in
    `cache`, serving the checkpoint with `arguments` added: the arm's own
    speculation options, or none for the package profile's defaults."""

    def __init__(self, directory, install, cache, arguments, dry=False, stub=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.dry = dry
        self.pidfile = self.directory / 'server-group.json'
        self.process = self.log = None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.env = {k: v for k, v in os.environ.items()
                    if k != 'OPENAI_API_KEY' and (not k.startswith('CHOPINFER_') or k in KEPT | {CHECKPOINT_ENV})}
        self.env.update(CHOPINFER_STEP_TRACE='1', PYTHONDONTWRITEBYTECODE='1')
        self.server_command = [
            str(Path(install) / 'bin/chopinfer'), 'serve', '--model', os.environ.get(CHECKPOINT_ENV, 'dry-run'),
            '--tensor-parallel-size', str(GPUS), '--model-cache-dir', str(cache), '--listen', f'127.0.0.1:{port}',
            '--served-model-name', MODEL, '--drain-timeout-ms', '5000', '--kv-pages', str(KV_PAGES), *SERVE,
            *arguments]
        launched = self.server_command
        if dry:
            launched = [sys.executable, '-B', str(stub), '--port', str(port), '--directory', str(self.directory),
                        '--kv-pages', str(KV_PAGES), '--kv-line', KV_LINE, '--model', MODEL, '--gpus', str(GPUS),
                        '--', *arguments]
        self.launch = [sys.executable, '-B', str(HERE / 'runner.py'), 'server-child', str(self.pidfile), *launched]

    def start(self):
        write(self.directory / 'launch.json', {'argv': self.launch, 'serve_argv': self.server_command,
                                              'dry_run': self.dry, 'startup_timeout_seconds': 240,
                                              'lifetime_timeout_seconds': 2100})
        self.log = (self.directory / 'server.log').open('w')
        started = time.monotonic()
        self.process = subprocess.Popen(self.launch, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env=self.env)
        deadline = started + remaining(30 if self.dry else 240)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'{self.directory.name} exited before readiness')
            try:
                health = json.loads(http(self.url, '/health', timeout=1))
                models = json.loads(http(self.url, '/v1/models', timeout=1))
                if MODEL not in [row['id'] for row in models['data']]:
                    raise RuntimeError('wrong model from /v1/models')
                self.readiness = time.monotonic() - started
                self.health = health
                write(self.directory / 'ready.json', {'seconds': self.readiness, 'health': health, 'models': models})
                return
            except (OSError, ValueError):
                time.sleep(.2)
        raise TimeoutError(f'{self.directory.name} startup exceeded its timeout')

    def stop(self):
        errors = []
        old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if self.pidfile.exists():
                try:
                    kill_group(json.loads(self.pidfile.read_text())['pgid'])
                except Exception as error:
                    errors.append(str(error))
            if self.process is not None:
                try:
                    kill_group(self.process.pid, self.process)
                except Exception as error:
                    errors.append(str(error))
            if self.log:
                self.log.close()
            write(self.directory / 'cleanup.json', {'passed': not errors, 'errors': errors})
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
        if errors:
            raise RuntimeError('; '.join(errors))


def same_power(reference, limits):
    return len(reference) == len(limits) and all(abs(a - b) <= .5 for a, b in zip(reference, limits))


def power_snapshot(run, label, dry):
    """Power limits, draw, memory and SM clocks of every GPU."""
    if dry:
        watts = float(os.environ.get('TUNE_DRY_POWER_W', '300'))
        if os.environ.get('TUNE_DRY_POWER_CHANGE_LABEL') == label:
            watts -= 50
        value = {'power_limits_w': [watts] * GPUS, 'enforced_power_limits_w': [watts] * GPUS, 'synthetic': True}
        write(run / f'power-{label}.json', value)
        return value
    command(['nvidia-smi', '--query-gpu=index,power.limit,enforced.power.limit,power.draw,memory.used,clocks.sm,'
             'clocks.max.sm', '--format=csv,noheader,nounits'], run / f'power-{label}.csv', 15)
    rows = [[x.strip() for x in line.split(',')] for line in (run / f'power-{label}.csv').read_text().splitlines()
            if line.strip()]
    if [row[0] for row in rows] != [str(i) for i in range(GPUS)]:
        raise RuntimeError(f'expected {GPUS} GPUs, found {[row[0] for row in rows]}')

    def number(text):
        try:
            return float(text)
        except ValueError:
            return None
    value = {'power_limits_w': [float(row[1]) for row in rows], 'enforced_power_limits_w': [float(row[2]) for row in rows],
             'power_draw_w': [number(row[3]) for row in rows], 'memory_used_mib': [number(row[4]) for row in rows],
             'sm_clock_mhz': [number(row[5]) for row in rows], 'max_sm_clock_mhz': [number(row[6]) for row in rows]}
    write(run / f'power-{label}.json', value)
    return value


def power_check(run, label, dry, reference=None):
    value = power_snapshot(run, label, dry)
    limits = value['power_limits_w']
    if EXPECTED_POWER_W is not None and not same_power([EXPECTED_POWER_W] * GPUS, limits):
        raise RuntimeError(f'{label}: power limits {limits} W differ from the expected {EXPECTED_POWER_W} W')
    if reference is not None and not same_power(reference, limits):
        raise RuntimeError(f'{label}: power limits changed during the run: {reference} W at start, {limits} W now')
    return value


def telemetry(run, when, dry, since, reference=None):
    power = power_check(run, when, dry, reference)
    if dry:
        value = {'power_limits_w': power['power_limits_w'], 'power': power, 'xids': [], 'journal_checked': True,
                 'synthetic': True}
        write(run / f'telemetry-{when}.json', value)
        return value
    code = command(['journalctl', '-k', '--since', since, '--no-pager', '-o', 'short-iso'], run / f'journal-{when}.log',
                   20, check=False)
    journal = (run / f'journal-{when}.log').read_text()
    if code != 0 or re.search('permission|not seeing|No journal files', journal, re.I):
        raise RuntimeError('journalctl -k could not check kernel Xids')
    xids = [line for line in journal.splitlines() if re.search(r'\bNVRM:.*\bXid\b', line)]
    value = {'power_limits_w': power['power_limits_w'], 'power': power, 'xids': xids, 'journal_checked': True}
    write(run / f'telemetry-{when}.json', value)
    if when == 'before':
        command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                run / 'compute-before.csv', 15)
        if (run / 'compute-before.csv').read_text().strip():
            raise RuntimeError('whole-machine job found existing GPU compute processes')
    return value


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'server-child':
        pidfile = Path(sys.argv[2])
        if os.getpid() != os.getpgrp():
            os.setsid()
        write(pidfile, {'pid': os.getpid(), 'pgid': os.getpgrp()})
        # The lifetime backstop shares this session even if the job dies.
        os.execv('/usr/bin/timeout', ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                      '2100s', *sys.argv[3:]])
    raise SystemExit('runner.py only runs as server-child; tune.py is the job')


if __name__ == '__main__':
    raise SystemExit(main())
