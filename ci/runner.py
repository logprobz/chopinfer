"""Run CI steps as supervised child processes with a bounded number of workers.

A step starts after every step it names in `after` has succeeded, in the
order steps were declared. Each step runs in its own process group and writes
its own log; the runner prints a finished step's log as one block. A failed
step stops the run (unless keep_going) and SIGINT or SIGTERM stops every
running group, so no step outlives the runner. A step that declares its
allowed unittest skips passes only when its unittest log also shows executed
tests, an OK verdict and no other skip."""
import contextlib
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import time


@dataclasses.dataclass
class Step:
    name: str
    argv: list
    env: dict = None
    cwd: str = None
    after: tuple = ()
    # Worker slots the step occupies while it runs.
    weight: int = 1
    # Print the log when the step succeeds as well as when it fails.
    echo: bool = True
    # For a required unittest -v run: the skip reasons it may report, as
    # regular expressions. None leaves the exit status alone.
    skips: tuple = None


@dataclasses.dataclass
class Result:
    name: str
    status: str
    exit: int = None
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    max_rss_kib: int = 0
    log: str = None
    started: float = None
    # Executed and skipped unittest cases, for a step that declares its skips.
    tests: int = None
    skipped: int = None


class Interrupted(Exception):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


# Signals received while interrupts are held, or None when they raise.
_held = None


def _interrupted(signum):
    """Raises the interruption the run stops on. Until the stop holds them
    itself, later signals are held, so none can cut the stop's entry short."""
    global _held
    _held = []
    raise Interrupted(signum)


def _interrupt(signum, frame):
    if _held is not None:
        _held.append(signum)
        return
    _interrupted(signum)


@contextlib.contextmanager
def _holding_interrupts(raise_held=True):
    """Hold SIGINT and SIGTERM for the block, then raise the first one held,
    or with `raise_held` false drop them.

    A step's start forks before it returns the PID the runner records, so an
    interruption in between would leave that step outside `stop`'s reach;
    one during `stop` would skip its SIGKILL of steps that ignore SIGTERM.
    The hold is not reentrant: `run` must not be called from inside another
    `run`'s step start in the same process."""
    global _held
    _held = []
    try:
        yield
    finally:
        held, _held = _held, None
        if held and raise_held:
            _interrupted(held[0])


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _exited(pids, poll=0.02):
    """Waits until one of `pids`, the runner's own children, exits, and
    returns it unreaped."""
    while True:
        for pid in pids:
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                return pid
        time.sleep(poll)


def stop(groups, grace=10.0):
    """Terminate the process groups the runner started, then kill what remains.

    Each group ID is the PID of a step the runner started. Leaders that exit
    stay unreaped until the final SIGKILL, so no other process can take their
    IDs in between."""
    groups = set(groups)
    for pid in groups:
        _signal_group(pid, signal.SIGTERM)
    alive = set(groups)
    deadline = time.monotonic() + grace
    while alive and time.monotonic() < deadline:
        for pid in list(alive):
            try:
                if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    alive.discard(pid)
            except ChildProcessError:
                alive.discard(pid)
        if alive:
            time.sleep(0.05)
    # A leader that exited may leave descendants that ignored SIGTERM.
    for pid in groups:
        _signal_group(pid, signal.SIGKILL)
    for pid in groups:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def run(steps, jobs, log_dir, keep_going=False, out=None, label="ci", grace=10.0):
    """Run `steps`; return their results in declaration order. On a failure or
    a signal, running steps get `grace` seconds to exit after SIGTERM."""
    global _held
    out = out or sys.stderr
    names = [step.name for step in steps]
    if len(set(names)) != len(names):
        raise ValueError("duplicate step names")
    unknown = {dependency for step in steps for dependency in step.after} - set(names)
    if unknown:
        raise ValueError("unknown dependencies: " + ", ".join(sorted(unknown)))
    os.makedirs(log_dir, exist_ok=True)
    results = {step.name: Result(step.name, "pending") for step in steps}
    pending = list(steps)
    running = {}
    stopping = False
    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or running:
            if not stopping:
                for step in list(pending):
                    states = [results[dependency].status for dependency in step.after]
                    if any(state not in ("pending", "running", "ok") for state in states):
                        results[step.name].status = "skipped"
                        pending.remove(step)
                        continue
                    busy = sum(running[pid][0].weight for pid in running)
                    if all(state == "ok" for state in states) and (not running or busy + step.weight <= jobs):
                        pending.remove(step)
                        with _holding_interrupts():
                            running[_start(step, results[step.name], log_dir)] = (step, time.monotonic())
            if not running:
                for step in pending:
                    results[step.name].status = "skipped"
                break
            # Find an exited step without reaping it: until it leaves
            # `running`, its ID must stay reserved for stop()'s group signals.
            # Only the runner's own steps are waited on, so children another
            # thread starts (the HEAD poll's git) stay that thread's to reap.
            pid = _exited(running)
            with _holding_interrupts():
                # The unreaped leader keeps its group's ID reserved, so this
                # reaches only processes the step left behind.
                _signal_group(pid, signal.SIGKILL)
                _, status, usage = os.wait4(pid, 0)
                step, started = running.pop(pid)
            result = results[step.name]
            result.exit = os.waitstatus_to_exitcode(status)
            result.seconds = time.monotonic() - started
            result.cpu_seconds = usage.ru_utime + usage.ru_stime
            result.max_rss_kib = usage.ru_maxrss
            result.status = "ok" if result.exit == 0 else "failed"
            if step.skips is not None:
                _account(step, result)
            _report(step, result, out, label)
            if result.status == "failed" and not keep_going:
                stopping = True
                for other in pending:
                    results[other.name].status = "skipped"
                pending.clear()
                if running:
                    # A signal during the stop is raised once the steps are reaped.
                    with _holding_interrupts():
                        stop(set(running), grace)
                        for other, _ in running.values():
                            results[other.name].status = "stopped"
                        running.clear()
    except Interrupted as interruption:
        # A second signal, such as a terminal's Ctrl-C after a wrapper's, must
        # not cut the stop short: the run is ending on the first one anyway.
        with _holding_interrupts(raise_held=False):
            stop(set(running), grace)
        for other, _ in running.values():
            results[other.name].status = "stopped"
        print(f"{label}: interrupted by signal {interruption.signum}; stopped "
              f"{len(running)} running step(s)", file=out, flush=True)
        raise
    finally:
        _held = None
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return [results[name] for name in names]


def _start(step, result, log_dir):
    result.log = os.path.join(log_dir, step.name.replace("/", "_") + ".log")
    result.status = "running"
    result.started = time.time()
    with open(result.log, "wb") as log:
        child = subprocess.Popen(step.argv, cwd=step.cwd, env=step.env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # The runner reaps children with os.wait4; keep Popen from waiting again.
    child.returncode = 0
    return child.pid


SUMMARY = re.compile(r"^Ran (\d+) tests? in [0-9.]+s$", re.M)
VERDICT = re.compile(r"^(OK|FAILED)(?: \((.*)\))?$", re.M)
SKIP = re.compile(r""" \.\.\. skipped (['"])(.*)\1$""", re.M)


def unittest_outcome(text):
    """(tests run including skips, verdicts, skip reasons, reported skip count) from a unittest -v log."""
    verdicts = VERDICT.findall(text)
    counted = sum(int(item.partition("=")[2]) for _, details in verdicts for item in details.split(", ")
                  if item.startswith("skipped="))
    return (sum(int(count) for count in SUMMARY.findall(text)), [verdict for verdict, _ in verdicts],
            [reason for _, reason in SKIP.findall(text)], counted)


def _account(step, result):
    """Fail a step whose unittest run executed nothing, did not pass or skipped
    a case it does not allow; append the counts and causes to its log."""
    with open(result.log, errors="replace") as log:
        ran, verdicts, skips, counted = unittest_outcome(log.read())
    # unittest counts a skipped case as run; only the rest executed a body.
    skipped = max(counted, len(skips))
    tests = ran - skipped
    result.tests, result.skipped = tests, skipped
    problems = [] if tests > 0 else ["no test executed"]
    if result.status == "ok" and (not verdicts or any(verdict != "OK" for verdict in verdicts)):
        problems.append("no unittest OK verdict")
    if counted != len(skips):
        problems.append(f"{counted} skips reported, {len(skips)} with a reason (run unittest with -v)")
    unexpected = {}
    for reason in skips:
        if not any(re.fullmatch(pattern, reason) for pattern in step.skips):
            unexpected[reason] = unexpected.get(reason, 0) + 1
    problems += [f"unexpected skip ({count}): {reason}" for reason, count in sorted(unexpected.items())]
    with open(result.log, "a") as log:
        log.write(f"{step.name}: {tests} tests executed, {skipped} skipped\n")
        for problem in problems:
            log.write(f"{step.name}: {problem}\n")
    if problems and result.status == "ok":
        result.status = "failed"


def _report(step, result, out, label):
    if step.echo or result.status != "ok":
        with open(result.log, errors="replace") as log:
            text = log.read()
        out.write(f"==> {step.name} ({result.status})\n{text}")
        if text and not text.endswith("\n"):
            out.write("\n")
    out.write(f"{label} {step.name}: {result.seconds:.3f}s (exit {result.exit})\n")
    out.flush()


def write_summary(path, results, **fields):
    with open(path, "w") as summary:
        json.dump({**fields, "steps": [dataclasses.asdict(result) for result in results]}, summary, indent=1)
        summary.write("\n")


def failed(results):
    return [result.name for result in results if result.status != "ok"]
