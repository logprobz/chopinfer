"""Process control for concurrency measurements under one absolute deadline
(seconds since the epoch). Nothing here depends on the model.

`stop` stops the server a redeploy pidfile names (qwen38-redeploy.sh and
mimo-redeploy.sh write it through redeploy-common.sh). As
redeploy-common.sh's stop_owned_server does, a pidfd pins the process and its
start time and ownership token must match before any signal; a stale or
foreign pidfile is refused without signalling. One reading of the clock,
after the ownership checks and before any signal, picks the path, so the
exit status never depends on how fast the process dies:
- time left beyond the escalation reserve: SIGTERM, with the time left
  before the reserve to exit, then SIGKILL;
- time left within the reserve: SIGKILL at once (escalated);
- no time left: SIGKILL, no wait, exit 1 with the pidfile kept.
The helper then waits for the exit and for NVIDIA compute accounting to
release the GPUs, every wait bounded by the deadline. The pidfile is removed
only once both are confirmed. Exit 0 after a clean stop, `ESCALATED` when
SIGKILL was needed (the process is gone and the GPUs are released, but the
stop failed), 1 on refusal, an expired deadline or a timeout (the pidfile
stays for a retry).

`measure` holds a filler (fill_batch.py) while a measurement command runs:
it starts the filler, waits for its `ready` line, runs the command, and
watches the filler throughout. A filler that exits, or reports a finished or
failed stream, before the command ends makes the measurement invalid: the
command is stopped and the helper exits `INVALID`. After the command, the
filler is sent SIGTERM and must acknowledge it (`stopped`, exit 0). Exit 0
for a valid measurement, the command's own nonzero status when it failed,
`INVALID` when the filler did not hold, 1 when the filler never became
ready.
"""

import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ESCALATED = 3
INVALID = 4
# Seconds kept for SIGKILL and the exit that follows.
KILL_RESERVE = 5.0
# Longest SIGTERM grace.
TERM_GRACE = 120.0


def load_wait():
    spec = importlib.util.spec_from_file_location(
        "redeploy_wait", Path(__file__).resolve().parent / "redeploy-wait.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def monotonic_deadline(deadline):
    """The absolute epoch deadline on the monotonic clock."""
    return time.monotonic() + (deadline - time.time())


def remaining(until):
    return max(0.0, until - time.monotonic())


def proc_state(pid):
    """(state, start time) of `pid`, or None once it is gone."""
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return suffix[0], suffix[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def exited(pid):
    state = proc_state(pid)
    return state is None or state[0] == "Z"


def wait_exit(pid, until):
    """Whether `pid` exits by `until`."""
    while not exited(pid):
        if time.monotonic() >= until:
            return False
        time.sleep(min(0.05, remaining(until)))
    return True


def stop(pidfile, deadline, wait_released=None, log=print, clock=time.monotonic):
    """Stops the pidfile's server by `deadline` (see the module doc). `clock`
    is the monotonic clock the path decision reads."""
    until = monotonic_deadline(deadline)
    path = Path(pidfile)
    if not path.exists():
        return 0
    fields = path.read_text().split()
    if len(fields) != 3 or not fields[0].isdigit():
        log(f"invalid pidfile {path}; refusing to signal any process")
        return 1
    pid, started, token = int(fields[0]), fields[1], fields[2]
    if wait_released is None:
        wait_released = load_wait().wait_released

    def released(pidfd):
        try:
            wait_released(pid, pidfd, [], until)
        except TimeoutError as error:
            log(f"server {pid}: {error}")
            return False
        return True

    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        if not released(None):
            return 1
        path.unlink()
        return 0
    try:
        current = proc_state(pid)
        if current is None or current[0] == "Z":
            if not released(pidfd):
                return 1
            path.unlink()
            return 0
        try:
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            environment = None
        if environment is None:
            if not released(pidfd):
                return 1
            path.unlink()
            return 0
        if current[1] != started or f"CHOPINFER_DEV_REDEPLOY_TOKEN={token}".encode() not in environment:
            log(f"pidfile {path} does not identify its original server; refusing to signal pid {pid}")
            return 1

        def kill():
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass

        left = until - clock()
        if left <= 0:
            # Nothing may be waited for; the server must not outlive the run.
            kill()
            log(f"server {pid}: the deadline has passed; sent SIGKILL, pidfile kept")
            return 1
        escalated = left <= KILL_RESERVE
        if escalated:
            log(f"server {pid}: {left:.1f} s left, within the escalation reserve; sending SIGKILL")
            kill()
            if not wait_exit(pid, until):
                log(f"server {pid} is still running at the deadline")
                return 1
        else:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            grace = min(clock() + TERM_GRACE, until - KILL_RESERVE)
            if not wait_exit(pid, grace):
                log(f"server {pid} did not exit after SIGTERM; sending SIGKILL")
                escalated = True
                kill()
                if not wait_exit(pid, until):
                    log(f"server {pid} is still running at the deadline")
                    return 1
        if not released(pidfd):
            return 1
        path.unlink()
        return ESCALATED if escalated else 0
    finally:
        os.close(pidfd)


def parent_death_signal():
    """The child gets SIGTERM when this helper dies, so no filler outlives it."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def lines(path):
    try:
        return Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []


def ended(path):
    """A filler line reporting a stream that finished or failed."""
    return next((line for line in lines(path) if line.startswith(("finished ", "failed "))), None)


def terminate(process, until):
    """Stops a child: SIGTERM, then SIGKILL, by `until`."""
    if process.poll() is not None:
        return process.returncode
    process.terminate()
    try:
        return process.wait(timeout=max(0.001, remaining(until) - KILL_RESERVE))
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return process.wait(timeout=max(0.001, remaining(until)))
        except subprocess.TimeoutExpired:
            return None


def measure(filler, log_path, command, deadline, ready_timeout=240.0, log=print):
    """Runs `command` while `filler` holds its streams (see the module doc)."""
    until = monotonic_deadline(deadline)
    with open(log_path, "w") as output:
        holder = subprocess.Popen(filler, stdout=output, stderr=subprocess.STDOUT,
                                  preexec_fn=parent_death_signal)
    try:
        ready_until = min(until, time.monotonic() + ready_timeout)
        while not any(line.startswith("ready ") for line in lines(log_path)):
            if holder.poll() is not None or ended(log_path) or time.monotonic() >= ready_until:
                log(f"filler did not become ready: {lines(log_path)[-3:]}")
                return 1
            time.sleep(min(0.2, remaining(ready_until)))
        log("filler ready")
        runner = subprocess.Popen(command, preexec_fn=parent_death_signal)
        try:
            broken = None
            while runner.poll() is None:
                if holder.poll() is not None:
                    broken = f"filler exited with status {holder.returncode}"
                elif line := ended(log_path):
                    broken = f"filler reported {line!r}"
                elif time.monotonic() >= until:
                    broken = "the deadline arrived"
                if broken:
                    log(f"measurement invalid: {broken} while the command ran")
                    terminate(runner, until)
                    return INVALID
                time.sleep(min(0.2, remaining(until)))
            status = runner.returncode
        finally:
            terminate(runner, until)
        # The filler must still hold every stream when the command ends.
        if holder.poll() is not None or ended(log_path):
            log(f"measurement invalid: the filler did not outlive the command "
                f"(status {holder.returncode}, {ended(log_path)!r})")
            return INVALID
        code = terminate(holder, until)
        if code != 0 or not any(line.startswith("stopped ") for line in lines(log_path)):
            log(f"measurement invalid: the filler did not acknowledge its stop (status {code})")
            return INVALID
        return status
    finally:
        terminate(holder, until)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    stopping = commands.add_parser("stop")
    stopping.add_argument("--pidfile", required=True)
    stopping.add_argument("--deadline", type=float, required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--deadline", type=float, required=True)
    measuring.add_argument("--filler-log", required=True)
    measuring.add_argument("--filler-arg", action="append", required=True,
                           help="one word of the filler command; repeat in order")
    measuring.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    args = parser.parse_args()
    if args.command == "stop":
        raise SystemExit(stop(args.pidfile, args.deadline))
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        parser.error("measure needs a command after --")
    raise SystemExit(measure(args.filler_arg, args.filler_log, command, args.deadline))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
