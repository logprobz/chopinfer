"""One host-wide queue for heavy CPU commands on a shared build host.

Every cargo build or test, `ci/quick.sh`, `ci/cpu.sh`, journal capture build
and qualification run on the host goes through this launcher, so the host
runs a known number of them at once. Usage:

    heavy.py [--label TEXT] COMMAND...   run COMMAND in a lane slot
    heavy.py --gate COMMAND...           run COMMAND ahead of lane jobs
    heavy.py --status                    holders, queue, waits, target size
    heavy.py --pause REASON | --resume   hold or release new lane admissions

State lives in `CHOPINFER_HEAVY_DIR`. A lane run writes a ticket under
`queue/` and waits, in ticket order, for one of `CHOPINFER_HEAVY_SLOTS`
(default 4) slots, or `CHOPINFER_HEAVY_SLOTS_DURING_GATE` (default 2) while a
gate holds `gate.lock`; a gate that starts later stops no running lane job.
While the `paused` file exists no lane job is admitted. A gate run waits only
for another gate. Every caller of one state directory must use the same
slot and thread settings (set them in the host's shared local.env).

An admitted lane command runs pinned to its slot's own set of
`CHOPINFER_HEAVY_THREADS` (default 8) CPUs, under `nice -n 19 ionice -c3`,
with the Cargo, test and native thread-pool variables capped at that count,
so the worker pools it starts (nvcc workers, nextest groups, BLAS and OpenMP
pools) stay inside it. It runs in the launcher's own process group, so a
caller that signals that group (tools/journal_proof/capture.py) reaches the
whole command. The command inherits the slot's lock, so the slot stays held
while the command, or any descendant that keeps the descriptor, lives. A
signal only the launcher receives goes on to the command; a terminal's
signal, which already reached the whole foreground group, is not sent again.
A command started under an admitted run or gate (marked by
`CHOPINFER_HEAVY_ADMITTED`) runs directly, whatever its class, instead of
waiting for a second slot or for its own ancestor's gate.

With `CHOPINFER_TARGET_ROOT` set and `CARGO_TARGET_DIR` unset, the command
builds into one target per worktree under the root: Cargo judges freshness
by source mtimes, so a target shared between worktrees can test another
checkout's code. The launcher never deletes a target; `--status` reports the
root's size. Each finished run appends one line to `runs.jsonl`.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import time

MARKER = "CHOPINFER_HEAVY_ADMITTED"
THREAD_VARIABLES = ("CARGO_BUILD_JOBS", "RUST_TEST_THREADS", "NEXTEST_TEST_THREADS", "CHOPINFER_CI_JOBS",
                    "CHOPINFER_CI_TEST_THREADS", "CHOPINFER_KERNEL_BUILD_JOBS", "CHOPINFER_GEOMETRY_JOBS",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# siginfo si_code of a signal the kernel sent, as a terminal's Ctrl-C is.
SI_KERNEL = 0x80
# State is shared by the host's users: files group-writable, directories setgid.
FILE_MODE, DIRECTORY_MODE = 0o664, 0o2775


class Refused(Exception):
    """A configuration the launcher will not run under (exit 2)."""


def number(env, name, default, minimum):
    value = env.get(name, "")
    try:
        result = int(value) if value else default
    except ValueError:
        raise Refused(f"{name} must be an integer, not {value!r}")
    if result < minimum:
        raise Refused(f"{name} must be at least {minimum}")
    return result


def share(path, mode):
    """Opens up a state file or directory this user created; others' stay as they are."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def make_directory(path):
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        share(path, DIRECTORY_MODE)


class Settings:
    def __init__(self, env):
        if not env.get("CHOPINFER_HEAVY_DIR"):
            raise Refused("set CHOPINFER_HEAVY_DIR to the shared state directory (see docs/development-loop.md)")
        self.dir = pathlib.Path(env["CHOPINFER_HEAVY_DIR"])
        self.slots = number(env, "CHOPINFER_HEAVY_SLOTS", 4, 1)
        self.during_gate = min(number(env, "CHOPINFER_HEAVY_SLOTS_DURING_GATE", 2, 0), self.slots)
        self.threads = number(env, "CHOPINFER_HEAVY_THREADS", 8, 1)
        try:
            self.poll = float(env.get("CHOPINFER_HEAVY_POLL_SECONDS") or 2)
        except ValueError:
            raise Refused("CHOPINFER_HEAVY_POLL_SECONDS must be a number")
        self.target_root = pathlib.Path(env["CHOPINFER_TARGET_ROOT"]) if env.get("CHOPINFER_TARGET_ROOT") else None
        self.warn_gb = number(env, "CHOPINFER_TARGET_WARN_GB", 250, 0)
        make_directory(self.dir)
        make_directory(self.dir / "queue")

    def path(self, name):
        return self.dir / name


def start_time(pid):
    """The process's start time in clock ticks, or None when it is gone or hidden."""
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] in ("Z", "X") else int(fields[19])


def alive(pid, start):
    """Whether the process `pid` that started at `start` still runs."""
    seen = start_time(pid)
    if seen is not None:
        return seen == start
    try:
        os.kill(pid, 0)
    except PermissionError:
        # Another user's process that /proc hides (hidepid): assume it is the same one.
        return True
    except OSError:
        return False
    return False


def parent(pid):
    try:
        return int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    share(temporary, FILE_MODE)
    os.replace(temporary, path)


def open_lock(path):
    """A descriptor for flock on `path`, or None when this user cannot open it.
    flock needs no write access, so a lock another user made stays usable."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, FILE_MODE)
    except OSError:
        return None
    try:
        if os.fstat(descriptor).st_uid == os.getuid():
            os.fchmod(descriptor, FILE_MODE)
    except OSError:
        pass
    return descriptor


def held(path):
    """Whether another process holds the lock on `path`. Call under queue.lock:
    a free lock is taken and released at once. A lock this user cannot open
    counts as held, so the queue never over-admits."""
    descriptor = open_lock(path)
    if descriptor is None:
        return True
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


@contextlib.contextmanager
def queue_mutex(settings):
    """Holds queue.lock, the lock every admission check runs under."""
    descriptor = open_lock(settings.path("queue.lock"))
    if descriptor is None:
        raise Refused(f"cannot open {settings.path('queue.lock')}; make {settings.dir} group-writable")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def ticket_owner(path):
    """(pid, start) from a ticket's name, `<order>-<pid>-<start>.json`, or None."""
    parts = path.stem.split("-")
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return None


def inherited(env):
    """True when an ancestor's admitted run (or gate) holds the marker's slot."""
    try:
        slot, pid, start = env.get(MARKER, "").split(":")
        pid, start = int(pid), int(start)
    except ValueError:
        return False
    if start_time(pid) != start:
        return False
    ancestor = os.getppid()
    while ancestor > 1 and ancestor != pid:
        ancestor = parent(ancestor)
    if ancestor != pid:
        return False
    settings = Settings(env)
    info = read_json(settings.path("gate.json" if slot == "gate" else f"slot.{slot}.json"))
    return bool(info) and info.get("pid") == pid and info.get("start") == start


def worktree():
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        top = ""
    return os.path.realpath(top or os.getcwd())


def target_for(settings, env):
    """The CARGO_TARGET_DIR to run with, or None to leave it unset."""
    top = worktree()
    if env.get("CARGO_TARGET_DIR"):
        target = pathlib.Path(env["CARGO_TARGET_DIR"])
    elif settings.target_root is not None:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(top)) or "root"
        target = settings.target_root / f"wt-{name}-{hashlib.sha256(top.encode()).hexdigest()[:12]}"
    else:
        return None
    if settings.target_root is not None:
        resolved = os.path.realpath(target)
        if resolved.startswith(os.path.realpath(settings.target_root) + os.sep):
            owner = pathlib.Path(resolved + ".worktree")
            if not owner.exists():
                # link() creates the owner file whole or not at all, so of two
                # worktrees claiming one target at once exactly one wins.
                owner.parent.mkdir(parents=True, exist_ok=True)
                claim = owner.with_name(owner.name + f".{os.getpid()}.tmp")
                claim.write_text(top + "\n")
                try:
                    os.link(claim, owner)
                except FileExistsError:
                    pass
                finally:
                    claim.unlink(missing_ok=True)
            recorded = owner.read_text().strip()
            if recorded != top:
                raise Refused(f"{target} belongs to worktree {recorded}; give each worktree its own target")
    return target


def lane_environment(settings, env, slot, pid, start):
    env = dict(env)
    for name in THREAD_VARIABLES:
        try:
            keep = 0 < int(env.get(name, "")) <= settings.threads
        except ValueError:
            keep = False
        if not keep:
            env[name] = str(settings.threads)
    env[MARKER] = f"{slot}:{pid}:{start}"
    return env


def cpu_sets(settings):
    allowed = sorted(os.sched_getaffinity(0))
    if settings.slots * settings.threads > len(allowed):
        raise Refused(f"{settings.slots} slots of {settings.threads} CPUs need more than the {len(allowed)} "
                      "CPUs this process may use; lower CHOPINFER_HEAVY_SLOTS or CHOPINFER_HEAVY_THREADS")
    return {slot: allowed[(slot - 1) * settings.threads:slot * settings.threads] for slot in range(1, settings.slots + 1)}


def record(settings, entry):
    """Appends one run to runs.jsonl; a failure to log never changes the run's status."""
    line = json.dumps(entry, sort_keys=True)
    try:
        descriptor = os.open(settings.path("runs.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, FILE_MODE)
        try:
            if os.fstat(descriptor).st_uid == os.getuid():
                os.fchmod(descriptor, FILE_MODE)
            os.write(descriptor, (line[:3900] + "\n").encode())
        finally:
            os.close(descriptor)
    except OSError as error:
        print(f"heavy: could not record the run in {settings.path('runs.jsonl')}: {error}", file=sys.stderr)


def describe(command):
    return " ".join(command[:8])[:300]


class Stop(Exception):
    def __init__(self, number):
        self.number = number


def interrupt(number, _frame):
    raise Stop(number)


def wait_for_slot(settings, ticket, info, slots):
    """Waits in ticket order; returns (slot, open lock descriptor) with the ticket removed."""
    queue_lock = open_lock(settings.path("queue.lock"))
    if queue_lock is None:
        raise Refused(f"cannot open {settings.path('queue.lock')}; make {settings.dir} group-writable")
    try:
        while True:
            fcntl.flock(queue_lock, fcntl.LOCK_EX)
            try:
                admitted = admit(settings, ticket, info, slots)
            finally:
                fcntl.flock(queue_lock, fcntl.LOCK_UN)
            if admitted is not None:
                return admitted
            time.sleep(settings.poll)
    finally:
        os.close(queue_lock)


def admit(settings, ticket, info, slots):
    """One admission check, under queue.lock: (slot, descriptor) or None.
    The pause check is here too, so no admission follows a --pause that
    returned, and a gate takes queue.lock once it holds gate.lock."""
    if settings.path("paused").exists():
        return None
    if not ticket.exists():
        # Removed by hand: put it back in its place.
        write_json(ticket, info)
    waiting = []
    for path in sorted(settings.path("queue").glob("*.json")):
        owner = ticket_owner(path)
        if owner is not None and not alive(*owner):
            path.unlink(missing_ok=True)
        elif owner is not None:
            waiting.append(path)
    limit = settings.during_gate if held(settings.path("gate.lock")) else settings.slots
    free = [slot for slot in slots if not held(settings.path(f"slot.{slot}.lock"))]
    room = limit - (len(slots) - len(free))
    if ticket not in waiting or waiting.index(ticket) >= room:
        return None
    for slot in free:
        descriptor = open_lock(settings.path(f"slot.{slot}.lock"))
        if descriptor is None:
            continue
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        ticket.unlink(missing_ok=True)
        return slot, descriptor
    return None


def run(settings, command, label, gate):
    for number in SIGNALS:
        signal.signal(number, interrupt)
    ticket = None
    try:
        env = dict(os.environ)
        target = target_for(settings, env)
        if target is not None:
            env["CARGO_TARGET_DIR"] = str(target)
        pid, start = os.getpid(), start_time(os.getpid())
        asked = time.time()
        info = {"pid": pid, "start": start, "label": label, "cwd": os.getcwd(), "command": describe(command),
                "asked": asked, "target": str(target) if target else None}
        if gate:
            slot = "gate"
            descriptor = open_lock(settings.path("gate.lock"))
            if descriptor is None:
                raise Refused(f"cannot open {settings.path('gate.lock')}; make {settings.dir} group-writable")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            # An admission check in progress finishes first; every later one
            # sees the gate.
            with queue_mutex(settings):
                pass
            env[MARKER] = f"gate:{pid}:{start}"
            argv, cpus = command, None
        else:
            sets = cpu_sets(settings)
            ticket = settings.path("queue") / f"{time.monotonic_ns():020d}-{pid}-{start}.json"
            write_json(ticket, info)
            slot, descriptor = wait_for_slot(settings, ticket, info, sorted(sets))
            ticket = None
            env = lane_environment(settings, env, slot, pid, start)
            argv, cpus = ["nice", "-n", "19", "ionice", "-c3", *command], sets[slot]
        # Held from here on and read below, so a signal can neither strand a
        # child nor reach the command twice.
        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    except Stop as stop:
        if ticket is not None:
            ticket.unlink(missing_ok=True)
        return 128 + stop.number
    admitted = time.time()
    info.update(slot=slot, admitted=admitted, cpus=cpus)
    info_path = settings.path("gate.json" if gate else f"slot.{slot}.json")
    write_json(info_path, info)
    os.set_inheritable(descriptor, True)

    def prepare():
        signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGNALS)
        if cpus:
            os.sched_setaffinity(0, cpus)

    try:
        child = subprocess.Popen(argv, env=env, pass_fds=(descriptor,), preexec_fn=prepare)
    except OSError as error:
        info_path.unlink(missing_ok=True)
        print(f"heavy: cannot start {command[0]}: {error}", file=sys.stderr)
        return 127
    # Signals pending now arrived before the child existed, terminal ones
    # included, so they reached the launcher alone.
    while (received := signal.sigtimedwait(SIGNALS, 0)) is not None:
        try:
            child.send_signal(received.si_signo)
        except ProcessLookupError:
            pass
    while child.poll() is None:
        received = signal.sigtimedwait(SIGNALS, 0.1)
        # A terminal's signal went to the whole foreground group, the command
        # included. Any other may have been sent to the launcher alone.
        if received is not None and received.si_code != SI_KERNEL:
            try:
                child.send_signal(received.si_signo)
            except ProcessLookupError:
                pass
    status = child.returncode
    finished = time.time()
    info_path.unlink(missing_ok=True)
    os.close(descriptor)
    code = status if status >= 0 else 128 - status
    record(settings, {"class": "gate" if gate else "lane", "slot": slot, "label": label, "cwd": info["cwd"],
                      "command": info["command"], "waited": round(admitted - asked, 1),
                      "ran": round(finished - admitted, 1), "exit": code, "finished": round(finished)})
    return code


def target_size(root):
    try:
        output = subprocess.run(["du", "-s", "--block-size=1G", str(root)], capture_output=True, text=True,
                                timeout=60).stdout
        return int(output.split()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def status(settings, out):
    now = time.time()
    paused = settings.path("paused")
    if paused.exists():
        out.write(f"paused: {paused.read_text().strip() or 'no reason given'}\n")
    with queue_mutex(settings):
        holders = [("gate", settings.path("gate.lock"), settings.path("gate.json"))]
        holders += [(f"slot {n}", settings.path(f"slot.{n}.lock"), settings.path(f"slot.{n}.json"))
                    for n in range(1, settings.slots + 1)]
        for name, lock, info_path in holders:
            if not held(lock):
                out.write(f"{name}: free\n")
                continue
            info = read_json(info_path) or {}
            ran = f"{now - info['admitted']:.0f} s" if "admitted" in info else "unknown time"
            out.write(f"{name}: held {ran}, pid {info.get('pid', '?')}, {info.get('label') or ''} "
                      f"{info.get('cwd', '')}: {info.get('command', '(no info)')}\n")
    waiting = []
    for path in sorted(settings.path("queue").glob("*.json")):
        owner = ticket_owner(path)
        if owner is not None and alive(*owner):
            waiting.append((path, read_json(path) or {}))
    out.write(f"queue: {len(waiting)} waiting\n")
    for position, (path, entry) in enumerate(waiting, 1):
        waited = f"{now - entry['asked']:.0f} s" if "asked" in entry else "?"
        out.write(f"  {position}. waited {waited}, pid {ticket_owner(path)[0]}, {entry.get('label') or ''} "
                  f"{entry.get('cwd', '')}: {entry.get('command', '(unreadable ticket)')}\n")
    waits = []
    try:
        lines = settings.path("runs.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("class") == "lane" and entry.get("finished", 0) > now - 86400 and "waited" in entry:
            waits.append(entry["waited"])
    if waits:
        out.write(f"lane runs in the last 24 h: {len(waits)}, median wait {statistics.median(waits):.0f} s, "
                  f"max {max(waits):.0f} s\n")
    if settings.target_root is not None:
        size = target_size(settings.target_root)
        if size is None:
            out.write(f"targets under {settings.target_root}: size unknown\n")
        else:
            warning = f" (over {settings.warn_gb} GB: delete idle targets)" if size > settings.warn_gb else ""
            out.write(f"targets under {settings.target_root}: {size} GB{warning}\n")
    return 0


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gate", action="store_true", help="run COMMAND ahead of lane jobs (waits only for another gate)")
    parser.add_argument("--label", default="", help="short description shown by --status")
    parser.add_argument("--status", action="store_true", help="print holders, the queue and recent waits")
    parser.add_argument("--pause", metavar="REASON", help="admit no lane job until --resume")
    parser.add_argument("--resume", action="store_true", help="admit lane jobs again")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if command and inherited(os.environ):
            os.execvp(command[0], command)
        settings = Settings(os.environ)
        if args.status:
            return status(settings, sys.stdout)
        if args.pause is not None:
            with queue_mutex(settings):
                settings.path("paused").write_text(args.pause + "\n")
                share(settings.path("paused"), FILE_MODE)
            return 0
        if args.resume:
            with queue_mutex(settings):
                settings.path("paused").unlink(missing_ok=True)
            return 0
        if not command:
            parser.error("give a command to run, or --status, --pause or --resume")
        return run(settings, command, args.label, args.gate)
    except Refused as refusal:
        print(f"heavy: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
