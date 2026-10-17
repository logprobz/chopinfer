"""CPU checks of the host-wide heavy-command queue (tools/dev/heavy.py):
admission order and limits, the gate, pause, nested admission, process-group
signals, CPU pinning, thread caps and per-worktree targets. Every process
these tests signal is one they started; all state is in a temporary
directory."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import termios
import time
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ROOT / "tools/dev/heavy.sh"
SPEC = importlib.util.spec_from_file_location("heavy", ROOT / "tools/dev/heavy.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
# siginfo si_code of a kernel-sent signal, such as a terminal's Ctrl-C.
SI_KERNEL = 0x80

# A heavy command stand-in: records that it started, then holds its slot
# until the test writes its release file.
HOLD = """
import json, os, pathlib, sys, time
name, directory = sys.argv[1], pathlib.Path(sys.argv[2])
facts = {"pid": os.getpid(), "nice": os.nice(0), "cpus": sorted(os.sched_getaffinity(0)),
         "env": {k: v for k, v in os.environ.items() if k.startswith(("CARGO", "CHOPINFER", "RUST", "NEXTEST", "OMP"))}}
(directory / (name + ".partial")).write_text(json.dumps(facts))
(directory / (name + ".partial")).rename(directory / (name + ".started"))
while not (directory / (name + ".release")).exists():
    time.sleep(0.02)
"""

# Ignores SIGTERM and starts a grandchild that ignores it too, as a stuck
# build step would.
RESIST = """
import pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""

# Records the si_code of each SIGINT it receives until its stop file
# appears. It takes them one at a time with sigtimedwait, so two deliveries
# stay two (a Python handler would run once for both).
COUNT = """
import json, pathlib, signal, sys
path = pathlib.Path(sys.argv[1])
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
received = []
path.write_text("ready")
while not path.with_suffix(".stop").exists():
    info = signal.sigtimedwait({signal.SIGINT}, 0.02)
    if info is not None:
        received.append(info.si_code)
path.write_text(json.dumps(received))
"""


# Holds slot.<slot>.lock as an older launcher or a stray descendant would,
# with a record of the given CPUs (JSON null: no record) and the pid given
# (0: its own), until its release file appears.
HOLDER = """
import fcntl, json, os, pathlib, sys, time
state, slot, cpus, pid, release = (pathlib.Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]),
                                   int(sys.argv[4]) or os.getpid(), pathlib.Path(sys.argv[5]))
lock = os.open(state / f"slot.{slot}.lock", os.O_RDONLY | os.O_CREAT, 0o664)
fcntl.flock(lock, fcntl.LOCK_EX)
if cpus is not None:
    start = int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]) if pid == os.getpid() else 1
    (state / f"slot.{slot}.json").write_text(json.dumps({"pid": pid, "start": start, "cpus": cpus}))
release.with_suffix(".held").write_text("")
while not release.exists():
    time.sleep(0.02)
"""


def gone(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state in ("Z", "X")


@unittest.skipUnless(len(os.sched_getaffinity(0)) >= 4, "needs four CPUs")
class Queue(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self.temporary.name)
        # The caller's own caps would be kept as lower values, so start clean.
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("CHOPINFER_HEAVY", "CHOPINFER_TARGET")) and k != "CARGO_TARGET_DIR"
                    and k not in LAUNCHER.THREAD_VARIABLES}
        self.env.update(CHOPINFER_HEAVY_DIR=str(self.dir / "state"), CHOPINFER_HEAVY_POLL_SECONDS="0.02",
                        CHOPINFER_HEAVY_SLOTS="2", CHOPINFER_HEAVY_SLOTS_DURING_GATE="1",
                        CHOPINFER_HEAVY_THREADS="1")
        self.processes = {}
        # Release files of processes the launcher did not start: HOLDERs and descendants.
        self.releases = []
        self.holders = []

    def tearDown(self):
        for name in self.processes:
            (self.dir / (name + ".release")).write_text("")
        for release in self.releases:
            release.write_text("")
        for holder in self.holders:
            holder.wait(timeout=20)
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()
            process.stderr.close()
        self.temporary.cleanup()

    def start(self, name, *options, env=None, command=None, launcher=None, affinity=None):
        argv = [*(launcher or [str(HEAVY)]), *options, "--label", name,
                *(command or [sys.executable, "-c", HOLD, name, str(self.dir)])]
        process = subprocess.Popen(argv, env={**self.env, **(env or {})}, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   preexec_fn=(lambda: os.sched_setaffinity(0, affinity)) if affinity else None)
        self.processes[name] = process
        return process

    def patched(self, patch):
        """The launcher, run with `patch` applied to its module first."""
        return [sys.executable, "-c",
                f"import importlib.util, os, pathlib, signal, sys\n"
                f"spec = importlib.util.spec_from_file_location('heavy', {str(ROOT / 'tools/dev/heavy.py')!r})\n"
                f"heavy = importlib.util.module_from_spec(spec); spec.loader.exec_module(heavy)\n"
                f"{patch}\n"
                f"sys.exit(heavy.main(sys.argv[1:]))\n"]

    def hold(self, name, slot, cpus, pid=0):
        """Starts a HOLDER of `slot`; returns once it holds the lock."""
        release = self.dir / (name + ".release")
        process = subprocess.Popen([sys.executable, "-c", HOLDER, str(self.dir / "state"), str(slot), json.dumps(cpus),
                                    str(pid), str(release)])
        self.releases.append(release)
        self.holders.append(process)
        deadline = time.monotonic() + 20
        while not release.with_suffix(".held").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(release.with_suffix(".held").exists())
        return process

    def records(self):
        path = self.dir / "state/runs.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def started(self, name, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (self.dir / (name + ".started")).exists():
                return json.loads((self.dir / (name + ".started")).read_text())
            time.sleep(0.02)
        return None

    def stays_waiting(self, *names):
        time.sleep(0.5)
        return all(not (self.dir / (name + ".started")).exists() for name in names)

    def release(self, name):
        (self.dir / (name + ".release")).write_text("")
        return self.processes[name].wait(timeout=20)

    def tickets(self):
        return sorted((self.dir / "state/queue").glob("*.json"))

    def labels(self):
        labels = []
        for path in self.tickets():
            try:
                labels.append(json.loads(path.read_text())["label"])
            except (OSError, ValueError):
                pass
        return labels

    def queued(self, count, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(self.tickets()) != count:
            time.sleep(0.02)
        return len(self.tickets()) == count

    def test_runs_beyond_the_slots_wait_and_start_in_ticket_order(self):
        self.start("a")
        self.start("b")
        self.assertTrue(self.started("a") and self.started("b"))
        self.start("c")
        self.assertTrue(self.queued(1))
        self.start("d")
        self.assertTrue(self.queued(2))
        self.assertTrue(self.stays_waiting("c", "d"))
        self.assertEqual(self.release("a"), 0)
        self.assertTrue(self.started("c"))
        self.assertTrue(self.stays_waiting("d"))
        self.release("b")
        self.assertTrue(self.started("d"))

    def test_a_gate_never_waits_and_lowers_the_limit_for_new_admissions_only(self):
        self.start("a")
        self.start("b")
        self.assertTrue(self.started("a") and self.started("b"))
        self.start("gate", "--gate")
        gate = self.started("gate")
        self.assertIsNotNone(gate, "a gate waits only for another gate")
        self.assertIsNone(self.processes["a"].poll())
        self.assertIsNone(self.processes["b"].poll())
        self.assertIn(":", gate["env"]["CHOPINFER_HEAVY_ADMITTED"])
        self.release("a")
        self.start("c")
        self.assertTrue(self.stays_waiting("c"), "one slot during a gate, and b holds it")
        self.release("b")
        self.assertTrue(self.started("c"))
        self.release("gate")

    def test_pause_holds_new_admissions_and_resume_keeps_their_order(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        subprocess.run([str(HEAVY), "--pause", "maintenance"], env=self.env, check=True)
        self.start("a", env=env)
        self.assertTrue(self.queued(1))
        self.start("b", env=env)
        self.assertTrue(self.queued(2))
        self.assertTrue(self.stays_waiting("a", "b"))
        subprocess.run([str(HEAVY), "--resume"], env=self.env, check=True)
        self.assertTrue(self.started("a"))
        self.assertTrue(self.stays_waiting("b"))
        self.release("a")
        self.assertTrue(self.started("b"))

    def test_a_dead_waiter_is_dropped_from_the_queue(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        self.start("a", env=env)
        self.assertTrue(self.started("a"))
        dead = self.start("dead", env=env)
        self.assertTrue(self.queued(1))
        os.killpg(dead.pid, signal.SIGKILL)
        dead.wait()
        self.assertEqual(len(self.tickets()), 1, "a killed waiter leaves its ticket")
        self.start("c", env=env)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and self.labels() != ["c"]:
            time.sleep(0.02)
        self.assertEqual(self.labels(), ["c"], "the next waiter drops the dead ticket")
        self.release("a")
        self.assertTrue(self.started("c"))
        self.assertEqual(self.tickets(), [])

    def test_a_waiter_that_is_signalled_leaves_the_queue(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        self.start("a", env=env)
        self.assertTrue(self.started("a"))
        waiter = self.start("b", env=env)
        self.assertTrue(self.queued(1))
        self.assertTrue(self.stays_waiting("b"))
        waiter.send_signal(signal.SIGINT)
        self.assertEqual(waiter.wait(timeout=20), 128 + signal.SIGINT)
        self.assertEqual(self.tickets(), [])
        record = json.loads((self.dir / "state/runs.jsonl").read_text().splitlines()[0])
        self.assertEqual((record["label"], record["slot"], record["exit"], record["ran"], record["abandoned"]),
                         ("b", None, 128 + signal.SIGINT, 0, True))
        self.assertGreaterEqual(record["waited"], 0.4)
        self.release("a")
        output = subprocess.run([str(HEAVY), "--status"], env={**self.env, **env}, capture_output=True, text=True,
                                check=True).stdout
        self.assertIn("lane runs in the last 24 h: 1,", output)
        self.assertIn("waits given up in the last 24 h: 1, longest", output)

    def test_a_second_signal_during_the_cancellation_is_held(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        self.start("a", env=env)
        self.assertTrue(self.started("a"))
        # The ticket's removal brings a SIGTERM, as a caller's timeout might.
        patch = ("real = pathlib.Path.unlink\n"
                 "def unlink(path, *args, **kwargs):\n"
                 "    real(path, *args, **kwargs)\n"
                 "    if path.parent.name == 'queue':\n"
                 "        os.kill(os.getpid(), signal.SIGTERM)\n"
                 "pathlib.Path.unlink = unlink")
        waiter = self.start("b", env=env, launcher=self.patched(patch))
        self.assertTrue(self.queued(1))
        waiter.send_signal(signal.SIGINT)
        self.assertEqual(waiter.wait(timeout=20), 128 + signal.SIGINT, waiter.stderr.read())
        self.assertEqual([(r["label"], r["exit"], r.get("abandoned")) for r in self.records()],
                         [("b", 128 + signal.SIGINT, True)])
        self.assertEqual(self.tickets(), [])

    def test_a_signal_during_the_admission_reaches_the_command_and_is_recorded(self):
        patch = ("real = heavy.lane_environment\n"
                 "def interrupted(*args):\n"
                 "    os.kill(os.getpid(), signal.SIGINT)\n"
                 "    return real(*args)\n"
                 "heavy.lane_environment = interrupted")
        process = self.start("a", launcher=self.patched(patch), command=["true"])
        code = process.wait(timeout=20)
        self.assertEqual([(r["label"], r["slot"], r["exit"], r.get("abandoned")) for r in self.records()],
                         [("a", 1, code, None)])

    def test_callers_with_different_layouts_never_share_a_cpu(self):
        # With one CPU per slot, an affinity of CPUs x, y gives slots x and y;
        # one of y, z gives y and z. "shifted" finds its slot 1 free by
        # number, but "second" runs on that CPU.
        x, y, z = sorted(os.sched_getaffinity(0))[:3]
        env = {"CHOPINFER_HEAVY_THREADS": "1"}
        self.start("first", env=env, affinity=[x, y])
        self.assertEqual(self.started("first")["cpus"], [x])
        self.start("second", env=env, affinity=[x, y])
        self.assertEqual(self.started("second")["cpus"], [y])
        self.release("first")
        self.start("shifted", env=env, affinity=[y, z])
        self.assertTrue(self.stays_waiting("shifted"))
        self.release("second")
        self.assertEqual(self.started("shifted")["cpus"], [y])

    def test_an_older_launchers_recorded_cpus_keep_an_overlapping_slot_waiting(self):
        env = {"CHOPINFER_HEAVY_THREADS": "1"}
        sets = LAUNCHER.cpu_sets(LAUNCHER.Settings({**self.env, **env}))
        # Slot 1, held by a launcher whose layout put it on this layout's slot 2.
        older = self.hold("older", 1, sets[2])
        self.start("new", env=env)
        self.assertTrue(self.stays_waiting("new"))
        (self.dir / "older.release").write_text("")
        older.wait(timeout=20)
        self.assertEqual(self.started("new")["cpus"], sets[1])

    def test_held_slots_without_a_live_record_count_both_layouts(self):
        settings = LAUNCHER.Settings({**self.env, "CHOPINFER_HEAVY_THREADS": "1"})
        allowed = sorted(os.sched_getaffinity(0))
        sets = {1: [1001], 2: [1002], 3: [1003], 4: [1004]}
        self.hold("live", 1, [2001])
        self.hold("unrecorded", 2, None)
        dead = subprocess.Popen(["true"])
        dead.wait()
        self.hold("orphaned", 3, [2003], pid=dead.pid)
        self.assertEqual(LAUNCHER.busy_cpus(settings, sets), {2001, 1002, allowed[1], 2003, 1003, allowed[2]})

    def test_a_descendant_that_keeps_the_slot_keeps_its_record(self):
        grand = ["bash", "-c", '"$0" -c "$1" grand "$2" &', sys.executable, HOLD, str(self.dir)]
        self.releases.append(self.dir / "grand.release")
        self.assertEqual(self.start("a", command=grand).wait(timeout=20), 0)
        self.assertIsNotNone(self.started("grand"))
        self.assertTrue(LAUNCHER.held(self.dir / "state/slot.1.lock"))
        record = json.loads((self.dir / "state/slot.1.json").read_text())
        self.assertEqual((record["label"], record["cpus"]), ("a", self.started("grand")["cpus"]))

    def test_a_nested_call_inherits_the_admission_and_a_forged_marker_queues(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        nested = (f"import subprocess, sys; sys.exit(subprocess.run([{str(HEAVY)!r}, sys.executable, '-c', "
                  f"{HOLD!r}, 'inner', {str(self.dir)!r}]).returncode)")
        self.start("outer", env=env, command=[sys.executable, "-c", nested])
        self.assertIsNotNone(self.started("inner"), "the nested call must not wait for a second slot")
        (self.dir / "inner.release").write_text("")
        self.assertEqual(self.processes["outer"].wait(timeout=20), 0)
        runs = (self.dir / "state/runs.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["label"] for line in runs], ["outer"])

        self.start("a", env=env)
        holder = self.started("a")
        # The holder's own marker, presented by a process it did not start.
        forged = {**env, "CHOPINFER_HEAVY_ADMITTED": holder["env"]["CHOPINFER_HEAVY_ADMITTED"]}
        self.start("forged", env=forged)
        self.assertTrue(self.stays_waiting("forged"))
        self.release("a")
        self.assertTrue(self.started("forged"))

    def test_signals_reach_the_command(self):
        # The caller's group signals reach a command that ignores SIGTERM, and
        # its grandchild, as tools/journal_proof/capture.py's stop_owned sends them.
        pid_file = self.dir / "grandchild.pid"
        process = self.start("resist", command=[sys.executable, "-c", RESIST, str(pid_file)])
        deadline = time.monotonic() + 20
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        grandchild = int(pid_file.read_text())
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(0.3)
        self.assertFalse(gone(grandchild), "SIGTERM is ignored")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)
        deadline = time.monotonic() + 10
        while not gone(grandchild) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(gone(grandchild))
        # A signal only the launcher receives goes on to the command.
        launcher = self.start("hold")
        self.assertTrue(self.started("hold"))
        launcher.send_signal(signal.SIGTERM)
        self.assertEqual(launcher.wait(timeout=20), 128 + signal.SIGTERM)

    def test_a_terminal_interrupt_reaches_the_command_once(self):
        # The terminal sends Ctrl-C to its whole foreground group, the command
        # included, so the launcher must not send it again (ci/runner.py's
        # stop would be cut short by a second one).
        count = self.dir / "count"
        controller, terminal = os.openpty()
        self.addCleanup(os.close, controller)

        def own_terminal():
            os.setsid()
            fcntl.ioctl(terminal, termios.TIOCSCTTY, 0)

        process = subprocess.Popen([str(HEAVY), sys.executable, "-c", COUNT, str(count)], env=self.env,
                                   stdin=terminal, stdout=terminal, stderr=terminal, preexec_fn=own_terminal)
        os.close(terminal)
        deadline = time.monotonic() + 20
        while not (count.exists() and count.read_text() == "ready") and time.monotonic() < deadline:
            time.sleep(0.02)
        os.write(controller, b"\x03")
        time.sleep(0.5)
        count.with_suffix(".stop").write_text("")
        self.assertEqual(process.wait(timeout=20), 0)
        self.assertEqual(json.loads(count.read_text()), [SI_KERNEL], "the terminal's only")

    def test_a_terminal_interrupt_while_the_command_starts_still_reaches_it(self):
        # Before the child exists the terminal's Ctrl-C reaches the launcher
        # alone, so a signal pending at the start goes on to the command.
        marker, count = self.dir / "spawning", self.dir / "count"
        spawn = (f"import importlib.util, pathlib, subprocess, sys, time\n"
                 f"spec = importlib.util.spec_from_file_location('heavy', {str(ROOT / 'tools/dev/heavy.py')!r})\n"
                 f"heavy = importlib.util.module_from_spec(spec); spec.loader.exec_module(heavy)\n"
                 f"real = subprocess.Popen\n"
                 f"class Slow(real):\n"
                 f"    def __init__(self, argv, *args, **kwargs):\n"
                 f"        if argv[0] == 'nice':\n"
                 f"            pathlib.Path({str(marker)!r}).write_text('')\n"
                 f"            time.sleep(1)\n"
                 f"        super().__init__(argv, *args, **kwargs)\n"
                 f"heavy.subprocess.Popen = Slow\n"
                 f"sys.exit(heavy.main(sys.argv[1:]))\n")
        controller, terminal = os.openpty()
        self.addCleanup(os.close, controller)

        def own_terminal():
            os.setsid()
            fcntl.ioctl(terminal, termios.TIOCSCTTY, 0)

        process = subprocess.Popen([sys.executable, "-c", spawn, sys.executable, "-c", COUNT, str(count)],
                                   env=self.env, stdin=terminal, stdout=terminal, stderr=terminal,
                                   preexec_fn=own_terminal)
        os.close(terminal)
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        os.write(controller, b"\x03")
        deadline = time.monotonic() + 20
        while not count.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.5)
        count.with_suffix(".stop").write_text("")
        code = process.wait(timeout=20)
        # The command dies of the SIGINT before it blocks it, or counts it.
        self.assertTrue(code == 128 + signal.SIGINT or json.loads(count.read_text()) != [],
                        f"exit {code}, command received {count.read_text() if count.exists() else 'nothing'}")

    def test_a_gate_and_a_pause_wait_for_an_admission_check_in_progress(self):
        state = self.dir / "state"
        state.mkdir()
        descriptor = os.open(state / "queue.lock", os.O_RDWR | os.O_CREAT, 0o664)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        pause = subprocess.Popen([str(HEAVY), "--pause", "maintenance"], env=self.env)
        self.start("gate", "--gate")
        time.sleep(0.5)
        self.assertIsNone(pause.poll(), "--pause returned while an admission check held queue.lock")
        self.assertIsNone(self.started("gate", timeout=0.1), "the gate started while an admission check ran")
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        self.assertEqual(pause.wait(timeout=20), 0)
        self.assertIsNotNone(self.started("gate"))
        subprocess.run([str(HEAVY), "--resume"], env=self.env, check=True)

    def test_a_nested_gate_runs_under_its_ancestors_gate(self):
        nested = subprocess.Popen([str(HEAVY), "--gate", str(HEAVY), "--gate", "true"], env=self.env,
                                  start_new_session=True)
        try:
            self.assertEqual(nested.wait(timeout=30), 0)
        except subprocess.TimeoutExpired:
            os.killpg(nested.pid, signal.SIGKILL)
            nested.wait()
            self.fail("the nested gate waited for its own ancestor's gate")

    def test_a_competing_claim_on_a_target_loses(self):
        # Another worktree claims the target between this run's check and its
        # claim; the claim must not overwrite it.
        root, target, other = self.dir / "targets", self.dir / "targets/shared", str(self.dir / "other")
        settings = LAUNCHER.Settings({**self.env, "CHOPINFER_TARGET_ROOT": str(root)})
        root.mkdir()
        link = os.link

        def racing(source, destination):
            Path(destination).write_text(other + "\n")
            link(source, destination)

        with unittest.mock.patch.object(LAUNCHER.os, "link", racing), \
             unittest.mock.patch.object(LAUNCHER, "worktree", lambda: str(self.dir / "mine")):
            with self.assertRaisesRegex(LAUNCHER.Refused, "belongs to worktree"):
                LAUNCHER.target_for(settings, {"CARGO_TARGET_DIR": str(target)})
        self.assertEqual(Path(str(target) + ".worktree").read_text().strip(), other)

    def test_the_run_log_is_group_writable(self):
        subprocess.run([str(HEAVY), "true"], env=self.env, check=True, preexec_fn=lambda: os.umask(0o022))
        self.assertEqual((self.dir / "state/runs.jsonl").stat().st_mode & 0o777, 0o664)

    @unittest.skipIf(os.geteuid() == 0, "root opens any file")
    def test_unreadable_state_never_over_admits_or_drops_a_waiter(self):
        state = self.dir / "state"
        state.mkdir()
        # A slot lock this user cannot open counts as held.
        (state / "slot.1.lock").touch(mode=0o000)
        self.start("a")
        self.assertEqual(self.started("a")["env"]["CHOPINFER_HEAVY_ADMITTED"].split(":")[0], "2")
        self.start("b")
        self.assertTrue(self.queued(1))
        self.assertTrue(self.stays_waiting("b"))
        # An unreadable ticket of a live waiter keeps its place.
        self.tickets()[0].chmod(0o000)
        self.start("c")
        self.assertTrue(self.queued(2))
        self.release("a")
        self.assertTrue(self.started("b"))
        self.assertTrue(self.stays_waiting("c"))

    @unittest.skipIf(os.geteuid() == 0, "root writes any file")
    def test_a_run_log_that_cannot_be_written_keeps_the_exit_status(self):
        state = self.dir / "state"
        state.mkdir()
        (state / "runs.jsonl").touch(mode=0o444)
        refused = subprocess.run([str(HEAVY), "bash", "-c", "exit 3"], env=self.env, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 3)
        self.assertIn("could not record the run", refused.stderr)

    def test_lane_runs_are_pinned_niced_and_capped(self):
        env = {"CHOPINFER_HEAVY_THREADS": "2", "CARGO_BUILD_JOBS": "64", "RUST_TEST_THREADS": "1"}
        self.start("a", env=env)
        facts = self.started("a")
        settings = LAUNCHER.Settings({**self.env, **env})
        sets = LAUNCHER.cpu_sets(settings)
        self.assertEqual(facts["cpus"], sets[1])
        self.assertEqual(facts["nice"], 19)
        self.assertEqual(facts["env"]["CARGO_BUILD_JOBS"], "2")
        self.assertEqual(facts["env"]["RUST_TEST_THREADS"], "1", "a lower value the caller set is kept")
        for name in ("NEXTEST_TEST_THREADS", "CHOPINFER_CI_JOBS", "CHOPINFER_KERNEL_BUILD_JOBS", "OMP_NUM_THREADS"):
            self.assertEqual(facts["env"][name], "2")
        self.assertTrue(facts["env"]["CHOPINFER_HEAVY_ADMITTED"].startswith("1:"))
        self.start("b", env=env)
        self.assertEqual(self.started("b")["cpus"], sets[2])

    def test_each_worktree_gets_its_own_target(self):
        root = self.dir / "targets"
        checkouts = []
        for name in ("one", "two"):
            checkout = self.dir / name
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            checkouts.append(os.path.realpath(checkout))
        env = {**self.env, "CHOPINFER_TARGET_ROOT": str(root)}
        show = [str(HEAVY), sys.executable, "-c", "import os; print(os.environ['CARGO_TARGET_DIR'])"]
        target = subprocess.run(show, env=env, cwd=checkouts[0], capture_output=True, text=True, check=True).stdout.strip()
        key = hashlib.sha256(checkouts[0].encode()).hexdigest()[:12]
        self.assertEqual(target, str(root / f"wt-one-{key}"))
        self.assertEqual(Path(target + ".worktree").read_text().strip(), checkouts[0])
        # Another worktree may not build into it.
        foreign = subprocess.run(show, env={**env, "CARGO_TARGET_DIR": target}, cwd=checkouts[1],
                                 capture_output=True, text=True)
        self.assertEqual(foreign.returncode, 2)
        self.assertIn("belongs to worktree", foreign.stderr)
        # A caller's own target outside the root is kept.
        private = str(self.dir / "private")
        kept = subprocess.run(show, env={**env, "CARGO_TARGET_DIR": private}, cwd=checkouts[1],
                              capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(kept, private)

    def test_exit_status_passes_through(self):
        for command, expected in ((["bash", "-c", "exit 3"], 3),
                                  ([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
                                   128 + signal.SIGTERM)):
            with self.subTest(command=command[-1]):
                self.assertEqual(subprocess.run([str(HEAVY), *command], env=self.env).returncode, expected)
        record = json.loads((self.dir / "state/runs.jsonl").read_text().splitlines()[0])
        self.assertEqual((record["class"], record["slot"], record["exit"]), ("lane", 1, 3))

    def test_status_lists_holders_the_pause_and_the_queue(self):
        env = {"CHOPINFER_HEAVY_SLOTS": "1"}
        self.start("holder", env=env)
        self.assertTrue(self.started("holder"))
        self.start("waiter", env=env)
        self.assertTrue(self.queued(1))
        subprocess.run([str(HEAVY), "--pause", "maintenance"], env=self.env, check=True)
        output = subprocess.run([str(HEAVY), "--status"], env={**self.env, **env}, capture_output=True, text=True,
                                check=True).stdout
        self.assertIn("paused: maintenance", output)
        self.assertIn("gate: free", output)
        self.assertRegex(output, r"slot 1: held .*holder")
        self.assertRegex(output, r"queue: 1 waiting\n  1\. waited .*waiter")
        subprocess.run([str(HEAVY), "--resume"], env=self.env, check=True)

    def test_refusals(self):
        missing = {k: v for k, v in self.env.items() if k != "CHOPINFER_HEAVY_DIR"}
        refused = subprocess.run([str(HEAVY), "true"], env=missing, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("CHOPINFER_HEAVY_DIR", refused.stderr)
        wide = {**self.env, "CHOPINFER_HEAVY_THREADS": str(len(os.sched_getaffinity(0)))}
        refused = subprocess.run([str(HEAVY), "true"], env=wide, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("CPUs", refused.stderr)


def cpu_list_text(cpus):
    """A sysfs CPU list, such as `0-5,24-29`."""
    runs = []
    for cpu in sorted(cpus):
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(f"{first}-{last}" if first != last else str(first) for first, last in runs)


def write_topology(root, cores, caches):
    """A sysfs CPU tree: `cores` lists each physical core's CPUs, `caches`
    each L3 cache's CPUs (empty for none)."""
    for core in cores:
        for cpu in core:
            (root / f"cpu{cpu}/topology").mkdir(parents=True)
            (root / f"cpu{cpu}/topology/thread_siblings_list").write_text(cpu_list_text(core) + "\n")
            for index, level in enumerate((1, 2)):
                (root / f"cpu{cpu}/cache/index{index}").mkdir(parents=True)
                (root / f"cpu{cpu}/cache/index{index}/level").write_text(f"{level}\n")
                (root / f"cpu{cpu}/cache/index{index}/shared_cpu_list").write_text(cpu_list_text(core) + "\n")
    for cache in caches:
        for cpu in cache:
            (root / f"cpu{cpu}/cache/index3").mkdir(parents=True)
            (root / f"cpu{cpu}/cache/index3/level").write_text("3\n")
            (root / f"cpu{cpu}/cache/index3/shared_cpu_list").write_text(cpu_list_text(cache) + "\n")


class CpuSets(unittest.TestCase):
    """Slot CPU sets on synthetic topologies."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def sets(self, slots, threads, cores, caches=(), allowed=None):
        root = self.root = self.dir / f"sys-{len(list(self.dir.iterdir()))}"
        root.mkdir()
        write_topology(root, cores, caches)
        settings = LAUNCHER.Settings({"CHOPINFER_HEAVY_DIR": str(self.dir / "state"),
                                      "CHOPINFER_HEAVY_SLOTS": str(slots), "CHOPINFER_HEAVY_THREADS": str(threads)})
        cpus = allowed if allowed is not None else [cpu for core in cores for cpu in core]
        return LAUNCHER.cpu_sets(settings, allowed=cpus, topology=root)

    def test_a_threadripper_gets_whole_cores_inside_each_cache(self):
        # 24 cores, SMT siblings n and n + 24, four L3 caches of six cores.
        cores = [[n, n + 24] for n in range(24)]
        caches = [list(range(6 * c, 6 * c + 6)) + list(range(24 + 6 * c, 30 + 6 * c)) for c in range(4)]
        sets = self.sets(5, 8, cores, caches)
        self.assertEqual((self.root / "cpu0/cache/index3/shared_cpu_list").read_text(), "0-5,24-29\n")
        for cache in caches:
            for cpu in cache:
                self.assertEqual(LAUNCHER.shared_l3(self.root / f"cpu{cpu}"), set(cache))
        self.assertEqual(sets, {1: [0, 1, 2, 3, 24, 25, 26, 27], 2: [6, 7, 8, 9, 30, 31, 32, 33],
                                3: [12, 13, 14, 15, 36, 37, 38, 39], 4: [18, 19, 20, 21, 42, 43, 44, 45],
                                5: [4, 5, 10, 11, 28, 29, 34, 35]})
        used = [cpu for cpus in sets.values() for cpu in cpus]
        self.assertEqual(len(used), len(set(used)), "no CPU is in two slots")
        self.assertEqual(sorted(set(range(48)) - set(used)), [16, 17, 22, 23, 40, 41, 46, 47])
        self.assertEqual(sorted(cpu for cpus in self.sets(6, 8, cores, caches).values() for cpu in cpus),
                         list(range(48)))

    def test_slots_that_fit_no_cache_take_the_largest_remainders_first(self):
        # Caches of 5, 1 and 3 cores: slot 2 takes the 3-core remainder and one
        # more core, two caches, rather than CPUs 4-7 across three.
        sets = self.sets(2, 4, [[n] for n in range(9)], [list(range(5)), [5], list(range(6, 9))])
        self.assertEqual(sets, {1: [0, 1, 2, 3], 2: [4, 6, 7, 8]})
        sets = self.sets(3, 4, [[n] for n in range(12)], [list(range(6)), list(range(6, 9)), list(range(9, 12))])
        self.assertEqual(sets, {1: [0, 1, 2, 3], 2: [6, 7, 8, 9], 3: [4, 5, 10, 11]})

    def test_siblings_stay_together_without_cache_information(self):
        self.assertEqual(self.sets(2, 4, [[n, n + 4] for n in range(4)]), {1: [0, 1, 4, 5], 2: [2, 3, 6, 7]})

    def test_an_affinity_without_siblings_uses_one_cpu_per_core(self):
        cores = [[n, n + 8] for n in range(8)]
        caches = [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
        self.assertEqual(self.sets(2, 4, cores, caches, allowed=list(range(8))), {1: [0, 1, 2, 3], 2: [4, 5, 6, 7]})

    def test_consecutive_cpus_when_whole_cores_cannot_make_a_slot(self):
        cores = [[n, n + 4] for n in range(4)]
        self.assertEqual(self.sets(2, 3, cores), {1: [0, 1, 2], 2: [3, 4, 5]}, "3 CPUs is no whole number of cores")
        self.assertEqual(self.sets(2, 2, cores, allowed=[0, 1, 2, 3, 4]), {1: [0, 1], 2: [2, 3]},
                         "one sibling of the affinity is missing, so cores differ in size")
        settings = LAUNCHER.Settings({"CHOPINFER_HEAVY_DIR": str(self.dir / "state"), "CHOPINFER_HEAVY_SLOTS": "2",
                                      "CHOPINFER_HEAVY_THREADS": "2"})
        self.assertEqual(LAUNCHER.cpu_sets(settings, allowed=[0, 1, 2, 3], topology=self.dir / "missing"),
                         {1: [0, 1], 2: [2, 3]}, "no topology")

    def test_cpu_lists(self):
        self.assertEqual(LAUNCHER.cpu_list("0-2,24-25,40\n"), {0, 1, 2, 24, 25, 40})


if __name__ == "__main__":
    unittest.main()
