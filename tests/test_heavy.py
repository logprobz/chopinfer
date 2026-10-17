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

    def tearDown(self):
        for name in self.processes:
            (self.dir / (name + ".release")).write_text("")
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()
            process.stderr.close()
        self.temporary.cleanup()

    def start(self, name, *options, env=None, command=None):
        argv = [str(HEAVY), *options, "--label", name, *(command or [sys.executable, "-c", HOLD, name, str(self.dir)])]
        process = subprocess.Popen(argv, env={**self.env, **(env or {})}, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes[name] = process
        return process

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
        self.assertEqual(facts["cpus"], sorted(os.sched_getaffinity(0))[:2])
        self.assertEqual(facts["nice"], 19)
        self.assertEqual(facts["env"]["CARGO_BUILD_JOBS"], "2")
        self.assertEqual(facts["env"]["RUST_TEST_THREADS"], "1", "a lower value the caller set is kept")
        for name in ("NEXTEST_TEST_THREADS", "CHOPINFER_CI_JOBS", "CHOPINFER_KERNEL_BUILD_JOBS", "OMP_NUM_THREADS"):
            self.assertEqual(facts["env"][name], "2")
        self.assertTrue(facts["env"]["CHOPINFER_HEAVY_ADMITTED"].startswith("1:"))
        self.start("b", env=env)
        self.assertEqual(self.started("b")["cpus"], sorted(os.sched_getaffinity(0))[2:4])

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


if __name__ == "__main__":
    unittest.main()
