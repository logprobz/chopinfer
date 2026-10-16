"""CPU checks of the concurrency process control
(tools/dev/concurrency-control.py): owned server stops under the deadline
and fillers that must hold a measurement. Every process these tests signal
is one they started."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "concurrency_control", ROOT / "tools/dev/concurrency-control.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)

# A server stand-in: optionally ignores SIGTERM, then sleeps.
SERVER = """
import signal, sys, time
if sys.argv[1] == "resist":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("up", flush=True)
time.sleep(60)
"""

# A filler stand-in: `ready` after a moment, then per mode: hold (and
# acknowledge SIGTERM), die (SIGKILL itself), finish (report an ended
# stream), mute (die on SIGTERM without acknowledging), never (no ready).
FILLER = """
import os, signal, sys, time
mode = sys.argv[1]
def stopped(*_):
    print("stopped 3", flush=True)
    os._exit(0)
if mode != "mute":
    signal.signal(signal.SIGTERM, stopped)
print("started 3 streams", flush=True)
if mode != "never":
    time.sleep(0.2)
    print("ready 3", flush=True)
time.sleep(0.5)
if mode == "die":
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "finish":
    print("finished 1 40", flush=True)
time.sleep(60)
"""


def start_time(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class Stop(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.pidfile = Path(self.dir.name, "server.pid")
        self.released = []
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()

    def server(self, mode="plain", token="owned"):
        env = dict(os.environ, CHOPINFER_DEV_REDEPLOY_TOKEN=token)
        child = subprocess.Popen([sys.executable, "-c", SERVER, mode], env=env,
                                 stdout=subprocess.PIPE)
        self.children.append(child)
        child.stdout.readline()
        return child

    def write(self, pid, started, token):
        self.pidfile.write_text(f"{pid} {started} {token}\n")

    def wait_released(self, pid, pidfd, serve_args, until):
        self.released.append((pid, CONTROL.exited(pid)))

    def stop(self, deadline_in=30.0, **kwargs):
        return CONTROL.stop(str(self.pidfile), time.time() + deadline_in,
                            kwargs.get("wait_released", self.wait_released), log=lambda _: None,
                            clock=kwargs.get("clock", time.monotonic))

    def test_stale_ownership_never_signals(self):
        child = self.server()
        for started, token in ((str(int(start_time(child.pid)) + 1), "owned"),
                               (start_time(child.pid), "other")):
            self.write(child.pid, started, token)
            self.assertEqual(self.stop(), 1)
            time.sleep(0.2)
            self.assertIsNone(child.poll(), f"{started} {token} signalled the process")
            self.assertTrue(self.pidfile.exists())
        self.assertEqual(self.released, [])
        self.pidfile.write_text("garbage\n")
        self.assertEqual(self.stop(), 1)
        self.assertIsNone(child.poll())

    def test_an_owned_server_stops_and_its_pidfile_goes_after_release(self):
        child = self.server()
        self.write(child.pid, start_time(child.pid), "owned")
        self.assertEqual(self.stop(), 0)
        self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
        self.assertEqual(self.released, [(child.pid, True)])
        self.assertFalse(self.pidfile.exists())

    def test_an_absent_server_only_waits_for_release(self):
        child = self.server()
        pid, started = child.pid, start_time(child.pid)
        child.kill()
        child.wait()
        self.write(pid, started, "owned")
        self.assertEqual(self.stop(), 0)
        self.assertEqual(self.released, [(pid, True)])
        self.assertFalse(self.pidfile.exists())

    # The path is decided from one clock reading before any signal, so these
    # exit statuses do not depend on how fast a loaded host runs the child.
    # Elapsed-time bounds only show that no SIGTERM grace was spent.

    def test_near_deadline_escalates_at_once_and_stays_within_it(self):
        child = self.server("resist")
        self.write(child.pid, start_time(child.pid), "owned")
        began = time.monotonic()
        # Less than the escalation reserve is left: SIGKILL at once.
        self.assertEqual(self.stop(deadline_in=CONTROL.KILL_RESERVE), CONTROL.ESCALATED)
        self.assertLess(time.monotonic() - began, CONTROL.TERM_GRACE / 4)
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)
        self.assertFalse(self.pidfile.exists())

    def test_the_path_follows_the_clock_read_before_signalling(self):
        # A clock one second short of the deadline escalates even though the
        # waits still have the real 30 seconds.
        child = self.server("resist")
        self.write(child.pid, start_time(child.pid), "owned")
        self.assertEqual(self.stop(clock=lambda: time.monotonic() + 29.0), CONTROL.ESCALATED)
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)
        self.assertFalse(self.pidfile.exists())
        # A clock past the deadline kills without waiting and keeps the pidfile.
        released = len(self.released)
        child = self.server("resist")
        self.write(child.pid, start_time(child.pid), "owned")
        self.assertEqual(self.stop(clock=lambda: time.monotonic() + 31.0), 1)
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)
        self.assertTrue(self.pidfile.exists())
        self.assertEqual(len(self.released), released)

    def test_an_expired_deadline_signals_and_keeps_the_pidfile(self):
        child = self.server("resist")
        self.write(child.pid, start_time(child.pid), "owned")
        began = time.monotonic()
        self.assertEqual(self.stop(deadline_in=0.0), 1)
        self.assertLess(time.monotonic() - began, CONTROL.TERM_GRACE / 4)
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)
        self.assertTrue(self.pidfile.exists())
        self.assertEqual(self.released, [])

    def test_a_release_timeout_keeps_the_pidfile_for_a_retry(self):
        child = self.server()
        self.write(child.pid, start_time(child.pid), "owned")

        def timeout(*_):
            raise TimeoutError("GPU 0 still held")

        self.assertEqual(self.stop(wait_released=timeout), 1)
        self.assertTrue(self.pidfile.exists())


class Measure(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.log = Path(self.dir.name, "filler.txt")

    # Readiness and ends are read from the filler's own lines, so outcomes do
    # not depend on timing; the generous deadlines and bounds only allow for
    # a loaded host starting Python slowly.
    def measure(self, mode, command_seconds=1.5, command_status=0, deadline_in=120.0,
                ready_timeout=60.0):
        command = [sys.executable, "-c",
                   f"import time, sys; time.sleep({command_seconds}); sys.exit({command_status})"]
        began = time.monotonic()
        status = CONTROL.measure([sys.executable, "-c", FILLER, mode], str(self.log), command,
                                 time.time() + deadline_in, ready_timeout=ready_timeout,
                                 log=lambda _: None)
        return status, time.monotonic() - began

    def test_a_holding_filler_is_stopped_with_an_acknowledgement(self):
        status, _ = self.measure("hold")
        self.assertEqual(status, 0)
        self.assertIn("stopped 3", self.log.read_text())

    def test_a_filler_killed_after_readiness_invalidates_the_measurement(self):
        status, seconds = self.measure("die", command_seconds=600)
        self.assertEqual(status, CONTROL.INVALID)
        self.assertLess(seconds, 300, "the command runs on after the filler died")

    def test_an_ended_stream_invalidates_the_measurement(self):
        status, _ = self.measure("finish", command_seconds=600)
        self.assertEqual(status, CONTROL.INVALID)

    def test_a_filler_that_dies_without_acknowledging_its_stop_is_invalid(self):
        status, _ = self.measure("mute")
        self.assertEqual(status, CONTROL.INVALID)

    def test_a_failed_command_keeps_its_status(self):
        status, _ = self.measure("hold", command_status=7)
        self.assertEqual(status, 7)

    def test_a_filler_that_never_becomes_ready_runs_nothing(self):
        status, _ = self.measure("never", ready_timeout=2.0)
        self.assertEqual(status, 1)

    def test_the_deadline_ends_a_measurement(self):
        status, seconds = self.measure("hold", command_seconds=600, deadline_in=30.0)
        self.assertEqual(status, CONTROL.INVALID)
        self.assertLess(seconds, 300)


if __name__ == "__main__":
    unittest.main()
