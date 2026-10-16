"""CPU evidence for the GPU runner's deadline and failure handling."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from run_kv_offload_lab import run_process, utc_deadline


class RunnerTests(unittest.TestCase):
    def test_deadline_kills_and_reaps_only_launched_pid(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "deadline.log"
            with self.assertRaises(TimeoutError):
                run_process([sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(30)"],
                            os.environ.copy(), log, time.time() + 0.5)
            pid = int(log.read_text().strip())
            with self.assertRaises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

    def test_nonzero_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                run_process([sys.executable, "-c", "raise SystemExit(7)"],
                            os.environ.copy(), Path(folder) / "failed.log", time.time() + 5)

    def test_expired_slot_launches_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "absent.log"
            with self.assertRaises(TimeoutError):
                run_process(["no-such-executable"], os.environ.copy(), log, time.time() - 1)
            self.assertFalse(log.exists())

    def test_gpu_run_requires_authorization_and_end_time(self):
        runner = Path(__file__).with_name("run_kv_offload_lab.py")
        result = subprocess.run([sys.executable, "-B", str(runner), "--manifest", "unused"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU run requires", result.stderr)
        self.assertEqual(utc_deadline("2026-09-25T20:00:00Z"), utc_deadline("2026-09-25T22:00:00+02:00"))


if __name__ == "__main__":
    unittest.main()
