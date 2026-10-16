"""CPU-only checks of tools/dev/qwen38-topology-lab.sh with a fake cargo."""

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/dev/qwen38-topology-lab.sh"

# The fake cargo builds instantly; a test step prints what FAKE_MODE asks for.
FAKE_CARGO = """#!/usr/bin/env bash
for arg in "$@"; do [[ $arg == --no-run ]] && exit 0; done
echo "ranks=$CHOPINFER_QWEN38_RANKS gpu=$CHOPINFER_KERNEL_GPU $*" >>"$FAKE_CALLS"
case $FAKE_MODE in
    pass) echo "test result: ok. 2 passed; 0 failed" ;;
    skip) echo "skipping GPU test: set CHOPINFER_KERNEL_GPU"; echo "test result: ok. 2 passed" ;;
    empty) echo "test result: ok. 0 passed" ;;
    fail) echo "test result: FAILED. 1 passed; 1 failed"; exit 101 ;;
    slow) exec sleep 30 ;;
esac
"""


class TopologyLab(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        tmp = Path(self.dir.name)
        self.cargo = tmp / "cargo"
        self.cargo.write_text(FAKE_CARGO)
        self.cargo.chmod(self.cargo.stat().st_mode | stat.S_IXUSR)
        self.calls = tmp / "calls"
        self.logs = tmp / "logs"

    def run_lab(self, *args, mode="pass"):
        env = dict(os.environ, CARGO=str(self.cargo), FAKE_MODE=mode,
                   FAKE_CALLS=str(self.calls), CHOPINFER_LOCAL_ENV="/nonexistent",
                   XDG_CONFIG_HOME="/nonexistent")
        return subprocess.run(["bash", str(SCRIPT), *args, "--logs-dir", str(self.logs)],
                              env=env, capture_output=True, text=True, timeout=120)

    def test_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_dry_run_prints_the_build_and_every_step(self):
        result = self.run_lab("--ranks", "2", "--gpu", "3", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "export CHOPINFER_QWEN38_RANKS=2 CHOPINFER_KERNEL_GPU=3")
        self.assertIn("--no-run", lines[2])
        commands = [line for line in lines[3:] if not line.startswith("#")]
        self.assertEqual(len(commands), 11)
        for command in commands:
            self.assertIn("--release", command)
            self.assertIn("-- --exact --test-threads=1 --nocapture", command)
        self.assertFalse(self.calls.exists())

    def test_rejects_other_rank_counts(self):
        for args in (["--ranks", "4"], ["--ranks", "8"], [], ["--ranks", "2", "--budget", "0"]):
            self.assertEqual(self.run_lab(*args, "--dry-run").returncode, 2, args)

    def test_passes_when_every_step_runs_tests(self):
        result = self.run_lab("--ranks", "2", "--gpu", "1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS: TP2 kernel lab", result.stdout)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 11)
        self.assertTrue(all(call.startswith("ranks=2 gpu=1 ") for call in calls))
        log = list(self.logs.glob("qwen38-topology-lab-tp2-*.log"))
        self.assertEqual(len(log), 1)
        self.assertIn("PASS: TP2", log[0].read_text())

    def test_fails_on_skipped_empty_or_failed_steps(self):
        for mode, reason in (("skip", "GPU tests skipped"), ("empty", "no test ran"),
                             ("fail", "exit 101")):
            result = self.run_lab("--ranks", "2", mode=mode)
            self.assertEqual(result.returncode, 1, mode)
            self.assertIn("FAIL: TP2 kernel lab", result.stdout)
            self.assertIn(f"qsa ({reason}", result.stdout)
            self.calls.unlink()

    def test_the_budget_stops_a_slow_step(self):
        start = time.monotonic()
        result = self.run_lab("--ranks", "2", "--budget", "2", mode="slow")
        self.assertEqual(result.returncode, 1)
        self.assertIn("qsa (timed out", result.stdout)
        self.assertLess(time.monotonic() - start, 20)


if __name__ == "__main__":
    unittest.main()
