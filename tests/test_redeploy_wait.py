"""CPU-only GPU release checks. NVIDIA queries and CUDA enumeration are fake."""

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("redeploy_wait", ROOT / "tools/dev/redeploy-wait.py")
WAIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAIT)

GPU = "7, GPU-a, 90000, 96000\n2, GPU-b, 80000, 96000\n"


class ReleaseWait(unittest.TestCase):
    def setUp(self):
        self.clock = 0
        self.apps = ["41, 88000, GPU-a\n"]
        self.calls = []
        self.poller = Mock()
        self.poller.poll.return_value = [(5, 1)]
        for target, value in (
            ("time.monotonic", lambda: self.clock),
            ("time.sleep", self.sleep),
            ("subprocess.run", self.run_fake),
            ("select.poll", lambda: self.poller),
        ):
            patcher = patch.object(getattr(WAIT, target.split('.')[0]), target.split('.')[1], value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a"})
        env.start()
        self.addCleanup(env.stop)

    def sleep(self, seconds):
        self.clock += seconds

    def run_fake(self, command, **kwargs):
        self.calls.append(command)
        self.assertGreater(kwargs["timeout"], 0)
        if command[0] != "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "GPU-b\nGPU-a\n", "")
        if command[1] == "--query-gpu=index,uuid,memory.used,memory.total":
            return subprocess.CompletedProcess(command, 0, GPU, "")
        self.assertEqual(command[1], "--query-compute-apps=pid,used_memory,gpu_uuid")
        value = self.apps.pop(0) if len(self.apps) > 1 else self.apps[0]
        return subprocess.CompletedProcess(command, 0, value, "")

    def test_waits_for_accounting_after_process_exit(self):
        self.apps = ["41, 88000, GPU-a\n", "41, 88000, GPU-a\n", ""]
        WAIT.wait_released(41, 5, ["--devices", "1"], 2)
        self.assertEqual(self.clock, 1)

    def test_waits_until_every_selected_gpu_releases_the_old_pid(self):
        self.apps = ["41, 88000, GPU-a\n41, 88000, GPU-b\n", "41, 88000, GPU-b\n", ""]
        WAIT.wait_released(41, None, [], 2)
        self.assertEqual(self.clock, 1)

    def test_waits_for_exit_even_with_no_compute_process(self):
        self.apps = [""]
        self.poller.poll.side_effect = [[], [], [(5, 1)]]
        WAIT.wait_released(41, 5, [], 2)
        self.assertEqual(self.clock, 1)
        self.poller.register.assert_called_once_with(5, WAIT.select.POLLIN)

    def test_timeout_names_physical_gpu_and_all_holders(self):
        self.apps = ["41, 80000, GPU-a\n92, 10000, GPU-a\n"]
        with self.assertRaisesRegex(TimeoutError, r"GPU 7 .*PID 41 .*PID 92"):
            WAIT.wait_released(41, 5, ["--devices=1"], 1)
        self.assertEqual(self.clock, 1)

    def test_other_gpus_and_unrelated_processes_do_not_block(self):
        self.apps = ["41, 80000, GPU-a\n92, 80000, GPU-b\n"]
        WAIT.wait_released(41, None, ["--devices", "0"], 1)
        self.assertEqual(self.clock, 0)

    def test_numeric_visible_ordinals_use_cuda_order_not_smi_indices(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}):
            WAIT.wait_released(41, None, ["--devices", "0"], 1)
        self.assertTrue(any(command[0] != "nvidia-smi" for command in self.calls))

    def test_reordered_visible_devices_match_host_rank_order(self):
        gpus = {name: [] for name in ("GPU-a", "GPU-b", "GPU-c", "GPU-d")}
        enumeration = subprocess.CompletedProcess([], 0, "GPU-d\nGPU-c\nGPU-b\nGPU-a\n", "")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,2,1,0"}), \
                patch.object(WAIT.subprocess, "run", return_value=enumeration):
            self.assertEqual(WAIT.selected_uuids(gpus, ["--devices", "1,0,2,3"], 1),
                             ["GPU-c", "GPU-d", "GPU-b", "GPU-a"])

    def test_live_pid_timeout_even_after_accounting_clears(self):
        self.apps = [""]
        self.poller.poll.return_value = []
        with self.assertRaisesRegex(TimeoutError, "old PID 41 still running; GPU"):
            WAIT.wait_released(41, 5, [], 0.5)

    def test_query_failure_fails_closed_and_is_bounded(self):
        with patch.object(WAIT.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 1)):
            with self.assertRaisesRegex(TimeoutError, "PID 41.*GPU query failed"):
                WAIT.wait_released(41, 5, [], 1)
        self.assertEqual(self.clock, 1)

    def test_malformed_compute_accounting_fails_closed(self):
        self.apps = ["N/A, 88000, GPU-a\n"]
        with self.assertRaisesRegex(TimeoutError, "GPU query failed"):
            WAIT.wait_released(41, None, [], 0.5)

    def test_invalid_devices_fail_closed(self):
        for devices in ("-1", "2", "1,1", "x"):
            self.clock = 0
            with self.assertRaisesRegex(TimeoutError, "GPU query failed"):
                WAIT.wait_released(41, None, ["--devices", devices], 0.5)


class ProcessFiles(unittest.TestCase):
    """The redeploy stop path reads a server's /proc files while it exits."""

    def test_a_running_process_reads_its_state_and_environment(self):
        child = subprocess.Popen(["sleep", "60"], env={"CHOPINFER_TEST_MARK": "1"})
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        state, started = WAIT.process_state(child.pid)
        self.assertIn(state, "RS")
        self.assertTrue(started.isdigit())
        self.assertIn(b"CHOPINFER_TEST_MARK=1", WAIT.process_environment(child.pid))

    def test_a_process_reaped_during_a_read_is_gone(self):
        # Reaping a process after its /proc files were opened fails the stat
        # read with ESRCH, not ENOENT, and leaves its environment empty.
        child = subprocess.Popen(["sleep", "60"])
        with open(f"/proc/{child.pid}/stat") as stat, open(f"/proc/{child.pid}/environ", "rb") as environ:
            child.kill()
            child.wait()
            with self.assertRaises(ProcessLookupError):
                stat.read()
            self.assertEqual(environ.read(), b"")
        reaped = ProcessLookupError(3, "No such process")
        with patch.object(WAIT.Path, "read_text", side_effect=reaped):
            self.assertIsNone(WAIT.process_state(child.pid))
        with patch.object(WAIT.Path, "read_bytes", side_effect=reaped):
            self.assertIsNone(WAIT.process_environment(child.pid))
        with patch.object(WAIT.Path, "read_bytes", return_value=b""):
            self.assertIsNone(WAIT.process_environment(child.pid))
        self.assertIsNone(WAIT.process_state(child.pid))
        self.assertIsNone(WAIT.process_environment(child.pid))


if __name__ == "__main__":
    unittest.main()
