"""Development comparisons reject invalid and mismatched benchmark receipts."""
import copy
import importlib.util
import os
import pathlib
import subprocess
import sys
import time
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("loop", ROOT / "tools/dev/benchmark-loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class Comparison(unittest.TestCase):
    def setUp(self):
        self.receipt = {"schema": "chopinfer-qualification/v1",
                        "passed": True, "check": "spec-bench-chopinfer-spec",
                        "server": {"model": "example"}, "workload_sha256": "same",
                        "concurrencies": [1], "output_tokens": 64,
                        "cohorts": [{"category": "chat", "concurrency": 1,
                                     "aggregate_tokens_per_second": 100}]}

    def test_reports_measured_delta(self):
        after = copy.deepcopy(self.receipt)
        after["cohorts"][0]["aggregate_tokens_per_second"] = 125
        self.assertEqual(loop.compare(self.receipt, after),
                         ["chat C1: 100.00 -> 125.00 tok/s (+25.00%)"])

    def test_rejects_failures_and_workload_changes(self):
        for field, value in (("passed", False), ("mechanics_only", True),
                             ("workload_sha256", "different"), ("output_tokens", 128),
                             ("cohorts", []), ("errors", ["failed"]),
                             ("skipped", ["workload"]), ("schema", "unknown")):
            with self.subTest(field=field):
                after = dict(self.receipt, **{field: value})
                with self.assertRaises(ValueError):
                    loop.compare(self.receipt, after)

    def test_dry_run_never_starts_a_server(self):
        result = subprocess.run(["python3", "-B", str(ROOT / "tools/dev/benchmark-loop.py"),
                                 "--cohort", "spec-c1", "--baseline", "/unused/new-baseline.json",
                                 "--save-baseline", "--dry-run"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CHOPINFER_DEV_SKIP_WARMUP=1", result.stdout)
        self.assertIn("--max-sequences 1 --drafter dflash", result.stdout)

    def test_deadline_terminates_and_collects_its_child(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            loop.run_bounded([sys.executable, "-c", "import time; time.sleep(30)"],
                             cwd=ROOT, env=os.environ, deadline=time.monotonic() + 0.1,
                             capture=True)

    def test_external_term_is_forwarded_to_the_child_group(self):
        command = [sys.executable, "-c",
                   "import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"]
        with self.assertRaises(SystemExit) as stopped:
            loop.run_bounded(command, cwd=ROOT, env=os.environ,
                             deadline=time.monotonic() + 10, capture=True)
        self.assertEqual(stopped.exception.code, 143)


class ContributorTimings(unittest.TestCase):
    def test_phase_totals_keep_empty_and_existing_targets_separate(self):
        samples = [dict(iteration=i, starting_target="empty" if i == 0 else "existing", phase=phase, seconds=seconds)
                   for i in range(3) for phase, seconds in [("compile", 1), ("compile", 3), ("check", 2)]]
        summary = loop.contributor_summary(samples)
        row = next(row for row in summary if row["phase"] == "compile" and row["starting_target"] == "existing")
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["median_seconds"], 4)
        self.assertEqual(len(summary), 4)

    def test_contributor_dry_run_uses_fixtures_without_a_server(self):
        for mode, case in (("qwen38", "contributor_tensor_names_shapes_scales_and_pairs"),
                           ("mimo26", "image_free_packages_plan_every_image")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/dev/benchmark-loop.py"),
                                         "--contributor", mode, "--output", str(pathlib.Path(directory) / "costs.json"), "--dry-run"],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(case, result.stdout)
                self.assertIn("package_and_verify_fixtures", result.stdout)
                self.assertNotIn("redeploy.sh", result.stdout)
                self.assertNotIn("spec_bench.py", result.stdout)

    def test_contributor_and_gpu_arguments_cannot_mix(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/dev/benchmark-loop.py"),
                                 "--contributor", "sdk", "--cohort", "spec-c1", "--output", "/unused/timings.json"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot use GPU", result.stderr)


if __name__ == "__main__":
    unittest.main()
