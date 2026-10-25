import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from sweep import (MAX_RSS_KIB, attempt_paths, check_result, command,
                   completed_piece, journal_proof, receipt_identity)


class BoundedSweep(unittest.TestCase):
    def test_resume_requires_successful_same_plan_receipts_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            args = SimpleNamespace(parts=2, tier="full", seed=7, test="route_sweep::fixture")
            receipt = root / "part-0.json"
            value = {"identity": digest + ":bundle-fixture:fixture:Full:7:4", "index": 0,
                     "count": 2, "samples": [], "tags": []}
            receipt.write_text(json.dumps(value, separators=(",", ":")))
            (root / "0.log").write_text("test route_sweep::fixture ...\npeak_rss_bytes=1024\ntest result: ok. 1 passed; 0 failed;\n")
            (root / "0.time").write_text("Maximum resident set size (kbytes): 1024\nExit status: 0\n")
            record = completed_piece(root, 0, args, digest, [])
            self.assertEqual(record["peak_rss_kib"], 1024)
            for changed in [(1, 2, digest, "full", 7), (0, 3, digest, "full", 7),
                            (0, 2, "b" * 64, "full", 7), (0, 2, digest, "full", 8)]:
                with self.assertRaises(RuntimeError):
                    receipt_identity(receipt, *changed)
            receipt.write_text(receipt.read_text() + " ")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                completed_piece(root, 0, args, digest, [record])
            (root / "0.time").write_text("Maximum resident set size (kbytes): 1024\nExit status: 137\n")
            with self.assertRaisesRegex(RuntimeError, "successful test"):
                completed_piece(root, 0, args, digest, [])

    def test_reaudit_requires_fresh_executed_journal_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "proof.json"
            value = {"schema": "chopinfer.route-journal-proof.v1", "build_sha256": "a" * 64,
                     "tests": [{"name": "fixture::journal_pin", "exit_code": 0,
                                "passed": 1, "failed": 0, "ignored": 0}]}
            proof.write_text(json.dumps(value))
            self.assertEqual(journal_proof(proof, "a" * 64), value)
            with self.assertRaisesRegex(RuntimeError, "executable"):
                journal_proof(proof, "b" * 64)
            value["tests"][0]["passed"] = 0
            proof.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "executed"):
                journal_proof(proof, "a" * 64)

    def test_retry_preserves_previous_attempt_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.log").write_text("interrupted")
            log, timing = attempt_paths(root, 0)
            self.assertEqual((log.name, timing.name), ("0-retry-1.log", "0-retry-1.time"))
            self.assertEqual((root / "0.log").read_text(), "interrupted")

    def test_each_piece_uses_queue_cgroup_deadline_and_rss_measurement(self):
        cmd = command(Path("fixture"), "route_sweep::profile", Path("timing"))
        self.assertIn("--property=MemoryMax=24G", cmd)
        self.assertIn("--property=MemorySwapMax=0", cmd)
        self.assertLess(cmd.index("systemd-run"), cmd.index("timeout"))
        self.assertLess(cmd.index("timeout"), cmd.index("/usr/bin/time"))
        self.assertIn("880s", cmd)
        self.assertIn("--kill-after=10s", cmd)
        self.assertIn("--test-threads=1", cmd)
        self.assertTrue(cmd[1].endswith("tools/dev/heavy.sh"))

    def test_fails_on_oom_zero_tests_missing_budget_and_excess_rss(self):
        log = "peak_rss_bytes=1024\ntest result: ok. 1 passed; 0 failed;"
        timing = "Maximum resident set size (kbytes): 1024\n"
        self.assertEqual(check_result(log, timing, 0), 1024)
        for output, measured, code in [
            (log, timing, 137), (log.replace("1 passed", "0 passed"), timing, 0),
            (log.replace("peak_rss_bytes=", "missing="), timing, 0),
            (log, "", 0),
            (log, f"Maximum resident set size (kbytes): {MAX_RSS_KIB + 1}\n", 0),
        ]:
            with self.assertRaises(RuntimeError):
                check_result(output, measured, code)


if __name__ == "__main__":
    unittest.main()
