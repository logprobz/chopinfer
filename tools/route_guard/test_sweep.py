import unittest
from pathlib import Path

from sweep import MAX_RSS_KIB, check_result, command


class BoundedSweep(unittest.TestCase):
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
