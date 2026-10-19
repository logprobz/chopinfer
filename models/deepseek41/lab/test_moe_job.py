import copy
import json
import unittest

import moe_job


def records():
    values = []
    for label in sorted(moe_job.expected_cases()):
        bank, phase, rows, routing = label.split("-")
        experts, top_k = bank[1:].split("k")
        values.append(dict(case=label, experts=int(experts), top_k=int(top_k),
                           phase=phase, rows=int(rows), routing=routing,
                           old_ms=1.0, active_ms=1.0, compact_ms=1.1, block_warps_ms=2.0,
                           bitwise=True, dropped_join_detected=phase != "Prefill",
                           speed_passed=True, kernel_ms=dict(gateup=[0.5]*3, down=[0.2]*3)))
    return values


class TimingGate(unittest.TestCase):
    def test_complete_table_and_noise_limit(self):
        rows = records()
        self.assertEqual(len(rows), 40)
        rows[0]["active_ms"] = 1.03
        self.assertEqual(moe_job.check_records(rows), [])
        self.assertIn(rows[0]["case"], moe_job.speed_table(rows))
        rows[0]["active_ms"] = 1.031
        self.assertTrue(moe_job.check_records(rows))

    def test_parity_alone_cannot_pass_the_reported_regression(self):
        rows = records()
        rows[0].update(old_ms=0.504368, active_ms=0.847328)
        self.assertTrue(moe_job.check_records(rows))
        rows[0]["active_ms"] = float("nan")
        self.assertTrue(moe_job.check_records(rows))

    def test_missing_boundary_or_join_control_fails(self):
        rows = records()
        self.assertTrue(moe_job.check_records(rows[:-1]))
        self.assertTrue(moe_job.check_records(rows + [copy.deepcopy(rows[0])]))
        rows[0]["dropped_join_detected"] = False
        self.assertTrue(moe_job.check_records(rows))

    def test_records_parse_with_libtest_prefix(self):
        row = records()[0]
        log = "test name ... MOE_RESULT " + json.dumps(row) + "\nok\n"
        self.assertEqual(moe_job.parse_records(log), [row])


if __name__ == "__main__":
    unittest.main()
