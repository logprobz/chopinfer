import sqlite3
import unittest

from serving_step_nsys import summarize


class ServingStepNsysTest(unittest.TestCase):
    def test_overlapping_kernels_are_charged_their_advance(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
            "deviceId INTEGER, start INTEGER, end INTEGER, shortName INTEGER);"
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(0, "gemm"), (1, "allreduce"), (2, "chop_gemm_qwen38_head_m1_v1")],
        )
        rows = []
        for step in range(3):
            base = step * 100_000
            rows += [
                (0, base, base + 20_000, 0),
                # Launched early under PDL, ends 30 us after the GEMM.
                (0, base + 5_000, base + 50_000, 1),
                (0, base + 55_000, base + 60_000, 2),
            ]
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", rows
        )
        report = summarize(connection, 0)
        self.assertEqual(list(report), [1])
        entry = report[1]
        # The first head closes a partial step, so two of three remain.
        self.assertEqual(entry["steps"], 2)
        self.assertEqual(entry["median_span_us"], 60.0)
        charged = {item["kernel"]: item["us"] for item in entry["budget"]}
        self.assertEqual(charged, {"gemm": 20.0, "allreduce": 30.0,
                                   "chop_gemm_qwen38_head_m1_v1": 10.0})


if __name__ == "__main__":
    unittest.main()
