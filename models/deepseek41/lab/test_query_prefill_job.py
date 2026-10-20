"""Exercise the exact job parser, including libtest-prefixed first markers."""
import json
import tempfile
from pathlib import Path
import unittest
import query_prefill_job as job


def recording(memcheck):
    lines = []
    for case in sorted(job.identities(memcheck)):
        lines.append("test test_name ... QUERY_PREFILL_CASE " + json.dumps({
            "case": case, "rows": 8192 if case.startswith("real-") else 64,
            "passed": True, "output_sha256": "a" * 64}))
    if not memcheck:
        for case in sorted(job.identities(True)):
            for hot in (False, True):
                lines.append("QUERY_PREFILL_TIMING " + json.dumps({
                    "case": case, "hot": hot, "rows": 8192,
                    "old_us": 200, "new_us": 150, "ratio": .75}))
    fields = "true cases=4 invariances=0 negatives=0 timings=0" if memcheck else (
        "false cases=28 invariances=12 negatives=2 timings=8")
    lines += [f"QUERY_PREFILL_COMPLETE memcheck={fields} passed=true",
              "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;",
              "========= ERROR SUMMARY: 0 errors"]
    return "\n".join(lines)


class Parser(unittest.TestCase):
    def test_full_recordings_and_projected_saving(self):
        for memcheck in (False, True):
            self.assertEqual(len(job.parse(recording(memcheck), memcheck)["cases"]), 4 if memcheck else 28)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for gpu in (0, 1):
                for name, memcheck in (("matrix", False), ("memcheck", True)):
                    (root / f"{name}-{gpu}.log").write_text(recording(memcheck))
            result = job.analyze(root)
            self.assertEqual(len(result["timings"]), 16)
            self.assertTrue(all(r["projected_default_chunk_saving_ms"] == 1 for r in result["timings"]))
            self.assertTrue(all(r["projected_full_row_schedule_saving_ms"] == 2 for r in result["timings"]))

    def test_reject_zero_skipped_missing_duplicate_wrong_shape_and_mismatch(self):
        good = recording(False)
        first = next(line for line in good.splitlines() if "QUERY_PREFILL_CASE " in line)
        for bad in (
            good.replace("1 passed", "0 passed"),
            good.replace("0 ignored", "1 ignored"),
            good.replace(first, ""),
            good + "\n" + first,
            good.replace('"rows": 8192', '"rows": 32', 1),
            good.replace('"passed": true', '"passed": false', 1),
            good.replace("invariances=12", "invariances=11"),
            good + "\nQUERY_PREFILL_MISMATCH row=0",
            good.replace('"old_us": 200', '"old_us": 0', 1),
        ):
            with self.assertRaises(ValueError):
                job.parse(bad, False)


if __name__ == "__main__":
    unittest.main()
