"""Replay the frozen log contract and reject incomplete qualification."""
import json
from pathlib import Path
import tempfile
import unittest
import hc_prefill_fused_report as report


def recording(memcheck):
    lines = ["test result: ok. 1 passed; 0 failed; 0 ignored;"]
    for index, case in enumerate(sorted(report.identities(memcheck))):
        lines.append("HC_PREFILL_FUSED_PROGRESS " + json.dumps({
            "event": "case", "case": case, "elapsed_s": float(index)}))
        lines.append("test prefix ... HC_PREFILL_FUSED_CASE " + json.dumps({
            "case": case, "rows": int(case.split(":rows=")[1].split(":")[0]),
            "passed": True, "output_sha256": "a" * 64}))
    for case in ([] if memcheck else sorted(report.timing_cases())):
        for hot in (False, True):
            lines.append("HC_PREFILL_FUSED_TIMING " + json.dumps({
                "case": case, "rows": int(case.split(":rows=")[1].split(":")[0]),
                "hot": hot, "old_launches": 3, "new_launches": report.NEW_LAUNCHES,
                "old_us": 100.0, "new_us": 80.0, "ratio": 0.8}))
    c = report.coverage()
    lines.append("HC_PREFILL_FUSED_COMPLETE " + (
        f"memcheck=true cases={c['memcheck_cases']} invariances=0 negatives=0 timings=0 passed=true"
        if memcheck else
        f"memcheck=false cases={c['cases']} invariances={c['invariances']} "
        f"negatives={c['negatives']} timings={c['timings']} passed=true"))
    if memcheck:
        lines.append("========= ERROR SUMMARY: 0 errors")
    return "\n".join(lines)


class Parser(unittest.TestCase):
    def test_complete_recording_and_analyzer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for gpu in (0, 1):
                (root / f"matrix-{gpu}.log").write_text(recording(False))
                (root / f"memcheck-{gpu}.log").write_text(recording(True))
            value = report.analyze(root)
            self.assertTrue(value["passed"])
            self.assertEqual(len(value["timings"]), 2 * report.coverage()["timings"])
            self.assertEqual(report.coverage()["cases"], 128)
            self.assertEqual(value["timings"][0]["saving_us"], 20)
            (root / "matrix-1.log").write_text(recording(False).replace("a" * 64, "b" * 64))
            with self.assertRaisesRegex(ValueError, "peer"):
                report.analyze(root)

    def test_reject_missing_duplicate_failed_and_zero_test(self):
        text = recording(False)
        for broken in [
            text.replace("1 passed", "0 passed"),
            text.replace("0 failed", "1 failed"),
            text.replace('"passed": true', '"passed": false', 1),
            text.replace("site0:rows=1:finite:r4c4n:widen=false", "wrong", 1),
            text.replace('"new_launches": 1', '"new_launches": 2', 1),
            text + "\nHC_PREFILL_FUSED_MISMATCH bad",
            text.replace("negatives=3", "negatives=0"),
            text.replace('"old_us": 100.0', '"old_us": 0.0', 1),
            text.replace('"rows": 8192', '"rows": 8191', 1),
            text + "\n" + next(line for line in text.splitlines() if "HC_PREFILL_FUSED_CASE " in line),
            "\n".join(l for l in text.splitlines() if "_PROGRESS" not in l),
            text.replace('"elapsed_s": 5.0', '"elapsed_s": 0.5', 1),
        ]:
            with self.assertRaises(ValueError):
                report.parse(broken, False)


class Diagnostics(unittest.TestCase):
    def test_last_started_names_the_running_case(self):
        text = recording(False)
        lines = text.splitlines()
        cut = next(i for i, l in enumerate(lines) if '"elapsed_s": 7.0' in l)
        partial = "\n".join(lines[:cut + 1])
        self.assertEqual(report.last_started(partial)["elapsed_s"], 7.0)
        with self.assertRaises(ValueError):
            report.parse(partial, False)
        legacy = "\n".join(l for l in lines[:cut] if "_PROGRESS" not in l)
        self.assertEqual(report.last_started(legacy)["event"], "after")


if __name__ == "__main__":
    unittest.main()
