"""Synthetic C1/C8 completion logs: complete runs pass, every gap is rejected."""
import copy
import json
import unittest
import hc_completion_report as report


def coverage():
    cases = []
    for rows in (1, 8):
        for step in range(2):
            for site in range(3):
                cases.append({"id": f"decode-{rows}:[{step}]:{site}:\"Mlp\"", "rows": rows,
                              "widening": site % 2 == 0})
    return {"cases": cases, "corpus_sha256": "c" * 64, "rounds": 7, "warmup": 2,
            "matrix_reports": len(cases) * 10, "timing_records": len(cases) * 10}


def matrix_fixture(cov, memcheck=False):
    cases = report.memcheck_cases(cov) if memcheck else cov["cases"]
    configurations = ["served"] if memcheck else report.CONFIGURATIONS
    reports, lines = [], ["running 1 test"]
    for c in cases:
        for v in report.VARIANTS:
            for k in configurations:
                served = k == "served"
                reports.append({"case": c["id"], "rows": c["rows"], "variant": v, "configuration": k,
                                "widen": c["widening"] if served else not c["widening"], "alias": served,
                                "passed": True, "failures": [], "output_sha256": "a" * 64})
        lines.append("HC_COMPLETION_CASE " + json.dumps(
            {"case": c["id"], "rows": c["rows"], "reports": len(report.VARIANTS) * len(configurations)}))
    checks = [] if memcheck else [{"case": a, "name": b, "passed": True} for a, b in sorted(report.labels(cov))]
    captured = [] if memcheck else [{"case": c["id"], "outputs": 12, "failures": [], "passed": True} for c in cases]
    negatives = 0 if memcheck else 10
    value = {"schema": "chopinfer.hc-completion-matrix.v1", "corpus_sha256": "" if memcheck else cov["corpus_sha256"],
             "cases": len(cases), "memcheck": memcheck, "reports": reports, "checks": checks,
             "captured": captured, "negatives": negatives, "passed": True}
    lines.append(f"HC_COMPLETION_RESULT memcheck={str(memcheck).lower()} cases={len(cases)} reports={len(reports)} "
                 f"checks={len(checks)} negatives={negatives} passed=true")
    lines.append("test replay ... ok")
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out")
    if memcheck:
        lines.append("========= ERROR SUMMARY: 0 errors")
    return value, "\n".join(lines)


def timing_fixture(cov):
    records, lines = [], []
    for c in cov["cases"]:
        for v in report.VARIANTS:
            for hot in (False, True):
                records.append({"case": c["id"], "rows": c["rows"], "variant": v, "widen": c["widening"],
                                "hot": hot, "rounds": 7, "warmup": 2, "reference_us": 20.0,
                                "candidate_us": 12.0, "paired_ratio": 0.6,
                                "reference_launches": 2, "candidate_launches": 1})
        lines.append("HC_COMPLETION_TIMING_CASE " + json.dumps({"case": c["id"], "rows": c["rows"], "records": 10,
                                                                "elapsed_s": 1.0}))
    lines.append(f"HC_COMPLETION_TIMING_RESULT cases={len(cov['cases'])} records={len(records)} PASS")
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out")
    return {"cases": len(cov["cases"]), "corpus_sha256": cov["corpus_sha256"], "records": records}, "\n".join(lines)


class Report(unittest.TestCase):
    def test_complete_runs_pass(self):
        cov = coverage()
        value, text = matrix_fixture(cov)
        self.assertTrue(report.matrix(value, text, cov)["passed"])
        value, text = matrix_fixture(cov, True)
        self.assertEqual(report.matrix(value, text, cov, memcheck=True)["reports"], 20)
        value, text = timing_fixture(cov)
        rows = report.timing(value, text, cov)
        self.assertEqual(len(rows), 2 * 5 * 2)
        for row in rows:
            self.assertEqual(row["boundaries"], [3, 3])
            self.assertAlmostEqual(row["saved_us_per_step"], 24.0)
            self.assertEqual(row["launches_saved_per_step"], 3)

    def test_matrix_gaps_are_rejected(self):
        cov = coverage()
        value, text = matrix_fixture(cov)
        broken = []
        v = copy.deepcopy(value); v["reports"].pop(); broken.append((v, text))
        v = copy.deepcopy(value); v["reports"][0]["passed"] = False; broken.append((v, text))
        v = copy.deepcopy(value); v["reports"][0]["alias"] = False; broken.append((v, text))
        v = copy.deepcopy(value); v["checks"].pop(); broken.append((v, text))
        v = copy.deepcopy(value); v["checks"].append(v["checks"][0]); broken.append((v, text))
        v = copy.deepcopy(value); v["captured"].pop(); broken.append((v, text))
        v = copy.deepcopy(value); v["negatives"] = 0; broken.append((v, text))
        broken.append((value, text.replace("1 passed", "0 passed")))
        broken.append((value, text.replace('"reports": 10', '"reports": 9', 1)))
        broken.append((value, text.replace("passed=true", "passed=false")))
        for v, t in broken:
            with self.assertRaises(ValueError):
                report.matrix(v, t, cov)
        value, text = matrix_fixture(cov, True)
        with self.assertRaises(ValueError):
            report.matrix(value, text.replace("0 errors", "1 errors"), cov, memcheck=True)
        failed = copy.deepcopy(value)
        failed["reports"][0].update(passed=False, failures=["normed"])
        failed["passed"] = False
        text = text.replace("passed=true", "passed=false").replace("test result: ok. 1 passed; 0 failed",
                                                                    "test result: FAILED. 0 passed; 1 failed")
        self.assertFalse(report.matrix(failed, text, cov, memcheck=True)["passed"])

    def test_timing_gaps_are_rejected(self):
        cov = coverage()
        value, text = timing_fixture(cov)
        for mutate in [lambda v: v["records"].pop(), lambda v: v["records"][0].update(candidate_us=0.0),
                       lambda v: v["records"][0].update(widen=not v["records"][0]["widen"]),
                       lambda v: v["records"][0].update(candidate_launches=2)]:
            v = copy.deepcopy(value)
            mutate(v)
            with self.assertRaises(ValueError):
                report.timing(v, text, cov)
        with self.assertRaises(ValueError):
            report.timing(value, text.replace("PASS", "FAIL"), cov)

    def test_peers(self):
        cov = coverage()
        a, _ = matrix_fixture(cov)
        b = copy.deepcopy(a)
        self.assertEqual(report.peers(a, b), [])
        b["reports"][3]["output_sha256"] = "b" * 64
        self.assertEqual(len(report.peers(a, b)), 1)


if __name__ == "__main__":
    unittest.main()
