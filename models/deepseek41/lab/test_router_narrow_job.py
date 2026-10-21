"""CPU checks of router_narrow_job's analyzer and summary on synthetic logs
of the format the frozen test binary prints."""
import json
import unittest

import router_narrow_job as job
from test_native_moe_job import mutate, timing


def log(worker, phase):
    memcheck = phase == "memcheck"
    lines = ["running 1 test", "test narrow::router_narrow_gpu ... "]
    for pattern, live in job.cases(worker, memcheck):
        lines.append("router_narrow_record " + json.dumps({
            "pattern": pattern, "live": live, "differing_bytes": {k: 0 for k in job.COMPARED},
            "candidate_widened_untouched": True, "bound_widened_consistent": True,
            "uncertified_rows": 5 if pattern == "outliers" and live > 1 else 0,
            "timing": timing(1.0, 0.9) if live == job.ROWS and not memcheck else None, "passed": True}))
    lines.append("router_narrow_done " + json.dumps({"worker": worker, "cases": len(job.cases(worker, memcheck)),
                                                     "failed": 0, "memcheck": memcheck}))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out")
    if memcheck:
        lines.append("========= ERROR SUMMARY: 0 errors")
    return "\n".join(lines)


class Analyzer(unittest.TestCase):
    def test_accepts_both_phases_and_summarizes(self):
        for phase in job.PHASES:
            parsed = [job.analyze(log(w, phase), w, phase) for w in (0, 1)]
        s = job.summarize(parsed)
        self.assertEqual(set(s), {"clean", "outliers"})
        self.assertAlmostEqual(s["clean"]["per_chunk_saving_ms"], job.LAYERS * 0.1, places=4)

    def test_rejects_differences_and_missing_or_misassigned_work(self):
        good = log(1, "pair")
        bad = [
            mutate(good, "router_narrow_record", lambda r: r["differing_bytes"].update(logits=3)),
            mutate(good, "router_narrow_record", lambda r: r["differing_bytes"].pop("flags")),
            mutate(good, "router_narrow_record", lambda r: r.update(candidate_widened_untouched=False)),
            mutate(good, "router_narrow_record", lambda r: r.update(bound_widened_consistent=False)),
            mutate(good, "router_narrow_record", lambda r: r.update(passed=False)),
            mutate(good, "router_narrow_record", lambda r: r.update(uncertified_rows=0)),
            mutate(good, "router_narrow_record", lambda r: r.update(timing=timing()) if r["live"] == 65 else None),
            mutate(good, "router_narrow_record", lambda r: r.update(timing=None)),
            mutate(good, "router_narrow_done", lambda r: r.update(failed=1)),
            "\n".join(l for l in good.splitlines() if '"live": 2049' not in l),
            good + "\n========= ERROR SUMMARY: 0 errors",
            good.replace("test result: ok. 1 passed", "test result: ok. 0 passed"),
        ]
        for text in bad:
            with self.assertRaises((ValueError, KeyError, TypeError)):
                job.analyze(text, 1, "pair")
        with self.assertRaises(ValueError):
            job.analyze(good, 0, "pair")
        memcheck = log(0, "memcheck")
        for text in (memcheck.replace("0 errors", "2 errors"), memcheck.replace("ERROR SUMMARY: 0 errors", "")):
            with self.assertRaises(ValueError):
                job.analyze(text, 0, "memcheck")


if __name__ == "__main__":
    unittest.main()
