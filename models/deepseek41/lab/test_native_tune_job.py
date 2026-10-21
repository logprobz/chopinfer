"""CPU checks of native_tune_job's analyzer and summary on synthetic logs of
the format the frozen test binary prints."""
import json
import unittest

import native_tune_job as job
from test_native_moe_job import mutate, timing


def log(worker, ms=1.0):
    lines = ["running 1 test", "test tune::native_moe_tune_gpu ... "]
    for case in job.CASES[worker]:
        kinds = [{"kind": k, "arms": [{"arm": a, "bitwise": True if a in job.BITWISE else None,
                                       "timing": timing(2.0, ms)} for a in job.ARMS]} for k in job.KINDS]
        lines.append("native_moe_tune " + json.dumps({"case": case, "rows": job.ROWS, "kinds": kinds}))
    lines.append("native_moe_tune_done " + json.dumps({"worker": worker, "cases": 2, "failed": 0}))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out")
    return "\n".join(lines)


class Analyzer(unittest.TestCase):
    def test_accepts_both_workers_and_summarizes(self):
        parsed = [job.analyze(log(w), w) for w in (0, 1)]
        summary = job.summarize(parsed)
        lean = summary["GateUp"]["lean"]
        self.assertEqual(len(lean["cases"]), 4)
        self.assertAlmostEqual(lean["per_chunk_saving_ms"], job.FULL_LAYERS * 1.0)

    def test_rejects_missing_failed_or_misassigned_work(self):
        good = log(0)
        arm = lambda name, change: lambda r: [change(a) for k in r["kinds"] for a in k["arms"] if a["arm"] == name]
        bad = [
            good.replace("test result: ok. 1 passed", "test result: ok. 0 passed"),
            "\n".join(l for l in good.splitlines() if "captured-l14-r0" not in l),
            mutate(good, "native_moe_tune", arm("lean", lambda a: a.update(bitwise=False))),
            mutate(good, "native_moe_tune", arm("paired", lambda a: a.update(bitwise=None))),
            mutate(good, "native_moe_tune", arm("no_mma", lambda a: a.update(bitwise=True))),
            mutate(good, "native_moe_tune", lambda r: r["kinds"][1]["arms"].pop()),
            mutate(good, "native_moe_tune", lambda r: r["kinds"].pop()),
            mutate(good, "native_moe_tune", lambda r: r.update(rows=2048)),
            mutate(good, "native_moe_tune", arm("mma_only", lambda a: a["timing"]["round_samples_ms"].pop())),
            mutate(good, "native_moe_tune_done", lambda r: r.update(failed=1)),
        ]
        for text in bad:
            with self.assertRaises((ValueError, KeyError)):
                job.analyze(text, 0)
        with self.assertRaises(ValueError):
            job.analyze(good, 1)


if __name__ == "__main__":
    unittest.main()
