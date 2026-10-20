"""Require the complete HC prefill matrix, memcheck and peer parity.

The arms are the served expansion-then-mix tiles with the input norm and
ceil quantizer folded in (one launch) against the served expand, mixer and
norm chain."""
import argparse
import json
import math
from pathlib import Path
import re

ROWS = (1, 7, 64, 192, 320, 2496, 8191, 8192)
PATTERNS = ("zero", "tiny", "maximum", "nan", "inf", "weight_nan", "cancellation", "sticky")
VARIANTS = ("r4c4n", "r8c8n")
OLD_LAUNCHES, NEW_LAUNCHES = 3, 1
NEGATIVES = 3


def coverage():
    """Per variant: 64 cases, 2 memcheck cases, 4 invariances and 24 timing
    records; one control per mutation."""
    n = len(VARIANTS)
    return {"cases": 64 * n, "memcheck_cases": 2 * n, "invariances": 4 * n,
            "negatives": NEGATIVES, "timings": 24 * n}


def identities(memcheck):
    cases = {f"site{s}:rows={rows}:finite:{variant}:widen={wide}"
             for s in ((0,) if memcheck else range(3))
             for rows in ((8192,) if memcheck else ROWS)
             for variant in VARIANTS for wide in ("false", "true")}
    if not memcheck:
        cases |= {f"edge:rows=7:{pattern}:{variant}:widen={wide}"
                  for pattern in PATTERNS for variant in VARIANTS
                  for wide in ("false", "true")}
    return cases


def timing_cases():
    return {case for case in identities(False) if not case.startswith("edge:")
            and (":rows=8192:" in case
                 or (case.startswith("site0:") and any(f":rows={n}:" in case for n in (64, 320, 2496))))}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def markers(text, marker):
    records = []
    for line in text.splitlines():
        at = line.find(marker + " ")
        if at >= 0:
            records.append(json.loads(line[at + len(marker) + 1:]))
    return records


def progress(text):
    """Start markers in log order, with the seconds each one ran until the
    next marker. The last entry names the work a killed worker was doing."""
    events = markers(text, "HC_PREFILL_FUSED_PROGRESS")
    elapsed = [e["elapsed_s"] for e in events]
    require(all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in elapsed)
            and elapsed == sorted(elapsed), "progress clock is not monotonic")
    return [dict(e, seconds=(elapsed[i + 1] - e["elapsed_s"]) if i + 1 < len(events) else None)
            for i, e in enumerate(events)]


def last_started(text):
    """The last case a worker began, from progress markers or, for logs
    written before them, the case after the last completed one."""
    events = progress(text)
    if events:
        return events[-1]
    done = markers(text, "HC_PREFILL_FUSED_CASE")
    return {"event": "after", "case": done[-1]["case"] if done else None, "elapsed_s": None}


def parse(text, memcheck, require_progress=True):
    require(re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
            == [("ok", "1", "0", "0")], "one executed passing test required")
    cases = markers(text, "HC_PREFILL_FUSED_CASE")
    expected = identities(memcheck)
    events = progress(text)
    if require_progress:
        started = [e["case"] for e in events if e["event"] == "case"]
        require(len(started) == len(expected) and set(started) == expected,
                "every case needs one start marker")
    require(len(cases) == len(expected) and {c["case"] for c in cases} == expected,
            "executed case identities differ")
    for case in cases:
        require(case["passed"] is True, "bitwise mismatch")
        require(re.fullmatch("[0-9a-f]{64}", case.get("output_sha256", "")), "missing digest")
        require(case["rows"] == int(case["case"].split(":rows=")[1].split(":")[0]), "row count differs")
    require("HC_PREFILL_FUSED_MISMATCH" not in text, "mismatch marker present")
    summary = re.findall(r"HC_PREFILL_FUSED_COMPLETE memcheck=(true|false) cases=(\d+) "
                         r"invariances=(\d+) negatives=(\d+) timings=(\d+) passed=(true|false)", text)
    c = coverage()
    require(summary == [("true", str(c["memcheck_cases"]), "0", "0", "0", "true") if memcheck else
                        ("false", str(c["cases"]), str(c["invariances"]), str(c["negatives"]),
                         str(c["timings"]), "true")],
            "completion coverage differs")
    timings = markers(text, "HC_PREFILL_FUSED_TIMING")
    require(len(timings) == (0 if memcheck else c["timings"]), "timing count differs")
    if not memcheck:
        require({(r["case"], r["hot"]) for r in timings}
                == {(c, h) for c in timing_cases() for h in (False, True)}, "timing identities differ")
    for row in timings:
        require(row["old_launches"] == OLD_LAUNCHES and row["new_launches"] == NEW_LAUNCHES,
                "chain launch count differs")
        require(row["rows"] == int(row["case"].split(":rows=")[1].split(":")[0]), "timing rows differ")
        require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] > 0
                    for k in ("old_us", "new_us", "ratio")), "invalid event interval")
    durations = [e["seconds"] for e in events if e["event"] in ("case", "timing") and e["seconds"] is not None]
    return {"cases": cases, "timings": timings,
            "longest_step_seconds": max(durations) if durations else None,
            "last_marker_seconds": events[-1]["elapsed_s"] if events else None}


def analyze(root):
    results = []
    for gpu in (0, 1):
        memcheck = (root / f"memcheck-{gpu}.log").read_text()
        require("ERROR SUMMARY: 0 errors" in memcheck, "clean memcheck summary missing")
        parse(memcheck, True)
        results.append(parse((root / f"matrix-{gpu}.log").read_text(), False))
    require({c["case"]: c["output_sha256"] for c in results[0]["cases"]}
            == {c["case"]: c["output_sha256"] for c in results[1]["cases"]}, "peer bits differ")
    return {"passed": True, "cases_per_gpu": coverage()["cases"],
            "memcheck_cases_per_gpu": coverage()["memcheck_cases"],
            "matrix_seconds": [r["last_marker_seconds"] for r in results],
            "longest_step_seconds": [r["longest_step_seconds"] for r in results],
            "timings": [dict(row, gpu=gpu, saving_us=row["old_us"] - row["new_us"])
                        for gpu, result in enumerate(results) for row in result["timings"]],
            "scope": "Checkpoint weights and seeded prefill streams. Whole-chain kernel events, no serving timing claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Both GPUs passed HC prefill fusion, seeded 8K memcheck, invariances and negative controls.")
