"""Require the complete HC prefill matrix, memcheck and peer parity."""
import argparse
import json
import math
from pathlib import Path
import re

ROWS = (1, 7, 64, 192, 320, 2496, 8191, 8192)
PATTERNS = ("zero", "tiny", "maximum", "nan", "inf", "weight_nan", "cancellation", "sticky")


def identities(memcheck):
    cases = {f"site{s}:rows={rows}:finite:r{tile}:widen={wide}"
             for s in ((0,) if memcheck else range(3))
             for rows in ((8192,) if memcheck else ROWS)
             for tile in (1, 4, 8) for wide in ("false", "true")}
    if not memcheck:
        cases |= {f"edge:rows=7:{pattern}:r{tile}:widen={wide}"
                  for pattern in PATTERNS for tile in (1, 4, 8)
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


def parse(text, memcheck):
    require(re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
            == [("ok", "1", "0", "0")], "one executed passing test required")
    cases = markers(text, "HC_PREFILL_FUSED_CASE")
    expected = identities(memcheck)
    require(len(cases) == len(expected) and {c["case"] for c in cases} == expected,
            "executed case identities differ")
    for case in cases:
        require(case["passed"] is True, "bitwise mismatch")
        require(re.fullmatch("[0-9a-f]{64}", case.get("output_sha256", "")), "missing digest")
        require(case["rows"] == int(case["case"].split(":rows=")[1].split(":")[0]), "row count differs")
    require("HC_PREFILL_FUSED_MISMATCH" not in text, "mismatch marker present")
    summary = re.findall(r"HC_PREFILL_FUSED_COMPLETE memcheck=(true|false) cases=(\d+) "
                         r"invariances=(\d+) negatives=(\d+) timings=(\d+) passed=(true|false)", text)
    require(summary == [("true", "6", "0", "0", "0", "true") if memcheck else
                        ("false", "192", "12", "3", "72", "true")], "completion coverage differs")
    timings = markers(text, "HC_PREFILL_FUSED_TIMING")
    require(len(timings) == (0 if memcheck else 72), "timing count differs")
    if not memcheck:
        require({(r["case"], r["hot"]) for r in timings}
                == {(c, h) for c in timing_cases() for h in (False, True)}, "timing identities differ")
    for row in timings:
        require(row["old_launches"] == 3 and row["new_launches"] == 1, "chain launch count differs")
        require(row["rows"] == int(row["case"].split(":rows=")[1].split(":")[0]), "timing rows differ")
        require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] > 0
                    for k in ("old_us", "new_us", "ratio")), "invalid event interval")
    return {"cases": cases, "timings": timings}


def analyze(root):
    results = []
    for gpu in (0, 1):
        memcheck = (root / f"memcheck-{gpu}.log").read_text()
        require("ERROR SUMMARY: 0 errors" in memcheck, "clean memcheck summary missing")
        parse(memcheck, True)
        results.append(parse((root / f"matrix-{gpu}.log").read_text(), False))
    require({c["case"]: c["output_sha256"] for c in results[0]["cases"]}
            == {c["case"]: c["output_sha256"] for c in results[1]["cases"]}, "peer bits differ")
    return {"passed": True, "cases_per_gpu": 192, "memcheck_cases_per_gpu": 6,
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
