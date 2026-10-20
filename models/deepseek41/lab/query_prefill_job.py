"""Validate and summarize the frozen prefill query fusion pair."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PATTERNS = (
    "finite", "zero", "denormal", "ties", "maximum", "nan_first",
    "nan_last", "inf", "weight_nan", "weight_inf", "sticky", "cancellation",
)
VARIANTS = ("cached", "reread")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def marker_records(text, marker):
    records = []
    for line in text.splitlines():
        at = line.find(marker + " ")
        if at >= 0:
            records.append(json.loads(line[at + len(marker) + 1:]))
    return records


def identities(memcheck):
    fixtures = ["real-L0", "real-L39"] + ([] if memcheck else list(PATTERNS))
    return {f"{fixture}:chop_prefill_query_norm_ceil_{variant}_v1"
            for fixture in fixtures for variant in VARIANTS}


def parse(text, memcheck):
    results = re.findall(
        r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    require(results == [("ok", "1", "0", "0")], "one executed passing test required")
    records = marker_records(text, "QUERY_PREFILL_CASE")
    require(len(records) == len(identities(memcheck)), "wrong executed case count")
    require({r["case"] for r in records} == identities(memcheck), "case identities differ")
    require(all(r["passed"] is True for r in records), "bitwise case failed")
    require(all(re.fullmatch("[0-9a-f]{64}", r.get("output_sha256", "")) for r in records),
            "output digest missing")
    require(all(r["rows"] == (8192 if r["case"].startswith("real-") else 64)
                for r in records), "case row count differs")
    require("QUERY_PREFILL_MISMATCH" not in text, "mismatch marker present")
    completed = re.findall(
        r"QUERY_PREFILL_COMPLETE memcheck=(true|false) cases=(\d+) "
        r"invariances=(\d+) negatives=(\d+) timings=(\d+) passed=(true|false)", text)
    expected = ("true", "4", "0", "0", "0", "true") if memcheck else (
        "false", "28", "12", "2", "8", "true")
    require(completed == [expected], "completion evidence differs")
    timings = marker_records(text, "QUERY_PREFILL_TIMING")
    require(len(timings) == (0 if memcheck else 8), "timing count differs")
    if not memcheck:
        require({(r["case"], r["hot"]) for r in timings}
                == {(case, hot) for case in identities(True) for hot in (False, True)},
                "timing identities differ")
    for row in timings:
        require(row["rows"] == 8192, "timing shape differs")
        require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k])
                    and row[k] > 0 for k in ("old_us", "new_us", "ratio")),
                "invalid timing interval")
    return {"cases": records, "timings": timings}


def analyze(directory):
    results = []
    for gpu in (0, 1):
        memcheck = (directory / f"memcheck-{gpu}.log").read_text()
        require("ERROR SUMMARY: 0 errors" in memcheck, "clean memcheck summary missing")
        parse(memcheck, True)
        result = parse((directory / f"matrix-{gpu}.log").read_text(), False)
        results.append(result)
    require({r["case"]: r["output_sha256"] for r in results[0]["cases"]}
            == {r["case"]: r["output_sha256"] for r in results[1]["cases"]},
            "peer output bits differ")
    return {
        "passed": True, "executed_cases_per_gpu": 28,
        "memcheck_cases_per_gpu": 4, "timing_intervals_per_gpu": 8,
        "timings": [
            dict(row, gpu=gpu,
                 projected_default_chunk_saving_ms=20 * (row["old_us"] - row["new_us"]) / 1000,
                 projected_full_row_schedule_saving_ms=40 * (row["old_us"] - row["new_us"]) / 1000)
            for gpu, result in enumerate(results) for row in result["timings"]
        ],
        "scope": "Paired kernel-event intervals. The round-seven profile has 20 eligible full-row query boundaries per default chunk; a full-row 40-layer schedule has 40.",
    }


def manifest(directory):
    records = []
    for path in sorted(directory.iterdir()):
        if path.name in ("SHA256SUMS", "manifest.json") or not path.is_file():
            continue
        require(not path.is_symlink(), "snapshot symlink refused")
        records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seal-manifest", action="store_true")
    args = parser.parse_args()
    if args.seal_manifest:
        result = manifest(args.directory)
        (args.directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"QUERY_PREFILL_SEALED files={len(result)}")
    else:
        result = analyze(args.directory)
        (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Both GPUs passed real 8K query fusion, memcheck and negative controls.")


if __name__ == "__main__":
    main()
