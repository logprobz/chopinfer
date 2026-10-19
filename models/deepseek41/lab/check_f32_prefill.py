#!/usr/bin/env python3
"""Validate prefill candidate evidence and report measured winning buckets."""
import argparse
import json
import math
from pathlib import Path

from check_f32_decode import check_host_times, records, write_host_times


def check(texts, expected):
    if len(texts) != expected["shards"]:
        raise ValueError("missing shard log")
    for ordinal, text in enumerate(texts):
        if "1 passed; 0 failed; 0 ignored" not in text:
            raise ValueError(f"shard {ordinal} did not pass")
        if "skipping" in text.lower() or "skipped" in text.lower():
            raise ValueError("GPU checks were skipped")
        for _, value in records(text)["BITWISE"]:
            if "case" not in value or int(value["case"]).bit_count() % len(texts) != ordinal:
                raise ValueError("case is missing its ID or is in the wrong shard")
    parsed = records("\n".join(texts))
    for kind, key in (("BITWISE", "bitwise_cases"), ("TIMING", "timing_cases"), ("LAUNCH", "launch_timings")):
        if len(parsed[kind]) != expected[key]:
            raise ValueError(f"{kind}: expected {expected[key]}, found {len(parsed[kind])}")
    check_host_times(parsed, expected)
    ids = [int(value["case"]) for _, value in parsed["BITWISE"]]
    if len(set(ids)) != len(ids) or set(ids) != set(range(expected["bitwise_cases"])):
        raise ValueError("case IDs are duplicated or incomplete")
    if not all(value.get("passed") for _, value in parsed["BITWISE"]):
        raise ValueError("bitwise case failed")
    for _, value in parsed["TIMING"]:
        numeric = [float(value[k]) for k in ("old_us", "new_us", "ratio", "floor_us", "floor_ratio")]
        if not all(math.isfinite(v) and v > 0 for v in numeric):
            raise ValueError("invalid paired timing")
        if not math.isclose(numeric[1] / numeric[3], numeric[4], rel_tol=1e-5):
            raise ValueError("inconsistent floor ratio")
    for _, value in parsed["LAUNCH"]:
        if not math.isfinite(float(value["us"])) or float(value["us"]) <= 0:
            raise ValueError("invalid launch timing")
    measured = []
    for entry in expected["production"]:
        for rows in entry["rows"]:
            bits = [v for name, v in parsed["BITWISE"] if name == entry["baseline"] and int(v["rows"]) == rows
                    and int(v["n"]) == entry["n"] and int(v["k"]) == entry["k"]]
            timing = [v for name, v in parsed["TIMING"] if name == entry["baseline"] and int(v["rows"]) == rows]
            if len(bits) != 1 or len(timing) != 1:
                raise ValueError("missing or duplicated entry/bucket evidence")
            value = timing[0]
            if value["case"] != bits[0]["case"]:
                raise ValueError("timing and bitwise case IDs differ")
            if not math.isclose(float(value["floor_us"]), entry["weight_bytes"] / expected["bytes_per_us"], rel_tol=1e-5):
                raise ValueError("floor does not use stored weight bytes")
            measured.append(dict(reference=entry["reference"], candidate=entry["candidate"], tile=entry["tile"],
                rows=rows, old_us=float(value["old_us"]), new_us=float(value["new_us"]), ratio=float(value["ratio"]),
                floor_ratio=float(value["floor_ratio"]), wins=float(value["ratio"]) <= expected["winning_ratio"]))
    best = {}
    for row in measured:
        if row["wins"]:
            key = (row["reference"], row["rows"])
            if key not in best or row["ratio"] < best[key]["ratio"]:
                best[key] = row
    first_wins = {}
    for row in best.values():
        first_wins[row["reference"]] = min(first_wins.get(row["reference"], row["rows"]), row["rows"])
    return {"measured": measured, "winning_buckets": list(best.values()), "first_measured_win": first_wins,
            "winning_ratio": expected["winning_ratio"], "serving_promotion": False,
            "note": "Measured points only. Do not interpolate a crossover or install an unmeasured row range."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    texts = [log.read_text() for log in args.logs]
    for log, text in zip(args.logs, texts):
        write_host_times(log, records(text))
    # Preserve every available timing even when correctness or completion fails.
    table = ["| Candidate | Rows | Old us | New us | Ratio | Weight floor us | New / floor |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in records("\n".join(texts))["TIMING"]:
        table.append("| " + " | ".join([name] + [value[k] for k in ("rows", "old_us", "new_us", "ratio", "floor_us", "floor_ratio")]) + " |")
    out = args.logs[0].parent
    (out / "prefill-timings.md").write_text("\n".join(table) + "\n")
    result = check(texts, json.loads(args.coverage.read_text()))
    (out / "prefill-crossovers.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Prefill bitwise coverage passed; {len(result['winning_buckets'])} measured winning buckets, serving unchanged")


if __name__ == "__main__":
    main()
