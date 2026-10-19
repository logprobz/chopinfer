#!/usr/bin/env python3
"""Check complete exact-F32 records and the 3 percent gate for routed buckets."""

import argparse
import json
import math
from pathlib import Path
import re


MAX_RATIO = 1.03
VERIFY_GOAL_RATIO = 0.2
RECORD = re.compile(r"(F32_(?:BITWISE|TIMING|LAUNCH|HOST|SETUP|POISON) .*)$")


def records(text):
    """Libtest may put its test-name prefix before the first record."""
    result = {kind: [] for kind in ("BITWISE", "TIMING", "LAUNCH", "HOST", "SETUP", "POISON")}
    for line in text.splitlines():
        match = RECORD.search(line)
        if not match:
            continue
        tag, label, *fields = match.group(1).split()
        values = {}
        for field in fields:
            if field == "PASS":
                values["passed"] = True
                continue
            key, separator, value = field.partition("=")
            if not separator or key in values:
                raise ValueError(f"malformed or duplicate field: {field}")
            values[key] = value
        result[tag.removeprefix("F32_")].append((label, values))
    return result


def check_host_times(parsed, expected):
    if "poison_checks" in expected:
        if len(parsed["POISON"]) != expected["poison_checks"] or not all(
                v.get("passed") and int(v["bytes"]) > 16 << 20 for _, v in parsed["POISON"]):
            raise ValueError("missing or failed poison expansion check")
    if "host_cases" not in expected:
        return
    if len(parsed["HOST"]) != expected["host_cases"]:
        raise ValueError(f"HOST: expected {expected['host_cases']}, found {len(parsed['HOST'])}")
    key = lambda record: (record[0], record[1].get("case"), record[1]["rows"])
    bitwise = {key(record) for record in parsed["BITWISE"]}
    host = [key(record) for record in parsed["HOST"]]
    if len(set(host)) != len(host) or set(host) != bitwise:
        raise ValueError("host record does not match a bitwise case")
    for _, value in parsed["HOST"]:
        times = [float(value[k]) for k in
                 ("fixture_ms", "setup_ms", "poison_ms", "comparison_ms", "timing_ms", "total_ms")]
        if not all(math.isfinite(t) and t >= 0 for t in times) or times[-1] <= 0:
            raise ValueError("invalid host time")
        if times[-1] + 0.1 < sum(times[:-1]):
            raise ValueError("inconsistent host time")


def write_host_times(log, parsed):
    """Keep wall-clock diagnostics even if a shard times out or the gate fails."""
    log.with_suffix(".host.json").write_text(json.dumps({
        "cases": parsed["HOST"], "weight_setup": parsed["SETUP"],
        "note": "Host wall time includes GPU waits and device readback; it is not CPU utilization."
    }, indent=2) + "\n")


def routed(label, rows, expected):
    entry = next((entry for entry in expected["production"] if entry["baseline"] == label), None)
    if entry is None:
        return False
    # Older receipts predate prefill promotion and routed only rows <=256.
    return rows in entry.get("routed_rows", [r for r in entry["rows"] if r <= 256])


def speed_goals(parsed, expected):
    buckets = [dict(entry=label, rows=int(value["rows"]), speedup=1 / float(value["ratio"]),
                    fivefold=float(value["ratio"]) <= VERIFY_GOAL_RATIO)
               for label, value in parsed["TIMING"]
               if 33 <= int(value["rows"]) <= 256 and routed(label, int(value["rows"]), expected)]
    return {"goal_speedup": 5, "met": sum(bucket["fivefold"] for bucket in buckets),
            "buckets": buckets, "pass_condition": False}


def check(text, expected):
    if "1 passed; 0 failed; 0 ignored" not in text:
        raise ValueError("GPU test did not pass")
    if "skipping" in text.lower() or "skipped" in text.lower():
        raise ValueError("GPU test did not execute every check")
    parsed = records(text)
    for kind, key in (("BITWISE", "bitwise_cases"), ("TIMING", "timing_cases"),
                      ("LAUNCH", "launch_timings")):
        if len(parsed[kind]) != expected[key]:
            raise ValueError(f"{kind}: expected {expected[key]}, found {len(parsed[kind])}")
    check_host_times(parsed, expected)
    for _, value in parsed["BITWISE"]:
        if not value.get("passed"):
            raise ValueError("bitwise record did not pass")
    for label, value in parsed["TIMING"]:
        old, new, ratio = (float(value[key]) for key in ("old_us", "new_us", "ratio"))
        if not all(math.isfinite(x) and x > 0 for x in (old, new, ratio)):
            raise ValueError(f"{label}: invalid paired timing")
        if routed(label, int(value["rows"]), expected) and ratio > MAX_RATIO:
            raise ValueError(f"{label} rows={value['rows']}: ratio {ratio} exceeds {MAX_RATIO}")
        floor, floor_ratio = float(value["floor_us"]), float(value["floor_ratio"])
        if not all(math.isfinite(x) and x > 0 for x in (floor, floor_ratio)):
            raise ValueError("invalid bandwidth floor")
        if not math.isclose(new / floor, floor_ratio, rel_tol=1e-5):
            raise ValueError("bandwidth floor ratio is inconsistent")
    for _, value in parsed["LAUNCH"]:
        micros = float(value["us"])
        if not math.isfinite(micros) or micros <= 0:
            raise ValueError("invalid launch timing")
    for case in expected["production"]:
        label = case["baseline"]
        for rows in case["rows"]:
            matches = [value for name, value in parsed["BITWISE"]
                       if name == label and int(value["rows"]) == rows
                       and int(value["n"]) == case["n"] and int(value["k"]) == case["k"]]
            if len(matches) != 1:
                raise ValueError(f"{label} rows={rows}: missing or duplicate bitwise case")
        for rows in case["timing_rows"]:
            matches = [value for name, value in parsed["TIMING"]
                       if name == label and int(value["rows"]) == rows]
            if len(matches) != 1:
                raise ValueError(f"{label} rows={rows}: missing or duplicate timing")
            floor = case["weight_bytes"] / expected["bytes_per_us"]
            if not math.isclose(float(matches[0]["floor_us"]), floor, rel_tol=1e-5):
                raise ValueError(f"{label}: floor does not use stored weight bytes")
    return parsed


def check_shards(texts, expected):
    count = expected.get("shards", 1)
    if len(texts) != count:
        raise ValueError(f"expected {count} shard logs, found {len(texts)}")
    for ordinal, text in enumerate(texts):
        if "1 passed; 0 failed; 0 ignored" not in text:
            raise ValueError(f"shard {ordinal}: GPU test did not pass")
        if count > 1:
            for _, value in records(text)["BITWISE"]:
                if "case" not in value:
                    raise ValueError("missing case ID")
                if int(value["case"]).bit_count() % count != ordinal:
                    raise ValueError(f"case {value['case']} is in the wrong shard")
    joined = "\n".join(texts)
    if count > 1:
        ids = [int(value["case"]) for _, value in records(joined)["BITWISE"]]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case IDs across shards")
        if set(ids) != set(range(expected["bitwise_cases"])):
            raise ValueError("shards omit expected case IDs")
    return check(joined, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.coverage.read_text())
    texts = [log.read_text() for log in args.logs]
    for log, text in zip(args.logs, texts):
        parsed = records(text)
        write_host_times(log, parsed)
        table = ["| Entry | Rows | Old us | New us | New / old | Weight floor us | New / floor |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for label, value in parsed["TIMING"]:
            table.append("| " + " | ".join([label] + [value[key] for key in
                ("rows", "old_us", "new_us", "ratio", "floor_us", "floor_ratio")]) + " |")
        log.with_suffix(".timings.md").write_text("\n".join(table) + "\n")
        log.with_suffix(".timings.json").write_text(json.dumps(parsed["TIMING"], indent=2) + "\n")
    try:
        check_shards(texts, expected)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    parsed = records("\n".join(texts))
    goals = speed_goals(parsed, expected)
    (args.logs[0].parent / "speed-goals.json").write_text(json.dumps(goals, indent=2) + "\n")
    print(f"{len(args.logs)} GPU logs passed: {len(parsed['BITWISE'])} bitwise PASS; routed buckets <= {MAX_RATIO}")
    print(f"Fivefold goal: {goals['met']}/{len(goals['buckets'])} routed verify buckets; informational only")



if __name__ == "__main__":
    main()
