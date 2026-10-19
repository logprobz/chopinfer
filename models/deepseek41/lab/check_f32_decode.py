#!/usr/bin/env python3
"""Check complete exact-F32 records and the fixed 3% paired timing gate."""

import argparse
import json
import math
from pathlib import Path
import re


MAX_RATIO = 1.03
RECORD = re.compile(r"(?:^|\s)(F32_(?:BITWISE|TIMING|LAUNCH) .*)$")


def records(text):
    """Libtest may put its test-name prefix before the first record."""
    result = {kind: [] for kind in ("BITWISE", "TIMING", "LAUNCH")}
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
    for _, value in parsed["BITWISE"]:
        if not value.get("passed"):
            raise ValueError("bitwise record did not pass")
    for label, value in parsed["TIMING"]:
        old, new, ratio = (float(value[key]) for key in ("old_us", "new_us", "ratio"))
        if not all(math.isfinite(x) and x > 0 for x in (old, new, ratio)):
            raise ValueError(f"{label}: invalid paired timing")
        if ratio > MAX_RATIO:
            raise ValueError(f"{label} rows={value['rows']}: ratio {ratio} exceeds {MAX_RATIO}")
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
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.coverage.read_text())
    for log in args.logs:
        check(log.read_text(), expected)
    print(f"{len(args.logs)} GPU logs passed completeness and routed/baseline <= {MAX_RATIO}")


if __name__ == "__main__":
    main()
