#!/usr/bin/env python3
"""Per-kernel budget of serving decode steps from an Nsight SQLite export.

A step on one GPU is every kernel after the previous decode head GEMM up to
and including the next one (`--head`, a regular expression on the kernel
name; its `_m<rows>_` suffix names the bucket). Kernels overlap under PDL,
so a kernel's own duration overstates its cost; the budget charges each
kernel the advance of the latest end time it causes (end minus the previous
running maximum end, floored at zero, so idle gaps go to the kernel after
them); the charges sum to the step span.

Usage: serving_step_nsys.py <export.sqlite> [--device N] [--head REGEX]
       [--json OUT]
"""

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict

HEAD = r"_head_m(\d+)_"


def steps(connection, device, head):
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    rows = connection.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId = ? ORDER BY start",
        (device,),
    ).fetchall()
    pattern = re.compile(head)
    result = []
    current = []
    for start, end, name in rows:
        current.append((start, end, names[name]))
        match = pattern.search(names[name])
        if match:
            result.append((int(match.group(1)), current))
            current = []
    # The first step starts at an arbitrary point of the capture.
    return result[1:]


def budget(kernels):
    """Critical-path charge per kernel name: (count, microseconds)."""
    charged = defaultdict(lambda: [0, 0.0])
    latest = kernels[0][0]
    for _, end, name in kernels:
        advance = max(0, end - latest)
        charged[name][0] += 1
        charged[name][1] += advance / 1000
        latest = max(latest, end)
    return charged


def summarize(connection, device, head=HEAD):
    by_bucket = defaultdict(list)
    for rows, kernels in steps(connection, device, head):
        by_bucket[rows].append(kernels)
    report = {}
    for rows, runs in sorted(by_bucket.items()):
        spans = [(k[-1][1] - k[0][0]) / 1000 for k in runs]
        median_span = statistics.median(spans)
        # The step closest to the median span stands for the bucket.
        typical = min(runs, key=lambda k: abs((k[-1][1] - k[0][0]) / 1000 - median_span))
        charged = budget(typical)
        report[rows] = {
            "steps": len(runs),
            "median_span_us": round(median_span, 1),
            "kernels": len(typical),
            "budget": sorted(
                (
                    {"kernel": name, "count": count, "us": round(us, 1)}
                    for name, (count, us) in charged.items()
                ),
                key=lambda entry: -entry["us"],
            ),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--head", default=HEAD)
    parser.add_argument("--json")
    args = parser.parse_args()
    connection = sqlite3.connect(args.sqlite)
    report = summarize(connection, args.device, args.head)
    for rows, entry in report.items():
        print(
            f"bucket {rows}: {entry['steps']} steps, median {entry['median_span_us']} us, "
            f"{entry['kernels']} kernels"
        )
        for item in entry["budget"]:
            print(f"  {item['count']:5d} {item['us']:9.1f} us  {item['kernel']}")
    if args.json:
        with open(args.json, "w") as output:
            json.dump(report, output, indent=1)


if __name__ == "__main__":
    main()
