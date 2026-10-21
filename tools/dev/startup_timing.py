"""Summarize startup phases without adding nested or overlapping timers."""

import argparse
import json
import math
import pathlib
import re


def summarize(text):
    result = {name: [] for name in ("weight_plan", "direct_load", "weight_upload",
                                    "weight_pipeline")}
    elapsed = {"bundle create": "bundle_create_seconds",
               "generation warmup": "warmup_seconds", "readiness": "readiness_seconds"}
    for line in text.splitlines():
        match = re.fullmatch(r"startup ([^:]+): (.*)", line)
        if not match:
            continue
        name, value = match.groups()
        if name in result:
            record = json.loads(value)
            if not isinstance(record, dict):
                raise ValueError(f"{name} is not an object")
            result[name].append(record)
        elif name in elapsed:
            number = re.match(r"([0-9.]+)s(?:\s|$)", value)
            if not number or elapsed[name] in result:
                raise ValueError(f"invalid or repeated {name}")
            result[elapsed[name]] = float(number[1])
    if not result["weight_upload"] or "readiness_seconds" not in result:
        raise ValueError("startup did not finish a weight upload and reach readiness")
    for record in result["weight_upload"]:
        if record.get("status") != "complete":
            raise ValueError("weight upload did not complete")
        size = record.get("uploaded_bytes")
        if type(size) is not int or size <= 0 or size != record.get("planned_bytes"):
            raise ValueError("uploaded bytes do not match a nonempty planned arena")
        seconds = record.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid upload duration")
    if any(record.get("status") != "complete" for record in result["weight_pipeline"]):
        raise ValueError("weight pipeline did not complete")
    for name in elapsed.values():
        if name in result and (not math.isfinite(result[name]) or result[name] < 0):
            raise ValueError(f"invalid {name}")
    result["timing_scope"] = "phases overlap; worker seconds are not elapsed seconds"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.log.read_text()), indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"startup summary: {error}\n")


if __name__ == "__main__":
    main()
