"""Summarize startup phases without adding nested or overlapping timers."""

import argparse
import json
import math
import pathlib
import re


def nonnegative(value, label, integer=False):
    kinds = (int,) if integer else (int, float)
    if type(value) not in kinds or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")


def diagnostics(counters):
    for name in ("io_timing", "buffer_timing", "shared_timing"):
        record = counters.get(name)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{name} is not an object")
        for field, value in record.items():
            if field in ("request_sizes", "request_gaps"):
                size = 8 if field == "request_sizes" else 7
                if not isinstance(value, list) or len(value) != size:
                    raise ValueError(f"invalid {name}.{field}")
                for count in value:
                    nonnegative(count, field, integer=True)
            else:
                nonnegative(value, f"{name}.{field}", integer=not field.endswith("_seconds"))


def summarize(text):
    result = {name: [] for name in ("direct_package", "weight_plan", "direct_load", "weight_upload",
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
    if result["direct_package"]:
        if len(result["direct_package"]) != 1:
            raise ValueError("multiple direct package preparations")
        measurements = result["direct_package"][0].get("measurements", {})
        seconds = measurements.get("total_seconds")
        nonnegative(seconds, "package preparation duration")
        result["package_preparation_seconds"] = seconds
        for stage in measurements.get("stages", {}).values():
            nonnegative(stage.get("seconds"), "package stage duration")
            nonnegative(stage.get("invocations"), "package stage invocation count", integer=True)
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
    if result["weight_pipeline"]:
        if len(result["weight_pipeline"]) != len(result["weight_upload"]):
            raise ValueError("pipeline and upload phase counts differ")
        for pipeline, upload in zip(result["weight_pipeline"], result["weight_upload"]):
            ranks = pipeline.get("ranks")
            if not isinstance(ranks, list) or not ranks:
                raise ValueError("pipeline has no rank counters")
            for rank in ranks:
                if not isinstance(rank, dict):
                    raise ValueError("pipeline rank is not an object")
                for name in ("bytes", "chunks"):
                    nonnegative(rank.get(name), f"rank {name}", integer=True)
                if (rank["bytes"] == 0) != (rank["chunks"] == 0):
                    raise ValueError("pipeline rank byte and chunk counts disagree")
                for name, value in rank.items():
                    if name.endswith("_seconds"):
                        nonnegative(value, name)
            if sum(rank["bytes"] for rank in ranks) != upload["uploaded_bytes"]:
                raise ValueError("pipeline rank bytes differ from the completed upload")
    for load in result["direct_load"]:
        diagnostics(load.get("cumulative", load))
        for phase in ("preparation", "upload"):
            diagnostics(load.get(phase, {}).get("counters", {}))
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
