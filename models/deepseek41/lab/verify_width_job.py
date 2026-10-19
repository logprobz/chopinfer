#!/usr/bin/env python3
"""Run a frozen served verify-width comparison with the existing perf runner."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import sys
import threading
import time

CONCURRENCIES = (1, 8, 32)
OUTPUT_TOKENS = 64


def bits(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("missing or nonfinite logprob")
    return struct.pack("!d", value).hex()


def canonical(response):
    if len(response["choices"]) != 1:
        raise ValueError("one completion is required")
    choice = response["choices"][0]
    scores = choice["logprobs"]
    ids = choice["token_ids"]
    if (choice["finish_reason"] != "length"
            or response["usage"]["completion_tokens"] != OUTPUT_TOKENS
            or len(ids) != OUTPUT_TOKENS or any(type(i) is not int for i in ids)):
        raise ValueError("incomplete completion")
    for name in ("tokens", "token_logprobs", "top_logprobs"):
        if len(scores[name]) != OUTPUT_TOKENS:
            raise ValueError("incomplete logprobs")
    if any(not isinstance(row, dict) or not 1 <= len(row) <= 5
           or any(not isinstance(key, str) for key in row)
           for row in scores["top_logprobs"]):
        raise ValueError("invalid top logprobs")
    return {"token_ids": ids, "tokens": scores["tokens"],
            "selected_bits": [bits(p) for p in scores["token_logprobs"]],
            "top_bits": [{key: bits(p) for key, p in sorted(row.items())}
                         for row in scores["top_logprobs"]]}


def compare(left, right):
    if not left or left.keys() != right.keys():
        raise ValueError("comparison coverage differs")
    differences = []
    for name, rows in left.items():
        if not rows or len(rows) != len(right[name]):
            raise ValueError("comparison request coverage differs")
        for item, (a, b) in enumerate(zip(rows, right[name])):
            if a["request"] != b["request"]:
                raise ValueError("comparison requests differ")
            for field in ("token_ids", "tokens", "selected_bits", "top_bits"):
                if a["canonical"][field] != b["canonical"][field]:
                    differences.append({"case": name, "item": item, "field": field})
    return {"bitwise": not differences, "differences": differences}


def capture(p, server, prompts):
    result = {}
    for concurrency in CONCURRENCIES:
        for temperature in (0, 1):
            barrier = threading.Barrier(concurrency)

            def request(index):
                body = {"model": p.MODEL, "prompt": prompts[index],
                        "max_tokens": OUTPUT_TOKENS, "min_tokens": OUTPUT_TOKENS,
                        "ignore_eos": True, "temperature": temperature,
                        "top_p": 1, "seed": 19 + index, "logprobs": 5,
                        "return_token_ids": True, "stream": False}
                barrier.wait(timeout=20)
                response = json.loads(p.http(server.url, "/v1/completions", body, timeout=240))
                if response["usage"]["prompt_tokens"] != len(prompts[index]):
                    raise ValueError("prompt coverage differs")
                return {"request": body, "canonical": canonical(response), "response": response}

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                result[f"c{concurrency}-t{temperature}"] = list(pool.map(request, range(concurrency)))
    return result


def width_costs(rows, exact):
    """Report observed full cycle costs, never invent isolated verify costs."""
    groups = {}
    for row in rows:
        if row.get("prefill_tokens") != 0 or not row.get("verify_items"):
            continue
        live = row["decode_rows"] + row["verify_rows"]
        capacity = live if exact else 1 << (live - 1).bit_length()
        micros = row["execution_micros"]
        if live < 1 or not math.isfinite(micros) or micros <= 0:
            raise ValueError("invalid measured cycle cost")
        key = (row["decode_rows"] + row["verify_items"], live, capacity)
        groups.setdefault(key, []).append(micros)
    return [{"items": key[0], "live_rows": key[1], "graph_rows": key[2],
             "samples": len(values), "median_cycle_micros": statistics.median(values)}
            for key, values in sorted(groups.items())]


def validate(stage):
    manifest = json.loads((stage / "staged.json").read_text())
    for path, digest in manifest["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("staged artifact changed: " + path)
    for arm in manifest["arms"].values():
        install = Path(arm["install"])
        for path in [install, *install.rglob("*")]:
            if path.stat().st_mode & 0o022:
                raise ValueError("writable installation: " + str(path))
    return manifest


def idle(p, output, label, dry):
    if dry:
        return
    log = output / ("idle-" + label + ".csv")
    p.command(["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader,nounits"], log, seconds=15)
    if log.read_text().strip():
        raise RuntimeError("GPU compute processes remain after " + label)


def run(args):
    stage = Path(args.stage).resolve()
    sys.path.insert(0, str(stage))
    import perf_runner as p
    manifest = validate(stage)
    if manifest.get("cpu_dry_run_only") and not args.dry_run:
        raise ValueError("CPU-only stage has no candidate installation")
    if not args.dry_run and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("the job must select all four GPUs explicitly")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "dry_run": args.dry_run, "servers": {},
              "comparisons": {}, "errors": [], "stage": manifest,
              "limitations": ["Confidence callback is pending; this stage measures exact-width graphs.",
                              "The 32-row package budget leaves no spare rows at full C32.",
                              "Cycle costs include drafting; they are not isolated verify kernel timings."]}
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, p.interrupted)
    try:
        result["before"] = p.telemetry(output, "before", args.dry_run, since)
        captures = {}
        prompts = None
        for arm, profile in (("base", "tp4-ar"), ("base", "tp4-dspark"), ("candidate", "tp4-dspark")):
            p.INSTALL = Path(manifest["arms"][arm]["install"])
            p.CACHE = Path(manifest["arms"][arm]["cache"])
            parent = output / arm
            parent.mkdir(exist_ok=True)
            server = p.Server(parent, profile, dry=args.dry_run, use_nsys=False)
            name = arm + "-" + profile
            record = result["servers"][name] = {}
            try:
                server.start()
                if prompts is None:
                    prompts = [json.loads(p.http(server.url, "/tokenize", {
                        "model": p.MODEL, "prompt": p.bench.stream_prompt("diverse", i),
                        "add_special_tokens": False}))['tokens'] for i in range(32)]
                    p.write(output / "prompts.json", prompts)
                first = capture(p, server, prompts)
                p.write(server.directory / "exactness.json", first)
                repeat = capture(p, server, prompts)
                p.write(server.directory / "repeat.json", repeat)
                result["comparisons"][name + "-repeat"] = compare(first, repeat)
                captures[name] = first
                if profile == "tp4-dspark":
                    record["decode"] = {str(c): p.decode(server, c) for c in CONCURRENCIES}
                record["observed_width_costs"] = width_costs(p.trace_rows(server.directory / "server.log"), arm == "candidate")
            finally:
                server.stop()
                idle(p, output, name, args.dry_run)
            p.write(output / "summary.json", result)
        for name in ("base-tp4-dspark", "candidate-tp4-dspark"):
            result["comparisons"][name + "-against-ar"] = compare(captures["base-tp4-ar"], captures[name])
        result["comparisons"]["base-candidate-dspark"] = compare(captures["base-tp4-dspark"], captures["candidate-tp4-dspark"])
        if not all(row["bitwise"] for row in result["comparisons"].values()):
            raise ValueError("served token or logprob bits differ; performance is not qualified")
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            idle(p, output, "final", args.dry_run)
            result["after"] = p.telemetry(output, "after", args.dry_run, since)
            if result["after"]["xids"]:
                raise ValueError("kernel Xids during job")
        except BaseException as error:
            result["status"] = "failed"
            result["errors"].append(str(error))
        p.write(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(output), "errors": result["errors"]}), flush=True)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(run(parser.parse_args()))
