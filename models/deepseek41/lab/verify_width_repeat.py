#!/usr/bin/env python3
"""Repeat verify-width measurements in balanced order on an idle CPU host."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import verify_width_job as exact

CONCURRENCIES = (1, 8, 16, 32)
ORDER = ("base", "candidate", "candidate", "base")


def require_idle(load):
    if load.get("idle") is not True or load.get("timing") != "evidence":
        raise ValueError("host timing is informational; arrange an idle CPU host before rerunning")


def decision(passes):
    if len(passes) != 4 or tuple(row["arm"] for row in passes) != ORDER:
        raise ValueError("need all four balanced passes")
    result = {}
    for c in CONCURRENCIES:
        rows = [row["decode"][str(c)] for row in passes]
        for row in rows:
            require_idle(row["host_load"])
        rates = [row["steady"]["aggregate_tokens_per_second"] for row in rows]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("invalid measured throughput")
        ratios = [rates[1] / rates[0], rates[2] / rates[3]]
        result[str(c)] = {"base_tokens_per_second": [rates[0], rates[3]],
                          "candidate_tokens_per_second": [rates[1], rates[2]],
                          "paired_ratios": ratios,
                          "median_base_tokens_per_second": statistics.median([rates[0], rates[3]]),
                          "median_candidate_tokens_per_second": statistics.median([rates[1], rates[2]])}
    control = all(row["decode"]["32"]["acceptance"]["verify_items"] == 0 for row in passes)
    stable = control and all(0.97 <= r <= 1.03 for r in result["32"]["paired_ratios"])
    no_regressions = all(r >= 0.97 for c in ("1", "8", "16") for r in result[c]["paired_ratios"])
    repeatable_gain = any(all(r > 1.03 for r in result[c]["paired_ratios"]) for c in ("1", "8", "16"))
    return {"concurrencies": result, "c32_has_no_verify": control,
            "stable_control": stable, "no_regressions": no_regressions,
            "repeatable_gain": repeatable_gain,
            "route_d14": stable and no_regressions and repeatable_gain,
            "rule": "both pairs: C32 within 3%, no C1/C8/C16 regression over 3%, and one concurrency gains over 3%"}


def measure(p, server, concurrency):
    # Keep the existing benchmark and steady-window calculation. Shorten the
    # wider cohorts so four fresh server passes fit one bounded GPU slot.
    tokens = 512 if concurrency <= 8 else 256
    directory = server.directory / f"decode-c{concurrency}"
    directory.mkdir()
    before = p.metrics(server, f"c{concurrency}-before")
    argv = [sys.executable, "-B", p.SOURCE / "tools/qual/bench.py", "--url", server.url,
            "--model", p.MODEL, "--concurrencies", str(concurrency), "--max-sequences", "32",
            "--output-tokens", str(tokens), "--prompts", "diverse", "--sampling", "greedy",
            "--timeout", "180", "--spec-metrics-arm", "chopinfer", "--prefill-tokens", "128",
            "--prefill-repeats", "8", "--prefill-prefix", f"short untimed auxiliary {time.time_ns()} "]
    p.command(argv, directory / "bench.log", seconds=210, env=p.env_for(directory))
    receipts = list((directory / "receipts").glob("*-bench.json"))
    if len(receipts) != 1:
        raise ValueError("expected one benchmark receipt")
    receipt = json.loads(receipts[0].read_text())
    if not receipt["passed"] or receipt["skipped"] or receipt["mechanics_only"]:
        raise ValueError("benchmark did not execute the requested workload")
    row = receipt["decode"][0]
    if len(row["requests"]) != concurrency or any(r["usage"]["completion_tokens"] != tokens for r in row["requests"]):
        raise ValueError("incomplete performance cohort")
    delta = p.spec_metrics.difference(before, p.metrics(server, f"c{concurrency}-after"))
    verified, accepted, drafted = (delta[key] for key in ("verify_items", "accepted_tokens", "draft_tokens"))
    return {"output_tokens": tokens, "steady": row["steady"], "cohort": row["cohort"],
            "host_load": receipt["host_load"], "receipt": str(receipts[0]), "metrics_delta": delta,
            "acceptance": {"verify_items": verified, "accepted_tokens": accepted, "draft_tokens": drafted,
                           "accepted_per_verify_item": accepted / verified if verified else None,
                           "accepted_fraction": accepted / drafted if drafted else None}}


def run(args):
    stage = Path(args.stage).resolve()
    sys.path.insert(0, str(stage))
    import perf_runner as p
    from common import host_load, host_load_sample
    manifest = exact.validate(stage)
    if not args.dry_run and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("the job must select all four GPUs explicitly")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "dry_run": args.dry_run, "stage": manifest,
              "passes": [], "comparisons": {}, "errors": [], "route_d9": False}
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, p.interrupted)

    def check_host(label):
        samples = [host_load_sample()]
        time.sleep(1)
        samples.append(host_load_sample())
        load = host_load(samples)
        p.write(output / f"host-{label}.json", load)
        if not args.dry_run:
            require_idle(load)

    try:
        check_host("before-loading")
        result["before"] = p.telemetry(output, "before", args.dry_run, since)
        prompts = None
        captures = []
        for number, arm in enumerate(ORDER):
            p.INSTALL = Path(manifest["arms"][arm]["install"])
            p.CACHE = Path(manifest["arms"][arm]["cache"])
            parent = output / f"pass-{number}-{arm}"
            parent.mkdir()
            server = p.Server(parent, "tp4-dspark", dry=args.dry_run, use_nsys=False)
            record = {"arm": arm, "decode": {}}
            result["passes"].append(record)
            try:
                server.start()
                record["readiness_seconds"] = server.readiness
                check_host(f"pass-{number}-ready")
                if prompts is None:
                    prompts = [json.loads(p.http(server.url, "/tokenize", {
                        "model": p.MODEL, "prompt": p.bench.stream_prompt("diverse", i),
                        "add_special_tokens": False}))["tokens"] for i in range(32)]
                    p.write(output / "prompts.json", prompts)
                captured = exact.capture(p, server, prompts, "exactness", CONCURRENCIES)
                captures.append(captured)
                record["exactness"] = exact.capture_stats(captured)
                if number:
                    compared = exact.compare(captures[0], captured)
                    result["comparisons"][f"pass-0-pass-{number}"] = compared
                    if not compared["bitwise"]:
                        raise ValueError("served token IDs or logprob bits differ")
                for c in CONCURRENCIES:
                    row = measure(p, server, c)
                    record["decode"][str(c)] = row
                    p.write(output / "summary.json", result)
                    if not args.dry_run:
                        require_idle(row["host_load"])
            finally:
                server.stop()
                exact.idle(p, output, f"pass-{number}", args.dry_run)
        if args.dry_run:
            result["recommendation"] = "CPU protocol check only; no routing evidence"
        else:
            result["performance"] = decision(result["passes"])
            result["recommendation"] = "D14 eligible for integration validation" if result["performance"]["route_d14"] else "hold D14"
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            exact.idle(p, output, "final", args.dry_run)
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
