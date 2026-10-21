#!/usr/bin/env python3
"""Run and audit the native MoE tuning pair (native_tune.md) on a GPU pair.

One phase: the frozen test binary's `tune::native_moe_tune_gpu` once per
worker with one logical GPU each. Worker 0 takes the balanced synthetic
case and captured layer 14 on rank 0, worker 1 the skewed synthetic case
and captured layer 39 on rank 1. Every arm of both projections must be
timed, and the bitwise arms must write the served bytes. The job fails
closed on any missing, duplicate or failed case or arm, zero executed
tests, a power cap other than 250 W, or an Xid in the kernel journal. The
record, listing, power, seal and timing helpers are native_moe_job's.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import native_moe_job as base

KINDS = ("GateUp", "Down")
ARMS = ("lean", "no_checks", "no_mma", "no_global", "mma_only", "paired", "paired_lean")
BITWISE = ("lean", "paired", "paired_lean")
CASES = {0: ("e384-balanced", "captured-l14-r0"), 1: ("e384-skewed", "captured-l39-r1")}
TEST = "tune::native_moe_tune_gpu"
CPU_TESTS = ("tune::tune_cpu_plan",)
BINARY = "bin/native-moe-test"
ROWS = 8192
# Full layers per 8192-row chunk of a cold 32K prompt (round 7 profile).
FULL_LAYERS = 20


def check_arm(arm, label):
    expected = True if arm["arm"] in BITWISE else None
    if arm["bitwise"] is not expected:
        raise ValueError(f"{label}: bitwise {arm['bitwise']}, expected {expected}")
    base.validate_timing(arm["timing"])


def analyze(text, worker):
    base.libtest(text, 1)
    rows = base.records(text, "native_moe_tune")
    base.unique(rows, lambda r: r["case"], CASES[worker], "tune")
    for r in rows:
        if r["rows"] != ROWS:
            raise ValueError(f"{r['case']}: {r['rows']} rows")
        if [k["kind"] for k in r["kinds"]] != list(KINDS):
            raise ValueError(f"{r['case']}: missing or extra projections")
        for k in r["kinds"]:
            if [a["arm"] for a in k["arms"]] != list(ARMS):
                raise ValueError(f"{r['case']} {k['kind']}: missing or extra arms")
            for a in k["arms"]:
                check_arm(a, f"{r['case']} {k['kind']} {a['arm']}")
    done = base.records(text, "native_moe_tune_done")
    if done != [{"worker": worker, "cases": len(rows), "failed": 0}]:
        raise ValueError("missing case completion")
    return rows


def summarize(parsed):
    """Report-only per-arm timing: the served and arm medians per case, the
    mean over cases, and the served minus arm saving per chunk."""
    out = {}
    rows = [r for p in parsed for r in p]
    for kind in KINDS:
        out[kind] = {}
        for arm in ARMS:
            t = [(r["case"], a["timing"]) for r in rows for k in r["kinds"] if k["kind"] == kind
                 for a in k["arms"] if a["arm"] == arm]
            served = sum(x["ms"][0] for _, x in t) / len(t)
            ms = sum(x["ms"][1] for _, x in t) / len(t)
            out[kind][arm] = {
                "cases": {c: {"served_ms": x["ms"][0], "arm_ms": x["ms"][1], "ratio": x["median_round_ratio"]} for c, x in t},
                "served_ms_mean": served, "arm_ms_mean": ms,
                "per_chunk_saving_ms": FULL_LAYERS * (served - ms)}
    return out


def environment(root, plan, worker=None):
    env = base.environment(root, plan, "timing", worker)
    env["CHOPINFER_NATIVE_MOE_PROBE"] = "0"
    return env


def test_args(root, name):
    return [str(root / BINARY), name, "--exact", "--include-ignored", "--nocapture"]


def run(root, plan, out):
    deadline = time.monotonic() + plan["budget"]["tune"]
    children = []
    try:
        for worker in (0, 1):
            log = out / f"tune-{worker}.log"
            handle = log.open("w")
            p = subprocess.Popen([*test_args(root, TEST), "--test-threads=1"], env=environment(root, plan, worker),
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((p, handle, log))
        for p, _, log in children:
            if p.wait(timeout=max(1, deadline - time.monotonic())):
                raise ValueError(f"tune child failed: {log}")
    finally:
        for p, h, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            h.close()
    return [analyze(log.read_text(), worker) for worker, (_, _, log) in enumerate(children)]


def dry_run(root, plan, out):
    base.seal(root, plan)
    env = environment(root, plan)

    def listing(argv, name, log):
        r = subprocess.run([*argv, "--list"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        (out / log).write_text(r.stdout)
        if r.returncode:
            raise ValueError(f"dry-run listing failed: {name}")
        base.check_listed(r.stdout, name)

    for i, t in enumerate(CPU_TESTS):
        listing(test_args(root, t), t, f"cpu-{i}.list")
        r = subprocess.run(test_args(root, t), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=600)
        (out / f"cpu-{i}.log").write_text(r.stdout)
        if r.returncode:
            raise ValueError(f"dry-run test failed: {t}")
        base.libtest(r.stdout, 1)
    listing([*test_args(root, TEST), "--test-threads=1"], TEST, "tune.list")
    (out / "dry-run.json").write_text(json.dumps({"cpu_tests": len(CPU_TESTS), "gpu_commands_listed": [TEST],
                                                  "gpu_launched": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise TimeoutError(f"job interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.root / "PLAN.json").read_text())
    if args.dry_run:
        dry_run(args.root, plan, args.out)
        return
    base.seal(args.root, plan)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 2:
        raise ValueError("a pair job needs two runner-assigned GPUs")
    started = int(time.time())
    verdict = {"phase": "tune", "exit_code": 1, "power_before": base.gpus(visible, True)}
    error = None
    try:
        parsed = run(args.root, plan, args.out)
        (args.out / "parsed.json").write_text(json.dumps(parsed) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summarize(parsed), indent=2) + "\n")
        verdict["exit_code"] = 0
    except Exception as e:  # recorded in the verdict, then re-raised
        error = f"{type(e).__name__}: {e}"
        verdict["error"] = error
    finally:
        try:
            verdict["power_after"] = base.gpus(visible, False)
        except Exception as e:
            verdict["power_after_error"] = str(e)
            verdict["exit_code"] = 1
        kernel = subprocess.run(["journalctl", "-k", "--no-pager", "--since", "@" + str(started)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        (args.out / "kernel.log").write_text(kernel.stdout)
        verdict["xids"] = [l for l in kernel.stdout.splitlines() if re.search(r"NVRM: Xid|Xid \(", l)]
        if kernel.returncode or verdict["xids"]:
            verdict["exit_code"] = 1
        verdict["elapsed_seconds"] = time.time() - started
        (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["exit_code"]:
        raise SystemExit(error or "power, journal or Xid check failed")
    print("native MoE tuning passed on both workers")


if __name__ == "__main__":
    main()
