#!/usr/bin/env python3
"""Run and audit the narrow pair (router_narrow.md) on a GPU pair.

Phases, each only after the previous passed: ``memcheck`` runs the frozen
test binary's ``narrow::router_narrow_gpu`` under compute-sanitizer
memcheck below 8192 live rows, ``pair`` runs every case with the paired
timing. Each phase runs once per worker with one logical GPU each (worker
0 the clean pattern, worker 1 the outliers) and fails closed on any
missing, duplicate or failed case, a differing byte, a written candidate
widened row, zero executed tests, a sanitizer error, a power cap other
than 250 W, or an Xid on its GPUs. The power, Xid and seal helpers are
router_wide_job's; the record, listing and timing helpers native_moe_job's.
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
import router_wide_job as wide

TEST = "narrow::router_narrow_gpu"
CPU_TESTS = ("narrow::router_narrow_cpu_plan",)
BINARY = "bin/router-norm-test"
PATTERNS = {0: "clean", 1: "outliers"}
LIVES = (1, 65, 2049, 8192)
MEMCHECK_LIVES = (1, 65, 2049)
COMPARED = ("normed", "codes", "scales", "status", "flags", "logits")
PHASES = ("memcheck", "pair")
ROWS = 8192
# Certifying norms and routers per rank in an 8192-row step (the TP4
# journal's 40 MoE layers); the per-chunk saving counts each once.
LAYERS = 40


def cases(worker, memcheck):
    return [(PATTERNS[worker], l) for l in (MEMCHECK_LIVES if memcheck else LIVES)]


def analyze(text, worker, phase):
    memcheck = phase == "memcheck"
    base.libtest(text, 1)
    rows = base.records(text, "router_narrow_record")
    base.unique(rows, lambda r: (r["pattern"], r["live"]), cases(worker, memcheck), "narrow")
    for r in rows:
        label = f"{r['pattern']} live {r['live']}"
        if r["passed"] is not True or sorted(r["differing_bytes"]) != sorted(COMPARED) \
                or any(r["differing_bytes"].values()):
            raise ValueError(f"{label}: outputs differ")
        if r["candidate_widened_untouched"] is not True or r["bound_widened_consistent"] is not True:
            raise ValueError(f"{label}: widened rows written by the candidate or inconsistent")
        if r["live"] == ROWS and not memcheck:
            base.validate_timing(r["timing"])
        elif r["timing"] is not None:
            raise ValueError(f"{label}: timed outside the pair's 8192-row case")
    if worker == 1 and not any(r["uncertified_rows"] > 0 for r in rows):
        raise ValueError("no outlier case reaches the SIMT path")
    done = base.records(text, "router_narrow_done")
    if done != [{"worker": worker, "cases": len(rows), "failed": 0, "memcheck": memcheck}]:
        raise ValueError("missing case completion")
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors?", text)
    if memcheck and summaries != ["0"]:
        raise ValueError("missing clean memcheck summary")
    if not memcheck and summaries:
        raise ValueError("the pair ran under the sanitizer")
    return rows


def summarize(parsed):
    """Report-only: each worker's 8192-row bound and candidate medians, their
    ratio, and the saving per chunk if every layer saved as much."""
    out = {}
    for rows in parsed:
        for r in rows:
            if r["timing"]:
                t = r["timing"]
                out[r["pattern"]] = {"bound_ms": t["ms"][0], "narrow_ms": t["ms"][1],
                                     "ratio": t["median_round_ratio"],
                                     "per_chunk_saving_ms": LAYERS * (t["ms"][0] - t["ms"][1])}
    return out


def environment(root, phase, worker=None):
    env = dict(os.environ)
    for key in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE"):
        env.pop(key, None)
    env.update(CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS=str(root / "cubins"),
               CHOPINFER_DEEPSEEK41_ROUTER_NARROW_MEMCHECK="1" if phase == "memcheck" else "0",
               CHOPINFER_KERNEL_TIMEOUT_SECS="600")
    if worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE="1",
                   CHOPINFER_DEEPSEEK41_ROUTER_NARROW_WORKER=str(worker))
    return env


def test_args(root, name):
    return [str(root / BINARY), name, "--exact", "--include-ignored", "--nocapture"]


def command(root, plan, phase):
    cmd = [*test_args(root, TEST), "--test-threads=1"]
    if phase == "memcheck":
        cmd = [plan["sanitizer"], "--tool", "memcheck", "--error-exitcode", "86", "--require-cuda-init", "yes",
               "--destroy-on-device-error", "context", *cmd]
    return cmd


def run_phase(root, plan, phase, out):
    deadline = time.monotonic() + plan["budget"][phase]
    children = []
    try:
        for worker in (0, 1):
            log = out / f"{phase}-{worker}.log"
            handle = log.open("w")
            p = subprocess.Popen(command(root, plan, phase), env=environment(root, phase, worker),
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((p, handle, log))
        for p, _, log in children:
            if p.wait(timeout=max(1, deadline - time.monotonic())):
                raise ValueError(f"{phase} child failed: {log}")
    finally:
        for p, h, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            h.close()
    return [analyze(log.read_text(), worker, phase) for worker, (_, _, log) in enumerate(children)]


def dry_run(root, plan, out):
    wide.seal(root, plan)
    env = environment(root, "pair")

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
    if not Path(plan["sanitizer"]).is_file():
        raise ValueError("missing sanitizer")
    listing([*test_args(root, TEST), "--test-threads=1"], TEST, "narrow.list")
    (out / "dry-run.json").write_text(json.dumps({"cpu_tests": len(CPU_TESTS), "gpu_commands_listed": [TEST],
                                                  "phases": list(PHASES), "gpu_launched": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=PHASES)
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
    wide.seal(args.root, plan)
    if args.phase is None:
        raise ValueError("--phase is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 2:
        raise ValueError("a pair job needs two runner-assigned GPUs")
    started = int(time.time())
    verdict = {"phase": args.phase, "exit_code": 1, "power_before": wide.gpus(visible, True)}
    error = None
    try:
        parsed = run_phase(args.root, plan, args.phase, args.out)
        (args.out / "parsed.json").write_text(json.dumps(parsed) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summarize(parsed), indent=2) + "\n")
        verdict["exit_code"] = 0
    except Exception as e:  # recorded in the verdict, then re-raised
        error = f"{type(e).__name__}: {e}"
        verdict["error"] = error
    finally:
        try:
            verdict["power_after"] = wide.gpus(visible, False)
        except Exception as e:
            verdict["power_after_error"] = str(e)
            verdict["exit_code"] = 1
        kernel = subprocess.run(["journalctl", "-k", "--no-pager", "--since", "@" + str(started)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        (args.out / "kernel.log").write_text(kernel.stdout)
        verdict["host_xids"], verdict["xids"] = wide.own_xids(kernel.stdout.splitlines(), wide.bus_ids(visible))
        if kernel.returncode or verdict["xids"]:
            verdict["exit_code"] = 1
        verdict["elapsed_seconds"] = time.time() - started
        (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["exit_code"]:
        raise SystemExit(error or "power, journal or Xid check failed")
    print(f"router narrow {args.phase} passed on both workers")


if __name__ == "__main__":
    main()
