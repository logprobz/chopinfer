#!/usr/bin/env python3
"""Run and audit the router-in-norm qualification on a GPU pair.

Phases (router_norm.md): ``memcheck`` (criterion 5 under compute-sanitizer),
``bitwise`` (criteria 1 to 4 and 6) and ``timing`` (criterion 7, which
counts only after the other phases pass). Each phase runs the frozen test
binary once per worker with one logical GPU each and fails closed on any
missing, duplicate or failed case, zero executed tests, a power cap other
than 250 W, or an Xid in the kernel journal. The record, listing, power and
timing helpers are native_moe_job's.

``--emulate`` runs a phase exactly as the job does, but on CPU: the frozen
binary's host emulation (CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE=1)
replaces the GPU, the sanitizer is left out, and a stand-in for nvidia-smi
reads busy before it reads idle. Its records are marked emulated, and the
analyzers accept emulated records only in that mode. It checks the job's
control flow before a queue slot is spent; it is not GPU evidence.
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

PATTERNS = ("clean", "outliers")
PRIORS = ("clear", "flagged")
LIVES = (1, 65, 513, 2048, 8191, 8192)
MEMCHECK_LIVES = (1, 65, 2048)
TIMED_ROWS = (8192, 2048, 512)
TESTS = {"bitwise": "router_norm_bitwise_gpu", "timing": "router_norm_timing_gpu"}
CPU_TESTS = ("router_norm_cpu_plan", "emulation_catches_a_faulty_certified_arm")
PHASE_TESTS = {"memcheck": "bitwise", "bitwise": "bitwise", "timing": "timing"}
COMPARED = ("normed", "widened", "codes", "scales", "status", "logits")
BINARY = "bin/router-norm-test"
EMULATED_BUDGET = 1200
# Round 7 profile, chunk 0 of a cold 32K prompt: 20 full layers; the
# upper-skip layers' router steps are counted separately.
FULL_LAYERS = 20


def cases(memcheck):
    """(pattern, prior, live) of the phase, as the test's `cases`."""
    priors = ("clear",) if memcheck else PRIORS
    lives = MEMCHECK_LIVES if memcheck else LIVES
    return [(p, s, l) for p in PATTERNS for s in priors for l in lives]


def check_case(r, label):
    if r["failures"] or r["passed"] is not True:
        raise ValueError(f"{label}: failed checks {r['failures']}")
    if sorted(r["differing_bytes"]) != sorted(COMPARED) or any(r["differing_bytes"].values()):
        raise ValueError(f"{label}: outputs differ")
    if r["repeat_bitwise"] is not True or r["graph_bitwise"] is not True:
        raise ValueError(f"{label}: repeat or replay differs")
    if r["certified_rows"] + r["uncertified_rows"] != r["live"]:
        raise ValueError(f"{label}: flags do not cover the live rows")
    if r["mutant_differs"] is not (r["uncertified_rows"] > 0):
        raise ValueError(f"{label}: negative control escaped or differs without an uncertified row")


def analyze_bitwise(text, phase, emulated=False):
    memcheck = phase == "memcheck"
    base.libtest(text, 1)
    rows = base.records(text, "router_norm_record")
    base.unique(rows, lambda r: (r["pattern"], r["prior"], r["live"]), cases(memcheck), "bitwise")
    for r in rows:
        label = f"{r['pattern']} {r['prior']} live {r['live']}"
        if r.get("emulated") is not emulated:
            raise ValueError(f"{label}: {'not ' if emulated else ''}an emulated record")
        check_case(r, label)
    # The outlier pattern reaches the uncertified path and the control.
    if not any(r["uncertified_rows"] > 0 and r["mutant_differs"] for r in rows):
        raise ValueError("no case has an uncertified live row")
    done = base.records(text, "router_norm_cases")
    if done != [{"executed": len(rows), "failed": 0, "memcheck": memcheck}]:
        raise ValueError("missing case completion")
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors?", text)
    if emulated:
        if summaries:
            raise ValueError("an emulated phase ran under the sanitizer")
    elif memcheck and summaries != ["0"]:
        raise ValueError("missing clean memcheck summary")
    return rows


def analyze_timing(text, emulated=False):
    base.libtest(text, 1)
    rows = base.records(text, "router_norm_timing")
    done = base.records(text, "router_norm_timed")
    if emulated:
        if rows or done != [{"executed": 0, "emulated": True}]:
            raise ValueError("emulated timing must report no samples")
        return []
    base.unique(rows, lambda r: r["rows"], TIMED_ROWS, "timing")
    for r in rows:
        base.validate_timing(r["timing"])
    if done != [{"executed": len(TIMED_ROWS)}]:
        raise ValueError("missing timing completion")
    return rows


def summarize(parsed):
    """Report-only timing aggregates and the per-chunk projection."""
    out = {"cases": sum(len(p.get("bitwise", [])) for p in parsed), "timing": {}}
    timed = [r for p in parsed for r in p.get("timing", [])]
    if not timed:
        return out
    for rows in TIMED_ROWS:
        t = [r["timing"] for r in timed if r["rows"] == rows]
        out["timing"][str(rows)] = {
            "served_ms_mean": sum(x["ms"][0] for x in t) / len(t),
            "certified_ms_mean": sum(x["ms"][1] for x in t) / len(t),
            "ratio_median": sorted(x["median_round_ratio"] for x in t)[len(t) // 2]}
    full = out["timing"]["8192"]
    out["timing"]["per_chunk_full_layers_ms"] = FULL_LAYERS * (full["served_ms_mean"] - full["certified_ms_mean"])
    return out


def emulated_gpus(busy=2):
    """A stand-in for `base.gpus` in emulated mode: the pair reads busy for
    `busy` calls, then idle at 250 W."""
    calls = []

    def gpus(visible, idle):
        calls.append(idle)
        if idle and len(calls) <= busy:
            raise ValueError("assigned GPU pair is not idle (emulated)")
        return [["0", "250.00", "0", "0"], ["1", "250.00", "0", "0"]]
    return gpus


def idle_pair(visible, wait=60.0, poll=2.0, gpus=None):
    """The pair's power and idle check, retried while the pair still reads
    busy after the previous phase on the same GPUs."""
    gpus = gpus or base.gpus
    deadline = time.monotonic() + wait
    while True:
        try:
            return gpus(visible, True)
        except ValueError as e:
            if "not idle" not in str(e) or time.monotonic() >= deadline:
                raise
            time.sleep(poll)


def seal(root, plan):
    expected = set(plan["files"]) | {"PLAN.json", "SHA256SUMS"} | set(plan["jobs"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.is_relative_to(root / "runs")}
    if actual != expected:
        raise ValueError(f"unsealed snapshot files: {sorted(actual ^ expected)}")
    for name, item in plan["files"].items():
        path = root / name
        st = path.lstat()
        if path.is_symlink() or st.st_mode & 0o022 or st.st_size != item["bytes"] or base.sha(path) != item["sha256"]:
            raise ValueError(f"changed or writable snapshot file: {name}")


def environment(root, phase, worker=None, emulated=False):
    env = dict(os.environ)
    for key in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE"):
        env.pop(key, None)
    env.update(CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS=str(root / "cubins"),
               CHOPINFER_DEEPSEEK41_ROUTER_NORM_MEMCHECK="1" if phase == "memcheck" else "0",
               CHOPINFER_KERNEL_TIMEOUT_SECS="600")
    if emulated:
        env.update(CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE="1", CUDA_VISIBLE_DEVICES="")
    elif worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE="1")
    return env


def test_args(root, name):
    return [str(root / BINARY), name, "--exact", "--include-ignored", "--nocapture"]


def command(root, plan, phase, emulated=False):
    cmd = [*test_args(root, TESTS[PHASE_TESTS[phase]]), "--test-threads=1"]
    if phase == "memcheck" and not emulated:
        cmd = [plan["sanitizer"], "--tool", "memcheck", "--error-exitcode", "86", "--require-cuda-init", "yes",
               "--destroy-on-device-error", "context", *cmd]
    return cmd


def run_phase(root, plan, phase, out, emulated=False):
    deadline = time.monotonic() + (EMULATED_BUDGET if emulated else plan["budget"][phase])
    parsed = [{}, {}]
    children = []
    try:
        for worker in (0, 1):
            log = out / f"{phase}-{worker}.log"
            handle = log.open("w")
            p = subprocess.Popen(command(root, plan, phase, emulated), env=environment(root, phase, worker, emulated),
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
    for worker, (_, _, log) in enumerate(children):
        text = log.read_text()
        if phase == "timing":
            parsed[worker]["timing"] = analyze_timing(text, emulated)
        else:
            parsed[worker]["bitwise"] = analyze_bitwise(text, phase, emulated)
    return parsed


def dry_run(root, plan, out):
    seal(root, plan)
    env = environment(root, "bitwise")

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
    listed = []
    for phase in PHASE_TESTS:
        cmd = command(root, plan, phase)
        if not Path(cmd[0]).is_file():
            raise ValueError(f"missing executable {cmd[0]}")
        argv = cmd[cmd.index(str(root / BINARY)):]
        listing(argv, TESTS[PHASE_TESTS[phase]], f"{phase}.list")
        listed.append({"phase": phase, "test": TESTS[PHASE_TESTS[phase]], "listed": 1})
    (out / "dry-run.json").write_text(json.dumps({"cpu_tests": len(CPU_TESTS), "gpu_commands_listed": listed,
                                                  "gpu_launched": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=sorted(PHASE_TESTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emulate", action="store_true", help="run --phase on CPU (module docstring)")
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
    seal(args.root, plan)
    if args.phase is None:
        raise ValueError("--phase is required")
    if args.emulate:
        visible, gpus = "emulated", emulated_gpus()
    else:
        visible, gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ""), base.gpus
        if len(visible.split(",")) != 2:
            raise ValueError("a pair job needs two runner-assigned GPUs")
    started = int(time.time())
    verdict = {"phase": args.phase, "exit_code": 1, "emulated": args.emulate,
               "power_before": idle_pair(visible, wait=60 if not args.emulate else 5,
                                         poll=2 if not args.emulate else 0.1, gpus=gpus)}
    error = None
    try:
        parsed = run_phase(args.root, plan, args.phase, args.out, args.emulate)
        (args.out / "parsed.json").write_text(json.dumps(parsed) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summarize(parsed), indent=2) + "\n")
        verdict["exit_code"] = 0
    except Exception as e:  # recorded in the verdict, then re-raised
        error = f"{type(e).__name__}: {e}"
        verdict["error"] = error
    finally:
        try:
            verdict["power_after"] = gpus(visible, False)
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
    print(f"router norm {args.phase} passed on both workers{' (emulated)' if args.emulate else ''}")


if __name__ == "__main__":
    main()
