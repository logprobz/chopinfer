#!/usr/bin/env python3
"""Measure contributor fixtures or rebuild and compare a cohort on assigned GPUs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def run_bounded(command, *, cwd, env, deadline, capture=False):
    """Forward deadlines and external interrupts to this command's own group."""
    process = None
    pending_signal = None

    def terminate():
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()

    def interrupted(signum, _frame):
        nonlocal pending_signal
        if process is None:
            pending_signal = signum
            return
        terminate()
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None, text=True) as process:
            if pending_signal is not None:
                interrupted(pending_signal, None)
            try:
                stdout, stderr = process.communicate(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                terminate()
                raise
            if process.returncode:
                if capture:
                    print(stdout, end="")
                    print(stderr, end="", file=sys.stderr)
                raise subprocess.CalledProcessError(process.returncode, command)
            return stdout, stderr
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def compare(before, after):
    for receipt in (before, after):
        if (receipt.get("passed") is not True or receipt.get("mechanics_only")
                or receipt.get("errors") or receipt.get("skipped")):
            raise ValueError("comparison requires two passing full-workload receipts")
        if (receipt.get("schema") != "chopinfer-qualification/v1"
                or receipt.get("check") != "spec-bench-chopinfer-spec"):
            raise ValueError("comparison requires speculation benchmark receipts")
    fields = ("check", "arm", "output_tokens", "sampling", "concurrencies", "max_sequences",
              "speculation_mode", "workload_sha256", "gsm8k_examples_sha256", "prompts")
    for field in fields:
        if before.get(field) != after.get(field):
            raise ValueError(f"baseline workload differs: {field}")
    if before["server"]["model"] != after["server"]["model"]:
        raise ValueError("baseline model differs")
    def rows(receipt):
        result = {}
        for row in receipt["cohorts"]:
            key = (row["category"], row["concurrency"])
            value = row["aggregate_tokens_per_second"]
            if key in result or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid or duplicate cohort measurement")
            result[key] = value
        if not result:
            raise ValueError("receipt contains no measured cohorts")
        return result
    old, new = rows(before), rows(after)
    if old.keys() != new.keys():
        raise ValueError("baseline cohort set differs")
    return [f"{category} C{count}: {old[category, count]:.2f} -> {value:.2f} tok/s "
            f"({100 * (value / old[category, count] - 1):+.2f}%)"
            for (category, count), value in sorted(new.items())]


def contributor_summary(samples):
    grouped = {}
    for iteration in sorted({row["iteration"] for row in samples}):
        phases = {}
        rows = [row for row in samples if row["iteration"] == iteration]
        for row in rows:
            phases[row["phase"]] = phases.get(row["phase"], 0) + row["seconds"]
        for phase, seconds in phases.items():
            grouped.setdefault((rows[0]["starting_target"], phase), []).append(seconds)
    return [{"starting_target":condition, "phase":phase, "samples":len(values),
             "median_seconds":statistics.median(values), "minimum_seconds":min(values),
             "maximum_seconds":max(values)} for (condition, phase), values in sorted(grouped.items())]


def measure_contributor(args):
    sys.path.insert(0, str(ROOT / "ci"))
    import contributor
    output = contributor.external_path(str(args.output), "--output")
    if output.exists():
        raise ValueError("contributor timing output already exists; choose a new file")
    if not output.parent.is_dir():
        raise ValueError("create the external timing output directory first")
    commands = contributor.steps(args.contributor)
    if args.dry_run:
        for step in commands:
            print(step["phase"] + ": " + shlex.join(step["argv"]))
        return
    env = contributor.environment()
    setup = contributor.preflight(args.contributor, env=env)
    deadline = time.monotonic() + args.timeout
    report = {"schema":"chopinfer-contributor-timing/v1", "mode":args.contributor,
              "passed":False, "qualified":False, "setup":setup, "samples":[],
              "source_revision":contributor.probe(["git", "rev-parse", "HEAD"], "Git source identity is required"),
              "conditions":{key:env.get(key) for key in ("CARGO_BUILD_JOBS", "CARGO_INCREMENTAL", "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG", "RUST_TEST_THREADS", "RUSTC_WRAPPER", "CHOPINFER_NVCC_WRAPPER")},
              "limits":["Registry cache is required; OS caches are not controlled",
                        "Compilation and checks are distinct phases; repeated unchanged runs are freshness measurements",
                        "Materialize includes resume and tamper checks; package and verify fixtures include their setup",
                        "A contributor pass does not satisfy the full merge or model qualification gates"],
              "execution_phases":{phase:{"status":"not_applicable", "reason":"fixture exercise does not execute a model"} for phase in ("load", "warmup", "first_result")}}
    with output.open("x") as stream:
        try:
            for iteration in range(args.repetitions):
                target = Path(setup["target"])
                condition = "existing" if target.exists() and any(target.iterdir()) else "empty"
                for step in commands:
                    row = contributor.execute(step, env=env, runner=lambda command:
                        run_bounded(command, cwd=ROOT, env=env, deadline=deadline, capture=True))
                    report["samples"].append(dict(row, iteration=iteration, starting_target=condition))
            report["passed"] = True
        except BaseException as error:
            report["error"] = str(error)
            raise
        finally:
            report["summary"] = contributor_summary(report["samples"])
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Contributor timings saved; inspect starting-target conditions and executed counts: " + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mimo", "qwen38"), default="mimo")
    parser.add_argument("--cohort", choices=("spec-c1", "spec-c16", "spec-c32"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--save-baseline", action="store_true", help="Create a new baseline, never overwrite")
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--timeout", type=int, default=1800, help="Whole loop deadline in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without building or using GPUs")
    parser.add_argument("redeploy_args", nargs=argparse.REMAINDER,
                        help="After --, checkpoint/model-cache/work/log paths for redeploy")
    parser.add_argument("--contributor", choices=("sdk", "qwen38", "mimo26"), help="Measure bounded fixture phases without a server")
    parser.add_argument("--output", type=Path, help="New external contributor timing file")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.contributor:
        if not args.output or args.cohort or args.baseline or args.save_baseline or args.redeploy_args:
            parser.error("--contributor requires --output and cannot use GPU cohort/baseline/redeploy options")
        if args.repetitions < 1 or args.timeout <= 0:
            parser.error("repetitions and timeout must be positive")
        measure_contributor(args)
        return
    if not args.cohort or not args.baseline:
        parser.error("GPU development loops require --cohort and --baseline")
    if args.output or args.repetitions != 1:
        parser.error("--output and --repetitions require --contributor")
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error("port must be 1..65535 and timeout must be positive")
    if args.save_baseline and args.baseline.exists():
        parser.error("baseline already exists")
    before = None if args.save_baseline else json.loads(args.baseline.read_text())
    count = int(args.cohort.removeprefix("spec-c"))
    if before is not None:
        compare(before, before)
        if (before.get("concurrencies") != [count] or before.get("max_sequences") != count
                or before.get("output_tokens") != 64 or before.get("speculation_mode") != "adaptive"):
            parser.error("baseline does not match the requested development cohort")
    run_id = before["run_id"] if before else f"dev-{time.time_ns()}"
    extra = args.redeploy_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    # Limit overrides to input/output paths. Capacity, arm and port must match
    # the measured workload, and prepare-only must not benchmark an old server.
    allowed = {"--checkpoint", "--model-cache-dir", "--work-dir", "--logs-dir"}
    # Serving loads the checkpoint directly now, so rank images and installed
    # bundles no longer exist. Name the replacement for flags kept by habit.
    removed = {"--rank-dir": "--checkpoint", "--drafter-rank-dir": "--checkpoint",
               "--install-root": "--model-cache-dir or --work-dir"}
    for flag in extra[::2]:
        if flag in removed:
            parser.error(f"redeploy override {flag} was removed; serving reads the checkpoint "
                         f"directly, use {removed[flag]}")
    if len(extra) % 2 or any(extra[i] not in allowed for i in range(0, len(extra), 2)):
        parser.error("redeploy overrides must be pairs of checkpoint/model-cache/work/log paths")
    if "--logs-dir" not in extra[::2]:
        checkout = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
        extra = [*extra, "--logs-dir", str(args.baseline.resolve().parent / "servers" / checkout / args.model)]
    redeploy = ["bash", f"tools/dev/{args.model}-redeploy.sh", *extra,
                "--port", str(args.port), "--max-sequences", str(count), "--drafter",
                "dflash" if args.model == "mimo" else "mtp"]
    benchmark = ["python3", "-B", "tools/qual/spec_bench.py", "--arm", "chopinfer-spec",
                 "--speculation", "adaptive", "--run-id", run_id, "--label", "development",
                 "--url", f"http://127.0.0.1:{args.port}", "--max-sequences", str(count),
                 "--concurrencies", str(count)]
    env = dict(os.environ, CHOPINFER_DEV_SKIP_WARMUP="1", CARGO_BUILD_JOBS="8",
               CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0")
    env.setdefault("CHOPINFER_QUAL_CONFIG", str(ROOT / "models" / ("mimo26" if args.model == "mimo" else "qwen38")
                                              / "qual.json"))
    env["CHOPINFER_QUAL_RECEIPTS"] = str(args.baseline.resolve().parent / "runs")
    preflight = [sys.executable, "-B", "-c",
                 "import sys; sys.path.insert(0, 'tools/qual'); "
                 "from common import startup; from spec_bench import gsm8k_questions; "
                 "startup(model_facts=True); gsm8k_questions(int(sys.argv[1]))", str(count)]
    if args.dry_run:
        print("fixture preflight: " + shlex.join(preflight))
        print("CHOPINFER_DEV_SKIP_WARMUP=1 " + shlex.join(["nice", "-n", "19", *redeploy]))
        print("warm cohort: " + shlex.join(benchmark))
        print("measured cohort: " + shlex.join(benchmark))
        return
    deadline = time.monotonic() + args.timeout
    def run(command, capture=False):
        return run_bounded(command, cwd=ROOT, env=env, deadline=deadline, capture=capture)
    run(preflight)
    run(["nice", "-n", "19", *redeploy])
    # The first cohort warms lazy graph captures after skipping startup waves.
    # Its receipt remains available but is excluded from the comparison.
    run(benchmark)
    stdout, stderr = run(benchmark, capture=True)
    print(stdout, end="")
    print(stderr, end="", file=sys.stderr)
    receipt_path = Path(json.loads(stdout.strip().splitlines()[-1])["receipt"])
    after = json.loads(receipt_path.read_text())
    compare(after, after)
    if args.save_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with args.baseline.open("x") as output:
            json.dump(after, output, indent=2)
            output.write("\n")
        print(f"Saved baseline: {args.baseline}")
    else:
        print("\n".join(compare(before, after)))
    print("Development timings only. Startup warmup was skipped; this is not qualification evidence.")


if __name__ == "__main__":
    main()
