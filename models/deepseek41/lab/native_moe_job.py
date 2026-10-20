#!/usr/bin/env python3
"""Run and audit the native MoE prefill qualification on a GPU pair.

Phases (native_moe_prefill.md): ``memcheck`` (criterion 6 and the probe
under compute-sanitizer), ``oracle`` (criteria 1 to 5 and the report-only
comparison, synthetic and captured) and ``timing`` (paired timing, which
counts only after the other phases pass). Each phase runs the frozen test
binary once per worker with one logical GPU each and fails closed on any
missing, duplicate or failed case, zero executed tests, a power cap other
than 250 W, or an Xid in the kernel journal.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

CASES = ("e384-balanced", "e384-skewed", "e128-skewed")
MEMCHECK_LIVES = (1, 65, 513, 2048)
LAYERS = (0, 13, 14, 39)
CHUNKS = (("input-0", 0, 8192), ("input-1", 8192, 65), ("input-2", 8192, 257))
VARIANTS = 3  # two other gate/up tiles and one other down tile
# Full libtest paths: the commands pass --exact, which matches whole names.
TESTS = {
    "synthetic": "synthetic::native_moe_synthetic_gpu",
    "captured": "captured::native_moe_captured_gpu",
}
CPU_TESTS = ("synthetic::cpu_plan", "captured::native_moe_captured_cpu_admission",
             "probe::probe_cases_isolate_one_product_and_cover_the_domain",
             "oracle::tests::bound_and_cosine_reject_a_dropped_chain")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def records(text, marker):
    # libtest can prefix the first line of output with the test's name.
    return [json.loads(line.split(marker + " ", 1)[1]) for line in text.splitlines() if marker + " " in line]


def libtest(text, count):
    results = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    if results != [("ok", str(count), "0", "0")]:
        raise ValueError(f"expected {count} passing test(s), got {results}")


def check_listed(text, name):
    """The sealed binary's --list output under a command's filter must name
    exactly that test, so a filter that selects nothing fails before a GPU run."""
    names = re.findall(r"^(\S+): test$", text, re.M)
    counts = re.findall(r"^(\d+) tests?, (\d+) benchmarks?$", text, re.M)
    if names != [name] or counts != [("1", "0")]:
        raise ValueError(f"filter {name!r} lists {names} ({counts})")


def unique(rows, key, expected, label):
    keys = [key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError(f"{label}: missing, extra or duplicate cases")


def validate_timing(t):
    rounds = t["round_samples_ms"]
    if len(rounds) != 5 or t["replays_per_sample"] != 1 or len(t["ms"]) != 2:
        raise ValueError("missing interleaved timing rounds")
    medians, ratios = [[], []], []
    for pair in rounds:
        if len(pair) != 2 or any(len(s) != 3 or any(not math.isfinite(v) or v <= 0 for v in s) for s in pair):
            raise ValueError("invalid timing samples")
        m = [sorted(s)[1] for s in pair]
        ratios.append(m[1] / m[0])
        for arm in range(2):
            medians[arm].append(m[arm])
    for arm in range(2):
        if not math.isclose(t["ms"][arm], sorted(medians[arm])[2], rel_tol=1e-6):
            raise ValueError("timing median does not match its samples")
    if not math.isclose(t["median_round_ratio"], sorted(ratios)[2], rel_tol=1e-6):
        raise ValueError("timing ratio does not match its rounds")
    return t["ms"], t["median_round_ratio"]


def check_stage(stage, label):
    if stage["rows"] <= 0 or stage["elements"] <= 0 or stage["failures"] or stage["cosine_failures"] or not stage["min_cosine"] >= 0.9999:
        raise ValueError(f"{label}: oracle bound or cosine failure")


def check_qualified(row, label, oracle):
    if row["failures"] or not row["passed"] or not row["serving_gate_matches_raw"]:
        raise ValueError(f"{label}: failed checks {row['failures']}")
    if len(row["variants"]) != VARIANTS or not all(v["bitwise"] for v in row["variants"]):
        raise ValueError(f"{label}: tile variants differ")
    if oracle:
        o = row["oracle"]
        for name in ("gate_up", "down", "chain_down", "exact_down"):
            check_stage(o[name], f"{label} {name}")
        if not o["omit_first_rejected"] or o["omit_first"]["failures"] <= 0:
            raise ValueError(f"{label}: the omit-first control escaped")
        if o["sampled_tokens"] != min(32, row["live"]):
            raise ValueError(f"{label}: wrong sampled rows")
    elif "oracle" in row:
        raise ValueError(f"{label}: unexpected oracle in this phase")


def analyze_synthetic(text, memcheck, oracle, timing, probe):
    libtest(text, 1)
    probes = records(text, "native_moe_probe")
    if len(probes) != int(probe):
        raise ValueError("missing or unexpected instruction probe")
    for p in probes:
        if p["failures"] or p["executed_outputs"] != 4 * 32 * 128 + 8 * 128:
            raise ValueError("instruction probe failed or incomplete")
    rows = records(text, "native_moe_synthetic")
    lives = MEMCHECK_LIVES if memcheck else (8192,)
    unique(rows, lambda r: (r["case"], r["live"]), [(c, l) for c in CASES for l in lives], "synthetic")
    for r in rows:
        label = f"{r['case']} live {r['live']}"
        if r["memcheck"] != memcheck:
            raise ValueError(f"{label}: wrong mode")
        check_qualified(r, label, oracle)
        if oracle:
            inv = r["invariance"]
            if [p["live"] for p in inv["prefixes"]] != [4097, 65, 1] or not all(p["bitwise_prefix"] for p in inv["prefixes"]) \
                    or not inv["graph_bitwise"] or not inv["repeat_bitwise"]:
                raise ValueError(f"{label}: invariance failure")
        if timing:
            for part in ("gate_up", "down", "both"):
                validate_timing(r["timing"][part])
        elif "timing" in r:
            raise ValueError(f"{label}: unexpected timing")
    done = records(text, "native_moe_synthetic_done")
    if done != [{"cases": len(CASES) * len(lives), "failed": 0, "memcheck": memcheck}]:
        raise ValueError("missing synthetic completion")
    if memcheck and len(re.findall(r"ERROR SUMMARY: 0 errors", text)) != 1:
        raise ValueError("missing clean memcheck summary")
    return dict(probe=probes, cases=rows)


def analyze_captured(text, worker, oracle, timing):
    libtest(text, 1)
    rows = records(text, "native_moe_captured")
    expected = [(l, r, p, pos, n) for l in LAYERS for r in (worker, worker + 2) for (p, pos, n) in CHUNKS]
    unique(rows, lambda r: (r["layer"], r["rank"], r["prompt"], r["position"], r["rows"]), expected, "captured")
    for r in rows:
        label = f"layer {r['layer']} rank {r['rank']} {r['prompt']}@{r['position']}"
        if r["worker"] != worker or r["authentication_failures"] or not r["ordered_sum_model_reproduces_capture"]:
            raise ValueError(f"{label}: capture authentication failed")
        check_qualified(r, label, oracle)
        if timing and r["rows"] == 8192:
            for part in ("gate_up", "down", "both"):
                validate_timing(r["timing"][part])
            tiles = r["timing"]["tiles"]
            if len(tiles) != VARIANTS + 2 or sum(t["serving_tile"] for t in tiles) != 2:
                raise ValueError(f"{label}: incomplete tile timing")
            for t in tiles:
                validate_timing(t["timing"])
        elif "timing" in r:
            raise ValueError(f"{label}: unexpected timing")
    done = records(text, "native_moe_captured_done")
    if done != [{"worker": worker, "cases": len(expected), "failed": 0}]:
        raise ValueError("missing captured completion")
    return rows


def summarize(parsed):
    """Report-only aggregates; none of these is a gate."""
    out = {"stages": {}, "no_larger_error": {}, "timing": {}}
    rows = [r for p in parsed for r in p.get("captured", [])] + [r for p in parsed for r in p.get("synthetic", {}).get("cases", [])]
    for name in ("gate_up", "down", "chain_down"):
        stages = [r["oracle"][name] for r in rows if "oracle" in r]
        if stages:
            out["stages"][name] = {"rows": sum(s["rows"] for s in stages),
                                   "worst_ratio": max(s["worst_ratio"] for s in stages),
                                   "min_cosine": min(s["min_cosine"] for s in stages)}
    for name in ("down_vs_exact", "ordered_sum_vs_exact"):
        v = [r["oracle"]["report_only"][name] for r in rows if "oracle" in r]
        if v:
            out["no_larger_error"][name] = {"cases": len(v), "no_larger": sum(x["no_larger_error"] for x in v),
                                            "rmse_ratio_max": max(x["rmse_ratio"] for x in v),
                                            "rmse_ratio_median": statistics.median(x["rmse_ratio"] for x in v)}
    captured = [r for p in parsed for r in p.get("captured", []) if "timing" in r]
    if captured:
        for part in ("gate_up", "down", "both"):
            ms = [r["timing"][part]["ms"] for r in captured]
            out["timing"][part] = {"cases": len(ms), "reference_ms_mean": statistics.mean(m[0] for m in ms),
                                   "native_ms_mean": statistics.mean(m[1] for m in ms),
                                   "ratio_median": statistics.median(r["timing"][part]["median_round_ratio"] for r in captured)}
        both = out["timing"]["both"]
        out["timing"]["per_chunk_40_layers_ms"] = {"reference": 40 * both["reference_ms_mean"], "native": 40 * both["native_ms_mean"]}
    return out


def gpus(visible, idle):
    output = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,power.limit,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 2 or any(abs(float(r[1]) - 250) > 0.5 for r in rows):
        raise ValueError("both assigned GPUs must have a 250 W cap")
    if idle and any(float(r[2]) > 0 or float(r[3]) > 1000 for r in rows):
        raise ValueError("assigned GPU pair is not idle")
    return [[x.strip() for x in r] for r in rows]


def seal(root, plan):
    expected = set(plan["files"]) | {"PLAN.json", "SHA256SUMS"} | set(plan["jobs"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.is_relative_to(root / "runs")}
    if actual != expected:
        raise ValueError(f"unsealed snapshot files: {sorted(actual ^ expected)}")
    for name, item in plan["files"].items():
        path = root / name
        st = path.lstat()
        if path.is_symlink() or st.st_mode & 0o022 or st.st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"changed or writable snapshot file: {name}")
    if sha(plan["ready"]["path"]) != plan["ready"]["sha256"]:
        raise ValueError("capture ready marker changed")
    if sha(Path(plan["banks"]["path"]) / "BANKS.json") != plan["banks"]["sha256"]:
        raise ValueError("bank manifest changed")


def environment(root, plan, phase, worker=None):
    env = dict(os.environ)
    for key in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS"):
        env.pop(key, None)
    env.update(CHOPINFER_DEEPSEEK41_NATIVE_MOE_CUBINS=str(root / "cubins"),
               CHOPINFER_NATIVE_MOE_READY=plan["ready"]["path"],
               CHOPINFER_NATIVE_MOE_BANKS=plan["banks"]["path"],
               CHOPINFER_NATIVE_MOE_MEMCHECK="1" if phase == "memcheck" else "0",
               CHOPINFER_NATIVE_MOE_TIMING="1" if phase == "timing" else "0",
               CHOPINFER_NATIVE_MOE_ORACLE="1" if phase == "oracle" else "0",
               CHOPINFER_NATIVE_MOE_PROBE="1" if phase in ("memcheck", "oracle") else "0",
               CHOPINFER_KERNEL_TIMEOUT_SECS="900")
    if worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE="1", CHOPINFER_NATIVE_MOE_WORKER=str(worker))
    return env


PHASE_TESTS = {"memcheck": ("synthetic",), "oracle": ("synthetic", "captured"), "timing": ("synthetic", "captured")}


def test_args(root, name):
    return [str(root / "bin/native-moe-test"), name, "--exact", "--include-ignored", "--nocapture"]


def command(root, plan, phase, test):
    cmd = [*test_args(root, TESTS[test]), "--test-threads=1"]
    if phase == "memcheck":
        cmd = [plan["sanitizer"], "--tool", "memcheck", "--error-exitcode", "86", "--require-cuda-init", "yes",
               "--destroy-on-device-error", "context", *cmd]
    return cmd


def run_phase(root, plan, phase, out):
    deadline = time.monotonic() + plan["budget"][phase]
    parsed = [{}, {}]
    for test in PHASE_TESTS[phase]:
        children = []
        try:
            for worker in (0, 1):
                log = out / f"{phase}-{test}-{worker}.log"
                handle = log.open("w")
                p = subprocess.Popen(command(root, plan, phase, test), env=environment(root, plan, phase, worker),
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((p, handle, log))
            for p, _, log in children:
                if p.wait(timeout=max(1, deadline - time.monotonic())):
                    raise ValueError(f"{phase} {test} child failed: {log}")
        finally:
            for p, h, _ in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                h.close()
        for worker, (_, _, log) in enumerate(children):
            text = log.read_text()
            if test == "synthetic":
                parsed[worker]["synthetic"] = analyze_synthetic(text, phase == "memcheck", phase == "oracle", phase == "timing", phase != "timing")
            else:
                parsed[worker]["captured"] = analyze_captured(text, worker, phase == "oracle", phase == "timing")
    return parsed


def dry_run(root, plan, out):
    seal(root, plan)
    env = environment(root, plan, "oracle")
    def listing(argv, name, log):
        r = subprocess.run([*argv, "--list"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        (out / log).write_text(r.stdout)
        if r.returncode:
            raise ValueError(f"dry-run listing failed: {name}")
        check_listed(r.stdout, name)

    for i, t in enumerate(CPU_TESTS):
        listing(test_args(root, t), t, f"cpu-{i}.list")
        r = subprocess.run(test_args(root, t), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        (out / f"cpu-{i}.log").write_text(r.stdout)
        if r.returncode:
            raise ValueError(f"dry-run test failed: {t}")
        libtest(r.stdout, 1)
    listed = []
    for phase in PHASE_TESTS:
        for test in PHASE_TESTS[phase]:
            cmd = command(root, plan, phase, test)
            if not Path(cmd[0]).is_file():
                raise ValueError(f"missing executable {cmd[0]}")
            # The job's own argv from the test binary on, with its filter and flags.
            argv = cmd[cmd.index(str(root / "bin/native-moe-test")):]
            listing(argv, TESTS[test], f"{phase}-{test}.list")
            listed.append({"phase": phase, "test": TESTS[test], "listed": 1})
    (out / "dry-run.json").write_text(json.dumps({"cpu_tests": len(CPU_TESTS), "gpu_commands_listed": listed,
                                                  "gpu_launched": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=sorted(PHASE_TESTS))
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
    seal(args.root, plan)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 2:
        raise ValueError("a pair job needs two runner-assigned GPUs")
    started = int(time.time())
    verdict = {"phase": args.phase, "exit_code": 1, "power_before": gpus(visible, True)}
    error = None
    try:
        parsed = run_phase(args.root, plan, args.phase, args.out)
        summary = summarize(parsed)
        (args.out / "parsed.json").write_text(json.dumps(parsed) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    print(f"native MoE {args.phase} passed on both workers")


if __name__ == "__main__":
    main()
