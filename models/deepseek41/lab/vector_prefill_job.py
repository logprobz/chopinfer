#!/usr/bin/env python3
"""Audit vector staging against current serving and frozen scalar controls."""
import argparse
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

PREFIX = "sm120::vector::"
ENV = "CHOPINFER_DEEPSEEK41_VECTOR_PREFILL_CUBINS"


def records(text, marker):
    return [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]


def passed(text, expected=1):
    results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    if not results or sum(int(r[0]) for r in results) != expected or any(int(f) or int(i) for _, f, i in results):
        raise ValueError("missing tests, failed tests, or unintended ignores")
    if re.search(r"skipp", text, re.I) or "FAILED" in text:
        raise ValueError("skipped or failed GPU work")


VARIANTS = ("pipe64", "wide128", "decoded128")


def timing_ratio(timing):
    rounds = timing["round_samples_ms"]
    if len(rounds) != 5 or timing["replays_per_sample"] != 1:
        raise ValueError("missing interleaved timing rounds")
    medians = [[], []]
    ratios = []
    for pair in rounds:
        if len(pair) != 2 or any(len(samples) != 3 or any(not math.isfinite(t) or t <= 0 for t in samples) for samples in pair):
            raise ValueError("invalid timing samples")
        values = [statistics.median(samples) for samples in pair]
        ratios.append(values[1] / values[0])
        for arm in range(2):
            medians[arm].append(values[arm])
    for arm in range(2):
        if not math.isclose(timing["ms"][arm], statistics.median(medians[arm]), rel_tol=1e-7):
            raise ValueError("incorrect timing median")
    return statistics.median(ratios)


def summarize(logs):
    if len(logs) != 2:
        raise ValueError("requires both devices")
    expected = {(e, r, route, v) for e in (128, 384) for r in (512, 2048, 4096, 8192)
                for route in ("balanced", "skewed") for v in VARIANTS}
    result = []
    for gpu, text in enumerate(logs):
        passed(text)
        rows = records(text, "vector_prefill_result ")
        keys = [(r["experts"], r["input_rows"], r["routing"], r["variant"]) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("incomplete or duplicate matrix")
        for row in rows:
            if row["serving_enabled"] or not row["bitwise"] or not row["scalar_bitwise"] or not row["clock_bitwise"] or row["boundary_replays"] != 4:
                raise ValueError("missing vector qualification controls")
            if row["variant"] == "pipe64" and not row["negative_control_caught"]:
                raise ValueError("missing negative control")
            for part in ("gateup", "down"):
                grouped = row["input_rows"] in (512, 2048, 4096) and not (
                    row["experts"] == 128 and row["input_rows"] == 2048 and part == "down")
                symbols = row["baseline"][part]
                pipeline = row["experts"] == 384 and row["input_rows"] == 8192 and part == "gateup"
                if pipeline:
                    prefix = "chop_exact_prefill_e384_before_gateup_decoded128_"
                    expected_symbols = [prefix + stage + "_v1" for stage in
                                        ("pack", "predecode", "r8", "r32", "r64", "r128")]
                    if symbols != expected_symbols:
                        raise ValueError("baseline omits the promoted gate-only route")
                else:
                    if len(symbols) != (4 if grouped else 1) or any("chop_exact" in symbol for symbol in symbols):
                        raise ValueError("incorrect serving baseline")
                    if any(symbol.endswith("_pack_v1") for symbol in symbols) != grouped:
                        raise ValueError("baseline bypasses measured grouped winner")
                queues = row["clock"]["parts"][part]
                if len(queues) != 4 or not any(q["items"] > 0 for q in queues):
                    raise ValueError("missing clock evidence")
                for q in queues:
                    if len(q["cta_phase_cycles"]) != 7 or any(c < 0 for c in q["cta_phase_cycles"]):
                        raise ValueError("invalid clock diagnostics")
            for arm in ("timing", "scalar_timing"):
                for part in ("gateup", "down", "projections", "full"):
                    measurement = row[arm][part]
                    measurement["paired_ratio"] = timing_ratio(measurement)
                    measurement["speedup"] = 1 / measurement["paired_ratio"]
            row["gpu"] = gpu
            result.append(row)
    wins = []
    # A losing down candidate must not veto an independently winning gate.
    for experts in (128, 384):
        for count in (512, 2048, 4096, 8192):
            for variant in VARIANTS:
                selected = [r for r in result if
                            (r["experts"], r["input_rows"], r["variant"]) == (experts, count, variant)]
                if len(selected) != 4:
                    raise ValueError("missing device/routing cross-product")
                for part in ("gateup", "down"):
                    ratios = [r["timing"][part]["paired_ratio"] for r in selected]
                    if max(ratios) < 0.97:
                        wins.append({"experts": experts, "input_rows": count, "variant": variant,
                                     "part": part, "worst_speedup": 1 / max(ratios)})
    return {"records": result, "cases_per_gpu": len(expected), "bitwise": True,
            "kernel_wins": wins, "serving_enabled": False,
            "scope": "vector experiment only; current serving includes decoded128 gate/up at E384/8192"}


def dry(snapshot, out):
    for path in (snapshot, *snapshot.rglob("*")):
        if path.stat().st_mode & 0o022:
            raise ValueError(f"unsealed snapshot path {path}")
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"snapshot digest mismatch: {name}")
    env = os.environ.copy()
    for key in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_KERNEL_REQUIRE"):
        env.pop(key, None)
    binary = snapshot / "vector-prefill-tests"
    command = [str(binary), "--exact", PREFIX + "vector_cpu_matrix", "--nocapture"]
    proc = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    text = proc.stdout + proc.stderr
    (out / "cpu.log").write_text(text)
    passed(text)
    metadata, = records(text, "vector_prefill_dry_run ")
    if metadata["required_ordinals"] != 1 or metadata["cases"] != 16 or tuple(metadata["variants"]) != VARIANTS or not metadata["embedded_modules"]:
        raise ValueError("incorrect worker GPU or matrix requirements")
    found = set()
    for module in metadata["modules"]:
        path = snapshot / "cubins" / (module + ".cubin")
        symbols = subprocess.check_output(["readelf", "--wide", "--symbols", str(path)], text=True)
        found.update(line.split()[-1] for line in symbols.splitlines() if " FUNC " in line and " GLOBAL " in line and " UND " not in line)
    if not set(metadata["symbols"]).issubset(found):
        raise ValueError(f"missing staged symbols: {set(metadata['symbols']) - found}")
    listed = subprocess.check_output([str(binary), "--list"], text=True, env=env)
    if PREFIX + "vector_gpu: test" not in listed:
        raise ValueError("missing GPU test")
    resources = json.loads((snapshot / "resources.json").read_text())
    if set(resources) != set(metadata["symbols"]) or any(r["local_bytes"] or r["stack_bytes"] for r in resources.values()):
        raise ValueError("missing entry resource audit or spilled candidate")
    metadata["resources"] = resources
    metadata["external_modules_present"] = True
    metadata["workers"] = [{"ordinal": i, "required_ordinals": 1} for i in range(2)]
    # This harness uses driver events around complete graphs, not the
    # kernel-harness eviction/probe timing path. Baselines are embedded.
    metadata["timing_modules"] = []
    (out / "inventory.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def power(visible, path):
    text = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"], text=True)
    path.write_text(text)
    rows = [line.split(",") for line in text.splitlines()]
    if len(rows) != 2 or any(abs(float(row[1]) - 250) > 0.01 for row in rows):
        raise ValueError("pair requires 250 W on both devices")


def run(snapshot, out):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 2:
        raise ValueError("runner must allocate a pair")
    start = int(time.time())
    children = []
    def stop(signum, _frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise RuntimeError(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        power(visible, out / "power-before.csv")
        processes = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        (out / "processes-before.txt").write_text(processes)
        if processes.strip():
            raise ValueError("allocated GPUs are not idle")
        for ordinal in range(2):
            env = os.environ.copy()
            env.pop("CHOPINFER_KERNEL_GPUS", None)
            env.pop("CHOPINFER_KERNEL_SHARED_DEVICE", None)
            env.update({"CHOPINFER_KERNEL_GPU": str(ordinal), "CHOPINFER_KERNEL_REQUIRE": "1",
                        "CHOPINFER_KERNEL_TIMEOUT_SECS": "780", ENV: str(snapshot / "cubins"),
                        "CHOPINFER_KERNEL_RECEIPTS": str(out / f"receipts-{ordinal}")})
            with (out / f"gpu{ordinal}.log").open("w") as log:
                children.append(subprocess.Popen([str(snapshot / "vector-prefill-tests"), "--exact", PREFIX + "vector_gpu", "--include-ignored", "--nocapture", "--test-threads=1"], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        deadline = time.monotonic() + 800
        codes = [child.wait(timeout=max(1, deadline - time.monotonic())) for child in children]
        if codes != [0, 0]:
            raise ValueError(f"GPU test exits: {codes}")
        summary = summarize([(out / f"gpu{i}.log").read_text() for i in range(2)])
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines = ["| GPU | E | Rows | Routing | Variant | Gate/up speedup | Down speedup | Combined speedup | Full speedup |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for row in summary["records"]:
            times = [f"{row['timing'][part]['speedup']:.3f}x" for part in ("gateup", "down", "projections", "full")]
            lines.append(f"| {row['gpu']} | {row['experts']} | {row['input_rows']} | {row['routing']} | {row['variant']} | {' | '.join(times)} |")
        (out / "timing.md").write_text("\n".join(lines) + "\n")
        scalar_lines = ["| GPU | E | Rows | Routing | Variant | Gate/up scalar/vector speedup | Down scalar/vector speedup | Pair speedup | Full speedup |",
                        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for row in summary["records"]:
            times = [f"{row['scalar_timing'][part]['speedup']:.3f}x" for part in ("gateup", "down", "projections", "full")]
            scalar_lines.append(f"| {row['gpu']} | {row['experts']} | {row['input_rows']} | {row['routing']} | {row['variant']} | {' | '.join(times)} |")
        (out / "scalar-control-timing.md").write_text("\n".join(scalar_lines) + "\n")
        print("Pair vector matrices complete; per-kernel wins recorded; vector serving remains disabled")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        power(visible, out / "power-after.csv")
        journal = subprocess.run(["journalctl", "-k", "--since", f"@{start}", "--no-pager"], text=True, capture_output=True)
        (out / "kernel.log").write_text(journal.stdout)
        (out / "journal.err").write_text(journal.stderr)
        xids = [line for line in journal.stdout.splitlines() if re.search(r"NVRM:.*Xid", line, re.I)]
        (out / "xids.txt").write_text("\n".join(xids))
        if journal.returncode or xids:
            raise ValueError("kernel journal unavailable or Xid detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.root / f"vector-run-{int(time.time())}-{os.getpid()}"
    out.mkdir(parents=True)
    code = 1
    try:
        dry(args.snapshot, out)
        if not args.dry_run:
            run(args.snapshot, out)
        code = 0
    finally:
        (out / "verdict.json").write_text(json.dumps({"exit_code": code, "gpu_launched": not args.dry_run}) + "\n")
        print(out)

if __name__ == "__main__":
    main()
