#!/usr/bin/env python3
"""Prepare E5 on CPU, then measure HC and linear kernels in one assigned GPU slot.

Source local.env first. Run requires --go, one idle device and a UTC end time.
All artifacts stay under $CHOPINFER_DATA/draft-recurrence. No device reservations are acquired.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import qwen38_e5_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[2]
TESTS = {"qwen38_hc_e5": "qwen38_hc_e5", "ops": "linear::e5::linear_e5",
         "qwen38_hc_up_mix": "qwen38_hc_up_mix_matches_the_unfused_chain",
         "qwen38_gemm": "qwen38_gemm_g4"}
SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
            "SchedulerStats", "WarpStateStats")


def data_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to("/data"):
        raise ValueError(f"path must be under /data: {path}")
    return path


def environment():
    os.umask(0o077)
    base = data_path(os.environ["CHOPINFER_DATA"]) / "draft-recurrence"
    env = dict(os.environ, CARGO_NET_OFFLINE="true", CARGO_BUILD_JOBS="8",
               CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
               CARGO_TARGET_DIR=str(base / "target"), TMPDIR=str(base / "tmp"),
               CHOPINFER_SCRATCH=str(base / "tmp"), PYTHONDONTWRITEBYTECODE="1",
               CHOPINFER_QUAL_CONFIG=str(ROOT / "models/qwen38/qual.json"))
    for name in ("tmp", "logs"):
        (base / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    return base, env


def call(command, env, log, timeout=None):
    with log.open("w") as output:
        output.write(json.dumps([str(x) for x in command]) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def deadline(value, now=None, mode="no-counters"):
    end = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if end.tzinfo is None or end.utcoffset().total_seconds() != 0:
        raise ValueError("end time must explicitly use UTC")
    remaining = end.timestamp() - (time.time() if now is None else now)
    minimum, maximum = (10, 15) if mode == "no-counters" else (15, 20)
    if remaining < minimum * 60:
        raise ValueError(f"E5 {mode} needs at least {minimum} minutes before its authorized end")
    return time.monotonic() + min(maximum * 60, remaining)


def test_command(prepared, name):
    return [prepared["binaries"][name]["path"], "--exact", TESTS[name], "--nocapture",
            "--test-threads=1"]


def ncu_command(executable, output, kernel, command, count=1):
    args = [executable, "--target-processes", "all", "--replay-mode", "application",
            "--cache-control", "none", "--clock-control", "none", "--kernel-name",
            "regex:" + kernel, "--launch-count", str(count), "--export", str(output)]
    for section in SECTIONS:
        args += ["--section", section]
    return args + command


def prepare(base, env, mode="no-counters"):
    for tool in ("cargo", "rustc", "nice", *(('nvcc', 'cuobjdump') if mode == "no-counters" else ('ncu',))):
        if not shutil.which(tool, path=env["PATH"]):
            raise ValueError(f"required tool missing: {tool}")
    version = subprocess.check_output(["rustc", "--version"], env=env, text=True)
    if not version.startswith("rustc 1.89.0 "):
        raise ValueError(f"expected Rust 1.89.0, got {version.strip()}")
    ncu = None
    if mode == "ncu":
        ncu = shutil.which("ncu", path=env["PATH"])
        sections = subprocess.check_output([ncu, "--list-sections"], env=env, text=True)
        if any(section not in sections for section in SECTIONS):
            raise ValueError("installed Nsight Compute lacks an E5 section")
    command = ["nice", "-n", "19", "cargo", "test", "-p", "kernel-lab", "--no-run",
               "--locked", "--offline", "--message-format=json",
               "--config", "profile.dev.package.kernel-core.opt-level=3",
               "--config", "profile.dev.package.kernel-lab.opt-level=3"]
    for name in TESTS:
        command += ["--test", name]
    build_env = dict(env)
    build_env.pop("CHOPINFER_KERNEL_GPU", None)
    build_env.pop("CHOPINFER_KERNEL_GPUS", None)
    log = base / "logs/e5-prepare.log"
    call(command, build_env, log)
    binaries = {}
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("reason") != "compiler-artifact":
            continue
        name = record["target"]["name"]
        path = record.get("executable")
        if name in TESTS and path:
            path = data_path(path)
            listed = subprocess.check_output([str(path), "--list"], env=build_env, text=True)
            if TESTS[name] + ": test" not in listed.splitlines():
                raise ValueError(f"required test absent: {TESTS[name]}")
            binaries[name] = {"path": str(path), "sha256": digest(path)}
    if set(binaries) != set(TESTS):
        raise ValueError("build did not produce every required test binary")
    result = {"binaries": binaries, "mode": mode, "ncu": ncu,
              "diagnostics": diagnostics.build(ROOT, base, env, call) if mode == "no-counters" else None,
              "ncu_version": subprocess.check_output([ncu, "--version"], env=env, text=True) if ncu else None,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (base / "e5-prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CPU preparation complete: " + str(base / "e5-prepared.json"))


def stop_group(process):
    # Popen starts a new session, so its process group contains only our children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def call_owned(command, env, log, finish, limit=240):
    remaining = min(limit, finish - time.monotonic() - 45)
    if remaining <= 0:
        raise TimeoutError("E5 deadline reached before the next command")
    with log.open("w") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stop_group(process)


def idle(device, env, log):
    rows = subprocess.check_output(["nvidia-smi", "--id=" + device,
        "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        env=env, text=True, timeout=10)
    processes = subprocess.check_output(["nvidia-smi", "--id=" + device,
        "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        env=env, text=True, timeout=10)
    log.write_text(rows + processes)
    values = [row.split(",") for row in rows.strip().splitlines()]
    if len(values) != 1 or values[0][0].strip() != device or int(values[0][-1]) >= 1000:
        raise ValueError("assigned device is missing or has at least 1000 MiB in use")
    if processes.strip():
        raise ValueError("assigned device has another compute process")
    uuid = values[0][1].strip()
    if not uuid.startswith("GPU-"):
        raise ValueError("assigned device has no physical GPU UUID")
    return uuid


def check_receipts(root):
    expected = {
        "qwen38-hc-e5": {(str(rows), mode) for rows in (16, 32) for mode in ("in", "mix", "chain")},
        "qwen38-linear-e5": {(shape, mode) for shape in ("16x1", "32x1", "16x2")
                            for mode in ("conv", "delta", "norm")
                            if (shape, mode) != ("16x2", "norm")},
    }
    counts = {}
    for kind, required in expected.items():
        found = set()
        for path in (root / "receipts" / kind).glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("measurement_label") != "exclusive-device":
                raise ValueError(f"shared-device receipt: {path}")
            if record.get("profile_only"):
                continue
            elapsed = record.get("median_us_per_call", 0)
            if not record.get("eager_graph_bits_equal") or not isinstance(elapsed, (int, float)) or not 0 < elapsed < float("inf"):
                raise ValueError(f"missing exactness or timing evidence: {path}")
            key = (str(record.get("rows")) if kind == "qwen38-hc-e5" else record.get("shape"), record.get("mode"))
            if key in found or key not in required:
                raise ValueError(f"unexpected or duplicate E5 case: {path}")
            found.add(key)
        if found != required:
            raise ValueError(f"incomplete E5 {kind} cases: {required - found}")
        counts[kind] = len(found)
    gates = list((root / "receipts/qwen38-hc-up-mix-g1").glob("*.json"))
    if len(gates) != 1:
        raise ValueError("missing or duplicate HC unfused-chain gate")
    gate = json.loads(gates[0].read_text())
    if (gate.get("measurement_label") != "exclusive-device"
            or not gate.get("graph_determinism_pdl")
            or not {16, 32}.issubset(gate.get("lives", []))):
        raise ValueError("HC gate lacks exclusive 16/32-row graph evidence")
    projections = set()
    for path in (root / "receipts/bench").glob("*.json"):
        record = json.loads(path.read_text())
        symbol = record.get("symbol")
        if symbol not in ("chop_gemm_qwen38_lin_in_c32_v1", "chop_gemm_qwen38_out_c32_v1"):
            raise ValueError(f"unexpected projection receipt: {path}")
        elapsed = record.get("chain_us_per_call", 0)
        if (record.get("measurement_label") != "exclusive-device" or record.get("m") != 32
                or not isinstance(elapsed, (int, float)) or not 0 < elapsed < float("inf")
                or symbol in projections):
            raise ValueError(f"invalid or duplicate projection receipt: {path}")
        projections.add(symbol)
    if len(projections) != 2:
        raise ValueError("missing 32-row projection measurement")
    counts.update(hc_gate=1, projections=2)
    return counts


def run(base, env, device, end, mode="no-counters"):
    if device not in ("0", "1", "2", "3"):
        raise ValueError("E5 requires one explicitly assigned device, 0 through 3")
    finish = deadline(end, mode=mode)
    prepared = json.loads((base / "e5-prepared.json").read_text())
    if prepared.get("mode") != mode:
        raise ValueError(f"rerun CPU preparation with --mode {mode}")
    artifacts = list(prepared["binaries"].values())
    if mode == "no-counters":
        artifacts.append(prepared["diagnostics"])
    for binary in artifacts:
        if digest(binary["path"]) != binary["sha256"]:
            raise ValueError("prepared binary changed; rerun CPU preparation")
    root = base / ("e5-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir()
    env = dict(env, CUDA_VISIBLE_DEVICES=device, CHOPINFER_KERNEL_GPU="0",
               CHOPINFER_QWEN38_RANKS="4", CHOPINFER_KERNEL_BENCH="1",
               CHOPINFER_KERNEL_RECEIPTS=str(root / "receipts"))
    for key in ("CHOPINFER_KERNEL_GPUS", "CHOPINFER_KERNEL_SHARED_DEVICE", "CHOPINFER_E5_PROFILE",
                "CHOPINFER_KERNEL_BENCH_FILTER", "CHOPINFER_KERNEL_BENCH_VARIANT"):
        env.pop(key, None)
    env.pop("CHOPINFER_E5_DIAGNOSTIC_CUBIN", None)
    env.pop("CHOPINFER_E5_DIAGNOSTIC_MANIFEST", None)
    if mode == "no-counters":
        env["CHOPINFER_E5_DIAGNOSTIC_CUBIN"] = prepared["diagnostics"]["path"]
        env["CHOPINFER_E5_DIAGNOSTIC_MANIFEST"] = str(Path(prepared["diagnostics"]["evidence_path"]) / "manifest.json")
    (root / "run.json").write_text(json.dumps({"device": device, "end_time": end,
        "prepared": prepared, "mode": mode, "sections": SECTIONS if mode == "ncu" else []}, indent=2) + "\n")
    assigned_uuid = idle(device, env, root / "device-before.txt")
    env["CUDA_VISIBLE_DEVICES"] = assigned_uuid
    run_record = json.loads((root / "run.json").read_text())
    run_record["assigned_uuid"] = assigned_uuid
    (root / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")

    def expired(_signum, _frame):
        raise TimeoutError("E5 authorized deadline reached")

    old_handlers = {sig: signal.signal(sig, expired) for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, finish - time.monotonic() - 45)
    try:
        if mode == "no-counters":
            call_owned([sys.executable, "-B", diagnostics.__file__, "--manifest", str(base / "e5-prepared.json"),
                        "--go", "--device", device, "--end-time", end], env, root / "occupancy.json.log", finish, limit=30)
            lines = (root / "occupancy.json.log").read_text().splitlines()
            (root / "occupancy.json").write_text("\n".join(lines[1:]) + "\n")
        call_owned(test_command(prepared, "qwen38_hc_up_mix"), env, root / "hc-exact.log", finish)
        for rows in (16, 32):
            for operation_mode in ("in", "mix", "chain"):
                arm_env = dict(env, CHOPINFER_HC_E5_ROWS=str(rows), CHOPINFER_HC_E5_MODE=operation_mode)
                call_owned(test_command(prepared, "qwen38_hc_e5"), arm_env,
                           root / f"hc-{rows}-{operation_mode}.log", finish)
        for shape in ("16x1", "32x1", "16x2"):
            for operation_mode in ("conv", "delta", "norm"):
                if operation_mode == "norm" and shape == "16x2":
                    continue
                arm_env = dict(env, CHOPINFER_LINEAR_E5_SHAPE=shape, CHOPINFER_LINEAR_E5_MODE=operation_mode)
                call_owned(test_command(prepared, "ops"), arm_env,
                           root / f"linear-{shape}-{operation_mode}.log", finish)
        for operation in ("lin_in_c32", "out_c32"):
            call_owned(test_command(prepared, "qwen38_gemm"), dict(env, CHOPINFER_KERNEL_BENCH_FILTER=operation),
                       root / f"projection-{operation}.log", finish)
        counts = check_receipts(root)
        if mode == "no-counters":
            baseline = {}
            for kind in ("qwen38-hc-e5", "qwen38-linear-e5"):
                for path in (root / "receipts" / kind).glob("*.json"):
                    record = json.loads(path.read_text())
                    if kind == "qwen38-hc-e5" and record["mode"] in ("mix", "chain"):
                        baseline[(str(record["rows"]), record["mode"])] = record["median_us_per_call"]
                    elif kind == "qwen38-linear-e5" and record["mode"] == "delta":
                        baseline[(record["shape"], "delta")] = record["median_us_per_call"]
            decision = diagnostics.decide(root, prepared["diagnostics"], baseline)
            (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
            (root / "summary.json").write_text(json.dumps({"collection_passed": True,
                "mode": mode, "timing_receipts": counts, "diagnostic_receipts": decision["diagnostic_receipts"],
                "decision": "decision.json", "occupancy": "occupancy.json", "serving_acceptance_run": False}, indent=2) + "\n")
        else:
            profiles = []
            for rows in (16, 32):
                for mode in ("mix", "chain"):
                    arm_env = dict(env, CHOPINFER_E5_PROFILE="1", CHOPINFER_HC_E5_ROWS=str(rows), CHOPINFER_HC_E5_MODE=mode)
                    label = f"ncu-hc-{rows}-{mode}"
                    kernel = r"chop_gemm_qwen38_hc_up_mix_decode_v1" if mode == "mix" else rf"chop_gemm_qwen38_(hc_in_c{rows}|hc_up_mix_decode)_v1"
                    command = ncu_command(prepared["ncu"], root / label, kernel,
                                          test_command(prepared, "qwen38_hc_e5"), 1 if mode == "mix" else 2)
                    profiles.append((label, command, arm_env))
            for shape in ("32x1", "16x2"):
                arm_env = dict(env, CHOPINFER_E5_PROFILE="1", CHOPINFER_LINEAR_E5_SHAPE=shape, CHOPINFER_LINEAR_E5_MODE="delta")
                label = f"ncu-linear-{shape}"
                profiles.append((label, ncu_command(prepared["ncu"], root / label,
                    r"chop_linear_qwen38_delta_verify_v1", test_command(prepared, "ops")), arm_env))
            for label, command, arm_env in profiles:
                call_owned(command, arm_env, root / (label + ".log"), finish, limit=180)
                report = root / (label + ".ncu-rep")
                if not report.is_file() or report.stat().st_size == 0:
                    raise ValueError("Nsight Compute produced no report: " + label)
                call_owned([prepared["ncu"], "--import", str(report), "--page", "raw", "--csv"],
                           env, root / (label + ".csv"), finish, limit=30)
            (root / "summary.json").write_text(json.dumps({"collection_passed": True,
                "timing_receipts": counts, "ncu_reports": [p[0] for p in profiles],
                "serving_acceptance_run": False}, indent=2) + "\n")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        idle(device, env, root / "device-after.txt")
    print("E5 collected: " + str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--mode", choices=("no-counters", "ncu"), default="no-counters")
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--end-time")
    args = parser.parse_args()
    if args.action == "run" and not (args.go and args.device and args.end_time):
        parser.error("run requires --go, --device and --end-time")
    base, env = environment()
    if args.action == "prepare":
        prepare(base, env, args.mode)
    else:
        run(base, env, args.device, args.end_time, args.mode)


if __name__ == "__main__":
    main()
