#!/usr/bin/env python3
"""Run the prebuilt MoE comparison, without compiling or queueing work."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "bin/moe-active-tests"
TESTS = ["moe_active::active_moe_backbone_gpu", "moe_active::active_moe_dspark_gpu"]


def expected_cases():
    result = set()
    for experts, top_k in [(384, 6), (128, 3)]:
        for phase, rows in [("Decode", [1, 2, 4, 8, 16, 32]),
                            ("Verify", [1, 2, 4, 5, 6, 8, 16, 32, 64, 128, 192]),
                            ("Prefill", [128])]:
            for row in rows:
                result.add(f"e{experts}k{top_k}-{phase}-{row}-Mixed")
        for routing in ["All", "Seeded"]:
            result.add(f"e{experts}k{top_k}-Verify-192-{routing}")
    return result


def parse_records(log):
    return [json.loads(line.partition("MOE_RESULT ")[2]) for line in log.splitlines() if "MOE_RESULT " in line]


def check_records(records):
    errors = []
    seen = set()
    for row in records:
        label = row["case"]
        if label in seen:
            errors.append(f"duplicate {label}")
        seen.add(label)
        for name in ["old_ms", "active_ms", "compact_ms", "block_warps_ms"]:
            if not math.isfinite(row[name]) or row[name] <= 0:
                errors.append(f"{label}: invalid {name}")
        if row["bitwise"] is not True:
            errors.append(f"{label}: parity failed")
        if row["phase"] != "Prefill" and row["dropped_join_detected"] is not True:
            errors.append(f"{label}: dropped join was not detected")
        if row["active_ms"] > row["old_ms"] * 1.03 or row["speed_passed"] is not True:
            errors.append(f"{label}: selected policy exceeds old by more than 3%")
    if seen != expected_cases():
        errors.append(f"case coverage differs: missing={sorted(expected_cases()-seen)}, extra={sorted(seen-expected_cases())}")
    return errors


def speed_table(records):
    lines = ["Active means the serving policy, including measured fallbacks.",
             "Compact and block-warps are diagnostic candidates, excluded from serving selection.",
             "case | old ms | active ms | active/old | compact ms | block-warps ms | gate old/active ms | down old/active ms"]
    for row in sorted(records, key=lambda r: (r["experts"], r["phase"], r["rows"], r["routing"])):
        gate = row["kernel_ms"]["gateup"]
        down = row["kernel_ms"]["down"]
        lines.append(f'{row["case"]} | {row["old_ms"]:.6f} | {row["active_ms"]:.6f} | '
                     f'{row["active_ms"]/row["old_ms"]:.4f} | {row["compact_ms"]:.6f} | {row["block_warps_ms"]:.6f} | '
                     f'{gate[0]:.6f}/{gate[1]:.6f} | {down[0]:.6f}/{down[1]:.6f}')
    return "\n".join(lines) + "\n"

def checked(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout

def preflight():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    listing = checked([str(BINARY), "moe_active::", "--list"])
    for name in TESTS + ["moe_active::active_moe_cpu_dry_run"]:
        assert name + ": test" in listing, name
    assert "libcuda.so.1" in checked(["ldd", str(BINARY)])
    env = os.environ.copy()
    for name in ["CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_KERNEL_REQUIRE"]:
        env.pop(name, None)
    dry = checked([str(BINARY), "moe_active::", "--skip", "_gpu", "--nocapture"], env=env)
    assert "3 passed; 0 failed" in dry and "skipping" not in dry
    (ROOT / "dry-run.log").write_text(dry)
    checked(["journalctl", "-k", "-n", "0", "--no-pager"])
    return {"cpu_dry_run": "passed", "tests": TESTS, "sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest()}

def main():
    record = preflight()
    (ROOT / "dry-run.json").write_text(json.dumps(record, indent=2) + "\n")
    if sys.argv[1:] == ["--dry-run"]:
        print(json.dumps(record))
        return
    assert sys.argv[1:] == ["--run"]
    pair = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert re.fullmatch(r"[0-9]+,[0-9]+", pair), "runner must name two idle GPU ordinals"
    devices = pair.split(",")
    assert devices[0] != devices[1]
    start = int(time.time())
    out = ROOT / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
    out.mkdir(parents=True)
    processes = []
    logs = []
    failed = False
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        power = checked(["nvidia-smi", "-i", pair, "--query-gpu=index,name,power.limit,power.draw,clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits"])
        (out / "conditions.csv").write_text(power)
        rows = [line.split(",") for line in power.strip().splitlines()]
        assert len(rows) == 2 and all(abs(float(row[2]) - 250) < 0.1 for row in rows), "expected 250 W cap"
        for device, selector in zip(devices, TESTS):
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("CHOPINFER_KERNEL_GPUS", None)
            env["CHOPINFER_KERNEL_GPU"] = device
            env["CHOPINFER_KERNEL_REQUIRE"] = "1"
            env["CHOPINFER_KERNEL_TIMEOUT_SECS"] = "1000"
            env["CHOPINFER_KERNEL_RECEIPTS"] = str(out / ("receipts-" + device))
            log = open(out / ("gpu-" + device + ".log"), "w")
            logs.append(log)
            processes.append(subprocess.Popen([str(BINARY), selector, "--exact", "--nocapture", "--test-threads=1"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        codes = [process.wait(timeout=max(1, 1100 - (time.time() - start))) for process in processes]
        for log in logs:
            log.flush()
        (out / "exit-codes.json").write_text(json.dumps(codes) + "\n")
        records = []
        for device in devices:
            output = (out / ("gpu-" + device + ".log")).read_text()
            records.extend(parse_records(output))
        table = speed_table(records)
        (out / "speed-table.txt").write_text(table)
        (out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        print(table, flush=True)
        errors = check_records(records)
        (out / "performance-errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        assert codes == [0, 0], codes
        assert not errors, errors
        for device in devices:
            output = (out / ("gpu-" + device + ".log")).read_text()
            assert "1 passed; 0 failed; 0 ignored" in output and "skipping" not in output
    except BaseException:
        failed = True
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        journal = checked(["journalctl", "-k", "--since", "@" + str(start), "--no-pager"])
        (out / "kernel-journal.log").write_text(journal)
        xids = [line for line in journal.splitlines() if re.search(r"NVRM:.*Xid", line, re.I)]
        (out / "xids.json").write_text(json.dumps(xids) + "\n")
        (out / "verdict.json").write_text(json.dumps({"passed": not failed and not xids, "power_limit_w": 250, "xids": len(xids), "speed_limit_ratio": 1.03}) + "\n")
        assert not xids, xids
    print(str(out))

if __name__ == "__main__":
    main()
