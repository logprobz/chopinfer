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
        for phase, rows in [("Decode", [1, 2, 4, 6, 8, 16, 32]),
                            ("Verify", [1, 2, 4, 5, 6, 8, 16, 32, 48, 64, 128, 192]),
                            ("Prefill", [128])]:
            for row in rows:
                result.add(f"e{experts}k{top_k}-{phase}-{row}-Mixed")
        for routing in ["All", "Seeded"]:
            result.add(f"e{experts}k{top_k}-Verify-192-{routing}")
        for routing in ["All", "Seeded", "Concentrated"]:
            result.add(f"e{experts}k{top_k}-Verify-64-{routing}")
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
        for name in ["old_ms", "active_ms", "compact_ms", "block_warps_ms", "streamed_ms", "streamed_old_ms", "previous_ms", "selected_paired_ms"]:
            if not math.isfinite(row[name]) or row[name] <= 0:
                errors.append(f"{label}: invalid {name}")
        candidate = row["rows"] <= 64 and row["phase"] != "Prefill"
        if row["streamed_candidate"] != candidate:
            errors.append(f"{label}: wrong streamed coverage")
        modes = ["old", "selected", "compact", "block_warps", "streamed", "previous"]
        if row["compared_modes"] != modes or row["compared_outputs"] != 12 or row["live_replays"] != [row["rows"],1,0,row["rows"]]:
            errors.append(f"{label}: incomplete bitwise comparison")
        compact = row["experts"] == 128 and row["top_k"] == 3 and row["phase"] == "Decode" and row["rows"] == 1
        if row["selected_compact"] != compact or row["selected_streamed"] != (candidate and not compact) or row["streamed_bitwise"] != candidate:
            errors.append(f"{label}: selected/streamed parity coverage differs")
        if row["kernel_order"] != modes:
            errors.append(f"{label}: wrong kernel timing order")
        if not row["top_k"] <= row["active_experts"] <= min(row["experts"], row["rows"] * row["top_k"]):
            errors.append(f"{label}: invalid active expert count")
        for projection, elements in [("gateup", 1152*5120), ("down", 5120*576)]:
            floor = row["active_experts"] * elements * (0.5 + 1/32) / 1.6e9
            if not math.isclose(row["weight_floor_ms"][projection], floor) or row["bandwidth_tb_s"] != 1.6:
                errors.append(f"{label}: invalid bandwidth floor")
            times = row["kernel_ms"][projection]
            ratios = row["floor_ratio"][projection]
            if len(times) != 6 or len(ratios) != 6:
                errors.append(f"{label}: missing kernel times or floor ratios")
            elif any(not math.isfinite(t) or t <= 0 or not math.isclose(r, t/floor, rel_tol=1e-6) for t, r in zip(times, ratios)):
                errors.append(f"{label}: invalid kernel time or floor ratio")
        if row["bitwise"] is not True:
            errors.append(f"{label}: parity failed")
        if row["phase"] != "Prefill" and row["dropped_join_detected"] is not True:
            errors.append(f"{label}: dropped join was not detected")
        if row["selected_paired_ms"] > row["previous_ms"] * 1.03:
            errors.append(f"{label}: selected policy exceeds previous selection by more than 3%")
        if row["active_ms"] > row["old_ms"] * 1.03:
            errors.append(f"{label}: selected policy exceeds serial baseline by more than 3%")
        if row["speed_passed"] is not True:
            errors.append(f"{label}: timing gate failed")
    if seen != expected_cases():
        errors.append(f"case coverage differs: missing={sorted(expected_cases()-seen)}, extra={sorted(seen-expected_cases())}")
    return errors


def speed_table(records):
    lines = ["Active means the serving policy, including measured fallbacks.",
             "TP4 decode/verify through 64 rows selects streamed, except DSpark Decode-1 selects compact columns; prefill and larger rows keep the old path.",
             "Weight-only floor uses unique active expert MXFP4 codes and UE8M0 scales at 1.6 TB/s.",
             "Warm graph/event timings include all projection stages; floor excludes activation, scratch and output traffic.",
             "case | experts active | old ms | active ms | active/old | compact ms | block-warps ms | streamed ms | streamed/paired-old | streamed selected | compact selected | previous ms | selected paired ms | selected/previous | gate old/active/block-warps/streamed ms | down old/active/block-warps/streamed ms | gate/down floor us | gate old/active/block-warps/streamed /floor | down old/active/block-warps/streamed /floor"]
    for row in sorted(records, key=lambda r: (r["experts"], r["phase"], r["rows"], r["routing"])):
        series = lambda values: "/".join(f"{values[i]:.5f}" for i in [0,1,3,4])
        eligible = row["selected_streamed"]
        lines.append(f'{row["case"]} | {row["active_experts"]} | {row["old_ms"]:.6f} | {row["active_ms"]:.6f} | '
                     f'{row["active_ms"]/row["old_ms"]:.4f} | {row["compact_ms"]:.6f} | {row["block_warps_ms"]:.6f} | '
                     f'{row["streamed_ms"]:.6f} | {row["streamed_ms"]/row["streamed_old_ms"]:.4f} | {eligible} | {row["selected_compact"]} | '
                     f'{row["previous_ms"]:.6f} | {row["selected_paired_ms"]:.6f} | {row["selected_paired_ms"]/row["previous_ms"]:.4f} | '
                     f'{series(row["kernel_ms"]["gateup"])} | {series(row["kernel_ms"]["down"])} | '
                     f'{row["weight_floor_ms"]["gateup"]*1000:.3f}/{row["weight_floor_ms"]["down"]*1000:.3f} | '
                     f'{series(row["floor_ratio"]["gateup"])} | {series(row["floor_ratio"]["down"])}')
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
    dry = checked([str(BINARY), "moe_active::", "--skip", "_gpu", "--skip", "prefill::", "--nocapture"], env=env)
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
            env["CHOPINFER_KERNEL_TIMEOUT_SECS"] = "400"
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
        (out / "profiling.json").write_text(json.dumps({"status":"disabled", "gates_verdict":False,
            "reason":"Profile separately; performance-counter access is not required for qualification."}) + "\n")
    except BaseException:
        failed = True
        raise
    finally:
        for process in processes:
            # A worker parent can exit while its child still
            # lives. Terminate the whole group even after the parent exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
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
