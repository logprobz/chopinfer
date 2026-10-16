#!/usr/bin/env python3
"""Run the assigned single-GPU few-expert baseline or comparison within 20 minutes."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    "moe::qwen38::qwen38_moe_smalle_bitwise_and_bench",
]
BENCHES = [
    ("moe::grouped_moe_mimo_47_layer_graph", "CHOPINFER_MOE_47_LAYER_BENCH"),
    ("moe::grouped_moe_mimo_random_route_47_layer_graph", "CHOPINFER_MOE_RANDOM_ROUTE_BENCH"),
    ("moe::grouped_moe_mimo_real_route_47_layer_graph", "CHOPINFER_MOE_REAL_ROUTE_BENCH"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", type=int)
    parser.add_argument("--binary", type=Path, required=True, help="Prebuilt kernel-lab ops test executable")
    parser.add_argument("--end", help="Assigned end time in ISO 8601 UTC, required for GPU execution")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--record", action="store_true", help="Create an immutable baseline before kernel edits")
    parser.add_argument("--preflight", action="store_true", help="Check selectors and fixtures without CUDA")
    parser.add_argument("--cubins", type=Path, help="Frozen baseline cubin directory, otherwise use embedded build")
    parser.add_argument("--qwen-logits", type=Path, help="Existing F32 router logits with at least 32 rows and 512 columns")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    binary = args.binary.resolve(strict=True)
    listing = subprocess.check_output([str(binary), "--list"], text=True, cwd=ROOT)
    for name in TESTS + [name for name, _ in BENCHES]:
        if listing.splitlines().count(name + ": test") != 1:
            raise RuntimeError(f"missing or ambiguous required test: {name}")
    fixture = json.loads((ROOT / "fixtures/mimo26/routing-locality.json").read_text())
    for tokens in (8, 16, 32, 64, 128, 256):
        if not any(row["tokens"] == tokens and row.get("bench_samples") for row in fixture["shapes"]):
            raise RuntimeError(f"missing real-route fixture for {tokens}")
    artifacts = {}
    if args.cubins:
        args.cubins = args.cubins.resolve(strict=True)
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.cubins.glob("*.cubin"))}
        if not artifacts:
            parser.error("--cubins contains no cubins")
    route_source = None
    if args.qwen_logits:
        args.qwen_logits = args.qwen_logits.resolve(strict=True)
        data = args.qwen_logits.read_bytes()
        if len(data) < 32 * 512 * 4 or len(data) % (512 * 4):
            parser.error("invalid Qwen F32 router dump dimensions")
        route_source = {"path": str(args.qwen_logits), "sha256": hashlib.sha256(data).hexdigest(),
                        "rows": len(data) // (512 * 4), "columns": 512, "dtype": "f32"}
    if args.preflight:
        print("Preflight passed: four exact selectors and every MiMo route bucket. No GPU work ran.")
        return
    if not args.end:
        parser.error("--end is required")
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if end.tzinfo is None:
        parser.error("--end must include a timezone")
    seconds = min(1200, end.timestamp() - time.time())
    if seconds <= 0:
        parser.error("the assigned window has ended")
    deadline = time.monotonic() + seconds
    base = Path(os.environ["CHOPINFER_DATA"]) / "kopt-moe-smalle"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = base / "logs" / stamp
    run.mkdir(parents=True)
    receipts = base / "receipts" / stamp
    receipts.mkdir(parents=True)
    (run / "artifacts.json").write_text(json.dumps({
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cubins": str(args.cubins) if args.cubins else "embedded build",
        "cubin_sha256": artifacts, "qwen_route_source": route_source,
    }, indent=2) + "\n")
    golden = args.golden.resolve()
    if not golden.is_relative_to(base.resolve()):
        parser.error("golden files must be inside the task's data directory")
    if args.record:
        golden.mkdir(parents=True, exist_ok=False)
        manifest = {
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
            "kernel_locks": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                             ("crates/kernel-lab/kernels.lock", "models/mimo26/bundle/kernels.lock")},
            "created_at": stamp,
        }
        (golden / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elif not golden.is_dir():
        parser.error("comparison requires an existing baseline")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("CHOPINFER_MOE_") or key in ("CHOPINFER_KERNEL_GPUS", "CHOPINFER_KERNEL_SHARED_DEVICE"):
            env.pop(key)
    env.update(CHOPINFER_KERNEL_GPU=str(args.device), CHOPINFER_MOE_SMALLE="1",
               CHOPINFER_MOE_SMALLE_GOLDEN=str(golden), CHOPINFER_MOE_SMALLE_RECORD="1" if args.record else "0",
               CHOPINFER_KERNEL_RECEIPTS=str(receipts), CHOPINFER_KERNEL_RECEIPT_DIR=str(receipts),
               CHOPINFER_MOE_GRAPH_LAYERS="24",
               CHOPINFER_MOE_REAL_ROUTE_SAMPLES="1", CHOPINFER_MOE_47_FUSED="0",
               CHOPINFER_QWEN38_RANKS="4", CHOPINFER_QWEN_MOE_RANK="1")
    if args.cubins:
        env["CHOPINFER_MOE_CUBIN_DIR"] = str(args.cubins)
    if args.qwen_logits:
        env["CHOPINFER_MOE_QWEN_LOGITS"] = str(args.qwen_logits)
    results = []
    tasks = [(name, {}) for name in TESTS]
    tasks += [(name, {flag: "1", "CHOPINFER_MOE_47_TOKENS": str(tokens)})
              for tokens in (16, 32) for name, flag in BENCHES]
    try:
        for index, (name, extra) in enumerate(tasks):
            remaining = deadline - time.monotonic()
            if remaining <= 3:
                raise TimeoutError("assigned GPU window expired")
            log = run / f"{index:02d}-{name.split('::')[-1]}.log"
            # timeout owns and reaps the test process and its children.
            cmd = ["timeout", "--signal=TERM", "--kill-after=3s", f"{max(0.1, remaining - 3):.1f}s",
                   "nice", "-n", "19", str(binary), name, "--exact", "--nocapture", "--test-threads=1"]
            with log.open("x") as output:
                status = subprocess.run(cmd, cwd=ROOT, env=env | extra, stdout=output, stderr=subprocess.STDOUT).returncode
            text = log.read_text()
            results.append({"test": name, "env": extra, "exit": status, "log": str(log)})
            (run / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results[-1]), flush=True)
            if status or "1 passed; 0 failed; 0 ignored" not in text or "skipping" in text.lower() or "SKIP:" in text:
                raise RuntimeError(f"required test failed or skipped: {log}")
        paths = list(receipts.rglob("*.json"))
        if not paths:
            raise RuntimeError("no GPU receipts")
        for path in paths:
            if json.loads(path.read_text()).get("measurement_label") != "exclusive-device":
                raise RuntimeError(f"shared-device measurement cannot qualify: {path}")
        print(f"Single-GPU run passed. Golden outputs: {golden}. Receipts: {receipts}")
    finally:
        (run / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
