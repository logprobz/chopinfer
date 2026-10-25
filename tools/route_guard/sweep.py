#!/usr/bin/env python3
"""Run a prebuilt profile sweep in separately admitted, memory-limited pieces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_RSS_KIB = 16 * 1024 * 1024


def command(binary: Path, test: str, timing: Path) -> list[str]:
    return [
        "bash", str(ROOT / "tools/dev/heavy.sh"), "--label", f"route-guard:{test}",
        "systemd-run", "--user", "--scope", "--quiet",
        "--property=MemoryMax=24G", "--property=MemorySwapMax=0",
        "timeout", "--kill-after=10s", "880s",
        "/usr/bin/time", "-v", "-o", str(timing),
        str(binary), "--exact", test, "--nocapture", "--test-threads=1",
    ]


def check_result(log: str, timing: str, code: int) -> int:
    if code:
        raise RuntimeError(f"sweep piece exited {code}")
    if not re.search(r"test result: ok\. 1 passed; 0 failed;", log):
        raise RuntimeError("sweep piece did not execute exactly one passing test")
    if "peak_rss_bytes=" not in log:
        raise RuntimeError("sweep piece omitted its enforced RSS budget report")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", timing)
    if match is None:
        raise RuntimeError("sweep piece omitted /usr/bin/time peak RSS")
    peak = int(match[1])
    if peak > MAX_RSS_KIB:
        raise RuntimeError(f"sweep peak RSS {peak} KiB exceeds {MAX_RSS_KIB} KiB")
    return peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="test executable built through heavy.sh")
    parser.add_argument("--test", required=True, help="one exact route_sweep profile test name")
    parser.add_argument("--out", required=True, type=Path, help="new output directory outside the checkout")
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--tier", choices=["full", "exhaustive"], default="full")
    parser.add_argument("--seed", type=int, default=int(time.time()) // 86400)
    args = parser.parse_args()
    if os.environ.get("CHOPINFER_HEAVY_ADMITTED"):
        parser.error("start this coordinator outside heavy.sh; it admits each piece separately")
    if "route_sweep::" not in args.test or not 1 <= args.parts <= 4096 or args.seed < 0:
        parser.error("provide one route_sweep test, 1..4096 parts and an unsigned seed")
    for tool in ("systemd-run", "timeout", "/usr/bin/time"):
        if shutil.which(tool) is None:
            parser.error(f"required tool is missing: {tool}")
    binary = args.binary.resolve(strict=True)
    output = args.out.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the checkout")
    output.mkdir(mode=0o700)
    # Cargo may replace its output between queue admissions. Pin this run to
    # one open executable inode before taking any coverage measurements.
    snapshot = output / "sweep-test"
    with binary.open("rb") as source, snapshot.open("xb") as target:
        shutil.copyfileobj(source, target)
    snapshot.chmod(0o500)
    binary = snapshot
    with binary.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    env = os.environ.copy()
    env.update(
        CHOPINFER_ROUTE_SWEEP=args.tier,
        CHOPINFER_ROUTE_PARTS_DIR=str(output),
        CHOPINFER_ROUTE_PARTS=str(args.parts),
        CHOPINFER_ROUTE_BUILD=digest,
        CHOPINFER_ROUTE_SEED=str(args.seed),
    )
    peaks = []
    resources = []
    for part in [*range(args.parts), "audit"]:
        with binary.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise RuntimeError("test executable changed during sweep")
        env["CHOPINFER_ROUTE_PART"] = str(part)
        timing = output / f"{part}.time"
        log = output / f"{part}.log"
        print(f"{args.test}: admit {part}/{args.parts}, seed={args.seed}, MemoryMax=24G, deadline=890s", flush=True)
        with log.open("x") as stream:
            result = subprocess.run(command(binary, args.test, timing), env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        # Preserve GNU time and the test's budget report even when a scope dies.
        timing_text = timing.read_text() if timing.exists() else ""
        if timing_text:
            print(timing_text, flush=True)
        measured = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", timing_text)
        elapsed = re.search(r"Elapsed \(wall clock\) time.*?:\s*(\S+)\s*$", timing_text, re.MULTILINE)
        resources.append({"piece": part, "exit_code": result.returncode,
                          "peak_rss_kib": int(measured[1]) if measured else None,
                          "elapsed": elapsed[1] if elapsed else None})
        (output / "piece-resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        peak = check_result(log.read_text(), timing_text, result.returncode)
        peaks.append(peak)
        print(f"{args.test}: {part} passed, peak_rss_kib={peak}", flush=True)
    summary = {"test": args.test, "tier": args.tier, "seed": args.seed,
               "parts": args.parts, "build_sha256": digest, "peak_rss_kib": max(peaks),
               "rss_budget_kib": MAX_RSS_KIB, "memory_max_gib": 24,
               "piece_resources": resources, "verdict": "passed"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
