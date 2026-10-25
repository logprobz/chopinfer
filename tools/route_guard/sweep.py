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
import stat
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_RSS_KIB = 16 * 1024 * 1024


def digest_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise RuntimeError("route artifact must be a regular file")
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def receipt_identity(path: Path, part: int, parts: int, digest: str, tier: str, seed: int) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise RuntimeError("missing, linked or oversized route receipt")
    with path.open("rb") as stream:
        header = stream.read(8192).decode("utf-8")
    match = re.match(r'^\{"identity":("(?:[^"\\]|\\.)*"),"index":(\d+),"count":(\d+),', header)
    if match is None:
        raise RuntimeError("invalid route receipt header")
    identity = json.loads(match[1])
    if (int(match[2]), int(match[3])) != (part, parts) or not identity.startswith(digest + ":"):
        raise RuntimeError("route receipt uses a different executable or piece plan")
    if not re.search(rf':{tier.capitalize()}:{seed}:[1-9][0-9]*$', identity):
        raise RuntimeError("route receipt uses a different tier or seed")
    return identity


def completed_piece(directory: Path, part: int, args, digest: str, resources: list) -> dict:
    receipt = directory / f"part-{part}.json"
    identity = receipt_identity(receipt, part, args.parts, digest, args.tier, args.seed)
    records = [item for item in resources if item["piece"] == part and item["exit_code"] == 0]
    record = records[-1].copy() if records else {"piece": part, "exit_code": 0}
    log_name, timing_name = record.get("log", f"{part}.log"), record.get("timing", f"{part}.time")
    if any(Path(name).name != name for name in (log_name, timing_name)):
        raise RuntimeError("invalid route resource filename")
    log, timing = (directory / log_name).read_text(), (directory / timing_name).read_text()
    if f"test {args.test} ..." not in log or not re.search(r"Exit status:\s*0\s*$", timing, re.MULTILINE):
        raise RuntimeError("completed receipt lacks a matching successful test")
    peak = check_result(log, timing, 0)
    digest_now = digest_file(receipt)
    if record.get("receipt_sha256", digest_now) != digest_now:
        raise RuntimeError("completed route receipt changed")
    return {**record, "log": log_name, "timing": timing_name, "peak_rss_kib": peak,
            "receipt_sha256": digest_now, "identity": identity}


def journal_proof(path: Path, digest: str) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise RuntimeError("missing, linked or oversized journal proof")
    proof = json.loads(path.read_text())
    if not isinstance(proof, dict) or proof.get("schema") != "chopinfer.route-journal-proof.v1" or proof.get("build_sha256") != digest:
        raise RuntimeError("journal proof does not match the audit executable")
    tests = proof.get("tests")
    if not isinstance(tests, list) or not tests or any(
        not isinstance(test, dict) or not test.get("name") or (test.get("exit_code"), test.get("passed"), test.get("failed"), test.get("ignored")) != (0, 1, 0, 0)
        for test in tests
    ):
        raise RuntimeError("journal proof needs executed, passing pin tests")
    return proof


def attempt_paths(output: Path, part) -> tuple[Path, Path]:
    stem = str(part)
    attempt = 0
    while (output / f"{stem}.log").exists() or (output / f"{stem}.time").exists():
        attempt += 1
        stem = f"{part}-retry-{attempt}"
    return output / f"{stem}.log", output / f"{stem}.time"


def command(binary: Path, test: str, timing: Path) -> list[str]:
    capture = timing.parent.name or binary.stem
    return [
        "bash", str(ROOT / "tools/dev/heavy.sh"), "--label", f"route-guard:{capture}:{test}:{timing.stem}",
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
    parser.add_argument("--resume", action="store_true", help="resume the same pinned executable and case plan")
    parser.add_argument("--metadata-audit", type=Path, help="complete capture to re-audit after a reviewed metadata-only change")
    parser.add_argument("--journal-proof", type=Path, help="fresh journal-pin proof for the audit executable")
    parser.add_argument("--reason", help="reviewed reason for a metadata-only re-audit")
    args = parser.parse_args()
    if os.environ.get("CHOPINFER_HEAVY_ADMITTED"):
        parser.error("start this coordinator outside heavy.sh; it admits each piece separately")
    if "route_sweep::" not in args.test or not 1 <= args.parts <= 4096 or not 0 <= args.seed < 2 ** 64:
        parser.error("provide one route_sweep test, 1..4096 parts and an unsigned seed")
    if args.metadata_audit and (args.resume or not args.journal_proof or not args.reason or not args.reason.strip()):
        parser.error("metadata re-audit needs a new output directory, fresh journal proof and reviewed reason")
    if not args.metadata_audit and (args.journal_proof or args.reason):
        parser.error("journal proof and reason apply only to --metadata-audit")
    for tool in ("systemd-run", "timeout", "/usr/bin/time"):
        if shutil.which(tool) is None:
            parser.error(f"required tool is missing: {tool}")
    binary = args.binary.resolve(strict=True)
    output = args.out.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the checkout")
    if args.resume:
        if not output.is_dir():
            parser.error("resume needs the existing capture directory")
    else:
        output.mkdir(mode=0o700)
    # Cargo may replace its output between queue admissions. Pin this run to
    # one open executable inode before taking any coverage measurements.
    snapshot = output / "sweep-test"
    if args.resume:
        if digest_file(binary) != digest_file(snapshot):
            raise RuntimeError("resume must use the original pinned executable")
    else:
        with binary.open("rb") as source, snapshot.open("xb") as target:
            shutil.copyfileobj(source, target)
        snapshot.chmod(0o500)
    binary = snapshot
    digest = digest_file(binary)
    plan = {"test": args.test, "tier": args.tier, "seed": args.seed, "parts": args.parts,
            "build_sha256": digest}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("resume uses a different executable or case plan")
    env = os.environ.copy()
    env.pop("CHOPINFER_ROUTE_CAPTURE_BUILD", None)
    env.pop("CHOPINFER_ROUTE_METADATA_ONLY", None)
    env.update(
        CHOPINFER_ROUTE_SWEEP=args.tier,
        CHOPINFER_ROUTE_PARTS_DIR=str(output),
        CHOPINFER_ROUTE_PARTS=str(args.parts),
        CHOPINFER_ROUTE_BUILD=digest,
        CHOPINFER_ROUTE_SEED=str(args.seed),
    )
    peaks = []
    resource_path = output / "piece-resources.json"
    resources = json.loads(resource_path.read_text()) if args.resume and resource_path.exists() else []
    if args.resume:
        for part in range(args.parts):
            if (output / f"part-{part}.json").exists():
                completed_piece(output, part, args, digest, resources)
    write_json(plan_path, plan)
    audit_metadata = {}
    if args.metadata_audit:
        capture = args.metadata_audit.resolve(strict=True)
        if capture.is_relative_to(ROOT):
            raise RuntimeError("capture must be outside the checkout")
        capture_digest = digest_file(capture / "sweep-test")
        proof = journal_proof(args.journal_proof, digest)
        capture_resources = capture / "piece-resources.json"
        records = json.loads(capture_resources.read_text()) if capture_resources.exists() else []
        verified = [completed_piece(capture, part, args, capture_digest, records) for part in range(args.parts)]
        if len({record["identity"] for record in verified}) != 1:
            raise RuntimeError("capture mixes different case plans")
        peaks.extend(record["peak_rss_kib"] for record in verified)
        for part, record in enumerate(verified):
            target = output / f"part-{part}.json"
            with (capture / target.name).open("rb") as source, target.open("xb") as copied:
                shutil.copyfileobj(source, copied)
            target.chmod(0o400)
            if digest_file(target) != record["receipt_sha256"]:
                raise RuntimeError("capture changed while sealing audit receipts")
        audit_metadata = {"capture_build_sha256": capture_digest, "audit_build_sha256": digest,
                          "metadata_only": True, "reason": args.reason, "journal_proof": proof,
                          "capture_pieces": verified}
        write_json(output / "capture-proof.json", audit_metadata)
        env.update(CHOPINFER_ROUTE_CAPTURE_BUILD=capture_digest, CHOPINFER_ROUTE_METADATA_ONLY="1")
    for part in (["audit"] if args.metadata_audit else [*range(args.parts), "audit"]):
        if digest_file(binary) != digest:
            raise RuntimeError("test executable changed during sweep")
        if args.resume and part != "audit" and (output / f"part-{part}.json").exists():
            record = completed_piece(output, part, args, digest, resources)
            for index in range(len(resources) - 1, -1, -1):
                item = resources[index]
                if item["piece"] == part and item["exit_code"] == 0 and item.get("log", f"{part}.log") == record["log"]:
                    resources[index] = record
                    break
            else:
                resources.append(record)
            write_json(resource_path, resources)
            peaks.append(record["peak_rss_kib"])
            print(f"{args.test}: retain verified piece {part}, peak_rss_kib={record['peak_rss_kib']}", flush=True)
            continue
        env["CHOPINFER_ROUTE_PART"] = str(part)
        log, timing = attempt_paths(output, part)
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
                          "elapsed": elapsed[1] if elapsed else None, "log": log.name, "timing": timing.name})
        write_json(resource_path, resources)
        log_text = log.read_text()
        peak = check_result(log_text, timing_text, result.returncode)
        if part != "audit":
            resources[-1] = completed_piece(output, part, args, digest, resources)
            write_json(resource_path, resources)
        elif args.metadata_audit and "case_plan_matches=true" not in log_text:
            raise RuntimeError("metadata re-audit omitted the complete case-plan check")
        peaks.append(peak)
        print(f"{args.test}: {part} passed, peak_rss_kib={peak}", flush=True)
    summary = {"test": args.test, "tier": args.tier, "seed": args.seed,
               "parts": args.parts, "build_sha256": digest, "peak_rss_kib": max(peaks),
               "rss_budget_kib": MAX_RSS_KIB, "memory_max_gib": 24,
               "piece_resources": resources, "verdict": "passed", **audit_metadata}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
