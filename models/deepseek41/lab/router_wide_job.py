#!/usr/bin/env python3
"""Prepare, run and audit the router tile pair (router_wide.md) on a GPU pair.

``--prepare`` writes the captured data the pair reads into ``--out``: per
captured layer, the MoE input codes of the published capture's 8192-row
chunk (rank 0), widened exactly to BF16, and the layer's router weights
from the checkpoint, with DATA.json naming each file's size and digest.

The job runs the frozen test binary's ``wide::router_wide_gpu`` once per
worker with one logical GPU each. Worker 0 takes captured layer 14, worker
1 captured layer 39, each with two synthetic cases. Every arm of every
case must write the served certified projection's logits byte for byte,
and the 8192-row cases must be timed. The job fails closed on any missing,
duplicate or failed case or arm, zero executed tests, a power cap other
than 250 W, or an Xid on its GPUs. The record, listing and timing helpers
are native_moe_job's.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

import native_moe_job as base

TEST = "wide::router_wide_gpu"
CPU_TESTS = ("wide::router_wide_cpu_plan",)
BINARY = "bin/router-norm-test"
LAYERS = (14, 39)
CASES = {0: (("captured-l14", 8192), ("outliers-8192", 8192), ("outliers-65", 65)),
         1: (("captured-l39", 8192), ("outliers-2049", 2049), ("clean-8192", 8192))}
ARMS = ("f32", "f64", "f128")
ROWS, K, N = 8192, 5120, 384
# The router projection's measured share of an 8192-row chunk (the served
# profile on the native default, every layer): the per-chunk saving scales
# it by the arm's time over the served projection's on the captured cases.
CHUNK_PROJECTION_MS = 13.75
IDLE_SETTLE_SECONDS = 30


def e4m3(code):
    """The E4M3 code's value (NaN for 0x7f and 0xff)."""
    sign = -1.0 if code & 0x80 else 1.0
    exp, man = (code >> 3) & 15, code & 7
    if exp == 15 and man == 7:
        return float("nan")
    if exp == 0:
        return sign * man / 8 * 2.0 ** -6
    return sign * (1 + man / 8) * 2.0 ** (exp - 7)


def bf16_bits(value):
    """The BF16 bits of `value`, which must be exact in BF16."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0xffff:
        raise ValueError(f"{value} is not a BF16 value")
    return bits >> 16


def widen_codes(codes, scales, rows, k):
    """BF16 rows (little-endian bytes) of E4M3 codes times their UE8M0
    group scales (one byte per 32 codes), exactly."""
    groups = k // 32
    if len(codes) != rows * k or len(scales) != rows * groups:
        raise ValueError("codes or scales have the wrong size")
    tables = {}
    out = bytearray(rows * k * 2)
    for g in range(rows * groups):
        s = scales[g]
        if s == 255:
            raise ValueError("scale 255 in the captured input")
        table = tables.get(s)
        if table is None:
            table = tables[s] = [struct.pack("<H", 0x7fc0 if e4m3(c) != e4m3(c) else bf16_bits(e4m3(c) * 2.0 ** (s - 127)))
                                 for c in range(256)]
        out[g * 64:(g + 1) * 64] = b"".join(table[c] for c in codes[g * 32:(g + 1) * 32])
    return bytes(out)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, item):
    data = Path(path).read_bytes()
    if len(data) != item["bytes"] or sha_bytes(data) != item["sha256"]:
        raise ValueError(f"capture file differs from its manifest: {path}")
    return data


def tensor(checkpoint, name):
    """One tensor's dtype, shape and bytes from a safetensors checkpoint."""
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())
    path = Path(checkpoint) / index["weight_map"][name]
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        item = header[name]
        begin, end = item["data_offsets"]
        f.seek(8 + size + begin)
        return item["dtype"], item["shape"], f.read(end - begin)


def prepare(ready, checkpoint, out):
    """Writes input-l{L}.bf16 and router-l{L}.bf16 per layer and DATA.json."""
    ready_doc = json.loads(Path(ready).read_text())
    manifest_bytes = Path(ready_doc["path"]).read_bytes()
    if sha_bytes(manifest_bytes) != ready_doc["sha256"]:
        raise ValueError("capture manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    out.mkdir(parents=True, exist_ok=False)
    files, sources = {}, {}
    for layer in LAYERS:
        found = []
        for c in manifest["captures"]:
            meta = json.loads(checked(Path(c["path"]) / "down.json", c["files"]["down.json"]))
            if (c["prompt"], meta["layer"], meta["rank"], meta["rows"], meta["positions"][0]) == ("input-0", layer, 0, ROWS, 0):
                found.append((c, meta))
        if len(found) != 1:
            raise ValueError(f"expected one 8192-row rank-0 capture of layer {layer}")
        c, _ = found[0]
        codes = checked(Path(c["path"]) / "input.e4m3", c["files"]["input.e4m3"])
        scales = checked(Path(c["path"]) / "input.ue8m0", c["files"]["input.ue8m0"])
        dtype, shape, weights = tensor(checkpoint, f"layers.{layer}.ffn.gate.weight")
        if dtype != "BF16" or shape != [N, K]:
            raise ValueError(f"layer {layer} router weight is {dtype} {shape}")
        for name, data in ((f"input-l{layer}.bf16", widen_codes(codes, scales, ROWS, K)),
                           (f"router-l{layer}.bf16", weights)):
            (out / name).write_bytes(data)
            files[name] = {"bytes": len(data), "sha256": sha_bytes(data)}
        sources[str(layer)] = {"capture": {k: c["files"][k] for k in ("input.e4m3", "input.ue8m0")},
                               "router": f"layers.{layer}.ffn.gate.weight"}
    (out / "DATA.json").write_text(json.dumps({"schema": "chopinfer.router-wide-data.v1", "files": files,
                                               "sources": sources}, indent=2) + "\n")


def analyze(text, worker):
    """Every case of the worker, every arm bitwise, the 8192-row cases timed."""
    base.libtest(text, 1)
    rows = base.records(text, "router_wide_record")
    base.unique(rows, lambda r: (r["case"], r["live"]), CASES[worker], "router tile")
    for r in rows:
        label = r["case"]
        if r["passed"] is not True or [a["arm"] for a in r["arms"]] != list(ARMS):
            raise ValueError(f"{label}: failed, or missing or extra arms")
        for a in r["arms"]:
            if a["bitwise"] is not True:
                raise ValueError(f"{label} {a['arm']}: differs from the served projection")
            if r["live"] == ROWS:
                base.validate_timing(a["timing"])
            elif a["timing"] is not None:
                raise ValueError(f"{label} {a['arm']}: timed below the served step")
    synthetic = [r for r in rows if not r["case"].startswith("captured")]
    if not any(r["uncertified_rows"] > 0 and r["simt_weight_groups"] > 0 for r in synthetic):
        raise ValueError("no synthetic case reaches the SIMT path")
    done = base.records(text, "router_wide_done")
    if done != [{"worker": worker, "cases": len(rows), "failed": 0}]:
        raise ValueError("missing case completion")
    return rows


def summarize(parsed):
    """Report-only: per arm, the served and arm medians of each timed case,
    and on the captured cases their means and the per-chunk saving
    (CHUNK_PROJECTION_MS times one minus their ratio)."""
    rows = [r for p in parsed for r in p]
    out = {}
    for arm in ARMS:
        t = [(r["case"], a["timing"]) for r in rows for a in r["arms"] if a["arm"] == arm and a["timing"]]
        captured = [x for c, x in t if c.startswith("captured")]
        served = sum(x["ms"][0] for x in captured) / len(captured)
        ms = sum(x["ms"][1] for x in captured) / len(captured)
        out[arm] = {"cases": {c: {"served_ms": x["ms"][0], "arm_ms": x["ms"][1], "ratio": x["median_round_ratio"]}
                              for c, x in t},
                    "captured_served_ms_mean": served, "captured_arm_ms_mean": ms,
                    "per_chunk_saving_ms": CHUNK_PROJECTION_MS * (1 - ms / served)}
    return out


def gpus(visible, idle):
    """Power caps, and with `idle` both GPUs idle, waiting out a previous
    job's utilization and memory for up to IDLE_SETTLE_SECONDS."""
    deadline = time.monotonic() + IDLE_SETTLE_SECONDS
    while True:
        output = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,power.limit,utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits"], text=True)
        rows = list(csv.reader(output.splitlines()))
        if len(rows) != 2 or any(abs(float(r[1]) - 250) > 0.5 for r in rows):
            raise ValueError("both assigned GPUs must have a 250 W cap")
        if not idle or all(float(r[2]) == 0 and float(r[3]) <= 1000 for r in rows):
            return [[x.strip() for x in r] for r in rows]
        if time.monotonic() >= deadline:
            raise ValueError("assigned GPU pair is not idle")
        time.sleep(1)


def bus_ids(visible):
    """The assigned GPUs' PCI addresses as Xid lines print them."""
    output = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=pci.bus_id",
                                      "--format=csv,noheader"], text=True)
    ids = set()
    for line in output.split():
        domain, bus, device = line.strip().split(":")
        ids.add(f"{domain[-4:]}:{bus}:{device.split('.')[0]}".lower())
    return ids


def own_xids(lines, buses):
    """The journal's Xid lines, and those on the assigned GPUs or naming no
    device (other jobs' Xids are not this job's)."""
    xids = [l for l in lines if re.search(r"NVRM: Xid|Xid \(", l)]
    out = []
    for l in xids:
        m = re.search(r"Xid \(PCI:([0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+)", l)
        if m is None or m.group(1).lower() in buses:
            out.append(l)
    return xids, out


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


def environment(root, worker=None):
    env = dict(os.environ)
    for key in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE"):
        env.pop(key, None)
    env.update(CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS=str(root / "cubins"),
               CHOPINFER_DEEPSEEK41_ROUTER_WIDE_DATA=str(root / "data"),
               CHOPINFER_KERNEL_TIMEOUT_SECS="600")
    if worker is not None:
        env.update(CHOPINFER_KERNEL_GPU=str(worker), CHOPINFER_KERNEL_REQUIRE="1",
                   CHOPINFER_DEEPSEEK41_ROUTER_WIDE_WORKER=str(worker))
    return env


def test_args(root, name):
    return [str(root / BINARY), name, "--exact", "--include-ignored", "--nocapture"]


def run(root, plan, out):
    deadline = time.monotonic() + plan["budget"]["tiles"]
    children = []
    try:
        for worker in (0, 1):
            log = out / f"tiles-{worker}.log"
            handle = log.open("w")
            p = subprocess.Popen([*test_args(root, TEST), "--test-threads=1"], env=environment(root, worker),
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((p, handle, log))
        for p, _, log in children:
            if p.wait(timeout=max(1, deadline - time.monotonic())):
                raise ValueError(f"tiles child failed: {log}")
    finally:
        for p, h, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            h.close()
    return [analyze(log.read_text(), worker) for worker, (_, _, log) in enumerate(children)]


def dry_run(root, plan, out):
    seal(root, plan)
    env = environment(root)

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
    listing([*test_args(root, TEST), "--test-threads=1"], TEST, "tiles.list")
    (out / "dry-run.json").write_text(json.dumps({"cpu_tests": len(CPU_TESTS), "gpu_commands_listed": [TEST],
                                                  "gpu_launched": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare", nargs=2, metavar=("READY", "CHECKPOINT"),
                        help="write the captured data into --out")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare[0], args.prepare[1], args.out)
        return

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
    verdict = {"phase": "tiles", "exit_code": 1, "power_before": gpus(visible, True)}
    error = None
    try:
        parsed = run(args.root, plan, args.out)
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
        verdict["host_xids"], verdict["xids"] = own_xids(kernel.stdout.splitlines(), bus_ids(visible))
        if kernel.returncode or verdict["xids"]:
            verdict["exit_code"] = 1
        verdict["elapsed_seconds"] = time.time() - started
        (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["exit_code"]:
        raise SystemExit(error or "power, journal or Xid check failed")
    print("router tile pair passed on both workers")


if __name__ == "__main__":
    main()
