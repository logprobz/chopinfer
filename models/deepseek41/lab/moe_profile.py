#!/usr/bin/env python3
"""CPU cubin audit and bounded Nsight launches for routed MoE candidates."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess

SECTIONS = ["LaunchStats", "Occupancy", "MemoryWorkloadAnalysis", "SchedulerStats", "WarpStateStats"]
ROWS = [1, 6, 8, 32, 48]


def command(binary, ncu, bank, rows, output):
    args = [str(ncu), "--target-processes", "all", "--kernel-name-base", "function",
            "--kernel-name", "regex:.*before_(gateup|down)_.*", "--cache-control", "none",
            "--clock-control", "none", "--export", str(output), "--force-overwrite"]
    for section in SECTIONS:
        args += ["--section", section]
    args += [str(binary), "moe_active::profile_moe_gpu", "--exact", "--nocapture", "--test-threads=1"]
    env = os.environ.copy()
    env["CHOPINFER_MOE_PROFILE_BANK"] = str(bank)
    env["CHOPINFER_MOE_PROFILE_ROWS"] = str(rows)
    return args, env


def check_profile(report, bank, rows):
    experts, top_k = [(384, 6), (128, 3)][bank]
    old_tile, old_capacity = (16, 32) if rows <= 32 else (64, 8192)
    active_capacity = 32 if rows <= 32 else 192
    stream_capacity = 8 if rows <= 8 else 64
    for kind, n, k, clamp in [("gateup", 1152, 5120, "_clamp4120000041200000"), ("down", 5120, 576, "")]:
        prefix = f"before_{kind}_e{experts}k{top_k}_n{n}_k{k}_r"
        suffixes = [f"{old_tile}_t{old_capacity}{clamp}_v1",
                    f"16_t{active_capacity}{clamp}_active_bw4_v1"]
        suffixes += [f"64_t{stream_capacity}{clamp}_stream_{stage}_v1" for stage in
                     (["pack", "small", "merge"] if rows <= 8 else ["pack", "small", "wide", "merge"])]
        for suffix in suffixes:
            assert prefix + suffix in report, prefix + suffix


def static(lock_dir, cubins, cuobjdump, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for mode, family, cubin in zip(["old", "block_warps", "streamed"],
                                  ["routed_moe", "routed_moe_active", "routed_moe_streamed"], cubins):
        records_in_lock = json.loads((lock_dir / (family + ".lock")).read_text())
        resource = next(c["resources"] for c in records_in_lock if c["name"] == cubin.stem)
        sass = subprocess.check_output([str(cuobjdump), "-sass", str(cubin)], text=True)
        (output / (mode + ".sass")).write_text(sass)
        functions = re.split(r"Function : (\S+)", sass)
        for i in range(1, len(functions), 2):
            name, body = functions[i:i+2]
            if "e384k6" not in name or not ("_n1152_k5120_" in name or "_n5120_k576_" in name):
                continue
            instructions = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)", body)
            histogram = dict(collections.Counter(instructions))
            r = next(r for r in resource if r["symbol"] == name)
            assert r["local_bytes"] == 0 and r["stack_bytes"] == 0, name
            records.append(dict(mode=mode, **r, instructions=histogram))
    (output / "resources.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["Static compiler resources, not achieved occupancy or device timing.",
             "mode | projection/tile/capacity | registers/thread | shared bytes | LDG.128 sites | local/stack bytes"]
    for r in records:
        name = r["symbol"].partition("before_")[2]
        wide = sum(n for op, n in r["instructions"].items() if op.startswith("LDG") and ".128" in op)
        lines.append(f'{r["mode"]} | {name} | {r["registers"]} | {r["shared_static_bytes"]} | {wide} | 0/0')
    lines += ["", "CTA counts for deterministic mixed routing; counts are analytical, not hardware measurements.",
              "rows | active experts | old useful gate/down | old launched gate/down | block-warps launched gate/down | streamed small work gate/down | streamed wide work gate/down"]
    for rows in ROWS:
        counts = collections.Counter(((0 if r % 3 == 0 else r * 37 % 384) + k) % 384 for r in range(rows) for k in range(6))
        tile = 16 if rows <= 32 else 64
        useful = sum((c + tile - 1) // tile for c in counts.values())
        padded = (rows * 6 + 384 * (tile - 1) + tile - 1) // tile
        small = sum(c <= 8 for c in counts.values())
        wide = sum(c > 8 for c in counts.values())
        lines.append(f'{rows} | {len(counts)} | {useful*9}/{useful*80} | {padded*9}/{padded*80} | '
                     f'{min(rows*6*36,376)}/{min(rows*6*320,376)} | {small*9*40}/{small*80*5} | {wide*9*40}/{wide*80*5}')
    (output / "static-table.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--cubins", type=Path, nargs=3)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--bank", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.static:
        static(args.lock_dir, args.cubins, args.cuobjdump, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for rows in ROWS:
        output = args.out / f"bank{args.bank}-rows{rows}"
        cmd, env = command(args.binary, args.ncu, args.bank, rows, output)
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=150)
        assert output.with_suffix(".ncu-rep").is_file(), output
        report = subprocess.check_output([str(args.ncu), "--import", str(output.with_suffix(".ncu-rep")),
                                          "--page", "raw", "--csv"], text=True)
        output.with_suffix(".csv").write_text(report)
        check_profile(report, args.bank, rows)
    print(f"profiled bank {args.bank}, rows {ROWS}")


if __name__ == "__main__":
    main()
