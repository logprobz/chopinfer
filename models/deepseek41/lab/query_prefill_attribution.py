"""Attribute recorded row-family launches and their adjacent tensor consumers."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3


def family(symbol):
    if "sinkhorn" in symbol or "_hc_" in symbol:
        return "HC"
    if "moe_ceil_w3456" in symbol or "swiglu" in symbol:
        return None
    if any(s in symbol for s in ("prefill_e384_", "moe_chain_mxfp4", "route_")):
        return None
    if "mxfp8_fold_" in symbol:
        shape = re.search(r"_n(\d+)_k(\d+)", symbol)
        if (not shape or "_admit_" not in symbol
                or tuple(map(int, shape.groups())) in
                {(25600, 6144), (576, 5120), (5120, 576), (4096, 1280)}):
            return None
        return "Norm and quantize"
    if any(s in symbol for s in ("engram_", "mxfp8_decode", "ceil_w6144")):
        return None
    if any(s in symbol for s in ("ceil", "rmsnorm", "q_norm", "rounding_widen", "_admit_")):
        return "Norm and quantize"
    return None


def kernels(path):
    if path.suffix == ".csv":
        with path.open() as source:
            records = list(csv.DictReader(source))
        for row in records:
            for key in ("deviceId", "start", "end", "gridX", "gridY", "blockX"):
                row[key] = int(row[key])
            row["exclusive_ms"] = float(row["exclusive_ms"])
        return records
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("PREFILL_PROFILE_INVALID: no kernel activity")
        return [dict(row) for row in connection.execute(
            "select k.*,s.value as symbol from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id=k.demangledName order by start,end")]


def analyze(path, gpu, window=None):
    rows = sorted((r for r in kernels(path) if r["deviceId"] == gpu), key=lambda r:(r["start"],r["end"]))
    if window is not None:
        start, end = window
        if any(r["start"] < start < r["end"] or r["start"] < end < r["end"] for r in rows):
            raise ValueError("PREFILL_PROFILE_INVALID: window cuts a kernel")
        rows = [r for r in rows if start <= r["start"] < end]
    rows = [r for r in rows if not r["symbol"].startswith("chop_sampling_")]
    if not rows:
        raise ValueError("PREFILL_PROFILE_INVALID: empty GPU capture")
    if not any(r["gridX"] == 8192 and (
            "q_norm_f32" in r["symbol"] or "prefill_query_norm_ceil" in r["symbol"])
            for r in rows):
        raise ValueError("PREFILL_PROFILE_INVALID: no full 8192-row query boundary")
    frontier = rows[0]["start"]
    groups = defaultdict(lambda: {"count": 0, "exclusive_ms": 0., "raw_ms": 0., "shapes": set()})
    pairs = defaultdict(lambda: {"count": 0, "exclusive_ms": 0.})
    for i, row in enumerate(rows):
        exclusive = max(0, row["end"]-max(row["start"], frontier))/1e6
        frontier = max(frontier, row["end"])
        symbol = row["symbol"]
        kind = row.get("family", family(symbol))
        if kind not in ("HC", "Norm and quantize"):
            continue
        if kind is None:
            continue
        record = groups[(kind, symbol)]
        record["count"] += 1
        record["exclusive_ms"] += exclusive
        record["raw_ms"] += (row["end"]-row["start"])/1e6
        record["shapes"].add((row["gridX"], row["gridY"], row["blockX"]))
        # These are launch neighbors, not inferred dataflow edges. The source
        # inventory establishes which neighbors share a tensor.
        pair = (rows[i-1]["symbol"] if i else "", symbol,
                rows[i+1]["symbol"] if i+1 < len(rows) else "",
                row["gridX"], row["gridY"], row["blockX"])
        pairs[pair]["count"] += 1
        pairs[pair]["exclusive_ms"] += exclusive
    return {
        "source": str(path), "gpu": gpu, "kernel_count": len(rows),
        "symbols": [dict(family=k[0], symbol=k[1], **{**v, "shapes": sorted(v["shapes"])})
                    for k,v in sorted(groups.items(), key=lambda kv:-kv[1]["exclusive_ms"])],
        "neighbors": [dict(producer=k[0], row_op=k[1], consumer=k[2],
                           grid=[k[3], k[4]], threads=k[5], **v) for k,v in pairs.items()],
        "scope": "Exclusive frontier attribution; adjacency requires source-level dataflow confirmation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, nargs=2)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.profile, args.gpu, args.window), indent=2) + "\n")


if __name__ == "__main__":
    main()
