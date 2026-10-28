#!/usr/bin/env python3
"""Attribute bounded DeepSeek steps without treating idle time as kernel work.

Read an Nsight SQLite export, step.v1 JSONL and bracketed host clock anchors.
Keep every rank separate. Bytes inferred from shapes are not DRAM counters.
"""

import argparse
import bisect
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import re
import sqlite3
import struct


def family(name):
    if name.startswith("chop_comm_") or "prefill_dma_sum" in name:
        return "collective"
    if "dspark" in name or "draft_" in name:
        return "drafter"
    if "engram_" in name or "ceil_w6144" in name:
        return "engram"
    if "_hc_" in name or "sinkhorn" in name:
        return "hyperconnection"
    if "head_f32" in name or "_head_m" in name:
        return "logits"
    if "shared_swiglu" in name:
        return "moe_shared"
    if "scale_tiles_expand" in name:
        return "moe_scale_expand"
    if "moe" in name or "_route_" in name or "prefill_e384" in name:
        if "gateup" in name or "gate_up" in name:
            return "moe_gate_up"
        if "down" in name:
            return "moe_down"
        if "route" in name:
            return "moe_router"
        return "moe_glue"
    if "compressor" in name or "compressed_key" in name:
        return "compressor"
    if "position_blocks" in name:
        return "indexer_topk" if "select" in name else "indexer_score"
    if "position_query" in name:
        return "indexer_query"
    if any(word in name for word in ("indexer", "index_query", "index_key", "candidate_block_ids", "topk")):
        return "indexer"
    if "latent_sources" in name:
        return "attention_mixed_sources" if "_mixed_" in name else "attention_window"
    if any(word in name for word in ("latent_", "window_", "rope_")):
        return "attention_glue"
    shape = re.search(r"_n(\d+)_k(\d+)", name)
    if shape:
        n, k = map(int, shape.groups())
        if (n, k) in ((576, 5120), (5120, 576)):
            return "moe_shared"
        if (n, k) == (25600, 6144):
            return "engram"
        if "dense_f32" in name:
            return "moe_router" if n == 384 else "compressor" if n == 512 else "indexer"
        if (n, k) == (4096, 1280):
            return "indexer"
        return "dense_projection"
    if "chop_fold_local_" in name:
        return "dense_projection"
    if any(word in name for word in ("ceil", "norm", "rounding")):
        return "norm_rounding"
    if "sampling_" in name:
        return "sampling"
    if any(word in name for word in ("row_embedding", "row_copy", "row_gather", "gather_rows", "state_", "token_")):
        return "copy_state"
    return "unclassified"


def exclusive(kernels, start, end):
    """Partition the window into kernel union and gaps, including PDL overlap.

    First-start ownership is an attribution convention, not causal speedup.
    Raw symbol durations remain available for rate calculations.
    """
    if not start < end:
        raise ValueError("empty time window")
    charged, raw, counts = defaultdict(int), defaultdict(int), defaultdict(int)
    frontier = start
    for row in sorted(kernels, key=lambda r: (r["start"], r["end"], r["name"])):
        a, b = max(start, row["start"]), min(end, row["end"])
        if a >= b:
            continue
        name = row["name"]
        raw[name] += b - a
        counts[name] += 1
        charged[family(name)] += max(0, b - max(a, frontier))
        frontier = max(frontier, b)
    busy = sum(charged.values())
    return {"window_ns": end - start, "kernel_union_ns": busy,
            "no_kernel_ns": end - start - busy, "exclusive_family_ns": dict(charged),
            "raw_symbol_ns": dict(raw), "launches": dict(counts)}


def clock_epoch(metadata, anchors):
    if len(anchors) < 4:
        raise ValueError("at least four bracketed clock anchors are required")
    low = min(a["unix_before_ns"] - a["monotonic_ns"] for a in anchors)
    high = max(a["unix_after_ns"] - a["monotonic_ns"] for a in anchors)
    if not 0 <= high - low <= 10_000_000:
        raise ValueError("clock bracket is invalid or wider than 10 ms")
    return metadata["systemClockNs"] + (low + high) // 2, high - low


def phase_widths(step):
    """Serving executes these phases separately; their widths must not be added."""
    return {key: step.get(key, 0) for key in ("prefill_tokens", "decode_rows", "verify_rows")}


def cohort_readiness(progress, count, outputs, minimum_remaining=64):
    """Require every decoder to have emitted, with room for request admission.

    This client observation is a precondition. The engine trace must still
    prove that the requested concurrency survived into the mixed step.
    """
    if len(progress) != count or count < 1:
        raise ValueError("not every decoder emitted a token")
    remaining = []
    for row in progress:
        emitted = row["emitted"]
        if not 1 <= emitted <= outputs or row.get("finished", False):
            raise ValueError("decoder drained before cohort readiness")
        remaining.append(outputs - emitted)
    if min(remaining) < minimum_remaining:
        raise ValueError("decoder output budget leaves too little admission margin")
    return {"minimum_remaining_tokens": min(remaining),
            "required_remaining_tokens": minimum_remaining, "decoders": count}


def moe_payload(rows, distinct_experts=None):
    """TP4 Flash's logical selected weight footprint and arithmetic, per layer.

    This excludes activation traffic, tile rereads and scale expansion traffic.
    An absent expert census stays absent; the uniform expectation is a model.
    """
    if rows < 1 or distinct_experts is not None and not 1 <= distinct_experts <= min(384, rows * 6):
        raise ValueError("invalid row or distinct expert count")
    return {"rows": rows, "distinct_experts": distinct_experts,
            "uniform_expected_experts": 384 * (1 - (1 - 6 / 384) ** rows),
            "weight_bytes": None if distinct_experts is None else 4_700_160 * distinct_experts,
            "weight_bytes_upper_bound": 4_700_160 * min(384, rows * 6),
            "flops": 6 * rows * 6 * 5120 * 576,
            "traffic_kind": "logical payload, not measured DRAM traffic"}


def expert_census(directory, decode_rows, prefill_rows=512):
    """Read existing untimed router taps for a single final-logits prefill.

    Taps execute full layers. Restrict the upper layers to the final-logits
    dependency cone before estimating the served phase's expert union.
    """
    directory = Path(directory)
    if prefill_rows != 512 or not 0 <= decode_rows <= 64:
        raise ValueError("census supports a 512-row prefill and at most 64 target decode/verify rows")
    if json.loads((directory / "completion.json").read_text())["completion"] != "full":
        raise ValueError("incomplete router capture")
    entries = [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]
    by_layer = {}
    for entry in entries:
        match = re.fullmatch(r"L(\d{2})\.router\.ids", entry["name"])
        if not match:
            continue
        layer = int(match[1])
        if (layer in by_layer or not 0 <= layer < 40 or entry["rank"] != 0
                or entry["dtype"] != "u32" or entry["cols"] != 6
                or entry["rows"] != 512 or entry["first_row"] != 0):
            raise ValueError("unexpected router tap shape, rank or layer")
        data = (directory / (entry["name"] + ".r0.bin")).read_bytes()
        if len(data) != 512 * 6 * 4:
            raise ValueError("router tensor byte length differs")
        ids = struct.unpack("<3072I", data)
        if max(ids) >= 384:
            raise ValueError("router expert ID exceeds bank")
        if any(len(set(ids[first:first + 6])) != 6 for first in range(0, len(ids), 6)):
            raise ValueError("router top-k contains duplicate experts")
        live = {36: 448, 37: 320, 38: 192, 39: 64}.get(layer, 512)
        counts = [0] * 384
        for expert in ids[(512 - live) * 6:]:
            counts[expert] += 1
        union = sum(n > 0 for n in counts)
        by_layer[layer] = {"prefill_rows": live, "route_counts": counts,
                           "prefill_experts": union,
                           "mixed_union_bounds": [union, min(384, union + 6 * decode_rows)],
                           "separate_phase_weight_bytes_bounds": [
                               4_700_160 * (union + (6 if decode_rows else 0)),
                               4_700_160 * (union + min(384, 6 * decode_rows))],
                           "prefill_payload": moe_payload(live, union),
                           "mma_route_rows": sum(16 * ((n + 15) // 16) for n in counts),
                           "tensor_sha256": hashlib.sha256(data).hexdigest()}
    if set(by_layer) != set(range(40)):
        raise ValueError("expected all 40 target router layers")
    return {"layers": by_layer, "decode_or_verify_live_rows": decode_rows,
            "scope": "Untimed full prefill taps restricted to the served final-logits cone; decoder union bounded.",
            "limitation": "Separate target phases can fetch shared experts twice. A union is not measured DRAM traffic."}


def analyze(database, steps, anchors):
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        metadata = dict(db.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME").fetchone())
        epoch, uncertainty = clock_epoch(metadata, anchors)
        kernels = [dict(row) for row in db.execute(
            "SELECT k.start,k.end,k.deviceId,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName ORDER BY k.start,k.end")]
    finally:
        db.close()
    by_rank = {rank: [row for row in kernels if row["deviceId"] == rank] for rank in range(4)}
    if any(not rows for rows in by_rank.values()):
        raise ValueError("missing rank kernel activity")
    indices = {}
    for rank, rows in by_rank.items():
        frontier = 0
        ends = []
        for row in rows:
            frontier = max(frontier, row["end"])
            ends.append(frontier)
        indices[rank] = ([row["start"] for row in rows], ends)
    records = []
    for step in steps:
        end = step["completed_unix_ns"] - epoch
        start = end - step["execution_us"] * 1000
        if any(start < rows[0]["start"] or end > indices[rank][1][-1]
               for rank, rows in by_rank.items()):
            continue
        ranks = {}
        for rank, rows in by_rank.items():
            starts, ends = indices[rank]
            first = bisect.bisect_right(ends, start)
            last = bisect.bisect_left(starts, end)
            ranks[str(rank)] = exclusive(rows[first:last], start, end)
        records.append({"step": step, "phase_widths": phase_widths(step),
                        "ranks": ranks})
    if not records:
        raise ValueError("no complete step windows inside the four-rank capture")
    return {"schema": "deepseek.mixed-profile.v1", "clock_bracket_ns": uncertainty,
            "steps": records, "dram_bandwidth": None,
            "limitations": ["Exclusive ownership is a convention for overlapping kernels.",
                            "Host execution windows include launch gaps and synchronization.",
                            "Nsight Systems durations do not measure DRAM bytes or distinct experts."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = []
    for line in args.steps.read_text().splitlines():
        if '"schema":"chopinfer.step.v1"' in line or '"schema": "chopinfer.step.v1"' in line:
            steps.append(json.loads(line[line.index("{"):]))
    result = analyze(args.sqlite, steps, json.loads(args.anchors.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
