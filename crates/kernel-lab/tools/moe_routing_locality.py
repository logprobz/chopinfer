#!/usr/bin/env python3
"""Expert locality of real MiMo routes for speculative verify batches.

Reads the router choices that `mimo_model_audit.rs::gpu_hidden_states`
writes (`<stream>/layerNN.ids`, int32 [tokens, 8]) and samples verify-like
batches: `sequences` distinct streams with `rows` consecutive tokens each.
For every layer it records the distinct experts of the batch and how many
routed rows the busiest expert holds, next to the independent-token
expectation 256 * (1 - (1 - 8/256)^n). It also keeps a few whole-model
samples per batch shape as token-major top-8 expert IDs for the MoE decode
benchmark.

Usage: moe_routing_locality.py AUDIT_DIR OUTPUT_JSON
"""

import json
import random
import statistics
import sys
from pathlib import Path

EXPERTS = 256
TOP_K = 8
SAMPLES = 256
BENCH_SAMPLES = 4
SHAPES = [(1, 8), (1, 16), (1, 32), (2, 16), (4, 8), (16, 2), (32, 1),
          (8, 1), (16, 1),
          # Phase-2 verify buckets of the speculation contract (64 rows).
          (1, 64), (2, 32), (4, 16), (8, 8), (16, 4),
          # K7 verify steps at C16 and C32 (128 and 256 rows). Appended last,
          # so the samples of the shapes above are unchanged.
          (16, 8), (32, 8)]
# Shapes whose samples the decode benchmark replays.
BENCH_SHAPES = {(1, 8), (1, 16), (1, 32), (16, 2), (32, 1), (1, 64), (16, 4),
                (16, 8), (32, 8)}


def load(audit):
    manifest = json.loads((audit / "manifest.json").read_text())
    layers = range(1, manifest["layers"])
    # GSM8K streams share the fixture's instruction; blocks start after it
    # so that repeated text does not count as locality.
    ids = [s["ids"] for s in manifest["streams"] if s["name"].startswith("gsm8k")]
    shared = 0
    while ids and all(len(i) > shared and i[shared] == ids[0][shared] for i in ids):
        shared += 1
    starts = {s["name"]: shared if s["name"].startswith("gsm8k") else 0
              for s in manifest["streams"]}
    streams = {}
    for stream in manifest["streams"]:
        name = stream["name"]
        routes = {}
        for layer in layers:
            data = (audit / name / f"layer{layer:02}.ids").read_bytes()
            ids = [int.from_bytes(data[i:i + 4], "little", signed=True)
                   for i in range(0, len(data), 4)]
            routes[layer] = [ids[t * TOP_K:(t + 1) * TOP_K]
                             for t in range(len(ids) // TOP_K)]
        streams[name] = routes
    return manifest, list(layers), streams, starts, shared


def independent(tokens):
    return EXPERTS * (1 - (1 - TOP_K / EXPERTS) ** tokens)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main():
    audit, output = Path(sys.argv[1]), Path(sys.argv[2])
    manifest, layers, streams, starts, shared = load(audit)
    lengths = {name: len(routes[layers[0]]) for name, routes in streams.items()}
    rng = random.Random(0x5eed)
    shapes = []
    for sequences, rows in SHAPES:
        eligible = [name for name, length in lengths.items()
                    if length - starts[name] >= rows]
        picks = []
        for _ in range(SAMPLES):
            chosen = rng.sample(eligible, sequences)
            picks.append([(name, rng.randrange(starts[name], lengths[name] - rows + 1))
                          for name in chosen])
        distinct_all, busiest_all, shared_all = [], [], []
        per_layer = {}
        for layer in layers:
            distinct, busiest = [], []
            for pick in picks:
                counts = {}
                for name, start in pick:
                    for token in streams[name][layer][start:start + rows]:
                        for expert in token:
                            counts[expert] = counts.get(expert, 0) + 1
                distinct.append(len(counts))
                busiest.append(max(counts.values()))
                # Routed rows whose expert serves at least one other row.
                shared_all.append(sum(c for c in counts.values() if c > 1)
                                  / (sequences * rows * TOP_K))
            distinct_all += distinct
            busiest_all += busiest
            per_layer[str(layer)] = {
                "distinct_mean": round(statistics.fmean(distinct), 2),
                "busiest_rows_mean": round(statistics.fmean(busiest), 2),
            }
        tokens = sequences * rows
        bench = []
        for pick in picks[:BENCH_SAMPLES] if (sequences, rows) in BENCH_SHAPES else []:
            # Token-major top-8 expert IDs of each layer, one hex byte each.
            bench.append({
                "streams": [[name, start] for name, start in pick],
                "layers": {
                    str(layer): bytes(expert for name, start in pick
                                      for token in streams[name][layer][start:start + rows]
                                      for expert in token).hex()
                    for layer in layers
                },
            })
        shapes.append({
            "sequences": sequences,
            "rows_per_sequence": rows,
            "tokens": tokens,
            "independent_distinct": round(independent(tokens), 2),
            "distinct": {
                "mean": round(statistics.fmean(distinct_all), 2),
                "p10": percentile(distinct_all, 0.10),
                "p50": percentile(distinct_all, 0.50),
                "p90": percentile(distinct_all, 0.90),
                "min": min(distinct_all),
                "max": max(distinct_all),
            },
            "busiest_expert_rows": {
                "mean": round(statistics.fmean(busiest_all), 2),
                "p90": percentile(busiest_all, 0.90),
                "max": max(busiest_all),
            },
            "shared_route_fraction": round(statistics.fmean(shared_all), 4),
            "per_layer": per_layer,
            "bench_samples": bench,
        })
    result = {
        "schema": "chopinfer.mimo26.routing-locality.v1",
        "source": {
            "harness": "crates/kernel-lab/tests/mimo_model_audit.rs gpu_hidden_states",
            "streams": {name: lengths[name] for name in sorted(lengths)},
            "tokens": sum(lengths.values()),
            "layers": [layers[0], layers[-1]],
            "prefill_capacity": manifest["prefill_capacity"],
            "gsm8k_shared_prefix_tokens_skipped": shared,
            "note": "teacher texts, and GSM8K instruction, question and reference "
                    "solution as plain text; router choices of the GPU prefill path",
        },
        "samples_per_shape_and_layer": SAMPLES,
        "shapes": shapes,
    }
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    for shape in shapes:
        print(f"{shape['sequences']:>2} x {shape['rows_per_sequence']:>2}: distinct "
              f"{shape['distinct']['mean']:6.1f} (p10 {shape['distinct']['p10']}, "
              f"p90 {shape['distinct']['p90']}) vs independent "
              f"{shape['independent_distinct']:6.1f}; busiest expert "
              f"{shape['busiest_expert_rows']['mean']:.1f} rows; shared routes "
              f"{shape['shared_route_fraction']:.2f}")


if __name__ == "__main__":
    main()
