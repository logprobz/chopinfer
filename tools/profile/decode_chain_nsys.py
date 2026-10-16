#!/usr/bin/env python3
"""Summarize timed decode-chain CUDA graph nodes from an Nsight SQLite export."""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


def median_us(values):
    return round(statistics.median(values) / 1000, 3)


def summarize(connection, ordinals):
    devices = {
        device_id: ordinal
        for device_id, ordinal in connection.execute(
            "SELECT id, cuDevice FROM TARGET_INFO_GPU"
        )
    }
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    rows = connection.execute(
        "SELECT deviceId, graphId, graphNodeId, start, end, shortName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL "
        "AND graphNodeId IS NOT NULL ORDER BY deviceId, graphId, start, graphNodeId"
    ).fetchall()
    by_device = {}
    for device, graph, node, start, end, name in rows:
        by_device.setdefault(device, {}).setdefault(graph, []).append(
            (node, start, end, names[name])
        )
    result = []
    timed_runs = {}
    for rank, ordinal in enumerate(ordinals):
        matching = [device for device, number in devices.items() if number == ordinal]
        if len(matching) != 1 or matching[0] not in by_device:
            raise ValueError(f"missing graph kernels on GPU {ordinal}")
        device = matching[0]
        graphs = by_device[device]
        graph = max(graphs, key=lambda key: len(set(row[0] for row in graphs[key])))
        events = graphs[graph]
        node_order = []
        for node, *_ in events:
            if node in node_order:
                break
            node_order.append(node)
        width = len(node_order)
        if width < 2 or len(events) != width * 6:
            raise ValueError(
                f"GPU {ordinal}: expected six complete graph launches, "
                f"found {len(events)} events with {width} nodes"
            )
        launches = [events[index : index + width] for index in range(0, len(events), width)]
        if any([event[0] for event in launch] != node_order for launch in launches):
            raise ValueError(f"GPU {ordinal}: graph node order changed across replays")
        if any([event[3] for event in launch] != [event[3] for event in launches[0]]
               for launch in launches):
            raise ValueError(f"GPU {ordinal}: graph kernel names changed across replays")
        # decode_chain.rs runs one validation replay, one timing warmup,
        # three timed rounds, then one final validation replay.
        timed = launches[2:5]
        timed_runs[rank] = timed
        stages = []
        for index in range(width):
            durations = [launch[index][2] - launch[index][1] for launch in timed]
            gaps = [
                launch[index][1] - launch[index - 1][2] if index else 0
                for launch in timed
            ]
            tails = [
                max(0, launch[index][2] - launch[index - 1][2]) if index
                else launch[index][2] - launch[index][1]
                for launch in timed
            ]
            stages.append(
                {
                    "index": index,
                    "kernel": timed[0][index][3],
                    "duration_us": median_us(durations),
                    "gap_before_us": median_us(gaps),
                    "tail_after_previous_us": median_us(tails),
                }
            )
        result.append(
            {
                "rank": rank,
                "gpu_ordinal": ordinal,
                "graph_id": graph,
                "trace_graph_us": median_us(
                    [max(event[2] for event in launch) - min(event[1] for event in launch)
                     for launch in timed]
                ),
                "stages": stages,
            }
        )
    width = len(result[0]["stages"])
    if any(len(rank["stages"]) != width for rank in result):
        raise ValueError("rank graph stage counts differ")
    critical = [
        max(max(event[2] for event in timed_runs[rank][sample]) for rank in timed_runs)
        - min(min(event[1] for event in timed_runs[rank][sample]) for rank in timed_runs)
        for sample in range(3)
    ]
    return {"trace_critical_path_us": median_us(critical), "ranks": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--layer", type=int, choices=(1, 5), required=True)
    shape = parser.add_mutually_exclusive_group(required=True)
    shape.add_argument("--batch", type=int, choices=(1, 8, 16, 32))
    shape.add_argument("--tokens", type=int, choices=(2048, 8192))
    parser.add_argument("--ordinals", default="0,1,2,3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ordinals = [int(value) for value in args.ordinals.split(",")]
    if not ordinals or len(set(ordinals)) != len(ordinals) or min(ordinals) < 0:
        parser.error("--ordinals requires distinct nonnegative GPU ordinals")
    with sqlite3.connect(args.sqlite) as connection:
        report = summarize(connection, ordinals)
    phase = "prefill" if args.tokens else "decode"
    report.update(
        {"layer": args.layer, "phase": phase, "source": args.sqlite.name,
         "timing_environment": "shared-device", "timed_replays": 3}
    )
    report["tokens" if args.tokens else "batch"] = args.tokens or args.batch
    print(f"layer {args.layer} {phase} {args.tokens or args.batch}: traced graph critical path "
          f"{report['trace_critical_path_us']:.3f} us (shared-device)")
    for rank in report["ranks"]:
        print(f"rank {rank['rank']} GPU {rank['gpu_ordinal']}: "
              f"graph {rank['trace_graph_us']:.3f} us")
        print("  #  duration us  gap us  tail us  kernel")
        for stage in rank["stages"]:
            print(f"  {stage['index']:2d} {stage['duration_us']:11.3f} "
                  f"{stage['gap_before_us']:7.3f} "
                  f"{stage['tail_after_previous_us']:8.3f}  {stage['kernel']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
