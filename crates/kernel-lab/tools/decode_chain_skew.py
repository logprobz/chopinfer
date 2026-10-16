#!/usr/bin/env python3
"""Split in-graph collective time into rank skew and protocol cost.

Reads the SQLite export written by tools/profile/decode_chain_nsys.sh and
aligns the four ranks' graph replays on the common Nsight timeline. For every
comm node it reports, per replay and as medians:

* start skew: how far each rank's graph started after the earliest one;
* arrival skew: how long each rank's comm kernel waited for the last rank's
  preceding kernel to end (the slowest rank's predecessor end minus its own);
* protocol: the comm kernel's end minus the last rank's predecessor end.

Usage: decode_chain_skew.py <profile.sqlite> [nodes-per-replay, default 14]
"""

import sqlite3
import statistics
import sys


def replays(connection, nodes):
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    rows = connection.execute(
        "SELECT deviceId, graphId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphNodeId IS NOT NULL ORDER BY start"
    ).fetchall()
    by_device = {}
    for device, graph, start, end, name in rows:
        by_device.setdefault(device, {}).setdefault(graph, []).append((start, end, names[name]))
    # The timed graph is the one each device replayed most often.
    kernels = {device: max(graphs.values(), key=len) for device, graphs in by_device.items()}
    devices = sorted(kernels)
    count = min(len(kernels[device]) // nodes for device in devices)
    return devices, kernels, count


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    nodes = int(sys.argv[2]) if len(sys.argv) == 3 else 14
    devices, kernels, count = replays(sqlite3.connect(sys.argv[1]), nodes)
    first = kernels[devices[0]]
    comm = [index for index in range(nodes) if "comm" in first[index][2]]
    summary = {index: ([], []) for index in comm}
    for replay in range(count):
        base = min(kernels[device][replay * nodes][0] for device in devices)
        starts = [(kernels[device][replay * nodes][0] - base) / 1000 for device in devices]
        print(f"replay {replay}: start skew us {' '.join(f'{value:.1f}' for value in starts)}")
        for index in comm:
            at = replay * nodes + index
            before = [kernels[device][at - 1][1] for device in devices]
            ends = [kernels[device][at][1] for device in devices]
            last = max(before)
            waits = [(last - value) / 1000 for value in before]
            protocol = [(end - last) / 1000 for end in ends]
            summary[index][0].append(max(waits))
            summary[index][1].append(max(protocol))
            print(
                f"  node {index} {first[index][2]}: arrival skew us "
                f"{' '.join(f'{value:.1f}' for value in waits)}; protocol us "
                f"{' '.join(f'{value:.1f}' for value in protocol)}"
            )
    for index in comm:
        waits, protocol = summary[index]
        print(
            f"node {index} median over {count} replays: largest arrival skew "
            f"{statistics.median(waits):.1f} us, protocol after last arrival "
            f"{statistics.median(protocol):.1f} us"
        )


if __name__ == "__main__":
    main()
