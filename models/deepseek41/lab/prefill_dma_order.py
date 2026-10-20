"""Validate ordered DMA edges by peer source and destination, across all ranks."""
import sqlite3
from pathlib import Path

SUM = 'chop_comm_owned_sum_round_f32_v1'
ROWS = [8192] * 41 + [r for n in range(2496, 512, -128) for r in [n, n]]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(path, expected_chunks=4):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        sums = db.execute('SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                          'JOIN StringIds s ON s.id=k.shortName WHERE s.value=? ORDER BY k.start', (SUM,)).fetchall()
        copies = db.execute('SELECT start,end,deviceId,srcDeviceId,dstDeviceId,bytes '
                            'FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=(SELECT id '
                            "FROM ENUM_CUDA_MEMCPY_OPER WHERE name='CUDA_MEMCPY_KIND_PTOP') ORDER BY start,end").fetchall()
    count = expected_chunks * len(ROWS)
    require(len(sums) == 4 * count and len(copies) == 4 * count * 6, 'wrong DMA trace counts')
    require(all(c[2] == c[3] and c[3] != c[4] for c in copies), 'peer activity device is not its source')
    by_rank = {rank: [k for k in sums if k[2] == rank] for rank in range(4)}
    incoming = {rank: [c for c in copies if c[4] == rank] for rank in range(4)}
    outgoing = {rank: [c for c in copies if c[3] == rank] for rank in range(4)}
    for rank in range(4):
        require(len(by_rank[rank]) == count and len(incoming[rank]) == 6 * count
                and len(outgoing[rank]) == 6 * count, 'wrong per-rank DMA counts')
    dependencies, global_gather = [], []
    legal_overlaps = 0
    for index in range(count):
        rows = ROWS[index % len(ROWS)]
        expected = [rows // 4 * 5120 * 4] * 3 + [rows // 4 * 5120 * 2] * 3
        for rank in range(4):
            kernel = by_rank[rank][index]
            rx = incoming[rank][6 * index:6 * index + 6]
            tx = outgoing[rank][6 * index:6 * index + 6]
            require([c[5] for c in rx] == expected and [c[5] for c in tx] == expected,
                    'peer bytes differ from the cone interval')
            require([c[3] for c in rx] == [(rank + 3 - phase) % 4 for phase in range(3)] * 2
                    and [c[4] for c in tx] == [(rank + 1 + phase) % 4 for phase in range(3)] * 2,
                    'peer phases differ from the direct schedule')
            scatter_end = max(c[1] for c in rx[:3])
            gather_start = min(c[0] for c in tx[3:])
            before, after = kernel[0] - scatter_end, gather_start - kernel[1]
            require(before >= 0, f'incoming scatter overlaps owner sum: rank={rank} index={index} gap_ns={before}')
            require(after >= 0, f'outgoing gather overlaps owner sum: rank={rank} index={index} gap_ns={after}')
            dependencies.append(dict(rank=rank, chunk=index // len(ROWS), index=index % len(ROWS), rows=rows,
                                     sum_start_ns=kernel[0], sum_end_ns=kernel[1], last_incoming_scatter_end_ns=scatter_end,
                                     first_outgoing_gather_start_ns=gather_start,
                                     scatter_to_sum_gap_ns=before, sum_to_gather_gap_ns=after))
            legal_overlaps += max(c[1] for c in tx[:3]) > kernel[0]
        # The implementation joins all owner sums before the first gather phase.
        gap = min(outgoing[r][6 * index + p][0] for r in range(4) for p in [3, 4, 5]) - max(by_rank[r][index][1] for r in range(4))
        require(gap >= 0, f'gather precedes global sum join: index={index} gap_ns={gap}')
        global_gather.append(gap)
    return dict(sums_checked=len(sums), peer_copies_checked=len(copies), chunks=expected_chunks, ranks=4,
                ordering_failures=[], min_incoming_scatter_to_sum_gap_ns=min(r['scatter_to_sum_gap_ns'] for r in dependencies),
                min_sum_to_outgoing_gather_gap_ns=min(r['sum_to_gather_gap_ns'] for r in dependencies),
                min_all_sums_to_any_gather_gap_ns=min(global_gather),
                valid_outgoing_scatter_overlap_count=legal_overlaps, dependencies=dependencies)
