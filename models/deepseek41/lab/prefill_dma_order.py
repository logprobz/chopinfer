"""Validate ordered DMA edges by peer source and destination, across all ranks."""
import sqlite3
from pathlib import Path

SUM = 'chop_comm_owned_sum_round_f32_v1'
BF16_SUM = 'chop_comm_owned_sum_bf16_v1'
BF16_ROUND = 'chop_comm_owned_round_bf16_v1'
SUMS = {'f32': SUM, 'bf16': BF16_SUM}
# Bytes per scattered partial element; the gather always carries BF16 output.
SCATTER_BYTES = {'f32': 4, 'bf16': 2}
ROWS = [8192] * 41 + [r for n in range(2496, 512, -128) for r in [n, n]]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pieces(rank, ranks=4):
    """Intervals a rank rounds for sending: all rows outside its own."""
    return 1 if rank in (0, ranks - 1) else 2


def audit(path, expected_chunks=4, payload='f32', split=False):
    """With `split`, a BF16 call may instead round each of its three sent
    intervals on its own, each before its own scatter reads it (a producer
    run per interval); the call's rounds are then those between the
    previous owner sum and its own."""
    require(payload in SUMS, 'unknown DMA payload')
    require(not split or payload == 'bf16', 'split rounding needs the BF16 payload')
    def kernels(db, name):
        return db.execute('SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                          'JOIN StringIds s ON s.id=k.shortName WHERE s.value=? ORDER BY k.start', (name,)).fetchall()
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        sums = kernels(db, SUMS[payload])
        used = {SUMS[payload]} | ({BF16_ROUND} if payload == 'bf16' else set())
        require(not any(kernels(db, n) for n in {SUM, BF16_SUM, BF16_ROUND} - used), 'another DMA payload executed')
        rounds = kernels(db, BF16_ROUND) if payload == 'bf16' else []
        copies = db.execute('SELECT start,end,deviceId,srcDeviceId,dstDeviceId,bytes '
                            'FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=(SELECT id '
                            "FROM ENUM_CUDA_MEMCPY_OPER WHERE name='CUDA_MEMCPY_KIND_PTOP') ORDER BY start,end").fetchall()
    count = expected_chunks * len(ROWS)
    require(len(sums) == 4 * count and len(copies) == 4 * count * 6, 'wrong DMA trace counts')
    rounded = {rank: [k for k in rounds if k[2] == rank] for rank in range(4)}
    if payload == 'bf16' and not split:
        require(all(len(rounded[r]) == pieces(r) * count for r in range(4)), 'wrong BF16 rounding counts')
    rounding_gaps = []
    split_calls = 0
    assigned = {rank: 0 for rank in range(4)}
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
        expected = [rows // 4 * 5120 * SCATTER_BYTES[payload]] * 3 + [rows // 4 * 5120 * 2] * 3
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
            if payload == 'bf16':
                # The rows a rank sends are rounded before its first scatter
                # reads them, and after its previous scatters finished reading.
                if split:
                    after_previous = by_rank[rank][index - 1][1] if index else float('-inf')
                    own = [k for k in rounded[rank] if after_previous <= k[0] < kernel[0]]
                    require(len(own) in (pieces(rank), 3), f'wrong BF16 rounding counts: rank={rank} index={index}')
                    assigned[rank] += len(own)
                else:
                    own = rounded[rank][pieces(rank) * index:pieces(rank) * (index + 1)]
                if split and len(own) == 3:
                    # One rounding per sent interval, in scatter phase order.
                    gap = min(tx[phase][0] - own[phase][1] for phase in range(3))
                    split_calls += 1
                else:
                    gap = min(c[0] for c in tx[:3]) - max(k[1] for k in own)
                require(gap >= 0, f'scatter reads unrounded rows: rank={rank} index={index} gap_ns={gap}')
                if index:
                    previous = max(c[1] for c in outgoing[rank][6 * index - 6:6 * index - 3])
                    reuse = min(k[0] for k in own) - previous
                    require(reuse >= 0, f'rounding overwrites rows still scattering: rank={rank} index={index} gap_ns={reuse}')
                rounding_gaps.append(gap)
        # The implementation joins all owner sums before the first gather phase.
        gap = min(outgoing[r][6 * index + p][0] for r in range(4) for p in [3, 4, 5]) - max(by_rank[r][index][1] for r in range(4))
        require(gap >= 0, f'gather precedes global sum join: index={index} gap_ns={gap}')
        global_gather.append(gap)
    require(not split or all(assigned[r] == len(rounded[r]) for r in range(4)), 'BF16 rounding outside any call')
    return dict(payload=payload, sums_checked=len(sums), rounds_checked=len(rounds), split_calls=split_calls,
                min_round_to_scatter_gap_ns=min(rounding_gaps) if rounding_gaps else None,
                peer_copies_checked=len(copies), chunks=expected_chunks, ranks=4,
                ordering_failures=[], min_incoming_scatter_to_sum_gap_ns=min(r['scatter_to_sum_gap_ns'] for r in dependencies),
                min_sum_to_outgoing_gather_gap_ns=min(r['sum_to_gather_gap_ns'] for r in dependencies),
                min_all_sums_to_any_gather_gap_ns=min(global_gather),
                valid_outgoing_scatter_overlap_count=legal_overlaps, dependencies=dependencies)
