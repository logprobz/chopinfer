#!/usr/bin/env python3
"""Audit and attribute a cold Flash TP4 sparse prefill Nsight SQLite export.

The qualified workload has four 8192-row chunks and 73 phased or DMA reductions
per chunk. Attention variants are reported independently. Peer activity is charged
to its source device. Owner-sum dependencies are checked separately against
incoming scatters and outgoing gathers on all four ranks. Kernel interval
unions include PDL waits and do not establish useful compute overlap.
"""
import argparse
import collections
import json
from pathlib import Path
import sqlite3

import prefill_dma_order

EMBED = 'chop_row_embedding_deepseek41_tp4_v1'
SUM = prefill_dma_order.SUM
ROUND = 'chop_comm_allreduce_phased_round_f32_v1'
SCATTER = 'chop_comm_reduce_scatter_f32_v1'
REUSE = 'chop_latent_sources_deepseek41_h16_d512_mixed_normalized_s1_r64_prefill_p32_reuse2_v1'
HIDDEN = 5120
ROWS = prefill_dma_order.ROWS


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def duration(spans):
    return sum(end - start for start, end in merged(spans))


def exact(first, second):
    if json.dumps(first, sort_keys=True, allow_nan=False) != json.dumps(second, sort_keys=True, allow_nan=False):
        raise RuntimeError('served output bits differ')


def analyze(path, dma, expected_chunks=4):
    if dma:
        prefill_dma_order.audit(path, expected_chunks)
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        kernels = db.execute(
            'SELECT k.start,k.end,s.value,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=0 ORDER BY k.start,k.end,s.value'
        ).fetchall()
        copies = db.execute(
            "SELECT start,end,'peer_dma',streamId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            'WHERE srcDeviceId=0 AND copyKind=(SELECT id FROM ENUM_CUDA_MEMCPY_OPER '
            "WHERE name='CUDA_MEMCPY_KIND_PTOP') ORDER BY start,end"
        ).fetchall()
    kernels = [row for row in kernels if not row[2].startswith('chop_sampling_')]
    starts = [index for index, row in enumerate(kernels) if row[2] == EMBED]
    if len(starts) != expected_chunks:
        raise RuntimeError('wrong number of prefill chunks')
    starts.append(len(kernels))
    chunks = []
    for chunk_index, (first, last) in enumerate(zip(starts, starts[1:])):
        body = kernels[first:last]
        begin = body[0][0]
        end = kernels[last][0] if last < len(kernels) else max(row[1] for row in body)
        peer = [row for row in copies if begin <= row[0] < end]
        if any(row[1] > end for row in peer):
            raise RuntimeError('peer copy crosses a chunk join')
        counts = collections.Counter(row[2] for row in body)
        expected_small = 8 if chunk_index == expected_chunks - 1 else 6
        paired = sum(n for name, n in counts.items() if 'latent_sources' in name and name.endswith('_reuse2_v1'))
        if (counts[SUM] != (73 if dma else 0) or counts[ROUND] != (0 if dma else 73)
                or counts[SCATTER] != expected_small or len(peer) != (438 if dma else 0)):
            raise RuntimeError(f'wrong sparse route: sum={counts[SUM]} round={counts[ROUND]} '
                               f'scatter={counts[SCATTER]} reuse={paired} copies={len(peer)}')
        activities = sorted([*body, *(row[:4] for row in peer)])
        frontier = begin
        totals, attributed = collections.Counter(), {}
        for start, finish, name, stream in activities:
            exclusive = max(0, finish - max(start, frontier))
            frontier = max(frontier, finish)
            totals[name] += exclusive
            attributed[(start, finish, name, stream)] = exclusive
        compute = [(a, b) for a, b, name, _ in body if not name.startswith('chop_comm_')]
        comm = ([(a, b) for a, b, name, _ in body if name.startswith('chop_comm_')]
                + [(row[0], row[1]) for row in peer])
        intersection = duration(compute) + duration(comm) - duration(compute + comm)
        reduce = [row for row in body if row[2] == (SUM if dma else ROUND)]
        groups = collections.defaultdict(lambda: dict(calls=0, rows=0, activity_ms=0, envelope_ms=0,
                                                       exclusive_ms=0, peer_bytes=0))
        reductions = []
        for index, (kernel, rows) in enumerate(zip(reduce, ROWS)):
            part, payload = [kernel], 0
            if dma:
                transfers = peer[index * 6:index * 6 + 6]
                expected = [rows // 4 * HIDDEN * 4] * 3 + [rows // 4 * HIDDEN * 2] * 3
                if [row[4] for row in transfers] != expected:
                    raise RuntimeError('peer copy bytes differ from the cone interval')
                # These are outgoing activity intervals. They are not the
                # incoming writes that the local owner sum must wait for.
                part.extend(row[:4] for row in transfers)
                payload = sum(row[4] for row in transfers)
            times = [row[:2] for row in part]
            row = dict(index=index, layer=None if index == 0 else (index - 1) // 2,
                       kind='embedding' if index == 0 else ('attention' if index % 2 else 'moe'), rows=rows,
                       activity_ms=duration(times) / 1e6,
                       envelope_ms=(max(b for a, b in times) - min(a for a, b in times)) / 1e6,
                       exclusive_ms=sum(attributed[item] for item in part) / 1e6, peer_bytes=payload)
            reductions.append(row)
            group = groups['full' if rows == 8192 else 'upper']
            group['calls'] += 1
            for field in ['rows', 'activity_ms', 'envelope_ms', 'exclusive_ms', 'peer_bytes']:
                group[field] += row[field]
        counts['peer_dma'] = len(peer)
        chunks.append(dict(
            offset=chunk_index * 8192, rows=8192, paired_attention_launches=paired,
            gpu_envelope_ms=(frontier - begin) / 1e6,
            exclusive_ms=sum(totals.values()) / 1e6, compute_union_ms=duration(compute) / 1e6,
            communication_union_ms=duration(comm) / 1e6, compute_communication_overlap_ms=intersection / 1e6,
            communication_outside_compute_ms=(duration(comm) - intersection) / 1e6,
            communication_exclusive_ms=sum(value for name, value in totals.items()
                                           if name.startswith('chop_comm_') or name == 'peer_dma') / 1e6,
            compute_interval_caveat='Intervals include PDL waits; overlap is not useful compute overlap.',
            reductions=reductions, reduction_groups=dict(groups),
            kernels={name: dict(exclusive_ms=value / 1e6, launches=counts[name]) for name, value in totals.items()}))
    return chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--dma', action='store_true')
    parser.add_argument('--chunks', type=int, default=4)
    parser.add_argument('--steps', type=Path, help='Saved step trace JSON array, for wall time')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.chunks < 1:
        parser.error('--chunks must be positive')
    chunks = analyze(args.trace, args.dma, args.chunks)
    if args.steps:
        steps = json.loads(args.steps.read_text())
        if len(steps) != len(chunks) or any(step['prefill_tokens'] != 8192 for step in steps):
            raise RuntimeError('wrong step trace chunk sizes')
        for chunk, step in zip(chunks, steps):
            chunk['wall_ms'] = step['execution_us'] / 1000
            chunk['outside_gpu_ms'] = chunk['wall_ms'] - chunk['gpu_envelope_ms']
            chunk['host'] = step.get('host', {})
    result = dict(chunks=chunks)
    if args.dma:
        result['ordering'] = prefill_dma_order.audit(args.trace, args.chunks)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
