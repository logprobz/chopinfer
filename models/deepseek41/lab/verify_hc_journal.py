#!/usr/bin/env python3
"""Audit the verify fusion against every original operand and untouched launch."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

ROUND = 'chop_row_rounding_round_deepseek41_w5120_v1'
EXPAND = 'chop_sinkhorn_hc_deepseek41_expand_f32_v1'
MIX = 'chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_split_v1'
WARP = 'chop_deepseek41_hc_mixer_probe_warp_v1'
NORM = 'chop_row_rmsnorm_deepseek41_w5120_v1'
WIDEN = 'chop_row_rounding_widen_deepseek41_w5120_v1'
CEIL = 'chop_deepseek41_moe_ceil_w5120_parallel_v1'
PARTIAL = 'chop_deepseek41_hc_round_chain_w4_v1'
TAIL = 'chop_deepseek41_hc_finish_norm_w32_v1'


def arguments(text):
    result, start, depth = [], 0, 0
    for at, char in enumerate(text):
        if char in '([{': depth += 1
        elif char in ')]}': depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[start:at].strip()); start = at + 1
        if depth < 0: raise ValueError('unbalanced journal argument')
    if depth: raise ValueError('unclosed journal argument')
    result.append(text[start:].strip())
    return result


def launches(text):
    streams = collections.defaultdict(list)
    for line in text.splitlines():
        if not line.startswith('launch '): continue
        match = re.fullmatch(r'launch (\d+) (\S+) (LaunchConfigV1 .*?) \[(.*)\]', line)
        if match is None: raise ValueError('unparsed launch: ' + line)
        stream, symbol, config, args = match.groups()
        streams[stream].append(dict(symbol=symbol, config=config, args=arguments(args)))
    if not streams: raise ValueError('empty launch journal')
    return dict(streams)


def require(condition, message):
    if not condition: raise ValueError(message)


def align_streams(before, after):
    """Match opaque stream/event IDs using unchanged dependency edges."""
    tags = {'record', 'wait', 'synchronize'}
    left = [parts for line in before.splitlines()
            if (parts := line.split()) and parts[0] in tags]
    right = [parts for line in after.splitlines()
             if (parts := line.split()) and parts[0] in tags]
    require(len(left) == len(right), 'dependency count changed')
    forward, reverse = {}, {}
    def bind(kind, old, new):
        key, value = (kind, old), (kind, new)
        require(forward.setdefault(key, value) == value and
                reverse.setdefault(value, key) == key, 'dependency identity changed')
    for old, new in zip(left, right):
        require(old[0] == new[0] and len(old) == len(new), 'dependency order changed')
        bind('stream', old[1], new[1])
        if old[0] in {'record', 'wait'}:
            bind('event', old[2], new[2])
        else:
            require(old[2:] == new[2:], 'synchronize arguments changed')
    old_streams = set(launches(before))
    new_streams = set(launches(after))
    for stream in new_streams:
        if ('stream', stream) not in reverse:
            require(stream in old_streams, 'unanchored stream renumbering')
            bind('stream', stream, stream)
    require({reverse['stream', s][1] for s in new_streams} == old_streams,
            'stream set changed')
    result = []
    for line in after.splitlines():
        if line.startswith('launch '):
            _, stream, rest = line.split(' ', 2)
            line = 'launch ' + reverse['stream', stream][1] + ' ' + rest
        result.append(line)
    return '\n'.join(result)


def align_buffers(before_meta, after_meta, after):
    """Use captured allocation order, rank and byte extent as buffer identity."""
    old, new = before_meta['buffers'], after_meta['buffers']
    require(len(old) == len(new) and len(set(old)) == len(old) and
            len(set(new)) == len(new), 'allocation count or identity changed')
    inverse = {}
    for a, b in zip(old, new):
        require(before_meta['ranks'][str(a)] == after_meta['ranks'][str(b)],
                'allocation rank, extent or order changed')
        inverse[(1 << 32) + b] = (1 << 32) + a
    return re.sub(r'raw: (\d+)',
                  lambda m: 'raw: ' + str(inverse.get(int(m[1]), int(m[1]))), after)


def config(grid_x, grid_y, block_x):
    return (f'LaunchConfigV1 {{ flags: 1, grid_x: {grid_x}, grid_y: {grid_y}, grid_z: 1, '
            f'block_x: {block_x}, block_y: 1, block_z: 1, shared_memory_bytes: 0, '
            'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')


def audit(before, after, rows, row_tiles=False):
    old, new = launches(before), launches(align_streams(before, after))
    require(old.keys() == new.keys(), 'stream set changed')
    total, widened, warps = 0, 0, 0
    for stream in old:
        a, b = old[stream], new[stream]
        removed_a, removed_b = set(), set()
        sites = {entry['args'][2]: i for i, entry in enumerate(a) if entry['symbol'] == MIX}
        for at, entry in enumerate(b):
            if entry['symbol'] == WARP:
                match = sites.get(entry['args'][2])
                require(match is not None, 'warp mixer has no original site')
                require(entry['args'] == a[match]['args'] + ['NullDevicePointer'], 'warp mixer operands changed')
                require(entry['config'] == a[match]['config'], 'warp mixer launch geometry changed')
                removed_a.add(match); removed_b.add(at); warps += 1
            if entry['symbol'] != TAIL: continue
            producer = 'chop_deepseek41_hc_round_rows_r2_v1' if row_tiles else PARTIAL
            require(at > 0 and b[at-1]['symbol'] == producer, 'tail is not adjacent to its producer')
            require(b[at-1]['config'] == config(160, rows//2 if row_tiles else rows,
                                              256 if row_tiles else 128),
                    'producer geometry differs from qualification')
            require(entry['config'] == config(1, rows, 1024), 'tail geometry differs from qualification')
            p, t = b[at-1]['args'], entry['args']
            require(len(p) == 9 and len(t) == 18, 'fused ABI changed')
            mix_at = sites.get(t[1])
            require(mix_at is not None, 'tail has no original mixer')
            m = a[mix_at]['args']
            expand_at = next((i for i in range(mix_at-1, -1, -1) if a[i]['symbol'] == EXPAND), None)
            require(expand_at is not None, 'tail lacks its preceding expansion')
            round_at = next((i for i in range(expand_at-1, -1, -1) if a[i]['symbol'] == ROUND), None)
            require(round_at is not None, 'producer lacks the original rounded sum')
            norm_at = mix_at + 1
            require(a[norm_at]['symbol'] == NORM, 'original norm is not after its mixer')
            quant_at = norm_at + 1
            widen_at = quant_at if a[quant_at]['symbol'] == WIDEN else None
            if widen_at is not None: quant_at += 1; widened += 1
            require(a[quant_at]['symbol'] == CEIL, 'original quantizer is absent')
            r, e, n, q = (a[i]['args'] for i in (round_at, expand_at, norm_at, quant_at))
            require(len(r)==3 and len(e)==6 and len(m)==12 and len(n)==7 and len(q)==6, 'reference ABI changed')
            require(p[:7] == [e[0], r[0], r[1], e[2], e[3], e[4], m[1]], 'producer boundary operands changed')
            require(p[2] == e[1] and p[5] == m[0], 'round/expand boundary changed')
            require(t[:8] == [m[0], *m[2:9]], 'finisher operands changed')
            require(p[7] == t[8], 'producer and finisher scratch differ')
            require(p[8] == t[9] == r[2] == e[5] == m[11] == n[5] == q[3], 'live-row guard changed')
            require(n[0] == m[5] and n[2:4] == ['NullDevicePointer']*2 and q[0] == n[4], 'norm input contract changed')
            wide = 'NullDevicePointer'
            if widen_at is not None:
                w = a[widen_at]['args']
                require(w[0] == n[4] and w[2] == n[5], 'widen input contract changed')
                wide = w[1]
            require(t[10:] == [n[1], n[4], wide, q[1], q[2], q[4], q[5], n[6]], 'norm outputs, epsilon or status changed')
            members = {round_at, expand_at, mix_at, norm_at, quant_at}
            if widen_at is not None: members.add(widen_at)
            require(not members & removed_a, 'original launch consumed twice')
            removed_a.update(members); removed_b.update([at-1, at]); total += 1
        kept_a = [e for i,e in enumerate(a) if i not in removed_a]
        kept_b = [e for i,e in enumerate(b) if i not in removed_b]
        if kept_a != kept_b:
            first = next(((x, y) for x, y in zip(kept_a, kept_b) if x != y), None)
            raise ValueError(f'unrelated launch changed on stream {stream}: '
                             f'counts {len(kept_a)}/{len(kept_b)}, first {first}')
    expected = rows == 32 if row_tiles else rows in (6, 32)
    require(total == (77*4 if expected else 0), 'wrong fused boundary count')
    require(widened == (47*4 if expected else 0), 'wrong widened boundary count')
    require(warps == (3*4 if expected else 0), 'wrong standalone warp count')
    before_count, after_count = sum(map(len, old.values())), sum(map(len, new.values()))
    require(before_count-after_count == (278*4 if expected else 0), 'wrong launch reduction')
    return dict(rows=rows, ranks=4, fused_boundaries_per_rank=total//4, widened_per_rank=widened//4,
                standalone_warps_per_rank=warps//4, before_per_rank=before_count//4,
                after_per_rank=after_count//4, launches_saved_per_rank=(before_count-after_count)//4,
                unrelated_launches_and_operands_equal=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row-tiles', action='store_true')
    args = parser.parse_args()
    records = []
    for rows in (6, 8, 16, 32):
        name = f'verify.tp4.r{rows}.Native.journal'
        before, after = args.before/name, args.after/name
        before_text, after_text = before.read_text(), after.read_text()
        before_meta = before.with_name(name + '.handles.json')
        after_meta = after.with_name(name + '.handles.json')
        require(before_meta.exists() == after_meta.exists(), 'missing allocation identities')
        if before_meta.exists():
            after_text = align_buffers(json.loads(before_meta.read_text()),
                                       json.loads(after_meta.read_text()), after_text)
        record = audit(before_text, after_text, rows, args.row_tiles)
        record['sha256'] = {label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in [('before',before),('after',after)]}
        if rows == 32 or (rows == 6 and not args.row_tiles):
            text = after_text
            line = next(l for l in text.splitlines() if TAIL in l and l.startswith('launch '))
            changed = re.sub(r'ScalarU32\(\d+\)\]$', 'ScalarU32(0)]', line)
            require(changed != line, 'epsilon negative control did not change the journal')
            try: audit(before.read_text(), text.replace(line,changed,1), rows, args.row_tiles)
            except ValueError: record['changed_epsilon_rejected'] = True
            else: raise ValueError('journal auditor accepted a changed epsilon')
        records.append(record)
    args.output.write_text(json.dumps(dict(records=records), indent=2)+'\n')
