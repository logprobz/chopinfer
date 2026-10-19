#!/usr/bin/env python3
"""Reconstruct the qualified hierarchy's serving journal from the 6c journal."""
import argparse
import bisect
import itertools
import json
from pathlib import Path
import re
from audit_indexer_guard_fast import fnv

MERGES = (
    'chop_position_indexer_deepseek41_r2_all_select_v3_live_merge_v1',
    'chop_position_indexer_deepseek41_r1_all_select_v3_live_merge_v1',
    'chop_position_blocks_deepseek41_r1_select_v3_valid_merge_v1',
)
EXTRA = tuple(s + suffix for s in MERGES for suffix in
              ('_hier_group_v1_negative_v1', '_hier_group_v1', '_hier_final_v1'))
# Cubins load in name order: position_hierarchical follows position_blocks.
PREVIOUS_LAST = 'chop_candidate_block_ids_deepseek41_gather_v3'
SPAN = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)')


def repartition(lines):
    """Replay graphs::pieces over shared_overlap's indivisible fork/join units."""
    result, at = [], 0
    while at < len(lines):
        if not lines[at].startswith('capture '):
            result.append(lines[at])
            at += 1
            continue
        start, graphs, body = lines[at], [], []
        while at < len(lines) and lines[at] == start:
            at += 1
            segment = []
            while not lines[at].startswith('captured '):
                segment.append(lines[at])
                at += 1
            end = lines[at].split()
            assert end[1] == start.split()[1]
            assert int(end[3]) == sum(s.startswith('launch ') for s in segment)
            graphs.append(end[:3])
            body.extend(segment)
            at += 1
        units, i = [], 0
        while i < len(body):
            if body[i].startswith('launch '):
                units.append([body[i]])
                i += 1
                continue
            first = i
            op, main, fork = body[i].split()
            wait, side, event = body[i+1].split()
            assert op == 'record' and wait == 'wait' and event == fork and main != side
            i += 2
            while body[i].startswith('launch '):
                assert body[i].split()[1] in (main, side)
                assert '_hier_' not in body[i], 'indexer overlaps shared expert work'
                i += 1
            op, stream, join = body[i].split()
            wait, target, event = body[i+1].split()
            assert (op, stream, wait, target, event) == ('record', side, 'wait', main, join)
            i += 2
            units.append(body[first:i])
        n = len(units)
        bounds = (0, n//32, n//6, n)
        chunks = [list(itertools.chain.from_iterable(units[a:b])) for a, b in zip(bounds, bounds[1:]) if a != b]
        assert len(chunks) == len(graphs) == 3
        for graph, chunk in zip(graphs, chunks):
            result.extend([start, *chunk, ' '.join(graph + [str(sum(s.startswith('launch ') for s in chunk))])])
    return result


def audit(old, new, ranks, bf16=False):
    if bf16:
        assert old == new, 'BF16 journal changed'
        return dict(identical=True, lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}')
    ends = [int(s.split()[1]) for s in old if s.startswith('resolve ') and s.split()[2] == PREVIOUS_LAST]
    assert len(ends) == ranks and ends == sorted(ends)

    def handle(value):
        value = int(value)
        slot = value & 0xffffffff
        # Fixed FakeHost image/KV/state namespaces never consume device IDs.
        return str(value if slot >= 1_000_000 else value + 10 * bisect.bisect_left(ends, slot))

    expected, merged, capturing = [], 0, {}
    for line in old:
        fields = line.split(' ')
        op = fields[0]
        columns = []
        if op in ('stream', 'event', 'capture', 'synchronize', 'zero', 'download', 'copy', 'launch', 'resolve', 'configure'):
            columns = [1]
        elif op in ('record', 'wait', 'captured', 'replay'):
            columns = [1, 2]
        elif op == 'upload' and fields[1] != 'arena':
            columns = [1]
        elif op == 'release' and len(fields) > 1:
            assert fields[1] in ('stream', 'event', 'graph')
            columns = [2]
        else:
            assert op in ('allocate', 'state', 'serving', 'release', 'upload'), op
        for i in columns:
            fields[i] = handle(fields[i])
        if op == 'capture':
            assert fields[1] not in capturing
            capturing[fields[1]] = 0
        if op == 'captured':
            # The fake host records the number of graph nodes as well.
            fields[3] = str(int(fields[3]) + capturing.pop(fields[1]))
        rewritten = re.sub(r'raw: (\d+)', lambda m: 'raw: ' + handle(m[1]), ' '.join(fields))
        if op == 'launch' and fields[2] in MERGES and fields[1] in capturing:
            rows = int(re.search(r'grid_y: (\d+)', rewritten)[1])
            assert 1 <= rows <= 32
            # These journals only capture Decode, with C2 text or C1 media.
            score = SPAN.search(rewritten)
            raw, offset, length = map(int, score.groups())
            size = rows * 4 * 513 * 8
            assert length == 256 << 20
            assert rows * (1 << 20) * 4 <= length - size
            extra = f'DeviceBuffer(DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset + length - size}, length: {size} }})'
            assert rewritten.endswith(']') and 'grid_x: 1,' in rewritten
            args = rewritten[:-1] + ', ' + extra + ']'
            group = args.replace(fields[2], fields[2] + '_hier_group_v1', 1).replace('grid_x: 1,', 'grid_x: 4,', 1)
            final = args.replace(fields[2], fields[2] + '_hier_final_v1', 1)
            expected.extend((group, final))
            merged += 1
            capturing[fields[1]] += 1
        else:
            expected.append(rewritten)
        if op == 'resolve' and fields[2] == PREVIOUS_LAST:
            last = int(fields[1])
            expected.extend(f'resolve {last + 2 + i} {symbol}' for i, symbol in enumerate(EXTRA))
    assert not capturing and merged == 8*ranks
    assert repartition(old) == old, 'baseline graph pieces do not match the scheduler'
    expected = repartition(expected)

    def groups(lines):
        return [(same, list(group)) for same, group in itertools.groupby(lines, lambda s: s == 'release')]
    a, b = groups(expected), groups(new)
    assert len(a) == len(b), 'release boundaries changed'
    extra_releases = 0
    for (release_a, aa), (release_b, bb) in zip(a, b):
        assert release_a == release_b
        if release_a:
            growth = len(bb) - len(aa)
            assert growth >= 0 and growth % 10 == 0, 'unexpected release delta'
            extra_releases += growth
        else:
            assert aa == bb, next(((x[:500], y[:500]) for x, y in zip(aa, bb) if x != y), 'operation count changed')
    assert extra_releases == 10*ranks
    assert len(new) == len(old) + 19*ranks + merged
    return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}', decode_merges=merged,
                added_resolves=9*ranks, added_releases=extra_releases,
                other_launches_arguments_grids_streams_and_allocations_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('head', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    files = sorted(args.base.glob('*.journal'))
    assert len(files) == 8 and {p.name for p in files} == {p.name for p in args.head.glob('*.journal')}
    report = {}
    for path in files:
        name = path.name
        old = path.read_text().splitlines()
        new = (args.head/name).read_text().splitlines()
        ranks = int(re.search(r'tp(\d+)', name)[1])
        bf16 = '.bf16.' in name
        report[name] = audit(old, new, ranks, bf16)
        text = '\n'.join(new)
        mutants = [text+'\nrelease', text.replace('grid_x: 192', 'grid_x: 193', 1),
                   text.replace('offset: 0', 'offset: 1', 1)]
        if not bf16:
            at = next(i for i, line in enumerate(new) if line.startswith('launch ') and '_hier_group_v1 ' in line)
            for before, after in [('_hier_group_v1 ', '_hier_group_v1_negative_v1 '), ('grid_x: 4,', 'grid_x: 3,'), ('ScalarU64(0)', 'ScalarU64(8)')]:
                broken = new.copy()
                broken[at] = broken[at].replace(before, after, 1)
                mutants.append('\n'.join(broken))
            broken = new.copy()
            broken[at], broken[at+1] = broken[at+1], broken[at]
            mutants.append('\n'.join(broken))
            scratch = list(SPAN.finditer(new[at]))[-1]
            mutants.append(text.replace(scratch[0], scratch[0].replace('offset: '+scratch[2], 'offset: 0'), 1))
            at = next(i for i, line in enumerate(new) if line.startswith('captured '))
            broken = new.copy()
            fields = broken[at].split()
            fields[3] = str(int(fields[3]) + 1)
            broken[at] = ' '.join(fields)
            mutants.append('\n'.join(broken))
            broken = new.copy()
            broken[at-1], broken[at] = broken[at], broken[at-1]
            mutants.append('\n'.join(broken))
        for broken in mutants:
            assert broken != text
            try:
                audit(old, broken.splitlines(), ranks, bf16)
            except AssertionError:
                pass
            else:
                raise AssertionError('journal negative control escaped')
        report[name]['negative_controls'] = len(mutants)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
