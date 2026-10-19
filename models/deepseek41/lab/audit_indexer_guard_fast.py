#!/usr/bin/env python3
"""Reconstruct the coarse-guard journal delta from the previous journal."""
import argparse
import bisect
import itertools
import json
from pathlib import Path
import re

SCORES = (
    'chop_position_indexer_deepseek41_r2_all_scores_v3_live_v1_mma_fp4_guard_v1',
    'chop_position_indexer_deepseek41_r1_all_scores_v3_live_v1_mma_fp4_guard_v1',
    'chop_position_blocks_deepseek41_r1_scores_v3_valid_v1_mma_fp4_guard_v1',
)
FAST = tuple(s.replace('_guard_v1', '_guard_fast_v1') for s in SCORES)
EXTRA = tuple(s + suffix for s in FAST for suffix in ('', '_counts_v1'))


def fnv(text):
    result = 0xcbf29ce484222325
    for b in text.encode():
        result = ((result ^ b) * 0x100000001b3) & ((1 << 64) - 1)
    return result


def audit(old, new, ranks, bf16=False):
    if bf16:
        assert old == new, 'BF16 journal changed'
        return dict(identical=True, lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}')
    ends = [int(s.split()[1]) for s in old if s.startswith('resolve ') and s.split()[2] == SCORES[-1]]
    assert len(ends) == ranks and ends == sorted(ends)

    def handle(value):
        value = int(value)
        slot = value & 0xffffffff
        # FakeHost supplies image/KV/state views in fixed namespaces starting
        # at IMAGE_SLOT=1,000,000; those never consume FakeDevice.next IDs.
        if slot >= 1_000_000:
            return str(value)
        return str(value + 7 * bisect.bisect_left(ends, slot))

    expected, renamed = [], 0
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
        if op == 'launch' and fields[2] in SCORES:
            fields[2] = FAST[SCORES.index(fields[2])]
            renamed += 1
        rewritten = re.sub(r'raw: (\d+)', lambda m: 'raw: ' + handle(m[1]), ' '.join(fields))
        expected.append(rewritten)
        if op == 'resolve' and fields[2] == SCORES[-1]:
            # One new module consumes a handle before its six qualified entries.
            last = int(fields[1])
            expected.extend(f'resolve {last + 2 + i} {symbol}' for i, symbol in enumerate(EXTRA))
    assert renamed > 0
    def groups(lines):
        return [(same, list(group)) for same, group in itertools.groupby(lines, lambda s: s == 'release')]
    a, b = groups(expected), groups(new)
    assert len(a) == len(b), 'release boundaries changed'
    extra_releases = 0
    for (release_a, aa), (release_b, bb) in zip(a,b):
        assert release_a == release_b
        if release_a:
            growth = len(bb) - len(aa)
            assert growth >= 0 and growth % 7 == 0, 'unexpected resource release delta'
            extra_releases += growth
        else:
            assert aa == bb, next(((x[:300],y[:300]) for x,y in zip(aa,bb) if x != y), 'operation count changed')
    assert extra_releases == 7*ranks
    assert len(new) == len(old) + 13*ranks
    return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}', renamed_scores=renamed,
                added_resolves=6*ranks, added_releases=extra_releases,
                launch_arguments_grids_and_stream_order_unchanged=True)


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
        ranks = int(re.search(r'tp(\d+)',name)[1])
        bf16 = '.bf16.' in name
        report[name] = audit(old,new,ranks,bf16)
        text = '\n'.join(new)
        mutants = [text+'\nrelease', text.replace('grid_x: 192','grid_x: 193',1),
                   text.replace('offset: 0','offset: 1',1)]
        if not bf16:
            mutants.extend([text.replace(FAST[0],SCORES[0],1),
                            text.replace('_counts_v1','_counts_v2',1)])
        for broken in mutants:
            assert broken != text
            try:
                audit(old,broken.splitlines(),ranks,bf16)
            except AssertionError:
                pass
            else:
                raise AssertionError('journal negative control escaped')
        report[name]['negative_controls'] = len(mutants)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
