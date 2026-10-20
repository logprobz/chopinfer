#!/usr/bin/env python3
"""Check the tile module delta and match context graphs to qualified descriptors."""
import argparse
import bisect
import itertools
import json
from pathlib import Path
import re

from audit_indexer_guard_fast import fnv

EXTRA = tuple(
    f'chop_position_indexer_deepseek41_r{ratio}_all_scores_v3_live_v1_mma_fp4_tile{tiles}_mode{mode}_trace{trace}_v1'
    for ratio in (2, 1)
    for tiles, mode, trace in ((1, 0, 0), (4, 0, 0), (1, 1, 0), (1, 2, 0), (1, 0, 1), (4, 0, 1)))


def audit(old, new, ranks, bf16=False):
    if bf16:
        assert old == new, 'BF16 journal changed'
        return dict(identical=True, lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}')
    resolves = [(int(s.split()[1]), s.split()[2]) for s in new
                if s.startswith('resolve ') and s.split()[2] in EXTRA]
    assert len(resolves) == 12 * ranks
    ends = []
    for i in range(ranks):
        group = resolves[12*i:12*i+12]
        assert [s for _, s in group] == list(EXTRA)
        assert [n for n, _ in group] == list(range(group[0][0], group[0][0]+12))
        ends.append(group[-1][0])
    assert ends == sorted(ends)

    def handle(value):
        value = int(value)
        slot = value & 0xffffffff
        return str(value if slot >= 1_000_000 else value - 13 * bisect.bisect_left(ends, slot))

    normalized = []
    for line in new:
        fields = line.split(' ')
        op = fields[0]
        if op == 'resolve' and fields[2] in EXTRA:
            continue
        if op == 'launch':
            assert fields[2] not in EXTRA, 'short-context fixture selected a tiled score'
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
        normalized.append(re.sub(r'raw: (\d+)', lambda m: 'raw: '+handle(m[1]), ' '.join(fields)))

    def groups(lines):
        return [(kind, list(part)) for kind, part in itertools.groupby(lines, lambda s: s == 'release')]

    a, b = groups(old), groups(normalized)
    assert len(a) == len(b)
    releases = 0
    for (ka, aa), (kb, bb) in zip(a, b):
        assert ka == kb
        if ka:
            growth = len(bb) - len(aa)
            assert growth >= 0 and growth % 13 == 0
            releases += growth
        else:
            assert aa == bb, next(((x[:300], y[:300]) for x, y in zip(aa, bb) if x != y), 'operation count changed')
    assert releases == 13 * ranks and len(new) == len(old) + 25 * ranks
    return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}',
                added_resolves=12*ranks, added_releases=releases,
                existing_launches_arguments_allocations_and_streams_unchanged=True)


def captures(lines):
    at = 0
    while at < len(lines):
        if not lines[at].startswith('capture '):
            at += 1
            continue
        start, body = lines[at], []
        while at < len(lines) and lines[at] == start:
            at += 1
            while not lines[at].startswith('captured '):
                body.append(lines[at])
                at += 1
            assert lines[at].split()[1] == start.split()[1]
            at += 1
        yield body


def scalars(text):
    return re.findall(r'(?:Scalar)?(U32|U64|I32)\((\d+)\)', text)


def audit_contexts(directory, inventory):
    plans = json.loads(inventory.read_text())['plans']
    seen = []
    for path in sorted(directory.glob('*.journal')):
        for body in captures(path.read_text().splitlines()):
            actual = [v for v in body if v.startswith('launch ') and
                      re.search(r' chop_position_(indexer|blocks)_', v) and
                      ('_pack_fp4_guard_v1 ' in v or '_mma_fp4_' in v or '_select_v3_' in v)]
            if not actual:
                continue
            scores = [v for v in actual if '_mma_fp4_' in v]
            assert len(scores) == 8
            rows = int(re.search(r'grid_y: (\d+)', scores[0])[1])
            verify = any('chop_latent_sources_' in v and '_verify' in v for v in body)
            layout = 'verify32' if verify else f'c{rows}'
            arm = 'combined' if any('_tile4_' in v for v in scores) else (
                'verify_hierarchy' if verify and any('_hier_group_v1' in v for v in actual) else 'accepted')
            wanted = [c for p in plans if p['layout'] == layout and p['arm'] == arm for c in p['calls']]
            assert len(actual) == len(wanted), (layout, arm, len(actual), len(wanted))
            for line, call in zip(actual, wanted):
                assert line.split()[2] == call['symbol'], (layout, arm, line.split()[2], call['symbol'])
                for kind in ('grid', 'block'):
                    for axis, value in zip(('x', 'y', 'z'), call[kind]):
                        assert f'{kind}_{axis}: {value},' in line
                assert 'flags: 1,' in line and 'shared_memory_bytes: 0,' in line
                assert scalars(line) == scalars(call['arguments']), call['symbol']
            for i, score in enumerate(scores):
                spans = re.findall(r'offset: (\d+), length: (\d+)', score)
                assert spans[7] == ('512' if i >= 4 else '0', '4')
                assert 'length: 268435456' in score
            seen.append((layout, arm))
    expected = {(layout, arm): 4 for layout in ('c8', 'c32', 'verify32') for arm in ('accepted', 'combined')}
    expected[('verify32', 'verify_hierarchy')] = 4
    assert len(seen) == 28
    assert {key: seen.count(key) for key in set(seen)} == expected
    return dict(captured_rank_programs=28, source_plans=224,
                qualified_descriptor_matches=True, graph_cache_replay_checked_by_rust=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('head', type=Path)
    p.add_argument('contexts', type=Path)
    p.add_argument('inventory', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    paths = sorted(a.base.glob('*.journal'))
    assert len(paths) == 8 and {p.name for p in paths} == {p.name for p in a.head.glob('*.journal')}
    report = {}
    for path in paths:
        old, new = path.read_text().splitlines(), (a.head/path.name).read_text().splitlines()
        ranks = int(re.search(r'tp(\d+)', path.name)[1])
        bf16 = '.bf16.' in path.name
        row = audit(old, new, ranks, bf16)
        text = '\n'.join(new)
        mutants = [text+'\nrelease', text.replace('offset: 0', 'offset: 1', 1),
                   text.replace('grid_x: 192', 'grid_x: 191', 1)]
        if not bf16:
            mutants += [text.replace(EXTRA[0], EXTRA[0]+'_bad', 1)]
        for mutant in mutants:
            assert mutant != text
            try:
                audit(old, mutant.splitlines(), ranks, bf16)
            except AssertionError:
                continue
            raise AssertionError('journal mutation was accepted')
        report[path.name] = dict(row, rejected_mutations=len(mutants))
    report['contexts'] = audit_contexts(a.contexts, a.inventory)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('INDEXER_WORKLOAD_JOURNAL_AUDIT_COMPLETE contexts=28 source_plans=224')


if __name__ == '__main__':
    main()
