#!/usr/bin/env python3
"""Audit the dedicated prefill arena's complete fake-host lifecycle journals."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from audit_decode_attention import bind, digest

DATA_BYTES = 75_632_640
TABLE_BYTES = 65_536
CHANGED = {'deepseek41-flash-tp4.native.journal', 'deepseek41-flash-tp4.media.journal'}
RAW = re.compile(r'raw: (\d+)')


def read(path):
    return path.read_text().splitlines(), json.loads(Path(str(path) + '.handles.json').read_text())


def audit(old, new, before, after):
    """Allow eight dedicated allocations and releases, with exact retained operations."""
    assert Counter(before['modules'].values()) == Counter(after['modules'].values())
    assert len(new) == len(old) + 16
    kv = new.index('serving kv')
    allocations = [i for i, line in enumerate(new) if line.startswith('allocate ')]
    assert len(allocations) == len(after['buffers']) == len(after['ranks'])
    additions = set()
    added_buffers = set()
    ranks = []
    for at, index in enumerate(allocations):
        if new[index] != f'allocate {DATA_BYTES} 256':
            continue
        assert index + 1 < kv and new[index + 1] == f'allocate {TABLE_BYTES} 256'
        assert allocations[at + 1] == index + 1
        data, table = after['buffers'][at:at + 2]
        rank, size = after['ranks'][str(data)]
        assert size == DATA_BYTES and after['ranks'][str(table)] == [rank, TABLE_BYTES]
        ranks.append(rank)
        additions.update((index, index + 1))
        added_buffers.update((data, table))
    assert ranks == [0, 1, 2, 3] and len(additions) == len(added_buffers) == 8
    kept = [line for i, line in enumerate(new) if i not in additions]
    # Bare releases carry no handles. Resource-accounting tests independently
    # check ownership; here require exactly eight additional teardown releases.
    assert kept[-8:] == ['release'] * 8
    kept = kept[:-8]
    assert len(kept) == len(old)
    maps = {kind: {} for kind in ('stream', 'event', 'graph', 'function', 'buffer', 'mapped')}
    for kind, keyword, column in [
        ('stream', 'stream', 1), ('event', 'event', 1), ('graph', 'captured', 2),
        ('function', 'resolve', 1), ('mapped', 'async mapped create', 3),
    ]:
        aa = [line.split() for line in old if line.startswith(keyword + ' ')]
        bb = [line.split() for line in kept if line.startswith(keyword + ' ')]
        assert len(aa) == len(bb), keyword
        for a, b in zip(aa, bb):
            if kind == 'function':
                assert a[2:] == b[2:]
            bind(maps[kind], int(a[column]), int(b[column]))
    alloc = [line for line in old if line.startswith('allocate ')]
    assert alloc == [line for line in kept if line.startswith('allocate ')]
    remaining = [b for b in after['buffers'] if b not in added_buffers]
    assert len(alloc) == len(before['buffers']) == len(remaining)
    for line, a, b in zip(alloc, before['buffers'], remaining):
        assert before['ranks'][str(a)] == after['ranks'][str(b)]
        assert before['ranks'][str(a)][1] == int(line.split()[1])
        bind(maps['buffer'], (1 << 32) + a, (1 << 32) + b)
    inverse = {k: {v: u for u, v in mapping.items()} for k, mapping in maps.items()}
    added_raw = {(1 << 32) + b for b in added_buffers}

    def normalize(line):
        words = line.split(' ')
        kind = words[0]

        def handle(at, category):
            words[at] = str(inverse[category][int(words[at])])

        if kind in ('stream', 'event'):
            handle(1, kind)
        elif kind in ('capture', 'synchronize', 'zero', 'download', 'copy', 'launch', 'peer-copy', 'gather'):
            handle(1, 'stream')
        elif kind == 'upload':
            if words[1] != 'arena':
                handle(1, 'stream')
        elif kind in ('record', 'wait'):
            handle(1, 'stream')
            handle(2, 'event')
        elif kind in ('captured', 'replay'):
            handle(1, 'stream')
            handle(2, 'graph')
        elif kind in ('resolve', 'configure'):
            handle(1, 'function')
        elif kind == 'release':
            if len(words) > 1:
                assert words[1] in ('graph', 'stream', 'event')
                handle(2, words[1])
        elif kind == 'async':
            assert words[1] == 'mapped' and words[2] in ('create', 'release')
            handle(3, 'mapped')
        else:
            assert kind in ('allocate', 'state', 'serving'), kind
        assert not any(int(m[1]) in added_raw for m in RAW.finditer(line))
        return RAW.sub(lambda m: 'raw: ' + str(inverse['buffer'].get(int(m[1]), int(m[1]))), ' '.join(words))

    for index, (a, b) in enumerate(zip(old, kept)):
        assert a == normalize(b), ('operation', index, a, b)
    return dict(old=digest(old), new=digest(new), added_allocations=8, added_releases=8,
                added_bytes_per_rank=DATA_BYTES + TABLE_BYTES, all_retained_operations_exact=True)


def negative_controls(old, new, before, after):
    mutants = []
    allocation = new.index(f'allocate {DATA_BYTES} 256')
    launch = next(i for i, line in enumerate(new) if line.startswith('launch ') and 'offset: ' in line)
    for index, text, replacement in [
        (allocation, str(DATA_BYTES), str(DATA_BYTES + 256)),
        (allocation + 1, str(TABLE_BYTES), str(TABLE_BYTES + 256)),
        (allocation, '256', '128'),
        (launch, 'grid_x: ', 'grid_x: 9'),
    ]:
        mutant = new.copy()
        mutant[index] = mutant[index].replace(text, replacement, 1)
        assert mutant != new
        mutants.append((mutant, after))
    mutant = new.copy()
    mutant[launch] = re.sub(r'offset: (\d+)', lambda m: f'offset: {int(m[1]) + 16}', mutant[launch], count=1)
    mutants.append((mutant, after))
    mutant = new.copy()
    mutant[allocation], mutant[allocation + 1] = mutant[allocation + 1], mutant[allocation]
    mutants.extend([(mutant, after), (new[:-1], after), (new + ['release'], after)])
    meta = copy.deepcopy(after)
    at = sum(line.startswith('allocate ') for line in new[:allocation])
    meta['ranks'][str(meta['buffers'][at])][0] = 3
    mutants.append((new, meta))
    for i, (lines, metadata) in enumerate(mutants):
        try:
            audit(old, lines, before, metadata)
        except (AssertionError, KeyError, IndexError, ValueError):
            continue
        raise AssertionError(f'mutation {i} escaped')
    return len(mutants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.base.glob('*.journal'))
    assert len(files) == 8 and {p.name for p in files} == {p.name for p in args.head.glob('*.journal')}
    results = {}
    controls = 0
    for path in files:
        old, before = read(path)
        new, after = read(args.head / path.name)
        if path.name in CHANGED:
            results[path.name] = audit(old, new, before, after)
            controls += negative_controls(old, new, before, after)
        else:
            assert old == new and before == after, path.name
            results[path.name] = dict(exact=True, old=digest(old), new=digest(new))
    assert CHANGED <= results.keys() and controls == 18
    args.output.write_text(json.dumps(dict(passed=True, journals=results, rejected_mutations=controls), indent=2) + '\n')
    for name in sorted(CHANGED):
        print(name, results[name]['old'], results[name]['new'])
    print(f'PASS prefill_arena_audit journals={len(files)} rejected_mutations={controls}')


if __name__ == '__main__':
    main()
