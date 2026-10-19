#!/usr/bin/env python3
"""Audit exact-row attention adoption without discarding retained operations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

CUBIN = 'latent_decode_staging_deepseek41_sm120'
SUFFIX = '_stage64_h0_b0_profile0_v1'
RAW = re.compile(r'raw: (\d+)')
BASE_PINS = {
    'tp1.native': (16208, '51b2f41159e4e61a'),
    'tp1.bf16': (16154, 'df0f35d783383abc'),
    'tp2.native': (32843, 'c6477b788b9023cc'),
    'tp2.bf16': (32735, '882854e11a40dd08'),
    'tp4.native': (66019, '531294d3234ec39b'),
    'tp4.bf16': (65763, '35ba745043b4ba72'),
    'tp2.media': (20803, 'd038f4dff171fa11'),
    'tp4.media': (41851, '3a24d94544043dce'),
}


def digest(lines):
    data = '\n'.join(lines).encode()
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return dict(lines=len(lines), fnv=f'{value:016x}', sha256=hashlib.sha256(data).hexdigest())


def bind(mapping, old, new):
    if old in mapping:
        assert mapping[old] == new, ('inconsistent handle', old, new)
    else:
        assert new not in mapping.values(), ('aliased handle', old, new)
        mapping[old] = new


def normalize_source(line, winners, changes):
    symbol = line.split()[2]
    if '_stage' not in symbol:
        return line
    assert symbol in winners, ('unexpected source', symbol)
    rows = 8 if '_decode_' in symbol else 32
    expected = dict(flags=1, grid_x=rows, grid_y=2, grid_z=8, block_x=128,
                    block_y=1, block_z=1, shared_memory_bytes=75008)
    for name, value in expected.items():
        match = re.search(r'\b' + name + r': (\d+)', line)
        assert match and int(match[1]) == value, (symbol, name)
    assert 'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 }' in line
    assert line.endswith(', NullDevicePointer]')
    assert line.count('NullDevicePointer') == 1
    changes[symbol] += 1
    line = line.removesuffix(', NullDevicePointer]') + ']'
    return line.replace(symbol, symbol.removesuffix(SUFFIX) + '_heads_v1', 1).replace(
        'shared_memory_bytes: 75008', 'shared_memory_bytes: 83200', 1)


def audit(base, head, base_meta, head_meta, symbols, added_module, changed_sources):
    winners = {symbol for symbol in symbols if symbol.endswith(SUFFIX)}
    assert len(symbols) == 110 and len(winners) == 6
    functions = {line.split()[1]: line.split()[2] for line in head if line.startswith('resolve ')}
    extras = [line for line in head if line.startswith('resolve ') and line.split()[2] in symbols]
    assert [line.split()[2] for line in extras] == symbols * 4, 'locked declaration order'
    assert Counter(head_meta['modules'].values())[CUBIN] == 4
    if added_module:
        assert not any(line.startswith('resolve ') and line.split()[2] in symbols for line in base)
    else:
        assert [line.split()[2] for line in base if line.startswith('resolve ') and line.split()[2] in symbols] == symbols * 4
    configured = Counter()
    kept = []
    changes = Counter()
    for line in head:
        fields = line.split()
        if fields[0] == 'configure' and functions[fields[1]] in symbols:
            symbol = functions[fields[1]]
            assert symbol in winners and fields[2:] == ['75008'], 'unqualified configuration'
            configured[symbol] += 1
            continue
        if added_module and fields[0] == 'resolve' and fields[2] in symbols:
            continue
        if fields[0] == 'launch' and fields[2].startswith('chop_latent_sources_') and '_stage' in fields[2]:
            assert changed_sources, 'unmeasured lifecycle route'
            line = normalize_source(line, winners, changes)
        kept.append(line)
    assert configured == Counter({symbol: 4 for symbol in winners})
    if changed_sources:
        expected = {symbol: (8 if '_window_' in symbol else 72 if '_p32_' in symbol else 80)
                    for symbol in winners}
        assert changes == Counter(expected), ('source substitution coverage', changes)
        assert not any(line.startswith('launch ') and line.split()[2].startswith('chop_latent_sources_') and '_stage' in line.split()[2] for line in base)
    else:
        assert not changes
    released = 4 * (len(symbols) + 1) if added_module else 0
    if released:
        assert kept[-released:] == ['release'] * released, 'added resources must retire at teardown'
        kept = kept[:-released]
    assert len(base) == len(kept), ('retained length', len(base), len(kept))

    def result():
        return dict(old=digest(base), new=digest(head), added_resolves=len(extras) if added_module else 0,
                    added_configurations=sum(configured.values()), added_releases=released,
                    source_substitutions=dict(changes), retained_operations=len(base))

    if not added_module:
        # The paired CPU test loads the same catalog in both arms. No handle
        # may move, including asynchronous gather records and their resources.
        assert base_meta == head_meta, 'paired resource identities changed'
        for index, (old, new) in enumerate(zip(base, kept)):
            assert old == new, ('paired operation differs', index, old, new)
        return result()

    maps = {kind: {} for kind in ['stream', 'event', 'graph', 'function', 'buffer']}
    for kind, keyword, column in [('stream', 'stream', 1), ('event', 'event', 1),
                                   ('graph', 'captured', 2), ('function', 'resolve', 1)]:
        old = [line.split() for line in base if line.startswith(keyword + ' ')]
        new = [line.split() for line in kept if line.startswith(keyword + ' ')]
        assert len(old) == len(new), keyword
        for a, b in zip(old, new):
            if keyword == 'resolve':
                assert a[2:] == b[2:], 'retained symbol changed'
            bind(maps[kind], int(a[column]), int(b[column]))
    old_alloc = [line for line in base if line.startswith('allocate ')]
    new_alloc = [line for line in kept if line.startswith('allocate ')]
    assert old_alloc == new_alloc, 'allocation sizes or order changed'
    assert len(old_alloc) == len(base_meta['buffers']) == len(head_meta['buffers'])
    for line, old, new in zip(old_alloc, base_meta['buffers'], head_meta['buffers']):
        assert base_meta['ranks'][str(old)] == head_meta['ranks'][str(new)], 'allocation rank or size changed'
        assert base_meta['ranks'][str(old)][1] == int(line.split()[1])
        bind(maps['buffer'], (1 << 32) + old, (1 << 32) + new)
    inverse = {kind: {new: old for old, new in values.items()} for kind, values in maps.items()}

    def normalize(line):
        words = line.split(' ')
        kind = words[0]
        def change(at, category):
            words[at] = str(inverse[category][int(words[at])])
        if kind in ['stream', 'event']:
            change(1, kind)
        elif kind in ['capture', 'synchronize', 'zero', 'download', 'copy', 'launch']:
            change(1, 'stream')
        elif kind == 'upload':
            if words[1] != 'arena':
                change(1, 'stream')
        elif kind in ['record', 'wait']:
            change(1, 'stream'); change(2, 'event')
        elif kind in ['captured', 'replay']:
            change(1, 'stream'); change(2, 'graph')
        elif kind in ['resolve', 'configure']:
            change(1, 'function')
        elif kind == 'release':
            if len(words) > 1:
                assert words[1] in ['graph', 'stream', 'event']
                change(2, words[1])
        else:
            assert kind in ['allocate', 'state', 'serving'], kind
        line = ' '.join(words)
        return RAW.sub(lambda match: 'raw: ' + str(inverse['buffer'].get(int(match[1]), int(match[1]))), line)

    for index, (old, new) in enumerate(zip(base, kept)):
        assert old == normalize(new), ('retained operation differs', index, old, new)
    return result()


def mutations(base, head, base_meta, head_meta, symbols):
    source = next(i for i, line in enumerate(head) if line.startswith('launch ') and SUFFIX in line)
    config = next(i for i, line in enumerate(head) if line.startswith('configure ') and line.endswith(' 75008'))
    merge = next(i for i, line in enumerate(head) if i > source and line.startswith('launch ') and 'chop_latent_sources_' in line and '_merge_' in line)
    resolve = next(i for i, line in enumerate(head) if line.startswith('resolve ') and SUFFIX in line)
    raw = RAW.search(head[source])[1]
    other = next(str((1 << 32) + slot) for slot in head_meta['buffers'] if str((1 << 32) + slot) != raw)
    pointers = list(re.finditer(r'DeviceBuffer\(DeviceBufferSpanV1 \{.*?\}\)', head[source]))
    assert len(pointers) >= 8
    bad = []
    for index, old, new in [
        (config, '75008', '75007'), (source, 'NullDevicePointer', pointers[0][0]),
        (source, 'grid_x: 8,', 'grid_x: 16,'), (source, 'grid_y: 2,', 'grid_y: 4,'),
        (source, 'flags: 1,', 'flags: 0,'), (source, 'raw: ' + raw, 'raw: ' + other),
        (source, 'cluster: LaunchClusterV1 { x: 1,', 'cluster: LaunchClusterV1 { x: 2,'),
        (merge, 'grid_z: 1,', 'grid_z: 2,'),
    ]:
        changed = head.copy(); changed[index] = changed[index].replace(old, new, 1)
        assert changed != head, 'mutation did not change its input'
        bad.append(changed)
    workspace = pointers[7]
    shifted = re.sub(r'offset: (\d+)', lambda match: f'offset: {int(match[1]) + 16}', workspace[0], count=1)
    changed = head.copy()
    changed[source] = head[source][:workspace.start()] + shifted + head[source][workspace.end():]
    assert changed != head
    bad.append(changed)
    scale = list(re.finditer(r'ScalarU32\((\d+)\)', head[source]))[5]
    changed = head.copy()
    changed[source] = head[source][:scale.start()] + f'ScalarU32({int(scale[1]) + 1})' + head[source][scale.end():]
    bad.append(changed)
    changed = head.copy(); del changed[resolve]; bad.append(changed)
    changed = head.copy(); changed.insert(source, 'release'); bad.append(changed)
    changed = head.copy(); changed[source], changed[merge] = changed[merge], changed[source]; bad.append(changed)
    for index, changed in enumerate(bad):
        try:
            audit(base, changed, base_meta, head_meta, symbols, False, True)
        except (AssertionError, KeyError, ValueError):
            continue
        raise AssertionError(f'mutation {index} was accepted')
    return len(bad)


def read(path):
    return path.read_text().splitlines(), json.loads(path.with_name(path.name + '.handles.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ['base', 'head', 'buckets', 'report']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    lock = Path(__file__).resolve().parents[1] / 'bundle/kernels.lock.d/latent_decode_staging.lock'
    module, = json.loads(lock.read_text())
    assert module['name'] == CUBIN
    symbols = [entry['symbol'] for entry in module['entries']]
    results = {}
    for suffix, pin in BASE_PINS.items():
        name = 'deepseek41-flash-' + suffix + '.journal'
        base, base_meta = read(args.base / name)
        head, head_meta = read(args.head / name)
        before = digest(base)
        assert (before['lines'], before['fnv']) == pin, 'baseline differs from the pinned reference'
        if suffix not in ['tp4.native', 'tp4.media']:
            assert base == head, ('unqualified topology/layout changed', name)
            assert CUBIN not in head_meta['modules'].values()
            results[name] = dict(identical=True, old=before, new=digest(head))
        else:
            results[name] = audit(base, head, base_meta, head_meta, symbols, True, False)
    name = 'deepseek41-flash-tp4.native.decode-buckets.'
    base, base_meta = read(args.buckets / (name + 'reference.journal'))
    head, head_meta = read(args.buckets / (name + 'serving.journal'))
    result = audit(base, head, base_meta, head_meta, symbols, False, True)
    controls = mutations(base, head, base_meta, head_meta, symbols)
    output = dict(passed=True, journals=results, measured_buckets=result, rejected_mutations=controls,
                  release_scope='Anonymous added releases are checked in final teardown, not identified individually.')
    args.report.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
