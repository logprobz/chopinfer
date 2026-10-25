#!/usr/bin/env python3
"""Audit unused retention declarations and the six unchanged verify bodies."""
import argparse
from collections import Counter
import copy
import difflib
import json
from pathlib import Path
import re

MODULE = 'deepseek41_dspark_retain_rows_sm120'
SYMBOL = 'chop_deepseek41_dspark_retain_rows_v1'
ROWS = (4, 6, 8, 12, 16, 32)
RAW = re.compile(r'(BundleDeviceBufferHandleV1 \{ raw: )(\d+)( \})')


def bind(mapping, old, new):
    assert old not in mapping and new not in mapping.values(), 'aliased handle'
    mapping[old] = new


def fnv(lines):
    value = 0xcbf29ce484222325
    for byte in '\n'.join(lines).encode():
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return f'0x{value:016x}'


def normalizer(base, head, old, new):
    """Map identities from allocation metadata and ordered creation records."""
    modules = [sorted(m['modules'].items(), key=lambda pair: int(pair[0]))
               for m in (old, new)]
    assert Counter(new['modules'].values()) - Counter(old['modules'].values()) == Counter({MODULE: 4})
    assert not Counter(old['modules'].values()) - Counter(new['modules'].values())
    retained_modules = [pair for pair in modules[1] if pair[1] != MODULE]
    assert [name for _, name in modules[0]] == [name for _, name in retained_modules]
    assert len(old['buffers']) == len(new['buffers'])
    maps = {kind: {} for kind in ('buffer', 'module', 'stream', 'event', 'graph', 'function', 'gather')}
    for (left, _), (right, _) in zip(modules[0], retained_modules):
        bind(maps['module'], int(left), int(right))
    for left, right in zip(old['buffers'], new['buffers']):
        assert old['ranks'][str(left)] == new['ranks'][str(right)], 'allocation rank or size changed'
        bind(maps['buffer'], left, right)

    def external(lines, owned):
        return {int(match[2]) for line in lines for match in RAW.finditer(line)
                if int(match[2]) and (int(match[2]) & 0xffffffff) not in owned}

    external_buffers = external(base, set(old['buffers']))
    assert external_buffers == external(head, set(new['buffers'])), 'external buffer identities changed'
    extras = [line for line in head if line.startswith('resolve ') and line.split()[2] == SYMBOL]
    assert len(extras) == 4 and len(set(extras)) == 4
    assert not any(SYMBOL in line for line in base)
    assert not any(line.startswith('launch ') and SYMBOL in line for line in head), 'retention route ran'
    extra_ids = {line.split()[1] for line in extras}
    assert not any(line.startswith('configure ') and line.split()[1] in extra_ids for line in head)
    kept = [line for line in head if line not in set(extras)]
    assert len(base) == len(kept), 'an undeclared operation was added or removed'
    for kind, prefix, column in (
        ('stream', 'stream ', 1), ('event', 'event ', 1),
        ('graph', 'captured ', 2), ('function', 'resolve ', 1),
        ('gather', 'async mapped create ', 3),
    ):
        left = [line.split() for line in base if line.startswith(prefix)]
        right = [line.split() for line in kept if line.startswith(prefix)]
        assert len(left) == len(right), prefix
        for a, b in zip(left, right):
            if kind == 'function':
                assert a[2:] == b[2:], 'resolved symbol order changed'
            bind(maps[kind], int(a[column]), int(b[column]))
    inverse = {kind: {right: left for left, right in mapping.items()}
               for kind, mapping in maps.items()}

    def normalize(line):
        fields = line.split(' ')
        kind = fields[0]

        def change(at, category):
            fields[at] = str(inverse[category][int(fields[at])])

        if kind in ('stream', 'event'):
            change(1, kind)
        elif kind in ('capture', 'synchronize', 'zero', 'download', 'copy', 'launch'):
            change(1, 'stream')
        elif kind == 'upload':
            if fields[1] != 'arena':
                change(1, 'stream')
        elif kind in ('record', 'wait'):
            change(1, 'stream')
            change(2, 'event')
        elif kind in ('captured', 'replay'):
            change(1, 'stream')
            change(2, 'graph')
        elif kind in ('resolve', 'configure'):
            change(1, 'function')
        elif kind == 'async':
            assert fields[1] == 'mapped'
            if fields[2] in ('create', 'release'):
                change(3, 'gather')
            elif fields[2] == 'wait':
                change(3, 'stream')
            else:
                assert fields[2] in ('ids', 'complete')
        elif kind == 'mapped':
            assert fields[1] == 'gather'
        else:
            assert kind in ('allocate', 'state', 'serving', 'release'), kind

        def buffer(match):
            raw = int(match[2])
            if raw == 0 or raw in external_buffers:
                return match[0]
            generation, slot = raw >> 32, raw & 0xffffffff
            return match[1] + str((generation << 32) | inverse['buffer'][slot]) + match[3]

        return RAW.sub(buffer, ' '.join(fields))

    return normalize, kept, maps


def load(directory, rows):
    path = directory / f'verify.tp4.r{rows}.Native.journal'
    return (path.read_text().splitlines(), Path(str(path) + '.full').read_text().splitlines(),
            json.loads(Path(str(path) + '.handles.json').read_text()))


def audit(base, head, output, rows):
    body_a, full_a, handles_a = load(base, rows)
    body_b, full_b, handles_b = load(head, rows)
    assert full_a[-len(body_a):] == body_a and full_b[-len(body_b):] == body_b
    normalize, kept, maps = normalizer(full_a, full_b, handles_a, handles_b)
    normalized = [normalize(line) for line in kept]
    diff = list(difflib.unified_diff(full_a, normalized, fromfile='base', tofile='normalized-head', lineterm=''))
    (output / f'r{rows}.diff').write_text('\n'.join(diff))
    (output / f'r{rows}.maps.json').write_text(json.dumps(maps, indent=2) + '\n')
    assert not diff, f'row {rows}: unexplained full-history difference'
    assert body_a == [normalize(line) for line in body_b], 'verify body changed'
    return dict(rows=rows, verify_lines=len(body_a), full_lines=len(full_a),
                old_fnv=fnv(body_a), new_fnv=fnv(body_b),
                added_modules=4, added_resolves=4, allocations_unchanged=True,
                verify_body_identical=True, retained_full_history_identical=True,
                handle_bijections={kind: len(mapping) for kind, mapping in maps.items()})


def controls(base, head):
    body_a, full_a, handles_a = load(base, 4)
    body_b, full_b, handles_b = load(head, 4)
    normalize, _, _ = normalizer(full_a, full_b, handles_a, handles_b)
    launch = next(i for i, line in enumerate(body_b) if line.startswith('launch '))
    refused = []
    for label, pattern in [('grid', r'grid_x: (\d+)'), ('threads', r'block_x: (\d+)'),
                           ('flags', r'flags: (\d+)'), ('offset', r'offset: (\d+)'),
                           ('length', r'length: (\d+)'), ('scalar', r'ScalarU32\((\d+)\)')]:
        changed = body_b.copy()
        at = next(i for i, line in enumerate(changed) if line.startswith('launch ') and re.search(pattern, line))
        match = re.search(pattern, changed[at])
        changed[at] = changed[at][:match.start(1)] + str(int(match[1]) + 1) + changed[at][match.end(1):]
        assert [normalize(line) for line in changed] != body_a, label
        refused.append(label)
    for label in ('symbol', 'buffer-alias', 'order', 'capture-count', 'missing-operation'):
        changed = body_b.copy()
        if label == 'symbol':
            parts = changed[launch].split(' ');parts[2] += '_wrong';changed[launch] = ' '.join(parts)
        elif label == 'buffer-alias':
            at = next(i for i, line in enumerate(changed) if line.startswith('launch ') and RAW.search(line))
            match = RAW.search(changed[at]);raw = int(match[2])
            slot = next(value for value in handles_b['buffers'] if value != (raw & 0xffffffff))
            changed[at] = changed[at][:match.start(2)] + str((raw & ~0xffffffff) | slot) + changed[at][match.end(2):]
        elif label == 'order':
            other = next(i for i in range(launch + 1, len(changed)) if changed[i] != changed[launch])
            changed[launch], changed[other] = changed[other], changed[launch]
        elif label == 'capture-count':
            at = next(i for i, line in enumerate(changed) if line.startswith('captured '))
            parts = changed[at].split(' ');parts[3] = str(int(parts[3]) + 1);changed[at] = ' '.join(parts)
        else:
            del changed[launch]
        assert [normalize(line) for line in changed] != body_a, label
        refused.append(label)
    bad = copy.deepcopy(handles_b)
    bad['ranks'][str(bad['buffers'][0])][1] += 4
    try:
        normalizer(full_a, full_b, handles_a, bad)
    except AssertionError:
        refused.append('allocation-size')
    else:
        raise AssertionError('changed allocation accepted')
    assert len(refused) == 12
    return refused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [audit(args.base, args.head, args.output, rows) for rows in ROWS]
    report = dict(passed=True, journals=results, negative_controls=controls(args.base, args.head))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
