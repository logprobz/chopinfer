#!/usr/bin/env python3
"""Audit HC producer substitutions and wide-fold module lifetime additions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from audit_moe_direct import bind, RAW

OLD = 'chop_deepseek41_hc_round_expand_mix_partials_v1'
WARPS = {2: 8, 3: 8, 4: 8, 5: 4, 6: 4, 7: 4}


def inventory():
    locks = Path(__file__).resolve().parents[1] / 'bundle/kernels.lock.d'
    modules = []
    for family in ('mxfp8_fold_verify64', 'mxfp8_fold_verify_local64'):
        modules.extend(json.loads((locks / (family + '.lock')).read_text()))
    return {m['name'] for m in modules}, {e['symbol'] for m in modules for e in m['entries']}


def normalize_histories(base, head, old_handles, new_handles, closed=True):
    modules, symbols = inventory()
    old_modules = Counter(old_handles['modules'].values())
    new_modules = Counter(new_handles['modules'].values())
    assert new_modules - old_modules == Counter({m: 4 for m in modules})
    assert not old_modules - new_modules
    maps = {kind: {} for kind in ('stream', 'event', 'graph', 'function', 'buffer', 'gather')}
    old_buffers, new_buffers = old_handles['buffers'], new_handles['buffers']
    assert len(old_buffers) == len(new_buffers)
    for old, new in zip(old_buffers, new_buffers):
        assert old_handles['ranks'][str(old)] == new_handles['ranks'][str(new)]
        bind(maps['buffer'], old, new)
    # Input/output views supplied by the fake host are not owned allocations.
    # Their complete raw identities must stay unchanged across the histories.
    def external(lines, owned):
        return {int(raw) for line in lines for raw in RAW.findall(line)
                if int(raw) and (int(raw) & 0xffffffff) not in owned}

    external_buffers = external(base, set(old_buffers))
    assert external_buffers == external(head, set(new_buffers))

    extra = {int(line.split()[1]): line.split()[2] for line in head
             if line.startswith('resolve ') and line.split()[2] in symbols}
    assert Counter(extra.values()) == Counter({s: 4 for s in symbols})
    assert not any(line.startswith('resolve ') and line.split()[2] in symbols for line in base)
    configs = Counter()
    kept = []
    for line in head:
        fields = line.split()
        if fields[0] == 'resolve' and int(fields[1]) in extra:
            continue
        if fields[0] == 'configure' and int(fields[1]) in extra:
            symbol = extra[int(fields[1])]
            match = re.search(r'_cluster64_n(\d+)_c\d+s(\d+)_', symbol)
            assert match and symbol.endswith('_packed_v1'), line
            tile, stages = map(int, match.groups())
            assert int(fields[2]) == stages * (64 + tile) * 132 + 64 * 4
            configs[symbol] += 1
            continue
        kept.append(line)
    assert len(configs) == 3 and set(configs.values()) == {4}
    assert all('_n512_k5120_' not in s for s in configs)
    releases = 4 * (len(symbols) + len(modules)) if closed else 0
    assert head.count('release') - base.count('release') == releases
    a = [line for line in base if line != 'release']
    b = [line for line in kept if line != 'release']
    for kind, keyword, column in (
        ('stream', 'stream', 1), ('event', 'event', 1),
        ('graph', 'captured', 2), ('function', 'resolve', 1),
        ('function', 'configure', 1), ('gather', 'async mapped create', 3),
    ):
        left = [line.split() for line in a if line.startswith(keyword + ' ')]
        right = [line.split() for line in b if line.startswith(keyword + ' ')]
        assert len(left) == len(right), keyword
        for old, new in zip(left, right):
            if keyword in ('resolve', 'configure'):
                assert old[2:] == new[2:]
            bind(maps[kind], int(old[column]), int(new[column]))
    inverse = {kind: {v: k for k, v in values.items()} for kind, values in maps.items()}

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
        elif kind == 'release':
            change(2, fields[1])
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
            assert kind in ('allocate', 'state', 'serving'), kind

        def buffer(match):
            raw = int(match[1])
            if raw == 0:
                return 'raw: 0'
            if raw in external_buffers:
                return 'raw: ' + str(raw)
            generation, slot = raw >> 32, raw & 0xffffffff
            return 'raw: ' + str((generation << 32) | inverse['buffer'][slot])

        return RAW.sub(buffer, ' '.join(fields))

    info = dict(identical=False, old_lines=len(base), new_lines=len(head),
                retained_records=len(a), added_modules=8, added_resolves=len(extra),
                added_configures=sum(configs.values()), added_releases=releases,
                allocations_unchanged=True,
                handle_bijections={kind: len(values) for kind, values in maps.items()})
    return a, [normalize(line) for line in b], normalize, info


def audit(base, head, old_handles, new_handles):
    if base == head:
        assert old_handles == new_handles
        return dict(identical=True, lines=len(base))
    a, b, _, info = normalize_histories(base, head, old_handles, new_handles)
    assert len(a) == len(b), (len(a), len(b))
    changed = Counter()
    for at, (old, new) in enumerate(zip(a, b)):
        if old == new:
            continue
        assert old.startswith('launch ') and OLD + ' ' in old, (at, old, new)
        rows = int(re.search(r'grid_y: (\d+),', old)[1])
        warps = WARPS[rows]
        assert 'grid_x: 20,' in old and 'block_x: 256,' in old
        expected = old.replace(OLD, f'chop_deepseek41_hc_round_chain_w{warps}_v1', 1)
        expected = expected.replace('grid_x: 20,', 'grid_x: 160,', 1)
        expected = expected.replace('block_x: 256,', f'block_x: {warps * 32},', 1)
        assert expected == new, (at, expected, new)
        changed[rows] += 1
    return dict(info, hc_producers=dict(changed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    old = {path.name: path for path in args.base.glob('*.journal')}
    new = {path.name: path for path in args.head.glob('*.journal')}
    assert old.keys() == new.keys() and len(old) == 6
    results = {}
    controls = []
    for name in sorted(old):
        metadata = lambda path: json.loads(Path(str(path) + '.handles.json').read_text())
        a, b = old[name].read_text().splitlines(), new[name].read_text().splitlines()
        ah, bh = metadata(old[name]), metadata(new[name])
        results[name] = audit(a, b, ah, bh)
        if not results[name]['identical']:
            assert 'tp4' in name
        if name.endswith('tp4.native.journal'):
            for label, needle, before, after in (
                ('producer operand', 'hc_round_chain_w8_v1 Launch', 'offset: 0,', 'offset: 4,'),
                ('producer threads', 'hc_round_chain_w8_v1 Launch', 'block_x: 256,', 'block_x: 128,'),
                ('retained binding', 'shared_swiglu', 'offset: 0,', 'offset: 4,'),
            ):
                at = next(i for i, line in enumerate(b) if needle in line and before in line)
                bad = b.copy()
                bad[at] = bad[at].replace(before, after, 1)
                try:
                    audit(a, bad, ah, bh)
                except (AssertionError, KeyError, ValueError):
                    controls.append(label)
                else:
                    raise AssertionError('mutation escaped: ' + label)
            for label, bad in (
                ('missing release', b[:b.index('release')] + b[b.index('release') + 1:]),
                ('missing operation', b[:len(b) // 2] + b[len(b) // 2 + 1:]),
            ):
                try:
                    audit(a, bad, ah, bh)
                except (AssertionError, KeyError, ValueError):
                    controls.append(label)
                else:
                    raise AssertionError('mutation escaped: ' + label)
    assert sum(not value['identical'] for value in results.values()) == 2
    assert len(controls) == 5
    args.report.write_text(json.dumps(dict(passed=True, journals=results, negative_controls=controls), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
