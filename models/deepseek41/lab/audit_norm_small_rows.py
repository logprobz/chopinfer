#!/usr/bin/env python3
"""Audit small cached-norm substitutions while retaining every other record."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def pin(lines):
    value = 0xcbf29ce484222325
    for byte in '\n'.join(lines).encode():
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return len(lines), value


def audit(base, head, tp4):
    assert len(base) == len(head), 'lifecycle operation count changed'
    changed = Counter()
    for index, (old, new) in enumerate(zip(base, head)):
        expected = old
        if tp4 and old.startswith('launch '):
            for kind, warps in [('input', 32), ('query', 16)]:
                symbol = f'chop_norm_quantize_deepseek41_{kind}_v1'
                if old.split()[2] != symbol:
                    continue
                before = 'grid_x: 2, grid_y: 1, grid_z: 1, block_x: 256,'
                assert before in old, 'unexpected norm bucket in lifecycle'
                after = f'grid_x: 1, grid_y: 2, grid_z: 1, block_x: {warps * 32},'
                expected = old.replace(symbol, symbol.replace('_v1', f'_cached{warps}_v1'), 1)
                expected = expected.replace(before, after, 1)
                changed[kind] += 1
        assert new == expected, f'unreviewed journal difference at line {index + 1}'
    assert changed == (Counter(input=320, query=160) if tp4 else Counter()), changed
    return dict(changed=changed, records=len(base), unchanged=len(base) - sum(changed.values()))


def controls(base, head):
    at = next(i for i, line in enumerate(head) if '_input_cached32_v1 ' in line)
    for old, replacement in [('ScalarU64(0)', 'ScalarU64(1)'), ('flags: 1,', 'flags: 0,'),
                             ('block_x: 1024,', 'block_x: 512,')]:
        assert old in head[at]
        bad = head.copy()
        bad[at] = bad[at].replace(old, replacement, 1)
        try:
            audit(base, bad, True)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted a mutated operand or launch configuration')
    bad = head.copy()
    del bad[next(i for i, line in enumerate(bad) if line.startswith('allocate '))]
    try:
        audit(base, bad, True)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted a missing lifecycle allocation')
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--head', required=True, type=Path)
    parser.add_argument('--base-revision', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    source = subprocess.check_output([
        'git', 'show', args.base_revision + ':models/deepseek41/bundle/src/fake_host_tests.rs',
    ], text=True)
    table = source.split('const FLASH_JOURNALS:', 1)[1].split('];', 1)[0]
    pins = {(name, layout.lower()): (int(count), int(digest, 16))
            for name, layout, count, digest in re.findall(
                r'"([^"]+)",\s*KvStorage::(\w+),\s*(?://[^\n]*\n\s*)*\((\d+), 0x([a-f0-9]+)\)', table)}
    assert len(pins) == 6, 'baseline source does not declare all six lifecycle pins'
    result = {'passed': True, 'journals': {}, 'negative_controls': 0}
    for ranks in [1, 2, 4]:
        for layout in ['native', 'bf16']:
            name = f'deepseek41-flash-tp{ranks}.{layout}.journal'
            old, new = [(root / name).read_text().splitlines() for root in [args.base, args.head]]
            before, after = pin(old), pin(new)
            assert before == pins[(f'deepseek41-flash-tp{ranks}', layout)], 'baseline pin differs'
            assert (args.base / (name + '.handles.json')).read_bytes() == (args.head / (name + '.handles.json')).read_bytes(), 'handles or resources changed'
            record = audit(old, new, ranks == 4)
            record.update(base_pin=[before[0], f'{before[1]:016x}'], head_pin=[after[0], f'{after[1]:016x}'],
                          base_sha256=hashlib.sha256((args.base / name).read_bytes()).hexdigest(),
                          head_sha256=hashlib.sha256((args.head / name).read_bytes()).hexdigest())
            result['journals'][name] = record
            if ranks == 4:
                result['negative_controls'] += controls(old, new)
    assert result['negative_controls'] == 8
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: six pinned lifecycle histories, only 320 input and 160 query substitutions per TP4 history; eight mutation controls')


if __name__ == '__main__':
    main()
