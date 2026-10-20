#!/usr/bin/env python3
"""Audit retained prefill substitutions without dropping other operations."""
import argparse
import hashlib
import json
from pathlib import Path
import re

OLD = 'chop_latent_sources_deepseek41_h16_d512_mixed_normalized_s1_r64_prefill_reuse2_v1'
MERGE = OLD.removesuffix('_reuse2_v1') + '_merge_v1'
NEW = OLD.removesuffix('_reuse2_v1') + '_head_shared_finish_v1'
ARG = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{.*?\}\)|NullDevicePointer|Scalar(?:U32|U64|F32|I32)\([^)]*\)')


def arguments(line):
    tail = line.split(' } [', 1)[1].removesuffix(']')
    tokens = ARG.findall(tail)
    assert ', '.join(tokens) == tail
    return tokens


def plan(line, rows, heads, shared):
    expected = (f'LaunchConfigV1 {{ flags: 1, grid_x: {rows}, grid_y: {heads}, grid_z: 1, '
                f'block_x: 256, block_y: 1, block_z: 1, shared_memory_bytes: {shared}, '
                'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')
    assert line.split(' ', 3)[3].split(' [', 1)[0] == expected


def audit(old, new, expected):
    at = 0
    changed = 0
    for line in new:
        if at < len(old) and line == old[at]:
            at += 1
            continue
        assert line.startswith('launch ') and line.split()[2] == NEW, ('unexpected operation', at)
        body, merge = old[at:at+2]
        assert body.startswith('launch ') and body.split()[2] == OLD
        assert merge.startswith('launch ') and merge.split()[2] == MERGE
        assert line.split()[1] == body.split()[1] == merge.split()[1]
        rows = int(re.search(r'grid_x: (\d+)', line)[1])
        assert 64 <= rows <= 2496 and rows % 128 == 64
        plan(line, rows, 1, 83200)
        plan(body, rows // 2, 2, 85504)
        merge_plan = (f'LaunchConfigV1 {{ flags: 1, grid_x: {rows}, grid_y: 16, grid_z: 1, '
                      'block_x: 128, block_y: 1, block_z: 1, shared_memory_bytes: 0, '
                      'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')
        assert merge.split(' ', 3)[3].split(' [', 1)[0] == merge_plan
        args, final, fused = arguments(body), arguments(merge), arguments(line)
        assert len(args) == 25 and len(final) == 5 and len(fused) == 28
        assert final[:2] == [args[7], args[2]]
        args[7] = 'NullDevicePointer'
        assert fused == args + [final[3], final[4], final[2]], ('fused arguments', at)
        at += 2
        changed += 1
    assert at == len(old) and changed == expected, (at, len(old), changed, expected)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.base.glob('*.journal'))
    assert len(files) == 99 and {p.name for p in files} == {p.name for p in args.head.glob('*.journal')}
    results = {}
    control = None
    for path in files:
        current = args.head / path.name
        before, after = path.read_bytes(), current.read_bytes()
        assert json.loads(Path(str(path)+'.handles.json').read_text()) == json.loads(Path(str(current)+'.handles.json').read_text())
        offset = int(re.search(r'offset-(\d+).journal$', path.name)[1])
        expected = (8 if offset in (0, 24576) else 80) if '-skip-true-' in path.name and offset < 32768 else 0
        old, new = before.decode().splitlines(), after.decode().splitlines()
        if expected == 0:
            assert before == after, path.name
        changes = audit(old, new, expected)
        results[path.name] = dict(substitutions=changes, before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(), other_operations_exact=True, handles_exact=True)
        if expected and control is None:
            control = (old, new, expected)
    old, new, expected = control
    at = next(i for i, line in enumerate(new) if line.startswith('launch ') and line.split()[2] == NEW and 'grid_x: 192,' in line)
    mutants = []
    for text, replacement in [('grid_x: 192,', 'grid_x: 64,'), ('shared_memory_bytes: 83200,', 'shared_memory_bytes: 83216,'), ('NullDevicePointer', 'ScalarU64(0)')]:
        lines = new.copy()
        lines[at] = lines[at].replace(text, replacement, 1)
        assert lines != new
        mutants.append(lines)
    lines = new.copy()
    lines[at] = re.sub(r'offset: (\d+)', lambda m: f'offset: {int(m[1])+16}', lines[at], count=1)
    mutants.append(lines)
    lines = new.copy()
    tokens = arguments(lines[at])
    tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
    lines[at] = lines[at].split(' } [', 1)[0] + ' } [' + ', '.join(tokens) + ']'
    mutants.extend([lines, new+['release']])
    for i, mutant in enumerate(mutants):
        try:
            audit(old, mutant, expected)
        except (AssertionError, IndexError, ValueError):
            continue
        raise AssertionError(f'mutation {i} escaped')
    args.output.write_text(json.dumps(dict(passed=True, journals=results, rejected_mutations=len(mutants)), indent=2)+'\n')
    print(f'PASS retained_prefill_audit journals={len(files)} substitutions={sum(r["substitutions"] for r in results.values())} mutations={len(mutants)}')


if __name__ == '__main__':
    main()
