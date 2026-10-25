#!/usr/bin/env python3
"""Audit Verify-6/12 fold substitutions and all six pinned full histories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from audit_small_row_lifecycle import normalize_histories

ARG = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{.*? \}\)|NullDevicePointer|Scalar(?:U32|U64)\(\d+\)')
SPAN = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)')
NULL = 'NullDevicePointer'
QUANTIZE = 'chop_deepseek41_dense_ceil_w8192_cached_v1'
NEXT_QUANTIZE = 'chop_deepseek41_dense_ceil_f32_w2048_parallel_v1'
SLICED = 'chop_deepseek41_dense_ceil_w8192_sliced_s32w8_v1'
SHAPES = {'qa': (1280, 5120, 1), 'oa': (1024, 4096, 2), 'ob': (5120, 2048, 1)}
SLABS = {'qa': 3276800, 'oa': 4194304, 'ob': 5242880}


def prefetch(id):
    n, k, groups = SHAPES[id]
    return f'chop_deepseek41_mxfp8_prefetch_n{n}_k{k}_g{groups}_f32_v3'


def status(id):
    n, k, groups = SHAPES[id]
    tail = {'qa': 'c8s5_status', 'oa': 'c8s4_statusq', 'ob': 'c2s4_statusp'}[id]
    return f'chop_deepseek41_mxfp8_fused_n{n}_k{k}_g{groups}_f32_cluster16_n32_{tail}_v1_packed_v1'


QB = 'chop_deepseek41_mxfp8_stream_n8192_k1280_g1_f32_v1'
QB_STATUS = 'chop_deepseek41_mxfp8_fused_n8192_k1280_g1_f32_local32_n48_s4x_status_v1_packed_v1'


def launch(line):
    head, tail = line.split(' [', 1)
    args = ARG.findall(tail[:-1])
    assert ', '.join(args) == tail[:-1], line
    return head.split(' ', 3), args


def view(raw, offset, length):
    return f'DeviceBuffer(DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset}, length: {length} }})'


def config(x, y, z, block, shared=0, cluster=1):
    return (f'LaunchConfigV1 {{ flags: 1, grid_x: {x}, grid_y: {y}, grid_z: {z}, '
            f'block_x: {block}, block_y: 1, block_z: 1, shared_memory_bytes: {shared}, '
            f'cluster: LaunchClusterV1 {{ x: 1, y: 1, z: {cluster} }} }}')


def stripped(lines):
    records, markers, captured = [], [], []
    nodes = Counter()
    inside = False
    for line in lines:
        if line.startswith('capture '):
            assert not inside
            inside = True
            markers.append(line)
        elif line.startswith('captured '):
            assert inside
            inside = False
            head, count = line.rsplit(' ', 1)
            markers.append(head)
            nodes[line.split()[1]] += int(count)
        else:
            records.append(line)
            captured.append(inside)
    assert not inside
    return records, captured, markers, nodes


def body(old, new, rows, handles):
    if rows not in (6, 12):
        assert old == new, rows
        return dict(identical=True, lines=len(old))
    a, ac, am, an = stripped(old)
    b, bc, bm, bn = stripped(new)
    assert am == bm
    pending, changed, removed = {}, Counter(), Counter()
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            assert ac[i] == bc[j]
            i += 1
            j += 1
            continue
        ah, aa = launch(a[i])
        bh, ba = launch(b[j])
        assert ah[1] == bh[1]
        stream, symbol = ah[1:3]
        consumed = 1
        if symbol == QUANTIZE:
            assert ah[3] == config(rows, 1, 1, 1024)
            assert bh[2:] == [SLICED, config(rows, 32, 1, 256)] and aa == ba
            id = 'quantizer'
        elif symbol in (QB, QB + '_reuse32_n64_v1_admit_v1'):
            if rows == 6:
                assert symbol == QB and len(aa) == 15 and aa[12] == 'ScalarU64(0)'
                expected = aa[:8] + [aa[11], aa[10], aa[13], 'ScalarU32(0)']
            else:
                fh, fa = launch(a[i + 1])
                assert fh[1:3] == [stream, QB + '_reuse32_n64_v1_packed_v1']
                assert len(aa) == 7 and len(fa) == 12
                assert aa[:2] == fa[:2] and aa[2:4] == [fa[8], fa[9]]
                assert aa[5:] == ['ScalarU64(0)', f'ScalarU32({rows})']
                expected = fa[:8] + [aa[4], fa[9], fa[10], 'ScalarU32(0)']
                consumed = 2
            assert bh[2:] == [QB_STATUS, config(171, 1, 1, 192)] and ba == expected
            id = 'qb'
        else:
            id = next(key for key in SHAPES if symbol == prefetch(key) + '_initialize_v3')
            fh, fa = launch(a[i + 1])
            assert fh[1:3] == [stream, prefetch(id) + '_packed_v1']
            assert len(aa) == 7 and len(fa) == 12
            assert aa[:2] == fa[:2] and aa[2:4] == [fa[8], fa[9]]
            assert aa[5:] == ['ScalarU64(0)', f'ScalarU32({rows})']
            assert fa[4:6] == [NULL, NULL] and fa[7] == NULL and fa[10] == aa[6]
            raw, offset, _ = map(int, SPAN.fullmatch(aa[2]).groups())
            assert offset + SLABS[id] <= handles['ranks'][str(raw & 0xffffffff)][1]
            expected = fa[:4] + [fa[6], NULL, NULL, NULL, NULL,
                                  view(raw, offset, SLABS[id]), aa[3], aa[4], aa[5], aa[6]]
            consumed = 2
            if id == 'oa':
                qh, qa = launch(a[i + 2])
                assert qh[1:3] == [stream, NEXT_QUANTIZE]
                assert qa[0] == fa[6] and qa[3:] == aa[3:6]
                scale_raw, scale_offset, scale_bytes = map(int, SPAN.fullmatch(qa[2]).groups())
                scale_slot = scale_raw & 0xffffffff
                flags_slot = handles['buffers'][handles['buffers'].index(scale_slot) + 1]
                assert handles['ranks'][str(flags_slot)] == handles['ranks'][str(scale_slot)]
                flags = view((scale_raw & ~0xffffffff) | flags_slot, scale_offset, scale_bytes)
                assert stream not in pending
                pending[stream] = flags
                expected[5:8] = [qa[1], qa[2], flags]
                consumed = 3
            elif id == 'ob':
                expected[8] = pending.pop(stream)
            geometry = {'qa': config(40, 1, 8, 128, 31744, 8),
                        'oa': config(32, 1, 16, 128, 25408, 8),
                        'ob': config(160, 1, 2, 128, 25408, 2)}[id]
            assert bh[2:] == [status(id), geometry] and ba == expected, (id, ba, expected)
        assert all(value == bc[j] for value in ac[i:i + consumed])
        if bc[j]:
            removed[stream] += consumed - 1
        changed[id] += 1
        i += consumed
        j += 1
    assert i == len(a) and j == len(b) and not pending
    assert changed == Counter({id: 160 for id in ('qa', 'qb', 'oa', 'ob', 'quantizer')})
    assert an == bn + removed and sum(removed.values()) == (640 if rows == 6 else 800)
    return dict(identical=False, old_lines=len(old), new_lines=len(new), changed=dict(changed),
                removed_capture_nodes=sum(removed.values()), ordered_operands_checked=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    results, controls = {}, []
    for rows in (4, 6, 8, 12, 16, 32):
        name = f'verify.tp4.r{rows}.Native.journal'
        old, new = args.base / name, args.head / name
        metadata = lambda path: json.loads(Path(str(path) + '.handles.json').read_text())
        old_handles, new_handles = metadata(old), metadata(new)
        a, b, normalize, info = normalize_histories(
            Path(str(old) + '.full').read_text().splitlines(),
            Path(str(new) + '.full').read_text().splitlines(), old_handles, new_handles, closed=False)
        x = old.read_text().splitlines()
        y = [normalize(line) for line in new.read_text().splitlines()]
        assert a[:-len(x)] == b[:-len(y)], 'creation or prefill changed'
        results[rows] = dict(body(x, y, rows, old_handles), creation_and_prefill_equal=True, inventory=info)
        if rows == 6:
            at = next(i for i, line in enumerate(y) if status('oa') in line)
            for label, before, after in [('operand', 'offset: 0,', 'offset: 4,'),
                                          ('threads', 'block_x: 128,', 'block_x: 256,'),
                                          ('rows', 'ScalarU32(6)', 'ScalarU32(5)')]:
                bad = y.copy()
                assert before in bad[at]
                bad[at] = bad[at].replace(before, after, 1)
                try:
                    body(x, bad, rows, old_handles)
                except (AssertionError, KeyError, ValueError, StopIteration):
                    controls.append(label)
                else:
                    raise AssertionError('mutation escaped: ' + label)
    assert len(controls) == 3
    result = dict(passed=True, journals=results, negative_controls=controls)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
