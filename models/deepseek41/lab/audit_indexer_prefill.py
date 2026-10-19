#!/usr/bin/env python3
"""Audit the bounded prefill journal change without ignoring any other bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

SPAN = re.compile(r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}')


def fnv(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return value


def spans(line):
    return [tuple(map(int, m.groups())) for m in SPAN.finditer(line)]


def span_text(span):
    b, offset, length = span
    return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {b} }}, offset: {offset}, length: {length} }}'


def replace_span(line, i, value):
    matches = list(SPAN.finditer(line))
    match = matches[i]
    return line[:match.start()] + span_text(value) + line[match.end():]


def scalar(line, bits, i, before, after):
    matches = list(re.finditer(rf'ScalarU{bits}\((\d+)\)', line))
    match = matches[i]
    assert int(match.group(1)) == before, (bits, i, before, line[:100])
    return line[:match.start()] + f'ScalarU{bits}({after})' + line[match.end():]


def packing(rows):
    codes = (rows * 32 * 513 * 8 + 255) & ~255
    scales = codes + rows * 2048
    status = scales + rows * 128
    return codes, scales, status, status + rows * 4


def candidate(rows, width):
    blocks = rows * ((width + 7) // 8) * 4
    offset = (blocks + 7) & ~7
    return offset, offset + rows * 32 * 2049 * 8


def audit(old, new, ranks, bf16=False):
    if bf16:
        assert old == new, 'BF16 journal changed'
        return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new).encode()):016x}', identical=True)
    visible_buffers = {spans(line)[1][0] for line in old if line.startswith('launch ')
                       and 'chop_candidate_block_ids_deepseek41_scores_v3_valid_v1 ' in line}
    digest = f'{fnv(b"".join(struct.pack("<I", n) for n in range(8193))):016x}'
    tables = {}
    stripped = []
    first_index = next(i for i, line in enumerate(new) if line.startswith('launch ') and 'chop_position_indexer_' in line)
    for i, line in enumerate(new):
        b = spans(line) if line.startswith('upload ') else []
        if b and b[0][1:] == (520, 32772):
            assert line.endswith(' ' + digest) and len(b) == 1 and i < first_index
            stream = int(line.split()[1])
            assert stream not in tables
            tables[stream] = b[0][0]
        else:
            stripped.append(line)
    assert len(tables) == ranks and len(stripped) == len(old)
    capturing = set()
    maximum = {}
    changed = allocations = 0
    for line, observed in zip(old, stripped):
        if line.startswith('capture '): capturing.add(int(line.split()[1]))
        if line.startswith('captured '): capturing.remove(int(line.split()[1]))
        if line.startswith('upload '):
            b = spans(line)
            if b and b[0][0] in visible_buffers:
                assert b[0][1] == 0 and b[0][2] % 4 == 0
                n = b[0][2] // 4
                # These pinned lifecycle/media fixtures have contiguous
                # prefills beginning at 0 or 64. Decode uploads may be ragged.
                starts = [start for start in (0, 64) if line.endswith(' ' + f'{fnv(b"".join(struct.pack("<i", p + 1) for p in range(start, start+n))):016x}')]
                maximum[int(line.split()[1])] = starts[0] + n if starts else None
        if line == observed:
            continue
        if line == 'allocate 520 256':
            assert observed == 'allocate 33292 256'
            allocations += 1
            continue
        assert line.startswith('launch '), (line[:120], observed[:120])
        stream = int(line.split()[1])
        assert stream not in capturing, 'decode/verify graph changed'
        symbol = line.split()[2]
        dense = symbol.startswith('chop_position_indexer_deepseek41_')
        producer = symbol.startswith('chop_candidate_block_ids_deepseek41_')
        assert dense or producer, symbol
        assert maximum.get(stream) is not None, 'unrecognized prefill metadata'
        pack = '_pack_fp4_guard_v1' in symbol
        rows = int(re.search(r'grid_' + ('x' if pack else 'y') + r': (\d+)', line)[1])
        assert 1 <= rows <= 64
        ratio = 2 if '_r2_' in symbol else 1
        page = 64 // ratio
        width = ((max(1, maximum[stream] // ratio) + page - 1) // page) * page
        expected = line
        b = spans(line)
        if dense:
            score = '_scores_' in symbol and not pack
            assert pack or score or '_select_' in symbol
            live_index, work_index = (1, 2) if pack else (7, 9) if score else (4, 7)
            assert b[live_index][0] == tables[stream] and b[live_index][2] == 4
            assert b[live_index][1] in (0, 4)
            expected = replace_span(expected, live_index, (b[live_index][0], 520 + rows*4, 4))
            old_layout, new_layout = packing(64), packing(rows)
            assert b[work_index][1:] == (0, old_layout[3])
            expected = replace_span(expected, work_index, (b[work_index][0], 0, new_layout[3]))
            if not pack:
                expected = scalar(expected, 32, 0, (1 << 20) // ratio, width)
            if pack or score:
                for j in range(3):
                    expected = scalar(expected, 64, j + (2 if score else 0), old_layout[j], new_layout[j])
        else:
            score = '_scores_' in symbol
            assert score or '_gather_' in symbol
            live_index = 2 if score else 1
            assert b[live_index][0] == tables[stream] and b[live_index][2] == 4
            assert b[live_index][1] in (0, 4)
            expected = replace_span(expected, live_index, (b[live_index][0], 520 + rows*4, 4))
            old_layout, new_layout = candidate(64, 1 << 20), candidate(rows, width)
            for i in ([3] if score else [0, 9]):
                assert b[i][1:] == (0, old_layout[1])
                expected = replace_span(expected, i, (b[i][0], 0, new_layout[1]))
            expected = scalar(expected, 32, 0, 1 << 20, width)
            if not score:
                expected = scalar(expected, 64, 0, old_layout[0], new_layout[0])
        assert expected == observed, f'unexpected binding change: {symbol}'
        changed += 1
    assert not capturing and allocations == ranks and changed > 0
    return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new).encode()):016x}', changed_launches=changed,
                count_tables=ranks, extra_metadata_bytes_per_rank=32772, all_other_bytes_identical=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = {}
    paths = sorted(args.baseline.glob('*.journal'))
    assert len(paths) == 8
    for path in paths:
        old = path.read_text().splitlines()
        current = args.candidate / path.name
        new = current.read_text().splitlines()
        ranks = int(re.search(r'tp(\d+)', path.name)[1])
        result = audit(old, new, ranks, '.bf16.' in path.name)
        # Independent mutations must be refused before any new digest is used.
        controls = 0
        for needle, replacement in [('flags: 1', 'flags: 0'), ('ScalarU32(512)', 'ScalarU32(511)'),
                                    ('_mma_fp4_guard_v1 ', '_mma_fp4_v1 ')]:
            at = next((i for i, s in enumerate(new) if s.startswith('launch ') and needle in s), None)
            if at is None: continue
            wrong = new.copy()
            wrong[at] = wrong[at].replace(needle, replacement, 1)
            try:
                audit(old, wrong, ranks, '.bf16.' in path.name)
            except AssertionError:
                controls += 1
            else:
                raise AssertionError(f'accepted mutation {path.name}: {needle}')
        assert controls >= 2
        result.update(negative_controls=controls, sha256=hashlib.sha256(current.read_bytes()).hexdigest())
        report[path.name] = result
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
