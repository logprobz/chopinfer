"""Audit the vector gate binding, leaving all unrelated operations unchanged."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

from audit_prefill_gate_journals import ARG, args, normalized, shared

STAGES = ('pack', 'predecode', 'r8', 'r32', 'r64', 'r128')


def logical(stream, experts, handles):
    def span(value):
        if value is None:
            return 'NullDevicePointer'
        _, handle, offset, length = value
        return ('DeviceBuffer(DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: '
                f'{handle} }}, offset: {offset}, length: {length} }})')
    return f'launch {stream} logical_gate_e{experts} ' + ' '.join(map(span, handles))


def collapse(lines, rows):
    output = []
    replacements = Counter()
    index = 0
    while index < len(lines):
        line = lines[index]
        vector = re.search(r'launch (\d+) chop_exact_prefill_e(384|128)_before_gateup_decoded128_pack_vector_v1 ', line)
        column = re.search(r'launch (\d+) \S*before_gateup_e(384)k6_n1152_k5120_r64_t8192_clamp4120000041200000_v1 ', line)
        column = column or re.search(r'launch (\d+) \S*before_gateup_e(128)k3_n1152_k5120_r64_t8192_clamp4120000041200000_v1 ', line)
        grouped = re.search(r'launch (\d+) (\S*before_gateup_e(128)k3_\S*_prefill)_pack_v1 ', line)
        if vector:
            stream, experts = vector[1], int(vector[2])
            assert rows == 8192 or (rows == 4096 and experts == 128)
            group = lines[index:index+6]
            assert len(group) == 6
            for position, (record, stage) in enumerate(zip(group, STAGES)):
                assert record.startswith(f'launch {stream} chop_exact_prefill_e{experts}_before_gateup_decoded128_{stage}_vector_v1 ')
                grid = 1 if stage == 'pack' else 376
                threads = 32 if stage == 'pack' else 256 if stage in ('predecode', 'r128') else 128
                smem = 0 if position < 2 else shared(int(stage[1:]))
                assert f'grid_x: {grid}, grid_y: 1, grid_z: 1,' in record
                assert f'block_x: {threads}, block_y: 1, block_z: 1,' in record
                assert f'shared_memory_bytes: {smem},' in record and 'flags: 1,' in record
            pack, decode = map(args, group[:2])
            compute = [args(r) for r in group[2:]]
            assert len(pack) == 5 and len(decode) == 4
            assert all(c == compute[0] for c in compute)
            c = compute[0]
            assert len(c) == 11 and c[-1] is None
            queue, decoded = pack[4], decode[3]
            queue_bytes = 37120 if experts == 384 else 15616
            assert queue[3] == queue_bytes + rows*5120*2
            assert decoded == ('span', queue[1], queue[2]+queue_bytes, rows*5120*2)
            assert c[0] == decoded and c[7] == queue and c[3] == decode[1]
            assert pack[3] == decode[2] == c[6]
            # Mutable queue and conversion storage must not overlap operands.
            for operand in [*pack[:4], *decode[:3], *c[1:7], *c[8:10]]:
                assert operand[1] != queue[1] or operand[2]+operand[3] <= queue[2] or operand[2] >= queue[2]+queue[3]
            h = [decode[0], c[1], c[2], decode[1], pack[1], c[4], c[5], c[6], c[8], c[9]]
            output.append(logical(stream, experts, h))
            replacements[experts] += 1
            index += 6
            continue
        if column and rows == 8192:
            h = args(line)
            assert len(h) == 10
            output.append(logical(column[1], int(column[2]), h))
            replacements[int(column[2])] += 1
            index += 1
            continue
        if grouped and rows == 4096:
            stream, prefix, experts = grouped[1], grouped[2], int(grouped[3])
            group = lines[index:index+4]
            assert len(group) == 4
            for record, stage in zip(group, ('pack', 'r8', 'r32', 'r64')):
                assert record.startswith(f'launch {stream} {prefix}_{stage}_v1 ')
            pack = args(group[0])
            compute = [args(record) for record in group[1:]]
            assert len(pack) == 5 and all(c == compute[0] for c in compute)
            c = compute[0]
            assert len(c) == 10 and c[7] == pack[4] and c[6] == pack[3]
            h = [c[0], c[1], c[2], c[3], pack[1], c[4], c[5], c[6], c[8], c[9]]
            output.append(logical(stream, experts, h))
            replacements[experts] += 1
            index += 4
            continue
        output.append(line)
        index += 1
    mapped, live = {}, set()
    for i, line in enumerate(output):
        match = re.fullmatch(r'async mapped (create|release) (\d+)', line)
        if match:
            action, handle = match.groups()
            if action == 'create':
                assert handle not in mapped
                mapped[handle] = len(mapped)
                live.add(handle)
            else:
                assert handle in live
                live.remove(handle)
            output[i] = f'async mapped {action} {mapped[handle]}'
    assert not live
    return normalized(output)[0], replacements


def expected_exports():
    directory = Path(__file__).resolve().parent/'../bundle/kernels.lock.d'
    result = Counter()
    for family in ('exact_moe_prefill', 'exact_moe_prefill_vector'):
        for module in json.loads((directory/f'{family}.lock').read_text()):
            result.update({entry['symbol']: 4 for entry in module['entries']})
    return result


def audit(old, new, rows):
    a, b = old.read_text().splitlines(), new.read_text().splitlines()
    resolved = lambda lines: Counter(l.split()[2] for l in lines if l.startswith('resolve '))
    ac, bc = resolved(a), resolved(b)
    expected = expected_exports() if 'tp4' in old.name else Counter()
    assert bc == ac+expected, old.name
    functions = {l.split()[1]: l.split()[2] for l in b if l.startswith('resolve ')}
    configs = Counter((functions[l.split()[1]], int(l.split()[2])) for l in b
                      if l.startswith('configure ') and functions[l.split()[1]].startswith('chop_exact_prefill_'))
    banks = (128, 384) if '-dspark-' in old.name else (384,)
    required = Counter({(f'chop_exact_prefill_e{e}_before_gateup_decoded128_r{r}_vector_v1', shared(r)): 4
                        for e in banks for r in (64, 128)}) if expected else Counter()
    assert configs == required, configs
    # All captured journals include instance destruction and both added modules.
    assert b.count('release')-a.count('release') == sum(expected.values())+(8 if expected else 0)
    left, n = collapse(a, rows)
    right, m = collapse(b, rows)
    assert n == m, (old.name, n, m)
    assert left == right, (old.name, next(((i, str(x)[:400], str(y)[:400]) for i, (x, y) in enumerate(zip(left, right)) if x != y), ('length', len(left), len(right))))
    expected_delta = sum((2 if rows == 4096 else 5)*count for count in m.values())
    assert sum(l.startswith('launch ') for l in b)-sum(l.startswith('launch ') for l in a) == expected_delta
    return dict(journal=old.name, gates=dict(n), extra_functions=sum(expected.values()),
                configured_vector_functions=sum(configs.values()),
                all_other_records_equal=True, input_output_bindings_equal=True, scratch_disjoint=True)


def run(root):
    results = []
    for directory in ('journals', 'prefill'):
        old_dir, new_dir = root/f'baseline-{directory}', root/f'serving-{directory}'
        assert {p.name for p in old_dir.glob('*.journal')} == {p.name for p in new_dir.glob('*.journal')}
        for old in sorted(old_dir.glob('*.journal')):
            rows = int(old.stem.split('-')[-1]) if directory == 'prefill' else 0
            results.append(audit(old, new_dir/old.name, rows))
    assert len(results) == 14, len(results)
    source = root/'serving-prefill/tp4-native-dspark-8192.journal'
    old = root/'baseline-prefill'/source.name
    lines = source.read_text().splitlines()
    reference = collapse(old.read_text().splitlines(), 8192)[0]
    controls = 0
    # Current full-instance prefills execute the backbone E384 MoE. DSpark
    # catch-up projects features and KV; E128 gate bindings are audited by
    # the typed launch test for both qualified prefill buckets.
    for experts in (384,):
        start = next(i for i, l in enumerate(lines) if f'chop_exact_prefill_e{experts}_before_gateup_decoded128_pack_vector_v1 ' in l)
        for stage in range(6):
            for field in ('grid_x:', 'block_x:', 'shared_memory_bytes:'):
                changed = lines.copy()
                changed[start+stage] = re.sub(re.escape(field)+r' (\d+)', lambda m: field+' '+str(int(m[1])+1), changed[start+stage], count=1)
                try:
                    collapse(changed, 8192)
                except AssertionError:
                    controls += 1
                else:
                    raise AssertionError('launch geometry control escaped')
        for operand in (0, 1, 2, 3, 4, 5, 6, 8, 9):
            changed = lines.copy()
            for position in range(start+2, start+6):
                match = list(ARG.finditer(changed[position]))[operand]
                begin, end = match.span()
                value = match[0].replace('offset: '+match[2]+',', 'offset: '+str(int(match[2])+16)+',')
                changed[position] = changed[position][:begin]+value+changed[position][end:]
            try:
                assert collapse(changed, 8192)[0] == reference
            except AssertionError:
                controls += 1
            else:
                raise AssertionError('operand control escaped')
    changed = lines.copy()
    mapped_release = next(i for i,l in enumerate(changed) if l.startswith('async mapped release '))
    changed[mapped_release] = 'async mapped release '+str(int(changed[mapped_release].split()[-1])+1)
    try:
        collapse(changed,8192)
    except AssertionError:
        controls += 1
    else:
        raise AssertionError('mapped gather release control escaped')
    result = dict(all_passed=True, pairs=results, negative_controls=controls)
    (root/'vector-journal-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Audited {len(results)} complete journal pairs and {controls} mutation controls')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: audit_vector_gate_journals.py EXTERNAL_AUDIT_ROOT')
    run(Path(sys.argv[1]))
