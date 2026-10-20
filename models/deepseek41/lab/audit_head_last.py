"""Audit ordered head replacements and retained graph records before pin updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_decode_attention import digest, read
from audit_selected_cache import canonical, graph_groups, shape, VIEW

CUBIN = 'f32_decode_last_deepseek41_sm120'
HEAD = 'chop_deepseek41_head_f32_m8192_n32320_k5120_v1_cached'
ATOM = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{.*?\}\)|ScalarU32\(\d+\)')


def arguments(line):
    body = line.split(' [', 1)[1][:-1]
    args = ATOM.findall(body)
    assert ', '.join(args) == body, 'unknown or malformed argument'
    return args


def original_pair(prepare, compute, meta):
    pa, ca = arguments(prepare), arguments(compute)
    assert len(pa) == 4 and len(ca) == 7
    rows = int(re.fullmatch(r'ScalarU32\((\d+)\)', pa[2])[1])
    assert rows in (1, 8, 32), 'unmeasured head row bucket'
    tile = 32 if rows == 32 else 8
    symbol = HEAD+f'_decode_v1_last_m{tile}_s1_p1_r0_v1'
    assert prepare.split()[2] == symbol+'_prepare_v1'
    assert compute.split()[2] == symbol
    assert prepare.split()[1] == compute.split()[1], 'head stream changed'
    shape(prepare, (rows, 1, 1), 0)
    shape(compute, (20200, 1, 1), 0)
    assert ca[:6] == [pa[0], ca[1], pa[3], pa[1], pa[2], ca[5]], 'head operand binding changed'
    views = [VIEW.fullmatch(arg) for arg in (ca[0], ca[1], ca[2], ca[3], ca[5], ca[6])]
    assert all(views)
    x, weight, work, live, domain, output = [tuple(map(int, m.groups())) for m in views]
    required = ((128+40*rows*32320*4+255)//256)*256+8080
    assert work[2] >= required, 'counter workspace is short'
    assert work[0] not in (x[0], weight[0], live[0], domain[0], output[0]), 'counter workspace aliases an operand'
    rank, allocated = meta['ranks'][str(work[0]-(1 << 32))]
    assert 0 <= rank < 4 and work[1]+work[2] <= allocated
    config = prepare.split(' LaunchConfigV1 ', 1)[1].split(' [', 1)[0]
    def launch(name, grid, args):
        cfg = config
        for field, value in zip(('grid_x', 'grid_y', 'grid_z'), grid):
            cfg = re.sub(r'\b'+field+r': \d+', field+': '+str(value), cfg)
        return f'launch {prepare.split()[1]} {name} LaunchConfigV1 {cfg} ['+', '.join(args)+']'
    return [launch(HEAD+'_rows_v1', (rows, 1, 1), pa),
            launch(HEAD+'_decode_v1'+('_reuse32_v1' if rows == 32 else ''), (2020, 1, 10), ca[:6]),
            launch(HEAD+'_reduce_v1', ((rows*32320+127)//128, 1, 1), [ca[2], ca[6], ca[3], ca[4]])], rows


def audit(before, after, om, nm, symbols, added):
    original_before, original_after = before, after
    before, old_graphs = graph_groups(before)
    after, new_graphs = graph_groups(after)
    assert [(g['mode'], len(g['pieces'])) for g in old_graphs] == [(g['mode'], len(g['pieces'])) for g in new_graphs]
    assert len(symbols) == 24
    resolves = [l for l in after if l.startswith('resolve ') and l.split()[2] in symbols]
    assert Counter(l.split()[2] for l in resolves) == Counter({s:4 for s in symbols})
    functions = {l.split()[1]:l.split()[2] for l in after if l.startswith('resolve ')}
    assert not any(l.startswith('configure ') and functions[l.split()[1]] in symbols for l in after)
    if added:
        assert Counter(nm['modules'].values()) == Counter(om['modules'].values())+Counter({CUBIN:4})
    else:
        assert om == nm, 'paired resources changed'
    assert [om['ranks'][str(s)] for s in om['buffers']] == [nm['ranks'][str(s)] for s in nm['buffers']], 'allocation rank, size or order changed'
    kept = []
    changed = Counter()
    at = 0
    while at < len(after):
        line = after[at]
        at += 1
        if added and line in resolves:
            continue
        if line.startswith('launch ') and '_last_m' in line:
            assert at < len(after)
            restored, rows = original_pair(line, after[at], nm)
            at += 1
            kept.extend(restored)
            changed[rows] += 1
        else:
            kept.append(line)
    assert changed, 'no selected head executed'
    releases = 4*(len(symbols)+1) if added else 0
    if releases:
        assert kept[-releases:] == ['release']*releases
        kept = kept[:-releases]
    # Paired graphs load the same resources, so compare their raw records,
    # including asynchronous mapped gathers. Only the lifecycle comparison
    # needs to translate handles after the additional module resolves.
    a = list(canonical(before, om)) if added else before
    b = list(canonical(kept, nm)) if added else kept
    assert len(a) == len(b), ('retained record count', len(a), len(b))
    for i, (old, new) in enumerate(zip(a,b)):
        assert old == new, ('retained operation differs', i, old, new)
    return dict(old=digest(original_before), new=digest(original_after), replacements=dict(changed),
                added_resolves=len(resolves) if added else 0, added_releases=releases,
                added_modules=4 if added else 0, removed_launches=sum(changed.values()),
                allocations_changed=0, old_graphs=old_graphs, new_graphs=new_graphs)


def controls(before, after, om, nm, symbols, added):
    start = next(i for i,l in enumerate(after) if l.startswith('launch ') and '_last_m' in l)
    other = next(i for i,l in enumerate(after) if l.startswith('launch ') and '_head_f32_m' not in l)
    resolve = next(i for i,l in enumerate(after) if l.startswith('resolve ') and l.split()[2] in symbols)
    allocation = next(i for i,l in enumerate(after) if l.startswith('allocate '))
    mutations = []
    for index, old, new in [(start,'grid_x: ','grid_x: 9'), (start+1,'grid_z: 1','grid_z: 2'),
                            (start+1,'_s1_p1_r0_','_s0_p1_r0_'), (start+1,'flags: 1','flags: 0'),
                            (start+1,'shared_memory_bytes: 0','shared_memory_bytes: 4'),
                            (other,'block_x: ','block_x: 9'), (allocation,' 256',' 128')]:
        bad = after.copy(); bad[index] = bad[index].replace(old,new,1)
        assert bad != after
        mutations.append(bad)
    bad = after.copy(); del bad[resolve]; mutations.append(bad)
    bad = after.copy(); del bad[start]; mutations.append(bad)
    args = arguments(after[start+1])
    for position, replacement in [(2,args[0]), (6,args[1])]:
        changed = args.copy(); changed[position] = replacement
        bad = after.copy(); bad[start+1] = after[start+1].split(' [',1)[0]+' ['+', '.join(changed)+']'
        mutations.append(bad)
    bad = after.copy(); bad.pop(); mutations.append(bad)
    for bad in mutations:
        try:
            audit(before,bad,om,nm,symbols,added)
        except (AssertionError,ValueError,KeyError,IndexError,TypeError):
            continue
        raise AssertionError('journal corruption escaped')
    return len(mutations)


def run(before, after, paired, lock):
    module = json.loads(lock.read_text())[0]
    assert module['name'] == CUBIN
    symbols = [e['symbol'] for e in module['entries']]
    results = {}; negatives = 0
    for path in sorted(before.glob('*.journal')):
        old, om = read(path); new, nm = read(after/path.name)
        if '-tp4.' not in path.name:
            assert old == new and om == nm
            results[path.name] = dict(unchanged=True, old=digest(old), new=digest(new))
        else:
            results[path.name] = audit(old,new,om,nm,symbols,True)
            negatives += controls(old,new,om,nm,symbols,True)
    assert len(results) == 8
    old, om = read(paired/'head-reference.journal')
    new, nm = read(paired/'head-serving.journal')
    results['paired-buckets'] = audit(old,new,om,nm,symbols,False)
    assert set(results['paired-buckets']['replacements']) == {1,8,32}
    negatives += controls(old,new,om,nm,symbols,False)
    assert negatives == 48
    return dict(passed=True,journals=results,rejected_mutations=negatives)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('before','after','paired','lock','output'):p.add_argument(name,type=Path)
    a=p.parse_args();result=run(a.before,a.after,a.paired,a.lock)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('HEAD_LAST_JOURNAL_AUDIT_OK journals=9 controls=48')
