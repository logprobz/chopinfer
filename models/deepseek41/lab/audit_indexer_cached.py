#!/usr/bin/env python3
"""Audit the cached selector's module-only delta in the short-context journals."""
import argparse
import bisect
import itertools
import json
from pathlib import Path
import re
from audit_indexer_guard_fast import fnv

EXTRA = tuple(f'chop_position_indexer_deepseek41_r{ratio}_all_select_v3_cached4_v1' for ratio in (2, 1))


def audit(old, new, ranks, bf16=False):
    if bf16:
        assert old == new, 'BF16 journal changed'
        return dict(identical=True, lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}')
    resolves = [(int(s.split()[1]), s.split()[2]) for s in new if s.startswith('resolve ') and s.split()[2] in EXTRA]
    assert len(resolves) == 2*ranks
    ends = []
    for i in range(ranks):
        pair = resolves[2*i:2*i+2]
        assert [p[1] for p in pair] == list(EXTRA) and pair[1][0] == pair[0][0]+1
        ends.append(pair[1][0])
    assert ends == sorted(ends)

    def handle(value):
        value = int(value)
        slot = value & 0xffffffff
        return str(value if slot >= 1_000_000 else value - 3*bisect.bisect_left(ends, slot))

    normalized = []
    for line in new:
        fields = line.split(' ')
        op = fields[0]
        if op == 'resolve' and fields[2] in EXTRA:
            continue
        if op == 'launch':
            assert fields[2] not in EXTRA, 'short-context fixture selected cached prefill'
        columns = []
        if op in ('stream', 'event', 'capture', 'synchronize', 'zero', 'download', 'copy', 'launch', 'resolve', 'configure'):
            columns = [1]
        elif op in ('record', 'wait', 'captured', 'replay'):
            columns = [1, 2]
        elif op == 'upload' and fields[1] != 'arena':
            columns = [1]
        elif op == 'release' and len(fields) > 1:
            assert fields[1] in ('stream', 'event', 'graph')
            columns = [2]
        else:
            assert op in ('allocate', 'state', 'serving', 'release', 'upload'), op
        for i in columns:
            fields[i] = handle(fields[i])
        normalized.append(re.sub(r'raw: (\d+)', lambda m: 'raw: '+handle(m[1]), ' '.join(fields)))
    def groups(lines):
        return [(kind, list(part)) for kind, part in itertools.groupby(lines, lambda s: s == 'release')]
    a,b=groups(old),groups(normalized)
    assert len(a)==len(b)
    releases=0
    for (ka,aa),(kb,bb) in zip(a,b):
        assert ka==kb
        if ka:
            growth=len(bb)-len(aa)
            assert growth>=0 and growth%3==0
            releases+=growth
        else:
            assert aa==bb, next(((x[:240],y[:240]) for x,y in zip(aa,bb) if x!=y),'operation count changed')
    assert releases==3*ranks and len(new)==len(old)+5*ranks
    return dict(lines=len(new), fnv=f'0x{fnv(chr(10).join(new)):016x}', added_resolves=2*ranks,
                added_releases=releases, existing_launches_arguments_allocations_and_streams_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path);parser.add_argument('head',type=Path);parser.add_argument('report',type=Path)
    args=parser.parse_args();paths=sorted(args.base.glob('*.journal'))
    assert len(paths)==8 and {p.name for p in paths}=={p.name for p in args.head.glob('*.journal')}
    result={}
    for path in paths:
        old=path.read_text().splitlines();new=(args.head/path.name).read_text().splitlines()
        ranks=int(re.search(r'tp(\d+)',path.name)[1]);bf16='.bf16.' in path.name
        row=audit(old,new,ranks,bf16);text='\n'.join(new)
        mutants=[text+'\nrelease',text.replace('offset: 0','offset: 1',1),text.replace('grid_x: 192','grid_x: 191',1)]
        if not bf16:mutants+=[text.replace(EXTRA[0],EXTRA[0]+'_bad',1)]
        for bad in mutants:
            assert bad!=text
            try:audit(old,bad.splitlines(),ranks,bf16)
            except AssertionError:pass
            else:raise AssertionError('journal mutation escaped')
        row['negative_controls']=len(mutants);result[path.name]=row
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
