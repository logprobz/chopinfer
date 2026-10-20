"""Audit new projection module handles before updating lifecycle journal pins."""
import argparse
from collections import Counter
import json
from pathlib import Path
from audit_selected_cache import canonical
from audit_decode_attention import digest

MODULES=('mxfp8_fold_reuse_deepseek41_sm120','f32_decode_reuse_deepseek41_sm120')


def audit(before,after,old_meta,new_meta,symbols):
    assert len(symbols)==17
    added=[line for line in after if line.startswith('resolve ') and line.split()[2] in symbols]
    assert Counter(line.split()[2] for line in added)==Counter({s:4 for s in symbols})
    functions={line.split()[1]:line.split()[2] for line in after if line.startswith('resolve ')}
    assert not any(line.startswith('configure ') and functions[line.split()[1]] in symbols for line in after)
    assert not any(line.startswith('launch ') and line.split()[2] in symbols for line in after), 'C1/C2 lifecycle must retain launches'
    assert Counter(new_meta['modules'].values())==Counter(old_meta['modules'].values())+Counter({m:4 for m in MODULES})
    assert len(old_meta['buffers'])==len(new_meta['buffers'])
    for a,b in zip(old_meta['buffers'],new_meta['buffers']):
        assert old_meta['ranks'][str(a)]==new_meta['ranks'][str(b)],'allocation rank or bytes differ'
    kept=[line for line in after if line not in added]
    releases=4*(len(symbols)+len(MODULES))
    assert kept[-releases:]==['release']*releases
    kept=kept[:-releases]
    a=list(canonical(before,old_meta));b=list(canonical(kept,new_meta))
    assert len(a)==len(b)
    for i,(old,new) in enumerate(zip(a,b)):
        assert old==new,('retained journal differs',i,old,new)
    return dict(old=digest(before),new=digest(after),added_resolves=len(added),added_releases=releases,
                added_modules=8,changed_launches=0,retained_lines=len(before))


def run(before,after,locks):
    symbols=[]
    for family in ('mxfp8_fold_reuse','f32_decode_reuse'):
        record=json.loads((locks/(family+'.lock')).read_text())[0]
        assert record['name'] in MODULES
        symbols.extend(e['symbol'] for e in record['entries'])
    records={};negative=0
    for path in sorted(before.glob('*.journal')):
        old=path.read_text().splitlines();new=(after/path.name).read_text().splitlines()
        om=json.loads(path.with_suffix('.journal.handles.json').read_text())
        nm=json.loads((after/(path.name+'.handles.json')).read_text())
        if '-tp4.' not in path.name:
            assert old==new and om==nm
            records[path.name]=dict(unchanged=True,old=digest(old),new=digest(new));continue
        records[path.name]=audit(old,new,om,nm,symbols)
        at=next(i for i,l in enumerate(new) if l.startswith('launch '))
        for kind in ('missing_resolve','missing_release','changed_launch','changed_allocation'):
            changed=new.copy()
            if kind=='missing_resolve':changed.remove(next(l for l in changed if l.startswith('resolve ') and l.split()[2] in symbols))
            elif kind=='missing_release':changed.pop()
            elif kind=='changed_launch':changed[at]=changed[at].replace('block_x: ','block_x: 9',1)
            else:
                index=next(i for i,l in enumerate(changed) if l.startswith('allocate '));changed[index]+=' changed'
            try:audit(old,changed,om,nm,symbols)
            except (AssertionError,ValueError):negative+=1
            else:raise AssertionError('journal mutant escaped: '+kind)
    assert len(records)==8 and negative==12
    return dict(passed=True,rejected_mutations=negative,journals=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('locks',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();result=run(a.before,a.after,a.locks);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PROJECTION_JOURNAL_AUDIT_OK journals=8 controls=12')
