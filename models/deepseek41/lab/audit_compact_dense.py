"""Audit measured dense substitutions and preserve all other journal records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from audit_prefill_gate_journals import normalized

ROOT=Path(__file__).resolve().parents[3]
SPAN=re.compile(r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}')
SHAPE=re.compile(r'chop_deepseek41_mxfp8_fold_m8192_n(\d+)_k(\d+)_g(\d+)_f32_v2')
COMPACT=re.compile(r'_compact_m(\d+)_n128_s(\d+)_v1(_admit_v1)?$')
ROWS=(128,512,2048,4096,8192)


def choices(summary):
    text=(ROOT/'models/deepseek41/bundle/src/sm120/mxfp8_compact.rs').read_text()
    selected={}
    for match in re.finditer(r'\((\d+), (\d+), (\d+), \[([0-3, ]+)\]\)',text):
        shape=tuple(map(int,match.group(1,2,3)));stages=list(map(int,match[4].split(',')))
        assert len(stages)==5
        for rows,stage in zip(ROWS,stages):
            if stage:selected[(*shape,rows)]=stage
    data=json.loads(summary.read_text());assert data['bitwise'] and len(data['records'])==400
    expected={}
    for row in data['both_device_winners']:
        key=tuple(row[k] for k in ('n','k','groups','rows'))
        if key not in expected or row['worst_speedup']>expected[key]['worst_speedup']:expected[key]=row
    assert selected=={k:r['stages'] for k,r in expected.items()} and len(selected)==66
    for key,stage in selected.items():
        records=[r for r in data['records'] if tuple(r[k] for k in ('n','k','groups','rows'))==key and r['stages']==stage]
        assert {r['device'] for r in records}=={0,1} and len(records)==2
        assert all(r['bitwise'] and r['exclusive']['paired_ratio']<.97 and r['event_ratio']<.97 for r in records)
    return selected


def plan(line,n,groups,rows,tile,threads,shared):
    for key,value in dict(flags=1,grid_x=(n+127)//128,grid_y=(rows+tile-1)//tile,grid_z=groups,block_x=threads,block_y=1,block_z=1,shared_memory_bytes=shared).items():
        assert re.search(r'\b'+key+r': (\d+)',line)[1]==str(value),(key,line)
    assert 'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 }' in line


def canonical(lines,selected):
    output=[];changed=0
    functions={l.split()[1]:l.split()[2] for l in lines if l.startswith('resolve ')}
    for line in lines:
        if line.startswith('configure ') and COMPACT.search(functions[line.split()[1]]):continue
        if not line.startswith('launch '):output.append(line);continue
        _,stream,symbol=line.split(' ',3)[:3]
        shape=SHAPE.match(symbol)
        if not shape:output.append(line);continue
        n,k,groups=map(int,shape.groups());rows=int(re.findall(r'ScalarU32\((\d+)\)',line)[-1])
        stage=selected.get((n,k,groups,rows));compact=COMPACT.search(symbol)
        if not stage:
            assert compact is None,('unqualified compact launch',symbol,rows)
            output.append(line);continue
        if compact:
            assert int(compact[2])==stage
            admit=bool(compact[3])
        elif symbol.endswith('_prefill_tma_v1_admit_v1'):admit=True
        elif symbol.endswith('_prefill_tma_v1'):admit=False
        else:output.append(line);continue
        if admit:
            if compact:assert int(compact[1])==64
            assert f'grid_x: {rows}, grid_y: 1, grid_z: 1,' in line
            assert 'block_x: 128, block_y: 1, block_z: 1,' in line and 'shared_memory_bytes: 0,' in line
            output.append(line.replace(symbol,f'logical_dense_admit_n{n}_k{k}_g{groups}',1));continue
        spans=[tuple(map(int,m.groups())) for m in SPAN.finditer(line)]
        if compact:
            tile=32 if rows==128 else 64
            assert int(compact[1])==tile and len(spans)==7
            plan(line,n,groups,rows,tile,128 if tile==32 else 256,stage*((tile+128)*132)+tile*4)
        else:
            assert len(spans)==9 and spans[:2]==spans[2:4]
            assert line.count('TensorMap(TensorMapV1')==2
            assert f'global_dims: [{k*groups}, {rows}, 0, 0, 0]' in line
            assert f'global_dims: [{k}, {n*groups}, 0, 0, 0]' in line
            plan(line,n,groups,rows,128,384,66560)
            spans=spans[2:];spans=[spans[i] for i in (0,2,1,3,4,5,6)]
        args=['DeviceBuffer(DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: %d }, offset: %d, length: %d })'%s for s in spans]
        scalars=re.findall(r'Scalar(?:U64|U32)\(\d+\)',line);assert len(scalars)==2
        output.append(f'launch {stream} logical_dense_n{n}_k{k}_g{groups} '+str(args+scalars))
        changed+=1
    return normalized(output)[0],changed


def audit(old,new,selected):
    a=old.read_text().splitlines();b=new.read_text().splitlines()
    resolved=lambda lines:Counter(l.split()[2] for l in lines if l.startswith('resolve '))
    ac,bc=resolved(a),resolved(b);assert not ac-bc
    extra=bc-ac;assert extra and all(COMPACT.search(s) or '_omit_first_v1' in s and '_compact_' in s for s in extra)
    ranks=int(re.search(r'tp(\d)',old.name)[1]);assert sum(extra.values())==280*ranks
    def configurations(lines):
        functions={l.split()[1]:l.split()[2] for l in lines if l.startswith('resolve ')}
        return Counter((functions[l.split()[1]],int(l.split()[2])) for l in lines if l.startswith('configure '))
    ca,cb=configurations(a),configurations(b);assert not ca-cb
    assert all(COMPACT.search(name) for name,_ in cb-ca)
    left,n=canonical(a,selected);right,m=canonical(b,selected)
    assert n==m
    assert left==right,(old.name,next(((i,x[:400],y[:400]) for i,(x,y) in enumerate(zip(left,right)) if x!=y),('length',len(left),len(right))))
    assert sum(l.startswith('launch ') for l in a)==sum(l.startswith('launch ') for l in b)
    assert b.count('release')-a.count('release')==(sum(extra.values())+ranks if old.name.startswith('deepseek') else 0)
    return dict(journal=old.name,replaced_compute_calls=n,extra_resolves=sum(extra.values()),all_other_records_equal=True,bindings_equal=True,allocations_equal=True)


def run(root,summary,base_revision):
    selected=choices(summary);results=[]
    for old_dir,new_dir in [('base','head'),('base-prefill','journals')]:
        old={p.name:p for p in (root/old_dir).glob('*.journal')};new={p.name:p for p in (root/new_dir).glob('*.journal')}
        assert old.keys()==new.keys() and old
        for name in sorted(old):results.append(audit(old[name],new[name],selected))
    assert len(results)==12
    # Authenticate the retained lifecycle bodies against the pre-change source pins.
    source=subprocess.check_output(['git','show',base_revision+':models/deepseek41/bundle/src/fake_host_tests.rs'],cwd=ROOT,text=True)
    pins=re.findall(r'"(deepseek41-flash-tp\d)"\s*,\s*KvStorage::(\w+)\s*,\s*\((\d+), (0x[0-9a-f]+)\)',source)
    assert len(pins)==6
    for name,storage,count,digest in pins:
        data=(root/'base'/f'{name}.{storage.lower()}.journal').read_bytes();value=0xcbf29ce484222325
        for byte in data:value=((value^byte)*0x100000001b3)&((1<<64)-1)
        assert len(data.splitlines())==int(count) and value==int(digest,16)
    # A real changed journal must reject misrouting, launch changes and operand edits.
    lines=(root/'journals/tp4-native-8192.journal').read_text().splitlines()
    reference=canonical(lines,selected)[0]
    index=next(i for i,l in enumerate(lines) if l.startswith('launch ') and '_compact_m64_n128_s2_v1 ' in l)
    controls=0
    for before,after in [('grid_y: 128','grid_y: 129'),('shared_memory_bytes: 50944','shared_memory_bytes: 50948'),('_s2_v1 ','_s3_v1 '),('offset: 0','offset: 16')]:
        changed=lines.copy();assert before in changed[index];changed[index]=changed[index].replace(before,after,1)
        try:assert canonical(changed,selected)[0]==reference
        except AssertionError:controls+=1
        else:raise AssertionError('journal mutation passed: '+before)
    report=dict(passed=True,base_revision=base_revision,buckets=66,journals=results,negative_controls=controls,scope='Pinned lifecycle and six dense prefill CPU journals; no GPU or full history proof')
    (root/'journal-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--base-revision',required=True)
    a=p.parse_args();run(a.root,a.summary,a.base_revision)
