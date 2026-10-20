#!/usr/bin/env python3
"""Audit exact router substitutions and every retained lifecycle record."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_moe_direct import bind, RAW

PREFIX = 'chop_deepseek41_moe_router_finish_'
SYMBOLS = {PREFIX+f'e{e}_k{k}_{stage}_v1' for e,k in [(384,6),(128,3)] for stage in ['prepare','finish']}
SPAN = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)')
ARG = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{.*? \}\)|ScalarU32\(\d+\)')


def launch(line):
    head,tail=line.split(' [',1)
    args=ARG.findall(tail[:-1])
    assert ', '.join(args)==tail[:-1],line
    return head.split(' ',3),args


def strip(lines,new):
    kept=[];groups=[];active={};capturing=False;removed=0
    for line in lines:
        f=line.split()
        if f[0]=='capture':
            assert not capturing and not active
            capturing=True;removed=0
        elif f[0]=='captured':
            assert capturing and not active
            f[3]=str(int(f[3])-removed);line=' '.join(f);capturing=False
        if f[0]=='launch':
            stream,symbol=f[1:3]
            begin=(symbol.startswith(PREFIX) and symbol.endswith('_prepare_v1')) if new else (capturing and symbol=='chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_rows_v1' and 'grid_x: 1,' in line)
            if begin:
                assert capturing and stream not in active
                active[stream]=[]
            if stream in active:
                assert ('moe_' in symbol or 'dense_f32_m8192_n384_k5120' in symbol),symbol
                active[stream].append(line);removed+=1
                if symbol.endswith('_down_finish_v1'):
                    groups.append(active.pop(stream))
                continue
        kept.append(line)
    assert not capturing and not active
    return kept,groups


def audit(base,head):
    if base==head:return dict(identical=True,lines=len(base))
    a,old_groups=strip(base,False);b,new_groups=strip(head,True)
    assert len(old_groups)==len(new_groups)
    extra=[x for x in b if x.startswith('resolve ') and x.split()[2] in SYMBOLS]
    assert Counter(x.split()[2] for x in extra)==Counter({s:4 for s in SYMBOLS})
    b=[x for x in b if x not in set(extra)]
    assert head.count('release')-base.count('release')==20
    a=[x for x in a if x!='release'];b=[x for x in b if x!='release']
    assert len(a)==len(b),(len(a),len(b))
    maps={kind:{} for kind in ['stream','event','graph','function','buffer']}
    for kind,key,column in [('stream','stream',1),('event','event',1),('graph','captured',2),('function','resolve',1),('function','configure',1)]:
        aa=[x.split() for x in a if x.startswith(key+' ')];bb=[x.split() for x in b if x.startswith(key+' ')]
        assert len(aa)==len(bb)
        for x,y in zip(aa,bb):
            if key=='resolve':assert x[2:]==y[2:]
            if key=='configure':assert x[2:]==y[2:]
            bind(maps[kind],int(x[column]),int(y[column]))
    inv={k:{v:a for a,v in m.items()} for k,m in maps.items()}
    def normalize(line):
        f=line.split(' ');kind=f[0]
        def change(i,k):f[i]=str(inv[k].get(int(f[i]),int(f[i])))
        if kind in ['stream','event']:change(1,kind)
        elif kind in ['capture','synchronize','zero','download','copy','launch']:change(1,'stream')
        elif kind=='upload':
            if f[1]!='arena':change(1,'stream')
        elif kind in ['record','wait']:change(1,'stream');change(2,'event')
        elif kind in ['captured','replay']:change(1,'stream');change(2,'graph')
        elif kind in ['resolve','configure']:change(1,'function')
        elif kind=='release':
            assert f[1] in ['graph','stream','event'];change(2,f[1])
        else:assert kind in ['allocate','state','serving'],kind
        return ' '.join(f)
    for index,(old,new) in enumerate(zip(a,b)):
        x=RAW.findall(old);y=RAW.findall(new);assert len(x)==len(y)
        for left,right in zip(x,y):bind(maps['buffer'],int(left),int(right))
        assert RAW.sub('raw: BUFFER',old)==RAW.sub('raw: BUFFER',normalize(new)),('retained record',index,old,new)
    def check_arg(old,new):
        left,right=SPAN.fullmatch(old),SPAN.fullmatch(new)
        if left and right:
            x,y=tuple(map(int,left.groups())),tuple(map(int,right.groups()))
            bind(maps['buffer'],x[0],y[0]);assert x[1:]==y[1:],(old,new)
        else:assert old==new,(old,new)
    def check_args(old,new):
        assert len(old)==len(new)
        for x,y in zip(old,new):check_arg(x,y)
    def span(arg):return tuple(map(int,SPAN.fullmatch(arg).groups()))
    def view(handle,offset,length):return f'DeviceBuffer(DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {handle} }}, offset: {offset}, length: {length} }})'
    for old,new in zip(old_groups,new_groups):
        assert len(old)==9 and len(new)==7
        old=[launch(x) for x in old];new=[launch(x) for x in new]
        old_names=['chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_'+s+'_v1' for s in ['rows','decode','reduce']]+['chop_moe_sqrt_softplus_row_bias_e384k6_t64_v1','chop_deepseek41_moe_floor_e384_k6_metadata_v1']
        assert [x[0][2] for x in old[:5]]==old_names
        assert new[0][0][2]==PREFIX+'e384_k6_prepare_v1'
        assert new[2][0][2]==PREFIX+'e384_k6_finish_v1'
        assert old[0][1][2]=='ScalarU32(1)'
        assert old[0][0][3]==new[0][0][3]
        assert new[2][0][3]==old[0][0][3].replace('block_x: 128,','block_x: 384,')
        for oh,nh in zip([x[0] for x in old[:1]+old[1:2]+old[5:]],[x[0] for x in new[:1]+new[1:2]+new[3:]]):
            bind(maps['stream'],int(oh[1]),int(nh[1]))
        root,origin,capacity=span(old[0][1][3])
        partial=(32+384*40)*4;arrival=(partial+255)//256*256;shift=arrival+256
        assert origin==0 and shift+span(old[4][1][7])[2]+256<=capacity
        arrive=view(root,arrival,4)
        check_args(old[0][1][:3]+[view(root,0,partial),arrive],new[0][1])
        assert old[1][0][2:]==new[1][0][2:];check_args(old[1][1],new[1][1])
        expected=[view(root,0,partial)]+old[2][1][1:]+old[3][1][1:7]+old[3][1][8:9]+old[4][1][1:5]+old[4][1][6:9]+[arrive]
        for i in [16,17]:
            h,o,n=span(expected[i]);assert h==root;expected[i]=view(h,o+shift,n)
        check_args(expected,new[2][1])
        for (oh,oa),(nh,na) in zip(old[5:],new[3:]):
            assert oh[2:]==nh[2:]
            shifted=[]
            for arg in oa:
                m=SPAN.fullmatch(arg)
                if m and int(m[1])==root:
                    h,o,n=span(arg);arg=view(h,o+shift,n)
                shifted.append(arg)
            check_args(shifted,na)
    return dict(identical=False,old_lines=len(base),new_lines=len(head),fused_layers=len(old_groups),added_resolves=16,added_releases=20,retained_records=len(a),retained_launches=sum(x.startswith('launch ') for x in a),workspace_allocations_unchanged=True,ordered_bindings_checked=True,handle_bijections={k:len(v) for k,v in maps.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--head',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    before={f.name:f.read_text().splitlines() for f in args.base.glob('*.journal')};after={f.name:f.read_text().splitlines() for f in args.head.glob('*.journal')}
    assert before.keys()==after.keys() and len(before)==8
    results={}
    for name in sorted(before):
        if before[name]!=after[name]:assert '-tp4.' in name
        results[name]=audit(before[name],after[name])
        expected=160 if name.endswith('tp4.media.journal') else 0
        assert results[name].get('fused_layers',0)==expected
    name='deepseek41-flash-tp4.media.journal';base=before[name];good=after[name]
    controls=[]
    for needle,old,new in [
        ('_prepare_v1 LaunchConfig','offset: 61696, length: 4','offset: 0, length: 4'),
        ('_finish_v1 LaunchConfig','ScalarU32(1069547520)','ScalarU32(1069547521)'),
        ('_finish_v1 LaunchConfig','block_x: 384','block_x: 128'),
        ('_gate_small_v1 LaunchConfig','offset: 61952,','offset: 61956,'),
        ('_down_finish_v1 LaunchConfig','length: 122880','length: 122876'),
    ]:
        at=next(i for i,line in enumerate(good) if needle in line and old in line)
        bad=good.copy();bad[at]=bad[at].replace(old,new,1)
        try:audit(base,bad)
        except (AssertionError,KeyError,ValueError):controls.append(needle)
        else:raise AssertionError('mutation escaped '+needle)
    # A retained shared binding is not allowed to change at all.
    at=next(i for i,line in enumerate(good) if line.startswith('launch ') and 'shared_swiglu' in line)
    bad=good.copy();bad[at]=bad[at].replace('length:','offset: 4, length:',1)
    try:audit(base,bad)
    except (AssertionError,KeyError,ValueError):controls.append('shared binding')
    else:raise AssertionError('shared mutation escaped')
    result=dict(passed=True,journals=results,negative_controls=controls)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
