#!/usr/bin/env python3
"""Audit direct-MoE journal substitutions, retaining every other operation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

OLD = [
    'chop_moe_sqrt_softplus_row_bias_route_decode_e384l384k6_t8_p64_v1',
    *['deepseek41_chop_moe_chain_mxfp4_a8ceil_before_gateup_e384k6_n1152_k5120_r64_t8_clamp4120000041200000_stream_'+s+'_v1'
      for s in ['pack','small','merge']],
    'chop_deepseek41_moe_ceil_w3456_parallel_v1',
    *['deepseek41_chop_moe_chain_mxfp4_a8ceil_before_down_e384k6_n5120_k576_r64_t8_stream_'+s+'_v1'
      for s in ['pack','small','merge']],
    'chop_moe_deepseek41_sum_add_f32_k6_w5120_l384_t8192_v1',
]
NEW = ['chop_moe_sqrt_softplus_row_bias_e384k6_t64_v1',
       'chop_deepseek41_moe_floor_e384_k6_metadata_v1',
       *['chop_deepseek41_moe_bandwidth_direct_e384_k6_'+s+'_v1'
         for s in ['gate_small','gate_finish','down_small']],
       'chop_deepseek41_moe_floor_e384_k6_down_finish_v1']
RAW = re.compile(r'raw: (\d+)')


def bind(mapping, old, new):
    if old in mapping:
        assert mapping[old] == new, ('inconsistent handle',old,new)
    else:
        assert new not in mapping.values(), ('aliased handle',old,new)
        mapping[old] = new


def retained(lines, replacement):
    kept, removed = [], []
    active = False
    launches = 0
    changed = []
    layers = 0
    for line in lines:
        fields = line.split()
        if fields[0] == 'capture':
            assert not active
            active = True; launches = 0; changed = []
        elif fields[0] == 'captured':
            assert active and launches == int(fields[3])
            assert len(changed) % len(replacement) == 0
            assert changed == replacement * (len(changed)//len(replacement)), 'changed route stage order'
            layers += len(changed)//len(replacement)
            fields[3] = str(launches-len(changed)); line = ' '.join(fields)
            active = False
        elif active and fields[0] == 'launch':
            launches += 1
            if fields[2] in replacement:
                changed.append(fields[2]); removed.append(line); continue
        kept.append(line)
    assert not active
    return kept,removed,layers


def audit(base, head, added_symbols, added_modules):
    if base == head:
        return dict(identical=True,lines=len(base))
    a,removed_a,layers_a = retained(base,OLD)
    b,removed_b,layers_b = retained(head,NEW)
    assert layers_a == layers_b == 160, 'expected forty MoE layers on four ranks'
    extras = [x for x in b if x.startswith('resolve ') and x.split()[2] in added_symbols]
    assert Counter(x.split()[2] for x in extras) == Counter({s:4 for s in added_symbols})
    extra_lines = set(extras)
    b = [x for x in b if x not in extra_lines]
    assert head.count('release') - base.count('release') == 4*(len(added_symbols)+added_modules)
    a = [x for x in a if x != 'release']; b = [x for x in b if x != 'release']
    assert len(a) == len(b)
    maps = {kind:{} for kind in ['stream','event','graph','function','buffer']}
    for kind,keyword,column in [('stream','stream',1),('event','event',1),('graph','captured',2),
                                ('function','resolve',1),('function','configure',1)]:
        aa=[x.split() for x in a if x.startswith(keyword+' ')]
        bb=[x.split() for x in b if x.startswith(keyword+' ')]
        assert len(aa)==len(bb)
        for x,y in zip(aa,bb):
            if keyword=='resolve':assert x[2]==y[2]
            if keyword=='configure':assert x[2:]==y[2:]
            bind(maps[kind],int(x[column]),int(y[column]))
    inverse={k:{v:a for a,v in m.items()} for k,m in maps.items()}
    def normalize(line):
        words=line.split(' ');kind=words[0]
        def change(at,typ):
            value=int(words[at]);words[at]=str(inverse[typ].get(value,value))
        if kind in ['stream','event']:change(1,kind)
        elif kind in ['capture','synchronize','zero','download','copy','launch']:change(1,'stream')
        elif kind=='upload':
            if words[1]!='arena':change(1,'stream')
        elif kind in ['record','wait']:change(1,'stream');change(2,'event')
        elif kind in ['captured','replay']:change(1,'stream');change(2,'graph')
        elif kind in ['resolve','configure']:change(1,'function')
        elif kind=='release':
            assert words[1] in ['graph','stream','event'];change(2,words[1])
        else:assert kind in ['allocate','state','serving'], kind
        return ' '.join(words)
    for index,(x,y) in enumerate(zip(a,b)):
        old_raw=RAW.findall(x);new_raw=RAW.findall(y)
        assert len(old_raw)==len(new_raw)
        for left,right in zip(old_raw,new_raw):bind(maps['buffer'],int(left),int(right))
        assert RAW.sub('raw: BUFFER',x)==RAW.sub('raw: BUFFER',normalize(y)), ('retained operation differs',index,x,y)
    return dict(identical=False,old_lines=len(base),new_lines=len(head),moe_layers=layers_a,
                old_replaced_launches=len(removed_a),new_replacement_launches=len(removed_b),
                retained_launches=sum(x.startswith('launch ') for x in a),
                retained_operations=len(a),added_resolves=len(extras),added_releases=4*(len(added_symbols)+added_modules),
                handle_bijections={k:len(v) for k,v in maps.items()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--head',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    locks=Path(__file__).resolve().parents[1]/'bundle/kernels.lock.d'
    modules=[]
    for family in ['routed_moe_floor','routed_moe_bandwidth']:
        modules.extend(json.loads((locks/(family+'.lock')).read_text()))
    symbols={e['symbol'] for m in modules for e in m['entries']}
    old={f.name:f for f in args.base.glob('*.journal')};new={f.name:f for f in args.head.glob('*.journal')}
    assert old.keys()==new.keys() and len(old)==8
    results={}
    for name in sorted(old):
        a=old[name].read_text().splitlines();b=new[name].read_text().splitlines()
        if a!=b:assert '-tp4.' in name
        results[name]=audit(a,b,symbols,len(modules))
    args.report.write_text(json.dumps(dict(passed=True,journals=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
