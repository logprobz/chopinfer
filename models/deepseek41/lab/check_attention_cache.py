#!/usr/bin/env python3
"""Require exact selected-KV cache coverage before accepting pair timings."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

FIELDS = ('context', 'rows', 'sequences', 'verify', 'verify_width', 'ratio', 'holes', 'inactive')
MODES = ('cold_sources', 'cold_attention', 'warm_attention')


def require(value, message):
    if not value:
        raise ValueError(message)


def records(text, marker):
    return [json.loads(line.split(marker+' ', 1)[1]) for line in text.splitlines() if marker+' ' in line]


def key(case):
    return tuple(case[name] for name in FIELDS)


def cases(shard):
    def shapes(context):
        return [dict(context=context, sequences=s, verify=v, verify_width=w, rows=s*w,
                     ratio=r, holes=False, inactive=0)
                for s,v,w in [(1,False,1),(8,False,1),(1,True,6),(8,True,4)] for r in (0,1,2)]
    matrix=[c for n in (1024,8192,32768,131072) for c in shapes(n)]
    edges=[]
    for i,n in enumerate((1,31,32,33,63,64,65,127,128,129,511,512,513,1023)):
        for c in shapes(n):
            c.update(holes=True,inactive=(0,1,c['rows'])[i%3]);edges.append(c)
    return matrix[shard::2],edges[shard::2]


def same(actual, expected, name):
    require(Counter(actual)==Counter(expected),name+' coverage differs')


def validate(text, shard):
    results=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    require(results==[('ok','1','0','0')],'one successful GPU test is required')
    require('GPU work skipped' not in text,'GPU execution skipped')
    require(text.count(f'CACHE_COMPLETE shard={shard} matrix=24 edges=84 negative=4')==1,'completion marker')
    matrix,edges=cases(shard)
    compared=records(text,'CACHE_CASE_OK')
    same(map(key,compared),map(key,matrix+edges),'bitwise')
    clocks=records(text,'CACHE_CLOCK')
    same((key(r['case']) for r in clocks),map(key,matrix+edges),'phase clocks')
    for r in clocks:
        c=r['case'];ctas=(c['rows']-c['inactive'])*(16 if c['rows']<=4 else 2)*8
        require(r['ctas']==ctas,'clock CTA count')
        values=r['cycles'];require(len(values)==8 and all(type(v) is int and v>=0 for v in values),'clock values')
        require(sum(values[:6])<=values[6] and (values[6]>0)==(ctas>0),'clock phase sum')
        require(values[7]==ctas//8*(2 if c['ratio']==0 else 10),'clock split tiles')
    timings=records(text,'CACHE_TIMING')
    same(((key(r['case']),r['mode']) for r in timings),((key(c),m) for c in matrix for m in MODES),'timing')
    graphs=records(text,'CACHE_GRAPH_OK')
    same(((key(r['case']),r['mode']) for r in graphs),((key(c),m) for c in matrix for m in MODES),'captured candidate')
    for r in timings:
        c=r['case'];cells=128 if c['ratio']==0 else 640
        require(r['cache_bytes']==[c['rows']*cells*1024,c['rows']*cells*4],'bounded cache bytes')
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in ('old_us','new_us','paired_ratio')),'finite timing')
        require(('_stage64_h0_b0_profile0_v1' in r['old_symbol'])==(c['rows'] in (8,32)),'current serving reference')
        compressed=0 if c['ratio']==0 else min(512,c['context']//c['ratio'])
        unique=c['sequences']*((min(128,c['context'])+c['verify_width']-1)*528+compressed*288)
        row=c['sequences']*(sum(min(128,c['context']+i)*528 for i in range(c['verify_width']))+c['verify_width']*compressed*288)
        require(r['selected_kv_bytes']==unique and r['row_kv_bytes']==row,'selected byte counts')
    predecode=records(text,'CACHE_PREDECODE_TIMING')
    same((key(r['case']) for r in predecode),map(key,matrix),'predecode timing')
    require(all(math.isfinite(r['us']) and r['us']>0 for r in predecode),'predecode latency')
    controls=records(text,'CACHE_NEGATIVE_OK')
    same(((c['rows'],c['verify']) for c in controls),[(1,False),(8,False),(6,True),(32,True)],'negative controls')
    require(all(c in matrix and c['ratio']!=0 for c in controls),'negative control cases')
    launches=records(text,'ATTENTION_LAUNCH')
    require(launches and records(text,'ATTENTION_CONFIGURE'),'launch and opt-in evidence')
    require(any('_cached_profile0_' in r['symbol'] for r in launches),'cached source not launched')
    for r in launches:
        require(r['dynamic_shared_bytes']<=r['configured_shared_bytes'],'missing dynamic shared opt-in')
        require(len(r['grid'])==3 and all(v>0 for v in r['grid']),'invalid grid')
        require(len(r['block'])==3 and 0<math.prod(r['block'])<=1024,'invalid block')
    return dict(cases=compared,clocks=clocks,timings=timings,graphs=graphs,predecode=predecode,controls=controls)


def synthetic(shard):
    matrix,edges=cases(shard)
    lines=[f'CACHE_COMPLETE shard={shard} matrix=24 edges=84 negative=4',
           'ATTENTION_CONFIGURE '+json.dumps(dict(symbol='source',dynamic_shared_bytes=73984)),
           'ATTENTION_LAUNCH '+json.dumps(dict(symbol='source_cached_profile0_v1',dynamic_shared_bytes=73984,
              configured_shared_bytes=73984,grid=[1,16,8],block=[128,1,1]))]
    for c in matrix+edges:
        lines.append('CACHE_CASE_OK '+json.dumps(c))
        ctas=(c['rows']-c['inactive'])*(16 if c['rows']<=4 else 2)*8
        lines.append('CACHE_CLOCK '+json.dumps(dict(case=c,ctas=ctas,cycles=[ctas]*6+[ctas*8,ctas//8*(2 if c['ratio']==0 else 10)])))
    seen=set()
    for c in matrix:
        cells=128 if c['ratio']==0 else 640
        compressed=0 if c['ratio']==0 else min(512,c['context']//c['ratio'])
        unique=c['sequences']*((min(128,c['context'])+c['verify_width']-1)*528+compressed*288)
        row=c['rows']*(min(128,c['context'])*528+compressed*288)
        for mode in MODES:
            lines.append('CACHE_GRAPH_OK '+json.dumps(dict(case=c,mode=mode)))
            lines.append('CACHE_TIMING '+json.dumps(dict(case=c,mode=mode,cache_bytes=[c['rows']*cells*1024,c['rows']*cells*4],
                old_symbol='source_stage64_h0_b0_profile0_v1' if c['rows'] in (8,32) else 'source_heads_v1',
                old_us=2,new_us=1,paired_ratio=.5,selected_kv_bytes=unique,row_kv_bytes=row)))
        lines.append('CACHE_PREDECODE_TIMING '+json.dumps(dict(case=c,us=1)))
        bucket=(c['rows'],c['verify'])
        if c['ratio'] and bucket not in seen:
            seen.add(bucket);lines.append('CACHE_NEGATIVE_OK '+json.dumps(c))
    lines.append('test result: ok. 1 passed; 0 failed; 0 ignored; 12 filtered out; finished in 1.0s')
    return '\n'.join('test prefix: '+line for line in lines)+'\n'


def self_test():
    for shard in (0,1):validate(synthetic(shard),shard)
    text=synthetic(0);first=next(line for line in text.splitlines() if 'CACHE_CASE_OK' in line)
    bad=[text.replace('1 passed','0 passed'),text.replace('0 failed','1 failed'),text.replace('0 ignored','1 ignored'),
         text.replace(first+'\n','',1),text+first+'\n',text.replace('CACHE_NEGATIVE_OK','MISSING_CONTROL'),
         text.replace('ATTENTION_CONFIGURE','MISSING_OPTIN'),text.replace('"configured_shared_bytes": 73984','"configured_shared_bytes": 49152'),
         text.replace('"new_us": 1','"new_us": NaN',1),text.replace('"ctas": 128','"ctas": 1',1),
         text.replace('CACHE_PREDECODE_TIMING','MISSING_PREDECODE'),text.replace('_stage64_h0_b0_profile0_v1','_heads_v1'),
         text+'GPU work skipped\n',text.replace('CACHE_GRAPH_OK','MISSING_GRAPH')]
    for i,value in enumerate(bad):
        try:validate(value,0)
        except ValueError:continue
        raise ValueError(f'negative parser control {i} passed')
    print('CACHE_PARSER_OK successes=2 rejected=14')


def main():
    if sys.argv[1:]==['--self-test']:
        self_test();return
    require(len(sys.argv)==4,'usage: check_attention_cache.py GPU0 GPU1 OUT')
    reports=[validate(Path(name).read_text(),i) for i,name in enumerate(sys.argv[1:3])]
    combined={k:[v for report in reports for v in report[k]] for k in reports[0]}
    combined['coverage']=dict(matrix=48,boundaries=168,poison_replays=432,profile_comparisons=216,graph_comparisons=144,negative_controls=8,timings=144,predecode_timings=48)
    out=Path(sys.argv[3]);(out/'measurements.json').write_text(json.dumps(combined,indent=2)+'\n')
    print('CACHE_PAIR_OK '+json.dumps(combined['coverage']))


if __name__=='__main__':main()
