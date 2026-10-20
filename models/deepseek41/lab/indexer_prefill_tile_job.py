#!/usr/bin/env python3
"""Require complete prefill tile bitwise, memcheck and paired timing evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess

CPU_TEST='indexer_mma::prefill::tile::cpu_prefill_tiles_preserve_abi_fallback_and_chunk_offsets'
GPU_TEST='indexer_mma::prefill::tile::gpu_prefill_tiles_real_edges_partitions_and_timing'


def cases():
    rows=[]
    for context in (8192,32768):
        for layer in (2,8,14,20,24):
            rows.append(dict(layer=layer,context=context,rows=32,kind=0,capture=len(rows)))
    for layer in (2,20,24):
        for context in (8192,16384,24576,32768):
            for n in (8192,8191,2496,65):
                rows.append(dict(layer=layer,context=context,rows=n,kind=0,capture=None))
        for kind in range(1,13):
            rows.append(dict(layer=layer,context=32768,rows=65,kind=kind,capture=None))
    assert len(rows)==94
    return rows


def records(text,marker):
    return [json.loads(line.split(marker+' ',1)[1]) for line in text.splitlines() if marker+' ' in line]


def timed(c):
    return c['capture'] is not None or c['rows']==8192 and c['kind']==0


def selected(memcheck):
    return {i for i,c in enumerate(cases()) if not memcheck or i<5 or c['kind']==3 or c['kind']>=8}


def batch_count(c):
    """Match contract::indexer::prefill and the compact fixture's row chunks."""
    rows=c['rows']
    if c['layer']>=24:return (rows+4095)//4096
    ratio=1 if c['layer']==20 else 2
    # Captures and ordinary fixtures end at context - 1. The empty/one
    # position fixture instead ends at position 1.
    visible=(2 if c['kind']==6 else c['context'])//ratio
    page=64//ratio
    width=(max(1,visible)+page-1)//page*page
    fit=min((256<<20)//(width*4),rows,512 if c['layer']==20 else 4096)
    # The contract rounds after limiting by the supplied row count, so
    # even 65 rows with ample score storage require two dense batches.
    if fit>=64:fit=fit//64*64
    assert fit>0
    return (rows+fit-1)//fit


def results(logs,memcheck=False):
    assert len(logs)==2
    seen=[];times=[];guards=[]
    shapes=cases()
    for shard,text in enumerate(logs):
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text)==[('ok','1','0','0')]
        expected=sorted(i for i in selected(memcheck) if i%2==shard)
        assert re.findall(r'PREFILL_TILE_COMPLETE shard=(\d+) cases=(\d+) memcheck=(\w+)',text)==[(str(shard),str(len(expected)),str(memcheck).lower())]
        found=records(text,'PREFILL_TILE_CASE')
        assert sorted(c['id'] for c in found)==expected
        for c in found:
            want=shapes[c['id']]
            assert {k:c[k] for k in want}==want
            assert c['live']==want['rows']-int(want['kind']==12)
            assert c['kept']==(31 if want['kind']==11 else 512)
            assert c['reuse_pipelines']==1
            assert c['raw']==4 and c['pipelines']==4 and c['partitions']==1 and c['controls']==2
            assert c['mismatches']==0 and c['memcheck'] is memcheck
            assert c['timed'] is (timed(want) and not memcheck)
            expected_batches=batch_count(want)
            assert c['batches']==expected_batches, (
                f"batch count: case={c['id']} layer={want['layer']} context={want['context']} "
                f"rows={want['rows']} kind={want['kind']} memcheck={memcheck} "
                f"expected={expected_batches} observed={c['batches']}")
        seen.extend(found);times.extend(records(text,'PREFILL_TILE_TIMING'));guards.extend(records(text,'PREFILL_TILE_GUARDS'))
    wanted={(i,stage,arm) for i,c in enumerate(shapes) if timed(c) and not memcheck
            for stage in ('pack_score','all') for arm in ('coarse','narrow','wide','tile')}
    assert len(times)==len(wanted) and {(t['id'],t['stage'],t['candidate']) for t in times}==wanted
    for t in times:
        assert t['exclusive'] is True and t['score_buffer_reused'] is True
        assert t['reference']==('public' if t['candidate']=='coarse' else 'coarse')
        assert len(t['median_ms'])==2 and all(math.isfinite(x) and x>0 for x in [*t['median_ms'],t['paired_ratio']])
        assert t['batches']==next(c['batches'] for c in seen if c['id']==t['id'])
    assert {g['id'] for g in guards}=={i for i,c in enumerate(shapes) if timed(c) and not memcheck}
    assert len(guards)==(0 if memcheck else 22)
    for g in guards:
        a,b=g['public'],g['coarse']
        assert a['admitted']==b['admitted'] and a['repairs']==b['repairs']
        assert b['admitted']>0, 'diagnostic entries must count actual scores'
        assert a['timed'] is False and b['timed'] is False
        assert 0<=b['repairs']<=b['admitted'] and 0<=b['coarse_safe']<=b['admitted'] and 0<=b['fine_checked']<=b['admitted']
        if shapes[g['id']]['layer']>=24:assert g['compact_tile']==b
        else:assert g['compact_tile'] is None
    n=len(seen)
    return dict(cases=n,raw=4*n,pipelines=4*n,reuse_pipelines=n,partitions=n,controls=2*n,mismatches=0,
                timing_pairs=len(times),guards=guards,timings=times,records=seen)


def receipts(paths):
    seen=[];pairs=samples=0
    for path in paths:
        row=json.loads(path.read_text())
        if 'id' not in row:continue
        i=row['id'];assert 0<=i<94 and row['mismatches']==0
        assert row['timed'] is timed(cases()[i])
        assert row['measurement_label']=='exclusive-device'
        seen.append(i)
        rounds=row['timing_rounds']
        summaries=[r for r in rounds if r.get('kind')=='paired_summary']
        assert len(summaries)==(8 if timed(cases()[i]) else 0)
        assert {s['operation'] for s in summaries}==({f'prefill-tile-{i}-{a}-{s}' for a in ('coarse','narrow','wide','tile') for s in ('pack_score','all')} if summaries else set())
        for summary in summaries:
            assert summary['rounds']==3
            raw=[r for r in rounds if r.get('measurement')==summary['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'],r['arm']) for r in raw)==[(r,a) for r in range(3) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>0 for r in raw)
            pairs+=1;samples+=len(raw)
    assert sorted(seen)==list(range(94)) and pairs==176 and samples==1056
    return dict(pairs=pairs,samples=samples)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def profile(path):
    """Replay the exclusive interval rule on the frozen offset-zero profile."""
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        rows=list(db.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=0 order by k.start,k.end'))
    assert sum(s.startswith('chop_row_embedding_') for s,_,_ in rows)==1
    front=0;count=0;total=0.;by={}
    for s,start,end in rows:
        exclusive=max(0,end-max(start,front))/1e6;front=max(front,end)
        indexer=('position_' in s or 'candidate_block_ids_' in s or 'index_key_' in s
                 or 'dense_f32_m8192_n128_k512' in s or 'dense_f32_m8192_n32_k5120' in s
                 or 'mxfp8_fold_m8192_n4096_k1280' in s)
        if indexer:
            count+=1;total+=exclusive
            x=by.setdefault(s,dict(count=0,exclusive_ms=0));x['count']+=1;x['exclusive_ms']+=exclusive
    assert count==228 and 90<total<105,(count,total)
    return dict(count=count,exclusive_ms=total,kernels=by,offsets=[0],later_offsets_measured_by_pair=[8192,16384,24576])


def validate(stage_path):
    root=stage_path.parent;stage=json.loads(stage_path.read_text())
    for name,sha in stage['sha256'].items():
        p=root/name
        assert not p.is_symlink() and digest(p)==sha,name
        assert p.stat().st_mode&0o022==0,name
    captures=json.loads((root/'snapshot/captures.json').read_text())['captures']
    assert len(captures)==10
    for i,c in enumerate(captures):
        path=Path(c['path']);meta=json.loads((path/'operands.json').read_text())
        completion=json.loads((path/'../completion.json').read_text())
        assert completion['completion']=='full' and completion['observed']==completion['expected']
        assert meta['layer']==cases()[i]['layer'] and meta['context']==cases()[i]['context']
        assert meta['phase_rows']==8192 and meta['sample_rows']==[j*8191//31 for j in range(32)]
        assert meta['positions']==[meta['context']-8192+j for j in meta['sample_rows']]
        assert (path/'query.bf16').stat().st_size==32*8192
        for name,sha in c['sha256'].items():assert digest(path/name)==sha,(i,name)
    profiles=[profile(Path(p['path'])) for p in stage['profiles']]
    for p in stage['profiles']:assert digest(Path(p['path']))==p['sha256']
    assert abs(sum(p['exclusive_ms'] for p in profiles)/2-95.5536)<0.01
    listed=subprocess.check_output([str(root/'snapshot/prefill-tile-tests'),'--list'],text=True,timeout=30)
    for test in (CPU_TEST,GPU_TEST):
        assert sum(line.startswith(test+': test') for line in listed.splitlines())==1
    assert GPU_TEST in (root/'snapshot/worker.sh').read_text()
    output=subprocess.check_output([str(root/'snapshot/prefill-tile-tests'),CPU_TEST,'--exact','--nocapture'],text=True,timeout=120)
    assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',output)==[('ok','1','0','0')]
    assert stage['job_timeout_seconds']==880 and stage['worker_timeout_seconds']==820
    # Two existing 59-second workload matrices, 276 s for the 23 small
    # memcheck cases, and 180 s for fixture/setup contention fit the worker.
    budget=2*59+23*12+180
    assert budget<stage['worker_timeout_seconds']
    return dict(profiles=profiles,cases=94,memcheck_cases=23,real_captures=10,real_rows=320,
                timing_pairs=176,raw_samples=1056,conservative_budget_seconds=budget,gpu_launched=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage:report=validate(a.stage)
    else:
        report=results([(a.run/f'gpu-{i}.log').read_text() for i in range(2)])
        report['memcheck']=results([(a.run/f'memcheck-{i}.log').read_text() for i in range(2)],True)
        for i in range(2):
            text=(a.run/f'sanitizer-{i}.log').read_text()
            assert re.findall(r'ERROR SUMMARY: (\d+) errors',text)==['0']
        report['receipts']=receipts([p for i in range(2) for p in (a.run/f'receipts-{i}').rglob('*.json')])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PREFILL_TILE_ANALYSIS_COMPLETE')


if __name__=='__main__':main()
