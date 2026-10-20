#!/usr/bin/env python3
"""Require complete fused candidate publication evidence before selection."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import indexer_prefill_tile_job as base

CPU_TEST='indexer_mma::prefill::tile::publish::cpu_publication_inventory_and_admission'
GPU_TEST='indexer_mma::prefill::tile::publish::gpu_publication_real_edges_and_timing'


def cases():
    rows=[]
    for source,c in enumerate(c for c in base.cases() if c['layer']==20):
        for kept in (1,31,2048):
            rows.append(dict(direct=False,source=source,context=c['context'],rows=c['rows'],
                             kind=c['kind'],capture=c['capture'],kept=kept))
    for width in (1,8,512,16384,16385,32768):
        for pattern in range(3):
            for kept in (1,31,2048):
                for live in (4,8):
                    rows.append(dict(direct=True,width=width,pattern=pattern,kept=kept,live=live))
    assert len(rows)==198
    return rows


def timed(c):
    return not c['direct'] and (c['capture'] is not None or c['kind']==0 and c['rows'] in (2496,8192))


def selected(c,memcheck):
    if not memcheck:return True
    if c['direct']:return c['width'] in (1,16385,32768) and c['pattern']==2 and c['live']==8
    return c['capture'] is not None or c['kind'] in (3,6,7,8,9,10,12)


def results(logs,memcheck=False):
    shapes=cases();seen=[];times=[]
    assert len(logs)==2
    for shard,text in enumerate(logs):
        expected={i for i,c in enumerate(shapes) if i%2==shard and selected(c,memcheck)}
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text)==[('ok','1','0','0')]
        assert re.findall(r'CANDIDATE_CACHED_COMPLETE shard=(\d+) cases=(\d+) memcheck=(\w+)',text)==[(str(shard),str(len(expected)),str(memcheck).lower())]
        found=base.records(text,'CANDIDATE_CACHED_CASE')
        assert len(found)==len(expected) and {c['id'] for c in found}==expected
        for row in found:
            want=shapes[row['id']]
            assert {k:row[k] for k in want}==want
            assert row['comparisons']==(4 if want['direct'] else 8)
            assert row['controls']==2 and row['mismatches']==0
            assert row['timed'] is (timed(want) and not memcheck)
            assert row['memcheck'] is memcheck
        seen.extend(found);times.extend(base.records(text,'CANDIDATE_CACHED_TIMING'))
    wanted={(i,s) for i,c in enumerate(shapes) if timed(c) and not memcheck for s in ('publish','all')}
    assert len(times)==len(wanted) and {(t['id'],t['stage']) for t in times}==wanted
    for t in times:
        assert t['exclusive'] is True
        assert t['score_buffer_reused'] is (t['stage']=='all')
        assert len(t['median_ms'])==2 and all(math.isfinite(x) and x>0 for x in [*t['median_ms'],t['paired_ratio']])
    assert len(seen)==(36 if memcheck else 198)
    return dict(cases=len(seen),comparisons=sum(c['comparisons'] for c in seen),
                controls=2*len(seen),mismatches=0,timing_pairs=len(times),records=seen,timings=times)


def receipts(paths):
    ids=[];pairs=samples=0
    for path in paths:
        row=json.loads(path.read_text())
        if 'id' not in row:continue
        i=row['id'];assert 0<=i<198 and row['mismatches']==0
        assert row['timed'] is timed(cases()[i])
        assert row['measurement_label']=='exclusive-device'
        ids.append(i)
        rounds=row['timing_rounds']
        summaries=[r for r in rounds if r.get('kind')=='paired_summary']
        assert {s['operation'] for s in summaries}==({f'candidate-cached-{i}-{stage}' for stage in ('publish','all')} if timed(cases()[i]) else set())
        assert len(summaries)==(2 if timed(cases()[i]) else 0)
        for summary in summaries:
            assert summary['rounds']==3
            raw=[r for r in rounds if r.get('measurement')==summary['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'],r['arm']) for r in raw)==[(r,a) for r in range(3) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>0 for r in raw)
            pairs+=1;samples+=len(raw)
    assert sorted(ids)==list(range(198)) and pairs==60 and samples==360
    return dict(pairs=pairs,samples=samples)


def validate(path):
    root=path.parent;stage=json.loads(path.read_text())
    for name,sha in stage['sha256'].items():
        p=root/name
        assert not p.is_symlink() and base.digest(p)==sha and p.stat().st_mode&0o022==0,name
    captures=json.loads((root/'snapshot/captures.json').read_text())['captures']
    assert len(captures)==10
    for i,c in enumerate(captures):
        p=Path(c['path']);meta=json.loads((p/'operands.json').read_text())
        assert meta['context']==base.cases()[i]['context'] and meta['layer']==base.cases()[i]['layer']
        assert meta['phase_rows']==8192 and meta['sample_rows']==[j*8191//31 for j in range(32)]
        completion=json.loads((p/'../completion.json').read_text())
        assert completion['completion']=='full' and completion['observed']==completion['expected']
        for name,sha in c['sha256'].items():assert base.digest(p/name)==sha
    profiles=[]
    assert len(stage['profiles'])==2
    for p in stage['profiles']:
        assert base.digest(Path(p['path']))==p['sha256']
        profiles.append(base.profile(Path(p['path'])))
    # Replay the executed scorer result on CPU; do not spend another GPU
    # job re-running it because its old batch-count parser was incorrect.
    run=Path(stage['baseline_run'])
    for name,sha in stage['baseline_sha256'].items():assert base.digest(run/name)==sha
    prior=base.results([(run/f'gpu-{i}.log').read_text() for i in (0,1)])
    assert prior['cases']==94 and prior['timing_pairs']==176
    assert base.results([(run/f'memcheck-{i}.log').read_text() for i in (0,1)],True)['cases']==23
    assert base.receipts([p for i in (0,1) for p in (run/f'receipts-{i}').rglob('*.json')])==dict(pairs=176,samples=1056)
    for i in (0,1):
        assert re.findall(r'ERROR SUMMARY: (\d+) errors',(run/f'sanitizer-{i}.log').read_text())==['0']
    binary=str(root/'snapshot/prefill-tile-tests')
    listed=subprocess.check_output([binary,'--list'],text=True,timeout=30)
    for test in (CPU_TEST,GPU_TEST):
        assert sum(line.startswith(test+': test') for line in listed.splitlines())==1
    assert GPU_TEST in (root/'snapshot/worker.sh').read_text()
    text=subprocess.check_output([binary,CPU_TEST,'--exact','--nocapture'],text=True,timeout=120)
    assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text)==[('ok','1','0','0')]
    assert stage['job_timeout_seconds']==880 and stage['worker_timeout_seconds']==820
    assert stage['cases']==198 and stage['memcheck_cases']==36
    return dict(cases=198,memcheck_cases=36,timing_pairs=60,raw_samples=360,
                baseline_cases=94,profiles=profiles,gpu_launched=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage:report=validate(a.stage)
    else:
        report=results([(a.run/f'gpu-{i}.log').read_text() for i in (0,1)])
        report['memcheck']=results([(a.run/f'memcheck-{i}.log').read_text() for i in (0,1)],True)
        for i in (0,1):
            assert re.findall(r'ERROR SUMMARY: (\d+) errors',(a.run/f'sanitizer-{i}.log').read_text())==['0']
        report['receipts']=receipts([p for i in (0,1) for p in (a.run/f'receipts-{i}').rglob('*.json')])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('CANDIDATE_CACHED_ANALYSIS_COMPLETE')


if __name__=='__main__':main()
