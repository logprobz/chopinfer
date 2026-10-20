#!/usr/bin/env python3
"""Require complete early-stop selection evidence before serving it."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import indexer_candidate_cached_job as cached
import indexer_prefill_tile_job as base

CPU_TEST='indexer_mma::prefill::tile::early::cpu_early_selection_inventory'
GPU_TEST='indexer_mma::prefill::tile::early::gpu_early_selection_real_edges_and_timing'
PATTERNS=6
STAGES=('select','all')


def cases():
    return base.cases()


def crafted(c):
    return c['capture'] is None and c['kind']==0 and c['rows']==65 and c['context']==32768


def selected(i,c,memcheck):
    return not memcheck or i<5 or c['kind']==3 or c['kind']>=8 or crafted(c)


def comparisons(c):
    return 4+(PATTERNS if crafted(c) else 0)


def results(logs,memcheck=False):
    shapes=cases();seen=[];times=[]
    assert len(logs)==2
    for shard,text in enumerate(logs):
        expected={i for i,c in enumerate(shapes) if i%2==shard and selected(i,c,memcheck)}
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text)==[('ok','1','0','0')]
        assert re.findall(r'SELECT_EARLY_COMPLETE shard=(\d+) cases=(\d+) memcheck=(\w+)',text)==[(str(shard),str(len(expected)),str(memcheck).lower())]
        found=base.records(text,'SELECT_EARLY_CASE')
        assert len(found)==len(expected) and {c['id'] for c in found}==expected
        for row in found:
            want=shapes[row['id']]
            assert {k:row[k] for k in want}==want
            assert row['live']==want['rows']-int(want['kind']==12)
            assert row['kept']==(31 if want['kind']==11 else 512)
            assert row['batches']==base.batch_count(want)
            assert row['comparisons']==comparisons(want)
            assert row['patterns']==(PATTERNS if crafted(want) else 0)
            assert row['controls']==2 and row['mismatches']==0
            assert row['timed'] is (base.timed(want) and not memcheck)
            assert row['memcheck'] is memcheck
        seen.extend(found);times.extend(base.records(text,'SELECT_EARLY_TIMING'))
    wanted={(i,s) for i,c in enumerate(shapes) if base.timed(c) and not memcheck for s in STAGES}
    assert len(times)==len(wanted) and {(t['id'],t['stage']) for t in times}==wanted
    for t in times:
        assert t['exclusive'] is True and t['score_buffer_reused'] is True
        assert t['batches']==next(c['batches'] for c in seen if c['id']==t['id'])
        assert len(t['median_ms'])==2 and all(math.isfinite(x) and x>0 for x in [*t['median_ms'],t['paired_ratio']])
    assert len(seen)==(26 if memcheck else 94)
    return dict(cases=len(seen),comparisons=sum(c['comparisons'] for c in seen),
                patterns=sum(c['patterns'] for c in seen),controls=2*len(seen),mismatches=0,
                timing_pairs=len(times),records=seen,timings=times,summary=summary(times))


def summary(times):
    """Total served and early medians per stage over the timed cases."""
    out={}
    for stage in STAGES:
        rows=[t for t in times if t['stage']==stage]
        if not rows:continue
        served=sum(t['median_ms'][0] for t in rows);early=sum(t['median_ms'][1] for t in rows)
        out[stage]=dict(cases=len(rows),served_ms=served,early_ms=early,saving_ms=served-early,
                        worst_paired_ratio=max(t['paired_ratio'] for t in rows))
    return out


def receipts(paths):
    ids=[];pairs=samples=0
    for path in paths:
        row=json.loads(path.read_text())
        if 'id' not in row:continue
        i=row['id'];assert 0<=i<94 and row['mismatches']==0
        timed=base.timed(cases()[i])
        assert row['timed'] is timed
        assert row['measurement_label']=='exclusive-device'
        ids.append(i)
        rounds=row['timing_rounds']
        summaries=[r for r in rounds if r.get('kind')=='paired_summary']
        assert {s['operation'] for s in summaries}==({f'select-early-{i}-{stage}' for stage in STAGES} if timed else set())
        assert len(summaries)==(2 if timed else 0)
        for s in summaries:
            assert s['rounds']==3
            raw=[r for r in rounds if r.get('measurement')==s['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'],r['arm']) for r in raw)==[(r,a) for r in range(3) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>0 for r in raw)
            pairs+=1;samples+=len(raw)
    assert sorted(ids)==list(range(94)) and pairs==44 and samples==264
    return dict(pairs=pairs,samples=samples)


def replay(text,memcheck=False):
    """Rewrite a recorded tile log into this job's records in place.

    The recorded test output keeps its real prefixes, interleaving and
    result lines; only the marker payloads change. Memcheck selection here
    adds the crafted cases, which the tile job did not run under memcheck.
    """
    out=[];shapes=cases();count=0
    for line in text.splitlines():
        if 'PREFILL_TILE_TIMING ' in line or 'PREFILL_TILE_GUARDS ' in line:continue
        if 'PREFILL_TILE_CASE ' in line:
            prefix,payload=line.split('PREFILL_TILE_CASE ',1);row=json.loads(payload);c=shapes[row['id']]
            if not selected(row['id'],c,memcheck):continue
            count+=1
            early={k:row[k] for k in (*c,'id','live','kept','batches','controls','mismatches','timed','memcheck')}
            early.update(comparisons=comparisons(c),patterns=PATTERNS if crafted(c) else 0)
            out.append(prefix+'SELECT_EARLY_CASE '+json.dumps(early))
            for stage in STAGES if row['timed'] else ():
                out.append('SELECT_EARLY_TIMING '+json.dumps(dict(id=row['id'],stage=stage,batches=row['batches'],
                    median_ms=[1.,.5],paired_ratio=.5,exclusive=True,score_buffer_reused=True)))
            continue
        if 'PREFILL_TILE_COMPLETE ' in line:
            shard=re.search(r'shard=(\d+)',line)[1]
            missing=[i for i,c in enumerate(shapes) if i%2==int(shard) and selected(i,c,memcheck) and i not in base.selected(memcheck)]
            for i in missing:
                c=shapes[i];count+=1
                out.append('SELECT_EARLY_CASE '+json.dumps(dict(c,id=i,live=c['rows']-int(c['kind']==12),
                    kept=31 if c['kind']==11 else 512,batches=base.batch_count(c),controls=2,mismatches=0,
                    timed=False,memcheck=memcheck,comparisons=comparisons(c),patterns=PATTERNS if crafted(c) else 0)))
            line=line.split('PREFILL_TILE_COMPLETE ',1)[0]+f'SELECT_EARLY_COMPLETE shard={shard} cases={count} memcheck={str(memcheck).lower()}'
        out.append(line)
    return '\n'.join(out)


def sanitizer(text):
    assert re.findall(r'ERROR SUMMARY: (\d+) errors',text)==['0']


def validate(path):
    root=path.parent;stage=json.loads(path.read_text())
    for name,sha in stage['sha256'].items():
        p=root/name
        assert not p.is_symlink() and base.digest(p)==sha and p.stat().st_mode&0o022==0,name
    captures=json.loads((root/'snapshot/captures.json').read_text())['captures']
    assert len(captures)==10
    for i,c in enumerate(captures):
        p=Path(c['path']);meta=json.loads((p/'operands.json').read_text())
        assert meta['context']==cases()[i]['context'] and meta['layer']==cases()[i]['layer']
        assert meta['phase_rows']==8192 and meta['sample_rows']==[j*8191//31 for j in range(32)]
        completion=json.loads((p/'../completion.json').read_text())
        assert completion['completion']=='full' and completion['observed']==completion['expected']
        for name,sha in c['sha256'].items():assert base.digest(p/name)==sha
    # The early module is the one the lab lock and the bundle lock pin.
    declared=json.loads((root/'snapshot/select-early-modules.json').read_text())
    assert len(declared)==1 and len(declared[0]['symbols'])==6
    module=root/'snapshot'/(declared[0]['module']+'.cubin')
    assert base.digest(module)==declared[0]['sha256']==stage['lab_lock_cubin_sha256']==stage['bundle_lock_cubin_sha256']
    # Replay the served-selector evidence this job compares against: the
    # tile qualification and the fused publisher run, with their parsers.
    for key,parse in (('tile_run',base),('candidate_run',cached)):
        run=Path(stage[key]['path'])
        for name,sha in stage[key]['sha256'].items():assert base.digest(run/name)==sha,(key,name)
        for i in (0,1):sanitizer((run/f'sanitizer-{i}.log').read_text())
        gpu=parse.results([(run/f'gpu-{i}.log').read_text() for i in (0,1)])
        memcheck=parse.results([(run/f'memcheck-{i}.log').read_text() for i in (0,1)],True)
        receipts_=parse.receipts([p for i in (0,1) for p in (run/f'receipts-{i}').rglob('*.json')])
        assert [gpu['cases'],gpu['timing_pairs'],memcheck['cases'],receipts_['pairs']]==stage[key]['expected'],key
    # The parser accepts the recorded tile output's real formatting.
    run=Path(stage['tile_run']['path'])
    for memcheck,name in ((False,'gpu'),(True,'memcheck')):
        replayed=results([replay((run/f'{name}-{i}.log').read_text(),memcheck) for i in (0,1)],memcheck)
        assert replayed['cases']==(26 if memcheck else 94) and replayed['timing_pairs']==(0 if memcheck else 44)
    binary=str(root/'snapshot/prefill-tile-tests')
    listed=subprocess.check_output([binary,'--list'],text=True,timeout=30)
    for test in (CPU_TEST,GPU_TEST):
        assert sum(line.startswith(test+': test') for line in listed.splitlines())==1
    assert GPU_TEST in (root/'snapshot/worker.sh').read_text()
    text=subprocess.check_output([binary,CPU_TEST,'--exact','--nocapture'],text=True,timeout=120)
    assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text)==[('ok','1','0','0')]
    assert stage['job_timeout_seconds']==880 and stage['worker_timeout_seconds']==820
    assert stage['cases']==94 and stage['memcheck_cases']==26
    # The tile run's 59-second matrices carried four arms and eight timing
    # pairs per timed case; this job runs two arms and two pairs. Budget
    # the full matrix twice, 12 s per memcheck case and 180 s of setup.
    budget=2*59+26*12+180
    assert budget<stage['worker_timeout_seconds']
    return dict(cases=94,memcheck_cases=26,crafted_cases=3,patterns=3*PATTERNS,timing_pairs=44,
                raw_samples=264,conservative_budget_seconds=budget,gpu_launched=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage:report=validate(a.stage)
    else:
        report=results([(a.run/f'gpu-{i}.log').read_text() for i in (0,1)])
        report['memcheck']=results([(a.run/f'memcheck-{i}.log').read_text() for i in (0,1)],True)
        for i in (0,1):sanitizer((a.run/f'sanitizer-{i}.log').read_text())
        report['receipts']=receipts([p for i in (0,1) for p in (a.run/f'receipts-{i}').rglob('*.json')])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('SELECT_EARLY_ANALYSIS_COMPLETE')


if __name__=='__main__':main()
