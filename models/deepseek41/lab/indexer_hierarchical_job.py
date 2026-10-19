#!/usr/bin/env python3
"""Check hierarchical selection coverage and every exclusive timing sample."""
import argparse
import json
import math
from pathlib import Path
import re
import indexer_guard_fast_job as guard
from indexer_guard_job import records

STAGES = ('merge', 'select', 'all')


def shapes():
    result = guard.shapes()
    for layer in (2,20,24):
        result.extend((layer,rows,1024,0) for rows in (1,32))
        result.extend((layer,6,131072,case) for case in range(20,33))
    assert len(result) == 147
    return result


def validate(path):
    stage = guard.validate(path)
    for name in ('cpu_hierarchy_keeps_original_keys_abi_and_scratch_disjoint',
                 'cpu_group_union_preserves_exact_keys_and_order',
                 'cpu_edge_scores_keep_signed_zero_and_poisoned_tails'):
        assert 'indexer_mma::hierarchical::'+name in stage['cpu_tests']
    return stage


def results(logs):
    assert len(logs) == 2
    expected = shapes()
    cases, timings = [], []
    for shard,text in enumerate(logs):
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',text) == [('ok','3','0','0')]
        assert re.findall(r'GUARD_MATRIX_COMPLETE shard=(\d+) shapes=(\d+) total=(\d+)',text) == [(str(shard),str(74-shard),'147')]
        assert re.findall(r'GUARD_REAL_COMPLETE shard=(\d+) captures=(\d+)',text) == [(str(shard),'105')]
        assert re.findall(r'SERVING_MMA_COMPLETE shard=(\d+) comparisons=(\d+)',text) == [(str(shard),'150')]
        here = records(text,'HIER_CASE')
        matrix = [c for c in here if 'matrix' in c['shape']]
        real = [c for c in here if 'capture' in c['shape']]
        assert sorted(c['shape']['matrix'] for c in matrix) == list(range(shard,147,2))
        assert sorted((c['shape']['capture'],c['shape']['rows']) for c in real) == [(i,r) for i in range(shard,30,2) for r in guard.ROWS]
        for c in matrix:
            s = c['shape']
            assert tuple(s[k] for k in ('layer','rows','offset','case')) == expected[s['matrix']]
            assert s['top_k'] == {21:128,22:1,23:511}.get(s['case'],512)
        for c in real:
            s = c['shape']
            assert s['context'] == guard.CONTEXTS[s['capture']]
            assert s['layer'] == (2,8,14,20,24)[s['capture']%5]
        for c in here:
            assert c['mismatches'] == 0
            assert c['old_launches'] == 4 and c['new_launches'] == 5
            assert c['intermediate_bytes'] == c['shape']['rows']*4*513*8
            control = c['shape'].get('case') == 20
            assert c['negative_control_rejected'] is control
            assert (c['negative_mismatches'] > 0) is control
        cases.extend(here)
        timings.extend(records(text,'HIER_TIMING'))
    assert len(cases) == 357
    wanted = {(guard.key(c['shape']),stage) for c in cases
              if c['shape'].get('case',0) == 0 for stage in STAGES}
    actual = [(guard.key(t['shape']),t['stage']) for t in timings]
    assert len(actual) == len(set(actual)) == 864 and set(actual) == wanted
    for t in timings:
        assert t['exclusive'] is True and len(t['median_ms']) == 2
        assert all(math.isfinite(x) and x > 0 for x in [*t['median_ms'],t['paired_ratio']])
    return dict(matrix_shapes=147,real_shapes=210,captures=30,poisoned_comparisons=300,
                negative_controls=3,timing_pairs=864,cases=cases,timings=timings)


def receipts(paths, report):
    found, measured = [], 0
    for path in paths:
        record = json.loads(path.read_text())
        if 'shape' not in record:
            continue
        assert record['measurement_label'] == 'exclusive-device' and record['mismatches'] == 0
        shape = record['shape']
        found.append(guard.key(shape))
        rounds = record['timing_rounds']
        summaries = [r for r in rounds if r.get('kind') == 'paired_summary']
        assert len(summaries) == (3 if shape.get('case',0) == 0 else 0)
        for summary in summaries:
            assert summary['rounds'] == 5
            samples = [r for r in rounds if r.get('measurement') == summary['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'],r['arm']) for r in samples) == [(r,a) for r in range(5) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0 for r in samples)
            measured += 1
    assert len(found) == len(set(found)) == 357 and measured == 864
    assert set(found) == {guard.key(c['shape']) for c in report['cases']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--logs',type=Path,nargs=2)
    parser.add_argument('--receipts',type=Path,nargs=2)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.stage:
        validate(args.stage)
        print('HIER_CPU_DRY_RUN_COMPLETE')
    if args.logs:
        report = results([p.read_text() for p in args.logs])
        receipts([p for root in args.receipts for p in root.rglob('*.json')],report)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('HIER_PAIR_COMPLETE matrix=147 real=210 poisoned=300 negative_controls=3 timing_pairs=864')


if __name__ == '__main__':
    main()
