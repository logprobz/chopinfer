#!/usr/bin/env python3
"""Qualify packed folds from interleaved timings and exclusive graph traces."""
import argparse
import csv
import io
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

ROWS = (1, 2, 4, 6, 8, 16, 32)
WIN = 0.97
LIMIT = 1.03


def records(text):
    for line in text.splitlines():
        offset = line.find('{')
        if offset < 0:
            continue
        try:
            item, _ = json.JSONDecoder().raw_decode(line[offset:])
        except ValueError:
            continue
        if isinstance(item, dict):
            yield item


def exclusive(events):
    """Charge overlap once and exclude idle gaps, including nested intervals."""
    frontier = 0
    charges = defaultdict(int)
    for _, start, end, name in sorted(events, key=lambda e: (e[1], e[2], e[0])):
        if end < start:
            raise ValueError('negative kernel interval')
        charges[name] += max(0, end - max(start, frontier))
        frontier = max(frontier, end)
    return dict(charges)


def graphs(path):
    with sqlite3.connect(path) as db:
        names = dict(db.execute('SELECT id,value FROM StringIds'))
        rows = db.execute('SELECT deviceId,graphId,graphNodeId,start,end,shortName '
                          'FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL '
                          'AND graphNodeId IS NOT NULL ORDER BY start').fetchall()
    grouped = defaultdict(list)
    for device, graph, node, start, end, name in rows:
        if graph:
            grouped[device, graph].append((node, start, end, names[name]))
    result = []
    for key, events in grouped.items():
        width = len({e[0] for e in events})
        if not any('mxfp8_' in e[3] for e in events):
            continue
        assert len(events) % width == 0, (key, 'incomplete graph')
        replay = [events[i:i+width] for i in range(0, len(events), width)]
        symbols = sorted(e[3] for e in replay[0])
        assert all(sorted(e[3] for e in r) == symbols for r in replay), key
        result.append(dict(key=key, start=events[0][1], symbols=symbols, replays=replay))
    return sorted(result, key=lambda g: g['start'])


def key(r):
    return r['n'], r['k'], r['groups'], r.get('paired', False), r['rows']


def with_trace(comparisons, trace, ahead=False):
    candidates = [g for g in trace if len(g['replays']) == (8 if ahead else 15)
                  and (any(s == 'chop_mxfp8_stream_prefetch_v1' for s in g['symbols'])
                       if ahead else len(g['symbols']) == 1 and '_mxfp8_stream_' in g['symbols'][0])]
    baselines = [g for g in trace if len(g['replays']) == (8 if ahead else 7)
                 and (len(g['symbols']) in (2,3) and sum('_mxfp8_stream_' in s for s in g['symbols']) == 2
                      if ahead else all('_mxfp8_stream_' not in s for s in g['symbols']))]
    assert len(candidates) == len(baselines) == len(comparisons), (len(candidates), len(baselines), len(comparisons))
    for r, a, b in zip(comparisons, baselines, candidates):
        assert a['symbols'] == sorted(r['baseline_symbols'])
        assert b['symbols'] == sorted(r['candidate_symbols'])
        stages = [[exclusive(replay) for replay in g['replays'][-5:]] for g in (a,b)]
        elapsed = [[sum(s.values()) / 1000 for s in arm] for arm in stages]
        assert all(math.isfinite(t) and t > 0 for arm in elapsed for t in arm)
        ratio = statistics.median(y/x for x,y in zip(*elapsed))
        r.update(exclusive_baseline_us=statistics.median(elapsed[0]),
                 exclusive_candidate_us=statistics.median(elapsed[1]), exclusive_paired_ratio=ratio,
                 exclusive_stages=[{s: statistics.median(sample.get(s,0) for sample in arm)/1000
                                    for s in {name for sample in arm for name in sample}} for arm in stages])
        if not ahead:
            r['exclusive_floor_ratio'] = r['exclusive_candidate_us']/r['floor_us']
    return comparisons


def traffic(path):
    text = Path(path).read_text()
    start = next((i for i,l in enumerate(text.splitlines()) if l.startswith('"ID",')), None)
    assert start is not None, f'no profiler CSV header: {path}'
    samples = defaultdict(dict)
    for r in csv.DictReader(io.StringIO('\n'.join(text.splitlines()[start:]))):
        if 'graph' not in r.get('Kernel Name','').lower():
            continue
        metric = r.get('Metric Name')
        if metric not in ('dram__bytes_read.sum','dram__bytes_write.sum'):
            continue
        assert r['Metric Unit'].lower() in ('byte','bytes'), r['Metric Unit']
        samples[r['ID']][metric] = float(r['Metric Value'].replace(',',''))
    assert len(samples) == 4, (path, len(samples))
    ordered = [samples[k] for k in sorted(samples,key=int)]
    assert all(len(r)==2 for r in ordered), path
    return statistics.median(sum(r.values()) for r in ordered[-3:])


def select(rows, ahead=False):
    by_key=defaultdict(list)
    for r in rows:
        assert r['bitwise']
        assert math.isfinite(r['paired_ratio']) and r['paired_ratio'] > 0
        by_key[key(r)].append(r)
    for k, group in by_key.items():
        assert len(group)==2 and {r['device'] for r in group}=={0,1}, k
        win=all(r['paired_ratio'] < WIN and r['exclusive_paired_ratio'] < WIN for r in group)
        if ahead:
            win &= all(r.get('dram_traffic_proven', False) and r['candidate_dram_bytes'] <= r['baseline_dram_bytes'] for r in group)
        for r in group:
            r['selected'] = ('ahead' if ahead else 'stream') if win else r.get('baseline_routing','baseline')
            r['selected_paired_ratio'] = r['paired_ratio'] if win else 1.0
            r['selected_exclusive_ratio'] = r['exclusive_paired_ratio'] if win else 1.0
            assert max(r['selected_paired_ratio'],r['selected_exclusive_ratio']) <= LIMIT
    return rows


def check_serving(rows, enabled):
    selected = [r for r in rows if key(r) in enabled]
    assert {key(r) for r in selected} == enabled
    assert len(selected) == 2 * len(enabled)
    assert all(r['bitwise'] and 0 < r['paired_ratio'] <= LIMIT
               and 0 < r['exclusive_paired_ratio'] <= LIMIT for r in selected)
    return len(selected)


def summarize(root):
    root=Path(root)
    inventory=json.loads((root/'inventory.json').read_text())
    expected={tuple(r) for r in inventory['matrix']}
    assert len(expected)==23*len(ROWS)
    all_rows=[];ahead_rows=[]
    for device in (0,1):
        raw=list(records((root/f'gpu{device}.log').read_text()))
        comparisons=[r for r in raw if r.get('phase')=='stream-comparison']
        assert len(comparisons)==len(expected) and {key(r) for r in comparisons}==expected
        assert all(r['device']==device and r['bitwise'] and r['decode_launches']==1 for r in comparisons)
        assert all(r.get('concurrent_replays')==3 for r in comparisons if r['rows']==1)
        prefill=[r for r in raw if r.get('phase')=='preserved-prefill']
        assert len(prefill)==20 and all(r['rows']==128 and r['bitwise'] for r in prefill)
        deferred=[r for r in raw if r.get('phase')=='deferred-admission']
        assert len(deferred)==2 and {r['rows'] for r in deferred}=={1,32}
        assert all(r.get('bitwise',True) for r in raw)
        all_rows.extend(with_trace(comparisons,graphs(root/f'fold{device}.sqlite')))
        ahead=[r for r in records((root/f'ahead{device}.log').read_text()) if r.get('phase')=='ahead-comparison']
        assert len(ahead)==4
        sequence=[r for r in records((root/f'sublayer{device}.log').read_text()) if r.get('phase')=='ahead-comparison']
        assert len(sequence)==3*len(ROWS)
        assert {(r['ranks'],r['rows']) for r in sequence}=={(tp,rows) for tp in (1,2,4) for rows in ROWS}
        for r in ahead+sequence:
            assert r['device']==device and r['bitwise']
            r.update(dram_traffic_proven=False,baseline_dram_bytes=None,candidate_dram_bytes=None,
                     dram_ratio=None,look_ahead_qualified=False)
        ahead_rows.extend(with_trace(ahead,graphs(root/f'ahead{device}.sqlite'),ahead=True))
        ahead_rows.extend(with_trace(sequence,graphs(root/f'sublayer{device}.sqlite'),ahead=True))
    serving={tuple(r) for r in inventory.get('serving_matrix',[])}
    serving_cases=check_serving(all_rows,serving)
    select(all_rows);select(ahead_rows,ahead=True)
    (root/'routed-matrix.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    (root/'ahead-matrix.json').write_text(json.dumps(ahead_rows,indent=2)+'\n')
    columns=['device','n','k','groups','paired','rows','ranks','baseline_routing','baseline_us','decode_us',
             'exclusive_baseline_us','exclusive_candidate_us','floor_us','exclusive_floor_ratio','paired_ratio','exclusive_paired_ratio','selected']
    with (root/'timings.csv').open('w') as f:
        w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    verdict=dict(passed=True,bitwise=True,timed_cases=len(all_rows),qualified_buckets=sum(r['selected']=='stream' for r in all_rows)//2,
                 ahead_qualified_buckets=sum(r['selected']=='ahead' for r in ahead_rows)//2,
                 routed_regressions=0,regression_limit=LIMIT,win_limit=WIN,
                 serving_routing_changed=bool(serving),serving_routed_cases=serving_cases,requires_integration=True,
                 sublayer_comparisons=sum(r.get('sequence')=='shared-expert' for r in ahead_rows),
                 look_ahead_dram_traffic_proven=False,look_ahead_qualified=False)
    (root/'routing-verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');summarize(parser.parse_args().run)
