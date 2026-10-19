#!/usr/bin/env python3
"""Require the complete guard-cost matrix, controls and raw paired samples."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import indexer_guard_job as original

ROWS = (1, 2, 4, 6, 8, 16, 32)
PAIRS = ((2, 3), (1, 3), (0, 3))  # guarded, unguarded, SIMT versus candidate
CONTEXTS = (8192,) * 10 + (32768,) * 5 + (8192,) * 5 + (32768,) * 5 + (131072,) * 5


def shapes():
    out = []
    for layer in (2, 20, 24):
        out.extend((layer, rows, offset, 0) for rows in ROWS
                   for offset in (8192, 32768, 131072))
        out.extend((layer, 8192, offset, 0) for offset in (0, 32768, 131072))
        out.extend((layer, 6, 1024, case) for case in range(1, 8))
    out.extend((layer, 6, 1024, case) for layer in (2, 20, 24) for case in range(8, 11))
    return out


def key(shape):
    return ('matrix', shape['matrix']) if 'matrix' in shape else ('capture', shape['capture'], shape['rows'])


def validate(path):
    stage = original.validate(path)
    assert 'indexer_mma::guard_fast::cpu_candidate_keeps_abi_grid_workspace_and_selection' in stage['cpu_tests']
    output = subprocess.check_output([stage['coarse_bound_binary']], text=True)
    result = re.fullmatch(r'GUARD_FAST_BOUND_COMPLETE accepted=(\d+) rejected=(\d+) negative_control_rejected=1\n', output)
    assert result and sum(map(int, result.groups())) == 100000
    assert all(int(n) > 1000 for n in result.groups())
    return stage


def results(logs):
    assert len(logs) == 2
    cases, timings = [], []
    expected = shapes()
    for shard, text in enumerate(logs):
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored', text) == [('ok', '3', '0', '0')]
        assert re.findall(r'GUARD_MATRIX_COMPLETE shard=(\d+) shapes=(\d+) total=(\d+)', text) == [(str(shard), '51', '102')]
        assert re.findall(r'GUARD_REAL_COMPLETE shard=(\d+) captures=(\d+)', text) == [(str(shard), '105')]
        assert re.findall(r'SERVING_MMA_COMPLETE shard=(\d+) comparisons=(\d+)', text) == [(str(shard), '150')]
        here = original.records(text, 'GUARD_CASE')
        matrix = [c for c in here if 'matrix' in c['shape']]
        captures = [c for c in here if 'capture' in c['shape']]
        assert sorted(c['shape']['matrix'] for c in matrix) == list(range(shard, 102, 2))
        assert sorted((c['shape']['capture'], c['shape']['rows']) for c in captures) == [
            (i, rows) for i in range(shard, 30, 2) for rows in ROWS]
        for c in matrix:
            s = c['shape']
            assert tuple(s[k] for k in ('layer', 'rows', 'offset', 'case')) == expected[s['matrix']]
        for c in captures:
            s = c['shape']
            assert s['context'] == CONTEXTS[s['capture']]
            assert s['layer'] == (2, 8, 14, 20, 24)[s['capture'] % 5]
        cases.extend(here)
        timings.extend(original.records(text, 'GUARD_TIMING'))
    assert len(cases) == 312
    for c in cases:
        assert c['guard_mismatches'] == 0 and c['fast'] is True
        assert 0 <= c['fallback_slots'] <= c['admitted']
        assert 0 <= c['coarse_slots'] <= c['admitted'] and 0 <= c['fine_slots'] <= c['admitted']
        assert c['counts_measured'] is (c['shape']['rows'] <= 32)
    controls = [c for c in cases if c['negative_control_rejected']]
    assert len(controls) == 3
    assert all(c['fallback_slots'] > 0 and c['unguarded_score_mismatches'] > 0 and c['fallback_rows'] == 0 for c in controls)
    for c in cases:
        n = c['shape'].get('case')
        if n == 8:
            assert c['coarse_slots'] == c['fallback_slots'] == 0 and c['fine_slots'] == c['admitted'] > 0
        if n == 9:
            assert 0 < c['coarse_slots'] < c['admitted'] and 0 < c['fallback_slots'] < c['fine_slots']
        if n == 10:
            assert c['coarse_slots'] == 0 and c['fallback_slots'] == c['fine_slots'] == c['admitted'] > 0
    wanted = {(key(c['shape']), pair) for c in cases
              if 'capture' in c['shape'] or c['shape']['case'] == 0 for pair in PAIRS}
    seen = [(key(t['shape']), tuple(t['arms'])) for t in timings]
    assert len(seen) == len(set(seen)) == 846 and set(seen) == wanted
    for t in timings:
        assert t['exclusive'] is True and len(t['median_ms']) == 2
        assert all(math.isfinite(v) and v > 0 for v in [*t['median_ms'], t['paired_ratio']])
    frequency = {}
    for c in cases:
        s = c['shape']
        if 'capture' not in s:
            continue
        label = f"{s['context']}/{s['rows']}"
        item = frequency.setdefault(label, dict(admitted=0, fallback_slots=0, fallback_rows=0, coarse_slots=0, fine_slots=0))
        for field in item:
            item[field] += c[field]
    assert set(frequency) == {f'{context}/{rows}' for context in (8192,32768,131072) for rows in ROWS}
    assert all(item['admitted'] > 0 for item in frequency.values())
    return dict(matrix_shapes=102, real_shapes=210, captures=30, poisoned_comparisons=300,
                negative_controls=3, timing_pairs=846, frequency=frequency, cases=cases, timings=timings)


def receipts(paths, report):
    wanted = {key(c['shape']) for c in report['cases']}
    found, measured = [], 0
    for path in paths:
        record = json.loads(path.read_text())
        if record.get('test') != 'indexer-guard-cost' and 'shape' not in record:
            continue
        assert record['measurement_label'] == 'exclusive-device' and record['fast'] is True
        assert record['mismatches'] == 0
        shape = record['shape']
        found.append(key(shape))
        rounds = record['timing_rounds']
        summaries = [r for r in rounds if r.get('kind') == 'paired_summary']
        assert len(summaries) == (3 if 'capture' in shape or shape['case'] == 0 else 0)
        for summary in summaries:
            assert summary['rounds'] == 5
            samples = [r for r in rounds if r.get('measurement') == summary['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'], r['arm']) for r in samples) == [(r,a) for r in range(5) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0 for r in samples)
            measured += 1
    assert len(found) == len(set(found)) == 312 and set(found) == wanted and measured == 846


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--logs', type=Path, nargs=2)
    parser.add_argument('--receipts', type=Path, nargs=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage:
        validate(args.stage)
        print('GUARD_FAST_CPU_DRY_RUN_COMPLETE')
    if args.logs:
        report = results([p.read_text() for p in args.logs])
        receipts([p for root in args.receipts for p in root.rglob('*.json')], report)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('GUARD_FAST_PAIR_COMPLETE matrix=102 real=210 poisoned=300 timing_pairs=846')


if __name__ == '__main__':
    main()
