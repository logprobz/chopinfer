#!/usr/bin/env python3
"""Check every frozen prefill batch case, negative control and paired interval."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

GPU_TEST = 'indexer_mma::prefill::gpu_batched_prefill_bitwise_and_timing'
CPU_TEST = 'indexer_mma::prefill::cpu_prefill_plans_fit_existing_allocations'
GUARD_FIXTURE_TEST = 'indexer_mma::guard::cpu_compact_matrix_uses_distinct_tags'


def shapes():
    out = []
    for layer in (2, 20):
        out.extend(dict(layer=layer, context=context, rows=rows, case=0)
                   for context in (8192, 32768, 131072)
                   for rows in (8192, 8191, 4096, 2496, 1984, 448, 65))
        out.extend(dict(layer=layer, context=8192, rows=65, case=case) for case in range(1, 7))
    return out


def passed(text, count):
    assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored', text) == [
        ('ok', str(count), '0', '0')], 'missing, skipped or failing test result'


def records(text, marker):
    return [json.loads(line.split(marker + ' ', 1)[1])
            for line in text.splitlines() if marker + ' ' in line]


def validate(path):
    stage = json.loads(path.read_text())
    for name, digest in stage['sha256'].items():
        artifact = Path(name)
        assert artifact.is_absolute(), name
        assert artifact.stat().st_mode & 0o022 == 0, name
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest, name
    listed = subprocess.check_output([stage['binary'], '--list'], text=True)
    for name in (CPU_TEST, GPU_TEST, GUARD_FIXTURE_TEST):
        assert f'{name}: test' in listed, name
    for name in (CPU_TEST, GUARD_FIXTURE_TEST):
        result = subprocess.check_output([stage['binary'], name, '--exact', '--nocapture'], text=True)
        passed(result, 1)
    result = subprocess.check_output([stage['contract_binary'], 'indexer::prefill::tests', '--nocapture'], text=True)
    passed(result, 3)
    assert stage['shapes'] == shapes()
    return stage


def results(logs):
    assert len(logs) == 2
    expected = shapes()
    all_cases, all_times = [], []
    for shard, text in enumerate(logs):
        passed(text, 1)
        footer = re.findall(r'PREFILL_BATCH_COMPLETE shard=(\d+) shapes=(\d+) total=(\d+)', text)
        assert footer == [(str(shard), '27', '54')], footer
        cases = records(text, 'PREFILL_BATCH_CASE')
        assert sorted(c['id'] for c in cases) == list(range(shard, len(expected), 2))
        for c in cases:
            assert {key: c[key] for key in ('layer', 'context', 'rows', 'case')} == expected[c['id']]
            assert c['mismatches'] == 0 and c['negative_control_rejected'] is True
            assert c['candidate_planes'] is (c['layer'] == 20)
            assert c['score_bytes'] <= 256 << 20 and c['workspace_bytes'] <= 546848768
            assert 1 <= c['new_batches'] <= c['old_batches']
        all_cases.extend(cases)
        all_times.extend(records(text, 'PREFILL_BATCH_TIMING'))
    wanted = {(i, stage) for i, s in enumerate(expected) if s['case'] == 0
              for stage in (('all', 'pack_score') if s['layer'] == 2 else ('all', 'pack_score', 'pack_score_select'))}
    actual = [(t['id'], t['stage']) for t in all_times]
    assert len(set(actual)) == len(actual) and set(actual) == wanted
    for t in all_times:
        assert t['exclusive'] is True and len(t['median_ms']) == 2
        assert all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in [*t['median_ms'], t['paired_ratio']])
    assert len(all_cases) == 54 and len(all_times) == 105
    return dict(shapes=54, negative_controls=54, timing_pairs=105,
                cases=sorted(all_cases, key=lambda c: c['id']), timings=all_times)


def receipts(paths):
    cases = []
    measured = 0
    for path in paths:
        record = json.loads(path.read_text())
        assert record['measurement_label'] == 'exclusive-device', path
        assert record['mismatches'] == 0
        shape = shapes()[record['id']]
        rounds = record['timing_rounds']
        summaries = [r for r in rounds if r.get('kind') == 'paired_summary']
        expected = 0 if shape['case'] else (2 if shape['layer'] == 2 else 3)
        assert len(summaries) == expected, path
        for summary in summaries:
            assert summary['rounds'] == 3
            samples = [r for r in rounds if r.get('measurement') == summary['measurement'] and 'elapsed_ms' in r]
            assert sorted((r['round'], r['arm']) for r in samples) == [(r, a) for r in range(3) for a in range(2)]
            assert all(math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0 for r in samples)
            measured += 1
        cases.append(record['id'])
    assert sorted(cases) == list(range(54)) and measured == 105


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--logs', type=Path, nargs=2)
    parser.add_argument('--receipts', type=Path, nargs=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage:
        validate(args.stage)
        print('PREFILL_BATCH_CPU_DRY_RUN_COMPLETE')
    if args.logs:
        report = results([p.read_text() for p in args.logs])
        receipts([p for root in args.receipts for p in root.rglob('*.json')])
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('PREFILL_BATCH_PAIR_COMPLETE shapes=54 negative_controls=54 timing_pairs=105')


if __name__ == '__main__':
    main()
