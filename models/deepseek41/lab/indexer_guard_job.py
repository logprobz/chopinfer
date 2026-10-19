#!/usr/bin/env python3
"""Verify sealed inputs and require every guarded-indexer job marker."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(path.read_text())


def validate(stage_path):
    stage = read(stage_path)
    assert 'indexer_mma::guard::cpu_compact_matrix_uses_distinct_tags' in stage['cpu_tests'], \
        'dry run must exercise the compact GPU fixture builder'
    for name, digest in stage['sha256'].items():
        path = Path(name)
        assert path.is_absolute() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        assert path.stat().st_mode & 0o022 == 0, name
    verifier = Path(stage['verifier'])
    spec = importlib.util.spec_from_file_location('capture_verifier', verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, captures = module.verify_capture(Path(stage['qualification']))
    assert str(manifest) == stage['manifest'] and len(captures['captures']) == 30
    listed = subprocess.check_output([stage['binary'], '--list'], text=True)
    for test in stage['tests']:
        assert f'{test}: test' in listed, test
    for test in stage['cpu_tests']:
        result = subprocess.check_output([stage['binary'], '--exact', test, '--nocapture'], text=True)
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored', result) == [('ok', '1', '0', '0')]
    result = subprocess.check_output([stage['bound_binary']], text=True)
    assert result == 'GUARD_BOUND_COMPLETE accepted=20000 negative_control_rejected=1\n'
    return stage


def records(text, marker):
    # libtest may put its test name before the first marker on a line.
    return [json.loads(line.split(marker + ' ', 1)[1]) for line in text.splitlines() if marker + ' ' in line]


def results(logs):
    for log in logs:
        assert re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored', log) == [('ok', '3', '0', '0')]
    text = '\n'.join(logs)
    matrix = re.findall(r'GUARD_MATRIX_COMPLETE shard=(\d+) shapes=(\d+) total=(\d+)', text)
    assert sorted(int(x[0]) for x in matrix) == [0, 1]
    assert sum(int(x[1]) for x in matrix) == 93 and all(x[2] == '93' for x in matrix)
    real = re.findall(r'GUARD_REAL_COMPLETE shard=(\d+) captures=(\d+)', text)
    assert sorted(int(x[0]) for x in real) == [0, 1] and sum(int(x[1]) for x in real) == 30
    failures = re.findall(r'SERVING_MMA_COMPLETE shard=(\d+) comparisons=(\d+)', text)
    assert sorted(int(x[0]) for x in failures) == [0, 1] and sum(int(x[1]) for x in failures) == 300
    cases = records(text, 'GUARD_CASE')
    timings = records(text, 'GUARD_TIMING')
    assert len(cases) == 123 and len(timings) == 204
    expected = {(json.dumps(case['shape'], sort_keys=True), pair)
                for case in cases if 'capture' in case['shape'] or case['shape'].get('case', 0) == 0
                for pair in ((0, 2), (1, 2))}
    actual = [(json.dumps(item['shape'], sort_keys=True), tuple(item['arms'])) for item in timings]
    assert len(set(actual)) == len(actual) and set(actual) == expected
    assert all(item['exclusive'] and len(item['median_ms']) == 2
               and all(isinstance(ms, (float, int)) and ms > 0 for ms in item['median_ms']) for item in timings)
    assert all(case['guard_mismatches'] == 0 for case in cases)
    controls = [case for case in cases if case['negative_control_rejected']]
    assert len(controls) == 3 and all(case['fallback_slots'] > 0 and case['unguarded_score_mismatches'] > 0 for case in controls)
    assert sorted(case['shape']['matrix'] for case in cases if 'matrix' in case['shape']) == list(range(93))
    assert sorted(case['shape']['capture'] for case in cases if 'capture' in case['shape']) == list(range(30))
    frequency = {}
    for case in cases:
        if 'capture' not in case['shape']:
            continue
        context = str(case['shape']['context'])
        item = frequency.setdefault(context, dict(admitted=0, fallback_slots=0, fallback_rows=0, rows=0))
        for field in ('admitted', 'fallback_slots', 'fallback_rows'):
            item[field] += case[field]
        item['rows'] += case['shape']['rows']
    assert set(frequency) == {'8192', '32768', '131072'}
    for item in frequency.values():
        assert item['admitted'] > 0
        item['fallback_fraction'] = item['fallback_slots'] / item['admitted']
    return dict(matrix_shapes=93, captures=30, poisoned_comparisons=300, negative_controls=3,
                timing_pairs=timings, real_fallback_frequency=frequency, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--logs', type=Path, nargs=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage:
        validate(args.stage)
        print('GUARD_CPU_DRY_RUN_COMPLETE')
    if args.logs:
        report = results([path.read_text() for path in args.logs])
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('GUARD_PAIR_COMPLETE shapes=93 captures=30 poisoned=300')


if __name__ == '__main__':
    main()
