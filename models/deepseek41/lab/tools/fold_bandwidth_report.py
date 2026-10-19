"""Audit ordered-fold candidates against serving, with both-device timing gates."""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from fold_stream_report import exclusive, graphs

MODES = ('cold', 'prefetched')
WIN = 0.97
LIMIT = 1.03


def marked(text, marker):
    result = []
    for line in text.splitlines():
        if marker + ' ' in line:
            result.append(json.JSONDecoder().raw_decode(line.split(marker + ' ', 1)[1])[0])
    return result


def check_log(text, passed=1):
    summaries = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    assert summaries == [(str(passed), '0', '0')], summaries
    assert not re.search(r'skipp', text, re.I) and 'FAILED' not in text


def key(row):
    return tuple(row[name] for name in ('n', 'k', 'groups', 'paired', 'rows', 'variant'))


def inventory(root):
    root = Path(root)
    cpu = (root / 'cpu.log').read_text()
    check_log(cpu)
    data, = marked(cpu, 'fold_bandwidth_cpu')
    assert data['required_ordinals'] == 1 and not data['serving_enabled']
    assert data['shapes'] == 23 and data['rows'] == [1, 2, 4, 6, 8, 16, 32, 64]
    assert data['topologies'] == [1, 2, 4]
    assert len(data['matrix']) == len({key(r) for r in data['matrix']}) == data['cases']
    modules = (root / 'modules.log').read_text()
    check_log(modules)
    required, = marked(modules, 'fold_bandwidth_modules')
    additional = marked(modules, 'fold_bandwidth_module')
    assert len(additional) == 2
    data['modules'] = required['modules'] + additional
    names = {m['name'] for m in data['modules']}
    assert 'probe_sm120' in names and 'mxfp8_fold_direct_sm120' in names
    assert len(names) == len(data['modules'])
    for device in (0, 1):
        worker = (root / f'worker{device}.log').read_text()
        check_log(worker)
        record, = marked(worker, 'fold_bandwidth_worker')
        assert record['selected_ordinals'] == [device] and record['required_ordinals'] == 1
        assert record['test'] == 'mxfp8_fold_bandwidth::direct_fold_gpu'
        assert record['gpu_launched'] is False
    (root / 'inventory.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


def paired_rounds(samples):
    assert len(samples) == 10
    measurements = {s['measurement'] for s in samples}
    assert len(measurements) == 1
    arms = [{}, {}]
    for sample in samples:
        arm, round_id = sample['arm'], sample['round']
        assert arm in (0, 1) and round_id in range(5)
        assert sample['order'] == ([0, 1] if round_id % 2 == 0 else [1, 0])
        value = sample['elapsed_ms'] * 1000
        assert math.isfinite(value) and value > 0
        assert round_id not in arms[arm]
        arms[arm][round_id] = value
    assert all(set(arm) == set(range(5)) for arm in arms)
    return [statistics.median(arm.values()) for arm in arms], statistics.median(
        arms[1][i] / arms[0][i] for i in range(5))


def add_events(rows, receipt):
    telemetry = defaultdict(list)
    for sample in receipt['timing_rounds']:
        if 'elapsed_ms' in sample:
            telemetry[sample['operation']].append(sample)
    expected = {r[mode]['label'] for r in rows for mode in MODES}
    assert set(telemetry) == expected
    for row in rows:
        for mode in MODES:
            times, ratio = paired_rounds(telemetry[row[mode]['label']])
            data = row[mode]
            assert math.isclose(ratio, data['paired_ratio'], rel_tol=2e-6)
            for name, value in zip(('baseline_us', 'candidate_us'), times):
                assert math.isclose(value, data[name], rel_tol=2e-6)
            data['paired_ratio'] = ratio


def add_exclusive(rows, traces):
    candidates = [g for g in traces if len(g['replays']) == 31
                  and len(g['symbols']) == 1 and '_mxfp8_direct_' in g['symbols'][0]]
    baselines = [g for g in traces if len(g['replays']) == 14
                 and all('_mxfp8_direct_' not in s for s in g['symbols'])]
    assert len(candidates) == len(baselines) == len(rows), (len(candidates), len(baselines), len(rows))
    for row, old, new in zip(rows, baselines, candidates):
        assert old['symbols'] == sorted(row['baseline_symbols'])
        assert new['symbols'] == sorted(row['candidate_symbols'])
        assert row['trace_replays'] == dict(baseline=14, candidate=31, boundary=8, admission=9,
                                           warmup_per_mode=2, rounds_per_mode=5)
        arms = [old['replays'], new['replays'][17:]]
        for offset, mode in zip((0, 7), MODES):
            stages = [[exclusive(replay) for replay in arm[offset+2:offset+7]] for arm in arms]
            samples = [[sum(s.values()) / 1000 for s in arm] for arm in stages]
            assert all(math.isfinite(x) and x > 0 for arm in samples for x in arm)
            medians = list(map(statistics.median, samples))
            ratio = statistics.median(b/a for a, b in zip(*samples))
            row[mode].update(exclusive_baseline_us=medians[0], exclusive_candidate_us=medians[1],
                             exclusive_paired_ratio=ratio, exclusive_floor_ratio=medians[1]/row['floor_us'],
                             exclusive_effective_tbps=row['weight_bytes']/(medians[1]*1e6),
                             exclusive_stages=[{name: statistics.median(s.get(name, 0) for s in arm)/1000
                                                for name in {n for s in arm for n in s}} for arm in stages])
    return rows


def select(rows):
    candidates = defaultdict(list)
    for row in rows:
        assert row['bitwise'] and row['clock_bitwise']
        candidates[key(row)].append(row)
    buckets = defaultdict(list)
    for candidate, pair in candidates.items():
        assert len(pair) == 2 and {r['device'] for r in pair} == {0, 1}, candidate
        ratios = [r[mode][field] for r in pair for mode in MODES
                  for field in ('paired_ratio', 'exclusive_paired_ratio')]
        assert all(math.isfinite(v) and v > 0 for v in ratios)
        # Cold and prefetched results must both improve on both devices.
        if max(ratios) < WIN:
            buckets[candidate[:-1]].append((max(ratios), candidate[-1]))
    selected = {k: min(values)[1] for k, values in buckets.items()}
    for row in rows:
        row['selected_variant'] = selected.get(key(row)[:-1], 'baseline')
        row['selected'] = row['variant'] == row['selected_variant']
        for mode in MODES:
            row[mode]['selected_ratio'] = row[mode]['paired_ratio'] if row['selected'] else 1.0
            row[mode]['selected_exclusive_ratio'] = row[mode]['exclusive_paired_ratio'] if row['selected'] else 1.0
            assert max(row[mode]['selected_ratio'], row[mode]['selected_exclusive_ratio']) <= LIMIT
    return selected


def summarize(root):
    root = Path(root)
    inv = json.loads((root/'inventory.json').read_text())
    expected = {key(r): r for r in inv['matrix']}
    all_rows = []
    for device in (0, 1):
        text = (root/f'gpu{device}.log').read_text()
        check_log(text)
        rows = marked(text, 'fold_bandwidth_result')
        assert len(rows) == len(expected) and {key(r) for r in rows} == set(expected)
        for row in rows:
            reference = expected[key(row)]
            assert row['device'] == device and row['bitwise'] and row['clock_bitwise']
            assert row['ranks'] == reference['ranks'] and row['baseline_stream'] == reference['baseline_stream']
            assert row['baseline_symbols'] == reference['baseline_symbols']
            assert row['boundary_replays'] == 8 and row['admission_cases'] == 9
            assert row['admission_graph_replays'] == 9
            assert row['global_partial_bytes'] == 0 and len(row['candidate_symbols']) == 1
            assert row['concurrent_replays'] == (3 if row['rows'] in (1, 8, 64) else 0)
            assert row['pdl_producer'] == (row['rows'] in (1, 8, 64))
            compulsory = row['n'] * row['k'] * row['groups'] * (2 if row['paired'] else 1) * 33 // 32
            assert row['weight_bytes'] == compulsory
            assert math.isclose(row['floor_us'], compulsory/1_800_000)
        probes = marked(text, 'fold_bandwidth_deferred')
        assert len(probes) == 2 and {r['rows'] for r in probes} == {1, 32}
        assert all(r['bitwise'] and r['clock_bitwise'] for r in probes)
        prefill=marked(text,'fold_bandwidth_prefill')
        assert len(prefill)==20 and all(r['bitwise'] and r['rows']==128 for r in prefill)
        assert {(r['n'],r['k'],r['groups'],r['paired']) for r in prefill} == {
            (r['n'],r['k'],r['groups'],r['paired']) for r in rows if not r['paired']}
        control, = marked(text, 'fold_bandwidth_control')
        assert control == {'omitted_chain_caught': True}
        receipts = list((root/f'receipts{device}'/'mxfp8-fold-bandwidth').glob('*.json'))
        receipts = [json.loads(p.read_text()) for p in receipts]
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt['measurement_label'] != 'shared-device' and receipt['cases'] == len(expected)
        add_events(rows, receipt)
        all_rows.extend(add_exclusive(rows, graphs(root/f'fold{device}.sqlite')))
    selected = select(all_rows)
    (root/'matrix.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    columns = ['device', 'n', 'k', 'groups', 'paired', 'rows', 'ranks', 'variant', 'cache',
               'baseline_us', 'candidate_us', 'paired_ratio', 'exclusive_baseline_us',
               'exclusive_candidate_us', 'exclusive_paired_ratio', 'floor_us', 'exclusive_floor_ratio',
               'exclusive_effective_tbps', 'selected_variant', 'selected']
    with (root/'timings.csv').open('w') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            for mode in MODES:
                writer.writerow(dict(row, **row[mode], cache=mode))
    verdict = dict(passed=True, cases=len(all_rows), bitwise=True, qualified_buckets=len(selected),
                   serving_enabled=False, routed_regressions=0, regression_limit=LIMIT,
                   win_limit=WIN, look_ahead_qualified=False)
    (root/'routing-verdict.json').write_text(json.dumps(verdict, indent=2)+'\n')
    print('Pair direct-fold matrices complete; both-device wins recorded; serving remains disabled')
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inventory', 'summarize'))
    parser.add_argument('root')
    options = parser.parse_args()
    print(json.dumps(globals()[options.mode](options.root)))
