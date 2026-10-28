#!/usr/bin/env python3
"""Run the depth tuner's parsers on a recorded GPU run, on CPU, before a
job that depends on them is queued.

Usage: python3 -B STAGE/harness/recorded.py RECORDED_RUN OUTPUT_DIR

RECORDED_RUN is an earlier tuner run on real servers: one directory per
server with server/server.log, server/*.prom and windows/*.json holding
each window's engine steps. For every server the check parses the KV pages
line, memory after warmup (one line per rank), weight uploads, the
preemption and eviction counters and the speculation metrics; for every
window it rebuilds the full-concurrency steps and the step cells from the
recorded steps. It then writes those windows as a costs run under
OUTPUT_DIR and tabulates it with report.costs, and fails unless every
server and window parsed and the costs table holds every recorded
full-concurrency shape.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402
import runner as r  # noqa: E402
import tune  # noqa: E402


def main(recorded, output):
    recorded, output = Path(recorded).resolve(), Path(output).resolve()
    config = json.loads((recorded / 'config.json').read_text())
    source_path = recorded / 'summary.json'
    source_body = source_path.read_bytes() if source_path.exists() else b'{}'
    source_summary = json.loads(source_body)
    if source_summary.get('status') == 'void' or source_summary.get('source_validity', {}).get('status') == 'void':
        raise ValueError('void source cannot be adapted or pooled')
    provenance = {'status': source_summary.get('status', 'unknown'),
                  'summary_sha256': hashlib.sha256(source_body).hexdigest() if source_path.exists() else None}
    model = config['model']
    gpus = model['tensor_parallel_size']
    kv_line = re.compile(re.escape(model['kv_pages_line']).replace(re.escape('{pages}'), r'(\d+)'))
    output.mkdir(parents=True, exist_ok=False)
    processes, windows, shapes = {}, 0, set()
    for directory in sorted(x.parents[1] for x in recorded.glob('*/server/server.log')):
        label = directory.name
        log = (directory / 'server' / 'server.log').read_text(errors='replace')
        memory = tune.memory_lines(log)
        if sorted(m['rank'] for m in memory['ranks']) != list(range(gpus)):
            raise SystemExit(f'{label}: memory after warmup for ranks {memory["ranks"]}, expected {gpus}')
        pages = [int(x) for x in kv_line.findall(log)]
        if not pages:
            raise SystemExit(f'{label}: no KV pages line')
        weights = tune.weight_source(log)
        if weights['source'] not in ('loaded', 'borrowed'):
            raise SystemExit(f'{label}: weight uploads {weights}')
        prom = directory / 'server' / 'final.prom'
        counters = None
        if prom.exists():
            text = prom.read_text()
            try:
                counters = tune.parse_counters(text, label)
            except RuntimeError as error:
                raise SystemExit(str(error)) from error

        steps = r.trace_rows(directory / 'server' / 'server.log')
        if not steps:
            raise SystemExit(f'{label}: no engine step records')
        processes[label] = {'depth': None, 'args': [], 'memory_after_warmup': memory, 'kv_pages_per_rank': pages,
                            'weights': weights, 'metrics': counters, 'step_records': len(steps)}
        for path in sorted((directory / 'windows').glob('*.json')):
            record = json.loads(path.read_text())
            parts = record['steps'] if record['steps'] and isinstance(record['steps'][0], list) else [record['steps']]
            c = record['concurrency']
            full = tune.merge_full([tune.full_concurrency(part, c) for part in parts])
            if not full['steps']:
                raise SystemExit(f'{label} {path.name}: no full-concurrency steps')
            cells = tune.merge_cells([tune.cells(part) for part in parts])
            for key in full['histogram']:
                shapes.add(key if '-R' in key else key.rsplit('-v', 1)[0])
            processes[label]['depth'] = record['depth']
            value = {k: record[k] for k in ('name', 'kind', 'concurrency', 'depth', 'request_tokens_per_second')}
            value.update(repeat=windows, full=full, cells=cells, energy=record.get('energy'))
            r.write(output / label / 'windows' / f"{record['name']}-{path.stem}.json", value)
            windows += 1
    if not processes or not windows:
        raise SystemExit(f'{recorded}: no recorded servers or windows')
    r.write(output / 'config.json', config)
    r.write(output / 'summary.json', {**source_summary, 'status': provenance['status'],
                                     'source_validity': provenance, 'processes': processes})
    costs = report.costs(output)
    missing = sorted(shapes - set(costs['cells']))
    if missing:
        raise SystemExit(f'costs table lacks recorded shapes {missing}')
    result = {'passed': True, 'recorded': str(recorded), 'servers': len(processes), 'windows': windows,
              'source_validity': provenance,
              'full_concurrency_shapes': len(shapes), 'cells': len(costs['cells'])}
    r.write(output / 'recorded-check.json', result)
    print(f"PASS: {len(processes)} recorded servers and {windows} windows parsed, {len(costs['cells'])} cost cells: "
          f'{output}')


def closed_loop(recorded, output, require_agreement=False):
    """Replay historical closed-loop receipts through the paired verdict core.

    Historical admission gaps stay explicit. These receipts cannot certify a
    new contract A job, even when their numerical verdict agrees.
    """
    import csv
    import hashlib
    import math
    import faults
    root, output = Path(recorded).resolve(), Path(output).resolve()
    evidence = {}

    def read(path):
        path = Path(path).resolve()
        if root not in path.parents:
            raise ValueError('recorded input escapes its run directory')
        body = path.read_bytes()
        evidence[str(path.relative_to(root))] = hashlib.sha256(body).hexdigest()
        return json.loads(body)

    summary, original = read(root / 'summary.json'), read(root / 'comparison.json')
    if summary['status'] != 'passed' or original['passed'] is not True or original.get('mechanics_only'):
        raise ValueError('historical run did not pass its harness')
    arms = summary['abba']['AR-vs-DSpark']
    labels = ('A1', 'B1', 'B2', 'A2')
    counts = set(original['rows'])
    if not counts or any(set(arms[label]['decode']) != counts for label in labels):
        raise ValueError('wrong historical visit or cell counts')
    power = summary['telemetry']['before']['power']['enforced_power_limits_w']
    if power != summary['telemetry']['after']['power']['enforced_power_limits_w']:
        raise ValueError('void historical job: changed power limits')
    samples = root / 'gpu-samples.csv'
    evidence[samples.name] = hashlib.sha256(samples.read_bytes()).hexdigest()
    seen = set()
    with samples.open() as stream:
        for row in csv.reader(stream):
            index = int(row[1])
            if float(row[5]) != power[index]:
                raise ValueError('void historical job: changed sampled power limits')
            if float(row[4]) <= 0:
                raise ValueError('missing historical clocks')
            seen.add(index)
    if seen != set(range(len(power))):
        raise ValueError('incomplete historical GPU samples')
    result, gaps, disagreements = {}, set(), []
    servers = [arms[label]['server'] for label in labels]
    if len(set(servers)) != 4:
        gaps.add('Historical paired DSpark visits reused a process.')
    mapping = {'decisive gain': 'resolved gain', 'decisive regression': 'resolved regression',
               'no change (within the A/A or B/B spread)': 'neutral',
               'no change (both head arms inside the base range)': 'neutral', 'read by hand': None}
    for label in labels:
        if summary['servers'][arms[label]['server']]['weights']['source'] != 'borrowed':
            raise ValueError('historical visit did not borrow keeper weights')
        faults.validate(faults.combine([w['faults'] for w in arms[label]['decode'].values()]))
    for c in sorted(counts, key=int):
        rates, identities, durations, energies = [], [], [], []
        for label in labels:
            window = arms[label]['decode'][c]
            start, end = window['window_monotonic_seconds']
            if end <= start or not math.isclose(end - start, window['window_seconds'], abs_tol=1e-6):
                raise ValueError('inconsistent historical window boundaries')
            durations.append(end - start)
            if not 14.9 <= end - start <= 15.2:
                gaps.add('Historical windows include durations other than 15 seconds.')
            faults.validate(window['faults'])
            actual_faults = faults.difference(window['fault_counters_before'], window['fault_counters_after'])
            if actual_faults != window['faults']:
                raise ValueError('historical fault counters disagree')
            requests = read(window['requests_path'])
            if len(requests) != window['requests']['total'] or len(requests) != window['requests']['completed']:
                raise ValueError('incomplete historical requests')
            identities_for_visit = {}
            tokens = 0
            for request in requests:
                usage, body = request['usage'], request['request']
                if request['error'] or request['finish_reason'] != 'length':
                    raise ValueError('failed historical request')
                if usage['prompt_tokens_details']['cached_tokens'] != 0:
                    raise ValueError('cached historical prompt tokens')
                if body['temperature'] != 1.0 or usage['completion_tokens'] != window['output_tokens']:
                    raise ValueError('wrong historical sampling or output count')
                key = (request['slot'], request['sequence'])
                if key in identities_for_visit:
                    raise ValueError('duplicate historical slot sequence')
                identities_for_visit[key] = (body['seed'], body['prompt'])
                if sum(n for _, n in request['events']) != window['output_tokens']:
                    raise ValueError('incomplete historical token events')
                tokens += sum(n for at, n in request['events'] if start <= at < end)
            if tokens != window['tokens']:
                raise ValueError('historical timed token count differs')
            rate = tokens / (end - start)
            if not math.isclose(rate, original['rows'][c]['arms'][label]['window_tok_s'], rel_tol=1e-10):
                raise ValueError('historical reported rate differs from events')
            rates.append(rate)
            identities.append(identities_for_visit)
            energies.append(window['energy']['j_per_token'])
        for a, b in ((0, 1), (2, 3)):
            common = identities[a].keys() & identities[b].keys()
            if len(common) < int(c) or any(identities[a][k] != identities[b][k] for k in common):
                raise ValueError('mismatched historical paired seeds or prompts')
        if {v[0] for v in identities[0].values()} & {v[0] for v in identities[2].values()}:
            raise ValueError('historical pairs share a seed set')
        decision = report.paired(rates)
        historical_label = original['rows'][c]['window_tok_s']['verdict']
        expected = mapping.get(historical_label)
        if decision['verdict'] != expected:
            disagreements.append({'concurrency': c, 'paired': decision['verdict'], 'harness': expected})
        result[c] = {**decision, 'rates': rates, 'window_seconds': durations,
                     'joules_per_token': energies, 'harness_verdict': expected, 'harness_label': historical_label}
    output.mkdir(parents=True, exist_ok=False)
    value = {'status': 'historical replay', 'passed': True, 'contract_a_admissible': False,
             'limitations': sorted(gaps | {'Historical records do not contain per-seed exact output proofs.'}),
             'pairs': {'AR/DSpark': result}, 'enforced_power_limits_w': power, 'evidence': evidence,
             'harness_agreement': not disagreements, 'disagreements': disagreements,
             'windows': len(counts) * 4, 'excluded_probes': sorted(set(arms) - set(labels))}
    r.write(output / 'ab.json', value)
    r.write(output / 'recorded-check.json', value)
    if require_agreement and disagreements:
        raise ValueError('paired verdict differs from recorded harness; see recorded-check.json')
    print(json.dumps({'passed': True, 'windows': value['windows'],
                      'verdicts': {c: v['verdict'] for c, v in result.items()}}))
    return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-loop', action='store_true')
    parser.add_argument('--require-agreement', action='store_true')
    parser.add_argument('recorded')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.closed_loop:
        closed_loop(args.recorded, args.output, args.require_agreement)
    else:
        main(args.recorded, args.output)
