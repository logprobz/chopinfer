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
            found = [tune.PREEMPTIONS.findall(text), tune.EVICTIONS.findall(text)]
            if any(len(x) != 1 for x in found):
                raise SystemExit(f'{label}: metrics lack one preemption and one eviction counter')
            counters = {'preemptions': float(found[0][0]), 'prefix_cache_evictions': float(found[1][0])}
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
                shapes.add(key.rsplit('-v', 1)[0])
            processes[label]['depth'] = record['depth']
            value = {k: record[k] for k in ('name', 'kind', 'concurrency', 'depth', 'request_tokens_per_second')}
            value.update(repeat=windows, full=full, cells=cells, energy=record.get('energy'))
            r.write(output / label / 'windows' / f"{record['name']}-{path.stem}.json", value)
            windows += 1
    if not processes or not windows:
        raise SystemExit(f'{recorded}: no recorded servers or windows')
    r.write(output / 'config.json', config)
    r.write(output / 'summary.json', {'processes': processes})
    costs = report.costs(output)
    missing = sorted(shapes - set(costs['cells']))
    if missing:
        raise SystemExit(f'costs table lacks recorded shapes {missing}')
    result = {'passed': True, 'recorded': str(recorded), 'servers': len(processes), 'windows': windows,
              'full_concurrency_shapes': len(shapes), 'cells': len(costs['cells'])}
    r.write(output / 'recorded-check.json', result)
    print(f"PASS: {len(processes)} recorded servers and {windows} windows parsed, {len(costs['cells'])} cost cells: "
          f'{output}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
