#!/usr/bin/env python3
"""Select a per-concurrency draft depth schedule from a depth tuner run, or
tabulate a costs run's step times per verify width.

Usage: python3 -I report.py RUN_DIR [RUN_DIR ...]
       python3 -I report.py --costs RUN_DIR

Several runs (for example a low- and a high-concurrency job) merge into one
schedule, each concurrency taken from the run that measured it. Each run
reads RUN_DIR/config.json and every timed window record (the reference pass
never counts) and writes RUN_DIR/tune.json and RUN_DIR/tune.md.

Per concurrency C, kind (greedy, sampled) and depth d, over every measurement
(one per timed repeat, or per process in runs with a server per repeat):
median and spread (max/min - 1) of request tok/s and full-concurrency tok/s,
median J/token, median stream p10, accepted over drafted by draft position
(summed counts; not conditional on the earlier positions), drafts per verify
item and whether every measurement served the same full-concurrency step
shapes.

Selection per C: a depth's score is the weighted geometric mean, over the
kinds in config["kinds"], of its median request tok/s over depth 0's. A depth
is stable when every kind served identical shapes and its request spread is at
most config["selection"]["spread"]. Among stable depths, the shallowest whose
score is within the margin of the best wins; the margin is the larger of
config["selection"]["margin"] and the best depth's request spread. A depth
whose full-concurrency shapes equal a shallower one's is noted; its tail
steps can still differ, so the margin rule alone decides. The schedule gives each measured C its depth, from that C up to the next
measured one, as a `depths` string (`items:depth,...`).
"""
import json
import math
import statistics
import sys
from pathlib import Path


def load(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    rows = []
    for directory in sorted(run.glob('*/windows')):
        for path in sorted(directory.glob('*.json')):
            record = json.loads(path.read_text())
            record.pop('exact', None)
            record.pop('steps', None)
            rows.append({**record, 'label': directory.parent.name})
    return config, rows


def spread(values):
    values = [v for v in values if v]
    return max(values) / min(values) - 1 if len(values) > 1 else 0.0


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def measure(rows):
    request = [r['request_tokens_per_second'] for r in rows]
    full = [r['full']['tokens_per_second'] for r in rows]
    positions, drafted, verified = {}, 0, 0
    for r in rows:
        a = r.get('acceptance') or {}
        drafted += a.get('draft_tokens') or 0
        verified += a.get('verify_items') or 0
        for p in a.get('per_position') or []:
            cell = positions.setdefault(p['position'], [0, 0])
            cell[0] += p['drafted']
            cell[1] += p['accepted']
    histograms = {json.dumps(r['full']['histogram'], sort_keys=True) for r in rows}
    return {'measurements': len(rows), 'request': request, 'request_median': median(request),
            'request_spread': spread(request), 'full_median': median(full), 'full_spread': spread(full),
            'joules_median': median([(r.get('energy') or {}).get('joules_per_token') for r in rows]),
            'stream_p10_median': median([(r.get('streams') or {}).get('p10') for r in rows]),
            'acceptance': {str(p): {'drafted': d, 'accepted': a, 'rate': a / d if d else None}
                           for p, (d, a) in sorted(positions.items())},
            'drafts_per_item': drafted / verified if verified else 0.0,
            'identical_shapes': len(histograms) == 1, 'histogram': rows[0]['full']['histogram']}


def select(config, cells):
    """The chosen depth and its evidence per concurrency."""
    kinds, rule = config['kinds'], config['selection']
    out = {}
    for c in config['model']['concurrencies']:
        depths = sorted({d for (cc, _, d) in cells if cc == c})
        if 0 not in depths:
            continue
        rows = {}
        for d in depths:
            present = [k for k in kinds if (c, k, d) in cells and (c, k, 0) in cells]
            if not present:
                continue
            weight = sum(kinds[k] for k in present)
            score = math.exp(sum(kinds[k] * math.log(cells[(c, k, d)]['request_median'] /
                                                   cells[(c, k, 0)]['request_median']) for k in present) / weight)
            same = next((e for e in depths if e < d and all(
                (c, k, e) in cells and cells[(c, k, e)]['histogram'] == cells[(c, k, d)]['histogram']
                and cells[(c, k, d)]['identical_shapes'] for k in present)), None)
            rows[d] = {'score': score, 'equivalent_to': same,
                       'stable': all(cells[(c, k, d)]['identical_shapes']
                                     and cells[(c, k, d)]['request_spread'] <= rule['spread'] for k in present),
                       'request_spread': max(cells[(c, k, d)]['request_spread'] for k in present)}
        stable = [d for d, v in rows.items() if v['stable']]
        if not stable:
            out[str(c)] = {'depth': 0, 'reason': 'no stable depth', 'depths': rows}
            continue
        best = max(stable, key=lambda d: rows[d]['score'])
        margin = max(rule['margin'], rows[best]['request_spread'])
        chosen = min(d for d in stable if rows[d]['score'] >= rows[best]['score'] * (1 - margin))
        out[str(c)] = {'depth': chosen, 'best': best, 'margin': margin, 'depths': rows,
                       'reason': 'best' if chosen == best else f'within {margin:.1%} of depth {best}'}
    return out


def schedule(selection):
    entries, last = [], None
    for c in sorted(selection, key=int):
        depth = selection[c]['depth']
        if depth != last:
            entries.append(f'{c}:{depth}')
            last = depth
    return ','.join(entries)


def evaluate(run):
    run = Path(run)
    config, rows = load(run)
    groups = {}
    for r in rows:
        groups.setdefault((r['concurrency'], r['kind'], r['depth']), []).append(r)
    cells = {key: measure(value) for key, value in groups.items()}
    selection = select(config, cells)
    out = {'depths': schedule(selection), 'selection': selection,
           'cells': {f'c{c}/{k}/d{d}': v for (c, k, d), v in sorted(cells.items())}}
    (run / 'tune.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    (run / 'tune.md').write_text(markdown(config, cells, selection, out['depths']))
    return out


def markdown(config, cells, selection, depths):
    def pair(c, d, field, form):
        values = [cells.get((c, k, d), {}).get(field) for k in ('greedy', 'sampled')]
        return ' / '.join(form.format(v) if v is not None else '-' for v in values)

    lines = [f"# {config['name']}: draft depth per concurrency", '',
             f"Schedule: `{depths}` (profile `depths`, or `--speculation-depths`).", '',
             'Per C and depth, greedy / sampled: median request tok/s (spread), full-concurrency tok/s, J/token, '
             'stream p10 tok/s, drafts per verify item, greedy accepted/drafted by position (not conditional), '
             'identical shapes, score (weighted request tok/s over depth 0).', '']
    for c in config['model']['concurrencies']:
        chosen = selection.get(str(c))
        if not chosen:
            continue
        lines += [f"## C{c}: depth {chosen['depth']} ({chosen['reason']})", '',
                  '| depth | request | spread | full | J/token | stream p10 | drafts/item | greedy accepted/drafted '
                  '| shapes | score |', '|---:|---|---|---|---|---|---|---|---|---:|']
        for d, row in sorted(chosen['depths'].items()):
            greedy = cells.get((c, 'greedy', d), {})
            accept = ' '.join(f"{v['rate']:.2f}" for v in greedy.get('acceptance', {}).values() if v['rate'] is not None)
            shapes = ' / '.join('yes' if cells.get((c, k, d), {}).get('identical_shapes') else 'no'
                                for k in ('greedy', 'sampled'))
            note = f" (= d{row['equivalent_to']})" if row['equivalent_to'] is not None else ''
            lines.append(f"| {d}{note} | {pair(c, d, 'request_median', '{:.1f}')} | "
                         f"{pair(c, d, 'request_spread', '{:.1%}')} | {pair(c, d, 'full_median', '{:.1f}')} | "
                         f"{pair(c, d, 'joules_median', '{:.3f}')} | {pair(c, d, 'stream_p10_median', '{:.1f}')} | "
                         f"{pair(c, d, 'drafts_per_item', '{:.2f}')} | {accept or '-'} | {shapes} | "
                         f"{row['score']:.3f} |")
        lines.append('')
    return '\n'.join(lines)


def merged(runs):
    """One schedule from several runs' selections."""
    selection = {}
    for run in runs:
        selection.update(evaluate(run)['selection'])
    return schedule(selection)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))] if values else None


def costs(run):
    """A costs run's step times per verify width: for every `i{items}-r{rows}`
    cell the median, p10 and p90 cycle time (from the previous step's
    completion) and median execution time over every timed window's steps,
    with the step count and the arms that served it; and per window
    (C, kind, depth) the median request and full-concurrency tok/s and
    J/token over repeats. Writes RUN_DIR/costs.json and RUN_DIR/costs.md."""
    run = Path(run)
    config, rows = load(run)
    table = {}
    for row in rows:
        for key, cell in (row.get('cells') or {}).items():
            into = table.setdefault(key, {'cycle_ms': [], 'execution_ms': [], 'arms': set()})
            into['cycle_ms'] += cell['cycle_ms']
            into['execution_ms'] += cell['execution_ms']
            into['arms'].add(row['label'])
    cells = {}
    for key, cell in table.items():
        items, rows_ = (int(x[1:]) for x in key.split('-'))
        cells[key] = {'items': items, 'rows': rows_, 'steps': len(cell['cycle_ms']),
                      'cycle_ms_median': median(cell['cycle_ms']), 'cycle_ms_p10': percentile(cell['cycle_ms'], .1),
                      'cycle_ms_p90': percentile(cell['cycle_ms'], .9),
                      'execution_ms_median': median(cell['execution_ms']), 'arms': sorted(cell['arms'])}
    groups = {}
    for row in rows:
        groups.setdefault((row['concurrency'], row['kind'], row['depth']), []).append(row)
    windows = {f'c{c}/{k}/d{d}': {
        'repeats': len(v), 'request_median': median([x['request_tokens_per_second'] for x in v]),
        'request_spread': spread([x['request_tokens_per_second'] for x in v]),
        'full_median': median([x['full']['tokens_per_second'] for x in v]),
        'joules_median': median([(x.get('energy') or {}).get('joules_per_token') for x in v])}
        for (c, k, d), v in sorted(groups.items())}
    memory = {}
    summary = run / 'summary.json'
    if summary.exists():
        for label, process in json.loads(summary.read_text()).get('processes', {}).items():
            memory[label] = process.get('gpu_memory_mib')
    out = {'cells': dict(sorted(cells.items(), key=lambda kv: (kv[1]['items'], kv[1]['rows']))),
           'windows': windows, 'gpu_memory_mib': memory}
    full = {key: cell for key, cell in out['cells'].items()
            if any(key == f"i{c}-r{c * (d + 1)}" for c in config['model']['concurrencies']
                   for d in range(config['model'].get('max_depth', 5) + 1))}
    out['summary'] = {key: round(cell['cycle_ms_median'], 2) for key, cell in full.items()
                      if cell['cycle_ms_median'] is not None}
    (run / 'costs.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    lines = [f"# {config['name']}: step cost per verify width", '',
             'Per (items, rows): steps, median / p10 / p90 cycle ms (completion to completion), median execution '
             'ms, arms.', '', '| items | rows | steps | cycle median | p10 | p90 | execution | arms |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    for cell in out['cells'].values():
        form = lambda v: '-' if v is None else f'{v:.2f}'
        lines.append(f"| {cell['items']} | {cell['rows']} | {cell['steps']} | {form(cell['cycle_ms_median'])} | "
                     f"{form(cell['cycle_ms_p10'])} | {form(cell['cycle_ms_p90'])} | "
                     f"{form(cell['execution_ms_median'])} | {' '.join(cell['arms'])} |")
    lines += ['', '| window | repeats | request tok/s | spread | full tok/s | J/token |', '|---|---:|---:|---:|---:|---:|']
    for key, value in windows.items():
        lines.append(f"| {key} | {value['repeats']} | {value['request_median']:.1f} | {value['request_spread']:.1%} | "
                     f"{value['full_median']:.1f} | "
                     + ('-' if value['joules_median'] is None else f"{value['joules_median']:.3f}") + ' |')
    lines += ['', 'GPU MiB used per arm (at readiness, after each warm window, at the end):', '']
    lines += [f'- {label}: {value}' for label, value in memory.items()]
    (run / 'costs.md').write_text('\n'.join(lines) + '\n')
    return out


if __name__ == '__main__':
    if sys.argv[1:2] == ['--costs']:
        print(json.dumps({'costs': costs(sys.argv[2])['summary']}))
    else:
        print(json.dumps({'depths': merged(sys.argv[1:])}))
