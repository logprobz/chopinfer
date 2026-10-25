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
import re
import statistics
import sys
from pathlib import Path


def load(run, allow_void=False):
    run = Path(run)
    summary = run / 'summary.json'
    if not allow_void and summary.exists() and json.loads(summary.read_text()).get('status') == 'void':
        raise ValueError('void jobs cannot be pooled')
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
    fallback = [r.get('fallback_launches') for r in rows]
    fallback = sum(fallback) if fallback and all(type(n) is int and n >= 0 for n in fallback) else None
    return {'fallback_launches': fallback, 'measurements': len(rows), 'request': request, 'request_median': median(request),
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
    kinds = [k for k in ('sampled', 'greedy') if k in config['kinds']]

    def pair(c, d, field, form):
        values = [cells.get((c, k, d), {}).get(field) for k in kinds]
        return ' / '.join(form.format(v) if v is not None else '-' for v in values)

    lines = [f"# {config['name']}: draft depth per concurrency", '',
             f"Schedule: `{depths}` (profile `depths`, or `--speculation-depths`).", '',
             f"Per C and depth, {' / '.join(kinds)}: median request tok/s (spread), full-concurrency tok/s, J/token, "
             f"stream p10 tok/s, drafts per verify item, {kinds[0]} accepted/drafted by position (not conditional), "
             'identical shapes, score (weighted request tok/s over depth 0).', '']
    for c in config['model']['concurrencies']:
        chosen = selection.get(str(c))
        if not chosen:
            continue
        lines += [f"## C{c}: depth {chosen['depth']} ({chosen['reason']})", '',
                  f"| depth | request | spread | full | J/token | stream p10 | drafts/item | {kinds[0]} accepted/drafted "
                  '| shapes | fallback launches | score |', '|---:|---|---|---|---|---|---|---|---|---:|---:|']
        for d, row in sorted(chosen['depths'].items()):
            first = cells.get((c, kinds[0], d), {})
            accept = ' '.join(f"{v['rate']:.2f}" for v in first.get('acceptance', {}).values() if v['rate'] is not None)
            shapes = ' / '.join('yes' if cells.get((c, k, d), {}).get('identical_shapes') else 'no' for k in kinds)
            note = f" (= d{row['equivalent_to']})" if row['equivalent_to'] is not None else ''
            lines.append(f"| {d}{note} | {pair(c, d, 'request_median', '{:.1f}')} | "
                         f"{pair(c, d, 'request_spread', '{:.1%}')} | {pair(c, d, 'full_median', '{:.1f}')} | "
                         f"{pair(c, d, 'joules_median', '{:.3f}')} | {pair(c, d, 'stream_p10_median', '{:.1f}')} | "
                         f"{pair(c, d, 'drafts_per_item', '{:.2f}')} | {accept or '-'} | {shapes} | "
                         f"{pair(c, d, 'fallback_launches', '{:.0f}')} | {row['score']:.3f} |")
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
    J/token over repeats; per arm its served rows, graph rows and widths,
    each rank's memory after warmup and the GPU memory samples. Writes
    RUN_DIR/costs.json and RUN_DIR/costs.md."""
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
        shape = re.fullmatch(r'i(\d+)-r(\d+)(?:-v(\d+)-R(\d+))?', key)
        if shape is None:
            raise ValueError(f'invalid step shape {key}')
        items, rows_ = map(int, shape.group(1, 2))
        cells[key] = {'items': items, 'rows': rows_, 'steps': len(cell['cycle_ms']),
                      'cycle_ms_median': median(cell['cycle_ms']), 'cycle_ms_p10': percentile(cell['cycle_ms'], .1),
                      'cycle_ms_p90': percentile(cell['cycle_ms'], .9),
                      'execution_ms_median': median(cell['execution_ms']), 'arms': sorted(cell['arms'])}
        if shape[3] is not None:
            cells[key].update(verify_items=int(shape[3]), reserved=int(shape[4]))
    groups = {}
    for row in rows:
        groups.setdefault((row['concurrency'], row['kind'], row['depth']), []).append(row)
    windows = {f'c{c}/{k}/d{d}': {
        'repeats': len(v), 'request_median': median([x['request_tokens_per_second'] for x in v]),
        'request_spread': spread([x['request_tokens_per_second'] for x in v]),
        'full_median': median([x['full']['tokens_per_second'] for x in v]),
        'joules_median': median([(x.get('energy') or {}).get('joules_per_token') for x in v])}
        for (c, k, d), v in sorted(groups.items())}
    memory, warmed = {}, {}
    summary = run / 'summary.json'
    if summary.exists():
        for label, process in json.loads(summary.read_text()).get('processes', {}).items():
            memory[label] = process.get('gpu_memory_mib')
            args = process.get('args') or []
            option = lambda flag: args[args.index(flag) + 1] if flag in args else None
            warmed[label] = {'speculative_rows': option('--speculative-rows'),
                             'graph_rows': option('--speculation-graph-rows'),
                             'widths': option('--speculation-widths'),
                             'ranks': (process.get('memory_after_warmup') or {}).get('ranks', [])}
    out = {'cells': dict(sorted(cells.items(), key=lambda kv: (kv[1]['items'], kv[1]['rows']))),
           'windows': windows, 'gpu_memory_mib': memory, 'memory_after_warmup': warmed}
    full = {key: cell for key, cell in out['cells'].items()
            if ('reserved' in cell and cell['items'] in config['model']['concurrencies'])
            or any(key == f"i{c}-r{c * (d + 1)}" for c in config['model']['concurrencies']
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
        shape = str(cell['rows'])
        if 'reserved' in cell:
            shape += f" (v{cell['verify_items']}, R{cell['reserved']})"
        lines.append(f"| {cell['items']} | {shape} | {cell['steps']} | {form(cell['cycle_ms_median'])} | "
                     f"{form(cell['cycle_ms_p10'])} | {form(cell['cycle_ms_p90'])} | "
                     f"{form(cell['execution_ms_median'])} | {' '.join(cell['arms'])} |")
    lines += ['', '| window | repeats | request tok/s | spread | full tok/s | J/token |', '|---|---:|---:|---:|---:|---:|']
    for key, value in windows.items():
        lines.append(f"| {key} | {value['repeats']} | {value['request_median']:.1f} | {value['request_spread']:.1%} | "
                     f"{value['full_median']:.1f} | "
                     + ('-' if value['joules_median'] is None else f"{value['joules_median']:.3f}") + ' |')
    gpus = config['model'].get('tensor_parallel_size', 1)
    lines += ['', 'Memory after warmup per arm and rank (rows and graph rows as served; `-` is the profile '
              'default): MiB taken since KV sizing (graphs) / MiB free.', '',
              '| arm | rows | graph rows | widths | ' + ' | '.join(f'rank {i}' for i in range(gpus)) + ' |',
              '|---|---|---|---|' + '---|' * gpus]
    for label, value in warmed.items():
        ranks = {m['rank']: f"{m['taken_mib']} / {m['free_mib']}" for m in value['ranks']}
        lines.append(f"| {label} | {value['speculative_rows'] or '-'} | {value['graph_rows'] or '-'} | "
                     f"{value['widths'] or '-'} | " + ' | '.join(ranks.get(i, '-') for i in range(gpus)) + ' |')
    lines += ['', 'GPU MiB used per arm (at readiness, after each warm window, at the end):', '']
    lines += [f'- {label}: {value}' for label, value in memory.items()]
    (run / 'costs.md').write_text('\n'.join(lines) + '\n')
    return out


def paired(values, change='speedup'):
    """Values are rates in chronological A1, B1, B2, A2 order."""
    if len(values) != 4 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('paired verdict needs four positive finite rates')
    a1, b1, b2, a2 = values
    d1, d2 = b1 / a1 - 1, b2 / a2 - 1
    mean, noise = (d1 + d2) / 2, abs(d1 - d2)
    gain = d1 > 0 and d2 > 0 and mean > noise
    regression = d1 < 0 and d2 < 0 and mean < -max(noise, .005)
    verdict = 'resolved gain' if gain else 'resolved regression' if regression else 'neutral'
    return {'d1': d1, 'd2': d2, 'mean': mean, 'spread': noise, 'verdict': verdict,
            'accepted': (not regression if change == 'simplification' else
                         gain and (change not in ('variant', 'table_row') or mean >= .015
                                   or math.isclose(mean, .015, abs_tol=1e-12)))}


def ab_validity(config, rows, summary):
    """Validate counts and provenance before any pair can be pooled."""
    import faults
    import tune
    visits = tune.visits(config)
    if summary.get('status', 'passed') != 'passed':
        raise ValueError('A/B job did not pass its runtime checks')
    expected = {(v['label'], kind, c, repeat): v for v in visits
                for repeat, windows in v['passes'] for kind, c in windows}
    actual = [(r['label'], r['kind'], r['concurrency'], r['repeat']) for r in rows]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('wrong visit or cell counts / visit order')
    processes = summary.get('processes', {})
    if set(processes) != {v['label'] for v in visits}:
        raise ValueError('wrong process visit counts')
    groups = [processes[v['label']].get('process_group') for v in visits]
    if any(type(group) is not int or group <= 0 for group in groups) or len(set(groups)) != len(visits):
        raise ValueError('A/B visits must use fresh process groups')
    if summary.get('exactness', {}).get('passed') is not True:
        raise ValueError('missing per-seed exactness evidence')
    snapshots = list(summary.get('power', {}).values())
    if not {'before', 'after'} <= set(summary.get('power', {})):
        raise ValueError('missing job power boundaries')
    for visit in visits:
        process = processes[visit['label']]
        if config['model'].get('speculation_mode') != 'off' and process.get('weights', {}).get('source') != 'borrowed':
            raise ValueError('DSpark visit lacks keeper borrowing')
        faults.validate(process.get('faults', {}))
        for boundary in ('power_before', 'power_after'):
            if boundary not in process:
                raise ValueError('missing visit power boundary')
            snapshots.append(process[boundary])
    limits = None
    changed = False
    for snapshot in snapshots:
        current, clocks = snapshot.get('enforced_power_limits_w'), snapshot.get('sm_clock_mhz')
        count = config['model'].get('tensor_parallel_size', 1)
        if (not current or len(current) != count or not clocks or len(clocks) != count
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in current + clocks)):
            raise ValueError('incomplete power or clock evidence')
        limits = current if limits is None else limits
        changed |= current != limits
    for row in rows:
        visit = expected[(row['label'], row['kind'], row['concurrency'], row['repeat'])]
        if row['arm'] != visit['arm'] or row['visit'] != visit['visit']:
            raise ValueError('wrong visit identity')
        if row.get('seed_set') != tune.seed_set(config, row['repeat'], row['concurrency']):
            raise ValueError('mismatched seeds')
        if row.get('window_seconds') != 15 or row.get('temperature') != 1.0:
            raise ValueError('A/B requires 15-second temperature 1.0 windows')
        if not row.get('synthetic') and not math.isclose(row.get('request_seconds', 0), 15, abs_tol=1e-6):
            raise ValueError('measured interval differs from 15 seconds')
        if row.get('cached_prompt_tokens') != 0:
            raise ValueError('cached prompt tokens in measured window')
        faults.validate(row.get('faults', {}))
        clocks = row.get('energy', {}).get('sm_mhz_range')
        if not clocks or len(clocks) != len(limits) or any(len(x) != 2 or x[0] <= 0 or x[1] < x[0] for x in clocks):
            raise ValueError('missing per-window clocks')
        window_limits = row.get('energy', {}).get('enforced_power_limits_w')
        if not window_limits or len(window_limits) != len(limits):
            raise ValueError('missing per-window enforced limits')
        if window_limits != limits:
            changed = True
    for visit in visits:
        measured = faults.combine([r['faults'] for r in rows if r['label'] == visit['label']])
        if any(measured[k] != processes[visit['label']]['faults'][k] for k in ('complete', 'major_faults')):
            raise ValueError('incomplete visit faults')
    return {'status': 'void' if changed or summary.get('status') == 'void' else 'valid',
            'enforced_power_limits_w': limits,
            'reason': 'power limits changed within job' if changed or summary.get('status') == 'void' else None}


def ab(run):
    """Contract A decisions preserve matched pairs and never pool void jobs."""
    run = Path(run)
    config, rows = load(run, allow_void=True)
    summary = json.loads((run / 'summary.json').read_text())
    validity = ({'status': 'void', 'reason': 'power limits changed within job'}
                if summary.get('status') == 'void' else ab_validity(config, rows, summary))
    out = {**validity, 'pairs': {}, 'weighted': {}, 'mechanics_only': summary.get('dry_run', False),
           'contract_a_admissible': validity['status'] == 'valid' and not summary.get('dry_run', False)}
    lines = [f"# {config['name']}: served A/B", '', f"Status: {validity['status']}", '',
             '| Pair / cell | d1 | d2 | mean | spread | verdict | weight |',
             '|---|---:|---:|---:|---:|---|---:|']
    if validity['status'] == 'valid':
        model = config['model']
        if (any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
                for w in model.get('cell_weights', {}).values())
                or not set(model.get('required_shapes', {})) <= {str(c) for c in model['concurrencies']}):
            raise ValueError('invalid served cell weights or target concurrencies')
        change = model.get('change', 'speedup')
        if change not in ('speedup', 'simplification', 'variant', 'table_row'):
            raise ValueError('unknown A/B change class')
        for baseline, treatment in model['pairs']:
            key = f'{baseline}/{treatment}'
            pair = out['pairs'][key] = {}
            decisions = []
            unmatched = []
            for c in model['concurrencies']:
                selected = sorted((r for r in rows if r['arm'] in (baseline, treatment)
                                   and r['concurrency'] == c), key=lambda r: r['visit'])
                repeats = model.get('concurrency_repeats', {}).get(str(c), model['repeats'])
                order = [r['arm'] for r in selected]
                wanted = [baseline, treatment, treatment, baseline] if repeats == 2 else [baseline, treatment]
                if order != wanted:
                    raise ValueError(f'{key} C{c}: visit order {order}, expected {wanted}')
                result = pair[str(c)] = {'order': order, 'abba': repeats == 2, 'arms': {}, 'cells': {}}
                if config.get('cold_engram'):
                    result['cold_engram'] = {row['label']: {'residency': row['cold_engram'],
                        'ttft_seconds': row['ttft_seconds'], 'mapped_gather_us': row['mapped_gather_us'],
                        'prefill_execution_us': row['prefill_execution_us']} for row in selected}
                for label in (baseline, treatment):
                    windows = [r for r in selected if r['arm'] == label]
                    value = measure(windows)
                    steps = sum(r['full']['steps'] for r in windows)
                    if not steps:
                        raise ValueError('no full-concurrency steps')
                    value.update(tokens_per_step=sum(r['full']['tokens'] for r in windows) / steps,
                                 allocated_share=sum(r['full'].get('allocated_steps', 0) for r in windows) / steps)
                    result['arms'][label] = value
                # Energy is separate and comparable only after all limits match.
                base, head = (result['arms'][x] for x in (baseline, treatment))
                result['energy_ratio'] = (head['joules_median'] / base['joules_median']
                                          if head['joules_median'] and base['joules_median'] else None)
                if repeats != 2:
                    result['verdict'] = 'short control'
                    continue
                result.update(paired([r['request_tokens_per_second'] for r in selected], change))
                weight = model.get('cell_weights', {}).get(f'decode:c{c}', 1 if c in (1, 2, 4, 8, 16, 32) else 0)
                result['weight'] = weight
                result['target'] = f'decode:c{c}' in model.get('target_cells', [f'decode:c{c}'])
                if weight > 0:
                    decisions.append((weight, result))
                normalized = []
                for row in selected:
                    cells = {}
                    for shape, data in row['cells'].items():
                        if not re.fullmatch(r'i\d+-r\d+(?:-v\d+-R\d+)?', shape):
                            raise ValueError('invalid served cell shape')
                        cells.setdefault(shape.split('-v')[0], []).extend(data['cycle_ms'])
                    normalized.append(cells)
                shapes = set().union(*normalized)
                required = model.get('required_shapes', {}).get(str(c), [])
                if not set(required) <= shapes:
                    raise ValueError('missing required served cells')
                for shape in sorted(shapes):
                    values = [median(cells.get(shape, [])) for cells in normalized]
                    match = re.fullmatch(r'i(\d+)-r(\d+)(?:-v\d+-R\d+)?', shape)
                    if not match:
                        raise ValueError('invalid served cell shape')
                    items, width = map(int, match.groups())
                    weight = (model.get('cell_weights', {}).get(f'verify:r{width}',
                               1 if width in (6, 8, 16, 32, 64) and width > items else 0) if items == c else 0)
                    if any(v is None for v in values):
                        if shape in required:
                            raise ValueError('missing required served cells in a visit')
                        result['cells'][shape] = {'verdict': 'unmatched', 'weight': 0}
                        if weight > 0:
                            unmatched.append(f'c{c}/{shape}')
                        continue
                    decision = {**paired([1 / v for v in values], change), 'weight': weight}
                    decision['target'] = f'c{c}/{shape}' in model.get('target_cells', [f'c{c}/{shape}'])
                    result['cells'][shape] = decision
                    if weight > 0:
                        decisions.append((weight, decision))
                for cell, decision in [(f'decode:c{c}', result), *result['cells'].items()]:
                    if 'd1' in decision:
                        lines.append(f"| {key} {cell} | {decision['d1']:.2%} | {decision['d2']:.2%} | "
                                     f"{decision['mean']:.2%} | {decision['spread']:.2%} | "
                                     f"{decision['verdict']} | {decision['weight']} |")
            if not decisions or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w, _ in decisions):
                raise ValueError('no valid weighted served cells')
            total = sum(w for w, _ in decisions)
            targets = [d for _, d in decisions if d['target']]
            if change != 'simplification' and not targets:
                raise ValueError('no weighted target cells')
            # A weighted average cannot hide a resolved cell regression.
            out['weighted'][key] = {'mean': sum(w * d['mean'] for w, d in decisions) / total,
                                    'cells': len(decisions), 'weight': total,
                                    'unmatched_served_cells': unmatched,
                                    'accepted': not unmatched and all(d['verdict'] != 'resolved regression' for _, d in decisions)
                                                and (change == 'simplification' or all(d['accepted'] for d in targets)),
                                    'verdict': ('resolved regression' if any(d['verdict'] == 'resolved regression' for _, d in decisions)
                                                else 'resolved gain' if not unmatched and all(d['verdict'] == 'resolved gain' for _, d in decisions)
                                                else 'neutral')}
    (run / 'ab.json').write_text(json.dumps(out, indent=2) + '\n')
    (run / 'ab.md').write_text('\n'.join(lines) + '\n')
    return out


if __name__ == '__main__':
    if sys.argv[1:2] == ['--ab']:
        print(json.dumps(ab(sys.argv[2])))
    elif sys.argv[1:2] == ['--costs']:
        print(json.dumps({'costs': costs(sys.argv[2])['summary']}))
    else:
        print(json.dumps({'depths': merged(sys.argv[1:])}))
