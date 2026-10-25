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
    if not config.get('cold_engram'):
        warmup = summary.get('page_cache_warmup', {})
        expected_warm = {(c, repeat) for c in config['model']['concurrencies'] for repeat in (0, 1)}
        windows = warmup.get('windows', [])
        if (warmup.get('completed') is not True or len(windows) != len(expected_warm)
                or {(w['concurrency'], w['repeat']) for w in windows} != expected_warm):
            raise ValueError('missing page-cache warm-up for both seed sets')
        first = min(row.get('priming', {}).get('window_started', 0) for row in rows)
        for window in windows:
            if (window.get('seed_set') != tune.seed_set(config, window['repeat'], window['concurrency'])
                    or window.get('cached_prompt_tokens') != 0 or window.get('requests', 0) < window['concurrency']
                    or not window.get('ended', first) < first):
                raise ValueError('invalid page-cache warm-up evidence')
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
        priming = row.get('priming', {})
        times = priming.get('per_slot_completed', {})
        if (priming.get('completed_requests') != row['concurrency']
                or {int(k) for k in times} != set(range(row['concurrency']))
                or max(times.values(), default=float('inf')) > priming.get('window_started', 0)
                or priming.get('last_completed') != max(times.values(), default=None)):
            raise ValueError('window opened before every slot completed priming')
        if type(row.get('replacement_submissions')) is not int or row['replacement_submissions'] < 0:
            raise ValueError('missing replacement count')
        if not isinstance(row.get('prefill_share'), (int, float)) or not 0 <= row['prefill_share']:
            raise ValueError('missing prefill share')
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


def distribution(values):
    values = [v for v in values if v is not None]
    return {'values': values, 'median': median(values), 'spread': spread(values),
            'min': min(values) if values else None, 'max': max(values) if values else None}


def energy_check(cells):
    """Provisional J/token veto, separate from timing and only at equal limits."""
    if not cells or any(len(v) != 4 or any(x is None or x <= 0 for x in v) for v in cells.values()):
        return {'comparable': False, 'veto': False, 'provisional': True, 'reason': 'missing paired energy'}
    paired_cells = {c: paired(values) for c, values in cells.items()}
    mean = statistics.fmean(row['mean'] for row in paired_cells.values())
    same_sign = all(row['mean'] > 0 for row in paired_cells.values())
    return {'comparable': True, 'provisional': True, 'mean': mean, 'same_sign': same_sign,
            'veto': mean > .03 and not math.isclose(mean, .03, abs_tol=1e-12) and same_sign,
            'cells': {c: {k: row[k] for k in ('d1', 'd2', 'mean', 'spread')} for c, row in paired_cells.items()},
            'rule': 'Equal-weight mean J/token increase above 3%, positive at every active concurrency.'}


def decision_cells(model, selected, c, change):
    """Only adequately sampled full-concurrency shapes can decide timing."""
    weights = model.get('cell_weights', {})
    minimum = model.get('min_cell_steps', 20)
    share = model.get('min_cell_time_share')
    if type(minimum) is not int or minimum < 1 or (share is not None and not 0 < share <= 1):
        raise ValueError('invalid minimum cell evidence')
    normalized = []
    for row in selected:
        cells = {}
        for shape, data in row['cells'].items():
            if not re.fullmatch(r'i\d+-r\d+(?:-v\d+-R\d+)?', shape):
                raise ValueError('invalid served cell shape')
            values = data['cycle_ms']
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('invalid step cycle time')
            cells.setdefault(shape.split('-v')[0], []).extend(values)
        normalized.append(cells)
    required = set(model.get('required_shapes', {}).get(str(c), []))
    shapes = set().union(*normalized) | required
    result = {}
    for shape in sorted(shapes):
        match = re.fullmatch(r'i(\d+)-r(\d+)', shape)
        if not match:
            raise ValueError('invalid required shape')
        items, width = map(int, match.groups())
        key = f'decode_step:c{c}' if width == items else f'verify:r{width}'
        default = int(items == c and ((width == items and c in (1, 2, 4, 8, 16, 32))
                                     or (width > items and width in (6, 8, 16, 32, 64))))
        nominal = weights.get(f'c{c}/{shape}', weights.get(key, default)) if items == c else 0
        values = [row.get(shape, []) for row in normalized]
        counts = [len(v) for v in values]
        shares = [sum(v) / (row['request_seconds'] * 1000) for v, row in zip(values, selected)]
        supported = [n >= minimum or (share is not None and fraction >= share)
                     for n, fraction in zip(counts, shares)]
        paired_shape = len(selected) == 4 and all(supported)
        eligible = items == c and paired_shape
        cell = {'steps': counts, 'time_share': shares, 'nominal_weight': nominal,
                'supported': supported, 'weight': nominal if eligible else 0, 'decision': eligible and nominal > 0}
        if shape in required and not eligible:
            raise ValueError(f'required shape {shape} is missing, unpaired or insufficiently sampled')
        if items != c or not any(supported):
            cell.update(verdict='informational', reason='insufficient full-concurrency steps or time share')
        elif supported in ([True, False, False, True], [False, True, True, False]):
            cell.update(verdict='informational', reason='arm-specific shape')
        elif not paired_shape:
            cell.update(verdict='unmatched', reason='coverage gap')
        else:
            cell.update(paired([1 / median(v) for v in values], change))
        result[shape] = cell
    return result


def validate_cell_weights(model):
    concurrencies = set(model['concurrencies'])
    for key, weight in model.get('cell_weights', {}).items():
        decode = re.fullmatch(r'decode(?:_step)?:c([1-9]\d*)', key)
        verify = re.fullmatch(r'verify:r([1-9]\d*)', key)
        shape = re.fullmatch(r'c([1-9]\d*)/i([1-9]\d*)-r([1-9]\d*)', key)
        valid = (decode and int(decode[1]) in concurrencies) or verify or (
            shape and int(shape[1]) in concurrencies and shape[1] == shape[2])
        if not valid or type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'invalid served cell weight: {key}')


def cold_report(selected):
    result = {}
    for arm in dict.fromkeys(row['arm'] for row in selected):
        rows = [row for row in selected if row['arm'] == arm]
        result[arm] = {field: distribution([median(row[field]) for row in rows]) for field in
                      ('ttft_seconds', 'prefill_execution_us', 'mapped_gather_us')}
        result[arm]['residency'] = [row['cold_engram'] for row in rows]
        result[arm]['prompt_lengths'] = [row.get('prompt_lengths', []) for row in rows]
    return result


def ab(run):
    """Contract A decisions preserve matched pairs and never pool void jobs."""
    run = Path(run)
    config, rows = load(run, allow_void=True)
    summary = json.loads((run / 'summary.json').read_text())
    validity = ({'status': 'void', 'reason': 'power limits changed within job'}
                if summary.get('status') == 'void' else ab_validity(config, rows, summary))
    cold = bool(config.get('cold_engram'))
    out = {**validity, 'pairs': {}, 'weighted': {}, 'mechanics_only': summary.get('dry_run', False),
           'contract_a_admissible': validity['status'] == 'valid' and not summary.get('dry_run', False) and not cold}
    lines = [f"# {config['name']}: served A/B", '', f"Status: {validity['status']}", '',
             '| Pair / cell | d1 | d2 | mean | spread | verdict | weight |',
             '|---|---:|---:|---:|---:|---|---:|']
    if validity['status'] == 'valid':
        model = config['model']
        validate_cell_weights(model)
        if not set(model.get('required_shapes', {})) <= {str(c) for c in model['concurrencies']}:
            raise ValueError('invalid target concurrencies')
        change = model.get('change', 'speedup')
        if change not in ('speedup', 'simplification', 'variant', 'table_row'):
            raise ValueError('unknown A/B change class')
        declared_targets = model.get('target_cells')
        if declared_targets is not None and (not declared_targets or len(set(declared_targets)) != len(declared_targets)):
            raise ValueError('target_cells must be a nonempty unique list')
        for baseline, treatment in model['pairs']:
            key = f'{baseline}/{treatment}'
            pair = out['pairs'][key] = {}
            decisions, available, gaps, energies = [], {}, [], {}
            for c in model['concurrencies']:
                selected = sorted((row for row in rows if row['arm'] in (baseline, treatment)
                                   and row['concurrency'] == c), key=lambda row: row['visit'])
                repeats = model.get('concurrency_repeats', {}).get(str(c), model['repeats'])
                order = [row['arm'] for row in selected]
                wanted = [baseline, treatment, treatment, baseline] if repeats == 2 else [baseline, treatment]
                if order != wanted:
                    raise ValueError(f'{key} C{c}: visit order {order}, expected {wanted}')
                result = pair[str(c)] = {'order': order, 'abba': repeats == 2, 'arms': {}, 'cells': {}}
                for arm in (baseline, treatment):
                    windows = [row for row in selected if row['arm'] == arm]
                    value = measure(windows)
                    steps = sum(row['full']['steps'] for row in windows)
                    if not steps and not cold:
                        raise ValueError('no full-concurrency steps')
                    value.update(tokens_per_step=sum(row['full']['tokens'] for row in windows) / steps if steps else None,
                                 allocated_share=sum(row['full'].get('allocated_steps', 0) for row in windows) / steps if steps else None)
                    result['arms'][arm] = value
                if cold:
                    result.update(verdict='cold, unmatched prompts', timing_accepted=False,
                                  cold_engram=cold_report(selected))
                    lines += [f'| {key} C{c} | - | - | - | - | cold, unmatched prompts | 0 |']
                    for arm, values in result['cold_engram'].items():
                        for field in ('ttft_seconds', 'prefill_execution_us', 'mapped_gather_us'):
                            value = values[field]
                            lines += [f"{arm} C{c} {field}: median {value['median']}, spread {value['spread']:.2%}."]
                    continue
                if repeats != 2:
                    result.update(verdict='short control', weight=0)
                    continue
                result.update(paired([row['request_tokens_per_second'] for row in selected], change))
                result['weight'] = model.get('cell_weights', {}).get(f'decode:c{c}', int(c in (1, 2, 4, 8, 16, 32)))
                result['cells'] = decision_cells(model, selected, c, change)
                current = {f'decode:c{c}': result}
                current.update({f'c{c}/{shape}': cell for shape, cell in result['cells'].items()})
                for name, cell in current.items():
                    aliases = [name]
                    if name == f'c{c}/i{c}-r{c}':
                        aliases.append(f'decode_step:c{c}')
                    for alias in aliases:
                        available[alias] = cell
                    cell['target'] = declared_targets is None or any(alias in declared_targets for alias in aliases)
                    if cell['weight'] > 0:
                        decisions.append((cell['weight'], cell))
                    if cell['verdict'] == 'unmatched' and cell['nominal_weight'] > 0:
                        gaps.append(name)
                    if 'd1' in cell:
                        lines.append(f"| {key} {name} | {cell['d1']:.2%} | {cell['d2']:.2%} | "
                                     f"{cell['mean']:.2%} | {cell['spread']:.2%} | {cell['verdict']} | {cell['weight']} |")
                    else:
                        lines.append(f"| {key} {name} | - | - | - | - | {cell['verdict']} | {cell['weight']} |")
                if any(cell['weight'] > 0 for cell in current.values()):
                    energies[str(c)] = [(row.get('energy') or {}).get('joules_per_token') for row in selected]
            if cold:
                out['weighted'][key] = {'verdict': 'cold, unmatched prompts', 'timing_accepted': False,
                                        'energy_check': {'comparable': False, 'veto': False,
                                                         'reason': 'unmatched prompts'}, 'coverage_gaps': []}
            else:
                for target in declared_targets or []:
                    cell = available.get(target, {})
                    if cell.get('weight', 0) <= 0 or 'd1' not in cell or 'd2' not in cell:
                        raise ValueError(f'target {target} is missing, unpaired, insufficiently sampled or has no positive weight')
                if not decisions:
                    raise ValueError('no valid weighted served cells')
                targets = [cell for _, cell in decisions if cell['target']]
                if change != 'simplification' and not targets:
                    raise ValueError('no weighted target cells')
                total = sum(weight for weight, _ in decisions)
                regression = any(cell['verdict'] == 'resolved regression' for _, cell in decisions)
                timing = not regression and (change == 'simplification' or all(cell['accepted'] for cell in targets))
                out['weighted'][key] = {'mean': sum(weight * cell['mean'] for weight, cell in decisions) / total,
                    'cells': len(decisions), 'weight': total, 'coverage_gaps': gaps, 'timing_accepted': timing,
                    'verdict': 'resolved regression' if regression else 'resolved gain' if
                        all(cell['verdict'] == 'resolved gain' for _, cell in decisions) else 'neutral',
                    'energy_check': energy_check(energies)}
            weighted = out['weighted'][key]
            lines += ['', f"{key}: {weighted['verdict']}; timing acceptance: {weighted['timing_accepted']}; "
                      f"energy veto: {weighted['energy_check']['veto']} (provisional).",
                      'Coverage gaps: ' + (', '.join(weighted['coverage_gaps']) or 'none') + '.', '']
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
