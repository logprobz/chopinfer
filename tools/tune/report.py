#!/usr/bin/env python3
"""Select a per-concurrency draft depth schedule from a depth tuner run.

Usage: python3 -I report.py RUN_DIR

Reads RUN_DIR/config.json and every timed process's window records (W0 is the
untimed warm-up and never counts) and writes RUN_DIR/tune.json and
RUN_DIR/tune.md.

Per concurrency C, kind (greedy, sampled) and depth d, over every measurement
(one per process): median and spread (max/min - 1) of request tok/s and
full-concurrency tok/s, median J/token, median stream p10, acceptance per
draft position (summed counts), drafts per verify item and whether every
measurement served the same full-concurrency step shapes.

Selection per C: a depth's score is the weighted geometric mean, over the
kinds in config["kinds"], of its median request tok/s over depth 0's. A depth
is stable when every kind served identical shapes and its request spread is at
most config["selection"]["spread"]. Among stable depths, the shallowest whose
score is within the margin of the best wins; the margin is the larger of
config["selection"]["margin"] and the best depth's request spread. A depth
that served the same shapes as a shallower one in every kind counts as that
one. The schedule gives each measured C its depth, from that C up to the next
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
    for directory in sorted(run.glob('d*p*/windows')):
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
        stable = [d for d, v in rows.items() if v['stable'] and v['equivalent_to'] is None]
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
             'stream p10 tok/s, drafts per verify item, greedy acceptance by position, identical shapes, score '
             '(weighted request tok/s over depth 0).', '']
    for c in config['model']['concurrencies']:
        chosen = selection.get(str(c))
        if not chosen:
            continue
        lines += [f"## C{c}: depth {chosen['depth']} ({chosen['reason']})", '',
                  '| depth | request | spread | full | J/token | stream p10 | drafts/item | greedy acceptance '
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


if __name__ == '__main__':
    print(json.dumps({'depths': evaluate(sys.argv[1])['depths']}))
