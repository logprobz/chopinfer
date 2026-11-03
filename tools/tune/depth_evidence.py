"""Depth-sweep evidence checks. These do not qualify a serving profile."""
import math
import re
from collections import Counter


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(config, rows, summary):
    """Require exactly the experiment declared by the runner before scoring."""
    import tune
    model = config['model']
    if summary.get('status') != 'passed':
        raise ValueError(f"job status is {summary.get('status', 'missing')}")
    if model.get('mode', 'sweep') != 'sweep' or summary.get('mode') != 'sweep':
        raise ValueError('depth recommendations require a depth sweep')
    if summary.get('dry_run') is not False or any(
            row.get('synthetic') or (row.get('energy') or {}).get('synthetic') for row in rows):
        raise ValueError('synthetic or unclassified captures are diagnostic only')
    for field in ('errors', 'greedy_differences', 'sampled_differences'):
        if summary.get(field) != []:
            raise ValueError(f'missing or failed runtime evidence: {field}')
    kinds = config['kinds']
    if not kinds or set(kinds) - {'greedy', 'sampled'} or not all(positive(v) for v in kinds.values()):
        raise ValueError('workload weights must be finite and positive')
    if any(type(config['selection'].get(key)) not in (int, float)
           or not 0 <= config['selection'][key] < 1 for key in ('spread', 'margin')):
        raise ValueError('invalid selection policy')
    cs = model['concurrencies']
    if (not cs or any(type(c) is not int or c <= 0 for c in cs) or sorted(set(cs)) != cs
            or type(model['repeats']) is not int or model['repeats'] <= 0
            or type(model['verify_rows']) is not int or model['verify_rows'] <= 0):
        raise ValueError('invalid sweep dimensions')
    arms = tune.arms(config)
    depths = [arm['depth'] for arm in arms]
    if (0 not in depths or any(type(d) is not int or d < 0 for d in depths)
            or len(set(depths)) != len(depths) or any('depths' in arm for arm in arms)
            or len({arm['label'] for arm in arms}) != len(arms)):
        raise ValueError('sweep needs a baseline and one uniform treatment per depth')
    visits = tune.visits(config)
    expected = {(v['label'], k, c, p): v for v in visits for p, windows in v['passes'] for k, c in windows}
    actual = [(r['label'], r['kind'], r['concurrency'], r['repeat']) for r in rows]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('missing, duplicate or unexpected timed windows')
    if summary.get('order') != tune.order(config):
        raise ValueError('wrong process order')
    processes = summary.get('processes', {})
    if set(processes) != {v['label'] for v in visits}:
        raise ValueError('missing or unexpected processes')
    count = tune.planned_windows(config)
    if summary.get('windows') != {'planned': count, 'executed': count}:
        raise ValueError('incomplete summary window counters')
    exact = Counter()
    for row, key in zip(rows, actual):
        visit = expected[key]
        c, kind, repeat = row['concurrency'], row['kind'], row['repeat']
        name = f'{kind}-c{c}'
        filename = f'{name}-r{repeat}'
        if (any(type(row.get(k)) is not int for k in ('concurrency', 'repeat', 'depth', 'visit'))
                or row.get('arm') != visit['arm'] or row['depth'] != visit['depth']
                or row['visit'] != visit['visit'] or row.get('name') != name
                or row.get('_file') != filename):
            raise ValueError('mislabeled timed window')
        groups = model.get('groups', {}).get(str(c), 1)
        tokens = config['tokens'].get(str(c), config['tokens']['default'])
        if (type(groups) is not int or groups <= 0 or type(tokens) is not int or tokens <= 0
                or row.get('groups') != groups or row.get('tokens') != tokens
                or not positive(row.get('request_seconds')) or not positive(row.get('request_tokens_per_second'))
                or not math.isclose(row['request_tokens_per_second'], groups * c * tokens / row['request_seconds'])):
            raise ValueError('invalid request rate or window size')
        full = row['full']
        histogram = full.get('histogram')
        if (not isinstance(histogram, dict) or not histogram
                or any(not re.fullmatch(fr'i{c}-r[1-9][0-9]*(?:-v[0-9]+)?(?:-R[0-9]+)?', key)
                       or type(n) is not int or n <= 0 for key, n in histogram.items())
                or type(full.get('steps')) is not int or full['steps'] != sum(histogram.values())
                or not all(positive(full.get(k)) for k in ('tokens', 'seconds', 'tokens_per_second'))
                or not math.isclose(full['tokens_per_second'], full['tokens'] / full['seconds'])):
            raise ValueError('invalid full-concurrency shape or rate evidence')
        for shape in histogram:
            match = re.fullmatch(r'i([0-9]+)-r([0-9]+)(?:-v([0-9]+))?(?:-R([0-9]+))?', shape)
            used, verifying, reserved = match.group(2, 3, 4)
            if (not c <= int(used) <= c * (row['depth'] + 1)
                    or verifying is not None and not 0 <= int(verifying) <= c
                    or reserved is not None and int(reserved) < int(used)):
                raise ValueError('full-concurrency shape exceeds its depth or reserved rows')
        process = processes[row['label']]
        recorded = process.get('windows', {}).get(filename, {})
        if any(recorded.get(k) != row[k] for k in ('request_tokens_per_second', 'full')):
            raise ValueError('window evidence differs from summary')
        exact[name] += 1
    if summary.get('exact') != dict(exact):
        raise ValueError('incomplete exactness counters')
    for visit in visits:
        process = processes[visit['label']]
        names = {f'{k}-c{c}-r{p}' for p, windows in visit['passes'] for k, c in windows}
        if (process.get('depth') != visit['depth'] or set(process.get('windows', {})) != names
                or any(process.get('metrics', {}).get(k) != 0
                       for k in ('preemptions', 'prefix_cache_evictions'))):
            raise ValueError('incomplete process windows or failed counters')
