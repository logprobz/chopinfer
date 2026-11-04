#!/usr/bin/env python3
"""The stateful replay acceptance verdict of one ABBA: four agentic replay
receipts, visits A1, B1, B2 and A2, each a fresh server with identical
settings (docs/contributing.md#stateful-replay-acceptance).

Per concurrency (a cell) and metric it reports each visit's value, and

    d1 = B1 / A1 - 1, d2 = B2 / A2 - 1, mean = (d1 + d2) / 2, spread = |d1 - d2|

with lower-is-better metrics inverted (A / B - 1), so a positive d is an
improvement. The cache-hit rate and sessions kept warm, in percentage
points, and stalls and stall time per 1,000 gaps can be zero in the
baseline, so they use differences (B - A, or A - B for stalls). A metric
whose samples a visit lacks (a percentile below its supported sample
count, a rate without prompt tokens, returns or eligible gaps) is
unavailable in that cell and gives no verdict.

A gain is resolved when d1 and d2 are both positive and the mean exceeds
the spread, in a declared metric and cell (`--gain`, `--cell`). A
regression is resolved in any metric and cell when d1 and d2 are both
negative and the mean is below -max(spread, 0.5%) for ratio metrics, or
below -spread for differences. Verdicts stay per metric and cell. A
resolved gain without a resolved regression replaces the old behavior;
gains and regressions together are a tradeoff, adoptable only as a
serving option value.

Joules per output token come from each visit's power samples (`--power`,
the agentic_trace_analysis format) over each cell's window, compared only
under identical power limits in every visit. An increase vetoes adoption
only when every cell has energy, both pairs show the increase in every
cell and its mean exceeds 3% in at least one; otherwise it is
informational.

Acceptance also needs the replay protocol and complete evidence:
mechanics-free visits on an idle host, run in A1, B1, B2, A2 order in
separate server processes, at temperature 1.0 with per-session cache salts
(group salts are a shared-prefix diagnostic); every power sample of a
visit carrying each GPU's power, enforced limit and SM clock, bracketing
its measured windows, with each GPU's limit constant within the visit and
the same in every visit (a change voids the visit or the job); and major
faults recorded for each visit, at most 1,000 in each (`--major-faults`).
"""
import argparse
import json
import math
from pathlib import Path
import sys

from agentic_metrics import LABEL, SAMPLE, TAIL_SAMPLES, stall_time_per_1000, tail
from agentic_trace_analysis import energy
from common import compared_timing

SCHEMA = 'chopinfer.agentic-verdict.v1'
VISITS = ('A1', 'B1', 'B2', 'A2')
# Each metric's rule: a ratio or a difference, and whether higher is better.
RULES = {
    'cache_hit_rate': ('difference', True),
    'sessions_kept_warm': ('difference', True),
    'ttft_p50_ms': ('ratio', False),
    'ttft_p95_ms': ('ratio', False),
    'return_ttft_p50_ms': ('ratio', False),
    'return_ttft_p95_ms': ('ratio', False),
    'event_gap_p50_ms': ('ratio', False),
    'event_gap_p95_ms': ('ratio', False),
    'stalls_per_1000_gaps': ('difference', False),
    'stall_ms_per_1000_gaps': ('difference', False),
    'output_tokens_per_second': ('ratio', True),
    'valid_sessions_per_minute': ('ratio', True),
}
# A ratio regression's mean must also fall below minus this.
REGRESSION_FLOOR = 0.005
# A joules-per-token increase above this can veto adoption.
ENERGY_VETO = 0.03
# Major faults a visit's measured windows may take before it is invalid.
MAJOR_FAULT_BUDGET = 1000
# Server counters each visit's cells report as evidence, never judged.
SERVER_COUNTERS = {
    'preemptions': 'vllm:num_preemptions_total',
    'prefix_evictions': 'chop_prefix_cache_evictions_total',
    'prefix_hits': 'chop_prefix_cache_hits_total',
    'ram_hits': 'chop_prefix_cache_ram_hits_total',
    'ram_demotions': 'chop_prefix_ram_demotions_total',
    'ram_promotions': 'chop_prefix_ram_promotions_total',
    'ram_missed_evictions': 'chop_prefix_ram_missed_evictions_total',
    'ram_headroom_waits': 'chop_prefix_ram_headroom_waits_total',
    'ram_fallbacks': 'chop_prefix_ram_fallbacks_total',
    'offload_stalls': 'chop_kv_offload_stalls_total',
}


def window_turns(level):
    start, end = level['window_start'], level['window_end']
    return [t for s in level['sessions'] if s.get('phase') != 'warmup' for t in s['turns']
            if start <= t['ended'] <= end]


def percentile(dist, name):
    """(value, None) of a supported percentile, else (None, why not)."""
    if not dist:
        return None, 'not recorded'
    row = tail(dist, name)
    if not row['supported']:
        return None, f"{row['samples']} samples; {name} needs {TAIL_SAMPLES[name]}"
    return row['value'], None


def rate(value, scale=1):
    if not value:
        return None, 'not recorded'
    if value.get('value') is None:
        return None, value.get('unavailable') or 'unavailable'
    return value['value'] * scale, None


def figures(level):
    """Every judged metric of one visit's cell: (value, None) or (None, reason)."""
    stalls = level['client_stalls']
    gaps = stalls['eligible_gaps']
    return {
        'cache_hit_rate': rate(level['cache_hit_rate'], 100),
        'sessions_kept_warm': rate(level.get('sessions_kept_warm'), 100),
        'ttft_p50_ms': percentile(level['ttft_ms'], 'p50'),
        'ttft_p95_ms': percentile(level['ttft_ms'], 'p95'),
        'return_ttft_p50_ms': percentile(level.get('return_ttft_ms'), 'p50'),
        'return_ttft_p95_ms': percentile(level.get('return_ttft_ms'), 'p95'),
        'event_gap_p50_ms': percentile(level['step_gap_ms'], 'p50'),
        'event_gap_p95_ms': percentile(level['step_gap_ms'], 'p95'),
        'stalls_per_1000_gaps': (stalls['per_1000_gaps'], None) if gaps else (None, 'no eligible gaps'),
        'stall_ms_per_1000_gaps': (stall_time_per_1000(stalls), None) if gaps else (None, 'no eligible gaps'),
        'output_tokens_per_second': rate(level['output_tokens_per_second']),
        'valid_sessions_per_minute': (level['valid_sessions_per_minute'], None),
    }


def counters(text, names):
    """{name: {labels: value}} of the wanted counters in a /metrics scrape."""
    found = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = SAMPLE.fullmatch(line)
        if not match or match[1] not in names:
            continue
        labels = ','.join(f'{key}={value}' for key, value in sorted(LABEL.findall(match[2] or ''))
                          if key not in ('model_name', 'engine'))
        series = found.setdefault(match[1], {})
        series[labels] = series.get(labels, 0) + float(match[3])
    return found


def server_counters(level):
    """Each SERVER_COUNTERS delta over the cell, in total and by label; None
    for a counter neither scrape exposes."""
    metrics = level.get('server_metrics') or {}
    before, after = metrics.get('before') or {}, metrics.get('after') or {}
    if before.get('error') or after.get('error') or 'text' not in before or 'text' not in after:
        return {key: None for key in SERVER_COUNTERS}
    names = set(SERVER_COUNTERS.values())
    old, new = counters(before['text'], names), counters(after['text'], names)
    result = {}
    for key, name in SERVER_COUNTERS.items():
        if name not in old and name not in new:
            result[key] = None
            continue
        labels = {label: new.get(name, {}).get(label, 0) - old.get(name, {}).get(label, 0)
                  for label in old.get(name, {}).keys() | new.get(name, {}).keys()}
        result[key] = dict(delta=sum(labels.values()), labels={k: v for k, v in sorted(labels.items()) if k})
    return result


def evidence(level):
    """Raw counts beside the judged rates: reported, never judged."""
    turns = window_turns(level)
    cached = [t.get('cached_tokens') for t in turns]
    stalls = level['client_stalls']
    kept = level.get('sessions_kept_warm') or {}
    warm = [t['warm'] for t in turns if type(t.get('warm')) is bool]
    return dict(requests=len(turns), cached_prompt_tokens=sum(cached) if all(type(v) is int for v in cached) else None,
                prompt_tokens=level['prompt_tokens']['value'], output_tokens=level['output_tokens']['value'],
                returns=kept.get('samples'), warm_returns=sum(warm), stall_count=stalls['count'],
                total_stall_ms=stalls['total_stall_ms'], eligible_gaps=stalls['eligible_gaps'],
                server=server_counters(level))


def judge(kind, higher, values):
    """d1, d2, mean and spread of (A1, B1, B2, A2), positive when B is
    better, and whether they resolve a gain or a regression; None when a
    ratio's base is not positive."""
    a1, b1, b2, a2 = values
    if kind == 'difference':
        d1, d2 = (b1 - a1, b2 - a2) if higher else (a1 - b1, a2 - b2)
    else:
        if min((a1, a2) if higher else (b1, b2)) <= 0:
            return None
        d1, d2 = (b1 / a1 - 1, b2 / a2 - 1) if higher else (a1 / b1 - 1, a2 / b2 - 1)
    mean, spread = (d1 + d2) / 2, abs(d1 - d2)
    floor = max(spread, REGRESSION_FLOOR) if kind == 'ratio' else spread
    return dict(d1=d1, d2=d2, mean=mean, spread=spread, gain=d1 > 0 and d2 > 0 and mean > spread,
                regression=d1 < 0 and d2 < 0 and mean < -floor)


def metric_verdict(name, rows, declared):
    """One metric of one cell from each visit's (value, reason)."""
    kind, higher = RULES[name]
    values = [rows[visit][0] for visit in VISITS]
    result = dict(kind=kind, higher_is_better=higher, declared=declared, values=dict(zip(VISITS, values)))
    if kind == 'difference' and name in ('cache_hit_rate', 'sessions_kept_warm'):
        result['unit'] = 'percentage points'
    missing = {visit: rows[visit][1] for visit in VISITS if rows[visit][0] is None}
    if missing:
        return dict(result, verdict='unavailable', unavailable=missing)
    if not all(math.isfinite(value) for value in values):
        return dict(result, verdict='unavailable', unavailable='nonfinite value')
    judged = judge(kind, higher, values)
    if judged is None:
        return dict(result, verdict='unavailable', unavailable='zero baseline rate')
    gain, regression = judged.pop('gain'), judged.pop('regression')
    verdict = ('resolved regression' if regression else 'resolved gain' if gain and declared
               else 'undeclared gain' if gain else 'unresolved')
    return dict(result, **judged, verdict=verdict)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def power_evidence(name, samples, levels):
    """(each GPU's enforced limit, problems) of one visit's power samples.
    Every sample must carry each GPU's power, enforced limit and SM clock,
    the samples must bracket every measured window of the visit at
    increasing times, and each GPU's limit must stay constant: a change
    voids the visit. The limits are None when the evidence is incomplete."""
    if not samples:
        return None, [f'{name}: power and clock samples not recorded']
    if not isinstance(samples, list) or not all(isinstance(sample, dict) for sample in samples):
        return None, [f'{name}: malformed power samples']
    problems, ranks = [], None
    for index, sample in enumerate(samples):
        if not finite(sample.get('at')):
            problems.append(f'{name}: power sample {index} has no time')
        for field in ('watts', 'limits', 'sm_clocks'):
            values = sample.get(field)
            if not isinstance(values, dict) or not values or not all(finite(value) for value in values.values()):
                problems.append(f'{name}: power sample {index} lacks per-GPU {field}')
            elif ranks is None:
                ranks = set(values)
            elif set(values) != ranks:
                problems.append(f'{name}: power sample {index} names other GPUs in {field}')
    if problems:
        return None, problems
    times = [sample['at'] for sample in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        problems.append(f'{name}: power sample times do not increase')
    start, end = min(level['window_start'] for level in levels), max(level['window_end'] for level in levels)
    if times[0] > start or times[-1] < end:
        problems.append(f'{name}: power samples do not bracket the measured windows')
    for rank in sorted(ranks):
        seen = sorted({sample['limits'][rank] for sample in samples})
        if len(seen) > 1:
            problems.append(f'{name}: GPU {rank} power limit changed within the visit ({seen}): the visit is void')
    return (None if problems else dict(samples[0]['limits'])), problems


def energy_cell(levels, power, limits):
    """Joules per output token of each visit's cell, and B against A, from
    power evidence `limits` (power_evidence) shows complete."""
    if any(limits.get(visit) is None for visit in VISITS):
        return dict(verdict='unavailable', unavailable='incomplete power evidence')
    if any(limits[visit] != limits['A1'] for visit in VISITS):
        return dict(verdict='unavailable', unavailable='power limits differ between visits')
    values = {}
    for visit in VISITS:
        level = levels[visit]
        try:
            value = energy(power[visit], level['window_start'], level['window_end'], level['output_tokens']['value'])
        except (ValueError, KeyError, TypeError) as error:
            return dict(verdict='unavailable', unavailable=f'{visit}: {error}')
        if value['joules_per_output_token'] is None:
            return dict(verdict='unavailable', unavailable=f"{visit}: {value['unavailable'] or 'no output tokens'}")
        values[visit] = value['joules_per_output_token']
    if min(values['A1'], values['A2']) <= 0:
        return dict(values=values, verdict='unavailable', unavailable='zero baseline energy')
    # Positive when B takes more energy per token.
    d1, d2 = values['B1'] / values['A1'] - 1, values['B2'] / values['A2'] - 1
    return dict(values=values, d1=d1, d2=d2, mean=(d1 + d2) / 2, spread=abs(d1 - d2),
                verdict='increase' if d1 > 0 and d2 > 0 else 'decrease' if d1 < 0 and d2 < 0 else 'mixed')


def process(level):
    """The listener process identities a level observed, when recorded."""
    budget = level.get('prefill_budget') or {}
    found = set()
    for point in ('startup_before', 'startup_after'):
        startup = budget.get(point) or {}
        if startup.get('pid') is None:
            return None
        found.add((startup['pid'], startup.get('start_ticks')))
    return found


def chronology(visits):
    """Each visit's measured span on the wall clock, or None when a receipt
    lacks its clock offsets; refuses visits that did not run one after the
    other in A1, B1, B2, A2 order."""
    spans = {}
    for name in VISITS:
        windows = []
        for level in visits[name]['levels']:
            offsets = level.get('realtime_offset_ns') or {}
            if not finite(offsets.get('start')) or not finite(offsets.get('end')):
                return None
            windows.append((level['window_start'] + offsets['start'] / 1e9,
                            level['window_end'] + offsets['end'] / 1e9))
        spans[name] = (min(start for start, _ in windows), max(end for _, end in windows))
    for left, right in zip(VISITS, VISITS[1:]):
        if spans[right][0] < spans[left][1]:
            raise ValueError(f'{right} did not start after {left} ended: the visits must run A1, B1, B2, A2 in order')
    return spans


def protocol_problems(profile):
    """Why identical replay settings are not the acceptance protocol."""
    problems = []
    if profile.get('cache_salt') != 'session':
        problems.append(f"cache salt {profile.get('cache_salt')!r}: shared-prefix diagnostics are informational; "
                        "acceptance isolates each session (--cache-salt session)")
    if profile.get('temperature') != 1:
        problems.append(f"temperature {profile.get('temperature')!r}: acceptance replays sample at temperature 1.0")
    return problems


def validate(visits):
    """Refuse four receipts that are not one ABBA of completed replays with
    identical settings: matched seeds within each pair, different seeds
    across pairs, the same arm in A1 and A2 and in B1 and B2, and the same
    cells, one replica each. Returns the cells."""
    for name, value in visits.items():
        if value.get('schema') != 'chopinfer-qualification/v1' or value.get('check') != 'agentic-mixed':
            raise ValueError(f'{name}: not an agentic replay receipt')
        if value.get('receipt_kind') != 'sweep' or value.get('aborted') or value.get('passed') is not True:
            raise ValueError(f'{name}: an aborted, failed or partial replay cannot support a verdict')
        if type(value.get('mechanics_only')) is not bool or not isinstance(value.get('arm'), str) or not value['arm']:
            raise ValueError(f'{name}: missing mechanics identity or arm label')
    first = visits['A1']
    for key in ('mechanics_only', 'workload_schema', 'source_revision', 'corpus_digest', 'workload_digest'):
        if any(value.get(key) != first.get(key) for value in visits.values()):
            raise ValueError(f'visits differ in {key}')
    profiles = [{k: v for k, v in value['profile'].items() if k != 'seed'} for value in visits.values()]
    if any(profile != profiles[0] for profile in profiles):
        raise ValueError('visits differ in settings other than the seed')
    if any(value['server'].get('model') != first['server'].get('model') for value in visits.values()):
        raise ValueError('visits served different models')
    seeds = {name: value['profile']['seed'] for name, value in visits.items()}
    if seeds['A1'] != seeds['B1'] or seeds['B2'] != seeds['A2'] or seeds['A1'] == seeds['B2']:
        raise ValueError('A1 and B1 must share one seed and B2 and A2 another')
    arms = {name: value['arm'] for name, value in visits.items()}
    if arms['A1'] != arms['A2'] or arms['B1'] != arms['B2'] or arms['A1'] == arms['B1']:
        raise ValueError('A1 and A2 must be one arm and B1 and B2 another')
    cells = None
    for name, value in visits.items():
        levels = [(level['concurrency'], level.get('replica', 0)) for level in value['levels']]
        if len({c for c, _ in levels}) != len(levels) or any(replica for _, replica in levels):
            raise ValueError(f'{name}: each cell needs exactly one replica')
        if cells is None:
            cells = [c for c, _ in levels]
        elif [c for c, _ in levels] != cells:
            raise ValueError('visits measured different cells')
    if not cells:
        raise ValueError('no measured cells')
    return cells


def fresh_processes(visits):
    """True when the receipts show four separate server processes, None
    when they do not identify them; refuses a process two visits share."""
    seen = {}
    for name, value in visits.items():
        found = [process(level) for level in value['levels']]
        if any(identity is None for identity in found):
            return None
        seen[name] = set().union(*found)
    for index, left in enumerate(VISITS):
        for right in VISITS[index + 1:]:
            if seen[left] & seen[right]:
                raise ValueError(f'{left} and {right} ran in one server process')
    return True


def verdict(visits, gains, cells=None, power=None, major_faults=None):
    """The ABBA verdict of `visits` ({A1, B1, B2, A2: receipt}) with the
    declared gain metrics and cells (default every cell), each visit's power
    samples and major-fault counts when recorded."""
    measured = validate(visits)
    unknown = sorted(set(gains) - set(RULES))
    if unknown or not gains:
        raise ValueError(f'declare at least one gain metric of {sorted(RULES)}' + (f'; unknown {unknown}' if unknown else ''))
    declared_cells = measured if cells is None else list(cells)
    if not set(declared_cells) <= set(measured) or not declared_cells:
        raise ValueError(f'declared cells {declared_cells} are not among the measured {measured}')
    power, major_faults = power or {}, major_faults or {}
    fresh = fresh_processes(visits)
    spans = chronology(visits)
    limits, power_problems = {}, []
    for name in VISITS:
        limits[name], found = power_evidence(name, power.get(name), visits[name]['levels'])
        power_problems += found
    results = []
    for concurrency in measured:
        levels = {name: next(level for level in visits[name]['levels'] if level['concurrency'] == concurrency)
                  for name in VISITS}
        rows = {name: figures(level) for name, level in levels.items()}
        metrics = {name: metric_verdict(name, {visit: rows[visit][name] for visit in VISITS},
                                        name in gains and concurrency in declared_cells) for name in RULES}
        cell_energy = energy_cell(levels, power, limits)
        results.append(dict(concurrency=concurrency, declared=concurrency in declared_cells, metrics=metrics,
                            energy=cell_energy, evidence={name: evidence(level) for name, level in levels.items()}))
    named = [(f"{name}@C{cell['concurrency']}", row['verdict']) for cell in results for name, row in cell['metrics'].items()]
    resolved_gains = [name for name, label in named if label == 'resolved gain']
    regressions = [name for name, label in named if label == 'resolved regression']
    unavailable_declared = [f"{name}@C{cell['concurrency']}" for cell in results for name, row in cell['metrics'].items()
                            if row['declared'] and row['verdict'] == 'unavailable']
    # A veto needs energy in every cell: the candidate is active in each.
    energies = [cell['energy'] for cell in results]
    complete = all('mean' in row for row in energies)
    increases = [row for row in energies if 'mean' in row]
    consistent = bool(increases) and all(row['d1'] > 0 and row['d2'] > 0 for row in increases) \
        and any(row['mean'] > ENERGY_VETO for row in increases)
    energy_veto = complete and consistent
    energy_note = None
    if consistent and not complete:
        missing = ', '.join(f"C{cell['concurrency']}" for cell in results if 'mean' not in cell['energy'])
        energy_note = f'energy unavailable in {missing}: the increases measured elsewhere are informational'
    timing = compared_timing({name: visits[name].get('host_load') for name in VISITS})
    problems = []
    if visits['A1']['mechanics_only']:
        problems.append('mechanics-only visits')
    if timing['timing'] != 'evidence':
        problems.append('timing is informational: ' + '; '.join(timing['reasons']))
    for name in VISITS:
        faults = major_faults.get(name)
        if faults is None:
            problems.append(f'{name}: major faults not recorded')
        elif faults > MAJOR_FAULT_BUDGET:
            problems.append(f'{name}: {faults} major faults exceed the budget of {MAJOR_FAULT_BUDGET}; rerun the visit')
    problems += power_problems
    if all(limits[name] is not None for name in VISITS) and any(limits[name] != limits['A1'] for name in VISITS):
        problems.append('power limits changed between visits: the job is void')
    if fresh is None:
        problems.append('server processes not identified')
    if spans is None:
        problems.append('visit chronology not recorded')
    problems += protocol_problems(visits['A1']['profile'])
    decision = ('tradeoff' if resolved_gains and regressions else 'regression' if regressions
                else 'replace' if resolved_gains else 'no resolved gain')
    return dict(schema=SCHEMA, declared=dict(gains=sorted(gains), cells=declared_cells),
                visits={name: dict(arm=visits[name]['arm'], seed=visits[name]['profile']['seed'],
                                   run_id=visits[name].get('run_id'), major_faults=major_faults.get(name),
                                   power_limits=limits[name], span=spans and spans[name])
                        for name in VISITS},
                arms=dict(A=visits['A1']['arm'], B=visits['B1']['arm']), mechanics_only=visits['A1']['mechanics_only'],
                fresh_processes=fresh, host_load=timing, cells=results,
                resolved_gains=resolved_gains, resolved_regressions=regressions,
                unavailable_declared=unavailable_declared, energy_veto=energy_veto, energy_note=energy_note,
                decision=decision,
                problems=problems, accepted=decision == 'replace' and not energy_veto and not problems)


def assignment(value):
    visit, separator, rest = value.partition('=')
    if not separator or visit not in VISITS or not rest:
        raise argparse.ArgumentTypeError(f'expected VISIT=VALUE with VISIT one of {", ".join(VISITS)}')
    return visit, rest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in VISITS:
        parser.add_argument(name.lower(), type=Path, metavar=name, help=f'visit {name} replay receipt')
    parser.add_argument('--gain', action='append', choices=sorted(RULES), default=[], required=True,
                        help='a metric the change declared it improves (repeatable)')
    parser.add_argument('--cell', action='append', type=int,
                        help='a concurrency the change declared it improves (repeatable; default every cell)')
    parser.add_argument('--power', action='append', type=assignment, default=[], metavar='VISIT=FILE',
                        help="a visit's power samples (agentic_trace_analysis --power format)")
    parser.add_argument('--major-faults', action='append', type=assignment, default=[], metavar='VISIT=N',
                        help="major faults over a visit's measured windows")
    args = parser.parse_args(argv)
    try:
        visits = {name: json.loads(getattr(args, name.lower()).read_text()) for name in VISITS}
        power = {visit: json.loads(Path(path).read_text()) for visit, path in args.power}
        faults = {}
        for visit, count in args.major_faults:
            if not count.isdigit():
                raise ValueError(f'{visit}: major faults must be a count')
            faults[visit] = int(count)
        result = verdict(visits, args.gain, args.cell, power, faults)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Verdict refused: {error}\n')
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
