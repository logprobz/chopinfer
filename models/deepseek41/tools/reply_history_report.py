#!/usr/bin/env python3
"""Counter-only passive reply-end opportunity, never measured reuse speedup.

Snapshot JSON carries deployment, profile, captured_at (Unix seconds), and
metrics (the unmodified Prometheus response). Keep snapshots outside the tree.
The same process must serve both snapshots. Optional request traces diagnose
coverage only; no trace field participates in the savings calculation.
"""
import argparse
import json
import math
from pathlib import Path
import re

CALIBRATION = 79 / 2025
CONTEXTS = ("0_32k", "32_64k", "64_128k", "128_192k", "192k_plus")
ROW_KINDS = ("candidate", "eligible", "repeated", "rounding", "unobserved", "ambiguous", "prefilled")
OUTCOMES = (
    "submitted compared no_history scope_without_ancestry before_reply inside_reply final_token "
    "complete_reply shorter_consumer ambiguous any_page eligible_turn finish_stop finish_tool_calls finish_length "
    "finish_cancelled finish_delivery_failed finish_frontend_stop evicted drop_capacity "
    "drop_allocation drop_unsupported late_annotation annotated_tokens unknown_tokens mixed_tokens "
    "end_tokens ignored_end_tokens mismatch_unknown mismatch_reasoning mismatch_content "
    "mismatch_tool_calls mismatch_end_token mismatch_mixed mismatch_first_quarter "
    "mismatch_middle_half mismatch_last_quarter retention_loss_possible fingerprints_hashed digest_probes"
).split()
LINE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+\-]+)\s*')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"\\]*)"')


def metric_key(name, **labels):
    return name, tuple(sorted(labels.items()))


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = LINE.fullmatch(line)
        if not match:
            # Irrelevant histograms may use labels outside this contract.
            if line.startswith(('chop_reply_history_', 'chop_prefix_cache_recomputed_generated_', 'chop_process_uptime_')):
                raise ValueError('malformed observation metric')
            continue
        name, raw_labels, raw = match.groups()
        pairs = LABEL.findall(raw_labels or '')
        labels = dict(pairs)
        if len(labels) != len(pairs):
            raise ValueError('duplicate metric label')
        key = metric_key(name, **labels)
        value = float(raw)
        if not math.isfinite(value) or value < 0 or key in values:
            raise ValueError('nonfinite, negative or duplicate metric')
        if name.endswith('_total') and not value.is_integer():
            raise ValueError('fractional counter')
        values[key] = value
    return values


def required(values, name, **labels):
    try:
        return values[metric_key(name, **labels)]
    except KeyError as error:
        raise ValueError(f'missing metric {name} {labels}') from error


def window(before, after):
    for snapshot in (before, after):
        if not all(key in snapshot for key in ('deployment', 'profile', 'captured_at', 'metrics')):
            raise ValueError('snapshot identity or time missing')
        if not snapshot['deployment'] or not snapshot['profile']:
            raise ValueError('empty snapshot identity')
    if (before['deployment'], before['profile']) != (after['deployment'], after['profile']):
        raise ValueError('snapshots are from different deployments or profiles')
    elapsed = after['captured_at'] - before['captured_at']
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid observation window')
    first, last = (parse_metrics(snapshot['metrics']) for snapshot in (before, after))
    if any(required(values, 'chop_reply_history_enabled') != 1 for values in (first, last)):
        raise ValueError('observer was disabled; zero counters are not evidence')
    uptime_delta = required(last, 'chop_process_uptime_seconds') - required(first, 'chop_process_uptime_seconds')
    if abs(uptime_delta - elapsed) > max(2, elapsed * .01):
        raise ValueError('process reset or snapshot clock mismatch')
    for key, value in first.items():
        if key[0].endswith('_total') and key in last and last[key] < value:
            raise ValueError('counter reset')
    def delta(name, **labels):
        value = required(last, name, **labels) - required(first, name, **labels)
        if value < 0:
            raise ValueError('counter reset')
        return int(value)
    counters = {label: delta('chop_reply_history_observations_total', outcome=label) for label in OUTCOMES}
    rows = {kind: [delta('chop_reply_history_rows_total', kind=kind, context=bucket) for bucket in CONTEXTS]
            for kind in ROW_KINDS}
    if sum(rows['repeated']) != delta('chop_prefix_cache_recomputed_generated_tokens_total'):
        raise ValueError('generated-page aliases disagree; snapshot may straddle an update')
    for bucket in range(len(CONTEXTS)):
        candidate, eligible, repeated, rounding, unknown, ambiguous, prefilled = (rows[kind][bucket] for kind in ROW_KINDS)
        if not (eligible <= candidate <= prefilled and repeated + rounding <= candidate
                and unknown <= prefilled and ambiguous <= prefilled):
            raise ValueError('overlapping or inconsistent row counters; take a coherent snapshot')
    if counters['eligible_turn'] > counters['submitted'] or counters['any_page'] > counters['submitted']:
        # Windows starting with live requests can have commits for admissions
        # before the first snapshot. They cannot establish rates with this cohort.
        raise ValueError('window crosses incomplete request lifetimes')
    gauges = {name: required(last, f'chop_reply_history_{name}') for name in ('entries', 'allocated_bytes')}
    if gauges['entries'] > 256 or gauges['allocated_bytes'] > 64 * 1024 * 1024:
        raise ValueError('observer allocation contract exceeded')
    return counters, rows, gauges, elapsed


def rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'fraction': numerator / denominator if denominator else None}


def cost_share(eligible, candidate, low, high, unknown):
    """Extreme eligible/noneligible per-row costs, allowing unknown candidate rows.

    Unknown rows may all be eligible or all be noneligible. They are never
    added to measured denominator counters. Their role is uncertainty only.
    """
    yes_low = sum(n * w for n, w in zip(eligible, low))
    yes_high = sum(n * w for n, w in zip(eligible, high))
    no_low = sum((d - n) * w for n, d, w in zip(eligible, candidate, low))
    no_high = sum((d - n) * w for n, d, w in zip(eligible, candidate, high))
    missing = sum(n * w for n, w in zip(unknown, high))
    lower = yes_low / (yes_low + no_high + missing) if yes_low else 0
    upper = (yes_high + missing) / (yes_high + missing + no_low) if yes_high + missing else 0
    return lower, upper


def verdict(bounds):
    return 'stop' if bounds[1] < .02 else 'continue_requires_approval' if bounds[0] > .02 else 'inconclusive'


def analyze(before, after, *, cost_low=None, cost_high=None, transfer=None,
            assumptions=None, independent_turns=None, turn_cost_ratio=1, optional_traces=None):
    counters, rows, gauges, elapsed = window(before, after)
    total = {kind: sum(values) for kind, values in rows.items()}
    submitted, compared = counters['submitted'], counters['compared']
    if not submitted or not total['prefilled']:
        raise ValueError('empty sample: need submitted requests and committed prefill rows')
    if cost_low is None: cost_low = [1] * 5
    if cost_high is None: cost_high = [1] * 5
    if len(cost_low) != 5 or len(cost_high) != 5 or any(
            not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo <= hi
            for lo, hi in zip(cost_low, cost_high)):
        raise ValueError('five finite positive ordered row-cost bounds are required')
    if transfer is not None and (len(transfer) != 2 or not 0 <= transfer[0] <= transfer[1]
                                 or not all(math.isfinite(v) for v in transfer)):
        raise ValueError('invalid workload-transfer bounds')
    if (transfer is not None or cost_low != [1]*5 or cost_high != [1]*5) and not assumptions:
        raise ValueError('cost and transfer bounds need an explicit evidence/assumptions description')
    # Ambiguous forks and retention loss can conceal a different deeper reply.
    # Bound their union by all committed prefill, not by adding overlapping rows.
    broad_loss = any(counters[name] for name in ('retention_loss_possible', 'drop_capacity', 'drop_allocation', 'drop_unsupported'))
    unknown = [max(0, p - d) if broad_loss else min(p - d, u)
               for p, d, u in zip(rows['prefilled'], rows['candidate'], rows['unobserved'])]
    if counters['ambiguous'] or broad_loss:
        # Known candidate rows may also have uncertain classification; broaden
        # eligible bounds separately rather than double-counting denominator.
        uncertain_eligible = [d for d in rows['candidate']]
        lower_share = 0
        upper_share = cost_share(uncertain_eligible, rows['candidate'], cost_low, cost_high, unknown)[1]
    else:
        lower_share, upper_share = cost_share(rows['eligible'], rows['candidate'], cost_low, cost_high, unknown)
    equal_cost = total['eligible'] / total['candidate'] if total['candidate'] else None
    coverage_complete = not any(unknown) and not broad_loss and not counters['ambiguous']
    zero_window = coverage_complete and total['candidate'] > 0 and total['eligible'] == 0
    comparable = after['profile'] == 'tp4-dspark'
    calibrated = comparable and transfer is not None
    if zero_window:
        bounds = (0, 0)
    elif calibrated:
        bounds = (min(1, CALIBRATION * transfer[0] * lower_share), min(1, CALIBRATION * transfer[1] * upper_share))
    else:
        bounds = (0, 1)
    future = (0, min(1, CALIBRATION * transfer[1])) if calibrated else (0, 1)
    sampling = 'No independence assumption supplied; the observed window does not bound future turns.'
    if independent_turns is not None:
        if not 1 <= independent_turns <= compared or not math.isfinite(turn_cost_ratio) or turn_cost_ratio < 1:
            raise ValueError('independent turns must be within compared turns and cost ratio at least one')
        # Hoeffding bounds for weighted numerator and denominator, union bound
        # alpha=.05. Request cost lies in [1,K] under the supplied assumption.
        epsilon = turn_cost_ratio * math.sqrt(math.log(4 / .05) / (2 * independent_turns))
        error = min(1, 2 * epsilon / (1 - epsilon)) if epsilon < 1 else 1
        if calibrated:
            future = (max(0, lower_share - error) * CALIBRATION * transfer[0],
                      min(1, upper_share + error) * CALIBRATION * transfer[1])
        sampling = f'Conditional 95% bounded-weight Hoeffding envelope, {independent_turns} independent turns, cost ratio <= {turn_cost_ratio}; correlated sessions invalidate it.'
    trace = {'provided': optional_traces is not None, 'complete': None}
    if optional_traces is not None:
        trace['records'] = len(optional_traces)
        valid = [row for row in optional_traces if isinstance(row, dict)
                 and row.get('schema') == 'chopinfer.request.v1'
                 and row.get('finish_reason') in ('stop', 'length', 'abort', 'error')
                 and isinstance(row.get('engine_request_id'), int)
                 and isinstance(row.get('prompt_tokens'), int)
                 and isinstance(row.get('generation_tokens'), int)]
        trace['complete'] = (len(valid) == len(optional_traces) == submitted
                             and len({row['engine_request_id'] for row in valid}) == len(valid))
    sensitivity = []
    for ratio in [1, 2, 4, 10]:
        fraction = cost_share(rows['eligible'], rows['candidate'], [1]*5, [ratio]*5, unknown)
        if counters['ambiguous'] or broad_loss:
            fraction = (0, cost_share(rows['candidate'], rows['candidate'], [1]*5, [ratio]*5, unknown)[1])
        sensitivity.append({'row_cost_ratio': ratio, 'eligible_cost_share': fraction,
                            'busy_fraction_if_transfer_one': [CALIBRATION * v for v in fraction]})
    return {
        'schema': 'reply-history-passive-report-v1',
        'deployment': after['deployment'], 'profile': after['profile'],
        'window_seconds': elapsed, 'observer_enabled': True, 'gauges': gauges,
        'counts': counters, 'rows': dict(zip(CONTEXTS, [dict(zip(ROW_KINDS, values)) for values in zip(*(rows[k] for k in ROW_KINDS))])),
        'totals': total, 'coverage_complete': coverage_complete, 'unknown_candidate_rows_upper': sum(unknown),
        'rates': {'any_page_per_submission': rate(counters['any_page'], submitted),
                  'whole_reply_per_comparison': rate(counters['complete_reply'], compared),
                  'exact_boundary_per_submission': rate(counters['eligible_turn'], submitted)},
        'estimate': {'calibration_date': '2026-11-01', 'historical_all_reply_busy_bound': CALIBRATION,
                     'historical_ratio': '79/2025', 'equal_cost_share': equal_cost,
                     'eligible_cost_share_bounds': [lower_share, upper_share], 'row_cost_low': cost_low,
                     'row_cost_high': cost_high, 'workload_transfer': transfer, 'assumptions': assumptions,
                     'window_busy_fraction_bounds': bounds, 'window_decision': verdict(bounds),
                     'future_busy_fraction_bounds': future, 'future_decision': verdict(future),
                     'sampling': sampling, 'sensitivity': sensitivity},
        'optional_traces': trace,
        'limitations': [
            'No reuse is enabled and no speedup was measured. Potential excludes future capture/restore overhead.',
            'Page matches describe a different design with interior reply anchors; only exact R-1 rows enter this estimate.',
            'Equal row cost is a scenario, not a conservative bound. Near-end tool mismatches may select expensive replies.',
            'Unsalted prompts share a scope, not a uniquely identified session. Missing history is not a mismatch.',
            'Region labels describe prior generated tokens, not renderer causes. Hidden and partial tokens remain unknown or mixed.',
            'DeepSeek may remove earlier reasoning and re-render DSML tool calls; channel labels alone cannot establish either cause.',
            'Generated and recomputed-seen counters overlap and must not be added.',
            'Terminal channel handoff does not prove receipt by the remote client.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cost-low', type=float, nargs=5)
    parser.add_argument('--cost-high', type=float, nargs=5)
    parser.add_argument('--transfer', type=float, nargs=2)
    parser.add_argument('--assumptions')
    parser.add_argument('--independent-turns', type=int)
    parser.add_argument('--turn-cost-ratio', type=float, default=1)
    parser.add_argument('--request-traces', type=Path)
    args = parser.parse_args()
    traces = None
    if args.request_traces:
        traces = []
        for line in args.request_traces.read_text().splitlines():
            try: traces.append(json.loads(line))
            except json.JSONDecodeError: traces.append(None)
    report = analyze(json.loads(args.before.read_text()), json.loads(args.after.read_text()),
                     cost_low=args.cost_low, cost_high=args.cost_high, transfer=args.transfer,
                     assumptions=args.assumptions, independent_turns=args.independent_turns,
                     turn_cost_ratio=args.turn_cost_ratio, optional_traces=traces)
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
