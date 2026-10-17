#!/usr/bin/env python3
"""Compare two verified arms of completed agentic-mixed v3 sweeps, a baseline
and a candidate, replica by replica: replica R of one arm repeats replica R
of the other. Each level reports every replica's pair and the spread of each
figure across replicas, never tails pooled over them. Two v2 receipts still
compare as they did (`compare_v2`); a v2 receipt never pairs with a v3 one.

An arm label is a prefill-budget mode (`fixed`, `tbt`, `decode-aware`),
alone or with a `-` suffix naming the arm's other server settings
(`fixed-mbt4096`); the mode is verified against each level's evidence. The first receipt is the
baseline, except that of two arms of different modes the `fixed` one is."""
import argparse
import hashlib
import json
from pathlib import Path

from agentic_metrics import replica_figures, span
from agentic_mixed import REVISION
from agentic_server import arm_mode, mode_evidence, verify_arm
from common import compared_timing


def validate_pair(left, right):
    labels = [left.get('arm'), right.get('arm')]
    if not all(isinstance(label, str) and arm_mode(label) for label in labels):
        raise ValueError('arm labels must name a verified mode (fixed, tbt or decode-aware), optionally with a -suffix')
    if labels[0] == labels[1]:
        raise ValueError('expected two different arms')
    for receipt in (left, right):
        if receipt.get('schema') != 'chopinfer-qualification/v1' or receipt.get('check') != 'agentic-mixed':
            raise ValueError('not an agentic qualification receipt')
        if type(receipt.get('mechanics_only')) is not bool or receipt.get('dry_run') != receipt['mechanics_only']:
            raise ValueError('inconsistent mechanics identity')
        if receipt.get('workload_schema') != REVISION or receipt.get('receipt_kind') != 'sweep':
            raise ValueError('comparison requires completed v3 sweep receipts, not checkpoints')
        if receipt.get('aborted') or not receipt.get('passed'):
            raise ValueError('aborted, insufficient or invalid sweep cannot support a performance comparison')
        profile = receipt['profile']
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != receipt['profile_digest']:
            raise ValueError('profile digest does not match recorded settings')
        expected = [(c, replica) for c in profile['levels'] for replica in range(profile['replicas'])]
        if [(level['concurrency'], level.get('replica')) for level in receipt['levels']] != expected:
            raise ValueError('recorded levels and replicas do not match the profile')
        for level in receipt['levels']:
            evidence = level['prefill_budget']
            actual = mode_evidence(level['server_metrics']['before'], level['server_metrics']['after'],
                                   receipt['server']['model'], evidence.get('startup_before'), evidence.get('startup_after'))
            check = verify_arm(receipt['arm'], actual)
            if actual['mode'] != evidence['mode'] or not check['passed']:
                raise ValueError(check['reason'] or 'stored mode disagrees with raw evidence')
    if not left.get('label') or left['label'] != right.get('label'):
        raise ValueError('comparison labels must be the same nonempty value')
    # `frontend` holds the declaration and re-parser digests the raw re-parse ran with.
    for key in ('source_revision', 'source_commit', 'corpus_digest', 'workload_digest', 'profile_digest', 'profile',
                'mechanics_only', 'dry_run', 'frontend'):
        if left.get(key) != right.get(key):
            raise ValueError(f'incompatible {key}')
    for key in ('model', 'models', 'context_window'):
        if left['server'].get(key) != right['server'].get(key):
            raise ValueError(f'incompatible server {key}')


ARMS = ('baseline', 'candidate')


def spread(replicas):
    """Each figure across one level's replica pairs: per arm the values,
    samples and their min, median and max; the per-replica candidate minus
    baseline differences and their span; and whether the arms' ranges overlap."""
    names = sorted(set().union(*(pair['figures'][arm] for pair in replicas for arm in ARMS)))
    figures = {}
    for name in names:
        arms = {}
        for arm in ARMS:
            rows = [pair['figures'][arm].get(name, {}) for pair in replicas]
            values = [row.get('value') for row in rows]
            arms[arm] = dict(values=values, samples=[row.get('samples', 0) for row in rows],
                             supported=[row.get('supported') for row in rows], **span(values))
        differences = [b - a if a is not None and b is not None else None
                       for a, b in zip(arms['baseline']['values'], arms['candidate']['values'])]
        overlap = None
        if arms['baseline']['complete'] and arms['candidate']['complete'] and replicas:
            overlap = (arms['candidate']['min'] <= arms['baseline']['max']
                       and arms['baseline']['min'] <= arms['candidate']['max'])
        figures[name] = dict(**arms, candidate_minus_baseline=dict(values=differences, **span(differences)),
                             ranges_overlap=overlap)
    return figures



def ordered(left, right):
    """(baseline, candidate): the arm whose mode is fixed when the arms'
    modes differ, else the first receipt."""
    modes = arm_mode(left['arm']), arm_mode(right['arm'])
    return (right, left) if modes[0] != modes[1] and modes[1] == 'fixed' else (left, right)


def compare(left, right):
    versions = {left.get('workload_schema'), right.get('workload_schema')}
    if len(versions) != 1:
        raise ValueError('receipt versions differ: compare two v2 or two v3 sweeps')
    if versions == {V2}:
        return compare_v2(left, right)
    validate_pair(left, right)
    receipts = dict(zip(ARMS, ordered(left, right)))
    pairs = []
    for baseline, candidate in zip(receipts['baseline']['levels'], receipts['candidate']['levels']):
        levels = dict(baseline=baseline, candidate=candidate)
        def coverage(level):
            return dict(valid=level['valid_sessions'], completed=level['measured_completed'],
                        errors=level['errors'], timeouts=level['timeouts'], failure_reasons=level['failure_reasons'],
                        arm_warnings=level['arm_verification'].get('warnings', []),
                        prompt_tokens=level['prompt_tokens'], output_tokens=level['output_tokens'],
                        drain_seconds=level['drain_seconds'], tool_results=level['tool_results'],
                        server_prefill_tokens_per_step=level['prefill']['server_prefill_tokens_per_step'],
                        server_histograms_unavailable=[{'name': h['name'], 'labels': h['labels'], 'reason': h['unavailable']}
                                                       for h in level['server_metrics']['histograms'] if h['unavailable']])
        def trajectories(level):
            return {(s['case_id'], t['turn_index']): (t['request_sha256'], t['answer_sha256'])
                    for s in level['sessions'] if s['phase'] != 'warmup' for t in s['turns']
                    if 'request_sha256' in t}
        a, b = trajectories(baseline), trajectories(candidate)
        shared = a.keys() & b.keys()
        # Every figure carries its sample count in each arm; a percentile also
        # whether that count supports it (agentic_metrics.TAIL_SAMPLES).
        figures = {arm: replica_figures(level) for arm, level in levels.items()}
        comparisons = {}
        for name in ('output_tokens_per_second', 'ttft_p95_ms', 'client_step_gap_p95_ms', 'client_stalls_per_1000_gaps'):
            x, y = figures['baseline'][name], figures['candidate'][name]
            comparisons[name] = dict(baseline=x, candidate=y, candidate_minus_baseline=y['value'] - x['value']
                                     if x['value'] is not None and y['value'] is not None else None)
        # Server timing histograms are primary and kept per metric/label set.
        server = {}
        for arm, level in levels.items():
            for hist in level['server_metrics']['histograms']:
                if any(part in hist['name'] for part in ('step_', 'inter_token_latency', 'decode_')):
                    key = hist['name'] + json.dumps(hist['labels'], sort_keys=True)
                    server.setdefault(key, {})[arm] = hist
        for pair in server.values():
            if all(arm in pair for arm in ARMS):
                pair['samples'] = {arm: pair[arm]['count'] for arm in ARMS}
                f, t = pair['baseline'], pair['candidate']
                pair['candidate_minus_baseline'] = {field: t[field] - f[field]
                                                    if t[field] is not None and f[field] is not None else None
                                                    for field in ('mean_ms',)}
                pair['stall_change_per_1000_bounds'] = {
                    'lower': t['stall_per_1000_lower'] - f['stall_per_1000_upper']
                    if t['stall_per_1000_lower'] is not None and f['stall_per_1000_upper'] is not None else None,
                    'upper': t['stall_per_1000_upper'] - f['stall_per_1000_lower']
                    if t['stall_per_1000_upper'] is not None and f['stall_per_1000_lower'] is not None else None}
        pairs.append(dict(concurrency=baseline['concurrency'], replica=baseline['replica'],
                          coverage={arm: coverage(level) for arm, level in levels.items()},
                          primary_server_step_evidence=list(server.values()), secondary_client_metrics=comparisons,
                          trajectories=dict(shared_turns=len(shared), baseline_turns=len(a), candidate_turns=len(b),
                                            equal_requests=sum(a[k][0] == b[k][0] for k in shared),
                                            equal_answers=sum(a[k][1] == b[k][1] for k in shared)),
                          figures=figures))
    results = []
    for concurrency in receipts['baseline']['profile']['levels']:
        replicas = [pair for pair in pairs if pair['concurrency'] == concurrency]
        results.append(dict(concurrency=concurrency, replicas=replicas, spread=spread(replicas)))
    # Throughput and latency compared across the arms count only when both
    # ran on an idle host (common.HOST_LOAD_RULE); the label gates nothing.
    return dict(schema='chopinfer.agentic-mixed-comparison.v2', label=left['label'], mechanics_only=left['mechanics_only'],
                arms={arm: receipts[arm]['arm'] for arm in ARMS},
                profile_digest=left['profile_digest'], levels=results,
                host_load=compared_timing({receipts[arm]['arm']: receipts[arm].get('host_load') for arm in ARMS}))


# Receipts of the v2 sweep (one pass per level, a tbt and a fixed arm) keep
# their own comparison.
V2 = 'chopinfer.agentic-mixed.v2'


def validate_pair_v2(left, right):
    """`validate_pair` of the v2 sweep, which ran one pass per level."""
    if {left.get('arm'), right.get('arm')} != {'tbt', 'fixed'}:
        raise ValueError('expected one tbt arm and one fixed arm')
    for receipt in (left, right):
        if receipt.get('schema') != 'chopinfer-qualification/v1' or receipt.get('check') != 'agentic-mixed':
            raise ValueError('not an agentic qualification receipt')
        if type(receipt.get('mechanics_only')) is not bool or receipt.get('dry_run') != receipt['mechanics_only']:
            raise ValueError('inconsistent mechanics identity')
        if receipt.get('workload_schema') != V2 or receipt.get('receipt_kind') != 'sweep':
            raise ValueError('comparison requires completed v2 sweep receipts, not checkpoints')
        if receipt.get('aborted') or not receipt.get('passed'):
            raise ValueError('aborted, insufficient or invalid sweep cannot support a performance comparison')
        profile = receipt['profile']
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != receipt['profile_digest']:
            raise ValueError('profile digest does not match recorded settings')
        if [level['concurrency'] for level in receipt['levels']] != profile['levels']:
            raise ValueError('recorded levels do not match the profile')
        for level in receipt['levels']:
            evidence = level['prefill_budget']
            actual = mode_evidence(level['server_metrics']['before'], level['server_metrics']['after'],
                                   receipt['server']['model'], evidence.get('startup_before'), evidence.get('startup_after'))
            check = verify_arm(receipt['arm'], actual)
            if actual['mode'] != evidence['mode'] or not check['passed']:
                raise ValueError(check['reason'] or 'stored mode disagrees with raw evidence')
    if not left.get('label') or left['label'] != right.get('label'):
        raise ValueError('comparison labels must be the same nonempty value')
    # `frontend` holds the declaration and re-parser digests the raw re-parse ran with.
    for key in ('source_revision', 'source_commit', 'corpus_digest', 'workload_digest', 'profile_digest', 'profile',
                'mechanics_only', 'dry_run', 'frontend'):
        if left.get(key) != right.get(key):
            raise ValueError(f'incompatible {key}')
    for key in ('model', 'models', 'context_window'):
        if left['server'].get(key) != right['server'].get(key):
            raise ValueError(f'incompatible server {key}')


def compare_v2(left, right):
    """The comparison of two v2 sweeps, unchanged: schema v1 of the output."""
    validate_pair_v2(left, right)
    arms = {r['arm']: r for r in (left, right)}
    results = []
    for fixed, tbt in zip(arms['fixed']['levels'], arms['tbt']['levels']):
        def coverage(level):
            return dict(valid=level['valid_sessions'], completed=level['measured_completed'],
                        errors=level['errors'], timeouts=level['timeouts'], failure_reasons=level['failure_reasons'],
                        arm_warnings=level['arm_verification'].get('warnings', []),
                        prompt_tokens=level['prompt_tokens'], output_tokens=level['output_tokens'],
                        drain_seconds=level['drain_seconds'],
                        server_histograms_unavailable=[{'name': h['name'], 'labels': h['labels'], 'reason': h['unavailable']}
                                                       for h in level['server_metrics']['histograms'] if h['unavailable']])
        def trajectories(level):
            return {(s['case_id'], t['turn_index']): (t['request_sha256'], t['answer_sha256'])
                    for s in level['sessions'] if s['phase'] != 'warmup' for t in s['turns']
                    if 'request_sha256' in t}
        a, b = trajectories(fixed), trajectories(tbt)
        shared = a.keys() & b.keys()
        comparisons = {}
        for name, path in [('output_tokens_per_second', ('output_tokens_per_second', 'value')),
                           ('ttft_p95_ms', ('ttft_ms', 'p95')),
                           ('client_step_gap_p95_ms', ('step_gap_ms', 'p95')),
                           ('client_stalls_per_1000_gaps', ('client_stalls', 'per_1000_gaps'))]:
            av, bv = fixed[path[0]][path[1]], tbt[path[0]][path[1]]
            comparisons[name] = dict(fixed=av, tbt=bv, tbt_minus_fixed=bv-av if av is not None and bv is not None else None)
        # Server timing histograms are primary and kept per metric/label set.
        server = {}
        for arm, level in [('fixed', fixed), ('tbt', tbt)]:
            for hist in level['server_metrics']['histograms']:
                if any(part in hist['name'] for part in ('step_', 'inter_token_latency', 'decode_')):
                    key = hist['name'] + json.dumps(hist['labels'], sort_keys=True)
                    server.setdefault(key, {})[arm] = hist
        for pair in server.values():
            if 'fixed' in pair and 'tbt' in pair:
                pair['tbt_minus_fixed'] = {field: pair['tbt'][field] - pair['fixed'][field]
                                           if pair['tbt'][field] is not None and pair['fixed'][field] is not None else None
                                           for field in ('mean_ms',)}
                f, t = pair['fixed'], pair['tbt']
                pair['stall_change_per_1000_bounds'] = {
                    'lower': t['stall_per_1000_lower'] - f['stall_per_1000_upper']
                    if t['stall_per_1000_lower'] is not None and f['stall_per_1000_upper'] is not None else None,
                    'upper': t['stall_per_1000_upper'] - f['stall_per_1000_lower']
                    if t['stall_per_1000_upper'] is not None and f['stall_per_1000_lower'] is not None else None}
        results.append(dict(concurrency=fixed['concurrency'], coverage={'fixed': coverage(fixed), 'tbt': coverage(tbt)},
                            primary_server_step_evidence=list(server.values()), secondary_client_metrics=comparisons,
                            trajectories=dict(shared_turns=len(shared), fixed_turns=len(a), tbt_turns=len(b),
                                              equal_requests=sum(a[k][0] == b[k][0] for k in shared),
                                              equal_answers=sum(a[k][1] == b[k][1] for k in shared))))
    # Throughput and latency compared across the arms count only when both
    # ran on an idle host (common.HOST_LOAD_RULE); the label gates nothing.
    return dict(schema='chopinfer.agentic-mixed-comparison.v1', label=left['label'], mechanics_only=left['mechanics_only'],
                profile_digest=left['profile_digest'], levels=results,
                host_load=compared_timing({arm: arms[arm].get('host_load') for arm in ('fixed', 'tbt')}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Comparison refused: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
