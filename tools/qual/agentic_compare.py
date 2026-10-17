#!/usr/bin/env python3
"""Compare a verified tbt/fixed pair of completed agentic-mixed v3 sweeps."""
import argparse
import hashlib
import json
from pathlib import Path

from agentic_metrics import tail
from agentic_mixed import REVISION
from agentic_server import mode_evidence, verify_arm
from common import compared_timing


def validate_pair(left, right):
    if {left.get('arm'), right.get('arm')} != {'tbt', 'fixed'}:
        raise ValueError('expected one tbt arm and one fixed arm')
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


def compare(left, right):
    validate_pair(left, right)
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
        # Every figure carries its sample count in each arm; a percentile also
        # whether that count supports it (agentic_metrics.TAIL_SAMPLES).
        figures = {
            'output_tokens_per_second': lambda level: dict(value=level['output_tokens_per_second']['value'],
                                                           samples=level['output_tokens_per_second']['samples']),
            'ttft_p95_ms': lambda level: tail(level['ttft_ms'], 'p95'),
            'client_step_gap_p95_ms': lambda level: tail(level['step_gap_ms'], 'p95'),
            'client_stalls_per_1000_gaps': lambda level: dict(value=level['client_stalls']['per_1000_gaps'],
                                                              samples=level['client_stalls']['eligible_gaps']),
        }
        comparisons = {}
        for name, figure in figures.items():
            x, y = figure(fixed), figure(tbt)
            comparisons[name] = dict(fixed=x, tbt=y, tbt_minus_fixed=y['value'] - x['value']
                                     if x['value'] is not None and y['value'] is not None else None)
        # Server timing histograms are primary and kept per metric/label set.
        server = {}
        for arm, level in [('fixed', fixed), ('tbt', tbt)]:
            for hist in level['server_metrics']['histograms']:
                if any(part in hist['name'] for part in ('step_', 'inter_token_latency', 'decode_')):
                    key = hist['name'] + json.dumps(hist['labels'], sort_keys=True)
                    server.setdefault(key, {})[arm] = hist
        for pair in server.values():
            if 'fixed' in pair and 'tbt' in pair:
                pair['samples'] = {arm: pair[arm]['count'] for arm in ('fixed', 'tbt')}
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
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Comparison refused: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
