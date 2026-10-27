#!/usr/bin/env python3
"""Join trace receipts to step timestamps and optional power samples offline."""
import argparse
import json
from pathlib import Path

from agentic_metrics import burst_intervals, distribution


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('reversed interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def output_events(row):
    """Use output timestamps, or reconstruct an older receipt's output timing.

    A token-ID/control SSE event is not evidence of an emitted text delta.
    Missing output timing stays unavailable instead of falling back to SSE.
    """
    if 'output_events' in row:
        return row['output_events']
    count = row.get('output_deltas')
    if count == 0:
        return []
    kinds, deltas = row.get('output_event_kinds', []), row.get('inter_delta_ms', [])
    if not count or len(kinds) != count or len(deltas) != count - 1 or row.get('ttft_ms') is None:
        raise ValueError('output event timing unavailable')
    times = [row['started'] + row['ttft_ms'] / 1000]
    for delta in deltas:
        times.append(times[-1] + delta / 1000)
    return [dict(at=at, kinds=kind) for at, kind in zip(times, kinds)]


def gaps(row, prefills, window_ms=None):
    window_ms = row.get('burst_window_ms', 2) if window_ms is None else window_ms
    return interval_gaps(burst_intervals(output_events(row), window_ms), prefills)


def raw_sse_gaps(row, prefills):
    times = [event['at'] for event in row['sse_events']]
    return interval_gaps(list(zip(times, times[1:])), prefills)


def interval_gaps(intervals, prefills):
    prefills = union(prefills)
    result = {}
    for threshold in (50, 100, 250, 1000):
        selected = [(a, b) for a, b in intervals if (b - a) * 1000 > threshold]
        result[str(threshold)] = dict(count=len(selected), eligible_gaps=len(intervals),
            total_gap_ms=sum((b - a) * 1000 for a, b in selected),
            excess_ms=sum((b - a) * 1000 - threshold for a, b in selected),
            prefill_overlap_ms=sum(max(0, min(b, y) - max(a, x)) * 1000 for a, b in selected for x, y in prefills))
    return result


def energy(samples, start, end, output_tokens):
    """Trapezoidal energy; samples must bracket the entire measured window."""
    if len(samples) < 2 or samples[0]['at'] > start or samples[-1]['at'] < end:
        return dict(joules=None, joules_per_output_token=None, unavailable='power samples do not bracket window')
    ranks = set(samples[0]['watts'])
    limits = samples[0]['limits']
    if not ranks or any(set(s['watts']) != ranks or s['limits'] != limits or 'sm_clocks' not in s for s in samples):
        raise ValueError('power limits changed or rank/clock evidence missing')
    total = 0
    for a, b in zip(samples, samples[1:]):
        if b['at'] <= a['at']:
            raise ValueError('power times must increase')
        lo, hi = max(start, a['at']), min(end, b['at'])
        if hi <= lo:
            continue
        x, y = sum(a['watts'].values()), sum(b['watts'].values())
        if min(x, y) < 0:
            raise ValueError('negative power')
        slope = (y - x) / (b['at'] - a['at'])
        total += (x + slope * ((lo + hi) / 2 - a['at'])) * (hi - lo)
    return dict(joules=total, joules_per_output_token=total / output_tokens if output_tokens else None, unavailable=None)


def analyze(level, steps=(), power=()):
    offsets = level['realtime_offset_ns']
    if offsets['start'] is None or offsets['end'] is None or abs(offsets['end'] - offsets['start']) > 10000000:
        raise ValueError('missing clock alignment or more than 10 ms clock drift')
    offset = (offsets['start'] + offsets['end']) / 2 / 1e9
    start, end = level['window_start'], level['window_end']
    selected, prefills = [], []
    for step in steps:
        if step.get('schema') != 'chopinfer.step.v1':
            continue
        completed = step['completed_unix_ns'] / 1e9 - offset
        began = completed - step['execution_us'] / 1e6
        if completed < start or began > end:
            continue
        selected.append(step)
        if step['prefill_tokens']:
            prefills.append((max(start, began), min(end, completed)))
    rows = [t for s in level['sessions'] for t in s['turns'] if start <= t['ended'] <= end]
    by_class = {}
    for row in rows:
        group = by_class.setdefault(row['append_class'], [])
        if row['ttft_ms'] is not None:
            group.append(row['ttft_ms'])
    output = sum(t['output_tokens'] or 0 for t in rows)
    valid = [t for t in rows if t['prompt_tokens'] is not None and t['cached_tokens'] is not None]
    verifies = [s for s in selected if s['verify_items']]
    decoded = []
    for row in rows:
        try:
            value = dict(thresholds=gaps(row, prefills, level.get('burst_window_ms')), unavailable=None)
        except ValueError as error:
            value = dict(thresholds=None, unavailable=str(error))
        decoded.append(dict(turn_index=row['turn_index'], request_sha256=row['request_sha256'], **value))
    return dict(concurrency=level['concurrency'], replica=level['replica'], requests=len(rows),
                requests_per_second=len(rows) / (end - start), output_tokens_per_second=output / (end - start),
                ttft_by_append_class={k: distribution(v) for k, v in by_class.items()},
                gaps=decoded,
                raw_sse_gaps=[dict(turn_index=t['turn_index'], request_sha256=t['request_sha256'],
                                  thresholds=raw_sse_gaps(t, prefills)) for t in rows],
                step_samples=len(selected), prefill_overlap_available=bool(selected),
                accepted_drafts_per_step=sum(s['accepted_tokens'] for s in verifies) / len(verifies) if verifies else None,
                computed_prefill_tokens=sum(t['prompt_tokens'] - t['cached_tokens'] for t in valid) if len(valid) == len(rows) else None,
                reused_prefill_tokens=sum(t['cached_tokens'] for t in valid) if len(valid) == len(rows) else None,
                anchor_survival=[t.get('anchor_survived') for t in rows if t['requested_idle_pages']],
                thinking_lcp_loss_tokens=[t.get('thinking_lcp_loss_tokens') for t in rows],
                lost_before_previous_answer_tokens=[t.get('lost_before_previous_answer_tokens') for t in rows],
                energy=energy(power, start, end, output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path)
    p.add_argument('--steps', type=Path)
    p.add_argument('--power', type=Path, help='JSON samples: at (monotonic seconds), watts/limits/sm_clocks by rank')
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_text())
    steps = []
    if args.steps:
        for line in args.steps.read_text().splitlines():
            at = line.find('{')
            if at >= 0 and 'chopinfer.step.v1' in line:
                steps.append(json.loads(line[at:]))
    power = json.loads(args.power.read_text()) if args.power else []
    if not receipt['levels']:
        raise ValueError('no measured levels')
    print(json.dumps([analyze(level, steps, power) for level in receipt['levels']], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
