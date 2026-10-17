"""Stable mixed-workload measurement math and Prometheus counter deltas."""
import math
import re
import statistics


def distribution(values):
    values = sorted(values)
    return {'samples': len(values), **{name: values[max(0, int(q * len(values)) - 1)] if values else None
                                     for name, q in [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1)]}}


# Samples a nearest-rank percentile needs before one sample is expected above
# it, 1 / (1 - q): with fewer it is the maximum or next to it, not a tail.
TAIL_SAMPLES = {'p50': 2, 'p95': 20, 'p99': 100}


def tail(dist, name):
    """Percentile NAME of a `distribution`, with its sample count and
    whether the count supports it (TAIL_SAMPLES)."""
    return dict(value=dist[name], samples=dist['samples'], supported=dist['samples'] >= TAIL_SAMPLES[name])


def metric(value=None, samples=0, missing=0, unavailable=None):
    return dict(value=value, samples=samples, missing=missing, unavailable=unavailable)


def ratio(rows, numerator, denominator):
    good = [r for r in rows if type(r.get(numerator)) is int and type(r.get(denominator)) is int
            and 0 <= r[numerator] <= r[denominator]]
    missing = len(rows) - len(good)
    total = sum(r[denominator] for r in good)
    if missing or not total:
        return metric(samples=len(good), missing=missing, unavailable='missing counters' if missing else 'zero denominator')
    return metric(sum(r[numerator] for r in good) / total, len(good))


def summarize(sessions, start, end, concurrency, stall_threshold_ms=100):
    measured = [s for s in sessions if s['phase'] != 'warmup']
    completed = [s for s in measured if start <= s['ended'] <= end]
    turns = [t for s in measured for t in s['turns'] if start <= t['ended'] <= end]
    generated = [t for t in turns if t['output_deltas']]
    used = [t for t in generated if type(t['output_tokens']) is int and t['output_tokens'] >= 0]
    missing = len(generated) - len(used)
    seconds = end - start
    throughput = metric(sum(t['output_tokens'] for t in used) / seconds, len(used)) if used and not missing else metric(
        samples=len(used), missing=missing, unavailable='completion usage missing or no generated turns')
    events = []
    for s in measured:
        for t in s['turns']:
            a, b = max(start, t['started']), min(end, t['ended'])
            if b > a:
                events.extend([(a, 1), (b, -1)])
    active = peak = 0
    occupied = 0.0
    previous = start
    for at, change in sorted(events):
        occupied += active * (at - previous)
        active += change
        peak = max(peak, active)
        previous = at
    valid = sum(s['valid'] for s in completed)
    return dict(concurrency=concurrency, window_start=start, window_end=end, window_seconds=seconds,
                warmup_sessions=len(sessions) - len(measured), measured_starts=len(measured),
                measured_completed=len(completed), drain_sessions=len(measured) - len(completed),
                valid_sessions=valid, invalid_sessions=len(completed) - valid,
                valid_sessions_per_minute=valid * 60 / seconds,
                completed_sessions_per_minute=len(completed) * 60 / seconds,
                patch_validity=metric(valid / len(completed), len(completed)) if completed else metric(unavailable='no completed sessions'),
                output_tokens_per_second=throughput, cache_hit_rate=ratio(turns, 'cached_tokens', 'prompt_tokens'),
                draft_acceptance=ratio(generated, 'accepted_draft_tokens', 'draft_tokens'),
                ttft_ms=distribution([t['ttft_ms'] for t in turns if t['ttft_ms'] is not None]),
                inter_delta_ms=distribution([v for t in turns for v in t['inter_delta_ms']]),
                request_ms=distribution([t['elapsed_seconds'] * 1000 for t in turns]),
                session_ms=distribution([s['elapsed_seconds'] * 1000 for s in completed]),
                peak_http_in_flight=peak, average_http_in_flight=occupied / seconds,
                tool_errors=sum(s['tool_errors'] for s in completed), tool_results=tool_results(completed),
                errors=sum(bool(s.get('runner_error')) or (not s['turns'] and bool(s.get('failure'))) for s in completed)
                       + sum(bool(t.get('failure')) for t in turns),
                timeouts=sum(bool(t.get('timed_out')) for t in turns),
                failure_reasons={reason: sum(failure_reason(s) == reason for s in completed if not s['valid'])
                                 for reason in sorted({failure_reason(s) for s in completed if not s['valid']})},
                prompt_tokens=token_total(turns, 'prompt_tokens'), output_tokens=token_total(generated, 'output_tokens'),
                tpot_ms=distribution([t['tpot_ms'] for t in turns if t.get('tpot_ms') is not None]),
                prompt_tokens_per_turn=distribution([t['prompt_tokens'] for t in turns if type(t.get('prompt_tokens')) is int]),
                uncached_prompt_tokens=distribution([t['prompt_tokens'] - t['cached_tokens'] for t in turns
                                                     if type(t.get('prompt_tokens')) is int and type(t.get('cached_tokens')) is int
                                                     and 0 <= t['cached_tokens'] <= t['prompt_tokens']]),
                **summarize_gaps(turns, stall_threshold_ms))


SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+.*)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,|$)')


def parse_metrics(text):
    """Preserve Chopinfer and reference-server series without combining them."""
    rows = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError('malformed Prometheus sample')
        name, raw_labels, raw_value = match.groups()
        if not name.startswith(('chop_', 'vllm:')):
            continue
        if not any(part in name for part in ('step_', 'decode_', 'prefix_cache', 'prefill_budget', 'mixed_budget', 'inter_token_latency_seconds', 'time_to_first_token_seconds', 'scheduler_info')):
            continue
        labels = []
        pos = 0
        for label in LABEL.finditer(raw_labels or ''):
            if label.start() != pos:
                raise ValueError('malformed metric labels')
            labels.append(label.groups())
            pos = label.end()
        if pos != len(raw_labels or '') or len({k for k, _ in labels}) != len(labels):
            raise ValueError('malformed metric labels')
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid metric value')
        key = (name, tuple(sorted(labels)))
        if key in rows:
            raise ValueError('duplicate metric series')
        rows[key] = value
    return rows


def metrics_delta(before, after):
    if before.get('error') or after.get('error'):
        return {'series': [], 'unavailable': 'metrics scrape failed'}
    try:
        a, b = parse_metrics(before['text']), parse_metrics(after['text'])
    except ValueError as error:
        return {'series': [], 'unavailable': str(error)}
    rows = []
    for name, labels in sorted(a.keys() | b.keys()):
        key = (name, labels)
        old, new = a.get(key), b.get(key)
        reason = None
        if old is None or new is None:
            reason = 'series missing at boundary'
        elif not name.endswith(('_total', '_count', '_sum', '_bucket')):
            reason = 'gauge; no counter delta'
        elif new < old:
            reason = 'counter reset'
        rows.append(dict(name=name, labels=dict(labels), before=old, after=new,
                         delta=new - old if reason is None else None, unavailable=reason,
                         gauge_change=new-old if reason == 'gauge; no counter delta' else None))
    return {'series': rows, 'unavailable': None if rows else 'no supported metrics exposed'}


def tool_results(sessions):
    """Tool results the sessions returned and those cut to the session's
    limit, in total and per limit: whether the limit bound at all."""
    by_limit = {}
    for s in sessions:
        if s.get('tool_result_bytes') is None:
            continue
        tier = by_limit.setdefault(str(s['tool_result_bytes']), dict(results=0, truncated=0))
        tier['results'] += s.get('tool_results', 0)
        tier['truncated'] += s.get('truncated_tool_results', 0)
    return dict(results=sum(t['results'] for t in by_limit.values()),
                truncated=sum(t['truncated'] for t in by_limit.values()),
                by_limit=dict(sorted(by_limit.items(), key=lambda item: int(item[0]))))


def token_total(rows, field):
    good = [r[field] for r in rows if type(r.get(field)) is int and r[field] >= 0]
    return metric(sum(good) if len(good) == len(rows) else None, len(good), len(rows) - len(good),
                  'usage missing' if len(good) != len(rows) else None)


# A session can be invalid by the model's own choice although the engine did
# its job: at temperature 0 Qwen3.8 sometimes ends its turn inside its
# thinking, or writes a tool call its grammar does not define, and
# batch-dependent numerics decide which sessions do (about 0.5% of sessions,
# under vLLM too). A level allows a few such sessions, but only when an
# independent parse of the raw generated tokens proves the shape
# (`model_outcome`); every other invalid session fails the level.
ENDED_IN_THINKING = 'ended inside thinking'
MALFORMED_CALL = 'malformed tool call'
MODEL_OUTCOMES = (ENDED_IN_THINKING, MALFORMED_CALL)
MODEL_OUTCOMES_PER_100 = 3
INVALID_SESSION = 'session did not finish a valid read, patch and final response'
# agentic_mixed.raw_class of a last turn that is reasoning and an end token,
# and of one whose reasoning is followed only by a tool region without a
# function header.
RAW_REASONING_ONLY = 'reasoning only'
RAW_MALFORMED_CALL = 'unparsed tool region'
RAW_OUTCOMES = {RAW_REASONING_ONLY: ENDED_IN_THINKING, RAW_MALFORMED_CALL: MALFORMED_CALL}
# agentic_mixed.raw_class of an earlier turn whose raw text agrees with its
# parsed reasoning, content and calls.
RAW_CALLS_AGREE = 'calls agree'


def model_outcome_limit(sessions):
    """Model outcomes a level allows: 3 per 100 measured sessions, rounded up."""
    return math.ceil(MODEL_OUTCOMES_PER_100 * sessions / 100)


def model_outcome(session):
    """(outcome, None) when an invalid session is proven to be a model outcome,
    else (None, why not).

    The decision is each turn's `raw_class`, the driver's own parse of the
    raw text of every token ID the turn generated, decoded through
    `/detokenize` after the prompt's assistant header. The last turn must be
    reasoning and one end token (the model ended its turn inside its
    thinking), or reasoning, its close and one tool region without a
    function header (a call the grammar does not define). Every earlier turn
    must agree with what was parsed from it (RAW_CALLS_AGREE), so the
    context the model ended in is the one it generated. The parsed events
    never decide; they and the session can only veto: every turn HTTP 200
    without failure or timeout, one streamed token ID per generated token,
    `tool_calls` with parsed calls before the last turn and `stop` on it, no
    tool arguments that failed to parse, the target read, and either a patch
    or no tool error. A missing field is a veto.
    """
    if session.get('runner_error') or session.get('failure') != INVALID_SESSION:
        return None, 'session failure'
    if session.get('argument_errors') is None:
        return None, 'no tool argument evidence (receipt predates it)'
    if session['argument_errors']:
        return None, 'tool arguments did not parse'
    if not session.get('read_before_edit'):
        return None, 'no code read'
    if not session.get('patch_applied') and session.get('tool_errors') != 0:
        return None, 'no patch after tool errors'
    turns = session.get('turns') or []
    if not turns:
        return None, 'no turns'
    for index, turn in enumerate(turns):
        last = index == len(turns) - 1
        if turn.get('status') != 200 or turn.get('failure') or turn.get('timed_out') is not False:
            return None, f'turn {index} did not complete cleanly'
        if turn.get('token_ids') is None:
            return None, f'turn {index} has no token evidence'
        if type(turn.get('output_tokens')) is not int or turn['token_ids'] != turn['output_tokens']:
            return None, f'turn {index} streamed {turn["token_ids"]} token IDs for {turn.get("output_tokens")} tokens'
        if turn.get('finish_reason') != ('stop' if last else 'tool_calls') or (not last and not turn.get('parsed_calls')):
            return None, f"turn {index} finished {turn.get('finish_reason')} with {turn.get('parsed_calls')} parsed calls"
        if not last and turn.get('raw_class') != RAW_CALLS_AGREE:
            return None, f"raw turn {index}: {turn.get('raw_class')}"
    raw = turns[-1].get('raw_class')
    if raw in RAW_OUTCOMES:
        return RAW_OUTCOMES[raw], None
    return None, f'raw last turn: {raw}'


def level_verdict(level):
    """One level's gate.

    Fails on an aborted or insufficient level, a failed arm verification,
    request errors and timeouts, and any invalid session that is not a proven
    model outcome (`model_outcome`). Model outcomes count against
    `model_outcome_limit` of the measured sessions: every non-warmup session,
    drained ones included.
    """
    sessions = [s for s in level['sessions'] if s['phase'] != 'warmup']
    outcomes, hard = {}, {}
    for s in sessions:
        if s['valid']:
            continue
        outcome, why = model_outcome(s)
        if outcome:
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        else:
            reason = failure_reason(s)
            if reason in ('no code read', 'no edit applied', 'no final answer'):
                failure = s.get('failure')
                reason = (f'session failure: {failure}' if failure not in (None, INVALID_SESSION)
                          else f'unverified {reason}: {why}')
            hard[reason] = hard.get(reason, 0) + 1
    problems = []
    if level.get('aborted'):
        problems.append(f"aborted: {level['aborted']}")
    if level.get('insufficient'):
        problems.append(f"insufficient: {level['insufficient']}")
    if not level['arm_verification']['passed']:
        problems.append(f"arm verification failed: {level['arm_verification'].get('reason')}")
    if level['errors']:
        problems.append(f"{level['errors']} errors")
    if level['timeouts']:
        problems.append(f"{level['timeouts']} timeouts")
    problems += [f'{count} x {reason}' for reason, count in sorted(hard.items())]
    count, limit = sum(outcomes.values()), model_outcome_limit(len(sessions))
    if count > limit:
        problems.append(f'{count} model outcomes above the limit of {limit} for {len(sessions)} measured sessions')
    return dict(passed=not problems, problems=problems, measured_sessions=len(sessions),
                model_outcomes=dict(sorted(outcomes.items())), model_outcome_count=count,
                model_outcome_limit=limit, hard_invalid=dict(sorted(hard.items())))


def failure_reason(session):
    failure = session.get('failure') or ''
    if session.get('runner_error'):
        return 'runner error'
    if 'output token limit' in failure:
        return 'output token limit'
    if 'turn safety ceiling' in failure:
        return 'turn safety ceiling'
    if 'Timeout' in failure or 'deadline' in failure:
        return 'timeout'
    if not session.get('read_before_edit'):
        return 'no code read'
    if not session.get('patch_applied'):
        return 'no edit applied'
    if not session.get('finished'):
        return 'no final answer'
    return 'request or response error'


def stall_stats(gaps, threshold_ms):
    stalls = [gap for gap in gaps if gap > threshold_ms]
    return dict(count=len(stalls), eligible_gaps=len(gaps),
                per_1000_gaps=1000 * len(stalls) / len(gaps) if gaps else None,
                total_stall_ms=sum(stalls), threshold_ms=threshold_ms)


def burst_metrics(events, window_ms, threshold_ms):
    """Adjacent output within window forms a burst; channel/tool transitions are flagged.

    Events from one socket read share a timestamp and are always grouped. The
    result is a client burst proxy, not an assertion about engine step identity.
    """
    bursts = []
    for event in events:
        at, kinds = event['at'], event['kinds']
        if bursts and (at - bursts[-1]['last']) * 1000 < window_ms:
            burst = bursts[-1]
            burst['events'] += 1
            burst['transition'] |= kinds != burst['last_kinds'] or len(kinds) > 1
            burst['last_kinds'] = kinds
            burst['last'] = at
        else:
            bursts.append(dict(first=at, last=at, events=1, first_kinds=kinds,
                               last_kinds=kinds, transition=len(kinds) > 1))
    gaps = []
    for a, b in zip(bursts, bursts[1:]):
        suspect = a['transition'] or b['transition'] or a['last_kinds'] != b['first_kinds']
        gaps.append(dict(ms=(b['first'] - a['first']) * 1000, holdback_possible=suspect))
    values = [gap['ms'] for gap in gaps]
    within = [gap['ms'] for gap in gaps if not gap['holdback_possible']]
    return dict(burst_window_ms=window_ms, burst_event_counts=[b['events'] for b in bursts],
                step_gaps=gaps, step_gap_ms=distribution(values), within_run_step_gap_ms=distribution(within),
                holdback_possible_gaps=sum(g['holdback_possible'] for g in gaps),
                client_stalls=stall_stats(values, threshold_ms),
                within_run_client_stalls=stall_stats(within, threshold_ms))


def summarize_gaps(turns, threshold=100):
    gaps = [g for turn in turns for g in turn.get('step_gaps', [])]
    values = [g['ms'] for g in gaps]
    within = [g['ms'] for g in gaps if not g['holdback_possible']]
    return dict(step_gap_ms=distribution(values), within_run_step_gap_ms=distribution(within),
                holdback_possible_gaps=sum(g['holdback_possible'] for g in gaps),
                client_stalls=stall_stats(values, threshold), within_run_client_stalls=stall_stats(within, threshold),
                request_max_gap_ms=distribution([t['step_gap_ms']['max'] for t in turns
                                                if t.get('step_gap_ms', {}).get('max') is not None]))


def span(values):
    """Min, median and max of the known VALUES, and whether every value is known."""
    known = sorted(v for v in values if v is not None)
    return dict(min=known[0] if known else None, median=statistics.median(known) if known else None,
                max=known[-1] if known else None, complete=len(known) == len(values))


def server_key(histogram):
    labels = ','.join(f'{k}={v}' for k, v in sorted(histogram['labels'].items()) if k not in ('model_name', 'engine'))
    return histogram['name'] + (f'{{{labels}}}' if labels else '')


def replica_figures(level):
    """One replica's figures, each {value, samples} and, for a percentile,
    whether its samples support it (`tail`): what the replica spread and
    agentic_compare read."""
    prefill = level['prefill']
    ratio = prefill['mixed_actual_over_predicted']
    figures = {
        'output_tokens_per_second': dict(value=level['output_tokens_per_second']['value'],
                                         samples=level['output_tokens_per_second']['samples']),
        'valid_sessions_per_minute': dict(value=level['valid_sessions_per_minute'], samples=level['measured_completed']),
        'session_p95_ms': tail(level['session_ms'], 'p95'),
        'ttft_p50_ms': tail(level['ttft_ms'], 'p50'),
        'ttft_p95_ms': tail(level['ttft_ms'], 'p95'),
        'client_step_gap_p95_ms': tail(level['step_gap_ms'], 'p95'),
        'client_stalls_per_1000_gaps': dict(value=level['client_stalls']['per_1000_gaps'],
                                            samples=level['client_stalls']['eligible_gaps']),
        'prompt_tokens_per_turn_p50': tail(prefill['prompt_tokens_per_turn'], 'p50'),
        'uncached_prompt_tokens_p50': tail(prefill['uncached_prompt_tokens'], 'p50'),
        'server_prefill_tokens_per_step': {k: prefill['server_prefill_tokens_per_step'][k] for k in ('value', 'samples')},
        'server_prefix_cache_reuse': {k: prefill['server_prefix_cache_reuse'][k] for k in ('value', 'samples')},
        'mixed_actual_over_predicted_mean': dict(value=ratio['mean'], samples=ratio['count']),
        'mixed_actual_over_predicted_p95_upper': dict(value=ratio['quantile_upper']['p95'], samples=ratio['count'],
                                                      supported=ratio['count'] >= TAIL_SAMPLES['p95']),
    }
    for histogram in level['server_metrics']['histograms']:
        if any(part in histogram['name'] for part in ('step_', 'inter_token_latency', 'decode_')):
            count = int(histogram['count'] or 0)
            key = server_key(histogram)
            figures[key + ' mean_ms'] = dict(value=histogram['mean_ms'], samples=count)
            figures[key + ' p95_upper_ms'] = dict(value=histogram['quantile_upper_ms']['p95'], samples=count,
                                                  supported=count >= TAIL_SAMPLES['p95'])
    return figures


def replica_spread(levels):
    """Per concurrency, each figure's value and samples in every replica
    (and, for a percentile, whether its samples support it; None otherwise),
    with their min, median and max. Replicas are repeated passes of one
    workload, so the spread is across replicas; no tail pools their samples."""
    groups = {}
    for level in levels:
        groups.setdefault(level['concurrency'], []).append(level)
    spreads = []
    for concurrency, group in groups.items():
        group = sorted(group, key=lambda level: level.get('replica', 0))
        figures = [replica_figures(level) for level in group]
        spread = {}
        for name in sorted(set().union(*figures)):
            values = [f.get(name, {}).get('value') for f in figures]
            spread[name] = dict(values=values, samples=[f.get(name, {}).get('samples', 0) for f in figures],
                                supported=[f.get(name, {}).get('supported') for f in figures], **span(values))
        spreads.append(dict(concurrency=concurrency, replicas=[level.get('replica', 0) for level in group], figures=spread))
    return spreads


def counter_total(delta, name):
    """The summed delta of every NAME series (any labels), or None when one is unavailable or none exists."""
    rows = [row for row in delta['series'] if row['name'] == name]
    if not rows or any(row['delta'] is None for row in rows):
        return None
    return sum(row['delta'] for row in rows)


def value_histogram(delta, name):
    """A histogram of plain values (not seconds) summed over its label sets:
    count, mean and bucket upper bounds for p50, p95 and p99 (bounds, not samples)."""
    count, total = counter_total(delta, name + '_count'), counter_total(delta, name + '_sum')
    buckets = {}
    for row in delta['series']:
        if row['name'] == name + '_bucket' and row['delta'] is not None:
            le = float(row['labels']['le'])
            buckets[le] = buckets.get(le, 0) + row['delta']
    good = bool(count) and total is not None
    upper = {}
    for label, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
        bound = next((le for le, n in sorted(buckets.items()) if good and n >= max(1, int(count * q))), None)
        upper[label] = bound if bound is not None and math.isfinite(bound) else None
    return dict(name=name, count=int(count or 0), mean=total / count if good else None, quantile_upper=upper)


def prefill_evidence(delta, summary):
    """How much prefill a window caused: the server's prefill tokens per
    prefill-carrying step and its prefix-cache reuse, the client's uncached
    prompt tokens per turn, and the TBT budget's actual-over-predicted ratio.
    A budget mode can only matter when prefill per step approaches its cap."""
    tokens, steps = counter_total(delta, 'chop_step_prefill_tokens_sum'), counter_total(delta, 'chop_step_prefill_tokens_count')
    reused, queried = (counter_total(delta, 'chop_prefix_cache_reused_tokens_total'),
                       counter_total(delta, 'chop_prefix_cache_queried_tokens_total'))
    return dict(server_prefill_tokens_per_step=metric(tokens / steps, int(steps)) if steps and tokens is not None
                else metric(unavailable='no prefill step counters'),
                server_prefix_cache_reuse=metric(reused / queried, int(queried)) if queried and reused is not None
                else metric(unavailable='no prefix cache counters'),
                uncached_prompt_tokens=summary['uncached_prompt_tokens'],
                prompt_tokens_per_turn=summary['prompt_tokens_per_turn'],
                mixed_actual_over_predicted=value_histogram(delta, 'chop_mixed_step_actual_over_predicted'))


def histogram_reports(delta, threshold_ms):
    """Per-series histogram evidence. Quantiles are bucket upper bounds, not samples.

    A non-bucket threshold yields bounds on the stall rate, never a made-up
    exact count. Compatibility aliases remain separate series.
    """
    groups = {}
    for row in delta['series']:
        name = row['name']
        suffix = next((s for s in ('_count', '_sum', '_bucket') if name.endswith(s)), None)
        if suffix is None:
            continue
        labels = {k: v for k, v in row['labels'].items() if k != 'le'}
        key = (name[:-len(suffix)], tuple(sorted(labels.items())))
        group = groups.setdefault(key, dict(name=key[0], labels=labels, count=None, sum=None, buckets=[], unavailable=None))
        if row['delta'] is None:
            group['unavailable'] = row['unavailable']
        elif suffix == '_bucket':
            group['buckets'].append((float(row['labels']['le']), row['delta']))
        else:
            group[suffix[1:]] = row['delta']
    reports = []
    for group in groups.values():
        if not group['name'].endswith('_seconds'):
            continue
        buckets = sorted(group.pop('buckets'))
        count = group['count']
        if count is None or group['sum'] is None:
            group['unavailable'] = group['unavailable'] or 'incomplete histogram'
        if buckets and (any(v < 0 or (count is not None and v > count) for _, v in buckets)
                        or any(b[1] < a[1] for a, b in zip(buckets, buckets[1:]))):
            group['unavailable'] = 'inconsistent histogram buckets'
        good = not group['unavailable'] and count and count > 0
        group['mean_ms'] = 1000 * group['sum'] / count if good else None
        group['quantile_upper_ms'] = {}
        for label, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
            upper = next((le for le, n in buckets if good and n >= max(1, int(count * q))), None)
            group['quantile_upper_ms'][label] = upper * 1000 if upper is not None and math.isfinite(upper) else None
        threshold = threshold_ms / 1000
        lower_bucket = max((n for le, n in buckets if le <= threshold), default=0)
        upper_bucket = min((n for le, n in buckets if le >= threshold), default=count or 0)
        group['stall_per_1000_lower'] = 1000 * (count - upper_bucket) / count if good else None
        group['stall_per_1000_upper'] = 1000 * (count - lower_bucket) / count if good else None
        group['stall_threshold_ms'] = threshold_ms
        reports.append(group)
    return sorted(reports, key=lambda r: (r['name'], sorted(r['labels'].items())))
