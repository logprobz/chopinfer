#!/usr/bin/env python3
"""Content-free declarations and finite session replay for agentic_mixed.

Only generated code, patches and tool-result text enter requests. Token
counts are calibrated with the served tokenizer, never a bytes/token rule.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import common
import agentic_mixed as mixed
from agentic_metrics import histogram_reports, level_verdict, metrics_delta, prefill_evidence, replica_spread, summarize
from agentic_server import mode_evidence, observe_startup, verify_arm

SCHEMA = 'chopinfer.agentic-trace.v1'
CLASSES = ('1-63', '64-511', '512-2047', '2048-8191', '8192+')
RANGES = ((1, 63), (64, 511), (512, 2047), (2048, 8191), (8192, 12288))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def append_class(tokens):
    return next((name for name, (_, high) in zip(CLASSES[:-1], RANGES) if tokens <= high), CLASSES[-1])


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError('trace fields must be exactly: ' + expected)


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'trace integer must be in [{low}, {high}]')


def validate(trace):
    fields(trace, 'schema seed groups sessions page_tokens')
    if trace['schema'] != SCHEMA or not isinstance(trace['seed'], str) or not 1 <= len(trace['seed']) <= 128:
        raise ValueError('invalid trace schema or seed')
    integer(trace['page_tokens'], 1, 65536)
    if not isinstance(trace['groups'], list) or not 1 <= len(trace['groups']) <= 64:
        raise ValueError('trace needs 1 to 64 groups')
    groups = set()
    for group in trace['groups']:
        fields(group, 'id context_tokens')
        integer(group['id'], 0, 63)
        integer(group['context_tokens'], 1, 500000)
        if group['id'] in groups:
            raise ValueError('duplicate group')
        groups.add(group['id'])
    if not isinstance(trace['sessions'], list) or not 1 <= len(trace['sessions']) <= 1024:
        raise ValueError('trace needs 1 to 1024 sessions')
    ids = set()
    for session in trace['sessions']:
        fields(session, 'id group turns')
        integer(session['id'], 0, 1023)
        integer(session['group'], 0, 63)
        if session['id'] in ids or session['group'] not in groups:
            raise ValueError('duplicate session or unknown group')
        ids.add(session['id'])
        if not isinstance(session['turns'], list) or not 1 <= len(session['turns']) <= 1000:
            raise ValueError('trace needs 1 to 1000 turns per session')
        for index, turn in enumerate(session['turns']):
            fields(turn, 'after append_tokens output_tokens seed think_seconds idle_pages')
            if (index == 0 and turn['after'] is not None) or (index > 0 and
                    (type(turn['after']) is not int or turn['after'] != index - 1)):
                raise ValueError('turn must depend on its immediate predecessor')
            integer(turn['append_tokens'], 1, 32768)
            integer(turn['output_tokens'], 1, 8192)
            integer(turn['seed'], 0, (1 << 63) - 1)
            integer(turn['idle_pages'], 0, 10000000)
            delay = turn['think_seconds']
            if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 3600:
                raise ValueError('invalid think time')
    if sum(len(s['turns']) for s in trace['sessions']) > 100000:
        raise ValueError('trace exceeds 100000 turns')
    return trace


def generate(sessions=8, turns=100, seed='agentic-trace-v1', contexts=(100000, 250000), idle_pages=4096):
    rng = random.Random(seed)
    classes = [index for index, count in enumerate((31, 40, 12, 7, 10)) for _ in range(count)]
    rng.shuffle(classes)
    rows = []
    for session in range(sessions):
        planned = []
        for turn in range(turns):
            ordinal = turn * sessions + session
            low, high = RANGES[classes[ordinal % 100]]
            planned.append(dict(after=turn - 1 if turn else None, append_tokens=rng.randint(low, high),
                                output_tokens=(128, 256, 550, 1266)[(turn + session) % 4],
                                seed=mixed.request_seed(seed, sessions, session, turn),
                                think_seconds=(0, 2, 10, 60)[(turn + session) % 4],
                                idle_pages=idle_pages if session == 0 and turn and turn % 4 == 0 else 0))
        rows.append(dict(id=session, group=session % len(contexts), turns=planned))
    return validate(dict(schema=SCHEMA, seed=seed, page_tokens=64,
                         groups=[dict(id=i, context_tokens=n) for i, n in enumerate(contexts)], sessions=rows))


def synthetic(seed, characters, kind='code'):
    """Deterministic varied source snippets, diffs and test results, not padding."""
    rng = random.Random(str(seed))
    rows, size, index = [], 0, 0
    while size < characters:
        a, b = rng.randrange(100000), rng.randrange(1, 1000)
        if kind == 'patch':
            row = f'--- a/module_{index}.py\n+++ b/module_{index}.py\n@@ -1 +1 @@\n-return value + {a}\n+return value % {b}\n'
        elif kind == 'result':
            row = f'test_bucket_{index}: input={a}, divisor={b}, remainder={a % b}, status=passed\n'
        else:
            row = (f'# file: module_{index}.py\ndef bucket_{index}(value: int) -> int:\n'
                   f'    return (value + {a}) % {b}\nassert bucket_{index}(0) == {a % b}\n')
        rows.append(row)
        size += len(row)
        index += 1
    return ''.join(rows)[:characters]


def tokenize(client, body, deadline):
    keys = ('model', 'messages', 'prompt', 'thinking', 'chat_template_kwargs', 'add_generation_prompt')
    request = {k: body[k] for k in keys if k in body}
    data = b''.join(chunk for _, chunk in client.chunks('/tokenize', request,
                       min(deadline, time.monotonic() + client.timeout), mixed.MAX_RESPONSE))
    return mixed.token_id_list(json.loads(data)['tokens'])


def fitted(client, model, seed, target, deadline, kind='code'):
    """Fit a synthetic text prefix to the token target, with a bounded search.

    Tokenization need not be monotone at the final byte. Search neighbouring
    prefixes too and refuse an unachievable target, never silently mislabel it.
    """
    text = synthetic(seed, min(mixed.MAX_REQUEST // 2, max(256, target * 12)), kind)
    low, high = 1, len(text)
    best = None
    for _ in range(24):
        if low > high:
            break
        mid = (low + high) // 2
        ids = tokenize(client, dict(model=model, prompt=text[:mid]), deadline)
        if best is None or abs(len(ids) - target) < best[0]:
            best = (abs(len(ids) - target), mid)
        if len(ids) == target:
            return text[:mid]
        if len(ids) < target:
            low = mid + 1
        else:
            high = mid - 1
    for end in range(max(1, best[1] - 32), min(len(text), best[1] + 32) + 1):
        if len(tokenize(client, dict(model=model, prompt=text[:end]), deadline)) == target:
            return text[:end]
    raise ValueError('synthetic token target cannot be fitted')


def lcp(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def check_prompt(expected, actual, usage):
    if expected != actual or usage != len(expected):
        raise ValueError('served prompt differs from /tokenize')


class PromptEvidence(mixed.TokenIds):
    def __init__(self):
        super().__init__()
        self.prompt = []

    def event(self, event):
        super().event(event)
        if event.get('prompt_token_ids') is not None:
            self.prompt = mixed.token_id_list(event['prompt_token_ids'])


def ready(turn, previous_end, now, pages, previous_pages):
    return now >= previous_end + turn['think_seconds'] and pages - previous_pages >= turn['idle_pages']


def salt(profile, group, session):
    mode = profile['cache_salt']
    if mode == 'none':
        return None
    key = session if mode == 'session' else group
    return digest([profile['run_id'], profile['seed'], mode, key])[:32]


def body_for(model, profile, messages, turn, cache_salt):
    body = mixed.request_body(profile, model, messages, output=turn['output_tokens'], seed=turn['seed'])
    body.update(return_token_ids=True, ignore_eos=True, thinking={'type': 'enabled'})
    if cache_salt:
        body['cache_salt'] = cache_salt
    return body


def request(client, body, deadline, expected=None, previous=None, stall_threshold_ms=100, burst_window_ms=2):
    expected = tokenize(client, body, deadline) if expected is None else expected
    evidence = PromptEvidence()
    row, answer = mixed.stream_request(client, body, deadline, stall_threshold_ms, burst_window_ms, evidence=evidence)
    row['request_sha256'] = digest({k: v for k, v in body.items() if k != 'cache_salt'})
    row['answer_sha256'] = digest(answer)
    row['output_ids_sha256'] = digest(evidence.ids)
    row['prompt_ids_sha256'] = digest(expected)
    row['response_lcp_tokens'] = lcp(previous, expected) if previous is not None else None
    row['response_prefix_lost_tokens'] = len(previous) - lcp(previous, expected) if previous is not None else None
    if not row['failure']:
        try:
            check_prompt(expected, evidence.prompt, row['prompt_tokens'])
            if not evidence.complete() or evidence.count != body['max_tokens'] or row['output_tokens'] != evidence.count:
                raise ValueError('incomplete output token evidence')
        except ValueError as error:
            row['failure'] = str(error)
    return row, answer, expected + evidence.ids


def run_level(client, model, profile, trace, concurrency):
    """Prime prefixes, then admit ready turns without occupying thinking slots.

    The page clock counts logical uncached token pages from completed
    requests. It is a pressure proxy, not physical allocation telemetry:
    snapshots, speculative scratch and copy-on-write are reported separately
    by the server metrics. No wall-time interval is called an eviction test.
    """
    validate(trace)
    profile = {**profile, 'seed': f"{profile['seed']}/C{concurrency}"}
    timing = {key: profile[key] for key in ('stall_threshold_ms', 'burst_window_ms')}
    setup_deadline = time.monotonic() + profile['session_timeout_seconds']
    contexts = {g['id']: fitted(client, model, [trace['seed'], g['id']], g['context_tokens'], setup_deadline)
                for g in trace['groups']}
    priming, primed, states = [], {}, []
    for session in trace['sessions']:
        cache_salt = salt(profile, session['group'], session['id'])
        messages = [dict(role='system', content='Review the synthetic code and describe the change.\n' + contexts[session['group']])]
        # The served frontend refuses a system-only chat. The priming user
        # turn is not retained, so sessions reuse the shared system prefix.
        prime_messages = messages + [dict(role='user', content='Read the synthetic files and prepare to review a patch.')]
        prime_body = body_for(model, profile, prime_messages, dict(output_tokens=1, seed=0), cache_salt)
        key = (session['group'], cache_salt)
        if key not in primed:
            row, _, ids = request(client, prime_body, setup_deadline, **timing)
            if row['failure']:
                raise ValueError('prefix priming failed: ' + row['failure'])
            priming.append(row)
            primed[key] = (ids[:-1], row)
        # Calibrate append texts before the measured interval, for both histories.
        appends = [fitted(client, model, [trace['seed'], session['id'], i], t['append_tokens'], setup_deadline,
                          ('patch', 'result', 'code')[i % 3]) for i, t in enumerate(session['turns'])]
        states.append(dict(spec=session, messages=messages, appends=appends, salt=cache_salt,
                           previous=primed[key][0], previous_prompt=len(primed[key][0]), next=0,
                           rows=[], end=None, pages=0, failure=None, baseline_cached=None))
    before = client.scrape()
    startup_before = observe_startup(client.url) if hasattr(client, 'url') else None
    load = [common.host_load_sample()]
    start = time.monotonic()
    offsets = dict(start=mixed.realtime_offset_ns(), end=None)
    deadline = start + profile['seconds']
    for state in states:
        state['end'] = start
    pages = 0
    aborted = None
    pending = {}

    def drive(state, at, page_clock):
        index = state['next']
        turn = dict(state['spec']['turns'][index])
        turn['seed'] = mixed.request_seed(profile['seed'], concurrency, state['spec']['id'], turn['seed'])
        messages = state['messages'] + [dict(role='user', content=state['appends'][index])]
        body = body_for(model, profile, messages, turn, state['salt'])
        request_deadline = min(deadline + client.timeout, at + client.timeout)
        ids = tokenize(client, body, request_deadline)
        row, answer, response = request(client, body, request_deadline, ids, state['previous'], **timing)
        row.update(turn_index=index, seed=turn['seed'], append_class=append_class(turn['append_tokens']),
                   append_tokens=turn['append_tokens'], requested_output_tokens=turn['output_tokens'],
                   submitted=at, idle_seconds=at - state['end'], intervening_token_pages=page_clock - state['pages'],
                   requested_idle_pages=turn['idle_pages'],
                   lost_before_previous_answer_tokens=max(0, state['previous_prompt'] - lcp(state['previous'], ids)))
        if index:
            kept = tokenize(client, {**body, 'chat_template_kwargs': {**body.get('chat_template_kwargs', {}), 'preserve_thinking': True}}, request_deadline)
            row['preserved_thinking_lcp_tokens'] = lcp(state['previous'], kept)
            row['thinking_lcp_loss_tokens'] = max(0, lcp(state['previous'], kept) - lcp(state['previous'], ids))
        if profile['trace_history'] == 'canonical':
            answer = dict(role='assistant', content=synthetic(turn['seed'], 256, 'result'),
                          reasoning_content='Check the remainder and compare the changed branch.')
        return row, messages + [answer], response, len(ids)

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while time.monotonic() < deadline:
            for state in states:
                if len(pending) == concurrency:
                    break
                if state['failure'] or state['next'] == len(state['spec']['turns']) or any(s is state for s in pending.values()):
                    continue
                turn = state['spec']['turns'][state['next']]
                if ready(turn, state['end'], time.monotonic(), pages, state['pages']):
                    pending[pool.submit(drive, state, time.monotonic(), pages)] = state
            if not pending:
                if all(s['failure'] or s['next'] == len(s['spec']['turns']) for s in states):
                    break
                time.sleep(min(.01, max(0, deadline - time.monotonic())))
                continue
            done, _ = wait(pending, timeout=min(.05, max(0, deadline - time.monotonic())), return_when=FIRST_COMPLETED)
            for future in done:
                state = pending.pop(future)
                try:
                    row, messages, response, prompt = future.result()
                except Exception as error:
                    state['failure'] = type(error).__name__
                    state['end'] = time.monotonic()
                    continue
                state['rows'].append(row)
                state.update(messages=messages, previous=response, previous_prompt=prompt,
                             next=state['next'] + 1, end=row['ended'], failure=row['failure'])
                if not row['failure'] and (row['cached_tokens'] is None or not 0 <= row['cached_tokens'] <= row['prompt_tokens']):
                    row['failure'] = state['failure'] = 'invalid cached-token usage'
                if not row['failure']:
                    used = row['prompt_tokens'] - row['cached_tokens'] + row['output_tokens']
                    pages += math.ceil(used / trace['page_tokens'])
                if state['baseline_cached'] is None:
                    state['baseline_cached'] = row['cached_tokens']
                row['first_prefix_retained'] = (row['cached_tokens'] >= state['baseline_cached']
                                          if state['baseline_cached'] and row['cached_tokens'] is not None else None)
                # Usage and rendered LCP do not reveal which frontier the server captured.
                row['anchor_survived'] = None
                row['anchor_survival_unavailable'] = 'served capture frontier and granularity unavailable'
                state['pages'] = pages
        if pending:
            aborted = 'trace window expired with requests in flight'
            client.cancel()
    except (Exception, KeyboardInterrupt) as error:
        aborted = type(error).__name__
        client.cancel()
    finally:
        # Collect every worker, including cancellations, before making a receipt.
        for future, state in pending.items():
            try:
                row, _, _, _ = future.result()
                state['rows'].append(row)
            except Exception as error:
                state['failure'] = type(error).__name__
            state['failure'] = state['failure'] or 'trace window expired'
            state['end'] = time.monotonic()
        pool.shutdown(wait=True, cancel_futures=True)
    end = max(start + 1e-9, time.monotonic())
    load.append(common.host_load_sample())
    after = client.scrape()
    offsets['end'] = mixed.realtime_offset_ns()
    startup_after = observe_startup(client.url) if hasattr(client, 'url') else None
    sessions = []
    for state in states:
        complete = state['next'] == len(state['spec']['turns']) and not state['failure']
        sessions.append(dict(index=state['spec']['id'], case_id=f"trace-{state['spec']['id']}",
                             group=state['spec']['group'], phase='measured', turns=state['rows'],
                             started=state['rows'][0]['started'] if state['rows'] else start,
                             ended=state['end'] if complete else end + 1e-9,
                             elapsed_seconds=state['end'] - start, valid=complete, finished=complete,
                             tool_errors=0, failure=state['failure'] or (None if complete else 'trace incomplete'),
                             runner_error=state['failure']))
    summary = summarize(sessions, start, end, concurrency, profile['stall_threshold_ms'])
    delta = metrics_delta(before, after)
    summary.update(sessions=sessions, replica=profile.get('replica', 0), seed=profile['seed'],
                   priming=priming, aborted=aborted, drain_seconds=0, realtime_offset_ns=offsets,
                   burst_window_ms=profile['burst_window_ms'], stall_threshold_ms=profile['stall_threshold_ms'],
                   host_load=common.host_load(load),
                   pressure=dict(token_pages=pages, page_tokens=trace['page_tokens'],
                                 basis='ceil((prompt - cached + output) / page_tokens) per completed request; excludes snapshot and COW pages'),
                   insufficient=None if all(s['valid'] for s in sessions) and len(sessions) >= profile['min_samples']
                   else 'trace incomplete or insufficient completed sessions',
                   server_metrics=dict(before=before, after=after, **delta,
                                       histograms=histogram_reports(delta, profile['stall_threshold_ms'])))
    summary['prefill'] = prefill_evidence(delta, summary)
    summary['prefill_budget'] = mode_evidence(before, after, model, startup_before, startup_after)
    summary['arm_verification'] = verify_arm(profile['arm'], summary['prefill_budget'])
    summary['verdict'] = level_verdict(summary)
    return summary


def main(args, levels):
    common.startup()
    trace = validate(json.loads(args.trace.read_text()))
    profile = {k: v for k, v in vars(args).items() if k not in
               ('trace', 'corpus', 'url', 'model', 'label', 'arm', 'dry_run', 'fake_prefill_budget', 'frontend', 'model_root')}
    profile.update(levels=levels, trace_digest=digest(trace), trace_schema=SCHEMA)
    result = dict(workload_schema=mixed.REVISION, source_revision=SCHEMA, source_commit=None,
                  corpus_digest=digest(trace), workload_digest=digest(trace), profile=profile,
                  profile_digest=digest(profile), frontend=None, label=args.label, arm=args.arm,
                  run_id=uuid.uuid4().hex, dry_run=args.dry_run, levels=[], decode=None,
                  aborted=None, passed=False, receipt_kind='sweep', level_receipts=[])
    from agentic_fake import fake_server
    context = fake_server({}, mode=args.fake_prefill_budget) if args.dry_run else nullcontext(args.url)
    identity = dict(url=None, model=args.model, models=None, context_window=None)
    try:
        with context as url:
            client = mixed.Transport(url, args.request_timeout_seconds, None if args.dry_run else os.environ.get('OPENAI_API_KEY'))
            listing = json.loads(client.get('/v1/models'))
            model = args.model or listing['data'][0]['id']
            selected = [m for m in listing['data'] if m['id'] == model]
            if len(selected) != 1:
                raise ValueError('selected model must occur once')
            identity.update(url=url, model=model, models=listing, context_window=selected[0].get('max_model_len'))
            for c in levels:
                for replica in range(args.replicas):
                    level = run_level(client, model, {**profile, 'arm': args.arm, 'run_id': result['run_id'],
                                      'replica': replica, 'seed': mixed.replica_seed(args.seed, replica)}, trace, c)
                    result['levels'].append(level)
                    if level['aborted']:
                        raise ValueError(level['aborted'])
    except (Exception, KeyboardInterrupt) as error:
        result['aborted'] = type(error).__name__ + ': ' + str(error)
    result['passed'] = not result['aborted'] and bool(result['levels']) and all(l['verdict']['passed'] for l in result['levels'])
    result['replica_spread'] = replica_spread(result['levels'])
    windows = [l['host_load']['samples'] for l in result['levels']]
    path = common.receipt('agentic-mixed', identity, result, args.dry_run, load_windows=windows)
    path.with_suffix('.md').write_text(mixed.markdown(result, common.host_load_windows(windows)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sessions', type=int, default=8)
    p.add_argument('--turns', type=int, default=100)
    p.add_argument('--seed', default='agentic-trace-v1')
    p.add_argument('--contexts', default='100000,250000')
    p.add_argument('--idle-pages', type=int, default=4096)
    a = p.parse_args()
    workload = generate(a.sessions, a.turns, a.seed, [int(n) for n in a.contexts.split(',')], a.idle_pages)
    with a.out.open('x') as out:
        json.dump(workload, out, indent=2)
        out.write('\n')
