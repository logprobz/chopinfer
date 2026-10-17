"""CPU tests for the pinned workload, stream framing and measurement contract."""
import hashlib
import io
from contextlib import contextmanager, redirect_stdout
from functools import partial
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import agentic_fake
from agentic_fake import EVENT_SECONDS, FakeClock, Reads, fake_server
from agentic_metrics import distribution, metrics_delta, parse_metrics, ratio, summarize
from agentic_mixed import Transport, main, markdown, parser, run_decode, run_level, session, sse_events, stream_request
from agentic_workload import CORPUS, Workspace, decode_context, load_corpus, stage_corpus, task_for, valid_session


@contextmanager
def fake_time(clock):
    """Run agentic_mixed, and every fake server started inside, on CLOCK."""
    with patch('agentic_mixed.time', clock), patch.object(agentic_fake, 'fake_server', partial(fake_server, clock=clock)):
        yield clock


@contextmanager
def large_monotonic_origin():
    """Real monotonic time as on a host up for 2**23 s (97 days), where
    seconds near the origin keep only about 1e-9 s of float precision."""
    real = time.monotonic
    with patch('time.monotonic', lambda: real() + 2.0 ** 23):
        yield


class ConfirmingTransport(Transport):
    """A Transport that names each streamed request (`fake_stream`) and,
    through READS, confirms to the fake server every chunk it has timed."""

    def __init__(self, url, timeout, reads):
        super().__init__(url, timeout)
        self.reads = reads

    def chunks(self, path, body, deadline, cap):
        if body is None:
            yield from super().chunks(path, body, deadline, cap)
            return
        stream = self.reads.open()
        inner = super().chunks(path, {**body, 'fake_stream': stream}, deadline, cap)
        try:
            for at, chunk in inner:
                self.reads.confirm(stream, len(chunk))
                yield at, chunk
        finally:
            inner.close()


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.groups = load_corpus()

    def test_pins_and_all_nine_tasks(self):
        tasks = [task_for(self.manifest, self.groups['coding'], 'runyard-coding-v1', 1, i) for i in range(120)]
        self.assertEqual(len({(t['focus_path'], t['kind']) for t in tasks}), 9)
        self.assertTrue(all(6000 <= len(t['prompt'].encode()) <= 11000 for t in tasks))
        self.assertEqual(tasks, [task_for(self.manifest, self.groups['coding'], 'runyard-coding-v1', 1, i) for i in range(120)])
        # Independent selection from Runyard's SHA256 key, unaffected by port revision.
        key = hashlib.sha256(b'runyard-coding-v1/C1/0/task-selection-v1').digest()
        self.assertEqual(tasks[0]['focus_path'], list(self.groups['coding'])[key[0] % 3])
        self.assertEqual(tasks[0]['kind'], ['bug fix', 'refactor', 'documentation'][key[1] % 3])
        self.assertNotEqual(tasks[0], task_for(self.manifest, self.groups['coding'], 'different', 1, 0))

    def test_tampered_corpus_refused(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for path in CORPUS.glob('*'):
                if path.is_file():
                    (directory / path.name).write_bytes(path.read_bytes())
            (directory / 'go.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                load_corpus(directory)

    def test_decode_context_pin_and_size(self):
        prompt, digest = decode_context(self.groups, 32768)
        self.assertEqual(len(prompt.encode()), 131303)
        self.assertEqual(digest, hashlib.sha256(prompt.encode()).hexdigest())
        self.assertEqual((prompt, digest), decode_context(self.groups, 32768))
        self.assertIn('internal/bench/runservice.go, read 1', prompt)

    def test_scorer_recovery_and_one_patch(self):
        w = Workspace({'a': 'hello\nunique', 'b': 'other'})
        args = json.dumps({'path': 'a', 'oldString': 'unique', 'newString': 'changed'})
        self.assertTrue(w.call('edit', args).startswith('tool error:'))
        self.assertFalse(valid_session(w, True, None))
        w.call('read', '{"path":"b"}')
        self.assertTrue(w.call('edit', args).startswith('tool error:'))
        w.call('read', '{"path":"a"}')
        self.assertEqual(w.call('edit', args), 'patch applied to a')
        self.assertTrue(valid_session(w, True, None))
        self.assertFalse(valid_session(w, False, None))
        self.assertFalse(valid_session(w, True, 'output token limit'))
        self.assertTrue(w.call('edit', args).startswith('tool error:'))
        self.assertTrue(valid_session(w, True, None))
        self.assertEqual(w.tool_errors, 3)

    def test_noop_and_ambiguous_patch_match_runyard(self):
        w = Workspace({'a': 'same same'})
        w.call('read', '{"path":"a"}')
        self.assertTrue(w.call('edit', '{"path":"a","oldString":"same","newString":"x"}').startswith('tool error:'))
        self.assertEqual(w.call('edit', '{"path":"a","oldString":"same same","newString":"same same"}'), 'patch applied to a')
        self.assertTrue(valid_session(w, True, None))

    def test_bounded_tools_and_paths(self):
        w = Workspace({'a': '\n'.join(str(i) for i in range(250))})
        result = w.call('read', '{"path":"a","limit":999}')
        self.assertIn('offset=201', result)
        self.assertNotIn('201: 200', result)
        for args in ['{"path":"../a"}', '{"path":"/a"}', '{"path":"a","offset":false}', '{"path":"a","offset":999}', 'null', '[]', '{}']:
            self.assertTrue(w.call('read', args).startswith('tool error:'), args)
        w = Workspace({'a': 'é' * 100}, 60)
        self.assertLessEqual(len(w.call('read', '{"path":"a"}').encode()), 60)
        w.call('list_files', '{}')
        self.assertEqual((w.results, w.truncated), (2, 1))


class StreamTests(unittest.TestCase):
    def test_sse_fragments_utf8_multiline_and_comments(self):
        raw = ': heartbeat\r\ndata: {"x":\r\ndata: "é"}\r\n\r\ndata:[DONE]\n\n'.encode()
        values = list(sse_events((i, raw[i:i+1]) for i in range(len(raw))))
        self.assertEqual(json.loads(values[0][1]), {'x': 'é'})
        self.assertEqual(values[1][1], '[DONE]')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            list(sse_events([(0, b'data: {')]))

    def test_fragmented_calls_usage_ttft_and_stalls(self):
        clock = FakeClock()
        started = clock.monotonic()
        events = [
            (.1, {'choices': [{'delta': {'role': 'assistant'}}]}),
            (.2, {'choices': [{'delta': {'reasoning_content': 'think'}}]}),
            (.3, {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call', 'function': {'name': 'read', 'arguments': '{"pa'}}]}}]}),
            (.8, {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'th":"a"}'}}]}, 'finish_reason': 'tool_calls'}]}),
            (.9, {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 12, 'prompt_tokens_details': {'cached_tokens': 50}},
                  'metrics': {'speculative_decoding': {'num_accepted_draft_tokens': 4, 'num_draft_tokens': 5}}})]
        class Client:
            timeout = 10
            def chunks(self, *_):
                for offset, event in events + [(1, None)]:
                    clock.advance(started + offset - clock.monotonic())
                    yield clock.monotonic(), b'data: ' + (json.dumps(event).encode() if event else b'[DONE]') + b'\n\n'
        with patch('agentic_mixed.time', clock):
            row, message = stream_request(Client(), {}, started + 10, stall_threshold_ms=150)
        self.assertIsNone(row['failure'])
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']), {'path': 'a'})
        self.assertEqual(row['output_deltas'], 3)
        self.assertAlmostEqual(row['ttft_ms'], 200, places=6)
        self.assertAlmostEqual(row['elapsed_seconds'], 1, places=9)
        self.assertAlmostEqual(row['inter_delta_summary_ms']['max'], 500, places=3)
        self.assertAlmostEqual(row['decode_tokens_per_second'], 20)
        self.assertEqual(row['accepted_draft_tokens'], 4)
        self.assertEqual(row['client_stalls']['count'], 1)
        self.assertEqual(row['within_run_client_stalls']['count'], 1)
        self.assertEqual(row['holdback_possible_gaps'], 1)
        self.assertNotIn('itl_ms', row)

    def test_ttft_with_a_large_monotonic_origin(self):
        """The controlled clock's assertions hold however long the host has been up."""
        with large_monotonic_origin():
            self.test_fragmented_calls_usage_ttft_and_stalls()

    def test_truncated_and_error_streams_fail_closed(self):
        class Client:
            timeout = 10
            def chunks(self, *_):
                yield time.monotonic(), b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n'
        row, _ = stream_request(Client(), {}, time.monotonic() + 10)
        self.assertIsNotNone(row['failure'])
        self.assertIsNone(row['output_tokens'])

    def test_arbitrary_endpoint_configuration_without_connection(self):
        client = Transport('https://example.invalid:9999/nested')
        self.assertEqual(client.parsed.path, '/nested')
        with self.assertRaises(ValueError):
            Transport('https://user:password@example.invalid')


class MathTests(unittest.TestCase):
    def test_nearest_rank_missing_and_ratios(self):
        self.assertEqual(distribution([1, 2, 3, 4])['p50'], 2)
        self.assertEqual(distribution(range(1, 101))['p95'], 95)
        self.assertIsNone(distribution([])['max'])
        self.assertEqual(ratio([{'a': 0, 'b': 10}], 'a', 'b')['value'], 0)
        self.assertIsNone(ratio([{'a': 2, 'b': 1}], 'a', 'b')['value'])
        missing = ratio([{'a': 1, 'b': 2}, {}], 'a', 'b')
        self.assertEqual((missing['samples'], missing['missing'], missing['value']), (1, 1, None))

    def test_tails_carry_sample_counts(self):
        """A tail percentile is reported with its sample count, and is
        supported only when at least one sample is expected above it."""
        from agentic_metrics import TAIL_SAMPLES, tail
        self.assertEqual(TAIL_SAMPLES, {'p50': 2, 'p95': 20, 'p99': 100})
        few = distribution(range(19))
        self.assertEqual(tail(few, 'p95'), dict(value=few['p95'], samples=19, supported=False))
        self.assertTrue(tail(distribution(range(20)), 'p95')['supported'])
        self.assertFalse(tail(distribution(range(99)), 'p99')['supported'])
        self.assertEqual(tail(distribution([]), 'p50'), dict(value=None, samples=0, supported=False))

    def test_window_turns_from_draining_sessions(self):
        def turn(a, b):
            return dict(started=a, ended=b, elapsed_seconds=b-a, output_deltas=2, output_tokens=10,
                        cached_tokens=2, prompt_tokens=4, accepted_draft_tokens=1, draft_tokens=2, ttft_ms=1, inter_delta_ms=[2])
        def outcome(phase, end, turns):
            return dict(phase=phase, ended=end, elapsed_seconds=end, turns=turns, valid=True, tool_errors=0)
        sessions = [outcome('warmup', 0, [turn(-1, 0)]), outcome('measured', 5, [turn(1, 3)]),
                    outcome('drain', 12, [turn(2, 4), turn(5, 12)])]
        row = summarize(sessions, 0, 10, 2)
        self.assertEqual((row['measured_completed'], row['drain_sessions']), (1, 1))
        self.assertEqual(row['output_tokens_per_second']['value'], 2)
        self.assertEqual(row['peak_http_in_flight'], 2)
        self.assertAlmostEqual(row['average_http_in_flight'], .9)
        sessions[1]['turns'][0]['output_tokens'] = None
        self.assertIsNone(summarize(sessions, 0, 10, 2)['output_tokens_per_second']['value'])

    def test_metrics_prefixes_labels_resets_and_missing(self):
        text = ''.join(f'{prefix}step_execution_seconds_count{{phase="mixed",model_name="a,b"}} 2\n' for prefix in ['chop_', 'vllm:'])
        before = {'text': text}
        after = {'text': text.replace('} 2', '} 5')}
        rows = metrics_delta(before, after)['series']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['delta'] == 3 and r['labels']['phase'] == 'mixed' for r in rows))
        self.assertTrue(all(r['delta'] is None for r in metrics_delta(after, before)['series']))
        self.assertTrue(all(r['delta'] is None for r in metrics_delta({'text': ''}, after)['series']))
        self.assertIsNotNone(metrics_delta({'error': '404'}, after)['unavailable'])
        legacy_prefix = bytes((114, 116, 120)).decode() + "llm_"
        self.assertEqual(parse_metrics(legacy_prefix + "step_execution_seconds_count 2\n"), {})
        with self.assertRaises(ValueError):
            parse_metrics(text + text)


class FakeIntegrationTests(unittest.TestCase):
    def test_arm_identity_does_not_change_comparison_digest(self):
        results = []
        with tempfile.TemporaryDirectory() as name:
            def receipt(check, identity, result, mechanics_only):
                self.assertEqual(check, 'agentic-mixed')
                self.assertTrue(mechanics_only)
                self.assertEqual(identity['model'], 'fake-agentic')
                results.append(result)
                kind = result['receipt_kind'] if result['receipt_kind'] == 'sweep' else f"{result['receipt_kind']}-{len(results)}"
                return Path(name) / (result['arm'] + '-' + kind + '.json')
            # The fake sweep's window is in event time, so its two sessions fit on any host.
            with patch('agentic_mixed.common.startup'), patch('agentic_mixed.common.receipt', side_effect=receipt), \
                    fake_time(FakeClock()), redirect_stdout(io.StringIO()):
                for arm in ['tbt', 'fixed']:
                    self.assertEqual(main(['--dry-run', '--levels', '1', '--seconds', '.5', '--warmup-waves', '0', '--min-samples', '2',
                                           '--no-decode', '--arm', arm, '--fake-prefill-budget', arm, '--label', 'comparison']), 0)
                    summary = (Path(name) / (arm + '-sweep.md')).read_text()
                    self.assertIn('Server step evidence', summary)
                    # Every percentile in the summary carries its sample count.
                    for line in summary.splitlines():
                        if re.match(r'\| 1 \| \d+ \| ', line):
                            cells = line.split(' | ')
                            self.assertRegex(line, r'p95 upper [0-9.]+\*? ms \(n=\d+\)')
                            self.assertEqual(sum(bool(re.search(r'\(n=\d+\)', cell)) for cell in cells), 5, line)
        results = [r for r in results if r['receipt_kind'] == 'sweep']
        self.assertEqual(results[0]['profile_digest'], results[1]['profile_digest'])
        self.assertEqual(results[0]['workload_digest'], results[1]['workload_digest'])
        self.assertNotEqual(results[0]['arm'], results[1]['arm'])

    def test_real_http_sweep_and_decode(self):
        """On the fake server's event clock, with each event confirmed read
        before the next, the window holds the same sessions and the decode
        probe the same interval on any host. run_level's waits and boundary
        scrape still run on real time: this checks session accounting, not
        when the boundary scrape happens."""
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.25', '--warmup-waves', '1', '--min-samples', '4']))
        clock, reads = FakeClock(), Reads()
        with fake_server(groups['coding'], clock=clock, reads=reads) as url, patch('agentic_mixed.time', clock):
            client = ConfirmingTransport(url, 5, reads)
            self.assertEqual(json.loads(client.get('/v1/models'))['data'][0]['id'], 'fake-agentic')
            row = run_level(client, 'fake-agentic', profile, manifest, groups['coding'], 1)
            self.assertIsNone(row['insufficient'])
            # 19 events per session: four end inside the 0.25 s window, the fifth drains.
            self.assertEqual((row['measured_completed'], row['drain_sessions']), (4, 1))
            self.assertEqual(row['warmup_sessions'], 1)
            self.assertEqual(row['invalid_sessions'], 0)
            self.assertTrue(all(s['valid'] for s in row['sessions']))
            self.assertEqual(row['draft_acceptance']['value'], .8)
            # The fake's prefix cache reuses each salted session's earlier turns.
            self.assertTrue(0 < row['cache_hit_rate']['value'] < 1)
            self.assertTrue(row['server_metrics']['series'])
            self.assertTrue(all(r['delta'] >= 0 for r in row['server_metrics']['series'] if r['delta'] is not None))
            probe = run_decode(client, 'fake-agentic', profile, groups, 262144)
            self.assertIsNone(probe['failure'])
            self.assertAlmostEqual(probe['decode_seconds'], EVENT_SECONDS, places=9)
            self.assertGreater(probe['decode_tokens_per_second'], 0)

    def test_too_few_sessions_for_a_p95_are_insufficient(self):
        """By default a level needs as many completed sessions as a p95 over
        them needs (agentic_metrics.TAIL_SAMPLES), and the help says why."""
        from agentic_metrics import TAIL_SAMPLES
        self.assertEqual(parser().parse_args([]).min_samples, TAIL_SAMPLES['p95'])
        self.assertIn('p95', next(a.help for a in parser()._actions if a.dest == 'min_samples'))
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.25', '--warmup-waves', '0']))
        clock, reads = FakeClock(), Reads()
        with fake_server(groups['coding'], clock=clock, reads=reads) as url, patch('agentic_mixed.time', clock):
            row = run_level(ConfirmingTransport(url, 5, reads), 'fake-agentic', profile, manifest, groups['coding'], 1)
        self.assertEqual(row['insufficient'], f"too few completed measured sessions: {row['measured_completed']} of 20")
        self.assertIn('insufficient', ' '.join(row['verdict']['problems']))

    def test_unconfirmed_reads(self):
        """A named stream whose reads are not confirmed in time ends in a
        stream error, and a client that confirms nothing keeps the real
        pacing sleep even on the event clock."""
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args([]))
        clock, reads = FakeClock(), Reads(seconds=.05)
        with fake_server(groups['coding'], clock=clock, reads=reads) as url:
            # Its confirmations go to another Reads, so the server never sees one.
            probe = run_decode(ConfirmingTransport(url, 5, Reads()), 'fake-agentic', profile, groups, 262144)
            self.assertEqual(probe['failure'], 'ValueError')
            self.assertIsNone(probe['decode_tokens_per_second'])
            before, started = clock.monotonic(), time.monotonic()
            row, answer = stream_request(Transport(url, 5), {'messages': [], 'max_tokens': 300}, started + 5)
            self.assertIsNone(row['failure'])
            self.assertEqual(answer['content'], '1 alpha beta gamma delta epsilon zeta eta theta\n')
            # Two deltas and the finish event, each followed by a real sleep and a clock step.
            self.assertGreaterEqual(time.monotonic() - started, 3 * EVENT_SECONDS)
            self.assertAlmostEqual(clock.monotonic() - before, 3 * EVENT_SECONDS, places=9)

    def test_unconfirmed_reads_with_a_large_monotonic_origin(self):
        with large_monotonic_origin():
            self.test_unconfirmed_reads()

    def test_session_cap_and_length_tool_call_failure(self):
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.3', '--warmup-waves', '0', '--max-sessions-per-c', '2']))
        clock, reads = FakeClock(), Reads()
        with fake_server(groups['coding'], clock=clock, reads=reads) as url, patch('agentic_mixed.time', clock):
            row = run_level(ConfirmingTransport(url, 5, reads), 'fake-agentic', profile, manifest, groups['coding'], 1)
            self.assertIn('cap reached', row['insufficient'])
            self.assertEqual(row['measured_completed'], 2)
        response = dict(started=0, ended=1, elapsed_seconds=1, failure=None, finish_reason='length')
        with patch('agentic_mixed.stream_request', return_value=(response, {'tool_calls': [{'id': 'bad'}]})):
            result = session(None, 'fake', profile, manifest, groups['coding'], 1, 0, 'measured')
        self.assertFalse(result['valid'])
        self.assertFalse(result['patch_applied'])
        self.assertEqual(result['failure'], 'output token limit')


class ReplicaTests(unittest.TestCase):
    """Repeated passes are replicas of one workload: each reports its own
    figures, and the receipt and comparison give their spread, never tails
    pooled across replicas."""

    def sweeps(self, *flags, arms=(('tbt', 'tbt'), ('fixed', 'fixed'))):
        """Fake sweeps of each (arm label, fake server mode) in ARMS, by label."""
        receipts = {}
        with tempfile.TemporaryDirectory() as name:
            def receipt(check, identity, result, mechanics_only):
                if result['receipt_kind'] == 'sweep':
                    receipts[result['arm']] = json.loads(json.dumps({**result, 'server': identity}))
                return Path(name) / f"{result['arm']}-{result['receipt_kind']}-{len(result['level_receipts'])}.json"
            with patch('agentic_mixed.common.startup'), patch('agentic_mixed.common.receipt', side_effect=receipt), \
                    fake_time(FakeClock()), redirect_stdout(io.StringIO()):
                for arm, mode in arms:
                    self.assertEqual(main(['--dry-run', '--levels', '1,2', '--seconds', '.4', '--warmup-waves', '0',
                                           '--min-samples', '2', '--no-decode', '--arm', arm, '--fake-prefill-budget', mode,
                                           '--label', 'replicas', *flags]), 0)
        for receipt in receipts.values():
            receipt.update(schema='chopinfer-qualification/v1', check='agentic-mixed', mechanics_only=True)
        return receipts

    def test_replicas_are_separate_passes_with_a_spread(self):
        self.assertEqual(parser().parse_args([]).replicas, 3)
        sweep = self.sweeps('--replicas', '2')['tbt']
        self.assertEqual([(l['concurrency'], l['replica']) for l in sweep['levels']], [(1, 0), (1, 1), (2, 0), (2, 1)])
        first, second = sweep['levels'][:2]
        # Replica 0 keeps the profile seed; replica 1 draws its own tasks and request seeds.
        self.assertEqual((first['seed'], second['seed']), ('runyard-coding-v1', 'runyard-coding-v1/replica-1'))
        self.assertNotEqual([s['prompt_sha256'] for s in first['sessions']], [s['prompt_sha256'] for s in second['sessions']])
        self.assertEqual([s['concurrency'] for s in sweep['replica_spread']], [1, 2])
        spread = sweep['replica_spread'][0]
        self.assertEqual(spread['replicas'], [0, 1])
        figure = spread['figures']['ttft_p95_ms']
        self.assertEqual(figure['values'], [first['ttft_ms']['p95'], second['ttft_ms']['p95']])
        self.assertEqual(figure['samples'], [first['ttft_ms']['samples'], second['ttft_ms']['samples']])
        self.assertEqual((figure['min'], figure['max']), (min(figure['values']), max(figure['values'])))
        throughput = spread['figures']['output_tokens_per_second']['values']
        self.assertEqual(throughput, [level['output_tokens_per_second']['value'] for level in (first, second)])
        summary = markdown(sweep)
        self.assertIn('Replica spread', summary)
        self.assertRegex(summary, r'\| 1 \| 1 \| ')

    def test_comparison_pairs_replicas_and_reports_their_spread(self):
        from agentic_compare import compare
        receipts = self.sweeps('--replicas', '2')
        with patch('agentic_compare.validate_pair'):
            compared = compare(receipts['tbt'], receipts['fixed'])
        self.assertEqual(compared['schema'], 'chopinfer.agentic-mixed-comparison.v2')
        # Of a tbt and fixed pair, fixed is the baseline in either order.
        self.assertEqual(compared['arms'], {'baseline': 'fixed', 'candidate': 'tbt'})
        self.assertEqual([level['concurrency'] for level in compared['levels']], [1, 2])
        level = compared['levels'][0]
        self.assertEqual([pair['replica'] for pair in level['replicas']], [0, 1])
        for pair in level['replicas']:
            # Replica r of one arm repeats replica r of the other.
            self.assertGreater(pair['trajectories']['shared_turns'], 0)
            self.assertEqual(pair['trajectories']['equal_requests'], pair['trajectories']['shared_turns'])
        figure = level['spread']['ttft_p95_ms']
        fixed = [l for l in receipts['fixed']['levels'] if l['concurrency'] == 1]
        tbt = [l for l in receipts['tbt']['levels'] if l['concurrency'] == 1]
        self.assertEqual(figure['baseline']['values'], [l['ttft_ms']['p95'] for l in fixed])
        self.assertEqual(figure['baseline']['samples'], [l['ttft_ms']['samples'] for l in fixed])
        self.assertEqual(figure['candidate_minus_baseline']['values'],
                         [b['ttft_ms']['p95'] - a['ttft_ms']['p95'] for a, b in zip(fixed, tbt)])
        self.assertIn(figure['ranges_overlap'], (True, False))
        self.assertNotIn('pooled', json.dumps(compared))

    def test_any_two_verified_arms_compare_in_order(self):
        """Two arms of one mode with different server settings (a fixed
        budget at another --max-batch-tokens) compare as baseline and
        candidate, in argument order; each label's mode prefix is verified."""
        from agentic_compare import compare, validate_pair
        from agentic_server import verify_arm
        fixed = {'mode': 'fixed', 'errors': [], 'selected_budget_observations': 0}
        tbt = {'mode': 'tbt', 'errors': [], 'selected_budget_observations': 3}
        self.assertTrue(verify_arm('fixed-mbt4096', fixed)['passed'])
        self.assertFalse(verify_arm('fixed-mbt4096', tbt)['passed'])
        self.assertTrue(verify_arm('tbt-150ms', tbt)['passed'])
        self.assertFalse(verify_arm('tbt-150ms', fixed)['passed'])
        receipts = self.sweeps('--replicas', '1', arms=(('fixed', 'fixed'), ('fixed-mbt4096', 'fixed')))
        compared = compare(receipts['fixed'], receipts['fixed-mbt4096'])
        self.assertEqual(compared['arms'], {'baseline': 'fixed', 'candidate': 'fixed-mbt4096'})
        self.assertEqual(set(compared['levels'][0]['replicas'][0]['coverage']), {'baseline', 'candidate'})
        self.assertEqual(compare(receipts['fixed-mbt4096'], receipts['fixed'])['arms']['baseline'], 'fixed-mbt4096')
        same = json.loads(json.dumps(receipts['fixed-mbt4096']))
        same['arm'] = 'fixed'
        with self.assertRaisesRegex(ValueError, 'two different arms'):
            validate_pair(receipts['fixed'], same)
        unlabeled = self.sweeps('--replicas', '1', arms=(('other', 'fixed'),))['other']
        with self.assertRaisesRegex(ValueError, 'arm labels'):
            validate_pair(receipts['fixed'], unlabeled)

    def test_a_fixed_arm_is_the_cross_mode_baseline_in_either_order(self):
        """The baseline of a fixed and tbt pair follows the arms' modes, not
        their exact labels, so suffixed labels never reverse the differences."""
        from agentic_compare import compare
        receipts = self.sweeps('--replicas', '1', arms=(('tbt-150ms', 'tbt'), ('fixed-mbt8192', 'fixed')))
        forward = compare(receipts['tbt-150ms'], receipts['fixed-mbt8192'])
        backward = compare(receipts['fixed-mbt8192'], receipts['tbt-150ms'])
        for compared in (forward, backward):
            self.assertEqual(compared['arms'], {'baseline': 'fixed-mbt8192', 'candidate': 'tbt-150ms'})
        self.assertEqual(forward['levels'], backward['levels'])
        figure = forward['levels'][0]['spread']['output_tokens_per_second']
        self.assertEqual(figure['candidate_minus_baseline']['values'],
                         [receipts['tbt-150ms']['levels'][0]['output_tokens_per_second']['value']
                          - receipts['fixed-mbt8192']['levels'][0]['output_tokens_per_second']['value']])

    def test_v2_pairs_still_compare_and_mixed_versions_are_refused(self):
        """Receipts of the v2 sweep, written by its own code (fixtures/agentic,
        dry runs with per-turn timing lists the comparison never reads
        removed), compare as before; a v2 receipt never pairs with a v3 one."""
        from agentic_compare import compare
        fixtures = CORPUS.parents[2] / 'fixtures' / 'agentic'
        old = {arm: json.loads((fixtures / f'v2-sweep-{arm}.json').read_text()) for arm in ('tbt', 'fixed')}
        self.assertEqual({r['workload_schema'] for r in old.values()}, {'chopinfer.agentic-mixed.v2'})
        for left, right in (('tbt', 'fixed'), ('fixed', 'tbt')):
            compared = compare(old[left], old[right])
            self.assertEqual(compared['schema'], 'chopinfer.agentic-mixed-comparison.v1')
            level = compared['levels'][0]
            self.assertEqual(level['concurrency'], 1)
            self.assertGreater(level['trajectories']['shared_turns'], 0)
            ttft = level['secondary_client_metrics']['ttft_p95_ms']
            self.assertEqual((ttft['fixed'], ttft['tbt']), (old['fixed']['levels'][0]['ttft_ms']['p95'],
                                                            old['tbt']['levels'][0]['ttft_ms']['p95']))
        new = self.sweeps('--replicas', '1', '--levels', '1')
        with self.assertRaisesRegex(ValueError, 'versions'):
            compare(old['tbt'], new['fixed'])
        with self.assertRaisesRegex(ValueError, 'versions'):
            compare(new['tbt'], old['fixed'])

    def test_comparison_refuses_a_missing_replica(self):
        from agentic_compare import validate_pair
        receipts = self.sweeps('--replicas', '2')
        broken = json.loads(json.dumps(receipts['fixed']))
        del broken['levels'][1]
        with self.assertRaisesRegex(ValueError, 'levels'):
            validate_pair(receipts['tbt'], broken)


def source_repository(directory, files):
    """A git repository at DIRECTORY with one commit of FILES {path: text}: (path, commit)."""
    for path, text in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text(text)
    def git(*args):
        return subprocess.run(['git', '-C', str(directory), *args], check=True, capture_output=True, text=True,
                              env={**os.environ, 'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.invalid',
                                   'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@example.invalid'}).stdout
    git('init', '-q')
    git('add', '.')
    git('commit', '-q', '-m', 'fixture')
    return directory, git('rev-parse', 'HEAD').strip()


def rust_source(lines, width):
    return ''.join(f'// line {i:05d} ' + 'x' * width + '\n' for i in range(lines))


class PressureTests(unittest.TestCase):
    """Prefill pressure: the pinned Runyard corpus repeats nine prompts the
    prefix cache absorbs, so a level must say how much prefill it caused, and
    the sweep must be able to defeat the cache and use a larger corpus."""

    def staged(self, directory, count=4):
        files = {f'crates/c{i}/src/lib.rs': rust_source(120 + 60 * i, 40) for i in range(6)}
        files['crates/tiny/src/lib.rs'] = 'fn f() {}\n'
        files['README.md'] = 'not source\n'
        repository, commit = source_repository(directory / 'repo', files)
        out = directory / 'corpus'
        manifest = stage_corpus(repository, commit, out, count=count)
        return repository, commit, out, manifest

    def sweep(self, manifest, files, *flags):
        profile = vars(parser().parse_args(['--levels', '2', '--seconds', '.5', '--warmup-waves', '0', *flags]))
        clock, reads = FakeClock(), Reads()
        with fake_server(files, clock=clock, reads=reads) as url, patch('agentic_mixed.time', clock):
            return run_level(ConfirmingTransport(url, 5, reads), 'fake-agentic', {**profile, 'run_id': 'run'},
                             manifest, files, 2)

    def test_stage_pins_source_outside_the_tree(self):
        with tempfile.TemporaryDirectory() as name:
            repository, commit, out, manifest = self.staged(Path(name))
            loaded, groups = load_corpus(out)
            self.assertEqual(loaded, manifest)
            self.assertEqual((manifest['source_commit'], manifest['revision']), (commit, 'chopinfer-source.v1'))
            self.assertEqual(len(groups['coding']), 4)
            # Rust sources inside the size bounds only, byte for byte.
            for path, text in groups['coding'].items():
                self.assertTrue(path.startswith('crates/c') and path.endswith('.rs'), path)
                self.assertEqual(text, (repository / path).read_text())
                self.assertEqual(set(manifest['tasks'][path]), {'bug fix', 'refactor', 'documentation'})
            runyard, _ = load_corpus()
            self.assertEqual((manifest['tools'], manifest['setup']), (runyard['tools'], runyard['setup']))
            # The same revision stages the same corpus; a used directory and the tree are refused.
            again = stage_corpus(repository, commit, Path(name) / 'again', count=4)
            self.assertEqual(again['corpus_digest'], manifest['corpus_digest'])
            with self.assertRaisesRegex(ValueError, 'not empty'):
                stage_corpus(repository, commit, out, count=4)
            with self.assertRaisesRegex(ValueError, 'outside the repository'):
                stage_corpus(repository, commit, repository / 'staged', count=4)

    def test_cache_salt_defeats_shared_prefixes_and_the_level_reports_prefill(self):
        manifest, groups = load_corpus()
        self.assertEqual(parser().parse_args([]).cache_salt, 'session')
        shared = self.sweep(manifest, groups['coding'], '--cache-salt', 'none')
        salted = self.sweep(manifest, groups['coding'])
        salts = [s['cache_salt'] for s in salted['sessions']]
        self.assertTrue(all(salts) and len(set(salts)) == len(salts))
        self.assertTrue(all(s['cache_salt'] is None for s in shared['sessions']))
        for level in (shared, salted):
            prefill = level['prefill']
            self.assertGreater(prefill['server_prefill_tokens_per_step']['samples'], 0)
            self.assertGreater(prefill['uncached_prompt_tokens']['samples'], 0)
        # Nine shared first prompts hit the cache without a salt; a salted session's first turn misses.
        self.assertGreater(shared['cache_hit_rate']['value'], salted['cache_hit_rate']['value'])
        self.assertGreater(salted['prefill']['server_prefill_tokens_per_step']['value'],
                           shared['prefill']['server_prefill_tokens_per_step']['value'])
        # The salt is not part of the request identity the arms compare.
        self.assertEqual([t['request_sha256'] for t in salted['sessions'][0]['turns']],
                         [t['request_sha256'] for t in shared['sessions'][0]['turns']])

    def test_a_staged_corpus_drives_the_sweep(self):
        with tempfile.TemporaryDirectory() as name:
            _, _, out, _ = self.staged(Path(name))
            manifest, groups = load_corpus(out)
            level = self.sweep(manifest, groups['coding'])
            self.assertGreater(level['measured_completed'], 0)
            self.assertTrue(all(s['valid'] for s in level['sessions']))
            self.assertTrue({s['focus_path'] for s in level['sessions']} <= set(groups['coding']))

    def test_the_tool_result_limit_binds_on_a_staged_corpus(self):
        """Runyard's 16, 32 or 64 KiB per-task limit never binds on its own
        corpus; a staged corpus sets 16 KiB, so a 200-line read meets its
        4 and 8 KiB tiers. Each session counts its truncated results."""
        runyard, groups = load_corpus()
        level = self.sweep(runyard, groups['coding'])
        self.assertEqual(level['tool_results']['truncated'], 0)
        self.assertGreater(level['tool_results']['results'], 0)
        with tempfile.TemporaryDirectory() as name:
            _, _, out, manifest = self.staged(Path(name))
            self.assertEqual(manifest['max_tool_result_bytes'], 16384)
            manifest, groups = load_corpus(out)
            tiers = {task_for(manifest, groups['coding'], 'seed', 1, i)['tool_result_bytes'] for i in range(64)}
            self.assertEqual(tiers, {4096, 8192, 16384})
            level = self.sweep(manifest, groups['coding'])
        counts = level['tool_results']
        self.assertGreater(counts['truncated'], 0)
        measured = [s for s in level['sessions'] if s['phase'] == 'measured']
        self.assertEqual(counts['truncated'], sum(s['truncated_tool_results'] for s in measured))
        self.assertEqual(counts['results'], sum(s['tool_results'] for s in measured))
        self.assertLessEqual(set(counts['by_limit']), {'4096', '8192', '16384'})
        self.assertEqual(sum(tier['truncated'] for tier in counts['by_limit'].values()), counts['truncated'])
        self.assertTrue(all(s['valid'] for s in level['sessions']))

    def test_main_records_the_staged_corpus_identity_and_no_path(self):
        with tempfile.TemporaryDirectory() as name:
            _, commit, out, manifest = self.staged(Path(name))
            receipts = []
            def receipt(check, identity, result, mechanics_only):
                receipts.append(json.dumps({k: v for k, v in result.items() if k != 'level_receipts'}))
                return Path(name) / f'{len(receipts)}.json'
            with patch('agentic_mixed.common.startup'), patch('agentic_mixed.common.receipt', side_effect=receipt), \
                    fake_time(FakeClock()), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--dry-run', '--levels', '1', '--seconds', '.5', '--warmup-waves', '0',
                                       '--no-decode', '--min-samples', '2', '--corpus', str(out)]), 0)
            sweep = json.loads(receipts[-1])
            self.assertEqual((sweep['source_commit'], sweep['corpus_digest']), (commit, manifest['corpus_digest']))
            self.assertTrue(all(name not in r for r in receipts))


if __name__ == '__main__':
    unittest.main()
