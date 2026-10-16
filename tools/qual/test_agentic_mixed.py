"""CPU tests for the pinned workload, stream framing and measurement contract."""
import hashlib
import io
from contextlib import contextmanager, redirect_stdout
from functools import partial
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import agentic_fake
from agentic_fake import EVENT_SECONDS, FakeClock, Reads, fake_server
from agentic_metrics import distribution, metrics_delta, parse_metrics, ratio, summarize
from agentic_mixed import Transport, main, parser, run_decode, run_level, session, sse_events, stream_request
from agentic_workload import CORPUS, Workspace, decode_context, load_corpus, task_for, valid_session


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
                return Path(name) / (result['arm'] + '-' + result['receipt_kind'] + '.json')
            # The fake sweep's window is in event time, so its two sessions fit on any host.
            with patch('agentic_mixed.common.startup'), patch('agentic_mixed.common.receipt', side_effect=receipt), \
                    fake_time(FakeClock()), redirect_stdout(io.StringIO()):
                for arm in ['tbt', 'fixed']:
                    self.assertEqual(main(['--dry-run', '--levels', '1', '--seconds', '.5', '--warmup-waves', '0',
                                           '--no-decode', '--arm', arm, '--fake-prefill-budget', arm, '--label', 'comparison']), 0)
                    self.assertIn('Server step evidence', (Path(name) / (arm + '-sweep.md')).read_text())
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
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.25', '--warmup-waves', '1']))
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
            self.assertEqual(row['cache_hit_rate']['value'], .5)
            self.assertTrue(row['server_metrics']['series'])
            self.assertTrue(all(r['delta'] >= 0 for r in row['server_metrics']['series'] if r['delta'] is not None))
            probe = run_decode(client, 'fake-agentic', profile, groups, 262144)
            self.assertIsNone(probe['failure'])
            self.assertAlmostEqual(probe['decode_seconds'], EVENT_SECONDS, places=9)
            self.assertGreater(probe['decode_tokens_per_second'], 0)

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


if __name__ == '__main__':
    unittest.main()
