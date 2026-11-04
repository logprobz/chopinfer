"""CPU checks for declarative trace scheduling and served token evidence."""
import argparse
import copy
from contextlib import redirect_stdout
import gzip
import io
import json
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from agentic_fake import fake_server
from agentic_mixed import HTTPStatusError, Transport, parser
import agentic_trace as trace


class TraceTests(unittest.TestCase):
    def test_generator_append_range(self):
        for bounds in ((1000, 2000), (1, 1), (32768, 32768)):
            with self.subTest(bounds=bounds):
                workload = trace.generate(4, 100, 'shape', [100000], 0, append_tokens=bounds)
                self.assertEqual(workload, trace.validate(copy.deepcopy(workload)))
                appends = [t['append_tokens'] for s in workload['sessions'] for t in s['turns']]
                self.assertTrue(all(bounds[0] <= tokens <= bounds[1] for tokens in appends))
        appends = [t['append_tokens'] for s in trace.generate(4, 100, 'shape', [100000], 0,
                                                              append_tokens=(1000, 2000))['sessions'] for t in s['turns']]
        self.assertGreater(len(set(appends)), 300)
        self.assertLess(min(appends), 1050)
        self.assertGreater(max(appends), 1950)

    def test_generator_default_append_range(self):
        # The digest of the trace the class mix generated before the range option existed.
        args = (4, 100, 'shape', [100000, 250000], 0)
        self.assertEqual(trace.digest(trace.generate(*args)),
                         '5e95100f9c53d35403bb34b5d196399173c8c57a42ad16d5b6827687436f8258')
        self.assertEqual(trace.generate(*args), trace.generate(*args, append_tokens=None))

    def test_append_range_parser(self):
        for value, bounds in (('1000:2000', (1000, 2000)), ('1:32768', (1, 32768)), ('7:7', (7, 7))):
            self.assertEqual(trace.append_range(value), bounds)
        for value in ('', '1', '1:', ':2', '1:2:3', 'a:2', '1.0:2', '0:2', '-1:2',
                      '2:1', '1:32769', '1_000:2000', '1: 2', '+1:2'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                trace.append_range(value)

    def test_generator_distribution_and_no_content(self):
        workload = trace.generate(4, 100, 'shape', [100000, 250000], 0)
        rows = [t for s in workload['sessions'] for t in s['turns']]
        self.assertEqual({k: sum(trace.append_class(t['append_tokens']) == k for t in rows)
                          for k in trace.CLASSES}, dict(zip(trace.CLASSES, [124, 160, 48, 28, 40])))
        self.assertEqual(sum(t['output_tokens'] for t in rows) / len(rows), 550)
        self.assertEqual({t['think_seconds'] for t in rows}, {0, 2, 10, 60})
        self.assertEqual(workload, trace.validate(copy.deepcopy(workload)))
        bad = copy.deepcopy(workload)
        bad['sessions'][0]['content'] = 'private input'
        with self.assertRaises(ValueError):
            trace.validate(bad)

    def test_cyclic_returns_park_no_session_and_validate(self):
        workload = trace.generate(4, 8, 'cyclic', [100000], 4096, returns='cyclic')
        self.assertEqual(workload['returns'], 'cyclic')
        self.assertNotIn('prefill_chunk_tokens', workload)
        self.assertEqual(workload, trace.validate(copy.deepcopy(workload)))
        self.assertTrue(all(t['idle_pages'] == 0 for s in workload['sessions'] for t in s['turns']))
        # Apart from parking, the turns are the default generator's.
        default = trace.generate(4, 8, 'cyclic', [100000], 0)
        self.assertEqual(workload['sessions'], default['sessions'])
        self.assertNotIn('returns', default)
        chunked = trace.generate(1, 1, 'chunk', [64], 0, prefill_chunk_tokens=8192)
        self.assertEqual(trace.validate(copy.deepcopy(chunked))['prefill_chunk_tokens'], 8192)
        for key, value in (('returns', 'random'), ('returns', None), ('prefill_chunk_tokens', 0),
                           ('prefill_chunk_tokens', 65537), ('prefill_chunk_tokens', 2048.0), ('order', 'cyclic')):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(workload)
                bad[key] = value
                with self.assertRaises(ValueError):
                    trace.validate(bad)
        with self.assertRaises(ValueError):
            trace.generate(1, 1, 'x', [64], 0, returns='random')

    def test_cyclic_admission_takes_the_least_advanced_session_first(self):
        states = [dict(name=name, next=index) for name, index in (('a', 2), ('b', 1), ('c', 1), ('d', 0))]
        self.assertEqual([s['name'] for s in trace.admission_order({'returns': 'cyclic'}, states)],
                         ['d', 'b', 'c', 'a'])
        self.assertIs(trace.admission_order({}, states), states)

    def replay_order(self, returns):
        workload = trace.generate(3, 2, 'order', [64], 0, returns=returns)
        for session in workload['sessions']:
            for turn in session['turns']:
                turn.update(think_seconds=0, append_tokens=20, output_tokens=4)
        profile = vars(parser().parse_args([]))
        profile.update(seconds=10, min_samples=3, trace_history='canonical', cache_salt='session',
                       arm='fixed', seed='order', run_id='test', session_timeout_seconds=10)
        with fake_server({}) as url:
            level = trace.run_level(Transport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        rows = sorted((t['started'], s['index'], t['turn_index']) for s in level['sessions'] for t in s['turns'])
        return [(session, turn) for _, session, turn in rows]

    def test_cyclic_returns_visit_every_session_before_any_returns_again(self):
        self.assertEqual(self.replay_order('cyclic'), [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])
        # The declared order keeps serving the first ready session.
        self.assertEqual(self.replay_order(None), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def warm_replay(self, cached):
        """A one-session, three-turn replay whose server reports cached(turn,
        previous) tokens: `previous` is the session's previous prompt."""
        workload = trace.generate(1, 3, 'warm', [64], 0, prefill_chunk_tokens=8)
        # Appends longer than the priming turn, so every prompt outgrows the previous one.
        for turn in workload['sessions'][0]['turns']:
            turn.update(think_seconds=0, append_tokens=64, output_tokens=2)
        profile = vars(parser().parse_args([]))
        profile.update(seed='warm', run_id='test', arm='fixed', seconds=5, min_samples=1, session_timeout_seconds=5)
        original, calls = trace.request, []

        def request(client, body, deadline, expected=None, previous=None, **kwargs):
            row, answer, response = original(client, body, deadline, expected, previous, **kwargs)
            if previous is not None:
                # The first return's previous prompt is its primer's.
                row['cached_tokens'] = cached(len(calls), calls[-1] if calls else len(previous))
                calls.append(row['prompt_tokens'])
            return row, answer, response

        with fake_server({}) as url, patch.object(trace, 'request', side_effect=request):
            level = trace.run_level(Transport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        return level['sessions'][0]['turns']

    def test_a_zero_first_hit_is_a_cold_return_not_a_missing_baseline(self):
        # Before, a zero first hit left the baseline unset, so every later
        # return read None.
        turns = self.warm_replay(lambda turn, previous: (0, previous, previous - 8)[turn])
        self.assertEqual([row['warm'] for row in turns], [False, True, True])
        self.assertTrue(all(row['warm_threshold_tokens'] == row['previous_prompt_tokens'] - 8 for row in turns))
        self.assertNotIn('first_prefix_retained', turns[0])

    def test_an_evicted_primer_is_recorded_as_a_miss(self):
        # The primer's anchors are gone; only a short shared prefix hits.
        turns = self.warm_replay(lambda turn, previous: (previous - 9, previous - 9, 1)[turn])
        self.assertEqual([row['warm'] for row in turns], [False, False, False])
        warm = self.warm_replay(lambda turn, previous: previous - 8)
        self.assertEqual([row['warm'] for row in warm], [True, True, True])
        # The threshold comes from the workload alone: both arms judge alike.
        self.assertEqual([row['warm_threshold_tokens'] for row in turns],
                         [row['warm_threshold_tokens'] for row in warm])

    def test_warm_threshold_is_one_chunk_short_and_never_below_one_token(self):
        self.assertEqual(trace.warm_threshold({}, 100000), 100000 - trace.PREFILL_CHUNK_TOKENS)
        self.assertEqual(trace.warm_threshold({'prefill_chunk_tokens': 8192}, 100000), 100000 - 8192)
        self.assertEqual(trace.warm_threshold({}, 2048), 1)
        self.assertEqual(trace.warm_threshold({}, 10), 1)

    def test_http_refusal_retains_bounded_body(self):
        with fake_server({}) as url:
            with self.assertRaises(HTTPStatusError) as caught:
                Transport(url, 2).get('/absent')
        self.assertEqual(caught.exception.status, 404)
        self.assertIn(b'404', caught.exception.response_body)
        self.assertLessEqual(len(caught.exception.response_body), 4096)

    def test_dependencies_are_strict_and_bounded(self):
        workload = trace.generate(2, 2, 'schedule', [64], 0)
        for value in [False, -1, float('nan')]:
            bad = copy.deepcopy(workload)
            bad['sessions'][0]['turns'][0]['think_seconds'] = value
            with self.assertRaises(ValueError):
                trace.validate(bad)
        workload['sessions'][0]['turns'][1]['after'] = 1
        with self.assertRaises(ValueError):
            trace.validate(workload)

    def test_fake_http_trace_lcp_and_cap(self):
        workload = trace.generate(3, 2, 'http', [64], 0)
        for session in workload['sessions']:
            for turn in session['turns']:
                turn.update(think_seconds=0, append_tokens=20, output_tokens=8)
        profile = vars(parser().parse_args([]))
        profile.update(seconds=5, min_samples=3, trace_history='generated', cache_salt='group',
                       arm='fixed', seed='http', run_id='test', session_timeout_seconds=5)
        with fake_server({}) as url:
            level = trace.run_level(Transport(url, 2), 'fake-agentic', profile, workload, 2)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        self.assertEqual(level['measured_completed'], 3)
        self.assertLessEqual(level['peak_http_in_flight'], 2)
        self.assertEqual(len(level['priming']), 1)
        for session in level['sessions']:
            first, second = session['turns']
            self.assertGreaterEqual(second['started'], first['ended'])
            self.assertIsNotNone(second['response_lcp_tokens'])
            self.assertTrue(all('at' in event for event in second['sse_events']))
            self.assertEqual(second['token_ids'], 8)

    def test_priming_has_an_answerable_turn_and_uses_template_kwargs(self):
        preserved = []
        class CheckedTransport(Transport):
            def chunks(self, path, body, deadline, cap):
                if body and 'messages' in body:
                    if not any(m['role'] == 'user' for m in body['messages']):
                        raise ValueError('chat has no turn to answer')
                    if body.get('chat_template_kwargs', {}).get('preserve_thinking'):
                        preserved.append(body)
                    if 'preserve_thinking' in body:
                        raise ValueError('preserve_thinking belongs in chat_template_kwargs')
                yield from super().chunks(path, body, deadline, cap)
        workload = trace.generate(1, 2, 'wire', [64], 0)
        for turn in workload['sessions'][0]['turns']:
            turn.update(think_seconds=0, append_tokens=20, output_tokens=8)
        profile = vars(parser().parse_args([]))
        profile.update(seconds=5, min_samples=1, trace_history='generated', cache_salt='group',
                       arm='fixed', seed='wire', run_id='test', session_timeout_seconds=5)
        with fake_server({}) as url:
            level = trace.run_level(CheckedTransport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        second = level['sessions'][0]['turns'][1]
        self.assertIsNotNone(second['thinking_lcp_loss_tokens'])
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]['messages'][-1]['role'], 'user')

    def test_scheduler_releases_slots_during_think_and_waits_for_pressure(self):
        # Eligibility uses completed predecessor time and traffic from other sessions.
        turn = dict(think_seconds=2, idle_pages=10)
        self.assertFalse(trace.ready(turn, 3, 4, 20, 5))
        self.assertFalse(trace.ready(turn, 3, 5, 14, 5))
        self.assertTrue(trace.ready(turn, 3, 5, 15, 5))

    def test_timeout_and_lcp_mismatch_fail_closed(self):
        class Client:
            timeout = .01
            def chunks(self, *_):
                raise TimeoutError('fixture')
                yield
        with self.assertRaises(TimeoutError):
            trace.tokenize(Client(), {'model': 'fake', 'prompt': 'x'}, time.monotonic() + 1)
        row, _, _ = trace.request(Client(), {'max_tokens': 1}, time.monotonic() + 1, expected=[1])
        self.assertTrue(row['timed_out'])
        self.assertEqual(row['failure'], 'TimeoutError')
        with self.assertRaisesRegex(ValueError, 'prompt'):
            trace.check_prompt([1, 2], [1, 3], 2)
        self.assertEqual(trace.lcp([1, 2, 3], [1, 2, 4]), 2)

    def test_analysis_overlap_and_missing_energy(self):
        from agentic_trace_analysis import gaps
        row = {'output_events': [dict(at=at, kinds=['content']) for at in (1, 1.2, 2.5)]}
        result = gaps(row, [(1.1, 1.3), (1.2, 1.4)])
        self.assertEqual(result['100']['count'], 2)
        self.assertAlmostEqual(result['100']['total_gap_ms'], 1500)
        self.assertAlmostEqual(result['100']['prefill_overlap_ms'], 300)
        self.assertAlmostEqual(result['100']['excess_ms'], 1300)

    def test_parked_session_does_not_hold_slot_and_unmet_pressure_fails(self):
        workload = trace.generate(2, 1, 'park', [32], 0)
        for session in workload['sessions']:
            session['turns'][0].update(think_seconds=0, append_tokens=4, output_tokens=2)
        workload['sessions'][0]['turns'][0].update(think_seconds=.1, idle_pages=1)
        profile = vars(parser().parse_args([]))
        profile.update(seconds=.4, min_samples=2, trace_history='generated', cache_salt='group',
                       arm='fixed', seed='park', run_id='test')
        with fake_server({}) as url:
            level = trace.run_level(Transport(url, 1), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        parked, other = level['sessions']
        self.assertGreaterEqual(parked['turns'][0]['started'], other['turns'][0]['ended'])
        self.assertGreaterEqual(parked['turns'][0]['intervening_token_pages'], 1)
        workload['sessions'][0]['turns'][0]['idle_pages'] = 100000
        with fake_server({}) as url:
            level = trace.run_level(Transport(url, 1), 'fake-agentic', profile, workload, 1)
        self.assertFalse(level['verdict']['passed'])
        self.assertIsNotNone(level['insufficient'])

    def test_trace_receipts_compare_and_bind_declaration(self):
        """Each run writes one compressed receipt and its summary; the
        comparator and the offline analysis read the files themselves."""
        import common
        from agentic_compare import compare, main as compare_main
        from agentic_mixed import main
        from agentic_trace_analysis import main as analysis_main
        workload = trace.generate(1, 1, 'receipts', [32], 0)
        workload['sessions'][0]['turns'][0].update(think_seconds=0, append_tokens=4, output_tokens=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps(workload))
            receipts = Path(directory) / 'receipts'
            with patch('common.startup'), patch.object(common, 'RECEIPTS', receipts), redirect_stdout(io.StringIO()):
                for arm in ('fixed-base', 'fixed-next'):
                    self.assertEqual(main(['--workload', 'trace', '--trace', str(path), '--dry-run',
                                           '--arm', arm, '--label', 'pair', '--levels', '1', '--replicas', '1',
                                           '--min-samples', '1', '--seconds', '2']), 0)
            written = sorted(receipts.glob('*.json.gz'))
            self.assertEqual(len(written), 2)
            self.assertEqual(sorted(p.name for p in receipts.iterdir()),
                             sorted([p.name for p in written] + [common.receipt_summary(p).name for p in written]))
            results = [common.read_receipt(p) for p in written]
            for raw, value in zip((p.read_bytes() for p in written), results):
                self.assertEqual(raw[4:8], bytes(4))
                self.assertEqual(gzip.decompress(raw).decode(),
                                 json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
                self.assertEqual((value['schema'], value['workload_schema'], value['mechanics_only']),
                                 ('chopinfer-qualification/v1', 'chopinfer.agentic-mixed.v3', True))
            with redirect_stdout(io.StringIO()) as compared:
                self.assertEqual(compare_main([str(p) for p in written]), 0)
            self.assertEqual(json.loads(compared.getvalue()), compare(*results))
            with redirect_stdout(io.StringIO()) as analyzed:
                analysis_main([str(written[0])])
            self.assertEqual(len(json.loads(analyzed.getvalue())), 1)
        report = compare(*results)
        self.assertEqual(report['levels'][0]['replicas'][0]['trajectories']['equal_requests'], 1)
        changed = copy.deepcopy(results[1])
        changed['workload_digest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'workload_digest'):
            compare(results[0], changed)


class ReceiptFormTests(unittest.TestCase):
    def setUp(self):
        import common
        self.common = common
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.value = dict(levels=[], passed=True, label='caf\u00e9')

    def write(self, compressed):
        with patch.object(self.common, 'RECEIPTS', self.root), redirect_stdout(io.StringIO()):
            return self.common.receipt('agentic-mixed', {'model': 'M'}, self.value, True, compressed=compressed)

    def test_compressed_and_plain_receipts_hold_the_same_value(self):
        compressed, plain = self.write(True), self.write(False)
        self.assertTrue(compressed.name.endswith('-agentic-mixed.json.gz'))
        self.assertTrue(plain.name.endswith('-agentic-mixed.json'))
        raw = compressed.read_bytes()
        # One gzip member with no stored name and a zero modification time.
        self.assertEqual((raw[:3], raw[3] & 0x08, raw[4:8]), (b'\x1f\x8b\x08', 0, bytes(4)))
        value, other = self.common.read_receipt(compressed), self.common.read_receipt(plain)
        self.assertEqual(other, json.loads(plain.read_text()))
        # The two writes differ only in their stamps.
        self.assertEqual({k: v for k, v in value.items() if k != 'at'}, {k: v for k, v in other.items() if k != 'at'})
        self.assertTrue(compressed.name.startswith(value['at']))
        self.assertEqual(value['label'], 'caf\u00e9')
        for path in (compressed, plain):
            self.assertEqual(self.common.receipt_summary(path), path.parent / (path.name.split('.')[0] + '.md'))

    def test_historical_plain_fixtures_still_read(self):
        fixture = Path(__file__).resolve().parents[2] / 'fixtures' / 'agentic' / 'v2-sweep-tbt.json'
        self.assertEqual(self.common.read_receipt(fixture), json.loads(fixture.read_text()))

    def test_truncated_or_corrupt_compressed_receipts_are_value_errors(self):
        raw = self.write(True).read_bytes()
        cases = dict(truncated=raw[:len(raw) // 2], unterminated=raw[:-8], corrupt=raw[:10] + bytes(len(raw) - 10),
                     not_json=gzip.compress(b'{"levels": ['))
        for name, data in cases.items():
            with self.subTest(name):
                path = self.root / f'{name}.json.gz'
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.common.read_receipt(path)

if __name__ == '__main__':
    unittest.main()
