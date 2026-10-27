"""CPU checks for declarative trace scheduling and served token evidence."""
import copy
from contextlib import redirect_stdout
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
        row = {'sse_events': [{'at': 1}, {'at': 1.2}, {'at': 2.5}]}
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
        from agentic_compare import compare
        from agentic_mixed import main
        workload = trace.generate(1, 1, 'receipts', [32], 0)
        workload['sessions'][0]['turns'][0].update(think_seconds=0, append_tokens=4, output_tokens=2)
        results = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps(workload))
            def receipt(check, identity, result, dry, **_):
                results.append(dict(result, schema='chopinfer-qualification/v1', check=check,
                                    server=identity, mechanics_only=dry))
                return Path(directory) / f'{len(results)}.json'
            with patch('common.startup'), patch('common.receipt', side_effect=receipt), redirect_stdout(io.StringIO()):
                for arm in ('fixed-base', 'fixed-next'):
                    self.assertEqual(main(['--workload', 'trace', '--trace', str(path), '--dry-run',
                                           '--arm', arm, '--label', 'pair', '--levels', '1', '--replicas', '1',
                                           '--min-samples', '1', '--seconds', '2']), 0)
        report = compare(*results)
        self.assertEqual(report['levels'][0]['replicas'][0]['trajectories']['equal_requests'], 1)
        changed = copy.deepcopy(results[1])
        changed['workload_digest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'workload_digest'):
            compare(results[0], changed)


if __name__ == '__main__':
    unittest.main()
