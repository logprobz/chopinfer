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
