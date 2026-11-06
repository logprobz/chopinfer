"""Regressions for trace measurement and coding-receipt compatibility."""
import copy
import hashlib
import json
import shlex
from pathlib import Path
import unittest
from unittest.mock import patch

import agentic_metrics as metrics
import agentic_mixed as mixed
import agentic_trace as trace


class TraceReviewTests(unittest.TestCase):
    def test_request_uses_nondefault_burst_and_stall_controls(self):
        class Client:
            timeout = 10
            def chunks(self, *_):
                events = [(0, {'prompt_token_ids': [1]}),
                          (.1, {'choices': [{'delta': {'content': 'a'}, 'token_ids': [2]}]}),
                          (.11, {'choices': [{'delta': {'content': 'b'}, 'token_ids': [3]}]}),
                          (.25, {'choices': [{'delta': {'content': 'c'}, 'token_ids': [4]}]}),
                          (.3, {'choices': [{'finish_reason': 'length'}],
                                'usage': {'prompt_tokens': 1, 'completion_tokens': 3}})]
                for at, event in events:
                    yield at, b'data: ' + json.dumps(event).encode() + b'\n\n'
                yield .4, b'data: [DONE]\n\n'
        row, _, _ = trace.request(Client(), {'max_tokens': 3}, 10, expected=[1],
                                  stall_threshold_ms=50, burst_window_ms=20)
        self.assertIsNone(row['failure'])
        self.assertEqual((row['stall_threshold_ms'], row['burst_window_ms']), (50, 20))
        self.assertEqual(row['burst_event_counts'], [2, 1])
        self.assertEqual([e['at'] for e in row['output_events']], [.1, .11, .25])
        self.assertEqual(row['client_stalls']['eligible_gaps'], 1)
        self.assertAlmostEqual(row['step_gaps'][0]['ms'], 150)
    def test_replay_propagates_controls_to_priming_and_turns(self):
        from agentic_fake import fake_server
        workload = trace.generate(1, 1, 'controls', [32], 0)
        workload['sessions'][0]['turns'][0].update(append_tokens=4, output_tokens=3, think_seconds=0)
        profile = vars(mixed.parser().parse_args([]))
        profile.update(seed='controls', run_id='test', arm='fixed', seconds=2, min_samples=1,
                       session_timeout_seconds=5, stall_threshold_ms=50, burst_window_ms=20)
        with fake_server({}) as url, patch.object(trace, 'request', wraps=trace.request) as calls:
            level = trace.run_level(mixed.Transport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        self.assertEqual(calls.call_count, 2)
        for call in calls.call_args_list:
            self.assertEqual(call.kwargs['stall_threshold_ms'], 50)
            self.assertEqual(call.kwargs['burst_window_ms'], 20)

    def test_analysis_excludes_prefill_and_control_waits(self):
        from agentic_trace_analysis import gaps
        output = [dict(at=at, kinds=['content']) for at in (1, 1.005, 1.15)]
        row = dict(output_events=output, burst_window_ms=20,
                   sse_events=[dict(at=at) for at in (0, 1, 1.005, 1.15, 1.9, 2.1)])
        values = gaps(row, [(0, .99)])['100']
        self.assertEqual(values['eligible_gaps'], 1)
        self.assertEqual(values['count'], 1)
        self.assertAlmostEqual(values['total_gap_ms'], 150)
        self.assertEqual(values['prefill_overlap_ms'], 0)

    def test_legacy_output_timing_and_missing_evidence(self):
        from agentic_trace_analysis import gaps, raw_sse_gaps
        row = dict(started=0, ttft_ms=1000, output_deltas=3,
                   inter_delta_ms=[5, 145], output_event_kinds=[['content']] * 3,
                   burst_window_ms=20, sse_events=[dict(at=at) for at in (0, 1, 1.005, 1.15, 2)])
        self.assertAlmostEqual(gaps(row, [])['100']['total_gap_ms'], 150)
        self.assertGreater(raw_sse_gaps(row, [])['100']['count'], 1)
        del row['output_event_kinds']
        with self.assertRaisesRegex(ValueError, 'output event timing unavailable'):
            gaps(row, [])

    def test_fallback_to_old_prefix_is_not_newest_anchor_survival(self):
        from agentic_fake import fake_server
        workload = trace.generate(1, 3, 'anchor', [64], 0)
        for turn in workload['sessions'][0]['turns']:
            turn.update(append_tokens=4, output_tokens=2, think_seconds=0)
        profile = vars(mixed.parser().parse_args([]))
        profile.update(seed='anchor', run_id='test', arm='fixed', seconds=3, min_samples=1)
        original = trace.request
        cached = iter([0, 10, 30, 10])
        def request(*args, **kwargs):
            row, answer, response = original(*args, **kwargs)
            row['cached_tokens'] = next(cached)
            return row, answer, response
        with fake_server({}) as url, patch.object(trace, 'request', side_effect=request):
            level = trace.run_level(mixed.Transport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        turns = level['sessions'][0]['turns']
        self.assertEqual([row['cached_tokens'] for row in turns], [10, 30, 10])
        self.assertEqual([row['anchor_survived'] for row in turns], [None, None, None])
        self.assertTrue(all(row['anchor_survival_unavailable'] for row in turns))
        # Prompts shorter than a prefill chunk are warm with any cached token.
        self.assertEqual([row['warm_threshold_tokens'] for row in turns], [1, 1, 1])
        self.assertEqual([row['warm'] for row in turns], [True, True, True])

    def test_coding_profile_pairs_with_pre_trace_receipt(self):
        from agentic_compare import compare
        from test_agentic_mixed import ReplicaTests
        receipts = ReplicaTests().sweeps('--replicas', '1',
                    arms=(('fixed-old', 'fixed'), ('fixed-new', 'fixed')))
        legacy = copy.deepcopy(receipts['fixed-old'])
        # The pre-trace coding parser had none of these fields. Its other
        # defaults, profile additions and digest encoding are unchanged.
        for key in ('workload', 'trace', 'trace_history'):
            legacy['profile'].pop(key, None)
        encoded = json.dumps(legacy['profile'], sort_keys=True, separators=(',', ':'))
        legacy['profile_digest'] = hashlib.sha256(encoded.encode()).hexdigest()
        current = receipts['fixed-new']
        self.assertEqual(encoded, json.dumps(current['profile'], sort_keys=True, separators=(',', ':')))
        self.assertEqual(legacy['profile_digest'], current['profile_digest'])
        report = compare(legacy, current)
        self.assertEqual(report['arms'], {'baseline': 'fixed-old', 'candidate': 'fixed-new'})
        for level in legacy['levels']:
            level['client_stalls'].pop('stall_ms_per_1000_gaps', None)
        old_report = compare(legacy, current)
        key = 'client_stall_ms_per_1000_gaps'
        self.assertEqual(report['levels'][0]['spread'][key], old_report['levels'][0]['spread'][key])
        self.assertIn(key, old_report['levels'][0]['replicas'][0]['figures']['baseline'])

    def test_readme_trace_has_feasible_pacing(self):
        text = (Path(__file__).resolve().parents[2] / 'docs/contributing.md').read_text()
        snippet = text.split('python3 -B tools/qual/agentic_trace.py', 1)[1].split('```', 1)[0]
        commands = snippet.replace('\\' + '\n', '').strip().splitlines()
        generator, replay = shlex.split(commands[0]), shlex.split(commands[1])
        def value(words, flag):
            return words[words.index(flag) + 1]
        workload = trace.generate(int(value(generator, '--sessions')), int(value(generator, '--turns')),
                                  contexts=[int(n) for n in value(generator, '--contexts').split(',')],
                                  idle_pages=int(value(generator, '--idle-pages')))
        think_bound = max(sum(turn['think_seconds'] for turn in session['turns'])
                          for session in workload['sessions'])
        self.assertLess(think_bound, float(value(replay, '--seconds')))
        self.assertTrue(all(turn['idle_pages'] == 0 for session in workload['sessions'] for turn in session['turns']))

    def test_generator_rotates_mix_within_each_session(self):
        for session in trace.generate(8, 4)['sessions']:
            self.assertEqual({t['think_seconds'] for t in session['turns']}, {0, 2, 10, 60})
            self.assertEqual({t['output_tokens'] for t in session['turns']}, {128, 256, 550, 1266})

    def test_stall_time_normalizes_by_eligible_gaps(self):
        short = metrics.stall_stats([200, 1], 100)
        long = metrics.stall_stats([200] + [1] * 9, 100)
        self.assertEqual(short['total_stall_ms'], long['total_stall_ms'])
        self.assertEqual(short['stall_ms_per_1000_gaps'], 100000)
        self.assertEqual(long['stall_ms_per_1000_gaps'], 20000)
        self.assertIsNone(metrics.stall_stats([], 100)['stall_ms_per_1000_gaps'])
        from agentic_trace_analysis import gaps
        row = dict(output_events=[dict(at=t, kinds=['content']) for t in (0, .2, .21)])
        value = gaps(row, [])['100']
        self.assertAlmostEqual(value['stall_ms_per_1000_gaps'], 100000)
        self.assertEqual(value['per_1000_gaps'], 500)

    def test_append_ttft_tails_require_supported_samples(self):
        from agentic_trace_analysis import analyze
        def level(count):
            rows = [dict(ended=2, append_class='1-63', ttft_ms=100 + i, output_tokens=1,
                         prompt_tokens=100, cached_tokens=10, sse_events=[], output_events=[],
                         turn_index=i, request_sha256=str(i), requested_idle_pages=1,
                         anchor_survived=True) for i in range(count)]
            return dict(realtime_offset_ns=dict(start=0, end=0), window_start=0, window_end=10,
                        concurrency=1, replica=0, sessions=[dict(turns=rows)], burst_window_ms=2)
        small = analyze(level(19))
        ttft = small['ttft_by_append_class']['1-63']
        self.assertEqual(ttft['samples'], 19)
        self.assertIsNone(ttft['p95'])
        self.assertFalse(ttft['supported']['p95'])
        self.assertTrue(ttft['unavailable']['p95'])
        self.assertEqual(ttft['max'], 118)
        enough = analyze(level(20))['ttft_by_append_class']['1-63']
        self.assertEqual(enough['p95'], 118)
        self.assertTrue(enough['supported']['p95'])
        self.assertIsNone(enough['p99'])
        # Old trace receipts' first-hit booleans are not newest-frontier evidence.
        self.assertEqual(small['anchor_survival'], [None] * 19)
        self.assertEqual(small['first_prefix_retention'], [True] * 19)
        # They predate warmth, so they report no returns.
        self.assertEqual(small['sessions_kept_warm']['unavailable'], 'no trace returns')
        self.assertEqual(small['return_ttft_ms']['samples'], 0)

    def test_sessions_kept_warm_and_return_ttft_per_cell(self):
        def row(warm, ttft):
            return dict(warm=warm, warm_threshold_tokens=100, ttft_ms=ttft)
        turns = [row(True, 10), row(False, 900), row(True, 20), row(True, 30)]
        figures = metrics.summarize_returns(turns)
        self.assertEqual(figures['sessions_kept_warm'], metrics.metric(.75, 4))
        self.assertEqual((figures['return_ttft_ms']['samples'], figures['return_ttft_ms']['p50']), (4, 20))
        # A return without a verdict leaves the share unavailable, not a tie.
        figures = metrics.summarize_returns(turns + [row(None, None)])
        self.assertIsNone(figures['sessions_kept_warm']['value'])
        self.assertEqual(figures['sessions_kept_warm']['missing'], 1)
        # A coding workload's turns are not trace returns.
        figures = metrics.summarize_returns([dict(ttft_ms=5)])
        self.assertEqual(figures['sessions_kept_warm']['unavailable'], 'no trace returns')
        self.assertEqual(figures['return_ttft_ms']['samples'], 0)

    def test_analysis_reports_warmth_return_ttft_rate_and_energy_per_cell(self):
        from agentic_trace_analysis import analyze
        rows = [dict(ended=1 + i / 10, append_class='1-63', ttft_ms=100 + i, output_tokens=10,
                     prompt_tokens=100, cached_tokens=90 if i % 4 else 0, sse_events=[], output_events=[],
                     turn_index=i, request_sha256=str(i), requested_idle_pages=0,
                     warm_threshold_tokens=50, warm=bool(i % 4)) for i in range(20)]
        level = dict(realtime_offset_ns=dict(start=0, end=0), window_start=0, window_end=10, concurrency=8,
                     replica=0, sessions=[dict(turns=rows)], burst_window_ms=2)
        power = [dict(at=at, watts={'0': 300, '1': 300}, limits={'0': 300, '1': 300}, sm_clocks={'0': 1, '1': 1})
                 for at in (0, 10)]
        cell = analyze(level, power=power)
        self.assertEqual(cell['sessions_kept_warm'], metrics.metric(.75, 20))
        self.assertEqual(cell['return_ttft_ms']['p95'], 118)
        self.assertTrue(cell['return_ttft_ms']['supported']['p95'])
        self.assertEqual(cell['output_tokens_per_second'], 20)
        self.assertEqual(cell['energy']['joules'], 6000)
        self.assertEqual(cell['energy']['joules_per_output_token'], 30)
        unpowered = analyze(level)
        self.assertIsNone(unpowered['energy']['joules_per_output_token'])
        self.assertIsNotNone(unpowered['energy']['unavailable'])

    def test_trace_levels_record_warmth_and_return_ttft(self):
        from agentic_fake import fake_server
        workload = trace.generate(2, 2, 'kept', [32], 0)
        for session in workload['sessions']:
            for turn in session['turns']:
                turn.update(append_tokens=4, output_tokens=2, think_seconds=0)
        profile = vars(mixed.parser().parse_args([]))
        profile.update(seed='kept', run_id='test', arm='fixed', seconds=5, min_samples=1, session_timeout_seconds=5)
        with fake_server({}) as url:
            level = trace.run_level(mixed.Transport(url, 2), 'fake-agentic', profile, workload, 1)
        self.assertTrue(level['verdict']['passed'], level['verdict'])
        turns = [t for s in level['sessions'] for t in s['turns']]
        self.assertEqual(level['sessions_kept_warm']['samples'], 4)
        self.assertEqual(level['sessions_kept_warm']['value'], sum(t['warm'] for t in turns) / 4)
        self.assertEqual(level['return_ttft_ms']['samples'], 4)



if __name__ == '__main__':
    unittest.main()
