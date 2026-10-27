"""Regressions for trace measurement and coding-receipt compatibility."""
import copy
import hashlib
import json
from pathlib import Path
import time
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
        self.assertEqual([row['first_prefix_retained'] for row in turns], [True, True, True])



if __name__ == '__main__':
    unittest.main()
