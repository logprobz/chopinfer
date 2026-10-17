"""Regression checks for independent mixed-benchmark review findings."""
import hashlib
import json
from pathlib import Path
import re
import time
import unittest
from unittest.mock import patch

import agentic_mixed
from agentic_fake import FakeClock
from agentic_mixed import Markup, parser, session, stream_request
from agentic_workload import Workspace, load_corpus
from test_agentic_markup import frontends, stop_texts, undeclared_ending
from test_agentic_mixed import fake_time


class ReviewRegressions(unittest.TestCase):
    def test_speculative_tokens_in_one_read_are_one_burst(self):
        start = time.monotonic()
        class Client:
            timeout = 10
            def chunks(self, *_):
                for step in range(4):
                    event = {'choices': [{'delta': {'content': 'x'}}]}
                    chunk = (b'data: ' + json.dumps(event).encode() + b'\n\n') * 3
                    yield start + .02 * (step + 1), chunk
                yield start + .09, b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"completion_tokens":12}}\n\ndata: [DONE]\n\n'
        row, _ = stream_request(Client(), {}, start + 10)
        self.assertIn('step_gap_ms', row)
        self.assertAlmostEqual(row['step_gap_ms']['p50'], 20, places=3)
        self.assertEqual(row['burst_event_counts'], [3, 3, 3, 3])
        self.assertNotIn('itl_ms', row)

    def test_deep_tool_arguments_are_errors_not_runner_crashes(self):
        workspace = Workspace({'a': 'hello'})
        result = workspace.call('read', '[' * 10000)
        self.assertTrue(result.startswith('tool error:'))
        self.assertEqual(workspace.tool_errors, 1)

    def test_unexpected_turn_exception_preserves_failed_session(self):
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args([]))
        with patch('agentic_mixed.stream_request', side_effect=RuntimeError('secret should not be recorded')):
            trace = session(None, 'fake', profile, manifest, groups['coding'], 1, 0, 'measured')
        self.assertFalse(trace['valid'])
        self.assertEqual(trace['failure'], 'runner error: RuntimeError')
        self.assertNotIn('secret', json.dumps(trace))

class MeasurementReviewTests(unittest.TestCase):
    def test_floor_percentiles_match_runyard(self):
        from agentic_metrics import distribution
        self.assertEqual(distribution([1, 2, 3, 4, 5])['p50'], 2)
        self.assertEqual(distribution(range(1, 31))['p95'], 28)

    def test_window_groups_nearby_reads_and_marks_tool_transitions(self):
        from agentic_metrics import burst_metrics
        events = [dict(at=at, kinds=[kind]) for at, kind in
                  [(0, 'reasoning'), (.001, 'reasoning'), (.020, 'reasoning'),
                   (.150, 'tool_args:0'), (.175, 'tool_args:0'), (.300, 'tool_args:1')]]
        row = burst_metrics(events, 2, 100)
        self.assertEqual(row['burst_event_counts'], [2, 1, 1, 1, 1])
        self.assertEqual(row['holdback_possible_gaps'], 2)
        self.assertEqual(row['client_stalls']['count'], 2)
        self.assertEqual(row['within_run_client_stalls']['count'], 0)
        self.assertEqual(row['client_stalls']['per_1000_gaps'], 500)
        self.assertAlmostEqual(row['client_stalls']['total_stall_ms'], 255)
        mixed = burst_metrics([{'at': 0, 'kinds': ['reasoning', 'content']},
                               {'at': .2, 'kinds': ['reasoning', 'content']}], 2, 100)
        self.assertTrue(mixed['step_gaps'][0]['holdback_possible'])

    def test_tpot_includes_terminal_delay_and_preserves_status(self):
        class Client:
            timeout = 10
            def chunks(self, *_):
                yield .02, b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                yield .05, b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}],"usage":{"completion_tokens":4}}\n\ndata: [DONE]\n\n'
        with patch('agentic_mixed.time.monotonic', side_effect=[0, .2]):
            row, _ = stream_request(Client(), {}, 10)
        self.assertAlmostEqual(row['tpot_ms'], 60)
        self.assertEqual(row['status'], 200)
        from agentic_mixed import HTTPStatusError
        with patch.object(Client, 'chunks', side_effect=HTTPStatusError(429)):
            row, _ = stream_request(Client(), {}, time.monotonic() + 10)
        self.assertEqual(row['status'], 429)
        self.assertIsNotNone(row['failure'])

    def test_error_timeout_and_reason_counts(self):
        from agentic_metrics import summarize
        base = dict(started=1, ended=2, elapsed_seconds=1, ttft_ms=None, output_deltas=0,
                    inter_delta_ms=[], failure='TimeoutError', timed_out=True, status=None)
        sessions = [dict(phase='measured', ended=3, elapsed_seconds=3, turns=[base], valid=False,
                         read_before_edit=False, patch_applied=False, finished=False, tool_errors=1, failure='TimeoutError'),
                    dict(phase='measured', ended=4, elapsed_seconds=4, turns=[], valid=False, tool_errors=0,
                         runner_error='RuntimeError', failure='runner error: RuntimeError')]
        row = summarize(sessions, 0, 10, 2)
        self.assertEqual((row['errors'], row['timeouts'], row['invalid_sessions']), (2, 1, 2))
        self.assertEqual(row['failure_reasons'], {'runner error': 1, 'timeout': 1})

    def test_real_render_names_and_histogram_stall_bounds(self):
        from agentic_metrics import metrics_delta, histogram_reports
        # Values and labels from runtime-engine/tests/metrics_contract.rs:225-235,
        # :719-740 (compatibility.rs render contract), extended with count/buckets.
        after = '''chop_step_gpu_seconds_count{model_name="fixture",engine="0"} 2
chop_step_gpu_seconds_sum{model_name="fixture",engine="0"} 0.00004
chop_prefill_budget_selected_tokens_count{model_name="fixture",engine="0"} 3
chop_prefill_budget_selected_tokens_sum{model_name="fixture",engine="0"} 5120
chop_prefill_budget_decode_estimate_seconds{model_name="fixture",engine="0",rows_le="1"} 0.004096
chop_prefill_budget_decode_estimate_samples{model_name="fixture",engine="0",rows_le="1"} 2
vllm:inter_token_latency_seconds_count{model_name="fixture",engine="0"} 10
vllm:inter_token_latency_seconds_sum{model_name="fixture",engine="0"} 0.8
vllm:inter_token_latency_seconds_bucket{model_name="fixture",engine="0",le="0.1"} 8
vllm:inter_token_latency_seconds_bucket{model_name="fixture",engine="0",le="+Inf"} 10
chop_decode_future_seconds_count 1
chop_decode_future_seconds_sum 0.2
'''
        before = '\n'.join(line.rsplit(' ', 1)[0] + ' 0' for line in after.splitlines())
        delta = metrics_delta({'text': before}, {'text': after})
        self.assertEqual(len(delta['series']), 12)
        self.assertIn('chop_prefill_budget_decode_estimate_seconds', [r['name'] for r in delta['series']])
        report = next(r for r in histogram_reports(delta, 100) if r['name'].startswith('vllm:inter_token'))
        self.assertEqual(report['stall_per_1000_lower'], 200)
        self.assertEqual(report['stall_per_1000_upper'], 200)
        report = next(r for r in histogram_reports(delta, 50) if r['name'].startswith('vllm:inter_token'))
        self.assertEqual((report['stall_per_1000_lower'], report['stall_per_1000_upper']), (200, 1000))

    def test_modes_are_independent_of_arm_and_absence_is_unknown(self):
        from agentic_server import mode_evidence, verify_arm
        before = {'text': 'chop_prefill_budget_selected_tokens_count 0\n'}
        after = {'text': 'chop_prefill_budget_selected_tokens_count 4\n'}
        # Both tbt and decode-aware select budgets, so the counter alone
        # establishes neither.
        evidence = mode_evidence(before, after, 'M')
        self.assertIsNone(evidence['mode'])
        self.assertFalse(verify_arm('tbt', evidence)['passed'])
        self.assertFalse(verify_arm('fixed', evidence)['passed'])
        unknown = mode_evidence({'text': ''}, {'text': ''}, 'M')
        self.assertIsNone(unknown['mode'])
        self.assertFalse(verify_arm('fixed', unknown)['passed'])
        conflict = mode_evidence(before, after, 'M', {'mode': 'fixed'})
        self.assertIsNone(conflict['mode'])
        self.assertIn('selected-budget observations under the fixed budget', conflict['errors'])
        fixed = mode_evidence({'text': ''}, {'text': ''}, 'M', {'mode': 'fixed'})
        self.assertTrue(verify_arm('fixed', fixed)['passed'])
        idle_tbt = mode_evidence({'text': ''}, {'text': ''}, 'M', {'mode': 'tbt'})
        checked = verify_arm('tbt', idle_tbt)
        self.assertTrue(checked['passed'])
        self.assertTrue(checked['warnings'])
        self.assertFalse(checked['policy_exercised'])

    def test_decode_aware_never_passes_as_tbt(self):
        """The runtime's tbt and decode-aware budgets are separate policies;
        each arm needs evidence specific to its own."""
        from agentic_server import mode_evidence, verify_arm
        selected = 'chop_prefill_budget_selected_tokens_count{model_name="M"} %d\n'
        decode_aware = 'chop_prefill_budget_decode_estimate_seconds{model_name="M",rows_le="1"} 0.004\n'
        tbt = ('chop_mixed_budget_samples{model_name="M"} 12\n'
               'chop_mixed_step_actual_over_predicted_count{model_name="M"} %d\n')
        def evidence(before, after, startup=None):
            return mode_evidence({'text': before}, {'text': after}, 'M', startup, startup)
        cases = [
            ('decode-aware startup', evidence(selected % 0, selected % 5, {'mode': 'decode-aware'}), 'decode-aware'),
            ('decode-aware metrics', evidence(selected % 0 + decode_aware, selected % 5 + decode_aware), 'decode-aware'),
            ('tbt metrics', evidence(selected % 0 + tbt % 0, selected % 5 + tbt % 3), 'tbt'),
            ('tbt startup', evidence(selected % 0, selected % 5, {'mode': 'tbt'}), 'tbt'),
        ]
        for name, observed, mode in cases:
            with self.subTest(name):
                self.assertEqual(observed['mode'], mode)
                for arm in ('tbt', 'tbt-150ms', 'decode-aware', 'fixed'):
                    self.assertEqual(verify_arm(arm, observed)['passed'], arm.split('-1')[0] == mode, arm)
        self.assertEqual(mode_evidence({'text': ''}, {'text': ''}, 'M', {'mode': 'decode-aware'})['mode'], 'decode-aware')
        mixed = evidence(selected % 0 + decode_aware, selected % 5 + tbt % 3)
        self.assertIsNone(mixed['mode'])
        self.assertIn('conflicting mode evidence', mixed['errors'])
        # A selected budget cannot come from the fixed policy.
        fixed = evidence(selected % 0, selected % 5, {'mode': 'fixed'})
        self.assertIsNone(fixed['mode'])
        self.assertFalse(verify_arm('fixed', fixed)['passed'])

    def test_request_seed_and_slot_assignment(self):
        import threading
        from agentic_mixed import run_level, request_seed
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '2', '--seconds', '.03', '--warmup-waves', '0',
                                            '--max-sessions-per-c', '2']))
        calls = {}
        lock = threading.Lock()
        # Sessions take no clock time, so both slots finish their two before the window ends.
        clock = FakeClock()
        def drive(client, model, profile, manifest, files, c, index, phase):
            with lock:
                calls.setdefault(index % c, []).append(index)
            if index % c == 0:
                time.sleep(.002)
            at = clock.monotonic()
            return dict(index=index, phase=phase, started=at, ended=at, elapsed_seconds=0,
                        turns=[], valid=True, tool_errors=0)
        class Client:
            def scrape(self):
                return {'text': ''}
        with patch('agentic_mixed.session', side_effect=drive), patch('agentic_mixed.time', clock):
            run_level(Client(), 'M', profile, manifest, groups['coding'], 2)
        self.assertEqual(calls, {0: [0, 2], 1: [1, 3]})
        self.assertEqual(request_seed('seed', 2, 'coding-1', 0), request_seed('seed', 2, 'coding-1', 0))
        self.assertNotEqual(request_seed('seed', 2, 'coding-1', 0), request_seed('seed', 2, 'coding-1', 1))

    def test_interrupt_during_final_scrape_preserves_current_level(self):
        from agentic_mixed import run_level
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.02', '--warmup-waves', '0',
                                            '--max-sessions-per-c', '2']))
        clock = FakeClock()
        def completed(*args):
            now = clock.monotonic()
            return dict(index=args[5], phase='measured', started=now, ended=now, elapsed_seconds=0,
                        turns=[], valid=True, tool_errors=0)
        class Client:
            count = 0
            def scrape(self):
                self.count += 1
                if self.count == 3:
                    raise KeyboardInterrupt()
                return {'text': ''}
        checkpoints = []
        # Sessions take no clock time, so both run before the window ends.
        with patch('agentic_mixed.session', side_effect=completed), patch('agentic_mixed.time', clock):
            result = run_level(Client(), 'M', profile, manifest, groups['coding'], 1, on_partial=checkpoints.append)
        self.assertEqual(result['aborted'], 'KeyboardInterrupt')
        self.assertEqual(len(result['sessions']), 2)
        self.assertEqual(len(checkpoints[0]['sessions']), 2)

    def test_drain_counters_are_separate_from_window(self):
        from agentic_mixed import run_level
        manifest, groups = load_corpus()
        profile = vars(parser().parse_args(['--levels', '1', '--seconds', '.02', '--warmup-waves', '0']))
        clock = FakeClock()
        def slow_session(*args):
            at = clock.monotonic()
            clock.advance(.06)
            return dict(index=0, phase='measured', started=at, ended=clock.monotonic(), elapsed_seconds=.06,
                        turns=[], valid=True, tool_errors=0)
        class Client:
            count = 0
            def scrape(self):
                self.count += 1
                return {'text': f'chop_step_execution_seconds_count{{phase="mixed"}} {self.count}\n'}
        # The one session starts inside the 0.02 s window and ends 0.06 s later on the clock.
        with patch('agentic_mixed.session', side_effect=slow_session), patch('agentic_mixed.time', clock):
            result = run_level(Client(), 'M', profile, manifest, groups['coding'], 1)
        self.assertEqual(result['server_metrics']['series'][0]['delta'], 1)
        self.assertEqual(result['drain_server_metrics']['series'][0]['delta'], 1)
        self.assertGreater(result['drain_seconds'], 0)
        self.assertEqual(result['drain_sessions'], 1)

class DurabilityAndComparisonTests(unittest.TestCase):
    def test_level_checkpoint_survives_unexpected_later_failure(self):
        import io
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        import common
        import agentic_mixed
        actual = agentic_mixed.run_level
        def fail_second(*args, **kwargs):
            if args[5] == 2:
                raise RuntimeError('do not persist this arbitrary detail')
            return actual(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(common, 'RECEIPTS', Path(directory)), patch.object(agentic_mixed, 'run_level', side_effect=fail_second), redirect_stdout(io.StringIO()):
                code = agentic_mixed.main(['--dry-run', '--levels', '1,2', '--seconds', '.3', '--warmup-waves', '0', '--no-decode'])
            receipts = [json.loads(p.read_text()) for p in Path(directory).glob('*.json')]
            self.assertEqual(code, 1)
            # One checkpoint per replica of the completed level.
            self.assertEqual([r['receipt_kind'] for r in receipts].count('level'), 3)
            partial = next(r for r in receipts if r['receipt_kind'] == 'partial')
            self.assertFalse(partial['passed'])
            self.assertEqual(partial['aborted'], 'runner error: RuntimeError')
            self.assertEqual(partial['levels'][0]['concurrency'], 1)
            self.assertNotIn('arbitrary detail', json.dumps(partial))

    def test_ctrl_c_writes_partial_without_overwriting_completed_level(self):
        import os
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, 'CHOPINFER_QUAL_RECEIPTS': directory}
            proc = subprocess.Popen([sys.executable, '-B', '-c',
                                     'import os, runpy, signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler); '
                                     'sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(sys.argv[0])); '
                                     'runpy.run_path(sys.argv[0], run_name="__main__")',
                                     str(Path(__file__).with_name('agentic_mixed.py')),
                                     '--dry-run', '--levels', '1,2', '--seconds', '1', '--warmup-waves', '0', '--no-decode'],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                until = time.monotonic() + 20
                completed_path = None
                while time.monotonic() < until:
                    for path in Path(directory).glob('*.json'):
                        try:
                            receipt = json.loads(path.read_text())
                        except ValueError:
                            continue
                        if receipt['receipt_kind'] == 'level':
                            completed_path = path
                            break
                    if completed_path:
                        break
                    if proc.poll() is not None:
                        stdout, stderr = proc.communicate()
                        self.fail('fake sweep exited before checkpoint: ' + (stdout + stderr).decode())
                    time.sleep(.02)
                self.assertIsNotNone(completed_path, 'level checkpoint was never written')
                original = completed_path.read_bytes()
                proc.send_signal(signal.SIGINT)
                stdout, stderr = proc.communicate(timeout=15)
                self.assertEqual(proc.returncode, 130, (stdout + stderr).decode())
                self.assertEqual(completed_path.read_bytes(), original)
                receipts = [json.loads(p.read_text()) for p in Path(directory).glob('*.json')]
                partials = [r for r in receipts if r['receipt_kind'] == 'partial']
                self.assertTrue(partials)
                self.assertTrue(all(not r['passed'] and r['aborted'] == 'KeyboardInterrupt' for r in partials))
                self.assertTrue(any(r['levels'] and r['levels'][0]['concurrency'] == 1 for r in partials))
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()

    def test_comparator_refuses_mislabeled_incompatible_and_fake_real_pairs(self):
        import copy
        import io
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        import common
        from agentic_mixed import main
        from agentic_compare import compare
        with tempfile.TemporaryDirectory() as directory:
            # The fake sweep's window is in event time, so its sessions fit on any host.
            with patch.object(common, 'RECEIPTS', Path(directory)), fake_time(FakeClock()), redirect_stdout(io.StringIO()):
                for arm in ('tbt', 'fixed'):
                    self.assertEqual(main(['--dry-run', '--levels', '1', '--seconds', '.4', '--warmup-waves', '0', '--min-samples', '2',
                                           '--no-decode', '--arm', arm, '--fake-prefill-budget', arm, '--label', 'review']), 0)
            receipts = [json.loads(p.read_text()) for p in Path(directory).glob('*.json')]
            a, b = [r for r in receipts if r['receipt_kind'] == 'sweep']
            self.assertTrue(all(r['host_load']['rule'] == common.HOST_LOAD_RULE for r in (a, b)))
            compared = compare(a, b)
            self.assertEqual(compared['host_load']['timing'],
                             'evidence' if a['host_load']['idle'] and b['host_load']['idle'] else 'informational')
            replica = compared['levels'][0]['replicas'][0]
            self.assertTrue(replica['primary_server_step_evidence'])
            # Each compared figure states its samples in both arms.
            for figure in replica['secondary_client_metrics'].values():
                for arm in ('baseline', 'candidate'):
                    self.assertIsInstance(figure[arm]['samples'], int)
            for name in ('ttft_p95_ms', 'client_step_gap_p95_ms'):
                figure = replica['secondary_client_metrics'][name]
                self.assertEqual(set(figure['baseline']), {'value', 'samples', 'supported'})
            for pair in replica['primary_server_step_evidence']:
                self.assertEqual(pair['samples'], {'baseline': pair['baseline']['count'],
                                                   'candidate': pair['candidate']['count']})
            trajectory = replica['trajectories']
            self.assertGreater(trajectory['shared_turns'], 0)
            self.assertEqual(trajectory['equal_requests'], trajectory['shared_turns'])
            self.assertEqual(trajectory['equal_answers'], trajectory['shared_turns'])
            variants = []
            for key, value in [('mechanics_only', False), ('workload_digest', 'changed'), ('label', 'other')]:
                bad = copy.deepcopy(b)
                bad[key] = value
                variants.append(bad)
            for key, value in [('seed', 'different'), ('levels', [2])]:
                bad = copy.deepcopy(b)
                bad['profile'][key] = value
                variants.append(bad)
            bad = copy.deepcopy(b)
            bad['server']['model'] = 'other'
            variants.append(bad)
            bad = copy.deepcopy(b)
            bad['levels'][0]['prefill_budget']['mode'] = 'unknown'
            variants.append(bad)
            # Both arms' turns must be re-parsed by the same frontend and re-parser.
            native = dict(sha256='a' * 64, tool_grammar='synthetic-lines-v1', reparser_sha256='b' * 64)
            a, b = ({**receipt, 'frontend': native} for receipt in (a, b))
            compare(a, b)
            for key in native:
                bad = copy.deepcopy(b)
                bad['frontend'][key] = 'changed'
                variants.append(bad)
            for bad in variants:
                with self.assertRaises(ValueError):
                    compare(a, bad)
            swapped_a, swapped_b = copy.deepcopy(a), copy.deepcopy(b)
            swapped_a['arm'], swapped_b['arm'] = b['arm'], a['arm']
            with self.assertRaises(ValueError):
                compare(swapped_a, swapped_b)

    def test_fake_mode_is_not_derived_from_arm_label(self):
        import tempfile
        from pathlib import Path
        import io
        from contextlib import redirect_stdout
        import common
        from agentic_mixed import main
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(common, 'RECEIPTS', Path(directory)), redirect_stdout(io.StringIO()):
                code = main(['--dry-run', '--levels', '1', '--seconds', '.3', '--warmup-waves', '0',
                             '--no-decode', '--arm', 'tbt', '--fake-prefill-budget', 'fixed'])
            self.assertEqual(code, 1)
            receipt = next(json.loads(p.read_text()) for p in Path(directory).glob('*.json') if json.loads(p.read_text())['receipt_kind'] == 'sweep')
            self.assertFalse(receipt['levels'][0]['arm_verification']['passed'])
            self.assertEqual(receipt['levels'][0]['prefill_budget']['mode'], 'fixed')


EOS, THINK_START, THINK_END = 248046, 248068, 248069
# Prompt tails: Qwen3.8's generation prompt opens the thinking block, MiMo's
# leaves it to the model, and a thinking-disabled prompt closes it.
QWEN_PROMPT = [248067, EOS, 198, 248045, 74455, 198, THINK_START, 198]
MIMO_PROMPT = [248067, EOS, 248045, 74455, 198]
NO_THINKING_PROMPT = QWEN_PROMPT + [271, THINK_END, 271]
# Decoded text of every token ID the scripted streams carry; `piece` adds
# synthetic IDs from SYNTHETIC on.
SYNTHETIC = 400000
VOCAB = {1: 'Use the tool.', 2: 'a', 3: 'b', 5: 'Done', 27: '<', 198: '\n', 271: '\n\n', 1076: '=edit>',
         1628: 'function', 2817: '=read>', 3833: 'One edit', 8977: ' applied.', 12345: ' Summary of the change.',
         21731: 'invoke', 74455: 'assistant', 248045: '<|im_start|>', 248058: '<tool_call>', 248059: '</tool_call>',
         248067: '</tool_response>', THINK_START: '<think>', THINK_END: '</think>', EOS: '<|im_end|>'}
ROOT = Path(__file__).resolve().parents[2]


# The recorded replay of RecordedOutcomes; its model's declared frontend
# gives every scripted stream here its markup.
OUTCOMES = ROOT / 'fixtures/qwen38/agentic-outcomes.json'


def served_markup():
    """The raw re-parse markup of the replay model's declared frontend."""
    return Markup(json.loads((ROOT / f'models/{OUTCOMES.parent.name}/compiler/src/frontend.json').read_text()))


def chop_events(tokens, finish, ids=True, usage=None, prompt=QWEN_PROMPT):
    """A Chopinfer stream: each token's IDs in their own chunk, then its delta."""
    events = [{'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                            **({'token_ids': []} if ids else {})}],
               **({'prompt_token_ids': prompt} if ids else {})}]
    for token, delta in tokens:
        if ids:
            events.append({'choices': [{'index': 0, 'delta': {}, 'token_ids': [token]}]})
        if delta:
            events.append({'choices': [{'index': 0, 'delta': delta, **({'token_ids': []} if ids else {})}]})
    events.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})
    events.append({'choices': [], 'usage': {'prompt_tokens': 10,
                                            'completion_tokens': len(tokens) if usage is None else usage}})
    return events + ['[DONE]']


def vllm_events(chunks, finish, end, prompt=QWEN_PROMPT):
    """A vLLM stream: each chunk's token IDs ride on the chunk with its delta;
    `end` IDs arrive with the finish reason."""
    events = [{'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'token_ids': []}], 'prompt_token_ids': prompt}]
    for ids, delta in chunks:
        events.append({'choices': [{'index': 0, 'delta': delta or {}, 'token_ids': ids}]})
    events.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish, 'token_ids': end}]})
    count = sum(len(ids) for ids, _ in chunks) + len(end)
    events.append({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': count}})
    return events + ['[DONE]']


def piece(text):
    """A synthetic token ID that decodes as TEXT."""
    found = next((i for i, t in VOCAB.items() if i >= SYNTHETIC and t == text), None)
    if found is None:
        found = SYNTHETIC + sum(i >= SYNTHETIC for i in VOCAB)
        VOCAB[found] = text
    return found


def region(name, arguments):
    """The raw text of one call to NAME with JSON ARGUMENTS (none when they
    do not parse), as Qwen3.8 writes it inside its tool region, each value in
    the framing the active function-parameter markup declares (one newline
    on each side when none is active)."""
    try:
        values = json.loads(arguments)
    except ValueError:
        values = {}
    markup = agentic_mixed.MARKUP
    lead, trail = (markup.value_lead, markup.value_trail) if getattr(markup, 'grammar', None) == \
        'xml-function-parameters-v1' else ('\n', '\n')
    return f'<function={name}>\n' + ''.join(
        f'<parameter={k}>{lead}{v if isinstance(v, str) else json.dumps(v)}{trail}</parameter>\n'
        for k, v in (values if isinstance(values, dict) else {}).items()) + '</function>\n'


def call_turn(name, arguments, ids=True, extra=()):
    """A tool-call turn whose raw tokens spell the call it streams; EXTRA
    adds raw regions (name, arguments) the stream does not deliver."""
    tokens = [(1, {'reasoning': 'Use the tool.'}), (THINK_END, None), (271, None), (248058, None),
              (piece('\n' + region(name, arguments)), {'tool_calls': [{'index': 0, 'id': f'call-{name}', 'type': 'function',
                                                                      'function': {'name': name, 'arguments': ''}}]}),
              (248059, {'tool_calls': [{'index': 0, 'function': {'arguments': arguments}}]})]
    for other in extra:
        tokens += [(198, None), (248058, None), (piece('\n' + region(*other)), None), (248059, None)]
    return chop_events(tokens + [(EOS, None)], 'tool_calls', ids)


def vocab_text(ids):
    return ''.join(VOCAB[i] for i in ids)


class ScriptedServer:
    """A transport that answers each chat request with the next scripted
    stream and /detokenize with `decode` (or fails it)."""

    timeout = 10

    def __init__(self, turns, decode=vocab_text):
        self.turns, self.decode, self.bodies, self.decoded = list(turns), decode, [], []
        self.status = None

    def chunks(self, path, body, deadline, cap):
        if path == '/detokenize':
            self.decoded.append(body['tokens'])
            if not self.decode:
                self.status = 404
                raise RuntimeError('HTTP 404')
            yield time.monotonic(), json.dumps({'prompt': self.decode(body['tokens'])}).encode()
            return
        self.bodies.append(body)
        self.status = 200
        for event in self.turns.pop(0):
            data = event if isinstance(event, str) else json.dumps(event)
            yield time.monotonic(), f'data: {data}\n\n'.encode()

    def last_status(self):
        return self.status


class LevelVerdict(unittest.TestCase):
    """Invalid sessions count as model outcomes only when the raw text of the
    last turn, decoded from its generated token IDs after the prompt's
    assistant header and re-parsed, is reasoning and an end token, or
    reasoning closed by a tool region without a function header; the streamed
    parsed deltas never decide."""

    @classmethod
    def setUpClass(cls):
        from agentic_workload import task_for
        cls.enterClassContext(patch.object(agentic_mixed, 'MARKUP', served_markup()))
        cls.manifest, groups = load_corpus()
        cls.files = groups['coding']
        cls.task = task_for(cls.manifest, cls.files, 'runyard-coding-v1', 1, 0)
        content = cls.files[cls.task['focus_path']]
        cls.span = next(line for line in content.splitlines() if len(line) > 12 and content.count(line) == 1)

    def run_session(self, final, phase='measured', decode=vocab_text, opening=None, ids=True):
        path = self.task['focus_path']
        turns = (opening or [call_turn('read', json.dumps({'path': path}), ids)]) + [
            call_turn('edit', json.dumps({'path': path, 'oldString': self.span, 'newString': self.span + ' '}), ids),
            final]
        server = ScriptedServer(turns, decode)
        profile = vars(parser().parse_args(['--temperature', '0']))
        trace = session(server, 'model', profile, self.manifest, self.files, 1, 0, phase)
        self.assertTrue(all(body.get('return_token_ids') for body in server.bodies))
        return trace, server

    def level(self, sessions, **fields):
        row = dict(sessions=sessions, aborted=None, insufficient=None, errors=0, timeouts=0,
                   arm_verification={'passed': True})
        row.update(fields)
        return row

    def verdict(self, trace):
        from agentic_metrics import level_verdict
        valid = dict(trace, valid=True)
        return level_verdict(self.level([valid] * 40 + [trace]))

    def ended_in_thinking(self, prompt=QWEN_PROMPT):
        return chop_events([(3833, {'reasoning': 'One edit'}), (8977, {'reasoning': ' applied.'}), (EOS, None)], 'stop',
                           prompt=prompt)

    def assert_outcome(self, trace, outcome):
        from agentic_metrics import model_outcome
        self.assertFalse(trace['valid'])
        self.assertEqual(model_outcome(trace), (outcome, None))
        verdict = self.verdict(trace)
        self.assertTrue(verdict['passed'], verdict['problems'])
        self.assertEqual(verdict['model_outcomes'], {outcome: 1})

    def assert_hard(self, trace, why):
        from agentic_metrics import model_outcome
        outcome, reason = model_outcome(trace)
        self.assertIsNone(outcome)
        self.assertEqual(reason, why)
        verdict = self.verdict(trace)
        self.assertFalse(verdict['passed'])
        self.assertEqual(verdict['model_outcome_count'], 0)

    def test_reasoning_and_end_token_is_a_model_outcome_in_both_stream_formats(self):
        from agentic_metrics import ENDED_IN_THINKING
        grouped = vllm_events([([3833], {'reasoning': 'One edit'}), ([8977], {'reasoning': ' applied.'})], 'stop', [EOS])
        for final in (self.ended_in_thinking(), grouped):
            trace, server = self.run_session(final)
            self.assertEqual(len(server.decoded), 6)
            self.assertEqual(server.decoded[-2:], [QWEN_PROMPT, [3833, 8977, EOS]])
            self.assertEqual([turn['raw_class'] for turn in trace['turns'][:-1]], ['calls agree'] * 2)
            self.assertEqual(trace['turns'][-1]['raw_class'], 'reasoning only')
            self.assert_outcome(trace, ENDED_IN_THINKING)
            self.assertNotIn('One edit', json.dumps(trace))

    def test_a_tool_region_without_a_function_header_is_a_malformed_call(self):
        from agentic_metrics import MALFORMED_CALL
        final = chop_events([(1, {'reasoning': 'Use the tool.'}), (THINK_END, None), (271, None), (248058, None),
                             (198, None), (27, None), (27, None), (21731, None), (248059, None), (EOS, None)], 'stop')
        trace, _ = self.run_session(final)
        self.assertEqual(trace['turns'][-1]['raw_class'], 'unparsed tool region')
        self.assert_outcome(trace, MALFORMED_CALL)

    def test_a_turn_ending_in_any_declared_stop_token_is_a_model_outcome(self):
        """The scripted turns are function-parameter calls, so every declared
        frontend in that grammar reads them; an undeclared ending stays
        hard."""
        from agentic_metrics import ENDED_IN_THINKING
        declared = [(model, frontend) for model, frontend in frontends()
                    if Markup(frontend).grammar == 'xml-function-parameters-v1']
        self.assertTrue(declared)
        for model, frontend in declared:
            markup = Markup(frontend)
            with patch.object(agentic_mixed, 'MARKUP', markup):
                for stop in stop_texts(frontend):
                    with self.subTest(model=model, stop=stop):
                        final = chop_events([(3833, {'reasoning': 'One edit'}), (8977, {'reasoning': ' applied.'}),
                                             (piece(stop), None)], 'stop')
                        trace, _ = self.run_session(final)
                        self.assertEqual(trace['turns'][-1]['raw_class'], 'reasoning only')
                        self.assert_outcome(trace, ENDED_IN_THINKING)
                ending = undeclared_ending(frontend, markup)
                with self.subTest(model=model, ending=ending):
                    final = chop_events([(3833, {'reasoning': 'One edit'}), (8977, {'reasoning': ' applied.'}),
                                         (piece(ending), None)], 'stop')
                    trace, _ = self.run_session(final)
                    self.assert_hard(trace, 'raw last turn: no end token')

    def test_the_prompt_decides_whether_generated_text_is_reasoning(self):
        from agentic_metrics import ENDED_IN_THINKING
        # A MiMo-style prompt leaves the thinking block to the model.
        opened = chop_events([(THINK_START, None), (3833, {'reasoning': 'One edit'}), (EOS, None)], 'stop',
                             prompt=MIMO_PROMPT)
        trace, _ = self.run_session(opened)
        self.assert_outcome(trace, ENDED_IN_THINKING)
        # Without a reasoning open, the same text is an answer the parser lost.
        for prompt in (MIMO_PROMPT, NO_THINKING_PROMPT):
            with self.subTest(prompt=prompt):
                trace, _ = self.run_session(self.ended_in_thinking(prompt))
                self.assert_hard(trace, 'raw last turn: thinking not open')
        trace, _ = self.run_session(self.ended_in_thinking([198, 3833]))
        self.assert_hard(trace, 'raw last turn: no assistant header')
        trace, _ = self.run_session(self.ended_in_thinking([]))
        self.assert_hard(trace, 'raw last turn: unavailable')

    def test_reviewer_parser_loss_streams_are_hard(self):
        # Confirm review, replay 1: visible reasoning, then one silent token
        # that is not the end token; finish stop, matching usage, [DONE].
        silent_tail = chop_events([(3833, {'reasoning': 'One edit'}), (8977, {'reasoning': ' applied.'}), (12345, None)],
                                  'stop')
        trace, _ = self.run_session(silent_tail)
        self.assert_hard(trace, 'raw last turn: no end token')
        # Confirm review, replay 2: one vLLM chunk carries the reasoning delta
        # and the IDs of the reasoning close and a dropped answer; only the end
        # token arrives with the finish reason.
        grouped = vllm_events([([3833, THINK_END, 12345], {'reasoning': 'One edit'})], 'stop', [EOS])
        trace, server = self.run_session(grouped)
        self.assertEqual(server.decoded[-1], [3833, THINK_END, 12345, EOS])
        self.assert_hard(trace, 'raw last turn: output after reasoning')
        # Tool-call grammar loss: a well-formed call the parser dropped.
        lost = chop_events([(1, {'reasoning': 'Use the tool.'}), (THINK_END, None), (271, None), (248058, None),
                            (27, None), (1628, None), (1076, None), (198, None), (248059, None), (EOS, None)], 'stop')
        trace, _ = self.run_session(lost)
        self.assert_hard(trace, 'raw last turn: output after reasoning')
        # A stream truncated before [DONE] fails its request.
        trace, _ = self.run_session(self.ended_in_thinking()[:-1])
        self.assertEqual(trace['failure'], 'ValueError')
        self.assert_hard(trace, 'session failure')

    def test_any_answer_call_or_markup_in_the_raw_turn_is_hard(self):
        cases = {
            # The parser dropped the reasoning close and the answer.
            'dropped answer': (chop_events([(5, {'reasoning': 'Done'}), (THINK_END, None), (12345, None), (EOS, None)],
                                           'stop'), 'output after reasoning'),
            # The answer streamed as reasoning.
            'answer as reasoning': (chop_events([(5, {'reasoning': 'Done'}), (THINK_END, None),
                                                 (12345, {'reasoning': ' Summary of the change.'}), (EOS, None)], 'stop'),
                                    'output after reasoning'),
            # Text around an unparsed tool region.
            'text and region': (chop_events([(5, {'reasoning': 'Done'}), (THINK_END, None), (12345, None),
                                             (248058, None), (27, None), (248059, None), (EOS, None)], 'stop'),
                                'output after reasoning'),
            # Two unparsed tool regions.
            'two regions': (chop_events([(5, {'reasoning': 'Done'}), (THINK_END, None), (248058, None), (27, None),
                                         (248059, None), (248058, None), (27, None), (248059, None), (EOS, None)],
                                        'stop'), 'output after reasoning'),
            # A call inside the reasoning, dropped.
            'call in reasoning': (chop_events([(1, {'reasoning': 'Use the tool.'}), (248058, None), (27, None),
                                               (1628, None), (2817, None), (248059, None), (EOS, None)], 'stop'),
                                  'call markup'),
            # Turn markup or a second reasoning open in the generation.
            'turn markup': (chop_events([(2, {'reasoning': 'a'}), (248045, None), (EOS, None)], 'stop'),
                            'turn markup'),
            'second open': (chop_events([(2, {'reasoning': 'a'}), (THINK_START, None), (EOS, None)], 'stop'),
                            'turn markup'),
            'two end tokens': (chop_events([(2, {'reasoning': 'a'}), (EOS, None), (EOS, None)], 'stop'),
                               'turn markup'),
            # An end token alone.
            'empty': (chop_events([(EOS, None)], 'stop'), 'empty'),
        }
        for name, (final, raw) in cases.items():
            with self.subTest(name):
                trace, _ = self.run_session(final)
                self.assert_hard(trace, f'raw last turn: {raw}')
        trace, _ = self.run_session(self.ended_in_thinking(), decode=None)
        self.assert_hard(trace, 'raw turn 0: unavailable')
        repeated = self.ended_in_thinking()
        repeated[1]['prompt_token_ids'] = QWEN_PROMPT
        trace, _ = self.run_session(repeated)
        self.assert_hard(trace, 'raw last turn: unavailable')

    def test_incomplete_or_abnormal_streams_are_vetoed(self):
        cases = {
            'lost call': (chop_events([(1, {'reasoning': 'Edit'}), (248058, None), (EOS, None)], 'tool_calls'),
                          'turn 2 finished tool_calls with 0 parsed calls'),
            'missing IDs': (chop_events([(3833, {'reasoning': 'One edit'}), (EOS, None)], 'stop', usage=3),
                            'turn 2 streamed 2 token IDs for 3 tokens'),
            'no IDs': (chop_events([(3833, {'reasoning': 'One edit'}), (EOS, None)], 'stop', ids=False),
                       'turn 2 has no token evidence'),
        }
        for name, (final, why) in cases.items():
            with self.subTest(name):
                trace, _ = self.run_session(final)
                self.assert_hard(trace, why)
        streams = {'no [DONE]': self.ended_in_thinking()[:-1],
                   'length': chop_events([(3833, {'reasoning': 'One edit'}), (EOS, None)], 'length'),
                   'no finish': chop_events([(3833, {'reasoning': 'One edit'})], None)}
        for name, final in streams.items():
            with self.subTest(name):
                trace, _ = self.run_session(final)
                self.assertNotEqual(trace['failure'], 'session did not finish a valid read, patch and final response')
                verdict = self.verdict(trace)
                self.assertFalse(verdict['passed'])
                self.assertEqual(verdict['model_outcome_count'], 0)

    def test_tool_argument_failures_and_missing_reads_are_hard(self):
        path = self.task['focus_path']
        opening = [call_turn('read', '{"path": '), call_turn('read', json.dumps({'path': path}))]
        trace, _ = self.run_session(self.ended_in_thinking(), opening=opening)
        self.assertEqual(trace['argument_errors'], 1)
        self.assert_hard(trace, 'tool arguments did not parse')
        trace, _ = self.run_session(self.ended_in_thinking(),
                                    opening=[call_turn('search', json.dumps({'query': 'def'}))])
        self.assert_hard(trace, 'no code read')

    def test_limit_counts_drained_sessions_and_ignores_warmup(self):
        from agentic_metrics import level_verdict, model_outcome_limit
        from agentic_mixed import level_passed
        self.assertEqual([model_outcome_limit(n) for n in (0, 1, 33, 34, 55, 100, 101)], [0, 1, 1, 2, 2, 3, 4])
        outcome, _ = self.run_session(self.ended_in_thinking(), phase='drain')
        valid = dict(outcome, valid=True, phase='measured', failure=None)
        broken, _ = self.run_session(chop_events([(3833, {'reasoning': 'x'}), (EOS, None)], 'length'), phase='warmup')
        verdict = level_verdict(self.level([valid] * 52 + [outcome] * 2 + [broken]))
        self.assertTrue(verdict['passed'], verdict['problems'])
        self.assertEqual((verdict['measured_sessions'], verdict['model_outcome_count'], verdict['model_outcome_limit']),
                         (54, 2, 2))
        over = self.level([valid] * 51 + [outcome] * 3)
        self.assertFalse(level_passed(over))
        self.assertIn('3 model outcomes above the limit of 2', level_verdict(over)['problems'][0])

    def test_level_failures_fail_however_few(self):
        from agentic_metrics import level_verdict
        trace, _ = self.run_session(self.ended_in_thinking())
        healthy = [dict(trace, valid=True, failure=None)] * 99
        runner = dict(trace, valid=False, failure='runner error: RuntimeError', runner_error='RuntimeError')
        self.assertEqual(level_verdict(self.level(healthy + [runner]))['hard_invalid'], {'runner error': 1})
        for fields, problem in ((dict(errors=1), '1 errors'), (dict(timeouts=2), '2 timeouts'),
                                (dict(aborted='KeyboardInterrupt'), 'aborted: KeyboardInterrupt'),
                                (dict(insufficient='too few completed measured sessions'),
                                 'insufficient: too few completed measured sessions'),
                                (dict(arm_verification={'passed': False, 'reason': 'mode unknown'}),
                                 'arm verification failed: mode unknown')):
            self.assertEqual(level_verdict(self.level(healthy, **fields))['problems'], [problem])


    # A final-turn model outcome counts only when every earlier turn's raw
    # text agrees with the reasoning, content and calls parsed from it.

    def test_an_earlier_call_the_parser_dropped_is_hard(self):
        # Confirm review P1: the read turn's raw tokens hold a second tool
        # region the stream never delivered; the last turn ends in thinking.
        path = self.task['focus_path']
        opening = [call_turn('read', json.dumps({'path': path}), extra=[('search', json.dumps({'query': 'def'}))])]
        trace, server = self.run_session(self.ended_in_thinking(), opening=opening)
        self.assertEqual(trace['turns'][0]['parsed_calls'], 1)
        self.assertEqual(len(server.decoded), 6)
        self.assert_hard(trace, 'raw turn 0: calls differ')

    def test_earlier_turns_must_agree_in_reasoning_content_names_and_values(self):
        path = self.task['focus_path']
        read = json.dumps({'path': path, 'offset': 1, 'limit': 20})
        trace, _ = self.run_session(self.ended_in_thinking(), opening=[call_turn('read', read)])
        self.assertEqual(trace['turns'][0]['raw_class'], 'calls agree')
        def delta(turn, field):
            return next(e['choices'][0]['delta'] for e in turn if isinstance(e, dict) and e['choices']
                        and field in json.dumps(e['choices'][0].get('delta')))

        cases = {
            'value': (lambda t: delta(t, 'arguments": "{')['tool_calls'][0]['function'].update(
                arguments=json.dumps({'path': path, 'offset': 2, 'limit': 20})), 'calls differ'),
            'name': (lambda t: delta(t, 'name": "read')['tool_calls'][0]['function'].update(name='list_files'),
                     'calls differ'),
            'reasoning': (lambda t: delta(t, 'reasoning').update(reasoning='Other.'), 'reasoning differs'),
            'content': (lambda t: delta(t, 'reasoning').update(content='Reading.'), 'content differs'),
        }
        for name, (change, why) in cases.items():
            with self.subTest(name):
                turn = call_turn('read', read)
                change(turn)
                trace, _ = self.run_session(self.ended_in_thinking(), opening=[turn, call_turn('read', read)])
                self.assertEqual(trace['turns'][0]['raw_class'], why)
                self.assert_hard(trace, f'raw turn 0: {why}')
        # Raw text that is not reasoning, content and well-formed regions.
        broken = call_turn('read', read)
        chunk = next(e for e in broken if isinstance(e, dict) and e['choices']
                     and any(i >= SYNTHETIC for i in e['choices'][0].get('token_ids') or []))
        chunk['choices'][0]['token_ids'] = [piece('\n<function=read>\n<parameter=path>\nx\n</function>\n')]
        trace, _ = self.run_session(self.ended_in_thinking(), opening=[broken])
        self.assert_hard(trace, 'raw turn 0: malformed call region')


def recorded_answer(text, tools):
    """What a Qwen3.8 parser streams for one raw turn: the reasoning before
    the close, the content before the first tool region, and each region's
    call with its parameters typed by the tool schema."""
    body = text.removesuffix('<|im_end|>')
    reasoning, _, after = body.partition('</think>')
    types = {t['function']['name']: {k: v.get('type') for k, v in t['function']['parameters']['properties'].items()}
             for t in tools}
    calls = []
    for raw in re.findall(r'<tool_call>(.*?)</tool_call>', after, re.S):
        function = re.search(r'<function=([^>]+)>(.*)</function>', raw, re.S)
        if function:
            name = function.group(1)
            arguments = {k: int(v) if types[name].get(k) == 'integer' else v
                         for k, v in re.findall(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', function.group(2), re.S)}
            calls.append([name, arguments])
    return dict(reasoning=reasoning, content=after.partition('<tool_call>')[0].strip(), calls=calls)


def recorded_stream(turn, answer, ids=None):
    """A Chopinfer stream of a recorded turn: its prompt tail, each
    generated ID in its own chunk, then the parsed ANSWER's deltas."""
    ids = turn['token_ids'] if ids is None else ids
    events = [{'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'token_ids': []}],
               'prompt_token_ids': turn['prompt_tail']}]
    events += [{'choices': [{'index': 0, 'delta': {}, 'token_ids': [i]}]} for i in ids]
    deltas = [{field: answer[field]} for field in ('reasoning', 'content') if answer[field]]
    deltas += [{'tool_calls': [{'index': index, 'id': f'call-{index}', 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
               for index, (name, arguments) in enumerate(answer['calls'])]
    events += [{'choices': [{'index': 0, 'delta': delta, 'token_ids': []}]} for delta in deltas]
    events.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': turn['finish_reason']}]})
    events.append({'choices': [], 'usage': {'prompt_tokens': turn['prompt_tokens'], 'completion_tokens': len(ids)}})
    return events + ['[DONE]']


class RecordedOutcomes(unittest.TestCase):
    """The two Qwen3.8 model outcomes of the 2026-10-01 Qwen agentic replay
    (fixtures/qwen38/agentic-outcomes.json, token IDs only), every recorded
    turn replayed through the driver with the checkpoint tokenizer's token
    bytes. Each turn's parsed stream is rebuilt from its raw tokens and must
    match the digest of what Chopinfer's parser streamed."""

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(patch.object(agentic_mixed, 'MARKUP', served_markup()))
        cls.fixture = json.loads(OUTCOMES.read_text())
        token_bytes = {int(k): bytes.fromhex(v) for k, v in cls.fixture['token_bytes'].items()}
        cls.decode = staticmethod(lambda ids: b''.join(token_bytes[i] for i in ids).decode('utf-8', 'replace'))
        cls.manifest, groups = load_corpus()
        cls.files = groups['coding']

    def answers(self, recorded):
        answers = [recorded_answer(self.decode(turn['token_ids']), self.manifest['tools']) for turn in recorded['turns']]
        for turn, answer in zip(recorded['turns'], answers):
            digest = hashlib.sha256(json.dumps(answer, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(digest, turn['parsed_digest'])
        return answers

    def replay(self, recorded, changes=None):
        """The driver's trace of RECORDED, with CHANGES {turn: (IDs, answer)}
        to the generated IDs or the parsed answer of single turns."""
        streams = []
        for index, (turn, answer) in enumerate(zip(recorded['turns'], self.answers(recorded))):
            ids, answer = (changes or {}).get(index, (None, answer))
            streams.append(recorded_stream(turn, answer, ids))
        profile = vars(parser().parse_args(['--temperature', '0']))
        trace = session(ScriptedServer(streams, self.decode), 'model', profile, self.manifest, self.files,
                        recorded['concurrency'], recorded['index'], recorded['phase'])
        self.assertEqual((trace['case_id'], len(trace['turns'])), (recorded['case_id'], len(streams)))
        return trace

    def test_recorded_outcomes_are_model_outcomes(self):
        from agentic_metrics import model_outcome
        classes = {'ended inside thinking': 'reasoning only', 'malformed tool call': 'unparsed tool region'}
        for recorded in self.fixture['sessions']:
            with self.subTest(recorded['case_id']):
                trace = self.replay(recorded)
                self.assertEqual(trace['failure'], recorded['recorded_failure'])
                self.assertEqual([turn['raw_class'] for turn in trace['turns']],
                                 ['calls agree'] * (len(trace['turns']) - 1) + [classes[recorded['outcome']]])
                self.assertEqual(model_outcome(trace), (recorded['outcome'], None))

    def test_recorded_sessions_with_parser_loss_are_hard(self):
        from agentic_metrics import model_outcome
        ended, malformed = self.fixture['sessions']
        last = len(ended['turns']) - 1
        # A dropped answer after a reasoning close in the last turn.
        ids = ended['turns'][-1]['token_ids']
        answer = self.answers(ended)[-1]
        changes = {last: (ids[:-1] + [THINK_END, 271, 3833] + ids[-1:], answer)}
        self.assertEqual(model_outcome(self.replay(ended, changes)), (None, 'raw last turn: output after reasoning'))
        # The malformed region given a function header (`<function=search`):
        # a call the parser lost.
        ids = list(malformed['turns'][-1]['token_ids'])
        at = ids.index(21731)
        changes = {len(malformed['turns']) - 1: (ids[:at] + [1628, 93260] + ids[at + 1:], self.answers(malformed)[-1])}
        self.assertEqual(model_outcome(self.replay(malformed, changes)), (None, 'raw last turn: output after reasoning'))
        # Confirm review P1 on recorded tokens: the first turn's second call
        # (list_files) dropped, and a read offset changed.
        first = self.answers(malformed)[0]
        changes = {0: (None, dict(first, calls=first['calls'][:1]))}
        self.assertEqual(model_outcome(self.replay(malformed, changes)), (None, 'raw turn 0: calls differ'))
        read = self.answers(ended)[1]
        changes = {1: (None, dict(read, calls=[[read['calls'][0][0], dict(read['calls'][0][1], offset=61)]]))}
        self.assertEqual(model_outcome(self.replay(ended, changes)), (None, 'raw turn 1: calls differ'))


if __name__ == '__main__':
    unittest.main()
