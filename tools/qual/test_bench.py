"""CPU checks for bench.py's speculation scrapes and decode windows."""

from qual_test_config import configure_receipts as setUpModule

import unittest
from unittest.mock import patch

import bench
from bench import cohort, steady


class SettledMetricsTests(unittest.TestCase):
    def test_waits_until_speculation_series_stop_changing(self):
        scrapes = iter(["chop_spec_verify_items_total 1\n",
                        "chop_spec_verify_items_total 2\nchop_requests_active 1\n",
                        "chop_spec_verify_items_total 2\nchop_requests_active 0\n"])
        with patch.object(bench, "metrics_text", lambda client: next(scrapes)):
            self.assertEqual(bench.settled_metrics(None, interval=0),
                             "chop_spec_verify_items_total 2\nchop_requests_active 0\n")

    def test_gives_up_after_its_attempts(self):
        counter = iter(range(100))
        with patch.object(bench, "metrics_text",
                          lambda client: f"chop_spec_verify_items_total {next(counter)}\n"):
            self.assertEqual(bench.settled_metrics(None, attempts=3, interval=0),
                             "chop_spec_verify_items_total 3\n")


class DecodeWindows(unittest.TestCase):
    def test_cohort_counts_the_tail_the_steady_window_ends_before(self):
        # One stream emits 4 tokens per step and finishes early; the other
        # emits one per step. Steps are 10 ms apart.
        fast = [(0.01 * step, 4) for step in range(16)]
        slow = [(0.01 * step, 1) for step in range(64)]
        window = steady([fast, slow], trim=8)
        whole = cohort([fast, slow])
        self.assertAlmostEqual(whole["seconds"], 0.63)
        self.assertEqual(whole["tokens"], 15 * 4 + 63)
        self.assertLess(whole["tokens_per_second"], window["aggregate_tokens_per_second"])

    def test_equal_streams_give_the_same_rate_in_both_windows(self):
        streams = [[(0.01 * step, 1) for step in range(100)] for _ in range(4)]
        self.assertAlmostEqual(cohort(streams)["tokens_per_second"], 400.0)
        self.assertAlmostEqual(steady(streams)["aggregate_tokens_per_second"], 400.0, delta=15)


class StreamPromptTests(unittest.TestCase):
    def test_numbered_streams_share_the_list_and_diverse_streams_differ(self):
        numbered = [bench.stream_prompt("numbered", index) for index in range(3)]
        self.assertTrue(all(prompt.startswith("Continue this numbered list") for prompt in numbered))
        diverse = [bench.stream_prompt("diverse", index) for index in range(128)]
        self.assertEqual(len(set(diverse)), 128)
        seeds = {prompt.split(" (request")[0] for prompt in diverse}
        self.assertEqual(seeds, set(bench.DIVERSE_SEEDS))
        self.assertEqual(len(bench.DIVERSE_SEEDS), 16)


class PrefillStreamTests(unittest.TestCase):
    def run_bench(self, token_text, include_usage=True):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path, body))
                choice = {"text": token_text, "finish_reason": "length"}
                if body.get("logprobs") is not None:
                    choice["logprobs"] = {"tokens": [token_text or "<think>"]}
                events = [{"choices": [choice]}]
                if include_usage:
                    events.append({"choices": [], "usage": {
                        "prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with patch("sys.argv", ["bench", "--url", url, "--mechanics-only"]), \
                     patch.object(bench.Client, "identity", return_value={"model": "fake"}), \
                     patch.object(bench, "settled_metrics", return_value=""), \
                     patch.object(bench, "receipt") as receipt:
                    with self.assertRaises(SystemExit) as result:
                        bench.main()
                return result.exception.code, receipt.call_args.args[2], requests
            finally:
                server.shutdown()
                thread.join()

    def test_hidden_first_token_has_prefill_timing(self):
        code, report, requests = self.run_bench("")
        self.assertEqual(code, 0, report["errors"])
        self.assertGreater(report["prefill"]["ttft_seconds"], 0)
        self.assertEqual(report["prefill"]["prompt_tokens"], 12)
        path, body = requests[-1]
        self.assertEqual(path, "/v1/completions")
        self.assertEqual(body["max_tokens"], 1)
        self.assertTrue(body["stream_options"]["include_usage"])

    def test_visible_first_token_has_prefill_timing(self):
        code, report, _ = self.run_bench("A")
        self.assertEqual(code, 0, report["errors"])
        self.assertGreater(report["prefill"]["throughput_tokens_per_second"], 0)

    def test_missing_usage_still_fails(self):
        code, report, _ = self.run_bench("A", include_usage=False)
        self.assertEqual(code, 1)
        self.assertIsNone(report["prefill"])

class ConfigurableCohortTests(unittest.TestCase):
    def test_success_receipt_counts_the_requested_cohorts(self):
        import sys

        class Client:
            def __init__(self, *args):
                pass

            def identity(self, *args):
                return {'model': 'fake'}

            def request(self, *args, **kwargs):
                return [], 0.01

        for concurrencies in [('1',), ('1', '2', '3'), ('32', '64', '128', '256')]:
            with self.subTest(concurrencies=concurrencies):
                receipts = []
                with patch.object(sys, 'argv', ['bench.py', '--mechanics-only',
                        '--max-sequences', '256', '--concurrencies', *concurrencies]), \
                     patch.object(bench, 'Client', Client), \
                     patch.object(bench, 'settled_metrics', side_effect=ValueError('no metrics')), \
                     patch.object(bench, 'token_events', return_value=(
                         [(10**12, 'x')], {'prompt_tokens': 10, 'completion_tokens': 8}, 'length')), \
                     patch.object(bench, 'receipt', side_effect=lambda _, server, result, mechanics,
                                  load_samples: receipts.append({**result, 'load_samples': load_samples})):
                    with self.assertRaises(SystemExit) as exit:
                        bench.main()
                self.assertEqual(receipts[0]['errors'], [])
                self.assertEqual(exit.exception.code, 0)
                self.assertTrue(receipts[0]['passed'])
                self.assertEqual(len(receipts[0]['decode']), len(concurrencies))
                # Samples bound the decode passes and the timed prefill.
                self.assertEqual(len(receipts[0]['load_samples']), 3)
                self.assertTrue(all(row and row['pressure_some_us'] for row in receipts[0]['load_samples']))


class QueuedCohortTests(unittest.TestCase):
    def test_disjoint_queued_streams_pass_only_with_explicit_cohort_measurement(self):
        import time
        def request(_client, _path, body, stream=False):
            count = body['max_tokens']
            index = int(body['prompt'].split('from ')[1].split(':')[0]) if count > 1 else 0
            start = time.monotonic() + 1 + index * 10
            events = [(start + i / 100, {'choices': [{'text': 'x', 'finish_reason': None}]}) for i in range(count)]
            events.append((start + count / 100, {'choices': [{'finish_reason': 'length'}],
                           'usage': {'completion_tokens': count, 'prompt_tokens': 32768}}))
            return events, count / 100
        for queued in (False, True):
            args = ['bench', '--concurrencies', '2', '--max-sequences', '2', '--output-tokens', '128',
                    '--prefill-repeats', '1'] + (['--cohort-only'] if queued else [])
            with patch('sys.argv', args), patch.object(bench.Client, 'request', request), \
                 patch.object(bench.Client, 'identity', return_value={'model': 'fake'}), \
                 patch.object(bench, 'settled_metrics', return_value=''), \
                 patch.object(bench, 'receipt') as receipt:
                with self.assertRaises(SystemExit) as caught:
                    bench.main()
            self.assertEqual(caught.exception.code, 0 if queued else 1)
            value = receipt.call_args.args[2]
            self.assertEqual(value['passed'], queued)
            if queued:
                self.assertIsNone(value['decode'][0]['steady'])
                self.assertGreater(value['decode'][0]['cohort']['tokens_per_second'], 0)


class FixedBudgetTests(unittest.TestCase):
    def run_fixed(self, sampling='greedy', broken=False):
        import time
        def request(_client, _path, body, stream=False):
            count = body['max_tokens']
            decode = count > 1
            if decode and broken and 'from 1:' in body['prompt']:
                raise ValueError('transport failed')
            forced = body.get('ignore_eos') is True and body.get('min_tokens') == count
            short = decode and (broken or not forced)
            emitted = 2 if short else count
            start = time.monotonic() + 1
            events = [(start + i / 100, {'choices': [{'text': 'x'}]}) for i in range(emitted)]
            events.append((start + emitted / 100, {'choices': [
                {'finish_reason': 'stop' if short else 'length'}],
                'usage': {'completion_tokens': emitted, 'prompt_tokens': 32768 if not decode else 18}}))
            return events, emitted / 100
        args = ['bench', '--concurrencies', '2', '--max-sequences', '2', '--output-tokens', '128',
                '--prefill-repeats', '1', '--cohort-only', '--sampling', sampling]
        with patch('sys.argv', args), patch.object(bench.Client, 'request', request), \
             patch.object(bench.Client, 'identity', return_value={'model': 'fake'}), \
             patch.object(bench, 'settled_metrics', return_value=''), \
             patch.object(bench, 'receipt') as receipt:
            with self.assertRaises(SystemExit) as caught:
                bench.main()
        return caught.exception.code, receipt.call_args.args[2]

    def test_fixed_budget_suppresses_normal_eos_in_both_sampling_modes(self):
        for sampling in ('greedy', 'server-default'):
            code, result = self.run_fixed(sampling)
            self.assertEqual(code, 0, result['errors'])
            self.assertEqual(len(result['decode'][0]['requests']), 2)

    def test_failed_cohort_preserves_every_request_and_has_no_throughput(self):
        code, result = self.run_fixed(broken=True)
        self.assertEqual(code, 1)
        rows = result['errors'][0]['requests']
        self.assertEqual([r['index'] for r in rows], [0, 1])
        self.assertEqual(rows[0]['finish_reason'], 'stop')
        self.assertEqual(rows[0]['usage']['completion_tokens'], 2)
        self.assertIn('transport failed', rows[1]['error'])
        self.assertEqual(result['decode'], [])


if __name__ == "__main__":
    unittest.main()
