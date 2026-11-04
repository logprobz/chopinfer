"""CPU checks for local serving process GPU memory accounting and the canary."""

from qual_test_config import setUpModule

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import subprocess
import threading
import unittest
from unittest.mock import patch

import json
import soak

from soak import (MEDIA_CASES, check_canaries, check_media, check_structured, classify_canary,
                  completion_canary, device_memory, media_requests, serving_pid)

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "serving process memory accounting and canary parsing with mocked tools"


def result(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout)


class SoakMemoryTests(unittest.TestCase):
    @patch("soak.subprocess.run")
    def test_serving_pid_uses_local_listener(self, run):
        run.return_value = result('LISTEN 0 128 127.0.0.1:8101 0.0.0.0:* users:(("chopinfer",pid=123,fd=8))\n')
        self.assertEqual(serving_pid("http://127.0.0.1:8101"), 123)

    @patch("soak.subprocess.run")
    def test_memory_counts_server_and_children_only(self, run):
        run.side_effect = [result("123 1\n124 123\n125 124\n900 1\n"),
                           result("123, 100\n124, 200\n125, 50\n900, 9000\n")]
        self.assertEqual(device_memory(123), 350)

    @patch("soak.subprocess.run")
    def test_missing_serving_allocation_fails(self, run):
        run.side_effect = [result("123 1\n900 1\n"), result("900, 9000\n")]
        with self.assertRaisesRegex(RuntimeError, "no GPU allocation"):
            device_memory(123)

    @patch("soak.process_tree", return_value={123, 124})
    @patch("soak.Path.read_text")
    def test_host_rss_counts_the_same_serving_tree(self, read, tree):
        from soak import host_memory
        read.side_effect = ["VmRSS:\t    1024 kB\nRssAnon:\t     512 kB\nRssFile:\t     256 kB\n"
                            "RssShmem:\t     256 kB\n",
                            "VmRSS:\t    2048 kB\nRssAnon:\t    1024 kB\nRssFile:\t     768 kB\n"
                            "RssShmem:\t     256 kB\n"]
        self.assertEqual(host_memory(123), {"host_rss_mib": 3, "host_rss_anon_mib": 1.5,
                                            "host_rss_file_mib": 1, "host_rss_shmem_mib": 0.5})
        # A part one process does not report is unknown for the tree; VmRSS
        # stays required.
        read.side_effect = ["VmRSS: 1024 kB\nRssAnon: 512 kB\nRssFile: 512 kB\n",
                            "VmRSS: 2048 kB\nRssAnon: 1024 kB\nRssFile: 768 kB\nRssShmem: 256 kB\n"]
        self.assertEqual(host_memory(123), {"host_rss_mib": 3, "host_rss_anon_mib": 1.5,
                                            "host_rss_file_mib": 1.25, "host_rss_shmem_mib": None})
        read.side_effect = ["RssAnon: 512 kB\n", "VmRSS: 2048 kB\n"]
        with self.assertRaisesRegex(RuntimeError, "no resident memory sample"):
            host_memory(123)

    @patch("soak.device_memory", return_value=350)
    @patch("soak.host_memory", return_value={"host_rss_mib": 3, "host_rss_anon_mib": 1.5,
                                             "host_rss_file_mib": 1, "host_rss_shmem_mib": 0.5})
    def test_samples_record_the_mapped_views_the_host_serves(self, host, device):
        # runtime-host mapped_views.rs: the resident bytes only while
        # residency is visible.
        visible = ("# TYPE chop_mapped_view_bytes gauge\nchop_mapped_view_bytes 8192\n"
                   "# TYPE chop_mapped_view_residency_visible gauge\nchop_mapped_view_residency_visible 1\n"
                   "# TYPE chop_mapped_view_resident_bytes gauge\nchop_mapped_view_resident_bytes 4096\n")
        hidden = ("# TYPE chop_mapped_view_bytes gauge\nchop_mapped_view_bytes 8192\n"
                  "# TYPE chop_mapped_view_residency_visible gauge\nchop_mapped_view_residency_visible 0\n")
        for text, expected in ((visible, {"mapped_view_bytes": 8192, "mapped_view_resident_bytes": 4096}),
                               (hidden, {"mapped_view_bytes": 8192}), ("chop_batches_total 3\n", {})):
            with self.subTest(expected), patch("soak.metrics_text", return_value=text):
                sample = soak.memory_sample(123, 0, None)
                self.assertEqual({key: sample[key] for key in sample if key.startswith("mapped_view")}, expected)
                self.assertEqual((sample["mib"], sample["host_rss_mib"], sample["host_rss_anon_mib"],
                                  sample["host_rss_file_mib"], sample["host_rss_shmem_mib"]), (350, 3, 1.5, 1, 0.5))
        with patch("soak.metrics_text", side_effect=OSError("connection refused")):
            self.assertEqual(soak.memory_sample(123, 0, None)["mapped_view_error"], "OSError: connection refused")
        # A gauge that is not a whole number of bytes is an error of the
        # scrape, beside a kept sample.
        for value in ("NaN", "+Inf", "-Inf", "-1", "8192.5", "x", ""):
            with self.subTest(value=value), patch("soak.metrics_text",
                                                  return_value=f"chop_mapped_view_bytes {value}\n"):
                sample = soak.memory_sample(123, 0, None)
                self.assertNotIn("mapped_view_bytes", sample)
                self.assertTrue(sample["mapped_view_error"].startswith(("ValueError", "IndexError")))
                self.assertEqual((sample["mib"], sample["host_rss_mib"]), (350, 3))

    def test_the_soak_samples_the_parts_and_mapped_views_of_its_server(self):
        argv = ["soak", "--duration", "0.3", "--concurrency", "2", "--server-pid", "123"]
        parts = {"host_rss_mib": 3, "host_rss_anon_mib": 1.5, "host_rss_file_mib": 1, "host_rss_shmem_mib": 0.5}
        with patch("sys.argv", argv), patch.object(soak, "Client", FakeClient), \
             patch.object(soak, "device_memory", return_value=350), \
             patch.object(soak, "host_memory", return_value=parts), \
             patch.object(soak, "metrics_text", return_value="chop_mapped_view_bytes 8192\n"), \
             patch.object(soak, "receipt") as receipt:
            with self.assertRaises(SystemExit):
                soak.main()
        report = receipt.call_args.args[2]
        self.assertGreaterEqual(len(report["memory_samples"]), 2)
        for sample in report["memory_samples"]:
            self.assertEqual({key: sample[key] for key in (*parts, "mapped_view_bytes")},
                             {**parts, "mapped_view_bytes": 8192})
        self.assertEqual(report["errors"], [])

    def test_the_recorded_parts_leave_the_memory_verdict_alone(self):
        limits = {"device": 512, "host": 512}
        rows = [{"elapsed_seconds": t, "mib": 1000, "host_rss_mib": 4000 + t // 4} for t in range(0, 1201, 30)]
        recorded = [{**row, "host_rss_anon_mib": 3000 + 9 * row["elapsed_seconds"], "host_rss_file_mib": 900,
                     "host_rss_shmem_mib": 100, "mapped_view_bytes": 2 ** 40,
                     "mapped_view_resident_bytes": 2 ** 30 * row["elapsed_seconds"]} for row in rows]
        for allowance in (None, {"host": 0, "device": 0}):
            self.assertEqual(soak.memory_rule(recorded, 1200, limits, 0, allowance),
                             soak.memory_rule(rows, 1200, limits, 0, allowance))



def run(*steps):
    """A canary answer from (token, logprob, runner-up, runner-up logprob) steps."""
    return {"tokens": [step[0] for step in steps], "logprobs": [step[1] for step in steps],
            "top_logprobs": [[{"token": step[0], "logprob": step[1]},
                              {"token": step[2], "logprob": step[3]}] for step in steps]}


class CanaryTests(unittest.TestCase):
    def test_same_tokens_are_exact(self):
        cold = run(("A", -0.1, "B", -2.0), ("C", -0.2, "D", -3.0))
        self.assertEqual(classify_canary(cold, cold)["verdict"], "exact")

    def test_a_flip_both_runs_nearly_tie_is_a_near_tie(self):
        cold = run(("A", -0.60, "B", -0.80), ("C", -0.2, "D", -3.0))
        warm = run(("B", -0.65, "A", -0.75), ("E", -0.2, "F", -3.0))
        verdict = classify_canary(cold, warm)
        self.assertEqual(verdict["verdict"], "near_tie")
        self.assertEqual(verdict["first_difference"], 0)
        self.assertAlmostEqual(verdict["cold_margin"], 0.20)
        self.assertAlmostEqual(verdict["other_margin"], 0.10)

    def test_a_confident_flip_is_a_restore_error(self):
        cold = run(("A", -0.01, "B", -5.0))
        warm = run(("B", -0.02, "A", -4.5))
        self.assertEqual(classify_canary(cold, warm)["verdict"], "restore_error")

    def test_a_token_outside_the_other_top_two_is_a_restore_error(self):
        cold = run(("A", -0.60, "C", -0.80))
        warm = run(("B", -0.65, "A", -0.75))
        verdict = classify_canary(cold, warm)
        self.assertEqual(verdict["verdict"], "restore_error")
        self.assertIsNone(verdict["cold_margin"])

    def test_prefix_drift_before_a_tie_is_a_restore_error(self):
        cold = run(("A", -0.1, "X", -3.0), ("C", -0.60, "D", -0.80))
        warm = run(("A", -1.1, "X", -3.0), ("D", -0.65, "C", -0.75))
        self.assertEqual(classify_canary(cold, warm)["verdict"], "restore_error")

    def test_an_early_stop_is_a_restore_error(self):
        cold = run(("A", -0.1, "B", -3.0), ("C", -0.1, "D", -3.0))
        warm = run(("A", -0.1, "B", -3.0))
        verdict = classify_canary(cold, warm)
        self.assertEqual((verdict["verdict"], verdict["first_difference"]), ("restore_error", 1))


def message(content, finish_reason="stop"):
    return {"message": {"content": content}, "finish_reason": finish_reason}



def with_cache(answer, cached_tokens):
    return {**answer, "usage": {"prompt_tokens_details": {"cached_tokens": cached_tokens}}}


class CanaryBaselineTests(unittest.TestCase):
    def test_completion_logprobs_become_canary_steps(self):
        value = {"usage": {"prompt_tokens": 3}, "choices": [{"text": "Rule 7", "logprobs": {
            "tokens": ["Rule", " 7"], "token_logprobs": [-0.64, -0.1],
            "top_logprobs": [{"Rule": -0.64, "I": -1.17}, {" 7": -0.1, " 8": -4.0}]}}]}
        answer = completion_canary(value)
        self.assertEqual(answer["tokens"], ["Rule", " 7"])
        self.assertEqual(answer["top_logprobs"][0], [{"token": "Rule", "logprob": -0.64},
                                                     {"token": "I", "logprob": -1.17}])
        self.assertEqual(classify_canary(answer, answer)["verdict"], "exact")

    def test_hits_are_compared_with_the_cold_canary_cut_at_the_anchor(self):
        # The 2026-09-25 soaks: cut at 2304, "Rule" leads "I" by 0.53 nats.
        cut = run(("Rule", -0.64, "I", -1.17), (" 7", -0.1, " 8", -4.0))
        canaries = [with_cache(cut, 0), with_cache(cut, 2304), with_cache(cut, 2304)]
        self.assertTrue(check_canaries(canaries, 2304)["canary_passed"])
        # An anchor evicted under load: the first later canary is cut again.
        canaries[1] = with_cache(cut, 0)
        self.assertTrue(check_canaries(canaries, 2304)["canary_passed"])

    def test_a_restore_error_a_missing_hit_or_a_hit_baseline_fails(self):
        cut = run(("Rule", -0.64, "I", -1.17), (" 7", -0.1, " 8", -4.0))
        wrong = run(("Topic", -0.2, "Rule", -3.0), (" 7", -0.1, " 8", -4.0))
        self.assertFalse(check_canaries(
            [with_cache(cut, 0), with_cache(cut, 2304), with_cache(wrong, 2304)],
            2304)["canary_passed"])
        self.assertFalse(check_canaries(
            [with_cache(cut, 0), with_cache(cut, 2304), with_cache(cut, 0)],
            2304)["canary_passed"])
        self.assertFalse(check_canaries(
            [with_cache(cut, 2304), with_cache(cut, 2304), with_cache(cut, 2304)],
            2304)["canary_passed"])
        self.assertFalse(check_canaries(
            [with_cache(cut, 0), with_cache(cut, 1024), with_cache(cut, 2304)],
            2304)["canary_passed"])

class SoakMixTests(unittest.TestCase):
    PERSON = {"name": "Ada", "age": 36, "status": "active", "kind": "person",
              "address": {"city": "Paris", "zip": "75001"}, "tags": ["a"], "active": True,
              "nickname": None, "note": None}

    def test_media_cases_exist(self):
        import soak
        module = soak.media_module()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(soak, "media_module", return_value=module), \
             patch.object(module, "messages", lambda case, fixture_dir: case["id"]):
            self.assertEqual([case for case, _ in media_requests(fixture_dir=Path(directory))], list(MEDIA_CASES))

    def test_the_square_must_be_blue(self):
        check_media("image-blue-square.png", message("Blue."))
        with self.assertRaises(ValueError):
            check_media("image-blue-square.png", message("Red."))
        with self.assertRaises(ValueError):
            check_media("video-red-then-blue", message(" "))

    def test_structured_output_must_satisfy_the_schema(self):
        from structured import PERSON
        valid = {key: self.PERSON[key] for key in PERSON["required"]}
        self.assertFalse(check_structured(message(json.dumps(valid))))
        self.assertTrue(check_structured(message('{"name": "A', "length")))
        with self.assertRaises(ValueError):
            check_structured(message('{"name": "A"'))
        with self.assertRaises(ValueError):
            check_structured(message(json.dumps(dict(valid, age=-1))))


class FakeClient:
    """A server whose structured answers cycle valid, cut off, invalid."""

    def __init__(self, url, timeout=900):
        self.lock = __import__("threading").Lock()
        self.structured = 0

    def identity(self, model=None):
        return {"url": "http://127.0.0.1:1", "model": "fake", "models": ["fake"], "health": None}

    def request(self, path, body=None, stream=False, headers=None):
        if "response_format" not in body:
            return {"id": "chatcmpl_plain", "choices": [message("Hello.")]}, 0.01
        with self.lock:
            self.structured += 1
            index = self.structured
        valid = json.dumps({key: SoakMixTests.PERSON[key] for key in __import__("structured").PERSON["required"]})
        choice = (message(valid), message('{"name": "Ada", "age": 3', "length"),
                  message('{"name": "Ada"}'))[index % 3]
        return {"id": f"chatcmpl_{index:04d}", "choices": [choice],
                "usage": {"completion_tokens": 512 if choice["finish_reason"] == "length" else 20}}, 0.01


class StructuredKeepTests(unittest.TestCase):
    def soak(self):
        argv = ["soak", "--remote-only", "--structured-mix", "--duration", "0.3", "--concurrency", "8"]
        with patch("sys.argv", argv), patch.object(soak, "Client", FakeClient), \
             patch.object(soak, "receipt") as receipt:
            with self.assertRaises(SystemExit):
                soak.main()
        return receipt.call_args.args[2]

    def test_cut_off_and_invalid_structured_outputs_are_kept(self):
        report = self.soak()
        kept = report["structured_kept"]
        self.assertEqual(kept["limit"], soak.KEEP_STRUCTURED)
        self.assertEqual(kept["total"], report["mixed_requests"]["structured_truncated"]
                         + sum("schema" in str(error.get("error")) for error in report["errors"]))
        self.assertGreater(kept["total"], soak.KEEP_STRUCTURED)
        self.assertEqual(len(kept["requests"]), soak.KEEP_STRUCTURED)
        reasons = {row["reason"] for row in kept["requests"]}
        self.assertIn("truncated", reasons)
        self.assertTrue(any(reason.startswith("schema") for reason in reasons))
        for row in kept["requests"]:
            self.assertEqual(row["request"]["response_format"], soak.STRUCTURED_FORMAT)
            self.assertTrue(row["response_id"].startswith("chatcmpl_"))
            self.assertIn(row["finish_reason"], ("length", "stop"))
            self.assertEqual(row["completion_tokens"], 512 if row["finish_reason"] == "length" else 20)
            self.assertTrue(row["content"].startswith('{"name": "Ada"'))

    def test_host_load_is_sampled_with_the_memory_samples(self):
        argv = ["soak", "--remote-only", "--duration", "0.3", "--concurrency", "2"]
        with patch("sys.argv", argv), patch.object(soak, "Client", FakeClient), \
             patch.object(soak, "receipt") as receipt:
            with self.assertRaises(SystemExit):
                soak.main()
        samples = receipt.call_args.kwargs["load_samples"]
        # The start, each sample during the soak, and the end once the last
        # request in flight finished, so even a soak whose start overruns
        # its deadline has its timed work bounded.
        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue(all(set(row) == {"at", "uptime_seconds", "load_average", "cpu_idle_ticks", "cpu_ticks",
                                         "cpus", "pressure_some_us"} for row in samples))

    def test_keeping_outputs_leaves_the_verdict_alone(self):
        report = self.soak()
        # Truncation still counts and a schema failure is still an error.
        self.assertGreater(report["mixed_requests"]["structured_truncated"], 0)
        self.assertTrue(any("schema" in str(error.get("error")) for error in report["errors"]))
        self.assertFalse(report["passed"])


class MetricsServer:
    """A local /metrics endpoint that answers every scrape with `body`, cut to
    its first `sent` bytes under the full Content-Length when `sent` is set."""

    def __init__(self, body, sent=None):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):
                handler.send_response(200)
                handler.send_header("Content-Type", "text/plain; version=0.0.4")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body if sent is None else body[:sent])

            def log_message(handler, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class MetricsClient(FakeClient):
    """A FakeClient whose /metrics is served at `url`."""

    def __init__(self, url, timeout=900):
        super().__init__(url, timeout)
        self.url, self.timeout = url, timeout


class MappedViewScrapeTests(unittest.TestCase):
    """The mapped-view gauges are recorded only: a scrape that fails in
    transport, decoding or parsing keeps each memory sample beside its error
    and leaves the soak verdict alone."""

    def soak(self, body, sent=None):
        server = MetricsServer(body, sent)
        self.addCleanup(server.close)
        argv = ["soak", "--url", server.url, "--duration", "0.3", "--concurrency", "2", "--server-pid", "123"]
        with patch("sys.argv", argv), patch.object(soak, "Client", MetricsClient), \
             patch.object(soak, "device_memory", return_value=350), \
             patch.object(soak, "host_memory", return_value={"host_rss_mib": 3, "host_rss_anon_mib": 1.5,
                                                             "host_rss_file_mib": 1, "host_rss_shmem_mib": 0.5}), \
             patch.object(soak, "receipt") as receipt:
            with self.assertRaises(SystemExit) as exited:
                soak.main()
        return exited.exception.code, receipt.call_args.args[2]

    def test_a_failed_scrape_keeps_the_sample_and_the_verdict(self):
        served = b"chop_mapped_view_bytes 8192\nchop_mapped_view_resident_bytes 4096\n"
        code, report = self.soak(served)
        self.assertEqual((code, report["passed"], report["errors"]), (0, True, []))
        self.assertGreaterEqual(len(report["memory_samples"]), 2)
        self.assertTrue(all(sample["mapped_view_bytes"] == 8192 for sample in report["memory_samples"]))
        for label, body, sent, cause in (
                ("truncated response", served * 4, 20, "IncompleteRead"),
                ("bad encoding", served + b"# \xff\xfe\n", None, "UnicodeDecodeError"),
                ("NaN gauge", b"chop_mapped_view_bytes NaN\n", None, "not a finite whole number of bytes")):
            with self.subTest(label):
                failed_code, failed = self.soak(body, sent)
                self.assertEqual((failed_code, failed["passed"], failed["errors"]), (code, report["passed"], []))
                self.assertEqual(len(failed["memory_samples"]), len(report["memory_samples"]))
                for sample in failed["memory_samples"]:
                    self.assertEqual((sample["mib"], sample["host_rss_mib"]), (350, 3))
                    self.assertNotIn("mapped_view_bytes", sample)
                    self.assertIn(cause, sample["mapped_view_error"])


class MemoryRule(unittest.TestCase):
    def samples(self, host, device=lambda t: 0, end=1200):
        return [{"elapsed_seconds": t, "mib": 1000 + device(t), "host_rss_mib": 4000 + host(t)}
                for t in range(0, end + 1, 30)]

    def test_the_defaults_are_the_plain_growth_rule(self):
        import random
        generator = random.Random(7)
        for _ in range(200):
            rows = [{"elapsed_seconds": 30 * i, "mib": generator.randint(0, 900),
                     "host_rss_mib": generator.randint(0, 900)} for i in range(41)]
            limits = {"device": generator.choice((256, 512)), "host": generator.choice((256, 512))}
            plain = all(max(row[key] for row in rows) - rows[0][key] <= limits[kind]
                        for kind, key in (("device", "mib"), ("host", "host_rss_mib")))
            rule = soak.memory_rule(rows, 1200, limits)
            self.assertEqual(rule["passed"], plain)
            self.assertEqual(rule["memory_growth_mib"], rule["steady_memory_growth_mib"])

    def test_the_warm_up_sets_the_steady_baseline_and_is_capped(self):
        rows = self.samples(lambda t: min(t, 120) * 8)
        rule = soak.memory_rule(rows, 1200, {"device": 512, "host": 512}, 180, {"host": 2560, "device": 1344})
        self.assertEqual((rule["steady_baseline_seconds"], rule["steady_host_memory_growth_mib"],
                          rule["host_memory_growth_mib"], rule["passed"]), (180, 0, 960, True))
        # A 60-second soak keeps three quarters of its interval measured.
        rule = soak.memory_rule(self.samples(lambda t: 0, end=60), 60, {"device": 512, "host": 512}, 180,
                                {"host": 0, "device": 0})
        self.assertEqual((rule["effective_warmup_seconds"], rule["steady_baseline_seconds"]), (15, 30))

    def test_invalid_declarations_are_refused(self):
        for warmup, allowance in ((-1, {"host": 0, "device": 0}), (1.5, {"host": 0, "device": 0}),
                                  (0, {"host": 0}), (0, {"host": -1, "device": 0}), (0, [0, 0])):
            with self.subTest(warmup=warmup, allowance=allowance), self.assertRaises(ValueError):
                soak.memory_rule(self.samples(lambda t: 0), 1200, {"device": 512, "host": 512}, warmup, allowance)


if __name__ == "__main__":
    unittest.main()
