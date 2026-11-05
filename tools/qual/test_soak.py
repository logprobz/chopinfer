"""CPU checks for local serving process GPU memory accounting and the canary."""

from qual_test_config import setUpModule

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
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


class SoakMediaFixtureTests(unittest.TestCase):
    """The media mix reads its fixtures from the declaration's
    soak_media_fixtures input when it states one (a model whose
    media_fixtures holds other media, such as image probes), else from
    media_fixtures."""

    def requested(self, **inputs):
        """The fixture directory the media mix's image request is built from
        under a declaration with `inputs`."""
        module = soak.media_module()
        seen = []
        with patch("common.qualification_config", return_value={"model": "soak-fixture", "inputs": inputs}), \
             patch.object(soak, "media_module", return_value=module), \
             patch.object(module, "messages", lambda case, fixture_dir: seen.append(fixture_dir)):
            soak.media_requests(["image-blue-square.png"])
        return seen

    def test_a_declared_soak_fixture_input_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.requested(media_fixtures=f"{directory}/probes",
                                            soak_media_fixtures=f"{directory}/functional"),
                             [Path(directory) / "functional"])

    def test_without_one_the_media_fixtures_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.requested(media_fixtures=f"{directory}/functional"),
                             [Path(directory) / "functional"])


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


LIMITS = {"device": 512, "host": 512}


def parts(files=lambda t: 0, end=1200):
    """Memory samples whose VmRSS is the sum of its recorded parts, RssFile
    grown by `files(t)`, beside a recorded mapped-view gauge."""
    rows = [{"elapsed_seconds": t, "mib": 1000, "host_rss_anon_mib": 9600, "host_rss_shmem_mib": 2100,
             "host_rss_file_mib": 13000 + files(t), "mapped_view_bytes": 200 * 2 ** 30}
            for t in range(0, end + 1, 30)]
    for row in rows:
        row["host_rss_mib"] = row["host_rss_anon_mib"] + row["host_rss_shmem_mib"] + row["host_rss_file_mib"]
    return rows


class HostMemoryScope(unittest.TestCase):
    """The declared HOST_MEMORY_SCOPE: VmRSS by default, whatever else a
    sample records; an unknown scope is refused."""

    def rule(self, rows, scope="rss"):
        return soak.memory_rule(rows, 1200, LIMITS, 0, None, scope)

    def test_the_default_scope_is_vmrss_and_adds_no_fields(self):
        fields = {"warmup_seconds", "effective_warmup_seconds", "steady_baseline_seconds", "cache_allowance_mib",
                  "memory_growth_mib", "steady_memory_growth_mib", "host_memory_growth_mib",
                  "steady_host_memory_growth_mib", "passed"}
        vmrss = [{"elapsed_seconds": t, "mib": 1000, "host_rss_mib": 4000 + t // 4} for t in range(0, 1201, 30)]
        for label, rows in (("VmRSS only", vmrss), ("with parts", parts(files=lambda t: t * 4))):
            with self.subTest(label):
                rule = self.rule(rows)
                self.assertEqual(rule, soak.memory_rule(rows, 1200, LIMITS))
                self.assertEqual(set(rule), fields)
        # A file-backed growth over the limit fails under VmRSS.
        self.assertFalse(self.rule(parts(files=lambda t: t * 4))["passed"])

    def test_an_unknown_scope_is_refused(self):
        # The removed anonymous_shmem scope is unknown too.
        for scope in ("vmrss", "anonymous", "anonymous_shmem", "", 1, ["rss"]):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "host memory scope"):
                self.rule(parts(), scope)
            with self.subTest(declared=scope), patch.object(soak, "HOST_MEMORY_SCOPE", scope), \
                    self.assertRaisesRegex(ValueError, "host memory scope"):
                soak.memory_rule(parts(), 1200, LIMITS)
        for scope in ("vmrss", "anonymous_shmem"):
            with self.subTest(argv=scope), patch("sys.argv", ["soak", "--host-memory-scope", scope, "--server-pid",
                                                              "123"]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as exited:
                soak.main()
            self.assertEqual(exited.exception.code, 2)


# The declared files of a server that maps two read-only tables, each as a
# large view and a small one, in the shape of a 2026-11-05 C32 GPU soak.
TABLES = ["tables/table-a.bin", "tables/table-b.bin"]
REAL = {"read_text": Path.read_text, "read_bytes": Path.read_bytes, "readlink": os.readlink}


def server_mappings(store, exempt=0, driver=0, anon=0, other=0, writable=0, later=None, protected=False):
    """The server's mappings as (range, perms, offset, path, Size kB, Rss
    kB): the four table views under `store` (the first grown by `exempt`
    kB), the driver's control node (pinned host buffers, grown by `driver`),
    anonymous memory (`anon`), an undeclared checkpoint file (`other`), a
    writable mapping of a declared file (`writable`) and, when `later` is
    not None, a mapping of a declared file the first sample did not see.
    `protected` makes the first table view writable, as mprotect would. The
    table views, libc and their Size and Rss are a 2026-11-05 GPU soak's;
    each other mapping spans 1 GiB (the anonymous one 8 GiB), room for the
    tests' growth."""
    table_a, table_b = (f"{store}/{name}" for name in TABLES)
    rows = [("74404d400000-745730b82000", "rw-s" if protected else "r--s", "00000000", table_a, 96001544,
             3905472 + exempt),
            ("745730d81000-7457e7f3e000", "r--s", "16e3781000", table_a, 3000052, 2672428),
            ("7428b2600000-743f96013000", "r--s", "00000000", table_b, 96004172, 3904596),
            ("743f96212000-74404d3e4000", "r--s", "16e3a12000", table_b, 3000136, 2612208),
            ("700000000000-700040000000", "rw-s", "00000000", "/dev/nvidiactl", 1048576, 223232 + driver),
            ("700100000000-700300000000", "rw-p", "00000000", "", 8388608, 1294020 + anon),
            ("767c89028000-767c891b1000", "r-xp", "00028000", "/usr/lib/libc.so.6", 1572, 1120),
            ("700400000000-700440000000", "r--p", "00000000", f"{store}/weights.bin", 1048576, 1000 + other),
            ("700500000000-700540000000", "rw-s", "00000000", table_b, 1048576, 1000 + writable)]
    if later is not None:
        rows.append(("700600000000-700640000000", "r--s", "00000000", table_a, 1048576, later))
    return rows


class FakeProc:
    """/proc of a serving tree: a wrapper (pid 100) whose command line holds
    the server's, and the server (pid 101), started in `cwd`. `mappings(n)`
    gives the server's mappings at the nth sample (the nth listing of the
    tree); `unreadable(n)` the pids whose smaps cannot be read then."""

    def __init__(self, cwd, mappings, argv=("chopinfer", "serve", "--model", "checkpoint"),
                 unreadable=lambda n: ()):
        self.cwd, self.mappings, self.argv, self.unreadable = cwd, mappings, list(argv), unreadable
        self.samples = -1
        self.smaps_reads = 0

    def rows(self, pid):
        if pid == 100:
            return [("55bc8aed4000-55bc8aed9000", "r-xp", "00002000", "/usr/bin/timeout", 20, 20)]
        return self.mappings(self.samples)

    def tree(self, root_pid):
        self.samples += 1
        return {100, 101}

    def status(self, pid):
        rows = self.rows(pid)
        anon = sum(rss for *_, path, size, rss in rows if not path)
        total = sum(rss for *_, rss in rows)
        return f"VmRSS:\t{total} kB\nRssAnon:\t{anon} kB\nRssFile:\t{total - anon} kB\nRssShmem:\t0 kB\n"

    def smaps(self, pid):
        self.smaps_reads += 1
        if pid in self.unreadable(self.samples):
            raise PermissionError(13, "Permission denied", f"/proc/{pid}/smaps")
        lines = []
        for span, perms, offset, path, size, rss in self.rows(pid):
            device, inode = ("00:00", 0) if not path else ("103:01", sum(map(ord, path)))
            lines += [f"{span} {perms} {offset} {device} {inode}" + (f"                 {path}" if path else ""),
                      f"Size:           {size:8} kB", "KernelPageSize:        4 kB",
                      f"Rss:            {rss:8} kB", f"Pss:            {rss:8} kB",
                      f"Private_Dirty:  {rss:8} kB", "Anonymous:             0 kB", "VmFlags: rd mr me sd"]
        return ("\n".join(lines) + "\n").encode()

    def read_text(self, path, *args, **kwargs):
        parts = str(path).split("/")
        if parts[:2] == ["", "proc"] and parts[2] in ("100", "101") and parts[3] == "status":
            return self.status(int(parts[2]))
        return REAL["read_text"](path, *args, **kwargs)

    def read_bytes(self, path):
        parts = str(path).split("/")
        if parts[:2] == ["", "proc"] and parts[2] in ("100", "101"):
            if parts[3] == "smaps":
                return self.smaps(int(parts[2]))
            if parts[3] == "cmdline":
                argv = ["/usr/bin/timeout", "0s", *self.argv] if parts[2] == "100" else self.argv
                return "".join(word + "\0" for word in argv).encode()
        return REAL["read_bytes"](path)

    def readlink(self, path, *args, **kwargs):
        if str(path) in ("/proc/100/cwd", "/proc/101/cwd"):
            return self.cwd
        return REAL["readlink"](path, *args, **kwargs)

    @contextmanager
    def mounted(self):
        with patch.object(soak, "process_tree", side_effect=self.tree), \
                patch.object(soak.Path, "read_text", autospec=True, side_effect=self.read_text), \
                patch.object(soak.Path, "read_bytes", autospec=True, side_effect=self.read_bytes), \
                patch.object(soak.os, "readlink", side_effect=self.readlink):
            yield self


class MappedViewScope(unittest.TestCase):
    """The rss_less_mapped_views scope judges VmRSS less the Rss of the
    declared files' read-only mappings that the serving tree holds at the
    first sample, attributed per mapping from smaps; every other mapping
    counts."""
    SCOPE = "rss_less_mapped_views"
    # 5,425 MiB of the tables' pages becoming resident over the soak.
    RESIDENT = 5425 * 1024

    def setUp(self):
        base = Path(os.path.realpath(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, base)
        # The server names its checkpoint relative to its working directory,
        # through a symbolic link; smaps names the files it resolves to.
        (base / "store").mkdir()
        (base / "work").mkdir()
        (base / "work" / "checkpoint").symlink_to(base / "store")
        self.store, self.cwd = str(base / "store"), str(base / "work")

    def proc(self, grow=lambda t: {}, **options):
        return FakeProc(self.cwd, lambda n: server_mappings(self.store, **grow(30 * n)), **options)

    def sampled(self, grow=lambda t: {}, files=TABLES, **options):
        """Samples every 30 s over 1200 s of a server whose mappings grow by
        `grow(t)` (server_mappings options), with the soak's MappedViews."""
        proc, views = self.proc(grow, **options), soak.MappedViews(list(files))
        with proc.mounted():
            rows = [{"elapsed_seconds": t, "mib": 1000, **soak.host_memory(100, views)} for t in range(0, 1201, 30)]
        return rows, views

    def rule(self, rows, scope=SCOPE):
        return soak.memory_rule(rows, 1200, LIMITS, 0, None, scope)

    def test_the_fake_mappings_are_realizable(self):
        # At the largest growth any test applies, each mapping's Size is its
        # address range and holds its Rss, and no two mappings overlap.
        grown = {"exempt": self.RESIDENT, **dict.fromkeys(("driver", "anon", "other", "writable", "later"), 600 * 1024)}
        for options in ({}, grown, {"protected": True}):
            with self.subTest(options=options):
                spans = []
                for span, perms, offset, path, size, rss in server_mappings(self.store, **options):
                    low, high = (int(edge, 16) for edge in span.split("-"))
                    self.assertEqual((high - low, rss <= size), (size * 1024, True), span)
                    spans.append((low, high))
                spans.sort()
                self.assertTrue(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), spans)

    def test_growth_of_the_exempt_mappings_passes(self):
        rows, views = self.sampled(lambda t: {"exempt": t * self.RESIDENT // 1200})
        rule = self.rule(rows)
        bound = (96001544 + 3000052 + 96004172 + 3000136) / 1024
        self.assertEqual((rule["passed"], rule["host_memory_scope"], rule["host_memory_growth_mib"],
                          rule["steady_host_memory_growth_mib"], rule["host_exempt_rss_mib"],
                          rule["host_exempt_bound_mib"]),
                         (True, self.SCOPE, 0, 0, (3905472 + 2672428 + 3904596 + 2612208 + self.RESIDENT) / 1024,
                          bound))
        for row in rows:
            self.assertEqual(row["host_judged_mib"], row["host_rss_mib"] - row["host_exempt_rss_mib"])
            self.assertEqual(row["host_exempt_bound_mib"], bound)
        # The server's four read-only table views, not the wrapper whose
        # command line names the same checkpoint, nor an undeclared file.
        self.assertEqual([(row["pid"], row["path"], row["perms"], row["offset"]) for row in views.mappings],
                         [(101, f"{self.store}/{TABLES[0]}", "r--s", "00000000"),
                          (101, f"{self.store}/{TABLES[0]}", "r--s", "16e3781000"),
                          (101, f"{self.store}/{TABLES[1]}", "r--s", "00000000"),
                          (101, f"{self.store}/{TABLES[1]}", "r--s", "16e3a12000")])
        # Judged by VmRSS the same soak fails.
        self.assertEqual((self.rule(rows, "rss")["passed"], self.rule(rows, "rss")["host_memory_growth_mib"]),
                         (False, 5425))

    def test_growth_in_the_driver_node_or_another_file_mapping_counts(self):
        for part in ("driver", "other", "anon"):
            with self.subTest(part):
                rule = self.rule(self.sampled(lambda t: {part: t * 600 * 1024 // 1200})[0])
                self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (600, False))
                rule = self.rule(self.sampled(lambda t: {part: t * 500 * 1024 // 1200,
                                                          "exempt": t * self.RESIDENT // 1200})[0])
                self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (500, True))

    def test_a_mapping_of_a_declared_file_after_the_first_sample_counts(self):
        rows, views = self.sampled(lambda t: {"later": t * 600 * 1024 // 1200} if t else {})
        rule = self.rule(rows)
        self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (600, False))
        self.assertEqual(len(views.mappings), 4)

    def test_a_writable_mapping_of_a_declared_file_is_not_exempt(self):
        rows, views = self.sampled(lambda t: {"writable": t * 600 * 1024 // 1200})
        self.assertEqual((self.rule(rows)["host_memory_growth_mib"], self.rule(rows)["passed"]), (600, False))
        self.assertNotIn("rw-s", {row["perms"] for row in views.mappings})
        # An exempt mapping made writable after the first sample counts from then on.
        rows, _ = self.sampled(lambda t: {"protected": t >= 600})
        rule = self.rule(rows)
        self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (3905472 / 1024, False))

    def test_an_unreadable_smaps_leaves_its_sample_without_evidence(self):
        rows, _ = self.sampled(unreadable=lambda n: (100,) if n == 7 else ())
        self.assertEqual([index for index, row in enumerate(rows) if row["host_judged_mib"] is None], [7])
        self.assertEqual((rows[7]["host_exempt_rss_mib"], rows[7]["host_exempt_bound_mib"]), (None, None))
        self.assertIn("process 100 (PermissionError", rows[7]["host_exempt_error"])
        self.assertEqual(rows[7]["host_rss_mib"], rows[6]["host_rss_mib"])
        with self.assertRaisesRegex(ValueError, "at 210 s has no host_exempt_rss_mib, host_exempt_bound_mib, "
                                                "host_judged_mib .*PermissionError"):
            self.rule(rows)
        # Unread at the first sample, the exempt mappings are unknown for
        # the whole soak.
        rows, views = self.sampled(unreadable=lambda n: (101,) if n == 0 else ())
        self.assertTrue(all(row["host_judged_mib"] is None for row in rows))
        self.assertTrue(all(row["host_exempt_error"].startswith("the first sample identified no exempt mappings")
                            for row in rows))
        self.assertEqual(views.mappings, [])
        with self.assertRaisesRegex(ValueError, "at 0 s has no"):
            self.rule(rows)

    def test_the_served_checkpoint_directory_must_be_named_once(self):
        for argv, exempt in ((("chopinfer", "serve", "--bundle", "package", "--checkpoint", "checkpoint"), 4),
                             (("chopinfer", "serve", "--model", self.store), 4),
                             (("chopinfer", "serve", "--model", f"{self.cwd}/elsewhere"), 0),
                             (("chopinfer", "serve", "--listen", "127.0.0.1:1"), None),
                             (("chopinfer", "serve", "--model", "checkpoint", "--checkpoint", "checkpoint"), None),
                             (("chopinfer", "serve", "--model"), None),
                             (("chopinfer", "verify", "--model", "checkpoint"), None)):
            with self.subTest(argv=argv):
                rows, views = self.sampled(argv=argv)
                if exempt is None:
                    self.assertIn("checkpoint director", rows[0]["host_exempt_error"])
                    self.assertTrue(all(row["host_judged_mib"] is None for row in rows))
                else:
                    self.assertEqual(len(views.mappings), exempt)
                    self.assertNotIn("host_exempt_error", rows[0])

    def test_a_sample_without_consistent_values_is_insufficient(self):
        keys = ("host_rss_mib", "host_exempt_rss_mib", "host_exempt_bound_mib", "host_judged_mib")
        for key in keys:
            for value in (None, float("nan"), float("inf"), -1, True, "1", "absent"):
                rows, _ = self.sampled()
                if value == "absent":
                    del rows[5][key]
                else:
                    rows[5][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, f"has no .*{key}"):
                    self.rule(rows)
        rows, _ = self.sampled()
        for key, value in (("host_judged_mib", rows[5]["host_judged_mib"] - 600),
                           ("host_exempt_rss_mib", rows[5]["host_exempt_bound_mib"] + 1),
                           ("host_exempt_bound_mib", rows[5]["host_exempt_rss_mib"] - 1)):
            changed = [dict(row) for row in rows]
            changed[5][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "beyond its bound"):
                self.rule(changed)
        # The judged series is VmRSS less the exempt Rss: a lower recorded
        # judged value cannot hide growth from the rule.
        rows, _ = self.sampled(lambda t: {"anon": t * 600 * 1024 // 1200})
        self.assertFalse(self.rule(rows)["passed"])

    def test_the_declared_files_belong_to_the_scope(self):
        self.assertEqual(soak.mapped_view_files(TABLES, self.SCOPE), TABLES)
        self.assertEqual(soak.mapped_view_files(), [])
        for files, scope in (([], self.SCOPE), (TABLES, "rss"), (["/abs.bin"], self.SCOPE),
                             (["../up.bin"], self.SCOPE), (["a//b.bin"], self.SCOPE), (["a/./b.bin"], self.SCOPE),
                             (["a/"], self.SCOPE), ([""], self.SCOPE), (["a.bin", "a.bin"], self.SCOPE),
                             ([1], self.SCOPE), ("a.bin", self.SCOPE), ([["a.bin"]], self.SCOPE)):
            with self.subTest(files=files, scope=scope), self.assertRaises(ValueError):
                soak.mapped_view_files(files, scope)
        for argv in (["--host-memory-scope", self.SCOPE], ["--mapped-view-file", TABLES[0]],
                     ["--host-memory-scope", self.SCOPE, "--mapped-view-file", "/abs.bin"]):
            with self.subTest(argv=argv), patch("sys.argv", ["soak", "--server-pid", "100", *argv]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as exited:
                soak.main()
            self.assertEqual(exited.exception.code, 2)

    def soak(self, proc, argv=()):
        with patch("sys.argv", ["soak", "--duration", "0.3", "--concurrency", "2", "--server-pid", "100", *argv]), \
                proc.mounted(), patch.object(soak, "Client", FakeClient), \
                patch.object(soak, "device_memory", return_value=350), \
                patch.object(soak, "metrics_text", return_value="chop_mapped_view_bytes 202758045696\n"), \
                patch.object(soak, "receipt") as receipt, self.assertRaises(SystemExit) as exited:
            soak.main()
        return exited.exception.code, receipt.call_args.args[2]

    def test_the_soak_records_the_scope_and_its_exempt_mappings(self):
        argv = ["--host-memory-scope", self.SCOPE, *[word for name in TABLES for word in ("--mapped-view-file", name)]]
        # The tables' resident pages grow 600 MiB a sample.
        grow = lambda t: {"exempt": t // 30 * 600 * 1024}
        for declared in ({"argv": argv}, {}):
            with self.subTest(declared=bool(declared)), \
                    patch.object(soak, "HOST_MEMORY_SCOPE", "rss" if declared else self.SCOPE), \
                    patch.object(soak, "MAPPED_VIEW_FILES", () if declared else tuple(TABLES)):
                code, report = self.soak(self.proc(grow), **declared)
                self.assertEqual((code, report["passed"], report["errors"]), (0, True, []))
                self.assertEqual((report["host_memory_scope"], report["mapped_view_files"],
                                  report["host_memory_growth_mib"], len(report["host_exempt_mappings"])),
                                 (self.SCOPE, TABLES, 0, 4))
                samples = report["memory_samples"]
                self.assertGreaterEqual(len(samples), 2)
                self.assertEqual(report["host_exempt_rss_mib"], max(row["host_exempt_rss_mib"] for row in samples))
                self.assertGreater(samples[-1]["host_exempt_rss_mib"], samples[0]["host_exempt_rss_mib"])
                self.assertTrue(all(row["host_judged_mib"] == samples[0]["host_judged_mib"] for row in samples))
        # The default scope reads no smaps, records none of it, and fails the
        # same soak on its VmRSS growth.
        proc = self.proc(grow)
        code, report = self.soak(proc)
        self.assertEqual((code, proc.smaps_reads), (1, 0))
        self.assertFalse({"host_memory_scope", "mapped_view_files", "host_exempt_mappings", "host_exempt_rss_mib",
                          "host_exempt_bound_mib"} & set(report))
        self.assertFalse(any("host_judged_mib" in row for row in report["memory_samples"]))

    def test_an_unreadable_smaps_fails_the_soak_and_keeps_its_receipt(self):
        argv = ["--host-memory-scope", self.SCOPE, "--mapped-view-file", TABLES[0]]
        code, report = self.soak(self.proc(unreadable=lambda n: (101,) if n == 1 else ()), argv)
        self.assertEqual((code, report["passed"]), (1, False))
        self.assertGreaterEqual(len(report["memory_samples"]), 2)
        self.assertEqual([list(error) for error in report["errors"]], [["memory"]])
        self.assertIn("has no host_exempt_rss_mib", report["errors"][0]["memory"])
        self.assertEqual(report["host_memory_scope"], self.SCOPE)



if __name__ == "__main__":
    unittest.main()
