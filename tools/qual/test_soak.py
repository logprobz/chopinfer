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


def server_mappings(store, exempt=0, driver=0, anon=0, other=0, writable=0, later=None, protected=False, gone=False,
                    first_perms=None):
    """The server's mappings as (range, perms, offset, path, Size kB, Rss
    kB): the four table views under `store` (the first grown by `exempt`
    kB), the driver's control node (pinned host buffers, grown by `driver`),
    anonymous memory (`anon`), an undeclared checkpoint file (`other`), a
    writable mapping of a declared file (`writable`) and, when `later` is
    not None, a mapping of a declared file the first sample did not see.
    `protected` makes the first table view writable, as mprotect would,
    `first_perms` gives it other permissions, as a replacement mmap at its
    address could, and `gone` leaves it out, as munmap would. The
    table views, libc and their Size and Rss are a 2026-11-05 GPU soak's;
    each other mapping spans 1 GiB (the anonymous one 8 GiB), room for the
    tests' growth."""
    table_a, table_b = (f"{store}/{name}" for name in TABLES)
    rows = [("74404d400000-745730b82000", "rw-s" if protected else first_perms or "r--s", "00000000",
             table_a, 96001544, 3905472 + exempt),
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
    return rows[1:] if gone else rows


def write_clean(path, data):
    """Write a file and flush it to disk, so that its page cache, as a
    checkpoint's long after it was written, is clean."""
    with open(path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class FakeProc:
    """/proc of a serving tree: a wrapper (pid 100) whose command line holds
    the server's, and the server (pid 101), started in `cwd`. `mappings(n)`
    gives the server's mappings at the nth sample (the nth listing of the
    tree), `start(n)` its start time; `unreadable(n)` the pids whose smaps
    cannot be read then; `counters(n, range)` a mapping's Anonymous,
    Private_Dirty, Shared_Dirty or Swap kB then, where they are not those
    of its kind. A mapped file that exists here has its device and inode,
    unless it is in `foreign` (another file object under its path); the
    processes' root is this one's."""

    def __init__(self, cwd, mappings, argv=("chopinfer", "serve", "--model", "checkpoint"),
                 unreadable=lambda n: (), start=lambda n: 5001, foreign=(), counters=lambda n, span: {}):
        self.cwd, self.mappings, self.argv, self.unreadable = cwd, mappings, list(argv), unreadable
        self.start, self.foreign, self.counters = start, set(foreign), counters
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
            if path in self.foreign:
                device, inode = "00:05", 999
            elif path and os.path.isfile(path):
                status = os.stat(path)
                device, inode = f"{os.major(status.st_dev):02x}:{os.minor(status.st_dev):02x}", status.st_ino
            # Anonymous and writable mappings hold dirty pages; read-only file
            # mappings clean page cache, unless the sample says otherwise.
            dirty = rss if not path or "w" in perms else 0
            counters = {"Shared_Dirty": 0, "Private_Dirty": dirty, "Anonymous": rss if not path else 0, "Swap": 0,
                        **self.counters(self.samples, span)}
            lines += [f"{span} {perms} {offset} {device} {inode}" + (f"                 {path}" if path else ""),
                      f"Size:           {size:8} kB", "KernelPageSize:        4 kB", "MMUPageSize:           4 kB",
                      f"Rss:            {rss:8} kB", f"Pss:            {rss:8} kB",
                      f"Shared_Clean:   {0:8} kB", f"Shared_Dirty:   {counters['Shared_Dirty']:8} kB",
                      f"Private_Clean:  {rss - dirty:8} kB", f"Private_Dirty:  {counters['Private_Dirty']:8} kB",
                      f"Anonymous:      {counters['Anonymous']:8} kB", f"Swap:           {counters['Swap']:8} kB",
                      f"SwapPss:        {counters['Swap'] + 7:8} kB", "VmFlags: rd mr me sd"]
        return ("\n".join(lines) + "\n").encode()

    def stat(self, pid):
        # Field 22, the start time, after a command name holding spaces and
        # parentheses.
        start = 5000 if pid == 100 else self.start(self.samples)
        return f"{pid} (serve (x) y) S 1 {pid} {pid} 0 -1 4194560 {' '.join(['0'] * 8)} 20 0 1 0 {start} 0 0\n"

    def read_text(self, path, *args, **kwargs):
        parts = str(path).split("/")
        if parts[:2] == ["", "proc"] and parts[2] in ("100", "101") and parts[3] in ("status", "stat"):
            return (self.status if parts[3] == "status" else self.stat)(int(parts[2]))
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
                patch.object(soak, "process_root", return_value="/"), \
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
        (base / "store" / "tables").mkdir()
        for name in TABLES:
            (base / "store" / name).write_bytes(b"table")

    @contextmanager
    def serving_this_process(self):
        """This process as the serving tree, its smaps, stat and status read
        from /proc, with a command line that serves the test checkpoint."""
        pid = os.getpid()
        argv = b"".join(word + b"\0" for word in (b"chopinfer", b"serve", b"--model", os.fsencode(self.store)))
        with patch.object(soak, "process_tree", return_value={pid}), \
                patch.object(soak.Path, "read_bytes", autospec=True,
                             side_effect=lambda path: argv if str(path) == f"/proc/{pid}/cmdline"
                             else REAL["read_bytes"](path)):
            yield pid

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

    @staticmethod
    def listed(row):
        """A sample's exempt mappings as (PID, range, permissions, offset),
        the integers spelled as smaps writes them."""
        return [(entry["pid"], "%x-%x" % tuple(entry["range"]), entry["perms"], "%08x" % entry["offset"])
                for entry in row["host_exempt_mappings"]]

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
        # Every sample lists the server's four read-only table views, not
        # the wrapper whose command line names the same checkpoint, nor an
        # undeclared file, with their Size and Rss.
        tables = [(101, "7428b2600000-743f96013000", "r--s", "00000000"),
                  (101, "743f96212000-74404d3e4000", "r--s", "16e3a12000"),
                  (101, "74404d400000-745730b82000", "r--s", "00000000"),
                  (101, "745730d81000-7457e7f3e000", "r--s", "16e3781000")]
        self.assertTrue(all(self.listed(row) == tables for row in rows))
        self.assertEqual([(entry["start_time"], entry["size_kib"], entry["rss_kib"])
                          for entry in rows[-1]["host_exempt_mappings"]],
                         [(5001, 96004172, 3904596), (5001, 3000136, 2612208), (5001, 96001544, 3905472 + self.RESIDENT),
                          (5001, 3000052, 2672428)])
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
        rows, _ = self.sampled(lambda t: {"later": t * 600 * 1024 // 1200} if t else {})
        rule = self.rule(rows)
        self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (600, False))
        self.assertEqual(len(rows[-1]["host_exempt_mappings"]), 4)

    def test_a_writable_mapping_of_a_declared_file_is_not_exempt(self):
        rows, _ = self.sampled(lambda t: {"writable": t * 600 * 1024 // 1200})
        self.assertEqual((self.rule(rows)["host_memory_growth_mib"], self.rule(rows)["passed"]), (600, False))
        self.assertEqual({entry["perms"] for entry in rows[0]["host_exempt_mappings"]}, {"r--s"})
        # An exempt mapping made writable after the first sample counts from then on.
        rows, _ = self.sampled(lambda t: {"protected": t >= 600})
        rule = self.rule(rows)
        self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (3905472 / 1024, False))

    def test_an_exempt_mapping_a_sample_lacks_or_holds_writable_is_retired(self):
        # Unmapped, or made writable, at 600 s, then read-only again with the
        # same identity and 600 MiB more resident: it counts from 600 s on.
        for change in ("gone", "protected"):
            with self.subTest(change):
                rows, _ = self.sampled(lambda t: {change: t == 600, "exempt": 600 * 1024 if t > 600 else 0})
                self.assertEqual(rows[-1]["host_judged_mib"] - rows[0]["host_judged_mib"],
                                 (3905472 + 600 * 1024) / 1024)
                self.assertEqual(rows[-1]["host_exempt_rss_mib"], (2672428 + 3904596 + 2612208) / 1024)
                self.assertFalse(self.rule(rows)["passed"])

    def test_a_view_that_turns_private_dirty_or_otherwise_permitted_is_retired(self):
        view = "74404d400000-745730b82000"
        views = (3905472 + 2672428 + 3904596 + 2612208) / 1024
        # Replaced at 600 s by a private mapping with 600 MiB of dirty
        # anonymous pages, then read-only again: it counts from then on.
        rows, _ = self.sampled(lambda t: {"first_perms": "r--p" if t >= 600 else None,
                                          "exempt": 600 * 1024 if t >= 600 else 0},
                               counters=lambda n, span: {"Anonymous": 600 * 1024, "Private_Dirty": 600 * 1024}
                               if span == view and n >= 20 else {})
        self.assertEqual(rows[-1]["host_judged_mib"] - rows[0]["host_judged_mib"], (3905472 + 600 * 1024) / 1024)
        self.assertEqual(rows[-1]["host_exempt_rss_mib"], views - 3905472 / 1024)
        self.assertFalse(self.rule(rows)["passed"])
        # Dirty, anonymous or swapped pages in it, or other permissions, at
        # 600 s alone retire it as well.
        for label, options in [*((counter, {"counters": lambda n, span, counter=counter: {counter: 4}
                                            if span == view and n == 20 else {}})
                                 for counter in ("Anonymous", "Private_Dirty", "Shared_Dirty", "Swap")),
                               ("r-xs", {"grow": lambda t: {"first_perms": "r-xs" if t == 600 else None}})]:
            with self.subTest(label):
                rows, _ = self.sampled(options.pop("grow", lambda t: {}), **options)
                self.assertEqual([row["host_exempt_rss_mib"] for row in rows[19:22]],
                                 [views, views - 3905472 / 1024, views - 3905472 / 1024])
                self.assertEqual(rows[-1]["host_judged_mib"] - rows[0]["host_judged_mib"], 3905472 / 1024)

    def test_a_first_sample_view_that_is_private_or_dirty_is_never_exempt(self):
        view = "74404d400000-745730b82000"
        for label, options in (("private", {"grow": lambda t: {"first_perms": "r--p"}}),
                               ("dirty", {"counters": lambda n, span: {"Shared_Dirty": 4} if span == view and n == 0
                                          else {}})):
            with self.subTest(label):
                rows, _ = self.sampled(options.pop("grow", lambda t: {}), **options)
                self.assertTrue(all(view not in {span for _, span, _, _ in self.listed(row)} for row in rows))
                self.assertEqual(len(rows[-1]["host_exempt_mappings"]), 3)
        # Each listed view records the counters the rule requires to be 0.
        rows, _ = self.sampled()
        self.assertTrue(all((entry["anonymous_kib"], entry["private_dirty_kib"], entry["shared_dirty_kib"],
                             entry["swap_kib"]) == (0, 0, 0, 0) for row in rows for entry in row["host_exempt_mappings"]))

    def test_a_real_view_replaced_by_a_private_dirty_one_counts(self):
        # This process maps 8 MiB of a declared file shared and read-only,
        # then, between two samples, maps it again at the same address,
        # offset and file, privately, fills it and makes it read-only again:
        # the same identity in smaps, now 8 MiB of anonymous dirty pages.
        # A clean shared read-only replacement is the case no sample can
        # tell apart: the same file's clean page cache at the same address.
        import ctypes
        import mmap
        size = 8 * 2 ** 20
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long)
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        path = Path(self.store) / "table.bin"
        write_clean(path, b"x" * size)
        descriptor = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, descriptor)
        for label, protection, flags, exempt in (("private dirty", mmap.PROT_READ | mmap.PROT_WRITE,
                                                  mmap.MAP_PRIVATE, 0),
                                                 ("clean shared", mmap.PROT_READ, mmap.MAP_SHARED, size)):
            with self.subTest(label):
                views = soak.MappedViews(["table.bin"])
                pointer = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
                self.assertNotEqual(pointer, ctypes.c_void_p(-1).value)
                ctypes.string_at(pointer, size)
                with self.serving_this_process() as pid:
                    first = soak.host_memory(pid, views)
                self.assertEqual(libc.munmap(pointer, size), 0)
                # MAP_FIXED_NOREPLACE: the same address, or a failure.
                again = libc.mmap(pointer, size, protection, flags | 0x100000, descriptor, 0)
                self.assertEqual(again, pointer, os.strerror(ctypes.get_errno()))
                if protection & mmap.PROT_WRITE:
                    ctypes.memset(again, 7, size)
                    self.assertEqual(libc.mprotect(again, size, mmap.PROT_READ), 0)
                else:
                    ctypes.string_at(again, size)
                with self.serving_this_process() as pid:
                    later = soak.host_memory(pid, views)
                self.assertEqual(libc.munmap(again, size), 0)
                self.assertEqual((first["host_exempt_rss_mib"], later["host_exempt_rss_mib"]),
                                 (size / 2 ** 20, exempt / 2 ** 20))

    def test_smaps_identities_are_integers_and_never_overlap(self):
        def parsed(text):
            with patch.object(soak.Path, "read_bytes", return_value=text.encode()):
                return soak.smaps(7)
        counters = "".join(f"{name}: 0 kB\n" for name in ("Anonymous", "Private_Dirty", "Shared_Dirty", "Swap"))
        mapping = lambda header, size=1024: f"{header}\nSize: {size} kB\nRss: 4 kB\n{counters}"
        view = "7428b2600000-7428b2700000 r--s 16e3a12000 103:01 11"
        self.assertEqual([(entry["key"], entry["perms"], entry["path"])
                          for entry in parsed(mapping(f"{view}                 /checkpoint/table a.bin")
                                              + mapping("7428b2700000-7428b2701000 rw-p 00000000 00:00 0 ", 4))],
                         [((0x7428b2600000, 0x7428b2700000, 0x103, 1, 11, 0x16e3a12000), "r--s",
                           "/checkpoint/table a.bin"),
                          ((0x7428b2700000, 0x7428b2701000, 0, 0, 0, 0), "rw-p", "")])
        for label, second in (("the same identity", view),
                              ("a padded spelling of it", "007428b2600000-007428b2700000 r--s 0016e3a12000 0103:001 11"),
                              ("an overlapping range at another offset", "7428b26ff000-7428b27ff000 r--s 00000000 103:01 11"),
                              ("an empty range", "7428b2800000-7428b2800000 r--s 00000000 103:01 11")):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "empty or overlapping address range"):
                parsed(mapping(view) + mapping(second))
        for header in ("7428b2600000 r--s 00000000 103:01 11", "7428b2600000-7428b2700000 r--s 0x0 103:01 11",
                       "7428b2600000-7428b2700000 r--s 00000000 103 11", "7428b2600000-7428b2700000 r--s 0 1:1 -1"):
            with self.subTest(header), self.assertRaisesRegex(ValueError, "malformed mapping line"):
                parsed(mapping(header))

    def test_an_alias_of_an_exempt_view_leaves_the_sample_without_evidence(self):
        # From 600 s smaps also holds a table view at the same address,
        # offset and file, spelled with leading zeros, or a view of that
        # file overlapping it at another offset; the soak does not exempt
        # either twice, nor exempt the process's views again.
        for label, row in (("padded", ("0074404d400000-00745730b82000", "r--s", "0000000000")),
                           ("overlapping", ("74404d401000-74404d501000", "r--s", "00001000"))):
            with self.subTest(label):
                def extra(n, row=row):
                    return [(*row, f"{self.store}/{TABLES[1]}", 1024, 1024)] if n >= 20 else []
                proc = FakeProc(self.cwd, lambda n: [*server_mappings(self.store), *extra(n)])
                views = soak.MappedViews(list(TABLES))
                with proc.mounted():
                    rows = [{"elapsed_seconds": t, "mib": 1000, **soak.host_memory(100, views)}
                            for t in range(0, 1201, 30)]
                self.assertEqual(rows[19]["host_exempt_rss_mib"], (3905472 + 2672428 + 3904596 + 2612208) / 1024)
                self.assertTrue(all(row["host_judged_mib"] is None for row in rows[20:]))
                self.assertIn("process 101 (ValueError: /proc/101/smaps has an empty or overlapping address range)",
                              rows[20]["host_exempt_error"])
                with self.assertRaisesRegex(ValueError, "at 600 s has no"):
                    self.rule(rows)

    def test_a_restarted_server_or_a_recycled_pid_counts(self):
        # The same PID and mappings, but another process from 600 s on.
        rows, _ = self.sampled(start=lambda n: 5001 if n < 20 else 9001)
        self.assertEqual([row["host_exempt_rss_mib"] for row in rows[19:21]],
                         [(3905472 + 2672428 + 3904596 + 2612208) / 1024, 0])
        self.assertEqual(rows[-1]["host_judged_mib"], rows[-1]["host_rss_mib"])
        self.assertFalse(self.rule(rows)["passed"])

    def test_a_real_mapping_unmapped_and_mapped_again_counts(self):
        # This process maps a declared file, unmaps it and maps it again at
        # the same address, offset and file: the same identity in smaps.
        import ctypes
        import mmap
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long)
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        path = Path(self.store) / "table.bin"
        write_clean(path, b"x" * mmap.PAGESIZE)
        descriptor = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, descriptor)

        def mapped(address, flags):
            pointer = libc.mmap(address, mmap.PAGESIZE, mmap.PROT_READ, flags, descriptor, 0)
            self.assertNotEqual(pointer, ctypes.c_void_p(-1).value, os.strerror(ctypes.get_errno()))
            self.assertEqual(ctypes.string_at(pointer, 1), b"x")
            return pointer

        views = soak.MappedViews(["table.bin"])

        def sample():
            with self.serving_this_process() as pid:
                return soak.host_memory(pid, views)

        pointer = mapped(None, mmap.MAP_SHARED)
        first = sample()
        self.assertEqual(libc.munmap(pointer, mmap.PAGESIZE), 0)
        absent = sample()
        # MAP_FIXED_NOREPLACE: the same address, or a failure.
        again = mapped(pointer, mmap.MAP_SHARED | 0x100000)
        self.addCleanup(libc.munmap, again, mmap.PAGESIZE)
        self.assertEqual(again, pointer)
        later = sample()
        self.assertEqual([row["host_exempt_rss_mib"] for row in (first, absent, later)],
                         [mmap.PAGESIZE / 2 ** 20, 0, 0])

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
        rows, _ = self.sampled(unreadable=lambda n: (101,) if n == 0 else ())
        self.assertTrue(all(row["host_judged_mib"] is None and row["host_exempt_mappings"] is None for row in rows))
        self.assertTrue(all(row["host_exempt_error"].startswith("the first sample identified no exempt mappings")
                            for row in rows))
        with self.assertRaisesRegex(ValueError, "at 0 s has no"):
            self.rule(rows)

    def test_the_served_checkpoint_directory_must_be_named_once(self):
        for argv, exempt in ((("chopinfer", "serve", "--bundle", "package", "--checkpoint", "checkpoint"), 4),
                             (("chopinfer", "serve", "--model", self.store), 4),
                             (("chopinfer", "serve", "--model", f"{self.cwd}/elsewhere"), "No such file"),
                             (("chopinfer", "serve", "--listen", "127.0.0.1:1"), "checkpoint director"),
                             (("chopinfer", "serve", "--model", "checkpoint", "--checkpoint", "checkpoint"),
                              "checkpoint director"),
                             (("chopinfer", "serve", "--model"), "checkpoint director"),
                             (("chopinfer", "verify", "--model", "checkpoint"), "checkpoint director")):
            with self.subTest(argv=argv):
                rows, views = self.sampled(argv=argv)
                if isinstance(exempt, str):
                    self.assertIn(exempt, rows[0]["host_exempt_error"])
                    self.assertTrue(all(row["host_judged_mib"] is None for row in rows))
                else:
                    self.assertEqual(len(rows[0]["host_exempt_mappings"]), exempt)
                    self.assertEqual([identity[:2] for identity in views.identities],
                                     [(name, f"{self.store}/{name}") for name in TABLES])
                    self.assertNotIn("host_exempt_error", rows[0])

    def test_a_declared_file_must_be_a_regular_file(self):
        # The declared table is a symbolic link to a device node whose
        # read-only mapping holds 600 MiB.
        for node in [node for node in ("/dev/nvidiactl", "/dev/null") if os.path.exists(node)]:
            table = Path(self.store) / TABLES[0]
            table.unlink()
            table.symlink_to(node)
            node_view = ("700700000000-700740000000", "r--s", "00000000", node, 1048576, 600 * 1024)
            proc = FakeProc(self.cwd, lambda n: [*server_mappings(self.store), node_view])
            views = soak.MappedViews(list(TABLES))
            try:
                with self.subTest(node), proc.mounted():
                    rows = [{"elapsed_seconds": t, "mib": 1000, **soak.host_memory(100, views)}
                            for t in range(0, 1201, 30)]
                    self.assertTrue(all(row["host_judged_mib"] is None for row in rows))
                    self.assertIsNone(views.identities)
                    self.assertIn(f"{TABLES[0]} of serving process 101 is not a regular file ({node})",
                                  rows[0]["host_exempt_error"])
                    with self.assertRaisesRegex(ValueError, "at 0 s has no"):
                        self.rule(rows)
            finally:
                table.unlink()
                table.write_bytes(b"table")

    def test_a_mapping_under_a_declared_path_of_another_file_object_counts(self):
        # Mappings named as the first table but of another device and inode.
        table = f"{self.store}/{TABLES[0]}"
        rows, _ = self.sampled(lambda t: {"exempt": t * self.RESIDENT // 1200}, foreign={table})
        self.assertEqual(self.listed(rows[0]), [(101, "7428b2600000-743f96013000", "r--s", "00000000"),
                                                (101, "743f96212000-74404d3e4000", "r--s", "16e3a12000")])
        rule = self.rule(rows)
        self.assertEqual((rule["host_memory_growth_mib"], rule["passed"]), (5425, False))

    def test_a_real_mapping_of_a_deleted_file_is_not_exempt(self):
        # This process maps a declared file, then the file is deleted and
        # another regular file takes its name, or the name smaps then shows.
        import mmap
        for label, declared, mapped, replaced in (("kept", "table.bin", "table.bin", None),
                                                  ("deleted", "table.bin", "table.bin", "table.bin"),
                                                  ("name collision", "another.bin (deleted)", "another.bin",
                                                   "another.bin (deleted)")):
            with self.subTest(label):
                path = Path(self.store) / mapped
                write_clean(path, b"x" * mmap.PAGESIZE)
                with path.open("rb") as stream:
                    view = mmap.mmap(stream.fileno(), mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ)
                self.addCleanup(view.close)
                self.assertEqual(view[0], ord("x"))
                if replaced:
                    path.unlink()
                    (Path(self.store) / replaced).write_bytes(b"y" * mmap.PAGESIZE)
                with self.serving_this_process() as pid:
                    row = soak.host_memory(pid, soak.MappedViews([declared]))
                mappings = [entry for entry in row["host_exempt_mappings"]]
                self.assertEqual(row["host_exempt_rss_mib"], mmap.PAGESIZE / 2 ** 20 if not replaced else 0)
                self.assertEqual(len(mappings), 0 if replaced else 1)
                view.close()
                (Path(self.store) / (replaced or mapped)).unlink()

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
        for key, value, message in (("host_judged_mib", rows[5]["host_judged_mib"] - 600, "other than VmRSS less"),
                                    ("host_exempt_rss_mib", rows[5]["host_exempt_bound_mib"] + 1, "listed mappings'"),
                                    ("host_exempt_bound_mib", rows[5]["host_exempt_rss_mib"] - 1, "listed mappings'"),
                                    ("host_exempt_rss_mib", rows[5]["host_rss_mib"] + 1, "listed mappings'")):
            changed = [dict(row) for row in rows]
            changed[5][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, f"at 150 s .*{message}"):
                self.rule(changed)

    def test_exempt_bytes_need_the_listed_mappings_of_every_sample(self):
        """The exempt Rss and bound must be those of the mappings each sample
        lists, each listed by the first sample and every sample since."""
        def drop(row, index):
            entry = row["host_exempt_mappings"].pop(index)
            row["host_exempt_rss_mib"] -= entry["rss_kib"] / 1024
            row["host_exempt_bound_mib"] -= entry["size_kib"] / 1024
            row["host_judged_mib"] = row["host_rss_mib"] - row["host_exempt_rss_mib"]

        def unlisted(rows):
            del rows[5]["host_exempt_mappings"]

        def emptied(rows):
            rows[5]["host_exempt_mappings"] = []

        def one_kept(rows):
            del rows[5]["host_exempt_mappings"][1:]

        def changed(key, value):
            def change(rows):
                rows[5]["host_exempt_mappings"][0][key] = value
            return change

        def duplicated(rows):
            entries = rows[5]["host_exempt_mappings"]
            entries[1] = dict(entries[0])

        def gone_then_back(rows):
            drop(rows[5], 0)

        def recycled(rows):
            for row in rows[5:]:
                for entry in row["host_exempt_mappings"]:
                    entry["start_time"] = 9001

        def overlapped(rows):
            # The second view moved onto the first's last page, at its own
            # offset.
            entries = rows[5]["host_exempt_mappings"]
            for row in rows:
                second = row["host_exempt_mappings"][1]
                second["range"] = [entries[0]["range"][1] - 4096, entries[0]["range"][1] - 4096 + 4096 * 1024]
                second["size_kib"], second["rss_kib"] = 4096, min(second["rss_kib"], 4096)
                row["host_exempt_rss_mib"] = sum(entry["rss_kib"] for entry in row["host_exempt_mappings"]) / 1024
                row["host_exempt_bound_mib"] = sum(entry["size_kib"] for entry in row["host_exempt_mappings"]) / 1024
                row["host_judged_mib"] = row["host_rss_mib"] - row["host_exempt_rss_mib"]

        def two_start_times(rows):
            for row in rows:
                row["host_exempt_mappings"][1]["start_time"] = 9001

        def remapped(rows):
            for row in rows[5:]:
                row["host_exempt_mappings"][0]["range"] = [0x100000000000, 0x101000000000]
                row["host_exempt_mappings"][0]["size_kib"] = 0x1000000000 // 1024

        for label, change, message in (
                ("no list", unlisted, "lists no exempt mappings"),
                ("an empty list", emptied, "listed mappings'"),
                ("one listed mapping", one_kept, "listed mappings'"),
                ("Rss over Size", changed("rss_kib", 96004172 + 1), "not a distinct clean shared read-only mapping"),
                ("Size other than the range", changed("size_kib", 96004172 + 1), "not a distinct clean shared read-only mapping"),
                ("a writable mapping", changed("perms", "rw-s"), "not a distinct clean shared read-only mapping"),
                ("an unknown field", changed("path", "/x"), "not a distinct clean shared read-only mapping"),
                ("a duplicate", duplicated, "not a distinct clean shared read-only mapping"),
                *((f"{key} spelled as smaps writes it", changed(key, spelled), "not a distinct clean shared read-only")
                  for key, spelled in (("range", "7428b2600000-743f96013000"), ("device", "103:01"),
                                       ("offset", "00000000"), ("range", [0x7428b2600000]),
                                       ("device", [True, 1]))),
                ("an overlapping mapping", overlapped, "at 0 s lists exempt mappings that overlap within a process"),
                ("a process with two start times", two_start_times, "at 0 s .* a process with two start times"),
                ("a mapping absent from a sample", gone_then_back, "at 180 s exempts a mapping"),
                ("another process", recycled, "at 150 s exempts a mapping"),
                ("another range", remapped, "at 150 s exempts a mapping")):
            rows, _ = self.sampled()
            change(rows)
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                self.rule(rows)
        # A mapping dropped for good, its bytes counted from then on, is
        # consistent evidence of their growth.
        rows, _ = self.sampled()
        for row in rows[5:]:
            drop(row, 0)
        self.assertEqual(self.rule(rows)["host_memory_growth_mib"], 3904596 / 1024)
        # The judged series is VmRSS less the exempt Rss: a lower recorded
        # judged value cannot hide growth from the rule.
        rows, _ = self.sampled(lambda t: {"anon": t * 600 * 1024 // 1200})
        self.assertFalse(self.rule(rows)["passed"])

    def test_the_declared_files_belong_to_the_scope(self):
        self.assertEqual(soak.mapped_view_files(TABLES, self.SCOPE), TABLES)
        self.assertEqual(soak.mapped_view_files(), [])
        for files, scope in (([], self.SCOPE), (TABLES, "rss"), (["/abs.bin"], self.SCOPE), (["a.bin (deleted)"], self.SCOPE),
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
                samples = report["memory_samples"]
                self.assertEqual((report["host_memory_scope"], report["mapped_view_files"],
                                  report["host_memory_growth_mib"], [len(row["host_exempt_mappings"]) for row in samples]),
                                 (self.SCOPE, TABLES, 0, [4] * len(samples)))
                identities = [(Path(self.store) / name).stat() for name in TABLES]
                self.assertEqual(report["mapped_view_file_identities"],
                                 [{"file": name, "path": f"{self.store}/{name}",
                                   "device": [os.major(status.st_dev), os.minor(status.st_dev)],
                                   "inode": status.st_ino} for name, status in zip(TABLES, identities)])
                self.assertGreaterEqual(len(samples), 2)
                self.assertEqual(report["host_exempt_rss_mib"], max(row["host_exempt_rss_mib"] for row in samples))
                self.assertGreater(samples[-1]["host_exempt_rss_mib"], samples[0]["host_exempt_rss_mib"])
                self.assertTrue(all(row["host_judged_mib"] == samples[0]["host_judged_mib"] for row in samples))
        # The default scope reads no smaps, records none of it, and fails the
        # same soak on its VmRSS growth.
        proc = self.proc(grow)
        code, report = self.soak(proc)
        self.assertEqual((code, proc.smaps_reads), (1, 0))
        self.assertFalse({"host_memory_scope", "mapped_view_files", "mapped_view_file_identities",
                          "host_exempt_rss_mib", "host_exempt_bound_mib"} & set(report))
        self.assertFalse(any({"host_judged_mib", "host_exempt_mappings"} & set(row) for row in report["memory_samples"]))

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
