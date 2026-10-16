"""CPU subprocess checks using the installed TP4 fixture and the real host."""
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "debug"
# CI compilers can delay both the host and the client reading its terminal chunks.
LOAD_MARGIN_SECONDS = 5
# How long a wait the test needs only to end (a start, a first token, a
# response, an exit) may take before the test fails as hung; no test checks a
# duration with it. The 13 tests take about a minute on a quiet host, but in
# gate d-r1 (2026-10-15, load 60 to 80 beside fresh builds) a 64-token stream
# at 20 ms a token, about 1.3 s alone, outlasted a 10 s wait. 20 minutes, as
# test_model_preparation.py's HANG_SECONDS. The drain deadlines the tests
# check keep their own bounds below.
HANG_SECONDS = 20 * 60


def wait_until(predicate, timeout=None):
    deadline = time.monotonic() + (HANG_SECONDS if timeout is None else timeout)
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("condition did not become true before deadline")


class ProcessLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([
            "cargo", "build", "--locked", "--offline", "-p", "runtime-host",
            "--bin", "chopinfer", "-p", "bundle-tp4-fixture",
            "--bin", "build-tp4-fixture", "--features", "fake-host-services",
        ], cwd=ROOT, check=True)
        cls.root = tempfile.TemporaryDirectory(prefix="chopinfer-lifecycle-")
        cls.package = Path(cls.root.name) / "package"
        # The package plans its program image; serving loads it from this
        # synthetic checkpoint.
        cls.checkpoint = Path(cls.root.name) / "checkpoint"
        subprocess.run([str(TARGET / "build-tp4-fixture"), "--fixture", "tp4-synthetic",
                        "--output", str(cls.package), "--checkpoint", str(cls.checkpoint)],
                       cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(dir=self.root.name)
        self.control = Path(self.work.name)
        self.log = self.control / "server.log"
        self.output = self.log.open("w")
        self.process = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=HANG_SECONDS)
        self.pool.shutdown(wait=True)
        self.output.close()
        self.work.cleanup()

    def start(self, timeout=30000, extra=(), package=None):
        self.drain_timeout_seconds = timeout / 1000
        env = dict(os.environ, TP4_FIXTURE_CONTROL_DIR=str(self.control))
        self.process = subprocess.Popen([
            str(TARGET / "chopinfer"), "serve", "--bundle", str(package or self.package),
            "--checkpoint", str(self.checkpoint), "--fake-host-services", "--listen", "127.0.0.1:0", "--max-sequences", "1",
            "--drain-timeout-ms", str(timeout), *extra,
        ], cwd=ROOT, env=env, stdout=self.output, stderr=subprocess.STDOUT)

        def listening():
            self.assertIsNone(self.process.poll(), self.log.read_text())
            for line in self.log.read_text().splitlines():
                if line.startswith('{"status":"listening"'):
                    return json.loads(line)["address"]
            return None

        self.address = wait_until(listening)
        self.assertIn("startup generation warmup:", self.log.read_text())
        self.assertEqual(self.request("GET", "/health")[0], 200)
        self.model = self.request("GET", "/v1/models")[1]["data"][0]["id"]
        (self.control / "entered").unlink()

    def request(self, method, path, payload=None, *, raw=False, headers=False):
        connection = http.client.HTTPConnection(self.address, timeout=HANG_SECONDS)
        try:
            body = None if payload is None else json.dumps(payload)
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read()
            result = response.status, body.decode() if raw else json.loads(body)
            return (*result, dict(response.getheaders())) if headers else result
        finally:
            connection.close()

    def generation(self, delay=150):
        (self.control / "delay-ms").write_text(str(delay))
        future = self.pool.submit(self.request, "POST", "/v1/completions", {
            "model": self.model, "prompt": "Hello", "max_tokens": 4, "temperature": 0,
        })
        wait_until(lambda: (self.control / "entered").exists())
        return future

    def test_kv_cache_dtype_serves_the_bf16_fixture_and_refuses_fp8(self):
        # The fixture declares only a BF16 layout: auto and bf16 serve it.
        for extra, how in [((), "auto, the package's preference"), (("--kv-cache-dtype", "bf16"), "requested")]:
            with self.subTest(extra=extra):
                self.start(extra=extra)
                self.assertIn(f"serving KV cache dtype: bf16 ({how}; the package supports bf16)",
                              self.log.read_text())
                self.process.send_signal(signal.SIGTERM)
                self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())
                self.output.close()
                self.output = self.log.open("w")
        refused = subprocess.run([
            str(TARGET / "chopinfer"), "serve", "--bundle", str(self.package),
            "--checkpoint", str(self.checkpoint), "--fake-host-services",
            "--listen", "127.0.0.1:0", "--kv-cache-dtype", "fp8",
        ], cwd=ROOT, capture_output=True, text=True, timeout=HANG_SECONDS)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("--kv-cache-dtype fp8: bundle", refused.stderr)
        self.assertIn("declares no fp8 KV layout; it supports bf16", refused.stderr)

    def relaid_package(self, name, prefers_fp8):
        """The fixture package with a second, FP8 KV layout of 64-byte pages.

        The module places tokens only in its own 128-byte pages and refuses any
        other geometry at create, so which layout a serve hands it is visible.
        """
        package = Path(self.root.name) / name
        if not package.exists():
            shutil.copytree(self.package, package)
            path = package / "manifest.json"
            manifest = json.loads(path.read_text())
            bf16 = manifest["kv_layout"]
            fp8 = dict(bf16, id="fixture.tp4.kv-fp8.v1", alignment_bytes=64,
                       bytes_per_token=2, page_bytes=64)
            manifest["kv_formats"] = [{"dtype": "bf16", "layout": bf16},
                                      {"dtype": "fp8_e4m3", "layout": fp8}]
            manifest["kv_layout"] = fp8 if prefers_fp8 else bf16
            path.write_text(json.dumps(dict(sorted(manifest.items())), separators=(",", ":")))
        return package

    def test_the_selected_kv_layout_reaches_the_module_without_the_serving_pool(self):
        # The fixture allocates physical KV itself, so the choice reaches it
        # through the instance's KV contract alone.
        for name, prefers_fp8, dtype, serves in [
                ("prefers-bf16", False, "fp8", False), ("prefers-bf16", False, "auto", True),
                ("prefers-fp8", True, "auto", False), ("prefers-fp8", True, "bf16", True)]:
            with self.subTest(name=name, dtype=dtype):
                package = self.relaid_package(name, prefers_fp8)
                if serves:
                    self.start(extra=("--kv-cache-dtype", dtype), package=package)
                    self.assertIn("serving KV cache dtype: bf16", self.log.read_text())
                    self.process.send_signal(signal.SIGTERM)
                    self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())
                    self.output.close()
                    self.output = self.log.open("w")
                    continue
                refused = subprocess.run([
                    str(TARGET / "chopinfer"), "serve", "--bundle", str(package),
                    "--checkpoint", str(self.checkpoint), "--fake-host-services",
                    "--listen", "127.0.0.1:0", "--max-sequences", "1",
                    "--kv-cache-dtype", dtype,
                ], cwd=ROOT, capture_output=True, text=True, timeout=HANG_SECONDS)
                self.assertNotEqual(refused.returncode, 0, refused.stdout)
                self.assertIn("serving KV cache dtype: fp8", refused.stdout)
                self.assertIn("refused this host", refused.stdout + refused.stderr)

    def test_sigterm_drains_and_refuses_new_work(self):
        self.start()
        generation = self.generation(300)
        self.process.send_signal(signal.SIGTERM)
        wait_until(lambda: "shutdown: draining" in self.log.read_text())
        status, error = self.request("POST", "/v1/completions", {})
        self.assertEqual(status, 503)
        self.assertEqual(error["error"]["type"], "InternalServerError")
        self.assertEqual(error["error"]["code"], 503)
        self.assertEqual(self.request("GET", "/health")[0], 503)
        self.assertEqual(generation.result(timeout=HANG_SECONDS)[0], 200)
        self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())

    def streaming_generation(self, tokens=64):
        first_token = threading.Event()
        chunks = []

        def stream():
            connection = http.client.HTTPConnection(
                self.address, timeout=HANG_SECONDS)
            try:
                connection.request("POST", "/v1/completions", json.dumps({
                    "model": self.model, "prompt": "Hello", "max_tokens": tokens,
                    "temperature": 0, "stream": True, "ignore_eos": True,
                    "stream_options": {"include_usage": True},
                }), {"Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status != 200:
                    raise AssertionError((response.status, response.read()))
                for line in response:
                    if line.startswith(b"data: "):
                        value = line[6:].strip()
                        if value == b"[DONE]":
                            return response.status, chunks
                        chunk = json.loads(value)
                        chunks.append(chunk)
                        if any(c.get("text") for c in chunk.get("choices", [])):
                            first_token.set()
                raise AssertionError(f"stream closed without [DONE]: {chunks}")
            finally:
                connection.close()

        future = self.pool.submit(stream)
        return future, first_token, chunks

    def test_streaming_response_finishes_during_drain(self):
        # Use the production drain budget when testing successful completion.
        self.start()
        (self.control / "delay-ms").write_text("20")
        generation, first_token, chunks = self.streaming_generation()
        if not first_token.wait(HANG_SECONDS):
            self.fail(f"no content: {generation.result(timeout=1)}")
        before = len(chunks)
        started = time.monotonic()
        deadline = started + self.drain_timeout_seconds + LOAD_MARGIN_SECONDS
        self.process.send_signal(signal.SIGTERM)
        wait_until(lambda: "shutdown: draining" in self.log.read_text())
        # Each request opens a fresh TCP connection while the stream is active.
        for method, path in (("GET", "/health"), ("POST", "/v1/completions")):
            status, error, headers = self.request(method, path, {}, headers=True)
            self.assertEqual(status, 503)
            self.assertEqual(headers.get("retry-after"), "1")
            self.assertEqual(error["error"]["message"], "server is draining")
        self.assertEqual(self.request("GET", "/metrics", raw=True)[0], 200)
        status, chunks = generation.result(timeout=max(0, deadline - time.monotonic()))
        self.assertEqual(status, 200)
        self.assertFalse(any("error" in chunk for chunk in chunks), chunks)
        self.assertGreater(len(chunks), before + 10)
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 64)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
        finished = time.monotonic()
        self.assertEqual(self.process.wait(timeout=LOAD_MARGIN_SECONDS), 0, self.log.read_text())
        self.assertLess(time.monotonic() - finished, LOAD_MARGIN_SECONDS)
        self.assertLess(time.monotonic(), deadline)

    def test_request_in_body_intake_gets_retry_hint_during_drain(self):
        # The test checks the retry hint, not drain timing, so its 64-token
        # stream must finish inside the drain. Under host load that stream
        # outlasted the production 30 s (an estimated 33 s); 90 s is more
        # than twice that.
        self.start(timeout=90000)
        (self.control / "delay-ms").write_text("20")
        generation, first_token, _ = self.streaming_generation()
        self.assertTrue(first_token.wait(HANG_SECONDS))
        connection = http.client.HTTPConnection(self.address, timeout=HANG_SECONDS)
        try:
            body = json.dumps({
                "model": self.model, "prompt": "Hello", "max_tokens": 1,
            }).encode()
            connection.putrequest("POST", "/v1/completions")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders()
            connection.send(body[:1])
            # Observe intake past admission middleware before sending SIGTERM:
            # a waiting completion reserves its declared length, at least the
            # 16 KiB minimum charge.
            reserved = max(len(body), 16 * 1024)
            wait_until(lambda: f"chop_intake_used_bytes {reserved}\n" in
                       self.request("GET", "/metrics", raw=True)[1])
            self.process.send_signal(signal.SIGTERM)
            wait_until(lambda: "shutdown: draining" in self.log.read_text())
            status, expected, headers = self.request(
                "POST", "/v1/completions", {}, headers=True)
            connection.send(body[1:])
            response = connection.getresponse()
            self.assertEqual(response.status, 503)
            self.assertEqual(response.status, status)
            self.assertEqual(json.loads(response.read()), expected)
            self.assertEqual(response.getheader("Retry-After"), "1")
            self.assertEqual(response.getheader("Retry-After"), headers["retry-after"])
        finally:
            connection.close()
        status, chunks = generation.result(timeout=HANG_SECONDS)
        self.assertEqual(status, 200)
        self.assertFalse(any("error" in chunk for chunk in chunks), chunks)
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 64)
        self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())

    def test_streaming_deadline_sends_terminal_error(self):
        self.start(timeout=100)
        (self.control / "delay-ms").write_text("20")
        generation, first_token, _ = self.streaming_generation()
        if not first_token.wait(HANG_SECONDS):
            self.fail(f"no content: {generation.result(timeout=1)}")
        started = time.monotonic()
        self.process.send_signal(signal.SIGTERM)
        status, chunks = generation.result(timeout=3)
        self.assertEqual(status, 200)
        self.assertEqual(chunks[-1]["error"]["code"], "shutdown_timeout")
        self.assertIn("drain deadline", chunks[-1]["error"]["message"])
        self.assertEqual(self.process.wait(timeout=1), 0, self.log.read_text())
        self.assertLess(time.monotonic() - started, 2)

    def wait_for_queued_request(self):
        def queued():
            _, metrics = self.request("GET", "/metrics", raw=True)
            return any(line.startswith("vllm:num_requests_waiting{")
                       and line.endswith(" 1") for line in metrics.splitlines())
        wait_until(queued)

    def test_scheduler_queued_stream_finishes_during_drain(self):
        # Use the production drain budget when testing successful completion.
        self.start()
        (self.control / "delay-ms").write_text("20")
        running, first_token, _ = self.streaming_generation()
        self.assertTrue(first_token.wait(HANG_SECONDS))
        queued, queued_token, _ = self.streaming_generation(tokens=8)
        self.wait_for_queued_request()
        self.assertFalse(queued_token.is_set())
        started = time.monotonic()
        deadline = started + self.drain_timeout_seconds + LOAD_MARGIN_SECONDS
        self.process.send_signal(signal.SIGTERM)
        for future, tokens in [(running, 64), (queued, 8)]:
            status, chunks = future.result(timeout=max(0, deadline - time.monotonic()))
            self.assertEqual(status, 200)
            self.assertFalse(any("error" in chunk for chunk in chunks), chunks)
            self.assertEqual(chunks[-1]["usage"]["completion_tokens"], tokens)
            self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(self.process.wait(timeout=LOAD_MARGIN_SECONDS), 0, self.log.read_text())
        self.assertLess(time.monotonic(), deadline)

    def test_deadline_refuses_scheduler_queued_request(self):
        self.start(timeout=100)
        (self.control / "delay-ms").write_text("20")
        running, first_token, _ = self.streaming_generation()
        self.assertTrue(first_token.wait(HANG_SECONDS))
        queued = self.pool.submit(self.request, "POST", "/v1/completions", {
            "model": self.model, "prompt": "Hello", "max_tokens": 4,
        })
        self.wait_for_queued_request()
        self.process.send_signal(signal.SIGTERM)
        status, error = queued.result(timeout=3)
        self.assertEqual(status, 503)
        self.assertIn("drain deadline", error["error"]["message"])
        error = running.result(timeout=3)[1][-1]["error"]
        self.assertEqual(error["code"], "shutdown_timeout")
        self.assertIn("drain deadline", error["message"])
        self.assertEqual(self.process.wait(timeout=1), 0, self.log.read_text())

    def test_sigint_exits_cleanly(self):
        self.start()
        self.process.send_signal(signal.SIGINT)
        self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())

    def test_drain_deadline_cancels_remaining_work(self):
        self.start(timeout=50)
        generation = self.generation(300)
        started = time.monotonic()
        self.process.send_signal(signal.SIGTERM)
        self.assertEqual(self.process.wait(timeout=3), 0, self.log.read_text())
        self.assertLess(time.monotonic() - started, 1.5)
        try:
            status, _ = generation.result(timeout=HANG_SECONDS)
            self.assertNotEqual(status, 200)
        except (OSError, http.client.HTTPException):
            pass

    def test_second_signal_exits_immediately(self):
        self.start()
        self.generation(2000)
        self.process.send_signal(signal.SIGTERM)
        wait_until(lambda: "shutdown: draining" in self.log.read_text())
        started = time.monotonic()
        self.process.send_signal(signal.SIGINT)
        self.assertEqual(self.process.wait(timeout=1), 0)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_fatal_bundle_error_exits_nonzero(self):
        self.start()
        (self.control / "fail").touch()
        generation = self.generation(0)
        try:
            status, _ = generation.result(timeout=HANG_SECONDS)
            self.assertGreaterEqual(status, 500)
        except (OSError, http.client.HTTPException):
            pass
        self.assertNotEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())
        self.assertIn("generation worker failed", self.log.read_text())

    def test_failed_warmup_never_opens_listener(self):
        (self.control / "fail").touch()
        with self.assertRaises(AssertionError):
            self.start()
        self.assertNotEqual(self.process.wait(timeout=HANG_SECONDS), 0)
        self.assertNotIn('"status":"listening"', self.log.read_text())
        self.assertIn("startup warmup failed", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
