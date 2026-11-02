"""What a served fake-host process logs at each level.

Every failed request writes one chopinfer.request-failure.v1 line at the
default level, routine bundle traces appear only at debug and then exactly as
the bundle wrote them, warn drops startup chatter, and the summary line
reports what the server saw. The TP4 fixture's control directory refuses a
batch (`refuse`), fails the worker (`fail`), slows each execute (`delay-ms`)
or selects an output token (`emit-token`), and appends every step-shape
record it logs to `emitted.jsonl`.
"""
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "debug"
# As in test_process_lifecycle.py: a bound only for waits that must end.
HANG_SECONDS = 20 * 60
FAILURE = "chopinfer.request-failure.v1"
SUMMARY = "chopinfer.summary.v1"
STEP_SHAPE = '"schema":"chopinfer.tp4-fixture.step-shape.v1"'
FIELDS = {
    "schema", "time", "level", "request_id", "client_request_id", "method", "path", "model",
    "status", "error_type", "code", "param", "message", "stage", "prompt_tokens",
    "generated_tokens", "elapsed_seconds", "peer",
}


def wait_until(predicate, timeout=HANG_SECONDS):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("condition did not become true before deadline")


class RequestLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([
            "cargo", "build", "--locked", "--offline", "-p", "runtime-host",
            "--bin", "chopinfer", "-p", "bundle-tp4-fixture",
            "--bin", "build-tp4-fixture", "--features", "fake-host-services",
        ], cwd=ROOT, check=True)
        cls.root = tempfile.TemporaryDirectory(prefix="chopinfer-request-log-")
        cls.package = Path(cls.root.name) / "package"
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
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=HANG_SECONDS)
        self.pool.shutdown(wait=True)
        self.output.close()
        self.work.cleanup()

    def start(self, *extra, environment=None):
        env = dict(os.environ, TP4_FIXTURE_CONTROL_DIR=str(self.control))
        env.pop("CHOPINFER_LOG_LEVEL", None)
        env.pop("CHOPINFER_LOG_SUMMARY_SECONDS", None)
        env.update(environment or {})
        self.process = subprocess.Popen([
            str(TARGET / "chopinfer"), "serve", "--bundle", str(self.package),
            "--checkpoint", str(self.checkpoint), "--fake-host-services",
            "--listen", "127.0.0.1:0", "--max-sequences", "1", "--queue-depth", "0", *extra,
        ], cwd=ROOT, env=env, stdout=self.output, stderr=subprocess.STDOUT)

        def listening():
            self.assertIsNone(self.process.poll(), self.log.read_text())
            for line in self.log.read_text().splitlines():
                if line.startswith('{"status":"listening"'):
                    return json.loads(line)["address"]
            return None

        self.address = wait_until(listening)
        self.model = self.request("GET", "/v1/models")[1]["data"][0]["id"]

    def request(self, method, path, payload=None, *, body=None):
        connection = http.client.HTTPConnection(self.address, timeout=HANG_SECONDS)
        try:
            if body is None and payload is not None:
                body = json.dumps(payload)
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            text = response.read().decode()
            try:
                return response.status, json.loads(text)
            except json.JSONDecodeError:
                return response.status, text
        finally:
            connection.close()

    def stop(self):
        self.process.send_signal(signal.SIGTERM)
        self.assertEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())

    def records(self, schema):
        found = []
        for line in self.log.read_text().splitlines():
            if line.startswith('{"schema":"' + schema + '"'):
                found.append(json.loads(line))
        return found

    def failures(self, count):
        """The failure lines once there are `count`, and no more after a pause."""
        wait_until(lambda: len(self.records(FAILURE)) >= count)
        time.sleep(0.5)
        lines = self.records(FAILURE)
        self.assertEqual(len(lines), count, self.log.read_text())
        return lines

    def stream(self, tokens=64):
        """A streamed completion, and an event set once its first token arrives."""
        first_token = threading.Event()

        def read():
            connection = http.client.HTTPConnection(self.address, timeout=HANG_SECONDS)
            try:
                connection.request("POST", "/v1/completions", json.dumps({
                    "model": self.model, "prompt": "Hello", "max_tokens": tokens,
                    "temperature": 0, "stream": True, "ignore_eos": True,
                }), {"Content-Type": "application/json"})
                response = connection.getresponse()
                chunks = []
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    value = line[6:].strip()
                    if value == b"[DONE]":
                        return response.status, chunks
                    chunks.append(json.loads(value))
                    if any(choice.get("text") for choice in chunks[-1].get("choices", [])):
                        first_token.set()
                return response.status, chunks
            finally:
                connection.close()

        return self.pool.submit(read), first_token

    def provoke_failures(self):
        """A malformed body, an unknown model, a capacity refusal and a batch
        the bundle refuses mid-stream, in that order."""
        status, _ = self.request("POST", "/v1/completions", body='{"model": ')
        self.assertEqual(status, 400)
        status, _ = self.request("POST", "/v1/completions",
                                 {"model": "missing.model", "prompt": "x"})
        self.assertEqual(status, 404)
        (self.control / "delay-ms").write_text("20")
        held, first_token = self.stream()
        self.assertTrue(first_token.wait(HANG_SECONDS))
        status, _ = self.request("POST", "/v1/completions",
                                 {"model": self.model, "prompt": "x", "max_tokens": 1})
        self.assertEqual(status, 429)
        (self.control / "refuse").touch()
        status, chunks = held.result(timeout=HANG_SECONDS)
        self.assertEqual(status, 200)
        self.assertIn("error", chunks[-1], chunks)
        (self.control / "refuse").unlink()
        (self.control / "delay-ms").unlink()
        self.assertEqual(self.request("GET", "/health")[0], 200)

    def check_failure_lines(self, lines):
        for line in lines:
            self.assertEqual(set(line), FIELDS | ({"json_error"} if "json_error" in line else set())
                             | ({"served"} if "served" in line else set())
                             | ({"output_refusal"} if "output_refusal" in line else set()), line)
            self.assertNotIn("output_refusal", line)
            self.assertTrue(line["peer"].startswith("127.0.0.1:"), line)
        malformed, unknown, capacity, refused = lines
        self.assertEqual((malformed["status"], malformed["level"]), (400, "warn"))
        self.assertEqual(malformed["json_error"]["category"], "eof")
        self.assertEqual((unknown["status"], unknown["model"]), (404, "missing.model"))
        self.assertIn(self.model, unknown["served"])
        self.assertEqual((capacity["status"], capacity["level"]), (429, "warn"))
        self.assertEqual((refused["level"], refused["stage"]), ("error", "generation"))
        self.assertTrue(refused["request_id"].endswith("_0"), refused)
        self.assertGreaterEqual(refused["generated_tokens"], 1)
        self.assertIn(f"generation request failed: request_id={refused['request_id']} ",
                      self.log.read_text())

    def test_default_level_logs_each_failure_once_and_no_step_traces(self):
        self.start()
        self.assertIn("startup logging: level=info", self.log.read_text())
        self.provoke_failures()
        self.check_failure_lines(self.failures(4))
        self.stop()
        emitted = (self.control / "emitted.jsonl").read_text()
        self.assertIn(STEP_SHAPE, emitted)
        self.assertNotIn(STEP_SHAPE, self.log.read_text())

    def test_a_refused_output_names_its_reason_and_token(self):
        self.start()
        declaration = json.loads((self.package / "components/frontend.json").read_text())
        token_id = next(token["id"] for token in declaration["vocabulary"]["boundary_tokens"]
                        if token["text"] == "</arg_value>")
        control = self.control / "emit-token"
        control.write_text(str(token_id))
        try:
            status, body = self.request("POST", "/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": "canary-7f3e9b"}],
                "max_tokens": 4, "temperature": 0, "stream": True,
            })
        finally:
            control.unlink()
        self.assertEqual(status, 200, body)
        chunks = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertIn("error", chunks[-1], chunks)
        self.assertEqual(chunks[-1]["error"]["code"], "invalid_model_output", chunks)
        self.assertNotIn("output_refusal", chunks[-1]["error"])
        line, = self.failures(1)
        self.assertEqual(line["code"], "invalid_model_output")
        self.assertEqual(line["message"], "backend failure")
        self.assertEqual(line["output_refusal"], {
            "reason": "malformed completion: completion contains an unexpected reserved boundary",
            "token_ordinal": 0, "token_id": token_id,
        })
        self.assertNotIn("canary-7f3e9b", json.dumps(line))

    def test_warn_keeps_failures_and_drops_startup_lines(self):
        self.start("--log-level", "warn")
        self.provoke_failures()
        self.check_failure_lines(self.failures(4))
        self.stop()
        startup = [line for line in self.log.read_text().splitlines() if line.startswith("startup ")]
        self.assertEqual(startup, [])

    def test_debug_prints_every_step_trace_as_the_bundle_wrote_it(self):
        self.start(environment={"CHOPINFER_LOG_LEVEL": "debug"})
        self.provoke_failures()
        self.check_failure_lines(self.failures(4))
        self.stop()
        emitted = (self.control / "emitted.jsonl").read_text().splitlines()
        self.assertTrue(emitted)
        printed = [line[len("bundle: "):] for line in self.log.read_text().splitlines()
                   if line.startswith("bundle: ") and STEP_SHAPE in line]
        self.assertEqual(printed, emitted)

    def test_summary_reports_requests_refusals_and_the_worker(self):
        self.start(environment={"CHOPINFER_LOG_SUMMARY_SECONDS": "1"})
        status, _ = self.request("POST", "/v1/completions",
                                 {"model": self.model, "prompt": "Hello", "max_tokens": 2})
        self.assertEqual(status, 200)
        status, _ = self.request("POST", "/v1/completions",
                                 {"model": "missing.model", "prompt": "x"})
        self.assertEqual(status, 404)

        def counted():
            lines = self.records(SUMMARY)
            refused = sum(line["refusals"]["unknown_model"] for line in lines)
            submitted = sum(line["requests"]["submitted"] for line in lines)
            return lines if refused == 1 and submitted >= 1 else None

        lines = wait_until(counted, timeout=30)
        self.assertEqual(lines[-1]["worker_state"], "ready")
        self.assertEqual(sum(line["failures"]["warn"] for line in lines), 1)
        self.assertGreaterEqual(sum(line["http_requests"] for line in lines), 2)
        self.stop()

    def test_a_fatal_worker_failure_still_logs_the_request_in_flight(self):
        self.start()
        (self.control / "fail").touch()
        generation = self.pool.submit(self.request, "POST", "/v1/completions", {
            "model": self.model, "prompt": "Hello", "max_tokens": 4,
        })
        self.assertNotEqual(self.process.wait(timeout=HANG_SECONDS), 0, self.log.read_text())
        try:
            generation.result(timeout=HANG_SECONDS)
        except (OSError, http.client.HTTPException):
            pass
        self.assertIn("fatal: generation worker failed", self.log.read_text())
        lines = self.records(FAILURE)
        self.assertEqual(len(lines), 1, self.log.read_text())
        self.assertEqual((lines[0]["level"], lines[0]["path"]), ("error", "/v1/completions"))


if __name__ == "__main__":
    unittest.main()
