"""CPU checks for fill_batch.py's requests and readiness report."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import tempfile
import unittest

import fill_batch


class FillBatchTests(unittest.TestCase):
    def test_every_stream_has_its_own_prompt_and_a_fixed_budget(self):
        bodies = [fill_batch.body("m", index, 64) for index in range(40)]
        self.assertEqual(len({body["prompt"] for body in bodies}), 40)
        self.assertTrue(all(body["max_tokens"] == body["min_tokens"] == 64 for body in bodies))
        self.assertTrue(all(body["stream"] for body in bodies))

    def test_ready_follows_every_first_token_and_ends_are_reported(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for _ in range(3):
                    event = {"choices": [{"text": "x"}]}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        lines = []
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                lock, started = threading.Lock(), []
                workers = [threading.Thread(target=fill_batch.hold, args=(
                    url, fill_batch.body("m", index, 3), index, 2, started, lock, lines.append))
                    for index in range(2)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(lines.count("ready 2"), 1)
        self.assertEqual(sorted(line for line in lines if line.startswith("finished")),
                         ["finished 0 3", "finished 1 3"])
        self.assertTrue(all(set(request) == {"model", "prompt", "max_tokens", "min_tokens",
                                             "stream"} for request in requests))

    def test_sigterm_is_acknowledged_while_streams_hold(self):
        import os
        import signal
        import subprocess
        import sys
        import time

        class Slow(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "m"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for _ in range(600):
                        self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.05)
                except OSError:
                    pass

        with tempfile.TemporaryDirectory() as receipts, \
                ThreadingHTTPServer(("127.0.0.1", 0), Slow) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fill_batch.py")
                child = subprocess.Popen(
                    [sys.executable, "-B", tool, "--url", f"http://127.0.0.1:{server.server_port}",
                     "--streams", "3", "--tokens", "500"],
                    stdout=subprocess.PIPE, text=True,
                    env=dict(os.environ, CHOPINFER_QUAL_RECEIPTS=receipts))
                lines = []
                for line in child.stdout:
                    lines.append(line.strip())
                    if line.startswith("ready "):
                        break
                child.send_signal(signal.SIGTERM)
                rest, _ = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0)
                self.assertIn("ready 3", lines)
                self.assertIn("stopped 3", rest.splitlines())
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
