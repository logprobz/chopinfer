"""Owned loopback fixture for agentic_mixed dry runs. No model or GPU required."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import threading
import time

EVENT_SECONDS = .003


class FakeClock:
    """Monotonic time that advances only as the fake server streams events (or
    a test advances it). Patched in as `agentic_mixed.time`, it fixes which
    sessions a sweep's window holds however loaded the host is. `run_level`'s
    waits and its boundary scrape still run on real time, so a test on this
    clock checks session accounting, not when the boundary scrape happens."""

    def __init__(self):
        self.lock = threading.Lock()
        # From zero, not the host's monotonic time, so sums of small steps
        # keep full precision however long the host has been up.
        self.now = 0.0

    def monotonic(self):
        with self.lock:
            return self.now

    def advance(self, seconds):
        with self.lock:
            self.now += seconds


class Reads:
    """How many bytes of each stream a ConfirmingTransport has read. The fake
    server sends a confirmed stream's next event only once the client has read
    the previous one, so each event reaches it in its own read and is timed
    separately whatever the host load."""

    def __init__(self, seconds=5):
        self.condition = threading.Condition()
        self.counts = {}
        self.ids = itertools.count()
        self.seconds = seconds

    def open(self):
        with self.condition:
            stream = next(self.ids)
            self.counts[stream] = 0
            return stream

    def confirm(self, stream, size):
        with self.condition:
            self.counts[stream] += size
            self.condition.notify_all()

    def wait(self, stream, size):
        """Whether the client read SIZE bytes of STREAM within `seconds`."""
        with self.condition:
            return self.condition.wait_for(lambda: self.counts.get(stream, 0) >= size, self.seconds)


class Unconfirmed(Exception):
    pass


@contextmanager
def fake_server(files, mode='fixed', clock=None, reads=None):
    """`clock` (a FakeClock) advances by the pacing interval per event. With
    `reads`, a request a confirming client names (`fake_stream`) waits for the client's read
    of each event instead of sleeping, and ends in a stream error if the read
    is not confirmed in time; any other request sleeps the pacing interval."""
    state = {'requests': 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == '/v1/models':
                raw = json.dumps({'data': [{'id': 'fake-agentic', 'max_model_len': 262144}]}).encode()
                content_type = 'application/json'
            elif self.path == '/metrics':
                with lock:
                    n = state['requests']
                raw = (f'chop_prefill_budget_mode{{mode="{mode}"}} 1\n'
                       f'chop_prefill_budget_selected_tokens_count {n if mode == "tbt" else 0}\n'
                       f'vllm:inter_token_latency_seconds_count {n * 2}\n'
                       f'vllm:inter_token_latency_seconds_sum {n * .006}\n'
                       f'vllm:inter_token_latency_seconds_bucket{{le="0.1"}} {n * 2}\n'
                       f'vllm:inter_token_latency_seconds_bucket{{le="+Inf"}} {n * 2}\n'
                       f'chop_step_execution_seconds_count{{phase="mixed"}} {n}\n'
                       f'chop_step_execution_seconds_sum{{phase="mixed"}} {n * .003}\n'
                       f'chop_step_execution_seconds_count{{phase="decode"}} {n * 2}\n'
                       f'chop_step_execution_seconds_sum{{phase="decode"}} {n * .006}\n'
                       f'chop_step_prefill_tokens_sum {n * 100}\n'
                       f'chop_step_prefill_tokens_count {n}\n'
                       f'chop_prefix_cache_reused_tokens_total {n * 50}\n'
                       f'chop_prefix_cache_queried_tokens_total {n * 100}\n').encode()
                content_type = 'text/plain'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            try:
                try:
                    self.chat()
                except Unconfirmed:
                    self.wfile.write(b'data: {"error": {"message": "client read not confirmed"}}\r\n\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when an interrupted benchmark cancels a stream.

        def chat(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                state['requests'] += 1
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            stream = request.get('fake_stream') if reads else None
            sent = 0

            def emit(event):
                nonlocal sent
                raw = b'data: ' + json.dumps(event).encode() + b'\r\n\r\n'
                self.wfile.write(raw)
                self.wfile.flush()
                sent += len(raw)
                if stream is None:
                    time.sleep(EVENT_SECONDS)
                elif not reads.wait(stream, sent):
                    raise Unconfirmed()
                if clock:
                    clock.advance(EVENT_SECONDS)

            def delta(value):
                emit({'choices': [{'index': 0, 'delta': value}]})

            messages = request['messages']
            if request.get('tools'):
                tool_count = sum(m['role'] == 'tool' for m in messages)
                path = messages[1]['content'].split(':', 1)[0][3:]
                if tool_count < 3:
                    name = ['list_files', 'read', 'edit'][tool_count]
                    first_line = files[path].split('\n')[0]
                    args = [{}, {'path': path}, {'path': path, 'oldString': first_line, 'newString': first_line}][tool_count]
                    encoded = json.dumps(args)
                    delta({'tool_calls': [{'index': 0, 'id': f'call-{tool_count}', 'type': 'function', 'function': {'name': name, 'arguments': ''}}]})
                    for part in (encoded[:len(encoded)//2], encoded[len(encoded)//2:]):
                        delta({'tool_calls': [{'index': 0, 'function': {'arguments': part}}]})
                    finish = 'tool_calls'
                else:
                    delta({'content': 'Applied '})
                    delta({'content': 'one edit.'})
                    finish = 'stop'
                output = 20
            else:
                delta({'content': '1 alpha beta gamma '})
                delta({'content': 'delta epsilon zeta eta theta\n'})
                finish, output = 'length', request['max_tokens']
            emit({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})
            if request.get('stream_options'):
                emit({'choices': [], 'usage': {'prompt_tokens': 2000, 'completion_tokens': output,
                                              'prompt_tokens_details': {'cached_tokens': 1000}},
                      'metrics': {'speculative_decoding': {'num_accepted_draft_tokens': 8, 'num_draft_tokens': 10}}})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    class FakeHTTPServer(ThreadingHTTPServer):
        request_queue_size = 2048

    server = FakeHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
