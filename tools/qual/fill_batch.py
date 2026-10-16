"""Hold a fixed number of decoding streams on a server, so the decode steps
another tool runs meanwhile carry at least that many rows.

Each of `--streams` requests continues its own diverse prompt (bench.py's
seeds) for `--tokens` tokens with `min_tokens` set, so none stops early.
Once every stream has emitted its first token the tool prints `ready N` and
keeps reading; a stream that ends prints `finished INDEX TOKENS`, and a
stream that fails prints `failed INDEX ERROR`. A runner starts it, waits for
`ready`, runs its measurement with S more concurrent requests (every step
then carries N to N + S decode rows, one graph bucket when both bounds are
in it), checks that no stream finished or failed, and stops it: on SIGTERM
the tool prints `stopped N` and exits 0, so a runner can tell an
acknowledged stop from a death (tools/dev/concurrency-control.py).
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.request

from bench import stream_prompt
from common import Client, arguments


def body(model, index, tokens):
    """The request of filler stream `index`: its own prompt, a fixed budget."""
    return {"model": model, "prompt": stream_prompt("diverse", index), "max_tokens": tokens,
            "min_tokens": tokens, "stream": True}


def hold(url, request, index, total, started, lock, report):
    """Streams one request to its end; the last of `total` streams to emit
    its first token reports that all are decoding."""
    tokens = 0
    try:
        data = json.dumps(request).encode()
        http = urllib.request.Request(url + "/v1/completions", data=data,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=None) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                if any(choice.get("text") for choice in event.get("choices", [])):
                    tokens += 1
                    if tokens == 1:
                        with lock:
                            started.append(index)
                            if len(started) == total:
                                report(f"ready {len(started)}")
        report(f"finished {index} {tokens}")
    except Exception as error:  # every failure is reported, none retried
        report(f"failed {index} {type(error).__name__}: {error}")


def main():
    parser = arguments(__doc__)
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.streams < 1 or args.tokens < 1:
        parser.error("streams and tokens must be positive")
    client = Client(args.url, args.timeout)
    model = client.identity(args.model)["model"]
    lock = threading.Lock()
    started = []

    def report(line):
        print(line, flush=True)

    def stopped(*_):
        print(f"stopped {args.streams}", flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, stopped)

    # The SIGTERM handler prints from this thread, so it must not run while
    # this thread prints. The streams start with SIGTERM blocked and keep it
    # blocked, so the signal reaches this thread alone, and this thread
    # unblocks it only after its print.
    held = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    threads = []
    for index in range(args.streams):
        thread = threading.Thread(
            target=hold,
            args=(client.url, body(model, index, args.tokens), index, args.streams, started,
                  lock, report),
            daemon=True)
        thread.start()
        threads.append(thread)
    try:
        print(f"started {args.streams} streams of {args.tokens} tokens at {time.time():.3f}",
              flush=True)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, held)
    for thread in threads:
        thread.join()
    sys.exit(0)


if __name__ == "__main__":
    main()
