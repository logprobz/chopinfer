"""Run a MiMo TP4 comparison within 20 minutes, with 64 clients and 32 live sequences.

Run only on idle GPUs. The command file must contain a JSON argv
array for a foreground, already-built TP4 chopinfer serve process. This runner
owns only that process and its load worker.

Arms run in order: fixed, decode-aware at its default ITL target, and the
original one-decode-step target when its full allowance still fits.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import multiprocessing
import os
import subprocess
import threading
import time

from common import Client
from bench import token_events
from spec_metrics import metrics_text

ARM_SECONDS = 200
STARTUP_SECONDS = 150
DRAIN_SECONDS = 90
WARMUP_DRAIN_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 120
WARMUP_SECONDS = 30
MAX_RUN_SECONDS = 1200
# The worst case of one arm; an arm starts only with this much time left.
ARM_RESERVE_SECONDS = ARM_SECONDS + WARMUP_SECONDS + STARTUP_SECONDS + WARMUP_DRAIN_SECONDS + DRAIN_SECONDS
# The host's default decode-aware target, the ITL p99 gate for that arm.
ITL_TARGET_MS = 200
PROTECTED = ("--max-sequences", "--queue-depth", "--admission", "--prefill-budget",
             "--prefill-itl-target-ms", "--prefill-decode-steps")
# (name, policy arguments, required)
ARMS = (
    ("fixed", ["--prefill-budget", "fixed"], True),
    ("decode-aware", ["--prefill-budget", "decode-aware"], True),
    ("decode-aware-ratio1", ["--prefill-budget", "decode-aware", "--prefill-decode-steps", "1"], False),
)
REQUIRED_ARMS = sum(required for _, _, required in ARMS)


def server_command(command, arm):
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a nonempty JSON argv array")
    if any(arg.split("=", 1)[0] in PROTECTED for arg in command):
        raise ValueError("runner supplies max-sequences, queue-depth, admission and prefill-budget flags")
    policy = {name: arguments for name, arguments, _ in ARMS}.get(arm)
    if policy is None:
        raise ValueError(f"unknown arm {arm}")
    return command + ["--max-sequences", "32", "--queue-depth", "1024",
                      "--admission", "optimistic"] + policy


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def request_body(model, worker, turn):
    # The same request index has the same prompt and sampling in both arms.
    size = (64, 1024, 4096)[(worker + turn) % 3]
    return {"model": model, "prompt": f"soak-{worker}-{turn}: " + "Explain this. " * size,
            "max_tokens": (128, 512, 1024)[(worker + turn) % 3],
            "temperature": 0 if (worker + turn) % 2 == 0 else 0.7,
            "top_p": 0.9, "seed": worker * 100000 + turn,
            "stream": True, "logprobs": 1,
            "stream_options": {"include_usage": True}}


def measure(url, model, path, duration=ARM_SECONDS, warmup=False):
    """Run in a child so the parent can enforce a deadline on stuck streams."""
    start = time.monotonic()
    end = start + duration
    lock = threading.Lock()
    with open(path, "x") as output:
        def worker(index):
            client = Client(url, timeout=REQUEST_TIMEOUT_SECONDS, retries=0)
            turn = 0
            while time.monotonic() < end:
                began = time.monotonic()
                row = {"worker": index, "turn": turn, "started_s": began - start}
                try:
                    body = request_body(model, index, turn)
                    if warmup:
                        body.update(prompt="Warmup: count from one to ten.", max_tokens=32, temperature=0)
                    events, _ = client.request("/v1/completions", body, stream=True)
                    tokens, usage, finish = token_events(events)
                    if not tokens or not usage or not finish or any("error" in event for _, event in events):
                        raise ValueError("incomplete stream, missing usage or missing tokens")
                    row.update(ttft_ms=(tokens[0][0] - began) * 1000, usage=usage,
                               finish=finish, tokens=[[at - start, n] for at, n in tokens])
                except Exception as error:
                    row["error"] = str(error)
                with lock:
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                turn += 1
        clients = 1 if warmup else 64
        with ThreadPoolExecutor(max_workers=clients) as pool:
            list(pool.map(worker, range(clients)))


def summarize(path, duration=ARM_SECONDS):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [row for row in rows if "error" in row]
    ttft, itl, count = [], [], 0
    for row in rows:
        if "error" in row:
            continue
        ttft.append(row["ttft_ms"])
        previous = None
        for at, n in row["tokens"]:
            if at <= duration:
                count += n
                if previous is not None:
                    itl.append((at - previous) * 1000)
                # Tokens delivered in one SSE event share an arrival timestamp.
                itl.extend([0.0] * (n - 1))
            previous = at
    return {"requests": len(rows), "errors": errors, "ttft_p99_ms": percentile(ttft, .99),
            "itl_p99_ms": percentile(itl, .99), "itl_p50_ms": percentile(itl, .5),
            "ttft_p50_ms": percentile(ttft, .5), "tokens_per_second": count / duration,
            "measurement_seconds": duration, "offered_clients": 64, "max_sequences": 32}


def step_summary(path, start=0, end=None):
    """Summarize chopinfer.step.v1 lines written between two log byte offsets."""
    with open(path, "rb") as log:
        log.seek(start)
        text = log.read(None if end is None else max(0, end - start)).decode("utf-8", "replace")
    steps = []
    for line in text.splitlines():
        if '"schema":"chopinfer.step.v1"' in line:
            try:
                steps.append(json.loads(line))
            except ValueError:
                continue
    def phase(step):
        if step["prefill_tokens"] and step["decode_rows"] + step["verify_items"]:
            return "mixed"
        return "prefill" if step["prefill_tokens"] else "decode"
    execution = {}
    for name in ("decode", "mixed", "prefill"):
        times = [step["execution_us"] / 1000 for step in steps if phase(step) == name]
        execution[name] = {"steps": len(times), "p50_ms": percentile(times, .5),
                           "p99_ms": percentile(times, .99),
                           "mean_ms": sum(times) / len(times) if times else None}
    prefill = [step["prefill_tokens"] for step in steps if step["prefill_tokens"]]
    # Actual over assumed time for trained steps the estimates could price.
    ratios = [step["execution_us"] / step["predicted_us"] for step in steps
              if step["trained"] and step.get("predicted_us")]
    return {"steps": len(steps), "execution": execution,
            "mixed_share": execution["mixed"]["steps"] / len(steps) if steps else None,
            "prefill_tokens_per_prefill_step": sum(prefill) / len(prefill) if prefill else None,
            "selected_caps": dict(sorted(Counter(step["prefill_cap"] for step in steps
                                                 if step["prefill_cap"] is not None).items())),
            "actual_over_predicted": {"steps": len(ratios), "p10": percentile(ratios, .1),
                                      "p50": percentile(ratios, .5), "p90": percentile(ratios, .9)}}


def gates(fixed, aware):
    """The tuning gates for the default decode-aware arm against fixed."""
    loss = 100 * (1 - aware["tokens_per_second"] / fixed["tokens_per_second"])
    itl = aware["itl_p99_ms"] <= ITL_TARGET_MS
    return {"itl_target_ms": ITL_TARGET_MS, "itl_p99_within_target": itl,
            "throughput_loss_percent": loss, "throughput_within_5_percent": loss <= 5,
            "throughput_within_10_percent": loss <= 10, "qualified": itl and loss <= 10}


def stop(process):
    """Stop only the owned PID; never signal a process group or another server."""
    if process is not None and process.poll() is None:
        process.kill()
        process.wait()


def arm_fits(required, remaining):
    """Required arms always start; the optional arm needs its whole allowance."""
    return required or remaining >= ARM_RESERVE_SECONDS


def slot_seconds(end_utc, now):
    cutoff = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    if cutoff.utcoffset() is None or cutoff.utcoffset().total_seconds() != 0:
        raise ValueError("end-utc must have an explicit UTC timezone")
    remaining = min(MAX_RUN_SECONDS, cutoff.timestamp() - now)
    required = REQUIRED_ARMS * ARM_RESERVE_SECONDS + 30
    if remaining < required:
        raise ValueError(f"slot needs at least {required} seconds before end-utc")
    return remaining


def run(args):
    remaining = slot_seconds(args.end_utc, time.time())
    # Also use monotonic time so wall-clock corrections cannot extend the slot.
    deadline = time.monotonic() + remaining
    command = json.loads(Path(args.command_file).read_text())
    for name, _, _ in ARMS:
        server_command(command, name)
    output = Path(args.output).resolve()
    if not output.is_relative_to("/data"):
        raise ValueError("output must be under /data")
    output.mkdir(parents=True, exist_ok=False)
    result = {"model": args.model, "end_utc": args.end_utc, "arms": {}, "steps": {}, "passed": False,
              "skipped": {}, "max_run_seconds": MAX_RUN_SECONDS, "command": command,
              "policies": {name: arguments for name, arguments, _ in ARMS},
              "load": "C64 clients queued into 32 live sequences; not live C64"}
    server = worker = None
    current = (None, True)
    timed_out = threading.Event()
    ownership = threading.Lock()
    def expire():
        timed_out.set()
        with ownership:
            if worker is not None and worker.is_alive():
                worker.kill()
            stop(server)
    timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
    timer.start()
    try:
        # Refuse an occupied endpoint rather than accidentally measure another server.
        try:
            Client(args.url, timeout=2, retries=0).request("/v1/models")
        except RuntimeError:
            pass
        else:
            raise ValueError("endpoint is already serving; slot owner must stop it first")
        for mode, _, required in ARMS:
            if timed_out.is_set():
                raise TimeoutError("hard end time reached")
            if not arm_fits(required, deadline - time.monotonic()):
                result["skipped"][mode] = f"less than {ARM_RESERVE_SECONDS} seconds remained"
                continue
            current = (mode, required)
            log_path = output / f"{mode}-server.log"
            with log_path.open("x") as log:
                with ownership:
                    if timed_out.is_set():
                        raise TimeoutError("hard end time reached")
                    # One chopinfer.step.v1 line per step shows the selected cap and times.
                    server = subprocess.Popen(server_command(command, mode), stdout=log, stderr=subprocess.STDOUT,
                                              env=dict(os.environ, CHOPINFER_STEP_TRACE="1"))
                ready_by = min(deadline, time.monotonic() + STARTUP_SECONDS)
                client = Client(args.url, timeout=2, retries=0)
                while True:
                    if server.poll() is not None or time.monotonic() >= ready_by:
                        raise RuntimeError(f"{mode} startup failed or timed out")
                    try:
                        client.request("/health")
                        client.identity(args.model)
                        break
                    except (RuntimeError, ValueError):
                        time.sleep(.5)
                for phase, duration in (("warmup", WARMUP_SECONDS), ("requests", ARM_SECONDS)):
                    path = output / f"{mode}-{phase}.jsonl"
                    if phase == "requests":
                        (output / f"{mode}-metrics-before.txt").write_text(metrics_text(client))
                    worker = multiprocessing.get_context("spawn").Process(
                        target=measure, args=(args.url, args.model, path, duration, phase == "warmup"))
                    # Step lines between these offsets belong to the measured window.
                    window = {"start": log_path.stat().st_size}
                    window_end = threading.Timer(duration, lambda: window.setdefault("end", log_path.stat().st_size))
                    with ownership:
                        if timed_out.is_set():
                            raise TimeoutError("hard end time reached")
                        worker.start()
                    window_end.start()
                    drain = WARMUP_DRAIN_SECONDS if phase == "warmup" else DRAIN_SECONDS
                    worker.join(max(0, min(duration + drain, deadline - time.monotonic())))
                    window_end.cancel()
                    if worker.is_alive():
                        worker.kill()
                        worker.join()
                        raise TimeoutError(f"{mode} {phase} did not drain before deadline")
                    if worker.exitcode != 0 or timed_out.is_set():
                        raise RuntimeError(f"{mode} {phase} worker failed: {worker.exitcode}")
                    stats = summarize(path, duration)
                    if phase == "warmup" and (not stats["requests"] or stats["errors"]):
                        raise RuntimeError(f"{mode} warmup failed; inspect raw requests")
                client.request("/health")
                (output / f"{mode}-metrics-after.txt").write_text(metrics_text(client))
                result["arms"][mode] = stats
                result["steps"][mode] = step_summary(log_path, window["start"], window.get("end"))
                stop(server)
            if mode == "decode-aware":
                fixed, aware = (result["arms"][name] for name in ("fixed", "decode-aware"))
                result["passed"] = all(arm["requests"] and not arm["errors"] and arm["itl_p99_ms"] is not None
                                       for arm in (fixed, aware)) and aware["itl_p99_ms"] <= fixed["itl_p99_ms"]
                if result["passed"]:
                    result["gates"] = gates(fixed, aware)
    except Exception as error:
        if current[1]:
            result["error"] = str(error)
        else:
            # The comparison of the required arms stands without the optional one.
            result["skipped"][current[0]] = f"failed: {error}"
    finally:
        with ownership:
            if worker is not None and worker.is_alive():
                worker.kill()
                worker.join()
            stop(server)
        timer.cancel()
        timer.join()
        if timed_out.is_set():
            result["passed"] = False
            result["error"] = "hard end time reached"
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end-utc", required=True, help="Hard cutoff, e.g. 2026-09-26T04:00:00Z")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
