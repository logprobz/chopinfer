#!/usr/bin/env python3
"""Compare served DSpark arms in balanced order on fresh TP4 servers.

Every pass starts a fresh server. Served token IDs and every reported
logprob bit must equal the first base pass at C1, C8, C16 and C32, greedy
and seeded sampling. Drafter blocks logged by both arms at the same request
and committed row count must carry the same proposal IDs and confidence
bits. The job then measures steady decode throughput, acceptance per draft
position and per-step costs by items, drafting items and rows.
"""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import sys
import threading
import time

import verify_width_job as exact

CONCURRENCIES = (1, 8, 16, 32)
ORDER = ("base", "candidate", "candidate", "base")
# Staggered phases admit requests in submission order, so engine request
# IDs map to prompts and drafter blocks join across servers.
IDENTITY = {8: 128, 16: 96}
STAGGER_SECONDS = 0.4
# The drafter log is asynchronous; let a phase's records land before the
# next phase's offset.
SETTLE_SECONDS = 0.5
RECORD = "chopinfer.deepseek41.drafter-confidence.v1"
STEP = "chopinfer.step.v1"


def log_offset(server):
    path = server.directory / "server.log"
    return path.stat().st_size if path.exists() else 0


def log_lines(server, start, end, schema):
    """Parsed JSON records of one schema between two server log offsets."""
    with (server.directory / "server.log").open("rb") as stream:
        stream.seek(start)
        data = stream.read(end - start)
    rows = []
    for line in data.decode(errors="replace").splitlines():
        if schema not in line:
            continue
        try:
            row = json.loads(line[line.index("{"):])
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") == schema:
            rows.append(row)
    return rows


def blocks(records, requests):
    """Committed drafter blocks keyed by (request order, committed rows).

    Engine request IDs rise in admission order. A phase whose IDs are not
    `requests` consecutive values cannot be mapped and returns None.
    """
    committed = [r for r in records if r.get("completion") == "committed"]
    ids = sorted({int(r["request_id"]) for r in committed})
    if len(ids) != requests or ids[-1] - ids[0] != requests - 1:
        return None
    result = {}
    for record in committed:
        key = (int(record["request_id"]) - ids[0], int(record["main_rows"]))
        value = (record["draft_ids"], record["confidence_f32_bits"])
        if result.setdefault(key, value) != value:
            raise ValueError(f"one server logged two different blocks at {key}")
    return result


def compare_blocks(base, other):
    if base is None or other is None:
        return {"conclusive": False, "compared": 0, "differences": []}
    common = sorted(base.keys() & other.keys())
    differences = [{"request": key[0], "main_rows": key[1]}
                   for key in common if base[key] != other[key]]
    return {"conclusive": bool(common), "compared": len(common),
            "base_blocks": len(base), "other_blocks": len(other),
            "differences": differences}


def step_costs(rows):
    """Median decode-only step time by items, drafting items and rows."""
    groups = {}
    for row in rows:
        if row.get("prefill_items") or row.get("prefill_tokens"):
            continue
        micros = row.get("execution_us")
        if not isinstance(micros, (int, float)) or not math.isfinite(micros) or micros <= 0:
            continue
        key = (row["decode_rows"] + row["verify_items"], row["verify_items"],
               row["decode_rows"] + row["verify_rows"])
        groups.setdefault(key, []).append(micros)
    return [{"items": key[0], "drafting_items": key[1], "rows": key[2],
             "steps": len(values), "median_us": statistics.median(values)}
            for key, values in sorted(groups.items())]


def identity(p, server, prompts, concurrency, tokens, name):
    """Greedy requests submitted one at a time, then run concurrently."""
    directory = server.directory / "responses" / name

    def request(index):
        time.sleep(index * STAGGER_SECONDS)
        body = {"model": p.MODEL, "prompt": prompts[index], "max_tokens": tokens,
                "min_tokens": tokens, "ignore_eos": True, "temperature": 0, "top_p": 1,
                "logprobs": 5, "return_token_ids": True, "stream": False}
        p.write(directory / f"{index:03d}.request.json", body)
        raw = p.http(server.url, "/v1/completions", body, timeout=240)
        path = directory / f"{index:03d}.response.json"
        path.write_text(raw, encoding="utf-8")
        try:
            response = json.loads(raw)
            normalized = exact.canonical(response, tokens=tokens)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{error}; raw response saved to {path}") from error
        return {"request": body, "canonical": normalized}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(request, range(concurrency)))


def measure(p, server, concurrency):
    tokens = 512 if concurrency <= 8 else 256
    directory = server.directory / f"decode-c{concurrency}"
    directory.mkdir()
    before = p.metrics(server, f"c{concurrency}-before")
    start = log_offset(server)
    argv = [sys.executable, "-B", p.SOURCE / "tools/qual/bench.py", "--url", server.url,
            "--model", p.MODEL, "--concurrencies", str(concurrency), "--max-sequences", "32",
            "--output-tokens", str(tokens), "--prompts", "diverse", "--sampling", "greedy",
            "--timeout", "180", "--spec-metrics-arm", "chopinfer", "--prefill-tokens", "128",
            "--prefill-repeats", "8", "--prefill-prefix", f"short untimed auxiliary {time.time_ns()} "]
    p.command(argv, directory / "bench.log", seconds=210, env=p.env_for(directory))
    receipts = list((directory / "receipts").glob("*-bench.json"))
    if len(receipts) != 1:
        raise ValueError("expected one benchmark receipt")
    receipt = json.loads(receipts[0].read_text())
    if not receipt["passed"] or receipt["skipped"] or receipt["mechanics_only"]:
        raise ValueError("benchmark did not execute the requested workload")
    row = receipt["decode"][0]
    if len(row["requests"]) != concurrency or any(
            r["usage"]["completion_tokens"] != tokens for r in row["requests"]):
        raise ValueError("incomplete performance cohort")
    delta = p.spec_metrics.difference(before, p.metrics(server, f"c{concurrency}-after"))
    verified = delta.get("verify_items", 0)
    accepted = delta.get("accepted_tokens", 0)
    drafted = delta.get("draft_tokens", 0)
    return {"output_tokens": tokens, "steady": row["steady"], "cohort": row["cohort"],
            "host_load": receipt["host_load"], "receipt": str(receipts[0]),
            "metrics_delta": delta,
            "acceptance": {"verify_items": verified, "accepted_tokens": accepted,
                           "draft_tokens": drafted,
                           "accepted_per_verify_item": accepted / verified if verified else None,
                           "accepted_fraction": accepted / drafted if drafted else None,
                           "per_position": delta.get("per_position", [])},
            "step_costs": step_costs(log_lines(server, start, log_offset(server), STEP))}


def decision(passes):
    if tuple(row["arm"] for row in passes) != ORDER:
        raise ValueError("need all four balanced passes")
    result = {}
    for c in CONCURRENCIES:
        rows = [row["decode"][str(c)] for row in passes]
        rates = [row["steady"]["aggregate_tokens_per_second"] for row in rows]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("invalid measured throughput")
        result[str(c)] = {
            "base_tokens_per_second": [rates[0], rates[3]],
            "candidate_tokens_per_second": [rates[1], rates[2]],
            "paired_ratios": [rates[1] / rates[0], rates[2] / rates[3]],
            "median_base_tokens_per_second": statistics.median([rates[0], rates[3]]),
            "median_candidate_tokens_per_second": statistics.median([rates[1], rates[2]]),
            "accepted_per_verify_item": [row["acceptance"]["accepted_per_verify_item"] for row in rows],
            "verify_items": [row["acceptance"]["verify_items"] for row in rows],
            "timing": [row["host_load"].get("timing") for row in rows]}
    return {"concurrencies": result,
            "timing_evidence": all(t == "evidence" for c in result.values() for t in c["timing"])}


def run(args):
    stage = Path(args.stage).resolve()
    sys.path.insert(0, str(stage))
    import perf_runner as p
    from common import host_load, host_load_sample
    manifest = exact.validate(stage)
    if not args.dry_run and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("the job must select all four GPUs explicitly")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "dry_run": args.dry_run, "stage": manifest,
              "passes": [], "comparisons": {}, "draft_comparisons": {}, "errors": []}
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, p.interrupted)

    def record_host(label):
        samples = [host_load_sample()]
        time.sleep(1)
        samples.append(host_load_sample())
        load = host_load(samples)
        p.write(output / f"host-{label}.json", load)
        return load

    try:
        result["host_before_loading"] = record_host("before-loading")
        result["before"] = p.telemetry(output, "before", args.dry_run, since)
        prompts = None
        captures = []
        drafts = []
        for number, arm in enumerate(ORDER):
            p.INSTALL = Path(manifest["arms"][arm]["install"])
            p.CACHE = Path(manifest["arms"][arm]["cache"])
            parent = output / f"pass-{number}-{arm}"
            parent.mkdir()
            server = p.Server(parent, "tp4-dspark", dry=args.dry_run, use_nsys=False)
            extra = manifest["arms"][arm].get("server_args", [])
            if extra and not args.dry_run:
                server.launch = [*server.launch, *extra]
                server.server_command = [*server.server_command, *extra]
            record = {"arm": arm, "server_args": extra, "decode": {}}
            result["passes"].append(record)
            try:
                server.start()
                record["readiness_seconds"] = server.readiness
                record["host_ready"] = record_host(f"pass-{number}-ready")
                if prompts is None:
                    prompts = [json.loads(p.http(server.url, "/tokenize", {
                        "model": p.MODEL, "prompt": p.bench.stream_prompt("diverse", i),
                        "add_special_tokens": False}))["tokens"] for i in range(32)]
                    p.write(output / "prompts.json", prompts)
                # Log offsets bound each phase; only C1 (greedy, then seeded,
                # in sequence) and the staggered phases map request IDs.
                phases = []
                captured = {}
                for c in CONCURRENCIES:
                    start = log_offset(server)
                    captured.update(exact.capture(p, server, prompts, "exactness", (c,)))
                    time.sleep(SETTLE_SECONDS)
                    phases.append((f"c{c}", start, log_offset(server), 2 if c == 1 else None))
                for c, tokens in IDENTITY.items():
                    start = log_offset(server)
                    captured[f"identity-c{c}"] = identity(p, server, prompts, c, tokens,
                                                          f"identity-c{c}")
                    time.sleep(SETTLE_SECONDS)
                    phases.append((f"identity-c{c}", start, log_offset(server), c))
                captures.append(captured)
                p.write(server.directory / "exactness.json", captured)
                record["exactness"] = exact.capture_stats(
                    {k: v for k, v in captured.items() if not k.startswith("identity")})
                parsed = {name: blocks(log_lines(server, start, stop, RECORD), requests)
                          for name, start, stop, requests in phases if requests}
                drafts.append(parsed)
                record["draft_blocks"] = {name: None if value is None else len(value)
                                          for name, value in parsed.items()}
                if number:
                    compared = exact.compare(captures[0], captured)
                    result["comparisons"][f"pass-0-pass-{number}"] = compared
                    if not compared["bitwise"]:
                        raise ValueError("served token IDs or logprob bits differ")
                    draft = {name: compare_blocks(drafts[0][name], parsed[name])
                             for name in parsed}
                    result["draft_comparisons"][f"pass-0-pass-{number}"] = draft
                    if any(row["differences"] for row in draft.values()):
                        raise ValueError("drafter proposals or confidence bits differ")
                for c in CONCURRENCIES:
                    record["decode"][str(c)] = measure(p, server, c)
                    p.write(output / "summary.json", result)
            finally:
                server.stop()
                exact.idle(p, output, f"pass-{number}", args.dry_run)
        result["performance"] = decision(result["passes"])
        result["drafts_conclusive"] = all(
            row["conclusive"] for compared in result["draft_comparisons"].values()
            for row in compared.values())
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            exact.idle(p, output, "final", args.dry_run)
            result["after"] = p.telemetry(output, "after", args.dry_run, since)
            if result["after"]["xids"]:
                raise ValueError("kernel Xids during job")
        except BaseException as error:
            result["status"] = "failed"
            result["errors"].append(str(error))
        p.write(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(output),
                      "errors": result["errors"]}), flush=True)
    return int(result["status"] != "passed")


def stub(args):
    """A CPU HTTP server with the served surface this job reads.

    Outputs depend only on the request, so every pass compares equal. Each
    completion logs drafter blocks and step traces as the server would.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    lock = threading.Lock()
    counters = dict(verify_items=0, draft_tokens=0, accepted_tokens=0, emitted_tokens=0, rows=0)
    state = {"request": 100}

    def emit(value):
        with lock:
            print(json.dumps(value, separators=(",", ":")), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def respond(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.respond({"status": "ok", "synthetic": True})
            elif self.path == "/v1/models":
                self.respond({"data": [{"id": args.model}]})
            elif self.path == "/metrics":
                with lock:
                    totals = counters.copy()
                lines = [f"chop_spec_{key}_total {value}" for key, value in totals.items()]
                for position in range(1, 6):
                    lines += [f'chop_spec_drafted_position_total{{position="{position}"}} '
                              f'{totals["draft_tokens"] // 5}',
                              f'chop_spec_accepted_position_total{{position="{position}"}} '
                              f'{totals["accepted_tokens"] // 5}']
                self.send_response(200)
                self.end_headers()
                self.wfile.write(("\n".join(lines) + "\n").encode())
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.respond({"tokens": list(range(40)), "count": 40})
                return
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            with lock:
                state["request"] += 1
                request = state["request"]
            budget = body["max_tokens"]
            prompt = body["prompt"]
            prompt_tokens = len(prompt) if isinstance(prompt, list) else 128
            usage = dict(prompt_tokens=prompt_tokens, completion_tokens=budget,
                         total_tokens=prompt_tokens + budget)
            seed = sum(prompt) if isinstance(prompt, list) else 7
            for main_rows in range(prompt_tokens, prompt_tokens + budget, 4):
                ids = [(seed + main_rows + i) % 1000 for i in range(5)]
                emit({"schema": "chopinfer.deepseek41.drafter-confidence.v1",
                      "request_id": str(request), "main_rows": main_rows,
                      "completion": "committed", "draft_ids": ids,
                      "confidence_f32_bits": [struct.unpack("<I", struct.pack("<f", i / 8))[0]
                                              for i in range(5)]})
                emit({"schema": "chopinfer.step.v1", "prefill_items": 0, "prefill_tokens": 0,
                      "decode_rows": 0, "verify_items": 1, "verify_rows": 6,
                      "execution_us": 20000 + main_rows % 7, "host": {}})
            if not body.get("stream"):
                tokens = [(seed + i) % 1000 for i in range(budget)]
                labels = [f"t{token}" for token in tokens]
                top = [{label: -0.125, "other": -2.5} for label in labels]
                self.respond({"choices": [{
                    "text": "".join(labels), "finish_reason": "length", "token_ids": tokens,
                    "logprobs": {"tokens": labels, "token_logprobs": [-0.125] * budget,
                                 "top_logprobs": top, "text_offset": list(range(budget))}}],
                    "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for step in range(budget):
                    time.sleep(.002)
                    event = {"choices": [{"text": "x", "logprobs": {"tokens": ["x"]},
                                          "finish_reason": "length" if step == budget - 1 else None}]}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
                                  "data: [DONE]\n\n").encode())
                self.wfile.flush()
                if budget > 1:
                    with lock:
                        counters["verify_items"] += 100
                        counters["draft_tokens"] += 500
                        counters["accepted_tokens"] += 350
                        counters["emitted_tokens"] += 450
                        counters["rows"] += 600
            except (BrokenPipeError, ConnectionResetError):
                pass

    class StubServer(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    StubServer(("127.0.0.1", args.port), Handler).serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    job = commands.add_parser("run")
    job.add_argument("--stage", required=True)
    job.add_argument("--output", required=True)
    job.add_argument("--dry-run", action="store_true")
    fake = commands.add_parser("stub")
    fake.add_argument("--port", type=int, required=True)
    fake.add_argument("--profile")
    fake.add_argument("--directory")
    fake.add_argument("--prefill-chunk", type=int)
    fake.add_argument("--model", default="DeepSeek-V4.1-Flash")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else stub(args)


if __name__ == "__main__":
    raise SystemExit(main())
