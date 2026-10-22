"""Sampled width matrix and a subsequent four-server ABBA confirmation."""
import argparse
import concurrent.futures
import collections
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import analysis

MODEL = "DeepSeek-V4.1-Flash"
CONFIGS = {
    "k0": (0, None),
    "uniform1": (1, "balanced"),
    "uniform2": (2, "balanced"),
    "uniform3": (3, "balanced"),
    "fixed5": (5, "fixed"),
    "adaptive": (5, "adaptive"),
    "balanced": (5, "balanced"),
}
ORDER = tuple(CONFIGS)
TOKENS = 192
WARM_TOKENS = 64
MIN_FULL_STEPS = 8
SCHEMA = "chopinfer.deepseek41.width-matrix.v1"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(text, schema):
    result = []
    for line in text.splitlines():
        if "diagnostic writer: dropped " in line:
            assert int(line.split("diagnostic writer: dropped ")[1].split()[0]) == 0
        if schema in line:
            row = json.loads(line[line.index("{"):])
            if row.get("schema") == schema:
                result.append(row)
    return result


def cache_ready(text, *, required=True):
    rows = records(text, "chopinfer.mapped-row-cache.v1")
    assert not any(r["phase"] == "budget-skip" for r in rows), "resident cache budget skipped"
    ready = [r for r in rows if r["phase"] == "ready"]
    if not ready and not required:
        return []
    assert len(ready) == 2 and len({r["cache_id"] for r in ready}) == 2, "both resident sets required"
    for r in ready:
        assert r["charged_bytes"] == 1 << 30 and r["budget_bytes"] == 2 << 30
        assert 0 < r["locked_bytes"] <= r["charged_bytes"]
        assert r["process_charged_bytes"] <= r["budget_bytes"]
    return ready


def faults(pid):
    root = Path("/proc") / str(pid)
    parts = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    io = {k: int(v) for k, v in (line.split(":", 1) for line in (root / "io").read_text().splitlines())}
    return dict(pid=pid, start_ticks=int(parts[19]), minflt=int(parts[7]), majflt=int(parts[9]),
                read_bytes=io["read_bytes"], at_ns=time.time_ns())


def fault_delta(before, after):
    assert before["pid"] == after["pid"] and before["start_ticks"] == after["start_ticks"]
    out = {k: after[k] - before[k] for k in ("minflt", "majflt", "read_bytes")}
    assert min(out.values()) >= 0
    return out


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
    # A process-group child must not survive its leader.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def case_label(config, concurrency):
    if concurrency == 16 and config in ("uniform2", "uniform3"):
        return f"{config}-budgeted-to-32-rows-C16"
    return f"{config}-C{concurrency}"


def fallback_report(steps, config, concurrency, *, required=True):
    full = [s for s in steps if not s["prefill_items"]
            and s["decode_rows"] + s["verify_items"] == concurrency]
    assert full, "no full-concurrency fallback observations"
    plans = [s["speculation"] for s in full if s.get("speculation") is not None]
    if required and config != "k0":
        assert plans, "policy diagnostics missing from frozen binary"
    reasons = collections.Counter(p["reason"] for p in plans)
    assert all(p["mode"] == CONFIGS[config][1] for p in plans)
    known = {"no_spare_rows", "no_eligible_items", "fixed", "balanced", "table",
             "prefill_fresh", "await_ar_baseline", "cooldown", "realized_slower",
             "estimated_slower", "sparse_cost", "probe", "drafting"}
    assert set(reasons) <= known
    zero = sum(s["verify_items"] == 0 for s in full)
    backoff = {k: reasons[k] for k in ("estimated_slower", "realized_slower", "cooldown", "sparse_cost")}
    return dict(full_steps=len(full), planned_steps=len(plans), no_plan_steps=len(full)-len(plans),
                reason_counts=dict(reasons), reason_rates_per_full_step={k: v/len(full) for k,v in reasons.items()},
                reason_rates_per_plan={k: v/len(plans) for k,v in reasons.items()},
                actual_zero_verify_steps=zero, actual_zero_verify_rate=zero/len(full),
                selected_zero_draft_plans=sum(p["planned_drafts"] == 0 for p in plans),
                backoff_steps=sum(backoff.values()), backoff_rate=sum(backoff.values())/len(full),
                backoff_counts=backoff,
                denominator="Full-concurrency completed steps, before discarding four for step-time statistics. Plans that were not invoked are counted separately.")


def audit_window(text, metrics, config, concurrency, *, exact_geometry=True, diagnostics=True):
    steps = analysis.parse(text, require_phase=config != "k0")
    audit = analysis.coverage(steps, metrics)
    assert audit["per_item_exact"], "acceptance labels remain incomplete"
    full = [s for s in steps if not s["prefill_items"]
            and s["decode_rows"] + s["verify_items"] == concurrency]
    assert len(full) >= MIN_FULL_STEPS + 4, "too few full-concurrency steps"
    steady = full[4:]
    assert len(steady) >= MIN_FULL_STEPS
    packets = records(text, "chopinfer.device-results.v1")
    epochs = {p["cycle_epoch"] for p in packets if p.get("kind") == "acceptance_packet"}
    pure_verify = [s for s in steps if not s["prefill_items"] and s["verify_items"]]
    for s in pure_verify:
        assert int(s["phase"]["step_id"]) in epochs, "missing device acceptance packet"
        assert s["live_rows"] <= 32
        if exact_geometry:
            assert s["row_capacity"] == s["live_rows"], "verify costs still use padded rows"
    if config in ("uniform1", "uniform2", "uniform3", "balanced"):
        cap = CONFIGS[config][0]
        depth = min(cap, (32 - concurrency) // concurrency)
        assert sum(s["shape"] == [depth] * concurrency for s in full) >= MIN_FULL_STEPS
    if config == "fixed5" and concurrency == 8:
        assert sum(s["shape"] == [5, 5, 5, 5, 4, 0, 0, 0] for s in full) >= MIN_FULL_STEPS
    if config == "k0":
        assert not pure_verify and metrics["draft_tokens"] == 0
    return dict(coverage=audit, fallback=fallback_report(steps, config, concurrency, required=diagnostics), full_steps=len(full), discarded_full_steps=4,
                full=analysis.group(steady), all_pure=analysis.group([
                    s for s in steps if not s["prefill_items"]]),
                device_verify_steps=len(pure_verify), device_packet_epochs=len(epochs)), steps


def choose_winner(receipt):
    assert receipt["schema"] == SCHEMA and receipt["status"] == "passed"
    assert receipt["mode"] == "matrix" and not receipt["dry_run"]
    assert receipt["server_count"] == 7 and receipt["case_count"] == 14
    assert receipt["bitwise"]
    rates = {r["config"]: {c["concurrency"]: c["steady"]["aggregate_tokens_per_second"]
                            for c in r["cohorts"]} for r in receipt["servers"]}
    assert set(rates) == set(CONFIGS)
    assert all(set(v) == {8, 16} and all(math.isfinite(x) and x > 0 for x in v.values())
               for v in rates.values())
    current = rates["adaptive"]
    # A single configuration must cover both concurrencies. Higher K caps
    # still obey the batch's 32-row limit at C16.
    scores = {name: math.sqrt(v[8] / current[8] * v[16] / current[16]) for name, v in rates.items()}
    winner = max(ORDER, key=lambda name: scores[name])
    if winner == "adaptive" or scores[winner] <= 1:
        raise ValueError("matrix has no candidate faster than current adaptive serving")
    return winner, scores


def confirmation_rates(servers, winner):
    ratios = {}
    separated = {}
    for c in (8, 16):
        current = [cohort for server in servers if server['config'] == 'adaptive'
                   for cohort in server['cohorts'] if cohort['concurrency'] == c]
        candidate = [cohort for server in servers if server['config'] == winner
                     for cohort in server['cohorts'] if cohort['concurrency'] == c]
        assert len(current) == len(candidate) == 2
        rate = lambda rows: sum(r['steady']['tokens'] for r in rows) / sum(r['steady']['shared_seconds'] for r in rows)
        ratios[str(c)] = rate(candidate) / rate(current)
        separated[str(c)] = min(r['steady']['aggregate_tokens_per_second'] for r in candidate) > max(
            r['steady']['aggregate_tokens_per_second'] for r in current)
    idle = all(c['host_load']['timing'] == 'evidence' for s in servers for c in s['cohorts'])
    return dict(candidate_over_current=ratios, repeats_separated=separated, host_idle=idle,
                recommended_default=winner if idle and all(separated.values()) else None,
                rule='Recommend only when both candidate repetitions beat both current repetitions at each concurrency and every measured host window is idle. No serving defaults are edited by this job.')


def markdown(result):
    power = "CPU simulation; no GPU power measurement" if result["dry_run"] else "250 W cap on each of four GPUs, checked before and after every server"
    lines = [f"Status: {result['status']}. {result['timing']}. {power}.", "",
             "| Case | tok/s | Verify steps | Timed zero-verify rate | Timed backoff rate | Warmup backoff rate |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for server in result['servers']:
        for c in server['cohorts']:
            f = c['step_audit']['fallback']
            lines.append(f"| {c['label']} | {c['steady']['aggregate_tokens_per_second']:.2f} | {c['step_audit']['device_verify_steps']} | {f['actual_zero_verify_rate']:.1%} | {f['backoff_rate']:.1%} | {c['warmup_fallback']['backoff_rate']:.1%} |")
    lines.extend(['', 'Per-reason counts and rates, exact cost snapshots, positional acceptance, rows, step times and process faults are in the JSON receipts.', '', 'C16 uniform K2 and K3 require 48 and 64 rows and are unsupported. Their cap configurations are measured under the existing 32-row limit, with budgeted labels.'])
    if result.get('confirmation'):
        lines.extend(['', 'Confirmation: ' + json.dumps(result['confirmation'], sort_keys=True)])
    return '\n'.join(lines) + '\n'


class Runner:
    def __init__(self, stage, output, mode, dry=False, winner=None):
        self.stage, self.output, self.mode, self.dry = Path(stage), Path(output), mode, dry
        self.output.mkdir(parents=True, exist_ok=False)
        self.settings = json.loads((self.stage / "config.json").read_text())
        self.source = Path(self.settings["source"])
        sys.path[:0] = [str(self.source / "tools/qual"), str(self.source / "models/deepseek41/lab")]
        self.common = importlib.import_module("common")
        self.bench = importlib.import_module("bench")
        self.metrics = importlib.import_module("spec_metrics")
        self.exact = importlib.import_module("verify_width_job")
        self.started = time.monotonic() if dry else float(os.environ.get("X7_JOB_STARTED_MONOTONIC", time.monotonic()))
        self.deadline = self.started + (90 if dry else 1140)
        self.server = None
        self.log = None
        self.reference = {}
        self.winner = winner
        self.started_unix = int(time.time())
        self.env = dict(os.environ, CHOPINFER_DEVICE_RESULTS="1", CHOPINFER_MAPPED_HOT_ROWS="1",
                        CHOPINFER_MAPPED_READ_AHEAD="0", CHOPINFER_MAPPED_HOT_ROWS_SET_BYTES=str(1 << 30),
                        CHOPINFER_MAPPED_HOT_ROWS_BYTES=str(2 << 30), CHOPINFER_STEP_TRACE="1",
                        CHOPINFER_STEP_PROFILE_EVERY="1", PYTHONDONTWRITEBYTECODE="1")
        for key in ("CHOPINFER_DSPARK_STATE_DIR", "CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK"):
            self.env.pop(key, None)

    def remaining(self, maximum):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("shared execution deadline; final cleanup reserve retained")
        return min(maximum, left)

    def command(self, args, timeout=10, cleanup=False):
        return subprocess.run(args, check=True, capture_output=True, text=True,
                              timeout=timeout if cleanup else self.remaining(timeout)).stdout

    def gpu_guard(self, label, idle=False):
        if self.dry:
            write(self.output / f"{label}-power.json", {str(i): 250 for i in range(4)})
            return
        text = self.command(["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"], cleanup=label == "final")
        powers = {int(a.strip()): float(b.strip()) for a, b in (line.split(",") for line in text.splitlines())}
        assert powers == {i: 250.0 for i in range(4)}, powers
        write(self.output / f"{label}-power.json", powers)
        if idle:
            processes = self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            assert not processes.strip(), "GPU compute processes are already active"
        journal = self.command(["journalctl", "-k", "--since", f"@{self.started_unix}", "--no-pager"], cleanup=label == "final")
        (self.output / f"{label}-kernel.log").write_text(journal)
        assert not re.search(r"NVRM.*Xid", journal), "GPU Xid during job"

    def argv(self, config, port):
        cap, mode = CONFIGS[config]
        if self.dry:
            return [sys.executable, "-B", str(Path(__file__).with_name("stub_server.py")), str(port), config]
        argv = [self.settings["binary"], "serve", "--bundle", self.settings["package"],
                "--checkpoint", self.settings["checkpoint"], "--profile",
                str(self.source / "models/deepseek41/profiles.json") + "#tp4-dspark",
                "--devices", "0,1,2,3", "--kv-cache-dtype", "auto", "--prefix-cache", "off",
                "--max-sequences", "16", "--max-batch-tokens", "8192", "--prefill-chunk-tokens", "8192",
                "--verify-warmup", "off", "--speculative-tokens", str(cap),
                "--served-model-name", MODEL, "--listen", f"127.0.0.1:{port}"]
        if cap:
            argv += ["--speculation", mode, "--draft-cost", "per-block", "--speculative-rows", "32"]
        return argv

    def query(self, route, body=None, raw=False, timeout=10, headers=None):
        request = urllib.request.Request(self.url + route,
                   None if body is None else json.dumps(body).encode(),
                   {"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(request, timeout=self.remaining(timeout)) as response:
            data = response.read().decode()
        return data if raw else json.loads(data)

    @contextlib.contextmanager
    def serving(self, config, epoch):
        self.directory = self.output / f"{epoch}-{config}"
        self.directory.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        argv = self.argv(config, port)
        write(self.directory / "command.json", argv)
        write(self.directory / "environment.json", {k: self.env[k] for k in self.env if k in (
            "CHOPINFER_DEVICE_RESULTS", "CHOPINFER_MAPPED_HOT_ROWS", "CHOPINFER_MAPPED_READ_AHEAD",
            "CHOPINFER_STEP_TRACE", "CHOPINFER_STEP_PROFILE_EVERY")})
        self.log = self.directory / "server.log"
        with self.log.open("wb") as log:
            self.server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                           env=self.env, start_new_session=True)
            try:
                write(self.directory / "process.json", {"pid": self.server.pid})
                start = time.monotonic()
                readiness_deadline = start + self.remaining(self.settings.get("startup_timeout", 170))
                while True:
                    assert self.server.poll() is None, "server exited during startup"
                    if time.monotonic() >= readiness_deadline:
                        raise TimeoutError("server readiness deadline")
                    try:
                        health = self.query("/health", timeout=1)
                        break
                    except InterruptedError:
                        raise
                    except (OSError, ValueError):
                        time.sleep(.1 if self.dry else .5)
                write(self.directory / "startup.json", {"seconds": time.monotonic() - start, "health": health})
                text = self.log.read_text()
                assert "warmup incomplete:" not in text
                yield
            finally:
                stop(self.server)
                write(self.directory / "cleanup.json", {"returncode": self.server.returncode})
                self.server = None

    def stream(self, index, concurrency, tokens, barrier):
        body = dict(model=MODEL, prompt=self.bench.stream_prompt("diverse", index),
                    max_tokens=tokens, min_tokens=tokens, ignore_eos=True, temperature=1.0,
                    top_p=1.0, seed=710+index, logprobs=5, return_token_ids=True,
                    stream=True, stream_options={"include_usage": True})
        headers = {"Content-Type": "application/json"}
        if self.dry:
            headers["X-X7-Test-Concurrency"] = str(concurrency)
        barrier.wait(timeout=self.remaining(10))
        started = time.time_ns()
        request = urllib.request.Request(self.url + "/v1/completions", json.dumps(body).encode(), headers)
        ids, labels, selected, tops, events = [], [], [], [], []
        usage, finish, done = None, None, False
        with urllib.request.urlopen(request, timeout=self.remaining(30)) as response:
            for line in response:
                self.remaining(30)
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(payload)
                assert not chunk.get("error"), chunk
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    part = choice.get("token_ids") or []
                    if not part:
                        continue
                    logprobs = choice["logprobs"]
                    assert all(len(logprobs[k]) == len(part) for k in ("tokens", "token_logprobs", "top_logprobs"))
                    ids.extend(part)
                    labels.extend(logprobs["tokens"])
                    selected.extend(logprobs["token_logprobs"])
                    tops.extend(logprobs["top_logprobs"])
                    events.append([time.time_ns()/1e9, len(part)])
        assert done and events
        response = {"usage": usage, "choices": [{"finish_reason": finish, "token_ids": ids,
                    "logprobs": {"tokens": labels, "token_logprobs": selected, "top_logprobs": tops}}]}
        canonical = self.exact.canonical(response, sampled=True, tokens=tokens)
        return dict(index=index, request=body, canonical=canonical, events=events, start_ns=started,
                    end_ns=time.time_ns(), usage=usage)

    def wave(self, concurrency, tokens):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
        barrier = threading.Barrier(concurrency)
        try:
            futures = [pool.submit(self.stream, i, concurrency, tokens, barrier) for i in range(concurrency)]
            return [f.result(timeout=self.remaining(45)) for f in futures]
        except BaseException:
            stop(self.server)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def scrape(self, path=None):
        text = self.query("/metrics", raw=True)
        if path is not None:
            Path(path).with_suffix('.prom').write_text(text)
        snapshot = self.metrics.snapshot(text, "chopinfer", MODEL)
        if path is not None:
            write(Path(path).with_suffix('.snapshot.json'), snapshot)
        return snapshot

    def cohort(self, config, concurrency):
        path = self.directory / f"C{concurrency}"
        path.mkdir()
        warm_before = self.scrape(path / "warm-before")
        warm_begin = self.log.stat().st_size
        warm = self.wave(concurrency, WARM_TOKENS)
        # A settled scrape and drained log establish the exact measurement boundary.
        write(path / "warm-outputs.json", warm)
        before = self.scrape(path / "warm-after")
        time.sleep(.02 if self.dry else .2)
        assert before == self.scrape(path / "warm-settled"), "warmup counters did not settle"
        begin = self.log.stat().st_size
        warm_text = self.log.read_bytes()[warm_begin:begin].decode()
        (path / "warmup.log").write_text(warm_text)
        write(path / "warm-boundary.json", dict(start_bytes=warm_begin, end_bytes=begin))
        warm_delta = self.metrics.difference(warm_before, before)
        write(path / "warm-metrics.json", dict(before=warm_before, after=before, delta=warm_delta))
        warm_steps = analysis.parse(warm_text, require_phase=config != "k0")
        warm_coverage = analysis.coverage(warm_steps, warm_delta)
        assert warm_coverage["per_item_exact"]
        warm_fallback = fallback_report(warm_steps, config, concurrency)
        write(path / "warmup.json", dict(metrics=warm_delta, coverage=warm_coverage,
                                        fallback=warm_fallback, outputs=warm))
        (path / "warmup-steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in warm_steps))
        process_before = faults(self.server.pid)
        load_before = self.common.host_load_sample()
        requests = self.wave(concurrency, TOKENS)
        write(path / "outputs.json", requests)
        after = self.scrape(path / "timed-after")
        time.sleep(.02 if self.dry else .2)
        assert after == self.scrape(path / "timed-settled"), "measurement counters did not settle"
        process_after = faults(self.server.pid)
        load_after = self.common.host_load_sample()
        text = self.log.read_bytes()[begin:].decode()
        (path / "timed.log").write_text(text)
        write(path / "timed-boundary.json", dict(start_bytes=begin, end_bytes=self.log.stat().st_size))
        write(path / "faults.json", {"before": process_before, "after": process_after})
        delta = self.metrics.difference(before, after)
        write(path / "metrics.json", {"before": before, "after": after, "delta": delta})
        audit, steps = audit_window(text, delta, config, concurrency)
        for a, b in zip(warm, requests):
            for field in ("token_ids", "tokens", "selected_bits", "top_bits"):
                assert a["canonical"][field] == b["canonical"][field][:WARM_TOKENS], "warm replay bits differ"
        key = str(concurrency)
        if key not in self.reference:
            self.reference[key] = requests
        parity = self.exact.compare({key: self.reference[key]}, {key: requests})
        assert parity["bitwise"], parity
        (path / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
        result = dict(label=case_label(config, concurrency), concurrency=concurrency,
                      sampling={"temperature": 1.0, "top_p": 1.0}, request_count=len(requests),
                      output_tokens=TOKENS, positional_acceptance=delta, step_audit=audit, warmup_fallback=warm_fallback,
                      cohort=self.bench.cohort([r["events"] for r in requests]),
                      steady=self.bench.steady([r["events"] for r in requests]), bitwise=parity,
                      faults=fault_delta(process_before, process_after),
                      host_load=self.common.host_load([load_before, load_after]))
        write(path / "summary.json", result)
        return result

    def run(self):
        self.gpu_guard("before", idle=True)
        servers = []
        if self.mode == "matrix":
            configurations = ORDER
            winner_meta = None
        else:
            pointer_path = self.stage / "matrix-result.json"
            if not pointer_path.is_file():
                raise ValueError("confirmation requires a completed passing matrix; run the matrix job first")
            pointer = json.loads(pointer_path.read_text())
            matrix_path = Path(pointer["path"])
            assert matrix_path.resolve().is_relative_to((self.stage / "runs").resolve())
            assert digest(matrix_path) == pointer["sha256"]
            matrix = json.loads(matrix_path.read_text())
            if self.dry:
                matrix["dry_run"] = False
            winner, scores = choose_winner(matrix)
            if self.winner is not None:
                assert self.winner == winner, "explicit winner differs from matrix selection"
            configurations = ("adaptive", winner, winner, "adaptive")
            winner_meta = dict(config=winner, scores=scores, matrix=str(matrix_path),
                               matrix_sha256=pointer["sha256"])
        try:
            for epoch, config in enumerate(configurations):
                with self.serving(config, epoch):
                    # Every width configuration owns one server, always with
                    # room for both cohorts. Alternate the cohort order.
                    order = (8, 16) if epoch % 2 == 0 else (16, 8)
                    cohorts = [self.cohort(config, c) for c in order]
                    all_text = self.log.read_text()
                    verified = any(s["verify_items"] > 0 for s in records(all_text, "chopinfer.step.v1"))
                    ready = cache_ready(all_text, required=verified)
                    servers.append(dict(config=config, epoch=epoch, cohorts=cohorts, resident_cache=ready,
                                        resident_cache_status="active" if ready else "enabled-unused-no-verification"))
                self.gpu_guard(f"after-{epoch}")
            expected = 7 if self.mode == "matrix" else 4
            assert len(servers) == expected and sum(len(s["cohorts"]) for s in servers) == expected*2
            result = dict(schema=SCHEMA, status="passed", mode=self.mode, dry_run=self.dry,
                          native_revision=self.settings["native_revision"], server_count=len(servers),
                          case_count=expected*2, bitwise=True, servers=servers, winner=winner_meta,
                          unsupported=[{"case": f"uniform{k}-C16", "required_rows": 16*(k+1),
                                        "supported_rows": 32} for k in (2, 3)],
                          timing="provisional matrix" if self.mode == "matrix" else "ABBA confirmation",
                          elapsed_seconds=time.monotonic()-self.started)
        finally:
            stop(self.server)
            self.gpu_guard("final")
        # Publish success only after final power, Xid and cleanup checks.
        result["elapsed_seconds"] = time.monotonic() - self.started
        if self.mode == "confirm":
            result["confirmation"] = confirmation_rates(servers, winner_meta['config'])
        write(self.output / "summary.json", result)
        (self.output / "summary.md").write_text(markdown(result))
        if self.mode == "matrix":
            pointer = dict(path=str(self.output / "summary.json"), sha256=digest(self.output / "summary.json"))
            temp = self.stage / "matrix-result.pending.json"
            write(temp, pointer)
            temp.replace(self.stage / "matrix-result.json")
        return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("matrix", "confirm"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    def interrupted(number, _frame):
        raise InterruptedError(f"signal {number}")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    runner = Runner(args.stage, args.output, args.mode, args.dry_run)
    try:
        result = runner.run()
        print(json.dumps({k: result[k] for k in ("status", "mode", "server_count", "case_count", "elapsed_seconds")}))
    except BaseException as error:
        write(runner.output / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
