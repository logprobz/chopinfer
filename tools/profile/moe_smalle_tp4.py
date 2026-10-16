#!/usr/bin/env python3
"""Run four prepared TP4 server arms sequentially, with a 25-minute deadline.

The JSON config has `arms`, in order: MiMo baseline, MiMo candidate, Qwen
baseline, Qwen candidate. Each arm supplies family, revision, model, url,
command (a foreground executable and argv), and optional env. Only the
server PIDs started here are stopped. Prepare binaries and packages first;
this runner does not build, deploy packages, or acquire GPU resources.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/qual"))
from common import Client
from bench import stream_prompt
from spec_bench import CHAT_PROMPTS, gsm8k_questions, workload


def worker(arm, out, run_id):
    out.mkdir(parents=True, exist_ok=True)
    cs = [1, 16, 32] if arm["family"] == "mimo" else [16]
    env = os.environ | {"CHOPINFER_QUAL_RECEIPTS": str(out / "receipts")}
    shared = ["--url", arm["url"], "--model", arm["model"], "--timeout", "180",
              "--concurrencies", *map(str, cs)]
    commands = [
        [sys.executable, "-B", str(ROOT / "tools/qual/bench.py"), *shared,
         "--output-tokens", "256", "--prefill-tokens", "256", "--sampling", "greedy", "--prompts", "numbered"],
    ]
    if arm["family"] == "mimo":
        commands.append([sys.executable, "-B", str(ROOT / "tools/qual/spec_bench.py"), *shared,
                         "--arm", "chopinfer-ar", "--label", "exclusive", "--run-id", run_id,
                         "--output-tokens", "64"])
    for index, command in enumerate(commands):
        with (out / f"bench-{index}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
    receipts = list((out / "receipts").glob("*.json"))
    if len(receipts) != len(commands):
        raise RuntimeError("missing benchmark receipts")
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt.get("passed") is not True or receipt.get("skipped") or receipt.get("errors"):
            raise RuntimeError(f"benchmark failed or skipped: {path}")
    client = Client(arm["url"], timeout=180, retries=0)
    equal = {}
    for c in cs:
        for category in (["numbered", "chat"] if arm["family"] == "mimo" else ["numbered"]):
            def one(index):
                if category == "chat":
                    body = workload(arm["model"], run_id, category, c, index,
                                    CHAT_PROMPTS[index % len(CHAT_PROMPTS)],
                                    "Answer in at least ten numbered steps with concrete examples.", 64)
                    body.pop("stream_options")
                    body["stream"] = False
                    endpoint = "/v1/chat/completions"
                else:
                    body = {"model": arm["model"], "prompt": stream_prompt("numbered", index),
                            "max_tokens": 64, "temperature": 0, "top_p": 1}
                    endpoint = "/v1/completions"
                body["return_token_ids"] = True
                response, _ = client.request(endpoint, body)
                ids = response["choices"][0].get("token_ids")
                if not ids or len(ids) != response["usage"]["completion_tokens"] or len(ids) != 64:
                    raise RuntimeError("greedy comparison needs exactly 64 emitted token IDs")
                return ids
            with ThreadPoolExecutor(max_workers=c) as pool:
                equal[f"{category}-c{c}"] = list(pool.map(one, range(c)))
    (out / "greedy.json").write_text(json.dumps(equal) + "\n")


def config(path):
    arms = json.loads(path.read_text())["arms"]
    expected = [(family, revision) for family in ("mimo", "qwen") for revision in ("baseline", "candidate")]
    if [(arm["family"], arm["revision"]) for arm in arms] != expected:
        raise ValueError("need MiMo baseline/candidate, then Qwen baseline/candidate")
    for arm in arms:
        if not isinstance(arm["command"], list) or not arm["command"] or not all(isinstance(x, str) for x in arm["command"]):
            raise ValueError("command must be an executable and argv, not shell text")
        Path(arm["command"][0]).resolve(strict=True)
        if not arm["url"].startswith("http://127.0.0.1:") or not arm["model"]:
            raise ValueError("each arm needs a local server URL and model")
    gsm8k_questions(32)
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--end", help="Assigned absolute end time, ISO 8601 with timezone")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--run-id", help=argparse.SUPPRESS)
    args = p.parse_args()
    arms = config(args.config)
    if args.worker is not None:
        worker(arms[args.worker], args.out, args.run_id)
        return
    if args.preflight:
        print("TP4 config and frozen workloads passed preflight. No servers or GPU work started.")
        return
    if not args.end:
        p.error("--end is required; run only on four idle GPUs")
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if end.tzinfo is None:
        p.error("end time must include a timezone")
    deadline = time.monotonic() + min(1500, end.timestamp() - time.time())
    base = Path(os.environ["CHOPINFER_DATA"]) / "kopt-moe-smalle"
    run_id = datetime.now(timezone.utc).strftime("smalle-%Y%m%dT%H%M%S%fZ")
    out = base / "tp4" / run_id
    out.mkdir(parents=True)
    results = []
    for index, arm in enumerate(arms):
        if deadline - time.monotonic() < 15:
            raise TimeoutError("assigned TP4 window has ended")
        # Refuse an occupied endpoint rather than testing an unrelated server.
        try:
            urllib.request.urlopen(arm["url"] + "/health", timeout=1).close()
        except urllib.error.URLError:
            pass
        else:
            raise RuntimeError(f"endpoint already running: {arm['url']}")
        arm_dir = out / f"{arm['family']}-{arm['revision']}"
        arm_dir.mkdir()
        with (arm_dir / "server.log").open("x") as log:
            server = subprocess.Popen(arm["command"], cwd=ROOT, env=os.environ | arm.get("env", {}),
                                      stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited: {arm_dir / 'server.log'}")
                    if deadline - time.monotonic() < 15:
                        raise TimeoutError("startup exceeded the assigned window")
                    try:
                        client = Client(arm["url"], timeout=1, retries=0)
                        client.request("/health")
                        client.identity(arm["model"])
                        break
                    except (RuntimeError, ValueError):
                        time.sleep(0.25)
                seconds = deadline - time.monotonic() - 12
                command = ["timeout", "--kill-after=2s", f"{seconds:.1f}s", sys.executable, "-B", str(Path(__file__).resolve()),
                           "--config", str(args.config.resolve()), "--worker", str(index),
                           "--out", str(arm_dir), "--run-id", run_id]
                with (arm_dir / "worker.log").open("x") as worker_log:
                    status = subprocess.run(command, cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT).returncode
                results.append({"family": arm["family"], "revision": arm["revision"], "exit": status})
                (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                if status:
                    raise RuntimeError(f"TP4 arm failed: {arm_dir}")
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                else:
                    server.wait()
        if arm["revision"] == "candidate":
            baseline = json.loads((out / f"{arm['family']}-baseline/greedy.json").read_text())
            candidate = json.loads((arm_dir / "greedy.json").read_text())
            if baseline != candidate:
                raise RuntimeError(f"greedy token IDs differ: {arm['family']}")
    print(f"TP4 benchmarks and greedy token-ID equality passed: {out}")


if __name__ == "__main__":
    main()
