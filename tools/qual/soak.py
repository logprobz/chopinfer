"""Run a mixed C32 chat soak and watch device memory for growth.

With --prefix-mix, a third of the workers hold multi-turn conversations and a
third share long system prompts, so the prefix cache hits, captures and evicts
under load. A greedy canary checks restores (prefix-cache contract, gate 6). Its
system prompt is one the workers share, so it runs under its own cache salt:
only anchors the canary captured can serve it. Before the load, a primer
shares the canary's tokens up to its last block boundary and then diverges
(prefix_exact.primer: it runs a block on, or past the next chunk, so that
neither its prompt-end anchor, even rounded down to a prefill granule, nor
its final-chunk anchor lies at or before the divergence), so the cold canary
that follows ends a prefill chunk at that anchor and captures there. The
soak retains this chunked baseline; a C1 cold one-shot prefill is also
numerically valid under the bitwise prefix-cache contract. After the load
the canary runs twice more; each either hits the anchor
or, if it was evicted, is cut there again, and the last one must hit. An
answer that differs under load follows the separate near-tie rule: at the first
differing token, each run ranks the other's token second within
NEAR_TIE_MARGIN and the common prefix stayed within PREFIX_DLOGPROB; any
other difference counts as a restore error. The soak fails on a restore
error, on a canary that did not hit, on live pages once idle, or when the
cache served no hit.

--media-mix sends every eighth request of each worker with one frozen image,
audio or video part (a server packaged with encoders); the blue-square image
must be answered blue. --structured-mix sends another eighth with the strict
JSON schema of structured.py at default sampling; an output that stops must
parse and satisfy the schema, and a length stop is counted as truncated. The
receipt keeps the first KEEP_STRUCTURED truncated or invalid outputs
(`structured_kept`: request body, response text, finish reason, completion
tokens and the server's response id) and their total.
"""

from concurrent.futures import ThreadPoolExecutor
import re
import statistics
import subprocess
import threading
import time
from urllib.parse import urlsplit

import json
from pathlib import Path
import importlib.util
import uuid

from common import (REASONING_TOKENS, answer_kwargs, host_load_sample, input_path, thinking_can_be_disabled,
                    Client, arguments, not_applicable, receipt)
from prefix_exact import BLOCK_TOKENS, primer, tokenize
from spec_metrics import metrics_text
from structured import PERSON, PERSON_PROMPT, schema_errors

# A model declares these in its soak check's constants (checks.soak.constants
# .soak) when its server fills a bounded cache during warm-up, such as pinned
# prefill graphs. The growth after WARMUP_SECONDS, measured from the first
# sample at or after it, must stay within the memory limits; the growth over
# the whole soak, from the first sample, within the limits plus
# CACHE_ALLOWANCE_MIB (host and device, the cache's declared bound). The
# warm-up is capped at a quarter of the duration. With the defaults both
# conditions are the plain rule: growth from the first sample within the
# limits.
WARMUP_SECONDS = 0
CACHE_ALLOWANCE_MIB = {"host": 0, "device": 0}


def memory_policy(warmup=None, allowance=None):
    """The declared (warm-up seconds, allowance) pair, validated."""
    warmup = WARMUP_SECONDS if warmup is None else warmup
    allowance = CACHE_ALLOWANCE_MIB if allowance is None else allowance
    if type(warmup) is not int or warmup < 0:
        raise ValueError("soak warm-up must be a non-negative integer number of seconds")
    if not isinstance(allowance, dict) or set(allowance) != {"host", "device"} or any(
            type(value) is not int or value < 0 for value in allowance.values()):
        raise ValueError("soak cache allowance must give non-negative integer host and device MiB")
    return warmup, dict(allowance)


def memory_rule(samples, duration, limits, warmup=None, allowance=None):
    """Growth of device (`mib`) and host (`host_rss_mib`) memory over the
    whole soak and after the warm-up, judged against `limits` ({"device",
    "host"} MiB) and the declared allowance."""
    warmup, allowance = memory_policy(warmup, allowance)
    effective = min(warmup, duration / 4)
    steady = [row for row in samples if row["elapsed_seconds"] >= effective]
    if not steady:
        raise ValueError("no memory sample after the soak warm-up")
    result = {"warmup_seconds": warmup, "effective_warmup_seconds": effective,
              "steady_baseline_seconds": steady[0]["elapsed_seconds"], "cache_allowance_mib": allowance}
    passed = True
    for kind, key, prefix in (("device", "mib", "memory"), ("host", "host_rss_mib", "host_memory")):
        whole = max(row[key] for row in samples) - samples[0][key]
        after = max(row[key] for row in steady) - steady[0][key]
        result[prefix + "_growth_mib"] = whole
        result["steady_" + prefix + "_growth_mib"] = after
        passed = passed and after <= limits[kind] and whole <= limits[kind] + allowance[kind]
    result["passed"] = passed
    return result


def media_module():
    path = input_path("media_cases")
    spec = importlib.util.spec_from_file_location("qualification_media_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = ("Say hello.", "Summarize why a cache can improve repeated reads in one sentence.",
           "List three ways to check a numerical result. " * 60)


SYSTEMS = tuple(f"You are assistant {index}. " + " ".join(
    f"Rule {rule}: answer with care about topic {index * 31 + rule}." for rule in range(160))
    for index in range(4))
CANARY = [{"role": "system", "content": SYSTEMS[0]},
          {"role": "user", "content": "State rule 7 in your own words."}]


# The chunked-prefill noise floor reached 0.413 max |dlogprob| between two
# chunk sizes without caching.
NEAR_TIE_MARGIN = 0.5
PREFIX_DLOGPROB = 0.5


def classify_canary(cold, other):
    """Whether `other` emits the cold canary's tokens, flips a near tie, or not."""
    tokens = cold["tokens"], other["tokens"]
    at = next((index for index, (x, y) in enumerate(zip(*tokens)) if x != y),
              None if len(tokens[0]) == len(tokens[1]) else min(map(len, tokens)))
    if at is None:
        return {"verdict": "exact"}
    drift = max((abs(x - y) for x, y in zip(cold["logprobs"][:at], other["logprobs"][:at])),
                default=0.0)
    result = {"verdict": "restore_error", "first_difference": at, "prefix_max_abs_dlogprob": drift,
              "cold_margin": None, "other_margin": None}
    if at >= len(tokens[0]) or at >= len(tokens[1]):
        # One run stopped where the other went on; the stop has no top-2 entry.
        return result

    def margin(run, chosen, runner_up):
        top = {entry["token"]: entry["logprob"] for entry in run["top_logprobs"][at]}
        return top[chosen] - top[runner_up] if chosen in top and runner_up in top else None

    a, b = tokens[0][at], tokens[1][at]
    result["cold_margin"] = margin(cold, a, b)
    result["other_margin"] = margin(other, b, a)
    if (result["cold_margin"] is not None and result["other_margin"] is not None
            and max(result["cold_margin"], result["other_margin"]) <= NEAR_TIE_MARGIN
            and drift <= PREFIX_DLOGPROB):
        result["verdict"] = "near_tie"
    return result


def completion_canary(value):
    """A canary answer from a /v1/completions response with top-2 logprobs."""
    choice = value["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {"content": choice.get("text"), "usage": value["usage"],
            "tokens": logprobs.get("tokens") or [],
            "logprobs": logprobs.get("token_logprobs") or [],
            "top_logprobs": [[{"token": token, "logprob": logprob}
                              for token, logprob in (top or {}).items()]
                             for top in logprobs.get("top_logprobs") or []]}


def cached(run):
    return (run["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def check_canaries(canaries, anchor):
    """Compares every later canary with the baseline cut at `anchor`."""
    baseline, later = canaries[0], canaries[1:]
    verdicts = [classify_canary(baseline, other) for other in later]
    return {"canary_anchor": anchor, "canary_verdicts": verdicts,
            "canary_cached_tokens": [cached(run) for run in canaries],
            "canary_passed": (cached(baseline) == 0
                              and all(cached(run) in (0, anchor) for run in later)
                              and bool(later) and cached(later[-1]) == anchor
                              and all(verdict["verdict"] != "restore_error"
                                      for verdict in verdicts))}


def metric(text, name):
    for line in text.splitlines():
        if line.startswith(name + " "):
            return int(float(line.split()[1]))
    return None


def prefix_messages(worker_id, turn, conversation):
    """The next request of a --prefix-mix worker."""
    kind = worker_id % 3
    if kind == 0:
        if not conversation or len(conversation) > 12:
            conversation[:] = [{"role": "user", "content": f"Worker {worker_id} turn {turn}: "
                                + PROMPTS[turn % len(PROMPTS)]}]
        return conversation
    if kind == 1:
        return [{"role": "system", "content": SYSTEMS[(worker_id + turn) % len(SYSTEMS)]},
                {"role": "user", "content": f"Worker {worker_id} asks question {turn}."}]
    return [{"role": "user", "content": f"Worker {worker_id} turn {turn}: "
             + PROMPTS[(worker_id + turn) % len(PROMPTS)]}]


MEDIA_CASES = ("image-blue-square.png", "video-red-then-blue")


def media_requests(case_ids=MEDIA_CASES, fixture_dir=None):
    """(case id, messages) for the media mix, each fixture read and verified once."""
    multimodal_cases = media_module()
    fixture_dir = fixture_dir or input_path("media_fixtures")
    by_id = {case["id"]: case for case in multimodal_cases.cases()}
    return [(case_id, multimodal_cases.messages(by_id[case_id], fixture_dir)) for case_id in case_ids]


def check_media(case_id, choice):
    content = (choice["message"].get("content") or "").strip()
    if not content:
        raise ValueError(f"{case_id}: empty answer")
    if case_id == "image-blue-square.png" and "blue" not in content.lower():
        raise ValueError(f"{case_id}: answered {content[:40]!r}, expected blue")


STRUCTURED_FORMAT = {"type": "json_schema",
                     "json_schema": {"name": "person", "schema": PERSON, "strict": True}}


# The truncated or invalid structured outputs a receipt keeps in full.
KEEP_STRUCTURED = 16


def kept_structured(worker_id, turn, elapsed, latency, body, value, reason):
    """A structured request worth inspecting: what was sent and returned."""
    choice = value["choices"][0]
    answer = choice.get("message") or {}
    return {"worker": worker_id, "turn": turn, "elapsed_seconds": elapsed, "latency_seconds": latency,
            "reason": reason, "request": body, "response_id": value.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (value.get("usage") or {}).get("completion_tokens"),
            "content": answer.get("content"), "reasoning_content": answer.get("reasoning_content")}


def check_structured(choice):
    """True when the output was truncated; raises on a grammar violation."""
    if choice.get("finish_reason") == "length":
        return True
    value = json.loads(choice["message"].get("content") or "")
    errors = schema_errors(value, PERSON)
    if errors:
        raise ValueError("schema: " + "; ".join(errors[:3]))
    return False


def serving_pid(url):
    endpoint = urlsplit(url)
    if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("GPU memory sampling requires a local server URL or --server-pid")
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    output = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"],
                            check=True, capture_output=True, text=True).stdout
    pids = {int(pid) for pid in re.findall(r"\bpid=(\d+)\b", output)}
    if len(pids) != 1:
        raise RuntimeError(f"expected one serving PID on port {port}, found {sorted(pids)}")
    return pids.pop()


def process_tree(root_pid):
    output = subprocess.run(["ps", "-eo", "pid=,ppid="],
                            check=True, capture_output=True, text=True).stdout
    parents = {}
    for line in output.splitlines():
        pid, parent = line.split()
        parents[int(pid)] = int(parent)
    if root_pid not in parents:
        raise RuntimeError(f"serving PID {root_pid} exited during GPU memory sampling")
    family = {root_pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in family}
        if children <= family:
            return family
        family.update(children)


def device_memory(root_pid):
    family = process_tree(root_pid)
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True).stdout
    matched = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, mib = (field.strip() for field in line.split(",", 1))
        if int(pid) in family:
            matched.append(int(mib))
    if not matched:
        raise RuntimeError(f"no GPU allocation found for serving PID {root_pid} or its children")
    return sum(matched)


def host_memory(root_pid):
    """Resident memory of the serving process and its current children, in MiB."""
    total = 0
    for pid in process_tree(root_pid):
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            if pid == root_pid:
                raise RuntimeError("serving process exited during memory sampling") from None
            continue
        match = re.search(r"^VmRSS:\s+(\d+) kB$", status, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"no resident memory sample for serving process {pid}")
        total += int(match[1])
    return total / 1024


def memory_sample(root_pid, started):
    return {"at": time.time(), "elapsed_seconds": time.monotonic() - started,
            "mib": device_memory(root_pid), "host_rss_mib": host_memory(root_pid)}


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.media_mix or args.structured_mix)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--memory-growth-mib", type=int, default=512)
    parser.add_argument("--host-memory-growth-mib", type=int, default=512)
    parser.add_argument("--prefix-mix", action="store_true",
                        help="mix prefix-cache hits and misses and check them")
    parser.add_argument("--media-mix", action="store_true",
                        help="send an eighth of requests with image, audio or video parts")
    parser.add_argument("--media-cases", type=lambda value: value.split(","), default=list(MEDIA_CASES),
                        help=f"comma-separated media mix cases, default {','.join(MEDIA_CASES)}")
    parser.add_argument("--structured-mix", action="store_true",
                        help="send an eighth of requests with a strict JSON schema")
    parser.add_argument("--remote-only", action="store_true",
                        help="HTTP load only; do not inspect processes or GPU memory")
    parser.add_argument("--server-pid", type=int,
                        help="local serving process PID (auto-detected from --url by default)")
    parser.add_argument("--warmup-seconds", type=int,
                        help="seconds before the steady memory baseline (default: the declared WARMUP_SECONDS)")
    parser.add_argument("--cache-allowance-mib", type=lambda value: dict(zip(("host", "device"), map(int, value.split(",")))),
                        help="HOST,DEVICE MiB of declared bounded cache growth (default: CACHE_ALLOWANCE_MIB)")
    args = parser.parse_args()
    if args.duration <= 0 or args.concurrency <= 0:
        parser.error("duration and concurrency must be positive")
    if args.server_pid is not None and args.server_pid <= 0:
        parser.error("server-pid must be positive")
    try:
        warmup, allowance = memory_policy(args.warmup_seconds, args.cache_allowance_mib)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    # Media and structured answers come without reasoning, or after it when
    # the frontend refuses disabled thinking.
    mixed_reasoning = (args.media_mix or args.structured_mix) and not thinking_can_be_disabled()
    no_thinking = answer_kwargs() if args.media_mix or args.structured_mix else {}
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    root_pid = None if args.mechanics_only or args.remote_only else (args.server_pid or serving_pid(args.url))
    duration = min(args.duration, 4) if args.mechanics_only else args.duration
    concurrency = 1 if args.mechanics_only else args.concurrency
    lock = threading.Lock()
    latencies = []
    errors = []
    completed = 0
    memory = []
    load = []
    if unknown := set(args.media_cases) - set(MEDIA_CASES):
        parser.error(f"unknown media cases {sorted(unknown)}")
    # The mix sends only media of modalities the server serves; the others
    # make the soak not applicable as a media check, never passed.
    kinds = {case_id: case_id.split("-", 1)[0] for case_id in args.media_cases}
    inapplicable = sorted({reason for kind in kinds.values()
                           if args.media_mix and (reason := not_applicable(server, kind))})
    media = media_requests([case_id for case_id, kind in kinds.items()
                            if not not_applicable(server, kind)]) if args.media_mix else []
    mixed = {"media": 0, "structured": 0, "structured_truncated": 0}
    kept = {"limit": KEEP_STRUCTURED, "total": 0, "requests": []}

    def keep(*record):
        with lock:
            kept["total"] += 1
            if len(kept["requests"]) < KEEP_STRUCTURED:
                kept["requests"].append(kept_structured(*record))

    canary_salt = f"soak-canary/{uuid.uuid4().hex}"
    canary_ids = tokenize(client, server["model"], CANARY) if args.prefix_mix else []
    canary_anchor = (len(canary_ids) - 1) // BLOCK_TOKENS * BLOCK_TOKENS

    def canary(prompt=None, max_tokens=48):
        body = {"model": server["model"], "prompt": prompt or canary_ids, "temperature": 0,
                "max_tokens": max_tokens, "logprobs": 2, "cache_salt": canary_salt}
        return completion_canary(client.request("/v1/completions", body)[0])

    canaries = []
    if args.prefix_mix:
        canary(primer(canary_ids, canary_anchor), 1)
        canaries.append(canary())

    def worker(worker_id):
        nonlocal completed
        turn = 0
        conversation = []
        while time.monotonic() < deadline:
            prompt = "Hi." if args.mechanics_only else PROMPTS[(worker_id + turn) % len(PROMPTS)]
            messages = [{"role": "user", "content": prompt}]
            if args.prefix_mix:
                messages = prefix_messages(worker_id, turn, conversation)
            body = {"model": server["model"], "messages": messages,
                    "temperature": 0, "max_tokens": 1 if args.mechanics_only else (8, 32, 96)[turn % 3]}
            kind = None
            if media and (worker_id + turn) % 8 == 0:
                kind = "media"
                case_id, messages = media[(worker_id + turn) // 8 % len(media)]
                body = {"model": server["model"], "messages": messages, "temperature": 0,
                        "max_tokens": REASONING_TOKENS if mixed_reasoning else 48,
                        "chat_template_kwargs": no_thinking}
            elif args.structured_mix and (worker_id + turn) % 8 == 4:
                kind = "structured"
                body = {"model": server["model"], "max_tokens": REASONING_TOKENS if mixed_reasoning else 512,
                        "messages": [{"role": "user", "content": PERSON_PROMPT}],
                        "chat_template_kwargs": no_thinking,
                        "response_format": STRUCTURED_FORMAT}
            try:
                value, latency = client.request("/v1/chat/completions", body)
                if not value.get("choices"):
                    raise ValueError("missing choices")
                truncated = False
                if kind == "media":
                    check_media(case_id, value["choices"][0])
                elif kind == "structured":
                    record = worker_id, turn, time.monotonic() - started, latency, body, value
                    try:
                        truncated = check_structured(value["choices"][0])
                    except ValueError as error:
                        keep(*record, str(error) if str(error).startswith("schema") else f"parse: {error}")
                        raise
                    if truncated:
                        keep(*record, "truncated")
                if kind:
                    with lock:
                        mixed[kind] += 1
                        mixed["structured_truncated"] += truncated
                elif args.prefix_mix and worker_id % 3 == 0:
                    conversation.append({"role": "assistant",
                                         "content": value["choices"][0]["message"].get("content") or ""})
                    conversation.append({"role": "user", "content": f"Continue, turn {turn + 1}."})
                with lock:
                    completed += 1
                    latencies.append(latency)
            except Exception as error:
                with lock:
                    errors.append({"worker": worker_id, "error": str(error)})
            turn += 1

    started = time.monotonic()
    deadline = started + duration

    def sample():
        load.append(host_load_sample())
        if not args.mechanics_only and not args.remote_only:
            try:
                memory.append(memory_sample(root_pid, started))
            except Exception as error:
                errors.append({"memory": str(error)})

    sample()
    with ThreadPoolExecutor(max_workers=concurrency + 1) as pool:
        futures = [pool.submit(worker, index) for index in range(concurrency)]
        while time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            sample()
        measured_duration = time.monotonic() - started
        for future in futures:
            future.result()
    growth = host_growth = None
    rule = {}
    if memory:
        rule = memory_rule(memory, duration, {"device": args.memory_growth_mib, "host": args.host_memory_growth_mib},
                           warmup, allowance)
        growth, host_growth = rule["memory_growth_mib"], rule["host_memory_growth_mib"]
    prefix = None
    if args.prefix_mix:
        canaries.extend([canary(), canary()])
        time.sleep(1)
        text = metrics_text(client)
        prefix = {"canaries": canaries, "canary_salt": canary_salt,
                  **check_canaries(canaries, canary_anchor),
                  "canaries_match": len({canary["content"] for canary in canaries}) == 1,
                  "idle_blocks_used": metric(text, "chop_kv_blocks_used"),
                  "idle_active_sequences": metric(text, "chop_kv_active_sequences"),
                  **{name: metric(text, f"chop_prefix_cache_{name}")
                     for name in ("hits_total", "reused_tokens_total", "captures_total",
                                  "capture_failures_total", "restore_failures_total",
                                  "evictions_total", "anchors", "blocks",
                                  "capture_microseconds_total", "restore_microseconds_total")}}
        if not prefix["canary_passed"]:
            errors.append({"prefix": "canary restore error or no canary hit"})
        if not (prefix["idle_blocks_used"] == 0 and prefix["idle_active_sequences"] == 0
                and prefix["hits_total"]):
            errors.append({"prefix": "live pages after idle, or no hits"})
    passed = completed > 0 and not errors and (args.mechanics_only or args.remote_only or (len(memory) >= 2 and rule["passed"]))
    receipt("soak", server, {"passed": passed, "skipped": ["C32 and device memory sampling"] if args.mechanics_only else [],
                            "duration_seconds": duration, "measured_duration_seconds": measured_duration,
                            "concurrency": concurrency,
                            "completed": completed, "errors": errors,
                            "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
                            "latency_max_seconds": max(latencies) if latencies else None,
                            "memory_samples": memory, "memory_growth_mib": growth,
                            "host_memory_growth_mib": host_growth,
                            "host_memory_growth_limit_mib": args.host_memory_growth_mib,
                            "memory_scope": "not_sampled_remote" if args.remote_only else "serving_process_tree", "server_pid": root_pid,
                            "memory_growth_limit_mib": args.memory_growth_mib,
                            **({key: rule[key] for key in ("warmup_seconds", "effective_warmup_seconds",
                                                           "steady_baseline_seconds", "cache_allowance_mib",
                                                           "steady_memory_growth_mib", "steady_host_memory_growth_mib")}
                               if rule else {}),
                            "prefix_mix": prefix,
                            "mixed_requests": mixed if media or args.structured_mix else None,
                            "structured_kept": kept if args.structured_mix else None,
                            "media_cases": [case_id for case_id, _ in media] if media else None,
                            **({"not_applicable": inapplicable} if inapplicable else {})},
            args.mechanics_only, load_samples=load)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
