"""Check that preempted requests produce what unpreempted requests produce.

Preemption GPU accuracy gate. Run the same prompts against a server with
`--admission reserved` and one with `--admission optimistic
--debug-preempt-every N`, then compare:

- `record` sends greedy completions at one concurrency and stores every
  emitted token with its logprob and top-2 alternatives, and the preemption
  counter before and after.
- `compare` pairs a reserved and a preempting record. Tokens before position
  N come from steps before any preemption and must be identical. Each later
  first divergence must be a near-tie: the reserved run's top-2 logprobs at
  that position are within `--tie-nats`. The reserved server must not have
  preempted and the preempting one must have.
- `compare --exact` also requires every request's tokens, logprobs and top
  alternatives to be bit-identical. With prefix caching on (the default), a
  preempted request whose capture succeeded and whose anchor survived
  restores it and continues its decode, so a C1 pair whose resumes all did so
  (MiMo with an ample KV pool) must match exactly. Near-ties remain the rule
  where requests prefill their emitted tokens again (a capture the bundle
  refused, as Qwen MTP does at page boundaries, no free snapshot page, an
  evicted anchor, a failed restore, prefix caching off) or share batches.
- `compare --ties-before-preemption` judges a first divergence before
  position N by the same near-tie rule as a later one. A batched pair (C16)
  needs it: its requests share steps with companions that differ between the
  two servers from the first step, so a near tie can flip before any
  preemption. A C1 pair has no companions and keeps the plain rule or
  `--exact`; the flag refuses `--exact` and `--baseline`.
- `compare --baseline` pairs two reserved records at the same concurrency, to
  see what batch composition alone changes.
- `burst` sends many requests at once and checks that every one completes;
  with `--require-preemptions`, that the server preempted during the burst.
  `--min-tokens N` keeps each request from stopping before N tokens. The
  receipt lists each request's prompt and output tokens and the peak of the
  running-request gauge, the only observed overlap; requests that completed
  were not necessarily resident together, so it claims no page demand. The
  preemption counter (`--require-preemptions`) is the gate.

The prompts are 16 frozen GSM8K questions, 16 fixed instructions and four
long prompts of about 8K tokens built from the GSM8K questions.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import threading
import time

from common import Client, arguments, receipt
from spec_bench import CHAT_PROMPTS, gsm8k_questions
from spec_metrics import metrics_text

TOP_LOGPROBS = 2
PREEMPTIONS = re.compile(r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)
RUNNING = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)
LONG_QUESTIONS = 130


def prompts():
    _, questions = gsm8k_questions(200)
    natural = [f"Question: {question}\nAnswer step by step.\n" for question in questions[:16]]
    chat = [f"{instruction}\n\n" for instruction in CHAT_PROMPTS]
    long = []
    for offset in (0, 50, 100, 150):
        chosen = [questions[(offset + index) % len(questions)] for index in range(LONG_QUESTIONS)]
        body = "\n\n".join(f"Problem {index + 1}: {question}" for index, question in enumerate(chosen))
        long.append(f"{body}\n\nSolve problem {LONG_QUESTIONS} step by step.\n")
    return [(f"natural-{index}", prompt) for index, prompt in enumerate(natural)] + \
        [(f"chat-{index}", prompt) for index, prompt in enumerate(chat)] + \
        [(f"long-{index}", prompt) for index, prompt in enumerate(long)]


def counter(text, pattern):
    match = pattern.search(text)
    return float(match[1]) if match else None


def completion(client, model, prompt, output_tokens, min_tokens=0):
    body = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
            "temperature": 0, "top_p": 1, "logprobs": TOP_LOGPROBS}
    if min_tokens:
        body["min_tokens"] = min_tokens
    try:
        response, seconds = client.request("/v1/completions", body)
    except (RuntimeError, ValueError) as error:
        return {"error": str(error)}
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    return {"tokens": logprobs["tokens"], "logprobs": logprobs["token_logprobs"],
            "top": logprobs["top_logprobs"], "finish_reason": choice["finish_reason"],
            "usage": response["usage"], "seconds": seconds}


class RunningSampler:
    """Samples the running-request gauge while a cohort runs."""

    def __init__(self, client):
        self.client, self.peak, self.stop = client, 0.0, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, counter(metrics_text(self.client), RUNNING) or 0.0)
            except OSError:
                pass
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def cohort(client, model, jobs, concurrency, output_tokens, min_tokens=0):
    before = counter(metrics_text(client), PREEMPTIONS)
    started = time.monotonic()
    with RunningSampler(client) as running, ThreadPoolExecutor(concurrency) as pool:
        rows = list(pool.map(lambda job: {"name": job[0], **completion(
            client, model, job[1], output_tokens, min_tokens)}, jobs))
    seconds = time.monotonic() - started
    after = counter(metrics_text(client), PREEMPTIONS)
    preempted = None if before is None or after is None else after - before
    return rows, {"preemptions": preempted, "peak_running": running.peak, "seconds": seconds}


def record(args):
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    jobs = prompts()
    rows, observed = cohort(client, server["model"], jobs, args.concurrency, args.output_tokens)
    errors = [row["name"] for row in rows if "error" in row]
    receipt(f"preempt-{args.arm}-c{args.concurrency}", server, {
        "passed": not errors, "errors": errors, "arm": args.arm, "concurrency": args.concurrency,
        "output_tokens": args.output_tokens,
        "prompts_sha256": hashlib.sha256("\n".join(prompt for _, prompt in jobs).encode()).hexdigest(),
        "requests": rows, **observed})
    return 0 if not errors else 1


def bits(value):
    """A value with every float as its IEEE bit pattern, so -0.0 != 0.0."""
    if isinstance(value, float):
        return struct.pack("!d", value)
    if isinstance(value, list):
        return [bits(item) for item in value]
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    return value


def top_gap(top):
    values = sorted(top.values(), reverse=True) if top else []
    return values[0] - values[1] if len(values) >= 2 else None


def compare_records(reserved, preempted, every, tie_nats, baseline=False, exact=False,
                    ties_before_preemption=False):
    """The gate over two records of the same prompts; returns the report.

    A baseline pair is two reserved records, which measures how far batch
    composition alone moves the outputs; it checks no preemption counts. An
    exact pair must also match bit for bit. With `ties_before_preemption` a
    divergence before `every` is judged by the near-tie rule too."""
    if ties_before_preemption and (exact or baseline):
        raise ValueError("ties before the first preemption point apply to neither exact nor baseline pairs")
    failures = []
    if reserved["prompts_sha256"] != preempted["prompts_sha256"] or \
            reserved["output_tokens"] != preempted["output_tokens"]:
        raise ValueError("records use different prompts or output limits")
    names = [row.get("name") for row in reserved["requests"]]
    if len(set(names)) != len(names) or None in names:
        raise ValueError("the reserved record has missing or duplicate request names")
    if [row.get("name") for row in preempted["requests"]] != names:
        raise ValueError("the records do not hold the same requests in the same order")
    if not baseline and reserved.get("preemptions") not in (0, 0.0):
        failures.append(f"the reserved server preempted {reserved.get('preemptions')} times")
    if not baseline and not preempted.get("preemptions"):
        failures.append("the preempting server did not preempt")
    requests, identical, prefixes, divergences, differences = 0, 0, [], [], []
    for left, right in zip(reserved["requests"], preempted["requests"]):
        name = left["name"]
        if "error" in left or "error" in right:
            failures.append(f"{name}: request failed")
            continue
        requests += 1
        prefix = 0
        for ours, theirs in zip(left["tokens"], right["tokens"]):
            if ours != theirs:
                break
            prefix += 1
        prefixes.append(prefix)
        same = left["tokens"] == right["tokens"]
        identical += same
        if exact and any(bits(left[field]) != bits(right[field])
                         for field in ("tokens", "logprobs", "top", "finish_reason")):
            position = next((index for index, pair in enumerate(zip(left["logprobs"], right["logprobs"],
                                                                        left["top"], right["top"]))
                             if bits(pair[0]) != bits(pair[1]) or bits(pair[2]) != bits(pair[3])), prefix)
            failures.append(f"{name}: not bit-identical from position {min(position, prefix)}")
        differences.extend(abs(left["logprobs"][position] - right["logprobs"][position])
                           for position in range(every, prefix))
        if same:
            continue
        # The first position the runs disagree on: a different token, or a
        # stop in one run where the other continues.
        longer = left if prefix < len(left["top"]) else right
        gap = top_gap(longer["top"][prefix]) if prefix < len(longer["top"]) else None
        # Each divergence keeps what judging it needs: where, which tokens
        # (None for a stop) and the reserved run's top-2 gap there.
        divergences.append({"name": name, "position": prefix, "top2_gap": gap,
                            "before_first_preemption": prefix < every,
                            "tokens": [row["tokens"][prefix] if prefix < len(row["tokens"]) else None
                                       for row in (left, right)]})
        if prefix < every and not ties_before_preemption:
            failures.append(f"{name}: diverges at {prefix}, before the first preemption point {every}")
        elif gap is None or gap >= tie_nats:
            failures.append(f"{name}: diverges at {prefix} with a top-2 gap of {gap}, not a near-tie")
    return {
        "passed": not failures, "failures": failures, "every": every, "tie_nats": tie_nats, "exact": exact,
        "ties_before_preemption": ties_before_preemption,
        "requests": requests, "identical_requests": identical,
        "mean_common_prefix": sum(prefixes) / len(prefixes) if prefixes else None,
        "divergences": divergences,
        "mean_abs_dlogprob_after_first_preemption": sum(differences) / len(differences) if differences else None,
        "positions_after_first_preemption": len(differences),
        "preemptions": {"reserved": reserved.get("preemptions"), "preempted": preempted.get("preemptions")},
    }


def compare(args):
    reserved = json.loads(Path(args.reserved).read_text())
    preempted = json.loads(Path(args.preempted).read_text())
    report = compare_records(reserved, preempted, args.every, args.tie_nats, args.baseline, args.exact,
                             args.ties_before_preemption)
    report.update({"reserved": args.reserved, "preempted_record": args.preempted})
    json.dump(report, sys.stdout, indent=1)
    print()
    return 0 if report["passed"] else 1


def burst(args):
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    questions = [prompt for name, prompt in prompts() if name.startswith("natural")]
    jobs = [(f"burst-{index}", f"Request {index}.\n{questions[index % len(questions)]}")
            for index in range(args.requests)]
    rows, observed = cohort(client, server["model"], jobs, args.concurrency, args.output_tokens,
                            args.min_tokens)
    errors = [row["name"] for row in rows if "error" in row
              or row.get("finish_reason") not in ("stop", "length")
              or len(row.get("tokens", [])) < args.min_tokens]
    failures = list(errors)
    # Per request only: requests that all completed need not have been
    # resident together, so their extremes bound no page demand.
    request_tokens = [{"name": row["name"], "prompt_tokens": row["usage"]["prompt_tokens"],
                       "completion_tokens": len(row["tokens"])} for row in rows if "usage" in row]
    if args.require_preemptions and not observed["preemptions"]:
        failures.append("the server did not preempt during the burst")
    receipt(f"preempt-burst-{args.arm}", server, {
        "passed": not failures, "errors": failures, "arm": args.arm,
        "concurrency": args.concurrency, "requests": len(rows), "output_tokens": args.output_tokens,
        "min_tokens": args.min_tokens,
        "completion_tokens": sum(len(row.get("tokens", [])) for row in rows),
        "request_tokens": request_tokens,
        "finish_reasons": sorted({row.get("finish_reason") for row in rows if "error" not in row}),
        **observed})
    return 0 if not failures else 1


def main():
    parser = arguments(__doc__)
    parser.add_argument("command", choices=("record", "compare", "burst"))
    parser.add_argument("--arm", default="arm", help="record, burst: receipt label")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--requests", type=int, default=64, help="burst: request count")
    parser.add_argument("--require-preemptions", action="store_true")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="burst: tokens each request emits before it may stop")
    parser.add_argument("--reserved", help="compare: the reserved record")
    parser.add_argument("--preempted", help="compare: the preempting record")
    parser.add_argument("--every", type=int, default=16, help="compare: --debug-preempt-every")
    parser.add_argument("--tie-nats", type=float, default=0.5)
    parser.add_argument("--exact", action="store_true",
                        help="compare: also require bit-identical tokens, logprobs and top alternatives")
    parser.add_argument("--baseline", action="store_true",
                        help="compare: both records are reserved; skip the preemption counts")
    parser.add_argument("--ties-before-preemption", action="store_true",
                        help="compare: judge divergences before --every by the near-tie rule (batched pairs)")
    args = parser.parse_args()
    if args.command == "compare":
        if not args.reserved or not args.preempted:
            parser.error("compare needs --reserved and --preempted")
        if args.ties_before_preemption and (args.exact or args.baseline):
            parser.error("--ties-before-preemption refuses --exact and --baseline")
        raise SystemExit(compare(args))
    if args.concurrency < 1 or args.output_tokens < 1:
        parser.error("concurrency and output tokens must be positive")
    if not 0 <= args.min_tokens <= args.output_tokens:
        parser.error("min tokens must be from 0 to the output tokens")
    raise SystemExit(record(args) if args.command == "record" else burst(args))


if __name__ == "__main__":
    main()
