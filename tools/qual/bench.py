"""Measure C1, C16 and C32 decode plus a 32K-token prefill on any endpoint.

Every stream continues the same numbered list by default. `--prompts diverse`
gives each stream its own prompt from a mix of prose, code, math, chat and
several languages, so a batch routes to more of a MoE's experts than
repetitive text does.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import statistics
import time

from common import Client, arguments, receipt
from concurrency import add_arguments, validate
from spec_metrics import difference, metrics_text, snapshot


# Seeds of the diverse prompts: stream i continues seed i mod 16 as request i.
DIVERSE_SEEDS = (
    "Write a detailed history of the printing press and its effect on literacy in Europe.",
    "Implement a thread-safe LRU cache in Rust with generics, then explain each method.",
    "Prove that the square root of 2 is irrational, then generalize the argument to primes.",
    "Explain how a transformer's attention works to a first-year engineering student.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the Krebs cycle step by step, naming every enzyme and product.",
    "Write a SQL schema for a library system and five example queries with joins.",
    "Translate into French and explain the grammar: the weather will change tomorrow.",
    "用中文详细解释量子纠缠的原理，并举一个日常生活中的类比。",
    "日本の四季について、それぞれの季節の行事と食べ物を詳しく説明してください。",
    "Solve step by step: a train leaves at 3 pm at 80 km/h; another at 4 pm at 100 km/h.",
    "Write a bash script that backs up a directory with rotation and explain the flags.",
    "Compare the economic policies of three different countries over the last century.",
    "Erkläre ausführlich, wie ein Verbrennungsmotor funktioniert und welche Arten es gibt.",
    "Draft a formal email to a customer apologizing for a delayed shipment, with options.",
    "Write CUDA code for a tiled matrix multiply and explain the shared memory reuse.",
)


def stream_prompt(kind, index):
    """The prompt stream `index` continues: the numbered list, or a diverse seed."""
    if kind == "diverse":
        return f"{DIVERSE_SEEDS[index % len(DIVERSE_SEEDS)]} (request {index})\n\n"
    return f"Continue this numbered list from {index}: 1, 2, 3, 4,"


def settled_metrics(client, attempts=10, interval=0.2):
    """A scrape whose speculation series equal the next scrape's, so that no
    step is counted in some series and not yet in others."""
    def speculation(text):
        return [line for line in text.splitlines()
                if line.startswith("chop_spec_")]
    text = metrics_text(client)
    for _ in range(attempts):
        time.sleep(interval)
        again = metrics_text(client)
        if speculation(again) == speculation(text):
            return again
        text = again
    return text


def token_events(events):
    rows = []
    usage = None
    finish = None
    for at, event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            logprobs = choice.get("logprobs") or {}
            count = len(logprobs.get("tokens") or [])
            if count == 0 and choice.get("text"):
                count = 1
            if count:
                rows.append((at, count))
    return rows, usage, finish


def steady(streams, trim=32):
    boundaries = []
    for events in streams:
        count = 0
        left = right = None
        for at, n in events:
            count += n
            if left is None and count >= trim:
                left = at
        count = 0
        for at, n in reversed(events):
            count += n
            if right is None and count >= trim:
                right = at
        if left is None or right is None:
            raise ValueError("not enough tokens for steady decode")
        boundaries.append((left, right))
    start = max(left for left, _ in boundaries)
    end = min(right for _, right in boundaries)
    if end <= start:
        raise ValueError("no shared steady decode interval")
    counts = [sum(n for at, n in events if start < at < end) for events in streams]
    if any(n == 0 for n in counts):
        raise ValueError("no decoded tokens in shared interval")
    seconds = end - start
    return {"shared_seconds": seconds, "tokens": sum(counts),
            "aggregate_tokens_per_second": sum(counts) / seconds,
            "mean_ms_per_token": statistics.mean(seconds / n * 1000 for n in counts)}


def cohort(streams):
    """Tokens after each stream's first token, from the first stream's first
    token to the last stream's last token. Unlike the steady window, this
    counts the tail after the fastest stream finishes."""
    start = min(events[0][0] for events in streams)
    end = max(events[-1][0] for events in streams)
    if end <= start:
        raise ValueError("no cohort decode interval")
    tokens = sum(n for events in streams for _, n in events[1:])
    return {"seconds": end - start, "tokens": tokens, "tokens_per_second": tokens / (end - start)}


def main():
    parser = arguments(__doc__)
    add_arguments(parser, (1, 16, 32))
    parser.add_argument("--cohort-only", action="store_true",
                        help="Measure whole cohorts, including queued clients; do not require a shared steady window")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--prefill-tokens", type=int, default=32768)
    parser.add_argument("--prefill-repeats", type=int,
                        help="Send exactly this many prompt units instead of sizing the prompt "
                             "with the endpoint's own tokenizer, so two endpoints get the same text")
    parser.add_argument("--prefill-prefix", default="",
                        help="Text before the prefill prompt; a new one per run defeats prefix caching")
    parser.add_argument("--spec-metrics-arm", choices=("auto", "chopinfer"),
                        default="auto", help="Counter family to capture from /metrics")
    parser.add_argument("--sampling", choices=("greedy", "server-default"), default="greedy",
                        help="Decode greedily, or omit temperature so the server's defaults apply; "
                             "server-default sets min_tokens to the decode budget")
    parser.add_argument("--prompts", choices=("numbered", "diverse"), default="numbered",
                        help="Every stream continues one numbered list, or each takes its own "
                             "prompt from a mix of topics and languages")
    args = parser.parse_args()
    validate(parser, args)
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    try:
        metrics_before = settled_metrics(client)
        metrics_error = None
    except Exception as error:
        metrics_before = None
        metrics_error = str(error)
    output_tokens = 8 if args.mechanics_only else args.output_tokens
    results = []
    errors = []

    def one(index):
        prompt = stream_prompt(args.prompts, index)
        started = time.monotonic()
        body = {"model": server["model"], "prompt": prompt, "max_tokens": output_tokens,
                "min_tokens": output_tokens, "ignore_eos": True,
                "logprobs": 1, "stream": True, "stream_options": {"include_usage": True}}
        if args.sampling == "greedy":
            body["temperature"] = 0
        events, latency = client.request("/v1/completions", body, stream=True)
        tokens, usage, finish = token_events(events)
        row = {"index": index, "ttft_seconds": tokens[0][0] - started if tokens else None,
               "latency_seconds": latency, "usage": usage, "finish_reason": finish,
               "events": tokens}
        if not tokens or usage is None:
            row["error"] = "missing streaming tokens or usage"
        elif not args.mechanics_only and (usage["completion_tokens"] != output_tokens or finish != "length"):
            row["error"] = "generation stopped before the fixed decode budget"
        return row

    def observed(index):
        try:
            return one(index)
        except Exception as error:
            return {"index": index, "error": str(error)}

    for concurrency in args.concurrencies:
        streams = []
        try:
            actual = 1 if args.mechanics_only else concurrency
            with ThreadPoolExecutor(max_workers=actual) as pool:
                streams = list(pool.map(observed, range(actual)))
            if any("error" in row for row in streams):
                raise ValueError("decode requests failed; see per-request errors")
            events = [row["events"] for row in streams]
            result = {"concurrency": concurrency, "requests": streams,
                      "steady": None if args.mechanics_only or args.cohort_only else steady(events),
                      "cohort": None if args.mechanics_only else cohort(events)}
            results.append(result)
        except Exception as error:
            errors.append({"concurrency": concurrency, "error": str(error), "requests": streams})
    try:
        unit = "A queue records each request and then validates the result. "
        if args.mechanics_only:
            prompt = unit
        elif args.prefill_repeats:
            prompt = args.prefill_prefix + unit * args.prefill_repeats
        else:
            probe, _ = client.request("/v1/completions", {"model": server["model"], "prompt": unit * 64,
                                                        "max_tokens": 1, "temperature": 0})
            tokens_per_unit = probe["usage"]["prompt_tokens"] / 64
            repeats = max(1, round(args.prefill_tokens / tokens_per_unit))
            prompt = args.prefill_prefix + unit * repeats
        started = time.monotonic()
        # A single generated special token can have no visible text. Request
        # logprobs, as decode does, so its SSE arrival still identifies TTFT.
        events, latency = client.request("/v1/completions", {
            "model": server["model"], "prompt": prompt, "max_tokens": 1,
            "temperature": 0, "logprobs": 1, "stream": True,
            "stream_options": {"include_usage": True}}, stream=True)
        tokens, usage, _ = token_events(events)
        if not tokens or usage is None:
            raise ValueError("prefill returned no token or usage")
        ttft = tokens[0][0] - started
        if not args.mechanics_only and abs(usage["prompt_tokens"] - args.prefill_tokens) > 128:
            raise ValueError("prefill prompt differs from 32K target by more than 128 tokens")
        prefill = {"prompt_tokens": usage["prompt_tokens"], "ttft_seconds": ttft,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "throughput_tokens_per_second": usage["prompt_tokens"] / ttft,
                   "latency_seconds": latency}
    except Exception as error:
        prefill = None
        errors.append({"prefill": str(error)})
    try:
        metrics_after = settled_metrics(client)
        if metrics_before is None:
            raise ValueError(f"before scrape failed: {metrics_error}")
        metrics_arm = args.spec_metrics_arm
        if metrics_arm == "auto":
            metrics_arm = "chopinfer"
        before = snapshot(metrics_before, metrics_arm, server["model"])
        after = snapshot(metrics_after, metrics_arm, server["model"])
        speculation = {"arm": metrics_arm, "before": before, "after": after,
                       "delta": difference(before, after)}
    except Exception as error:
        # G9 also benchmarks endpoints without speculation metrics. Preserve its
        # pass criteria while making a missing capture explicit in the receipt.
        speculation = {"available": False, "error": str(error)}
    passed = not errors and len(results) == len(args.concurrencies) and prefill is not None
    receipt("bench", server, {"passed": passed, "skipped": ["requested concurrency, steady decode and prefill size"] if args.mechanics_only else [],
                             "errors": errors, "decode": results,
                             "max_sequences": args.max_sequences, "concurrencies": args.concurrencies, "prefill": prefill,
                             "output_tokens": output_tokens, "sampling": args.sampling,
                             "prompts": args.prompts, "cohort_only": args.cohort_only,
                             "ignore_eos": True, "min_tokens": output_tokens,
                             "speculation": speculation}, args.mechanics_only)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
