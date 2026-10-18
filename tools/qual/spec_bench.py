"""Measure frozen GSM8K and chat prompts at selected concurrency with speculation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import re
import statistics
import time

from common import input_path, thinking_control, Client, arguments, host_load_sample, receipt
from concurrency import add_arguments, validate
from spec_metrics import difference, metrics_text, snapshot, within_output
from step_overlap_ab import step_lines, summarize_steps


FIXTURE = None
CONCURRENCIES = (1, 4, 16)
CHAT_PROMPTS = (
    "Explain how a queue handles producers and consumers during a burst of work.",
    "Describe a careful method for finding a bug in a distributed service.",
    "Explain why a fixed random seed helps compare two inference servers.",
    "Describe the stages of a compiler from source text to machine code.",
    "Explain the difference between latency and throughput in serving systems.",
    "Describe how to check whether a numerical result is reproducible.",
    "Explain how a cache can change a benchmark without changing the model.",
    "Describe a reliable way to compare two implementations of an API.",
    "Explain what a percentile means when measuring request latency.",
    "Describe how a tokenizer maps text into model input identifiers.",
    "Explain why a streaming response needs a final usage record.",
    "Describe how concurrency changes the memory needs of a server.",
    "Explain how to validate the shape of a binary data structure.",
    "Describe a process for checking a model against frozen examples.",
    "Explain why counter resets matter when measuring a running service.",
    "Describe how to make a performance measurement repeatable.",
)


def gsm8k_questions(count=16):
    manifest = json.loads(((FIXTURE or input_path("gsm8k")) / "manifest.json").read_text())
    raw = ((FIXTURE or input_path("gsm8k")) / "examples.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("frozen GSM8K examples changed")
    examples = json.loads(raw)
    if len(examples) != manifest["expected_count"] or not examples:
        raise ValueError("frozen GSM8K example count changed")
    return manifest, [examples[index % len(examples)]["question"] for index in range(count)]


def workload(model, run_id, category, concurrency, index, question, instruction, output_tokens,
             *, ignore_eos=False):
    # A unique first message prevents shared prefixes inside one cohort. The
    # same run ID and body are sent to every arm for a fair comparison.
    body = {"model": model,
            "messages": [
                {"role": "system", "content": f"Spec qualification {run_id} {category} C{concurrency} item {index}. "
                 + instruction},
                {"role": "user", "content": question}],
            "temperature": 0, "top_p": 1, "max_tokens": output_tokens,
            **thinking_control(True),
            "stream": True, "stream_options": {"include_usage": True}}
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def chat_events(events):
    first = None
    usage = None
    finish = None
    for at, event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if first is None and any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
                first = at
    if first is None or usage is None:
        raise ValueError("chat stream omitted content or final usage")
    return first, usage, finish


def trace_window(path, offset, concurrency):
    with path.open("rb") as log:
        log.seek(offset)
        steps = step_lines(log.read().decode())
        end_offset = log.tell()
    if not steps:
        raise ValueError("requested step trace contains no engine steps")
    return {"path": str(path), "start_byte": offset, "end_byte": end_offset,
            "decode": summarize_steps([s for s in steps if not s.get("verify_items")], concurrency),
            "verify": summarize_steps([s for s in steps if s.get("verify_items")], concurrency)}


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--arm", choices=("chopinfer-ar", "chopinfer-spec"), required=True)
    parser.add_argument("--label", default="exclusive", help="Comparison group, e.g. exclusive or shared-device")
    parser.add_argument("--run-id", required=True, help="Use the same unique ID for all arms; restart each server")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Request exactly the token budget even if the model emits EOS")
    parser.add_argument("--server-log", type=Path, help="Optional CHOPINFER_STEP_TRACE log; request bodies stay unchanged")
    add_arguments(parser, CONCURRENCIES)
    parser.add_argument("--speculation", choices=("fixed", "adaptive"), default="fixed",
                        help="Policy configured on the speculative server; adaptive may choose AR")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.run_id):
        parser.error("run-id must contain only letters, digits, underscores or dashes")
    if args.output_tokens < 32 or args.output_tokens > 512:
        parser.error("output-tokens must be between 32 and 512")
    if args.mechanics_only:
        parser.error("speculation benchmark requires the full fixed-length workload")
    validate(parser, args)
    manifest, questions = gsm8k_questions(max(args.concurrencies))
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    metric_arm = "chopinfer"
    instructions = {
        "gsm8k": manifest["instruction"] + " Explain your reasoning in at least ten numbered steps before the final answer.",
        "chat": "Answer in at least ten numbered steps with concrete examples.",
    }
    prompts = {"gsm8k": questions, "chat": CHAT_PROMPTS}
    cohorts = []
    errors = []
    fingerprints = []
    # The timed work: each completed cohort's window, sampled at its start
    # and end and judged alone (common.host_load_windows), so the setup and
    # the scrapes between cohorts are never judged.
    windows = []
    for category, texts in prompts.items():
        for concurrency in args.concurrencies:
            bodies = [workload(server["model"], args.run_id, category, concurrency, index,
                               texts[index % len(texts)], instructions[category], args.output_tokens,
                               ignore_eos=args.ignore_eos)
                      for index in range(concurrency)]
            fingerprints.extend(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                                for body in bodies)
            try:
                before = snapshot(metrics_text(client), metric_arm, server["model"])
                offset = args.server_log.stat().st_size if args.server_log else None

                def one(index):
                    started = time.monotonic()
                    events, latency = client.request("/v1/chat/completions", bodies[index], stream=True)
                    first, usage, finish = chat_events(events)
                    if usage["completion_tokens"] != args.output_tokens or finish != "length":
                        raise ValueError(f"item {index} generated {usage['completion_tokens']} of "
                                         f"{args.output_tokens} tokens, finish_reason={finish}")
                    return {"index": index, "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "finish_reason": finish, "latency_seconds": latency,
                            "ttft_seconds": first - started}

                window = [host_load_sample()]
                window_start = time.monotonic()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    requests = list(pool.map(one, range(concurrency)))
                window_end = time.monotonic()
                windows.append([*window, host_load_sample()])
                after = snapshot(metrics_text(client), metric_arm, server["model"])
                counters = difference(before, after)
                adaptive = args.arm == "chopinfer-spec" and args.speculation == "adaptive"
                if args.arm != "chopinfer-ar" and (not counters["available"] or
                                                    (not adaptive and counters["verify_items"] == 0)):
                    raise ValueError("speculation arm produced no verify-step counters")
                counters.update(within_output(counters, len(requests),
                                              sum(row["completion_tokens"] for row in requests)))
                seconds = max(row["latency_seconds"] for row in requests)
                cohorts.append({"category": category, "concurrency": concurrency,
                                "window_monotonic_seconds": [window_start, window_end],
                                "requests": requests, "metrics_before": before,
                                "metrics_after": after, "speculation": counters,
                                "aggregate_tokens_per_second": concurrency * args.output_tokens / seconds,
                                "mean_ms_per_token": statistics.mean(
                                    row["latency_seconds"] / args.output_tokens * 1000
                                    for row in requests)})
                if args.server_log:
                    cohorts[-1]["step_trace"] = trace_window(args.server_log, offset, concurrency)
            except Exception as error:
                errors.append({"category": category, "concurrency": concurrency, "error": str(error)})
    workload_hash = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    if args.arm == "chopinfer-spec" and not any(
            row["speculation"].get("verify_items", 0) for row in cohorts):
        errors.append({"error": "speculation arm produced no verify steps in the entire run"})
    passed = not errors and len(cohorts) == 2 * len(args.concurrencies)
    receipt(f"spec-bench-{args.arm}", server, {
        "passed": passed, "arm": args.arm, "label": args.label, "run_id": args.run_id,
        "workload_sha256": workload_hash, "gsm8k_examples_sha256": manifest["examples_sha256"],
        "output_tokens": args.output_tokens, "sampling": {"temperature": 0, "top_p": 1, "ignore_eos": args.ignore_eos},
        "concurrencies": args.concurrencies,
        "max_sequences": args.max_sequences,
        "speculation_mode": args.speculation if args.arm == "chopinfer-spec" else None,
        "errors": errors, "cohorts": cohorts}, False, load_windows=windows)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
