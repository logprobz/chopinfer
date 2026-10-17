"""Check that prefix-cache hits keep greedy outputs and cut follow-up TTFT.

`record` runs three scenarios one request at a time against one server and
stores each request's streamed tokens, logprobs, time to first token and
`usage.prompt_tokens_details.cached_tokens`:

- multi_turn: a user turn of about N tokens, then a follow-up turn that repeats
  it with the first answer, for each prefix length N;
- system: three questions under one system prompt of about N tokens, since an
  anchor at the shared blocks exists from the second use on;
- partial: three prompts sharing about 100 tokens, which ends inside a block.

Record the same `--run-id` against a server with `--prefix-cache off` (the
baseline), one with it on (the candidate), and optionally a second baseline
with a different `--prefill-chunk-tokens` (the chunked-prefill noise floor).
Restart the server between records so no cache survives. `compare` requires
the hits it expects to have happened, and at least as many requests matching
the baseline's tokens as the floor has, with logprob differences within the
floor's. Without a floor every request must match. The follow-up TTFT it
reports carries `host_load`: timing evidence only when both the baseline and
the candidate record ran on an idle host (common.compared_timing); the label
changes no verdict.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from common import answer_kwargs, Client, arguments, compared_timing, host_load_sample, receipt

WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor",
         "indigo", "juniper", "kelp", "lagoon", "meadow", "nimbus", "onyx", "prairie")
# Extra slack over the scale, so a follow-up still fits after its answer.
TOKENS_PER_RECORD = 16
MEAN_DLOGPROB_SLACK = 0.01
MAX_DLOGPROB_SLACK = 0.05


def text(tokens, salt):
    """About `tokens` MiMo tokens of numbered records that never repeat."""
    lines = []
    for index in range(max(1, tokens // TOKENS_PER_RECORD)):
        word = WORDS[(index * 7 + salt) % len(WORDS)]
        lines.append(f"Record {salt}-{index}: {word} {index * 31 % 997} {word[::-1]}.")
    return "\n".join(lines)


def scenarios(lengths, run_id, answer_tokens):
    """Yield (scenario, length, name, messages or a callable of prior answers)."""
    salt = int(hashlib.sha256(run_id.encode()).hexdigest()[:6], 16)
    for length in lengths:
        first = [{"role": "user", "content": f"Run {run_id}.\n{text(length, salt + length)}\n"
                  "Which word appears in the last record?"}]
        yield "multi_turn", length, "turn1", first
        yield "multi_turn", length, "turn2", lambda answers, first=first: first + [
            {"role": "assistant", "content": answers["turn1"]},
            {"role": "user", "content": "Which word appears in the first record?"}]
        system = f"Run {run_id}.\n{text(length, salt + 2 * length)}"
        for index, question in enumerate(("Name the first word.", "Name the last word.",
                                           "How many records are there?")):
            yield "system", length, f"question{index}", [
                {"role": "system", "content": system}, {"role": "user", "content": question}]
    shared = f"Run {run_id}. " + text(96, salt + 7)
    for index, tail in enumerate(("Count the records.", "Repeat the first record.",
                                  "Name any word you see.")):
        yield "partial", 100, f"tail{index}", [{"role": "user", "content": f"{shared} {tail}"}]


def stream_chat(client, model, messages, answer_tokens):
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": answer_tokens, "logprobs": True, "top_logprobs": 1,
            "stream": True, "stream_options": {"include_usage": True},
            # Reasoning would spend the answer budget before any content; a
            # frontend that refuses disabled thinking compares its reasoning.
            "chat_template_kwargs": answer_kwargs()}
    started = time.monotonic()
    events, _ = client.request("/v1/chat/completions", body, stream=True)
    first, tokens, logprobs, content, usage = None, [], [], [], None
    for at, event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if first is None and (delta.get("content") or delta.get("reasoning")):
                first = at
            for entry in (choice.get("logprobs") or {}).get("content") or []:
                tokens.append(entry["token"])
                logprobs.append(entry["logprob"])
            content.append(choice.get("delta", {}).get("content") or "")
    if usage is None:
        raise ValueError("the stream carried no usage")
    return {"tokens": tokens, "logprobs": logprobs, "content": "".join(content),
            "ttft_seconds": None if first is None else first - started,
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def record(args):
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    lengths = [64] if args.mechanics_only else [int(value) for value in args.lengths.split(",")]
    rows, answers = [], {}
    load = [host_load_sample()]
    for scenario, length, name, messages in scenarios(lengths, args.run_id, args.answer_tokens):
        if callable(messages):
            messages = messages(answers)
        row = stream_chat(client, server["model"], messages, args.answer_tokens)
        answers[name] = row["content"]
        rows.append({"scenario": scenario, "length": length, "name": name,
                     "messages_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                     **row})
    load.append(host_load_sample())
    receipt(f"prefix-cache-{args.arm}", server, {
        "passed": all(row["tokens"] for row in rows), "arm": args.arm, "run_id": args.run_id,
        "lengths": lengths, "answer_tokens": args.answer_tokens, "requests": rows},
        args.mechanics_only, load_samples=load)


def pair_stats(left, right):
    """Exactness and logprob differences over each request's common prefix."""
    exact, differences, prefixes = 0, [], []
    for a, b in zip(left, right):
        prefix = 0
        for x, y in zip(a["tokens"], b["tokens"]):
            if x != y:
                break
            prefix += 1
        prefixes.append(prefix)
        exact += a["tokens"] == b["tokens"]
        differences.extend(abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(prefix))
    return {"requests": len(prefixes), "exact_requests": exact,
            "scored_positions": len(differences),
            "scored_requests": sum(prefix > 0 for prefix in prefixes),
            "mean_common_prefix": statistics.mean(prefixes) if prefixes else None,
            "mean_abs_dlogprob": statistics.mean(differences) if differences else None,
            "max_abs_dlogprob": max(differences, default=None)}


def load(path):
    return json.loads(Path(path).read_text())


def request_keys(value):
    return [(row["scenario"], row["length"], row["name"], row["messages_sha256"])
            for row in value["requests"]]


def validate_record(value):
    if value.get("passed") is not True:
        raise ValueError("record did not pass")
    if not isinstance(value["run_id"], str) or not value["run_id"]:
        raise ValueError("run ID is missing")
    if not value["server"]["model"]:
        raise ValueError("model is missing")
    lengths = value["lengths"]
    answer_tokens = value["answer_tokens"]
    if (not isinstance(lengths, list) or not lengths
            or any(type(length) is not int or length <= 0 for length in lengths)
            or len(set(lengths)) != len(lengths)
            or type(answer_tokens) is not int or answer_tokens <= 0):
        raise ValueError("declared lengths are invalid")
    expected = [(scenario, length, name) for scenario, length, name, _ in
                scenarios(lengths, value["run_id"], answer_tokens)]
    keys = request_keys(value)
    if [key[:3] for key in keys] != expected or any(not key[3] for key in keys):
        raise ValueError("request coverage or identity is incomplete")
    for row in value["requests"]:
        tokens, scores = row["tokens"], row["logprobs"]
        if (not isinstance(tokens, list) or not 0 < len(tokens) <= answer_tokens
                or not isinstance(scores, list) or len(tokens) != len(scores)
                or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
            raise ValueError("token scores are incomplete or nonfinite")


def comparable(baseline, other):
    validate_record(other)
    for key in ("run_id", "lengths", "answer_tokens"):
        if baseline[key] != other[key]:
            raise ValueError(f"records have different {key}")
    if baseline["server"]["model"] != other["server"]["model"]:
        raise ValueError("records have different models")
    if request_keys(baseline) != request_keys(other):
        raise ValueError("records ran different requests")


def compare(args):
    try:
        compare_valid(args)
    except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        print(json.dumps({"passed": False, "errors": [str(error)]}))
        raise SystemExit(1) from error


def compare_valid(args):
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    validate_record(baseline)
    comparable(baseline, candidate)
    stats = pair_stats(candidate["requests"], baseline["requests"])
    floor = None
    if args.floor:
        floor_record = load(args.floor)
        comparable(baseline, floor_record)
        floor = pair_stats(floor_record["requests"], baseline["requests"])
    for label, comparison in (("candidate", stats), ("floor", floor)):
        if comparison is not None and comparison["scored_requests"] != comparison["requests"]:
            raise ValueError(f"{label}: each request needs a scored common position")
    # The follow-up turn, the third system question and the third partial
    # prompt are the requests an anchor must serve.
    expected = [row for row in candidate["requests"]
                if row["name"] in ("turn2", "question2", "tail2")]
    missed = [(row["scenario"], row["length"], row["name"]) for row in expected
              if not row["cached_tokens"]]
    ttft = []
    for cold, warm in zip(baseline["requests"], candidate["requests"]):
        if warm["name"] == "turn2" and cold["ttft_seconds"] and warm["ttft_seconds"]:
            ttft.append({"length": warm["length"], "prompt_tokens": warm["prompt_tokens"],
                         "cached_tokens": warm["cached_tokens"],
                         "baseline_seconds": cold["ttft_seconds"],
                         "candidate_seconds": warm["ttft_seconds"],
                         "ratio": warm["ttft_seconds"] / cold["ttft_seconds"],
                         "new_token_fraction": 1 - (warm["cached_tokens"] or 0) / warm["prompt_tokens"]})
    within_floor = floor is None or (
        stats["scored_positions"] >= floor["scored_positions"]
        and stats["mean_abs_dlogprob"] <= floor["mean_abs_dlogprob"] + MEAN_DLOGPROB_SLACK
        and stats["max_abs_dlogprob"] <= floor["max_abs_dlogprob"] + MAX_DLOGPROB_SLACK)
    # Greedy outputs can flip at near-tied tokens between two chunk sizes
    # without any cache, so a floor sets how many exact requests to expect.
    exact_needed = stats["requests"] if floor is None else floor["exact_requests"]
    passed = stats["exact_requests"] >= exact_needed and not missed and within_floor
    json.dump({"passed": passed, "run_id": baseline["run_id"], "candidate": stats,
               "floor": floor, "missed_hits": missed, "follow_up_ttft": ttft,
               "host_load": compared_timing({"baseline": baseline.get("host_load"),
                                             "candidate": candidate.get("host_load")})},
              sys.stdout, indent=1)
    print()
    raise SystemExit(0 if passed else 1)


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.command != "compare")
    parser.add_argument("command", choices=("record", "compare"))
    parser.add_argument("--arm", choices=("baseline", "candidate", "floor"), default="candidate")
    parser.add_argument("--run-id", default="prefix-001")
    parser.add_argument("--lengths", default="1024,32768,131072")
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--floor")
    args = parser.parse_args()
    if args.command == "compare":
        if not (args.baseline and args.candidate):
            parser.error("compare needs --baseline and --candidate")
        compare(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
