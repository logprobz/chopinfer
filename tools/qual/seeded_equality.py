#!/usr/bin/env python3
"""Record and compare exact seeded outputs before and after a runtime change.

Compare the same AR or MTP configuration across builds. This is a preservation
check, not an AR-versus-MTP numerical accuracy gate. The existing greedy
comparison and verify_exact gate remain required. Only C1 captures qualify
as an exact HTTP gate. Concurrent captures require --diagnostic when compared;
use fixed-batch replay for concurrent exactness evidence.
"""

import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math

from common import Client, arguments, receipt
from verify_exact import exact_path, load_exact
from spec_metrics import difference, metrics_text, snapshot


PROFILES = {
    "unfiltered": {"seed": 17, "temperature": 0.8, "top_p": 1.0, "top_k": -1,
                   "min_p": 0.0, "repetition_penalty": 1.0,
                   "frequency_penalty": 0.0, "presence_penalty": 0.0},
    "filtered-history": {"seed": 29, "temperature": 0.8, "top_p": 0.95, "top_k": 40,
                         "min_p": 0.05, "repetition_penalty": 1.1,
                         "frequency_penalty": 0.2, "presence_penalty": 0.1},
}
TOP = 5


def score_bits(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing or nonfinite logprob")
    # Preserve the exact API number, including signed zero, without a tolerance.
    return float(value).hex()


def output(choice, maximum):
    ids = choice.get("token_ids")
    probs = choice.get("logprobs")
    if not isinstance(ids, list) or not ids or len(ids) > maximum or any(
            type(token) is not int or token < 0 for token in ids):
        raise ValueError("missing, empty or invalid token IDs")
    if not isinstance(probs, dict):
        raise ValueError("missing logprobs")
    arrays = [probs.get(key) for key in ("tokens", "token_logprobs", "top_logprobs")]
    if any(not isinstance(values, list) or len(values) != len(ids) for values in arrays):
        raise ValueError("token and logprob lengths differ")
    if any(not isinstance(token, str) for token in arrays[0]):
        raise ValueError("invalid token labels")
    for top in arrays[2]:
        # Completion maps include the selected token and collapse duplicate
        # decoded labels (runtime-server/src/response.rs:2273).
        if not isinstance(top, dict) or not 1 <= len(top) <= TOP + 1 or any(
                not isinstance(key, str) for key in top):
            raise ValueError("missing top-logprob evidence")
        for value in top.values():
            score_bits(value)
    for value in arrays[1]:
        score_bits(value)
    reason = choice.get("finish_reason")
    if reason not in ("length", "stop") or (reason == "length" and len(ids) != maximum):
        raise ValueError("invalid finish reason or truncated length response")
    return {"token_ids": ids, "tokens": arrays[0], "logprobs": arrays[1],
            "top_logprobs": arrays[2], "finish_reason": reason}


def request_cases(exact, prompt_tokens, max_tokens):
    cases = {}
    for text in exact["texts"]:
        prompt = text["prompt_token_ids"][:min(prompt_tokens, len(text["prompt_token_ids"]) - 1)]
        if not prompt:
            raise ValueError("reference has an empty prompt prefix")
        for profile, sampling in PROFILES.items():
            name = text["name"] + "/" + profile
            if name in cases:
                raise ValueError("reference has duplicate request names")
            cases[name] = {"prompt": prompt, "max_tokens": max_tokens, **sampling,
                           "logprobs": TOP, "return_token_ids": True, "skip_special_tokens": False}
    if not cases:
        raise ValueError("reference has no requests")
    return cases


def record(args):
    path = exact_path(args.exact)
    cases = request_cases(load_exact(path), args.prompt_tokens, args.max_tokens)
    client = Client(args.url, timeout=args.timeout, retries=0)
    identity = client.identity(args.model)
    before = snapshot(metrics_text(client), "chopinfer", identity["model"])
    def capture(item):
        name, body = item
        response, _ = client.request("/v1/completions", {"model": identity["model"], **body})
        choices = response.get("choices", [])
        if len(choices) != 1:
            raise ValueError("expected exactly one completion")
        return name, {"request": body, "output": output(choices[0], args.max_tokens)}
    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = dict(pool.map(capture, cases.items()))
    after = snapshot(metrics_text(client), "chopinfer", identity["model"])
    counters = difference(before, after)
    check_speculation(counters, args.k)
    # No timed work, so no host load samples: the receipt's timing reads informational.
    receipt(f"seeded-equality-{args.arm}", identity, {
        "arm": args.arm, "k": args.k, "concurrency": args.concurrency,
        "diagnostic": args.concurrency > 1,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
        "requests": rows, "speculation": counters, "passed": True})


def check_speculation(counters, k):
    # runtime-engine/src/metrics.rs publishes these series only after a
    # verified draft. A successfully scraped AR server therefore has none.
    if k == 0 and counters.get("available") is False and counters.keys() <= {"available", "step_rows"}:
        return
    if not counters.get("available"):
        raise ValueError("speculation counters are missing")
    verified = counters.get("verify_items", 0)
    drafts = counters.get("draft_tokens", 0)
    if (k == 0 and (verified or drafts)) or (k > 0 and (verified <= 0 or drafts <= 0)):
        raise ValueError("record did not exercise its declared AR or MTP path")
    if k == 1 and drafts != verified:
        raise ValueError("record did not exercise fixed K1")


def canonical(value):
    if isinstance(value, float):
        return ("float", score_bits(value))
    if isinstance(value, dict):
        return tuple((key, canonical(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(canonical(item) for item in value)
    return value


def compare_records(first, second, *, diagnostic=False):
    for record in (first, second):
        if record.get("passed") is not True or record.get("mechanics_only") is not False:
            raise ValueError("record is not a passing full capture")
        if not record.get("requests") or type(record.get("concurrency")) is not int or record["concurrency"] < 1:
            raise ValueError("expected nonempty evidence and positive concurrency")
        if not diagnostic and (record["concurrency"] != 1 or record.get("diagnostic", False)):
            raise ValueError("concurrent HTTP records are diagnostic only; use --diagnostic")
        check_speculation(record.get("speculation", {}), record.get("k", -1))
    for key in ("reference_sha256", "prompt_tokens", "max_tokens", "concurrency", "k"):
        if first.get(key) is None or first[key] != second.get(key):
            raise ValueError(f"records differ in {key}")
    if set(first["requests"]) != set(second["requests"]):
        raise ValueError("records ran different requests")
    for name in sorted(first["requests"]):
        left, right = (record["requests"][name] for record in (first, second))
        if left["request"] != right["request"]:
            raise ValueError(f"{name}: sampling parameters or prompt differ")
        for row in (left, right):
            value = row["output"]
            output({"token_ids": value["token_ids"], "finish_reason": value["finish_reason"],
                    "logprobs": {"tokens": value["tokens"], "token_logprobs": value["logprobs"],
                                 "top_logprobs": value["top_logprobs"]}}, first["max_tokens"])
        for field in ("token_ids", "tokens", "logprobs", "top_logprobs", "finish_reason"):
            if canonical(left["output"][field]) != canonical(right["output"][field]):
                return {"request": name, "field": field}
    return None


def main():
    parser = arguments(__doc__)
    parser.add_argument("command", choices=("record", "compare"))
    parser.add_argument("--diagnostic", action="store_true", help="compare without claiming an exact gate")
    parser.add_argument("--exact")
    parser.add_argument("--arm")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("receipts", nargs="*")
    args = parser.parse_args()
    if args.mechanics_only:
        parser.error("exact preservation requires full captures")
    if args.command == "record":
        if not args.arm or not args.exact or args.k < 0 or args.prompt_tokens < 1 or args.max_tokens < 1 or args.concurrency < 1:
            parser.error("record needs --arm, --exact, positive --concurrency and token limits")
        record(args)
        return 0
    if len(args.receipts) != 2:
        parser.error("compare needs two receipts")
    first, second = (json.loads(Path(path).read_text()) for path in args.receipts)
    difference = compare_records(first, second, diagnostic=args.diagnostic)
    print(json.dumps({"passed": None if args.diagnostic else difference is None,
                      "diagnostic": args.diagnostic, "equal": difference is None, "first_difference": difference,
                      "requests": len(first["requests"]), "receipts": args.receipts}, indent=2))
    return 0 if difference is None else 1


if __name__ == "__main__":
    sys.exit(main())
