#!/usr/bin/env python3
"""One capture phase against a running server, timed in unix ns.

Every phase appends one JSON entry (start/end unix ns, so a trace can be cut
by it) to OUT (a JSON file holding {"model": ..., "phases": [...]}). Prompts
are built from a fixed word list and sized through the server's /tokenize to
the largest word count whose prompt is at most the requested tokens, so any
model's tokenizer gets the same prompt lengths. A prompt that must miss the
prefix cache starts with a nonce derived from --seed and the phase name, so
two runs send the same token IDs and their outputs compare bit for bit.

  prefill  one prompt of at most --tokens tokens, max_tokens 1, logprobs 5.
  teacher  one prompt of at most --tokens tokens, echo, prompt_logprobs 5,
           max_tokens 1, logprobs 5: every prompt row's logprobs (teacher
           forcing).
  warm     the decode prefix (at most --prefix-tokens tokens), max_tokens 1,
           so the decode phases' prompts hit the prefix cache. It must
           complete: one choice finishing by length with one token, or by
           stop (end of sequence) with at most one.
  decode   --concurrency streaming requests at once, each the decode prefix
           plus its own short tail (named by the phase and index), greedy,
           ignore_eos, --max-tokens tokens. Records each request's first and
           last arrival and the steady window [latest first token, earliest
           last token] in which every request decodes. A stream must end
           with a finish reason and [DONE] and carry no error event; what it
           saw is recorded and validation refuses an incomplete one.
  gsm8k    --count frozen GSM8K questions (the examples and manifest named by
           --gsm8k-examples and --gsm8k-manifest, checked by digest) as chat
           completions, greedy, --max-tokens tokens, --concurrency requests at
           a time in question order; records each answer's text, generated
           token IDs and token count (no logprobs: an answer-level gate).
  determinism
           tools/qual/determinism.py record (--qual DIR) against the server,
           scored by --scorer (a chopinfer binary), with its image request
           when /health lists the image modality; records the receipt and
           its digest.

Every request but warm and determinism records its generated token IDs
(return_token_ids) and finish reason. A phase the catalog (evidence.py)
declares is validated after it is recorded: a capture missing a requested
token ID, logprob, prompt logprob or top-k entry, or short of the requested
requests, tokens or prompt length, exits 1.

Usage: client.py URL OUT.json PHASE NAME [--tokens N] [--prefix-tokens N]
       [--concurrency N] [--max-tokens N] [--count N] [--seed S]
       [--gsm8k-examples FILE --gsm8k-manifest FILE] [--qual DIR --scorer FILE]
"""

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import threading
import time
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import evidence  # noqa: E402

WORDS = (
    "the history of printing begins with seals and woodblocks long before movable type "
    "presses spread from city to city as merchants bought paper ink and skilled labour "
    "scholars argued about texts while readers in markets learned new words every season "
    "rivers carried barges of books between monasteries universities and noble houses"
).split()


def model_id(url):
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as response:
        return json.load(response)["data"][0]["id"]


def text(words, offset=0):
    return " ".join(WORDS[(offset + i) % len(WORDS)] for i in range(words))


def nonce(seed, name):
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16]


def gsm8k_questions(count, examples_path, manifest_path):
    if not (examples_path and manifest_path):
        raise SystemExit("gsm8k needs --gsm8k-examples and --gsm8k-manifest")
    manifest = json.load(open(manifest_path))
    raw = open(examples_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]:
        raise SystemExit("frozen GSM8K examples changed")
    examples = json.loads(raw)
    return manifest["instruction"], [examples[i % len(examples)]["question"] for i in range(count)]


def post(url, body, stream=False, path="/v1/completions"):
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=900)
    connection.request("POST", path, json.dumps(body),
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    if not stream:
        value = json.load(response)
        if not isinstance(value, dict) or "error" in value:
            raise RuntimeError(f"error response: {json.dumps(value)[:500]}")
        return value
    # A stream counts only when it ends with a finish reason and [DONE] and
    # carries no error event; what it saw is recorded, so the capture is
    # refused by validation rather than lost.
    arrivals, pieces, usage, logprobs, ids = [], [], None, [], []
    finish, done, errors = None, False, []
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        if not isinstance(chunk, dict) or "error" in chunk:
            errors.append(json.dumps(chunk)[:500])
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not all(isinstance(c, dict) and c.get("index", 0) == 0
                                                    for c in choices):
            errors.append(f"invalid choices: {json.dumps(choices)[:500]}")
            continue
        for choice in choices:
            if finish is not None:
                errors.append("a choice after the finish reason")
            if choice.get("text"):
                arrivals.append(time.time_ns())
                pieces.append(choice["text"])
            if choice.get("logprobs"):
                logprobs.append(choice["logprobs"])
            if choice.get("token_ids") is not None:
                if isinstance(choice["token_ids"], list):
                    ids.extend(choice["token_ids"])
                else:
                    errors.append("token_ids is not a list")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if usage is None:
        errors.append("no usage")
    return {"arrivals": arrivals, "text": "".join(pieces), "usage": usage or {}, "logprobs": logprobs,
            "token_ids": ids, "finish_reason": finish, "done": done, "errors": errors}


def count(url, model, prompt):
    response = post(url, {"model": model, "prompt": prompt}, path="/tokenize")
    tokens = response.get("tokens")
    if not (isinstance(tokens, list) and response.get("count") == len(tokens)):
        raise SystemExit(f"/tokenize returned no token count: {response.get('count')}")
    return len(tokens)


def sized(url, model, make, target):
    """make(words) for the largest word count whose prompt is at most `target`
    tokens: a deterministic search, so every arm sends the same prompt."""
    if count(url, model, make(1)) > target:
        raise SystemExit(f"a one-word prompt exceeds {target} tokens")
    low, high = 1, 2
    while count(url, model, make(high)) <= target:
        low, high = high, high * 2
    while high - low > 1:
        middle = (low + high) // 2
        if count(url, model, make(middle)) <= target:
            low = middle
        else:
            high = middle
    return make(low)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("phase", choices=("prefill", "teacher", "warm", "decode", "gsm8k", "determinism"))
    parser.add_argument("name")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--prefix-tokens", type=int, default=evidence.PREFIX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", default="ab-serving")
    parser.add_argument("--gsm8k-examples")
    parser.add_argument("--gsm8k-manifest")
    parser.add_argument("--qual", help="determinism: the tools/qual directory")
    parser.add_argument("--scorer", help="determinism: the chopinfer binary that scores the record")
    args = parser.parse_args()
    model = model_id(args.url)
    entry = {"phase": args.name, "kind": args.phase}
    # Prompts are sized before the timed window starts.
    if args.phase in ("warm", "decode"):
        tag = f"[{nonce(args.seed, 'decode-prefix')}] "
        prefix = sized(args.url, model, lambda words: tag + text(words), args.prefix_tokens)
    elif args.phase in ("prefill", "teacher"):
        tag, offset = f"[{nonce(args.seed, args.name)}] ", 11 if args.phase == "teacher" else 0
        prompt = sized(args.url, model, lambda words: tag + text(words, offset), args.tokens)
    elif args.phase == "gsm8k":
        instruction, questions = gsm8k_questions(args.count, args.gsm8k_examples, args.gsm8k_manifest)
    start = time.time_ns()
    if args.phase == "prefill":
        result = post(args.url, {"model": model, "prompt": prompt, "max_tokens": 1,
                                 "temperature": 0, "logprobs": 5, "return_token_ids": True})
        choice = result["choices"][0]
        entry.update(requests=1, prompt_tokens=result["usage"]["prompt_tokens"],
                     outputs=[{"text": choice.get("text"), "token_ids": choice.get("token_ids"),
                               "logprobs": choice.get("logprobs"),
                               "finish_reason": choice.get("finish_reason")}])
    elif args.phase == "teacher":
        result = post(args.url, {"model": model, "prompt": prompt, "max_tokens": 1,
                                 "temperature": 0, "logprobs": 5, "echo": True,
                                 "prompt_logprobs": 5, "return_token_ids": True})
        choice = result["choices"][0]
        entry.update(requests=1, prompt_tokens=result["usage"]["prompt_tokens"],
                     outputs=[{"text": choice.get("text"), "token_ids": choice.get("token_ids"),
                               "logprobs": choice.get("logprobs"),
                               "prompt_logprobs": choice.get("prompt_logprobs")
                               or result.get("prompt_logprobs"),
                               "finish_reason": choice.get("finish_reason")}])
    elif args.phase == "gsm8k":
        results = [None] * len(questions)
        errors = []

        def ask(index):
            try:
                results[index] = post(args.url, {
                    "model": model, "max_tokens": args.max_tokens, "temperature": 0,
                    "return_token_ids": True, "messages": [{"role": "system", "content": instruction},
                                 {"role": "user", "content": questions[index]}]},
                    path="/v1/chat/completions")
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(f"{index}: {error}")

        for first in range(0, len(questions), args.concurrency):
            threads = [threading.Thread(target=ask, args=(i,))
                       for i in range(first, min(first + args.concurrency, len(questions)))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if errors or any(r is None for r in results):
            raise SystemExit(f"gsm8k requests failed: {errors}")
        entry.update(
            requests=len(results),
            concurrency=args.concurrency,
            prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in results),
            completion_tokens=sum(r["usage"]["completion_tokens"] for r in results),
            outputs=[{"text": r["choices"][0]["message"].get("content"),
                      "reasoning": r["choices"][0]["message"].get("reasoning_content"),
                      "token_ids": r["choices"][0].get("token_ids"),
                      "finish_reason": r["choices"][0].get("finish_reason"),
                      "completion_tokens": r["usage"]["completion_tokens"]} for r in results])
    elif args.phase == "determinism":
        if not (args.qual and args.scorer):
            raise SystemExit("determinism needs --qual and --scorer")
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as response:
            image = "image" in json.load(response).get("modalities", [])
        receipts = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"receipts-{args.name}")
        os.makedirs(receipts, exist_ok=False)
        arm = os.path.basename(os.path.dirname(os.path.abspath(args.out)))
        command = [sys.executable, "-B", os.path.join(args.qual, "determinism.py"), "record", "--url", args.url,
                   "--arm", arm, "--run-id", args.seed] + (["--image"] if image else [])
        subprocess.run(command, check=True, env=dict(os.environ, CHOPINFER_QUAL_RECEIPTS=receipts,
                                                      CHOPINFER_EVAL=args.scorer))
        found = [os.path.join(receipts, name) for name in os.listdir(receipts) if name.endswith(".json")]
        if len(found) != 1:
            raise SystemExit(f"expected one determinism receipt, found {len(found)}")
        record = json.load(open(found[0]))
        entry.update(requests=len(record.get("requests") or []), image=image, receipt=found[0],
                     receipt_sha256=hashlib.sha256(open(found[0], "rb").read()).hexdigest(),
                     passed=record.get("passed"))
    elif args.phase == "warm":
        result = post(args.url, {"model": model, "prompt": prefix, "max_tokens": 1,
                                 "temperature": 0, "return_token_ids": True})
        choices = result.get("choices") or [{}]
        entry.update(requests=1, prompt_tokens=result.get("usage", {}).get("prompt_tokens"),
                     completion_tokens=result.get("usage", {}).get("completion_tokens"),
                     choices=len(result.get("choices") or []),
                     outputs=[{"text": choices[0].get("text"), "token_ids": choices[0].get("token_ids"),
                               "finish_reason": choices[0].get("finish_reason")}])
    else:
        results = [None] * args.concurrency
        errors = []

        def one(index):
            prompt = f"{prefix} [{args.name} {index}] " + text(evidence.TAIL_WORDS, offset=7 * index)
            try:
                results[index] = post(args.url, {
                    "model": model, "prompt": prompt, "max_tokens": args.max_tokens,
                    "temperature": 0, "ignore_eos": True, "logprobs": 1, "return_token_ids": True,
                    "stream": True, "stream_options": {"include_usage": True}}, stream=True)
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(f"{index}: {error}")

        threads = [threading.Thread(target=one, args=(i,)) for i in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors or any(r is None or not r["arrivals"] for r in results):
            raise SystemExit(f"decode requests failed: {errors}")
        first = max(r["arrivals"][0] for r in results)
        last = min(r["arrivals"][-1] for r in results)
        tokens = sum(r["usage"].get("completion_tokens", 0) for r in results)
        entry.update(
            requests=args.concurrency,
            prompt_tokens=sum(r["usage"].get("prompt_tokens", 0) for r in results),
            completion_tokens=tokens,
            steady_start_unix_ns=first, steady_end_unix_ns=last,
            steady_ms=round((last - first) / 1e6, 1),
            ttft_ms=sorted(round((r["arrivals"][0] - start) / 1e6, 1) for r in results),
            ms_per_token=sorted(round((r["arrivals"][-1] - r["arrivals"][0]) / 1e6
                                      / max(1, r["usage"].get("completion_tokens", 0) - 1), 2)
                                for r in results),
            outputs=[dict({"text": r["text"], "token_ids": r["token_ids"], "logprobs": r["logprobs"],
                           "finish_reason": r["finish_reason"], "done": r["done"]},
                          **({"stream_errors": r["errors"]} if r["errors"] else {})) for r in results])
    end = time.time_ns()
    entry.update(start_unix_ns=start, end_unix_ns=end, wall_ms=round((end - start) / 1e6, 1))
    if entry.get("completion_tokens"):
        entry["tokens_per_s"] = round(entry["completion_tokens"] / ((end - start) / 1e9), 1)
    elif entry.get("prompt_tokens"):
        entry["prompt_tokens_per_s"] = round(entry["prompt_tokens"] / ((end - start) / 1e9), 1)
    record = {"model": model, "phases": []}
    if os.path.exists(args.out):
        record = json.load(open(args.out))
    record["phases"].append(entry)
    with open(args.out + ".tmp", "w") as output:
        json.dump(record, output, indent=1)
    os.replace(args.out + ".tmp", args.out)
    summary = {key: value for key, value in entry.items() if key != "outputs"}
    print(json.dumps(summary), flush=True)
    # A capture counts only when it holds every number the phase asked for:
    # refuse it here, after recording it.
    if args.phase == "warm":
        try:
            evidence.warm_complete(entry)
        except evidence.Refusal as error:
            raise SystemExit(f"invalid capture: {error}")
    declared = evidence.catalog().get(args.name, {}).get("spec")
    if declared is not None:
        try:
            compare.validate(entry, declared, args.name)
        except compare.Invalid as error:
            raise SystemExit(f"invalid capture: {error}")


if __name__ == "__main__":
    main()
