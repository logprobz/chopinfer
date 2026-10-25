#!/usr/bin/env python3
"""Served-output determinism: the same C1 greedy requests, bit for bit.

Speculation shapes and prefill chunks may be chosen from measured timings.
They must not change what the server returns. `record` sends a fixed request
set one request at a time (C1), greedy, 48 output tokens with top-5
logprobs, and stores each answer's text, finish reason, token IDs, selected
and top logprobs and usage:

- four short completions;
- one long completion (about 2,800 MiMo or 2,900 Qwen tokens: its prefill spans
  two 2048-token chunks), sent twice;
- with --image, one chat request with a small generated PNG;
- the long prompt as token IDs from /tokenize (`long-ids`), cold;
- per anchor A in HIT_ANCHORS (1024, a page and prefill granule multiple, and
  2000, neither), a primer of the first A token IDs with one output token
  (`long-primer-A`), then the whole token-ID prompt under the primer's salt
  (`long-hit-A`).

`cache_salt` isolates cache keys; captures under one salt cannot serve
another (docs/serving-request-surface.md). Every request but a hit carries a
fresh salt and must report zero cached tokens, so it prefills cold; the second
long send is therefore a second cold prefill of the same prompt within one
server start. A hit reuses its primer's salt and must report
0 < cached_tokens <= A: MiMo captures at the primer's end, Qwen at the last
64-token granule at or below it.

`compare LEFT RIGHT` requires every request of two records to be identical:
texts, finish reasons, token IDs, tokens, and selected and top logprobs with
exact float equality (bit patterns, no tolerance). Within each record the two
cold long sends must be identical, and so must every prefix hit and the cold
token-ID send. It reports the first differing request, field and position, and
the largest |difference| of a logprob at a position both records reached with
the same tokens.

With --server-log, `record` binds the live server to the gate's checkpoint
pin and that startup log (common.bind_checkpoint) and confirms after the
last request that the same process answered every one (confirm_live); the
record's server then names the checkpoint, package and startup nonce, which
a gate can require of both arms.

    python3 tools/qual/determinism.py record --url URL --arm spec --run-id RUN --image --server-log LOG
    python3 tools/qual/determinism.py compare LEFT_RECEIPT RIGHT_RECEIPT
"""

import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import uuid
import zlib

from common import Client, answer_control, arguments, bind_checkpoint, confirm_live, receipt
import experimental_determinism

SHORT_PROMPTS = {
    "capital": "The capital of France is",
    "haiku": "Write a haiku about the ocean.",
    "fibonacci": "def fibonacci(n):",
    "arithmetic": "Question: What is 17 multiplied by 23? Answer:",
}
MAX_TOKENS = 48
TOP_LOGPROBS = 5
SAMPLING = {"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "top_logprobs": TOP_LOGPROBS}
COMPLETION = {"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "logprobs": TOP_LOGPROBS,
              "return_token_ids": True}
# The profiles' prefill chunk. The long prompt must need more than one.
PREFILL_CHUNK_TOKENS = 2048
LONG_RULES = 92
LONG, LONG_REPEAT, IMAGE = "long", "long-cold-repeat", "image"
LONG_IDS = "long-ids"
# Primer lengths in long-prompt tokens: a page and granule multiple, and neither.
HIT_ANCHORS = (1024, 2000)
PRIMERS = {f"long-primer-{anchor}": anchor for anchor in HIT_ANCHORS}
HITS = {f"long-hit-{anchor}": anchor for anchor in HIT_ANCHORS}
IMAGE_SIDE = 256
IMAGE_QUESTION = "Name the four colors in this image, clockwise from the top left."
COLD_REPEAT = {"supported": True, "mechanism": "cache_salt",
               "note": "each cold request carries a fresh cache_salt and must report zero cached tokens, "
                       "so both long sends prefill cold within one server start"}
PREFIX_HIT = {"supported": True, "mechanism": "cache_salt",
              "note": "each long-hit-A reuses the salt of its A-token primer and must report "
                      "0 < cached_tokens <= A; its answer must equal the cold long-ids send"}


def long_prompt():
    """92 numbered rules and a question: about 2,800 MiMo or 2,900 Qwen tokens."""
    colors = ("red", "blue", "green", "yellow", "white", "black", "orange")
    goods = ("apples", "bolts", "candles", "maps", "nails", "paper", "rope", "salt",
             "tea", "wire", "yarn")
    markers = ("square", "circle", "triangle", "star", "arrow")
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
    rules = [f"Rule {n}: A {colors[n % 7]} box that holds {goods[n % 11]} must be placed on shelf "
             f"{n % 9 + 1}, next to the {markers[n % 5]} marker, and checked every {days[n % 6]}."
             for n in range(1, LONG_RULES + 1)]
    return ("The warehouse keeps every box in order with the rules below.\n\n" + "\n".join(rules)
            + "\n\nQuestion: Under rule 58, what does the box hold, on which shelf is it placed, "
              "and on which day is it checked? Answer:")


def image_png():
    """A 256 x 256 RGB PNG of four solid quadrants, generated bit for bit."""
    quadrants = ((220, 30, 30), (30, 160, 60), (30, 60, 220), (240, 210, 40))
    half = IMAGE_SIDE // 2
    rows = []
    for y in range(IMAGE_SIDE):
        left, right = (quadrants[0], quadrants[1]) if y < half else (quadrants[3], quadrants[2])
        rows.append(b"\x00" + bytes(left) * half + bytes(right) * half)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    header = struct.pack(">IIBBBBB", IMAGE_SIDE, IMAGE_SIDE, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def hit_names():
    return [LONG_IDS] + [name for anchor in HIT_ANCHORS
                         for name in (f"long-primer-{anchor}", f"long-hit-{anchor}")]


def request_names(image):
    return [*SHORT_PROMPTS, LONG, LONG_REPEAT] + ([IMAGE] if image else []) + hit_names()


def request_set(image, control=None):
    """(name, path, body) of every text request, without model and cache salt."""
    requests = [(name, "/v1/completions", dict(COMPLETION, prompt=prompt))
                for name, prompt in SHORT_PROMPTS.items()]
    long = dict(COMPLETION, prompt=long_prompt())
    requests += [(LONG, "/v1/completions", long), (LONG_REPEAT, "/v1/completions", dict(long))]
    if image:
        url = "data:image/png;base64," + base64.b64encode(image_png()).decode("ascii")
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": IMAGE_QUESTION}]}],
                "temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "logprobs": True,
                "top_logprobs": TOP_LOGPROBS, "return_token_ids": True}
        if control:
            body.update({key: value for key, value in control.items() if value})
        requests.append((IMAGE, "/v1/chat/completions", body))
    return requests


def hit_requests(token_ids):
    """(name, path, body) of the token-ID requests over the long prompt's IDs."""
    requests = [(LONG_IDS, "/v1/completions", dict(COMPLETION, prompt=list(token_ids)))]
    for anchor in HIT_ANCHORS:
        requests += [(f"long-primer-{anchor}", "/v1/completions",
                      dict(COMPLETION, prompt=list(token_ids[:anchor]), max_tokens=1)),
                     (f"long-hit-{anchor}", "/v1/completions", dict(COMPLETION, prompt=list(token_ids)))]
    return requests


def ids_digest(token_ids):
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def request_digest(requests):
    canonical = json.dumps([[name, path, body] for name, path, body in requests],
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse(path, response):
    """The answer fields of one non-streaming response."""
    choice = response["choices"][0]
    if path == "/v1/chat/completions":
        message = choice["message"]
        rows = (choice.get("logprobs") or {}).get("content") or []
        text = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tokens = [row["token"] for row in rows]
        logprobs = [row["logprob"] for row in rows]
        top = [[[entry["token"], entry["logprob"]] for entry in row["top_logprobs"]] for row in rows]
    else:
        rows = choice.get("logprobs") or {}
        text, reasoning = choice["text"], None
        tokens = rows.get("tokens") or []
        logprobs = rows.get("token_logprobs") or []
        # Kept as ordered pairs: receipts sort object keys.
        top = [list(map(list, (row or {}).items())) for row in rows.get("top_logprobs") or []]
    return {"text": text, "reasoning": reasoning, "finish_reason": choice["finish_reason"],
            "token_ids": choice.get("token_ids"), "tokens": tokens, "logprobs": logprobs,
            "top_logprobs": top, "usage": response.get("usage")}


def cached_tokens(row):
    return ((row.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def cache_error(row):
    """Why a row's reported cached tokens break its rule, or None."""
    name, cached = row["name"], cached_tokens(row)
    if name in HITS:
        if type(cached) is int and 0 < cached <= HITS[name]:
            return None
        return f"{name}: a prefix hit requires reported 0 < cached_tokens <= {HITS[name]}, got {cached}"
    return None if type(cached) is int and cached == 0 else \
        f"{name}: a cold request requires reported cached_tokens=0, got {cached}"


def send(client, model, name, path, body, salt=None):
    salt = salt or f"determinism-{name}-{uuid.uuid4().hex}"
    response, seconds = client.request(path, dict(body, model=model, cache_salt=salt))
    row = {"name": name, "path": path, "cache_salt": salt, "seconds": seconds, **parse(path, response)}
    if error := cache_error(row):
        raise ValueError(error)
    return row


def tokenize(client, model, prompt):
    response, _ = client.request("/tokenize", {"model": model, "prompt": prompt})
    tokens = response.get("tokens")
    if not (isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens)
            and response.get("count") == len(tokens)):
        raise ValueError(f"/tokenize returned no valid token IDs for the long prompt: {response.get('count')}")
    return tokens


def record(args):
    experimental_pin = experimental_determinism.selected(args)
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    if getattr(args, "server_log", None):
        bind_checkpoint(server, args.server_log, args.url, experimental_pin=experimental_pin)
    requests = request_set(args.image, answer_control() if args.image else None)
    # One request at a time: C1.
    rows = [send(client, server["model"], name, path, body) for name, path, body in requests]
    token_ids = tokenize(client, server["model"], long_prompt())
    hits = hit_requests(token_ids)
    salts = {}
    for name, path, body in hits:
        # A hit reuses its primer's salt, so only the primer's capture can serve it.
        salt = salts.get(f"long-primer-{HITS[name]}") if name in HITS else None
        rows.append(send(client, server["model"], name, path, body, salt))
        salts[name] = rows[-1]["cache_salt"]
    requests += hits
    value = {"arm": args.arm, "run_id": args.run_id, "concurrency": 1, "sampling": SAMPLING,
             "request_set": {"names": [name for name, _, _ in requests], "sha256": request_digest(requests),
                             "long_prompt_sha256": hashlib.sha256(long_prompt().encode()).hexdigest(),
                             "long_prompt_tokens": len(token_ids),
                             "long_prompt_token_ids_sha256": ids_digest(token_ids)},
             "image": {"included": args.image,
                       "png_sha256": hashlib.sha256(image_png()).hexdigest() if args.image else None,
                       "reason": args.image_reason or ("the server serves image inputs" if args.image
                                                       else "the server does not serve image inputs")},
             "cold_repeat": COLD_REPEAT, "prefix_hit": dict(PREFIX_HIT, anchors=list(HIT_ANCHORS)),
             "requests": rows, "passed": True}
    if experimental_pin is not None:
        value.update(experimental_determinism=experimental_pin.marker(), qualified=False, server=server)
    check(value)
    confirm_live(client, server, experimental_pin=experimental_pin)
    compared = compare(value, value, experimental=experimental_pin is not None)
    value["cold_repeat"] = dict(COLD_REPEAT, **compared["cold_repeat"]["left"])
    value["prefix_hit"] = dict(value["prefix_hit"], **compared["prefix_hit"]["left"])
    # Its per-request seconds are diagnostics of an accuracy check, not timing evidence, so no host load
    # samples bound them and the receipt's timing reads informational.
    receipt(f"determinism-{args.arm}", server, value, False)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check(value):
    """Raise ValueError unless `value` is a complete determinism record: every
    request cold except the prefix hits, which report 0 < cached <= anchor."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"determinism record: {message}")
    require(isinstance(value, dict) and value.get("passed") is True and not value.get("mechanics_only"),
            "not a passing model record")
    require(value.get("concurrency") == 1 and value.get("sampling") == SAMPLING, "wrong workload")
    image = (value.get("image") or {}).get("included")
    require(type(image) is bool, "missing image decision")
    names = request_names(image)
    requests = value.get("requests") or []
    require((value.get("request_set") or {}).get("names") == names and value["request_set"].get("sha256"),
            "wrong request set")
    require([row.get("name") for row in requests] == names, "missing or reordered requests")
    require((value.get("cold_repeat") or {}).get("supported") is True, "missing cold long repeat")
    prefix_hit = value.get("prefix_hit") or {}
    require(prefix_hit.get("supported") is True and prefix_hit.get("anchors") == list(HIT_ANCHORS),
            "missing prefix hits")
    salts = [row.get("cache_salt") for row in requests]
    rows = {row["name"]: row for row in requests}
    own = [row.get("cache_salt") for row in requests if row["name"] not in HITS]
    require(all(isinstance(salt, str) and salt for salt in salts) and len(set(own)) == len(own)
            and all(rows[name].get("cache_salt") == rows[f"long-primer-{anchor}"].get("cache_salt")
                    for name, anchor in HITS.items()),
            "every request needs its own cache salt, and each hit its primer's")
    for row in requests:
        name = row["name"]
        usage = row.get("usage") or {}
        require(row.get("finish_reason") in ("stop", "length"), f"{name}: bad finish reason")
        require(isinstance(row.get("text"), str), f"{name}: missing text")
        require(cache_error(row) is None,
                f"{name}: not a prefix hit within its anchor" if name in HITS
                else f"{name}: not cold (cached tokens reported)")
        ids = row.get("token_ids")
        require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids),
                f"{name}: empty or invalid token IDs")
        require(len(ids) == usage.get("completion_tokens")
                and len(ids) <= (1 if name in PRIMERS else MAX_TOKENS),
                f"{name}: token IDs differ from usage")
        tokens, logprobs, top = row.get("tokens"), row.get("logprobs"), row.get("top_logprobs")
        require(isinstance(tokens, list) and tokens and isinstance(logprobs, list) and isinstance(top, list)
                and len(tokens) == len(logprobs) == len(top), f"{name}: incomplete logprobs")
        require(all(finite(x) for x in logprobs), f"{name}: invalid selected logprobs")
        require(all(isinstance(entries, list) and entries and all(
                    isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str) and finite(entry[1])
                    for entry in entries) for entries in top), f"{name}: invalid top logprobs")
    prompt = rows[LONG]["usage"].get("prompt_tokens")
    require(type(prompt) is int and prompt > PREFILL_CHUNK_TOKENS,
            f"long prompt has {prompt} tokens; it must exceed one {PREFILL_CHUNK_TOKENS}-token prefill chunk")
    require(rows[LONG_REPEAT]["usage"].get("prompt_tokens") == prompt, "the long sends prefilled different prompts")
    count = value["request_set"].get("long_prompt_tokens")
    require(type(count) is int and count > max(HIT_ANCHORS) and value["request_set"].get("long_prompt_token_ids_sha256"),
            "missing long prompt token IDs")
    require(all(rows[name]["usage"].get("prompt_tokens") == count for name in (LONG_IDS, *HITS))
            and all(rows[name]["usage"].get("prompt_tokens") == anchor for name, anchor in PRIMERS.items()),
            "the token-ID sends prefilled the wrong number of tokens")


def repeat_difference(value):
    return compare(value, value)["cold_repeat"]["left"]


def hit_difference(value):
    return compare(value, value)["prefix_hit"]["left"]


def compare(left, right, *, experimental=False):
    """Bit-identical answers across two records, between the cold long sends
    within each, and between each prefix hit and its record's cold token-ID
    send; see the module docstring."""
    import kinds
    operation = "score_capture" if experimental else "score"
    return kinds.call(operation, kinds.KINDS["determinism"], left, right)["metrics"]


def describe(result):
    """One line naming the first failure of a comparison."""
    parts = []
    if not result["equal"]:
        d = result["first_difference"]
        parts.append(f"{result['left_arm']} and {result['right_arm']} differ at {d['request']} "
                     f"{d['field']} position {d['position']} ({d['left']!r} vs {d['right']!r})")
    for side in ("left", "right"):
        repeat = result["cold_repeat"][side]
        if not repeat["equal"]:
            d = repeat["first_difference"]
            parts.append(f"{result[side + '_arm']}: the two cold long sends differ at {d['field']} "
                         f"position {d['position']} ({d['left']!r} vs {d['right']!r})")
        hit = result["prefix_hit"][side]
        if not hit["equal"]:
            d = hit["first_difference"]
            parts.append(f"{result[side + '_arm']}: {d['request']} differs from the cold {LONG_IDS} send at "
                         f"{d['field']} position {d['position']} ({d['left']!r} vs {d['right']!r})")
    return "; ".join(parts + [f"max |difference| {result['max_abs_difference']!r}"])


def compare_command(args):
    left, right = (json.loads(Path(path).read_text()) for path in (args.left, args.right))
    result = compare(left, right, experimental=getattr(args, "experimental", False))
    json.dump(dict(result, left=args.left, right=args.right), sys.stdout, indent=1)
    print()
    if not result["passed"]:
        print(describe(result), file=sys.stderr)
    raise SystemExit(0 if result["passed"] else 1)


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.command == "record" and (args.image or args.server_log))
    parser.add_argument("command", choices=("record", "compare"))
    parser.add_argument("--arm", help="record: receipt label, e.g. ar, spec or spec-repeat")
    parser.add_argument("--run-id")
    parser.add_argument("--image", action="store_true", help="record: also send the image chat request")
    parser.add_argument("--image-reason", help="record: why the image request is or is not sent")
    parser.add_argument("--server-log", type=Path, help="record: bind the live server to the pin and this startup log")
    parser.add_argument("--experimental", action="store_true",
                        help="record or compare strictly pinned diagnostic captures, never release qualification")
    parser.add_argument("--experimental-pin", type=Path,
                        help="record: sealed complete option map, checkpoint identity and startup nonce")
    parser.add_argument("--experimental-pin-sha256", help="record: SHA256 of the sealed pin file")
    parser.add_argument("left", nargs="?", help="compare: the first record")
    parser.add_argument("right", nargs="?", help="compare: the second record")
    args = parser.parse_args()
    if args.command == "record":
        if not args.arm or not args.run_id:
            parser.error("record needs --arm and --run-id")
        record(args)
    else:
        if args.experimental_pin is not None or args.experimental_pin_sha256 is not None:
            parser.error("compare uses the pins already recorded in its input receipts")
        if not args.left or not args.right:
            parser.error("compare needs two records")
        compare_command(args)


if __name__ == "__main__":
    main()
