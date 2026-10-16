"""Prefix-cache hits against cold prefills, bit for bit, and against exact.

A hit restores exactly the state a cold prefill held at the anchor and
prefills the rest as another chunk, and prefill rows do not depend on chunk
boundaries, so a hit is bitwise equal to a one-shot cold prefill
(docs/numerics.md, "Prefix-cache accuracy"). This checks that at every
position of the gate's exact reference (fixtures/<model>/teacher-exact-reference.json)
under three conditions:

- cold: a one-shot prefill of ids[:p];
- chunked: a cold prefill of ids[:p] with a chunk that ends at the anchor B;
- hit: a real prefix-cache hit at B that prefills [B, p).

B is the last page boundary before p, as a hit would use. Each position has
its own cache salt and four completions of one token: a primer ids[:B] plus
other tokens (`primer`: they run to B + one page, or to one past the next
multiple of the 2,048-token chunk when B is at least a chunk, so neither the
primer's prompt-end anchor, which a prefill granule moves down to a multiple
of it, nor its final-chunk anchor lies at or before B; the server has then seen
the pages of [0, B) but holds no anchor that ids[:p] can use), ids[:p] (it
captures at the shared pages, which ends a chunk at B: the chunked arm),
ids[:p] again (the hit), and ids[:p] under a fresh salt (the cold arm).
Positions run only to B + page, so the hit prefills [B, p) as one chunk,
exactly like the chunked arm's second chunk. The scored row is the last
prefill row, whose output the server forces to the reference token (start it
with `--debug-reference-tokens` naming the file `references` writes),
reporting that token's raw logprob and the row's top logprobs.

The gate requires, per position:

- cache: `usage.prompt_tokens_details.cached_tokens` as the anchor rules of
  docs/serving-kv-pool.md ("Where prefix anchors are captured") give it, with
  the model's prefill granule from `qualification.prefix.prefill_granule`
  and the server's reuse unit: 0 for every request but the hit, which
  resumes the longest anchor the chunked arm left before p (`resumed`).
  The unit is one KV page on every rank of a context-parallel group,
  `kv_page_tokens * profile.context_parallel_size` from the bound server's
  /health, recorded as `reuse_tokens` (`reuse_tokens`). The chunked arm
  captures the shared prefix in whole units, its prompt end on the granule
  and where its final chunk starts, each once it spans a unit, so without
  context parallelism the hit resumes B rounded down to the granule, and at
  size N > 1 a hit below one unit, or past the shared units when p ends on
  a page, resumes less;
- equality: the hit and the chunked rows equal the cold rows bit for bit
  (raw logprob and top logprobs; the token is forced);
- accuracy: the hit rows no farther from exact than the cold rows (the
  paired rule of limits.py), which bounds how far a failing hit moved.

With `prompt_end` in the policy (or `--prompt-end-stride`) a second arm
resumes prompt-end anchors: for each position p it picks a primer end B one
to five tokens before p that is not a page boundary, sends ids[:B], then
ids[:p] (the hit) and ids[:p] under a fresh salt (cold). The hit resumes B
rounded down to the granule once that spans a reuse unit, and nothing
otherwise (`prompt_end_resumed`). Its hit rows must equal
its cold rows and pass the same accuracy rule.

`record` also runs two checks that need no reference, when the policy enables
them:

- media: image prompts with identical text but different images (another
  color, and a pair of transposed grids with identical patch rows)
  must share neither an anchor past the image nor encoder rows, while a
  repeat of the same image must hit. The text after the image is sized so
  each prompt spans two reuse units past the image (`media_cases`); a case
  that cannot is refused before scoring starts;
- acceptance: on a server with a drafter, greedy continuations after a hit,
  after the chunked prefill and after a cold prefill must be the same tokens,
  and acceptance after hits must not trail cold acceptance by more than the
  margin, since restored drafter history drives the drafts after a hit.

`record --split C` writes a separate receipt for prompts longer than the
reference: prompt-end anchors at multiples of C, over the reference texts and
`--split-prompt-tokens` synthetic prompts. The first row is compared at the
requested concurrency; full greedy continuations are compared serially on an
otherwise idle server to control decode batch shape. Both passes require
exact tokens, logprobs and top logprobs. `report
--split-record` adds such receipts to the verdict.

`report` recomputes the statistics and the gate from a record, or from
records over disjoint `--texts` (each from its own server start, which keeps
a long reference text within one job) that together score every position
once; a strided or partial set does not pass. `record --shard I/N` scores the
I-th of N contiguous slices of the position and prompt-end sets, balanced by
the prompt tokens each position prefills, so a long text spreads over N
server starts; the media and acceptance checks run in shard 1 only, and
`report` needs every shard of N exactly once (`report --slice` judges one
shard on its own slice, the quick suite's screen, which no declared check
uses). All main and split receipts
must name the same package, checkpoint, metadata, source commit, bundle,
model and effective profile. Equal build identities permit process restarts;
each receipt must bind its own startup nonce. Coverage uses the declared
`qualification.prefix.block_tokens`; other page sizes are diagnostic only.

Qwen3.8 uses its bound F32 teacher reference. Both models require the
media-key and drafter-acceptance checks declared in their prefix policy.

The release preview's determinism gate (tools/qual/determinism.py) checks
hits and chunk sizes on every run; this tool covers every page boundary of
the reference, off-page primer ends whose prompt-end anchors are rounded
down to the prefill granule, media keys and drafter acceptance.
"""

import common
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import struct
import sys
import zlib

from common import (qualification_config, input_path, answer_kwargs, Client, arguments, bind_checkpoint,
                    bound_live, confirm_live, receipt)
import limits as rules
from prefix_cache import TOKENS_PER_RECORD, text as record_text
from spec_metrics import difference, metrics_text, snapshot
from teacher import TOP1_SIGMAS, clustered_interval
import verify_exact
from verify_exact import ROW_MEAN_MARGIN, detokenized, disagrees, same_top1

BLOCK_TOKENS = 64
TOP_LOGPROBS = 5
ARMS = ("cold", "chunked", "hit")
# The profiles' prefill chunk. A prompt longer than one chunk also leaves an
# anchor where its final chunk starts (docs/serving-kv-pool.md).
PREFILL_CHUNK_TOKENS = 2048
PROMPT_END_ARMS = ("cold", "hit")
ACCEPTANCE_LENGTHS = (256, 1024, 4096)
# Suffix lengths after a split anchor: one token, a few, and a whole chunk
# (larger values are cut to the chunk).
SPLIT_DELTAS = (1, 2, 3, 17, 1 << 30)
SPLIT_TOKENS = 4
# Acceptance after hits may trail cold acceptance by two binomial standard
# errors, or by this much when there are few drafts.
ACCEPTANCE_MARGIN = 0.02


def positions(texts, block=BLOCK_TOKENS, stride=1):
    """(text, p, B) for every scored position p past the first block, with B
    the last block boundary before p."""
    scored = []
    for text in texts:
        ids = text["prompt_token_ids"]
        for position in range(max(text["score_from"], block + 1), len(ids), stride):
            scored.append((text, position, (position - 1) // block * block))
    return scored


def load_exact(path):
    """The exact reference, refused unless it holds the pinned bytes when it
    is the gate's pinned exact input."""
    exact = verify_exact.load_exact(Path(path))
    validate_policy(exact)
    return exact


def validate_policy(exact, recorded=None):
    policy = prefix_policy()
    if exact.get("schema") != policy["reference_schema"]:
        raise ValueError(f"reference schema {exact.get('schema')!r} differs from configured "
                         f"{policy['reference_schema']!r}")
    if recorded is not None and recorded.get("family") != policy["family"]:
        raise ValueError(f"recorded family {recorded.get('family')!r} differs from configured "
                         f"{policy['family']!r}")


def prefix_policy():
    return qualification_config()["qualification"]["prefix"]


def shard_entries(entries, index, count):
    """The `index`-th (from 1) of `count` contiguous slices of the scored
    entries (text, position, anchor), balanced by the tokens each position
    prefills: its position."""
    total = sum(position for _, position, _ in entries)
    cumulative, chosen = 0, []
    for entry in entries:
        # The entry belongs to the slice its cost's midpoint falls in.
        middle = cumulative + entry[1] / 2
        cumulative += entry[1]
        if min(count, int(middle * count // total) + 1 if total else 1) == index:
            chosen.append(entry)
    return chosen


def parse_shard(value):
    index, _, count = (value or "").partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise ValueError(f"--shard expects I/N with 1 <= I <= N, not {value!r}")
    return int(index), int(count)


def prompt_end_positions(texts, block=BLOCK_TOKENS, stride=1):
    """(text, p, B) for every scored position p, with B a prompt end one to
    five tokens before p that is not a block boundary and spans a block."""
    scored = []
    for text in texts:
        ids = text["prompt_token_ids"]
        for position in range(max(text["score_from"], block + 2), len(ids), stride):
            anchor = position - 1 - position % 5
            if anchor % block == 0:
                anchor -= 1
            if anchor >= block:
                scored.append((text, position, anchor))
    return scored


def primer(ids, anchor, chunk=None, block=BLOCK_TOKENS):
    """ids[:anchor] and a token that differs from ids[anchor], repeated, so
    the server has seen the blocks before the anchor but holds no anchor ids
    can use.

    The primer's prompt-end anchor sits at its end, or at the last multiple
    of the prefill granule (at most a block) at or before it, so the primer
    runs one block past `anchor`. A primer longer than one chunk also leaves
    an anchor where its final chunk starts, at or before `anchor`: such a
    primer runs to one past the next multiple of `chunk`, so its final chunk
    starts after it diverges."""
    chunk = PREFILL_CHUNK_TOKENS if chunk is None else chunk
    other = next(token for token in ids if token != ids[anchor])
    length = anchor + block if anchor < chunk else (anchor // chunk + 1) * chunk + 1
    return ids[:anchor] + [other] * (length - anchor)


def reuse_tokens(server):
    """The tokens in which `server` (a record's server identity) finds shared
    prefixes and captures prompt ends: one KV page on every rank of its
    context-parallel group, as its /health reports them
    (docs/serving-kv-pool.md#context-parallelism). A host that predates
    `kv_page_tokens` served context-parallel size 1 over the declared block."""
    serving = (server.get("health") or {}).get("serving") or {}
    size = (serving.get("profile") or {}).get("context_parallel_size", 1)
    page = serving.get("kv_page_tokens")
    if type(size) is not int or size < 1:
        raise SystemExit(f"refused: the server reports context-parallel size {size!r}")
    if page is None and size == 1:
        return prefix_policy()["block_tokens"]
    if type(page) is not int or page < 1:
        raise SystemExit(f"refused: the server reports context-parallel size {size} and KV page tokens "
                         f"{page!r}")
    return page * size


def record_reuse(recorded):
    """The reuse unit of a record's server, which the record states as it
    was scored with."""
    reuse = reuse_tokens(recorded.get("server") or {})
    if recorded.get("reuse_tokens", reuse) != reuse:
        raise SystemExit(f"refused: the record was scored with reuse_tokens {recorded['reuse_tokens']!r}, "
                         f"its server reuses {reuse}")
    return reuse


def resumed(position, anchor, reuse):
    """The cached tokens of a main-arm hit at `position`: the longest anchor
    before it that the chunked arm left (the primer leaves none the prompt
    can use). The chunked arm, a cold request ids[:position], captures as
    the scheduler does (runtime-scheduler `capture_points`), each point once
    it spans a unit of `reuse` tokens: the prefix it shares with the primer,
    which sent ids[:anchor], in whole units; its prompt end on the granule;
    and where its final chunk of the profile's prefill chunk starts, a whole
    number of chunks after the last capture before the prompt end."""
    granule = prefix_policy()["prefill_granule"]
    points = []
    shared = min(anchor // reuse * reuse, (position - 1) // reuse * reuse) // granule * granule
    if shared >= reuse:
        points.append(shared)
    end = position // granule * granule
    if end >= reuse:
        points.append(end)
    start = max((tokens for tokens in points if tokens < position), default=0)
    repeat = (start + (position - start - 1) // PREFILL_CHUNK_TOKENS * PREFILL_CHUNK_TOKENS) // granule * granule
    if repeat > start and repeat >= reuse:
        points.append(repeat)
    return max((tokens for tokens in points if tokens < position), default=0)


def prompt_end_resumed(anchor, reuse):
    """The cached tokens of a hit after a primer that ended at `anchor`: its
    prompt end on the granule, once that spans a unit of `reuse` tokens. The
    primer's final-chunk point lies at or before that end, so it adds none."""
    tokens = granule_floor(anchor)
    return tokens if tokens >= reuse else 0


def granule_floor(tokens):
    """Where a prompt-end anchor at `tokens` sits: the last multiple of the
    model's prefill granule at or before it."""
    granule = prefix_policy()["prefill_granule"]
    return tokens // granule * granule


def complete(client, model, prompt, salt, max_tokens=1, forced=True):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "top_p": 1, "logprobs": TOP_LOGPROBS, "return_token_ids": True,
            "skip_special_tokens": False, "cache_salt": salt}
    if forced or max_tokens > 1:
        body["ignore_eos"] = True
    response, seconds = client.request("/v1/completions", body)
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    usage = response["usage"]
    return {"token_ids": choice.get("token_ids") or [], "logprob": logprobs["token_logprobs"],
            "top": logprobs["top_logprobs"], "seconds": seconds,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def score_position(client, model, text, position, anchor, run_id):
    """The three arms' forced rows at one position, and their cached tokens."""
    ids = text["prompt_token_ids"]
    salt = f"prefix-exact/{run_id}/{text['name']}/{position}"
    requests = {"primer": complete(client, model, primer(ids, anchor), salt, forced=False),
                "chunked": complete(client, model, ids[:position], salt),
                "hit": complete(client, model, ids[:position], salt),
                "cold": complete(client, model, ids[:position], salt + "/cold")}
    rows = {}
    for arm in ARMS:
        result = requests[arm]
        if result["token_ids"][:1] != [ids[position]]:
            raise RuntimeError(f"{text['name']}@{position}: the server does not force reference "
                               "tokens; start it with --debug-reference-tokens")
        rows[arm] = {"logprob": result["logprob"][0], "top": result["top"][0]}
    return {"text": text["name"], "category": text["category"], "position": position,
            "anchor": anchor, **rows,
            "cached": {name: result["cached_tokens"] for name, result in requests.items()}}


def score_prompt_end(client, model, text, position, anchor, run_id):
    """The hit and cold forced rows at one position, the hit resuming a
    prompt-end anchor at `anchor`, and their cached tokens."""
    ids = text["prompt_token_ids"]
    salt = f"prefix-exact/{run_id}/prompt-end/{text['name']}/{position}"
    requests = {"primer": complete(client, model, ids[:anchor], salt, forced=False),
                "hit": complete(client, model, ids[:position], salt),
                "cold": complete(client, model, ids[:position], salt + "/cold")}
    rows = {}
    for arm in PROMPT_END_ARMS:
        result = requests[arm]
        if result["token_ids"][:1] != [ids[position]]:
            raise RuntimeError(f"{text['name']}@{position}: the server does not force reference "
                               "tokens; start it with --debug-reference-tokens")
        rows[arm] = {"logprob": result["logprob"][0], "top": result["top"][0]}
    return {"text": text["name"], "category": text["category"], "position": position,
            "anchor": anchor, **rows,
            "cached": {name: result["cached_tokens"] for name, result in requests.items()}}


def expected_cached(row, reuse, arms=ARMS):
    """The cached tokens the anchor rules give each request of a scored
    position on a server reusing units of `reuse` tokens: the hit resumes
    the main arm's anchors (`resumed`, arms with a chunked request) or the
    primer's prompt end (`prompt_end_resumed`), every other request prefills
    cold."""
    hit = resumed(row["position"], row["anchor"], reuse) if "chunked" in arms \
        else prompt_end_resumed(row["anchor"], reuse)
    return {"primer": 0, **{arm: hit if arm == "hit" else 0 for arm in arms}}


def cache_errors(scored, reuse, arms=ARMS):
    """Positions whose requests did not hit or miss as the anchor rules give."""
    errors = []
    for row in scored:
        expected = expected_cached(row, reuse, arms)
        if row["cached"] != expected:
            errors.append({"text": row["text"], "position": row["position"],
                           "expected": expected, "cached": row["cached"]})
    return errors


def interval(values, texts):
    return clustered_interval(values, texts) if len(set(texts)) > 1 else None


def paired(test, baseline, texts):
    """The paired rule (limits.py) of two arms on the same positions.

    Each entry is (distance, disagrees); the test arm may sit at most
    ROW_MEAN_MARGIN farther than the baseline on average, and its extra top-1
    disagreements must stay within TOP1_SIGMAS of the discordant count.
    """
    gap = [t[0] - b[0] for t, b in zip(test, baseline)]
    return {**rules.paired(test, baseline, mean_margin=ROW_MEAN_MARGIN, sigmas=TOP1_SIGMAS),
            "mean_gap_ci95": interval(gap, texts)}


def summarize(scored, exact, strings, variant="exact", arms=ARMS):
    """Distances from exact and from cold, bitwise equal rows, and the
    accuracy gate."""
    if not scored:
        return {"positions": 0}
    reference = {text["name"]: text for text in exact["texts"]}
    texts = [row["text"] for row in scored]
    versus_exact = {arm: [] for arm in arms}
    for row in scored:
        text = reference[row["text"]]
        index = row["position"] - text["score_from"]
        token = text["prompt_token_ids"][row["position"]]
        for arm in arms:
            versus_exact[arm].append((abs(row[arm]["logprob"] - text[variant]["logprob"][index]),
                                      disagrees(row[arm], token, text[variant]["top1"][index],
                                                strings)))
    versus_cold = {arm: [(abs(row[arm]["logprob"] - row["cold"]["logprob"]),
                          not same_top1(row[arm], row["cold"])) for row in scored]
                   for arm in arms if arm != "cold"}
    count = len(scored)

    def distance(arm):
        rows = versus_cold[arm]
        return {"mean_abs_dlogprob": statistics.mean(d for d, _ in rows),
                "max_abs_dlogprob": max(d for d, _ in rows),
                "top1_differs": sum(f for _, f in rows),
                "bitwise_equal_rows": sum(row[arm] == row["cold"] for row in scored)}
    pairs = [(test, base) for test, base in (("hit", "cold"), ("chunked", "cold"),
                                             ("hit", "chunked"))
             if test in arms and base in arms]
    summary = {
        "positions": count,
        "texts": len(set(texts)),
        "vs_exact": {arm: {"mean_abs_dlogprob": statistics.mean(d for d, _ in rows),
                           "max_abs_dlogprob": max(d for d, _ in rows),
                           "top1_disagreement": sum(f for _, f in rows) / count}
                     for arm, rows in versus_exact.items()},
        "paired_vs_exact": {
            f"{test}_minus_{base}": {
                "mean": statistics.mean(t[0] - b[0] for t, b in
                                        zip(versus_exact[test], versus_exact[base])),
                "ci95": interval([t[0] - b[0] for t, b in
                                  zip(versus_exact[test], versus_exact[base])], texts)}
            for test, base in pairs},
        "distances": {f"{arm}_vs_cold": distance(arm) for arm in versus_cold},
        # Hit rows no farther from exact than cold one-shot rows. Equal rows
        # pass it; a failing hit shows how far it moved.
        "gate_hit_vs_cold": paired(versus_exact["hit"], versus_exact["cold"], texts),
    }
    return summary


def anchor_group(anchor):
    policy = prefix_policy()
    return policy["anchor_labels"][int(anchor >= policy["anchor_boundary"])]


def report_scores(scored, exact, strings, arms=ARMS):
    groups = {}
    for row in scored:
        groups.setdefault(anchor_group(row["anchor"]), []).append(row)
    scores = {"all": summarize(scored, exact, strings, arms=arms)}
    scores["by_anchor"] = {name: summarize(rows, exact, strings, arms=arms)
                           for name, rows in sorted(groups.items())}
    # One Qwen3.8 text has no category.
    categories = {str(r["category"]) for r in scored}
    scores["by_category"] = {name: summarize([r for r in scored if str(r["category"]) == name],
                                             exact, strings, arms=arms)
                             for name in sorted(categories)}
    return scores


def block_colors(width, height, x, y):
    """Eight colors, one per 32x32 block in row-major block order."""
    block = (y // 32) * (width // 32) + x // 32
    return (block * 29 % 256, (255 - block * 23) % 256, block * 17 % 256)


def png(width, height, color):
    """An RGB PNG; `color` is one RGB triple or a function of (width, height, x, y)."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    def pixel(x, y):
        return bytes(color(width, height, x, y) if callable(color) else color)
    rows = b"".join(b"\0" + b"".join(pixel(x, y) for x in range(width)) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


MEDIA_PAIRS = (
    ("color", "What color is the image? Answer with one color word.",
     (256, 256, (0, 0, 255)), (256, 256, (255, 0, 0))),
    # With 16-pixel patches merged 2x2, these grids are
    # [1, 8, 4] and [1, 4, 8], but the eight blocks come in the same order, so
    # the patch rows and the placeholder count are identical; only the media
    # key's geometry tells them apart.
    ("geometry", "List the colors you see, in reading order.",
     (64, 128, block_colors), (128, 64, block_colors)),
)


# The trailing text's starting size (record_text's tokens) and how many
# times media_cases may grow it to reach two reuse units past the media.
MEDIA_TEXT_TOKENS = 192
MEDIA_SIZING_ROUNDS = 8


def media_messages(image, question, run_id, text_tokens=MEDIA_TEXT_TOKENS):
    parts = []
    if image is not None:
        # The encoder cache outlives a record and does not see the salt, so
        # each run shades its images differently (64 shades).
        shade = int(hashlib.sha256(run_id.encode()).hexdigest()[:2], 16) % 64
        width, height, color = image

        def shaded(*at):
            return tuple(max(0, value - shade)
                         for value in (color(*at) if callable(color) else color))
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                      + base64.b64encode(png(width, height, shaded)).decode("ascii")}})
    # Text after the image; media_cases sizes it to the server's reuse unit.
    parts.append({"type": "text", "text": f"Run {run_id}.\n{record_text(text_tokens, 5)}\n{question}"})
    return [{"role": "user", "content": parts}]


def chat(client, model, messages, salt):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 8,
            "logprobs": True, "top_logprobs": 2, "cache_salt": salt,
            "chat_template_kwargs": answer_kwargs()}
    response, _ = client.request("/v1/chat/completions", body)
    choice = response["choices"][0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    return {"content": choice["message"].get("content"),
            "tokens": [entry["token"] for entry in entries],
            "logprobs": [entry["logprob"] for entry in entries],
            "cached_tokens": (response["usage"].get("prompt_tokens_details") or {})
            .get("cached_tokens")}


def tokenize(client, model, messages):
    response, _ = client.request("/tokenize", {
        "model": model, "messages": messages,
        "chat_template_kwargs": answer_kwargs()})
    return response["tokens"]


def first_difference(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def counter(text, name):
    """The sum of every sample of one counter in a /metrics scrape."""
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith((name + " ", name + "{")))


def media_cases(client, model, run_id, reuse):
    """Each media pair's prompts, with the text after the image grown until
    both prompts span two units of `reuse` tokens past the media start: a
    hit is found and captured only in whole units (docs/serving-kv-pool.md,
    "Context parallelism"), so a shorter prompt can never show the positive
    control's hit, nor a wrongly shared anchor past the media. Refuses a
    case that does not reach that length."""
    cases = []
    for name, question, first, second in MEDIA_PAIRS:
        text_tokens = MEDIA_TEXT_TOKENS
        for _ in range(MEDIA_SIZING_ROUNDS):
            first_messages = media_messages(first, question, run_id, text_tokens)
            second_messages = media_messages(second, question, run_id, text_tokens)
            text_only = tokenize(client, model, media_messages(None, question, run_id, text_tokens))
            first_ids = tokenize(client, model, first_messages)
            second_ids = tokenize(client, model, second_messages)
            media_start = first_difference(first_ids, text_only)
            short = media_start + 2 * reuse - min(len(first_ids), len(second_ids))
            if short <= 0:
                break
            # Records run about TOKENS_PER_RECORD tokens; grow by half again.
            text_tokens += -(-short * 3 // 2 // TOKENS_PER_RECORD) * TOKENS_PER_RECORD
        else:
            raise SystemExit(f"refused: media case {name!r} spans {len(first_ids)} and {len(second_ids)} "
                             f"prompt tokens after {MEDIA_SIZING_ROUNDS} sizing rounds, short of its media "
                             f"start {media_start} plus two reuse units of {reuse} tokens")
        cases.append({"name": name, "question": question, "images": (first, second),
                      "messages": (first_messages, second_messages), "media_start": media_start,
                      "prompt_tokens": (len(first_ids), len(second_ids)), "text_tokens": text_tokens})
    return cases


def media_check(client, model, run_id, reuse, cases=None):
    """Same text, different images: no anchor or encoder rows may be shared."""
    checked = []
    for sized in cases or media_cases(client, model, run_id, reuse):
        name, media_start = sized["name"], sized["media_start"]
        first, second = sized["images"]
        first_messages, second_messages = sized["messages"]
        salt = f"prefix-exact/{run_id}/media/{name}"
        # The third repeat hits: the second captures at the shared units, or
        # already hits the first's prompt end or final chunk.
        repeats = [chat(client, model, first_messages, salt) for _ in range(3)]
        before = counter(metrics_text(client), "chop_media_cache_hits_total")
        other = chat(client, model, second_messages, salt)
        encoder_hits = counter(metrics_text(client), "chop_media_cache_hits_total") - before
        other_cold = chat(client, model, second_messages, salt + "/cold")
        case = {
            "name": name, "media_start": media_start, "reuse_tokens": reuse,
            "text_tokens": sized["text_tokens"],
            "first": {"image": first[:2], "content": repeats[0]["content"],
                      "prompt_tokens": sized["prompt_tokens"][0],
                      "cached_tokens": [run["cached_tokens"] for run in repeats]},
            "second": {"image": second[:2], "content": other["content"],
                       "prompt_tokens": sized["prompt_tokens"][1],
                       "cached_tokens": other["cached_tokens"], "encoder_cache_hits": encoder_hits,
                       "cold_content": other_cold["content"],
                       "same_tokens_as_cold": other["tokens"] == other_cold["tokens"],
                       "max_abs_dlogprob_vs_cold": max(
                           (abs(a - b) for a, b in zip(other["logprobs"], other_cold["logprobs"])),
                           default=0.0)},
            # The positive control: the same image does share its anchor.
            "same_image_hit": (repeats[2]["cached_tokens"] or 0) > media_start,
            "shared_past_media": (other["cached_tokens"] or 0) > media_start,
            "shared_encoder_rows": encoder_hits > 0}
        case["passed"] = (case["same_image_hit"] and not case["shared_past_media"]
                          and not case["shared_encoder_rows"])
        checked.append(case)
    return {"cases": checked, "reuse_tokens": reuse, "passed": all(case["passed"] for case in checked)}


def acceptance_prompts(client, model, run_id, count):
    prompts = []
    for index in range(count):
        length = ACCEPTANCE_LENGTHS[index % len(ACCEPTANCE_LENGTHS)]
        messages = [{"role": "user", "content": (
            f"Run {run_id}, prompt {index}.\n{record_text(length, 11 + index)}\n"
            "Rewrite the last three records as a numbered list, then explain what the "
            "numbers could mean.")}]
        prompts.append(tokenize(client, model, messages))
    return prompts


def generated(client, model, ids, salt, tokens):
    before = snapshot(metrics_text(client), "chopinfer", model)
    result = complete(client, model, ids, salt, tokens, forced=False)
    counters = difference(before, snapshot(metrics_text(client), "chopinfer", model))
    return {"cached_tokens": result["cached_tokens"], "token_ids": result["token_ids"],
            "draft_tokens": counters.get("draft_tokens", 0),
            "accepted_tokens": counters.get("accepted_tokens", 0)}


def acceptance_check(client, model, run_id, count, tokens, reuse):
    """Greedy continuations and draft acceptance after hits, chunked prefills
    and cold prefills, on a server reusing units of `reuse` tokens."""
    runs = []
    for index, ids in enumerate(acceptance_prompts(client, model, run_id, count)):
        anchor = (len(ids) - 1) // BLOCK_TOKENS * BLOCK_TOKENS
        salt = f"prefix-exact/{run_id}/acceptance/{index}"
        complete(client, model, primer(ids, anchor), salt, forced=False)
        arms = {"chunked": generated(client, model, ids, salt, tokens),
                "hit": generated(client, model, ids, salt, tokens),
                "cold": generated(client, model, ids, salt + "/cold", tokens)}
        runs.append({"prompt_tokens": len(ids), "anchor": anchor, **arms})
    return summarize_acceptance(runs, reuse)


def summarize_acceptance(runs, reuse):
    totals = {arm: {"draft_tokens": sum(run[arm]["draft_tokens"] for run in runs),
                    "accepted_tokens": sum(run[arm]["accepted_tokens"] for run in runs)}
              for arm in ARMS}
    for value in totals.values():
        value["rate"] = (value["accepted_tokens"] / value["draft_tokens"]
                         if value["draft_tokens"] else None)
    missed = [index for index, run in enumerate(runs)
              if run["hit"]["cached_tokens"] != resumed(run["prompt_tokens"], run["anchor"], reuse)
              or run["chunked"]["cached_tokens"] or run["cold"]["cached_tokens"]]
    # Greedy continuations after a hit or a chunked prefill are the cold ones.
    differing = [index for index, run in enumerate(runs)
                 if run["hit"]["token_ids"] != run["cold"]["token_ids"]
                 or run["chunked"]["token_ids"] != run["cold"]["token_ids"]]
    result = {"runs": runs, "totals": totals, "cache_errors": missed,
              "differing_tokens": differing,
              "hit_tokens_equal_chunked": sum(run["hit"]["token_ids"] == run["chunked"]["token_ids"]
                                              for run in runs),
              "hit_tokens_equal_cold": sum(run["hit"]["token_ids"] == run["cold"]["token_ids"]
                                           for run in runs)}
    hit, cold = totals["hit"], totals["cold"]
    if not hit["draft_tokens"] or not cold["draft_tokens"]:
        result.update(passed=not missed and not differing, skipped="the server drafted nothing")
        return result
    pooled = ((hit["accepted_tokens"] + cold["accepted_tokens"])
              / (hit["draft_tokens"] + cold["draft_tokens"]))
    error = math.sqrt(pooled * (1 - pooled)
                      * (1 / hit["draft_tokens"] + 1 / cold["draft_tokens"]))
    result["margin"] = max(ACCEPTANCE_MARGIN, 2 * error)
    result["passed"] = not missed and not differing and \
        hit["rate"] >= cold["rate"] - result["margin"]
    return result


def split_positions(prompts, chunk, deltas=SPLIT_DELTAS, max_anchors=None):
    """(prompt index, p, B) with B a multiple of `chunk` and p = B + delta
    within the prompt (delta at most `chunk`), B the end of a primer whose
    prompt-end anchor the request ids[:p] resumes. `max_anchors` spreads at
    most that many anchors evenly over each prompt."""
    scored = []
    for index, ids in enumerate(prompts):
        anchors = list(range(chunk, len(ids), chunk))
        if max_anchors and len(anchors) > max_anchors:
            step = (len(anchors) - 1) / (max_anchors - 1) if max_anchors > 1 else 0
            anchors = sorted({anchors[round(i * step)] for i in range(max_anchors)})
        for anchor in anchors:
            for delta in sorted({min(delta, chunk) for delta in deltas}):
                if delta > 0 and anchor + delta <= len(ids):
                    scored.append((index, anchor + delta, anchor))
    return sorted(scored, key=lambda entry: (entry[0], entry[2], entry[1]))


def score_split(client, model, name, ids, position, anchor, run_id, tokens=SPLIT_TOKENS):
    """Greedy continuations of ids[:p] after a hit at the prompt-end anchor
    of the primer ids[:anchor], and after a cold prefill: every row must be
    equal."""
    salt = f"prefix-exact/{run_id}/split/{name}/{position}"
    requests = {"primer": complete(client, model, ids[:anchor], salt, forced=False),
                "hit": complete(client, model, ids[:position], salt, tokens, forced=False),
                "control": complete(client, model, ids[:position], salt + "/cold", tokens,
                                    forced=False)}
    return {"text": name, "position": position, "anchor": anchor,
            **{arm: {key: requests[arm][key] for key in ("token_ids", "logprob", "top")}
               for arm in ("hit", "control")},
            "cached": {arm: result["cached_tokens"] for arm, result in requests.items()}}


def score_split_passes(client, model, prompts, entries, run_id, concurrency):
    """Concurrent prefill rows, then isolated continuations on an idle server.

    Decode reduction order depends on batch shape. Separate salts keep both
    passes cold at their primers and controls.
    """
    def score(entry, phase, tokens):
        index, position, anchor = entry
        name, ids = prompts[index]
        return score_split(client, model, name, ids, position, anchor,
                           f"{run_id}/{phase}", tokens)

    prefill = scored_concurrently(concurrency, lambda entry: score(entry, "prefill", 1), entries)
    rows = [score(entry, "continuation", SPLIT_TOKENS) for entry in entries]
    for row, first in zip(rows, prefill):
        row["concurrent_prefill"] = first
    return rows


def summarize_split(rows, reuse, tokens=SPLIT_TOKENS):
    """Every hit equal to its cold control, and hits and misses as the anchor
    rules give on a server reusing units of `reuse` tokens."""
    differing = [{"text": row["text"], "position": row["position"], "anchor": row["anchor"]}
                 for row in rows if row["hit"] != row["control"]]
    errors = [{"text": row["text"], "position": row["position"], "cached": row["cached"]}
              for row in rows
              if row["cached"] != {"primer": 0, "hit": prompt_end_resumed(row["anchor"], reuse),
                                   "control": 0}]
    row_errors = [{"text": row["text"], "position": row["position"], "arm": arm,
                   "expected": tokens, "lengths": lengths}
                  for row in rows for arm in ("hit", "control")
                  for lengths in [{key: len(row[arm][key])
                                   for key in ("token_ids", "logprob", "top")}]
                  if any(length != tokens for length in lengths.values())]
    compared = [(a, b) for row in rows
                for a, b in zip(row["hit"]["logprob"], row["control"]["logprob"])]
    prefill = [row["concurrent_prefill"] for row in rows if "concurrent_prefill" in row]
    if prefill:
        concurrent = summarize_split(prefill, reuse, tokens=1)
        differing += concurrent["differing"]
        errors += concurrent["cache_errors"]
        row_errors += concurrent["row_errors"]
        compared += [(a, b) for row in prefill
                     for a, b in zip(row["hit"]["logprob"], row["control"]["logprob"])]
    return {"positions": len(rows), "anchors": sorted({row["anchor"] for row in rows}),
            "max_abs_dlogprob": max((abs(a - b) for a, b in compared), default=0.0),
            "differing": differing, "cache_errors": errors, "row_errors": row_errors,
            "passed": bool(rows) and not differing and not errors and not row_errors}


def equal_rows(scored, arm):
    """Every row of `arm` bitwise equal to the cold row at its position."""
    return bool(scored) and all(row[arm] == row["cold"] for row in scored)


def evaluate(recorded, exact):
    """The gate: expected hits, hit and chunked rows equal to cold, accuracy
    against exact, the prompt-end arm, media and acceptance."""
    validate_policy(exact, recorded)
    reuse = record_reuse(recorded)
    scored = recorded["positions"]
    prompt_end = recorded.get("prompt_end") or []
    errors = cache_errors(scored, reuse) + cache_errors(prompt_end, reuse, PROMPT_END_ARMS)
    scores = report_scores(scored, exact, recorded["detokenized"]) if scored else None
    checks = {"cache": not errors,
              "hit_equals_cold": equal_rows(scored, "hit"),
              "chunked_equals_cold": equal_rows(scored, "chunked"),
              "hit_vs_cold": bool(scores) and scores["all"]["gate_hit_vs_cold"]["passed"]}
    result = {}
    if prompt_end:
        result["prompt_end_scores"] = report_scores(prompt_end, exact, recorded["detokenized"],
                                                    PROMPT_END_ARMS)
        checks["prompt_end_hit_equals_cold"] = equal_rows(prompt_end, "hit")
        checks["prompt_end_hit_vs_cold"] = \
            result["prompt_end_scores"]["all"]["gate_hit_vs_cold"]["passed"]
    if recorded.get("media") is not None:
        checks["media"] = recorded["media"]["passed"]
    if recorded.get("acceptance") is not None:
        checks["acceptance"] = recorded["acceptance"]["passed"]
    return {"passed": all(checks.values()), "checks": checks, "reuse_tokens": reuse, "cache_errors": errors,
            "scores": scores, **result}


def synthetic_prompt(client, model, tokens, salt):
    """At least `tokens` token IDs of never-repeating records, cut to `tokens`."""
    ids = []
    while len(ids) < tokens:
        text = record_text(max(64, (tokens - len(ids)) * 6 // 5), salt + len(ids))
        ids += tokenize(client, model, [{"role": "user", "content": text}])
    return ids[:tokens]


def record_split(args, client, server, model, exact):
    """Hits at prompt-end anchors every `--split` tokens, over the reference
    texts and synthetic prompts: its own receipt, which `report
    --split-record` combines."""
    texts = [text for text in exact["texts"]
             if not args.texts or text["name"] in args.texts.split(",")]
    prompts = [(text["name"], text["prompt_token_ids"]) for text in texts]
    prompts += [(f"synthetic-{tokens}", synthetic_prompt(client, model, tokens, 7 + index))
                for index, tokens in enumerate(args.split_prompt_tokens)]
    entries = split_positions([ids for _, ids in prompts], args.split,
                              max_anchors=args.split_max_anchors)
    rows = score_split_passes(client, model, prompts, entries, args.run_id, args.concurrency)
    reuse = reuse_tokens(server)
    summary = summarize_split(rows, reuse)
    confirm_live(client, server)
    receipt("prefix-exact-split", server, {
        "run_id": args.run_id, "chunk": args.split, "reuse_tokens": reuse,
        "prefill_concurrency": args.concurrency, "continuation_concurrency": 1,
        "family": prefix_policy()["family"],
        "prompts": {name: len(ids) for name, ids in prompts}, "rows": rows, **summary})
    raise SystemExit(0 if summary["passed"] else 1)


def bind_server(args, server, path):
    """Bind the record to the live server's checkpoint when the gate pins the
    exact reference it scores against."""
    if verify_exact.pins_exact(path):
        try:
            bind_checkpoint(server, args.server_log, args.url)
        except ValueError as error:
            raise SystemExit(f"refused: {error}")


def unpinned(path):
    """A gate that pins its exact input scores other references as diagnostics only."""
    return verify_exact.exact_pinned() and not verify_exact.pins_exact(path)


def scored_concurrently(concurrency, score, entries):
    """score(entry) for every entry, in order, `concurrency` at a time. Each
    entry's requests carry their own salt and run in sequence."""
    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(score, entries))


def skipped_checks(recorded, exact):
    """The parts of the gate a record (or merged records) left out: a check
    the policy enables that did not run, or scored positions other than
    every position of every reference text, once each. A shard record
    (record --shard I/N) is judged against its own slice, and runs the media
    and acceptance checks only as shard 1."""
    policy = prefix_policy()
    shard = recorded.get("shard")
    select = (lambda entries: shard_entries(entries, shard["index"], shard["count"])) if shard else (lambda entries: entries)
    skipped = [name for name in ("media", "acceptance")
               if policy[name] and recorded.get(name) is None and (not shard or shard["index"] == 1)]
    block = policy["block_tokens"]
    if recorded["block_tokens"] != block:
        skipped.append("policy block tokens")
    expected = sorted((text["name"], position, anchor)
                      for text, position, anchor in select(positions(exact["texts"], block)))
    scored = sorted((row["text"], row["position"], row["anchor"])
                    for row in recorded["positions"])
    complete = scored == expected
    if policy["prompt_end"] or recorded.get("prompt_end"):
        expected = sorted((text["name"], position, anchor)
                          for text, position, anchor in select(prompt_end_positions(exact["texts"], block)))
        scored = sorted((row["text"], row["position"], row["anchor"])
                        for row in recorded.get("prompt_end") or [])
        complete = complete and scored == expected
    if not complete:
        skipped.append("full position set")
    return skipped


def serving_identity(recorded):
    """Canonical build and effective profile, independent of process restarts.

    Equal package, metadata and source identities name the build across
    restarts. Each receipt must still bind its own live startup nonce.
    """
    server = recorded.get("server") or {}
    live = (server.get("health") or {}).get("serving") or {}
    bound = server.get("checkpoint") or {}

    def require(condition, field):
        if not condition:
            raise SystemExit(f"invalid serving identity: {field}")

    require("rescored_from" not in server, "rescored_from")
    for key, size in (("checkpoint_identity", 64), ("package_identity", 64),
                      ("metadata_identity", 64), ("source_commit", 40), ("startup_nonce", 32)):
        value = live.get(key)
        require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value), key)
        if key != "source_commit":
            require(bound.get(key) == value, f"checkpoint.{key} differs from health")
    for key in ("bundle_id", "version"):
        require(isinstance(live.get(key), str) and live[key].strip(), key)
    names = live.get("served_model_names")
    require(isinstance(names, list) and names and all(isinstance(name, str) and name for name in names),
            "served_model_names")
    require(isinstance(server.get("model"), str) and server["model"] in names, "model")
    profile = live.get("profile")
    require(isinstance(profile, dict), "profile")
    for key in ("name", "kv_cache_dtype"):
        require(isinstance(profile.get(key), str) and profile[key].strip(), f"profile.{key}")
    for key in ("tensor_parallel_size", "max_sequences"):
        require(type(profile.get(key)) is int and profile[key] > 0, f"profile.{key}")
    require(type(profile.get("prefix_cache")) is bool, "profile.prefix_cache")
    spec = profile.get("speculation")
    require(isinstance(spec, dict), "profile.speculation")
    require(isinstance(spec.get("mode"), str) and spec["mode"].strip(), "profile.speculation.mode")
    require("drafter" in spec and (spec["drafter"] is None or
            isinstance(spec["drafter"], str) and spec["drafter"].strip()), "profile.speculation.drafter")
    require(type(spec.get("tokens")) is int and spec["tokens"] >= 0, "profile.speculation.tokens")
    return {key: value for key, value in live.items() if key != "startup_nonce"} | {"model": server["model"]}


def common_serving_identity(records):
    identities = [serving_identity(value) for value in records]
    first = identities[0]
    for identity in identities[1:]:
        for key in first.keys() | identity.keys():
            if first.get(key) != identity.get(key):
                raise SystemExit(f"the records differ in serving identity {key}")
    return first


def input_provenance(records, paths=None):
    return [{"run_id": value.get("run_id"), "server": value["server"],
             "reference_sha256": value.get("reference_sha256"),
             **({"path": str(paths[index]), "sha256": hashlib.sha256(
                 Path(paths[index]).read_bytes()).hexdigest()} if paths is not None else {})}
            for index, value in enumerate(records)]


def merged_records(records):
    """Records of one reference and policy, each over its own texts (record
    --texts), as one record. A check that ran in several records must pass
    in each."""
    identity = common_serving_identity(records)
    first = records[0]
    shards = [value.get("shard") for value in records]
    if any(shards):
        counts = {(shard or {}).get("count") for shard in shards}
        if None in counts or len(counts) != 1 or sorted(shard["index"] for shard in shards) != \
                list(range(1, counts.pop() + 1)):
            raise SystemExit("shard records must be every shard of one count, once each")
    for key in ("family", "reference_sha256", "block_tokens", "reuse_tokens", "top_logprobs"):
        if any(value.get(key) != first.get(key) for value in records):
            raise SystemExit(f"the records differ in {key}")
    merged = {**first, "shard": None, "serving_identity": identity, "inputs": input_provenance(records),
              "run_id": [value["run_id"] for value in records],
              "positions": [row for value in records for row in value["positions"]],
              "prompt_end": [row for value in records for row in value.get("prompt_end") or []],
              "detokenized": {key: text for value in records
                              for key, text in value["detokenized"].items()}}
    for name in ("media", "acceptance"):
        ran = [value[name] for value in records if value.get(name) is not None]
        if name == "media" and ran and all(row.get("status") == "not_applicable" for row in ran):
            if any(row != ran[0] for row in ran):
                raise SystemExit("the records differ in unserved media evidence")
            merged[name] = ran[0]
            continue
        merged[name] = ran[0] if len(ran) == 1 else \
            {"records": ran, "passed": all(value["passed"] for value in ran)} if ran else None
    return merged


def record(args):
    path = Path(args.exact or input_path("exact"))
    exact = load_exact(path)
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    bind_server(args, server, path)
    model = server["model"]
    policy = prefix_policy()
    reuse = reuse_tokens(server)
    media = policy["media"] and not args.skip_media
    shard = dict(zip(("index", "count"), parse_shard(args.shard))) if args.shard else None
    select = (lambda entries: shard_entries(entries, shard["index"], shard["count"])) if shard else (lambda entries: entries)
    if shard and shard["index"] > 1:
        # Shard 1 carries the media and acceptance checks.
        media, args.acceptance_prompts = False, 0
    media_reason = common.not_applicable(server, "image") if policy["media"] else None
    if args.split:
        record_split(args, client, server, model, exact)
    # Sized before the long scoring passes, so a prompt that cannot reach
    # the reuse unit refuses at once.
    media_prompts = media_cases(client, model, args.run_id, reuse) if media and not media_reason else None
    texts = [text for text in exact["texts"]
             if not args.texts or text["name"] in args.texts.split(",")]
    scored = scored_concurrently(args.concurrency, lambda entry: score_position(
        client, model, *entry, args.run_id), select(positions(texts, args.block_tokens, args.stride)))
    prompt_end_stride = (args.stride if policy["prompt_end"] else 0) if args.prompt_end_stride is None \
        else args.prompt_end_stride
    prompt_end = scored_concurrently(args.concurrency, lambda entry: score_prompt_end(
        client, model, *entry, args.run_id),
        select(prompt_end_positions(texts, args.block_tokens, prompt_end_stride)) if prompt_end_stride else [])
    top1_ids = [token for text in texts for token in text["exact"]["top1"]]
    acceptance = args.acceptance_prompts > 0 and policy["acceptance"]
    recorded = {
        "run_id": args.run_id, "shard": shard, "block_tokens": args.block_tokens, "reuse_tokens": reuse, "stride": args.stride,
        "prompt_end_stride": prompt_end_stride, "top_logprobs": TOP_LOGPROBS,
        "family": prefix_policy()["family"],
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "positions": scored, "prompt_end": prompt_end,
        "detokenized": detokenized(client, model, top1_ids),
        "media": ({"kind": "image", "status": "not_applicable", "passed": None, "reason": media_reason}
                  if media_reason else media_check(client, model, args.run_id, reuse, media_prompts) if media else None),
        "acceptance": acceptance_check(client, model, args.run_id, args.acceptance_prompts,
                                       args.acceptance_tokens, reuse) if acceptance else None}
    recorded["server"] = server
    result = evaluate(recorded, exact)
    if media_reason:
        recorded["not_applicable"] = [media_reason]
        result["passed"] = None
    skipped = skipped_checks(recorded, exact)
    if unpinned(path):
        skipped.append("pinned exact reference")
    confirm_live(client, server)
    receipt("prefix-exact", server, {**recorded, **result,
                                     "passed": result["passed"] and not skipped,
                                     "skipped": skipped})
    successful = (all(flag is True for key, flag in result["checks"].items() if key != "media")
                  if media_reason else result["passed"])
    raise SystemExit(0 if successful and not skipped else 1)


def load_record(path, groups):
    """Refuse missing anchors before scoring or checking coverage."""
    recorded = json.loads(Path(path).read_text())
    for group in groups:
        for index, row in enumerate(recorded.get(group) or []):
            if "anchor" not in row:
                raise SystemExit(f"refused: {path}: {group}[{index}] is missing anchor")
    return recorded


def report(args):
    path = Path(args.exact or input_path("exact"))
    exact = load_exact(path)
    records = [load_record(value, ("positions", "prompt_end")) for value in args.record]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(value["reference_sha256"] != digest for value in records):
        raise SystemExit("the record was scored against a different exact reference")
    split_records = [load_record(split, ("rows",)) for split in args.split_record]
    if verify_exact.pins_exact(path) and not all(bound_live(value.get("server", {}))
                                                 for value in (*records, *split_records)):
        raise SystemExit("a record's server was not bound live to the pinned checkpoint")
    identity = common_serving_identity([*records, *split_records])
    if args.slice:
        # One shard judged on its own slice: the quick suite's screen, never
        # the declared check, which needs every position.
        if len(records) != 1 or not records[0].get("shard") or split_records:
            raise SystemExit("--slice reports one shard record (record --shard I/N) alone")
        recorded = records[0]
    else:
        recorded = merged_records(records)
    result = evaluate(recorded, exact)
    result["skipped"] = skipped_checks(recorded, exact)
    result["passed"] = result["passed"] and not result["skipped"]
    if args.slice:
        result["slice"] = recorded["shard"]
    if split_records:
        for split in split_records:
            validate_policy(exact, split)
        splits = [summarize_split(split["rows"], record_reuse(split)) for split in split_records]
        result = {"main": result, "splits": splits,
                  "passed": result["passed"] and all(split["passed"] for split in splits)}
    if unpinned(path):
        print(f"{path} is not the pinned reference: diagnostic only", file=sys.stderr)
        result["passed"] = False
    if recorded.get("not_applicable"):
        if unpinned(path):
            raise SystemExit("unserved media evidence still requires the pinned exact reference")
        result.update(schema="chopinfer-qualification/v1", server=recorded["server"],
                      not_applicable=recorded["not_applicable"])
        import kinds
        kinds.inapplicable_reason(result, {"module": "prefix_exact"})
        if recorded.get("media") != {"kind": "image", "status": "not_applicable", "passed": None,
                                      "reason": recorded["not_applicable"][0]}:
            raise SystemExit("invalid unserved prefix media evidence")
        result["passed"] = None
    result["serving_identity"] = identity
    result["inputs"] = input_provenance([*records, *split_records], [*args.record, *args.split_record])
    json.dump(result, sys.stdout, indent=1)
    print()
    if result.get("not_applicable"):
        scored = kinds.rescore(kinds.of("prefix_exact"), result, None,
                              {"module": "prefix_exact", "rules": [{"path": ["passed"], "op": "eq", "value": True}]})
        raise SystemExit(0 if scored["status"] == "not_applicable" else 1)
    raise SystemExit(0 if result["passed"] else 1)


def references(args):
    """The file `serve --debug-reference-tokens` names: every reference text."""
    exact = load_exact(args.exact or input_path("exact"))
    Path(args.out).write_text(json.dumps(
        {"sequences": [text["prompt_token_ids"] for text in exact["texts"]]}) + "\n")


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("command", choices=("record", "report", "references"))
    parser.add_argument("--exact", default=None,
                        help="exact reference path (default: gate config)")
    parser.add_argument("--out", help="references: the file for --debug-reference-tokens")
    parser.add_argument("--run-id", default="prefix-exact-001",
                        help="record: salts every request, so reruns on one server share nothing")
    parser.add_argument("--block-tokens", type=int,
                        help="record: page size; overrides are diagnostic only")
    parser.add_argument("--stride", type=int, default=1, help="record: score every Nth position")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="record: positions scored at once (each under its own salt)")
    parser.add_argument("--prompt-end-stride", type=int,
                        help="record: score every Nth position of the prompt-end arm (0 skips "
                             "it; default --stride when the policy enables the arm, else 0)")
    parser.add_argument("--texts", help="record: comma-separated text names (partial run)")
    parser.add_argument("--shard", help="record: I/N, the I-th of N slices of every position (see report)")
    parser.add_argument("--slice", action="store_true",
                        help="report: judge one shard record on its own slice (the quick suite's screen)")
    parser.add_argument("--skip-media", action="store_true",
                        help="record: the server has no image encoder")
    parser.add_argument("--acceptance-prompts", type=int, default=12,
                        help="record: prompts for the draft acceptance check (0 skips it)")
    parser.add_argument("--acceptance-tokens", type=int, default=128)
    parser.add_argument("--record", action="append", default=[],
                        help="report: a prefix-exact receipt; several (record --texts) "
                             "must cover every position together")
    parser.add_argument("--server-log", type=Path,
                        help="record: the server's startup log; when given, it must be the live server's")
    parser.add_argument("--split", type=int,
                        help="record: only the split receipt, with anchors every this many tokens")
    parser.add_argument("--split-prompt-tokens", type=int, action="append", default=[],
                        help="record --split: add a synthetic prompt of this many tokens")
    parser.add_argument("--split-max-anchors", type=int,
                        help="record --split: at most this many anchors per prompt")
    parser.add_argument("--split-record", action="append", default=[],
                        help="report: a split receipt that must pass too")
    args = parser.parse_args()
    if args.command == "report":
        if not args.record:
            parser.error("report needs --record")
        report(args)
    elif args.command == "references":
        if not args.out:
            parser.error("references needs --out")
        references(args)
    else:
        if args.block_tokens is None:
            args.block_tokens = prefix_policy()["block_tokens"]
        if args.shard:
            try:
                parse_shard(args.shard)
            except ValueError as error:
                parser.error(str(error))
            if args.texts or args.split:
                parser.error("--shard slices every text's positions; it takes neither --texts nor --split")
        if args.block_tokens < 1 or args.stride < 1 or args.concurrency < 1 or args.acceptance_prompts < 0 \
                or (args.prompt_end_stride or 0) < 0 or (args.split is not None
                                                          and args.split < args.block_tokens):
            parser.error("block tokens and stride must be positive")
        record(args)


if __name__ == "__main__":
    main()
