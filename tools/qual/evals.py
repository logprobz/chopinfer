"""Paired downstream evals for any OpenAI-compatible endpoint.

Suites: the full GSM8K test split, a stratified MMLU-Pro subset and a
synthetic RULER-style long-context set (multi-key needle, multi-value needle
and variable tracking at 32K and 128K). Every arm gets byte-identical request
bodies with greedy decoding and the same token limits.

  evals.py fetch                   download and pin the datasets
  evals.py run --arm NAME ...      run one arm (resumable)
  evals.py compare [--run-id ID] --arms A B [--receipt ARM=RECEIPT ...]
          [--stored ARM=RECEIPT ...] [--bound]
                                   paired scores, McNemar exact test, receipt

`run --server-log LOG` binds the live server to the gate's checkpoint pin
and that startup log, and confirms at the end that the same process answered
every request. A run reuses a cached answer only when a capture confirmed
under the same producer sealed it and its row names that producer
(checkpoint, package, metadata, version and source identities, serving
profile and model options, served model and its modalities, the checkpoint
pin a bound run was bound to, and request settings) and request body;
every other item is asked again, and the receipt counts the reused rows
(`resumed`).

An arm's receipt records the SHA-256 of its results file (`results_sha256`),
and compare refuses an arm whose results no longer have it. compare finds
an arm's receipt among the receipts by `--run-id`, its results in that
run's directory. `--receipt ARM=RECEIPT` names an arm's receipt instead,
such as the candidate arm an earlier check run recorded under its own run
ID, or an arm recorded in an earlier gate run (a pre-recorded reference
arm): it must be that arm's receipt, recording its run ID, its results file
and their digest, and compare reads the results file it names, held to that
digest. `--stored ARM=RECEIPT` names an arm of an earlier run, such as the
correctness runner's stored last good: its receipt and the results file the
receipt names, held to the receipt's digest. `--run-id` is needed only for
an arm neither named nor stored. `compare --bound`, a qualification
comparison, refuses a stored arm, requires the digest of every arm, found or
named, and that both arms' servers report one version and source commit
(their packages may differ, as two checkpoints' do), and records `bound:
true`; without it, an older receipt without a digest is still compared.
The receipt records the number of items both arms answered (`items`, the
paired coverage a gate admits) and the run ID each arm's receipt names
(`arm_runs`).
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
import urllib.parse
import urllib.request

from common import (startup, thinking_kwargs, qualification_config, receipt_root, Client, receipt, bind_checkpoint,
                    checkpoint_pin, confirm_live)

SIGNIFICANCE = 0.05
from gsm8k import extract as extract_number, parse as parse_number


CACHE = Path(os.environ.get("CHOPINFER_EVALS_DIR")
             or Path(os.environ.get("CHOPINFER_DATA", Path.home() / "chopinfer-data")) / "evals")
SOURCES = {
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test"},
    "mmlu_pro": {"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test"},
    "wikitext": {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "test"},
}
ROWS_API = "https://datasets-server.huggingface.co/rows"
HUB_API = "https://huggingface.co/api/datasets/"
SUITES = ("gsm8k", "mmlu_pro", "longctx")
SEED = 20260924
MMLU_PRO_SUBSET = 1200
GSM8K_INSTRUCTION = "Solve the math problem. Put your final numeric answer on its own line as #### NUMBER."
REASONING_TOKENS = 8192
LONG_LENGTHS = (32768, 131072)
LONG_TASKS = ("multikey", "multivalue", "vartrack")
LONG_ITEMS = 50
LONG_TOKENS = 128
LONG_FILL = 0.94
LETTERS = "ABCDEFGHIJ"
WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo",
         "juniper", "kestrel", "lagoon", "marble", "nectar", "orchid", "pewter", "quartz",
         "raven", "saffron", "tundra", "umber", "violet", "willow", "xenon", "yarrow", "zephyr")


def fetch_rows(source):
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({**{k: source[k] for k in ("dataset", "config", "split")},
                                        "offset": offset, "length": 100})
        # The rows API rate-limits bursts, so pace pages and back off on errors.
        for attempt in range(8):
            try:
                time.sleep(1)
                with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
                    page = json.load(response)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(120, 5 * 2 ** attempt))
        total = page["num_rows_total"]
        rows.extend(row["row"] for row in page["rows"])
        offset += len(page["rows"])
        if not page["rows"]:
            raise RuntimeError(f"{source['dataset']}: empty page at offset {offset}")
    return rows


def fetch():
    """Download each dataset once and record its hub revision and sha256."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for name, source in SOURCES.items():
        target = CACHE / f"{name}.json"
        if name in manifest and target.exists():
            continue
        with urllib.request.urlopen(HUB_API + source["dataset"], timeout=60) as response:
            revision = json.load(response)["sha"]
        rows = fetch_rows(source)
        data = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        target.write_bytes(data)
        manifest[name] = {**source, "revision": revision, "rows": len(rows),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "via": ROWS_API,
                          "fetched_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"{name}: {len(rows)} rows, revision {revision}")
    return manifest


def load(name, manifest):
    data = (CACHE / f"{name}.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[name]["sha256"]:
        raise ValueError(f"cached {name} does not match its manifest sha256")
    return json.loads(data)


def chat(messages, max_tokens, thinking):
    return {"messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": thinking_kwargs(thinking)}


def gsm8k_items(rows):
    items = []
    for index, row in enumerate(rows):
        items.append({"id": f"gsm8k/{index}", "suite": "gsm8k", "group": "gsm8k",
                      "expected": extract_number(row["answer"]),
                      "body": chat([{"role": "system", "content": GSM8K_INSTRUCTION},
                                    {"role": "user", "content": row["question"]}],
                                   REASONING_TOKENS, True)})
    return items


def mmlu_pro_subset(rows):
    """Proportional allocation per category (largest remainder), seeded sample within each."""
    categories = {}
    for row in rows:
        categories.setdefault(row["category"], []).append(row)
    quota = {name: MMLU_PRO_SUBSET * len(group) / len(rows) for name, group in categories.items()}
    counts = {name: int(value) for name, value in quota.items()}
    remainder = sorted(quota, key=lambda name: (counts[name] - quota[name], name))
    for name in remainder[:MMLU_PRO_SUBSET - sum(counts.values())]:
        counts[name] += 1
    selected = []
    for name in sorted(categories):
        group = sorted(categories[name], key=lambda row: row["question_id"])
        selected.extend(random.Random(f"{SEED}:{name}").sample(group, counts[name]))
    return sorted(selected, key=lambda row: row["question_id"])


def mmlu_pro_items(rows):
    items = []
    for row in mmlu_pro_subset(rows):
        options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(row["options"]))
        prompt = (f"The following is a multiple choice question about {row['category']}. Think "
                  "step by step and then finish your answer with \"The answer is (X)\" where X "
                  f"is the correct letter choice.\n\nQuestion: {row['question']}\nOptions:\n{options}")
        items.append({"id": f"mmlu_pro/{row['question_id']}", "suite": "mmlu_pro",
                      "group": row["category"], "expected": row["answer"],
                      "body": chat([{"role": "user", "content": prompt}], REASONING_TOKENS, True)})
    return items


def extract_choice(text):
    """MMLU-Pro's reference extraction order; unparsed answers count as wrong."""
    text = text or ""
    for pattern in (r"answer is \(?([A-J])\)?", r".*[aA]nswer:\s*\(?([A-J])\)?"):
        match = re.findall(pattern, text)
        if match:
            return match[-1]
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def paragraphs(rows):
    text = "".join(row["text"] for row in rows)
    for token, replacement in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
        text = text.replace(token, replacement)
    return [line.strip() for line in text.split("\n")
            if line.strip() and not line.strip().startswith("=")]


def haystack(corpus, rng, characters):
    start = rng.randrange(len(corpus))
    chosen, total = [], 0
    while total < characters:
        paragraph = corpus[(start + len(chosen)) % len(corpus)]
        chosen.append(paragraph)
        total += len(paragraph) + 2
    return chosen


def insert(chosen, rng, sentences):
    """Insert sentences at seeded depths, keeping their order."""
    depths = sorted(rng.randrange(len(chosen) + 1) for _ in sentences)
    out = list(chosen)
    for offset, (depth, sentence) in enumerate(zip(depths, sentences)):
        out.insert(depth + offset, sentence)
    return "\n\n".join(out)


def variable_names(rng, count):
    names = set()
    while len(names) < count:
        names.add("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)))
    names = sorted(names)
    rng.shuffle(names)
    return names


def long_item(corpus, task, length, index):
    rng = random.Random(f"{SEED}:{task}:{length}:{index}")
    chosen = haystack(corpus, rng, int(length * LONG_FILL * qualification_config()["qualification"]["chars_per_token"]))
    item = {"id": f"longctx/{task}-{length // 1024}k/{index}", "suite": "longctx",
            "group": f"{task}-{length // 1024}k"}
    if task == "multikey":
        keys = [f"{a}-{b}" for a, b in (rng.sample(WORDS, 2) for _ in range(4))]
        values = rng.sample(range(1_000_000, 10_000_000), 4)
        target = rng.randrange(4)
        needles = [f"One of the special magic numbers for {key} is: {value}."
                   for key, value in zip(keys, values)]
        rng.shuffle(needles)
        prompt = ("Some special magic numbers are hidden within the following text. Make sure to "
                  "memorize them. I will quiz you about the numbers afterwards.\n\n"
                  f"{insert(chosen, rng, needles)}\n\nWhat is the special magic number for "
                  f"{keys[target]} mentioned in the provided text? Answer with the number only.")
        item["expected"] = {"all": [str(values[target])],
                            "none": [str(v) for i, v in enumerate(values) if i != target]}
    elif task == "multivalue":
        key = "-".join(rng.sample(WORDS, 2))
        values = rng.sample(range(1_000_000, 10_000_000), 4)
        needles = [f"One of the special magic numbers for {key} is: {value}." for value in values]
        prompt = ("Some special magic numbers are hidden within the following text. Make sure to "
                  "memorize them. I will quiz you about the numbers afterwards.\n\n"
                  f"{insert(chosen, rng, needles)}\n\nWhat are all the special magic numbers for "
                  f"{key} mentioned in the provided text? List every one of them.")
        item["expected"] = {"all": [str(v) for v in values], "none": []}
    else:
        names = variable_names(rng, 10)
        values = rng.sample(range(10_000, 100_000), 2)
        chains = []
        for chain, value in zip((names[:5], names[5:]), values):
            chains.append([f"VAR {chain[0]} = {value}"]
                          + [f"VAR {b} = VAR {a}" for a, b in zip(chain, chain[1:])])
        # Interleave the two chains while keeping each chain's order.
        statements, cursors = [], [0, 0]
        while cursors != [5, 5]:
            pick = rng.choice([c for c in (0, 1) if cursors[c] < 5])
            statements.append(chains[pick][cursors[pick]])
            cursors[pick] += 1
        prompt = ("Memorize and track the chain of variable assignments hidden in the following "
                  f"text.\n\n{insert(chosen, rng, statements)}\n\nQuestion: Find all variables "
                  f"that are assigned the value {values[0]} in the text above. Answer with the "
                  "variable names separated by spaces.")
        item["expected"] = {"all": names[:5], "none": names[5:]}
    item["body"] = chat([{"role": "user", "content": prompt}], LONG_TOKENS, False)
    return item


def long_items(rows):
    corpus = paragraphs(rows)
    return [long_item(corpus, task, length, index)
            for length in LONG_LENGTHS for task in LONG_TASKS for index in range(LONG_ITEMS)]


def build_items(manifest, suites):
    builders = {"gsm8k": (gsm8k_items, "gsm8k"), "mmlu_pro": (mmlu_pro_items, "mmlu_pro"),
                "longctx": (long_items, "wikitext")}
    items = []
    for suite in suites:
        builder, source = builders[suite]
        items.extend(builder(load(source, manifest)))
    return items


def score(item, content):
    if item["suite"] == "gsm8k":
        extracted = extract_number(content)
        return extracted, extracted is not None and extracted == item["expected"]
    if item["suite"] == "mmlu_pro":
        extracted = extract_choice(content)
        return extracted, extracted == item["expected"]
    words = set(re.findall(r"[A-Za-z0-9]+", re.sub(r"(?<=\d),(?=\d)", "", content or "")))
    expected = item["expected"]
    return None, all(v in words for v in expected["all"]) and not any(v in words for v in expected["none"])


def body_digest(item):
    return hashlib.sha256(json.dumps(item["body"], sort_keys=True).encode()).hexdigest()


def workload_digest(items):
    return hashlib.sha256(json.dumps([[item["id"], item["body"]] for item in items],
                                     sort_keys=True).encode()).hexdigest()


def run_dir(run_id):
    return CACHE / "runs" / run_id


def read_results(path):
    results = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            results[row["id"]] = row
    return results


def seal_segments(seal):
    """The segments `seal` records, oldest first: each confirmed capture's
    results length and SHA-256 when it confirmed, and the producer it
    confirmed under. A seal from before segments names no producer."""
    if not seal.exists():
        return []
    sealed = json.loads(seal.read_text())
    if "segments" in sealed:
        return sealed["segments"]
    return [{"bytes": sealed["bytes"], "sha256": sealed["sha256"], "producer": None}]


def sealed_results(path, made_by):
    """The latest rows of `path` that a capture confirmed under `made_by`
    sealed. A capture appends a segment to `<arm>.seal.json` once the live
    server is confirmed, recording the producer, binding included, it
    confirmed under; so an unbound capture's rows never serve a bound run,
    rows a refused or unfinished capture appended are set aside to
    `<arm>.unsealed.jsonl` and asked again, and sealed bytes that changed
    refuse the run."""
    seal = path.with_suffix(".seal.json")
    data = path.read_bytes() if path.exists() else b""
    latest, start = {}, 0
    for segment in seal_segments(seal):
        end = segment["bytes"]
        if end < start or len(data) < end or hashlib.sha256(data[:end]).hexdigest() != segment["sha256"]:
            raise SystemExit(f"refused: {path} is not the results its validated captures sealed in {seal}; "
                             "move both aside or use another --run-id")
        for line in data[start:end].decode().splitlines():
            row = json.loads(line)
            latest[row["id"]] = row, segment["producer"]
        start = end
    if tail := data[start:]:
        with path.with_suffix(".unsealed.jsonl").open("ab") as aside:
            aside.write(tail)
        path.write_bytes(data[:start])
        rows = tail.count(b"\n")
        print(f"{path}: set aside {rows} rows no validated capture sealed", flush=True)
    return {key: row for key, (row, confirmed) in latest.items() if confirmed == made_by}


def seal_results(path, made_by, server):
    """Seal the rows `path` gained as confirmed under `made_by`, recording
    the bound process's evidence (`server`'s checkpoint record, if bound),
    and return the file's digest."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    seal = path.with_suffix(".seal.json")
    segments = seal_segments(seal)
    if not segments or segments[-1]["bytes"] != len(data):
        segments.append({"bytes": len(data), "sha256": digest, "producer": made_by,
                         "bound": server.get("checkpoint")})
        seal.write_text(json.dumps({"segments": segments}) + "\n")
    return digest


# The serving identity fields a cached answer must share with the server it is reused for:
# the checkpoint and build, then the serving configuration (profile: tensor and
# context parallelism, speculation, KV cache dtype, prefix cache and sequence
# capacity; resolved model options).
PRODUCER = ("checkpoint_identity", "package_identity", "metadata_identity", "version", "source_commit",
            "profile", "model_options")


def served_modalities(client, model):
    """The modalities /v1/models lists for the served model."""
    models, _ = client.request("/v1/models")
    return next((row.get("modalities") for row in models["data"] if row["id"] == model), None)


# The checkpoint pin fields a bound run's answers are bound to.
BINDING = ("repository", "revision", "served_model_name", "source_lock_sha256", "checkpoint_identity")


def producer(server, modalities, binding, args):
    """What an answer depends on beyond its request body (which carries its
    sampling settings): the serving process's checkpoint, build and serving
    configuration, the served model and its `modalities`, the checkpoint pin
    the run was bound and confirmed under (`binding`, None when unbound) and
    the run's request settings."""
    serving = (server.get("health") or {}).get("serving") or {}
    return {**{key: serving.get(key) for key in PRODUCER}, "model": server["model"], "modalities": modalities,
            "binding": binding, "top_logprobs": args.top_logprobs, "mechanics_only": bool(args.mechanics_only)}


def run(args):
    manifest = fetch()
    items = build_items(manifest, args.suites)
    if args.long_lengths:
        items = [item for item in items if item["suite"] != "longctx"
                 or item["group"].rsplit("-", 1)[1] in args.long_lengths]
    if args.limit:
        per_group = Counter()
        kept = []
        for item in items:
            key = item["suite"] if item["suite"] != "longctx" else item["group"]
            if per_group[key] < args.limit:
                per_group[key] += 1
                kept.append(item)
        items = kept
    client = Client(args.url, args.timeout, retries=1)
    server = client.identity(args.model)
    binding = None
    if getattr(args, "server_log", None):
        bind_checkpoint(server, args.server_log, args.url)
        pin = checkpoint_pin()
        binding = {key: pin.get(key) for key in BINDING}
    made_by = producer(server, served_modalities(client, server["model"]), binding, args)
    bodies = {item["id"]: body_digest(item) for item in items}
    directory = run_dir(args.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.arm}.jsonl"
    # Only answers sealed under this producer, to the same request bodies, are reused.
    done = {key: row for key, row in sealed_results(path, made_by).items()
            if "error" not in row and row.get("producer") == made_by and row.get("body_sha256") == bodies.get(key)}
    lock = threading.Lock()
    pending = [item for item in items if item["id"] not in done]
    print(f"{args.arm}: {len(items)} items, {len(pending)} pending, {len(items) - len(pending)} reused", flush=True)

    def request(item):
        body = {"model": server["model"], **item["body"]}
        if args.mechanics_only:
            body["max_tokens"] = 16
        if args.top_logprobs:
            body.update({"logprobs": True, "top_logprobs": args.top_logprobs})
        try:
            answer, latency = client.request("/v1/chat/completions", body)
            choice = answer["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            extracted, correct = score(item, content)
            row = {"id": item["id"], "suite": item["suite"], "group": item["group"],
                   "body_sha256": body_digest(item), "correct": correct and choice["finish_reason"] != "length",
                   "extracted": extracted, "finish_reason": choice["finish_reason"],
                   "prompt_tokens": (answer.get("usage") or {}).get("prompt_tokens"),
                   "completion_tokens": (answer.get("usage") or {}).get("completion_tokens"),
                   "latency_seconds": latency, "content": content,
                   "reasoning": message.get("reasoning_content") or message.get("reasoning"),
                   "producer": made_by}
            if item["suite"] == "gsm8k":
                row["rule"] = parse_number(content)[1]
            if args.top_logprobs:
                row["logprobs"] = (choice.get("logprobs") or {}).get("content")
        except Exception as error:
            row = {"id": item["id"], "suite": item["suite"], "group": item["group"],
                   "error": str(error), "producer": made_by}
        with lock:
            with path.open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    long_pending = [item for item in pending if item["suite"] == "longctx"]
    short_pending = [item for item in pending if item["suite"] != "longctx"]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(request, short_pending))
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.long_concurrency)) as pool:
        list(pool.map(request, long_pending))
    confirm_live(client, server)
    results_sha256 = seal_results(path, made_by, server)
    results = read_results(path)
    rows = [results.get(item["id"], {"id": item["id"], "error": "missing"}) for item in items]
    rows = [row if row.get("producer") == made_by else {"id": row["id"], "error": "not answered by this server"}
            for row in rows]
    errors = [{"id": row["id"], "error": row["error"]} for row in rows if "error" in row]
    suites = {}
    for row in rows:
        if "error" not in row:
            entry = suites.setdefault(row["suite"], {"count": 0, "correct": 0, "length": 0})
            entry["count"] += 1
            if row["suite"] == "gsm8k":
                entry["boxed"] = entry.get("boxed", 0) + (row.get("rule") == "boxed")
            entry["correct"] += row["correct"]
            entry["length"] += row["finish_reason"] == "length"
    for entry in suites.values():
        entry["accuracy"] = entry["correct"] / entry["count"]
    # Per-item latencies stay in the results file, whose answers can come from an earlier run, so no host load
    # samples could bound them: the receipt's timing reads informational.
    receipt(f"evals-{args.arm}", server, {
        "passed": not errors, "skipped": ["model quality"] if args.mechanics_only else [],
        "run_id": args.run_id, "arm": args.arm, "suites": suites, "errors": errors[:50],
        "error_count": len(errors), "items": len(items), "limit": args.limit,
        "concurrency": args.concurrency, "long_concurrency": args.long_concurrency,
        "top_logprobs": args.top_logprobs, "datasets": manifest,
        "workload_sha256": workload_digest(items), "results": str(path),
        "producer": made_by, "resumed": len(items) - len(pending),
        "results_sha256": results_sha256}, args.mechanics_only)
    raise SystemExit(0 if not errors else 1)


def mcnemar(only_a, only_b):
    """Exact two-sided binomial test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(correct, count):
    if count == 0:
        return [None, None]
    z = 1.959964
    p = correct / count
    centre = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [centre - half, centre + half]


def token_margin(row, part, offset):
    """Top-2 logprob margin of the output token covering character `offset` of
    `part` ("reasoning" or "content"). vLLM and current Chopinfer list every
    generated token including <think>...</think>; Chopinfer receipts before
    the logprob parity change list content tokens only, so their reasoning
    margins are unavailable."""
    tokens = row.get("logprobs") or []
    texts = [token["token"] for token in tokens]
    if texts and texts[0] == "<think>":
        close = texts.index("</think>") if "</think>" in texts else len(texts)
        span = tokens[1:close] if part == "reasoning" else tokens[close + 1:]
    elif part == "content":
        span = tokens
    else:
        return None
    # The texts first differ at `offset`, so the diverging token either ends
    # exactly there (the other arm chose a longer token) or covers it.
    margins, position = [], 0
    for token in span:
        position += len(token["token"])
        if position >= offset:
            top = sorted((t["logprob"] for t in token.get("top_logprobs") or []), reverse=True)
            if len(top) >= 2:
                margins.append(top[0] - top[1])
            if position > offset:
                break
    return min(margins, default=None)


def divergence(row_a, row_b):
    """First character where two greedy outputs differ, in the reasoning or the
    content, and each arm's top-2 margin at that token where its logprobs
    cover it. A margin near zero means a near-tie flipped."""
    for part in ("reasoning", "content"):
        a, b = row_a.get(part) or "", row_b.get(part) or ""
        if a != b:
            offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return {"part": part, "offset": offset, "text": [a[offset:offset + 20], b[offset:offset + 20]],
                    "margins": [token_margin(row_a, part, offset), token_margin(row_b, part, offset)]}
    return None


def paired(rows_a, rows_b):
    ids = [key for key in rows_a if key in rows_b]
    a = sum(rows_a[key]["correct"] for key in ids)
    b = sum(rows_b[key]["correct"] for key in ids)
    only_a = [key for key in ids if rows_a[key]["correct"] and not rows_b[key]["correct"]]
    only_b = [key for key in ids if rows_b[key]["correct"] and not rows_a[key]["correct"]]
    identical = [key for key in ids
                 if (rows_a[key]["content"], rows_a[key].get("reasoning"))
                 == (rows_b[key]["content"], rows_b[key].get("reasoning"))]
    def only(test):
        return [sum(test(rows_a[k]) and not test(rows_b[k]) for k in ids),
                sum(test(rows_b[k]) and not test(rows_a[k]) for k in ids)]

    return {"count": len(ids), "correct": [a, b],
            "accuracy": [a / len(ids), b / len(ids)] if ids else [None, None],
            "accuracy_ci95": [wilson(a, len(ids)), wilson(b, len(ids))],
            "length_finishes": [sum(rows_a[k]["finish_reason"] == "length" for k in ids),
                                sum(rows_b[k]["finish_reason"] == "length" for k in ids)],
            "only_first_correct": len(only_a), "only_second_correct": len(only_b),
            "mcnemar_p": mcnemar(len(only_a), len(only_b)),
            "correctness_agreement": (len(ids) - len(only_a) - len(only_b)) / len(ids) if ids else None,
            "identical_outputs": len(identical),
            # Items where only one arm stopped for length, or gave no parsable answer.
            "only_length": only(lambda row: row["finish_reason"] == "length"),
            "only_unparsed": only(lambda row: row["suite"] != "longctx" and row["extracted"] is None),
            "prompt_token_mismatch_ids": [k for k in ids if rows_a[k].get("prompt_tokens")
                                          != rows_b[k].get("prompt_tokens")],
            "only_first_correct_ids": only_a, "only_second_correct_ids": only_b,
            "discordant_divergence": {k: divergence(rows_a[k], rows_b[k]) for k in only_a + only_b}}


def flags(report, first, second):
    """Differences worth a root cause: significant or one-sided discordance,
    one-sided length stops or unparsed answers, and prompt token mismatches."""
    found = []
    for name, value in {**report["suites"], **report["groups"]}.items():
        pair = [value["only_first_correct"], value["only_second_correct"]]
        loser = first if pair[1] > pair[0] else second
        if value["mcnemar_p"] < SIGNIFICANCE:
            found.append(f"{name}: {loser} loses significantly ({pair[0]} vs {pair[1]} "
                         f"discordant, p={value['mcnemar_p']:.3g})")
        elif min(pair) == 0 and max(pair) >= 4:
            found.append(f"{name}: one-sided discordance, {loser} loses all {max(pair)}")
        for field, label in (("only_length", "length stops"), ("only_unparsed", "unparsed answers")):
            counts = value[field]
            if mcnemar(*counts) < SIGNIFICANCE:
                found.append(f"{name}: {label} differ ({first} {counts[0]}, {second} {counts[1]})")
        if name in report["suites"] and value["prompt_token_mismatch_ids"]:
            found.append(f"{name}: {len(value['prompt_token_mismatch_ids'])} items with different "
                         "prompt token counts")
    return found


def named_receipts(values, arms, option="--receipt"):
    """{arm: receipt path} of `--receipt ARM=RECEIPT` (or `--stored`) values."""
    named = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in arms or arm in named:
            raise SystemExit(f"{option} needs ARM=RECEIPT naming one compared arm once: {value}")
        named[arm] = Path(path)
    return named


# The serving identity fields of the code that served an arm: two
# checkpoints' arms may have different packages, never different code.
BUILD = ("version", "source_commit")


def build(value):
    """The code identity an arm receipt's server reported."""
    serving = ((value.get("server") or {}).get("health") or {}).get("serving") or {}
    return {key: serving.get(key) for key in BUILD}


def compare(args):
    first, second = args.arms
    named = named_receipts(args.receipt, args.arms)
    stored = named_receipts(args.stored, args.arms, "--stored")
    if both := named.keys() & stored.keys():
        raise SystemExit(f"--receipt and --stored name {sorted(both)}")
    if args.bound and stored:
        raise SystemExit("--bound compares arms of this build, not a stored arm")
    arm_receipts, results_paths = {}, {}
    for arm in args.arms:
        if arm in stored:
            # A stored arm of an earlier run: its receipt and the results
            # file it names, held to the receipt's digest below.
            value = json.loads(stored[arm].read_text())
            if not str(value.get("check", "")).startswith("evals-") or not value.get("results_sha256") \
                    or not value.get("results"):
                raise SystemExit(f"{stored[arm]} is not an evals arm's receipt with its results and digest")
            arm_receipts[arm] = value
            results_paths[arm] = Path(value["results"])
            continue
        if arm in named:
            # A named arm of any run: bound by the run ID, the results file
            # and the digest its own receipt records.
            value = json.loads(named[arm].read_text())
            if (value.get("check"), value.get("arm")) != (f"evals-{arm}", arm) \
                    or not isinstance(value.get("run_id"), str) or not value["run_id"] \
                    or not value.get("results_sha256") or not value.get("results"):
                raise SystemExit(f"{named[arm]} is not the {arm} arm's receipt with its results and digest")
            arm_receipts[arm] = value
            results_paths[arm] = Path(value["results"])
            continue
        if args.run_id is None:
            raise SystemExit(f"--run-id is needed to find the {arm} arm's receipt")
        results_paths[arm] = run_dir(args.run_id) / f"{arm}.jsonl"
        candidates = sorted(receipt_root().glob(f"*-evals-{arm}.json"))
        matching = [json.loads(path.read_text()) for path in candidates]
        matching = [r for r in matching if r.get("run_id") == args.run_id]
        if not matching:
            raise SystemExit(f"no evals receipt for arm {arm} and run {args.run_id}")
        arm_receipts[arm] = matching[-1]
    for arm, value in arm_receipts.items():
        results = results_paths[arm]
        recorded = value.get("results_sha256")
        if args.bound and not recorded:
            raise SystemExit(f"the {arm} arm's receipt records no results digest: --bound needs both")
        if recorded and (not results.is_file() or hashlib.sha256(results.read_bytes()).hexdigest() != recorded):
            raise SystemExit(f"{results} is not the results the {arm} arm's receipt recorded")
    builds = {arm: build(value) for arm, value in arm_receipts.items()}
    if args.bound and (None in builds[first].values() or builds[first] != builds[second]):
        raise SystemExit(f"--bound compares arms of one build: {builds}")
    rows = {arm: {key: row for key, row in read_results(results_paths[arm]).items()
                  if "error" not in row} for arm in args.arms}
    # Arms may cover different subsets (--suites, --limit); score the items
    # both ran, and refuse any whose request bodies differ.
    common = rows[first].keys() & rows[second].keys()
    if changed := sorted(k for k in common
                         if rows[first][k]["body_sha256"] != rows[second][k]["body_sha256"]):
        raise SystemExit(f"{len(changed)} items have different request bodies, e.g. {changed[:5]}")
    unpaired = {arm: len(rows[arm].keys() - common) for arm in args.arms}
    rows = {arm: {k: rows[arm][k] for k in common} for arm in args.arms}
    digests = {arm: r["workload_sha256"] for arm, r in arm_receipts.items()}
    suites = sorted({row["suite"] for row in rows[first].values()})
    report = {"suites": {}, "groups": {}}
    for suite in suites:
        pick = lambda arm: {k: r for k, r in rows[arm].items() if r["suite"] == suite}
        report["suites"][suite] = paired(pick(first), pick(second))
        for group in sorted({r["group"] for r in rows[first].values() if r["suite"] == suite}):
            pick_group = lambda arm: {k: r for k, r in rows[arm].items()
                                      if r["suite"] == suite and r["group"] == group}
            report["groups"][f"{suite}/{group}"] = paired(pick_group(first), pick_group(second))
    worse = [suite for suite, row in report["suites"].items()
             if row["only_second_correct"] > row["only_first_correct"] and row["mcnemar_p"] < SIGNIFICANCE]
    runs = {arm: arm_receipts[arm].get("run_id") for arm in args.arms}
    identity = {"arms": {arm: arm_receipts[arm]["server"] for arm in args.arms},
                "run_id": args.run_id, "arm_runs": runs, "workload_sha256": digests}
    result = {"passed": not worse, "first_significantly_worse": worse, "arms": list(args.arms),
              "flags": flags(report, first, second), "items": len(common), "unpaired_items": unpaired,
              "named_receipts": {arm: str(path) for arm, path in named.items()},
              "stored_receipts": {arm: str(path) for arm, path in stored.items()}, "bound": args.bound,
              "builds": builds,
              "run_id": args.run_id, "arm_runs": runs, "datasets": arm_receipts[first]["datasets"],
              "settings": {arm: {k: arm_receipts[arm][k] for k in
                                 ("concurrency", "long_concurrency", "top_logprobs", "limit", "at")}
                           for arm in args.arms},
              **report}
    # Compares saved receipts. No timed work, so no host load samples: the receipt's timing reads
    # informational.
    path = receipt("evals-compare", identity, result)
    path.with_suffix(".md").write_text(markdown(result))
    raise SystemExit(0 if result["passed"] else 1)


def markdown(result):
    first, second = result["arms"]
    runs = sorted({str(run) for run in result["arm_runs"].values()})
    lines = [f"# Evals: {first} vs {second} (run{'s' if len(runs) > 1 else ''} {', '.join(runs)})", "",
             "Greedy decoding, identical request bodies. McNemar p is the exact two-sided "
             "binomial test on the discordant pairs.", "",
             f"| suite | n | {first} | {second} | only {first} | only {second} | McNemar p "
             "| same correctness | identical outputs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name, value):
        accuracy = [f"{a:.2%}" if a is not None else "-" for a in value["accuracy"]]
        return (f"| {name} | {value['count']} | {accuracy[0]} | {accuracy[1]} | "
                f"{value['only_first_correct']} | {value['only_second_correct']} | "
                f"{value['mcnemar_p']:.3g} | {value['correctness_agreement']:.2%} | "
                f"{value['identical_outputs']} |")

    lines += [row(name, value) for name, value in result["suites"].items()]
    lines += ["", "Groups:", "", lines[4], lines[5]]
    lines += [row(name, value) for name, value in result["groups"].items()]
    lines += ["", f"First arm significantly worse (p < {SIGNIFICANCE}): "
              f"{', '.join(result['first_significantly_worse']) or 'none'}.", "", "Flags:", ""]
    lines += [f"- {flag}" for flag in result["flags"]] or ["- none"]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    runner = commands.add_parser("run")
    runner.add_argument("--url", default="http://127.0.0.1:8101")
    runner.add_argument("--model", help="Defaults to the first model from /v1/models")
    runner.add_argument("--timeout", type=float, default=3600)
    runner.add_argument("--arm", required=True)
    runner.add_argument("--run-id", required=True)
    runner.add_argument("--suites", type=lambda v: v.split(","), default=list(SUITES))
    runner.add_argument("--long-lengths", type=lambda v: v.split(","),
                        help="Long-context lengths to run, e.g. 32k (default all)")
    runner.add_argument("--limit", type=int, help="Items per suite (per task and length for longctx)")
    runner.add_argument("--concurrency", type=int, default=32)
    runner.add_argument("--long-concurrency", type=int, default=4)
    runner.add_argument("--top-logprobs", type=int, default=0)
    runner.add_argument("--mechanics-only", action="store_true")
    runner.add_argument("--server-log", type=Path, help="bind the live server to the pin and this startup log")
    comparer = commands.add_parser("compare")
    comparer.add_argument("--run-id", help="the run whose receipts and results supply each arm neither "
                                           "named nor stored")
    comparer.add_argument("--arms", nargs=2, required=True)
    comparer.add_argument("--receipt", action="append", default=[], metavar="ARM=RECEIPT",
                          help="an arm's receipt of any run, instead of the receipts directory's, read "
                               "with the results file it names")
    comparer.add_argument("--stored", action="append", default=[], metavar="ARM=RECEIPT",
                          help="an arm of an earlier run: its receipt, read with the results file it names")
    comparer.add_argument("--bound", action="store_true",
                          help="require every arm's results digest (a qualification comparison)")
    args = parser.parse_args()
    startup(model_facts=args.command == "run")
    if args.command == "fetch":
        fetch()
    elif args.command == "run":
        if unknown := set(args.suites) - set(SUITES):
            parser.error(f"unknown suites: {sorted(unknown)}")
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
