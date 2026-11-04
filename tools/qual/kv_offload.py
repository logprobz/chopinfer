"""Record preemption swap and resume; Rust judges paired off/swap records.

The correctness runner owns the four servers. This module only captures a
running arm, or reports the Rust rule's verdict over two saved records.

The pressure scenario's C16 cohort must evict a prefix-cache anchor
(`c16_prefix_evictions`). A fill phase after the warm cohort makes that hold
by construction rather than by timing. runtime-kv's pool gives an anchor its
sequence's leading pages and S fresh state pages, and when an append or a
snapshot finds too few free pages it evicts anchors least recently used
first, counting each; pages an anchor holds stay held until it is evicted.
The fill reads the pinned pool from the gauges (stripes and tokens) and the
cohort's longest prompt, L stripes, from C1, then sends distinct token-ID
fillers of k = min(L - FILL_SLACK_PAGES, FILL_CHUNK_TOKENS / stripe) stripes
one at a time until it has evicted every anchor cached when it began and one
filler anchor. The pool then holds only filler anchors, k + S pages each,
and fewer than k + S + FILL_SLACK_PAGES free pages. The longest C16 prompt
can hit no anchor, so with its own capture it needs L + S pages beyond them:
more than are free, whatever swaps return meanwhile, so C16 evicts. A fill
that reaches its bound first records `precondition: false`.

The prefix-pressure arm (`kv-offload-prefix-pressure`, record `--arm tier
--scenario prefix-pressure`) runs the pressure workload with the RAM prefix
tier on (`--kv-offload-prefix on`) and no forced swaps. The fill evicts
every anchor C1 and the warm cohort left, so C16 can find them only in the
host tier. Its judge pairs it with the pressure scenario's off arm: both
records complete, without a failed C1, warm, fill or control request, the
off arm passing its own end gates, every C1 request bitwise equal, the
fill's precondition met, published promotions (result "ok") and RAM hits
above zero, the tier's missed demotions recorded, no transfer pending,
swapped request or stall at the end. The tier's counters must be exposed: missing evidence never
becomes zero. The tier holds its entries after the run, so its host pages
are reported, not required to drain.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import time

from common import Client, arguments, qualification_config, receipt
import kinds
import limits
from preempt_compare import prompts
from spec_metrics import metrics_text

OUTPUT_TOKENS = 96
CONCURRENCY = 16
# A filler fits one prefill chunk (2,048 tokens in every profile), so its
# only anchor is at its end.
FILL_CHUNK_TOKENS = 2048
# Pages a filler may hold beyond its anchor (a decode, draft or copied last
# page) until it finishes.
FILL_SLACK_PAGES = 4
FILL_TEXT = "Fill the prefix cache with distinct anchors before the concurrent cohort. "
# The RAM prefix tier's counters (prefix-pressure arm), plain and by label,
# and its gauges.
TIER_COUNTERS = {"ram_hits": "chop_prefix_cache_ram_hits_total",
                 "ram_reused_tokens": "chop_prefix_cache_ram_reused_tokens_total",
                 "headroom_waits": "chop_prefix_ram_headroom_waits_total",
                 "ram_evictions": "chop_prefix_ram_evictions_total"}
TIER_LABELLED = {"demotions": "chop_prefix_ram_demotions_total",
                 "promotions": "chop_prefix_ram_promotions_total",
                 "missed_evictions": "chop_prefix_ram_missed_evictions_total",
                 "ram_fallbacks": "chop_prefix_ram_fallbacks_total"}
TIER_GAUGES = {"ram_anchors": "chop_prefix_ram_anchors", "ram_bytes": "chop_prefix_ram_bytes",
               "stalled_transfers": "chop_kv_offload_stalled_transfers"}
PENDING_GAUGE = "chop_kv_offload_oldest_pending_seconds"
# How long the prefix-pressure record waits for the tier's last copies
# before its final snapshot.
IDLE_SECONDS = 30
POOL_GAUGES = {"capacity_pages": "chop_kv_block_capacity", "capacity_tokens": "chop_kv_capacity_tokens",
               "used_pages": "chop_kv_blocks_used", "cached_pages": "chop_prefix_cache_blocks",
               "anchors": "chop_prefix_cache_anchors"}
COUNTERS = {
    "preemptions": "vllm:num_preemptions_total",
    "swap_out": "chop_kv_offload_swap_out_total",
    "swap_in": "chop_kv_offload_swap_in_total",
    "exact_resumes": "chop_prefix_restore_resumed_decoding_total",
    "stalls": "chop_kv_offload_stalls_total",
    "prefix_evictions": "chop_prefix_cache_evictions_total",
    "verified_items": "chop_spec_verify_items_total",
}


def configuration_option(configuration, flag, default):
    words = configuration.get("serve", [])
    return int(words[words.index(flag) + 1]) if flag in words else default


def pressure_pages(policy, page_tokens, parallel):
    pages, remainder = divmod(policy["pressure_tokens"], page_tokens * parallel)
    if not pages or remainder:
        raise ValueError("kv_offload.pressure_tokens must hold a whole number of KV stripes")
    return pages


def arms(facts, configuration):
    """Serve overrides, with pressure pages counted in context-parallel stripes."""
    policy = facts["kv_offload"] | configuration.get("kv_offload", {})
    pages = (pressure_pages(policy, configuration["page_tokens"], configuration_option(configuration, "--context-parallel-size", configuration.get("context_parallel_size", 1)))
             if "page_tokens" in configuration else "{kv_offload_pressure_pages}")
    depth = configuration_option(configuration, "--speculative-tokens", configuration.get("depth", facts["depth"]))
    result = {}
    for scenario in ("plain", "pressure"):
        for arm in ("off", "swap"):
            words = ["--admission", "optimistic", "--max-sequences", "16",
                     "--max-preemptions", str(policy["max_preemptions"]),
                     "--kv-offload-min-tokens", "1", "--kv-offload-prefix", "off",
                     "--prefix-cache", "on" if scenario == "pressure" else "off",
                     "--speculative-tokens", str(depth if scenario == "pressure" else 0),
                     "--kv-offload-gib", str(policy["budget_gib"] if arm == "swap" else 0)]
            if arm == "swap":
                words += ["--debug-swap-every", str(policy["swap_every"])]
            if scenario == "pressure":
                words += ["--kv-pages", str(pages), "--speculation", "fixed"]
            result[f"kv-offload-{scenario}-{arm}"] = words
    # The pressure off arm's settings with the RAM prefix tier on.
    tier = list(result["kv-offload-pressure-off"])
    tier[tier.index("--kv-offload-prefix") + 1] = "on"
    tier[tier.index("--kv-offload-gib") + 1] = str(policy["budget_gib"])
    result["kv-offload-prefix-pressure"] = tier
    return result


def RUNNER_ARMS(model):
    """The arms depend on the configuration: the runner calls the returned
    function with its model facts and configuration."""
    return arms


RUNNER_RECORDS = {
    f"kv_offload_{scenario}_{arm}": (
        f"kv-offload-{scenario}-{arm}", "kv_offload.py",
        ["record", "--url", "{url}", "--arm", arm, "--scenario", scenario, "--timeout", "600"])
    for scenario in ("plain", "pressure") for arm in ("off", "swap")
}
RUNNER_RECORDS["kv_offload_prefix_pressure_tier"] = (
    "kv-offload-prefix-pressure", "kv_offload.py",
    ["record", "--url", "{url}", "--arm", "tier", "--scenario", "prefix-pressure", "--timeout", "600"])


def metrics(text, offload=True, tier=False):
    """Required counters and gauges, with the RAM prefix tier's under `tier`;
    missing evidence never becomes zero."""
    samples = {}
    wanted = {*COUNTERS.values(), "chop_kv_offload_host_pages",
              "chop_kv_offload_swapped_requests", "chop_kv_offload_fallback_total"}
    if tier:
        wanted |= {*TIER_COUNTERS.values(), *TIER_LABELLED.values(), *TIER_GAUGES.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+(\S+)', line)
        if match:
            name, labels, raw = match.groups()
            if name not in wanted:
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(f"invalid metric {name}: {raw}")
            samples.setdefault(name, []).append((labels or "", int(value)))
    def read(name, labels=None):
        rows = [value for key, value in samples.get(name, []) if labels is None or key == labels]
        if not rows and (name == "chop_spec_verify_items_total" or
                         (not offload and name.startswith("chop_kv_offload_"))):
            return 0
        if not rows:
            raise ValueError(f"missing metric {name}{labels or ''}")
        return sum(rows)
    result = {key: read(name) for key, name in COUNTERS.items()}
    result["host_pages_used"] = read("chop_kv_offload_host_pages", '{kind="used"}')
    result["swapped_requests"] = read("chop_kv_offload_swapped_requests")
    result["fallbacks"] = {}
    for labels, value in samples.get("chop_kv_offload_fallback_total", []):
        match = re.fullmatch(r'\{reason="([^"]+)"\}', labels)
        if match is None:
            raise ValueError("invalid offload fallback labels")
        result["fallbacks"][match[1]] = value
    if tier:
        result["tier"] = tier_metrics(text, samples, read)
    return result


def tier_metrics(text, samples, read):
    """The RAM prefix tier's counters and gauges. A labelled family may not
    have a sample yet, but its TYPE line must show the server exports it."""
    result = {key: read(name) for key, name in {**TIER_COUNTERS, **TIER_GAUGES}.items()}
    for key, name in TIER_LABELLED.items():
        if name not in samples and f"# TYPE {name} counter" not in text.splitlines():
            raise ValueError(f"missing metric {name}")
        result[key] = {}
        for labels, value in samples.get(name, []):
            match = re.fullmatch(r'\{[a-z_]+="([^"]+)"\}', labels)
            if match is None:
                raise ValueError(f"invalid {name} labels")
            result[key][match[1]] = value
    rows = [line.split()[-1] for line in text.splitlines() if line.startswith(PENDING_GAUGE + " ")]
    if len(rows) != 1:
        raise ValueError(f"missing metric {PENDING_GAUGE}")
    pending = float(rows[0])
    if not math.isfinite(pending) or pending < 0:
        raise ValueError(f"invalid metric {PENDING_GAUGE}: {rows[0]}")
    result["pending_micros"] = math.ceil(pending * 1e6)
    return result


def pool(text):
    """The KV pool gauges: capacity, used and anchor-only pages, anchors."""
    values = {}
    for key, name in POOL_GAUGES.items():
        rows = [line.split()[-1] for line in text.splitlines() if line.startswith(name + " ")]
        if len(rows) != 1:
            raise ValueError(f"missing metric {name}")
        value = float(rows[0])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid metric {name}: {rows[0]}")
        values[key] = int(value)
    return values


def fill_plan(capacity_pages, capacity_tokens, longest_prompt_tokens, anchors):
    """The fill for a pool of `capacity_pages` stripes holding
    `capacity_tokens`, a cohort whose longest prompt has
    `longest_prompt_tokens` and `anchors` anchors cached (module doc).
    `filler_pages` 0 means no filler can satisfy the bound."""
    stripe, remainder = divmod(capacity_tokens, capacity_pages) if capacity_pages else (0, 1)
    if not stripe or remainder:
        raise ValueError("the KV pool must hold a whole number of tokens per stripe")
    longest = -(-longest_prompt_tokens // stripe)
    pages = max(0, min(longest - FILL_SLACK_PAGES, FILL_CHUNK_TOKENS // stripe))
    return {"capacity_pages": capacity_pages, "stripe_tokens": stripe, "longest_prompt_pages": longest,
            "filler_pages": pages, "filler_tokens": pages * stripe, "anchors": anchors,
            "evictions": anchors + 1,
            "max_fillers": 2 * -(-capacity_pages // pages) + anchors + 4 if pages else 0}


def tokenize(client, model, text, special=True):
    response, _ = client.request("/tokenize", {"model": model, "prompt": text, "add_special_tokens": special})
    return response["tokens"]


def fill(client, model, c1, snapshot):
    """The pressure scenario's fill phase (module doc), sent one at a time."""
    gauges = pool(metrics_text(client))
    longest = max((row.get("usage") or {}).get("prompt_tokens", 0) for row in c1)
    plan = fill_plan(gauges["capacity_pages"], gauges["capacity_tokens"], longest, gauges["anchors"])
    start = snapshot()["prefix_evictions"]
    evictions, rows = 0, []
    body = tokenize(client, model, FILL_TEXT, special=False) if plan["filler_pages"] else []
    if plan["filler_pages"] and not body:
        raise ValueError("the fill text tokenized to nothing")
    for index in range(plan["max_fillers"]):
        if evictions >= plan["evictions"]:
            break
        ids = tokenize(client, model, f"Filler {index} of the offload pressure check. ")
        ids = (ids + body * -(-plan["filler_tokens"] // len(body)))[:plan["filler_tokens"]]
        row = completion(client, model, (f"fill-{index}", ids), 1)
        rows.append({key: row[key] for key in ("name", "prompt_sha256", "finish_reason", "usage", "error")
                     if key in row})
        evictions = snapshot()["prefix_evictions"] - start
    return {"plan": plan, "pool_before": gauges, "pool_after": pool(metrics_text(client)),
            "evictions": evictions, "precondition": bool(plan["filler_pages"]) and evictions >= plan["evictions"],
            "rows": rows}


def delta(before, after):
    result = {key: after[key] - before[key] for key in COUNTERS}
    result["fallbacks"] = {key: after["fallbacks"].get(key, 0) - before["fallbacks"].get(key, 0)
                           for key in before["fallbacks"].keys() | after["fallbacks"].keys()
                           if after["fallbacks"].get(key, 0) != before["fallbacks"].get(key, 0)}
    if any(value < 0 for key, value in result.items() if key != "fallbacks") or any(
            value < 0 for value in result["fallbacks"].values()):
        raise ValueError("metrics reset during capture")
    if "tier" in before and "tier" in after:
        result["tier"] = tier_delta(before["tier"], after["tier"])
    return result


def tier_delta(before, after):
    """The tier counters' changes; gauges have none."""
    result = {key: after[key] - before[key] for key in TIER_COUNTERS}
    for key in TIER_LABELLED:
        result[key] = {label: after[key].get(label, 0) - before[key].get(label, 0)
                       for label in before[key].keys() | after[key].keys()
                       if after[key].get(label, 0) != before[key].get(label, 0)}
    if any(value < 0 for key, value in result.items() if key in TIER_COUNTERS) or any(
            value < 0 for key in TIER_LABELLED for value in result[key].values()):
        raise ValueError("tier metrics reset during capture")
    return result


def idle(snapshot):
    """Whether the host tier has finished its copies: no transfer pending or
    overdue and no request swapped out."""
    tier = snapshot["tier"]
    return not tier["pending_micros"] and not tier["stalled_transfers"] and not snapshot["swapped_requests"]


def settle(snapshot, seconds=IDLE_SECONDS):
    """The first snapshot that finds the host tier idle, or the last one
    taken within `seconds`, which the judge then finds busy."""
    end = time.monotonic() + seconds
    while True:
        value = snapshot()
        if idle(value) or time.monotonic() >= end:
            return value
        time.sleep(.5)


def completion(client, model, job, output_tokens):
    name, prompt = job
    data = prompt.encode() if isinstance(prompt, str) else json.dumps(prompt).encode()
    row = {"name": name, "prompt_sha256": hashlib.sha256(data).hexdigest()}
    try:
        response, _ = client.request("/v1/completions", {
            "model": model, "prompt": prompt, "max_tokens": output_tokens,
            "temperature": 0, "top_p": 1, "logprobs": 2, "return_token_ids": True})
        choice = response["choices"][0]
        scores = choice["logprobs"]
        row.update(token_ids=choice["token_ids"], tokens=scores["tokens"],
                   logprobs=scores["token_logprobs"], top=scores["top_logprobs"],
                   finish_reason=choice["finish_reason"], usage=response["usage"])
    except (RuntimeError, ValueError, KeyError) as error:
        row["error"] = str(error)
    return row


def record(args):
    client = Client(args.url, args.timeout, retries=0)
    server = client.identity(args.model)
    jobs = prompts()
    deadline = time.monotonic() + args.timeout
    def snapshot():
        client.timeout = max(0.001, deadline - time.monotonic())
        if time.monotonic() >= deadline:
            raise TimeoutError("offload capture timed out")
        return metrics(metrics_text(client), offload=args.arm != "off", tier=args.arm == "tier")
    before = snapshot()
    c1 = []
    for job in jobs:
        start = snapshot()
        row = completion(client, server["model"], job, args.output_tokens)
        row["observed"] = delta(start, snapshot())
        c1.append(row)
    c1_after = snapshot()
    def concurrent():
        snapshot()
        with ThreadPoolExecutor(args.concurrency) as pool:
            return list(pool.map(lambda job: completion(client, server["model"], job, args.output_tokens), jobs))
    pressure = args.scenario in ("pressure", "prefix-pressure")
    warm = concurrent() if pressure else []
    filled = fill(client, server["model"], c1, snapshot) if pressure else None
    c16_before = snapshot()
    c16 = concurrent()
    c16_after = snapshot()
    repeat = concurrent() if args.arm == "off" else []
    # The tier's last demotions may still be copying when C16 ends.
    after = settle(snapshot) if args.arm == "tier" else snapshot()
    failed = any("error" in row for rows in (c1, warm, (filled or {}).get("rows", []), c16, repeat)
                 for row in rows)
    # No timed work, so no host load samples: the receipt's timing reads informational.
    receipt(f"kv-offload-{args.scenario}-{args.arm}", server, {
        "passed": not failed, "arm": args.arm, "scenario": args.scenario,
        "output_tokens": args.output_tokens, "concurrency": args.concurrency,
        "c1": c1, "warm": warm, "fill": filled, "c16": c16, "repeat": repeat,
        "before": before, "c1_after": c1_after,
        "c16_before": c16_before, "c16_after": c16_after, "after": after})
    return int(failed)


def report(args):
    off = json.loads(args.off.read_text())
    config = qualification_config()
    speculative = config["preview"]["speculation"]["tokens"] > 0
    policy = {"facts": config["kv_offload"], "speculative": speculative}
    if args.tier:
        # The prefix-pressure arm, judged against the pressure scenario's off arm.
        tier = json.loads(args.tier.read_text())
        value = limits.one("kv_offload_prefix", off=off, tier=tier, **policy)
        receipt(f"kv-offload-{tier['scenario']}", tier.get("server"),
                {**value, "records": {"off": off, "tier": tier}, "policy": policy})
        return 0 if value["passed"] else 1
    swap = json.loads(args.swap.read_text())
    value = limits.one("kv_offload", off=off, swap=swap, **policy)
    # Judges saved records. No timed work, so no host load samples: the receipt's timing reads informational.
    receipt(f"kv-offload-{off['scenario']}", swap.get("server"),
            {**value, "records": {"off": off, "swap": swap}, "policy": policy})
    return 0 if value["passed"] else 1


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.command == "record")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("record")
    capture.add_argument("--url", required=True)
    capture.add_argument("--arm", choices=("off", "swap", "tier"), required=True)
    capture.add_argument("--scenario", choices=("plain", "pressure", "prefix-pressure"), required=True)
    capture.add_argument("--timeout", type=float, default=600)
    capture.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS)
    capture.add_argument("--concurrency", type=int, default=CONCURRENCY)
    score = sub.add_parser("report")
    score.add_argument("--off", type=Path, required=True)
    against = score.add_mutually_exclusive_group(required=True)
    against.add_argument("--swap", type=Path)
    against.add_argument("--tier", type=Path, help="the prefix-pressure arm, judged against the pressure off arm")
    args = parser.parse_args()
    if args.command == "record" and (args.arm == "tier") != (args.scenario == "prefix-pressure"):
        parser.error("the tier arm records the prefix-pressure scenario, and only it")
    return record(args) if args.command == "record" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
