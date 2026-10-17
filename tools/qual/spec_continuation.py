"""Score speculative continuations against the reference on their own histories.

A speculative and an AR server stop agreeing token for token once their
outputs diverge, and spec_verify scores only the common prefix. Here each
arm's own continuation is a fixed token sequence: the reference runner
(tools/refrun) teacher-forces it, so every reference row conditions on
exactly the history the arm had when it emitted that token, and every
emitted position is scored.

`record` sends the frozen sets below to one server and stores, per request,
the prompt and emitted token IDs (`return_token_ids`), each emitted token's
raw logprob (scored against the unmodified logits) and the top 20 by label,
the finish reason, the speculation counters of each set and the server's
checkpoint binding and serving profile. Record the speculative candidate
(`--arm chopinfer-spec`) and the AR control (`--arm chopinfer-ar`, the same
install with `--speculative-tokens 0`) with one `--run-id`.

`workload` turns a record into a reference-runner workload: one sequence
per request, named SET-INDEX, its emitted IDs forced after the prompt and
scored from the first of them. Run it through `tools.refrun run` in the gate
variant's mode and the spread mode.

`score` joins a record with the runner's mode files and reports, per set
and pooled over the greedy sets, coverage per request and in aggregate,
mean |dlogprob| at the emitted token, top-1 flips on greedy sets (the
emitted token is the arm's top-1) by the near-tie rule, the spread
variant's same measures against the gate variant, the spread rule's limits,
and the reference's entropy profile. A request whose reference rows do not
read its own prompt and emitted IDs is refused, never shortened. On the
sampled set (temperature 1, top_p 1) each emitted token is a draw from the
arm, so the mean of logprob_arm - logprob_reference there estimates the mean
KL divergence of the arm from the reference over the whole vocabulary.

`reference` runs one arm's record through the reference runner on this
process's GPUs, in the modes and under the checkpoint lock, adapter and
environment of the gate's pinned teacher reference (its `binding`), and
writes the scores with an execution record: what ran on which GPUs, which
entries were computed and which came from the reference cache with their
own provenance.

`compare` reads a speculative and an AR record with their references and
reports both arms, the candidate-minus-control differences with
request-clustered intervals, and the verdict of spec-continuation.v1
(docs/numerics.md#speculative-continuations). A mode file is read only when
its policy is that mode's run of the pinned teacher reference, and a
reference entry not computed as that reference was (lock, adapter,
reference code, environment, outputs and mode policy) is refused like one
of another history. Both arms must be bound to the gate's pinned checkpoint
on two processes of one install (binary source commit, direct package and
metadata) and one profile apart from speculation, the candidate drafting
and the control not. The receipt keeps the GPU execution of each
reference apart from this CPU scoring.

The correctness runner records both arms on the servers it starts
(RUNNER_RECORDS), runs `reference` for each after the configuration's
servers have stopped (RUNNER_OFFLINE_RECORDS) and then runs the check
without a server (RUNNER_CHECK_ARM).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_TOKENS = 64
SAMPLED_TOKENS = 128
SEEDS = (1, 2)
TOP_LOGPROBS = 20
GREEDY = {"temperature": 0, "top_p": 1}
SAMPLED = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
# The reference runner's mode labels: the gate variant and the variants
# whose spread raises the limits.
GATE_MODE = "f32"
SPREAD_MODES = ("bf16",)
# A position whose reference top-5 mass is below this has a 0.95 nucleus the
# top 5 cannot hold: the class top-5 nucleus analyses do not measure.
NUCLEUS = 0.95
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_ROUNDS = 2000
RULE = "spec-continuation.v1"
# The pinned reference whose binding the continuation references must share
# (the teacher's exact reference), and the variant each mode label scores.
BINDING_INPUT = "exact"
MODE_VARIANTS = {"f32": "exact", "bf16": "bf16"}
EXECUTION_SCHEMA = "chopinfer.spec-continuation.reference-execution.v1"
_SHARED = ["--url", "{url}", "--model", "{name}", "--timeout", "120", "--server-log", "{server_log}",
           "--run-id", "{run_id}"]
# The correctness runner's records: each arm's continuations on the servers
# it starts, then each arm's reference once the servers have stopped and the
# GPUs are free; the check itself needs no server.
RUNNER_RECORDS = {"continuation_spec": ("spec", "spec_continuation.py", ["record", *_SHARED, "--arm", "chopinfer-spec"]),
                  "continuation_ar": ("ar", "spec_continuation.py", ["record", *_SHARED, "--arm", "chopinfer-ar"])}
RUNNER_OFFLINE_RECORDS = {
    "continuation_spec_reference": ("spec_continuation.py", ["reference", "--record", "{continuation_spec}",
                                                             "--checkpoint", "{checkpoint}", "--out", "{run_dir}"]),
    "continuation_ar_reference": ("spec_continuation.py", ["reference", "--record", "{continuation_ar}",
                                                           "--checkpoint", "{checkpoint}", "--out", "{run_dir}"])}
RUNNER_CHECK_ARM = "offline"
REFERENCE_SECONDS = 3300

# Open-ended prompts with many plausible continuations, frozen for the
# sampled set beside the benchmark's chat prompts.
ENTROPY_PROMPTS = (
    "Write the opening paragraph of a story set in a lighthouse during a storm.",
    "List twelve unusual names for a small coffee shop, one per line.",
    "Describe an imaginary animal that lives in deserts and how it finds water.",
    "Write a short poem about a train station at midnight.",
    "Invent a holiday and describe how people celebrate it.",
    "Write a letter from a traveler to a friend describing a market in a foreign city.",
    "Describe the view from a mountain cabin on the first morning of winter.",
    "Suggest ten names for a robot that tends a garden, with a reason for each.",
    "Write a dialogue between a cat and a houseplant.",
    "Describe a meal that a chef might cook to celebrate the end of a long journey.",
    "Write a fairy tale that begins with a lost key.",
    "Describe a city of the far future from the point of view of a street musician.",
    "Write a product description for a lamp that changes color with the weather.",
    "Tell a short legend that explains why the sea is salty.",
    "Describe a dream about flying over a forest at dawn.",
    "Write the first page of a diary kept by a scientist on a remote island.",
)


def prompt_sets():
    """The frozen sets: name to sampling, output tokens and (prompt, seed)
    requests."""
    import spec_verify
    from spec_bench import CHAT_PROMPTS
    base = spec_verify.prompt_sets()
    greedy = {"copy": base["copy"], "natural": base["natural"], "open": list(CHAT_PROMPTS)}
    sets = {name: {"sampling": dict(GREEDY), "max_tokens": OUTPUT_TOKENS,
                   "requests": [{"prompt": prompt, "seed": None} for prompt in prompts]}
            for name, prompts in greedy.items()}
    sets["sampled"] = {"sampling": dict(SAMPLED), "max_tokens": SAMPLED_TOKENS,
                       "requests": [{"prompt": prompt, "seed": seed} for seed in SEEDS
                                    for prompt in (*CHAT_PROMPTS, *ENTROPY_PROMPTS)]}
    return sets


def sets_digest(sets):
    """One digest of every set's prompts, seeds, sampling and lengths."""
    return hashlib.sha256(json.dumps(sets, sort_keys=True).encode()).hexdigest()


def completion(client, model, request, sampling, max_tokens):
    body = {"model": model, "prompt": request["prompt"], "max_tokens": max_tokens,
            "logprobs": TOP_LOGPROBS, "return_token_ids": True, **sampling}
    if request["seed"] is not None:
        body["seed"] = request["seed"]
    response, seconds = client.request("/v1/completions", body)
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    return {"prompt_ids": choice.get("prompt_token_ids") or response.get("prompt_token_ids"),
            "ids": choice.get("token_ids"), "tokens": logprobs["tokens"], "logprobs": logprobs["token_logprobs"],
            "top": logprobs["top_logprobs"], "finish_reason": choice["finish_reason"], "seed": request["seed"],
            "seconds": seconds}


def record(args):
    from common import Client, bind_checkpoint, confirm_live, receipt, serving_profile
    from spec_metrics import difference, metrics_text, snapshot
    from spec_verify import engine_counters
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    try:
        bind_checkpoint(server, args.server_log, args.url)
        serving = serving_profile(server)
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
    sets = prompt_sets()
    recorded = {}
    for name, frozen in sets.items():
        text = metrics_text(client)
        before, engine_before = snapshot(text, "chopinfer", server["model"]), engine_counters(text)
        # A set's requests run together (at most --concurrency at once), so
        # verify rows come in the row buckets of a loaded server.
        width = min(args.concurrency or len(frozen["requests"]), len(frozen["requests"]))
        with ThreadPoolExecutor(max_workers=width) as pool:
            rows = list(pool.map(lambda request: completion(client, server["model"], request, frozen["sampling"],
                                                            frozen["max_tokens"]), frozen["requests"]))
        text = metrics_text(client)
        after, engine_after = snapshot(text, "chopinfer", server["model"]), engine_counters(text)
        recorded[name] = {"sampling": frozen["sampling"], "max_tokens": frozen["max_tokens"],
                          "concurrency": width, "requests": rows, "speculation": difference(before, after),
                          "generated_tokens": engine_after["chop_generation_tokens_total"]
                          - engine_before["chop_generation_tokens_total"]}
    confirm_live(client, server)
    # Its per-request seconds are diagnostics of an accuracy check, not timing evidence, so no host load
    # samples bound them and the receipt's timing reads informational.
    receipt(f"spec-continuation-{args.arm}", server, {
        "passed": True, "arm": args.arm, "run_id": args.run_id, "serving": serving,
        "sets_sha256": sets_digest(sets), "sets": recorded}, False)


def sequence_name(name, index):
    return f"{name}-{index:03d}"


def workload(value, model, names=None):
    """The reference-runner workload of a record (of its sets `names`, all by
    default): every request's emitted IDs forced after its prompt, scored
    from the first emitted token."""
    sequences = []
    if names is not None and not set(names) <= set(value["sets"]):
        raise ValueError(f"the record has no set {sorted(set(names) - set(value['sets']))}")
    for name, group in value["sets"].items():
        if names is not None and name not in names:
            continue
        for index, row in enumerate(group["requests"]):
            prompt, ids = row.get("prompt_ids"), row.get("ids")
            if not prompt or not ids:
                raise ValueError(f"{sequence_name(name, index)}: the record has no prompt or emitted token IDs")
            sequences.append({"name": sequence_name(name, index), "category": name, "token_ids": list(prompt),
                              "score_from": len(prompt), "answer": {"policy": "forced", "ids": list(ids)}})
    return {"schema": "chopinfer.refrun.workload.v1", "model": model, "sequences": sequences}


def canonical_sha256(value):
    """tools/model_sdk/diagnostic.py's canonical JSON digest, as refrun keys use it."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ids_sha256(ids):
    """tools/refrun/keys.py's digest of a token ID list."""
    return canonical_sha256(list(ids))


def binding_modes(binding):
    """Each mode label's run of the pinned reference: its mode, dequant rule and
    policy digest, checked to score the variant this check reads."""
    modes = (binding or {}).get("modes") or {}
    for label, variant in MODE_VARIANTS.items():
        if (modes.get(label) or {}).get("variant") != variant:
            raise ValueError(f"the pinned reference has no {label} run of its {variant} variant")
    return {label: modes[label] for label in MODE_VARIANTS}


def unbound(entry, mode, binding):
    """Why a reference entry was not computed as the pinned reference was,
    or None: its key must be its own digest's and share the pinned binding's
    model, checkpoint lock, reference code, adapter, environment and outputs,
    and the mode's policy."""
    key = entry.get("key") or {}
    shared = binding["key"]
    if entry.get("key_sha256") != canonical_sha256(key):
        return "key"
    if key.get("model") != binding.get("model"):
        return "model"
    if (key.get("checkpoint") or {}).get("source_lock_sha256") != shared["checkpoint"]["source_lock_sha256"]:
        return "checkpoint lock"
    for section in ("reference_code", "adapter", "environment", "outputs"):
        if key.get(section) != shared[section]:
            return section.replace("_", " ")
    run = binding["modes"][mode]
    if key.get("mode") != {"name": run["mode"], "policy_sha256": run["policy_sha256"]}:
        return "mode policy"
    return None


def reference_rows(entry, row):
    """The reference scores of one request when they read exactly its own
    prompt and emitted IDs; otherwise the reason they do not."""
    prompt, ids = row.get("prompt_ids"), row.get("ids")
    if entry is None:
        return None, "no reference entry"
    sequence = (entry.get("key") or {}).get("sequence") or {}
    scores = entry.get("scores") or {}
    if not prompt or not ids:
        return None, "no prompt or emitted token IDs"
    if sequence.get("token_ids_sha256") != ids_sha256(prompt) or sequence.get("score_from") != len(prompt) \
            or (sequence.get("answer") or {}).get("ids_sha256") != ids_sha256(ids) \
            or (sequence.get("answer") or {}).get("policy") != "forced":
        return None, "the reference read another history"
    if scores.get("score_from") != len(prompt) or scores.get("token") != list(ids):
        return None, "the reference rows force other tokens"
    columns = ("logprob", "top1", "topk_ids", "topk_logits", "topk_logprobs")
    if any(not isinstance(scores.get(column), list) or len(scores[column]) != len(ids) for column in columns):
        return None, "the reference rows do not cover every emitted token"
    return scores, None


def mass(logprobs, count):
    return sum(math.exp(value) for value in logprobs[:count])


def bucket(scores, position):
    """The reference's entropy class at a position: whether its top 5, or its
    top 20, holds the 0.95 nucleus."""
    if mass(scores["topk_logprobs"][position], 5) >= NUCLEUS:
        return "top5"
    if mass(scores["topk_logprobs"][position], 20) >= NUCLEUS:
        return "top20"
    return "beyond_top20"


def score_set(name, group, modes, binding):
    """Per-request coverage and per-position scores of one set against each
    mode's rows. `modes` maps a mode label to {sequence name: entry}; every
    entry must be computed as the pinned reference with `binding` was."""
    import limits
    greedy = group["sampling"].get("temperature") == 0
    requests, positions = [], []
    for index, row in enumerate(group["requests"]):
        label = sequence_name(name, index)
        ids, logprobs = row.get("ids"), row.get("logprobs")
        ids = ids if isinstance(ids, list) else []
        logprobs = logprobs if isinstance(logprobs, list) else []
        emitted = len(ids)
        rows, reason = {}, None
        if not ids or not all(type(token) is int and token >= 0 for token in ids):
            reason = "no emitted token IDs"
        elif len(logprobs) != emitted or not all(type(value) in (int, float) and math.isfinite(value)
                                                 for value in logprobs):
            reason = "logprob rows differ from the emitted token IDs"
        for mode, entries in modes.items():
            if reason is None:
                rows[mode], reason = reference_rows(entries.get(label), row)
            if reason is None and (field := unbound(entries[label], mode, binding)):
                reason = f"the reference was not computed as the pinned reference: {field}"
        scored = 0 if reason else emitted
        requests.append({"name": label, "emitted": emitted, "scored": scored, "unscored": emitted - scored,
                         "refused": reason, "finish_reason": row.get("finish_reason")})
        if reason:
            continue
        gate = rows[GATE_MODE]
        for position, (token, logprob) in enumerate(zip(ids, logprobs, strict=True)):
            positions.append({"request": label, "offset": position, "token": token,
                              "difference": abs(logprob - gate["logprob"][position]),
                              "excess": logprob - gate["logprob"][position],
                              "bucket": bucket(gate, position),
                              "spread": {mode: abs(rows[mode]["logprob"][position] - gate["logprob"][position])
                                         for mode in SPREAD_MODES},
                              "flip_args": (gate["top1"][position], token,
                                            [list(pair) for pair in zip(gate["topk_ids"][position],
                                                                        gate["topk_logits"][position])]),
                              "spread_flip_args": {mode: (gate["top1"][position], rows[mode]["top1"][position],
                                                          [list(pair) for pair in zip(
                                                              gate["topk_ids"][position],
                                                              gate["topk_logits"][position])])
                                                   for mode in SPREAD_MODES}})
    if greedy and positions:
        flips = limits.flips([p["flip_args"] for p in positions])
        for position, flip in zip(positions, flips, strict=True):
            position["flip"] = flip
        for mode in SPREAD_MODES:
            for position, flip in zip(positions, limits.flips([p["spread_flip_args"][mode] for p in positions]),
                                      strict=True):
                position.setdefault("spread_flip", {})[mode] = flip
    for position in positions:
        del position["flip_args"], position["spread_flip_args"]
    return requests, positions


def complete_request(row):
    """A request is covered when it emitted tokens, none was refused and every
    one was scored; a refused request is never complete, whatever it emitted."""
    return row["refused"] is None and row["emitted"] > 0 and row["scored"] == row["emitted"]


def coverage(requests):
    emitted = sum(row["emitted"] for row in requests)
    scored = sum(row["scored"] for row in requests)
    complete = sum(map(complete_request, requests))
    return {"requests": len(requests), "complete_requests": complete,
            "refused_requests": sum(row["refused"] is not None for row in requests),
            "emitted_positions": emitted, "scored_positions": scored, "unscored_positions": emitted - scored,
            "fraction": scored / emitted if emitted else None,
            "complete": bool(requests) and complete == len(requests)}


def measures(positions, greedy):
    """Mean |dlogprob| and, for greedy sets, the top-1 counts of an arm and of
    each spread variant over the same positions."""
    import limits
    count = len(positions)
    arm = {"scored": count, "mean_abs_dlogprob": statistics.mean(p["difference"] for p in positions) if count else None,
           "max_abs_dlogprob": max((p["difference"] for p in positions), default=None)}
    spread = {mode: {"mean_abs_dlogprob": statistics.mean(p["spread"][mode] for p in positions) if count else None}
              for mode in SPREAD_MODES}
    if greedy and count:
        arm.update(limits.top1_summary([p["flip"] for p in positions], count))
        for mode in SPREAD_MODES:
            spread[mode].update(limits.top1_summary([p["spread_flip"][mode] for p in positions], count))
    return arm, spread


def judged(positions, greedy):
    """The arm's measures, the spread rule's limits over the same positions
    and whether the arm meets them (the mean only on a sampled set)."""
    import limits
    arm, spread = measures(positions, greedy)
    if not positions:
        return {"arm": arm, "spread": spread, "limits": None, "within": None}
    variants = [{"mean_abs_dlogprob": value["mean_abs_dlogprob"],
                 "top1_decisive": value.get("top1_decisive", 0.0),
                 "top1_near_tie": value.get("top1_near_tie", 0.0),
                 "top1_disagreement": value.get("top1_disagreement", 0.0)} for value in spread.values()]
    bounds = limits.spread_limits(variants, len(positions))
    if greedy:
        within = limits.within({key: arm[key] for key in ("mean_abs_dlogprob", "top1_decisive", "top1_near_tie",
                                                          "top1_disagreement")}, bounds)
    else:
        within = arm["mean_abs_dlogprob"] <= bounds["mean_abs_dlogprob"]
    return {"arm": arm, "spread": spread, "limits": bounds, "within": within}


def entropy_profile(positions):
    profile = {}
    for name in ("top5", "top20", "beyond_top20"):
        chosen = [p for p in positions if p["bucket"] == name]
        profile[name] = {"positions": len(chosen),
                         "mean_abs_dlogprob": statistics.mean(p["difference"] for p in chosen) if chosen else None}
    return profile


def clustered_interval(values, clusters):
    """95% request-clustered bootstrap interval of a mean (teacher.py's
    resampling, seed and rounds)."""
    groups = {}
    for value, cluster in zip(values, clusters, strict=True):
        groups.setdefault(cluster, []).append(value)
    groups = list(groups.values())
    if len(groups) < 2:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.mean([value for _ in groups for value in generator.choice(groups)])
                   for _ in range(BOOTSTRAP_ROUNDS))
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]]


def kl_estimate(positions):
    """Mean logprob_arm - logprob_reference at sampled tokens: an unbiased
    estimate of the mean KL(arm || reference) over the visited histories,
    with its interval and the Pinsker bound on mean total variation."""
    if not positions:
        return None
    values = [p["excess"] for p in positions]
    interval = clustered_interval(values, [p["request"] for p in positions])
    upper = interval[1] if interval else statistics.mean(values)
    return {"positions": len(values), "estimate": statistics.mean(values), "ci95": interval,
            "total_variation_bound": math.sqrt(max(upper, 0.0) / 2)}


def score(value, modes, binding):
    """One arm's report: per set and pooled over the greedy sets, against
    reference rows computed as the pinned reference with `binding` was."""
    binding_modes(binding)
    if set(modes) != set(MODE_VARIANTS):
        raise ValueError(f"the reference has modes {sorted(modes)}, not {sorted(MODE_VARIANTS)}")
    report, pooled_requests, pooled_positions = {"arm": value["arm"], "run_id": value["run_id"],
                                                  "sets_sha256": value.get("sets_sha256"), "sets": {},
                                                  "_record": value}, [], []
    for name, group in value["sets"].items():
        greedy = group["sampling"].get("temperature") == 0
        requests, positions = score_set(name, group, modes, binding)
        entry = {"greedy": greedy, "coverage": coverage(requests), "requests": requests,
                 **judged(positions, greedy), "entropy": entropy_profile(positions)}
        if not greedy:
            entry["kl"] = kl_estimate(positions)
        report["sets"][name] = entry
        if greedy:
            pooled_requests += requests
            pooled_positions += positions
        report.setdefault("_positions", {})[name] = positions
    report["greedy_pooled"] = {"coverage": coverage(pooled_requests), **judged(pooled_positions, True),
                               "entropy": entropy_profile(pooled_positions)}
    return report


def reference_scores(directory):
    """The directory of a reference run's mode files: a `reference` output's
    `refrun`, or a plain runner output."""
    directory = Path(directory)
    return directory / "refrun" if (directory / "refrun").is_dir() else directory


def execution(directories):
    """The GPU execution records of an arm's reference runs, None for a run
    made outside `reference` (a plain runner output)."""
    records = []
    for directory in directories:
        path = Path(directory) / "execution.json"
        records.append(json.loads(path.read_text()) if path.is_file() else None)
    return records


def load_modes(directories, binding):
    """{mode: {sequence name: entry}} from the mode files of one or more
    reference runs (an arm's sets may be split over several runs). A file is
    read as a mode only when its policy is that mode's run of the pinned
    reference (`binding`): its name, dequantization rule and digest."""
    runs = binding_modes(binding)
    modes = {mode: {} for mode in (GATE_MODE, *SPREAD_MODES)}
    for directory in directories:
        directory = reference_scores(directory)
        for mode, entries in modes.items():
            path = Path(directory) / f"{mode}.json"
            document = json.loads(path.read_text())
            if document.get("schema") != "chopinfer.refrun.scores.v1":
                raise ValueError(f"{path}: not a reference-runner score file")
            policy, run = document.get("policy"), runs[mode]
            if not isinstance(policy, dict) or policy.get("name") != run["mode"] \
                    or policy.get("dequant") != run["dequant"] or canonical_sha256(policy) != run["policy_sha256"]:
                raise ValueError(f"{path}: not the pinned reference's {mode} run ({run['mode']}, "
                                 f"{run['dequant']} dequantization)")
            for entry in document["entries"]:
                if entry["name"] in entries:
                    raise ValueError(f"{path}: {entry['name']} appears twice")
                entries[entry["name"]] = entry
    return modes


def install_identity(value, arm):
    """The install a record's server ran: the binary's source commit and the
    direct package and metadata it loaded, as its /health reported them when
    the record was bound. A record without them, or whose binding disagrees
    with its own /health, is refused."""
    server = value.get("server") or {}
    checkpoint = server.get("checkpoint") or {}
    live = (server.get("health") or {}).get("serving") or {}
    identity = {"source_commit": live.get("source_commit"), "package_identity": checkpoint.get("package_identity"),
                "metadata_identity": checkpoint.get("metadata_identity")}
    if not isinstance(identity["source_commit"], str) or not identity["source_commit"] \
            or not isinstance(identity["package_identity"], str) or not identity["package_identity"]:
        raise ValueError(f"the {arm} record names no install identity (binary source commit and direct package)")
    if any(live.get(key) != checkpoint.get(key) for key in ("checkpoint_identity", "startup_nonce",
                                                            "package_identity", "metadata_identity")):
        raise ValueError(f"the {arm} record's checkpoint binding disagrees with its own /health")
    return identity


def drafted(value):
    """Draft tokens and verify items over a record's sets."""
    groups = value["sets"].values()
    return (sum((group.get("speculation") or {}).get("draft_tokens", 0) for group in groups),
            sum((group.get("speculation") or {}).get("verify_items", 0) for group in groups))


def compare(spec, ar, frozen=None):
    """The rule's verdict over a speculative and an AR score, with the
    reported candidate-minus-control differences. `frozen` (prompt_sets())
    is what both arms must have recorded: its digest and every request. Both
    arms must come from one checkpoint and served name on two processes of
    one topology, the candidate drafting and the control not."""
    if spec["arm"] != "chopinfer-spec" or ar["arm"] != "chopinfer-ar" or spec["run_id"] != ar["run_id"]:
        raise ValueError("comparison needs the speculative and AR arms of one run")
    if spec["sets_sha256"] != ar["sets_sha256"] or (frozen and spec["sets_sha256"] != sets_digest(frozen)):
        raise ValueError("the arms recorded different frozen sets")
    if set(spec["sets"]) != set(ar["sets"]) or (frozen and set(spec["sets"]) != set(frozen)):
        raise ValueError("the arms recorded different sets")
    from spec_verify import compare_servers
    bound = compare_servers(spec["_record"], ar["_record"])
    if bound is None:
        raise ValueError("both arms must be bound to the pinned checkpoint by their servers' startup identity")
    profiles = {arm: {key: value for key, value in bound["serving"][arm].items() if key != "speculation"}
                for arm in ("spec", "ar")}
    if profiles["spec"] != profiles["ar"]:
        raise ValueError(f"the arms served different profiles apart from speculation: {profiles}")
    installs = {arm: install_identity(value["_record"], arm) for arm, value in (("spec", spec), ("ar", ar))}
    if installs["spec"] != installs["ar"]:
        raise ValueError(f"the arms come from different installs: {installs}")
    bound = {**bound, "install": installs["spec"]}
    if min(drafted(spec["_record"])) <= 0 or max(drafted(ar["_record"])) > 0:
        raise ValueError("the speculative arm must draft and verify, and the AR control must not")
    insufficient, failed, rows = [], [], {}
    if frozen:
        for arm in (spec, ar):
            for name, group in frozen.items():
                if arm["sets"][name]["coverage"]["requests"] != len(group["requests"]):
                    insufficient.append(f"{arm['arm']} {name}: not every frozen request was recorded")
    names = list(spec["sets"])
    if any(group["greedy"] for group in spec["sets"].values()):
        names.append("greedy_pooled")
    for name in names:
        left = spec["greedy_pooled"] if name == "greedy_pooled" else spec["sets"][name]
        right = ar["greedy_pooled"] if name == "greedy_pooled" else ar["sets"][name]
        if not (left["coverage"]["complete"] and right["coverage"]["complete"]):
            insufficient.append(f"{name}: incomplete coverage")
        elif right["within"] is not True:
            insufficient.append(f"{name}: the AR control misses the limits")
        elif left["within"] is not True:
            failed.append(name)
        row = {"spec": left["arm"].get("mean_abs_dlogprob"), "ar": right["arm"].get("mean_abs_dlogprob")}
        positions = (spec["_positions"].get(name), ar["_positions"].get(name)) if name != "greedy_pooled" else None
        if positions and positions[0] and positions[1]:
            row["difference_ci95"] = difference_interval(*positions)
        rows[name] = row
    import limits
    high_entropy = {}
    for arm in (spec, ar):
        # Sampled positions whose reference 0.95 nucleus the top 5 cannot hold.
        high_entropy[arm["arm"]] = sum(p["bucket"] != "top5" for name, group in arm["sets"].items()
                                       if not group["greedy"] for p in arm["_positions"][name])
        if high_entropy[arm["arm"]] < limits.LARGE_SET_POSITIONS:
            insufficient.append(f"{arm['arm']}: {high_entropy[arm['arm']]} sampled positions beyond the "
                                f"top-5 nucleus, fewer than {limits.LARGE_SET_POSITIONS}")
    # A candidate that misses a limit the control met fails whatever else
    # is insufficient.
    status = "fail" if failed else "insufficient" if insufficient else "pass"
    return {"rule": RULE, "status": status, "passed": status == "pass", "failed_sets": failed,
            "insufficient": insufficient, "bound": bound,
            "high_entropy_positions": high_entropy, "mean_abs_dlogprob": rows,
            "kl": {"spec": spec["sets"].get("sampled", {}).get("kl"), "ar": ar["sets"].get("sampled", {}).get("kl")}}


def difference_interval(spec_positions, ar_positions):
    """95% request-clustered bootstrap interval of the candidate's mean
    |dlogprob| minus the control's; the arms' positions are not paired."""
    def groups(positions):
        clusters = {}
        for p in positions:
            clusters.setdefault(p["request"], []).append(p["difference"])
        return list(clusters.values())
    left, right = groups(spec_positions), groups(ar_positions)
    if len(left) < 2 or len(right) < 2:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    differences = sorted(
        statistics.mean([v for _ in left for v in generator.choice(left)])
        - statistics.mean([v for _ in right for v in generator.choice(right)])
        for _ in range(BOOTSTRAP_ROUNDS))
    return [differences[int(0.025 * len(differences))], differences[int(0.975 * len(differences)) - 1]]


def pinned_binding(name=BINDING_INPUT):
    """The binding block of the gate's pinned reference `name`, read through
    the input lock."""
    from common import input_bytes
    binding = json.loads(input_bytes(name)).get("binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("key"), dict):
        raise ValueError(f"the pinned {name} reference has no binding to compute continuations as it was")
    binding_modes(binding)
    return binding


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_command(binding, checkpoint, workload_path, output, devices):
    """The reference runner's command for one workload, in the pinned
    reference's modes, lock and adapter."""
    adapters = [path for path in binding["key"]["adapter"]["files"] if not path.startswith("tools/refrun/")]
    if len(adapters) != 1:
        raise ValueError(f"the pinned reference names no single model adapter: {adapters}")
    runs = binding_modes(binding)
    lock = binding["lock"]
    python = os.environ.get("CHOPINFER_REFRUN_PYTHON")
    if not python:
        raise ValueError("set CHOPINFER_REFRUN_PYTHON to the reference runner's environment")
    return [python, "-B", "-m", "tools.refrun", "run", "--adapter", adapters[0], "--checkpoint", str(checkpoint),
            "--lock", lock["path"], *(["--lock-variant", lock["variant"]] if lock.get("variant") else []),
            "--workload", str(workload_path), "--modes", ",".join(runs),
            *[word for label, run in runs.items() for word in ("--dequant", f"{label}={run['dequant']}")],
            "--devices", ",".join(f"cuda:{index}" for index in range(devices)), "--output", str(output)]


def entry_origins(output, started):
    """Per mode, the entries this run computed and those the reference cache
    held, the latter with the provenance they were stored with."""
    from common import refcache
    cache = refcache()
    run = json.loads((output / "run.json").read_text())
    store = cache.Store(run["cache"]) if run.get("cache") else None
    origins = {}
    for mode in MODE_VARIANTS:
        computed, cached = [], []
        for entry in json.loads((output / f"{mode}.json").read_text())["entries"]:
            found = store.get(entry["key_sha256"]) if store is not None else None
            provenance = (found or {}).get("entry", {}).get("provenance", {})
            row = {"name": entry["name"], "key_sha256": entry["key_sha256"],
                   **{key: provenance.get(key) for key in ("created", "devices", "git_head")}}
            (computed if found is None or (provenance.get("created") or 0) >= started else cached).append(row)
        origins[mode] = {"computed": len(computed), "cache_hits": len(cached), "cached": cached}
    return origins


def reference(args):
    """One arm's reference on this process's GPUs, with its execution record."""
    binding = pinned_binding()
    value = json.loads(args.record.read_text())
    args.out.mkdir(parents=True)
    workload_path = args.out / "workload.json"
    workload_path.write_text(json.dumps(workload(value, binding["model"])) + "\n")
    visible = [device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device]
    if not visible:
        raise SystemExit("refused: no GPUs are visible (CUDA_VISIBLE_DEVICES)")
    output = args.out / "refrun"
    command = reference_command(binding, args.checkpoint, workload_path, output, len(visible))
    # The execution record names the GPUs the reference ran on: a query that
    # fails or lists none refuses before the reference runner starts.
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,"
                                "utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"refused: the GPU query failed ({error}); the reference did not run")
    gpus = query.stdout.strip().splitlines()
    if query.returncode != 0 or not gpus:
        raise SystemExit(f"refused: the GPU query failed (nvidia-smi exit {query.returncode}, "
                         f"{len(gpus)} GPUs listed: {query.stderr.strip()}); the reference did not run")
    started_at, started, clock = utc(), time.time(), time.monotonic()
    with (args.out / "refrun.log").open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    record = {"schema": EXECUTION_SCHEMA, "arm": value["arm"], "record_sha256": file_sha256(args.record),
              "workload_sha256": file_sha256(workload_path), "command": command,
              "cuda_visible_devices": visible, "gpus": gpus, "load_average": list(os.getloadavg()),
              "started": started_at, "finished": utc(),
              "seconds": round(time.monotonic() - clock, 1), "exit": code}
    if code == 0:
        run = json.loads((output / "run.json").read_text())
        record["refrun"] = {key: run.get(key) for key in ("devices", "git_head", "load", "modes")}
        record["entries"] = entry_origins(output, started)
    (args.out / "execution.json").write_text(json.dumps(record, indent=1) + "\n")
    raise SystemExit(code)


def public(report):
    return {key: value for key, value in report.items() if not key.startswith("_")}


def authenticate(bound, binding, config):
    """Refuse captures or a reference of another checkpoint than the gate's:
    the arms' bound checkpoint and served name are the gate's pin, and the
    pinned reference's model and checkpoint lock are the gate's."""
    pin = config["qualification"]["checkpoint"]
    if bound["checkpoint_identity"] != pin["checkpoint_identity"] or bound["model"] != pin["served_model_name"]:
        raise ValueError(f"the arms served {bound['checkpoint_identity']} as {bound['model']}, not the gate's "
                         f"pinned checkpoint")
    if binding.get("model") != config["model"] \
            or binding["key"]["checkpoint"]["source_lock_sha256"] != pin["source_lock_sha256"]:
        raise ValueError("the pinned reference is of another model or checkpoint lock than the gate's")


def judge(spec_path, spec_references, ar_path, ar_references):
    """The check's receipt: the verdict, both arms' reports, each reference's
    GPU execution and this CPU scoring, kept apart."""
    from common import qualification_config
    started_at, clock = utc(), time.monotonic()
    binding = pinned_binding()
    spec = score(json.loads(Path(spec_path).read_text()), load_modes(spec_references, binding), binding)
    ar = score(json.loads(Path(ar_path).read_text()), load_modes(ar_references, binding), binding)
    result = compare(spec, ar, prompt_sets())
    authenticate(result["bound"], binding, qualification_config())
    return {**result, "spec": public(spec), "ar": public(ar),
            "execution": {"gpu": {"spec": execution(spec_references), "ar": execution(ar_references)},
                          "cpu_scoring": {"started": started_at, "seconds": round(time.monotonic() - clock, 1),
                                          "python": sys.version.split()[0], "load_average": list(os.getloadavg()),
                                          "binding_input": BINDING_INPUT, "refset_sha256": binding.get("refset_sha256"),
                                          "inputs_sha256": {"spec": file_sha256(spec_path), "ar": file_sha256(ar_path)}}}}


def main():
    from common import arguments
    parser = arguments(__doc__, model_facts=lambda args: args.command == "record")
    parser.add_argument("command", choices=("record", "workload", "reference", "score", "compare"))
    parser.add_argument("--arm", choices=("chopinfer-ar", "chopinfer-spec"))
    parser.add_argument("--run-id")
    parser.add_argument("--server-log", type=Path,
                        help="record: the server's startup log; when given, it must be the live server's")
    parser.add_argument("--record", type=Path, help="workload, score: an arm's record")
    parser.add_argument("--reference-model", help="workload: the model name the reference adapter expects")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="record: requests in flight at once (default: a whole set)")
    parser.add_argument("--sets", help="workload: comma-separated sets (default: all)")
    parser.add_argument("--reference", type=Path, action="append",
                        help="score: a reference run's output directory (repeatable)")
    parser.add_argument("--spec", type=Path, help="compare: the speculative record")
    parser.add_argument("--spec-reference", type=Path, action="append",
                        help="compare: a reference run of the speculative arm (repeatable)")
    parser.add_argument("--ar", type=Path, help="compare: the AR record")
    parser.add_argument("--ar-reference", type=Path, action="append",
                        help="compare: a reference run of the AR arm (repeatable)")
    parser.add_argument("--out", type=Path, help="workload: where to write it; reference: a new directory")
    parser.add_argument("--checkpoint", type=Path, help="reference: the checkpoint directory")
    args = parser.parse_args()
    if args.command == "record":
        if not args.arm or not args.run_id:
            parser.error("record needs --arm and --run-id")
        record(args)
    elif args.command == "workload":
        if not args.record or not args.reference_model or not args.out:
            parser.error("workload needs --record, --reference-model and --out")
        names = args.sets.split(",") if args.sets else None
        args.out.write_text(json.dumps(workload(json.loads(args.record.read_text()), args.reference_model, names))
                            + "\n")
    elif args.command == "reference":
        if not args.record or not args.out or not args.checkpoint:
            parser.error("reference needs --record, --checkpoint and --out")
        reference(args)
    elif args.command == "score":
        if not args.record or not args.reference:
            parser.error("score needs --record and --reference")
        binding = pinned_binding()
        json.dump(public(score(json.loads(args.record.read_text()), load_modes(args.reference, binding), binding)),
                  sys.stdout, indent=1)
        print()
    else:
        if not all((args.spec, args.spec_reference, args.ar, args.ar_reference)):
            parser.error("compare needs --spec, --spec-reference, --ar and --ar-reference")
        json.dump(judge(args.spec, args.spec_reference, args.ar, args.ar_reference), sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
