"""Score verify, decode and prefill rows at the same positions against exact.

The server must run with `serve --debug-reference-tokens FILE` naming the file
`references` writes, and with `--drafter none` so drafts come from the host.
Serving refuses to start while the retired CHOPINFER_DEBUG_REFERENCE_TOKENS
variable is set.
The server then drafts the reference continuation and emits the reference
token on every row, reporting that token's raw logprob, so a completion from
a reference prefix scores every later position of the text (teacher forcing).

`record` runs every text of the exact reference from `score_from + phase`
for each phase, so with K+1 phases each position is scored once at every
verify depth. Where references that share a prefix diverge, the server
cannot force a token; requests stop before such a position and resume after
it (`forced_runs`), so the positions a record scores follow from the
reference alone. The first token of a request comes from the last prefill row;
after it each step verifies c = min(K, remaining - 1) drafts rounded down to
a verify-row bucket and emits c + 1 tokens, rows 0..c. A server without
speculation scores every later position with a decode row. Each record also
holds the prompt-logprob (prefill) scores of the same texts. At concurrency 1
the speculation counters must match the simulated steps exactly.

`report` pairs an AR record with verify records (or another AR record, for
example at higher concurrency) and writes, pooled and per verify depth, those
rows, the AR decode rows and the prefill rows at the same positions against
exact, their distance from AR, and teacher.py's relative gate with the rows
under test as the server and AR decode as the baseline. Each `--control` is
a further AR record held to what `--ar` must meet; a position then pairs
only where every draw has a decode row, and its AR error is the mean over
`--ar` and the controls, while the top-1 counts keep `--ar`'s rows.
With `--pool-verify` the `--verify` records are draws of one candidate (for
example speculative rows at several item buckets): the mean rule reads the
mean of their row errors at the positions every draw pairs, while each record
keeps its own top-1 bound and coverage.

`--exact` picks an explicit reference file. Otherwise the gate configuration
selects the reference. Verify row buckets also come from the gate configuration.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from common import (Client, arguments, bind_checkpoint, bound_live, confirm_live, input_path, pinned_kind,
                    pinned_reference, qualification_config, read_config, receipt)
import declaration
from spec_metrics import difference, metrics_text, snapshot
from step_overlap_ab import step_lines
from teacher import EXACT_VARIANTS, clustered_interval, relative, top_one

TOP_LOGPROBS = 5
# The pooled gate allows verify rows to sit this much farther from exact than
# AR decode rows at the same positions. On the row-local deterministic build
# each AR decode bucket is one fixed draw: at C1, B8, B16, C32 and B64 the AR
# rows' mean error is 0.0538 to 0.0600 (sd 0.0026), so one control moves
# verdicts. Against the mean of four draws (--control) a candidate's gap has
# an sd of about 0.0029, against 0.0037 for one draw.
ROW_MEAN_MARGIN = 0.005
# Verify rows from host-forced drafts are scored once per phase and depth;
# such a record pairs at least this many rows, the teacher reference's
# scored positions. Decode-row records instead must pair exactly the
# positions forcing can reach (coverage).
HOST_DRAFTED_MINIMUM_POSITIONS = 4108
def exact_path(value):
    """Use a reference path, an explicitly named gate, or the selected gate."""
    if value is not None:
        try:
            declaration.split(value)
            gate = declaration.named(value)
        except ValueError:
            gate = None
        if gate is not None and gate.is_file():
            return input_path("exact", config=read_config(str(gate.resolve())))
        else:
            return Path(value)
    return input_path("exact")


def exact_pinned():
    """Whether the gate pins its exact input: MiMo's teacher reference,
    Qwen3.8's bound F32 teacher reference."""
    return pinned_kind(input_path("exact")) is not None


def pins_exact(path):
    """Whether `path` is the gate's pinned exact input, which must then hold
    the pinned bytes."""
    return exact_pinned() and Path(path).resolve() == input_path("exact").resolve()


def load_exact(path):
    """The reference at `path`, checked against the gate's pin when it is the
    pinned exact input, so records cannot be scored against a stale one."""
    if pins_exact(path):
        try:
            return pinned_reference(pinned_kind(path), path)
        except ValueError as error:
            raise SystemExit(f"refused: {error}")
    return json.loads(path.read_text())


def variants(exact):
    """The exact variants every text of the reference has."""
    return [variant for variant in EXACT_VARIANTS
            if all(variant in text for text in exact["texts"])]


def draft_count(k, remaining, buckets=None):
    """Drafts the engine verifies with `remaining` tokens left, as a bucketed K."""
    if buckets is None:
        buckets = qualification_config()["qualification"]["verify_buckets"]
    proposed = min(k, remaining - 1)
    if proposed <= 0:
        return 0
    return max((rows for rows in buckets if rows <= proposed + 1), default=1) - 1


def row_labels(k, count, available=None):
    """The row kind of each of `count` tokens a fully accepted request emits.

    `available(emitted)` is how many reference tokens the host drafter can
    propose after `emitted` tokens; where references share a prefix it is
    shorter than the rest of the text.
    """
    labels = [{"row": "prefill"}] if count else []
    while len(labels) < count:
        drafts = draft_count(k, count - len(labels))
        if available is not None:
            drafts = min(drafts, available(len(labels)))
        if drafts == 0:
            labels.append({"row": "decode"})
            continue
        labels.extend({"row": "verify", "drafts": drafts, "depth": depth}
                      for depth in range(drafts + 1))
    return labels


def agreed_continuations(texts, ids):
    """For each history length h, the reference tokens after ids[:h] on which
    every reference that starts with ids[:h] agrees (the host drafter's reach)."""
    agreed = []
    for h in range(len(ids)):
        matches = [text["prompt_token_ids"] for text in texts
                   if len(text["prompt_token_ids"]) > h and text["prompt_token_ids"][:h] == ids[:h]]
        length = len(ids) - h
        for other in matches:
            same = 0
            while same < length and h + same < len(other) and other[h + same] == ids[h + same]:
                same += 1
            length = min(length, same)
        agreed.append(length)
    return agreed


def expected_counters(labels, emitted, matched):
    """Verify items, drafts and rows the engine counts for one request with
    these row labels. After the first unforced token the history leaves the
    reference, so the host drafts nothing more."""
    items = drafts = 0
    for label in labels[:min(emitted, matched + 1)]:
        if label["row"] == "verify" and label["depth"] == 0:
            items += 1
            drafts += label["drafts"]
    return {"verify_items": items, "draft_tokens": drafts, "rows": items + drafts}


def forced_runs(agreed, start, end):
    """(start, count) of each request a record makes from `start` to `end`.

    The server forces a token only where every reference sharing the history
    agrees on it (agreed[position] >= 1). At a position where they diverge
    it samples its own token, so a request ends before it and the next one
    starts after it; which positions a record scores then depends only on the
    references, not on what the server samples there."""
    while start < end:
        count = 0
        while start + count < end and agreed[start + count] >= 1:
            count += 1
        if count:
            yield start, count
        start += count + 1


def forced_request(client, model, ids, start, count=None):
    """`count` tokens (default: the rest) from ids[:start]; returns the
    reference-matching prefix."""
    count = len(ids) - start if count is None else count
    body = {"model": model, "prompt": ids[:start], "max_tokens": count,
            "temperature": 0, "top_p": 1, "logprobs": TOP_LOGPROBS, "return_token_ids": True,
            "ignore_eos": True, "skip_special_tokens": False}
    response, seconds = client.request("/v1/completions", body)
    choice = response["choices"][0]
    emitted = choice.get("token_ids") or []
    logprobs = choice["logprobs"]
    if len(logprobs["token_logprobs"]) != len(emitted):
        raise ValueError(f"{len(emitted)} tokens but {len(logprobs['token_logprobs'])} logprobs")
    matched = 0
    while matched < len(emitted) and emitted[matched] == ids[start + matched]:
        matched += 1
    return {"start": start, "maximum": count, "emitted": len(emitted), "matched": matched,
            "finish_reason": choice["finish_reason"], "seconds": seconds,
            "logprob": logprobs["token_logprobs"][:matched],
            "top": logprobs["top_logprobs"][:matched]}


def record_pass(client, model, text, k, phase, agreed):
    """Teacher-forces one text from score_from + phase, one request per run
    of forced positions (forced_runs)."""
    ids = text["prompt_token_ids"]
    positions, requests = {}, []
    for start, count in forced_runs(agreed, text["score_from"] + phase, len(ids)):
        result = forced_request(client, model, ids, start, count)
        if result["matched"] != count:
            raise RuntimeError(f"{text['name']}: the server left the reference at position "
                               f"{start + result['matched']}; start it with --debug-reference-tokens")
        labels = row_labels(k, count, lambda emitted, start=start: agreed[start + emitted])
        for offset in range(count):
            positions[start + offset] = {"logprob": result["logprob"][offset],
                                         "top": result["top"][offset], **labels[offset]}
        requests.append({**{key: result[key] for key in
                            ("start", "maximum", "emitted", "matched", "finish_reason", "seconds")},
                         "expected": expected_counters(labels, result["emitted"], result["matched"])})
    return {"phase": phase, "requests": requests,
            "positions": {str(position): row for position, row in sorted(positions.items())}}


def prefill_capture(client, model, text):
    body = {"model": model, "prompt": text["prompt_token_ids"], "echo": True, "logprobs": TOP_LOGPROBS,
            "prompt_logprobs": TOP_LOGPROBS, "return_token_ids": True, "max_tokens": 1,
            "temperature": 0}
    response, _ = client.request("/v1/completions", body)
    choice = response["choices"][0]
    rows = choice.get("prompt_logprobs") or response.get("prompt_logprobs")
    ids = text["prompt_token_ids"]
    captured = {}
    for position in range(text["score_from"], len(ids)):
        row = rows[position]
        captured[str(position)] = {"logprob": row[str(ids[position])]["logprob"],
                                   "top1": top_one(row)}
    return captured


def detokenized(client, model, ids):
    strings = {}
    for token in sorted(set(ids)):
        response, _ = client.request("/detokenize", {"model": model, "tokens": [token]})
        strings[str(token)] = response["prompt"]
    return strings


# AR decode buckets trace evidence can establish: the 8- and 16-row graphs
# that further AR draws use as controls, and the 32-row path and wider.
DECODE_BUCKETS = (8, 16, 32, 64, 128, 256)


def decode_bucket_coverage(steps, bucket):
    """Require every decoding step in the capture window to use this bucket.

    The runner holds filler streams until capture ends. Prefill-only steps
    prove nothing; mixed verify steps cannot establish an AR decode bucket.
    """
    decoded = [step for step in steps if not step.get("prefill_tokens", 0)
               and (step.get("decode_rows", 0) or step.get("verify_items", 0))]
    invalid = [step.get("batch") for step in decoded
               if step.get("verify_items", 0)
               or step.get("row_capacity") != bucket]
    return {"bucket": bucket, "steps": len(decoded), "invalid_batches": invalid,
            "mixed_prefill_steps": sum(bool(step.get("prefill_tokens", 0)) and
                                       bool(step.get("decode_rows", 0)) for step in steps),
            "passed": bool(decoded) and not invalid}


def decode_bucket_trace(trace, bucket):
    return {**decode_bucket_coverage(step_lines(trace.decode(errors="replace")), bucket),
            "sha256": hashlib.sha256(trace).hexdigest()}


def bucket_matches(recorded, bucket):
    evidence = recorded.get("decode_bucket", {})
    return bucket is None or (evidence.get("bucket") == bucket and evidence.get("passed") is True
                              and type(evidence.get("steps")) is int and evidence["steps"] > 0
                              and evidence.get("invalid_batches") == [])


def row_capacity_coverage(observed, required, speculative):
    """Require executed capacities in committed target steps, never requested concurrency.

    Mixed prefill steps are absent from the serving histogram. A speculative
    capacity needs verification rows and drafting activity; AR needs pure
    decode activity. Missing or dropped observations cannot establish coverage.
    """
    observed_only = required is None
    required = sorted(set(required or []))
    histogram = observed.get("step_rows", {}) if isinstance(observed, dict) else {}
    histogram = histogram if isinstance(histogram, dict) else {}
    valid = (histogram.get("available") is True and type(histogram.get("dropped")) is int
             and histogram["dropped"] == 0 and isinstance(histogram.get("bins"), list))
    counts = {}
    seen = set()
    for row in histogram.get("bins", []) if isinstance(histogram.get("bins"), list) else []:
        if not isinstance(row, dict) or not all(type(row.get(key)) is int and row[key] >= 0
                for key in ("decode_rows", "verify_rows", "row_capacity", "steps")):
            valid = False
            continue
        decode, verify, capacity, steps = (row[key] for key in
                                           ("decode_rows", "verify_rows", "row_capacity", "steps"))
        key = (decode, verify, capacity)
        if key in seen or not steps or not decode + verify or capacity < decode + verify:
            valid = False
        seen.add(key)
        if not speculative and verify:
            valid = False
        if (verify if speculative else decode and not verify) and steps > 0:
            counts[capacity] = counts.get(capacity, 0) + steps
    if speculative:
        valid = valid and all(type(observed.get(key)) is int and observed[key] > 0
                              for key in ("verify_items", "draft_tokens"))
    else:
        valid = valid and not any(observed.get(key, 0) for key in ("verify_items", "draft_tokens"))
    missing = [capacity for capacity in required if not counts.get(capacity)]
    return {"required": required, "observed": counts, "missing": missing,
            "complete_observations": bool(valid),
            "passed": bool(valid and counts and (observed_only or required) and not missing)}


def record(args):
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    if exact_pinned():
        try:
            bind_checkpoint(server, args.server_log, args.url)
        except ValueError as error:
            raise SystemExit(f"refused: {error}")
    model = server["model"]
    path = exact_path(args.exact)
    exact = load_exact(path)
    texts = [text for text in exact["texts"]
             if not args.texts or text["name"] in args.texts.split(",")]
    phases = range(args.phases if args.phases is not None else args.k + 1)
    before = snapshot(metrics_text(client), "chopinfer", model)
    pooled = getattr(args, "pooled_decode", False)
    label_k = 0 if pooled else args.k
    jobs = [(text, phase) for text in texts for phase in phases]
    bucket = getattr(args, "decode_bucket", None)
    offset = args.server_log.stat().st_size if bucket else None
    with ThreadPoolExecutor(args.concurrency) as pool:
        agreed = {text["name"]: agreed_continuations(exact["texts"], text["prompt_token_ids"])
                  for text in texts}
        passes = list(pool.map(lambda job: record_pass(client, model, job[0], label_k, job[1],
                                                       agreed[job[0]["name"]]), jobs))
    trace = None
    if bucket:
        with args.server_log.open("rb") as log:
            log.seek(offset)
            captured_trace = log.read()
            trace = {**decode_bucket_trace(captured_trace, bucket),
                     "start_byte": offset, "end_byte": log.tell()}
    after = snapshot(metrics_text(client), "chopinfer", model)
    observed = difference(before, after)
    expected = {"verify_items": 0, "draft_tokens": 0, "rows": 0}
    for recorded in passes:
        for request in recorded["requests"]:
            counts = request["expected"]
            expected = {key: expected[key] + counts[key] for key in expected}
    counters_match = all(observed.get(key, 0) == value for key, value in expected.items())
    # At mixed MTP/AR concurrency, per-request verify depth is unavailable.
    # Score every post-prefill output position without inventing depths.
    # The counters include any filler, so they establish activity only.
    passed = (observed.get("draft_tokens", 0) > 0 and observed.get("verify_items", 0) > 0
              if pooled else counters_match or args.concurrency > 1)
    if trace is not None:
        passed = passed and trace["passed"]
    capacities = getattr(args, "row_capacity", None)
    required_shapes = getattr(args, "require_row_shapes", False)
    capacity = row_capacity_coverage(observed, capacities, args.k > 0) if capacities or required_shapes else None
    if capacity is not None:
        passed = passed and capacity["passed"]
    by_text = {}
    for (text, _), recorded in zip(jobs, passes):
        by_text.setdefault(text["name"], []).append(recorded)
    prefill = {text["name"]: prefill_capture(client, model, text) for text in texts}
    top1_ids = [token for text in texts for token in text["exact"]["top1"]]
    result = {
        "passed": passed, "arm": args.arm, "k": args.k, "pooled_decode": pooled,
        "concurrency": args.concurrency, "top_logprobs": TOP_LOGPROBS,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "speculation": {"observed": observed, "expected": None if pooled else expected,
                        "match": None if pooled else counters_match},
        "texts": [{"name": text["name"], "category": text["category"],
                   "passes": by_text[text["name"]]} for text in texts],
        "prefill": prefill,
        "detokenized": detokenized(client, model, top1_ids),
        "skipped": [] if pooled or counters_match or args.concurrency == 1
        else ["row labels are simulated; counters differ at concurrency above 1"],
    }
    if trace is not None:
        result["decode_bucket"] = trace
    if capacity is not None:
        result["row_capacity_coverage"] = capacity
    confirm_live(client, server)
    receipt(f"verify-exact-{args.arm}", server, result)
    raise SystemExit(0 if result["passed"] else 1)


def server_top1(row):
    """Whether the row's top-1 is the forced reference token, and the top-1 string.

    Output logprobs name alternatives by string, so the reference token is
    recognized by its logprob: it is top-1 when no alternative is higher.
    """
    best = max(row["top"], key=row["top"].get) if row["top"] else None
    return best is None or row["logprob"] >= row["top"][best], best


def disagrees(row, reference, exact_top1, strings):
    """Whether the row's top-1 differs from the exact top-1 token."""
    is_reference, best = server_top1(row)
    if exact_top1 == reference:
        return not is_reference
    return is_reference or best != strings.get(str(exact_top1))


def same_top1(left, right):
    left_reference, left_best = server_top1(left)
    right_reference, right_best = server_top1(right)
    return left_reference == right_reference and (left_reference or left_best == right_best)


def rows_by_position(recorded, kind, drafts=None):
    """{(text, position): row} of one row kind, with at most one row per position."""
    rows = {}
    for text in recorded["texts"]:
        for scored in text["passes"]:
            for position, row in scored["positions"].items():
                if row["row"] != kind or (drafts is not None and row.get("drafts") != drafts):
                    continue
                key = (text["name"], int(position), row.get("depth", 0))
                rows.setdefault(key, row)
    return rows


def summarize(values, texts):
    if not values:
        return None
    return {"mean": statistics.mean(values), "ci95": clustered_interval(values, texts)
            if len(set(texts)) > 1 else None}


def decode_rows(recorded):
    return {(name, position): row
            for (name, position, _), row in rows_by_position(recorded, "decode").items()}


def paired_positions(ar, other, exact, kind, variant, controls=()):
    """Rows of `kind` in `other` paired with AR decode rows at the same position.

    With further AR `controls`, a position pairs only where every control has
    a decode row, and its `ar` error is the mean over `ar` and the controls;
    the top-1 comparisons keep `ar`'s row."""
    reference = {text["name"]: text for text in exact["texts"]}
    strings = {**ar["detokenized"], **other["detokenized"]}
    decode = decode_rows(ar)
    control_rows = [decode_rows(control) for control in controls]
    paired = []
    for (name, position, depth), row in sorted(rows_by_position(other, kind).items()):
        if (name, position) not in decode or any((name, position) not in rows for rows in control_rows):
            continue
        text = reference[name]
        index = position - text["score_from"]
        token = text["prompt_token_ids"][position]
        exact_logprob = text[variant]["logprob"][index]
        exact_top1 = text[variant]["top1"][index]
        ar_row = decode[(name, position)]
        errors = [abs(rows[(name, position)]["logprob"] - exact_logprob)
                  for rows in [decode, *control_rows]]
        prefill = other["prefill"][name][str(position)]
        paired.append({
            "text": name, "position": position, "depth": depth,
            "row": abs(row["logprob"] - exact_logprob),
            "ar": sum(errors) / len(errors),
            "prefill": abs(prefill["logprob"] - exact_logprob),
            "row_vs_ar": abs(row["logprob"] - ar_row["logprob"]),
            "row_flip": disagrees(row, token, exact_top1, strings),
            "ar_flip": disagrees(ar_row, token, exact_top1, strings),
            "prefill_flip": prefill["top1"] != exact_top1,
            "row_vs_ar_flip": not same_top1(row, ar_row),
            **({"draws": errors} if controls else {})})
    return paired


def aggregate(paired):
    """Means, top-1 rates and the teacher-style relative gate of paired rows."""
    count = len(paired)
    if not count:
        return {"positions": 0}
    mean = {key: statistics.mean(p[key] for p in paired)
            for key in ("row", "ar", "prefill", "row_vs_ar")}
    flips = {key: sum(p[key] for p in paired)
             for key in ("row_flip", "ar_flip", "prefill_flip", "row_vs_ar_flip")}
    gap = [p["row"] - p["ar"] for p in paired]
    texts = [p["text"] for p in paired]
    # teacher.relative compares two engines on the same exact positions; here
    # the row kind under test plays the server and AR decode the baseline.
    gate = relative([{"difference": p["row"], "disagrees": p["row_flip"], "text": p["text"]}
                     for p in paired],
                    [{"difference": p["ar"], "disagrees": p["ar_flip"], "text": p["text"]}
                     for p in paired])
    # Pooled depths score each position once per depth, and a position's
    # flips are correlated across depths, so the flip counts' spread grows
    # with the replication; at worst it is sqrt(replication) times larger.
    replication = count / len({(p["text"], p["position"]) for p in paired})
    gate["replication"] = replication
    gate["top1_margin"] *= math.sqrt(replication)
    gate["mean_margin"] = ROW_MEAN_MARGIN
    gate["passed"] = (gate["mean_abs_dlogprob_gap"] <= ROW_MEAN_MARGIN
                      and gate["top1_server_only"] - gate["top1_baseline_only"] <= gate["top1_margin"])
    return {"positions": count, "mean_abs_dlogprob": mean,
            "top1_disagreement": {key: value / count for key, value in flips.items()},
            "top1_counts": flips,
            "row_minus_ar": {"mean": statistics.mean(gap),
                             "ci95": clustered_interval(gap, texts) if len(set(texts)) > 1 else None},
            "gate_against_ar": gate}


def compare(ar, other, exact, kind="verify", variant="exact", controls=()):
    """Per depth and pooled: `kind` rows, AR decode (the mean over `ar` and
    any `controls`) and prefill against exact."""
    paired = paired_positions(ar, other, exact, kind, variant, controls)
    depths = sorted({p["depth"] for p in paired})
    report = {"all": aggregate(paired),
              "by_depth": {str(depth): aggregate([p for p in paired if p["depth"] == depth])
                           for depth in depths}}
    # The first token of each request comes from the last prefill row, which
    # uses the decode logits head; prompt logprobs use the F32 head.
    reference = {text["name"]: text for text in exact["texts"]}
    last = []
    for (name, position, _), row in rows_by_position(other, "prefill").items():
        text = reference[name]
        exact_logprob = text[variant]["logprob"][position - text["score_from"]]
        last.append((abs(row["logprob"] - exact_logprob),
                     abs(other["prefill"][name][str(position)]["logprob"] - exact_logprob)))
    report["prefill_last_row"] = {
        "positions": len(last),
        "mean_abs_dlogprob": {
            "last_prefill_row": statistics.mean(a for a, _ in last) if last else None,
            "prompt_logprobs_f32": statistics.mean(b for _, b in last) if last else None}}
    return report


def require_equal_top1(compared):
    """The strict check the relative gate is not: every paired row picks the
    same top-1 token as AR decode at its position."""
    return compared["all"].get("top1_counts", {}).get("row_vs_ar_flip", 0) == 0


def relabel(recorded, exact):
    """Recomputes row labels from each request, so records written before the
    drafter's reach was modelled score every row by its true kind."""
    texts = {text["name"]: text for text in exact["texts"]}
    for text in recorded["texts"]:
        agreed = agreed_continuations(exact["texts"], texts[text["name"]]["prompt_token_ids"])
        for scored in text["passes"]:
            for request in scored["requests"]:
                labels = row_labels(0 if recorded.get("pooled_decode") else recorded["k"], request["maximum"],
                                    lambda emitted, start=request["start"]: agreed[start + emitted])
                for offset in range(request["matched"]):
                    row = scored["positions"][str(request["start"] + offset)]
                    for key in ("drafts", "depth"):
                        row.pop(key, None)
                    row.update(labels[offset])
    return recorded


def record_phases(recorded):
    return sorted({scored["phase"] for text in recorded["texts"] for scored in text["passes"]})


def decode_positions(exact, phases):
    """{(text, position)} a k=0 or pooled record of `exact` with these phases
    scores with a decode row: every row of a forced run but its first, which
    is the request's last prefill row."""
    positions = set()
    for text in exact["texts"]:
        ids = text["prompt_token_ids"]
        agreed = agreed_continuations(exact["texts"], ids)
        for phase in phases:
            for start, count in forced_runs(agreed, text["score_from"] + phase, len(ids)):
                positions.update((text["name"], start + offset) for offset in range(1, count))
    return positions


def coverage(ar, other, exact, kind, compared, controls=()):
    """Whether the comparison scored all it should. Decode rows (k=0 and
    pooled records) must pair exactly the positions forcing reaches in both
    records, derived from the reference; host-drafted verify rows at least
    HOST_DRAFTED_MINIMUM_POSITIONS. Each further AR control, a (path,
    record) pair, must score every position the comparison expects: for
    decode rows the candidate's, for verify rows the AR record's."""
    def detail(expected):
        return [{"path": path, "scored": len(scored), "missing": [list(key) for key in sorted(expected - scored)]}
                for path, scored in ((path, set(decode_rows(control))) for path, control in controls)]

    if kind == "verify":
        positions = compared["all"]["positions"]
        result = {"rule": "host-drafted", "minimum": HOST_DRAFTED_MINIMUM_POSITIONS,
                  "positions": positions, "passed": positions >= HOST_DRAFTED_MINIMUM_POSITIONS}
        if controls:
            result["controls"] = detail(decode_positions(exact, record_phases(ar)))
            result["passed"] = result["passed"] and not any(row["missing"] for row in result["controls"])
        return result
    scored = set(decode_rows(ar)) & set(decode_rows(other))
    for _, control in controls:
        scored &= set(decode_rows(control))
    expected = (decode_positions(exact, record_phases(ar))
                & decode_positions(exact, record_phases(other)))
    result = {"rule": "every forced decode position", "expected": len(expected), "scored": len(scored),
              "missing": [list(key) for key in sorted(expected - scored)],
              "unexpected": [list(key) for key in sorted(scored - expected)],
              "passed": scored == expected}
    if controls:
        result["controls"] = detail(expected)
    return result


def load_control(control_path, exact, digest, live):
    """A further AR draw, held to what the gate requires of `--ar`: an AR
    record that drafted nothing, of the same reference, bound live when the
    reference is pinned."""
    control = relabel(json.loads(Path(control_path).read_text()), exact)
    observed = control.get("speculation", {}).get("observed", {})
    if (control.get("k") != 0 or control.get("pooled_decode")
            or observed.get("draft_tokens", 0) or observed.get("verify_items", 0)):
        raise SystemExit(f"control {control_path} is not an AR record that drafted nothing")
    if control.get("reference_sha256") != digest:
        raise SystemExit("records were scored against a different exact reference")
    if live and not bound_live(control["server"]):
        raise SystemExit("a record's server was not bound live to the pinned checkpoint")
    return control


def recorded_bucket(recorded, required):
    """The decode bucket a record's trace evidence names and, when buckets are
    `required`, whether that evidence passed."""
    bucket = (recorded.get("decode_bucket") or {}).get("bucket")
    result = {"decode_bucket": bucket}
    if required:
        result["bucket_covered"] = bucket is not None and bucket_matches(recorded, bucket)
    return result


def control_draws(args, ar, controls, paired, required):
    """Each AR draw's mean error over the paired positions, and their mean."""
    errors = [statistics.mean(p["draws"][index] for p in paired) if paired else None
              for index in range(len(controls) + 1)]
    draws = [{"role": "ar", "path": args.ar, "arm": ar.get("arm"), "concurrency": ar.get("concurrency"),
              **recorded_bucket(ar, False), "mean_abs_dlogprob": errors[0]}]
    draws += [{"role": "control", "path": path, "arm": control.get("arm"),
               "concurrency": control.get("concurrency"), **recorded_bucket(control, required),
               "mean_abs_dlogprob": error} for (path, control), error in zip(controls, errors[1:])]
    return {"positions": len(paired), "draws": draws,
            "mean_abs_dlogprob": statistics.mean(errors) if paired else None}


def top_one_within(gate):
    """The relative gate's top-1 rule alone."""
    return gate["top1_server_only"] - gate["top1_baseline_only"] <= gate["top1_margin"]


def pooled_candidate(ar, candidates, exact, kind, controls=()):
    """Several (path, record) draws of one candidate scored as one: at each
    position every draw pairs, the mean of their row errors against the AR
    error (the mean over `ar` and any `controls`). The mean rule is
    aggregate's; the text-clustered CI95 is reported, not gated."""
    paired = [{(p["text"], p["position"], p["depth"]): p
               for p in paired_positions(ar, other, exact, kind, "exact", controls)}
              for _, other in candidates]
    shared = sorted(set.intersection(*(set(rows) for rows in paired)))
    draws = [{"path": path, "arm": other.get("arm"), "concurrency": other.get("concurrency"),
              "mean_abs_dlogprob": statistics.mean(rows[key]["row"] for key in shared) if shared else None}
             for (path, other), rows in zip(candidates, paired)]
    result = {"positions": len(shared), "draws": draws, "mean_margin": ROW_MEAN_MARGIN}
    if not shared:
        return {**result, "passed": False}
    gap = [statistics.mean(rows[key]["row"] for rows in paired) - paired[0][key]["ar"] for key in shared]
    result["mean_abs_dlogprob"] = {"candidate": statistics.mean(draw["mean_abs_dlogprob"] for draw in draws),
                                   "ar": statistics.mean(paired[0][key]["ar"] for key in shared)}
    result["row_minus_ar"] = summarize(gap, [name for name, _, _ in shared])
    result["passed"] = result["row_minus_ar"]["mean"] <= ROW_MEAN_MARGIN
    return result


def bound_serving_identity(server):
    """Keep an input arm's own live identity after checking its binding."""
    from common import checkpoint_pin, live_identity
    pin = checkpoint_pin()
    identity = live_identity(server)
    if not bound_live(server) or server.get("model") != pin["served_model_name"] or any(
            identity.get(key) != server["checkpoint"].get(key)
            for key in ("checkpoint_identity", "startup_nonce", "package_identity", "metadata_identity")):
        raise ValueError("verification input health differs from its bound process")
    return dict(identity)


def report(args):
    path = exact_path(args.exact)
    exact = load_exact(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ar = relabel(json.loads(Path(args.ar).read_text()), exact)
    output = {"ar": args.ar, "ar_concurrency": ar["concurrency"], "ar_arm": ar.get("arm"),
              "ar_k": ar.get("k"), "ar_speculation": ar.get("speculation", {}).get("observed", {}), "records": {}}
    bucket = getattr(args, "decode_bucket", None)
    controls = [(control_path, load_control(control_path, exact, digest, pins_exact(path)))
                for control_path in getattr(args, "control", None) or []]
    records = tuple(control for _, control in controls)
    candidates = []
    for record_path in args.verify:
        other = relabel(json.loads(Path(record_path).read_text()), exact)
        if {ar["reference_sha256"], other["reference_sha256"]} != {digest}:
            raise SystemExit("records were scored against a different exact reference")
        if pins_exact(path) and not all(bound_live(value["server"]) for value in (ar, other)):
            raise SystemExit("a record's server was not bound live to the pinned checkpoint")
        kind = "verify" if other["k"] and not other.get("pooled_decode") else "decode"
        candidates.append((record_path, other, kind))
        output["records"][record_path] = {
            "arm": other["arm"], "k": other["k"], "concurrency": other["concurrency"],
            "kind": kind, "pooled_decode": other.get("pooled_decode", False),
            "speculation": other.get("speculation", {}).get("observed", {}),
            **{variant: compare(ar, other, exact, kind, variant, records)
                             for variant in variants(exact)}}
        record = output["records"][record_path]
        if pins_exact(path):
            if "bound" not in output:
                output["bound"] = {"ar": bound_serving_identity(ar.get("server", {})), "candidates": {}}
            output["bound"]["candidates"][record_path] = bound_serving_identity(other["server"])
        record["coverage"] = coverage(ar, other, exact, kind, record["exact"], controls)
        if bucket is not None:
            record["decode_bucket"] = other.get("decode_bucket")
            record["bucket_covered"] = bucket_matches(other, bucket)
            record["ar_decode_bucket"] = ar.get("decode_bucket")
            record["ar_bucket_covered"] = bucket_matches(ar, getattr(args, "ar_decode_bucket", 32))
        capacities = getattr(args, "row_capacity", None)
        ar_capacities = getattr(args, "ar_row_capacity", None)
        required_shapes = getattr(args, "require_row_shapes", False)
        if capacities or required_shapes:
            record["row_capacity_coverage"] = row_capacity_coverage(
                other.get("speculation", {}).get("observed", {}), capacities, other["k"] > 0)
        if ar_capacities or required_shapes:
            record["ar_row_capacity_coverage"] = row_capacity_coverage(
                ar.get("speculation", {}).get("observed", {}), ar_capacities, False)
        if controls:
            # The gate reads the mean over every AR draw; today's statistic
            # against `--ar` alone stays as a diagnostic.
            record["controls"] = control_draws(args, ar, controls,
                                               paired_positions(ar, other, exact, kind, "exact", records),
                                               bucket is not None)
            record["single_control"] = compare(ar, other, exact, kind, "exact")["all"]
    for record in output["records"].values():
        record["equal_top1_to_ar"] = require_equal_top1(record["exact"])
    pool = getattr(args, "pool_verify", False)
    if pool:
        if len({kind for _, _, kind in candidates}) != 1:
            raise SystemExit("--pool-verify needs records of one row kind")
        output["pooled_candidate"] = pooled_candidate(ar, [(path, other) for path, other, _ in candidates],
                                                      exact, candidates[0][2], records)
    json.dump(output, sys.stdout, indent=1)
    print()

    def gated(record):
        # A comparison without shared positions has no gate statistics and fails.
        gate = record["exact"]["all"].get("gate_against_ar")
        if gate is None:
            return False
        # Pooled draws keep their own top-1 bound; the mean rule is the pool's.
        return top_one_within(gate) if pool else gate["passed"] is True

    passed = all(gated(record)
                 and record["coverage"]["passed"]
                 and record.get("bucket_covered", True)
                 and record.get("ar_bucket_covered", True)
                 and record.get("row_capacity_coverage", {}).get("passed", True)
                 and record.get("ar_row_capacity_coverage", {}).get("passed", True)
                 and all(draw.get("bucket_covered", True)
                         for draw in record.get("controls", {}).get("draws", []))
                 and (record["equal_top1_to_ar"]
                      or not getattr(args, "require_equal_top1", False))
                 for record in output["records"].values())
    if pool:
        passed = passed and output["pooled_candidate"]["passed"]
    if exact_pinned() and not pins_exact(path):
        print(f"{path} is not the pinned reference: diagnostic only", file=sys.stderr)
        passed = False
    raise SystemExit(0 if passed else 1)


def references(args):
    exact = load_exact(exact_path(args.exact))
    Path(args.out).write_text(json.dumps(
        {"sequences": [text["prompt_token_ids"] for text in exact["texts"]]}) + "\n")


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("command", choices=("references", "record", "report"))
    parser.add_argument("--exact", help="reference path or gate config name (default: selected gate)")
    parser.add_argument("--out", help="references: the file for serve --debug-reference-tokens")
    parser.add_argument("--arm", help="record: receipt label, e.g. ar or k7")
    parser.add_argument("--server-log", type=Path,
                        help="record: the server's startup log; when given, it must be the live server's")
    parser.add_argument("--k", type=int, default=0,
                        help="record: the server's --speculative-tokens (0 without speculation)")
    parser.add_argument("--phases", type=int, help="record: start offsets (default K + 1)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--decode-bucket", type=int, choices=DECODE_BUCKETS,
                        help="record/report: require trace evidence for this AR decode bucket")
    parser.add_argument("--require-row-shapes", action="store_true",
                        help="record/report: require complete observed AR and speculative batch shapes")
    parser.add_argument("--row-capacity", action="append", type=int,
                        help="record/report: require this observed target row capacity (repeatable)")
    parser.add_argument("--ar-row-capacity", action="append", type=int,
                        help="report: require this observed AR control row capacity (repeatable)")
    parser.add_argument("--ar-decode-bucket", type=int, choices=DECODE_BUCKETS, default=32,
                        help="report: control bucket required with --decode-bucket (default: 32)")
    parser.add_argument("--pooled-decode", action="store_true",
                        help="record: pool post-prefill MTP/AR rows without simulated depths; require drafting activity")
    parser.add_argument("--texts", help="record: comma-separated text names")
    parser.add_argument("--ar", help="report: the AR record")
    parser.add_argument("--control", action="append", default=[],
                        help="report: a further AR record; the gate compares against the mean over --ar and these")
    parser.add_argument("--require-equal-top1", action="store_true",
                        help="report: also fail when any row's top-1 differs from AR's")
    parser.add_argument("--verify", action="append", default=[],
                        help="report: a verify record, or another AR record to compare decode rows")
    parser.add_argument("--pool-verify", action="store_true",
                        help="report: gate the mean over the --verify records' row errors as one candidate")
    args = parser.parse_args()
    if any(capacity <= 0 for capacity in (args.row_capacity or []) + (args.ar_row_capacity or [])):
        parser.error("row capacities must be positive")
    if args.command == "references":
        if not args.out:
            parser.error("references needs --out")
        references(args)
    elif args.command == "record":
        if not args.arm or args.k < 0 or args.concurrency < 1:
            parser.error("record needs --arm, K >= 0 and concurrency >= 1")
        if args.decode_bucket and (not args.server_log or args.k or args.pooled_decode):
            parser.error("decode-bucket needs server-log and an AR record (K=0, no pooled-decode)")
        record(args)
    else:
        if not args.ar or not args.verify:
            parser.error("report needs --ar and --verify")
        report(args)


if __name__ == "__main__":
    main()
