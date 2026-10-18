"""Qualify declared media workloads against bound reference-code scores.

Capture formats are selected by qualification.media, independent of model names.
Per-item scoring rules and near-tie limits stay in the pinned reference.
"""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import tempfile
from functools import lru_cache

from common import (answer_budget, answer_control, thinking_control, Client, arguments, bind_checkpoint, confirm_live,
                    entries_top1, not_applicable, pinned_binding, pinned_reference, pinned_bytes, qualification_config, input_path, resolve_input, read_config,
                    reference_path, receipt, rows_digest, served_entries)
from exact_reference import check_scores, gap_top, top
import limits as rules
import declaration


def media_config(config=None):
    """Add capture-format data to historical policies that predate it.

    Reference pins and thresholds continue to come from the recorded policy.
    """
    config = qualification_config() if config is None else config
    if "media" in config["qualification"]:
        return config
    name = config["model"] + (f".{config['variant']}" if config.get("variant") else "")
    declared = read_config(str(declaration.named(name)))
    return {**config, "qualification": {**config["qualification"],
                                        "media": declared["qualification"]["media"]},
            "inputs": {**declared["inputs"], **config["inputs"]}}


def media_policy():
    return media_config()["qualification"]["media"]


def case_module():
    return cached_case_module(str(input_path("media_cases")))


@lru_cache(maxsize=None)
def cached_case_module(path):
    spec = importlib.util.spec_from_file_location("media_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    return case_module().cases()


def messages(case, fixtures, *, media_bytes=None):
    return case_module().messages(case, fixtures, media_bytes=media_bytes)


GATE_VARIANT = "exact"
# Floors of the teacher-forced limits, raised to each case's own reference
# spread by the spread rule (limits.py, case_limits).
MEAN_ABS_LOGPROB_LIMIT = rules.MEAN_FLOOR
LARGE_SET_MEAN_LIMIT = rules.LARGE_SET_MEAN_FLOOR
LARGE_SET_POSITIONS = rules.LARGE_SET_POSITIONS
TOP1_DISAGREEMENT_LIMIT = rules.TOP1_FLOOR
MEAN_STEP = rules.MEAN_STEP
# Greedy answers may part where the reference ranks the served token within
# the near-tie limit (limits.NEAR_TIE_NATS, 0.3 nats, inclusive) of its
# top-1.
NEAR_TIE_LIMIT = rules.NEAR_TIE_NATS
# An image probe's greedy budget, raised to its reference answer's length: a
# model that reasons first (the answer holds its reasoning) would otherwise
# end inside the reasoning with no content.
PROBE_MAX_TOKENS = 64


def audio_near_tie_limit():
    """The declared encoder spread limit; historical policies use current format data."""
    return media_policy()["audio_near_tie"]


def verified(path, schema):
    from refs import fixture_bytes
    data = fixture_bytes(path)
    manifest = json.loads(fixture_bytes(path.parent / "manifest.json"))
    digest = hashlib.sha256(data).hexdigest()
    if manifest["files"].get(path.name) != digest:
        raise ValueError(f"reference sha256 mismatch: {path}")
    value = json.loads(data)
    if value.get("schema") != schema:
        raise ValueError(f"unsupported multimodal reference schema: {path}")
    expected_ids = [case["id"] for case in cases()]
    actual_ids = [case["id"] for case in value["cases"]]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("multimodal reference case set differs from frozen workload")
    return value, digest, manifest


def media_variants():
    """The variants the gate pins for its media reference: the gate variant
    and the variants whose spread sets each case's limits. Every case
    carries all of them."""
    variants = tuple(qualification_config()["qualification"]["references"][media_policy()["reference"]]["variants"])
    if GATE_VARIANT not in variants or len(variants) < 2:
        raise ValueError(f"the pinned media variants {variants} need {GATE_VARIANT} and a spread variant")
    return variants


def load_reference(path, exact_path=None):
    """The frozen capture's cases, each with its `exact` reference scores when
    `exact_path` is given; the capture's digest and the fixture hashes."""
    reference, digest, manifest = verified(path, media_policy()["capture_schema"])
    from declaration import check_media_evidence
    check_media_evidence(media_policy(), reference["cases"])
    if exact_path is not None:
        spec = qualification_config()["qualification"]["references"][media_policy()["reference"]]
        exact, _, _ = verified(exact_path, spec["schema"])
        if exact["capture_sha256"] != digest:
            raise ValueError("the exact reference scores another capture")
        variants = media_variants()
        for case, scores in zip(reference["cases"], exact["cases"], strict=True):
            if scores.get("id") != case["id"] or scores.get("kind") != case["kind"]:
                raise ValueError("exact reference media kinds or IDs differ from the capture")
            ids = case["forced_prompt_token_ids"][case["answer_token_start"]:]
            if scores["answer_token_ids"] != ids:
                raise ValueError(f"{case['id']}: exact reference scores other answer tokens")
            for name in variants:
                check_scores(scores.get(name), ids, f"{case['id']} {name}")
            case["exact"] = {name: scores[name] for name in variants}
    return reference, digest, manifest["fixture_sha256"]


def gate_flips(gate, served):
    """The near-tie rule's flip (limits.flip) at each answer position of the
    served top-1 tokens `served` against the gate variant's stored top-k."""
    positions = [(top1, token, pairs) for token, top1, pairs in zip(served, gate["top1"], gap_top(gate), strict=True)]
    return rules.flips(positions) if positions else []


def case_limits(expected):
    """The teacher-forced limits of one case: the floors, raised to its
    reference variants scored against the gate variant."""
    gate = expected["exact"][GATE_VARIANT]
    spread = []
    for name, rows in expected["exact"].items():
        if name == GATE_VARIANT:
            continue
        count = len(rows["logprob"])
        spread.append({
            "mean_abs_dlogprob": sum(abs(a - b) for a, b in zip(rows["logprob"], gate["logprob"], strict=True))
            / count,
            **rules.top1_summary(gate_flips(gate, rows["top1"]), count)})
    return rules.spread_limits(spread, len(gate["logprob"]), mean_floor=MEAN_ABS_LOGPROB_LIMIT,
                               large_set_mean_floor=LARGE_SET_MEAN_LIMIT,
                               large_set_positions=LARGE_SET_POSITIONS, top1_floor=TOP1_DISAGREEMENT_LIMIT,
                               mean_step=MEAN_STEP)


def top_one(row):
    if not isinstance(row, dict) or not row:
        raise ValueError("empty top-logprob row")
    return entries_top1(served_entries(row))


def forced_rows(expected, observed):
    """The served rows of the reference answer's positions."""
    ids = expected["forced_prompt_token_ids"]
    rows = observed.get("prompt_logprobs")
    if observed.get("prompt_token_ids") != ids or not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("forced prompt token IDs or logprob rows differ from capture")
    answer = []
    for position in range(expected["answer_token_start"], len(ids)):
        row = rows[position]
        if not isinstance(row, dict) or str(ids[position]) not in row:
            raise ValueError(f"missing answer token logprob at position {position}")
        answer.append(row)
    if not answer:
        raise ValueError("no answer positions scored")
    return answer




def forced_capture(expected, observed):
    """Validated answer columns, retained for offline rescoring."""
    rows = forced_rows(expected, observed)
    ids = expected["forced_prompt_token_ids"][expected["answer_token_start"]:]
    import kinds
    columns = {"logprob": [row[str(token)]["logprob"] for row, token in zip(rows, ids, strict=True)],
               "top1": [top_one(row) for row in rows], "served_rows": [served_entries(row) for row in rows]}
    return columns


def score_forced(expected, observed):
    """The served answer rows within the case's reference-spread limits."""
    import kinds
    columns = forced_capture(expected, observed)
    constants = {key: {"value": globals()[key]} for key in
                 ("GATE_VARIANT", "MEAN_ABS_LOGPROB_LIMIT", "LARGE_SET_MEAN_LIMIT", "LARGE_SET_POSITIONS",
                  "TOP1_DISAGREEMENT_LIMIT", "MEAN_STEP")}
    return kinds.call("score_capture", kinds.KINDS["media_forced"], columns, expected,
                      {"constants": {"multimodal": constants}})["metrics"]




def greedy_agreement(expected, text, *, kind=None, generated_ids=None):
    """Walk the served greedy answer along the reference answer. Every served
    token up to and including the first that departs from the reference
    answer must be the exact reference's top-1 there, or within the near-tie
    limit of it (listed in its top 5). Past the reference answer's end nothing
    is checked. A served answer equal to the reference text is walked on the
    reference tokens, so retokenization cannot move it."""
    import kinds
    capture = {"greedy": {"exact": " ".join(text.split()) == " ".join(expected["answer"].split()),
                           "generated_ids": generated_ids,
                           "near_tie_limit": audio_near_tie_limit() if kind == "audio" else NEAR_TIE_LIMIT}}
    return kinds.call("score_capture", kinds.KINDS["media_forced"], capture, expected,
                      {"constants": {"multimodal": {"GATE_VARIANT": {"value": GATE_VARIANT}}}})["metrics"]


def expected_answer(case, fixture_manifest):
    """The declared manifest clauses, or no expectation for this item."""
    policy = media_policy()["expectations"]
    spec = policy.get("files", {}).get(case.get("file")) or policy.get("kinds", {}).get(case["kind"])
    if spec is None:
        return None
    value = fixture_manifest[spec["field"]]
    return [value] if spec["single"] else value


def ordered_clauses(answer, clauses):
    normalized = " ".join(answer.casefold().split())
    cursor = 0
    for clause in clauses:
        position = normalized.find(clause.casefold(), cursor)
        if position < 0:
            return False
        cursor = position + len(clause)
    return True


def evaluate_case(client, model, case, expected, fixture_dir, fixture_manifest, *, media_bytes=None):
    prompt = messages(case, fixture_dir, media_bytes=media_bytes)
    body = {"model": model, "messages": prompt,
            **answer_control(),
            "temperature": 0, "max_tokens": answer_budget(256), "return_token_ids": True}
    answer, latency = client.request("/v1/chat/completions", body)
    text = answer["choices"][0]["message"].get("content") or ""
    count = answer["usage"]["prompt_tokens"]
    expected_count = len(expected["prompt_token_ids"])
    clauses = expected_answer(case, fixture_manifest)
    result = {"id": case["id"], "kind": case["kind"], "answer": text,
              "reference_answer": expected["answer"],
              "fixture_expectation_match": clauses is None or ordered_clauses(text, clauses),
              "prompt_tokens": count, "reference_prompt_tokens": expected_count,
              "token_count_match": count == expected_count,
              "latency_seconds": latency}
    try:
        forced_body = {**body,
                       "messages": prompt + [{"role": "assistant", "content": expected["answer"]}],
                       "continue_final_message": True, "max_tokens": 1,
                       "prompt_logprobs": 20, "return_token_ids": True}
        forced, _ = client.request("/v1/chat/completions", forced_body)
        result["logprobs"] = score_forced(expected, forced)
        result["greedy"] = greedy_agreement(
            expected, text, kind=case["kind"], generated_ids=answer["choices"][0].get("token_ids"))
    except Exception as error:
        result["logprobs"] = {"passed": False, "error": str(error)}
        result["greedy"] = {"passed": False, "error": str(error)}
    result["passed"] = (result["greedy"]["passed"] and result["fixture_expectation_match"]
                        and result["token_count_match"] and result["logprobs"]["passed"])
    return result


def run(client, server, reference, fixture_dir, fixture_hashes):
    from refs import read_member
    config = qualification_config()
    fixture_manifest = json.loads(read_member(config, "media_fixtures", fixture_dir / "manifest.json"))
    rows = []
    for case, expected in zip(cases(), reference["cases"], strict=True):
        clauses = expected_answer(case, fixture_manifest)
        if clauses is not None and not ordered_clauses(expected["answer"], clauses):
            raise ValueError(f"{case['id']}: reference answer contradicts {clauses}")
        if reason := not_applicable(server, case["kind"]):
            rows.append({"id": case["id"], "kind": case["kind"], "status": "not_applicable",
                         "reason": reason, "passed": None})
            continue
        try:
            media_bytes = None
            if case.get("file"):
                media_bytes = read_member(config, "media_fixtures", fixture_dir / case["file"])
                actual = hashlib.sha256(media_bytes).hexdigest()
                if actual != fixture_hashes[case["file"]]:
                    raise ValueError("fixture sha256 differs from capture")
            rows.append(evaluate_case(client, server["model"], case, expected,
                                      fixture_dir, fixture_manifest, media_bytes=media_bytes))
        except Exception as error:
            rows.append({"id": case["id"], "passed": False, "error": str(error)})
    return applicable({"limits": receipt_limits()}, rows)


def applicable(result, rows):
    """`result` with `rows`, passed when every row the server's modalities
    serve passed. Rows of a modality it does not serve are not applicable:
    their reasons make the receipt's check not applicable, never passed."""
    served = [row for row in rows if row.get("status") != "not_applicable"]
    result.update(passed=bool(served) and all(row["passed"] for row in served),
                  errors=[row for row in served if not row["passed"]], rows=rows)
    if reasons := sorted({row["reason"] for row in rows if row.get("status") == "not_applicable"}):
        result["not_applicable"] = reasons
    return result


def exit_status(result):
    """0 when every applicable row passed, or none failed and the rest are
    not applicable: the gate then judges the receipt not applicable."""
    return 0 if result["passed"] or (result.get("not_applicable") and not result["errors"]) else 1


def receipt_limits(audio_near_tie=None):
    """The floors and near-tie limits a media receipt records."""
    return {"mean_abs_dlogprob_floor": MEAN_ABS_LOGPROB_LIMIT, "top1_decisive_floor": TOP1_DISAGREEMENT_LIMIT,
            "top1_near_tie_floor": rules.NEAR_TIE_FLOOR, "top1_disagreement_floor": rules.TOP1_TOTAL_FLOOR,
            "top1_near_tie": rules.NEAR_TIE_NATS,
            "near_tie": NEAR_TIE_LIMIT, "audio_near_tie": audio_near_tie_limit() if audio_near_tie is None else audio_near_tie}


def load_probes(path=None):
    """The probes, checked against the capture manifest and image hashes."""
    path = path or input_path("media_capture", config=media_config())
    from refs import fixture_bytes
    data = fixture_bytes(path)
    manifest = json.loads(fixture_bytes(path.parent / "manifest.json"))
    digest = hashlib.sha256(data).hexdigest()
    if manifest["files"][path.name]["sha256"] != digest:
        raise ValueError(f"probe sha256 mismatch: {path}")
    probes = json.loads(data)
    names = [probe["name"] for probe in probes]
    if len(set(names)) != len(names):
        raise ValueError("duplicate probe names")
    hashes = json.loads(fixture_bytes(path.parent / "identity.json"))["images"]
    return probes, hashes, digest


def probe_references(capture_sha256, path=None):
    """The declared reference runner's answers on the verified capture,
    in the form the checks read: the answer with its token bytes, the greedy steps
    ([token, logprob, top 5 as [token, logprob, bytes]]), and the forced rows
    ([token, logprob, top-1]) of the answer as a served chat renders it.
    Each free-text probe also gets its limits, the spread rule (limits.py)
    with the reference rule's floors and step over the rule's variants
    (the BF16 and f64 runs) scored against the F32 run like a server, their
    top-1 flips judged by the near-tie rule against the F32 run's stored
    top-k (`top`, which also judges a served answer's flips), and, per
    position, the F32 margin with every variant's top-1 and margin for the
    near-tie report."""
    spec = qualification_config()["qualification"]["references"][media_policy()["reference"]]
    path = path or reference_path(media_policy()["reference"])
    from refs import fixture_bytes
    data = json.loads(fixture_bytes(path))
    if data.get("schema") != spec["schema"]:
        raise ValueError("unsupported image probe reference schema")
    if data["provenance"]["capture_sha256"] != capture_sha256:
        raise ValueError(f"image probe references were measured on capture "
                         f"{data['provenance']['capture_sha256']}, not {capture_sha256}")
    rule, items = data["rule"], data["probes"]
    kinds = {name: item["kind"] for name, item in items.items()}
    wanted = media_policy()["items"]
    if kinds != wanted:
        raise ValueError(f"image probe references must be exactly {wanted}, not {kinds}")
    probes = {name: reference_probe(name, item, rule, data["token_bytes"]) for name, item in items.items()}
    return probes, rule, data["provenance"]


def reference_probe(name, item, rule, token_bytes):
    """One probe of the reference file in the form the checks read."""
    variants = [GATE_VARIANT, *rule["variants"]]
    forced = item.get("answer_token_ids")
    if not isinstance(forced, list) or not forced:
        raise ValueError(f"{name}: no forced answer")
    for variant in variants:
        check_scores(item.get(variant), forced, f"{name} {variant}")
    answer = item["answer"]
    steps = item["greedy" if item["kind"] == "short" else "generated"]
    if steps["answer"] != answer["token_ids"]:
        raise ValueError(f"{name}: the steps generated another answer")

    def tokens(ids):
        return [token_bytes[str(token)] for token in ids]
    gate = item[GATE_VARIANT]
    probe = {"kind": item["kind"], "inputs": item["inputs"],
             "answer": {**answer, "tokens": tokens(answer["token_ids"])},
             "steps": [[token, logprob, [list(row) for row in zip(ids, logprobs, tokens(ids))]]
                       for token, logprob, ids, logprobs in zip(answer["token_ids"], steps["logprob"],
                                                                steps["topk_ids"], steps["topk_logprobs"],
                                                                strict=True)],
             "forced": [list(row) for row in zip(forced, gate["logprob"], gate["top1"], strict=True)],
             "scored": len(forced)}
    if item["kind"] == "free-text":
        spread = [{"mean_abs_dlogprob": statistics.mean(abs(a - b) for a, b in zip(
                       item[variant]["logprob"], gate["logprob"], strict=True)),
                   **rules.top1_summary(gate_flips(gate, item[variant]["top1"]), len(forced))}
                  for variant in rule["variants"]]
        probe["limits"] = rules.spread_limits(spread, len(forced), mean_floor=rule["mean_floor"],
                                              top1_floor=rule["top1_floor"], mean_step=rule["mean_step"])
        probe["spread"] = dict(zip(rule["variants"], spread))
        probe["top"] = gap_top(gate)
        probe["positions"] = {"reference_margin": gate["margin"],
                              "variants": {variant: {"top1": item[variant]["top1"], "margin": item[variant]["margin"]}
                                           for variant in rule["variants"]}}
    return probe


def probe_images(generated, functional, hashes):
    """Probe image name to PNG bytes: blue-square and frame-00 from the functional
    functional fixtures, the rest regenerated by the capture's images.py
    into `generated` (existing files are kept). Each must match the capture."""
    spec = importlib.util.spec_from_file_location(
        "media_capture_images", input_path("media_images"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {name: functional / path for name, path in media_policy()["fixture_images"].items()}
    paths |= module.write_all(generated)
    images = {}
    for name, expected in hashes.items():
        if name in media_policy()["fixture_images"]:
            from refs import read_member
            data = read_member(media_config(), "media_fixtures", paths[name])
        else:
            # Generated images are authenticated by their captured digest below.
            data = paths[name].read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{name}: image sha256 differs from the capture")
        images[name] = data
    return images


def probe_messages(probe, images):
    """The captured request: the images in order, then the question."""
    parts = [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(images[name]).decode("ascii")}}
        for name in probe["images"]]
    return [{"role": "user", "content": parts + [{"type": "text", "text": probe["text"]}]}]


def answer_tokens(choice):
    """The answer's tokens as bytes, ending with the stop token when the
    answer stopped."""
    content = (choice.get("logprobs") or {}).get("content")
    if not isinstance(content, list):
        raise ValueError("server returned no answer logprobs")
    tokens = [bytes(entry["bytes"]) if entry.get("bytes") is not None
              else entry["token"].encode() for entry in content]
    if choice.get("finish_reason") == "stop" and (not tokens or tokens[-1] != media_policy()["stop_token"].encode()):
        tokens.append(media_policy()["stop_token"].encode())
    return tokens


def reference_greedy_agreement(reference, observed, near_tie):
    """Accept a served answer equal to the reference's, or one that follows it
    until a token the reference ranks within `near_tie` of its own there.
    `observed` is a chat completion choice with logprobs."""
    import kinds
    captured = {"probe_greedy": {"tokens": [list(token) for token in answer_tokens(observed)],
                                  "near_tie": near_tie}}
    return kinds.call("score_capture", kinds.KINDS["media_forced"], captured, reference)["metrics"]


def probe_forced_agreement(probe, reference, forced):
    """The reference answer teacher-forced through the server: the mean
    |dlogprob| of its tokens against the reference's own teacher-forced
    logprobs, and the share of positions whose served top-1 differs from the
    reference's top-1 there.
    The reference forced the answer as the chat template renders it (an
    answer's leading space trimmed), so the served tokens must equal its.
    `forced` is the server's response with prompt logprobs and token IDs."""
    ids, rows, prompt = forced.get("prompt_token_ids"), forced.get("prompt_logprobs"), probe["token_ids"]
    if (not isinstance(ids, list) or ids[:len(prompt)] != prompt or not isinstance(rows, list)
            or len(rows) != len(ids)):
        raise ValueError("forced prompt differs from the captured prompt")
    expected = reference["forced"]
    served = ids[len(prompt):]
    if len(served) != len(expected):
        raise ValueError(f"forced answer has {len(served)} tokens, the reference's {len(expected)}")
    logprobs, top1s, flips = [], [], []
    for offset, (token, _, top1) in enumerate(expected):
        position = len(prompt) + offset
        row, key = rows[position], str(ids[position])
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"missing answer token logprob at position {position}")
        if ids[position] != token:
            raise ValueError(f"forced answer token {offset} is {ids[position]}, the reference's {token}")
        logprobs.append(row[key]["logprob"])
        served = entries_top1(served_entries(row))
        top1s.append(served)
        if served != top1:
            flips.append((offset, served, {int(item): value["logprob"] for item, value in row.items()}))
    import kinds
    captured = {"probe_forced": {"logprob": logprobs, "top1": top1s}}
    result = kinds.call("score_capture", kinds.KINDS["media_forced"], captured, reference)["metrics"]
    return {**result, "near_tie_report": probe_near_ties(reference, flips)}


def probe_near_ties(reference, flips):
    """A non-gating account of each served top-1 disagreement: the reference's
    margin there (its top-1 logprob minus the best other's), the served
    margin for the reference's token, and every reference variant's top-1
    and margin. A flip is a near tie when a reference variant also flips
    there, or when some variant's margin differs from the reference's by more
    than the reference margin itself; otherwise it is confident.
    `flips` are (offset, served top-1, served logprobs by token)."""
    positions = reference.get("positions")
    if positions is None:
        return {"gates": False, "available": False}
    rows = []
    for offset, served, logprobs in flips:
        top1 = reference["forced"][offset][2]
        others = [value for token, value in logprobs.items() if token != top1]
        served_margin = (logprobs[top1] - max(others) if top1 in logprobs
                         else min(logprobs.values()) - max(logprobs.values()))
        margin = positions["reference_margin"][offset]
        variants = {name: {"top1": values["top1"][offset],
                           "margin": None if values["margin"] is None else values["margin"][offset]}
                    for name, values in positions["variants"].items()}
        flipped = sorted(name for name, value in variants.items() if value["top1"] != top1)
        moved = sorted(name for name, value in variants.items()
                       if value["margin"] is not None and abs(value["margin"] - margin) > abs(margin))
        rows.append({"position": offset, "served_top1": served, "reference_top1": top1,
                     "reference_margin": margin, "served_margin": served_margin,
                     "variants": variants, "variants_flipping": flipped, "margin_moves": moved,
                     "near_tie": bool(flipped or moved)})
    near = sum(row["near_tie"] for row in rows)
    return {"gates": False, "available": True, "flips": rows, "near_tie": near,
            "confident": len(rows) - near}




def evaluate_probe(client, model, probe, images, reference, near_tie):
    """One probe against its HF reference `reference`: a short answer gates on
    the greedy answer, a free-text one on the reference answer teacher-forced
    (its greedy verdict is recorded)."""
    thinking = media_policy().get("thinking", False)
    budget = max(PROBE_MAX_TOKENS, len(reference["answer"]["token_ids"]))
    body = {"model": model, "messages": probe_messages(probe, images),
            "max_tokens": budget if thinking else answer_budget(budget),
            "temperature": 0, "logprobs": True, "top_logprobs": 5,
            **(thinking_control(True) if thinking else answer_control())}
    response, latency = client.request("/v1/chat/completions", body)
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
    clauses = media_policy()["expected"].get(probe["name"])
    count = response["usage"]["prompt_tokens"]
    free_text = reference["kind"] == "free-text"
    result = {"id": probe["name"], "answer": text, "reference_answer": reference["answer"]["text"],
              "fixture_expectation_match": clauses is None or ordered_clauses(text, clauses),
              "prompt_tokens": count, "reference_prompt_tokens": len(probe["token_ids"]),
              "token_count_match": count == len(probe["token_ids"]), "latency_seconds": latency}
    if reasoning:
        result["reasoning"] = reasoning
    agreements = [("greedy", lambda: reference_greedy_agreement(reference, choice, near_tie))]
    for key, agreement in agreements:
        try:
            result[key] = agreement()
        except Exception as error:
            result[key] = {"passed": False, "error": str(error)}
    gate = result["greedy"]
    if free_text:
        result["greedy"]["gates"] = False
        forced = {**body, "messages": body["messages"] + [
            {"role": "assistant", "content": reference["answer"]["text"]}],
            "continue_final_message": True, "max_tokens": 1, "prompt_logprobs": 5,
            "return_token_ids": True}
        del forced["logprobs"], forced["top_logprobs"]
        try:
            response, _ = client.request("/v1/chat/completions", forced)
            gate = result["teacher_forced"] = probe_forced_agreement(probe, reference, response)
        except Exception as error:
            gate = result["teacher_forced"] = {"passed": False, "error": str(error)}
    result["passed"] = (gate["passed"] and result["fixture_expectation_match"]
                        and result["token_count_match"])
    return result


def run_probes(client, server, probes, images, references, rule, selected=None):
    names = sorted(probe["name"] for probe in probes)
    if names != sorted(media_policy()["items"]) or sorted(references) != names:
        raise ValueError(f"captured probes and references are not the pinned image-probe set: {names}")
    rows = []
    for probe in probes:
        if selected is not None and probe["name"] not in selected:
            continue
        clauses = media_policy()["expected"].get(probe["name"])
        answer = references[probe["name"]]["answer"]["text"]
        if clauses is not None and not ordered_clauses(answer, clauses):
            raise ValueError(f"{probe['name']}: reference answer contradicts {clauses}")
        if reason := not_applicable(server, "image"):
            rows.append({"id": probe["name"], "kind": "image", "status": "not_applicable",
                         "reason": reason, "passed": None})
            continue
        try:
            rows.append(evaluate_probe(client, server["model"], probe, images,
                                        references[probe["name"]], rule["near_tie"]))
        except Exception as error:
            rows.append({"id": probe["name"], "passed": False, "error": str(error)})
    if selected is not None and {row["id"] for row in rows} != set(selected):
        raise ValueError(f"unknown probes: {sorted(set(selected) - {row['id'] for row in rows})}")
    return applicable({"limits": {"near_tie": rule["near_tie"], "teacher_forced": {
        name: references[name]["limits"] for name, item in references.items() if item["kind"] == "free-text"}}}, rows)


MIXED_CASES = ("single-image", "two-images", "two-resized-images", "image-video")


def evaluate_mixed(client, model, expected, body):
    """Force the bound answer through the served chat template and media placement."""
    if len(expected["answer_token_ids"]) < 32 or not expected.get("answer"):
        raise ValueError("mixed reference needs a full answer with at least 32 scored positions")
    body = {**body, "model": model, "messages": [*body["messages"], {"role": "assistant", "content": expected["answer"]}],
            "continue_final_message": True, "max_tokens": 1, "prompt_logprobs": 20, "return_token_ids": True}
    forced, _ = client.request("/v1/chat/completions", body)
    expected = {**expected, "forced_prompt_token_ids": expected["prompt_token_ids"] + expected["answer_token_ids"],
                "answer_token_start": len(expected["prompt_token_ids"]),
                "exact": {variant: expected[variant] for variant in ("exact", "bf16", "f64")}}
    scored = score_forced(expected, forced)
    return {"id": expected["id"], "kind": expected.get("kind", "image"),
            "passed": scored["passed"], "logprobs": scored, "capture": forced_capture(expected, forced),
            "prompt_token_ids": forced["prompt_token_ids"], "prompt_token_ids_match": True}


def mixed_reference(config, *, live=False, revision=None):
    from kinds import source
    from common import recorded_binding
    spec = config["qualification"]["references"]["media_mixed"]
    path = source(reference_path("media_mixed", config), revision, external=isinstance(spec["path"], dict))
    raw = pinned_bytes("media_mixed", path, config)
    value = pinned_reference("media_mixed", path, config) if live else json.loads(raw)
    binding = pinned_binding("media_mixed", value, config) if live else recorded_binding(value)
    by_id = {case["id"]: case for case in value["cases"]}
    if set(by_id) != {*MIXED_CASES, "32-images"}:
        raise ValueError("mixed reference case set differs")
    if any(len(by_id[name]["answer_token_ids"]) < 32 or not by_id[name].get("answer") for name in MIXED_CASES):
        raise ValueError("mixed reference needs full answers with at least 32 scored positions each")
    cases = [{**by_id[name], "kind": "audio" if name.endswith("audio") else "video" if name.endswith("video") else "image",
              "exact": {v: by_id[name][v] for v in ("exact", "bf16", "f64")}} for name in MIXED_CASES]
    return {"sha256": hashlib.sha256(raw).hexdigest(), "cases": cases, "binding": binding}


def run_mixed(client, server, config):
    from refs import read_member
    reference = mixed_reference(config, live=True)
    by_id = {case["id"]: case for case in reference["cases"]}
    directory = resolve_input("media_mixed_requests", config=config)
    rows = []
    for name in MIXED_CASES:
        body = json.loads(read_member(config, "media_mixed_requests", directory / (name + ".json")))
        try:
            if reason := not_applicable(server, by_id[name]["kind"]):
                rows.append({"id": name, "kind": by_id[name]["kind"], "status": "not_applicable", "reason": reason, "passed": None})
                continue
            expected = {**by_id[name], **by_id[name]["exact"]}
            rows.append(evaluate_mixed(client, server["model"], expected, body))
        except Exception as error:
            rows.append({"id": name, "passed": False, "error": str(error)})
    return applicable({"case_count": len(rows), "reference_sha256": reference["sha256"],
                       "reference_binding": reference["binding"]}, rows)


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--exact-reference", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--probes", nargs="+", help="run only these declared probes")
    parser.add_argument("--images", type=Path, help="directory for regenerated probe PNGs")
    args = parser.parse_args()
    if args.mechanics_only:
        parser.error("media qualification requires the captured model and media encoder")
    policy = media_policy()
    config = media_config()
    name = policy["reference"]
    exact = args.exact_reference or reference_path(name, config)
    capture = args.reference or resolve_input("media_capture", config=config)
    fixtures = args.fixtures or resolve_input("media_fixtures", config=config)
    # Every row scores in chopinfer eval: without it each row would keep only
    # the error, and the receipt nothing to rescore.
    import kinds
    try:
        scorer = kinds.eval_binary()
    except RuntimeError as error:
        raise SystemExit(f"refused: {error}")
    if not scorer.is_file():
        raise SystemExit(f"refused: the eval binary {scorer} does not exist")
    if not os.access(scorer, os.X_OK):
        raise SystemExit(f"refused: the eval binary {scorer} is not executable")
    client = Client(args.url, args.timeout)
    try:
        binding = pinned_binding(name, pinned_reference(name, exact, config), config)
        server = client.identity(args.model)
        bind_checkpoint(server, args.server_log, args.url, config)
        if policy["format"] == "image-probes":
            probes, hashes, digest = load_probes(capture)
            with tempfile.TemporaryDirectory() as scratch:
                images = probe_images(args.images or Path(scratch), fixtures, hashes)
            references, rule, provenance = probe_references(digest, exact)
            result = run_probes(client, server, probes, images, references, rule, args.probes)
            result["capture_sha256"] = digest
            result["reference"] = {"file": config["qualification"]["references"][name]["path"],
                                   "sha256": hashlib.sha256(pinned_bytes(name, exact, config)).hexdigest(),
                                   "provenance": provenance}
        else:
            reference, digest, hashes = load_reference(capture, exact)
            result = run(client, server, reference, fixtures, hashes)
            result["reference_sha256"] = digest
            result["exact_reference_sha256"] = hashlib.sha256(pinned_bytes(name, exact, config)).hexdigest()
            if "media_mixed" in config["qualification"]["references"]:
                result["mixed"] = run_mixed(client, server, config)
                result["passed"] &= result["mixed"]["passed"]
                result["errors"].extend(result["mixed"]["errors"])
                reasons = set(result.get("not_applicable", [])) | set(result["mixed"].get("not_applicable", []))
                if reasons:
                    result["not_applicable"] = sorted(reasons)
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
    confirm_live(client, server)
    result["reference_binding"] = binding
    # Its per-request seconds are diagnostics of an accuracy check, not timing evidence, so no host load
    # samples bound them and the receipt's timing reads informational.
    receipt(policy["receipt"], server, result)
    raise SystemExit(exit_status(result))


if __name__ == "__main__":
    main()
