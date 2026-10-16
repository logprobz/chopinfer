#!/usr/bin/env python3
"""Validate fixed coverage on both sides, then compare gated outputs by IEEE bits.

Usage: compare.py BEFORE_DIR AFTER_DIR [OUT.json] --spec SPEC.json
Each directory holds the capture `capture.json`. The spec (schema version 2, see
tools/ab/serving/README.md) declares every phase, its request and token counts,
prompt-token bounds and teacher layout. Exit 0: valid complete captures and
identical gated phases; 1: failed evidence; 2: invalid arguments or
specification, or report I/O failure.
"""
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

NONFINITE = {"nan", "positive_infinity", "negative_infinity"}


class Invalid(ValueError):
    pass


def need(condition, where, message):
    if not condition:
        raise Invalid(f"{where}: {message}")


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Invalid(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path):
    # Nonfinite score literals are checked using each phase's explicit policy.
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except RecursionError as error:
        raise Invalid("JSON nesting is too deep") from error
    return value, hashlib.sha256(raw).hexdigest()


def exact_keys(value, required, optional, where):
    need(type(value) is dict, where, "expected an object")
    need(required <= value.keys(), where, f"missing fields {sorted(required - value.keys())}")
    need(value.keys() <= required | optional, where,
         f"unknown fields {sorted(value.keys() - required - optional)}")


def validate_spec(spec):
    exact_keys(spec, {"version", "phases"}, {"ignored_phases"}, "spec")
    need(type(spec["version"]) is int and spec["version"] == 2, "spec", "version must be 2")
    phases = spec["phases"]
    need(type(phases) is dict and phases, "spec", "phases must be a nonempty object")
    ignored = spec.get("ignored_phases", [])
    need(type(ignored) is list and all(type(x) is str and x for x in ignored),
         "spec", "ignored_phases must be a list of names")
    need(len(set(ignored)) == len(ignored) and not set(ignored) & phases.keys(),
         "spec", "ignored phase names must be unique and disjoint from required phases")
    for name, phase in phases.items():
        where = f"spec {name}"
        need(type(name) is str and bool(name.strip()), where, "empty phase name")
        exact_keys(phase, {"kind", "gated", "requests", "max_tokens", "top", "prompt_tokens"},
                   {"teacher", "allow_nonfinite_logprobs"}, where)
        need(phase["kind"] in ("prefill", "teacher", "decode"), where, "unknown phase kind")
        need(type(phase["gated"]) is bool, where, "gated must be a boolean")
        for key in ("requests", "max_tokens", "top"):
            need(integer(phase[key], 1), where, f"{key} must be a positive integer")
        # These capture records expose one phase-wide prompt count, not per-output counts.
        if phase["kind"] != "decode":
            need(phase["requests"] == 1, where, "prefill/teacher format supports one request")
        bounds = phase["prompt_tokens"]
        exact_keys(bounds, {"min", "max"}, set(), where + " prompt_tokens")
        need(integer(bounds["min"], 1) and integer(bounds["max"], 1)
             and bounds["min"] <= bounds["max"], where, "invalid prompt-token bounds")
        allowed = phase.get("allow_nonfinite_logprobs", [])
        need(type(allowed) is list and all(type(x) is str and x in NONFINITE for x in allowed)
             and len(set(allowed)) == len(allowed), where, "invalid nonfinite policy")
        if phase["kind"] == "teacher":
            teacher = phase.get("teacher")
            exact_keys(teacher, {"offset", "prompt_ids"}, set(), where + " teacher")
            need(type(teacher["offset"]) is int and teacher["offset"] in (0, 1), where,
                 "teacher offset must be 0 or 1")
            need(bounds["min"] > teacher["offset"], where, "teacher must have scored positions")
            need(teacher["prompt_ids"] in
                 ("prompt_token_ids", "scored_token_ids", "score_map_text"), where,
                 "unknown teacher prompt-ID source")
        else:
            need("teacher" not in phase, where, "teacher layout on a non-teacher phase")
    need(any(p["gated"] for p in phases.values()), "spec", "at least one equality gate is required")
    return spec


def same(a, b):
    """IEEE-bit equality with equal JSON types at every node: 1 and 1.0, or
    True and 1, differ. Iterative, so deep nesting cannot exhaust the stack."""
    pending = [(a, b)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return False
            pending.extend((left[k], right[k]) for k in left)
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right))
        elif isinstance(left, float):
            if struct.pack(">d", left) != struct.pack(">d", right):
                return False
        elif left != right:
            return False
    return True


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    left, right = ordered[middle - 1:middle + 1]
    total = left + right
    # Preserve subnormal pairs; halve first only when the sum overflows.
    if type(total) is float and not math.isfinite(total):
        return left / 2 + right / 2
    return total / 2


def logprob(value, spec, where):
    need(type(value) in (int, float), where, "logprob must be a number, not bool/null/string")
    if type(value) is float and not math.isfinite(value):
        kind = "nan" if math.isnan(value) else (
            "positive_infinity" if value > 0 else "negative_infinity")
        need(kind in spec.get("allow_nonfinite_logprobs", []), where,
             f"{kind} logprob is not explicitly allowed")


def ids(value, count, where):
    need(type(value) is list and len(value) == count, where, f"expected exactly {count} token IDs")
    need(all(integer(x) and x <= 0xffffffff for x in value), where, "token IDs must be u32 integers")
    return value


def choice(value, count, offset, spec, where):
    exact_keys(value, {"tokens", "token_logprobs", "top_logprobs"}, {"text_offset"}, where)
    for key in ("tokens", "token_logprobs", "top_logprobs"):
        need(type(value.get(key)) is list and len(value[key]) == count,
             where, f"{key} must cover exactly {count} positions")
    if "text_offset" in value:
        need(type(value["text_offset"]) is list and len(value["text_offset"]) == count
             and all(integer(x) for x in value["text_offset"]), where,
             "text_offset must cover every position with integer offsets")
    for i, (token, score, top) in enumerate(zip(value["tokens"], value["token_logprobs"],
                                              value["top_logprobs"])):
        at = f"{where} position {i}"
        need(type(token) is str, at, "token must be a string")
        if i < offset:
            need(score is None and top is None, at, "unscored prefix must be null")
            continue
        logprob(score, spec, at)
        need(type(top) is dict and len(top) >= spec["top"], at, "missing/short top-k map")
        for key, number in top.items():
            logprob(number, spec, f"{at} top {key!r}")


def teacher_ids(output, prompt, spec, where):
    layout = spec["teacher"]
    offset, source = layout["offset"], layout["prompt_ids"]
    count = prompt - offset
    supplied = {}
    if "prompt_token_ids" in output:
        supplied["prompt_token_ids"] = ids(output["prompt_token_ids"], prompt,
                                             where + " prompt_token_ids")[offset:]
    if "scored_token_ids" in output:
        supplied["scored_token_ids"] = ids(output["scored_token_ids"], count,
                                             where + " scored_token_ids")
    if source != "score_map_text":
        need(source in supplied, where, f"required {source} is missing")
    maps = output.get("prompt_logprobs")
    need(type(maps) is list and len(maps) == prompt, where,
         f"prompt_logprobs must contain {prompt} positions, including {offset} null prefix slots")
    selected = []
    for i, row in enumerate(maps):
        at = f"{where} prompt position {i}"
        if i < offset:
            need(row is None, at, "unscored prefix must be null")
            continue
        need(type(row) is dict and len(row) >= spec["top"], at, "missing/short prompt score map")
        ranked = set()
        for key, record in row.items():
            need(type(key) is str and re.fullmatch(r"0|[1-9][0-9]*", key)
                 and int(key) <= 0xffffffff, at, "score-map keys must be canonical u32 IDs")
            exact_keys(record, {"decoded_token", "logprob"}, {"rank"}, at)
            need(type(record["decoded_token"]) is str, at, "decoded_token must be a string")
            logprob(record.get("logprob"), spec, at + f" ID {key}")
            if "rank" in record:
                need(integer(record["rank"], 1), at, "rank must be a positive integer")
                need(record["rank"] not in ranked, at, "duplicate top-k rank")
                ranked.add(record["rank"])
        need(set(range(1, spec["top"] + 1)) <= ranked, at, "incomplete ranked top-k scores")
        token = output["logprobs"]["tokens"][i]
        if source == "score_map_text":
            matches = [int(key) for key, rec in row.items() if rec["decoded_token"] == token]
            need(len(matches) == 1, at, "archive prompt ID has no unique decoded-token match")
            token_id = matches[0]
        else:
            token_id = supplied[source][i - offset]
        need(str(token_id) in row and row[str(token_id)]["decoded_token"] == token,
             at, "prompt ID is not backed by the echoed token's score entry")
        selected.append(token_id)
    need(len(selected) == count and count > 0, where, "incomplete scored prompt-ID coverage")
    for name, values in supplied.items():
        need(values == selected, where, f"{name} disagrees with scored prompt IDs")
    return {"prompt_id_source": source, "offset": offset, "scored_positions": count,
            "scored_ids_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}


def validate(phase, spec, where):
    need(type(phase) is dict and phase.get("kind") == spec["kind"], where, "wrong/missing kind")
    need(type(phase.get("requests")) is int and phase["requests"] == spec["requests"],
         where, "wrong/missing request count")
    prompt = phase.get("prompt_tokens")
    bounds = spec["prompt_tokens"]
    need(integer(prompt, 1) and bounds["min"] <= prompt <= bounds["max"], where,
         f"prompt_tokens {prompt!r} outside [{bounds['min']}, {bounds['max']}]")
    outputs = phase.get("outputs")
    need(type(outputs) is list and len(outputs) == spec["requests"], where,
         f"expected {spec['requests']} nonempty request captures")
    completion = spec["requests"] * spec["max_tokens"]
    if spec["kind"] == "decode" or "completion_tokens" in phase:
        need(type(phase.get("completion_tokens")) is int and phase["completion_tokens"] == completion,
             where, f"completion_tokens must be {completion}")
    for key in ("wall_ms", "tokens_per_s", "prompt_tokens_per_s"):
        if key in phase:
            v = phase[key]
            need(finite_number(v) and v >= 0, where,
                 f"{key} must be finite and nonnegative")
    if "ms_per_token" in phase:
        need(type(phase["ms_per_token"]) is list and phase["ms_per_token"]
             and all(finite_number(v) and v >= 0
                     for v in phase["ms_per_token"]), where, "invalid timing samples")
    evidence = []
    for r, output in enumerate(outputs):
        at = f"{where} request {r}"
        need(type(output) is dict and type(output.get("text")) is str, at, "missing output/text")
        kind = spec["kind"]
        # Every request records its generated IDs and how it finished; text
        # and logprobs alone do not identify the served tokens.
        required, optional = {"text", "logprobs", "token_ids", "finish_reason"}, set()
        if kind == "teacher":
            optional = {"prompt_logprobs", "prompt_token_ids", "scored_token_ids"}
        elif kind == "decode":
            required, optional = required | {"done"}, {"stream_errors"}
        exact_keys(output, required, optional, at)
        if kind == "decode":
            need("stream_errors" not in output, at, f"the stream reported errors {output.get('stream_errors')}")
            need(output["done"] is True, at, "the stream ended without [DONE]")
        need(output["finish_reason"] in ("stop", "length"), at, "finish_reason must be stop or length")
        # choices.token_ids are generated IDs; they are not prompt IDs.
        ids(output["token_ids"], spec["max_tokens"], at + " completion token_ids")
        if kind == "prefill":
            choice(output.get("logprobs"), spec["max_tokens"], 0, spec, at)
        elif kind == "teacher":
            choice(output.get("logprobs"), prompt + spec["max_tokens"], spec["teacher"]["offset"],
                   spec, at)
            evidence.append(teacher_ids(output, prompt, spec, at))
        else:
            events = output.get("logprobs")
            need(type(events) is list and events, at, "missing/empty streamed logprob events")
            count = 0
            for index, event in enumerate(events):
                need(type(event) is dict and type(event.get("tokens")) is list
                     and event["tokens"], at, "empty/malformed stream event")
                n = len(event["tokens"])
                choice(event, n, 0, spec, f"{at} event {index}")
                count += n
            need(count == spec["max_tokens"], at,
                 f"stream covers {count} tokens, expected {spec['max_tokens']}")
    return evidence


def load(directory, spec):
    record, digest = read_json(Path(directory) / "capture.json")
    need(type(record) is dict and type(record.get("model")) is str and record["model"],
         str(directory), "missing capture/model")
    phases = record.get("phases")
    need(type(phases) is list and phases, str(directory), "empty/missing phase captures")
    by_name = {}
    for phase in phases:
        need(type(phase) is dict and type(phase.get("phase")) is str and phase["phase"],
             str(directory), "malformed phase")
        name = phase["phase"]
        need(name not in by_name, str(directory), f"duplicate phase {name}")
        by_name[name] = phase
    unknown = by_name.keys() - spec["phases"].keys() - set(spec.get("ignored_phases", []))
    need(not unknown, str(directory), f"undeclared phases {sorted(unknown)}")
    return record["model"], by_name, digest


def compare(before_dir, after_dir, spec):
    validate_spec(spec)
    report = {"version": 2, "before": str(before_dir), "after": str(after_dir),
              "valid": True, "identical": True, "errors": [], "phases": [],
              "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    sides = []
    for label, directory in (("before", before_dir), ("after", after_dir)):
        try:
            sides.append(load(directory, spec))
        except (OSError, ValueError, UnicodeError) as error:
            sides.append(None)
            report["errors"].append(f"{label}: {error}")
    if any(side is None for side in sides):
        report.update(valid=False, identical=False)
        return report
    report["capture_sha256"] = [side[2] for side in sides]
    if sides[0][0] != sides[1][0]:
        report["errors"].append("capture model identifiers differ")
        report.update(valid=False, identical=False)
    for name, phase_spec in spec["phases"].items():
        entry = {"phase": name, "kind": phase_spec["kind"], "gated": phase_spec["gated"],
                 "problems": [], "identical": False}
        pair = []
        teacher = []
        for label, (_, by_name, _) in zip(("before", "after"), sides):
            phase = by_name.get(name)
            pair.append(phase)
            try:
                need(phase is not None, f"{label} {name}", "required phase missing")
                teacher.append(validate(phase, phase_spec, f"{label} {name}"))
            except Invalid as error:
                entry["problems"].append(str(error))
        if entry["problems"]:
            report.update(valid=False, identical=False)
        else:
            old, new = pair
            fields = ("requests", "prompt_tokens", "completion_tokens")
            entry["counts"] = [[p.get(k) for k in fields] for p in pair]
            entry["differing_requests"] = [i for i, (a, b) in enumerate(zip(old["outputs"], new["outputs"]))
                                            if not same(a, b)]
            entry["identical"] = not entry["differing_requests"] and same(*entry["counts"])
            entry["wall_ms"] = [p.get("wall_ms") for p in pair]
            entry["tokens_per_s"] = [p.get("tokens_per_s", p.get("prompt_tokens_per_s")) for p in pair]
            entry["ms_per_token_median"] = [median(p["ms_per_token"])
                                             if p.get("ms_per_token") else None for p in pair]
            if phase_spec["kind"] == "teacher":
                entry["teacher_coverage"] = teacher
            if phase_spec["gated"] and not entry["identical"]:
                report["identical"] = False
        report["phases"].append(entry)
    return report


def describe(report):
    lines = list(report["errors"])
    for phase in report["phases"]:
        if phase["problems"]:
            verdict = "INVALID: " + "; ".join(phase["problems"])
        elif phase["identical"]:
            verdict = "bit-identical"
        else:
            verdict = f"DIFFERS requests {phase['differing_requests']} counts {phase['counts']}"
        mode = "gated" if phase["gated"] else "reported (coverage required)"
        lines.append(f"{phase['phase']}: {verdict}; {mode}; wall_ms {phase.get('wall_ms')}")
    lines.append("PASS: all declared phases valid; gated phases bit-identical" if report["identical"]
                 else "FAIL: invalid evidence or different gated output")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_intermixed_args(argv)
    try:
        spec, spec_digest = read_json(args.spec)
        validate_spec(spec)
        if args.out:
            destination = Path(args.out).resolve()
            protected = {args.spec.resolve(), (Path(args.before) / "capture.json").resolve(),
                         (Path(args.after) / "capture.json").resolve()}
            need(destination not in protected, "output", "would overwrite an input or spec")
        report = compare(args.before, args.after, spec)
        report["spec_file"] = str(args.spec)
        report["spec_file_sha256"] = spec_digest
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("\n".join(describe(report)))
        return 0 if report["identical"] else 1
    except (OSError, ValueError, UnicodeError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
