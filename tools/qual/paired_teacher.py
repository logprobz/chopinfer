"""Compare a candidate's complete teacher capture with an accepted one by the
declared margins.

The accepted capture is the runner's stored last-good or seed capture of the
same configuration and arm (tools/qual/README.md#correctness-runner); the
margins are the check's constants.paired_teacher.PROFILE_POLICIES entry for
the live serving profile.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import common
import kinds
import limits
import teacher
import validation_evidence as evidence


POLICY = {}
PROFILE_POLICIES = {}
MARGINS = ("mean_margin", "category_mean_margin", "top1_delta", "category_alpha", "min_clusters",
           "tail_growth_nats", "tail_net_allowance", "allow_sparse_pooled")
MAX_CAPTURE_BYTES = 64 << 20


def digest(value):
    return hashlib.sha256(evidence.canonical(value)).hexdigest()


def read_json(path):
    return evidence.read_json(path, max_bytes=MAX_CAPTURE_BYTES)


def require(condition, message):
    if not condition:
        raise kinds.Insufficient("paired teacher: " + message)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def profile_key(profile):
    """Select the effective mode; alpha.2 keeps the default name after AR overrides."""
    ranks = profile.get("tensor_parallel_size")
    speculation = profile.get("speculation", {})
    tokens = speculation.get("tokens")
    require(type(ranks) is int and ranks > 0 and type(tokens) is int and tokens >= 0,
            "incomplete effective serving profile")
    mode = "ar" if tokens == 0 else speculation.get("drafter")
    require(mode in ("ar", "mtp", "dflash"), "unknown effective speculation mode")
    return f"tp{ranks}-{mode}"


def policy_name(profile, names):
    """The policy a serving profile selects among `names`: its profile_key, or
    `KEY-cpC` when it runs context parallelism C > 1 and that key is declared."""
    name = profile_key(profile)
    size = profile.get("context_parallel_size")
    size = 1 if size is None else size
    require(type(size) is int and size > 0, "invalid context-parallel size")
    return f"{name}-cp{size}" if size > 1 and f"{name}-cp{size}" in names else name


def selected_policy(check, server=None):
    """The margins for the live profile among PROFILE_POLICIES, or POLICY
    when a check declares one set for every profile."""
    constants = (check or {}).get("constants", {}).get("paired_teacher", {})
    profiles = constants.get("PROFILE_POLICIES", {}).get("value", PROFILE_POLICIES)
    if not profiles:
        return constants.get("POLICY", {}).get("value", POLICY)
    live = common.live_identity(server or {})
    name = policy_name(live.get("profile", {}), profiles)
    require(name in profiles, "live profile has no declared paired margins")
    return profiles[name]


def policy(check, server=None):
    """The declared margins for the live profile."""
    value = selected_policy(check, server)
    require(isinstance(value, dict) and set(value) == {"schema", "approval", "rationale", "reference_sha256", *MARGINS},
            "incomplete margins")
    require(value["schema"] == "paired-teacher-margins/v1", "unknown margins schema")
    require(all(isinstance(value[key], str) and value[key].strip() for key in ("approval", "rationale")),
            "margins need their approval and rationale")
    require(sha(value["reference_sha256"]), "missing reference pin")
    for key in ("mean_margin", "category_mean_margin", "top1_delta", "category_alpha", "tail_growth_nats"):
        require(kinds.finite(value[key]) and value[key] >= 0, "invalid margin: " + key)
    require(value["top1_delta"] <= 1 and 0 < value["category_alpha"] < 1
            and value["tail_growth_nats"] > 0, "invalid rate or tail threshold")
    require(type(value["min_clusters"]) is int and value["min_clusters"] >= 2
            and type(value["tail_net_allowance"]) is int and value["tail_net_allowance"] >= 0, "invalid count margin")
    require(type(value["allow_sparse_pooled"]) is bool, "invalid sparse pooled policy")
    return value


def capture_rows(capture, reference, strict=True):
    kinds.admit(capture)
    require(capture.get("schema") == "chopinfer-qualification/v1", "not a qualification capture")
    require(capture.get("reference_sha256") == reference["sha256"], "different reference")
    rows, texts = capture.get("rows"), reference["texts"]
    require(isinstance(rows, list) and [row.get("name") for row in rows] == [text["name"] for text in texts],
            "missing, duplicate or reordered texts")
    captured = []
    for row, text in zip(rows, texts, strict=True):
        if strict:
            require(row.get("prompt_token_ids_sha256") == teacher.ids_digest(text["prompt_token_ids"])
                    and row.get("score_from") == text["score_from"] and row.get("category") == text["category"],
                    text["name"] + ": different inputs or scored positions")
        entries = row.get("served_rows")
        tokens = text["prompt_token_ids"][text["score_from"]:]
        require(isinstance(entries, list) and len(entries) == len(tokens) and bool(tokens), "incomplete scored rows")
        require(row.get("served_rows_sha256") == common.rows_digest(entries), "served rows digest differs")
        logprob, top1 = [], []
        for token, items in zip(tokens, entries, strict=True):
            require(isinstance(items, list) and bool(items) and all(
                isinstance(item, list) and len(item) == 3 and type(item[0]) is int and item[0] >= 0
                and kinds.finite(item[1]) and item[1] <= 0
                and (item[2] is None or type(item[2]) is int and item[2] > 0) for item in items), "invalid served row")
            require(len({item[0] for item in items}) == len(items), "duplicate served token")
            actual = [item[1] for item in items if item[0] == token]
            require(len(actual) == 1, "missing scored token")
            logprob.append(actual[0])
            top1.append(common.entries_top1(items))
        captured.append({"name": text["name"], "category": text["category"], "logprob": logprob, "top1": top1})
    return captured


def capture_scores(capture, reference, strict=True):
    return teacher.score(capture_rows(capture, reference, strict), reference["texts"], reference["gate"])


def severity(row):
    return 0 if row["flip"] is None else 2 if row["flip"]["decisive"] else 1


def sign_test(added, removed):
    """One-sided binomial tail over worsened and improved whole texts."""
    count = added + removed
    if added == 0:
        return 1.0
    if added <= count // 2:
        return 1.0 - sum(math.comb(count, index) for index in range(added)) / (1 << count)
    return sum(math.comb(count, index) for index in range(added, count + 1)) / (1 << count)


def holm(categories, alpha):
    """Step-down family control over the category severity harm tests."""
    continuing = True
    for index, (name, row) in enumerate(sorted(categories.items(), key=lambda item: (item[1]["harm_p"], item[0]))):
        threshold = alpha / (len(categories) - index)
        rejected = continuing and row["harm_p"] < threshold
        row.update(holm_threshold=threshold, harm_rejected=rejected)
        row["passed"] = row["passed"] and not rejected
        continuing = rejected


def reference_name(check):
    arguments = (check or {}).get("args", [])
    return arguments[arguments.index("--reference-name") + 1] if "--reference-name" in arguments else "teacher"


def teacher_check(config, check):
    selected = [row for row in config["checks"] if row["module"] == "teacher"
                and reference_name(row) == reference_name(check)]
    require(len(selected) == 1, "no unique shipped teacher check for this reference")
    require(not selected[0].get("informational_reason"), "shipped teacher check is informational")
    return selected[0]


def absolute_score(capture, reference, captured):
    check = reference.get("absolute_check")
    require(isinstance(check, dict), "missing shipped teacher policy")
    kind = kinds.KINDS["teacher_forced"]
    kind.verify(capture, reference, check)
    recomputed = kinds.call("score_capture", kind, captured, reference, check)["metrics"]
    fields = recomputed.keys()
    require(fields <= capture.keys(), "incomplete recorded absolute summary")
    differences = [key for key in sorted(fields) if not limits.bit_equal(capture[key], recomputed[key])]
    metrics = {**kind.recorded(capture, check), **recomputed}
    status, predicates = kind.judge(metrics, check)
    return {"metrics": metrics, "predicates": predicates, "status": "pass" if status == "pass" and not differences else "fail",
            "summary_matches": not differences, "summary_differences": differences,
            "passed": status == "pass" and not differences}


def scope(candidate, accepted, margins, *, pooled=True):
    texts = [row["text"] for row in candidate]
    clusters = len(set(texts))
    interval_gate = pooled and clusters >= margins["min_clusters"]
    same = limits.bit_equal(candidate, accepted)
    relative = teacher.relative(candidate, accepted, mean_margin=margins["mean_margin"], sigmas=0.0,
                                with_interval=interval_gate)
    errors, top1, added, removed = [], [], 0, 0
    text_changes = {}
    for c, a in zip(candidate, accepted, strict=True):
        errors.append(c["difference"] - a["difference"])
        top1.append(int(c["disagrees"]) - int(a["disagrees"]))
        added += severity(c) > severity(a)
        removed += severity(c) < severity(a)
        text_changes[c["text"]] = text_changes.get(c["text"], 0) + severity(c) - severity(a)
    top_interval = None
    if interval_gate:
        top_interval = [0.0, 0.0] if same else teacher.clustered_interval(top1, texts)
    top_rate = statistics.mean(top1)
    tail_added = sum(value > margins["tail_growth_nats"] for value in errors)
    tail_removed = sum(value < -margins["tail_growth_nats"] for value in errors)
    mean_interval = relative["mean_abs_dlogprob_gap_ci95"]
    mean_limit = margins["mean_margin"] if pooled else margins["category_mean_margin"]
    mean_passed = (mean_interval[1] if interval_gate else relative["mean_abs_dlogprob_gap"]) <= mean_limit
    top_passed = (top_interval[1] if interval_gate else top_rate) <= margins["top1_delta"]
    tail_passed = tail_added - tail_removed <= margins["tail_net_allowance"]
    sufficient = not pooled or interval_gate or margins["allow_sparse_pooled"]
    if interval_gate:
        interval_mode = "clustered_ci95"
    elif not sufficient:
        interval_mode = "insufficient"
    else:
        interval_mode = "point_sparse" if clusters < margins["min_clusters"] else "category_point"
    worsened = sum(value > 0 for value in text_changes.values())
    improved = sum(value < 0 for value in text_changes.values())
    return {"candidate": teacher.summary(candidate), "accepted": teacher.summary(accepted),
            "paired": {"mean_gap": relative["mean_abs_dlogprob_gap"], "candidate_only": relative["top1_server_only"],
                       "accepted_only": relative["top1_baseline_only"]}, "bit_identical": same, "mean_gap_ci95": mean_interval,
            "mean_error_change": relative["mean_abs_dlogprob_gap"], "mean_limit": mean_limit,
            "text_clusters": clusters, "minimum_clusters": margins["min_clusters"],
            "interval_mode": interval_mode,
            "evidence_sufficient": sufficient, "allow_sparse_pooled": margins["allow_sparse_pooled"],
            "top1_net_rate": top_rate, "top1_net_rate_ci95": top_interval,
            "top1_delta": margins["top1_delta"], "top1_gated": pooled, "top1_passed": top_passed if pooled else None,
            "severity_added": added, "severity_removed": removed,
            "harm_test": "text_sign", "worsened_texts": worsened, "improved_texts": improved,
            "tied_texts": clusters - worsened - improved, "harm_p": sign_test(worsened, improved),
            "candidate_accepted_top1_disagreements": sum(
                c["served"] != a["served"] for c, a in zip(candidate, accepted, strict=True)),
            "tail": {"added": tail_added, "removed": tail_removed, "net_added": tail_added - tail_removed,
                     "growth_nats": margins["tail_growth_nats"], "net_allowance": margins["tail_net_allowance"],
                     "gated": pooled, "passed": tail_passed},
            "mean_passed": mean_passed, "passed": sufficient and mean_passed and (top_passed and tail_passed if pooled else True)}


def verify_arms(value, reference, check):
    """Both arms' scores and absolute verdicts under the declared margins.
    A capture must hold every reference text's scored rows."""
    kinds.admit(value)
    declared = policy(check, server=value.get("qualified_server"))
    require(reference and reference["sha256"] == declared["reference_sha256"], "margins name another reference")
    teacher.check_reference(reference["texts"], [text["name"] for text in reference["texts"]], reference["variants"])
    require(value.get("policy_sha256") == digest(declared), "margins changed after the comparison")
    require(set(value.get("arms", {})) == {"accepted", "candidate"}, "missing arm")
    scores, absolute = {}, {}
    for arm in ("accepted", "candidate"):
        capture = value["arms"][arm]
        captured = capture_rows(capture, reference)
        scores[arm] = teacher.score(captured, reference["texts"], reference["gate"])
        absolute[arm] = absolute_score(capture, reference, captured)
    require(absolute["accepted"]["passed"], "accepted arm fails shipped absolute gate or recorded summary")
    return declared, scores, absolute


def compare(value, reference, check):
    declared, scores, absolute = verify_arms(value, reference, check)
    candidate, accepted = scores["candidate"], scores["accepted"]
    pooled = scope(candidate, accepted, declared)
    require(pooled["evidence_sufficient"], "pooled scope has too few texts for non-inferiority")
    groups = {}
    for c, a in zip(candidate, accepted, strict=True):
        group = groups.setdefault(c["category"], ([], []))
        group[0].append(c)
        group[1].append(a)
    categories = {name: scope(c, a, declared, pooled=False) for name, (c, a) in sorted(groups.items())}
    holm(categories, declared["category_alpha"])
    starts = {text["name"]: text["score_from"] for text in reference["texts"]}
    affected = [{"text": c["text"], "category": c["category"], "position":
                 starts[c["text"]] + c["offset"],
                 "error_change": c["difference"] - a["difference"], "candidate_flip": c["flip"], "accepted_flip": a["flip"]}
                for c, a in zip(candidate, accepted, strict=True)
                if c["difference"] != a["difference"] or c["served"] != a["served"]]
    absolute_failed = [arm for arm, row in absolute.items() if not row["passed"]]
    numerical = not absolute_failed and pooled["passed"] and all(row["passed"] for row in categories.values())
    return {"passed": numerical, "pooled": pooled, "by_category": categories,
            "failed_categories": [name for name, row in categories.items() if not row["passed"]],
            "affected_positions": affected, "absolute": absolute, "absolute_failed_arms": absolute_failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True, help="the stored accepted capture")
    parser.add_argument("--candidate", type=Path, required=True, help="this run's capture")
    parser.add_argument("--check-id", default="paired_teacher")
    parser.add_argument("--reference-name")
    parser.add_argument("--url", required=True, help="the live candidate server, for its serving profile")
    parser.add_argument("--model")
    args = parser.parse_args()
    config = common.qualification_config()
    check = next((row for row in config["checks"] if row["id"] == args.check_id), None)
    try:
        declared_reference = reference_name(check)
        require(args.reference_name is None or args.reference_name == declared_reference, "reference name differs from declaration")
        arms = {"accepted": read_json(args.accepted), "candidate": read_json(args.candidate)}
        client = common.Client(args.url)
        qualified_server = client.identity(args.model)
        common.bind_checkpoint(qualified_server, None, args.url)
        declared = policy(check, server=qualified_server)
        reference = kinds.KINDS["paired_teacher"].reference(check, config)
        value = {"schema": "chopinfer-qualification/v1", "check": args.check_id, "mechanics_only": False,
                 "policy_sha256": digest(declared), "arms": arms, "qualified_server": qualified_server,
                 "accepted_capture_sha256": digest(arms["accepted"]), "accepted_path": str(args.accepted)}
        result = compare(value, reference, check)
        common.confirm_live(client, qualified_server)
        common.receipt(args.check_id, value["arms"]["candidate"].get("server", {}), {**value, **result}, False)
        print(json.dumps(result, indent=1))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        raise SystemExit("refused: " + str(error)) from None
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
