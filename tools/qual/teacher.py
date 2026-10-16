"""Teacher-forced scoring helpers of the teacher gate.

`chopinfer eval run` captures and scores a teacher check (the shared
teacher_forced kind, docs/check-kinds.md). These helpers score a capture the
same way for the Python readers that compare captures: paired_teacher.py,
teacher_arms.py, verify_exact.py, prefix_exact.py and the conformance
vectors. A capture is scored against the exact chain on every scored
position of the reference, pooled and per text category. Its limits are the
contract floors below, raised to the reference's own spread over the same
positions: each other variant the gate pins for the reference scored against
the gate variant like a server. The top-1 limits apply to decisive flips,
near ties and all flips together (limits.py, near-tie rule). The module's
constants are the ones a declared check may set
(`checks.<id>.constants.teacher`).
"""

import hashlib
import json
import random
import re
import statistics
import sys

from common import entries_top1, qualification_config, served_entries
from exact_reference import check_scores, gap_top
import limits as rules


# MiMo's exact variants; verify_exact and prefix_exact score each one present.
# The teacher gate takes its variants from the pinned reference (teacher_variants).
EXACT_VARIANTS = ("exact",)
GATE_VARIANT = "exact"
# Only a declared check may opt in; missing spread evidence never selects it.
ALLOW_FLOOR_ONLY = False
# Floors of the numerical contract, applied against the exact chain by the
# spread rule (limits.py) to the pooled positions and to each category.
MAX_MEAN = rules.MEAN_FLOOR
LARGE_SET_MEAN = rules.LARGE_SET_MEAN_FLOOR
LARGE_SET_POSITIONS = rules.LARGE_SET_POSITIONS
MAX_TOP1 = rules.TOP1_FLOOR
MEAN_STEP = rules.MEAN_STEP
# The spread rule's headroom factor on the worst variant mean (limits.py);
# 1 unless the check declares one.
SPREAD_MEAN_FACTOR = rules.MEAN_FACTOR
# The default margins of `relative`, which compares two engines scored on
# the same exact positions. Correct BF16 implementations that differ only in
# rounding placement sit 0.09-0.10 from exact with a spread of about 0.01
# (numerics-audit.md), so the mean margin is twice that spread. Top-1 flips
# are compared on the positions where exactly one engine disagrees with
# exact: under equal accuracy their difference has standard deviation
# sqrt(discordant), and the margin is two of those.
MEAN_MARGIN = 0.02
TOP1_SIGMAS = 2.0
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_ROUNDS = 2000


def top_one(row):
    return entries_top1(served_entries(row))


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def response_prompt_rows(observed):
    choice = observed["choices"][0]
    ids = choice.get("prompt_token_ids") or observed.get("prompt_token_ids")
    rows = choice.get("prompt_logprobs") or observed.get("prompt_logprobs")
    if not isinstance(ids, list) or not isinstance(rows, list) or len(ids) != len(rows):
        raise ValueError("missing prompt token IDs or logprob rows")
    return ids, rows


def score(captured, reference, variant):
    """Per-position |dlogprob| and top-1 flip (limits.flip, against the
    variant's stored top-k gaps) of a capture against one exact variant."""
    positions, flips = [], []
    for text, exact in zip(captured, reference, strict=True):
        row = exact[variant]
        top = gap_top(row)
        if len(text["logprob"]) != len(row["logprob"]):
            raise ValueError(f"{text['name']}: {len(text['logprob'])} scores for "
                             f"{len(row['logprob'])} reference positions")
        for index, (logprob, top1) in enumerate(zip(text["logprob"], text["top1"], strict=True)):
            flips.append((row["top1"][index], top1, top[index]))
            positions.append({"text": text["name"], "category": text["category"], "offset": index, "served": top1,
                              "difference": abs(logprob - row["logprob"][index]),
                              "exact_margin": row["margin"][index]})
    return [{**{key: position[key] for key in ("text", "category", "offset", "served", "difference")},
             "disagrees": flip is not None, "flip": flip, "exact_margin": position["exact_margin"]}
            for position, flip in zip(positions, rules.flips(flips) if flips else [], strict=True)]


def summary(positions):
    count = len(positions)
    return {"scored": count,
            "mean_abs_dlogprob": statistics.mean(p["difference"] for p in positions) if count else None,
            "max_abs_dlogprob": max((p["difference"] for p in positions), default=None),
            **rules.top1_summary((p["flip"] for p in positions), count)}


def variant_scores(texts, variant):
    """One reference variant's rows as a capture, for scoring like a server."""
    return [{"name": text["name"], "category": text["category"], **text[variant]} for text in texts]


def teacher_variants(name="teacher"):
    """The variants the gate pins for its reference `name` (the teacher texts,
    or another pinned set such as Qwen3.8's long-context texts): the gate
    variant and the variants whose spread sets the limits."""
    variants = tuple(qualification_config()["qualification"]["references"][name]["variants"])
    if type(ALLOW_FLOOR_ONLY) is not bool:
        raise ValueError("ALLOW_FLOOR_ONLY must be boolean")
    floor = ALLOW_FLOOR_ONLY and GATE_VARIANT == "exact" and variants == ("exact",)
    if not floor and (GATE_VARIANT not in variants or len(variants) < 2):
        raise ValueError(f"the pinned teacher variants {variants} need {GATE_VARIANT} and a spread variant")
    return variants


def floor_only_reference(reference, variants):
    """Check the opt-in's evidence after normal pin and producer authentication."""
    if type(ALLOW_FLOOR_ONLY) is not bool:
        raise ValueError("ALLOW_FLOOR_ONLY must be boolean")
    if not ALLOW_FLOOR_ONLY:
        return False
    binding = reference.get("binding")
    modes = binding.get("modes") if isinstance(binding, dict) else None
    mode = modes.get("f32") if isinstance(modes, dict) else None
    from common import ROOT
    policy = json.loads((ROOT / "tools/refrun/mode-policies.json").read_text())["f32"]["policy_sha256"]["exact"]
    if (GATE_VARIANT != "exact" or tuple(variants) != ("exact",)
            or not isinstance(binding, dict) or binding.get("schema") != "chopinfer.refcache.binding.v1"
            or not isinstance(modes, dict) or set(modes) != {"f32"} or not isinstance(mode, dict)
            or (mode.get("mode"), mode.get("variant"), mode.get("dequant"), mode.get("policy_sha256"))
            != ("f32", "exact", "exact", policy)
            or not isinstance(binding.get("workload_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", binding["workload_sha256"]) is None):
        raise ValueError("floor-only reference needs a bound F32 exact variant")
    texts = reference.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(text, dict) for text in texts):
        raise ValueError("floor-only reference needs complete scored texts")
    check_reference(texts, [text.get("name") for text in texts], variants)
    if sum(len(text["exact"]["logprob"]) for text in texts) < rules.LARGE_SET_POSITIONS:
        raise ValueError("floor-only reference needs at least 1000 scored positions")
    return True


def reference_limits(texts, gate_variant=GATE_VARIANT, variants=None):
    """Each other variant of `variants` (default: the pinned ones), scored
    against the gate variant, and the gate limits that spread implies: the
    contract floors for this many positions, raised to the worst variant's
    mean |dlogprob| rounded up to MEAN_STEP and its decisive top-1 rate."""
    variants = teacher_variants() if variants is None else variants
    spread = {variant: score(variant_scores(texts, variant), texts, gate_variant)
              for variant in variants if variant != gate_variant}
    positions = sum(len(text[gate_variant]["logprob"]) for text in texts)
    limits = rules.spread_limits((summary(rows) for rows in spread.values()), positions,
                                 mean_floor=MAX_MEAN, large_set_mean_floor=LARGE_SET_MEAN,
                                 large_set_positions=LARGE_SET_POSITIONS, top1_floor=MAX_TOP1,
                                 mean_step=MEAN_STEP, mean_factor=SPREAD_MEAN_FACTOR)
    return limits, spread


def check_reference(texts, names, variants=None):
    """Raise ValueError unless the reference scores exactly the frozen texts
    `names`, in order, each with every pinned variant complete over its
    scored positions. A missing variant must not shrink the spread to the
    floors."""
    variants = teacher_variants() if variants is None else variants
    if [text.get("name") for text in texts] != names or len(set(names)) != len(names):
        raise ValueError("the reference does not score the frozen teacher texts in order")
    for text in texts:
        ids, start = text.get("prompt_token_ids"), text.get("score_from")
        if not isinstance(ids, list) or not isinstance(start, int) or not 1 <= start < len(ids):
            raise ValueError(f"{text['name']}: invalid token IDs or score_from")
        if not isinstance(text.get("category"), str):
            raise ValueError(f"{text['name']}: no category")
        for variant in variants:
            check_scores(text.get(variant), ids[start:], f"{text['name']} {variant}")


def clustered_interval(differences, texts):
    """95% text-clustered bootstrap interval of the mean paired difference."""
    clusters = {}
    for value, text in zip(differences, texts):
        clusters.setdefault(text, []).append(value)
    groups = list(clusters.values())
    generator = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [value for _ in groups for value in generator.choice(groups)]
        means.append(statistics.mean(sample))
    means.sort()
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]]


def relative(server, baseline, *, mean_margin=None, sigmas=None, with_interval=True):
    """Server against a baseline engine, both scored against the same exact
    positions (limits.paired)."""
    gate = rules.paired([(s["difference"], s["disagrees"]) for s in server],
                        [(b["difference"], b["disagrees"]) for b in baseline],
                        mean_margin=MEAN_MARGIN if mean_margin is None else mean_margin,
                        sigmas=TOP1_SIGMAS if sigmas is None else sigmas)
    paired = [s["difference"] - b["difference"] for s, b in zip(server, baseline, strict=True)]
    interval = None
    if with_interval:
        interval = ([0.0, 0.0] if rules.bit_equal(server, baseline) else
                    clustered_interval(paired, [s["text"] for s in server]))
    return {
        "mean_abs_dlogprob_gap": gate["mean_gap"],
        "mean_abs_dlogprob_gap_ci95": interval,
        "mean_margin": gate["mean_margin"],
        "top1_server_only": gate["top1_test_only"],
        "top1_baseline_only": gate["top1_baseline_only"],
        "top1_margin": gate["top1_margin"],
        "passed": gate["passed"],
    }


def evaluate(captured, reference, result, variants=None, binding=None):
    """Score a capture against the reference into `result`; the per-variant
    positions and the verdict. The pooled scores must meet the limits of the
    whole reference's spread, and each category those of its own spread, so
    good categories cannot dilute a bad one."""
    variants = teacher_variants() if variants is None else variants
    floor_only_reference({"texts": reference, "binding": binding}, variants)
    scores = {variant: score(captured, reference, variant) for variant in variants}
    import kinds
    constants = {key: {"value": globals()[key]} for key in
                 ("GATE_VARIANT", "MAX_MEAN", "LARGE_SET_MEAN", "LARGE_SET_POSITIONS", "MAX_TOP1", "MEAN_STEP",
                  "SPREAD_MEAN_FACTOR")}
    payload = {"texts": reference, "variants": list(variants)}
    if ALLOW_FLOOR_ONLY:
        constants["ALLOW_FLOOR_ONLY"] = {"value": True}
        payload["binding"] = binding
    metrics = kinds.call("score_capture", kinds.KINDS["teacher_forced"], captured,
                         payload,
                         {"constants": {"teacher": constants}})["metrics"]
    passed = metrics.pop("passed")
    result.update(metrics)
    return scores, passed


if __name__ == "__main__":
    # The capture CLI retired: refuse it rather than exit 0 without a receipt.
    sys.exit("teacher.py no longer captures a teacher check: run `chopinfer eval run --check ID` "
             "against the model's qual.json")
