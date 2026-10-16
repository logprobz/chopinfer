"""Write the conformance vectors of the shared check kinds (tests/qual-vectors).

Each vector is a small synthetic receipt, with its reference or baseline
record and the declared check, and the verdict kinds.rescore gives it:
status, metrics and error. test_qual_vectors.py requires this code to
reproduce every committed vector, and `chopinfer eval` runs the same files.
Regenerate only for a reviewed change of a kind's contract
(docs/check-kinds.md), and review the vectors' diff.

    python3 -B tools/qual/vectors.py
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import common
from exact_reference import gap_top
import kinds
import limits

OUTPUT = Path(__file__).resolve().parents[2] / "tests/qual-vectors"
SCHEMA = "chopinfer-qual-vectors/v1"
REFERENCE_SHA256 = "a" * 64


def rule(path, op, value):
    return {"path": path, "op": op, "value": value, "source": "docs/check-kinds.md"}


def vector(kind, name, receipt, check, reference=None, covers=()):
    """A conformance vector: the inputs, the contract clauses it pins
    (`covers`) and this code's verdict on them."""
    result = kinds.rescore(kinds.KINDS[kind], copy.deepcopy(receipt), copy.deepcopy(reference), check)
    expected = {"status": result["status"]}
    if result["metrics"] is not None:
        expected["metrics"] = result["metrics"]
    if result.get("error"):
        expected["error"] = result["error"]
    if result.get("reason"):
        expected["reason"] = result["reason"]
    return {"name": name, "covers": list(covers), "check": check, "reference": reference, "receipt": receipt,
            "expected": expected}


def receipt(check, **fields):
    return {"schema": "chopinfer-qualification/v1", "check": check, "mechanics_only": False, "skipped": [],
            "errors": [], **fields}


def changed(value, change):
    value = copy.deepcopy(value)
    change(value)
    return value


FILLER = 900


def scores(tokens, logprobs, top1s=None):
    """Consistent reference scores (exact_reference.check_scores, runner
    layout) of `tokens` with these token logprobs and top-1 tokens (default:
    the token itself). A top-1 other than the token sits halfway between the
    token's logprob and zero; fillers follow 1 nat apart."""
    columns = {"logprob": [], "top1": [], "margin": [], **top_columns([])}
    for position, (token, logprob) in enumerate(zip(tokens, logprobs)):
        top1 = token if top1s is None else top1s[position]
        top5 = [[token, logprob]] if top1 == token else [[top1, logprob / 2], [token, logprob]]
        top5 += [[FILLER + index, logprob - 1 - index] for index in range(5 - len(top5))]
        columns["logprob"].append(logprob)
        columns["top1"].append(top1)
        columns["margin"].append(top5[0][1] - top5[1][1])
        columns["topk_ids"].append([pair[0] for pair in top5])
        columns["topk_logits"].append([pair[1] for pair in top5])
        columns["topk_logprobs"].append([pair[1] for pair in top5])
    return columns


def ranked(tokens, top5s):
    """Reference scores whose top-5 rows are given: each token's logprob
    is its row entry."""
    return {"logprob": [dict(map(tuple, top5))[token] for token, top5 in zip(tokens, top5s)],
            "top1": [top5[0][0] for top5 in top5s], "margin": [top5[0][1] - top5[1][1] for top5 in top5s],
            **top_columns(top5s)}


def top_columns(rows):
    """Synthetic runner columns with logits equal to logprobs."""
    return {"topk_ids": [[pair[0] for pair in row] for row in rows],
            "topk_logits": [[pair[1] for pair in row] for row in rows],
            "topk_logprobs": [[pair[1] for pair in row] for row in rows]}


# The near-tie rule's reference rows (top1-near-tie.v1): the reference ranks
# TIE_A first; TIE_B sits exactly 0.3 below it in a "tie" row (-0.2 - -0.5)
# and 0.30000000000000004 below it in an "over" row (-0.1 - -0.4); TIE_D is
# outside the stored top 5.
TIE_A, TIE_B, TIE_C, TIE_D = 61, 62, 63, 99


def served_rows(tokens, logprobs, top1s):
    """`served_rows` and their digest as eval's teacher capture and multimodal.py record
    them: each scored token at its served logprob, and a served top-1 that
    differs from it ranked above it at half its logprob."""
    rows = []
    for token, logprob, top1 in zip(tokens, logprobs, top1s, strict=True):
        row = {str(token): {"logprob": logprob, "rank": 1 if top1 == token else 2}}
        if top1 != token:
            row[str(top1)] = {"logprob": logprob / 2, "rank": 1}
        rows.append(common.served_entries(row))
    return {"served_rows": rows, "served_rows_sha256": common.rows_digest(rows)}


def tie_row(top):
    return [[TIE_A, top], [TIE_B, top - 0.3], [TIE_C, top - 0.4], [FILLER, top - 1], [FILLER + 1, top - 2]]


def tie_rows(ties, overs):
    """`ties` rows where TIE_B is a near tie, then `overs` where it is not."""
    return [tie_row(-0.2)] * ties + [tie_row(-0.1)] * overs


def teacher_vectors():
    import teacher

    def text(name, category, ids, exact, spread, exact_top1=None, spread_top1=None):
        tokens = ids[1:]
        return {"name": name, "category": category, "prompt_token_ids": ids, "score_from": 1,
                "exact": scores(tokens, exact, exact_top1), "bf16": scores(tokens, spread, spread_top1)}

    def reference_of(texts):
        return {"sha256": REFERENCE_SHA256, "gate": "exact", "variants": ["exact", "bf16"], "texts": texts}

    def run(texts, served, served_top1=None, variants=("exact", "bf16")):
        """The receipt of a server whose token logprobs are `served` (per text)."""
        capture = [{"name": item["name"], "category": item["category"], "logprob": logprobs,
                    "top1": served_top1[index] if served_top1 else item["exact"]["top1"]}
                   for index, (item, logprobs) in enumerate(zip(texts, served))]
        result = {"errors": [], "skipped": [], "reference_sha256": REFERENCE_SHA256,
                  "rows": [{"name": item["name"], "prompt_tokens": len(item["prompt_token_ids"]),
                            "scored": len(item["exact"]["logprob"]), "latency_seconds": 0.1,
                            **served_rows(item["prompt_token_ids"][item["score_from"]:], text["logprob"],
                                          text["top1"])} for item, text in zip(texts, capture)]}
        _, passed = teacher.evaluate(capture, texts, result, variants=list(variants))
        return receipt("teacher", passed=passed, **result)

    texts = [text("t0", "chat", [1, 2, 3, 4], [-0.5, -1.0, -2.0], [-0.52, -1.1, -2.0], [2, 3, 9], [2, 3, 9]),
             text("t1", "code", [5, 6, 7], [-0.25, -3.0], [-0.25, -3.2], [6, 8], [6, 7])]
    reference = reference_of(texts)
    close = [[value + 0.01 for value in item["exact"]["logprob"]] for item in texts]
    far = [close[0], [value - 0.6 for value in close[1]]]
    check = {"rules": [rule(["passed"], "eq", True), rule(["mean_abs_dlogprob"], "le", 0.15),
                       rule(["top1_disagreement"], "le", 0.08), rule(["failed_categories"], "eq", []),
                       rule(["scored"], "ge", 5)]}
    passing, failing = run(texts, close), run(texts, far)
    vectors = [
        vector("teacher_forced", "pass", passing, check, reference),
        vector("teacher_forced", "fail: a category beyond its spread limits", failing, check, reference),
        vector("teacher_forced", "fail: a stored verdict its limits do not support is recomputed",
               changed(failing, lambda v: v.update(passed=True, failed_categories=[])), check, reference),
        vector("teacher_forced", "insufficient: a text without a row",
               changed(passing, lambda v: v["rows"].pop()), check, reference),
        vector("teacher_forced", "insufficient: a category missing",
               changed(passing, lambda v: v["by_category"].pop("code")), check, reference),
        vector("teacher_forced", "insufficient: scored positions inflated",
               changed(passing, lambda v: v.update(scored=100000)), check, reference),
        vector("teacher_forced", "insufficient: no limits", changed(passing, lambda v: v.pop("limits")),
               check, reference),
        vector("teacher_forced", "insufficient: another reference",
               changed(passing, lambda v: v.update(reference_sha256="b" * 64)), check, reference),
        vector("teacher_forced", "insufficient: the reference lacks a pinned variant on a text", passing, check,
               changed(reference, lambda r: r["texts"][1].pop("bf16")), covers=["missing variant"]),
        vector("teacher_forced", "insufficient: a reference logprob above zero", passing, check,
               changed(reference, lambda r: r["texts"][0]["exact"]["logprob"].__setitem__(0, 0.5)),
               covers=["invalid reference scores"]),
        vector("teacher_forced", "insufficient: a reference top-1 that is not its top-5 first", passing, check,
               changed(reference, lambda r: r["texts"][0]["bf16"]["top1"].__setitem__(0, 7)),
               covers=["invalid reference scores"]),
    ]
    # The spread rule at its decimal step: one scored position, so the
    # pooled and the category limits are the same. |-0.75 - -0.5| is 0.25
    # exactly; |-0.55 - -0.3| is 0.25000000000000006 in binary64, which
    # rounds up to 0.3. The served position sits 0.25, 0.25 + 1 ulp or 0.27
    # from the gate variant.
    spread_check = {"rules": [rule(["passed"], "eq", True), rule(["failed_categories"], "eq", [])]}
    exact_step = [text("s0", "chat", [1, 2], [-0.5], [-0.75])]
    overshoot = [text("s0", "chat", [1, 2], [-0.3], [-0.55])]
    vectors += [
        vector("teacher_forced", "pass: a spread of exactly 0.25 limits the mean to 0.25, inclusive",
               run(exact_step, [[-0.75]]), spread_check, reference_of(exact_step),
               covers=["rounded_up exact multiple", "limit inclusive"]),
        vector("teacher_forced", "fail: one ulp above a 0.25 limit", run(exact_step, [[-0.7500000000000001]]),
               spread_check, reference_of(exact_step), covers=["limit inclusive"]),
        vector("teacher_forced", "fail: 0.27 above a 0.25 limit", run(exact_step, [[-0.77]]), spread_check,
               reference_of(exact_step), covers=["rounded_up exact multiple"]),
        vector("teacher_forced", "pass: a spread of 0.25000000000000006 rounds the limit up to 0.3",
               run(overshoot, [[-0.57]]), spread_check, reference_of(overshoot),
               covers=["rounded_up binary overshoot"]),
    ]
    # The near-tie rule over one category: `ties` tie rows, then `overs`
    # over rows (tie_rows). A served top-1 differs from the reference's where
    # `served` maps a position to a token; the spread variant's rows differ
    # where `spread` maps a position to a row.
    def ties(served, spread=None, counts=(24, 26)):
        rows = tie_rows(*counts)
        tokens = [TIE_A] * len(rows)
        exact = ranked(tokens, rows)
        variant = ranked(tokens, [spread.get(index, row) for index, row in enumerate(rows)]) if spread else exact
        item = {"name": "n0", "category": "chat", "prompt_token_ids": [1, *tokens], "score_from": 1,
                "exact": exact, "bf16": variant}
        top1 = [served.get(index, TIE_A) for index in range(len(rows))]
        return run([item], [exact["logprob"]], [top1]), reference_of([item])

    def promoted(token, top):
        """A spread-variant row whose top-1 is `token`, TIE_A 0.3 below it."""
        return [[token, top], [TIE_A, top - 0.3], [TIE_C, top - 0.4], [FILLER, top - 1], [FILLER + 1, top - 2]]

    def flips(positions, token):
        return dict.fromkeys(positions, token)

    def variant(near, decisive):
        """Spread rows: near ties at the tie rows `near`, decisive flips at the over rows `decisive`."""
        return {**{index: promoted(TIE_B, -0.2) for index in near},
                **{index: promoted(TIE_D, -0.1) for index in decisive}}

    # 50 positions: 24 tie rows (0-23), then 26 over rows (24-49). The floors
    # allow 4 near ties, 4 decisive flips and 5 flips in all.
    edges, reference = ties({**flips(range(4), TIE_B), 24: TIE_B})
    five_decisive, _ = ties(flips(range(24, 29), TIE_D))
    mixed = {**flips(range(3), TIE_B), **flips(range(24, 27), TIE_D)}

    def against(served, spread):
        """(receipt, check, reference) of a served capture against a spread variant."""
        receipt_, reference_ = ties(served, spread)
        return receipt_, spread_check, reference_

    spread_near = against(flips(range(5), TIE_B), variant(range(5), ()))
    spread_decisive = against(flips(range(24, 29), TIE_D), variant((), range(24, 29)))
    spread_total = against(mixed, variant(range(3), range(24, 27)))
    vectors += [
        vector("teacher_forced", "pass: four flips at exactly 0.3 nats are near ties (8%) and one at "
               "0.30000000000000004 decisive, 10% of 50 positions in all", edges, spread_check, reference,
               covers=["top-1 near tie at limit", "top-1 near tie above limit", "near-tie rate inclusive",
                       "total rate inclusive"]),
        vector("teacher_forced", "fail: five near-tie flips, 10% of the positions", ties(flips(range(5), TIE_B))[0],
               spread_check, reference, covers=["near-tie rate bound"]),
        vector("teacher_forced", "pass: a served token outside the stored top 5 is decisive",
               ties({**flips(range(4), TIE_B), 24: TIE_D})[0], spread_check, reference,
               covers=["top-1 outside top-k"]),
        vector("teacher_forced", "fail: five decisive flips, 10% of the positions", five_decisive, spread_check,
               reference, covers=["decisive rate bound"]),
        vector("teacher_forced", "fail: three near-tie and three decisive flips, within each limit but 12% in all",
               ties(mixed)[0], spread_check, reference, covers=["total rate bound"]),
        vector("teacher_forced", "pass: a spread variant's five near ties raise the near-tie limit to 10%",
               *spread_near, covers=["near-tie limit from spread"]),
        vector("teacher_forced", "pass: a spread variant's five decisive flips raise the decisive limit to 10%",
               *spread_decisive, covers=["decisive limit from spread"]),
        vector("teacher_forced", "pass: a spread variant's six flips raise the total limit to 12%",
               *spread_total, covers=["total limit from spread"]),
        vector("teacher_forced", "fail: stored near-tie verdicts the reference does not support are recomputed",
               changed(five_decisive, lambda v: (
                   [flip.update(gap=0.1, decisive=False) for flip in v["top1_flips"]],
                   v.update(passed=True, failed_categories=[], top1_decisive=0.0, top1_decisive_flips=0,
                            top1_near_ties=5, top1_near_tie=0.1))),
               spread_check, reference, covers=["flips recomputed"]),
        vector("teacher_forced", "insufficient: no top-1 flips (a receipt before the near-tie rule)",
               changed(edges, lambda v: v.pop("top1_flips")), spread_check, reference, covers=["flips required"]),
        vector("teacher_forced", "insufficient: the top-1 flips do not add up to the disagreements",
               changed(edges, lambda v: v["top1_flips"].pop()), spread_check, reference, covers=["flips required"]),
        vector("teacher_forced", "insufficient: a recorded flip agreeing with the reference's top-1",
               changed(edges, lambda v: v["top1_flips"][0].update(served=TIE_A)), spread_check, reference,
               covers=["flips required"]),
    ]
    # The served rows bind each flip's token: a receipt that names a near-tie
    # candidate where its rows show a token outside the top 5 is refused.
    outside, _ = ties({**flips(range(4), TIE_B), 24: TIE_D})

    def rows_changed(change):
        def apply(value):
            change(value["rows"][0])
        return changed(outside, apply)

    vectors += [
        vector("teacher_forced", "insufficient: a recorded served token its served rows do not show",
               changed(outside, lambda v: v["top1_flips"][4].update(served=TIE_B, gap=0.30000000000000004)),
               spread_check, reference, covers=["served tokens from rows"]),
        vector("teacher_forced", "insufficient: served rows that differ from their digest",
               rows_changed(lambda row: row["served_rows"][24][0].__setitem__(0, TIE_B)), spread_check, reference,
               covers=["served rows bound"]),
        vector("teacher_forced", "insufficient: a served row without its scored token",
               rows_changed(lambda row: row.update(served_rows=[[[TIE_C, -0.2, 1]], *row["served_rows"][1:]],
                                                   served_rows_sha256=common.rows_digest(
                                                       [[[TIE_C, -0.2, 1]], *row["served_rows"][1:]]))),
               spread_check, reference, covers=["served rows bound"]),
        vector("teacher_forced", "insufficient: no served rows (a receipt before they were recorded)",
               rows_changed(lambda row: row.pop("served_rows")), spread_check, reference,
               covers=["served rows required"]),
    ]
    # The total cap at the two scopes the reference policy names: Qwen TP2's
    # prose-frozen category (12 flips of 141, 9 near ties, against an HF
    # variant with 11, 7 near ties) passes; the MiMo chat capture of the
    # confirm review (83 of 486: 38 near ties and 45 decisive, against a
    # variant with 65: 20 near ties and 45 decisive) fails on the cap alone,
    # the variant's 65 of 486.
    # The cap is the variant's exact rate: 21 flips of 200 (10.5%) set it, so
    # 21 served flips pass and 22 (11%, what rounding up would allow) fail.
    exact_cap = variant(range(50, 58), range(150, 163))
    at_cap = ties({**flips(range(10), TIE_B), **flips(range(100, 111), TIE_D)}, exact_cap, counts=(100, 100))
    over_cap = ties({**flips(range(11), TIE_B), **flips(range(100, 111), TIE_D)}, exact_cap, counts=(100, 100))
    vectors += [
        vector("teacher_forced", "pass: 21 of 200 flips, at a variant's exact 10.5% total", at_cap[0], spread_check,
               at_cap[1], covers=["total limit unrounded"]),
        vector("teacher_forced", "fail: 22 of 200 flips, above a variant's exact 10.5% total", over_cap[0],
               spread_check, over_cap[1], covers=["total limit unrounded"]),
    ]
    tp2 = ties({**flips(range(9), TIE_B), **flips(range(60, 63), TIE_D)}, variant(range(20, 27), range(70, 74)),
               counts=(60, 81))
    chat = ties({**flips(range(38), TIE_B), **flips(range(200, 245), TIE_D)},
                variant(range(100, 120), range(300, 345)), counts=(200, 286))
    vectors += [
        vector("teacher_forced", "pass: 12 of 141 flips (9 near ties), as Qwen TP2 prose-frozen, within the 10% cap",
               tp2[0], spread_check, tp2[1], covers=["total cap calibration"]),
        vector("teacher_forced", "fail: 83 of 486 flips (38 near ties, 45 decisive) above a variant's 65, as the "
               "MiMo chat capture", chat[0], spread_check, chat[1], covers=["total cap calibration", "total rate bound"]),
    ]
    # Category power (variant-spread.v2): a 141-position category of 60 tie
    # rows and 81 over rows, beside a 220-position category without flips
    # so that the pooled set passes (28 flips of 361 stay under its 0.08).
    # The BF16 run flips the tie rows `variant_near` (near ties). A top-1
    # limit within two binomial standard errors of the BF16 run's own rate
    # cannot judge the category, which then passes by the paired rule
    # against that run.
    def small_category(served, variant_near, alone=False, spread="bf16"):
        rows = tie_rows(60, 81)
        tokens = [TIE_A] * len(rows)
        exact = ranked(tokens, rows)
        own = ranked(tokens, [promoted(TIE_B, -0.2) if index in variant_near else row
                              for index, row in enumerate(rows)])
        items = [{"name": "f0", "category": "prose-frozen", "prompt_token_ids": [1, *tokens], "score_from": 1,
                  "exact": exact, spread: own}]
        top1 = [[served.get(index, TIE_A) for index in range(len(rows))]]
        if not alone:
            big = ranked([TIE_A] * 220, tie_rows(0, 220))
            items.append({"name": "p0", "category": "prose", "prompt_token_ids": [1, *[TIE_A] * 220],
                          "score_from": 1, "exact": big, spread: big})
            top1.append([TIE_A] * 220)
        variants = ("exact", spread)
        reference_ = {**reference_of(items), "variants": list(variants)}
        return run(items, [item["exact"]["logprob"] for item in items], top1, variants), reference_

    kv = small_category({**flips(range(9), TIE_B), **flips(range(14, 21), TIE_B)}, range(14))
    doubled = small_category({**flips(range(10), TIE_B), **flips(range(14, 32), TIE_B)}, range(14))
    powered = small_category(flips(range(15), TIE_B), ())
    alone = small_category({**flips(range(9), TIE_B), **flips(range(14, 21), TIE_B)}, range(14), alone=True)
    no_bf16 = small_category({**flips(range(9), TIE_B), **flips(range(14, 21), TIE_B)}, range(14), spread="f64")
    vectors += [
        vector("teacher_forced", "pass: 16 of 141 near ties against the BF16 run's 14 (9 shared), whose 0.10 "
               "limits sit within two standard errors of that run's rate, pass by the paired rule (7 against 5)",
               kv[0], spread_check, kv[1], covers=["category power fallback", "paired top-1 margin"]),
        vector("teacher_forced", "fail: 28 of 141 near ties against the BF16 run's 14 (10 shared): 18 against 4 "
               "exceeds the paired margin", doubled[0], spread_check, doubled[1],
               covers=["category power fallback", "paired top-1 margin"]),
        vector("teacher_forced", "fail: 15 of 141 near ties where the BF16 run has none: the limit has power, "
               "no fallback", powered[0], spread_check, powered[1], covers=["category power"]),
        vector("teacher_forced", "fail: the pooled set never falls back, though its one category passes by the "
               "paired rule", alone[0], spread_check, alone[1], covers=["pooled never falls back"]),
        vector("teacher_forced", "fail: without a BF16 run a category is held to its limits alone",
               no_bf16[0], spread_check, no_bf16[1], covers=["category power needs the BF16 run"]),
    ]
    return vectors


def modality_vectors(passing, reference, check, probes=False):
    """Unserved media is validated against a bound server, never passed."""
    pin = "d" * 64
    check = {**check, "module": "multimodal", "checkpoint_identity": pin}
    base = copy.deepcopy(passing)
    base["server"] = {"health": {"modalities": ["text", "image", "video", "audio"],
                                 "serving": {"checkpoint_identity": pin, "startup_nonce": "a" * 32}}}
    def partial(served):
        value = copy.deepcopy(base)
        value["server"]["health"]["modalities"] = served
        reasons = set()
        for index, row in enumerate(value["rows"]):
            modality = "image" if probes else row["kind"]
            if modality not in served:
                reason = f"the server does not serve {modality} input (--modalities {','.join(served)})"
                reasons.add(reason)
                value["rows"][index] = {"id": row["id"], "kind": modality, "status": "not_applicable",
                                         "reason": reason, "passed": None}
        value["not_applicable"] = sorted(reasons)
        value["passed"] = False
        return value
    prefix = "probes" if probes else "media"
    items = [("served set", base, "pass"), ("all unserved", partial(["text"]), "not_applicable")]
    mixed = partial(["text", "image"])
    if not probes:
        items.append(("served and unserved", mixed, "not_applicable"))
        failed = copy.deepcopy(mixed)
        failed["rows"][0]["logprobs"].update(mean_abs_dlogprob=5.0, max_abs_dlogprob=5.0)
        items.append(("served failure beside unserved", failed, "fail"))
    for label, change in [
            ("wrong checkpoint", lambda v: v["server"]["health"]["serving"].update(checkpoint_identity="e" * 64)),
            ("missing nonce", lambda v: v["server"]["health"]["serving"].pop("startup_nonce")),
            ("invalid served set", lambda v: v["server"]["health"].update(modalities=["text", "unknown"])),
            ("invented reason", lambda v: v.update(not_applicable=["invented"])),
            ("missing row", lambda v: v["rows"].pop()),
            ("served row marked unserved", lambda v: v["server"]["health"].update(modalities=["text", "image", "video", "audio"])),
    ]:
        items.append((label, changed(partial(["text"]), change), "insufficient"))
    result = []
    for label, value, status in items:
        item = vector("media_forced", f"{prefix} modalities: {label}", value, check, reference,
                      covers=["served modalities", "bound unserved evidence"])
        assert item["expected"]["status"] == status, item
        result.append(item)
    return result


def media_vectors():
    A, B, C = 41, 42, 43

    def case(identifier, kind, tokens, exact, spread):
        return {"id": identifier, "kind": kind, "answer_token_ids": tokens, "exact": {"exact": exact, "bf16": spread}}

    def reference_of(cases):
        return {"sha256": REFERENCE_SHA256, "gate": "exact", "variants": ["exact", "bf16"], "cases": cases, "audio_near_tie": 2.3}

    def row(item, mean, largest=None, greedy=None, served=()):
        """An item's receipt row; `served` lists (offset, token) where the
        served top-1 differs from the reference's, recorded as multimodal.py
        records them."""
        scored = len(item["answer_token_ids"])
        gate = item["exact"]["exact"]
        flipped = limits.flips([(gate["top1"][offset], token, gap_top(gate)[offset]) for offset, token in served]) \
            if served else []
        flips = [{"offset": offset, "served": token, **flip} for (offset, token), flip in zip(served, flipped)]
        top1 = list(gate["top1"])
        for offset, token in served:
            top1[offset] = token
        values = {"scored": scored, "mean_abs_dlogprob": mean,
                  "max_abs_dlogprob": scored * mean if largest is None else largest,
                  **limits.top1_summary(flips, scored), "top1_flips": flips,
                  **served_rows(item["answer_token_ids"], gate["logprob"], top1), "limits": {}, "passed": True}
        return {"id": item["id"], "kind": item["kind"], "answer": "a", "reference_answer": "a",
                "fixture_expectation_match": True, "prompt_tokens": 10, "reference_prompt_tokens": 10,
                "token_count_match": True, "latency_seconds": 0.1, "logprobs": values,
                "greedy": greedy or {"passed": True, "exact": True}, "passed": True}

    def run(cases, rows):
        return receipt("multimodal", passed=all(r["passed"] for r in rows), exact_reference_sha256=REFERENCE_SHA256,
                       rows=rows, limits={})

    cases = [case("image-0", "image", [4, 5, 6], scores([4, 5, 6], [-0.1, -0.2, -0.3]),
                  scores([4, 5, 6], [-0.3, -0.2, -0.3])),
             case("audio-0", "audio", [7, 8], scores([7, 8], [-1.0, -0.5]), scores([7, 8], [-1.0, -0.5], [7, 9]))]
    reference = reference_of(cases)
    passing = run(cases, [row(cases[0], 0.05), row(cases[1], 0.02)])
    check = {"rules": [rule(["passed"], "eq", True), rule(["exact_reference_sha256"], "eq", REFERENCE_SHA256)]}

    def impossible(label, change):
        return vector("media_forced", f"insufficient: {label}",
                      changed(passing, lambda v: v["rows"][0]["logprobs"].update(change)), check, reference,
                      covers=["impossible scores"])

    vectors = [
        vector("media_forced", "pass", passing, check, reference, covers=["runner layout"]),
        vector("media_forced", "fail: an item beyond its limits, though stored as passing",
               changed(passing, lambda v: v["rows"][1]["logprobs"].update(mean_abs_dlogprob=0.4,
                                                                         max_abs_dlogprob=0.8)), check, reference),
        vector("media_forced", "fail: an item's recorded greedy failure",
               changed(passing, lambda v: v["rows"][0]["greedy"].update(passed=False, error="no response")),
               check, reference),
        vector("media_forced", "insufficient: no items", changed(passing, lambda v: v.update(rows=[])),
               check, reference),
        vector("media_forced", "insufficient: an item scored over other positions",
               changed(passing, lambda v: v["rows"][0]["logprobs"].update(scored=2)), check, reference),
        vector("media_forced", "insufficient: another reference",
               changed(passing, lambda v: v.update(exact_reference_sha256="b" * 64)), check, reference),
        vector("media_forced", "insufficient: no reference", passing, check),
        vector("media_forced", "insufficient: the reference lacks a pinned variant on an item", passing, check,
               changed(reference, lambda r: r["cases"][1]["exact"].pop("bf16")), covers=["missing variant"]),
        vector("media_forced", "insufficient: a reference logprob above zero", passing, check,
               changed(reference, lambda r: r["cases"][0]["exact"]["exact"]["logprob"].__setitem__(0, 0.25)),
               covers=["invalid reference scores"]),
        impossible("negative mean and max |dlogprob|", {"mean_abs_dlogprob": -1, "max_abs_dlogprob": -1}),
        impossible("a negative top-1 disagreement", {"top1_disagreement": -1}),
        impossible("a top-1 disagreement above one", {"top1_disagreement": 1.5}),
        impossible("a top-1 disagreement no count of the positions gives", {"top1_disagreement": 0.4}),
        impossible("a mean above the max", {"mean_abs_dlogprob": 0.05, "max_abs_dlogprob": 0}),
    ]
    # The spread rule at its decimal step, as for teacher_forced, over one answer position.
    exact_step = [case("image-0", "image", [A], scores([A], [-0.5]), scores([A], [-0.75]))]
    overshoot = [case("image-0", "image", [A], scores([A], [-0.3]), scores([A], [-0.55]))]
    vectors += [
        vector("media_forced", "pass: a spread of exactly 0.25 limits the mean to 0.25, inclusive",
               run(exact_step, [row(exact_step[0], 0.25)]), check, reference_of(exact_step),
               covers=["rounded_up exact multiple", "limit inclusive"]),
        vector("media_forced", "fail: one ulp above a 0.25 limit",
               run(exact_step, [row(exact_step[0], 0.25000000000000006)]), check, reference_of(exact_step),
               covers=["limit inclusive"]),
        vector("media_forced", "fail: 0.27 above a 0.25 limit", run(exact_step, [row(exact_step[0], 0.27)]), check,
               reference_of(exact_step), covers=["rounded_up exact multiple"]),
        vector("media_forced", "pass: a spread of 0.25000000000000006 rounds the limit up to 0.3",
               run(overshoot, [row(overshoot[0], 0.27)]), check, reference_of(overshoot),
               covers=["rounded_up binary overshoot"]),
    ]
    # The greedy walk's near-tie rule: the served answer departs at the
    # first position to B, whose logprob is `gap` below the top-1 A.
    # 0.0 - 0.3 is exactly the image limit 2 * 0.15; -0.1 - -0.4 is
    # 0.30000000000000004. -1.0 - -3.3 is exactly the audio limit 2.3;
    # -0.05 - -2.35 is 2.3000000000000003.
    def tie(kind, top, served, label, covers):
        top5 = [[A, top], [B, served], [C, served - 1], [FILLER, served - 2], [FILLER + 1, served - 3]]
        reference_scores = ranked([A, A], [top5, top5])
        item = case(f"{kind}-0", kind, [A, A], reference_scores, reference_scores)
        greedy = {"passed": True, "exact": False, "divergence": 0, "served_token_id": B, "reference_token_id": A}
        return vector("media_forced", label, run([item], [row(item, 0.0, greedy=greedy)]), check,
                      reference_of([item]), covers=covers)

    vectors += [
        tie("image", -0.2, -0.5, "pass: a served token 0.3 below the top-1, at the image near-tie limit",
            ["near tie at limit"]),
        tie("image", -0.1, -0.4, "fail: a served token 0.30000000000000004 below the top-1",
            ["near tie above limit"]),
        tie("audio", -1.0, -3.3, "pass: a served token 2.3 below the top-1, at the audio near-tie limit",
            ["audio near tie at limit"]),
        tie("audio", -0.05, -2.35, "fail: a served audio token 2.3000000000000003 below the top-1",
            ["audio near tie above limit"]),
    ]
    walked = cases[0]
    vectors += [
        vector("media_forced", "fail: a served token outside the reference top 5",
               run([walked], [row(walked, 0.05, greedy={"passed": True, "exact": False, "divergence": 1,
                                                        "served_token_id": 77, "reference_token_id": 5})]),
               check, reference_of([walked]), covers=["outside top 5"]),
        vector("media_forced", "fail: the served answer ended inside the reference answer",
               run([walked], [row(walked, 0.05, greedy={"passed": True, "exact": False, "divergence": 2})]),
               check, reference_of([walked]), covers=["ended inside"]),
    ]
    # The near-tie rule over one item's 13 answer positions: 6 tie rows,
    # then 7 over rows (see tie_rows). One flip (7.7%) fits every limit;
    # two (15.4%) exceed the 10% total cap.
    tied = ranked([TIE_A] * 13, tie_rows(6, 7))
    item = case("image-0", "image", [TIE_A] * 13, tied, tied)
    near_one = run([item], [row(item, 0.0, served=[(0, TIE_B)])])
    one_each = run([item], [row(item, 0.0, served=[(0, TIE_B), (6, TIE_B)])])
    two_near = run([item], [row(item, 0.0, served=[(0, TIE_B), (1, TIE_B)])])
    vectors += [
        vector("media_forced", "pass: a flip at exactly 0.3 nats is a near tie, one of 13 positions", near_one,
               check, reference_of([item]), covers=["top-1 near tie at limit"]),
        vector("media_forced", "pass: a flip at 0.30000000000000004 is decisive, one of 13 positions",
               run([item], [row(item, 0.0, served=[(6, TIE_B)])]), check, reference_of([item]),
               covers=["top-1 near tie above limit"]),
        vector("media_forced", "fail: a near-tie and a decisive flip, within each limit but 15% in all", one_each,
               check, reference_of([item]), covers=["total rate bound"]),
        vector("media_forced", "fail: two near-tie flips, 15% of the positions", two_near, check,
               reference_of([item]), covers=["near-tie rate bound"]),
        vector("media_forced", "fail: a stored near-tie count the flips do not support is recomputed",
               changed(two_near, lambda v: v["rows"][0]["logprobs"].update(top1_near_ties=0, top1_near_tie=0.0,
                                                                          top1_decisive_flips=2,
                                                                          top1_decisive=2 / 13)),
               check, reference_of([item]), covers=["flips recomputed"]),
        vector("media_forced", "insufficient: an item without its top-1 flips",
               changed(near_one, lambda v: v["rows"][0]["logprobs"].pop("top1_flips")), check,
               reference_of([item]), covers=["flips required"]),
        vector("media_forced", "insufficient: a recorded served token its served rows do not show",
               changed(near_one, lambda v: v["rows"][0]["logprobs"]["top1_flips"][0].update(served=TIE_C)), check,
               reference_of([item]), covers=["served tokens from rows"]),
        vector("media_forced", "insufficient: served rows that differ from their digest",
               changed(near_one, lambda v: v["rows"][0]["logprobs"]["served_rows"][0][0].__setitem__(1, -0.01)),
               check, reference_of([item]), covers=["served rows bound"]),
        vector("media_forced", "insufficient: an item without its served rows",
               changed(near_one, lambda v: v["rows"][0]["logprobs"].pop("served_rows")), check,
               reference_of([item]), covers=["served rows required"]),
    ]
    # Category power for a media case (variant-spread.v2): 51 answer
    # positions, 30 tie rows then 21 over rows; the BF16 run's near ties at
    # rows 0-4 put its rate (5 of 51) within two standard errors of the 0.10
    # limits, so the case also passes by the paired rule against that run.
    small = ranked([TIE_A] * 51, tie_rows(30, 21))
    own = ranked([TIE_A] * 51, [[[TIE_B, -0.2], [TIE_A, -0.5], [TIE_C, -0.6], [FILLER, -1.2], [FILLER + 1, -2.2]]
                                if index < 5 else row for index, row in enumerate(tie_rows(30, 21))])
    powered = case("image-1", "image", [TIE_A] * 51, small, own)
    shared = run([powered], [row(powered, 0.0, served=[(index, TIE_B) for index in (0, 1, 2, 3, 10, 11)])])
    apart = run([powered], [row(powered, 0.0, served=[(index, TIE_B) for index in range(10, 24)])])
    vectors += [
        vector("media_forced", "pass: 6 of 51 near ties against the BF16 run's 5 (4 shared) pass by the paired rule",
               shared, check, reference_of([powered]), covers=["case power fallback"]),
        vector("media_forced", "fail: 14 of 51 near ties apart from the BF16 run's 5 exceed the paired margin",
               apart, check, reference_of([powered]), covers=["case power fallback"]),
    ]
    return vectors + modality_vectors(passing, reference, check)


def reference_vectors():
    """The pin and binding rules of reading a reference (common.pinned_reference
    and pinned_binding), which kinds.rescore cannot exercise because it is
    given the reference already read. The checkpoint pin is taken as
    verified against its lock."""
    checkpoint = {"lock": "models/example/checkpoint-files.json", "repository": "example/model",
                  "revision": "0" * 40, "served_model_name": "Example", "source_lock_sha256": "c" * 64,
                  "checkpoint_identity": "d" * 64}
    provenance = {"checkpoint": {key: checkpoint[key] for key in
                                 ("repository", "revision", "served_model_name", "source_lock_sha256",
                                  "checkpoint_identity")}}
    tokens = [2, 3]
    fixture = {"schema": "chopinfer.example.teacher-reference.v1", "provenance": provenance,
               "texts": [{"name": "t0", "category": "chat", "prompt_token_ids": [1, *tokens], "score_from": 1,
                          "exact": scores(tokens, [-0.5, -1.0]), "bf16": scores(tokens, [-0.52, -1.1])}]}

    def entry(name, value, covers, pin=None, raw=None):
        data = raw if raw is not None else json.dumps(value, sort_keys=True)
        spec = {"sha256": hashlib.sha256(data.encode()).hexdigest(), "schema": fixture["schema"],
                "variants": ["exact", "bf16"], **(pin or {})}
        result = read_reference(spec, checkpoint, data)
        return {"name": name, "covers": covers, "pin": spec, "checkpoint": checkpoint, "fixture": data,
                "expected": result}

    other_lock = {"lock": {"path": "models/other/checkpoint-files.json", "variant": None},
                  "digests": {"lock": "e" * 64}}
    return [
        entry("accepted: an unbound fixture is read as unbound (transitional)", fixture, ["unbound"]),
        entry("refused: a binding to another checkpoint lock", dict(fixture, binding=other_lock),
              ["bound to another lock"]),
        entry("refused: bytes other than the pinned digest", fixture, ["digest mismatch"],
              pin={"sha256": "f" * 64}),
        entry("refused: another schema", dict(fixture, schema="chopinfer.example.teacher-reference.v0"),
              ["schema mismatch"]),
        entry("refused: provenance of another checkpoint revision",
              changed(fixture, lambda v: v["provenance"]["checkpoint"].update(revision="1" * 40)),
              ["provenance mismatch"]),
        entry("refused: a pinned variant missing from a text",
              changed(fixture, lambda v: v["texts"][0].pop("bf16")), ["missing variant"]),
    ]


def read_reference(spec, checkpoint, data):
    """What reading a fixture with `data` bytes under `spec` gives: accepted
    with its binding status, or refused with the reason."""
    import tempfile
    from unittest.mock import patch
    import common
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "reference.json"
        path.write_text(data)
        config = {"qualification": {"references": {"teacher": {**spec, "path": str(path)}},
                                    "checkpoint": checkpoint}}
        try:
            with patch.object(common, "checkpoint_pin", return_value=checkpoint):
                value = common.pinned_reference("teacher", path, config)
                binding = common.pinned_binding("teacher", value, config)
        except ValueError as error:
            return {"status": "refused", "error": str(error).replace(str(path), "<fixture>")}
    return {"status": "accepted", "binding": binding}


def greedy_vectors():
    baseline = receipt("greedy-equality-ar", passed=True, arm="ar", k=0, concurrency=1, prompt_tokens=4,
                       max_tokens=3, cold_cache=True, reference_sha256=REFERENCE_SHA256,
                       requests={"t0": [5, 6, 7], "t1": [8, 9]}, finish_reasons={"t0": "length", "t1": "stop"})
    record = dict(baseline, check="greedy-equality-spec", arm="spec", k=3)
    exact, informational = {"rule": "exact"}, {"rule": "informational"}
    other = changed(record, lambda v: v["requests"].update(t0=[5, 6, 4]))
    return [
        vector("greedy_pin", "pass", record, exact, baseline),
        vector("greedy_pin", "fail: a token differs", other, exact, baseline),
        vector("greedy_pin", "informational: a token differs under rule informational", other, informational,
               baseline),
        vector("greedy_pin", "fail: a finish reason differs",
               changed(record, lambda v: v["finish_reasons"].update(t1="length")), exact, baseline),
        vector("greedy_pin", "insufficient: a request without a finish reason",
               changed(record, lambda v: v["finish_reasons"].pop("t1")), exact, baseline),
        vector("greedy_pin", "insufficient: no baseline", record, exact),
    ]


def determinism_vectors():
    import determinism as module
    names = module.request_names(False)
    count = 2800

    def row(name):
        tokens = 1 if name in module.PRIMERS else 2
        salt = f"salt-long-primer-{module.HITS[name]}" if name in module.HITS else f"salt-{name}"
        prompt = module.PRIMERS.get(name) or (count if name.startswith("long") else 8)
        return {"name": name, "cache_salt": salt, "text": " ab"[:tokens + 1], "reasoning": None,
                "finish_reason": "length", "token_ids": [1, 2][:tokens], "tokens": [" a", "b"][:tokens],
                "logprobs": [-0.125, -0.25][:tokens], "top_logprobs": [[[" a", -0.125]], [["b", -0.25]]][:tokens],
                "usage": {"prompt_tokens": prompt, "completion_tokens": tokens,
                          "prompt_tokens_details": {"cached_tokens": module.HITS[name] // 64 * 64
                                                    if name in module.HITS else 0}}}

    def record(arm):
        return receipt(f"determinism-{arm}", passed=True, arm=arm, run_id="run", concurrency=1,
                       sampling=module.SAMPLING,
                       request_set={"names": names, "sha256": "c" * 64, "long_prompt_tokens": count,
                                    "long_prompt_token_ids_sha256": "d" * 64},
                       image={"included": False, "png_sha256": None, "reason": "no image inputs"},
                       cold_repeat=dict(module.COLD_REPEAT, equal=True),
                       prefix_hit=dict(module.PREFIX_HIT, anchors=list(module.HIT_ANCHORS), equal=True),
                       requests=[row(name) for name in names])

    left, right = record("spec"), record("ar")
    check = {}
    return [
        vector("determinism", "pass", left, check, right),
        vector("determinism", "fail: a logprob differs by one ulp across the arms",
               changed(left, lambda v: v["requests"][0]["logprobs"].__setitem__(1, -0.25000000000000006)),
               check, right),
        vector("determinism", "fail: a prefix hit differs from its cold send",
               changed(left, lambda v: v["requests"][-1].update(token_ids=[1, 3])), check,
               changed(right, lambda v: v["requests"][-1].update(token_ids=[1, 3]))),
        vector("determinism", "insufficient: a request missing",
               changed(left, lambda v: v["requests"].pop(0)), check, right),
        vector("determinism", "insufficient: no record of the other arm", left, check),
    ]


def api_vectors():
    rows = [{"name": name, "passed": True, "latency_seconds": 0.01} for name in kinds.API_CASES]
    passing = receipt("api", passed=True, rows=rows, stream_text="Hello")
    check = {"rules": [rule(["passed"], "eq", True)]}
    return [
        vector("api", "pass", passing, check),
        vector("api", "fail: a case failed", changed(passing, lambda v: (
            v["rows"][3].update(passed=False), v.update(passed=False, errors=[{"name": v["rows"][3]["name"],
                                                                              "error": "missing choice"}]))), check),
        vector("api", "fail: a failed case under a passing verdict is recomputed",
               changed(passing, lambda v: v["rows"][3].update(passed=False)), check),
        vector("api", "insufficient: a case missing", changed(passing, lambda v: v["rows"].pop()), check),
        vector("api", "insufficient: no cases", changed(passing, lambda v: v.update(rows=[])), check),
    ]


def structured_vectors():
    from structured import summarize
    settings = {"cases": list(kinds.STRUCTURED_CASES), "repeat": 1, "stream": True, "max_tokens": 512,
                "reasoning_max_tokens": 4096, "temperature": 1.0, "seed": None}
    rows = [{"name": case + suffix, "stream": bool(suffix), "status": "passed", "latency_seconds": 0.5,
             "completion_tokens": 10, "tokens_per_second": 20.0, "finish_reason": "stop"}
            for case in kinds.STRUCTURED_CASES for suffix in ("", "+stream")]

    def run(rows):
        cases = {}
        for row in rows:
            cases.setdefault(row["name"], []).append(row)
        failed = [{"name": row["name"], "error": row.get("error", "")} for row in rows if row["status"] == "failed"]
        return receipt("structured", passed=not failed, errors=failed, settings=settings, rows=rows,
                       summary={"overall": summarize(rows), "cases": {k: summarize(v) for k, v in cases.items()}})

    check = {"rules": [rule(["passed"], "eq", True), rule(["summary", "overall", "validity_rate"], "ge", 1.0),
                       rule(["summary", "overall", "refused"], "eq", 0),
                       rule(["summary", "overall", "truncated"], "eq", 0)]}
    passing = run(rows)
    truncated = run([dict(row, status="truncated", finish_reason="length") if index == 5 else row
                     for index, row in enumerate(rows)])
    return [
        vector("structured", "pass", passing, check),
        vector("structured", "fail: a truncated output", truncated, check),
        vector("structured", "fail: a stored summary its rows do not support is recomputed",
               changed(truncated, lambda v: v.update(summary=passing["summary"])), check),
        vector("structured", "insufficient: a case missing", changed(passing, lambda v: v["rows"].pop()), check),
        vector("structured", "insufficient: the settings skip a case",
               changed(passing, lambda v: v["settings"].update(cases=settings["cases"][1:])), check),
    ]


def probe_vectors():
    reference = {"format": "image-probes", "sha256": REFERENCE_SHA256, "rule": {"near_tie": 0.3},
                 "order": ["short", "description"],
                 "probes": {"short": {"kind": "short", "scored": 1,
                                      "steps": [[1, -0.1, [[1, -0.1, [65]], [2, -0.4, [66]], [3, -0.6, [67]]]]]},
                            "description": {"kind": "free-text", "scored": 2,
                                            "forced": [[10, -0.1, 10], [20, -0.1, 20]],
                                            "top": [[[10, -0.1], [11, -0.5]], [[20, -0.1], [21, -0.5]]],
                                            "limits": {"mean_abs_dlogprob": 0.15, "top1_decisive": 0.08,
                                                       "top1_near_tie": 0.08, "top1_disagreement": 0.1}}}}
    passing = receipt("multimodal", reference={"sha256": REFERENCE_SHA256}, passed=True, rows=[
        {"id": "short", "greedy": {"exact": False, "divergence": 0, "reference_margin": 0.3, "passed": True},
         "fixture_expectation_match": True, "token_count_match": True, "passed": True},
        {"id": "description", "teacher_forced": {"scored": 2, "mean_abs_dlogprob": 0.15,
          "top1_disagreement": 0.0, "passed": True, "near_tie_report": {"available": True, "gates": False,
          "flips": [], "near_tie": 0, "confident": 0}},
         "fixture_expectation_match": True, "token_count_match": True, "passed": True}])
    check = {"rules": [rule(["passed"], "eq", True)]}
    cases = [
        ("probe limits inclusive", "pass", lambda value: None, ("probe near tie at limit", "probe forced at limit")),
        ("probe margin above limit", "fail", lambda value: value["rows"][0]["greedy"].update(reference_margin=0.5),
         ("probe near tie above limit",)),
        ("probe margin is the reference token's own gap", "insufficient", lambda value: value["rows"][0]["greedy"].update(reference_margin=0.0),
         ("probe self gap",)),
        ("probe margin not in reference top-k", "insufficient", lambda value: value["rows"][0]["greedy"].update(reference_margin=0.30001),
         ("probe margin not in reference top-k",)),
        ("probe forced above limit", "fail",
         lambda value: value["rows"][1]["teacher_forced"].update(mean_abs_dlogprob=0.15001), ("probe forced above limit",)),
        ("probe incomplete", "insufficient", lambda value: value["rows"].pop(), ("probe coverage",)),
        ("probe impossible count", "insufficient",
         lambda value: value["rows"][1]["teacher_forced"].update(top1_disagreement=0.1), ("probe impossible scores",)),
        ("probe wrong reference", "insufficient", lambda value: value["reference"].update(sha256="b" * 64), ("probe digest",)),
        ("probe missing flip evidence", "insufficient",
         lambda value: value["rows"][1]["teacher_forced"].pop("near_tie_report"), ("probe flip evidence",)),
    ]
    result = []
    for name, status, change, covers in cases:
        item = vector("media_forced", name, changed(passing, change), check, reference, covers)
        assert item["expected"]["status"] == status, name
        result.append(item)
    return result + modality_vectors(passing, reference, check, probes=True)


def frontend_vectors():
    passing = {name: {"rows": 2, "reference_rows": 2, "mismatches": [], "media_refused": [], "agrees": True, "key": name}
               for name in ("chat-renders.json", "tokenizer-golden.json")}
    check = {"rules": [rule([name, "agrees"], "eq", True) for name in passing]}
    reference = {"files": {name: {"input": "frontend_capture" if name == "chat-renders.json" else "tokenizer_goldens", "file": name if name == "chat-renders.json" else "", "sha256": REFERENCE_SHA256, "rows": 2} for name in passing}}
    passing["reference"] = reference
    cases = [
        ("exact IDs", "pass", lambda value: None, ("golden equality",)),
        ("mismatched IDs", "fail", lambda value: value["chat-renders.json"].update(agrees=False, mismatches=[1]),
         ("golden mismatch",)),
        ("missing tokenizer", "insufficient", lambda value: value.pop("tokenizer-golden.json"), ("golden missing",)),
        ("zero renders", "insufficient", lambda value: value["chat-renders.json"].update(rows=0), ("golden empty",)),
        ("wrong golden binding", "insufficient", lambda value: value["reference"]["files"]["chat-renders.json"].update(sha256="b" * 64), ("golden binding",)),
        ("unequal golden counts", "insufficient", lambda value: value["chat-renders.json"].update(reference_rows=3), ("golden counts",)),
        ("forged reference count", "insufficient", lambda value: value["chat-renders.json"].update(rows=1, reference_rows=1), ("golden pinned counts",)),
        ("contradictory equality", "insufficient", lambda value: value["chat-renders.json"].update(mismatches=[1]),
         ("golden consistency",)),
    ]
    result = []
    for name, status, change, covers in cases:
        item = vector("frontend_goldens", name, changed(passing, change), check, reference, covers=covers)
        assert item["expected"]["status"] == status, name
        result.append(item)
    return result


def all_media_vectors():
    return media_vectors() + probe_vectors()


GENERATORS = {"teacher_forced": teacher_vectors, "media_forced": all_media_vectors, "greedy_pin": greedy_vectors,
              "determinism": determinism_vectors, "api": api_vectors, "structured": structured_vectors,
              "references": reference_vectors, "frontend_goldens": frontend_vectors}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="directory for the reviewed vector files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, generate in GENERATORS.items():
        version = kinds.KINDS[name].version if name in kinds.KINDS else "references.v1"
        value = {"schema": SCHEMA, "kind": name, "version": version, "vectors": generate()}
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
        print(f"{name}: {', '.join(vector['expected']['status'] for vector in value['vectors'])}")


if __name__ == "__main__":
    main()
