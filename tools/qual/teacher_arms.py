"""Compare two bound teacher arms on the gate's pinned teacher reference.

    python3 tools/qual/teacher_arms.py --arm NAME=RECEIPT=CAPTURE --arm NAME=RECEIPT=CAPTURE

Each arm is a passing live teacher receipt (chopinfer eval run) and the
capture its run wrote (--capture-out), such as one serving configuration's arm recorded
by an earlier gate run and another's by this one, on a host that serves one
configuration at a time. An arm is refused unless its receipt passed live
(not rescored, not partial, not mechanics), its capture holds the served rows
the receipt recorded (each text's served-rows digest), is the whole file the
receipt's run wrote (`capture_sha256`, so its per-position scores are the
ones that run produced) under the same server identity, both name the
gate's pinned teacher reference, and its server
loaded the gate's pinned checkpoint. The two arms must share the checkpoint,
the build (package identity, version and source commit: serving options
such as the KV element or context-parallel size are what differ) and the
pinned reference's workload: its text names, prompt token digests and score
starts, with a score for each of its scored positions.

The receipt (`teacher-arms`) reports each arm against the reference's gate
variant (teacher.score: pooled and per category mean and maximum |dlogprob|,
decisive and near-tie top-1 rates), the second arm's mean |dlogprob| minus
the first's, and the two arms against each other (mean |logprob difference|
and the share of positions whose top-1 differs). It passes when both arms
are bound and comparable; the differences are reported, not gated: a margin
between two serving configurations is a separate, approved policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from common import (bound_live, checkpoint_pin, input_path, pinned_reference, qualification_config, receipt,
                    rows_digest)
import teacher

# The serving identity fields two arms of one build share.
BUILD = ("package_identity", "version", "source_commit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arm(name, receipt_path, capture_path, config):
    """The arm's receipt and capture, checked as described above."""
    value = json.loads(Path(receipt_path).read_text())
    capture = json.loads(Path(capture_path).read_text())
    require(value.get("schema") == "chopinfer-qualification/v1"
            and value.get("check") == config["qualification"]["teacher_receipt"],
            f"{name}: {receipt_path} is not a teacher receipt")
    require(value.get("passed") is True and not value.get("mechanics_only") and not value.get("skipped")
            and not value.get("errors"), f"{name}: the teacher receipt did not pass a complete live run")
    require(capture.get("schema") == config["qualification"]["teacher_capture_schema"],
            f"{name}: {capture_path} is not a teacher capture")
    pinned = config["qualification"]["references"]["teacher"]["sha256"]
    require(value.get("reference_sha256") == capture.get("reference_sha256") == pinned,
            f"{name}: the receipt and capture must both name the pinned teacher reference")
    require(value.get("capture_sha256") == hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
            f"{name}: the capture is not the one its receipt's run wrote (capture_sha256)")
    require(capture.get("server") == value.get("server"), f"{name}: the capture's server is not the receipt's")
    require(bound_live(value["server"], config), f"{name}: the receipt's server was not bound live to the pin")
    rows = value.get("rows") or []
    texts = capture.get("texts") or []
    require([row.get("name") for row in rows] == [text["name"] for text in texts]
            and all(row.get("served_rows_sha256") == rows_digest(text["served_rows"])
                    for row, text in zip(rows, texts)),
            f"{name}: the capture does not hold the served rows its receipt recorded")
    return value, capture


def workload(capture):
    return [[text["name"], text["prompt_token_ids_sha256"], text["score_from"]] for text in capture["texts"]]


def scores_reference(capture, reference, name):
    """Refuse a capture whose texts are not the pinned reference's workload,
    each scored at every scored position."""
    expected = [[text["name"], teacher.ids_digest(text["prompt_token_ids"]), text["score_from"]] for text in reference]
    require(workload(capture) == expected, f"{name}: the capture did not score the pinned reference's workload")
    for text, pinned in zip(capture["texts"], reference):
        count = len(pinned[teacher.GATE_VARIANT]["logprob"])
        require(len(text["logprob"]) == len(text["top1"]) == len(text["served_rows"]) == count,
                f"{name}: {text['name']} holds other than its {count} scored positions")


def plain(positions):
    count = len(positions)
    return {"positions": count,
            "mean_abs_dlogprob": statistics.mean(p["difference"] for p in positions),
            "max_abs_dlogprob": max(p["difference"] for p in positions),
            "top1_disagreement": sum(p["disagrees"] for p in positions) / count}


def report(positions):
    groups = {}
    for p in positions:
        groups.setdefault(p["category"], []).append(p)
    return {"pooled": {**plain(positions), **teacher.summary(positions)},
            "by_category": {name: {**plain(rows), **teacher.summary(rows)} for name, rows in sorted(groups.items())}}


def between(first, second):
    """The two arms' captures against each other, pooled and per category."""
    groups = {}
    for a, b in zip(first["texts"], second["texts"], strict=True):
        require(len(a["logprob"]) == len(b["logprob"]), f"{a['name']}: the arms scored different positions")
        rows = groups.setdefault(a["category"], [])
        rows.extend((abs(x - y), p != q) for x, y, p, q in zip(a["logprob"], b["logprob"], a["top1"], b["top1"]))

    def summarize(rows):
        return {"positions": len(rows), "mean_abs_logprob_difference": statistics.mean(d for d, _ in rows),
                "max_abs_logprob_difference": max(d for d, _ in rows),
                "top1_differs": sum(flip for _, flip in rows) / len(rows)}

    pooled = [row for rows in groups.values() for row in rows]
    return {"pooled": summarize(pooled),
            "by_category": {name: summarize(rows) for name, rows in sorted(groups.items())}}


def compare(arms, config=None):
    """The teacher-arms result of `arms` ([(name, receipt, capture)], two)."""
    config = qualification_config() if config is None else config
    require(len(arms) == 2 and arms[0][0] != arms[1][0], "compare exactly two differently named arms")
    loaded = [(name, *load_arm(name, path, capture, config)) for name, path, capture in arms]
    (first, first_receipt, first_capture), (second, second_receipt, second_capture) = loaded
    require(workload(first_capture) == workload(second_capture), "the arms scored different workloads")
    pin = checkpoint_pin(config)["checkpoint_identity"]
    require(first_receipt["server"]["checkpoint"]["checkpoint_identity"]
            == second_receipt["server"]["checkpoint"]["checkpoint_identity"] == pin,
            "the arms served different checkpoints")
    builds = [{key: ((value["server"].get("health") or {}).get("serving") or {}).get(key) for key in BUILD}
              for value in (first_receipt, second_receipt)]
    require(builds[0] == builds[1], f"the arms ran different builds: {builds[0]} and {builds[1]}")
    reference_path = input_path("teacher_reference", config=config)
    reference = pinned_reference("teacher", reference_path, config)["texts"]
    for name, _, capture in loaded:
        scores_reference(capture, reference, name)
    scored = {name: report(teacher.score(capture["texts"], reference, teacher.GATE_VARIANT))
              for name, _, capture in loaded}
    mean = {name: value["pooled"]["mean_abs_dlogprob"] for name, value in scored.items()}
    difference = {"pooled": mean[second] - mean[first],
                  "by_category": {category: scored[second]["by_category"][category]["mean_abs_dlogprob"]
                                  - row["mean_abs_dlogprob"]
                                  for category, row in scored[first]["by_category"].items()}}
    rows = [{"name": a["name"], "category": a["category"], "positions": len(a["logprob"])}
            for a in first_capture["texts"]]
    profiles = {name: ((value["server"].get("health") or {}).get("serving") or {}).get("profile")
                for name, value, _ in loaded}
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {"passed": True, "arms": [first, second],
            "reference_sha256": config["qualification"]["references"]["teacher"]["sha256"],
            "reference": str(reference_path), "checkpoint_identity": pin, "profiles": profiles,
            "inputs": {name: {"receipt": str(path), "receipt_sha256": sha256(path),
                              "capture": str(capture), "capture_sha256": sha256(capture)}
                       for name, path, capture in arms},
            "scored": sum(row["positions"] for row in rows), "rows": rows, "against_reference": scored,
            "mean_abs_dlogprob_difference": difference, "between_arms": between(first_capture, second_capture)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=RECEIPT=CAPTURE")
    args = parser.parse_args()
    arms = []
    for value in args.arm:
        parts = value.split("=")
        if len(parts) != 3 or not all(parts):
            parser.error(f"--arm needs NAME=RECEIPT=CAPTURE: {value}")
        arms.append(tuple(parts))
    try:
        result = compare(arms)
    except (OSError, ValueError, KeyError) as error:
        result = {"passed": False, "errors": [str(error)], "arms": [name for name, _, _ in arms]}
    servers = {}
    for name, path, _ in arms:
        try:
            servers[name] = json.loads(Path(path).read_text()).get("server")
        except (OSError, ValueError):
            servers[name] = None
    receipt("teacher-arms", {"arms": servers}, result)
    print(json.dumps({key: result.get(key) for key in ("passed", "errors", "mean_abs_dlogprob_difference")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
