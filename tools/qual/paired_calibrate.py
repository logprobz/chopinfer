"""Report accepted repeat spread beside the reviewed neutral-change margins: a
margin tool for a model's declared paired_teacher margins."""

import argparse
import json
from pathlib import Path

import common
import kinds
import paired_teacher as paired
import teacher


def calibrate(first, second, reference, proposed):
    """Report repeat spread and retain the reviewed neutral-change margins.
    The two captures are accepted repeats of one install from two server
    starts."""
    scores = []
    for capture in (first, second):
        captured = paired.capture_rows(capture, reference)
        paired.require(paired.absolute_score(capture, reference, captured)["passed"],
                       "accepted repeat fails the absolute gate")
        scores.append(teacher.score(captured, reference["texts"], reference["gate"]))
    paired.require(common.live_identity(first["server"])["startup_nonce"] !=
                   common.live_identity(second["server"])["startup_nonce"], "repeats use one startup")
    directions = []
    for candidate, accepted in ((scores[0], scores[1]), (scores[1], scores[0])):
        pooled = paired.scope(candidate, accepted, proposed)
        paired.require(pooled["evidence_sufficient"], "repeat lacks pooled CI coverage")
        groups = {}
        for c, a in zip(candidate, accepted, strict=True):
            group = groups.setdefault(c["category"], ([], []))
            group[0].append(c)
            group[1].append(a)
        categories = {name: paired.scope(c, a, proposed, pooled=False)
                      for name, (c, a) in sorted(groups.items())}
        paired.holm(categories, proposed["category_alpha"])
        paired.require(pooled["passed"] and all(row["passed"] for row in categories.values()),
                       "repeat spread exceeds the approved proposal")
        directions.append({"pooled": pooled, "categories": categories})
    observed_spread = {
        "mean_margin": max(0.0, *(d["pooled"]["mean_gap_ci95"][1] for d in directions)),
        "category_mean_margin": max(0.0, *(row["mean_error_change"] for d in directions
                                          for row in d["categories"].values())),
        "top1_delta": max(0.0, *(d["pooled"]["top1_net_rate_ci95"][1] for d in directions)),
        "tail_net_allowance": max(0, *(d["pooled"]["tail"]["net_added"] for d in directions)),
    }
    margins = {key: proposed[key] for key in observed_spread}
    return {"schema": "paired-repeat-calibration/v1", "captures": [paired.digest(first), paired.digest(second)],
            "reference_sha256": reference["sha256"],
            "margins": margins, "observed_spread": observed_spread, "directions": directions,
            "rationale": "Report the maximum nonnegative repeat upper CI95 for pooled mean and top-one, "
                         "the largest category point increase and net tail count. Deterministic repeat "
                         "spread does not bound legitimate neutral numerics changes. Retain the reviewed "
                         "proposal: 0.005-nat pooled upper CI95, 0.01-nat category point mean, 1% pooled "
                         "top-one upper CI95, Holm alpha 0.05 and two net tail positions above 0.5 nats. "
                         "The accepted alpha.2 TP2-versus-TP4 neutral evidence passes these rules. "
                         "The procedure was approved on 2026-10-03 and margin adoption corrected "
                         "on 2026-10-04; zero repeat spread must not adopt zero margins."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = common.qualification_config()
    template = paired.read_json(Path(__file__).parent / "templates/paired-teacher.json")
    reference = kinds.KINDS["paired_teacher"].reference(template, config)
    policy = template["constants"]["paired_teacher"]["POLICY"]["value"]
    result = calibrate(paired.read_json(args.first), paired.read_json(args.second), reference, policy)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
