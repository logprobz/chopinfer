"""Probe where the AR and speculative GSM8K arms part, and judge the pair.

Greedy GSM8K answers of two arms part at a first differing output token. When
that token's two candidates are a near tie, which arm takes which reading is
decided by the step's batch shape, not by speculation (docs/numerics.md,
"Batch shape"). For every item exactly one arm answered correctly, this reads
both arms' retained responses (gsm8k.py --response-dir), each refused unless
its bytes are the ones its score receipt row bound (`response_sha256`) and it
reads as that row does. It finds the first output token where they differ
and teacher-forces the prompt and the shared prefix on a live server, one
request at a time with a fresh cache salt: the prompt-logprobs row that
predicts the branch gives both branch tokens' logprobs and ranks. Prompt
scoring does not use speculation, so an AR and a speculative server measure
the same model. The gap is logprob(AR token) - logprob(speculative token).
A measured item is a near-tie branch when both tokens rank within the top 20
and the gap is at most limits.NEAR_TIE_NATS in magnitude (limits.near_tie),
and decisive otherwise. An item that could not be measured (a missing or
unbound response, an arm that ended where the other continued, a failed
request) is unmeasured.

`pair` is the paired rule of the release preview's gsm8k-paired check: both
arms score at least the model's declared floor (preview.gsm8k_floor, out of
200), every lost item is measured, and speculation
loses at most one item decisively. It recomputes each verdict from the
recorded measurement. Near-tie losses are recorded, not counted; lost items
without a measurement, or without a probe, are refused as insufficient
evidence.

  gsm8k_branches.py --url URL --model NAME --ar AR_RECEIPT --spec SPEC_RECEIPT

reads each arm's GSM8K receipt and the `responses` directory beside it and
writes a `gsm8k-branches` receipt.
"""

import hashlib
import json
from pathlib import Path
import uuid

from common import Client, arguments, receipt
from gsm8k import parse, screen_floor
import limits as rules
from teacher import response_prompt_rows


CHECK = "gsm8k-branches"
TOP_LOGPROBS = 20
DECISIVE_LOSSES = 1


def discordant(ar, spec):
    """The items speculation lost (AR correct, speculative wrong) and gained."""
    left, right = ({row["index"]: row["correct"] for row in value["rows"]} for value in (ar, spec))
    if left.keys() != right.keys():
        raise ValueError("GSM8K arms scored different items")
    lost = sorted(index for index in left if left[index] and not right[index])
    gained = sorted(index for index in left if not left[index] and right[index])
    return lost, gained


def token_ids(retained):
    """The prompt and generated token IDs of a retained chat response."""
    response = retained["response"]
    prompt, generated = response.get("prompt_token_ids"), response["choices"][0].get("token_ids")
    if not isinstance(prompt, list) or not isinstance(generated, list):
        raise ValueError("the response has no token IDs")
    return prompt, generated


def branch(ar, spec):
    """The shared prompt, the shared generated prefix and each arm's first
    token after it (None for an arm that ended there)."""
    if ar["body"]["messages"] != spec["body"]["messages"]:
        raise ValueError("the arms answered different prompts")
    (prompt, left), (other, right) = token_ids(ar), token_ids(spec)
    if prompt != other:
        raise ValueError("the arms' prompt token IDs differ")
    offset = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
    if offset == len(left) == len(right):
        raise ValueError("the arms generated the same tokens")
    return (prompt, left[:offset], left[offset] if offset < len(left) else None,
            right[offset] if offset < len(right) else None)


def ranked(value):
    """A prompt-logprob entry's rank when it is within the top TOP_LOGPROBS,
    else None: the actual prompt token is listed without a rank (or past it)
    when it is outside."""
    rank = value.get("rank")
    return rank if isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= TOP_LOGPROBS else None


def probe_branch(client, model, prompt, prefix, ar_token, spec_token):
    """Both branch tokens' logprobs and ranks in the server's row after
    `prefix`, and their gap."""
    ids = prompt + prefix + [ar_token]
    body = {"model": model, "prompt": ids, "echo": True, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": TOP_LOGPROBS, "return_token_ids": True, "cache_salt": uuid.uuid4().hex}
    observed, _ = client.request("/v1/completions", body)
    returned, rows = response_prompt_rows(observed)
    if returned != ids:
        raise ValueError("returned prompt token IDs differ from the request")
    row = {int(token): value for token, value in rows[-1].items()}
    ar, spec = row[ar_token], row.get(spec_token)
    logprobs = {token: value["logprob"] for token, value in row.items()}
    return {"ar_logprob": ar["logprob"], "ar_rank": ranked(ar),
            "spec_logprob": None if spec is None else spec["logprob"],
            "spec_rank": None if spec is None else ranked(spec),
            "gap": None if spec is None else ar["logprob"] - spec["logprob"],
            "top": sorted(([token, value] for token, value in logprobs.items()), key=lambda pair: -pair[1])}


def verdict(item):
    """An item's verdict from its recorded measurement: unmeasured without
    one; a near-tie branch when both branch tokens rank within the top
    TOP_LOGPROBS and their gap is a near tie; decisive otherwise."""
    if not item.get("measured"):
        return "unmeasured"
    ranks = (item.get("ar_rank"), item.get("spec_rank"))
    if not all(isinstance(rank, int) and 1 <= rank <= TOP_LOGPROBS for rank in ranks):
        return "decisive"
    return "near_tie" if rules.near_tie(item["ar_logprob"] - item["spec_logprob"]) else "decisive"


def bound_response(directory, row):
    """The retained response of a scored row, refused unless its bytes are
    the ones the score receipt recorded and it reads as the row does."""
    path = directory / f"{row['index']}.json"
    data = path.read_bytes()
    if not row.get("response_sha256") or hashlib.sha256(data).hexdigest() != row["response_sha256"]:
        raise ValueError(f"the retained response {path} is not the one its score receipt recorded")
    retained = json.loads(data)
    choice = retained["response"]["choices"][0]
    if (parse(choice["message"].get("content"))[0], choice["finish_reason"]) != (row["actual"], row["finish_reason"]):
        raise ValueError(f"the retained response {path} does not read as its scored row")
    return retained


def probe_item(client, model, rows, outcome, ar_responses, spec_responses):
    """One discordant item, from the scored rows of both arms. An item that
    cannot be measured records why and stays unmeasured."""
    ar_row, spec_row = rows
    item = {"index": ar_row["index"], "outcome": outcome, "measured": False}
    try:
        ar, spec = bound_response(ar_responses, ar_row), bound_response(spec_responses, spec_row)
        item["responses"] = {"ar": ar_row["response_sha256"], "spec": spec_row["response_sha256"]}
        prompt, prefix, ar_token, spec_token = branch(ar, spec)
        item.update(prompt_tokens=len(prompt), branch_offset=len(prefix), ar_token=ar_token, spec_token=spec_token)
        if ar_token is None or spec_token is None:
            raise ValueError("one arm ended where the other continued")
        item.update(probe_branch(client, model, prompt, prefix, ar_token, spec_token), measured=True)
        if item["ar_rank"] is None or item["spec_rank"] is None:
            item["reason"] = f"a branch token is outside the top {TOP_LOGPROBS}"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        item["reason"] = str(error)
    item["verdict"] = verdict(item)
    return item


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(client, model, ar_path, spec_path):
    """Every discordant item of two GSM8K receipts, probed in index order."""
    ar, spec = (json.loads(Path(path).read_text()) for path in (ar_path, spec_path))
    lost, gained = discordant(ar, spec)
    left, right = ({row["index"]: row for row in value["rows"]} for value in (ar, spec))
    outcomes = sorted([(index, "lost") for index in lost] + [(index, "gained") for index in gained])
    items = [probe_item(client, model, (left[index], right[index]), outcome, Path(ar_path).parent / "responses",
                        Path(spec_path).parent / "responses") for index, outcome in outcomes]
    return {"ar": {"receipt": str(ar_path), "sha256": digest(ar_path)},
            "spec": {"receipt": str(spec_path), "sha256": digest(spec_path)},
            "near_tie_nats": rules.NEAR_TIE_NATS, "top_logprobs": TOP_LOGPROBS,
            "lost": lost, "gained": gained, "items": items}


def pair(ar, spec, branches=None, floor=None):
    """The paired GSM8K rule over two arms' receipts and, when speculation
    lost items, the probe of their branches (`probe`'s result for these
    receipts), at `floor` correct per arm (default: the declared floor of
    CHOPINFER_QUAL_CONFIG's model). Each item's verdict is recomputed from
    its recorded measurement, which must come from the responses the
    receipts bound. Raises ValueError with the reason when the pair fails."""
    floor = screen_floor() if floor is None else floor
    if ar["examples_sha256"] != spec["examples_sha256"] or ar["sampling"] != spec["sampling"]:
        raise ValueError("GSM8K arms used different inputs")
    left, right = ({row["index"]: row for row in value["rows"]} for value in (ar, spec))
    if left.keys() != right.keys() or len(left) != 200:
        raise ValueError("GSM8K arms lost items")
    if any(left[index]["expected"] != right[index]["expected"] for index in left):
        raise ValueError("GSM8K answers differ")
    lost, gained = discordant(ar, spec)
    for value in (ar, spec):
        if not value["correct"] == sum(row["correct"] for row in value["rows"]) >= floor:
            raise ValueError(f"GSM8K screen below {floor}/200 or inconsistent score; lost={lost}, gained={gained}")
    result = {"passed": True, "paired": 200, "ar_correct": ar["correct"], "spec_correct": spec["correct"],
              "lost": lost, "gained": gained}
    if branches is None:
        if lost:
            raise ValueError(f"insufficient evidence: speculation lost GSM8K items {lost} and no branch probe ran")
        return result
    if (branches.get("near_tie_nats"), branches.get("top_logprobs")) != (rules.NEAR_TIE_NATS, TOP_LOGPROBS):
        raise ValueError("the branch probe used another near-tie rule")
    items = {item["index"]: dict(item, verdict=verdict(item)) for item in branches["items"]}
    if (branches["lost"], branches["gained"]) != (lost, gained) or sorted(items) != sorted(lost + gained):
        raise ValueError("the branch probe judged other GSM8K items than these receipts")
    for index, item in items.items():
        bound = {"ar": left[index].get("response_sha256"), "spec": right[index].get("response_sha256")}
        if item.get("measured") and (not all(bound.values()) or item.get("responses") != bound):
            raise ValueError(f"the branch probe of item {index} measured other responses than the receipts bound")
    unmeasured = [index for index in lost if items[index]["verdict"] == "unmeasured"]
    if unmeasured:
        raise ValueError(f"insufficient evidence: the branch probe did not measure lost GSM8K items {unmeasured}")
    decisive = [index for index in lost if items[index]["verdict"] == "decisive"]
    result.update(branches=[items[index] for index in sorted(items)], decisive_lost=decisive,
                  near_tie_lost=[index for index in lost if index not in decisive])
    if len(decisive) > DECISIVE_LOSSES:
        raise ValueError(f"speculation lost GSM8K items decisively beyond the C8 tolerance: {decisive}; "
                         f"near-tie losses: {result['near_tie_lost']}; gained: {gained}")
    return result


def main():
    parser = arguments(__doc__)
    parser.add_argument("--ar", type=Path, required=True, help="The AR arm's GSM8K receipt")
    parser.add_argument("--spec", type=Path, required=True, help="The speculative arm's GSM8K receipt")
    args = parser.parse_args()
    if args.mechanics_only:
        parser.error("the branch probe needs both arms' scored GSM8K receipts")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    result = probe(client, server["model"], args.ar, args.spec)
    try:
        verdict = pair(json.loads(args.ar.read_text()), json.loads(args.spec.read_text()), result)
        result["pair"] = {"passed": True, **{key: verdict[key] for key in ("decisive_lost", "near_tie_lost")}}
    except (ValueError, KeyError) as error:
        result["pair"] = {"passed": False, "reason": str(error)}
    # The probe passes when it measured; the pair's verdict belongs to the caller.
    receipt(CHECK, server, {"passed": True, "errors": [], "skipped": [], **result})


if __name__ == "__main__":
    main()
