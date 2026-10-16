"""Validate reference-runner scores before qualification.

Every token position has logprob, top-1, logit margin and aligned top-k
ID, logit and logprob columns. Short or inconsistent columns are refused.
"""

import math

RUNNER_COLUMNS = ("logprob", "top1", "margin", "topk_ids", "topk_logits", "topk_logprobs")
TOP = 5
# Values recomputed from the same f64 row agree to rounding of the JSON
# float text, far below any gated difference.
TOLERANCE = 1e-9


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def token_id(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def top(scores):
    """Per-position token IDs and logprobs, best first."""
    return [[list(pair) for pair in zip(ids, logprobs, strict=True)]
            for ids, logprobs in zip(scores["topk_ids"], scores["topk_logprobs"], strict=True)]


def gap_top(scores):
    """Per-position token IDs and logits, preserving the runner's gaps."""
    return [[list(pair) for pair in zip(ids, logits, strict=True)]
            for ids, logits in zip(scores["topk_ids"], scores["topk_logits"], strict=True)]


def check_scores(scores, tokens, where):
    """Require complete runner scores with consistent top-k IDs and values."""
    if not isinstance(scores, dict) or not tokens:
        raise ValueError(f"{where}: no scores")
    for column in RUNNER_COLUMNS:
        values = scores.get(column)
        if not isinstance(values, list) or len(values) != len(tokens):
            raise ValueError(f"{where}: {column} has {len(values) if isinstance(values, list) else 'no'} "
                             f"entries for {len(tokens)} positions")
    for index, (token, logprob, top1, margin, ids, logits, logprobs) in enumerate(zip(
            tokens, *(scores[column] for column in RUNNER_COLUMNS), strict=True)):
        at = f"{where} position {index}"
        if not token_id(token) or not finite(logprob) or logprob > TOLERANCE:
            raise ValueError(f"{at}: token or logprob is invalid")
        rows = (ids, logits, logprobs)
        if (any(not isinstance(row, list) or len(row) != TOP for row in rows)
                or not all(map(token_id, ids)) or not all(map(finite, logits + logprobs))):
            raise ValueError(f"{at}: the top-k rows are not {TOP} token IDs with finite logits and logprobs")
        if len(set(ids)) != TOP or any(a < b for values in (logits, logprobs) for a, b in zip(values, values[1:])):
            raise ValueError(f"{at}: the top-k rows repeat a token or are out of order")
        if top1 != ids[0] or not finite(margin):
            raise ValueError(f"{at}: top1 is not the first top-k token")
        if not math.isclose(margin, logits[0] - logits[1], abs_tol=TOLERANCE):
            raise ValueError(f"{at}: margin differs from the top-k logit gap")
        if logprob > logprobs[0] + TOLERANCE or (token in ids and not math.isclose(
                logprob, logprobs[ids.index(token)], abs_tol=TOLERANCE)):
            raise ValueError(f"{at}: logprob disagrees with the top-k logprobs")
