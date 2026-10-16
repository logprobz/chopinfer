"""Near-tie analysis over runner scores, without torch.

A gate that meets a divergence asks how close the reference was to the token
actually produced, which need not be its runner-up: `margin` is only top-1
minus top-2. `gap` reads the reference's logit gap between any two tokens of
a scored row from its `logit` and `topk_logits` columns (score.py), in nats,
subtracted in f64 like `margin`. Differences of `topk_logprobs` are not a
substitute: log-softmax can round distinct logits to one logprob.
"""


def logits(scores, row):
    """{token: logit} known at a scored row: its top-k and its forced token."""
    known = dict(zip(scores["topk_ids"][row], scores["topk_logits"][row]))
    known[scores["token"][row]] = scores["logit"][row]
    return known


def gap(scores, row, token, alternative):
    """logit(token) - logit(alternative) at a scored row, or None when either
    token is neither in the row's top-k nor its forced token."""
    known = logits(scores, row)
    if token not in known or alternative not in known:
        return None
    return known[token] - known[alternative]
