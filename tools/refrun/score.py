"""Scores of one sequence: teacher forcing and greedy continuation.

Position p >= score_from is scored by the logits at p - 1:

- `logit` / `logprob`: the forced token's logit and log-softmax value;
- `top1`: the highest logit, ties to the lower token ID;
- `margin`: top-1 minus top-2 logit, subtracted in f64;
- `topk_ids` / `topk_logits` / `topk_logprobs`: the top 20 in that order.

`margins.gap` gives the logit gap between any two of a row's tokens, for
near-tie decisions about a token other than the runner-up.

Ranks come from a stable descending sort of the logits, so equal logits go
to the lower token ID and top1 is the greedy argmax. Ranking the rounded
logprobs instead would tie distinct logits that log-softmax rounds together.
Logits are emitted as their exact f64 widening. Log-softmax runs in the
policy's `log_softmax` dtype (f64 for f64) and is emitted as f64.

Scores hold only what the key determines: a sequence's name and category
belong to the run entry, so equal keys give equal score bytes.
"""

import torch

from tools.refrun import adapter, modes

TOPK = 20
ROWS = 256


def rows(logits, tokens, dtype, topk=TOPK):
    """Columns of per-row scores: logits[i] scored against tokens[i]."""
    columns = {"token": [], "logit": [], "logprob": [], "top1": [], "margin": [], "topk_ids": [],
               "topk_logits": [], "topk_logprobs": []}
    if torch.promote_types(logits.dtype, dtype) != dtype:
        raise ValueError(f"{logits.dtype} logits are wider than the {dtype} log-softmax")
    tokens = torch.as_tensor(tokens, device=logits.device)
    for start in range(0, len(tokens), ROWS):
        # Widening to `dtype` is exact, so the logits keep their order.
        block = logits[start:start + ROWS].to(dtype)
        logprobs = torch.log_softmax(block, dim=-1)
        if not torch.isfinite(logprobs).all():
            raise ValueError("non-finite logprobs")
        chosen = tokens[start:start + ROWS]
        forced = logprobs.gather(1, chosen[:, None])[:, 0]
        top, ids = torch.sort(block, dim=-1, descending=True, stable=True)
        top, ids = top[:, :topk].double().cpu(), ids[:, :topk]
        columns["token"] += chosen.tolist()
        columns["logit"] += block.gather(1, chosen[:, None])[:, 0].double().tolist()
        columns["logprob"] += forced.double().tolist()
        columns["top1"] += ids[:, 0].tolist()
        columns["margin"] += (top[:, 0] - top[:, 1]).tolist()
        columns["topk_ids"] += ids.tolist()
        columns["topk_logits"] += top.tolist()
        columns["topk_logprobs"] += logprobs.gather(1, ids).double().tolist()
    return columns


def media_inputs(reference, sequence, device, dtype):
    if not sequence.media:
        return {}
    if reference.media_inputs is None:
        raise ValueError(f"{sequence.name} has media but the adapter takes none")
    return reference.media_inputs(sequence, device, dtype)


def teacher_forced(reference, sequence, policy, device):
    """Scores of the forced positions of one sequence, in one forward pass."""
    dtype = modes.torch_dtype(policy["dtype"])
    ids = sequence.forced_ids
    keep = len(ids) - sequence.score_from + 1
    forward = reference.forward or adapter.hf_forward
    with modes.active(policy):
        inputs = torch.tensor([ids], device=device)
        logits = forward(reference.model, inputs, keep, media_inputs(reference, sequence, device, dtype))
        if logits.shape[0] != keep:
            raise ValueError(f"forward returned {logits.shape[0]} logit rows, not {keep}")
        scored = rows(logits[:-1], ids[sequence.score_from:], modes.torch_dtype(policy["log_softmax"]))
    return {"score_from": sequence.score_from, **scored}


def greedy(reference, sequence, policy, device):
    """The greedy answer to a sequence's prompt and each step's scores."""
    dtype = modes.torch_dtype(policy["dtype"])
    decode = reference.decode or adapter.hf_decode
    with modes.active(policy):
        inputs = torch.tensor([sequence.token_ids], device=device)
        answer, logits = decode(reference.model, inputs, media_inputs(reference, sequence, device, dtype),
                                sequence.answer.max_new_tokens, set(sequence.answer.stop_ids))
        steps = rows(logits, answer, modes.torch_dtype(policy["log_softmax"]))
    return answer, {"answer": answer, **steps}
