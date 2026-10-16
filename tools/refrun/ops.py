"""Exact eager replacements for fused or memory-bound upstream kernels.

`attention` is softmax attention computed per block of queries, so long
sequences (vision towers, long prompts) need only one block of scores at a
time. Each block is HF's eager math: scores = q k^T * scale (+ mask),
softmax, then probabilities x v. HF's eager path takes the softmax in F32 and
casts back to the query dtype; here the softmax runs in F32 or the input
dtype, whichever is wider, so f64 runs keep f64 probabilities.
"""

import torch

CHUNK = 1024


def attention(query, key, value, scale, mask=None, chunk=CHUNK):
    """[..., Lq, D] queries over [..., Lk, D] keys and values; `mask` is
    additive and broadcasts to [..., Lq, Lk]."""
    wide = torch.float64 if query.dtype == torch.float64 else torch.float32
    blocks = []
    for start in range(0, query.shape[-2], chunk):
        scores = torch.matmul(query[..., start:start + chunk, :], key.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + (mask if mask.shape[-2] == 1 else mask[..., start:start + chunk, :])
        probabilities = torch.softmax(scores, dim=-1, dtype=wide).to(query.dtype)
        blocks.append(torch.matmul(probabilities, value))
    return torch.cat(blocks, dim=-2)


def repeat_kv(states, groups):
    """[B, Hkv, L, D] to [B, Hkv * groups, L, D], each KV head repeated in place."""
    if groups == 1:
        return states
    batch, heads, length, dim = states.shape
    return states[:, :, None].expand(batch, heads, groups, length, dim).reshape(batch, heads * groups, length, dim)


def hf_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **_kwargs):
    """An HF attention-interface function: [B, H, L, D] in, [B, L, H, D] out."""
    if dropout:
        raise ValueError("reference runs have no dropout")
    groups = getattr(module, "num_key_value_groups", 1)
    out = attention(query, repeat_kv(key, groups), repeat_kv(value, groups), scaling, attention_mask, CHUNK)
    return out.transpose(1, 2).contiguous(), None


def register(name="refrun_chunked"):
    """Registers `hf_attention` with transformers under `name`, with HF's
    eager mask builder: HF builds no mask for an attention implementation
    the mask registry does not know, and `hf_attention` adds the mask it is
    given (causal or sliding-window additive mask, or None for
    bidirectional calls)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(name, hf_attention)
    AttentionMaskInterface.register(name, eager_mask)
    return name
