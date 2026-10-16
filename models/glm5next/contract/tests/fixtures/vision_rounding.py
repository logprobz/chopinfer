"""Regenerates vision-rounding.json: the vision tower's per-head q and k
RMSNorm and rotary as the unmodified transformers glm5_next modules compute
them in float32 on the CPU (synthetic inputs and weights, no checkpoint).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python vision_rounding.py vision-rounding.json

Run it with the reference venv (transformers 5.16.1, torch 2.14.0). Values
are stored as their float32 bits.
"""
import json
import sys

import torch
import transformers.models.glm5_next.modeling_glm5_next as M
from transformers.vision_utils import get_vision_position_ids

torch.manual_seed(7)
heads, head_dim, eps, h, w = 2, 64, 1e-5, 2, 4
rows = h * w
q = torch.randn(rows, heads, head_dim) * 3
k = torch.randn(rows, heads, head_dim) * 3
q_norm, k_norm = M.Glm5NextRMSNorm(head_dim, eps), M.Glm5NextRMSNorm(head_dim, eps)
with torch.no_grad():
    q_norm.weight.copy_(1 + 0.3 * torch.randn(head_dim))
    k_norm.weight.copy_(1 + 0.3 * torch.randn(head_dim))
    rotary = M.Glm5NextVisionRotaryEmbedding(head_dim // 2)
    angles = rotary(get_vision_position_ids(torch.tensor([[1, h, w]]), 2))
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos(), angles.sin()
    q_out, k_out = M.apply_rotary_pos_emb_vision(q_norm(q), k_norm(k), cos, sin)


def bits(tensor):
    return tensor.contiguous().view(torch.int32).flatten().tolist()


json.dump(
    {
        "heads": heads, "head_dim": head_dim, "eps": eps, "grid": [1, h, w],
        "q": bits(q), "k": bits(k), "q_weight": bits(q_norm.weight.data),
        "k_weight": bits(k_norm.weight.data), "cos": bits(cos), "sin": bits(sin),
        "q_out": bits(q_out), "k_out": bits(k_out),
    },
    open(sys.argv[1], "w"),
)
