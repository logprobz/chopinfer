"""Group already-exclusive target charges; retain unfamiliar kernels explicitly."""
import collections
import math


def family(name):
    if "_before_gateup_" in name:
        return "MoE gate/up"
    if "_before_down_" in name:
        return "MoE down"
    if "moe_ceil_" in name or "moe_" in name and "_sum_add_" in name:
        return "MoE rounding and combine"
    if "chop_moe_" in name or "_n384_k5120_" in name:
        return "MoE routing"
    if any(s in name for s in ("shared_swiglu", "_n576_k5120_", "_n5120_k576_")):
        return "Shared expert"
    if name.startswith("chop_comm_"):
        return "Communication"
    if "sinkhorn_hc_" in name:
        return "HC"
    if "_head_f32_" in name:
        return "LM head"
    if "latent_sources_" in name:
        return "Latent attention"
    if any(s in name for s in ("position_", "candidate_", "_index_key_")):
        return "Indexer"
    if "_mxfp8_" in name and "_n" in name or "_dense_" in name:
        return "Dense projections and rounding"
    if any(s in name for s in ("latent_", "compress", "_rope_")):
        return "KV compression and RoPE"
    return "Other row, norm and Engram"


def summarize(kernels):
    result = collections.defaultdict(float)
    for name, value in kernels.items():
        result[family(name)] += value
    assert math.isclose(sum(kernels.values()), sum(result.values()), abs_tol=1e-9)
    return dict(sorted(result.items()))
