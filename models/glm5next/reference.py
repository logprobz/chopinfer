"""GLM 5.3 Flash reference adapter: HF `glm5_next` (transformers 5.16.1) on the GPUs.

The runner (tools/refrun) builds `Glm5NextForConditionalGeneration` once per
mode on the meta device in the mode's dtype, as `from_pretrained(dtype=D)`
would, and fills it from the shared store. Every stored weight stays
resident as the checkpoint stores it and is dequantized per call
(tools/refrun/quant.py), so the four GPUs hold the checkpoint's own bytes
(about 80 GB each without the MTP layer) in every mode:

- Every `nn.Linear` of the text model and the vision tower is a stored
  `QuantLinear`: FP8 E4M3 with F32 inverse scales per 128 x 128 block (the
  MLA projections, dense, shared and routed MLPs), NVFP4 (packed E2M1, E4M3
  scale per 16, F32 `weight_scale_2`; `input_scale` is not read, the
  reference is A16) and MXFP8 (E4M3, E8M0 scale per 32) by the stored
  tensors' names and dtypes, and BF16 for the rest. The one exception is
  the indexer's `weights_proj`, whose weight HF reads directly; it is cast.
  The token embedding gathers BF16 rows and widens them.
- Routed experts are `QuantStack`s whose items dequantize one expert per
  call (gate and up stacked as HF's fused `gate_up_proj`). They run
  `grouped_experts`, a copy of HF's `grouped_mm` experts forward with each
  expert's GEMM as one `torch.mm` over its rows, as HF's grouped fallback
  computes it: the eight weighted expert outputs are summed in F32 and
  rounded once, never per expert as HF's eager loop does.
- The checkpoint stores the KDA conv as `q_conv1d`, `k_conv1d` and
  `v_conv1d`; HF's loader concatenates them (q, k, v) into `conv1d`, and so
  does the adapter. The forget gate's `f_a_proj`, `f_b_proj`, `dt_bias` and
  `A_log` live under `forget_gate`, and `hc_{attn,ffn}_{fn,base,scale}`
  under `{attn,ffn}_hc` (transformers' conversion mapping).
- In bf16, `KEEP_F32` tensors (HF's `_keep_in_fp32_modules_strict`) and
  every tensor the checkpoint stores in F32 (mHC base and scale) stay F32;
  everything else is cast to the mode dtype, as `from_pretrained` would.
- HF's torch KDA (chunk 64, `chunk_kimi_delta_attention`), conv and indexer
  run; the adapter refuses to build when a kernel package (`fla`,
  `causal_conv1d`, `kernels`, `flash_attn`) could replace them.
- The cache keeps each sparse layer's normalized KV latent instead of HF's
  expanded K and V (`cached_latent` stands in for `expand_kv`), and sparse
  attention (`sparse_attention`) is HF's eager math in blocks of query rows,
  each expanding only the keys its rows select with HF's own `expand_kv`.
  HF's mask gives every other key exactly zero weight, so this changes the
  order of the sums, not the math. The vision tower's attention runs in
  query chunks round robin over the GPUs, as Qwen's adapter does.
- The indexer chooses its pools by a stable descending sort, ties to the
  lower pool index (`indexer_forward`, otherwise HF's code). This is the
  one deliberate departure from HF's code path: HF's `torch.topk` orders
  equal scores by kernel, and relu makes exact ties (often at zero) common,
  so HF's selection changes with the prefill chunking. Lower-index ties are
  the engine's rule.
- A forward pass embeds the prompt, merges the image features, then runs
  the text model in chunks (`chunk_rows`, at most `PREFILL_ROWS`, fewer at
  long context, since the indexer holds F32 [rows, 32, pools] scores) with
  a `DynamicCache`; each chunk is marked for the tap writer. Greedy decoding
  continues from that cache (HF's recurrent KDA step).
- The caches hold 512 latent and 257 indexer values per token in every
  sparse layer, so a sequence whose caches would pass `KV_BUDGET_BYTES` on
  some GPU is refused before it runs (`check_context`).

f64 is the unmodified upstream code in f64: HF computes some steps in F32
whatever the model dtype (`UPSTREAM_F32`), and they stay F32. The MTP layer
(layer 45) is not loaded; HF does not run it. Pixel rows come from the
workload (the processor's output); the processor never runs here, and
video is refused.
"""

import hashlib
import importlib.util
from pathlib import Path
import re

import torch
from torch import nn
import torch.nn.functional as F

from tools.refrun import adapter, ops, quant, taps as tap_writer
from tools.refrun.place import Unit
from tools.refrun.taps import Hook

MODEL = "glm5next"
TEXT = "model.language_model"
FP8 = quant.fp8_block(128, 128)
MXFP8 = quant.Format("e4m3", "e8m0", 1, 32)
# Prefill rows per chunk, a multiple of ROW_ALIGN (the KDA chunk) so the
# chunks keep HF's 64-row KDA grid. HF's KDA chunk path holds about three
# F32 [64 heads, rows, 64, 128] tensors (2 MiB per row each).
PREFILL_ROWS = 512
ROW_ALIGN = 64
# One F32 [rows, index heads, pools] indexer score tensor; HF holds about
# three at once.
INDEXER_SCORE_BYTES = 2 << 30
# Sparse attention runs in blocks of at most QUERY_ROWS query rows, each
# expanding the keys its rows select: one block's [heads, rows, keys]
# scores, and its kv_b_proj output and expanded keys.
QUERY_ROWS = 64
ATTENTION_SCORE_BYTES = 1 << 30
EXPAND_BYTES = 4 << 30
# The latent and indexer caches on one GPU. The fullest GPU holds 78.2 GiB
# of weights and three sparse layers; growing one layer's cache copies it once.
KV_BUDGET_BYTES = 4 << 30
VISION_CHUNK = 1024
SPARSE_ATTENTION = "refrun_glm5next_sparse"
VISION_ATTENTION = "refrun_glm5next_vision"
EXPERTS = "refrun_glm5next_grouped"
KERNEL_PACKAGES = ("fla", "causal_conv1d", "kernels", "flash_attn")
KEEP_F32 = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")
# Checkpoint names to HF module names (transformers' conversion mapping).
RENAMES = (("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
           ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
           ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
           ("self_attn.A_log", "self_attn.forget_gate.A_log"),
           ("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"), ("hc_attn_scale", "attn_hc.scale"),
           ("hc_ffn_fn", "ffn_hc.fn"), ("hc_ffn_base", "ffn_hc.base"), ("hc_ffn_scale", "ffn_hc.scale"))
CONV = ("q_conv1d", "k_conv1d", "v_conv1d")
# SHA256 of the source of the Glm5NextTextIndexer.forward that indexer_forward copies.
HF_INDEXER_FORWARD = "9ae93cd369a383f335ea2f4f0fbc39088d60a6c239a6bf150494784535067a7d"
UPSTREAM_F32 = {
    "Glm5NextTextRMSNorm / Glm5NextRMSNorm": "normalize x.float(), round to the input dtype, then scale",
    "Glm5NextTextHyperConnection": "mixing weights from x.float() and fn.float(); Sinkhorn in F32",
    "Glm5NextTextTopkRouter": "router logits, sigmoid scores and top-k weights in F32",
    "Glm5NextTextForgetGate": "g from forget_gate.float(), dt_bias.float() and A_log.float()",
    "chunk_kimi_delta_attention / recurrent_kimi_delta_attention": "run in F32",
    "Glm5NextTextRMSNormGated": "normalizes and gates in F32",
    "Glm5NextTextIndexer": "pool softmax, scores and head weights in F32",
    "eager_attention_forward (sparse_attention)": "softmax in F32",
    "Glm5NextVisionRotaryEmbedding / apply_rotary_pos_emb_vision": "F32 inv_freq; rotates q and k in F32",
}


def hf_modules():
    from transformers.models.glm5_next import configuration_glm5_next, modeling_glm5_next
    return configuration_glm5_next, modeling_glm5_next


def reference_code(_checkpoint):
    """The glm5_next configuration and modeling files, and the experts
    dispatch whose grouped_mm forward `grouped_experts` copies."""
    from transformers.integrations import moe
    configuration, modeling = hf_modules()
    files = {f"transformers/models/glm5_next/{Path(m.__file__).name}": Path(m.__file__)
             for m in (configuration, modeling)}
    files["transformers/integrations/moe.py"] = Path(moe.__file__)
    return files


def hf_name(name):
    """The HF module path of a checkpoint tensor (before the conv concatenation)."""
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def text_layers(config):
    return (config.get("text_config") or config)["num_hidden_layers"]


def read(name):
    """Whether the runner loads a stored tensor: not the activation scales
    of an NVFP4 export (the reference is A16)."""
    return not name.endswith(".input_scale")


def units(checkpoint):
    """Vision tower, embeddings, each decoder layer, final norm, head. The
    MTP layer (index num_hidden_layers) is not run and not loaded."""
    names = sorted(n for n in checkpoint.entries if read(n))
    layers = text_layers(checkpoint.config)

    def owned(prefix):
        return tuple(n for n in names if n.startswith(prefix))
    return [Unit("model.visual", owned("model.visual.")),
            Unit(f"{TEXT}.embed_tokens", owned(f"{TEXT}.embed_tokens.")),
            *(Unit(f"{TEXT}.layers.{layer}", owned(f"{TEXT}.layers.{layer}.")) for layer in range(layers)),
            Unit(f"{TEXT}.norm", owned(f"{TEXT}.norm.")),
            Unit("lm_head", owned("lm_head."))]


def stored_format(store, weight):
    """(format, [rows, cols], stored scale names) of a stored weight, from
    the dtypes and names of its tensors."""
    entries = store.checkpoint.entries
    base, entry = weight[:-len("weight")], entries[weight]
    rows, cols = entry.shape[0], (entry.shape[1] if len(entry.shape) > 1 else 1)
    if entry.dtype == "BF16":
        return quant.BF16, (rows, cols), ()
    if entry.dtype == "F8_E4M3" and base + "weight_scale_inv" in entries:
        return FP8, (rows, cols), (base + "weight_scale_inv",)
    scale = entries.get(base + "weight_scale")
    if entry.dtype == "U8" and scale is not None and scale.dtype == "F8_E4M3" and base + "weight_scale_2" in entries:
        global_scale = store.get(base + "weight_scale_2")
        if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
            raise ValueError(f"{base}weight_scale_2 is not one F32 global scale")
        return quant.nvfp4(global_scale.item()), (rows, 2 * cols), (base + "weight_scale", base + "weight_scale_2")
    if entry.dtype == "F8_E4M3" and scale is not None and scale.dtype in ("U8", "F8_E8M0"):
        return MXFP8, (rows, cols), (base + "weight_scale",)
    raise ValueError(f"{weight}: no supported format for {entry.dtype} codes with these scales")


def stored_tensor(store, weight, slot):
    fmt, shape, scales = stored_format(store, weight)
    return store.tensor(fmt, weight, scales[0] if scales else None, shape, slot), scales


def stored_linear(store, weight, bias=None):
    fmt, shape, scales = stored_format(store, weight)
    return store.linear(fmt, weight, scales[0] if scales else None, shape, bias=bias), scales


class GateUp(nn.Module):
    """One expert's gate and up projections as HF's fused [2I, H] weight,
    dequantized into one scratch buffer."""

    def __init__(self, gate, up):
        super().__init__()
        self.gate, self.up = gate, up

    def value(self):
        rows, cols = self.gate.shape
        out = quant.scratch(self.gate.codes.device, self.gate.dtype, "gate_up", 2 * rows * cols).view(2 * rows, cols)
        self.gate.dequantize(self.gate.dtype, self.gate.rule, out[:rows])
        self.up.dequantize(self.up.dtype, self.up.rule, out[rows:])
        return out


class StoredEmbedding(nn.Module):
    """Stands in for `embed_tokens`: gathers the stored BF16 rows and widens
    them to the mode dtype (exact)."""

    def __init__(self, table, dtype):
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.dtype = dtype

    def forward(self, ids):
        return F.embedding(ids, self.table).to(self.dtype)


def grouped_experts(self, hidden_states, top_k_index, top_k_weights):
    """transformers.integrations.moe.grouped_mm_experts_forward on stored
    experts: tokens sorted by expert (stable), each expert's rows through
    its dequantized weights with one torch.mm (HF's _grouped_mm_fallback),
    the routing weight applied to each expert output, then the top-k
    outputs summed in their (F32) product dtype and cast once."""
    num_top_k, num_tokens, hidden_dim = top_k_index.size(-1), hidden_states.size(0), hidden_states.size(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids, stable=True)
    selected = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]
    counts = torch.bincount(expert_ids_g.cpu(), minlength=self.num_experts).tolist()

    def grouped(rows, stack):
        out, start = None, 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            weight = stack[expert]
            if out is None:
                out = torch.zeros(rows.size(0), weight.size(0), device=rows.device, dtype=weight.dtype)
            torch.mm(rows[start:start + count].to(weight.dtype), weight.transpose(0, 1),
                     out=out[start:start + count])
            start += count
        return out
    projected = self._apply_gate(grouped(selected, self.gate_up_proj))
    projected = grouped(projected, self.down_proj)
    weighted = projected * sample_weights_g.unsqueeze(-1)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=perm.device)
    weighted = weighted[inverse]
    return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def cached_latent(kv_nope, k_rot):
    """Stands in for `Glm5NextTextAttention.expand_kv` in the forward pass:
    the cache keeps the normalized latent [B, 1, L, kv_lora_rank] and the
    rope part [B, 1, L, qk_rope_head_dim] (empty), and `sparse_attention`
    expands the keys it reads with HF's own `expand_kv`."""
    return kv_nope, k_rot


def lower_pools(index_scores, select_k):
    """The `select_k` best pools of each query, best first, ties to the lower
    pool index: a stable descending sort. torch.topk leaves the order of
    equal scores to the kernel, and the indexer's relu makes exact ties
    (often at zero) common, so HF's choice among them changes with the score
    width and so with the prefill chunking."""
    return torch.sort(index_scores, dim=-1, descending=True, stable=True).indices[..., :select_k]


@torch.no_grad()
def indexer_forward(self, hidden_states, q_resid, attention_mask, past_key_values):
    """Glm5NextTextIndexer.forward (transformers 5.16.1) with one change: the
    pools are chosen by `lower_pools` instead of `index_scores.topk`. Scores,
    the causal and padding filters, pool expansion, the incomplete tail and
    the -1 padding are HF's code, unchanged."""
    batch_size, seq_len = hidden_states.shape[:2]
    hidden_shape = (batch_size, seq_len, -1, self.head_dim)

    q = self.wq_b(q_resid).view(hidden_shape)
    k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

    gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
    valid_channel = attention_mask.to(k.dtype)[..., None]

    packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

    kv_len = seq_len
    current_length = seq_len
    if past_key_values is not None:
        cache_layer = past_key_values.layers[self.layer_idx]
        packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
        kv_len = cache_layer.keys.shape[-2]
        current_length = cache_layer.get_seq_length()

    valid_keys = packed_states[..., -1].bool()
    visible_tokens = self.get_visible_tokens(valid_keys=valid_keys, q_length=seq_len, current_length=current_length)

    pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * self.softmax_scale)

    weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1))
    valid_candidates = pool_visible & pool_valid[:, None]

    index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)

    select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
    selected = lower_pools(index_scores, select_k)
    batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]

    topk_indices = selected_indices.flatten(-2)
    topk_indices = topk_indices.masked_fill(~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)

    output_width = self.index_topk
    if self.index_kpool_always_select_tail:
        topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
        output_width += self.index_kpool - 1

    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
    topk_indices = topk_indices[..., :output_width]
    topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
    return topk_indices.to(torch.int32)


def pin_ties(indexer):
    """Installs `indexer_forward` on one HF indexer module, refusing an HF
    indexer whose forward is not the one it copies."""
    import inspect
    source = hashlib.sha256(inspect.getsource(type(indexer).forward).encode()).hexdigest()
    if source != HF_INDEXER_FORWARD:
        raise RuntimeError(f"HF's indexer forward is {source}, not the {HF_INDEXER_FORWARD} indexer_forward copies")
    indexer.forward = indexer_forward.__get__(indexer)
    return indexer


def query_blocks(mask, heads, per_key, width):
    """[start, stop) blocks of query rows and each block's key columns (the
    keys any of its rows selects): at most QUERY_ROWS rows, halved until the
    block's expanded K and V fit EXPAND_BYTES and its scores
    ATTENTION_SCORE_BYTES."""
    visible = mask[0, 0] == 0
    if not bool(visible.any(dim=1).all()):
        raise ValueError("a query selects no key; HF would attend to every key")
    blocks, start = [], 0
    while start < visible.shape[0]:
        rows = QUERY_ROWS
        while True:
            columns = visible[start:start + rows].any(dim=0).nonzero().flatten()
            size = columns.numel()
            if rows == 1 or (size * per_key <= EXPAND_BYTES and heads * rows * size * width <= ATTENTION_SCORE_BYTES):
                break
            rows //= 2
        stop = min(visible.shape[0], start + rows)
        blocks.append((start, stop, columns))
        start = stop
    return blocks


def sparse_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **_kwargs):
    """HF's eager_attention_forward over the latent cache, in blocks of query
    rows. `key` and `value` are the cached latent and rope part
    (`cached_latent`); each block expands only the keys its rows select with
    HF's `expand_kv` (the same kv_b_proj on the same latent rows), then takes
    the scores, the additive top-k mask, softmax in F32 cast to the query
    dtype, and the values. Keys no row of the block selects get exactly zero
    weight in HF's softmax, so leaving them out changes only the order of
    the sums. [B, H, L, D] in, [B, L, H, D] out."""
    from transformers.models.glm5_next.modeling_glm5_next import repeat_kv
    if dropout:
        raise ValueError("reference runs have no dropout")
    if module.__dict__.get("expand_kv") is not cached_latent:
        raise ValueError("sparse attention reads the latent cache; expand_kv must be cached_latent")
    if query.shape[0] != 1:
        raise ValueError("reference runs one sequence per forward")
    expand = type(module).expand_kv
    heads, width = query.shape[1], max(query.element_size(), 4)
    # kv_b_proj's output and the expanded keys, per selected key.
    per_key = heads * (module.qk_nope_head_dim + module.v_head_dim + module.qk_head_dim) * query.element_size()
    blocks = []
    for start, stop, columns in query_blocks(attention_mask, heads, per_key, width):
        keys, values = expand(module, key[:, :, columns], value[:, :, columns])
        keys = repeat_kv(keys, module.num_key_value_groups)
        values = repeat_kv(values, module.num_key_value_groups)
        weights = torch.matmul(query[:, :, start:stop], keys.transpose(2, 3)) * scaling
        weights = weights + attention_mask[:, :, start:stop][..., columns]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        blocks.append(torch.matmul(weights, values))
    return torch.cat(blocks, dim=2).transpose(1, 2).contiguous(), None


def vision_attention(devices):
    """HF's attention interface for one image's vision attention, query
    chunks spread round robin over `devices`. Each chunk attends to all of
    the image's keys: HF's SDPA in f32 and bf16, ops.attention (f64
    softmax) in f64; [B, H, L, D] in, [B, L, H, D] out."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def attend(module, query, key, value, attention_mask, scaling=None, dropout=0.0, is_causal=False, **kwargs):
        if attention_mask is not None or dropout or is_causal:
            raise ValueError("vision attention is bidirectional, without a mask or dropout")
        scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
        home, copies, parts = query.device, {}, []
        for index, start in enumerate(range(0, query.shape[2], VISION_CHUNK)):
            device = torch.device(devices[index % len(devices)]) if query.shape[2] > VISION_CHUNK else home
            if device not in copies:
                copies[device] = tuple(t.to(device) for t in (query, key, value))
            q, k, v = copies[device]
            block = q[:, :, start:start + VISION_CHUNK]
            if query.dtype == torch.float64:
                parts.append(ops.attention(block, k, v, scaling, chunk=VISION_CHUNK).transpose(1, 2))
            else:
                parts.append(sdpa_attention_forward(module, block, k, v, None, scaling=scaling, dropout=0.0,
                                                    is_causal=False, **kwargs)[0])
        return torch.cat([part.to(home) for part in parts], dim=1), None
    return attend


def check_kernels():
    for package in KERNEL_PACKAGES:
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"{package} is installed and could replace HF's torch KDA, conv, norms or attention")


def implementations(text, vision):
    """The pinned sparse attention and experts on every text config object
    the modules read, the chunked vision attention on the vision ones."""
    for config in text:
        config._attn_implementation = SPARSE_ATTENTION
        config._experts_implementation = EXPERTS
    for config in vision:
        config._attn_implementation = VISION_ATTENTION


def register(devices):
    from transformers import AttentionInterface
    from transformers.integrations.moe import ExpertsInterface
    AttentionInterface.register(SPARSE_ATTENTION, sparse_attention)
    AttentionInterface.register(VISION_ATTENTION, vision_attention(devices))
    ExpertsInterface.register(EXPERTS, grouped_experts)


def parameter_dtype(name, stored, dtype):
    """The dtype `from_pretrained(dtype=D)` gives a floating tensor: F32 in
    bf16 for HF's strict F32 modules and the tensors stored in F32."""
    if dtype == torch.bfloat16 and (any(key in name for key in KEEP_F32) or stored == torch.float32):
        return torch.float32
    return dtype


def build(config, placement):
    """The HF model in `config.dtype` around the store's weights."""
    configuration, hf = hf_modules()
    check_kernels()
    register(placement.devices)
    store, dtype = config.store, config.dtype
    hf_config = configuration.Glm5NextConfig.from_dict(dict(config.checkpoint.config))
    text_config = hf_config.text_config
    implementations([text_config], [hf_config.vision_config])
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = hf.Glm5NextForConditionalGeneration(hf_config)
    finally:
        torch.set_default_dtype(default)
    implementations([m.config for m in model.modules()
                     if isinstance(getattr(m, "config", None), configuration.Glm5NextTextConfig)],
                    [m.config for m in model.modules()
                     if isinstance(getattr(m, "config", None), configuration.Glm5NextVisionConfig)])
    devices = dict(placement.units)

    def device_of(path):
        while path not in devices:
            path = path.rpartition(".")[0]
        return torch.device(devices[path])
    stored = store.tensors
    names = {}

    def mapped(checkpoint_names, path):
        for name in checkpoint_names:
            names[name] = path
    # Every linear but indexer.weights_proj is a stored QuantLinear.
    linears = [(path, module) for path, module in model.named_modules()
               if isinstance(module, nn.Linear) and not path.endswith("indexer.weights_proj")]
    by_hf = {hf_name(name): name for name in stored}
    for path, module in linears:
        weight = by_hf.get(path + ".weight")
        if weight is None:
            raise ValueError(f"{path} has no stored weight")
        bias = by_hf.get(path + ".bias") if module.bias is not None else None
        linear, scales = stored_linear(store, weight, bias)
        parent, _, child = path.rpartition(".")
        setattr(model.get_submodule(parent), child, linear)
        mapped([weight, *scales, *([bias] if bias else [])], path)
    text = model.model.language_model
    text.embed_tokens = StoredEmbedding(store.get(f"{TEXT}.embed_tokens.weight"), dtype)
    mapped([f"{TEXT}.embed_tokens.weight"], f"{TEXT}.embed_tokens")
    for layer, block in enumerate(text.layers):
        prefix = f"{TEXT}.layers.{layer}."
        if text_config.mlp_layer_types[layer] != "sparse":
            continue
        experts, stacks = block.mlp.experts, {"gate_up": [], "down": []}
        for expert in range(experts.num_experts):
            base = f"{prefix}mlp.experts.{expert}."
            gate, gate_scales = stored_tensor(store, base + "gate_proj.weight", "gate")
            up, up_scales = stored_tensor(store, base + "up_proj.weight", "up")
            down, down_scales = stored_tensor(store, base + "down_proj.weight", "down")
            stacks["gate_up"].append(GateUp(gate, up))
            stacks["down"].append(down)
            path = f"{prefix}mlp.experts"
            mapped([base + "gate_proj.weight", *gate_scales], f"{path}.gate_up_proj.items.{expert}.gate")
            mapped([base + "up_proj.weight", *up_scales], f"{path}.gate_up_proj.items.{expert}.up")
            mapped([base + "down_proj.weight", *down_scales], f"{path}.down_proj.items.{expert}")
        del experts._parameters["gate_up_proj"], experts._parameters["down_proj"]
        experts.gate_up_proj = quant.QuantStack(stacks["gate_up"])
        experts.down_proj = quant.QuantStack(stacks["down"])
    state = {}
    for name, tensor in stored.items():
        if name in names:
            continue
        path = hf_name(name)
        if any(path.endswith(f"self_attn.{part}.weight") for part in CONV):
            # HF's loader concatenates q, k and v into one conv (conversion mapping).
            if not path.endswith("self_attn.q_conv1d.weight"):
                continue
            parts = [path.replace("q_conv1d", part) for part in CONV]
            path = path.replace("q_conv1d", "conv1d")
            tensor = torch.cat([stored[by_hf[part]] for part in parts])
            mapped([by_hf[part] for part in parts], path)
        else:
            names[name] = path
        state[path] = tensor.to(parameter_dtype(path, tensor.dtype, dtype)) if tensor.is_floating_point() else tensor
    _, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"stored tensors with no place in the HF model: {unexpected[:5]}")
    # Computed, not stored: the vision rotary table, in F32 as HF builds it.
    vision = hf_config.vision_config
    model.model.visual.rotary_pos_emb = hf.Glm5NextVisionRotaryEmbedding(
        vision.hidden_size // vision.num_heads // 2).to(device_of("model.visual"))
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{len(left)} tensors not loaded, e.g. {left[:5]}")
    model.eval()
    for block in text.layers:
        if block.block_type != "linear_attention":
            block.self_attn.expand_kv = cached_latent
            if block.self_attn.indexer is not None:
                pin_ties(block.self_attn.indexer)
    sparse = [device_of(f"{TEXT}.layers.{layer}") for layer, kind in enumerate(text_config.layer_types)
              if kind != "linear_attention"]
    plan = Plan(text_config, sparse)
    return adapter.Reference(
        model, names, layout=layout(text_config, config.policy["dequant"]), media_inputs=media_inputs,
        taps=taps, forward=plan.forward, decode=plan.decode,
        replacements={
            "nn.Linear (all but indexer.weights_proj)": "stored QuantLinear (FP8 128x128, NVFP4, MXFP8 or BF16), "
                                                        "dequantized per call",
            "embed_tokens": "StoredEmbedding: BF16 rows gathered, then widened",
            "mlp.experts": f"{EXPERTS}: HF's grouped_mm experts forward over QuantStacks, one expert's weights "
                           "dequantized per call, one torch.mm per expert",
            "expand_kv (sparse layers)": "cached_latent: the cache holds the normalized latent, not expanded K and V",
            "indexer.forward": "indexer_forward: HF's code with the pools chosen by a stable descending sort (ties "
                               "to the lower pool index) instead of torch.topk; the one departure from HF's "
                               "selection",
            "sparse attention": f"{SPARSE_ATTENTION}: HF's eager attention in query blocks of at most {QUERY_ROWS} "
                                "rows, each over the keys its rows select, expanded by HF's expand_kv",
            "vision attention": f"query chunks of {VISION_CHUNK} over all keys of an image, round robin over "
                                "the GPUs: HF sdpa_attention_forward in f32 and bf16, ops.attention in f64",
            "Glm5NextModel.forward": "the same steps, text model in chunks (chunk_rows) with a DynamicCache"})


def layout(text_config, rule):
    return {"fp8": "E4M3 with F32 inverse scales per 128x128 block (MLA q_a, q_b, kv_a and o, dense, shared and "
                   "routed MLPs); the f32 gate needs dequant rounded",
            "nvfp4": "packed E2M1, E4M3 scale per 16, F32 weight_scale_2; input_scale not read",
            "mxfp8": "E4M3 with an E8M0 scale per 32",
            "experts": "gate and up per expert stacked as gate_up_proj",
            "kda_conv": "q_conv1d, k_conv1d and v_conv1d concatenated in that order as conv1d",
            "renames": [list(pair) for pair in RENAMES],
            "bf16_f32": "KEEP_F32 tensors and F32-stored tensors stay F32 in bf16",
            "keep_f32": list(KEEP_F32), "mtp": "layer 45 not loaded or run",
            "dequant": rule, "prefill_rows": PREFILL_ROWS, "row_align": ROW_ALIGN,
            "indexer_ties": "lower pool index (stable sort), not torch.topk's kernel order",
            "kv_cache": "normalized latent and rope part per sparse layer; K and V expanded per query block",
            "indexer_score_bytes": INDEXER_SCORE_BYTES, "query_rows": QUERY_ROWS,
            "attention_score_bytes": ATTENTION_SCORE_BYTES, "expand_bytes": EXPAND_BYTES,
            "kv_budget_bytes": KV_BUDGET_BYTES, "vision_chunk": VISION_CHUNK, "kda_chunk": 64,
            "index_topk": text_config.index_topk, "index_kpool": text_config.index_kpool,
            "upstream_f32": UPSTREAM_F32}


def media_inputs(sequence, device, _dtype):
    """F32 pixel rows of every image in order, and their [t, h, w] grids."""
    pixels, grids = [], []
    for media in sequence.media:
        if media.kind == "video":
            raise ValueError(f"{sequence.name}: the GLM 5.3 Flash reference does not take video")
        if media.kind != "image" or media.grid is None or media.tensor.dtype != "f32" or media.grid.dtype != "i64":
            raise ValueError(f"{sequence.name}: GLM 5.3 Flash takes F32 image rows with an I64 grid")
        pixels.append(torch.frombuffer(bytearray(media.tensor.data), dtype=torch.float32).view(media.tensor.shape))
        grids.append(torch.frombuffer(bytearray(media.grid.data), dtype=torch.int64).view(-1, 3))
    return {"pixel_values": torch.cat(pixels).to(device), "image_grid_thw": torch.cat(grids).to(device)}


def chunk_rows(context, index_heads, kpool):
    """Prefill rows for a chunk ending at `context` keys: PREFILL_ROWS, fewer
    when the indexer's F32 [rows, heads, pools] scores would pass
    INDEXER_SCORE_BYTES, in whole ROW_ALIGN rows."""
    pools = -(-context // kpool)
    rows = min(PREFILL_ROWS, INDEXER_SCORE_BYTES // (4 * index_heads * pools))
    return max(ROW_ALIGN, rows // ROW_ALIGN * ROW_ALIGN)


def chunks(length, index_heads, kpool):
    """The [start, end) prefill chunks of a `length`-token prompt."""
    out, start = [], 0
    while start < length:
        end = min(length, start + chunk_rows(min(length, start + PREFILL_ROWS), index_heads, kpool))
        out.append((start, end))
        start = end
    return out


def cache_values(config):
    """Values cached per token in one sparse layer: the latent and rope part,
    and the indexer's key, gate scores and valid flag."""
    return config.kv_lora_rank + config.qk_rope_head_dim + 2 * config.index_head_dim + 1


class Plan:
    """The forward and decode schedule of one built model."""

    def __init__(self, text_config, sparse_devices):
        self.config, self.sparse = text_config, sparse_devices

    def check_context(self, length, dtype):
        """Refuses a sequence whose caches would pass KV_BUDGET_BYTES on a
        GPU: per token in every sparse layer, the latent and rope part and
        the indexer's packed key, gate scores and valid flag."""
        per_token = cache_values(self.config) * torch.empty((), dtype=dtype).element_size()
        for device in set(self.sparse):
            layers = self.sparse.count(device)
            if layers * per_token * length > KV_BUDGET_BYTES:
                most = KV_BUDGET_BYTES // (layers * per_token)
                raise ValueError(f"{length} tokens need {layers * per_token * length} bytes of latent and indexer "
                                 f"cache on {device}; the {dtype} reference holds at most {most} tokens")

    def prefill(self, model, ids, media, keep):
        """`Glm5NextModel.forward`'s steps over the prompt `ids` [1, L]: the
        cache and the logits of the last `keep` positions."""
        from transformers.cache_utils import DynamicCache
        inner = model.model
        embeds = inner.get_input_embeddings()(ids)
        if media:
            features = torch.cat(inner.get_image_features(media["pixel_values"], media["image_grid_thw"]).pooler_output)
            features = features.to(embeds.device, embeds.dtype)
            mask, _ = inner.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
            embeds = embeds.masked_scatter(mask, features)
        cache = DynamicCache(config=self.config)
        length, hidden = ids.shape[1], []
        for start, end in chunks(length, self.config.index_n_heads, self.config.index_kpool):
            with tap_writer.chunk(start, end):
                out = inner.language_model(inputs_embeds=embeds[:, start:end], past_key_values=cache, use_cache=True)
            first = max(start, length - keep)
            if first < end:
                hidden.append(out.last_hidden_state[:, first - start:])
        return cache, model.lm_head(torch.cat(hidden, dim=1))

    def forward(self, model, ids, keep, media):
        self.check_context(ids.shape[1], model.model.language_model.norm.weight.dtype)
        return self.prefill(model, ids, media, keep)[1][0]

    def decode(self, model, ids, media, max_new_tokens, stop_ids):
        """Greedy continuation from the prefill cache, one token per step."""
        self.check_context(ids.shape[1] + max_new_tokens, model.model.language_model.norm.weight.dtype)
        inner = model.model
        cache, logits = self.prefill(model, ids, media, 1)
        answer, rows = [], []
        row = logits[0, -1]
        while True:
            token = int(torch.argmax(row))
            answer.append(token)
            rows.append(row)
            if token in stop_ids or len(answer) == max_new_tokens:
                return answer, torch.stack(rows)
            step = torch.tensor([[token]], device=ids.device)
            out = inner.language_model(inputs_embeds=inner.get_input_embeddings()(step), past_key_values=cache,
                                       use_cache=True)
            row = model.lm_head(out.last_hidden_state)[0, -1]


def selection(value):
    """Indexer top-k key positions as u32 rows; HF's -1 (no key) becomes 2^32 - 1."""
    return value.to(torch.int64) & 0xFFFFFFFF


def taps(model):
    """Tap names (the runner's tap grammar) to the HF values carrying them.
    Text taps have one row per position: streams taps hold the four mHC
    streams (4 x hidden), KDA taps per-head values regrouped to [positions,
    heads x dim]. `x.indexer.ids` is a layer's selected key positions (a
    set; 2^32 - 1 where HF has none), `x.latent` the normalized KV latent
    and `x.q_latent` the normalized query latent the indexer reads."""
    text = model.model.language_model
    media = ("image",)
    names = {"embed.out": Hook(f"{TEXT}.layers.0", "input"),
             "final.x.hc.head": f"{TEXT}.hc_head",
             "final.norm": f"{TEXT}.norm",
             "enc.vision.out": Hook("model.visual.merger", rows=media)}
    for block in range(len(model.model.visual.blocks)):
        names[f"enc.vision.B{block:02d}.out"] = Hook(f"model.visual.blocks.{block}", rows=media)
    for layer, block in enumerate(text.layers):
        path, scope = f"{TEXT}.layers.{layer}", f"L{layer:02d}"
        attn = f"{path}.self_attn"
        names |= {
            f"{scope}.in.streams": Hook(path, "input"),
            f"{scope}.x.hc.attn.post": Hook(f"{path}.attn_hc", "output", 0),
            f"{scope}.x.hc.attn.comb": Hook(f"{path}.attn_hc", "output", 1),
            f"{scope}.x.hc.attn": Hook(f"{path}.attn_hc", "output", 2),
            f"{scope}.attn.in": f"{path}.input_layernorm",
            f"{scope}.attn.core": Hook(f"{attn}.o_proj", "input"),
            f"{scope}.attn.out": Hook(attn, "output", 0),
            f"{scope}.x.hc.mlp.streams": Hook(f"{path}.ffn_hc", "input"),
            f"{scope}.x.hc.mlp.post": Hook(f"{path}.ffn_hc", "output", 0),
            f"{scope}.x.hc.mlp.comb": Hook(f"{path}.ffn_hc", "output", 1),
            f"{scope}.x.hc.mlp": Hook(f"{path}.ffn_hc", "output", 2),
            f"{scope}.mlp.in": f"{path}.post_attention_layernorm",
            f"{scope}.mlp.out": f"{path}.mlp",
            f"{scope}.out": path,
        }
        if block.block_type == "linear_attention":
            names |= {f"{scope}.attn.q": f"{attn}.q_proj", f"{scope}.attn.k": f"{attn}.k_proj",
                      f"{scope}.attn.v": f"{attn}.v_proj",
                      f"{scope}.x.kda.g": f"{attn}.forget_gate",
                      f"{scope}.x.kda.beta": f"{attn}.b_proj",
                      f"{scope}.x.kda.core": Hook(f"{attn}.o_norm", "input", 0),
                      f"{scope}.x.kda.gate": Hook(f"{attn}.o_norm", "input", 1)}
        else:
            names |= {f"{scope}.x.q_latent": f"{attn}.q_a_layernorm",
                      f"{scope}.attn.q": f"{attn}.q_b_proj",
                      f"{scope}.x.latent": f"{attn}.kv_a_layernorm"}
            if block.self_attn.indexer is not None:
                names |= {f"{scope}.x.indexer.k": f"{attn}.indexer.k_norm",
                          f"{scope}.x.indexer.ids": Hook(f"{attn}.indexer", "output", 0, selection)}
        if text.config.mlp_layer_types[layer] == "sparse":
            names |= {f"{scope}.router.logits": Hook(f"{path}.mlp.gate", "output", 0),
                      f"{scope}.router.weights": Hook(f"{path}.mlp.gate", "output", 1),
                      f"{scope}.router.ids": Hook(f"{path}.mlp.gate", "output", 2),
                      f"{scope}.moe.routed": f"{path}.mlp.experts",
                      f"{scope}.moe.shared": f"{path}.mlp.shared_experts"}
    return names
