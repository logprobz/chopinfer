"""Qwen3.8 reference adapter: HF `qwen4_exp` (transformers 5.16.1) on the GPUs.

The runner (tools/refrun) builds `Qwen4ExpForConditionalGeneration` once per
mode on the meta device in the mode's dtype, as `from_pretrained(dtype=D)`
would, and fills it from the shared store:

- BF16 tensors (attention, gated delta net, hyper-connections, norms,
  router, shared expert, PLE projections, embeddings, head, vision tower)
  are cast to the mode dtype; I64 buffers are kept.
- Routed experts stay FP8 E4M3 with BF16 inverse scales per 128 x 128 block
  on their layer's GPU. HF's eager expert loop reads `gate_up_proj[e]` and
  `down_proj[e]`; those are `QuantStack`s whose items dequantize one
  expert's weights per call (gate and up stacked as HF's fused weight).
- The 47.7 GiB n-gram table stays in its shards on the host. The rows a
  prompt hashes to are read per call and multiplied by the table's per-tensor
  BF16 `weight_scale`, as vLLM and the engine do (HF 5.16.1 applies none).
- Dequantization follows the mode policy's rule for experts and n-gram rows
  alike. The model is the FP8 weights times their scales, so the f32 gate
  reference and the f64 run use `exact`. `hf-bf16` (the F32 product rounded
  to BF16, an artifact of HF's `dequantize=True` loading) belongs to the
  bf16 spread variant, which runs as HF BF16 does; `MODE:hf-bf16` exists
  only to compare with references made that way.
- Text attention and experts run HF's eager code. The vision tower's
  attention runs in query chunks of `VISION_CHUNK` rows over all of an
  image's keys, spread round robin over the GPUs: HF's SDPA call per chunk
  in f32 and bf16, `ops.attention` (f64 softmax) in f64, where SDPA would
  hold 16 heads of 64,800 squared scores.
- A forward pass runs `Qwen4ExpModel.forward`'s steps: image features
  merged into the token embeddings, M-RoPE positions from
  `compute_3d_position_ids` with the grids and image token types the
  processor gives, then the text model in chunks of `PREFILL_CHUNK` tokens
  with a `DynamicCache`, so eager attention over a 16K-token prompt holds one
  chunk of scores; each chunk is marked for the tap writer (`taps.chunk`).
  Greedy decoding continues from that cache.

f64 is the unmodified upstream code in f64: HF computes some steps in F32
whatever the model dtype (norms, the gated delta rule, the router softmax,
the eager softmax, QSA scoring, rotary tables), and they stay F32. Each run
records the list (`UPSTREAM_F32`) in its layout. A router or QSA near-tie
below that F32 rounding can therefore resolve differently from a pure f64
computation. Pixel rows come
from the workload (the processor's output); the processor never runs here.
The QSA indexer and the gated delta net run HF's torch code; the adapter
refuses to build when a kernel package (`fla`, `kernels`) could replace it.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

from tools.refrun import adapter, ops, quant, taps as tap_writer
from tools.refrun.place import HostRows, Unit
from tools.refrun.taps import Hook

MODEL = "qwen38"
TEXT = "model.language_model"
EXPERT = quant.fp8_block(128, 128)
VISION_CHUNK = 1024
PREFILL_CHUNK = 2048
NGRAM_READERS = 8
VISION_ATTENTION = "refrun_qwen38_vision"
UPSTREAM_F32 = {
    "Qwen4ExpTextRMSNorm": "normalizes x.float() and multiplies by 1 + weight.float()",
    "Qwen4ExpTextRMSNormGated": "normalizes and gates in F32",
    "torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule": "run in F32",
    "Qwen4ExpTextGatedDeltaNet": "decay g from A_log.float() and a.float()",
    "Qwen4ExpTextTopKRouter": "softmax of the router logits in F32",
    "eager_attention_forward": "softmax in F32",
    "Qwen4ExpTextQSAIndexer": "pooled keys and block scores in F32",
    "Qwen4ExpTextRotaryEmbedding / Qwen4ExpVisionRotaryEmbedding": "F32 inv_freq and tables",
    "apply_rotary_pos_emb_vision": "rotates q and k in F32",
}


def hf_modules():
    from transformers.models.qwen4_exp import configuration_qwen4_exp, modeling_qwen4_exp
    return configuration_qwen4_exp, modeling_qwen4_exp


def reference_code(_checkpoint):
    configuration, modeling = hf_modules()
    return {f"transformers/models/qwen4_exp/{Path(m.__file__).name}": Path(m.__file__)
            for m in (configuration, modeling)}


def ngram_prefix(layer):
    return f"{TEXT}.layers.{layer}.ple.ple_embedding.ngram_embedding."


def ple_layers(text_config):
    return [layer - 1 for layer in text_config.get("ple_layer_ids") or []]


def units(checkpoint):
    """Vision tower, embeddings, rotary tables, each decoder layer, final
    mixer, head; the n-gram shards as host units. MTP tensors are not run."""
    names = sorted(checkpoint.entries)
    text = checkpoint.config["text_config"]
    tables = {layer: ngram_prefix(layer) for layer in ple_layers(text)}

    def owned(prefix):
        return tuple(n for n in names if n.startswith(prefix)
                     and not any(n.startswith(t + "shard_") for t in tables.values()))
    out = [Unit("model.visual", owned("model.visual.")),
           Unit(f"{TEXT}.embed_tokens", owned(f"{TEXT}.embed_tokens.")),
           Unit(f"{TEXT}.rotary_emb", ())]
    for layer in range(text["num_hidden_layers"]):
        out.append(Unit(f"{TEXT}.layers.{layer}", owned(f"{TEXT}.layers.{layer}.")))
    out += [Unit(f"{TEXT}.hyper_connection_mixer", owned(f"{TEXT}.hyper_connection_mixer.")),
            Unit("lm_head", owned("lm_head."))]
    for prefix in tables.values():
        out.append(Unit(prefix.rstrip("."), tuple(n for n in names if n.startswith(prefix + "shard_")), host=True))
    return out


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


class NGramRows(nn.Module):
    """Stands in for the n-gram nn.Embedding: reads the hashed rows from the
    host shards and dequantizes them (E4M3 times the per-tensor scale) under
    the mode's rule, on the PLE layer's device."""

    def __init__(self, checkpoint, shards, scale, padded_rows, dtype, rule, device):
        super().__init__()
        self.tables = [HostRows(checkpoint, name) for name in shards]
        shapes = {table.entry.shape for table in self.tables}
        if len(shapes) != 1:
            raise ValueError(f"n-gram shards differ in shape: {sorted(shapes)}")
        self.rows_per_shard, self.cols = shapes.pop()
        if self.rows_per_shard * len(self.tables) != padded_rows:
            raise ValueError(f"{len(self.tables)} shards of {self.rows_per_shard} rows are not HF's "
                             f"{padded_rows}-row table")
        self.dtype, self.rule = dtype, rule
        self.register_buffer("scale", scale, persistent=False)
        # HF moves the row ids to `weight.device`.
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=device), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        by_shard = {}
        for position, row in enumerate(flat):
            shard, offset = divmod(row, self.rows_per_shard)
            by_shard.setdefault(shard, ([], []))
            by_shard[shard][0].append(position)
            by_shard[shard][1].append(offset)
        codes = torch.empty((len(flat), self.cols), dtype=torch.uint8)
        with ThreadPoolExecutor(NGRAM_READERS) as pool:
            for (positions, _), rows in zip(by_shard.values(), pool.map(
                    lambda item: self.tables[item[0]].rows(item[1][1]), by_shard.items())):
                codes[positions] = rows.view(torch.uint8)
        codes = codes.to(self.weight.device)
        fmt = quant.Format("e4m3", "float", len(flat), self.cols)
        values = quant.dequantize(fmt, codes, self.scale.view(1, 1), (len(flat), self.cols), self.dtype, self.rule)
        return values.view(*ids.shape, self.cols)


def vision_attention(devices):
    """HF's attention interface for one image's vision attention, query
    chunks spread round robin over `devices`. Each chunk attends to all of
    the image's keys; [B, H, L, D] in, [B, L, H, D] out."""
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
    for package in ("fla", "kernels"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"{package} is installed and could replace HF's torch gated delta rule or norms")


def build(config, placement):
    """The HF model in `config.dtype` around the store's weights."""
    from transformers import AttentionInterface
    configuration, hf = hf_modules()
    check_kernels()
    store, dtype, rule = config.store, config.dtype, config.policy["dequant"]
    hf_config = configuration.Qwen4ExpConfig.from_dict(dict(config.checkpoint.config))
    text_config = hf_config.text_config
    AttentionInterface.register(VISION_ATTENTION, vision_attention(placement.devices))
    implementations([hf_config.text_config], [hf_config.vision_config])
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = hf.Qwen4ExpForConditionalGeneration(hf_config)
    finally:
        torch.set_default_dtype(default)
    configs = {id(m.config): m.config for m in model.modules() if isinstance(getattr(m, "config", None),
                                                                           configuration.Qwen4ExpTextConfig)}
    vision = {id(m.config): m.config for m in model.modules() if isinstance(getattr(m, "config", None),
                                                                          configuration.Qwen4ExpVisionConfig)}
    implementations([text_config, *configs.values()], [hf_config.vision_config, *vision.values()])
    devices = dict(placement.units)

    def device_of(path):
        while path not in devices:
            path = path.rpartition(".")[0]
        return torch.device(devices[path])
    names, state = {}, {}
    parameters = {n for n, _ in model.named_parameters()}
    tables = {ngram_prefix(layer) for layer in ple_layers(config.checkpoint.config["text_config"])}
    for name, tensor in store.tensors.items():
        if ".mlp.experts." in name or any(name.startswith(t) for t in tables):
            continue
        state[name] = tensor.to(dtype) if name in parameters and tensor.is_floating_point() else tensor
        names[name] = name
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"stored tensors with no place in the HF model: {unexpected[:5]}")
    text = model.model.language_model
    inter, hidden = text_config.moe_intermediate_size, text_config.hidden_size
    for layer, block in enumerate(text.layers):
        base = f"{TEXT}.layers.{layer}.mlp.experts"
        stacks = {"gate_up": [], "down": []}
        for expert in range(text_config.num_experts):
            prefix = f"{base}.{expert}."

            def stored(projection, shape, slot):
                codes = prefix + projection + ".weight"
                return store.tensor(EXPERT, codes, codes + "_scale_inv", shape, slot)
            gate, up = stored("gate_proj", (inter, hidden), "gate"), stored("up_proj", (inter, hidden), "up")
            stacks["gate_up"].append(GateUp(gate, up))
            stacks["down"].append(stored("down_proj", (hidden, inter), "down"))
            for projection, path in (("gate_proj", f"gate_up_proj.items.{expert}.gate"),
                                     ("up_proj", f"gate_up_proj.items.{expert}.up"),
                                     ("down_proj", f"down_proj.items.{expert}")):
                names[prefix + projection + ".weight"] = names[prefix + projection + ".weight_scale_inv"] = \
                    f"{base}.{path}"
        experts = block.mlp.experts
        del experts._parameters["gate_up_proj"], experts._parameters["down_proj"]
        experts.gate_up_proj = quant.QuantStack(stacks["gate_up"])
        experts.down_proj = quant.QuantStack(stacks["down"])
        if block.ple is not None:
            prefix = ngram_prefix(layer)
            embedding = block.ple.ple_embedding
            divisor = text_config.make_ngram_vocab_size_divisible_by
            shards = sorted((n for n in config.checkpoint.entries if n.startswith(prefix + "shard_")),
                            key=lambda n: int(n[len(prefix + "shard_"):].split(".")[0]))
            if len(shards) != text_config.split_ngram_parts:
                raise ValueError(f"layer {layer}: {len(shards)} n-gram shards, config says "
                                 f"{text_config.split_ngram_parts}")
            embedding.ngram_embedding = NGramRows(
                config.checkpoint, shards, store.get(prefix + "weight_scale"),
                math.ceil(embedding.total_vocab_size / divisor) * divisor, dtype, rule, device_of(prefix.rstrip(".")))
            names[prefix + "weight_scale"] = prefix.rstrip(".")
            names |= {shard: "host:ngram_rows" for shard in shards}
    # Computed, not stored: the rotary tables, in F32 as HF builds them.
    text.rotary_emb = hf.Qwen4ExpTextRotaryEmbedding(text_config).to(device_of(f"{TEXT}.rotary_emb"))
    visual, vision = model.model.visual, hf_config.vision_config
    visual.rotary_pos_emb = hf.Qwen4ExpVisionRotaryEmbedding(vision.hidden_size // vision.num_heads // 2).to(
        device_of("model.visual"))
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{len(left)} tensors not loaded, e.g. {left[:5]}")
    model.eval()
    return adapter.Reference(
        model, names, layout=layout(text_config, rule), media_inputs=media_inputs, taps=taps,
        forward=forward, decode=decode,
        replacements={
            "mlp.experts.gate_up_proj, mlp.experts.down_proj": "QuantStack of FP8 128x128-block QuantTensors, "
                                                               "one expert dequantized per call",
            "ple.ple_embedding.ngram_embedding": "host row gather (place.HostRows) times the BF16 weight_scale",
            "vision attention": f"query chunks of {VISION_CHUNK} over all keys of an image, round robin over "
                                "the GPUs: HF sdpa_attention_forward in f32 and bf16, ops.attention in f64",
            "Qwen4ExpModel.forward": f"the same steps, text model in chunks of {PREFILL_CHUNK} tokens with a "
                                     "DynamicCache"})


def implementations(text, vision):
    """Eager text attention and experts, the chunked vision attention, on
    every config object the modules read."""
    for config in text:
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
    for config in vision:
        config._attn_implementation = VISION_ATTENTION


def layout(text_config, rule):
    return {"experts": "FP8 E4M3 [I, H] and [H, I] per expert, BF16 inverse scales per 128x128 block; gate and "
                       "up stacked as gate_up_proj",
            "ngram_embedding": f"E4M3 rows in {text_config.split_ngram_parts} host shards, row r in shard "
                               "r // rows_per_shard, times the per-tensor BF16 weight_scale",
            "dequant": rule, "prefill_chunk": PREFILL_CHUNK, "vision_chunk": VISION_CHUNK,
            "upstream_f32": UPSTREAM_F32}


def media_inputs(sequence, device, _dtype):
    """F32 pixel rows of every image in order, and their [t, h, w] grids."""
    pixels, grids = [], []
    for media in sequence.media:
        if media.kind != "image" or media.grid is None or media.tensor.dtype != "f32" or media.grid.dtype != "i64":
            raise ValueError(f"{sequence.name}: Qwen3.8 takes F32 image rows with an I64 grid")
        pixels.append(torch.frombuffer(bytearray(media.tensor.data), dtype=torch.float32).view(media.tensor.shape))
        grids.append(torch.frombuffer(bytearray(media.grid.data), dtype=torch.int64).view(-1, 3))
    return {"pixel_values": torch.cat(pixels).to(device), "image_grid_thw": torch.cat(grids).to(device)}


def prefill(model, ids, media, keep):
    """`Qwen4ExpModel.forward`'s steps over the prompt `ids` [1, L]: the cache
    and the logits of the last `keep` positions."""
    from transformers.cache_utils import DynamicCache
    inner, config = model.model, model.config
    embeds = inner.get_input_embeddings()(ids)
    grid = media.get("image_grid_thw")
    types = None
    if media:
        features = torch.cat(inner.get_image_features(media["pixel_values"], grid, return_dict=True).pooler_output)
        features = features.to(embeds.device, embeds.dtype)
        mask, _ = inner.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
        embeds = embeds.masked_scatter(mask, features)
        types = (ids == config.image_token_id).int()
    inner.rope_deltas = None
    positions = inner.compute_3d_position_ids(input_ids=ids, inputs_embeds=embeds, image_grid_thw=grid,
                                              mm_token_type_ids=types)
    cache = DynamicCache(config=config.text_config)
    length, hidden = ids.shape[1], []
    for start in range(0, length, PREFILL_CHUNK):
        end = min(length, start + PREFILL_CHUNK)
        with tap_writer.chunk(start, end):
            out = inner.language_model(inputs_embeds=embeds[:, start:end],
                                       position_ids=None if positions is None else positions[..., start:end],
                                       past_key_values=cache, use_cache=True, ple_input_ids=ids[:, start:end])
        first = max(start, length - keep)
        if first < end:
            hidden.append(out.last_hidden_state[:, first - start:])
    return cache, model.lm_head(torch.cat(hidden, dim=1))


def forward(model, ids, keep, media):
    return prefill(model, ids, media, keep)[1][0]


def decode(model, ids, media, max_new_tokens, stop_ids):
    """Greedy continuation from the prefill cache, one token per step."""
    inner = model.model
    cache, logits = prefill(model, ids, media, 1)
    answer, rows = [], []
    row = logits[0, -1]
    while True:
        token = int(torch.argmax(row))
        answer.append(token)
        rows.append(row)
        if token in stop_ids or len(answer) == max_new_tokens:
            return answer, torch.stack(rows)
        step = torch.tensor([[token]], device=ids.device)
        embeds = inner.get_input_embeddings()(step)
        positions = inner.compute_3d_position_ids(input_ids=step, inputs_embeds=embeds, past_key_values=cache)
        out = inner.language_model(inputs_embeds=embeds, position_ids=positions, past_key_values=cache,
                                   use_cache=True, ple_input_ids=step)
        row = model.lm_head(out.last_hidden_state)[0, -1]


def taps(model):
    """Tap names (REFERENCE-RUNNER.md 4.1, the runner's tap grammar) to the
    HF values carrying them; tapdiff maps the engine's Qwen dump names onto
    the same sites. Text taps have one row per position (the gated delta
    rule's per-head core output is regrouped to [positions, value heads x
    head dim]); vision taps have the tower's rows, `enc.vision.out` the
    merged rows that replace the image tokens."""
    text = model.model.language_model
    media = ("image", "video")
    names = {"embed.out": Hook(f"{TEXT}.layers.0", "input"),
             "final.x.hc.mixer": f"{TEXT}.hyper_connection_mixer",
             "enc.vision.out": Hook("model.visual.merger", rows=media)}
    for block in range(len(model.model.visual.blocks)):
        names[f"enc.vision.B{block:02d}.out"] = Hook(f"model.visual.blocks.{block}", rows=media)
    for layer, block in enumerate(text.layers):
        path, scope = f"{TEXT}.layers.{layer}", f"L{layer:02d}"
        names |= {
            f"{scope}.in.streams": Hook(path, "input"),
            f"{scope}.x.hc.attn.xn": f"{path}.attn_hyper_connection.hc_norm",
            f"{scope}.x.hc.attn": Hook(f"{path}.attn_hyper_connection", "output", 0),
            f"{scope}.x.hc.mlp.xn": f"{path}.mlp_hyper_connection.hc_norm",
            f"{scope}.x.hc.mlp": Hook(f"{path}.mlp_hyper_connection", "output", 0),
            f"{scope}.router.logits": Hook(f"{path}.mlp.gate", "output", 0),
            f"{scope}.router.weights": Hook(f"{path}.mlp.gate", "output", 1),
            f"{scope}.router.ids": Hook(f"{path}.mlp.gate", "output", 2),
            f"{scope}.moe.routed": f"{path}.mlp.experts",
            f"{scope}.moe.shared": f"{path}.mlp.shared_expert",
            f"{scope}.mlp.out": f"{path}.mlp",
            f"{scope}.out": path,
        }
        if block.layer_type == "linear_attention":
            width = block.linear_attn.value_dim
            names |= {f"{scope}.x.gdn.qkv": f"{path}.linear_attn.in_proj_qkv",
                      f"{scope}.x.gdn.z": f"{path}.linear_attn.in_proj_z",
                      f"{scope}.x.gdn.core": Hook(f"{path}.linear_attn.norm", "input", 0,
                                                  lambda value, width=width: value.reshape(-1, width)),
                      f"{scope}.attn.core": Hook(f"{path}.linear_attn.out_proj", "input"),
                      f"{scope}.attn.out": f"{path}.linear_attn"}
        else:
            names |= {f"{scope}.attn.core": Hook(f"{path}.self_attn.o_proj", "input"),
                      f"{scope}.attn.out": Hook(f"{path}.self_attn", "output", 0)}
        if block.ple is not None:
            names[f"{scope}.x.ple.out"] = Hook(f"{path}.attn_hyper_connection", "input")
    return names
