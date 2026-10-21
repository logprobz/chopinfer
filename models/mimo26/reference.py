"""MiMo V2.6 Flash reference adapter for the GPU reference runner (tools/refrun).

It runs the vendor's modeling_mimo_v2.py and configuration_mimo_v2.py from
the checkpoint directory, executed from the bytes whose sha256 is pinned in
VENDOR (never trust_remote_code), with the audio tokenizer from
audio_tokenizer/ (EXTRA_SHARDS).

Checkpoint facts the vendor code does not read as stored, stated once here:

- `self_attn.qkv_proj` is stored as 4 TP chunks of [Q/4 | K/4 | V/4] rows,
  FP8 E4M3 on 128x128 blocks with F32 scales whose grid restarts in every
  chunk (108 scale rows for 13,568 rows, 116 for 14,848). The vendor splits
  the projection contiguously, so the rows are permuted to Q | K | V, each
  keeping its own scale row (`qkv_rows`). Negative control:
  reference_contiguous_qkv.py, which must score garbage.
- Routed experts are MXFP4 (U8 packed E2M1 codes, U8 E8M0 scales per 32 K);
  the layer-0 MLP is FP8 like the QKV without chunks; `o_proj` is BF16.
- `visual.merger.ln_q` stores a weight only, which fits both an RMSNorm
  and a LayerNorm without bias. The adapter's default RMSNorm departs from
  the vendor code on GPU evidence (see MERGERS), not on a storage fact.
- `visual.merger.mlp` stores no biases although the vendor's linears have
  them; they run without one, as HF's zero-initialized missing bias would.
- `tie_word_embeddings` is false although the vendor class declares
  `_tied_weights_keys`; `lm_head.weight` is stored and must differ from the
  embedding table.

The merger normalization, `visual.merger.ln_q` (`MERGERS`, chosen by
`build(merger=...)`; reference_merger_layernorm.py runs the vendor's choice):

- LayerNorm: the pinned vendor modeling_mimo_v2.py (MiMoVisionPatchMerger,
  nn.LayerNorm, whose bias the checkpoint does not store and HF would
  initialize to zero); NeMo Automodel's MiMo-V2.6 vision (LayerNorm,
  bias=False). The vendor merger is a known upstream discrepancy: it also
  declares merger linear biases the checkpoint lacks.
- RMSNorm (default): vLLM's mimo_v2_omni and SGLang's mimo_vl (the
  Qwen2.5-VL merger), the serving engine, and the encoder oracle that the
  current media references come from. A trained LayerNorm would usually
  keep an affine bias; none is stored.
- vision_config has no normalization selector. Paired F32 GPU probes with
  this adapter (2026-10-01; 313 objective image probes plus 74 earlier)
  found LayerNorm no better overall and significantly worse on OCR and tiny
  numbers, so RMSNorm stays the default. docs/serving-mimo26.md (vision
  merger normalization) has the numbers. The tiny CPU parity test applies
  the same choice on both sides, so it validates the implementation of each
  choice, not the trained one.

Replacements (`REPLACEMENTS`) compute exactly the vendor's operations in F32
and BF16 runs; in f64 they keep f64 where the vendor casts to F32 (RMSNorm
variance, router logits, eager softmax, vision RoPE, the audio quantizer).
Rotary tables stay F32 as the vendor computes them. The vendor's mask
construction passes transformers 5.3 keywords that 5.16.1 refuses, so
`forward` builds the eager masks with 5.16.1's own functions and passes the
mapping the vendor model accepts. Answers must be forced: no decode.
"""

import json
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from tools.refrun import adapter, ops, quant
from tools.refrun.place import Unit
from tools.refrun.taps import Hook

MODEL = "mimo26"
VENDOR = {
    "configuration_mimo_v2.py": "773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e",
    "modeling_mimo_v2.py": "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad",
    "audio_tokenizer/config.json": "e0702adae37947e0c980c38bae58ffa0d48bd492d523afe814aa4d73f008c7d1",
}
EXTRA_SHARDS = {"audio_tokenizer/model.safetensors": "audio_tokenizer."}
PACKAGE = "chopinfer_mimo26_vendor"
STEMS = ("configuration_mimo_v2", "modeling_mimo_v2")
TP_CHUNKS = 4
MERGERS = {
    "rmsnorm": "RMSNorm over the stored weight, overriding the vendor's LayerNorm on GPU probe evidence "
               "(vLLM, SGLang, the engine and the encoder oracle use RMSNorm)",
    "layernorm": "the vendor's LayerNorm without bias over the stored weight (NeMo uses the same); "
                 "a known upstream discrepancy",
}
FP8 = (128, 128)
REPLACEMENTS = {
    "MiMoV2RMSNorm, Qwen2RMSNorm": "plain module (no hub kernel), variance in F32 or wider",
    "eager_attention_forward": "the same math per block of 1,024 queries, softmax in F32 or wider",
    "MiMoV2MoEGate.forward": "the same routing, logits in F32 or wider",
    "_apply_rotary_pos_emb_vision": "the same rotation in F32 or wider",
    "AudioTokenizerEncoder.encode": "the codes path, quantizer in F32 or wider",
    "F.scaled_dot_product_attention (vision tower, audio tokenizer)": "ops.attention for f64 inputs",
    "visual.merger.ln_q": "per MERGERS: RMSNorm (default) or the vendor's LayerNorm without bias",
    "qkv_proj, layer-0 mlp, experts": "QuantLinear",
    "causal masks": "transformers 5.16.1 eager masks passed as the vendor's mask mapping",
}


def wide(dtype):
    """F32, or f64 for f64 inputs: where the vendor casts to F32."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def vendor(checkpoint):
    """The vendor modeling module, executed from its pinned bytes, with the
    module-level replacements applied. Loaded once per process; every call
    first checks this checkpoint's vendor files against the pins, so a
    reused module always stands for the bytes the caller's checkpoint holds."""
    for stem in STEMS:
        checkpoint.pinned(f"{stem}.py", VENDOR[f"{stem}.py"])
    name = f"{PACKAGE}.modeling_mimo_v2"
    if name not in sys.modules:
        modules = load_vendor(checkpoint, PACKAGE)
        replace_functions(modules["modeling_mimo_v2"])
    module = sys.modules[name]
    if module.EXECUTED_SHA256 != {stem: VENDOR[f"{stem}.py"] for stem in STEMS}:
        raise ValueError("the loaded vendor module is not the pinned source")
    return module


def load_vendor(checkpoint, package):
    """configuration_mimo_v2 and modeling_mimo_v2 as submodules of `package`;
    the modeling module records the sha256 of the bytes it executed."""
    root = types.ModuleType(package)
    root.__path__ = []
    sys.modules[package] = root
    loaded = {}
    for stem in STEMS:
        module = types.ModuleType(f"{package}.{stem}")
        module.__package__ = package
        module.__file__ = str(checkpoint.root / f"{stem}.py")
        sys.modules[module.__name__] = module
        source = checkpoint.pinned(f"{stem}.py", VENDOR[f"{stem}.py"])
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        loaded[stem] = module
    loaded["modeling_mimo_v2"].EXECUTED_SHA256 = {stem: VENDOR[f"{stem}.py"] for stem in STEMS}
    return loaded


def replace_functions(modeling):
    modeling.eager_attention_forward = eager_attention
    modeling._apply_rotary_pos_emb_vision = rotary_vision
    functional = {key: getattr(F, key) for key in dir(F) if not key.startswith("__")}
    modeling.F = types.SimpleNamespace(**{**functional, "scaled_dot_product_attention": scaled_dot_product})


def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None, **_kwargs):
    """modeling_mimo_v2.eager_attention_forward per block of queries: sinks
    concatenated as a last column and dropped after the softmax."""
    if dropout:
        raise ValueError("reference runs have no dropout")
    keys_ = ops.repeat_kv(key, module.num_key_value_groups)
    values = ops.repeat_kv(value, module.num_key_value_groups)
    blocks = []
    for start in range(0, query.shape[-2], ops.CHUNK):
        block = query[:, :, start:start + ops.CHUNK]
        weights = torch.matmul(block, keys_.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, start:start + block.shape[-2], :keys_.shape[-2]]
        if sinks is not None:
            sink = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(block.shape[0], -1, block.shape[-2], -1)
            weights = torch.cat([weights, sink], dim=-1)
        weights = weights - weights.max(dim=-1, keepdim=True).values
        probabilities = F.softmax(weights, dim=-1, dtype=wide(query.dtype)).to(query.dtype)
        if sinks is not None:
            probabilities = probabilities[..., :-1]
        blocks.append(torch.matmul(probabilities, values))
    return torch.cat(blocks, dim=2).transpose(1, 2).contiguous(), None


def rotary_vision(q, k, cos, sin):
    """_apply_rotary_pos_emb_vision in F32, or f64 for f64 inputs."""
    q_dtype, k_dtype, dtype = q.dtype, k.dtype, wide(q.dtype)
    q, k = q.to(dtype), k.to(dtype)
    cos, sin = cos.unsqueeze(-2).to(dtype), sin.unsqueeze(-2).to(dtype)
    half = q.shape[-1] // 2

    def rotate(x):
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return ((q * cos) + (rotate(q) * sin)).to(q_dtype), ((k * cos) + (rotate(k) * sin)).to(k_dtype)


def scaled_dot_product(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                       enable_gqa=False):
    """The encoders' SDPA calls: torch's for F32 and BF16, the query-chunked
    exact attention for f64."""
    if query.dtype != torch.float64:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                              is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if dropout_p or is_causal or enable_gqa:
        raise ValueError("the encoders' attention passes masks, without dropout or GQA expansion")
    return ops.attention(query, key, value, query.shape[-1] ** -0.5 if scale is None else scale, attn_mask)


class RMSNorm(nn.Module):
    """MiMoV2RMSNorm's and Qwen2RMSNorm's forward as a plain module, with the
    variance in F32, or f64 for f64 inputs."""

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(wide(dtype))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype)


def route(gate, hidden_states):
    """MiMoV2MoEGate.forward (sigmoid scores, noaux_tc, inference) with the
    logits in F32, or f64 for f64 inputs."""
    if gate.scoring_func != "sigmoid" or gate.topk_method != "noaux_tc" or gate.training:
        raise NotImplementedError("the reference routes only sigmoid noaux_tc inference")
    tokens = hidden_states.shape[0] * hidden_states.shape[1]
    dtype = wide(hidden_states.dtype)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    scores = F.linear(hidden_states.type(dtype), gate.weight.type(dtype), None).sigmoid()
    choice = scores.view(tokens, -1) + gate.e_score_correction_bias.unsqueeze(0)
    group_scores = choice.view(tokens, gate.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=gate.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        tokens, gate.n_group, gate.n_routed_experts // gate.n_group).reshape(tokens, -1)
    _, topk_idx = torch.topk(choice.masked_fill(~score_mask.bool(), float("-inf")), k=gate.top_k, dim=-1,
                             sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if gate.top_k > 1 and gate.norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight * gate.routed_scaling_factor


def tokenizer_encode(encoder, input_features, input_lens=None, output_length=None, return_codes_only=False,
                     n_q=None, use_quantizer=True):
    """AudioTokenizerEncoder.encode's codes path (what tokenize_audio_batch
    calls), with the quantizer in F32, or f64 for an f64 encoder."""
    if not (return_codes_only and use_quantizer):
        raise NotImplementedError("the reference only tokenizes audio into codes")
    modeling = sys.modules[type(encoder).__module__]
    if output_length is None:
        output_length = encoder.get_output_length(input_lens)
    input_features = modeling._at_unpack_hidden_states(input_features, input_lens)
    hidden_states, output_length, *_ = encoder.get_features(
        input_features=input_features.transpose(1, 2), output_length=output_length)
    dtype = wide(hidden_states.dtype)
    encoder.quantizer.to(dtype)
    return encoder.quantizer.encode(hidden_states.to(dtype), n_q=n_q), output_length


def qkv_rows(q, k, v, chunks=TP_CHUNKS):
    """The stored row of each row of the vendor's contiguous Q | K | V: the
    checkpoint keeps chunk c's [Q/c | K/c | V/c] rows one after another."""
    if q % chunks or k % chunks or v % chunks:
        raise ValueError(f"QKV widths {q}, {k}, {v} do not split into {chunks} chunks")
    widths = (q // chunks, k // chunks, v // chunks)
    stride = sum(widths)
    order = []
    for part, width in enumerate(widths):
        offset = sum(widths[:part])
        for chunk in range(chunks):
            order.extend(range(chunk * stride + offset, chunk * stride + offset + width))
    return order


def reference_code(checkpoint):
    import transformers
    from transformers.models.qwen2 import modeling_qwen2
    from pathlib import Path
    upstream = Path(transformers.__file__).parent
    for file, digest in VENDOR.items():
        checkpoint.pinned(file, digest)
    return {**{f"mimo26/{file}": checkpoint.root / file for file in VENDOR},
            "transformers/models/qwen2/modeling_qwen2.py": Path(modeling_qwen2.__file__),
            "transformers/masking_utils.py": upstream / "masking_utils.py",
            "transformers/integrations/sdpa_attention.py": upstream / "integrations/sdpa_attention.py"}


ENCODERS = ("visual", "audio_tokenizer", "speech_embeddings", "audio_encoder")


def units(checkpoint):
    """Encoders, embeddings, rotary tables, each decoder layer, final norm,
    head. MTP layers and the audio tokenizer's decoder are not run."""
    names = sorted(checkpoint.entries)

    def owned(prefix):
        return tuple(n for n in names if n.startswith(prefix))
    layers = checkpoint.config["num_hidden_layers"]
    return [Unit("visual", owned("visual.")), Unit("audio_tokenizer", owned("audio_tokenizer.encoder.")),
            Unit("speech_embeddings", owned("speech_embeddings.")), Unit("audio_encoder", owned("audio_encoder.")),
            Unit("model.embed_tokens", owned("model.embed_tokens.")), Unit("model.rotary_emb", ()),
            Unit("model.swa_rotary_emb", ()),
            *(Unit(f"model.layers.{i}", owned(f"model.layers.{i}.")) for i in range(layers)),
            Unit("model.norm", owned("model.norm.")), Unit("lm_head", owned("lm_head."))]


def configs(checkpoint, modeling):
    text = sys.modules[f"{modeling.__package__}.configuration_mimo_v2"].MiMoV2Config.from_dict(
        dict(checkpoint.config))
    text._attn_implementation = "eager"
    tokenizer = modeling.MiMoAudioTokenizerConfig(**json.loads(checkpoint.pinned(
        "audio_tokenizer/config.json", VENDOR["audio_tokenizer/config.json"])))
    return text, tokenizer


def encoders(modeling, text_config, tokenizer_config):
    """The four encoder modules, built as MiMoV2ForCausalLM builds them, in
    one container with the model's module paths."""
    container = nn.Module()
    container.visual = modeling.MiMoVisionTransformer(modeling._as_namespace(text_config.vision_config))
    audio = modeling._as_namespace(text_config.audio_config)
    container.speech_embeddings = modeling._build_speech_embeddings(audio)
    container.audio_encoder = modeling.MiMoAudioEncoder(audio)
    container.audio_tokenizer = modeling.MiMoAudioTokenizer(tokenizer_config)
    return container


def build(config, placement, qkv="tp-chunks", merger="rmsnorm"):
    """The vendor MiMoV2ForCausalLM in `config.dtype` around the shared stored
    weights. `qkv="contiguous"` reads the stored QKV rows as the vendor's
    contiguous split (the negative control); `merger` picks the merger
    normalization (MERGERS)."""
    modeling = vendor(config.checkpoint)
    text_config, tokenizer_config = configs(config.checkpoint, modeling)
    with torch.device("meta"):
        model = modeling.MiMoV2ForCausalLM(text_config).eval()
        model.audio_tokenizer = modeling.MiMoAudioTokenizer(tokenizer_config)
    if model.config.tie_word_embeddings:
        raise ValueError("this adapter expects tie_word_embeddings false")
    model.model.rotary_emb = modeling.MiMoV2RotaryEmbedding(text_config, is_swa=False).to(
        placement.device("model.rotary_emb"))
    model.model.swa_rotary_emb = modeling.MiMoV2RotaryEmbedding(text_config, is_swa=True).to(
        placement.device("model.swa_rotary_emb"))
    names, layout = finish(model, config, placement, qkv, merger)
    head, table = model.lm_head.weight, model.model.embed_tokens.weight
    if head is table or (not head.is_meta and torch.equal(head, table.to(head.device))):
        raise ValueError("lm_head and the embedding table are one tensor; the checkpoint stores two")
    layout["lm_head"] = "stored separately from model.embed_tokens (tie_word_embeddings false), checked unequal"
    return adapter.Reference(model, names, layout=layout, replacements=REPLACEMENTS,
                             media_inputs=media_inputs(model), taps=taps,
                             forward=forward, decode=decode)


def build_encoders(config, placement, merger="rmsnorm"):
    """The encoders alone (`encoders`), for encoder row checks."""
    modeling = vendor(config.checkpoint)
    text_config, tokenizer_config = configs(config.checkpoint, modeling)
    with torch.device("meta"):
        container = encoders(modeling, text_config, tokenizer_config).eval()
    names, layout = finish(container, config, placement, merger=merger)
    return adapter.Reference(container, names, layout=layout, replacements=REPLACEMENTS,
                             media_inputs=media_inputs(container))


def finish(model, config, placement, qkv="tp-chunks", merger="rmsnorm"):
    """Replaces modules, rebuilds rotary tables, and loads the stored weights
    into a model built on the meta device. Returns the tensor map and the
    layout facts applied."""
    store, dtype, checkpoint = config.store, config.dtype, config.checkpoint
    if merger not in MERGERS:
        raise ValueError(f"unknown merger normalization {merger}; expected one of {sorted(MERGERS)}")
    modeling = vendor(checkpoint)
    first = placement.device(ENCODERS[0])
    if any(placement.device(unit) != first for unit in ENCODERS):
        raise ValueError("the encoders must share a device")
    visual, tokenizer = model.visual, model.audio_tokenizer.encoder
    local = model.audio_encoder.input_local_transformer
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(2 * visual.rotary_pos_emb.inv_freq.shape[0]).to(first)
    tc = tokenizer.config
    tokenizer.position_embedding = modeling.AudioTokenizerRotaryEmbedding(
        tc.rope_theta, tc.d_model // tc.encoder_attention_heads, tokenizer.max_source_positions,
        tc.rope_type).to(first)
    local.rotary_emb = type(local.rotary_emb)(config=local.config).to(first)
    local.embed_tokens = None
    for index in (0, 2):
        if f"visual.merger.mlp.{index}.bias" in checkpoint.entries:
            raise ValueError("the checkpoint now stores merger biases; the adapter assumes none")
        visual.merger.mlp[index].bias = None
    tokenizer.encode = types.MethodType(tokenizer_encode, tokenizer)
    with torch.device("meta"):
        norm = visual.merger.ln_q
        visual.merger.ln_q = nn.RMSNorm(norm.normalized_shape, eps=norm.eps) if merger == "rmsnorm" \
            else nn.LayerNorm(norm.normalized_shape, eps=norm.eps, bias=False)
        for path, module in list(model.named_modules()):
            if type(module).__name__ in ("MiMoV2RMSNorm", "Qwen2RMSNorm"):
                parent, _, leaf = path.rpartition(".")
                setattr(model.get_submodule(parent), leaf, RMSNorm(module.weight.shape[0], module.variance_epsilon))
    for module in model.modules():
        if type(module).__name__ == "MiMoV2MoEGate":
            module.forward = types.MethodType(route, module)

    names, state = {}, {}
    quantized = set()
    for name in store.tensors:
        entry = checkpoint.entries[name]
        if entry.dtype == "F8_E4M3" or (entry.dtype == "U8" and f"{name}_scale" in checkpoint.entries):
            path = name.removesuffix(".weight")
            linear = quantized_linear(store, checkpoint, model, name, path, qkv)
            parent, _, leaf = path.rpartition(".")
            setattr(model.get_submodule(parent), leaf, linear)
            scale = f"{name}_scale_inv" if entry.dtype == "F8_E4M3" else f"{name}_scale"
            names[name] = names[scale] = path
            quantized.update((name, scale))
    for name, tensor in store.tensors.items():
        if name not in quantized:
            state[name] = tensor.to(dtype)
            names[name] = name
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(f"stored tensors without a model path: {result.unexpected_keys[:3]}")
    if store.tensors and placement.devices[0] != "meta":
        left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
        if left:
            raise ValueError(f"{len(left)} model tensors have no stored value, e.g. {left[:3]}")
    layout = {
        "visual.merger.ln_q": f"weight only; {merger}: {MERGERS[merger]}",
        "visual.merger.mlp": "no stored biases: the linears run without one (HF would initialize them to zero)",
        "audio_tokenizer": "audio_tokenizer/model.safetensors, encoder and quantizer only",
    }
    if hasattr(model, "model"):
        layout.update({
            "qkv_proj": (f"FP8 E4M3 {FP8[0]}x{FP8[1]} blocks, F32 scales; {TP_CHUNKS} TP chunks of "
                         "[Q/4|K/4|V/4], scale grid restarting per chunk; "
                         + ("read contiguously (negative control)" if qkv == "contiguous"
                            else "rows permuted to the vendor's Q|K|V")),
            "experts": "MXFP4: U8 packed E2M1 codes, U8 E8M0 scales per 32 K",
            "mlp": "layer-0 dense MLP FP8 128x128 blocks, F32 scales",
            "o_proj": "BF16",
        })
    return names, layout


def quantized_linear(store, checkpoint, model, name, path, qkv):
    entry = checkpoint.entries[name]
    if entry.dtype == "U8":
        rows, packed = entry.shape
        return store.linear(quant.MXFP4, name, f"{name}_scale", (rows, 2 * packed))
    rows, cols = entry.shape
    if not path.endswith(".qkv_proj"):
        return store.linear(quant.fp8_block(*FP8), name, f"{name}_scale_inv", (rows, cols))
    attention = model.get_submodule(path.removesuffix(".qkv_proj"))
    widths = (attention.q_size, attention.k_size, attention.v_size)
    if sum(widths) != rows:
        raise ValueError(f"{name} has {rows} rows, not Q|K|V {widths}")
    order = None if qkv == "contiguous" else qkv_rows(*widths)
    if qkv not in ("contiguous", "tp-chunks"):
        raise ValueError(f"unknown QKV reading {qkv}")
    return store.linear(quant.fp8_block(*FP8, chunk_rows=rows // TP_CHUNKS), name, f"{name}_scale_inv",
                        (rows, cols), order=order)


def tensor(spec, device):
    dtypes = {"bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64, "i64": torch.int64,
              "i32": torch.int32}
    return torch.frombuffer(bytearray(spec.data), dtype=dtypes[spec.dtype]).reshape(spec.shape).to(device)


def audio_codes(model, mel, segment_size):
    """[codes, channels] RVQ codes of one [frames, mels] spectrogram, as
    get_audio_feature tokenizes it."""
    modeling = sys.modules[type(model.audio_tokenizer).__module__]
    (codes,) = modeling.tokenize_audio_batch([mel], model.audio_tokenizer.encoder, segment_size=segment_size)
    return codes


def media_inputs(model):
    """Forward kwargs from processor outputs: BF16 or F32 pixel rows with
    their [t, h, w] grid for images and video, F32 mel frames for audio
    (tokenized here, under the mode)."""
    segment = model.audio_encoder.audio_segment_size

    def inputs(sequence, device, _dtype):
        kinds = {}
        for item in sequence.media:
            kinds.setdefault(item.kind, []).append(item)
        limits = {"image": 32, "video": 1, "audio": 1}
        if any(kind not in limits or len(items) > limits[kind] for kind, items in kinds.items()):
            raise ValueError(f"{sequence.name}: at most 32 images, one video and one audio item")
        kwargs = {}
        for kind, keys in (("image", ("pixel_values", "image_grid_thw")),
                           ("video", ("video_pixel_values", "video_grid_thw"))):
            if kind in kinds:
                kwargs[keys[0]] = torch.cat([tensor(item.tensor, device) for item in kinds[kind]], dim=0)
                kwargs[keys[1]] = torch.cat([tensor(item.grid, device) for item in kinds[kind]], dim=0)
        if "audio" in kinds:
            kwargs["audio_codes"] = audio_codes(model, tensor(kinds["audio"][0].tensor, device), segment)
        return kwargs
    return inputs


def forward(model, ids, keep, media):
    """The last `keep` rows of logits of one sequence, without a cache."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    shape = torch.empty((1, ids.shape[1], 1), dtype=model.model.embed_tokens.weight.dtype, device=ids.device)
    masks = {"full_attention": create_causal_mask(model.config, shape, None, None),
             "sliding_window_attention": create_sliding_window_causal_mask(model.config, shape, None, None)}
    return model(input_ids=ids, attention_mask=masks, use_cache=False, logits_to_keep=keep, **media).logits[0]


def decode(*_args, **_kwargs):
    raise NotImplementedError("the MiMo reference scores forced answers only")


def taps(model):
    """The tap map (tools/refrun/taps.py names). Layer 0's MLP is dense; the
    MoE layers' gate returns (ids, weights). The vision tower runs once for
    the images and once for the video of a sequence, so its rows are the
    image rows, then the video rows."""
    names = {"embed.out": "model.embed_tokens", "enc.vision.out": Hook("visual", rows=("image", "video")),
             "enc.audio.out": Hook("audio_encoder", rows=("audio",))}
    for i, layer in enumerate(model.model.layers):
        stem, scope = f"model.layers.{i}", f"L{i:02d}"
        names |= {f"{scope}.in": Hook(stem, "input"), f"{scope}.attn.in": f"{stem}.input_layernorm",
                  f"{scope}.attn.core": Hook(f"{stem}.self_attn.o_proj", "input"),
                  f"{scope}.attn.out": f"{stem}.self_attn", f"{scope}.mlp.in": f"{stem}.post_attention_layernorm"}
        if hasattr(layer.mlp, "gate"):
            names |= {f"{scope}.router.ids": f"{stem}.mlp.gate",
                      f"{scope}.router.weights": Hook(f"{stem}.mlp.gate", index=1)}
        names |= {f"{scope}.mlp.out": f"{stem}.mlp", f"{scope}.out": stem}
    names["final.norm"] = "model.norm"
    return names
