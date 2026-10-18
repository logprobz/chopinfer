"""Qwen3.8 reference adapter for ModelOpt mixed-precision exports.

The same HF `qwen4_exp` model, forward, decode, placement and taps as
reference.py (loaded from it, pinned by `REFERENCE_SHA256`), around an
export whose `config.json` declares `quantization_config` with
`quant_method: modelopt`, `quant_algo: MIXED_PRECISION` and a
`quantized_layers` entry for every quantized module. Each declared module
of the text model must hold exactly the tensors its algorithm stores:

- `NVFP4` and `W4A16_NVFP4` (group 16): packed E2M1 codes `weight` (even K
  in the low nibble), one E4M3 `weight_scale` per 16 K and an F32
  `weight_scale_2`. The weight is code x (E4M3 x weight_scale_2), as
  ModelOpt's NVFP4 dequantization and the engine's decode define it. NVFP4
  (W4A4) also stores an F32 `input_scale` for activation quantization; the
  reference does not apply it: activations stay in the mode dtype, as the
  engine serves these exports with BF16 activation operands (A16).
- `MXFP8` (group 32): E4M3 codes `weight` with one UE8M0 `weight_scale`
  byte per 32 K: code x 2^(byte - 127), as HF's `Fp8Dequantize` decodes
  MXFP8.
- `FP8` (the n-gram table only): E4M3 rows times the per-tensor
  `weight_scale`, read on the host as reference.py does.

Routed experts declared NVFP4 run HF's eager expert loop over `QuantStack`s
as reference.py's FP8 experts do; NVFP4 n-gram shards (`shard_N.weight` and
`shard_N.weight_scale`, one F32 `weight_scale_2` for the table) are gathered
by row on the host; other declared projections are `QuantLinear`s.

Coverage is exact: every stored tensor is consumed by name (a declared
module's codes and scales, a host table shard, or a parameter or buffer of
the HF model), or is an MTP source (`mtp.*`, stored and declared but not run
by a teacher-forced reference; the run's layout counts them). A tensor of no
executed module, a declared module without the tensors its algorithm stores,
an undeclared expert layer, n-gram shards other than exactly shard_0 to
shard_(n-1), another algorithm, an NVFP4 global scale that is not positive
and finite, or an E4M3 block scale that is negative or NaN (including the
rows a host table gathers) refuses to build or to score.

Dequantization rules (tools/refrun/quant.py): `exact` is the exact product
(f64); E2M1 x E4M3 x an F32 global scale can need 30 significant bits, so
F32 uses `rounded`, the exact product rounded once (`--dequant f32=rounded`);
`hf-bf16` forms the F32 product (E4M3 scale times global first, as ModelOpt
does) and rounds it to BF16 for the bf16 spread mode.

Stored F32 parameters (the hyper-connection and PLE norm weights of an
mxfp8-dense-nvfp4 export) never narrow: they stay F32 in the bf16 mode and
widen exactly in f64. HF's `Qwen4ExpTextRMSNorm` computes
`1 + weight.float()` in F32, so every mode reads the stored values; HF's own
BF16 loading would round them to BF16.
Stored BF16 tensors are cast to the mode dtype as in reference.py.
"""

import hashlib
import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn

from tools.refrun import adapter, quant
from tools.refrun.place import HostRows

REFERENCE = Path(__file__).with_name("reference.py")
# reference.py is executed code of this adapter, outside the runner's adapter
# files: a change to it must also change this file, so bound mixed references
# go stale with it.
REFERENCE_SHA256 = "b09e38ac51f59fee3d76a4ef3ff00c11a5c9407a20a0e1a3c861df70f42ba114"
if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
    raise RuntimeError(f"{REFERENCE.name} is not the pinned {REFERENCE_SHA256}; review and repin")
_spec = importlib.util.spec_from_file_location("refrun_adapter_qwen38_reference", REFERENCE)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

MODEL = reference.MODEL
TEXT = reference.TEXT
units = reference.units
MXFP8 = quant.Format("e4m3", "e8m0", 1, 32)
# Declared algorithm -> group size along K (None: a per-tensor scale).
ALGORITHMS = {"NVFP4": 16, "W4A16_NVFP4": 16, "MXFP8": 32, "FP8": None}
NVFP4 = ("NVFP4", "W4A16_NVFP4")
# The MTP layers' tensors: stored, declared, and not run by a teacher-forced reference.
MTP = "mtp."


def reference_code(checkpoint):
    return {**reference.reference_code(checkpoint), "models/qwen38/reference.py": REFERENCE}


def declaration(config):
    """{module: algorithm} of the text model's declared quantized modules,
    and {module: algorithm} of declared modules this reference does not run
    (MTP). Refuses another export recipe or algorithm."""
    recipe = config.get("quantization_config") or {}
    if recipe.get("quant_method") != "modelopt" or recipe.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("not a ModelOpt MIXED_PRECISION export; block-FP8 checkpoints use reference.py")
    if recipe.get("kv_cache_quant_algo") is not None:
        raise ValueError(f"KV cache quantization {recipe['kv_cache_quant_algo']} is not supported")
    layers = recipe.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("quantization_config declares no quantized_layers")
    run, skipped = {}, {}
    for module, spec in sorted(layers.items()):
        algorithm = spec.get("quant_algo")
        if module.startswith(MTP):
            skipped[module] = algorithm
            continue
        if not module.startswith(TEXT + "."):
            raise ValueError(f"{module}: {algorithm} outside the text model and its MTP layers is not supported")
        if algorithm not in ALGORITHMS or spec.get("group_size") != ALGORITHMS[algorithm]:
            raise ValueError(f"{module}: unsupported {algorithm} with group size {spec.get('group_size')}")
        run[module] = algorithm
    return run, skipped


class Tensors:
    """The stored tensors of the placed units, checked against the formats
    the declaration names. `used` collects every stored name consumed by a
    declared module."""

    def __init__(self, checkpoint, store):
        self.entries, self.store, self.used = checkpoint.entries, store, set()

    def expect(self, name, dtype, shapes, algorithm, host=False):
        """`name` as the algorithm stores it, loaded on its unit's device
        unless it is a `host` table."""
        entry = self.entries.get(name)
        loaded = host or name in self.store.tensors
        if entry is None or not loaded or entry.dtype != dtype or list(entry.shape) not in shapes:
            found = "nothing" if entry is None else f"{entry.dtype} {list(entry.shape)}"
            raise ValueError(f"{name}: {algorithm} stores {dtype} {shapes[0]}, the export has {found}")
        self.used.add(name)
        return name

    def nvfp4(self, prefix, rows, cols, algorithm):
        """Names of an NVFP4 [rows, cols] weight; NVFP4 (W4A4) also has an input_scale."""
        names = (self.expect(prefix + "weight", "U8", [[rows, cols // 2]], algorithm),
                 self.expect(prefix + "weight_scale", "F8_E4M3", [[rows, cols // 16]], algorithm),
                 self.expect(prefix + "weight_scale_2", "F32", [[], [1]], algorithm))
        extra = prefix + "input_scale"
        if algorithm == "NVFP4":
            self.expect(extra, "F32", [[], [1]], algorithm)
        elif extra in self.entries:
            raise ValueError(f"{extra}: {algorithm} is weight-only")
        return names

    def mxfp8(self, prefix, rows, cols):
        return (self.expect(prefix + "weight", "F8_E4M3", [[rows, cols]], "MXFP8"),
                self.expect(prefix + "weight_scale", "U8", [[rows, cols // 32]], "MXFP8"))

    def global_scales(self, names):
        """The F32 weight_scale_2 values (one device read for all of them),
        each positive and finite as an NVFP4 global must be."""
        values = torch.stack([self.store.get(name).reshape(()) for name in names]).cpu().tolist()
        for name, value in zip(names, values):
            if not 0 < value < math.inf:
                raise ValueError(f"{name}: NVFP4 global scale {value} is not positive and finite")
        return values

    def local_scales(self, names):
        """Refuses an NVFP4 E4M3 block scale with its sign bit set or a NaN
        code (one device read for all of `names`)."""
        codes = torch.cat([self.store.get(name).view(torch.uint8).reshape(-1) for name in names])
        if bool(invalid_e4m3_scales(codes).any()):
            bad = next(name for name in names if bool(invalid_e4m3_scales(self.store.get(name).view(torch.uint8)).any()))
            raise ValueError(f"{bad}: an E4M3 block scale is negative or NaN")


def invalid_e4m3_scales(codes):
    """Bytes of E4M3 scales that are negative (sign bit) or NaN (S.1111.111)."""
    return ((codes & 0x80) != 0) | ((codes & 0x7F) == 0x7F)


class NVFP4Rows(nn.Module):
    """Stands in for the n-gram nn.Embedding of an NVFP4 table: reads the
    hashed rows' codes and E4M3 scales from the host shards and dequantizes
    them (times the table's F32 global scale) under the mode's rule, on the
    PLE layer's device."""

    def __init__(self, checkpoint, shards, global_scale, padded_rows, cols, dtype, rule, device):
        super().__init__()
        self.codes = [HostRows(checkpoint, name + ".weight") for name in shards]
        self.scales = [HostRows(checkpoint, name + ".weight_scale") for name in shards]
        rows = {table.entry.shape[0] for table in self.codes + self.scales}
        if len(rows) != 1 or rows.pop() * len(shards) != padded_rows:
            raise ValueError(f"{len(shards)} NVFP4 n-gram shards are not HF's {padded_rows}-row table")
        self.rows_per_shard, self.cols = self.codes[0].entry.shape[0], cols
        self.format, self.dtype, self.rule = quant.nvfp4(global_scale), dtype, rule
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=device), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        by_shard = {}
        for position, row in enumerate(flat):
            shard, offset = divmod(row, self.rows_per_shard)
            by_shard.setdefault(shard, ([], []))
            by_shard[shard][0].append(position)
            by_shard[shard][1].append(offset)
        codes = torch.empty((len(flat), self.cols // 2), dtype=torch.uint8)
        scales = torch.empty((len(flat), self.cols // 16), dtype=torch.uint8)

        def read(item):
            shard, (_, offsets) = item
            return self.codes[shard].rows(offsets), self.scales[shard].rows(offsets)
        with ThreadPoolExecutor(reference.NGRAM_READERS) as pool:
            for (positions, _), (code_rows, scale_rows) in zip(by_shard.values(), pool.map(read, by_shard.items())):
                codes[positions] = code_rows.view(torch.uint8)
                scales[positions] = scale_rows.view(torch.uint8)
        if bool(invalid_e4m3_scales(scales).any()):
            raise ValueError("an n-gram row's E4M3 block scale is negative or NaN")
        device = self.weight.device
        values = quant.dequantize(self.format, codes.to(device), scales.to(device), (len(flat), self.cols),
                                  self.dtype, self.rule)
        return values.view(*ids.shape, self.cols)


def stored_dtype(tensor, dtype):
    """The mode dtype, except that a stored F32 tensor never narrows."""
    return tensor.dtype if tensor.dtype == torch.float32 and dtype == torch.bfloat16 else dtype


def build(config, placement):
    """The HF model in `config.dtype` around the store's weights."""
    from transformers import AttentionInterface
    configuration, hf = reference.hf_modules()
    reference.check_kernels()
    store, dtype, rule, checkpoint = config.store, config.dtype, config.policy["dequant"], config.checkpoint
    declared, skipped = declaration(checkpoint.config)
    # Every stored tensor is read by an executed module or is an MTP source.
    executed = {n for unit in units(checkpoint) for n in unit.tensors}
    unexecuted = sorted(set(checkpoint.entries) - executed)
    stray = [n for n in unexecuted if not n.startswith(MTP)]
    if stray:
        raise ValueError(f"{stray[0]}: a stored tensor outside every executed module that is not an MTP source")
    hf_config = configuration.Qwen4ExpConfig.from_dict(dict(checkpoint.config))
    text_config = hf_config.text_config
    AttentionInterface.register(reference.VISION_ATTENTION, reference.vision_attention(placement.devices))
    reference.implementations([text_config], [hf_config.vision_config])
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
    reference.implementations([text_config, *configs.values()], [hf_config.vision_config, *vision.values()])
    devices = dict(placement.units)

    def device_of(path):
        while path not in devices:
            path = path.rpartition(".")[0]
        return torch.device(devices[path])
    stored, names = Tensors(checkpoint, store), {}
    text = model.model.language_model
    tables = {reference.ngram_prefix(layer) for layer in reference.ple_layers(checkpoint.config["text_config"])}
    experts = {f"{TEXT}.layers.{layer}.mlp.experts" for layer in range(text_config.num_hidden_layers)}
    linears = {}
    for module, algorithm in declared.items():
        try:
            target = model.get_submodule(module)
        except AttributeError:
            raise ValueError(f"{module}: declared quantized, but the HF model has no such module") from None
        if module in experts:
            if algorithm not in NVFP4:
                raise ValueError(f"{module}: {algorithm} experts are not supported")
        elif module + "." in tables:
            continue
        elif algorithm in NVFP4 + ("MXFP8",) and isinstance(target, nn.Linear):
            linears[module] = algorithm
        else:
            raise ValueError(f"{module}: {algorithm} is not declared on a linear projection, the experts or the "
                             "n-gram table")
    undeclared = sorted(experts - set(declared))
    if undeclared:
        raise ValueError(f"{undeclared[0]}: experts the export does not declare quantized")

    # Declared projections: QuantLinears over the stored codes and scales.
    for module, algorithm in sorted(linears.items()):
        linear = model.get_submodule(module)
        rows, cols = linear.out_features, linear.in_features
        if linear.bias is not None:
            raise ValueError(f"{module}: a quantized projection with a bias")
        if algorithm == "MXFP8":
            codes, scale = stored.mxfp8(module + ".", rows, cols)
            fmt = MXFP8
        else:
            codes, scale, second = stored.nvfp4(module + ".", rows, cols, algorithm)
            stored.local_scales([scale])
            fmt = quant.nvfp4(stored.global_scales([second])[0])
            names[second] = module
            if algorithm == "NVFP4":
                names[module + ".input_scale"] = module
        parent, _, child = module.rpartition(".")
        setattr(model.get_submodule(parent), child, store.linear(fmt, codes, scale, (rows, cols)))
        names[codes] = names[scale] = module

    # Routed experts: NVFP4 QuantTensors per expert, gate and up stacked as HF's gate_up_proj.
    inter, hidden = text_config.moe_intermediate_size, text_config.hidden_size
    for layer, block in enumerate(text.layers):
        base = f"{TEXT}.layers.{layer}.mlp.experts"
        algorithm = declared[base]
        shapes = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
        found = {(expert, projection): stored.nvfp4(f"{base}.{expert}.{projection}.", *shape, algorithm)
                 for expert in range(text_config.num_experts) for projection, shape in shapes.items()}
        scales = dict(zip(found, stored.global_scales([value[2] for value in found.values()])))
        stored.local_scales([value[1] for value in found.values()])
        stacks = {"gate_up": [], "down": []}
        for expert in range(text_config.num_experts):
            tensors = {}
            for projection, path in (("gate_proj", f"gate_up_proj.items.{expert}.gate"),
                                     ("up_proj", f"gate_up_proj.items.{expert}.up"),
                                     ("down_proj", f"down_proj.items.{expert}")):
                codes, scale, second = found[(expert, projection)]
                slot = projection.removesuffix("_proj")
                tensors[projection] = store.tensor(quant.nvfp4(scales[(expert, projection)]), codes, scale,
                                                   shapes[projection], slot)
                prefix = f"{base}.{expert}.{projection}."
                for suffix in ("weight", "weight_scale", "weight_scale_2") + (("input_scale",) if algorithm == "NVFP4"
                                                                                else ()):
                    names[prefix + suffix] = f"{base}.{path}"
            stacks["gate_up"].append(reference.GateUp(tensors["gate_proj"], tensors["up_proj"]))
            stacks["down"].append(tensors["down_proj"])
        experts_module = block.mlp.experts
        del experts_module._parameters["gate_up_proj"], experts_module._parameters["down_proj"]
        experts_module.gate_up_proj = quant.QuantStack(stacks["gate_up"])
        experts_module.down_proj = quant.QuantStack(stacks["down"])

    # The n-gram tables: host rows, FP8 per-tensor as reference.py or NVFP4.
    ngram = {}
    for layer, block in enumerate(text.layers):
        if block.ple is None:
            continue
        prefix = reference.ngram_prefix(layer)
        module = prefix.rstrip(".")
        algorithm = declared.get(module)
        embedding = block.ple.ple_embedding
        divisor = text_config.make_ngram_vocab_size_divisible_by
        padded = math.ceil(embedding.total_vocab_size / divisor) * divisor
        cols = embedding.ngram_embedding.embedding_dim
        # Row r is in shard r // rows: the shards must be exactly shard_0 .. shard_(n-1).
        shards = [f"shard_{index}" for index in range(text_config.split_ngram_parts)]
        found = {n[len(prefix):].split(".")[0] for n in checkpoint.entries if n.startswith(prefix + "shard_")}
        if found != set(shards):
            raise ValueError(f"layer {layer}: n-gram shards are not shard_0 to shard_{len(shards) - 1}: missing "
                             f"{sorted(set(shards) - found)[:3]}, unexpected {sorted(found - set(shards))[:3]}")
        rows = padded // len(shards)
        if algorithm == "FP8":
            for shard in shards:
                stored.expect(f"{prefix}{shard}.weight", "F8_E4M3", [[rows, cols]], algorithm, host=True)
            scale = prefix + "weight_scale"
            if scale not in store.tensors or store.get(scale).numel() != 1:
                raise ValueError(f"{scale}: FP8 stores one per-tensor scale")
            stored.used.add(scale)
            embedding.ngram_embedding = reference.NGramRows(
                checkpoint, [f"{prefix}{shard}.weight" for shard in shards], store.get(scale), padded, dtype, rule,
                device_of(module))
            names[scale] = module
            names |= {f"{prefix}{shard}.weight": "host:ngram_rows" for shard in shards}
        elif algorithm in NVFP4:
            for shard in shards:
                stored.expect(f"{prefix}{shard}.weight", "U8", [[rows, cols // 2]], algorithm, host=True)
                stored.expect(f"{prefix}{shard}.weight_scale", "F8_E4M3", [[rows, cols // 16]], algorithm, host=True)
            second = stored.expect(prefix + "weight_scale_2", "F32", [[], [1]], algorithm)
            embedding.ngram_embedding = NVFP4Rows(
                checkpoint, [prefix + shard for shard in shards], stored.global_scales([second])[0], padded, cols,
                dtype, rule, device_of(module))
            names[second] = module
            names |= {f"{prefix}{shard}.{name}": "host:ngram_rows" for shard in shards
                      for name in ("weight", "weight_scale")}
        else:
            raise ValueError(f"{module}: the export declares {algorithm} for the n-gram table, not FP8 or NVFP4")
        ngram[module] = algorithm

    # Every host table member is a consumed shard tensor.
    host = sorted({n for unit in units(checkpoint) if unit.host for n in unit.tensors} - stored.used)
    if host:
        raise ValueError(f"{host[0]}: a stored n-gram table tensor the table does not read")
    # Everything else is stored BF16 or F32 (I64 buffers kept) and must have a
    # place in the HF model: only consumed names are skipped.
    state, kept = {}, []
    parameters = {n for n, _ in model.named_parameters()}
    for name, tensor in store.tensors.items():
        if name in stored.used:
            continue
        entry = checkpoint.entries[name]
        if entry.dtype not in ("BF16", "F32", "I64"):
            raise ValueError(f"{name}: stored {entry.dtype}, but the export declares no quantized module for it")
        target = stored_dtype(tensor, dtype)
        state[name] = tensor.to(target) if name in parameters and tensor.is_floating_point() else tensor
        if entry.dtype == "F32":
            kept.append(name)
        names[name] = name
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"stored tensors with no place in the HF model: {unexpected[:5]}")
    text.rotary_emb = hf.Qwen4ExpTextRotaryEmbedding(text_config).to(device_of(f"{TEXT}.rotary_emb"))
    visual, vision_config = model.model.visual, hf_config.vision_config
    visual.rotary_pos_emb = hf.Qwen4ExpVisionRotaryEmbedding(
        vision_config.hidden_size // vision_config.num_heads // 2).to(device_of("model.visual"))
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{len(left)} tensors not loaded, e.g. {left[:5]}")
    model.eval()
    return adapter.Reference(
        model, names, layout=layout(declared, skipped, len(unexecuted), ngram, kept, rule), media_inputs=reference.media_inputs,
        taps=reference.taps, forward=reference.forward, decode=reference.decode,
        replacements={
            "mlp.experts.gate_up_proj, mlp.experts.down_proj": "QuantStack of NVFP4 QuantTensors, one expert "
                                                               "dequantized per call",
            "declared projections": "QuantLinear over the stored codes and scales, dequantized per call",
            "ple.ple_embedding.ngram_embedding": "host row gather (place.HostRows): FP8 rows times the per-tensor "
                                                 "scale, or NVFP4 rows and scales times the global scale",
            "vision attention": f"query chunks of {reference.VISION_CHUNK} over all keys of an image, round robin "
                                "over the GPUs: HF sdpa_attention_forward in f32 and bf16, ops.attention in f64",
            "Qwen4ExpModel.forward": f"the same steps, text model in chunks of {reference.PREFILL_CHUNK} tokens "
                                     "with a DynamicCache"})


def layout(declared, skipped, unexecuted, ngram, kept, rule):
    counts = {}
    for module, algorithm in declared.items():
        role = module.split(".layers.", 1)[-1].split(".", 1)[-1]
        counts.setdefault(algorithm, {}).setdefault(role, 0)
        counts[algorithm][role] += 1
    return {"export": "ModelOpt MIXED_PRECISION; quantization_config.quantized_layers by algorithm and role",
            "declared": counts, "not_run": {"declared": dict(sorted(skipped.items())), "mtp_tensors": unexecuted},
            "nvfp4": "packed E2M1 (even K low nibble) x (E4M3 weight_scale per 16 K x F32 weight_scale_2); "
                     "input_scale (W4A4 activation calibration) not applied",
            "mxfp8": "E4M3 x 2^(UE8M0 weight_scale - 127) per 32 K",
            "experts": "per expert [I, H] gate and up, [H, I] down; gate and up stacked as gate_up_proj",
            "ngram_embedding": {module: algorithm for module, algorithm in sorted(ngram.items())},
            "f32_parameters": {"count": len(kept), "rule": "stored F32 kept F32 in bf16, widened exactly in f64"},
            "dequant": rule, "prefill_chunk": reference.PREFILL_CHUNK, "vision_chunk": reference.VISION_CHUNK,
            "upstream_f32": reference.UPSTREAM_F32}
