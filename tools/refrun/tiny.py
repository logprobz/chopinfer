"""A synthetic adapter: a tiny random HF Llama with quantized MLP weights.

`write_checkpoint` writes random weights for a small config: BF16 attention,
norms, embeddings and head; per layer an FP8 `gate_proj` on a chunked 20 x 16
block grid (partial tail blocks in every chunk), an MXFP4 `up_proj` and a
BF16 `down_proj`, all three served through QuantLinear. The unit tests and
the calibration job run it through the runner; it needs no real checkpoint.
"""

import hashlib
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from tools.refrun import adapter, quant, taps as tap_names
from tools.refrun.place import Unit

MODEL = "tiny"
CONFIG = {"vocab_size": 96, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
          "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 256,
          "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "attention_bias": False,
          "mlp_bias": False, "architectures": ["LlamaForCausalLM"], "model_type": "llama"}
GATE = quant.fp8_block(20, 16, chunk_rows=48)
LOCK = "checkpoint-files.json"


def random_codes(generator, shape, avoid=()):
    codes = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    for code in avoid:
        codes[codes == code] = 0x38
    return codes


def write_checkpoint(root, seed=0):
    """Writes config.json, model.safetensors and a source lock under `root`."""
    from safetensors.torch import save_file
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    config = LlamaConfig(**CONFIG)
    hidden, inter = config.hidden_size, config.intermediate_size
    with torch.device("meta"):
        shapes = {n: p.shape for n, p in LlamaForCausalLM(config).named_parameters()}
    tensors = {}
    for name, shape in shapes.items():
        if ".mlp." in name:
            continue
        scale = 1.0 if "norm" in name else 0.2
        tensors[name] = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        tensors[prefix + "gate_proj.weight"] = random_codes(generator, (inter, hidden), (0x7F, 0xFF))
        grid = GATE.scale_shape(inter, hidden)
        tensors[prefix + "gate_proj.weight_scale_inv"] = (
            torch.rand(grid, generator=generator) * 0.002 + 0.0005).to(torch.bfloat16)
        tensors[prefix + "up_proj.weight"] = random_codes(generator, (inter, hidden // 2))
        tensors[prefix + "up_proj.weight_scale"] = torch.randint(
            118, 124, quant.MXFP4.scale_shape(inter, hidden), generator=generator, dtype=torch.uint8)
        tensors[prefix + "down_proj.weight"] = (
            torch.randn((hidden, inter), generator=generator) * 0.1).to(torch.bfloat16)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(config.to_json_string())
    lock = {}
    for file in ("config.json", "model.safetensors"):
        data = (root / file).read_bytes()
        lock[file] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / LOCK).write_text(json.dumps(lock, indent=2, sort_keys=True))
    return root


def reference_code(_checkpoint):
    return {"transformers/models/llama/modeling_llama.py": Path(modeling_llama.__file__)}


def units(checkpoint):
    """Embeddings, rotary tables, each decoder layer, final norm, head."""
    layers = checkpoint.config["num_hidden_layers"]
    names = sorted(checkpoint.entries)

    def owned(prefix):
        return tuple(n for n in names if n.startswith(prefix))
    return [Unit("model.embed_tokens", owned("model.embed_tokens.")), Unit("model.rotary_emb", ()),
            *(Unit(f"model.layers.{i}", owned(f"model.layers.{i}.")) for i in range(layers)),
            Unit("model.norm", owned("model.norm.")), Unit("lm_head", owned("lm_head."))]


def mlp_weights(store, layer, shape):
    """The layer's gate, up and down QuantLinears (shared by every mode)."""
    prefix = f"model.layers.{layer}.mlp."
    inter, hidden = shape
    return {"gate_proj": store.linear(GATE, prefix + "gate_proj.weight",
                                      prefix + "gate_proj.weight_scale_inv", (inter, hidden)),
            "up_proj": store.linear(quant.MXFP4, prefix + "up_proj.weight",
                                    prefix + "up_proj.weight_scale", (inter, hidden)),
            "down_proj": store.linear(quant.BF16, prefix + "down_proj.weight", None, (hidden, inter))}


def taps(model):
    """The tap name map: canonical sites of every layer, embed and final."""
    names = {"embed.out": "model.embed_tokens", "final.norm": "model.norm"}
    for layer in range(len(model.model.layers)):
        path, scope = f"model.layers.{layer}", f"L{layer:02d}"
        names |= {f"{scope}.in": tap_names.Hook(path, "input"), f"{scope}.attn.in": f"{path}.input_layernorm",
                  f"{scope}.attn.q": f"{path}.self_attn.q_proj",
                  f"{scope}.attn.core": tap_names.Hook(f"{path}.self_attn.o_proj", "input"),
                  f"{scope}.attn.out": f"{path}.self_attn", f"{scope}.mlp.in": f"{path}.post_attention_layernorm",
                  f"{scope}.mlp.out": f"{path}.mlp", f"{scope}.out": path}
    return names


def build(config, placement):
    """The HF Llama in `config.dtype`: BF16 tensors cast from the store,
    MLP linears replaced by the store's QuantLinears."""
    hf_config = LlamaConfig(**config.checkpoint.config)
    hf_config._attn_implementation = "eager"
    default = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    try:
        model = LlamaForCausalLM(hf_config).eval()
    finally:
        torch.set_default_dtype(default)
    names = {}
    for unit, device in placement.units:
        model.get_submodule(unit).to(device)
    state = {}
    for name in config.store.tensors:
        if ".mlp." not in name:
            state[name] = config.store.get(name).to(config.dtype)
            names[name] = name
    model.load_state_dict(state, strict=False, assign=True)
    shape = (hf_config.intermediate_size, hf_config.hidden_size)
    for layer, block in enumerate(model.model.layers):
        for projection, linear in mlp_weights(config.store, layer, shape).items():
            setattr(block.mlp, projection, linear)
            path = f"model.layers.{layer}.mlp.{projection}"
            names[path + ".weight"] = path
            if linear.fmt.element != "bf16":
                names[path + (".weight_scale_inv" if projection == "gate_proj" else ".weight_scale")] = path
    return adapter.Reference(model, names, layout={"gate_proj": "fp8 20x16 blocks, scale grid restarts every 48 rows"},
                             replacements={"mlp.*_proj": "QuantLinear"}, taps=taps)
