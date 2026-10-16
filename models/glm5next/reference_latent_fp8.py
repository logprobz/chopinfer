"""GLM 5.3 Flash reference with the served FP8 latent KV: an ablation adapter
and the text layer check's floor.

The same HF `glm5_next` model, forward, decode, placement and taps as
reference.py (loaded from it, pinned by `REFERENCE_SHA256`), with one
change: every sparse layer's normalized KV latent (`kv_a_layernorm`'s
output, the row the engine's latent cache stores and the only input of
`kv_b_proj`) goes through the engine's FP8 latent KV format before any
key or value is expanded from it, the query row's own key included (the
engine writes a step's latent rows before its attention reads them).

The round trip sits at `kv_b_proj`'s input, the latent rows the attention
expands (reference.py's sparse attention reads the cached latent and
expands the keys each query block selects), not at `kv_a_layernorm`'s
output. The round trip acts on each token's row alone, so the values the
attention reads are the same either way, and the `x.latent` tap, which
reads `kv_a_layernorm`'s output, keeps the served tap's boundary: the
normalized latent before the FP8 write (bundle taps/text.rs reads it after
LatentNorm, before LatentKvWrite). Later layers still carry the upstream
FP8 effects.

The round trip is the served writer's (crates/kernels-sm120/cuda/latent/
kv_write.cuh, kernel-core's `LatentKvFormat::Fp8E4M3Block128`): per 128
latent elements of a token, scale = max |x| / 448 divided in F32 with
round to nearest (`__fdiv_rn`; 1 for an all-zero block), code =
E4M3(x / scale) with the same F32 division, rounded to nearest even and
saturated to +-448 (`cvt.rn.satfinite`), and the reader's value
E4M3(code) * scale in F32. `latent_fp8` reproduces exactly those codes,
scales and F32 values from F32 input on CPU and CUDA (`f32_div`).

What the arms do not reproduce of the served path:

- f32 arm: the writer quantizes the BF16 latent; here the F32 latent is
  quantized as it is, without rounding it to BF16 first, so its codes can
  differ from the served ones. The arm isolates FP8 storage over F32.
- bf16 arm: the latent is BF16, as the writer reads it, so its codes and
  scales are the writer's for HF's BF16 latent; but each decoded F32 value E4M3(code) * scale is rounded to
  BF16 before kv_b_proj expands it, where the engine applies the scale to
  F32 attention partials and never rounds the product.
- Everything else stays HF's: the naive (expanded) attention instead of
  the engine's absorbed form, BF16 attention scores in the bf16 arm, HF's
  weight rounding. The indexer's keys and the KDA layers are not touched.

So the arms measure the FP8 latent cache inside HF's math; matching them
does not prove that every served deviation comes from FP8 storage.

As a diagnostic, run it with `--no-cache` and compare its scores with the
adopted references (ACC1, the FP8 KV share of the served deviation):

    python3 -B -m tools.refrun run --adapter models/glm5next/reference_latent_fp8.py \\
        --checkpoint CKPT --lock models/glm5next/checkpoint-files.json --workload W.json \\
        --modes f32,bf16 --dequant f32=rounded --output OUT --no-cache

Its bf16 arm is also the floor of qual.json's `text_layers` check
(`ADAPTERS`, README.md "Layer reference"): the served FP8 latent's error
is measured against reference.py's plain F32 run, and its floor is this
arm's distance from that run, so the layer reference holds one run of each
(`f32/` with reference.py, `bf16/` with this adapter, `--modes bf16`).
A gate reference's keys name this file's digest: changing it stales that
reference.
"""

import hashlib
import importlib.util
from pathlib import Path

import torch

REFERENCE = Path(__file__).with_name("reference.py")
# reference.py is executed code of this adapter, outside the runner's adapter
# files: a change to it must also change this file.
REFERENCE_SHA256 = "fa0736511316277b0f29fba208e1d85b3d54291a735f9ca0433ca799e41bdea3"
if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
    raise RuntimeError(f"{REFERENCE.name} is not the pinned {REFERENCE_SHA256}; review and repin")
_spec = importlib.util.spec_from_file_location("refrun_adapter_glm5next_reference", REFERENCE)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

MODEL = reference.MODEL
units = reference.units
reference_code = reference.reference_code
# kernel-core latent_attention::LATENT_FP8_SCALE_BLOCK, and E4M3's largest finite value.
SCALE_BLOCK = 128
E4M3_MAX = 448.0


def f32_div(numerator, denominator):
    """numerator / denominator rounded once to F32 (`__fdiv_rn`), from F32
    tensors on any device: an f64 tensor-by-tensor quotient rounded to F32.
    The f64 quotient of two F32 values is correctly rounded and 53 >= 2 * 24
    + 2 bits, so rounding it to F32 is the correctly rounded F32 quotient.
    Dividing by a Python scalar would not do: PyTorch's CUDA kernel
    multiplies by the scalar's rounded reciprocal instead."""
    return (numerator.double() / denominator.double()).float()


def latent_fp8(x):
    """The engine's FP8 latent round trip of `x` [..., latent], in x's dtype
    (crates/kernels-sm120/cuda/latent/kv_write.cuh): per block, scale =
    __fdiv_rn(amax, 448) (1 for an all-zero block), code =
    E4M3_satfinite(__fdiv_rn(x, scale)), value = E4M3(code) * scale in F32."""
    width = x.shape[-1]
    if width % SCALE_BLOCK:
        raise ValueError(f"latent width {width} is not a multiple of {SCALE_BLOCK}")
    blocks = x.float().reshape(*x.shape[:-1], width // SCALE_BLOCK, SCALE_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), f32_div(amax, torch.full_like(amax, E4M3_MAX)))
    codes = f32_div(blocks, scale.expand_as(blocks)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(x.shape).to(x.dtype)


def quantize_expanded(_module, args):
    """kv_b_proj's input, the latent rows the attention expands, through the
    round trip."""
    (latent,) = args
    return (latent_fp8(latent),)


def build(config, placement):
    """reference.build, then the FP8 round trip on the latent every sparse
    layer's kv_b_proj expands."""
    built = reference.build(config, placement)
    text = built.model.model.language_model
    for block in text.layers:
        if block.block_type != "linear_attention":
            block.self_attn.kv_b_proj.register_forward_pre_hook(quantize_expanded)
    built.replacements["kv_b_proj input (sparse layers)"] = (
        "the cached normalized latent through the FP8 E4M3 latent KV round trip, one F32 scale per 128 "
        "(max |x| / 448); kv_a_layernorm's output (the x.latent tap) is the unencoded latent")
    built.layout["kv_cache"] = "FP8 E4M3 latent round trip (reference_latent_fp8.py), " + built.layout["kv_cache"]
    return built
