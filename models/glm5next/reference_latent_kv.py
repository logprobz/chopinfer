"""GLM 5.3 Flash reference with a quantized latent KV: a format-study adapter.

The same HF `glm5_next` model, forward, decode, placement and taps as
reference.py (loaded from it, pinned by `REFERENCE_SHA256`), with every
sparse layer's normalized KV latent (`kv_a_layernorm`'s output, the row the
engine's latent cache stores and the only input of `kv_b_proj`) sent through
the round trip of one candidate latent KV format, `FORMATS[FORMAT]`, before
any key or value is expanded from it, the query row's own key included.

A format quantizes each block of `block` latent elements with one F32 scale:
`amax / max` (`exact`, 1 for an all-zero block) or the power of two at or
above it (`pow2`), where `max` is the element type's largest finite value
(E4M3 448, E5M2 57344, INT8 127). Codes are `x / scale` in F32, rounded to
nearest even and saturated to +-max; the reader's value is code * scale,
rounded once to the mode dtype. `e4m3-b128` is the served
`LatentKvFormat::Fp8E4M3Block128` (reference_latent_fp8.py's round trip).

It is a diagnostic, never a gate reference (KV1: is the served FP8 latent
format the most accurate one at its memory?). The caller picks the format
by setting the loaded module's `FORMAT` before `build`; run it with
`--no-cache`.
"""

import hashlib
import importlib.util
from dataclasses import dataclass
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

ELEMENTS = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0), "int8": (None, 127.0)}


@dataclass(frozen=True)
class LatentFormat:
    element: str
    block: int
    scale: str = "exact"

    def token_bytes(self, latent):
        """Cache bytes per token: one-byte codes and an F32 scale per block."""
        return latent + 4 * (latent // self.block)

    def describe(self):
        return (f"{self.element.upper()} codes, one F32 scale per {self.block} latent elements "
                f"({'max |x| / ' if self.scale == 'exact' else 'the power of two at or above max |x| / '}"
                f"{ELEMENTS[self.element][1]:g})")


FORMATS = {
    "e4m3-b128": LatentFormat("e4m3", 128),
    "e4m3-b32": LatentFormat("e4m3", 32),
    "e4m3-b512": LatentFormat("e4m3", 512),
    "e4m3-b128-pow2": LatentFormat("e4m3", 128, "pow2"),
    "e5m2-b128": LatentFormat("e5m2", 128),
    "int8-b128": LatentFormat("int8", 128),
    "int8-b64": LatentFormat("int8", 64),
    "int8-b32": LatentFormat("int8", 32),
}
# The served format; the caller sets another before build.
FORMAT = "e4m3-b128"


def round_trip(x, form):
    """The round trip of `x` [..., latent] through `form`, in x's dtype."""
    width = x.shape[-1]
    if width % form.block:
        raise ValueError(f"latent width {width} is not a multiple of {form.block}")
    dtype, top = ELEMENTS[form.element]
    blocks = x.float().reshape(*x.shape[:-1], width // form.block, form.block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # A tensor divisor: CUDA divides by a Python scalar as a product with its
    # rounded reciprocal, which is not the F32 division the writer rounds.
    ratio = amax / torch.full_like(amax, top)
    if form.scale == "pow2":
        # 2^ceil(log2(ratio)) from the exponent, exact: frexp gives
        # ratio = m 2^e with m in [0.5, 1), and m == 0.5 is already a power.
        mantissa, exponent = torch.frexp(ratio)
        ratio = torch.ldexp(torch.ones_like(ratio), torch.where(mantissa == 0.5, exponent - 1, exponent))
    scale = torch.where(amax == 0, torch.ones_like(amax), ratio)
    scaled = (blocks / scale).clamp(-top, top)
    codes = torch.round(scaled) if dtype is None else scaled.to(dtype).float()
    return (codes * scale).reshape(x.shape).to(x.dtype)


def build(config, placement):
    """reference.build, then the `FORMAT` round trip on every sparse layer's latent."""
    form = FORMATS[FORMAT]
    built = reference.build(config, placement)
    text = built.model.model.language_model

    def quantize_latent(_module, _inputs, output):
        return round_trip(output, form)
    for block in text.layers:
        if block.block_type != "linear_attention":
            block.self_attn.kv_a_layernorm.register_forward_hook(quantize_latent)
    built.replacements["kv_a_layernorm (sparse layers)"] = f"output through the {FORMAT} round trip: {form.describe()}"
    built.layout["kv_cache"] = f"{FORMAT} latent round trip (reference_latent_kv.py), " + built.layout["kv_cache"]
    return built
