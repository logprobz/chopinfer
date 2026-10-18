"""DeepSeek V4.1 Flash pure-F32 reference adapter (tools/refrun).

reference.py's adapter (loaded from it, pinned by `REFERENCE_SHA256`) with
one change: in the f32 mode the quantizers read the F32 activations as they
are instead of rounded to BF16 (reference_kernel.POLICY "mode"). It runs only
the f32 mode; bf16 inputs are the same under both policies, and f64 has no
untrained-input rule. Its keys differ from reference.py's by the adapter
file, so the pure-F32 and F32-TQ references never share a cache entry.
"""

import hashlib
import importlib.util
from pathlib import Path

import torch

REFERENCE = Path(__file__).resolve().with_name("reference.py")
REFERENCE_SHA256 = "7e2d7f82e73bdc493b9055c334ef1f56f5e82a648d5d8cab1d3108d8f42724c2"


def load_reference():
    data = REFERENCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError(f"reference.py is not the pinned {REFERENCE_SHA256}; update REFERENCE_SHA256")
    spec = importlib.util.spec_from_file_location("refrun_deepseek41_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_reference()
MODEL = base.MODEL
reference_code = base.reference_code
units = base.units
taps = base.taps


def build(config, placement, **options):
    if config.dtype != torch.float32:
        raise ValueError(f"the pure-F32 reference runs the f32 mode only, not {config.dtype}")
    return base.build(config, placement, quantizer_inputs="mode", **options)
