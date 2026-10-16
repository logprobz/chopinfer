"""Qwen3.8's pinned HF reference with 1024-token prefill chunks.

The 128K workload keeps the same eager attention, F32 mode and exact FP8
weight dequantization. Smaller query chunks bound the attention score
matrices. This producer has its own cache identity; existing references
continue to use reference.py and its unchanged chunking.
"""

import hashlib
import importlib.util
from pathlib import Path

REFERENCE = Path(__file__).with_name("reference.py")
REFERENCE_SHA256 = "b09e38ac51f59fee3d76a4ef3ff00c11a5c9407a20a0e1a3c861df70f42ba114"
if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
    raise RuntimeError(f"{REFERENCE.name} is not the pinned {REFERENCE_SHA256}; review and repin")
_spec = importlib.util.spec_from_file_location("refrun_adapter_long_context_base", REFERENCE)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

PREFILL_CHUNK = 1024
reference.PREFILL_CHUNK = PREFILL_CHUNK
MODEL = reference.MODEL
units = reference.units
build = reference.build


def reference_code(checkpoint):
    return {**reference.reference_code(checkpoint), "models/qwen38/reference.py": REFERENCE}
