"""Negative control for reference.py: the stored TP-chunked QKV rows read as
the vendor's contiguous Q | K | V split, everything else unchanged.

Its scores must be far from the reference (mean |dlogprob| > 5 on the
teacher texts); otherwise the TP-chunk layout claim is unproven. Never a
reference: its key names this file, and reference.py as executed code.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("refrun_adapter_mimo26_reference",
                                               Path(__file__).with_name("reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

MODEL = reference.MODEL
EXTRA_SHARDS = reference.EXTRA_SHARDS
units = reference.units


def reference_code(checkpoint):
    return {**reference.reference_code(checkpoint), "models/mimo26/reference.py": Path(reference.__file__)}


def build(config, placement):
    return reference.build(config, placement, qkv="contiguous")
