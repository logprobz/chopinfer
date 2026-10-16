"""reference.py with the vendor's merger normalization: `visual.merger.ln_q`
as a LayerNorm without bias over the stored weight, everything else
unchanged.

The vendor's LayerNorm is a known upstream discrepancy; GPU probes kept
reference.py's RMSNorm as the default (reference.MERGERS). Running both
shows how much the media references depend on the choice. Its key names
this file, and reference.py as executed code.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("refrun_adapter_mimo26_reference",
                                               Path(__file__).with_name("reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

MODEL = reference.MODEL
EXTRA_SHARDS = reference.EXTRA_SHARDS
ENCODERS = reference.ENCODERS
units = reference.units


def reference_code(checkpoint):
    return {**reference.reference_code(checkpoint), "models/mimo26/reference.py": Path(reference.__file__)}


def build(config, placement):
    return reference.build(config, placement, merger="layernorm")


def build_encoders(config, placement):
    return reference.build_encoders(config, placement, merger="layernorm")
