"""The adapter interface between the runner and a model's upstream code.

A model adapter, models/<m>/reference.py, is a module with:

- `MODEL`: the model name used in keys ("qwen38", "mimo26").
- `reference_code(checkpoint) -> {label: Path}`: the upstream source files
  the reference executes (vendor modeling files, the transformers modeling
  module). Their sha256 values go into the cache key.
- `units(checkpoint) -> [place.Unit]`: placed modules in model order, each
  with the checkpoint tensors it owns; `host=True` units stay on the host.
- `build(config, placement) -> Reference`: the upstream model for one mode,
  built in `config.dtype` around the shared stored weights.

The runner loads every non-host unit's tensors once (`config.store`), then
builds, runs and drops one model per mode.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from tools.refrun import place


@dataclass(frozen=True)
class BuildConfig:
    checkpoint: place.Checkpoint
    policy: dict
    dtype: torch.dtype
    store: place.Store


@dataclass
class Reference:
    """One mode's model and the facts the runner records about it.

    - `tensor_names`: every stored checkpoint tensor to the module, parameter
      or buffer path it feeds, or to "host:<use>" for host-resident tables.
      The runner refuses a stored tensor the map leaves out.
    - `layout`: checkpoint layout facts the adapter applies (for example a
      QKV projection stored in TP chunks), as canonical JSON.
    - `replacements`: upstream kernels or modules replaced, and by what.
    - `media_inputs(sequence, device, dtype) -> dict`: forward kwargs for
      the sequence's media tensors.
    - `taps(model) -> {tap name: module path}`: the tap name map (W2).
    - `forward(model, ids, keep, media) -> logits [keep, V]` and
      `decode(model, ids, media, max_new_tokens, stop_ids) -> (answer, logits)`
      override the plain HF calls below.
    """
    model: nn.Module
    tensor_names: dict
    layout: dict = field(default_factory=dict)
    replacements: dict = field(default_factory=dict)
    media_inputs: Callable = None
    taps: Callable = None
    forward: Callable = None
    decode: Callable = None


def hf_forward(model, ids, keep, media):
    """The last `keep` rows of HF logits for one sequence, without a cache."""
    return model(input_ids=ids, use_cache=False, logits_to_keep=keep, **media).logits[0]


def hf_decode(model, ids, media, max_new_tokens, stop_ids):
    """Greedy continuation with the model's own cache: argmax of each step's
    logits (first maximum, so ties go to the lower ID), until a stop ID."""
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1, **media)
    answer, rows = [], []
    while True:
        row = out.logits[0, -1]
        token = int(torch.argmax(row))
        answer.append(token)
        rows.append(row)
        if token in stop_ids or len(answer) == max_new_tokens:
            return answer, torch.stack(rows)
        step = torch.tensor([[token]], device=ids.device)
        out = model(input_ids=step, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)


def check_tensor_names(reference, stored):
    """Every stored tensor is mapped, and every mapped path exists."""
    missing = sorted(set(stored) - set(reference.tensor_names))
    if missing:
        raise ValueError(f"adapter leaves {len(missing)} stored tensors unmapped, e.g. {missing[:3]}")
    for name, path in reference.tensor_names.items():
        if path.startswith("host:"):
            continue
        for getter in (reference.model.get_submodule, reference.model.get_parameter,
                       reference.model.get_buffer):
            try:
                getter(path)
                break
            except AttributeError:
                continue
        else:
            raise ValueError(f"{name} maps to {path}, which the model does not have")
