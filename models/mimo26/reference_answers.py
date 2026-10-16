"""MiMo reference with cached greedy generation for reference answers.

Forced scoring uses reference.py unchanged. This separate adapter binds the
additional decode path without invalidating existing forced-only captures.
"""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("refrun_adapter_mimo26_reference",
                                               Path(__file__).with_name("reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
MODEL = reference.MODEL
EXTRA_SHARDS = reference.EXTRA_SHARDS
units = reference.units


def reference_code(checkpoint):
    return {**reference.reference_code(checkpoint), "models/mimo26/reference.py": Path(reference.__file__)}


def decode(model, ids, media, max_new_tokens, stop_ids):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    cache = DynamicCache(config=model.config)
    answer, rows = [], []
    for _ in range(max_new_tokens):
        shape = torch.empty((1, ids.shape[1], 1), dtype=model.model.embed_tokens.weight.dtype, device=ids.device)
        masks = {"full_attention": create_causal_mask(model.config, shape, None, cache),
                 "sliding_window_attention": create_sliding_window_causal_mask(model.config, shape, None, cache)}
        out = model(input_ids=ids, attention_mask=masks, past_key_values=cache,
                    use_cache=True, logits_to_keep=1, **media)
        row = out.logits[0, -1]
        token = int(row.argmax())
        answer.append(token)
        rows.append(row)
        if token in stop_ids:
            break
        cache = out.past_key_values
        ids = torch.tensor([[token]], device=ids.device)
        media = {}
    return answer, torch.stack(rows)


def build(config, placement):
    result = reference.build(config, placement)
    result.decode = decode
    return result
