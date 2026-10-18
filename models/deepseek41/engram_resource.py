"""Generate DeepSeek V4.1's Engram hash resource from the pinned vendor code.

The Engram hash reads a compressed token map (129,280 raw ids onto 99,092
normalized ids), one odd multiplier per (layer, lookback) and the bucket
primes. The vendor builds them at load time from the tokenizer with
`tokenizers` normalizers, numpy's PCG64 and sympy (`inference/engram.py`).
This script runs that pinned code once, offline, and writes the token map
and the multipliers of every layer id below 64 as one binary resource the
package ships. The primes follow from a geometry's bucket base alone, so the
contract crate draws them itself; it parses and verifies the resource
(`own_ops::engram::HashTables`, `contract/resources/engram-hash.v2.bin`).

Usage, with the reference interpreter (`$CHOPINFER_REFRUN_PYTHON`):

    CKPT="$CHOPINFER_DEEPSEEK_CHECKPOINT"
    python models/deepseek41/engram_resource.py "$CKPT" OUT.bin [--cases CASES.json]
    python models/deepseek41/engram_resource.py "$CKPT" \
        --check models/deepseek41/contract/resources/engram-hash.v2.bin

`--cases` also writes vendor `NgramHashState` outputs for fixed sequences
(the contract test's fixture): under the checkpoint's own layout and under
two small layouts of other layer ids and bucket bases. `--check`
regenerates and compares bytes.

Layout (little-endian): magic `DS41ENG2`; u32 vocab size, compressed vocab
size, raw pad id, compressed pad id, max n-gram size, layer ids (64); i64
multipliers `[layer id][lookback]`; u32 token map `[raw id]`.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# The versions the resource was generated with. The normalizers and the RNG
# define the map and the multipliers, so another version refuses.
PINNED = {"tokenizers": "0.23.2", "transformers": "5.16.1", "numpy": "2.5.3", "sympy": "1.14.0"}
MAGIC = b"DS41ENG2"
# Multipliers are written for every layer id below this.
LAYER_IDS = 64
# Small layouts the cases also cover: (layer ids, bucket base, table rows).
LAYOUTS = {"layers-1-3-base-1000": ([1, 3], 1000), "layers-2-5-base-4000": ([2, 5], 4000)}


def vendor(checkpoint: Path):
    """The checkpoint's own `inference/engram.py`, its config and tokenizer."""
    import numpy
    import sympy
    import tokenizers
    import transformers

    found = {
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "sympy": sympy.__version__,
    }
    if found != PINNED:
        sys.exit(f"pinned versions {PINNED} required, found {found}")
    sys.path.insert(0, str(checkpoint / "inference"))
    import engram  # noqa: E402  the vendor module, pinned with the weights

    config = json.loads((checkpoint / "inference" / "config.json").read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(checkpoint))
    return engram, config, tokenizer


class Args:
    """The vendor's ModelArgs fields NgramHashState reads."""

    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)
        self.max_batch_size = 1
        self.max_seq_len = 256


def state_of(engram, config, tokenizer, layer_ids=None, base=None):
    """The vendor's hash state under the config, or under a small layout."""
    args = Args(config)
    if layer_ids is not None:
        args.engram_layer_ids = layer_ids
        args.engram_vocab_size = base
        args.engram_num_embeddings = [0] * len(layer_ids)
    layout = engram.EngramLayout.from_args(args)
    return engram.NgramHashState(args, layout, tokenizer), layout


def generate(checkpoint: Path):
    engram, config, tokenizer = vendor(checkpoint)
    state, layout = state_of(engram, config, tokenizer)
    token_map = state.token_map.tolist()
    if len(token_map) != config["vocab_size"]:
        sys.exit(f"token map has {len(token_map)} ids, config {config['vocab_size']}")
    for layer, rows in zip(layout.primes, layout.num_embeddings):
        if sum(p for per_ngram in layer for p in per_ngram) != rows:
            sys.exit("bucket primes do not sum to the table rows")
    compressed = config["engram_compressed_vocab_size"]
    max_ngram = layout.max_ngram_size
    multipliers = engram.compute_hash_multipliers(tuple(range(LAYER_IDS)), max_ngram, compressed)
    if multipliers[list(layout.layer_ids)].tolist() != state.multipliers.tolist():
        sys.exit("per-layer-id multipliers differ from the state's")
    head = struct.pack(
        "<6I", config["vocab_size"], compressed, config["engram_pad_id"], int(state.pad_id), max_ngram, LAYER_IDS
    )
    body = bytearray(MAGIC + head)
    for row in multipliers.tolist():
        body += struct.pack(f"<{len(row)}q", *row)
    body += struct.pack(f"<{len(token_map)}I", *token_map)
    return bytes(body), (engram, config, tokenizer, state)


def cases(state, config, seed=20261017):
    """Vendor hashes of fixed sequences: chunk splits, pad ids and dead spans."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    out = []
    for index in range(12):
        length = int(rng.integers(1, 48))
        ids = rng.integers(0, config["vocab_size"], size=length)
        if index % 4 == 1:
            ids[rng.integers(0, length, size=max(1, length // 4))] = config["engram_pad_id"]
        mask = None
        if index % 3 == 2:
            mask = np.ones(length, dtype=bool)
            start = int(rng.integers(0, length))
            mask[start : min(length, start + int(rng.integers(1, 8)))] = False
        split = int(rng.integers(0, length + 1)) if index % 2 else length
        parts = []
        for begin, end in ((0, split), (split, length)):
            if end > begin:
                chunk_mask = None if mask is None else torch.tensor(mask[None, begin:end])
                parts.append(state(torch.tensor(ids[None, begin:end]), begin, chunk_mask)[0])
        hashes = torch.cat(parts, 0)  # [positions, layers, columns], int64
        out.append(
            {
                "ids": ids.tolist(),
                "text": None if mask is None else mask.tolist(),
                "split": split,
                "first_row": hashes[0].flatten().tolist(),
                "sha256": hashlib.sha256(hashes.numpy().astype("<i8").tobytes()).hexdigest(),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--check", type=Path)
    options = parser.parse_args()
    data, (engram, config, tokenizer, state) = generate(options.checkpoint)
    digest = hashlib.sha256(data).hexdigest()
    if options.check:
        pinned = options.check.read_bytes()
        if pinned != data:
            sys.exit(f"{options.check} differs from the regenerated resource ({digest})")
        print(f"{options.check} matches the vendor code: sha256 {digest}")
    if options.out:
        options.out.write_bytes(data)
        print(f"wrote {options.out}: {len(data)} bytes, sha256 {digest}")
    if options.cases:
        layouts = {}
        for index, (name, (layer_ids, base)) in enumerate(LAYOUTS.items()):
            small, layout = state_of(engram, config, tokenizer, layer_ids, base)
            layouts[name] = {
                "layer_ids": layer_ids,
                "bucket_base": base,
                "table_rows": [sum(p for n in layer for p in n) for layer in layout.primes],
                "cases": cases(small, config, seed=index + 1)[:4],
            }
        document = {
            "resource_sha256": digest,
            "versions": PINNED,
            "cases": cases(state, config),
            "layouts": layouts,
        }
        options.cases.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        print(f"wrote {options.cases}")


if __name__ == "__main__":
    main()
