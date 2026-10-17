"""Generate DeepSeek V4.1 Flash's Engram hash resource from the pinned vendor code.

The Engram hash reads a compressed token map (129,280 raw ids onto 99,092
normalized ids), one odd multiplier per (layer, lookback) and the bucket
primes. The vendor builds them at load time from the tokenizer with
`tokenizers` normalizers, numpy's PCG64 and sympy (`inference/engram.py`).
This script runs that pinned code once, offline, and writes the result as one
binary resource the package ships; the contract crate parses and verifies it
(`own_ops::engram::HashTables`, `contract/resources/engram-hash.v1.bin`).

Usage, with the reference interpreter (`$CHOPINFER_REFRUN_PYTHON`):

    CKPT="$CHOPINFER_DEEPSEEK_CHECKPOINT"
    python models/deepseek41/engram_resource.py "$CKPT" OUT.bin [--cases CASES.json]
    python models/deepseek41/engram_resource.py "$CKPT" \
        --check models/deepseek41/contract/resources/engram-hash.v1.bin

`--cases` also writes vendor `NgramHashState` outputs for fixed sequences
(the contract test's fixture). `--check` regenerates and compares bytes.

Layout (little-endian): magic `DS41ENG1`; u32 vocab size, compressed vocab
size, raw pad id, compressed pad id, Engram bucket base (`engram_vocab_size`),
layers, max n-gram size, heads; u32 layer ids; i64 multipliers
`[layer][lookback]`; u32 primes `[layer][n-gram size - 2][head]`; u32 token
map `[raw id]`.
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
MAGIC = b"DS41ENG1"


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


def generate(checkpoint: Path):
    engram, config, tokenizer = vendor(checkpoint)
    args = Args(config)
    layout = engram.EngramLayout.from_args(args)
    state = engram.NgramHashState(args, layout, tokenizer)
    token_map = state.token_map.tolist()
    if len(token_map) != config["vocab_size"]:
        sys.exit(f"token map has {len(token_map)} ids, config {config['vocab_size']}")
    compressed = config["engram_compressed_vocab_size"]
    head = struct.pack(
        "<8I",
        config["vocab_size"],
        compressed,
        config["engram_pad_id"],
        int(state.pad_id),
        config["engram_vocab_size"],
        len(layout.layer_ids),
        layout.max_ngram_size,
        layout.n_heads,
    )
    body = bytearray(MAGIC + head)
    body += struct.pack(f"<{len(layout.layer_ids)}I", *layout.layer_ids)
    for row in state.multipliers.tolist():
        body += struct.pack(f"<{len(row)}q", *row)
    for layer in layout.primes:
        for per_ngram in layer:
            body += struct.pack(f"<{len(per_ngram)}I", *per_ngram)
    body += struct.pack(f"<{len(token_map)}I", *token_map)
    for layer, rows in zip(layout.primes, layout.num_embeddings):
        if sum(p for per_ngram in layer for p in per_ngram) != rows:
            sys.exit("bucket primes do not sum to the table rows")
    return bytes(body), state, config


def cases(state, config):
    """Vendor hashes of fixed sequences: chunk splits, pad ids and dead spans."""
    import numpy as np
    import torch

    rng = np.random.default_rng(20261017)
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
    data, state, config = generate(options.checkpoint)
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
        document = {
            "resource_sha256": digest,
            "versions": PINNED,
            "cases": cases(state, config),
        }
        options.cases.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        print(f"wrote {options.cases}")


if __name__ == "__main__":
    main()
