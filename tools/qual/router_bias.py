"""The router correction biases a layer comparison needs to rebuild selection
scores, extracted once from a pinned checkpoint into a small sidecar.

A router that selects on `sigmoid(logit) + bias` (a noaux_tc-style router)
taps its F32 logits, not the biased scores; the bias is a checkpoint
constant. `extract` reads each layer's bias tensor named by `--tensor` (a
pattern with `{layer}`) from the checkpoint's safetensors shards, after
verifying the index and every shard it reads against the checkpoint lock
(byte length and SHA-256 of the whole file, hashed once after the read), and
writes the sidecar:

    {"schema": "chopinfer.router-bias.v1", "model", "tensor",
     "checkpoint": {"lock", "source_lock_sha256", "files": {shard: sha256}},
     "experts", "layers": {"<layer>": base64 of the F32 values, little endian}}

The sidecar is pinned like any qualification input; `load` reads it back.
Only F32 and BF16 tensors are read (BF16 widens to F32 exactly).

    python3 tools/qual/router_bias.py extract --checkpoint DIR \\
        --lock models/<m>/checkpoint-files.json --model M \\
        --tensor 'model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias' --out FILE
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.refrun import keys  # noqa: E402

SCHEMA = "chopinfer.router-bias.v1"
BLOCK = 1 << 24


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(lock, name):
    entry = lock.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"the checkpoint lock does not pin {name}")
    return entry


def verify(checkpoint, lock, name):
    """Checks checkpoint file `name`'s byte length and SHA-256 against the
    lock. Called after the file's bytes were read, so a file that differs,
    before the read or while it is read, is refused in one pass."""
    entry, path = pinned(lock, name), Path(checkpoint) / name
    if path.stat().st_size != entry["byte_length"]:
        raise ValueError(f"{name} holds {path.stat().st_size} bytes, the lock pins {entry['byte_length']}")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{name} differs from the checkpoint lock's sha256")


def read_tensor(path, name):
    """The F32 values of tensor `name` in a safetensors file, as bytes."""
    with open(path, "rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
        entry = header.get(name)
        if entry is None:
            raise ValueError(f"{path.name} holds no tensor {name}")
        start, end = entry["data_offsets"]
        stream.seek(8 + length + start)
        data = stream.read(end - start)
    count = 1
    for size in entry["shape"]:
        count *= size
    if entry["dtype"] == "F32" and len(data) == 4 * count:
        return data
    if entry["dtype"] == "BF16" and len(data) == 2 * count:
        return b"".join(b"\0\0" + data[i:i + 2] for i in range(0, len(data), 2))
    raise ValueError(f"{name} is {entry['dtype']} {entry['shape']}, not an F32 or BF16 vector")


def extract(checkpoint, lock_path, model, pattern):
    """The sidecar of every layer whose bias tensor `pattern` names."""
    if "{layer}" not in pattern:
        raise ValueError("--tensor must name {layer}")
    lock = keys.read_lock(lock_path)
    pinned(lock, "model.safetensors.index.json")
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_bytes())["weight_map"]
    verify(checkpoint, lock, "model.safetensors.index.json")
    regex = re.compile(re.escape(pattern).replace(re.escape("{layer}"), r"(\d+)") + r"\Z")
    tensors = {int(match.group(1)): name for name in index if (match := regex.match(name))}
    if not tensors:
        raise ValueError(f"the checkpoint index names no tensor matching {pattern}")
    files, layers, experts = {}, {}, set()
    for layer, name in sorted(tensors.items()):
        shard = index[name]
        files[shard] = pinned(lock, shard)["sha256"]
        data = read_tensor(Path(checkpoint) / shard, name)
        experts.add(len(data) // 4)
        layers[str(layer)] = base64.b64encode(data).decode("ascii")
    if len(experts) != 1:
        raise ValueError(f"the bias tensors have different lengths: {sorted(experts)}")
    # Each shard is hashed once, after its tensors were read.
    for shard in files:
        verify(checkpoint, lock, shard)
    source = keys.checkpoint(lock, [])["source_lock_sha256"]
    # The lock is named as the repository names it, never by a local path.
    try:
        named = Path(lock_path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f"{lock_path} is not a checkpoint lock of this repository") from None
    return {"schema": SCHEMA, "model": model, "tensor": pattern,
            "checkpoint": {"lock": named, "source_lock_sha256": source, "files": files},
            "experts": experts.pop(), "layers": layers}


def load(data):
    """(sidecar without values, {layer: tuple of F32 values as floats}) of
    sidecar bytes or a parsed sidecar; refuses a malformed one."""
    sidecar = json.loads(data) if isinstance(data, (bytes, str)) else data
    if not isinstance(sidecar, dict) or sidecar.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} sidecar")
    experts, layers = sidecar.get("experts"), sidecar.get("layers")
    if not isinstance(experts, int) or experts <= 0 or not isinstance(layers, dict) or not layers:
        raise ValueError("the router bias sidecar names no experts or layers")
    values = {}
    for layer, text in layers.items():
        if not layer.isdigit():
            raise ValueError(f"router bias layer {layer!r} is not a layer number")
        raw = base64.b64decode(text, validate=True)
        if len(raw) != 4 * experts:
            raise ValueError(f"router bias layer {layer} holds {len(raw) // 4} values, not {experts}")
        values[int(layer)] = struct.unpack(f"<{experts}f", raw)
    return {key: value for key, value in sidecar.items() if key != "layers"}, values


def tapdiff_args(spec, config, scratch):
    """tapdiff's --router arguments for a declared router spec (the ROUTER
    constant of a layer check); none for an empty spec. A spec whose `bias`
    names a declared input reads that input as the reference lock
    authenticates it, requires it to belong to the declaration's model and
    checkpoint pin, and hands tapdiff a private copy in `scratch`."""
    if not spec:
        return []
    args = ["--router", json.dumps(spec, sort_keys=True)]
    if spec.get("bias"):
        import common
        from refs import verify_input
        key = spec["bias"]
        source = config.get("inputs", {}).get(key)
        if source is None:
            raise ValueError(f"ROUTER names the bias input {key}, which the declaration does not declare")
        path = common.declared_path(source)
        if path is None or not path.is_file():
            raise ValueError(f"the router bias input {key} is missing")
        data = verify_input(config, key, path)
        data = path.read_bytes() if data is None else data
        sidecar, _ = load(data)
        if sidecar.get("model") != config.get("model"):
            raise ValueError(f"the router bias input {key} belongs to model {sidecar.get('model')!r}")
        pinned = common.checkpoint_pin(config)["source_lock_sha256"]
        if (sidecar.get("checkpoint") or {}).get("source_lock_sha256") != pinned:
            raise ValueError(f"the router bias input {key} was extracted from another checkpoint")
        copy = Path(scratch) / "router-bias.json"
        copy.write_bytes(data)
        args += ["--router-bias", str(copy)]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("extract", help="extract the router biases of a checkpoint into a sidecar")
    run.add_argument("--checkpoint", required=True, type=Path, help="the checkpoint directory")
    run.add_argument("--lock", required=True, type=Path, help="its checkpoint lock (checkpoint-files.json)")
    run.add_argument("--model", required=True, help="the model the sidecar belongs to")
    run.add_argument("--tensor", required=True, help="the bias tensor name with {layer}")
    run.add_argument("--out", required=True, type=Path, help="new sidecar file")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    sidecar = extract(args.checkpoint, args.lock, args.model, args.tensor)
    args.out.write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "layers": len(sidecar["layers"]), "experts": sidecar["experts"],
                      "shards": len(sidecar["checkpoint"]["files"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
