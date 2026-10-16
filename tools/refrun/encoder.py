"""The generic encoder reference: an encoder declaration evaluated in torch.

    python3 -m tools.refrun.encoder --declaration models/<m>/contract/encoders/<name>.json \\
        --adapter models/<m>/reference.py --checkpoint DIR --lock models/<m>/checkpoint-files.json \\
        [--lock-variant V] --workload WORKLOAD.json --modes exact,policy,reference-bf16 \\
        --output DIR [--devices cuda:0,...] [--blocks] [--tap-bytes N]

It reads a declaration (crates/encoder-spec, `chopinfer.encoder.v1`) and
evaluates its stages in order over each media item of the workload that the
declaration's recipes take, with one torch implementation per component
variant id (encoder_ops.VARIANTS, and a model's own `<model>:` variants in
models/<model>/encoder_reference.py). A variant without one is refused by
name. Weights come from the checkpoint the runner verifies against its lock
(stamps.py, place.Checkpoint and place.Store), through the declaration's
weight mapping; the adapter supplies only the checkpoint's extra shards.

Modes (`--modes`, in run order):

- `exact`: every value and product in f64, no rounding (`f64` arithmetic);
- `policy`: the declaration's precision policy; `policy:<id>` evaluates the
  same stages under another policy (MiMo's bf16-legacy.v1 declarations
  under f32-pairs.v1, labelled `policy-f32-pairs.v1`);
- `reference-bf16`: unfused BF16 PyTorch, as HF or vLLM run the encoder in
  BF16.

encoder_ops.py states each arithmetic's rounding points. Every mode runs
under the runner's torch flags (modes.FLAGS: IEEE F32 matmuls, deterministic
algorithms) and repeats its first item, which must give the same bytes.

Output: `<output>/<label>.json` per mode (schema chopinfer.refrun.encoder-rows.v1:
the mode document and, per sequence, its key document, `key_sha256` and the
sha256 of every item's output rows), `<output>/taps/<label>/step<NNNN>`
with one step per sequence that has media of this encoder (taps.py format,
encoder axis, each item's rows recorded under its kind and item), and
`run.json` (provenance). Taps (`tap_map`):

- `enc.<tower>.out`: the rows the language model reads;
- `enc.<tower>.x.unrounded`: the same before the output rounding to BF16
  (the input of a final `cast.bf16.v1`, else of the encoder's own rounding);
- `enc.<tower>.x.codes`: a quantize stage's codes;
- with `--blocks`, `enc.<tower>.B<nn>.out`: each transformer block's output,
  numbered across the declaration's transformer stages, always in the rows'
  own order: a block under a `reorder` puts its rows back before the tap.

`<tower>` is `vision` for image and video recipes, `audio` for audio.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

from tools.model_sdk.diagnostic import canonical
from tools.refrun import encoder_ops as ops, keys, modes, place, run, stamps, taps, workload

SCHEMA = "chopinfer.refrun.encoder-rows.v1"
MODE_SCHEMA = "chopinfer.refrun.encoder-mode.v1"
DECLARATION_SCHEMA = "chopinfer.encoder.v1"
MODES = ("exact", "policy", "reference-bf16")
PRECISIONS = {"f32-pairs.v1": "pairs", "bf16-legacy.v1": "legacy"}
TOWERS = {"image": "vision", "video": "vision", "audio": "audio"}
PAYLOADS = {"f32": torch.float32, "bf16": torch.bfloat16}
PATCHIFY = ("merge-blocks.v1",)
LAYOUTS = ("layout.tokens.v1",)


class Declaration:
    """One encoder declaration file, read as JSON. The encoder-spec crate
    validates declarations; this reads the fields the reference evaluates."""

    def __init__(self, path):
        self.path = Path(path)
        self.bytes = self.path.read_bytes()
        self.data = json.loads(self.bytes)
        if self.data.get("schema") != DECLARATION_SCHEMA:
            raise ValueError(f"{path} is not a {DECLARATION_SCHEMA} declaration")
        self.model, self.name, self.precision = self.data["model"], self.data["name"], self.data["precision"]
        self.stages = self.data["stages"]
        self.input, self.sequences = self.data["input"], self.data["sequences"]
        self.kinds = tuple(sorted({recipe["tag"].split(".")[0] for recipe in self.input["recipes"]}))
        towers = {TOWERS.get(kind) for kind in self.kinds}
        if len(towers) != 1 or None in towers:
            raise ValueError(f"{self.name}: recipes for {self.kinds} name no single tower")
        self.tower = towers.pop()
        if self.input.get("patchify", PATCHIFY[0]) not in PATCHIFY:
            raise ops.refused(f"patchify {self.input['patchify']}")
        if self.data["output"]["op"] not in LAYOUTS:
            raise ops.refused(self.data["output"]["op"])

    @property
    def sha256(self):
        return hashlib.sha256(self.bytes).hexdigest()

    def ops(self):
        """Every component id the declaration names, stages and slots."""
        found = []

        def walk(value):
            if isinstance(value, dict):
                if isinstance(value.get("op"), str):
                    found.append(value["op"])
                for key, item in value.items():
                    if key != "weights":
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(self.stages)
        walk(self.data["output"])
        return found


def block_set(text, count):
    """The members of a block set (`even`, `odd`, or `a,b-c` items) below `count`."""
    if text in ("even", "odd"):
        return set(range(0 if text == "even" else 1, count, 2))
    members = set()
    for item in text.split(","):
        first, _, last = item.partition("-")
        members.update(range(int(first), int(last or first) + 1))
    if not members or max(members) >= count:
        raise ValueError(f"block set {text} names a block past {count}")
    return members


def provider(declaration):
    """The model's own variants (models/<model>/encoder_reference.py) when the
    declaration names any, as (repo path, module); else (None, None)."""
    if not any(":" in op for op in declaration.ops()):
        return None, None
    relative = f"models/{declaration.model}/encoder_reference.py"
    path = keys.ROOT / relative
    if not path.exists():
        raise ops.refused(next(op for op in declaration.ops() if ":" in op))
    spec = importlib.util.spec_from_file_location(f"refrun_encoder_{declaration.model}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    foreign = [op for op in module.VARIANTS if not op.startswith(f"{declaration.model}:")]
    if module.MODEL != declaration.model or foreign:
        raise ValueError(f"{relative} registers variants outside {declaration.model}: {foreign}")
    return relative, module


def variants(declaration):
    """Every variant id the reference implements for this declaration, to its class."""
    # Norm and attention slots are built by the transformer from their ids.
    table = {**{op: None for op in ops.NORMS}, **{op: None for op in ops.ATTENTION}, **ops.VARIANTS,
             **ops.ROTARY, **ops.MLP, **ops.REORDER}
    _, module = provider(declaration)
    if module is not None:
        table.update(module.VARIANTS)
    for op in declaration.ops():
        if op not in table:
            raise ops.refused(op)
    return table


class Names:
    """Declaration tensor names to the runner checkpoint's: the declaration's
    `checkpoint.prefix`, its file for the name (`checkpoint.files`, longest
    matching prefix), and the prefix the checkpoint gives that file's
    tensors when it is an extra shard."""

    def __init__(self, declaration, checkpoint, extra):
        source = declaration.data["checkpoint"]
        self.prefix, self.files = source.get("prefix", ""), source.get("files", {})
        self.checkpoint, self.extra = checkpoint, dict(extra)

    def __call__(self, name):
        full = self.prefix + name
        matches = [prefix for prefix in self.files if full.startswith(prefix)]
        file = self.files[max(matches, key=len)] if matches else None
        stored = self.extra.get(file, "") + full
        entry = self.checkpoint.entries.get(stored)
        if entry is None or (file is not None and entry.file != file):
            raise ValueError(f"declaration tensor {full} is not in {file or 'the checkpoint'}")
        return stored


class Weights:
    """The declaration's weight mapping over stored tensors: `load(stored
    name)` gives a tensor in its stored dtype on `device`. Transforms are
    exact (`flatten`, `taps_first`, `pad_rows`, `pad_cols`,
    `space_to_depth_cols`, `widen_f32`)
    except `narrow_bf16`, a bf16-legacy.v1 rounding applied only under
    `legacy`. Under `bf16` every role is rounded to BF16, as a BF16 load of
    the upstream model casts every parameter. A bias the checkpoint lacks
    (`zeros`) is F32 zeros, or its siblings' dtype inside `concat_rows`."""

    def __init__(self, declaration, names, load, arithmetic, device):
        self.declaration, self.names, self.load = declaration, names, load
        self.arithmetic, self.device = arithmetic, device

    def stage(self, index):
        stage = self.declaration.stages[index]
        sources = stage.get("weights", {})
        if "blocks" in stage:
            return BlockWeights(self, sources, stage["blocks"])
        return {role: self.role(source) for role, source in sources.items()}

    def role(self, source, block=None, blocks=None):
        tensor = self.resolve(source, block, blocks)
        if tensor is not None and self.arithmetic == "bf16" and tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        return tensor

    def resolve(self, source, block=None, blocks=None):
        """The tensor of one role, or None for a block outside the source's `blocks`."""
        if isinstance(source, str):
            source = {"tensor": source}
        if "zeros" in source:
            return torch.zeros(source["zeros"], dtype=torch.float32, device=self.device)
        if "concat_rows" in source:
            parts = [self.resolve(part, block, blocks) for part in source["concat_rows"]]
            dtype = next((p.dtype for p, s in zip(parts, source["concat_rows"])
                          if not (isinstance(s, dict) and "zeros" in s)), torch.float32)
            return torch.cat([p.to(dtype) for p in parts])
        if "concat_range" in source:
            return torch.cat([self.transform(self.tensor(source["concat_range"].replace("{j}", str(j)), block),
                                             source.get("transforms", [])) for j in range(source["count"])])
        if "blocks" in source:
            if blocks is None:
                raise ValueError(f"{source['tensor']} has a block set outside a transformer")
            if block not in block_set(source["blocks"], blocks):
                return None
        return self.transform(self.tensor(source["tensor"], block), source.get("transforms", []))

    def tensor(self, name, block):
        if "{i}" in name:
            if block is None:
                raise ValueError(f"{name} names a block outside a transformer")
            name = name.replace("{i}", str(block))
        return self.load(self.names(name))

    def transform(self, tensor, transforms):
        for step in transforms:
            if step == "flatten":
                tensor = tensor.reshape(tensor.shape[0], -1)
            elif step == "taps_first":
                tensor = tensor.permute(0, 2, 1).contiguous()
            elif step == "widen_f32":
                tensor = tensor.to(torch.float32)
            elif step == "narrow_bf16":
                if self.arithmetic == "legacy":
                    tensor = tensor.to(torch.bfloat16)
            elif isinstance(step, dict) and set(step) == {"pad_rows"}:
                pad = step["pad_rows"] - tensor.shape[0]
                tensor = torch.cat([tensor, tensor.new_zeros((pad, *tensor.shape[1:]))])
            elif isinstance(step, dict) and set(step) == {"pad_cols"}:
                pad = step["pad_cols"] - tensor.shape[1]
                tensor = torch.cat([tensor, tensor.new_zeros((tensor.shape[0], pad))], dim=1)
            elif isinstance(step, dict) and set(step) == {"space_to_depth_cols"}:
                # Columns `c * f * f + ky * f + kx` to `(ky * f + kx) * c + c`.
                cell = step["space_to_depth_cols"] ** 2
                tensor = tensor.reshape(tensor.shape[0], -1, cell).transpose(1, 2).reshape(tensor.shape[0], -1)
            else:
                raise ops.refused(f"weight transform {step}")
        return tensor


class BlockWeights:
    """A transformer stage's roles, resolved per block (`{i}`); a role
    whose source has a block set is absent outside it."""

    def __init__(self, weights, sources, blocks):
        self.weights, self.sources, self.blocks = weights, sources, blocks

    def block(self, index):
        resolved = {role: self.weights.role(source, index, self.blocks) for role, source in self.sources.items()}
        return {role: tensor for role, tensor in resolved.items() if tensor is not None}


class Context:
    """What a stage's constructor reads besides its parameters and weights."""

    def __init__(self, numerics, devices, table):
        self.numerics, self.devices, self.table = numerics, devices, table

    def rotary(self, params):
        return self.table[params["op"]](params, self.numerics)

    def reorder(self, op):
        return self.table[op]

    def mlp(self, params, weights):
        return self.table[params["op"]](params, weights, self.numerics)

    @staticmethod
    def blocks(text, count):
        return block_set(text, count)


class Boundary(nn.Module):
    """The rows the language model reads: rounded to BF16, except under `f64`."""

    def __init__(self, numerics):
        super().__init__()
        self.numerics = numerics

    def forward(self, x):
        return x if self.numerics.exact else x.to(torch.bfloat16)


@dataclass(frozen=True)
class Item:
    """One media item: its payload rows and, for patches, the `(t, h, w)` grid."""
    kind: str
    payload: torch.Tensor
    grid: tuple = None


class Encoder(nn.Module):
    """A declaration's stages under one mode's numerics. `forward(item)` gives
    the item's output rows. Stages before a joining stage (a group op) run
    per segment; the joining stage reads the segments' rows in order."""

    def __init__(self, declaration, weights, numerics, devices, table):
        super().__init__()
        self.declaration, self.numerics = declaration, numerics
        context = Context(numerics, devices, table)
        stages = []
        for index, stage in enumerate(declaration.stages):
            params = {key: value for key, value in stage.items() if key not in ("op", "weights")}
            stages.append(table[stage["op"]](stage["op"], params, weights.stage(index), context))
        self.stages = nn.ModuleList(stages)
        self.boundary = Boundary(numerics)

    def segments(self, item):
        rows, kind = item.payload.shape[0], self.declaration.sequences["kind"]
        if kind in ("item", "temporal-grid"):
            t, h, w = item.grid
            if t * h * w != rows:
                raise ValueError(f"a {item.grid} grid has {t * h * w} patches, not {rows} rows")
            lengths = [rows] if kind == "item" else [h * w] * t
            return [(item.payload, ops.Rows(lengths, item.grid))]
        if kind == "segments":
            size = self.declaration.sequences["input_rows"]
            return [(item.payload[start:start + size], ops.Rows([min(size, rows - start)]))
                    for start in range(0, rows, size)]
        raise ops.refused(f"sequences {kind}")

    def forward(self, item):
        join = next((i for i, stage in enumerate(self.stages) if getattr(stage, "joins", False)), len(self.stages))
        parts = []
        for x, rows in self.segments(item):
            for stage in self.stages[:join]:
                x, rows = stage(x, rows), stage.geometry(rows)
            parts.append((x, rows))
        if len(parts) == 1:
            x, rows = parts[0]
        else:
            x, rows = torch.cat([x for x, _ in parts]), ops.Rows([sum(r.count for _, r in parts)])
        for stage in self.stages[join:]:
            x, rows = stage(x, rows), stage.geometry(rows)
        return self.boundary(x)


def tap_map(encoder, blocks=True):
    """Tap names to the encoder's modules (module docstring)."""
    declaration = encoder.declaration
    tower, kinds = declaration.tower, declaration.kinds
    # The output rounding: a final cast stage, else the boundary (a final
    # `linear.pairs.v1` with `output: bf16` leaves its rounding to it).
    last = len(declaration.stages) - 1
    rounding = f"stages.{last}" if declaration.stages[last]["op"] == "cast.bf16.v1" else "boundary"
    names = {f"enc.{tower}.out": taps.Hook("", rows=kinds),
             f"enc.{tower}.x.unrounded": taps.Hook(rounding, "input", rows=kinds)}
    block = 0
    for index, (stage, spec) in enumerate(zip(encoder.stages, declaration.stages)):
        if isinstance(stage, ops.Transformer):
            for inner in range(len(stage.blocks)):
                if blocks:
                    names[f"enc.{tower}.B{block:02d}.out"] = taps.Hook(f"stages.{index}.blocks.{inner}", rows=kinds)
                block += 1
        elif spec["op"].split(":")[-1].startswith("quantize."):
            names[f"enc.{tower}.x.codes"] = taps.Hook(f"stages.{index}", rows=kinds)
    return names


def items(sequence, declaration, device):
    """The sequence's media items this encoder takes, in order."""
    out = []
    for media in sequence.media:
        if media.kind not in declaration.kinds:
            continue
        tensor, payload = media.tensor, declaration.input["payload"]
        if tensor.dtype != payload or len(tensor.shape) != 2 or tensor.shape[1] != declaration.input["columns"]:
            raise ValueError(f"{sequence.name}: {media.kind} rows {tensor.dtype} {list(tensor.shape)}, the "
                             f"declaration takes {payload} [rows, {declaration.input['columns']}]")
        dtype = PAYLOADS[payload]
        rows = torch.frombuffer(bytearray(tensor.data), dtype=dtype).view(tensor.shape).to(device)
        grid = None
        if media.grid is not None:
            values = torch.frombuffer(bytearray(media.grid.data), dtype=torch.int64).tolist()
            if media.grid.dtype != "i64" or len(values) != 3:
                raise ValueError(f"{sequence.name}: one (t, h, w) I64 grid per {media.kind} item")
            grid = tuple(values)
        out.append(Item(media.kind, rows, grid))
    return out


def stored_names(declaration, names):
    """Every stored tensor the declaration's weight mapping reads."""
    seen = []

    def record(stored):
        entry = names.checkpoint.entries[stored]
        if stored not in seen:
            seen.append(stored)
        return torch.empty(entry.shape, dtype=place.DTYPES[entry.dtype], device="meta")
    weights = Weights(declaration, names, record, "legacy", "meta")
    for index, stage in enumerate(declaration.stages):
        resolved = weights.stage(index)
        if isinstance(resolved, BlockWeights):
            for block in range(stage["blocks"]):
                resolved.block(block)
    return seen


def arithmetic(mode, precision):
    return {"exact": "f64", "reference-bf16": "bf16"}.get(mode) or PRECISIONS[precision]


def mode_document(mode, precision):
    """Every numeric choice of one mode, as canonical JSON (keyed)."""
    return {"schema": MODE_SCHEMA, "name": mode, "precision": precision,
            "arithmetic": arithmetic(mode, precision), "flags": modes.FLAGS, "environment": modes.ENVIRONMENT,
            "sdpa_backend": "math", "attention_chunk": ops.CHUNK, "score_bytes": ops.SCORE_BYTES,
            "compile": False, "cuda_graphs": False}


def mode_list(text, declaration):
    """(label, mode, precision) per `--modes` item: `exact`, `policy`,
    `policy:<precision>`, `reference-bf16`."""
    out = []
    for item in text.split(","):
        mode, _, precision = item.partition(":")
        if mode not in MODES or (precision and mode != "policy"):
            raise ValueError(f"unknown mode {item}; expected {MODES} or policy:<precision>")
        if mode == "policy":
            precision = precision or declaration.precision
            if precision not in PRECISIONS:
                raise ops.refused(f"precision policy {precision}")
            label = "policy" if precision == declaration.precision else f"policy-{precision}"
        else:
            precision, label = None, mode
        out.append((label, mode, precision))
    labels = [label for label, _, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"--modes repeats a mode: {labels}")
    return out


def build(declaration, names, tensors, mode, precision, devices):
    """The declaration's Encoder for one mode over loaded `tensors` (stored
    name -> tensor on devices[0])."""
    numerics = ops.Numerics(arithmetic(mode, precision))
    weights = Weights(declaration, names, tensors.__getitem__, numerics.arithmetic, torch.device(devices[0]))
    return Encoder(declaration, weights, numerics, devices, variants(declaration)).eval()


def check_pins(declaration, checkpoint, verified):
    """The declaration's config pins against the checkpoint's verified
    config files; returns the pinned files it could not check (outside the lock)."""
    unchecked = []
    for pin in declaration.data["checkpoint"].get("config", []):
        if pin["file"] not in verified:
            unchecked.append(pin["file"])
            continue
        value = json.loads((checkpoint.root / pin["file"]).read_bytes())
        for part in [p for p in pin["pointer"].split("/") if p]:
            value = value[part]
        wrong = {key: value.get(key) for key, expected in pin["values"].items() if value.get(key) != expected}
        if wrong:
            raise ValueError(f"{pin['file']}{pin['pointer']} differs from {declaration.name}'s pins: {wrong}")
    return unchecked


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluate an encoder declaration over a workload's media.")
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--adapter", required=True, help="repo path of models/<m>/reference.py (extra shards)")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--lock-variant")
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--modes", default="exact,policy,reference-bf16")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stamps", type=Path, default=stamps.default_cache())
    parser.add_argument("--blocks", action="store_true", help="also tap every transformer block's output")
    parser.add_argument("--tap-bytes", type=int, default=taps.BUDGET, help="tap byte budget per sequence")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    devices = args.devices.split(",")
    modes.prepare_environment(cuda=any(d.startswith("cuda") for d in devices))
    declaration = Declaration(args.declaration)
    resolved = args.declaration.resolve()
    declaration_path = resolved.relative_to(keys.ROOT).as_posix() if resolved.is_relative_to(keys.ROOT) \
        else str(resolved)
    reference_adapter, _ = run.load_adapter(args.adapter)
    model, sequences = workload.load(args.workload)
    if not model == reference_adapter.MODEL == declaration.model:
        raise ValueError(f"workload {model}, adapter {reference_adapter.MODEL}, declaration {declaration.model}")
    sequences = [s for s in sequences if set(m.kind for m in s.media) & set(declaration.kinds)]
    if not sequences:
        raise ValueError(f"the workload has no {'/'.join(declaration.kinds)} media")
    plan = mode_list(args.modes, declaration)
    provider_path, _ = provider(declaration)
    variants(declaration)
    args.output.mkdir(parents=True, exist_ok=False)

    lock = keys.read_lock(args.lock)
    if args.lock_variant is not None:
        lock = lock[args.lock_variant]
    root = args.checkpoint
    first = ["config.json", place.INDEX] if (root / place.INDEX).exists() else ["config.json"]
    verified = stamps.verify(root, lock, first, args.stamps)
    extra = getattr(reference_adapter, "EXTRA_SHARDS", {})
    verified.update(stamps.verify(root, lock, [*place.shard_files(root), *extra], args.stamps))
    checkpoint = place.Checkpoint(root, extra)
    unchecked = check_pins(declaration, checkpoint, verified)
    names = Names(declaration, checkpoint, extra)
    stored = stored_names(declaration, names)
    placement = place.plan([("encoder", checkpoint.stored_bytes(stored))], devices[:1])
    started = time.perf_counter()
    store = place.Store(checkpoint, placement, [place.Unit("encoder", tuple(stored))])
    common = {
        "model": model,
        "checkpoint": keys.checkpoint(lock, sorted({"config.json", *checkpoint.files(stored)})),
        "declaration": {"path": declaration_path, "sha256": declaration.sha256},
        "code": {"files": keys.repo_files([*keys.ENCODER_NUMERICS, *([provider_path] if provider_path else [])])},
        "environment": keys.environment(run.device_label(devices)),
    }
    provenance = {"argv": sys.argv if argv is None else argv, "git_head": run.git_head(),
                  "declaration": common["declaration"], "unchecked_pins": unchecked,
                  "load": {"bytes": store.read_bytes, "seconds": time.perf_counter() - started},
                  "devices": [torch.cuda.get_device_name(d) if d.startswith("cuda") else d for d in devices],
                  "modes": {}}
    for label, mode, precision in plan:
        started = time.perf_counter()
        document = mode_document(mode, precision)
        encoder = build(declaration, names, store.tensors, mode, precision, devices)
        tap_names = tap_map(encoder, args.blocks)
        capture = taps.Taps(encoder, tap_names, args.output / "taps" / label, None, args.tap_bytes)
        entries, seconds = [], {}
        with modes.active(document):
            for position, sequence in enumerate(sequences):
                began = time.perf_counter()
                media = items(sequence, declaration, devices[0])
                with capture.step(sequence.name, sequence.token_ids, [], {m.kind for m in sequence.media}):
                    outputs = []
                    for item in media:
                        with taps.item(item.kind):
                            outputs.append(encoder(item))
                if position == 0:
                    # Outside a step the hooks record nothing.
                    again = [encoder(item) for item in media]
                    if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outputs, again)):
                        raise RuntimeError(f"{label}: {sequence.name} encoded differently on a repeat")
                seconds[sequence.name] = time.perf_counter() - began
                key = keys.encoder_document(
                    media=[m for m in keys.sequence(sequence)["media"] if m["kind"] in declaration.kinds],
                    mode={"name": label, "policy_sha256": keys.sha256(canonical(document))},
                    outputs={"taps": sorted(tap_names)}, **common)
                entries.append({"name": sequence.name, "key_sha256": keys.key_sha256(key), "key": key,
                                "items": [{"kind": item.kind, "rows": list(out.shape), "dtype": str(out.dtype),
                                           "sha256": hashlib.sha256(out.contiguous().view(torch.uint8).cpu()
                                                                    .numpy().tobytes()).hexdigest()}
                                          for item, out in zip(media, outputs)]})
                print(json.dumps({"mode": label, "sequence": sequence.name,
                                  "seconds": round(seconds[sequence.name], 1)}), file=sys.stderr, flush=True)
        capture.remove()
        stamps.confirm(root, verified)
        (args.output / f"{label}.json").write_bytes(canonical(
            {"schema": SCHEMA, "mode": document, "entries": entries}) + b"\n")
        provenance["modes"][label] = {"seconds": time.perf_counter() - started, "sequence_seconds": seconds,
                                      "repeat_bitwise_equal": True, "taps": sorted(tap_names)}
        del encoder
        if any(d.startswith("cuda") for d in devices):
            torch.cuda.empty_cache()
    (args.output / "run.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


if __name__ == "__main__":
    main()
