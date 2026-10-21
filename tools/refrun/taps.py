"""Per-layer taps of a reference forward pass, in the engine's dump format.

An adapter's `taps(model)` maps tap names to the values that carry them: a
module path (that module's output, the first element of a tuple output) or
a `Hook` (a module's input or output, an argument index or keyword, and an
optional transform). `Taps` registers forward hooks for the selected names
once per model; they record only inside `Taps.step`, so a pass outside a
step runs untouched.

Names follow the tap grammar (tools/refrun/README.md, "Taps and tapdiff"):
`<scope>.<site>` with scope `embed`, `L<nn>`, `final`, `enc.<tower>.B<nn>`
or `mtp.L<n>`, and a canonical site
(`SITES`) or a model-specific `x.<...>` site, as in `L03.attn.out`,
`final.norm` or `L05.x.hc.pre`. `enc.<tower>` alone scopes the tower's
output (`enc.vision.out`, the rows merged into the sequence).

Each tap declares its row axis (`Hook.rows`): sequence positions, or the
encoder output rows of the media kinds it serves, in the order the model
calls the encoder for them (`("image", "video")`: one call for a
sequence's images, then one for its videos). An encoder tap fires only for
sequences with media of those kinds, and is expected only there. Each
call's rows are its kind's items, one per row of the `grid_thw` the
encoder call receives (on the tapped module or an ancestor, as the vision
towers take it), split in proportion to each grid row's patches; a call
without a grid is one item. A caller that encodes one media item per call
(tools/refrun/encoder.py) wraps each in `item(kind)` instead: every
encoder tap's rows inside are that item's, the kind's next in the step.
They are recorded as rows `index..` of occurrence `item` of
that `kind`, the engine's encoder row keys.

Each step writes the format of the engine dump
(models/qwen38/bundle/src/dump.rs), so tools/qual/tapdiff.py reads both:

- `<dir>/step<NNNN>/<name>.r0.bin`: the selected rows, row-major, in the
  tapped tensor's own dtype (`bf16`, `f32`, `f64`; integers as `u32`,
  booleans as `u8`);
- `index.jsonl`: `{"name", "rank", "first_row", "rows", "cols", "dtype"}` per
  tap, in the order the hooks fired (program order), plus `"axis":
  "encoder"` for an encoder tap;
- `items.json`: one item, the sequence (`request` is its name) with the
  positions and tokens of the stored rows, a row map, and `encoder`: per
  encoder tap, its rows as `[{"request", "start", "kind", "item", "index",
  "rows"}]`;
- `capture.json` (schema `chopinfer.taps.v1`) before the pass and
  `completion.json` after it.

A forward pass that runs the sequence in chunks (an adapter's chunked
prefill) opens `chunk(start, stop)` around each call: each text tap then
has stop - start rows at positions start..stop, the chunks must arrive in
order, and a tap is written once its rows cover the sequence. A tap whose
rows stop short leaves the step incomplete. Without a chunk, a text tap has
one row per position of the whole sequence.

Rows: every position below 2048, then every 8th, plus the last 16
(`rows`), or an explicit list of positions. Sites are selected by glob
(`x.hc.*`); the default is `DEFAULT_SITES`.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
from typing import Callable

import torch

from tools.model_sdk import diagnostic

SCHEMA = "chopinfer.taps.v1"
SITES = ("in", "in.streams", "attn.in", "attn.q", "attn.k", "attn.v", "attn.core", "attn.out",
         "mlp.in", "router.logits", "router.ids", "router.weights", "moe.routed", "moe.shared",
         "mlp.out", "out", "norm", "logits.topk")
DEFAULT_SITES = ("in", "in.streams", "attn.out", "mlp.in", "router.logits", "router.ids",
                 "router.weights", "mlp.out", "out", "norm")
SCOPE = r"embed|L\d{2,3}|final|enc\.[a-z0-9_]+(?:\.B\d{2,3})?|mtp\.L\d+"
NAME = re.compile(rf"(?P<scope>{SCOPE})\.(?P<site>{'|'.join(map(re.escape, SITES))}|x(?:\.[a-z0-9_]+)+)")
DENSE_ROWS, STRIDE, TAIL = 2048, 8, 16
BUDGET = 4 << 30
# The steps open now, innermost last; `chunk` marks the calls of the last.
OPEN = []


@contextmanager
def chunk(start, stop):
    """Marks the forward calls inside as positions [start, stop) of the open
    step's sequence. Without an open step it does nothing, so adapters call
    it whether or not taps are recorded."""
    step = OPEN[-1] if OPEN else None
    if step is not None:
        step.open_chunk(start, stop)
    try:
        yield
    finally:
        if step is not None:
            step.window = None


@contextmanager
def item(kind):
    """Marks the encoder calls inside as the open step's next media item of
    `kind`. Without an open step it does nothing."""
    step = OPEN[-1] if OPEN else None
    if step is not None:
        step.open_item(kind)
    try:
        yield
    finally:
        if step is not None:
            step.item = None


def site(name):
    """The site of a tap name; refuses a name outside the grammar."""
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"tap name {name!r} is not <scope>.<site>")
    return match["site"]


def rows(length, positions=None):
    """The positions of a `length`-row sequence whose rows are written."""
    if positions is None:
        return [p for p in range(length) if p < DENSE_ROWS or p % STRIDE == 0 or p >= length - TAIL]
    chosen = sorted(set(positions))
    if chosen and (chosen[0] < 0 or chosen[-1] >= length):
        raise ValueError(f"tap rows {chosen[0]}..{chosen[-1]} outside a {length}-row sequence")
    return chosen


@dataclass(frozen=True)
class Hook:
    """Where a tap's value is: `module`'s output or input, element or
    argument `index` (an int, or a keyword name for an input), then
    `transform(value)` if given. With `rows="position"` the value has one
    row per position, with or without a leading batch dimension of 1;
    with a tuple of media kinds its rows are encoder output rows."""
    module: str
    hook: str = "output"
    index: object = 0
    transform: Callable = None
    rows: object = "position"


def hook_of(spec):
    spec = Hook(spec) if isinstance(spec, str) else spec
    if spec.hook not in ("input", "output"):
        raise ValueError(f"hook must be input or output, not {spec.hook!r}")
    if spec.rows != "position" and not (isinstance(spec.rows, tuple) and spec.rows):
        raise ValueError(f"rows must be position or a tuple of media kinds, not {spec.rows!r}")
    return spec


def encode(name, value):
    """(dtype name, bytes) of a [rows, cols] CPU tensor."""
    if value.dtype == torch.bool:
        return "u8", value.to(torch.uint8).numpy().tobytes()
    if value.is_floating_point():
        dtype = {torch.bfloat16: "bf16", torch.float32: "f32", torch.float64: "f64"}[value.dtype]
        return dtype, (value.view(torch.int16) if dtype == "bf16" else value).numpy().tobytes()
    value = value.to(torch.int64)
    if value.numel() and (value.min() < 0 or value.max() >= 1 << 32):
        raise ValueError(f"tap {name} has integers outside u32")
    return "u32", value.numpy().astype("<u4").tobytes()


class Step:
    """One forward pass's tap directory, open until `finish`. `encoder`
    maps each encoder tap to the media kinds of its calls, and `media`
    holds the kinds of the sequence's media items."""

    def __init__(self, directory, name, token_ids, positions, expected, budget, encoder=None, media=()):
        self.dir, self.positions, self.budget, self.name = Path(directory), positions, budget, name
        self.media = set(media)
        # The open `item` context's (kind, item), and each kind's items so far.
        self.item, self.items = None, {}
        self.length, self.expected, self.written, self.bytes = len(token_ids), expected, [], 0
        self.encoder, self.pending, self.segments = encoder or {}, {}, {}
        self.select = None
        # Chunked passes: the open window, the taps seen in it, and per tap
        # its selected rows so far and the next position it must start at.
        self.window, self.seen, self.parts, self.covered = None, set(), {}, {}
        self.tokens = [token_ids[p] for p in positions]
        diagnostic.write_new(self.dir / "capture.json", {
            "schema": SCHEMA, "producer": "refrun", "program": "reference", "ranks": 1,
            "request": name, "seq_len": self.length, "rows": len(positions), "budget_bytes": budget,
            "expected": [{"name": tap, "rank": 0, "sharding": "replicated"} for tap in expected],
            "execution": "eager-synchronized"})
        self.index = (self.dir / "index.jsonl").open("x")

    def open_chunk(self, start, stop):
        if self.window is not None:
            raise ValueError("tap chunks cannot nest")
        if not 0 <= start < stop <= self.length:
            raise ValueError(f"tap chunk {start}..{stop} outside a {self.length}-row sequence")
        self.window, self.seen = (start, stop), set()

    def open_item(self, kind):
        if self.item is not None:
            raise ValueError("media items cannot nest")
        if kind not in self.media:
            raise ValueError(f"a {kind} item in a sequence without {kind} media")
        self.item = (kind, self.items.get(kind, 0))
        self.items[kind] = self.item[1] + 1

    def write(self, name, value, shares=None):
        """Records a tap's value; `shares`, an encoder call's items' shares
        of its rows (None for one item)."""
        if name in self.written or name in self.seen:
            raise ValueError(f"tap {name} recorded twice in one pass")
        value = value.detach()
        if name in self.encoder:
            if name not in self.expected:
                raise ValueError(f"tap {name} fired for a sequence without {'/'.join(self.encoder[name])} media")
            if value.dim() >= 3 and value.shape[0] == 1:
                value = value[0]
            label = shares if self.item is None else self.item
            self.pending.setdefault(name, []).append((value.reshape(value.shape[0], -1).cpu(), label))
            return
        self.flush()
        start, stop = self.window or (0, self.length)
        if value.dim() >= 2 and value.shape[0] == 1 and stop - start != 1:
            value = value[0]
        if value.dim() == 0 or value.shape[0] != stop - start:
            raise ValueError(f"tap {name} has shape {tuple(value.shape)}, not {stop - start} rows")
        if self.window is None:
            if self.select is None or self.select.device != value.device:
                self.select = torch.tensor(self.positions, device=value.device)
            self.store(name, value.index_select(0, self.select).reshape(len(self.positions), -1).cpu())
            return
        self.seen.add(name)
        if self.covered.get(name, 0) != start:
            raise ValueError(f"tap {name} rows {start}..{stop} do not follow its rows 0..{self.covered.get(name, 0)}")
        local = torch.tensor([p - start for p in self.positions if start <= p < stop], device=value.device,
                             dtype=torch.long)
        # Flatten before selecting, so a chunk that selects no row keeps the width.
        self.parts.setdefault(name, []).append(value.reshape(stop - start, -1).index_select(0, local).cpu())
        self.covered[name] = stop
        if stop == self.length:
            self.store(name, torch.cat(self.parts.pop(name)))

    def flush(self):
        """Writes the encoder taps recorded so far, at their place in program
        order: before the next sequence tap."""
        for name, parts in self.pending.items():
            segments = self.split(name, [(part.shape[0], shares) for part, shares in parts])
            self.store(name, torch.cat([part for part, _ in parts]), {"axis": "encoder"})
            self.segments[name] = segments
        self.pending = {}

    def split(self, name, calls):
        """The segments of an encoder tap's `calls`, (rows, item shares or
        None, or the (kind, item) of an `item` context): one call per kind
        it serves that the sequence holds, in its order, each its kind's
        items in order; or each call one item."""
        labeled = [isinstance(label, tuple) for _, label in calls]
        if any(labeled):
            if not all(labeled):
                raise ValueError(f"tap {name} fired both inside and outside media items")
            segments, start, seen = [], 0, {}
            for total, (kind, item) in calls:
                index = seen.get((kind, item), 0)
                segments.append({"request": self.name, "start": start, "kind": kind, "item": item,
                                 "index": index, "rows": total})
                seen[(kind, item)], start = index + total, start + total
            return segments
        kinds = [kind for kind in self.encoder[name] if kind in self.media]
        if len(calls) != len(kinds):
            raise ValueError(f"tap {name} fired {len(calls)} times for media kinds {kinds}")
        segments, start = [], 0
        for kind, (total, shares) in zip(kinds, calls):
            shares = shares or [1]
            whole = sum(shares)
            if not whole or any(share * total % whole for share in shares):
                raise ValueError(f"tap {name}: {kind} items of shares {shares} do not split {total} rows")
            items = [share * total // whole for share in shares]
            for item, count in enumerate(items):
                segments.append({"request": self.name, "start": start, "kind": kind, "item": item,
                                 "index": 0, "rows": count})
                start += count
        return segments

    def store(self, name, value, extra=None):
        dtype, data = encode(name, value)
        self.bytes += len(data)
        if self.bytes > self.budget:
            raise ValueError(f"taps exceed the {self.budget}-byte budget at {name}")
        with (self.dir / f"{name}.r0.bin").open("xb") as stream:
            stream.write(data)
        self.index.write(json.dumps({"name": name, "rank": 0, "first_row": 0, "rows": value.shape[0],
                                     "cols": value.shape[1], "dtype": dtype, **(extra or {})},
                                    separators=(",", ":")) + "\n")
        self.written.append(name)

    def finish(self, error=None):
        """Writes the pending encoder taps, items.json and completion.json;
        returns the error that ended the step, if any."""
        if error is None:
            try:
                self.flush()
            except ValueError as failed:
                error = failed
        self.index.close()
        positions, tokens = self.positions, self.tokens
        diagnostic.write_new(self.dir / "items.json", {
            "program": "reference", "rows": len(positions), "dump_rows": [0, len(positions)],
            "items": [{"request": self.name, "start": 0, "rows": len(positions), "prefix": 0,
                       "seq_len": self.length, "tokens": tokens, "positions": positions}],
            "positions": positions, "token_ids": tokens,
            "row_map": [{"row": row, "item": 0, "position": p, "token": t, "padding": False}
                        for row, (p, t) in enumerate(zip(positions, tokens))],
            "encoder": self.segments})
        missing = [name for name in self.expected if name not in self.written]
        if error is None and self.parts:
            name = next(iter(self.parts))
            error = ValueError(f"tap {name} has rows 0..{self.covered[name]} of {self.length}")
        full = error is None and not missing
        diagnostic.write_new(self.dir / "completion.json", {
            "completion": "full" if full else "incomplete", "expected": len(self.expected),
            "observed": len(self.written), "missing": missing,
            "reason": None if full else (str(error) if error else "taps did not fire")})
        return error


class Taps:
    """The forward hooks of one model's selected taps, and the step counter
    of the directory they write to."""

    def __init__(self, model, tap_map, root, sites=DEFAULT_SITES, budget=BUDGET):
        self.root, self.budget, self.active, self.handles, self.step_index = Path(root), budget, None, [], 0
        self.encoder, self.grid, watched = {}, None, set()
        self.names = [name for name in tap_map
                      if sites is None or any(fnmatchcase(site(name), pattern) for pattern in sites)]
        for name in tap_map:
            site(name)
        if not self.names:
            raise ValueError("no taps selected")
        for name in self.names:
            spec = hook_of(tap_map[name])
            if spec.rows != "position":
                self.encoder[name] = spec.rows
                # The encoder call's grid, on the tapped module or an ancestor.
                parts = spec.module.split(".") if spec.module else []
                watched |= {".".join(parts[:depth]) for depth in range(len(parts) + 1)}
            module = model.get_submodule(spec.module)
            if spec.hook == "output":
                handle = module.register_forward_hook(self.output_hook(name, spec))
            else:
                handle = module.register_forward_pre_hook(self.input_hook(name, spec), with_kwargs=True)
            self.handles.append(handle)
        # A grid holds from its module's call, before the taps' own input
        # hooks, until that call returns, after their output hooks.
        for path in sorted(watched):
            module = model.get_submodule(path)
            self.handles.append(module.register_forward_pre_hook(self.grid_hook, with_kwargs=True, prepend=True))
            self.handles.append(module.register_forward_hook(self.grid_done))

    def grid_hook(self, module, _args, kwargs):
        if kwargs.get("grid_thw") is not None and self.grid is None:
            self.grid = (module, kwargs["grid_thw"])

    def grid_done(self, module, _args, _output):
        if self.grid is not None and self.grid[0] is module:
            self.grid = None

    def record(self, name, spec, value):
        if self.active is None:
            return
        value = spec.transform(value) if spec.transform else value
        if name in self.encoder:
            grid = self.grid[1].reshape(-1, 3) if self.grid is not None else None
            shares = None if grid is None else [int(share) for share in grid.prod(dim=1)]
            self.active.write(name, value, shares)
        else:
            self.active.write(name, value)

    def output_hook(self, name, spec):
        def hook(_module, _args, output):
            if isinstance(output, (tuple, list)):
                output = output[spec.index]
            elif spec.index != 0:
                raise ValueError(f"tap {name}: {spec.module} returns one value, not element {spec.index}")
            self.record(name, spec, output)
        return hook

    def input_hook(self, name, spec):
        def hook(_module, args, kwargs):
            value = kwargs[spec.index] if isinstance(spec.index, str) else args[spec.index]
            self.record(name, spec, value)
        return hook

    @contextmanager
    def step(self, name, token_ids, positions=None, media=()):
        """Records the taps of the passes inside into the next step directory;
        `media` holds the kinds of the sequence's media items."""
        directory = self.root / f"step{self.step_index:04d}"
        self.step_index += 1
        self.root.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        expected = [tap for tap in self.names if tap not in self.encoder or set(self.encoder[tap]) & set(media)]
        self.grid = None
        self.active = Step(directory, name, list(token_ids), rows(len(token_ids), positions), expected,
                           self.budget, self.encoder, media)
        OPEN.append(self.active)
        try:
            yield self.active
        except BaseException as error:
            self.active.finish(error)
            raise
        else:
            error = self.active.finish()
            if error is not None:
                raise error
        finally:
            OPEN.remove(self.active)
            self.active = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
