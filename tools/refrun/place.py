"""Weight placement over the GPUs and bounded checkpoint loading.

Placement splits the adapter's units (decoder layers, embeddings, head) into
contiguous runs over the devices, balanced by stored bytes. Activations hop
devices at unit boundaries through forward pre-hooks.

Loading never maps a shard. Several readers run at once, one shard each:
a reader reads runs of adjacent tensors with pread into its own ring of
pinned staging slots, copies each tensor to its device, and drops the file
range from the page cache with posix_fadvise(DONTNEED). Host RSS stays at
the readers' staging rings plus Python state. Tables too large for the GPUs
(n-gram or engram rows) stay in their shards and are gathered by row
(`HostRows`).
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import ctypes
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import struct
import threading
import time

import torch

from tools.refrun import quant

DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "BOOL": torch.bool,
}
SLOT_BYTES = 256 << 20
SLOTS = 2
READERS = 8
# One cold reader measured fastest at 4 MiB requests (calibration).
REQUEST_BYTES = 4 << 20


@dataclass(frozen=True)
class Entry:
    file: str
    dtype: str
    shape: tuple
    offset: int
    nbytes: int


def read_header(path):
    """The tensor entries of one safetensors file, with absolute offsets."""
    fd = os.open(path, os.O_RDONLY)
    try:
        (length,) = struct.unpack("<Q", os.pread(fd, 8, 0))
        header = json.loads(os.pread(fd, length, 8))
    finally:
        os.close(fd)
    entries = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        shape = tuple(spec["shape"])
        width = DTYPES[spec["dtype"]].itemsize
        if end - begin != math.prod(shape) * width:
            raise ValueError(f"{path}: {name} spans {end - begin} bytes, not {spec['dtype']} {list(shape)}")
        entries[name] = Entry(Path(path).name, spec["dtype"], shape, 8 + length + begin, end - begin)
    return entries


INDEX = "model.safetensors.index.json"


def shard_files(root):
    """The safetensors files of a checkpoint directory: the index's shards, or
    the single `model.safetensors`."""
    index = Path(root) / INDEX
    return sorted(set(json.loads(index.read_text())["weight_map"].values())) \
        if index.exists() else ["model.safetensors"]


class Checkpoint:
    """Tensor directory of a safetensors checkpoint (sharded or single file).
    `extra` maps further safetensors files outside the index (relative paths,
    such as an audio tokenizer's own file) to the prefix their tensor names
    take. `read` lists every file whose bytes it parsed."""

    def __init__(self, root, extra=None):
        self.root = Path(root)
        files = shard_files(self.root)
        extra = dict(extra or {})
        self.read = ["config.json", *([INDEX] if (self.root / INDEX).exists() else []), *files, *extra]
        self.entries = {}
        sources = [(file, "") for file in files] + list(extra.items())
        for file, prefix in sources:
            for name, entry in read_header(self.root / file).items():
                if prefix + name in self.entries:
                    raise ValueError(f"tensor {prefix + name} appears in two shards")
                self.entries[prefix + name] = replace(entry, file=file)
        self.config = json.loads((self.root / "config.json").read_text())

    def pinned(self, file, sha256):
        """The bytes of a file the source lock does not cover (upstream code
        and its configs shipped with the checkpoint), refused unless they
        have the adapter's pinned sha256."""
        data = (self.root / file).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"{file} is not the pinned {sha256}")
        return data

    def stored_bytes(self, names):
        return sum(self.entries[name].nbytes for name in names)

    def files(self, names):
        return sorted({self.entries[name].file for name in names})


@dataclass(frozen=True)
class Unit:
    """A module the adapter places as a whole, and the checkpoint tensors it owns."""
    module: str
    tensors: tuple
    host: bool = False


@dataclass(frozen=True)
class Placement:
    devices: tuple
    units: tuple  # ((module path, device), ...) in model order

    def device(self, module):
        return dict(self.units)[module]

    @property
    def first(self):
        return self.units[0][1]

    @property
    def last(self):
        return self.units[-1][1]

    def document(self):
        return {"devices": list(self.devices), "units": [list(u) for u in self.units]}


def plan(sizes, devices):
    """Contiguous runs of `sizes` ((module, bytes) in model order) over
    `devices`: each unit goes to the device its byte midpoint falls in."""
    total = sum(size for _, size in sizes)
    units, before = [], 0
    for module, size in sizes:
        index = (2 * before + size) * len(devices) // (2 * total) if total else 0
        units.append((module, devices[min(index, len(devices) - 1)]))
        before += size
    return Placement(tuple(devices), tuple(units))


def moved(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(moved(v, device) for v in value)
    if isinstance(value, dict):
        return {k: moved(v, device) for k, v in value.items()}
    return value


def attach(model, placement):
    """Moves every placed unit's inputs to its device before it runs."""
    handles = []
    for module, device in placement.units:
        def hook(_module, args, kwargs, device=torch.device(device)):
            return moved(args, device), moved(kwargs, device)
        handles.append(model.get_submodule(module).register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def read_into(fd, view, offset, request=None):
    """pread exactly len(view) bytes at `offset` into a writable memoryview,
    at most `request` bytes per call."""
    done = 0
    while done < len(view):
        end = len(view) if request is None else min(len(view), done + request)
        count = os.preadv(fd, [view[done:end]], offset + done)
        if count <= 0:
            raise OSError(f"short read at offset {offset + done}")
        done += count


def host_view(tensor):
    return memoryview((ctypes.c_ubyte * tensor.nbytes).from_address(tensor.data_ptr())).cast("B")


def runs(entries, limit):
    """`entries` ((name, Entry) in file order) grouped into runs of adjacent
    tensors read with one pass: each run spans at most `limit` bytes, except
    a single larger tensor, which is read in slot-sized parts."""
    groups, current, end = [], [], None
    for name, entry in entries:
        span = entry.offset + entry.nbytes - current[0][1].offset if current else entry.nbytes
        if current and entry.offset == end and span <= limit:
            current.append((name, entry))
        else:
            if current:
                groups.append(current)
            current = [(name, entry)]
        end = entry.offset + entry.nbytes
    if current:
        groups.append(current)
    return groups


class Staging:
    """One reader's ring of host slots (pinned when any target is a GPU),
    with the copy events that must finish before a slot is reused."""

    def __init__(self, slot_bytes, slots, pinned):
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=pinned) for _ in range(slots)]
        self.events = [[] for _ in range(slots)]
        self.turn = 0

    def next(self):
        index = self.turn % len(self.slots)
        self.turn += 1
        for event in self.events[index]:
            event.synchronize()
        self.events[index] = []
        return index

    def drain(self):
        for events in self.events:
            for event in events:
                event.synchronize()
        self.events = [[] for _ in self.slots]


class Store:
    """Checkpoint tensors resident on their units' devices, loaded once and
    shared by every mode's model.

    `readers` shards are read at once, each by one thread with its own
    staging ring of `slots` x `slot_bytes`. `read_seconds` sums the readers'
    time in pread; `load_seconds` is the wall time of the whole load."""

    def __init__(self, checkpoint, placement, units, slot_bytes=SLOT_BYTES, slots=SLOTS, readers=READERS):
        self.checkpoint = checkpoint
        self.device = {}
        for unit in units:
            if not unit.host:
                for name in unit.tensors:
                    self.device[name] = torch.device(placement.device(unit.module))
        pinned = any(d.type == "cuda" for d in self.device.values())
        self.slot_bytes = slot_bytes
        self.tensors, self.quantized = {}, {}
        self.read_bytes, self.read_seconds = 0, 0.0
        self.lock = threading.Lock()
        by_file = {}
        for name in self.device:
            by_file.setdefault(checkpoint.entries[name].file, []).append(name)
        started = time.perf_counter()
        workers = max(1, min(readers, len(by_file)))
        rings = queue.Queue()
        for _ in range(workers):
            rings.put(Staging(slot_bytes, slots, pinned))

        def load(file):
            staging = rings.get()
            try:
                self.load_file(file, by_file[file], staging)
            finally:
                staging.drain()
                rings.put(staging)
        with ThreadPoolExecutor(workers) as pool:
            for _ in pool.map(load, sorted(by_file)):
                pass
        self.load_seconds = time.perf_counter() - started

    def read(self, fd, view, offset):
        started = time.perf_counter()
        read_into(fd, view, offset, REQUEST_BYTES)
        with self.lock:
            self.read_seconds += time.perf_counter() - started
            self.read_bytes += len(view)

    def load_file(self, file, names, staging):
        """Reads one shard's tensors in file order, run by run."""
        entries = sorted(((n, self.checkpoint.entries[n]) for n in names), key=lambda item: item[1].offset)
        fd = os.open(self.checkpoint.root / file, os.O_RDONLY)
        try:
            for run in runs(entries, self.slot_bytes):
                outs = [(name, entry, torch.empty(entry.nbytes, dtype=torch.uint8, device=self.device[name]))
                        for name, entry in run]
                first = run[0][1].offset
                total = run[-1][1].offset + run[-1][1].nbytes - first
                for start in range(0, total, self.slot_bytes):
                    count = min(self.slot_bytes, total - start)
                    index = staging.next()
                    slot = staging.slots[index]
                    self.read(fd, host_view(slot)[:count], first + start)
                    devices = set()
                    for name, entry, out in outs:
                        # The part of this tensor inside [start, start + count) of the run.
                        begin = entry.offset - first
                        lo, hi = max(begin, start), min(begin + entry.nbytes, start + count)
                        if lo >= hi:
                            continue
                        with torch.cuda.device(out.device) if out.device.type == "cuda" else nullcontext():
                            out[lo - begin:hi - begin].copy_(slot[lo - start:hi - start], non_blocking=True)
                        devices.add(out.device)
                    for device in devices:
                        if device.type == "cuda":
                            with torch.cuda.device(device):
                                event = torch.cuda.Event()
                                event.record()
                            staging.events[index].append(event)
                os.posix_fadvise(fd, first, total, os.POSIX_FADV_DONTNEED)
                for name, entry, out in outs:
                    self.tensors[name] = out.view(DTYPES[entry.dtype]).reshape(entry.shape)
        finally:
            os.close(fd)

    def get(self, name):
        return self.tensors[name]

    def tensor(self, fmt, codes, scale, shape, slot, order=None):
        """The QuantTensor over stored `codes` and `scale`, shared by every
        mode's model. `order` permutes its rows (see QuantTensor.permute_rows)."""
        return self.shared(codes, order, lambda: quant.QuantTensor(
            fmt, self.get(codes), self.get(scale) if scale else None, shape, slot=slot))

    def linear(self, fmt, codes, scale, shape, bias=None, order=None):
        """As `tensor`, as a QuantLinear with an optional stored bias."""
        return self.shared(codes, order, lambda: quant.QuantLinear(
            fmt, self.get(codes), self.get(scale) if scale else None, shape,
            bias=self.get(bias) if bias else None))

    def shared(self, key, order, make):
        if key not in self.quantized:
            tensor = make()
            if order is not None:
                tensor.permute_rows(order)
            self.quantized[key] = tensor
        return self.quantized[key]


class HostRows:
    """Rows of a host-resident table, read on demand without mapping its shard."""

    def __init__(self, checkpoint, name):
        self.entry = checkpoint.entries[name]
        self.path = checkpoint.root / self.entry.file
        self.dtype = DTYPES[self.entry.dtype]
        self.row_bytes = self.entry.nbytes // self.entry.shape[0]

    def rows(self, indices):
        out = torch.empty((len(indices), self.row_bytes), dtype=torch.uint8)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            for i, row in enumerate(indices):
                if not 0 <= row < self.entry.shape[0]:
                    raise IndexError(f"row {row} outside {self.entry.shape[0]}")
                offset = self.entry.offset + row * self.row_bytes
                read_into(fd, host_view(out[i]), offset)
                os.posix_fadvise(fd, offset, self.row_bytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return out.view(self.dtype).reshape(len(indices), *self.entry.shape[1:])
