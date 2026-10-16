"""The generic encoder reference on tiny synthetic declarations, on CPU.

Three declarations cover every shared variant the committed ones use: a
patch tower under f32-pairs.v1 (patch linear, learned position grid, 2D
rotary, LayerNorm blocks, padded GELU MLP, merger), a legacy patch tower
(RMSNorm, GQA with a window, key-0 sinks and a full-attention override,
column-reordered blocks, SwiGLU, temporal grids) and a legacy audio-like
encoder (strided convolutions over segments, causal and causal-window
attention, a carry, a group join and a grouped transformer). `exact` must
match a direct f64 implementation written here from the definitions, block
by block; the other modes must stay near it with their own output dtype.
Needs torch; skipped without it.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import torch.nn.functional as F
    from tools.refrun import encoder, encoder_ops as ops, keys, modes, place, stamps, workload
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]

PATCH = {
    "schema": "chopinfer.encoder.v1", "model": "tiny", "name": "tiny-vision", "precision": "f32-pairs.v1",
    "checkpoint": {"prefix": "vis.", "config": [{"file": "config.json", "pointer": "/vision", "values": {"depth": 2}}]},
    "input": {"recipes": [{"tag": "image.patch-grid.v1"}], "patchify": "merge-blocks.v1", "payload": "f32",
              "columns": 12, "capacity_rows": 1024},
    "sequences": {"kind": "item"}, "image_layout": "view-order.v1",
    "stages": [
        {"op": "linear.pairs.v1", "out": 16, "bias": True,
         "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]}, "bias": "patch.bias"}},
        {"op": "position.learned-grid.bilinear-align-corners.f32.v1", "table": [3, 3],
         "weights": {"table": "pos.weight"}},
        {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 16,
         "norm": {"op": "norm.layer-f32.v1", "eps": 1e-6},
         "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 2, "kv_heads": 2, "head_dim": 8,
                       "qkv_bias": True, "proj_bias": True, "mask": "full"},
         "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 2, "theta": 10000, "style": "neox",
                    "table": "table.hf-f32-angle.v1"},
         "mlp": {"op": "mlp.gelu.v1", "hidden": 20, "pad_to": 24, "gelu": "tanh", "bias": True},
         "weights": {
             "norm1.weight": "blocks.{i}.n1.weight", "norm1.bias": "blocks.{i}.n1.bias",
             "qkv.weight": "blocks.{i}.qkv.weight", "qkv.bias": "blocks.{i}.qkv.bias",
             "proj.weight": "blocks.{i}.proj.weight", "proj.bias": "blocks.{i}.proj.bias",
             "norm2.weight": "blocks.{i}.n2.weight", "norm2.bias": "blocks.{i}.n2.bias",
             "up.weight": {"tensor": "blocks.{i}.fc1.weight", "transforms": [{"pad_rows": 24}]},
             "up.bias": {"tensor": "blocks.{i}.fc1.bias", "transforms": [{"pad_rows": 24}]},
             "down.weight": {"tensor": "blocks.{i}.fc2.weight", "transforms": [{"pad_cols": 24}]},
             "down.bias": "blocks.{i}.fc2.bias"}},
        {"op": "norm.layer-f32.v1", "eps": 1e-6, "weights": {"weight": "merger.n.weight", "bias": "merger.n.bias"}},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.pairs.v1", "out": 24, "bias": True, "gelu": "erf",
         "weights": {"weight": "merger.fc1.weight", "bias": "merger.fc1.bias"}},
        {"op": "linear.pairs.v1", "out": 8, "bias": True, "output": "bf16",
         "weights": {"weight": "merger.fc2.weight", "bias": "merger.fc2.bias"}},
    ],
    "output": {"op": "layout.tokens.v1", "hidden": 8, "positions": "grid-3axis.v1", "media_mask": False},
    "parallel": ["replicated"],
}

WINDOWED = {
    "schema": "chopinfer.encoder.v1", "model": "tiny", "name": "tiny-windowed", "precision": "bf16-legacy.v1",
    "checkpoint": {"prefix": "win.", "config": []},
    "input": {"recipes": [{"tag": "video.timestamped-grids.v1"}], "patchify": "merge-blocks.v1", "payload": "bf16",
              "columns": 12, "capacity_rows": 1024},
    "sequences": {"kind": "temporal-grid"}, "image_layout": "program-order-stacked.v1",
    "stages": [
        {"op": "linear.bf16.v1", "out": 16, "bias": False,
         "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]}}},
        {"op": "transformer.pre-norm.v1", "blocks": 4, "width": 16,
         "norm": {"op": "norm.rms.v1", "eps": 1e-6},
         "attention": {"op": "attention.varlen.bf16.v1", "q_heads": 4, "kv_heads": 2, "head_dim": 4,
                       "qkv_bias": True, "proj_bias": True, "mask": {"window": [2, 3]}, "sink": "key0"},
         "overrides": [{"blocks": "0", "attention": {"mask": "full", "sink": "none"}}],
         "reorder": [{"blocks": "2-3", "reorder": {"op": "reorder.merge-group-columns.v1"}}],
         "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 1, "theta": 10000, "style": "neox",
                    "table": "table.hf-f32-angle.v1"},
         "mlp": {"op": "mlp.swiglu.v1", "hidden": 12, "bias": True, "round_linear": True},
         "weights": {
             "norm1.weight": "blocks.{i}.n1.weight", "qkv.weight": "blocks.{i}.qkv.weight",
             "qkv.bias": "blocks.{i}.qkv.bias",
             "sinks": {"tensor": "blocks.{i}.sinks", "transforms": ["widen_f32"], "blocks": "1-3"},
             "proj.weight": "blocks.{i}.proj.weight", "proj.bias": "blocks.{i}.proj.bias",
             "norm2.weight": "blocks.{i}.n2.weight",
             "gate_up.weight": {"concat_rows": ["blocks.{i}.gate.weight", "blocks.{i}.up.weight"]},
             "gate_up.bias": {"concat_rows": ["blocks.{i}.gate.bias", "blocks.{i}.up.bias"]},
             "down.weight": "blocks.{i}.down.weight", "down.bias": "blocks.{i}.down.bias"}},
        {"op": "norm.rms.v1", "eps": 1e-6, "weights": {"weight": "merger.n.weight"}},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.bf16-round-linear.v1", "out": 32, "bias": False, "gelu": "erf",
         "weights": {"weight": "merger.fc1.weight"}},
        {"op": "linear.bf16.v1", "out": 8, "bias": False, "weights": {"weight": "merger.fc2.weight"}},
    ],
    "output": {"op": "layout.tokens.v1", "hidden": 8, "positions": "sequential", "media_mask": False},
    "parallel": ["replicated"],
}

SEGMENTED = {
    "schema": "chopinfer.encoder.v1", "model": "tiny", "name": "tiny-audio", "precision": "bf16-legacy.v1",
    "checkpoint": {"files": {"aud.": "model.safetensors"}, "config": []},
    "input": {"recipes": [{"tag": "audio.log-mel.v1"}], "payload": "f32", "columns": 4, "capacity_rows": 1024},
    "sequences": {"kind": "segments", "input_rows": 10}, "image_layout": "program-order-stacked.v1",
    "stages": [
        {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 1, "pad": [1, 1], "out": 8, "bias": True, "gelu": "erf",
         "round_linear": True,
         "weights": {"weight": {"tensor": "aud.conv1.weight", "transforms": ["taps_first"]}, "bias": "aud.conv1.bias"}},
        {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 2, "pad": [1, 1], "out": 8, "bias": True, "gelu_rows": "erf",
         "weights": {"weight": {"tensor": "aud.conv2.weight", "transforms": ["taps_first"]}, "bias": "aud.conv2.bias"}},
        {"op": "transformer.pre-norm.v1", "blocks": 3, "width": 8,
         "norm": {"op": "norm.layer.v1", "eps": 1e-5},
         "attention": {"op": "attention.varlen.bf16.v1", "q_heads": 2, "kv_heads": 2, "head_dim": 4,
                       "qkv_bias": True, "proj_bias": True, "mask": "causal"},
         "overrides": [{"blocks": "even", "attention": {"mask": {"causal_window": 2}}}],
         "carry": [{"save_after": 0, "add_after": 2}],
         "mlp": {"op": "mlp.gelu.v1", "hidden": 16, "gelu": "erf", "bias": True, "round_linear": True},
         "weights": {
             "norm1.weight": "aud.l{i}.n1.weight", "norm1.bias": "aud.l{i}.n1.bias",
             "qkv.weight": {"concat_rows": ["aud.l{i}.q.weight", "aud.l{i}.k.weight", "aud.l{i}.v.weight"]},
             "qkv.bias": {"concat_rows": ["aud.l{i}.q.bias", {"zeros": 8}, "aud.l{i}.v.bias"]},
             "proj.weight": "aud.l{i}.o.weight", "proj.bias": "aud.l{i}.o.bias",
             "norm2.weight": "aud.l{i}.n2.weight", "norm2.bias": "aud.l{i}.n2.bias",
             "up.weight": "aud.l{i}.fc1.weight", "up.bias": "aud.l{i}.fc1.bias",
             "down.weight": "aud.l{i}.fc2.weight", "down.bias": "aud.l{i}.fc2.bias"}},
        {"op": "norm.layer.v1", "eps": 1e-5, "weights": {"weight": "aud.n.weight", "bias": "aud.n.bias"}},
        {"op": "conv1d.im2col.v1", "kernel": 2, "stride": 2, "pad": [0, 1], "out": 8, "bias": False, "gelu": "erf",
         "round_linear": True,
         "weights": {"weight": {"tensor": "aud.down.weight", "transforms": ["taps_first"]}}},
        {"op": "group.repeat-last.v1", "rows": 4},
        {"op": "transformer.pre-norm.v1", "blocks": 1, "width": 8, "sequences": {"kind": "groups", "rows": 4},
         "norm": {"op": "norm.rms.v1", "eps": 1e-6},
         "attention": {"op": "attention.varlen.bf16.v1", "q_heads": 2, "kv_heads": 2, "head_dim": 4,
                       "qkv_bias": True, "proj_bias": True, "mask": "full"},
         "mlp": {"op": "mlp.swiglu.v1", "hidden": 8, "bias": True, "round_linear": True},
         "weights": {
             "norm1.weight": "aud.g{i}.n1.weight", "qkv.weight": "aud.g{i}.qkv.weight",
             "qkv.bias": "aud.g{i}.qkv.bias", "proj.weight": "aud.g{i}.o.weight", "proj.bias": {"zeros": 8},
             "norm2.weight": "aud.g{i}.n2.weight", "gate_up.weight": "aud.g{i}.gu.weight",
             "gate_up.bias": {"zeros": 16}, "down.weight": "aud.g{i}.d.weight", "down.bias": {"zeros": 8}}},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.bf16.v1", "out": 8, "bias": False, "weights": {"weight": "aud.proj.weight"}},
    ],
    "output": {"op": "layout.tokens.v1", "hidden": 8, "positions": "sequential", "media_mask": False},
    "parallel": ["replicated"],
}


def shapes():
    """Every stored tensor of the three declarations, [name] -> shape."""
    out = {"vis.patch.weight": (16, 3, 2, 2), "vis.patch.bias": (16,), "vis.pos.weight": (9, 16),
           "vis.merger.n.weight": (16,), "vis.merger.n.bias": (16,), "vis.merger.fc1.weight": (24, 64),
           "vis.merger.fc1.bias": (24,), "vis.merger.fc2.weight": (8, 24), "vis.merger.fc2.bias": (8,),
           "win.patch.weight": (16, 12), "win.merger.n.weight": (16,), "win.merger.fc1.weight": (32, 64),
           "win.merger.fc2.weight": (8, 32),
           "aud.conv1.weight": (8, 4, 3), "aud.conv1.bias": (8,), "aud.conv2.weight": (8, 8, 3),
           "aud.conv2.bias": (8,), "aud.n.weight": (8,), "aud.n.bias": (8,), "aud.down.weight": (8, 8, 2),
           "aud.proj.weight": (8, 32)}
    for i in range(2):
        b = f"vis.blocks.{i}."
        out |= {b + "n1.weight": (16,), b + "n1.bias": (16,), b + "qkv.weight": (48, 16), b + "qkv.bias": (48,),
                b + "proj.weight": (16, 16), b + "proj.bias": (16,), b + "n2.weight": (16,), b + "n2.bias": (16,),
                b + "fc1.weight": (20, 16), b + "fc1.bias": (20,), b + "fc2.weight": (16, 20), b + "fc2.bias": (16,)}
    for i in range(4):
        b = f"win.blocks.{i}."
        out |= {b + "n1.weight": (16,), b + "qkv.weight": (32, 16), b + "qkv.bias": (32,),
                b + "proj.weight": (16, 16), b + "proj.bias": (16,), b + "n2.weight": (16,),
                b + "gate.weight": (12, 16), b + "up.weight": (12, 16), b + "gate.bias": (12,), b + "up.bias": (12,),
                b + "down.weight": (16, 12), b + "down.bias": (16,)}
        if i:
            out[b + "sinks"] = (4,)
    for i in range(3):
        b = f"aud.l{i}."
        out |= {b + "n1.weight": (8,), b + "n1.bias": (8,), b + "q.weight": (8, 8), b + "k.weight": (8, 8),
                b + "v.weight": (8, 8), b + "q.bias": (8,), b + "v.bias": (8,), b + "o.weight": (8, 8),
                b + "o.bias": (8,), b + "n2.weight": (8,), b + "n2.bias": (8,), b + "fc1.weight": (16, 8),
                b + "fc1.bias": (16,), b + "fc2.weight": (8, 16), b + "fc2.bias": (8,)}
    out |= {"aud.g0.n1.weight": (8,), "aud.g0.qkv.weight": (24, 8), "aud.g0.qkv.bias": (24,),
            "aud.g0.o.weight": (8, 8), "aud.g0.n2.weight": (8,), "aud.g0.gu.weight": (16, 8),
            "aud.g0.d.weight": (8, 8)}
    return out


def write_checkpoint(root, seed=0):
    """config.json, model.safetensors (BF16 weights) and checkpoint-files.json."""
    from safetensors.torch import save_file
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in shapes().items():
        scale = 0.1 if name.endswith(("n1.weight", "n2.weight", "n.weight")) else 0.4
        value = torch.randn(shape, generator=generator) * scale
        if name.endswith(("n1.weight", "n2.weight", "n.weight")):
            value += 1.0
        tensors[name] = value.to(torch.bfloat16)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"vision": {"depth": 2}}))
    lock = {file: {"byte_length": (root / file).stat().st_size,
                   "sha256": hashlib.sha256((root / file).read_bytes()).hexdigest()}
            for file in ("config.json", "model.safetensors")}
    (root / "checkpoint-files.json").write_text(json.dumps(lock))
    return tensors


def setUpModule():
    if torch is None:
        return
    modes.prepare_environment(cuda=False)
    global RACY_NS
    RACY_NS, stamps.RACY_NS = stamps.RACY_NS, 100_000_000


def tearDownModule():
    if torch is not None:
        stamps.RACY_NS = RACY_NS


# A direct f64 implementation of the three declarations ----------------------


def merge_positions(t, h, w):
    out = []
    for _ in range(t):
        for bh in range(h // 2):
            for bw in range(w // 2):
                for dh in range(2):
                    for dw in range(2):
                        out.append((2 * bh + dh, 2 * bw + dw))
    return out


def layer_norm(x, w, b, eps):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def rms_norm(x, w, eps):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rotary(x, heads, angles):
    """NeoX rotation of every head of x [n, heads * d] by angles [n, d/2]."""
    n, d = x.shape[0], x.shape[1] // heads
    t = x.view(n, heads, d)
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    a, b = t[..., :d // 2], t[..., d // 2:]
    return torch.cat([a * c - b * s, b * c + a * s], -1).reshape(n, -1)


def attend(q, k, v, heads, kv_heads, allowed, sink=None):
    """One sequence: per query head, softmax over the allowed keys, a key-0
    sink logit added where key 0 is allowed."""
    n, d = q.shape[0], q.shape[1] // heads
    out = torch.zeros_like(q)
    for h in range(heads):
        g = h // (heads // kv_heads)
        scores = q[:, h * d:(h + 1) * d] @ k[:, g * d:(g + 1) * d].T * d ** -0.5
        if sink is not None:
            scores[:, 0] += sink[h]
        scores = scores.masked_fill(~allowed, float("-inf"))
        out[:, h * d:(h + 1) * d] = torch.softmax(scores, -1) @ v[:, g * d:(g + 1) * d]
    return out


def mask_of(kind, n):
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    return {"full": torch.ones(n, n, dtype=torch.bool), "causal": j <= i,
            "window": (j >= i - 2) & (j <= i + 3), "causal2": (j >= i - 2) & (j <= i)}[kind]


class Direct:
    """The tiny declarations in f64 from their definitions, recording each
    block's output in the rows' own order."""

    def __init__(self, tensors):
        self.w = {name: t.double() for name, t in tensors.items()}
        self.blocks = []

    def linear(self, x, name, bias=True):
        return F.linear(x, self.w[name + ".weight"], self.w[name + ".bias"] if bias else None)

    def patch(self, pixels, grid):
        w = self.w
        t, h, width = grid
        x = F.linear(pixels.double(), w["vis.patch.weight"].reshape(16, -1), w["vis.patch.bias"])
        # The learned grid resampled by F.interpolate (bilinear, align_corners).
        table = w["vis.pos.weight"].T.reshape(1, 16, 3, 3)
        grid_rows = F.interpolate(table, size=(h, width), mode="bilinear", align_corners=True)[0]
        positions = merge_positions(t, h, width)
        x = x + torch.stack([grid_rows[:, r, c] for r, c in positions])
        inverse = 10000.0 ** (-torch.arange(2, dtype=torch.float64) / 2)
        pos = torch.tensor(positions, dtype=torch.float64)
        angles = torch.cat([pos[:, :1] * inverse, pos[:, 1:] * inverse], -1)
        for i in range(2):
            b = f"vis.blocks.{i}."
            y = layer_norm(x, w[b + "n1.weight"], w[b + "n1.bias"], 1e-6)
            qkv = self.linear(y, b + "qkv")
            q, k, v = qkv[:, :16], qkv[:, 16:32], qkv[:, 32:]
            q, k = rotary(q, 2, angles), rotary(k, 2, angles)
            x = x + self.linear(attend(q, k, v, 2, 2, mask_of("full", x.shape[0])), b + "proj")
            y = layer_norm(x, w[b + "n2.weight"], w[b + "n2.bias"], 1e-6)
            x = x + self.linear(F.gelu(self.linear(y, b + "fc1"), approximate="tanh"), b + "fc2")
            self.blocks.append(x)
        x = layer_norm(x, w["vis.merger.n.weight"], w["vis.merger.n.bias"], 1e-6).reshape(-1, 64)
        return self.linear(F.gelu(self.linear(x, "vis.merger.fc1")), "vis.merger.fc2")

    def windowed(self, pixels, grid):
        w = self.w
        t, h, width = grid
        x = pixels.double() @ w["win.patch.weight"].T
        frame = h * width
        columns = []
        for f in range(t):
            for bw in range(width // 2):
                for bh in range(h // 2):
                    first = f * frame + (bh * (width // 2) + bw) * 4
                    columns.extend(range(first, first + 4))
        columns = torch.tensor(columns)
        pos = torch.tensor(merge_positions(t, h, width), dtype=torch.float64)
        angles = torch.cat([pos[:, :1], pos[:, 1:]], -1)
        for i in range(4):
            b = f"win.blocks.{i}."
            if i == 2:
                x, angles = x[columns], angles[columns]
            y = rms_norm(x, w[b + "n1.weight"], 1e-6)
            qkv = self.linear(y, b + "qkv")
            q, k, v = rotary(qkv[:, :16], 4, angles), rotary(qkv[:, 16:24], 2, angles), qkv[:, 24:]
            parts = []
            for f in range(t):
                rows = slice(f * frame, (f + 1) * frame)
                allowed = mask_of("full" if i == 0 else "window", frame)
                parts.append(attend(q[rows], k[rows], v[rows], 4, 2, allowed, None if i == 0 else w[b + "sinks"]))
            x = x + self.linear(torch.cat(parts), b + "proj")
            y = rms_norm(x, w[b + "n2.weight"], 1e-6)
            x = x + self.linear(F.silu(self.linear(y, b + "gate")) * self.linear(y, b + "up"), b + "down")
            self.blocks.append(x[torch.argsort(columns)] if i >= 2 else x)
        x = x[torch.argsort(columns)]
        x = rms_norm(x, w["win.merger.n.weight"], 1e-6).reshape(-1, 64)
        return F.gelu(x @ w["win.merger.fc1.weight"].T) @ w["win.merger.fc2.weight"].T

    def conv(self, x, name, stride, pad, bias=True):
        padded = torch.cat([x.new_zeros(pad[0], x.shape[1]), x, x.new_zeros(pad[1], x.shape[1])])
        y = F.conv1d(padded.T[None], self.w[name + ".weight"], self.w[name + ".bias"] if bias else None,
                     stride=stride)
        return y[0].T

    def audio(self, mel):
        w = self.w
        codes = []
        for start in range(0, mel.shape[0], 10):
            x = mel[start:start + 10].double()
            x = F.gelu(self.conv(x, "aud.conv1", 1, (1, 1)))
            x = F.gelu(self.conv(x, "aud.conv2", 2, (1, 1)))
            n = x.shape[0]
            saved = None
            for i in range(3):
                b = f"aud.l{i}."
                y = layer_norm(x, w[b + "n1.weight"], w[b + "n1.bias"], 1e-5)
                q, k, v = self.linear(y, b + "q"), F.linear(y, w[b + "k.weight"]), self.linear(y, b + "v")
                x = x + self.linear(attend(q, k, v, 2, 2, mask_of("causal2" if i % 2 == 0 else "causal", n)),
                                    b + "o")
                y = layer_norm(x, w[b + "n2.weight"], w[b + "n2.bias"], 1e-5)
                x = x + self.linear(F.gelu(self.linear(y, b + "fc1")), b + "fc2")
                self.blocks.append(x)
                if i == 0:
                    saved = x
            x = layer_norm(x + saved, w["aud.n.weight"], w["aud.n.bias"], 1e-5)
            codes.append(F.gelu(self.conv(x, "aud.down", 2, (0, 1), bias=False)))
        x = torch.cat(codes)
        x = x[torch.arange(-(-x.shape[0] // 4) * 4).clamp(max=x.shape[0] - 1)]
        y = rms_norm(x, w["aud.g0.n1.weight"], 1e-6)
        qkv = self.linear(y, "aud.g0.qkv")
        parts = [attend(qkv[r:r + 4, :8], qkv[r:r + 4, 8:16], qkv[r:r + 4, 16:], 2, 2, mask_of("full", 4))
                 for r in range(0, x.shape[0], 4)]
        x = x + F.linear(torch.cat(parts), w["aud.g0.o.weight"])
        y = rms_norm(x, w["aud.g0.n2.weight"], 1e-6)
        gate, up = F.linear(y, w["aud.g0.gu.weight"]).chunk(2, -1)
        x = x + F.linear(F.silu(gate) * up, w["aud.g0.d.weight"])
        self.blocks.append(x)
        return F.linear(x.reshape(-1, 32), w["aud.proj.weight"])


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@unittest.skipIf(torch is None, "needs torch")
class TinyDeclarations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.tensors = write_checkpoint(cls.root / "checkpoint")
        cls.checkpoint = place.Checkpoint(cls.root / "checkpoint")
        generator = torch.Generator().manual_seed(1)
        cls.pixels = torch.randn((24, 12), generator=generator)
        cls.frames = torch.randn((2 * 16, 12), generator=generator).to(torch.bfloat16)
        cls.mel = torch.randn((27, 4), generator=generator)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def encode(self, data, mode, item, precision=None, blocks=None):
        path = self.root / f"{data['name']}.json"
        path.write_text(json.dumps(data))
        declaration = encoder.Declaration(path)
        names = encoder.Names(declaration, self.checkpoint, {})
        if mode == "policy":
            precision = precision or declaration.precision
        model = encoder.build(declaration, names, self.tensors, mode, precision, ["cpu"])
        recorded = []
        if blocks is not None:
            for name, hook in encoder.tap_map(model).items():
                if ".B" in name:
                    model.get_submodule(hook.module).register_forward_hook(
                        lambda _m, _a, out, name=name: recorded.append((name, out)))
        with modes.active(encoder.mode_document(mode, precision)):
            out = model(item)
        if blocks is not None:
            blocks.extend(recorded)
        return out

    def check_exact(self, data, item, direct_out, direct):
        recorded = []
        out = self.encode(data, "exact", item, blocks=recorded)
        self.assertEqual(out.dtype, torch.float64)
        self.assertLess(relative(out, direct_out), 1e-6)
        self.assertEqual(len(recorded), len(direct.blocks))
        for (name, ours), theirs in zip(recorded, direct.blocks):
            self.assertLess(relative(ours, theirs), 1e-6, name)

    def test_patch_tower_exact_matches_the_definitions(self):
        direct = Direct(self.tensors)
        item = encoder.Item("image", self.pixels, (1, 4, 6))
        self.check_exact(PATCH, item, direct.patch(self.pixels, (1, 4, 6)), direct)

    def test_windowed_tower_exact_matches_the_definitions(self):
        direct = Direct(self.tensors)
        item = encoder.Item("video", self.frames, (2, 4, 4))
        self.check_exact(WINDOWED, item, direct.windowed(self.frames, (2, 4, 4)), direct)

    def test_segmented_encoder_exact_matches_the_definitions(self):
        direct = Direct(self.tensors)
        item = encoder.Item("audio", self.mel)
        self.check_exact(SEGMENTED, item, direct.audio(self.mel), direct)

    def test_modes_stay_near_exact_in_their_own_dtype(self):
        cases = [(PATCH, encoder.Item("image", self.pixels, (1, 4, 6))),
                 (WINDOWED, encoder.Item("video", self.frames, (2, 4, 4))),
                 (SEGMENTED, encoder.Item("audio", self.mel))]
        for data, item in cases:
            exact = self.encode(data, "exact", item)
            for mode, precision, bound in [("policy", None, 0.03), ("reference-bf16", None, 0.05),
                                           ("policy", "f32-pairs.v1", 0.01)]:
                out = self.encode(data, mode, item, precision)
                self.assertEqual(out.dtype, torch.bfloat16, (data["name"], mode))
                error = relative(out, exact)
                self.assertLess(error, bound, (data["name"], mode, precision))
                self.assertGreater(error, 0, (data["name"], mode, precision))
                again = self.encode(data, mode, item, precision)
                self.assertTrue(torch.equal(out.view(torch.int16), again.view(torch.int16)))

    def test_pairs_round_only_where_the_policy_says(self):
        # Under f32-pairs.v1 a tower of BF16-exact payload stays within pair
        # rounding of exact before its output rounding.
        item = encoder.Item("image", self.pixels.to(torch.bfloat16).float(), (1, 4, 6))
        recorded = []
        exact = self.encode(PATCH, "exact", item, blocks=recorded)
        pairs = []
        self.encode(PATCH, "policy", item, blocks=pairs)
        for (_, ours), theirs in zip(pairs, [out for _, out in recorded]):
            self.assertEqual(ours.dtype, torch.float32)
            self.assertLess(relative(ours, theirs), 1e-4)
        del exact

    def test_the_unrounded_tap_precedes_either_output_rounding(self):
        # f32-pairs.v1 rounds its output in the last linear (`output: bf16`)
        # or in a final cast; the tap reads the F32 rows before either.
        cast = copy.deepcopy(PATCH)
        cast["name"] = "tiny-vision-cast"
        cast["stages"][-1].pop("output")
        cast["stages"].append({"op": "cast.bf16.v1"})
        item = encoder.Item("image", self.pixels, (1, 4, 6))
        outputs = []
        for data in (PATCH, cast):
            path = self.root / f"{data['name']}.json"
            path.write_text(json.dumps(data))
            declaration = encoder.Declaration(path)
            model = encoder.build(declaration, encoder.Names(declaration, self.checkpoint, {}), self.tensors,
                                  "policy", declaration.precision, ["cpu"])
            hook = encoder.tap_map(model)["enc.vision.x.unrounded"]
            captured = []
            model.get_submodule(hook.module).register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
            with modes.active(encoder.mode_document("policy", declaration.precision)):
                out = model(item)
            self.assertEqual(captured[0].dtype, torch.float32, data["name"])
            self.assertTrue(torch.equal(captured[0].to(torch.bfloat16), out), data["name"])
            outputs.append((captured[0], out))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))

    def test_unimplemented_variants_are_refused_by_name(self):
        data = copy.deepcopy(PATCH)
        data["stages"][4] = {"op": "merge.space-to-depth.v1", "factor": 2, "pad": "zero"}
        path = self.root / "refused.json"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(NotImplementedError, "`merge.space-to-depth.v1`"):
            encoder.variants(encoder.Declaration(path))
        fixture = ROOT / "crates/encoder-spec/tests/fixtures/dsv41-vision.json"
        with self.assertRaisesRegex(NotImplementedError, "`patchify row-major.v1`"):
            encoder.Declaration(fixture)
        data = json.loads(fixture.read_text())
        data["input"]["patchify"] = "merge-blocks.v1"
        data["output"]["op"] = "layout.tokens.v1"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(NotImplementedError, "`merge.space-to-depth.v1`"):
            encoder.variants(encoder.Declaration(path))
        data["stages"] = [{"op": "acme:quantize.codes.v1"}]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(NotImplementedError, "`acme:quantize.codes.v1`"):
            encoder.variants(encoder.Declaration(path))

    def test_block_sets_and_modes(self):
        self.assertEqual(encoder.block_set("even", 5), {0, 2, 4})
        self.assertEqual(encoder.block_set("1-3,7", 8), {1, 2, 3, 7})
        with self.assertRaises(ValueError):
            encoder.block_set("1-8", 8)
        path = self.root / "modes.json"
        path.write_text(json.dumps(WINDOWED))
        declaration = encoder.Declaration(path)
        self.assertEqual(encoder.mode_list("exact,policy,policy:f32-pairs.v1,reference-bf16", declaration),
                         [("exact", "exact", None), ("policy", "policy", "bf16-legacy.v1"),
                          ("policy-f32-pairs.v1", "policy", "f32-pairs.v1"),
                          ("reference-bf16", "reference-bf16", None)])
        for bad in ("f32", "exact:f32-pairs.v1", "policy,policy", "policy:f16.v1"):
            with self.assertRaises((ValueError, NotImplementedError)):
                encoder.mode_list(bad, declaration)


@unittest.skipIf(torch is None, "needs torch")
class Arithmetic(unittest.TestCase):
    def test_pairs_carry_sixteen_bits(self):
        x = torch.randn(4096, generator=torch.Generator().manual_seed(2))
        hi, lo = ops.halves(x)
        self.assertTrue(torch.equal(hi, ops.bf16(hi)))
        self.assertTrue(torch.equal(lo, ops.bf16(lo)))
        self.assertLessEqual(float(((x - hi - lo).abs() / x.abs()).max()), 2.0 ** -16)
        exact = x.to(torch.bfloat16).float()
        self.assertTrue(torch.equal(ops.pair(exact), exact))

    def test_legacy_linear_rounds_before_its_activation_only_when_asked(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(5, 7, generator=generator).to(torch.bfloat16)
        weight = torch.randn(6, 7, generator=generator).to(torch.bfloat16)
        accumulator = F.linear(x.float(), weight.float())
        legacy = ops.Numerics("legacy")
        fused = ops.Gemm(legacy, weight, activation="erf")(x)
        rounded = ops.Gemm(legacy, weight, activation="erf", round_linear=True)(x)
        self.assertTrue(torch.equal(fused, F.gelu(accumulator).to(torch.bfloat16)))
        self.assertTrue(torch.equal(rounded, F.gelu(accumulator.to(torch.bfloat16).float()).to(torch.bfloat16)))

    def test_pair_attention_terms(self):
        generator = torch.Generator().manual_seed(4)
        q, k, v = (torch.randn(40, 8, generator=generator) for _ in range(3))
        exact = ops.attention(q.double(), k.double(), v.double(), [40], 2, 2, 4, "full", None, 0.5,
                              ops.Numerics("f64"), ["cpu"])
        pairs = ops.attention(q, k, v, [40], 2, 2, 4, "full", None, 0.5, ops.Numerics("pairs"), ["cpu"])
        self.assertEqual(pairs.dtype, torch.float32)
        self.assertLess(relative(pairs, exact), 1e-4)
        self.assertGreater(relative(pairs, exact), 1e-7)

    def test_legacy_probabilities_enter_as_bf16_terms(self):
        # The BF16 varlen kernel splits each probability into BF16 terms:
        # two at 128 to 32,767 visible keys, three below 128.
        generator = torch.Generator().manual_seed(1)
        for rows in (128, 40):
            q, k, v = (torch.randn(rows, 4, generator=generator).to(torch.bfloat16) for _ in range(3))
            got = ops.attention(q, k, v, [rows], 1, 1, 4, "full", None, 0.5, ops.Numerics("legacy"), ["cpu"])
            scores = (q.float() @ k.float().T) * 0.5
            p = (scores - scores.max(-1, keepdim=True).values).exp()
            first = ops.bf16(p)
            second = ops.bf16(p - first)
            terms = first + second if rows >= 128 else first + second + ops.bf16(p - first - second)
            want = ((terms @ v.float()) / p.sum(-1, keepdim=True)).to(torch.bfloat16)
            self.assertTrue(torch.equal(got, want), rows)
            unsplit = ((p @ v.float()) / p.sum(-1, keepdim=True)).to(torch.bfloat16)
            if rows == 128:
                self.assertFalse(torch.equal(got, unsplit))
        self.assertEqual([ops.probability_terms(k) for k in (1, 127, 128, 32767, 32768)], [3, 3, 2, 2, 1])
        self.assertEqual(ops.probability_terms(32768, least=2), 2)

    def test_attention_batches_equal_one_sequence_at_a_time(self):
        generator = torch.Generator().manual_seed(5)
        q, k, v = (torch.randn(12, width, generator=generator, dtype=torch.float64) for width in (8, 4, 4))
        sink = torch.randn(2, generator=generator, dtype=torch.float64)
        numerics = ops.Numerics("f64")
        together = ops.attention(q, k, v, [4, 4, 4], 2, 1, 4, {"window": [1, 1]}, sink, 0.5, numerics, ["cpu"])
        alone = torch.cat([ops.attention(q[r:r + 4], k[r:r + 4], v[r:r + 4], [4], 2, 1, 4, {"window": [1, 1]},
                                         sink, 0.5, numerics, ["cpu"]) for r in (0, 4, 8)])
        self.assertTrue(torch.equal(together, alone))

    def test_rotary_table_follows_the_f32_rule(self):
        table = ops.Axial2d({"pairs_per_axis": 18, "theta": 10000, "style": "neox",
                             "table": "table.hf-f32-angle.v1"}, ops.Numerics("pairs"))
        frequencies = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32) / 36))
        self.assertLessEqual(float((table.frequencies - frequencies).abs().max() / frequencies.max()), 2 ** -23)
        cos, sin = table.table(torch.tensor([[3, 7]]))
        angle = torch.cat([3 * table.frequencies, 7 * table.frequencies])
        self.assertTrue(torch.equal(cos[0], angle.double().cos().float()))
        self.assertTrue(torch.equal(sin[0], angle.double().sin().float()))

    def test_1d_rotary_table_follows_the_f32_rule(self):
        table = ops.Rope1d({"pairs": 32, "theta": 640000, "style": "neox", "table": "table.hf-f32-angle.v1",
                            "positions": "within-sequence"}, ops.Numerics("pairs"))
        frequencies = 1.0 / (640000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        self.assertLessEqual(float((table.frequencies - frequencies).abs().max() / frequencies.max()), 2 ** -23)
        cos, sin = table.table(ops.within_sequence([2, 4]))
        self.assertEqual(tuple(cos.shape), (6, 32))
        angle = 3 * table.frequencies
        self.assertTrue(torch.equal(cos[5], angle.double().cos().float()))
        self.assertTrue(torch.equal(sin[5], angle.double().sin().float()))
        self.assertTrue(torch.equal(cos[2], torch.ones(32)))


@unittest.skipIf(torch is None, "needs torch")
class VisionVariants(unittest.TestCase):
    """The q/k RMSNorm attention, clamped SwiGLU, norm-GELU-MLP projector,
    zero-column pad and space-to-depth weight columns, each against its
    definition. A model's tests run them in a whole tower against its
    vendor's."""

    def setUp(self):
        self.generator = torch.Generator().manual_seed(21)

    def randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.generator, dtype=torch.float64) * scale

    def test_space_to_depth_columns_and_a_block_merge_are_the_strided_conv(self):
        weights = encoder.Weights(None, None, None, "f64", "cpu")
        conv = self.randn(5, 3, 2, 2)
        rows = self.randn(6 * 4, 3)
        permuted = weights.transform(conv, ["flatten", {"space_to_depth_cols": 2}])
        merged = ops.Consecutive("merge.consecutive.v1", {"rows": 4}, {}, None)(rows, None)
        # Each run of 4 rows is one 2 x 2 block, row-major inside the block.
        image = rows.view(6, 2, 2, 3).permute(0, 3, 1, 2)
        expected = F.conv2d(image, conv, stride=2).view(6, 5)
        self.assertLess(relative(merged @ permuted.T, expected), 1e-14)

    def test_zero_columns_leave_a_gemm_unchanged(self):
        rows, weight = self.randn(7, 6), self.randn(4, 6)
        pad = ops.ZeroColumns("pad.zero-columns.v1", {"columns": 8}, {}, None)
        padded = pad(rows, None)
        self.assertEqual(padded.shape, (7, 8))
        self.assertTrue(torch.equal(padded[:, :6], rows) and not padded[:, 6:].any())
        wide = encoder.Weights(None, None, None, "f64", "cpu").transform(weight, [{"pad_cols": 8}])
        gemm = ops.Gemm(ops.Numerics("f64"), wide)
        self.assertTrue(torch.equal(gemm(padded), ops.Gemm(ops.Numerics("f64"), weight)(rows)))
        with self.assertRaises(ValueError):
            ops.ZeroColumns("pad.zero-columns.v1", {"columns": 6}, {}, None)(rows, None)

    def test_clamped_swiglu_clamps_after_the_bias(self):
        limit = 1.5
        weights = {"gate_up.weight": self.randn(8, 5), "gate_up.bias": self.randn(8),
                   "down.weight": self.randn(5, 4), "down.bias": self.randn(5)}
        x = self.randn(9, 5, scale=3.0)
        mlp = ops.ClampedSwigluMlp({"hidden": 4, "bias": True, "limit": limit}, weights, ops.Numerics("f64"))
        gate_up = x @ weights["gate_up.weight"].T + weights["gate_up.bias"]
        gate, up = gate_up[:, :4], gate_up[:, 4:]
        self.assertTrue(bool((gate > limit).any()) and bool((up < -limit).any()))
        expected = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit) @ weights["down.weight"].T \
            + weights["down.bias"]
        self.assertLess(relative(mlp(x), expected), 1e-14)
        x[0, 0] = float("nan")
        self.assertTrue(bool(mlp(x)[0].isnan().all()))

    def test_head_norms_follow_each_mode(self):
        x, weight = self.randn(3, 2 * 8, scale=3.0), 1 + self.randn(8, scale=0.3)
        heads = x.view(3, 2, 8)
        exact = heads * torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
        norm = ops.HeadRmsNorm(ops.Numerics("f64"), 1e-5, weight)
        self.assertLess(relative(norm(x, 2, 8), exact.view(3, 16)), 1e-15)
        # Under bf16: F32 normalization rounded to BF16, then the BF16 weight.
        low = x.to(torch.bfloat16)
        t = low.float().view(3, 2, 8)
        expected = weight.to(torch.bfloat16) * (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + ops.f32(1e-5))) \
            .to(torch.bfloat16)
        out = ops.HeadRmsNorm(ops.Numerics("bf16"), 1e-5, weight)(low, 2, 8)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, expected.view(3, 16)))
        pairs = ops.HeadRmsNorm(ops.Numerics("pairs"), 1e-5, weight)(x.float(), 2, 8)
        self.assertEqual(pairs.dtype, torch.float32)
        self.assertLess(relative(pairs, exact.view(3, 16)), 1e-6)
        with self.assertRaisesRegex(NotImplementedError, "qk-rms"):
            ops.HeadRmsNorm(ops.Numerics("legacy"), 1e-5, weight)

    def test_the_projector_is_norm_gelu_then_mlp(self):
        limit = 2.0
        weights = {"norm.weight": 1 + self.randn(6, scale=0.3), "norm.bias": self.randn(6, scale=0.3),
                   "gate_up.weight": self.randn(10, 6, scale=2.0), "down.weight": self.randn(6, 5)}
        context = encoder.Context(ops.Numerics("f64"), ["cpu"], ops.MLP)
        params = {"norm": {"op": "norm.layer-f32.v1", "eps": 1e-5}, "gelu": "erf",
                  "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 5, "bias": False, "limit": limit}}
        projector = ops.Projector("projector.norm-gelu-mlp.v1", params, weights, context)
        x = self.randn(4, 6, scale=4.0)
        hidden = F.gelu(F.layer_norm(x, (6,), weights["norm.weight"], weights["norm.bias"], 1e-5))
        gate, up = (hidden @ weights["gate_up.weight"].T).chunk(2, dim=-1)
        expected = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)) @ weights["down.weight"].T
        self.assertLess(relative(projector(x, None), expected), 1e-14)


@unittest.skipIf(torch is None, "needs torch")
class CommandLine(unittest.TestCase):
    def test_run_writes_rows_taps_and_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_checkpoint(root / "checkpoint")
            declaration = root / "tiny-vision.json"
            declaration.write_text(json.dumps(PATCH))
            pixels = torch.randn((24, 12), generator=torch.Generator().manual_seed(6))
            (root / "pixels.f32").write_bytes(pixels.numpy().tobytes())
            (root / "grid.i64").write_bytes(torch.tensor([1, 4, 6]).numpy().tobytes())
            image = {"kind": "image", "tensor": {"path": "pixels.f32", "dtype": "f32", "shape": [24, 12]},
                     "grid": {"path": "grid.i64", "dtype": "i64", "shape": [1, 3]},
                     "processor": {"code_sha256": "0" * 64, "config_sha256": "1" * 64}}
            (root / "workload.json").write_text(json.dumps({
                "schema": workload.SCHEMA, "model": "tiny", "sequences": [
                    {"name": "two-images", "category": "image", "token_ids": [1, 2, 3], "score_from": 1,
                     "media": [image, image]},
                    {"name": "text", "category": "text", "token_ids": [1, 2], "score_from": 1}]}))
            provenance = encoder.main([
                "--declaration", str(declaration), "--adapter", str(ROOT / "tools/refrun/tiny.py"),
                "--checkpoint", str(root / "checkpoint"), "--lock", str(root / "checkpoint/checkpoint-files.json"),
                "--workload", str(root / "workload.json"), "--devices", "cpu", "--blocks",
                "--modes", "exact,policy,reference-bf16", "--output", str(root / "out"),
                "--stamps", str(root / "stamps")])
            self.assertEqual(sorted(provenance["modes"]), ["exact", "policy", "reference-bf16"])
            policy = json.loads((root / "out/policy.json").read_text())
            (entry,) = policy["entries"]
            self.assertEqual(entry["name"], "two-images")
            self.assertEqual([item["rows"] for item in entry["items"]], [[6, 8], [6, 8]])
            self.assertEqual(entry["items"][0]["sha256"], entry["items"][1]["sha256"])
            self.assertLessEqual(set(keys.ENCODER_NUMERICS), set(entry["key"]["code"]["files"]))
            self.assertEqual(entry["key"]["mode"]["name"], "policy")
            self.assertEqual(keys.key_sha256(entry["key"]), entry["key_sha256"])
            step = root / "out/taps/policy/step0000"
            index = [json.loads(line) for line in (step / "index.jsonl").read_text().splitlines()]
            rows = {line["name"]: (line["rows"], line["dtype"]) for line in index}
            self.assertEqual(rows["enc.vision.B00.out"], (48, "f32"))
            self.assertEqual(rows["enc.vision.x.unrounded"], (12, "f32"))
            self.assertEqual(rows["enc.vision.out"], (12, "bf16"))
            self.assertEqual(json.loads((step / "completion.json").read_text())["completion"], "full")
            # One call per image: each tap's rows are recorded per item.
            segments = json.loads((step / "items.json").read_text())["encoder"]
            for name, rows in (("enc.vision.out", 6), ("enc.vision.B00.out", 24)):
                self.assertEqual(segments[name], [
                    {"request": "two-images", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": rows},
                    {"request": "two-images", "start": rows, "kind": "image", "item": 1, "index": 0,
                     "rows": rows}], name)
            out = torch.frombuffer(bytearray((step / "enc.vision.out.r0.bin").read_bytes()), dtype=torch.bfloat16)
            self.assertEqual(hashlib.sha256(out[:48].view(torch.uint8).numpy().tobytes()).hexdigest(),
                             entry["items"][0]["sha256"])
            exact = json.loads((root / "out/exact.json").read_text())["entries"][0]
            self.assertNotEqual(exact["key_sha256"], entry["key_sha256"])
            self.assertEqual(exact["items"][0]["dtype"], "torch.float64")


if __name__ == "__main__":
    unittest.main()
