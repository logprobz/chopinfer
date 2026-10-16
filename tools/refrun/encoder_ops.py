"""Torch implementations of the encoder component variants, one per id.

The generic encoder reference (encoder.py) builds a declaration's stages
from `VARIANTS`, which maps each shared variant id of crates/encoder-spec to
the class that evaluates it; an id without one is refused by name. A
model's own variants (`<model>:<kind>.<name>.v<N>`) live in
models/<model>/encoder_reference.py under the same interfaces.

A variant id fixes what a stage computes and, under the precision policy it
belongs to, where it rounds. `Numerics` is how one mode evaluates any
variant:

| Arithmetic | Mode | Rows between ops | Products |
|---|---|---|---|
| `f64` | `exact` | f64 | f64; no rounding anywhere, not even at the output |
| `pairs` | `policy`, `f32-pairs.v1` | F32 | GEMM activation operands and attention q, k, v as BF16 hi+lo pairs, F32 accumulation; one BF16 rounding at the output |
| `legacy` | `policy`, `bf16-legacy.v1` | BF16 | BF16 operands, F32 accumulation, the epilogue on the F32 accumulator, one BF16 rounding per stored row |
| `bf16` | `reference-bf16` | BF16 | unfused BF16 PyTorch: each op rounds its own output to BF16 |

Common to every arithmetic:

- Norms are two-pass over the row with a correctly rounded rsqrt (f64 in
  `f64`, F32 otherwise); `bf16` uses torch's BF16 LayerNorm and RMSNorm.
- Attention takes F32 (f64) scores and softmax and divides by the softmax
  denominator after the probability-value product. Under `pairs` the
  scores are q hi k hi + q hi k lo + q lo k hi. Under `pairs` and `legacy`
  each probability enters the product as BF16 terms (`probability_terms`:
  the rounded value and its rounded residuals, as the varlen kernels split
  them), times the pair of v under `pairs` and the BF16 v under `legacy`.
  The kernels take the term count from the fewest keys any row of a warp
  sees; here each row takes its own. The engine's accumulation order
  (tiles, online softmax, one product per term) is not emulated.
- Host tables (rotary angles, interpolation taps) are the variant's own F32
  values in every mode, widened in `f64`.
- Scalars (eps, the attention scale) are rounded to F32 except in `f64`.
- `round_linear` (bf16-legacy.v1 only) rounds a linear output to BF16
  before its activation under `legacy`; `bf16` always does.

Variant rounding points beyond the table:

- attention.varlen.qk-rms.pairs.v1 normalizes each q and k head of the
  stored qkv rows before the rotary: as `Norm` (`rms`) in `f64` and
  `pairs`; under `bf16` as the reference's own RMSNorm module computes it,
  the F32 mean square and `x * rsqrt` rounded to BF16, then the BF16 weight
  product (HeadRmsNorm). The block norms keep `Norm`'s rules.
- mlp.swiglu-clamped.v1 clamps the gate and up halves of the gate-up
  accumulator (after its bias) before SiLU and the product: in the
  accumulator dtype, BF16 under `bf16`. A clamp is exact and keeps a NaN.
- projector.norm-gelu-mlp.v1 stores its norm's rows, takes GELU as a pass
  over them (`Numerics.gelu`), then runs its MLP slot, with no residual.
- pad.zero-columns.v1 appends zero columns to the stored rows: exact.
"""

import struct

import torch
import torch.nn.functional as F
from torch import nn

ARITHMETIC = ("f64", "pairs", "legacy", "bf16")
CHUNK = 1024
# One attention score buffer holds at most this many bytes.
SCORE_BYTES = 2 << 30


def f32(value):
    """`value` rounded to F32, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(x):
    """`x` rounded to BF16 (nearest even), in `x`'s dtype."""
    return x.to(torch.bfloat16).to(x.dtype)


def halves(x):
    """An F32 operand as a BF16 hi+lo pair: hi = bf16(x), lo = bf16(x - hi)."""
    hi = bf16(x)
    return hi, bf16(x - hi)


def pair(x):
    """An F32 operand as its BF16 hi+lo pair, summed back in F32 (within
    2^-24 of the pair's own value)."""
    hi, lo = halves(x)
    return hi + lo


class Numerics:
    """How one mode evaluates every variant (module docstring)."""

    def __init__(self, arithmetic):
        if arithmetic not in ARITHMETIC:
            raise ValueError(f"unknown arithmetic {arithmetic}; expected one of {ARITHMETIC}")
        self.arithmetic = arithmetic
        self.rows = {"f64": torch.float64, "pairs": torch.float32}.get(arithmetic, torch.bfloat16)
        self.compute = torch.float64 if arithmetic == "f64" else torch.float32

    @property
    def exact(self):
        return self.arithmetic == "f64"

    def scalar(self, value):
        return float(value) if self.exact else f32(value)

    def store(self, x):
        return x.to(self.rows)

    def add(self, x, y):
        """A residual add of two stored row tensors."""
        return self.store(x.to(self.rows) + y.to(self.rows))

    def gelu(self, x, kind):
        """GELU of stored rows as a separate pass."""
        approximate = {"erf": "none", "tanh": "tanh"}[kind]
        if self.arithmetic == "bf16":
            return F.gelu(x.to(torch.bfloat16), approximate=approximate)
        return self.store(F.gelu(x.to(self.compute), approximate=approximate))


def gelu(x, kind):
    return F.gelu(x, approximate={"erf": "none", "tanh": "tanh"}[kind])


class Gemm(nn.Module):
    """`x @ weight^T + bias`, with an optional GELU on the accumulator."""

    def __init__(self, numerics, weight, bias=None, activation=None, round_linear=False):
        super().__init__()
        self.numerics, self.activation, self.round_linear = numerics, activation, round_linear
        arithmetic = numerics.arithmetic
        dtype = {"f64": torch.float64, "pairs": torch.float32, "legacy": torch.float32}.get(arithmetic, torch.bfloat16)
        if arithmetic in ("legacy", "bf16"):
            weight = weight.to(torch.bfloat16)
            bias = None if bias is None else bias.to(torch.bfloat16)
        self.weight = weight.to(dtype)
        self.bias = None if bias is None else bias.to(dtype)

    def accumulate(self, x):
        """The product plus bias, before any activation: BF16 under `bf16`,
        else the accumulator dtype."""
        arithmetic = self.numerics.arithmetic
        if arithmetic == "bf16":
            return F.linear(x.to(torch.bfloat16), self.weight, self.bias)
        if arithmetic == "pairs":
            x = pair(x.to(torch.float32))
        elif arithmetic == "legacy":
            x = x.to(torch.bfloat16).to(torch.float32)
        else:
            x = x.to(torch.float64)
        return F.linear(x, self.weight, self.bias)

    def forward(self, x):
        y = self.accumulate(x)
        if self.activation is not None:
            if self.numerics.arithmetic == "legacy" and self.round_linear:
                y = bf16(y)
            y = gelu(y, self.activation)
        return self.numerics.store(y)


def rsqrt(value, numerics):
    """1 / sqrt(value), correctly rounded to F32 outside `f64`."""
    if numerics.exact:
        return 1.0 / torch.sqrt(value)
    return (1.0 / torch.sqrt(value.to(torch.float64))).to(torch.float32)


class Norm(nn.Module):
    """LayerNorm (`layer`) or RMSNorm (`rms`) over each row with a learned
    weight (and bias)."""

    def __init__(self, numerics, kind, eps, weight, bias=None):
        super().__init__()
        self.numerics, self.kind = numerics, kind
        self.eps = numerics.scalar(eps)
        dtype = torch.bfloat16 if numerics.arithmetic == "bf16" else numerics.compute
        self.weight = weight.to(dtype)
        self.bias = None if bias is None else bias.to(dtype)
        if (kind == "layer") != (bias is not None):
            raise ValueError("a LayerNorm has a weight and a bias, an RMSNorm a weight only")

    def forward(self, x):
        n = self.numerics
        if n.arithmetic == "bf16":
            x = x.to(torch.bfloat16)
            if self.kind == "layer":
                return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
            return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
        x = x.to(n.compute)
        if self.kind == "layer":
            x = x - x.mean(-1, keepdim=True)
        scaled = x * rsqrt((x * x).mean(-1, keepdim=True) + self.eps, n) * self.weight
        return n.store(scaled if self.bias is None else scaled + self.bias)


NORMS = {"norm.layer.v1": "layer", "norm.layer-f32.v1": "layer", "norm.rms.v1": "rms", "norm.rms-f32.v1": "rms"}


def norm(op, params, numerics, weight, bias=None):
    if op not in NORMS:
        raise refused(op)
    return Norm(numerics, NORMS[op], params["eps"], weight, bias)


def refused(op):
    return NotImplementedError(f"the encoder reference has no implementation of `{op}`")


# Rows and their geometry ------------------------------------------------------


class Rows:
    """What a run's rows are: the attention sequences they form (lengths in
    order) and, for patch rows, the `(t, h, w)` merge-block grid they follow."""

    def __init__(self, lengths, grid=None):
        self.lengths, self.grid = tuple(lengths), grid

    @property
    def count(self):
        return sum(self.lengths)

    def __repr__(self):
        return f"Rows({list(self.lengths)}, grid={self.grid})"


def merge_block_positions(grid):
    """(row, column) of every patch of a `(t, h, w)` grid in merge-block
    order (block rows, block columns, the 2 x 2 patches of each block),
    frame after frame: [t*h*w, 2]."""
    t, h, w = grid
    blocks_h, blocks_w = torch.meshgrid(torch.arange(h // 2), torch.arange(w // 2), indexing="ij")
    inner_h, inner_w = torch.meshgrid(torch.arange(2), torch.arange(2), indexing="ij")
    rows = (2 * blocks_h[:, :, None, None] + inner_h).reshape(-1)
    cols = (2 * blocks_w[:, :, None, None] + inner_w).reshape(-1)
    return torch.stack([rows, cols], dim=-1).repeat(t, 1)


def within_sequence(lengths):
    """Each row's index inside its sequence."""
    return torch.cat([torch.arange(length) for length in lengths])


# Rotary -----------------------------------------------------------------------


def rotate(x, heads, head_dim, rotary_dims, cos, sin, numerics, unfused_bf16=False):
    """NeoX rotation of the first `rotary_dims` dims of every head of
    [rows, heads * head_dim] by per-row `cos`, `sin` [rows, rotary_dims / 2]:
    dim d pairs with d + rotary_dims / 2. In the compute dtype, then stored;
    with `unfused_bf16` under `legacy` and `bf16`, as unfused BF16 PyTorch:
    the table, both products and the sum each rounded to BF16."""
    rows = x.shape[0]
    half = rotary_dims // 2
    t = x.view(rows, heads, head_dim)
    if unfused_bf16 and numerics.arithmetic in ("legacy", "bf16"):
        t, cos, sin = t.to(torch.bfloat16), cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    else:
        t, cos, sin = t.to(numerics.compute), cos.to(numerics.compute), sin.to(numerics.compute)
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = t[..., :half], t[..., half:rotary_dims]
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, t[..., rotary_dims:]], dim=-1)
    return numerics.store(out.reshape(rows, heads * head_dim))


class Axial2d:
    """rotary.axial-2d.v1 with table.hf-f32-angle.v1: `pairs_per_axis`
    frequencies, the F32 exponent 2i / (2 P), an f64 power of the F32 theta
    rounded to F32, its F32 reciprocal, the F32 product with the patch row
    (first P angles) and column (next P), and cos and sin of that angle in
    f64 rounded to F32."""

    axes = "grid2d"
    TABLES = ("table.hf-f32-angle.v1",)

    def __init__(self, params, numerics):
        if params["table"] not in self.TABLES:
            raise refused(params["table"])
        if params["style"] != "neox":
            raise refused(f"rotary style {params['style']}")
        self.numerics = numerics
        count = params["pairs_per_axis"]
        theta = f32(params["theta"])
        self.frequencies = torch.tensor(
            [f32(1.0 / f32(theta ** f32((2 * i) / (2 * count)))) for i in range(count)], dtype=torch.float32)
        self.rotary_dims = 4 * count

    def table(self, positions):
        angles = (positions.to(torch.float32)[:, :, None] * self.frequencies.to(positions.device)).reshape(
            positions.shape[0], -1)
        return angles.double().cos().float(), angles.double().sin().float()

    def apply(self, x, heads, head_dim, cos, sin):
        return rotate(x, heads, head_dim, self.rotary_dims, cos, sin, self.numerics)


class Rope1d:
    """rotary.1d.v1 with table.hf-f32-angle.v1: `pairs` frequencies of
    Axial2d's rule (exponent 2i / (2 pairs)) times the row's index in its
    sequence."""

    axes = "sequence"
    TABLES = Axial2d.TABLES

    def __init__(self, params, numerics):
        if params["table"] not in self.TABLES:
            raise refused(params["table"])
        if params["style"] != "neox" or params["positions"] != "within-sequence":
            raise refused(f"rotary {params['style']} over {params['positions']}")
        self.numerics = numerics
        count = params["pairs"]
        theta = f32(params["theta"])
        self.frequencies = torch.tensor(
            [f32(1.0 / f32(theta ** f32((2 * i) / (2 * count)))) for i in range(count)], dtype=torch.float32)
        self.rotary_dims = 2 * count

    def table(self, positions):
        angles = positions.to(torch.float32)[:, None] * self.frequencies.to(positions.device)
        return angles.double().cos().float(), angles.double().sin().float()

    def apply(self, x, heads, head_dim, cos, sin):
        return rotate(x, heads, head_dim, self.rotary_dims, cos, sin, self.numerics)


ROTARY = {"rotary.axial-2d.v1": Axial2d, "rotary.1d.v1": Rope1d}


def merge_group_columns(grid):
    """reorder.merge-group-columns.v1: the merge blocks of each frame walked
    down each column of blocks, the 4 rows of a block in order. Destination
    row -> source row."""
    t, h, w = grid
    order = []
    for frame in range(t):
        base = frame * h * w
        for block_w in range(w // 2):
            for block_h in range(h // 2):
                first = base + (block_h * (w // 2) + block_w) * 4
                order.extend(range(first, first + 4))
    return torch.tensor(order)


REORDER = {"reorder.merge-group-columns.v1": merge_group_columns}


# Attention --------------------------------------------------------------------

# Every one runs `attention` under the mode's arithmetic; the q/k RMSNorm
# variant first normalizes q and k (HeadRmsNorm).
ATTENTION = ("attention.varlen.bf16.v1", "attention.varlen.pairs.v1", "attention.varlen.qk-rms.pairs.v1")
QK_RMS = "attention.varlen.qk-rms.pairs.v1"


class HeadRmsNorm(nn.Module):
    """RMSNorm of every `head_dim` slice of [rows, heads * head_dim] stored
    rows with one weight [head_dim] (module docstring for its rounding)."""

    def __init__(self, numerics, eps, weight):
        super().__init__()
        self.numerics = numerics
        if numerics.arithmetic == "legacy":
            raise refused(f"{QK_RMS} under bf16-legacy.v1")
        if numerics.arithmetic == "bf16":
            self.eps, self.weight = f32(eps), weight.to(torch.bfloat16)
        else:
            self.norm = Norm(numerics, "rms", eps, weight)

    def forward(self, x, heads, head_dim):
        rows = x.shape[0]
        t = x.reshape(rows * heads, head_dim)
        if self.numerics.arithmetic == "bf16":
            t = t.to(torch.float32)
            scaled = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
            out = self.weight * scaled
        else:
            out = self.norm(t)
        return out.reshape(rows, heads * head_dim)


def probability_terms(keys, least=1):
    """BF16 terms per probability of a row that sees `keys` keys
    (kernels-sm120 cuda/attention/varlen.cuh `probability_terms`): three
    below 128 keys, two below 32,768, one from there; at least `least`
    (two for attention.varlen.pairs.v1)."""
    return max(least, 1 if keys >= 32768 else 2 if keys >= 128 else 3)


def split_probabilities(probabilities, seen, length, least):
    """The probabilities as the sum of their BF16 terms: bf16(p), then
    bf16 of each remaining residual, as many as each row's
    `probability_terms`."""
    first = bf16(probabilities)
    rest = probabilities - first
    second = bf16(rest)
    third = bf16(rest - second)
    if seen is None:
        terms = probability_terms(length, least)
        return first + (second if terms >= 2 else 0) + (third if terms >= 3 else 0)
    keys = seen.sum(-1, keepdim=True)
    two = (keys < 32768) | (least >= 2)
    return first + second * two.to(second.dtype) + third * (keys < 128).to(third.dtype)


def visible(mask, queries, keys):
    """Which keys [len(keys)] each query [len(queries)] of one sequence sees,
    or None when it sees all: `full`, `causal`, `{"window": [l, r]}` (keys
    q - l ..= q + r) or `{"causal_window": n}` (keys q - n ..= q)."""
    i, j = queries[:, None], keys[None, :]
    if mask == "full":
        return None
    if mask == "causal":
        return j <= i
    if isinstance(mask, dict) and set(mask) == {"window"}:
        left, right = mask["window"]
        return (j >= i - left) & (j <= i + right)
    if isinstance(mask, dict) and set(mask) == {"causal_window"}:
        return (j >= i - mask["causal_window"]) & (j <= i)
    raise refused(f"attention mask {mask}")


def attention(q, k, v, lengths, heads, kv_heads, head_dim, mask, sink, scale, numerics, devices):
    """Self-attention over consecutive sequences of `lengths` rows. q is
    [rows, heads * head_dim] and k, v [rows, kv_heads * head_dim] stored
    rows; query head h reads key/value head h // (heads / kv_heads). A
    key-0 `sink` [heads] adds to each head's logit of the sequence's first
    key where the mask lets the query see it. Equal-length sequences are
    batched; query chunks go round robin over `devices`. Returns stored rows."""
    n = numerics
    pairs = n.arithmetic == "pairs"
    home = q.device
    out = torch.empty((q.shape[0], heads * head_dim), dtype=n.compute, device=home)
    groups = heads // kv_heads
    starts, offset = {}, 0
    for length in lengths:
        starts.setdefault(length, []).append(offset)
        offset += length
    if offset != q.shape[0]:
        raise ValueError(f"sequences of {offset} rows over {q.shape[0]} rows")
    turn = 0
    for length, firsts in starts.items():
        index = (torch.tensor(firsts)[:, None] + torch.arange(length)).to(home)

        def heads_of(x, count):
            return x.to(n.compute)[index].view(len(firsts), length, count, head_dim).transpose(1, 2)
        qs, ks, vs = heads_of(q, heads), heads_of(k, kv_heads), heads_of(v, kv_heads)
        if groups > 1:
            ks, vs = ks.repeat_interleave(groups, dim=1), vs.repeat_interleave(groups, dim=1)
        width = n.compute.itemsize
        chunk = max(1, min(CHUNK, length, SCORE_BYTES // (heads * length * width)))
        batch = max(1, min(len(firsts), SCORE_BYTES // (heads * chunk * length * width)))
        for first in range(0, len(firsts), batch):
            # Each device works from its own copies and keeps its results
            # until the batch is done, so the devices run side by side.
            copies, results = {}, []
            for start in range(0, length, chunk):
                device = torch.device(devices[turn % len(devices)])
                turn += 1
                if device not in copies:
                    qb, kb, vb = (x[first:first + batch].to(device) for x in (qs, ks, vs))
                    copies[device] = (qb, halves(kb), pair(vb)) if pairs else (qb, kb, vb)
                queries, keys_, values = copies[device]
                block = queries[:, :, start:start + chunk]
                if pairs:
                    (q_hi, q_lo), (k_hi, k_lo) = halves(block), keys_
                    scores = q_hi @ k_hi.transpose(-1, -2)
                    scores += q_hi @ k_lo.transpose(-1, -2)
                    scores += q_lo @ k_hi.transpose(-1, -2)
                else:
                    scores = block @ keys_.transpose(-1, -2)
                scores *= scale
                if sink is not None:
                    scores[..., 0] += sink.to(device=device, dtype=scores.dtype).view(1, -1, 1)
                rows = torch.arange(start, min(start + chunk, length), device=device)
                seen = visible(mask, rows, torch.arange(length, device=device))
                if seen is not None:
                    scores.masked_fill_(~seen, float("-inf"))
                scores -= scores.amax(-1, keepdim=True)
                probabilities = scores.exp_()
                denominator = probabilities.sum(-1, keepdim=True)
                if n.arithmetic in ("pairs", "legacy"):
                    probabilities = split_probabilities(probabilities, seen, length, 2 if pairs else 1)
                results.append((start, (probabilities @ values) / denominator))
                del scores, probabilities
            for start, result in results:
                rows_out = index[first:first + batch, start:start + chunk].reshape(-1)
                out.index_copy_(0, rows_out, result.transpose(1, 2).reshape(-1, heads * head_dim).to(home))
    return n.store(out)


# Stages -----------------------------------------------------------------------


class Stage(nn.Module):
    """A declaration stage: `forward(x, rows)` gives its output rows, and
    `geometry(rows)` what they are."""

    def geometry(self, rows):
        return rows


class Linear(Stage):
    """linear.bf16.v1, linear.bf16-round-linear.v1, linear.pairs.v1. The
    BF16 output of `output: bf16` is the encoder's output rounding
    (encoder.Boundary)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.gemm = Gemm(context.numerics, weights["weight"], weights.get("bias"), params.get("gelu"),
                         round_linear=op == "linear.bf16-round-linear.v1")

    def forward(self, x, rows):
        return self.gemm(x)


class Conv1d(Stage):
    """conv1d.im2col.v1 over each sequence of rows: output row r reads the
    zero-padded input rows r*stride .. r*stride + kernel, taps before
    channels, as one GEMM; `gelu` on its accumulator or `gelu_rows` as a
    pass over its stored rows."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics = context.numerics
        self.kernel, self.stride, self.pad = params["kernel"], params["stride"], params["pad"]
        weight = weights["weight"]
        self.gemm = Gemm(self.numerics, weight.reshape(weight.shape[0], -1), weights.get("bias"),
                         params.get("gelu"), params.get("round_linear", False))
        self.gelu_rows = params.get("gelu_rows")

    def out_rows(self, length):
        return (length + sum(self.pad) - self.kernel) // self.stride + 1

    def geometry(self, rows):
        return Rows([self.out_rows(length) for length in rows.lengths])

    def forward(self, x, rows):
        columns, offset = [], 0
        for length in rows.lengths:
            part = x[offset:offset + length]
            offset += length
            padded = torch.cat([part.new_zeros(self.pad[0], part.shape[1]), part,
                                part.new_zeros(self.pad[1], part.shape[1])])
            windows = padded.unfold(0, self.kernel, self.stride)[:self.out_rows(length)]
            columns.append(windows.transpose(1, 2).reshape(windows.shape[0], -1))
        y = self.gemm(torch.cat(columns))
        return y if self.gelu_rows is None else self.numerics.gelu(y, self.gelu_rows)


class LearnedGrid(Stage):
    """position.learned-grid.bilinear-align-corners.f32.v1: each patch adds
    the four bilinear taps of a learned `table` grid with `align_corners=True`.
    Taps and weights are F32 (HF's `get_vision_interpolation_indices_and_weights`);
    the weighted rows are summed in tap order and added in the compute dtype
    (under BF16 arithmetic, as HF BF16: the F32 sum rounded to BF16, then a
    BF16 add)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics = context.numerics
        self.side = params["table"]
        self.table_rows = weights["table"]

    def taps(self, grid, device):
        positions = merge_block_positions(grid).to(device)
        _, h, w = grid
        indices, weights = [], []
        for axis, size, side in ((0, h, self.side[0]), (1, w, self.side[1])):
            source = positions[:, axis].to(torch.float32) * (side - 1) / max(size - 1, 1)
            floor = torch.floor(source)
            offsets = torch.arange(2, device=device)
            indices.append((floor.long()[:, None] + offsets).clamp(0, side - 1))
            weights.append((1 - (source[:, None] - floor[:, None] - offsets).abs()).clamp(min=0))
        index = (indices[0][:, :, None] * self.side[1] + indices[1][:, None, :]).reshape(-1, 4)
        weight = (weights[0][:, :, None] * weights[1][:, None, :]).reshape(-1, 4)
        return index, weight

    def forward(self, x, rows):
        n = self.numerics
        index, weight = self.taps(rows.grid, x.device)
        low = n.arithmetic in ("legacy", "bf16")
        table = self.table_rows.to(torch.bfloat16 if low else n.compute)
        weight = weight if low else weight.to(n.compute)
        term = table[index[:, 0]] * weight[:, :1]
        for tap in range(1, 4):
            term = term + table[index[:, tap]] * weight[:, tap:tap + 1]
        if low:
            return n.store(x.to(torch.bfloat16) + term.to(torch.bfloat16))
        return n.store(x.to(n.compute) + term)


class NormStage(Stage):
    def __init__(self, op, params, weights, context):
        super().__init__()
        self.norm = norm(op, params, context.numerics, weights["weight"], weights.get("bias"))

    def forward(self, x, rows):
        return self.norm(x)


class Consecutive(Stage):
    """merge.consecutive.v1: each `rows` consecutive rows read as one."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.rows = params["rows"]

    def geometry(self, rows):
        if any(length % self.rows for length in rows.lengths):
            raise ValueError(f"sequences {list(rows.lengths)} do not come in whole groups of {self.rows}")
        return Rows([length // self.rows for length in rows.lengths])

    def forward(self, x, rows):
        return x.reshape(-1, self.rows * x.shape[1])


class RepeatLast(Stage):
    """group.repeat-last.v1: the item's rows (or code vectors), padded to a
    multiple of `rows` by repeating the last."""

    joins = True

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.rows = params["rows"]

    def geometry(self, rows):
        return Rows([-(-rows.count // self.rows) * self.rows])

    def forward(self, x, rows):
        padded = -(-x.shape[0] // self.rows) * self.rows
        return x[torch.arange(padded, device=x.device).clamp(max=x.shape[0] - 1)]


class Cast(Stage):
    """cast.bf16.v1: the output rounding to BF16 (none under `f64`)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics = context.numerics

    def forward(self, x, rows):
        return x if self.numerics.exact else x.to(torch.bfloat16)


class Block(nn.Module):
    """One pre-norm block: x += proj(attention(rotary(qkv(norm1(x))))), then
    x += mlp(norm2(x)), under the block's reorder: its rows are permuted
    before norm1 and put back after the second add, so its output is in the
    order it was given (`taps` sees canonical rows)."""

    def __init__(self, transformer, index, weights, context):
        super().__init__()
        n = self.numerics = context.numerics
        params = transformer.params
        self.devices = context.devices
        self.reorder, (self.mask, sink_kind) = transformer.reorder_at(index), transformer.attention_at(index)
        attention = params["attention"]
        self.heads, self.kv_heads, self.head_dim = attention["q_heads"], attention["kv_heads"], attention["head_dim"]
        self.scale = n.scalar(attention["scale"] if "scale" in attention else self.head_dim ** -0.5)
        if sink_kind == "key0":
            self.sink = weights["sinks"].to(torch.float32).to(n.compute)
        elif sink_kind == "none":
            self.sink = None
        else:
            raise refused(f"attention sink {sink_kind}")
        norm_op = params["norm"]["op"]
        self.norm1 = norm(norm_op, params["norm"], n, weights["norm1.weight"], weights.get("norm1.bias"))
        self.norm2 = norm(norm_op, params["norm"], n, weights["norm2.weight"], weights.get("norm2.bias"))
        self.qkv = Gemm(n, weights["qkv.weight"], weights.get("qkv.bias"))
        self.proj = Gemm(n, weights["proj.weight"], weights.get("proj.bias"))
        self.mlp = context.mlp(params["mlp"], weights)
        self.rotary = transformer.rotary
        self.qk_norms = None
        if attention["op"] == QK_RMS:
            eps = attention["qk_norm_eps"]
            self.qk_norms = (HeadRmsNorm(n, eps, weights["q_norm.weight"]),
                             HeadRmsNorm(n, eps, weights["k_norm.weight"]))

    def forward(self, x, rows):
        n = self.numerics
        order = None if self.reorder is None else self.reorder(rows.grid).to(x.device)
        if order is not None:
            x = x[order]
        qkv = self.qkv(self.norm1(x))
        q_width, kv_width = self.heads * self.head_dim, self.kv_heads * self.head_dim
        q, k, v = qkv[:, :q_width], qkv[:, q_width:q_width + kv_width], qkv[:, q_width + kv_width:]
        if self.qk_norms is not None:
            q = self.qk_norms[0](q, self.heads, self.head_dim)
            k = self.qk_norms[1](k, self.kv_heads, self.head_dim)
        if self.rotary is not None:
            positions = merge_block_positions(rows.grid) if self.rotary.axes == "grid2d" \
                else within_sequence(rows.lengths)
            positions = positions.to(x.device)
            if order is not None:
                positions = positions[order]
            cos, sin = self.rotary.table(positions)
            q = self.rotary.apply(q, self.heads, self.head_dim, cos, sin)
            k = self.rotary.apply(k, self.kv_heads, self.head_dim, cos, sin)
        attended = attention(q, k, v, rows.lengths, self.heads, self.kv_heads, self.head_dim, self.mask,
                             self.sink, self.scale, n, self.devices)
        x = n.add(x, self.proj(attended))
        x = n.add(x, self.mlp(self.norm2(x)))
        if order is not None:
            x = x[torch.argsort(order)]
        return x


class GeluMlp(nn.Module):
    """mlp.gelu.v1: down(gelu(up(x)))."""

    def __init__(self, params, weights, numerics):
        super().__init__()
        self.up = Gemm(numerics, weights["up.weight"], weights.get("up.bias"), params["gelu"],
                       params.get("round_linear", False))
        self.down = Gemm(numerics, weights["down.weight"], weights.get("down.bias"))

    def forward(self, x):
        return self.down(self.up(x))


class SwigluMlp(nn.Module):
    """mlp.swiglu.v1: down(silu(gate) * up), gate the first half of one
    gate-up projection."""

    def __init__(self, params, weights, numerics):
        super().__init__()
        self.numerics, self.round_linear = numerics, params.get("round_linear", False)
        self.gate_up = Gemm(numerics, weights["gate_up.weight"], weights.get("gate_up.bias"))
        self.down = Gemm(numerics, weights["down.weight"], weights.get("down.bias"))

    def forward(self, x):
        n = self.numerics
        gate, up = self.gate_up.accumulate(x).chunk(2, dim=-1)
        if n.arithmetic == "legacy" and self.round_linear:
            gate, up = bf16(gate), bf16(up)
        return self.down(n.store(F.silu(gate) * up))


class ClampedSwigluMlp(nn.Module):
    """mlp.swiglu-clamped.v1: down(silu(min(gate, limit)) * clamp(up,
    -limit, limit)), gate the first half of one gate-up projection, the
    clamps on the accumulator after its bias."""

    def __init__(self, params, weights, numerics):
        super().__init__()
        self.numerics, self.limit = numerics, numerics.scalar(params["limit"])
        self.gate_up = Gemm(numerics, weights["gate_up.weight"], weights.get("gate_up.bias"))
        self.down = Gemm(numerics, weights["down.weight"], weights.get("down.bias"))

    def forward(self, x):
        gate, up = self.gate_up.accumulate(x).chunk(2, dim=-1)
        gate, up = gate.clamp(max=self.limit), up.clamp(min=-self.limit, max=self.limit)
        return self.down(self.numerics.store(F.silu(gate) * up))


MLP = {"mlp.gelu.v1": GeluMlp, "mlp.swiglu.v1": SwigluMlp, "mlp.swiglu-clamped.v1": ClampedSwigluMlp}


class Transformer(Stage):
    """transformer.pre-norm.v1: its blocks in order, the per-block attention
    `overrides` and `reorder`, `carry` (block `save_after`'s output added to
    block `add_after`'s) and the stage's own `sequences` (`groups` of rows)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.params = params
        self.rotary = None if "rotary" not in params else context.rotary(params["rotary"])
        if self.rotary is not None and self.rotary.rotary_dims > params["attention"]["head_dim"]:
            raise ValueError("the rotary rotates more dims than a head has")
        if params["attention"]["op"] not in ATTENTION:
            raise refused(params["attention"]["op"])
        self.reorders = [(context.blocks(r["blocks"], params["blocks"]), context.reorder(r["reorder"]["op"]))
                         for r in params.get("reorder", [])]
        self.overrides = [(context.blocks(o["blocks"], params["blocks"]), o["attention"])
                          for o in params.get("overrides", [])]
        self.sequences = params.get("sequences")
        if self.sequences is not None and self.sequences["kind"] not in ("groups", "item"):
            raise refused(f"transformer sequences {self.sequences['kind']}")
        self.numerics = context.numerics
        self.blocks = nn.ModuleList([Block(self, i, weights.block(i), context) for i in range(params["blocks"])])

    def attention_at(self, index):
        attention = self.params["attention"]
        mask, sink = attention["mask"], attention.get("sink", "none")
        for members, change in self.overrides:
            if index in members:
                mask, sink = change.get("mask", mask), change.get("sink", sink)
        return mask, sink

    def reorder_at(self, index):
        for members, reorder in self.reorders:
            if index in members:
                return reorder
        return None

    def own(self, rows):
        if self.sequences is None:
            return rows
        if self.sequences["kind"] == "item":
            return Rows([rows.count], rows.grid)
        size = self.sequences["rows"]
        if rows.count % size:
            raise ValueError(f"{rows.count} rows are not whole groups of {size}")
        return Rows([size] * (rows.count // size))

    def forward(self, x, rows):
        rows = self.own(rows)
        carried = {}
        for index, block in enumerate(self.blocks):
            x = block(x, rows)
            for carry in self.params.get("carry", []):
                if carry["save_after"] == index:
                    carried[index] = x
                if carry["add_after"] == index:
                    x = self.numerics.add(x, carried.pop(carry["save_after"]))
        return x


class Projector(Stage):
    """projector.norm-gelu-mlp.v1: mlp(gelu(norm(x))) with no residual; the
    norm's weights are `norm.weight` and `norm.bias`, the MLP's its own
    roles. The BF16 output of `output: bf16` is the encoder's output
    rounding (encoder.Boundary)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics, self.gelu = context.numerics, params["gelu"]
        self.norm = norm(params["norm"]["op"], params["norm"], self.numerics, weights["norm.weight"],
                         weights.get("norm.bias"))
        self.mlp = context.mlp(params["mlp"], weights)

    def forward(self, x, rows):
        return self.mlp(self.numerics.gelu(self.norm(x), self.gelu))


class ZeroColumns(Stage):
    """pad.zero-columns.v1: the rows widened to `columns` with zeros."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.columns = params["columns"]

    def forward(self, x, rows):
        if self.columns <= x.shape[1]:
            raise ValueError(f"pads {x.shape[1]} columns to {self.columns}, not wider")
        return F.pad(x, (0, self.columns - x.shape[1]))


class Tokens(Stage):
    """layout.tokens.v1: the rows go to the language model as they are."""

    def __init__(self, op, params, weights, context):
        super().__init__()

    def forward(self, x, rows):
        return x


# Stage variants: id -> class. Slot variants: NORMS, ROTARY, MLP, REORDER and
# the attention ids Transformer accepts.
VARIANTS = {
    "linear.bf16.v1": Linear, "linear.bf16-round-linear.v1": Linear, "linear.pairs.v1": Linear,
    "conv1d.im2col.v1": Conv1d,
    "position.learned-grid.bilinear-align-corners.f32.v1": LearnedGrid,
    "norm.layer.v1": NormStage, "norm.layer-f32.v1": NormStage, "norm.rms.v1": NormStage,
    "norm.rms-f32.v1": NormStage,
    "transformer.pre-norm.v1": Transformer,
    "merge.consecutive.v1": Consecutive,
    "group.repeat-last.v1": RepeatLast,
    "cast.bf16.v1": Cast,
    "projector.norm-gelu-mlp.v1": Projector,
    "pad.zero-columns.v1": ZeroColumns,
    "layout.tokens.v1": Tokens,
}
