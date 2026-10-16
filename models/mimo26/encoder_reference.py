"""MiMo's own encoder variants for the generic encoder reference
(tools/refrun/encoder.py), one torch implementation per `mimo26:` id that
models/mimo26/contract/src/encoders/spec.rs registers. They follow the
interfaces of tools/refrun/encoder_ops.py and its arithmetic table.

The rotary tables are the bundle's host tables bit for bit (kernel-core
`AngleRule::MulRecipPowF32`, which MiMo's bundle builds with Rust F32
`powf`, `cos` and `sin`): those Rust functions call the C library's
`powf`, `cosf` and `sinf`, so the tables here call the same functions
through ctypes. These F32 functions are not always the f64 result rounded
to F32 (`cosf` and `sinf` differ from it by one ULP at some angles).
`test_encoder_reference.py` holds the bundle tables' digests.
"""

import ctypes
import ctypes.util

import torch

from tools.refrun import encoder_ops as ops

MODEL = "mimo26"
LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
for _name, _arguments in (("powf", 2), ("cosf", 1), ("sinf", 1)):
    getattr(LIBM, _name).restype = ctypes.c_float
    getattr(LIBM, _name).argtypes = [ctypes.c_float] * _arguments
# (theta, denominator, frequencies) -> (cos, sin) [positions, frequencies], grown as needed.
TABLES = {}


def f32_table(positions, pairs, denominator, theta):
    """cos and sin [rows, pairs] of `position * (1 / theta.powf(2i / d))` in
    F32 (kernel-core `AngleRule::MulRecipPowF32`): positions [rows, axes]
    (non-negative integers) take `pairs / axes` frequencies each."""
    theta = ops.f32(theta)
    per_axis = pairs // positions.shape[1]
    key = (theta, denominator, per_axis)
    count = int(positions.max()) + 1
    if key not in TABLES or TABLES[key][0].shape[0] < count:
        frequencies = [ops.f32(1.0 / LIBM.powf(theta, ops.f32((2 * i) / denominator))) for i in range(per_axis)]
        angles = [[ops.f32(float(position) * frequency) for frequency in frequencies] for position in range(count)]
        TABLES[key] = tuple(torch.tensor([[function(angle) for angle in row] for row in angles], dtype=torch.float32)
                            for function in (LIBM.cosf, LIBM.sinf))
    index = positions.long().cpu()
    return tuple(table[index].reshape(positions.shape[0], -1).to(positions.device) for table in TABLES[key])


class Axial2dPowf:
    """mimo26:rotary.axial-2d-f32-powf.v1: `pairs_per_axis` frequencies of
    the F32 table per patch row then column, over 2 * pairs_per_axis pairs."""

    axes = "grid2d"

    def __init__(self, params, numerics):
        if params["style"] != "neox":
            raise ops.refused(f"rotary style {params['style']}")
        self.numerics, self.theta = numerics, params["theta"]
        self.pairs = 2 * params["pairs_per_axis"]
        self.rotary_dims = 2 * self.pairs

    def table(self, positions):
        return f32_table(positions, self.pairs, self.pairs, self.theta)

    def apply(self, x, heads, head_dim, cos, sin):
        return ops.rotate(x, heads, head_dim, self.rotary_dims, cos, sin, self.numerics)


class Rope1dBf16:
    """mimo26:rotary.1d-bf16-unfused.v1: the F32 table over the row's index
    in its sequence, applied as unfused BF16 PyTorch under BF16 arithmetic
    (the table, both products and the sum rounded to BF16)."""

    axes = "sequence"

    def __init__(self, params, numerics):
        if params["style"] != "neox" or params["positions"] != "within-sequence":
            raise ops.refused(f"rotary {params['style']} over {params['positions']}")
        self.numerics, self.theta, self.pairs = numerics, params["theta"], params["pairs"]
        self.rotary_dims = 2 * self.pairs

    def table(self, positions):
        return f32_table(positions[:, None], self.pairs, self.rotary_dims, self.theta)

    def apply(self, x, heads, head_dim, cos, sin):
        return ops.rotate(x, heads, head_dim, self.rotary_dims, cos, sin, self.numerics, unfused_bf16=True)


class Rvq20(ops.Stage):
    """mimo26:quantize.rvq20.v1: per stage, the codebook entry nearest the
    remaining residual (sum of squared differences, ties to the lower
    code), subtracted from it. The residual starts from the rows; F32, f64
    under `f64`. Codebooks are as the weight mapping gives them: BF16 under
    `legacy` (`narrow_bf16`), the stored F32 otherwise. `bf16` takes the
    distance as the vendor's EuclideanCodebook.quantize does, |x|^2 -
    2 x.e + |e|^2 in F32, whose cancellation can pick another entry where
    two are near. Output: codes [rows, stages] (I64)."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics, self.sizes = context.numerics, params["codebooks"]
        tables = weights["codebooks"].to(context.numerics.compute)
        if tables.shape[0] != sum(self.sizes):
            raise ValueError(f"codebooks have {tables.shape[0]} rows, not {sum(self.sizes)}")
        self.tables = list(tables.split(self.sizes))

    def forward(self, x, rows):
        residual = x.to(self.numerics.compute)
        codes = []
        for table in self.tables:
            if self.numerics.arithmetic == "bf16":
                embed = table.t()
                chosen = (-(residual.pow(2).sum(1, keepdim=True) - 2 * residual @ embed
                            + embed.pow(2).sum(0, keepdim=True))).max(dim=-1).indices
            else:
                step = max(1, ops.SCORE_BYTES // (table.numel() * table.element_size()))
                chosen = torch.cat([((residual[start:start + step, None, :] - table) ** 2).sum(-1).argmin(-1)
                                    for start in range(0, residual.shape[0], step)])
            residual = residual - table[chosen]
            codes.append(chosen)
        return torch.stack(codes, dim=1)


class Sum20(ops.Stage):
    """mimo26:embed.sum20.v1: row r sums table s row code[r][s] over the
    tables in order; under BF16 arithmetic each add rounds to BF16."""

    def __init__(self, op, params, weights, context):
        super().__init__()
        self.numerics, self.vocab = context.numerics, params["vocab"]
        low = context.numerics.arithmetic in ("legacy", "bf16")
        self.tables = weights["tables"].to(torch.bfloat16 if low else context.numerics.compute)

    def forward(self, codes, rows):
        total = None
        for stage in range(codes.shape[1]):
            row = self.tables[stage * self.vocab + codes[:, stage]]
            total = row if total is None else total + row
        return self.numerics.store(total)


# The `-f32` ids are the same stages under f32-pairs.v1: F32 rows in and
# out and the stored F32 codebooks, which `Rvq20` and `Sum20` already take
# outside BF16 arithmetic.
VARIANTS = {
    "mimo26:rotary.axial-2d-f32-powf.v1": Axial2dPowf,
    "mimo26:rotary.1d-bf16-unfused.v1": Rope1dBf16,
    "mimo26:quantize.rvq20.v1": Rvq20,
    "mimo26:quantize.rvq20-f32.v1": Rvq20,
    "mimo26:embed.sum20.v1": Sum20,
    "mimo26:embed.sum20-f32.v1": Sum20,
}
