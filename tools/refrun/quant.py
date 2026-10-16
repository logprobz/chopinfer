"""Exact dequantization of checkpoint weights, per call, in the mode's dtype.

Formats (codes row-major, K contiguous):

- FP8 E4M3 blocks of any shape with a BF16 or F32 scale per block. A chunked
  grid restarts at every `chunk_rows` rows (a checkpoint stored in TP chunks),
  so each chunk's partial tail block gets its own scale row.
- MXFP4: packed E2M1 (even K in the low nibble) with one E8M0 scale per 32 K.
- NVFP4: packed E2M1 with one E4M3 scale per 16 K, times one F32 global scale.
- BF16.

Rules:

- `exact`: the weight in the mode dtype must equal its exact product.
  E4M3 x BF16 needs at most 4 + 8 significant bits and E2M1 x E8M0 is a
  power-of-two shift, so both are exact in F32. E4M3 x F32 and NVFP4's
  global product may not be; `check_exact` compares against the f64 product
  (exact: at most 2 + 4 + 24 significant bits) and refuses a mismatch.
- `rounded`: the exact product rounded once to the mode dtype (F32 or f64),
  for checkpoints whose F32 block scales make the product inexact in F32
  (MiMo's FP8 projections). One IEEE multiply of two F32 values is that
  rounding; NVFP4's three-factor product is formed in f64 first.
- `hf-bf16`: the F32 product rounded to BF16, as upstream BF16 loading does,
  then widened (exactly) to the mode dtype.

A `QuantTensor` keeps checkpoint-dtype codes and scales on its device and
dequantizes into a per-(device, dtype, slot) scratch buffer that the next
dequantization into the same slot overwrites. Ops queued on one stream read
it before that happens; a caller holding two weights at once uses two slots.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

RULES = ("exact", "rounded", "hf-bf16")


def e4m3(code):
    """OCP FP8 E4M3 (`float8_e4m3fn`): bias 7, no infinities, S.1111.111 is NaN."""
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0:
        return sign * math.ldexp(mantissa, -9)
    return sign * math.ldexp(8 + mantissa, exponent - 10)


def e2m1(code):
    """OCP FP4 E2M1: bias 1, no infinities or NaN; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6."""
    sign = -1.0 if code & 0x8 else 1.0
    exponent, mantissa = (code >> 1) & 0x3, code & 0x1
    if exponent == 0:
        return sign * mantissa * 0.5
    return sign * math.ldexp(2 + mantissa, exponent - 2)


def e8m0(code):
    """OCP E8M0 scale: 2^(code - 127); 0xFF is NaN."""
    return math.nan if code == 0xFF else math.ldexp(1.0, code - 127)


E4M3 = tuple(e4m3(code) for code in range(256))
E2M1 = tuple(e2m1(code) for code in range(16))
E8M0 = tuple(e8m0(code) for code in range(256))
# One byte of packed E2M1: (even K, odd K) = (low nibble, high nibble).
E2M1_PAIRS = tuple((E2M1[b & 0xF], E2M1[b >> 4]) for b in range(256))


@dataclass(frozen=True)
class Format:
    element: str          # "e4m3", "e2m1" or "bf16"
    scale: str = None     # "float" (the scale tensor's own BF16 or F32), "e8m0" or "e4m3"
    block_rows: int = 1
    block_cols: int = 1
    chunk_rows: int = None
    global_scale: float = None

    def scale_shape(self, rows, cols):
        chunk = self.chunk_rows or rows
        if rows % chunk:
            raise ValueError(f"{rows} rows do not split into chunks of {chunk}")
        return [rows // chunk * -(-chunk // self.block_rows), -(-cols // self.block_cols)]

    def scale_rows(self, rows):
        """The scale row of every stored weight row (a chunk starts a new grid)."""
        chunk = self.chunk_rows or rows
        per_chunk = -(-chunk // self.block_rows)
        return [r // chunk * per_chunk + r % chunk // self.block_rows for r in range(rows)]


BF16 = Format("bf16")
MXFP4 = Format("e2m1", "e8m0", 1, 32)


def fp8_block(block_rows=128, block_cols=128, chunk_rows=None):
    return Format("e4m3", "float", block_rows, block_cols, chunk_rows)


def nvfp4(global_scale):
    return Format("e2m1", "e4m3", 1, 16, global_scale=float(global_scale))


_tables = {}


def table(values, dtype, device):
    """A decode table on `device`, built once per (table, dtype, device)."""
    key = (id(values), dtype, torch.device(device))
    if key not in _tables:
        _tables[key] = torch.tensor(values, dtype=torch.float64).to(device=device, dtype=dtype)
    return _tables[key]


# Table gathers run in F32, which holds every E2M1, E4M3 and E8M0 value
# exactly (2^-127 as a subnormal), and widen exactly; an f64 gather of the
# packed E2M1 pairs measured 30x slower on the GPUs.
def decode_elements(codes, fmt, dtype, cols):
    """The format's element values, unscaled, in `dtype` (always exact)."""
    if fmt.element == "e4m3":
        return codes.view(torch.float8_e4m3fn).to(dtype)
    pairs = table(E2M1_PAIRS, torch.float32, codes.device)
    return pairs.index_select(0, codes.reshape(-1).int()).view(codes.shape[0], cols).to(dtype)


def decode_scales(scale, fmt, dtype):
    if fmt.scale == "float":
        return scale.to(dtype)
    values = E8M0 if fmt.scale == "e8m0" else E4M3
    codes = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    decoded = table(values, torch.float32, scale.device).index_select(0, codes.reshape(-1).int())
    return decoded.view(scale.shape).to(dtype)


def scaled(values, scales, fmt, row_index, out):
    """values x the block scale of each element, written to `out`."""
    rows, cols = values.shape
    block_rows = fmt.block_rows
    if row_index is not None:
        scales, block_rows = scales.index_select(0, row_index), 1
    if rows % block_rows == 0 and cols % fmt.block_cols == 0:
        shape = (rows // block_rows, block_rows, cols // fmt.block_cols, fmt.block_cols)
        torch.mul(values.view(shape), scales[:, None, :, None], out=out.view(shape))
    else:
        full = scales.repeat_interleave(block_rows, 0)[:rows].repeat_interleave(fmt.block_cols, 1)
        torch.mul(values, full[:, :cols], out=out)
    return out


def dequantize(fmt, codes, scale, shape, dtype, rule, row_index=None, out=None):
    """The [rows, cols] weight in `dtype` under `rule`, written to `out` if given."""
    if rule not in RULES:
        raise ValueError(f"unknown dequantization rule {rule}")
    rows, cols = shape
    out = torch.empty(shape, dtype=dtype, device=codes.device) if out is None else out
    if fmt.element == "bf16":
        return out.copy_(codes)
    if rule == "rounded" and dtype not in (torch.float32, torch.float64):
        raise ValueError(f"the rounded rule needs an F32 or f64 weight, not {dtype}")
    if rule == "rounded" and fmt.global_scale is not None:
        return out.copy_(dequantize(fmt, codes, scale, shape, torch.float64, "exact", row_index))
    compute = torch.float32 if rule == "hf-bf16" else dtype
    values = decode_elements(codes, fmt, compute, cols)
    scales = decode_scales(scale, fmt, compute)
    if fmt.global_scale is not None:
        # An F32 value, so the scalar converts exactly to F32 or f64.
        scales = scales * fmt.global_scale
    if rule in ("exact", "rounded"):
        return scaled(values, scales, fmt, row_index, out)
    product = scaled(values, scales, fmt, row_index, torch.empty(shape, dtype=compute, device=codes.device))
    return out.copy_(product.to(torch.bfloat16))


_scratch = {}


def scratch(device, dtype, slot, numel):
    """A reusable buffer of at least `numel` elements; see the module notes."""
    key = (torch.device(device), dtype, slot)
    if key not in _scratch or _scratch[key].numel() < numel:
        _scratch.pop(key, None)
        _scratch[key] = torch.empty(numel, dtype=dtype, device=device)
    return _scratch[key][:numel]


def release_scratch():
    _scratch.clear()


class QuantTensor(nn.Module):
    """One checkpoint weight [rows, cols], stored as the checkpoint stores it."""

    def __init__(self, fmt, codes, scale, shape, slot="linear"):
        super().__init__()
        rows, cols = shape
        packed = cols // 2 if fmt.element == "e2m1" else cols
        if fmt.element == "e2m1" and cols % 2:
            raise ValueError("packed E2M1 needs an even K")
        if tuple(codes.shape) != (rows, packed):
            raise ValueError(f"codes {list(codes.shape)} do not hold a [{rows}, {cols}] {fmt.element} weight")
        if fmt.element == "e4m3" and codes.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("FP8 codes must be uint8 or float8_e4m3fn")
        if fmt.element == "e2m1" and codes.dtype != torch.uint8:
            raise ValueError("packed E2M1 codes must be uint8")
        if fmt.element == "bf16" and (codes.dtype != torch.bfloat16 or scale is not None):
            raise ValueError("a BF16 weight is a bfloat16 tensor without scales")
        if fmt.element != "bf16":
            if list(scale.shape) != fmt.scale_shape(rows, cols):
                raise ValueError(f"scale {list(scale.shape)} is not the {fmt.scale_shape(rows, cols)} grid")
            if fmt.scale == "float" and scale.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("block scales must be BF16 or F32")
        if codes.dtype == torch.float8_e4m3fn:
            # Kept as bytes, so a module-wide dtype cast cannot decode them.
            codes = codes.view(torch.uint8)
        self.fmt, self.shape, self.slot = fmt, (rows, cols), slot
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("row_index", None, persistent=False)
        if fmt.chunk_rows is not None:
            self.row_index = torch.tensor(fmt.scale_rows(rows), device=codes.device)
        self.dtype, self.rule, self.checked = None, None, set()

    def permute_rows(self, order):
        """Stores the rows in `order` (stored row i becomes order[i]'s row),
        keeping each row's own scale row."""
        order = torch.as_tensor(order, device=self.codes.device)
        if self.fmt.element != "bf16":
            index = self.row_index if self.row_index is not None else torch.tensor(
                self.fmt.scale_rows(self.shape[0]), device=self.codes.device)
            self.row_index = index.index_select(0, order)
        self.codes = self.codes.index_select(0, order)
        self.checked.clear()

    def configure(self, dtype, rule, check=True):
        self.dtype, self.rule = dtype, rule
        if check and (dtype, rule) not in self.checked:
            check_exact(self, dtype, rule)
            self.checked.add((dtype, rule))

    def dequantize(self, dtype, rule, out=None):
        return dequantize(self.fmt, self.codes, self.scale, self.shape, dtype, rule, self.row_index, out)

    def value(self):
        out = scratch(self.codes.device, self.dtype, self.slot, self.shape[0] * self.shape[1])
        return self.dequantize(self.dtype, self.rule, out.view(self.shape))


class QuantLinear(QuantTensor):
    """`F.linear` over a weight dequantized per call; the bias is cast once."""

    def __init__(self, fmt, codes, scale, shape, bias=None, slot="linear"):
        super().__init__(fmt, codes, scale, shape, slot)
        self.in_features, self.out_features = shape[1], shape[0]
        self.register_buffer("stored_bias", bias, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def configure(self, dtype, rule, check=True):
        super().configure(dtype, rule, check)
        if self.stored_bias is not None:
            self.bias = self.stored_bias.to(dtype)

    def forward(self, x):
        return F.linear(x, self.value(), self.bias)


class QuantStack(nn.Module):
    """Stands in for a stacked [E, rows, cols] weight that upstream code
    indexes per expert: `stack[e]` is expert e's dequantized weight."""

    def __init__(self, tensors):
        super().__init__()
        self.items = nn.ModuleList(tensors)

    def __getitem__(self, index):
        return self.items[int(index)].value()

    def __len__(self):
        return len(self.items)


def check_exact(tensor, dtype, rule):
    """Refuses a non-finite weight, and under `exact` any element whose
    mode-dtype value differs from the f64 product."""
    value = tensor.dequantize(dtype, rule)
    if not torch.isfinite(value).all():
        raise ValueError(f"dequantized weight {list(tensor.shape)} is not finite")
    if rule == "exact" and dtype != torch.float64:
        if not torch.equal(value.double(), tensor.dequantize(torch.float64, "exact")):
            raise ValueError(f"{tensor.fmt} weight is not exact in {dtype}; use f64, rounded or hf-bf16")


def configure(model, dtype, rule, check=True):
    """Sets every quantized weight of `model` to the mode's dtype and rule."""
    for module in model.modules():
        if isinstance(module, QuantTensor):
            module.configure(dtype, rule, check)


def check_codecs(device):
    """The device's native E4M3 conversion equals the definition for every
    code in every mode dtype (the dequantizers rely on it)."""
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    expected = torch.tensor(E4M3, dtype=torch.float64)
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        native = codes.view(torch.float8_e4m3fn).to(dtype).double().cpu()
        nan = expected.isnan()
        same = ((native == expected) & (native.signbit() == expected.signbit())) | (native.isnan() & nan)
        if not same.all():
            raise RuntimeError(f"native E4M3 to {dtype} conversion differs from the definition")
