"""Torch replacements for the pinned vendor `kernel.py` (tilelang 0.1.12) of
DeepSeek V4.1 Flash, seeded into `sys.modules["kernel"]` when reference.py
executes the vendor's model.py.

Each function has the vendor's signature and return types. `LEDGER` records,
per replacement, the vendor lines it transcribes, its reduction order, its
casts and what each mode does; `POLICY` holds the one switch the modes share
(the quantizer inputs). The rules:

- The quantizers (`act_quant`, `fp4_act_quant`) are discrete choices. They
  compute exactly what the vendor kernels compute, in F32 on BF16 inputs (the
  vendor's kernels declare their input BF16), in every mode: under the
  trained-quantizer-input policy (`POLICY.quantizer_inputs == "bfloat16"`,
  the default, F32-TQ and F64-TQ) an F32 or f64 input is first rounded to
  BF16 once (an f64 input by round-to-odd to F32 then to nearest BF16, so it
  is a single correct rounding), so every mode picks the same codes and
  scales. A quantize-dequantize (`inplace=True`) writes code x scale, which is
  exact in BF16, F32 and f64.
- The FP8 and FP4 GEMMs follow the vendor's per-K-block structure: each K
  block's partial product (32 products of exactly representable values) is
  formed exactly in f64, rounded once to the accumulator (F32, or f64 in f64
  mode), multiplied by the activation then the weight scale, and added to a
  cleared accumulator in K order (`kernel.py:248-270`, `:530-554`). The
  output is the mode's default dtype (BF16 in bf16 mode, as the vendor's
  kernel writes; F32 and f64 keep the accumulator).
- `sparse_attn` in bf16 mode follows the vendor's online normalization over
  tiles of 64 selected keys, with the unnormalized probabilities cast to BF16
  before the PV product (`kernel.py:358-387`). In F32 and f64 modes it is one
  softmax pass over all selected keys with the row maximum, without that
  cast.
- `hc_split_sinkhorn` transcribes the vendor's Sinkhorn kernel
  (`kernel.py:421-460`) in the order of its generated code: each
  multiply-add fused, every sum over the 4 streams tilelang's butterfly, in
  F32 (f64 in f64 mode).

Where the vendor's lowering (tensor-core accumulation inside a block,
reduction trees, fused multiply-adds, `exp` and division intrinsics) can
differ from these definitions, the ledger says so; the executed vendor
control (reference_vendor_control.py) measures it.
"""

from dataclasses import dataclass
import struct

import torch

from tools.refrun import quant

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_TILE = 64  # sparse_attn's tile of selected keys (kernel.py:324)
# Sizes of the f64 working sets the GEMM and attention loops hold at once.
GEMM_PARTIAL_BYTES = 1 << 29
ATTENTION_BYTES = 1 << 28


def f32(value):
    """A Python float rounded to the nearest F32, as tilelang emits a float32
    literal."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


# The vendor kernels' float32 constants (kernel.py:46, 76, 131, 162, 165).
FP8_MAX_INV = f32(1 / FP8_MAX)
FP4_MAX_INV = f32(1.0 / FP4_MAX)
FP8_AMAX_FLOOR = f32(1e-4)
FP4_AMAX_FLOOR_E8M0 = 6 * (2 ** -126)
FP4_AMAX_FLOOR_E4M3 = 6 * (2 ** -9)


@dataclass
class Policy:
    """`quantizer_inputs`: "bfloat16" rounds every quantizer input to BF16
    first (the trained-quantizer-input modes, F32-TQ and F64-TQ); "mode"
    quantizes the F32 input as it is (the pure F32 reference). BF16 inputs
    are the same under both."""
    quantizer_inputs: str = "bfloat16"

    def document(self):
        return {"quantizer_inputs": self.quantizer_inputs}


POLICY = Policy()
QUANTIZER_INPUTS = ("bfloat16", "mode")


def set_policy(quantizer_inputs):
    if quantizer_inputs not in QUANTIZER_INPUTS:
        raise ValueError(f"unknown quantizer input policy {quantizer_inputs}; expected one of {QUANTIZER_INPUTS}")
    POLICY.quantizer_inputs = quantizer_inputs


def accumulator(dtype):
    """F32, or f64 for f64 inputs."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def round_odd_f32(x):
    """f64 to F32 with round-to-odd: the truncation toward zero, with the
    last bit set when inexact. Rounding that to BF16 (16 fewer bits) is the
    single correct rounding of x."""
    nearest = x.to(torch.float32)
    exact = nearest.double() == x
    away = nearest.double().abs() > x.abs()
    bits = nearest.view(torch.int32)
    # One step toward zero: the magnitude's bit pattern minus one.
    truncated = torch.where(away, bits - 1, bits)
    odd = torch.where(exact, truncated, truncated | 1)
    return odd.view(torch.float32)


def to_bf16(x):
    """x rounded once to the nearest BF16 (ties to even). torch's f64 to BF16
    conversion rounds through F32 and can round twice."""
    if x.dtype == torch.bfloat16:
        return x
    if x.dtype == torch.float64:
        x = round_odd_f32(x)
    return x.to(torch.bfloat16)


def quantizer_input(x):
    """The values a quantizer reads, in F32: x as BF16 under the trained
    policy (and always for BF16 inputs), else the F32 input itself."""
    if x.dtype == torch.bfloat16 or POLICY.quantizer_inputs == "bfloat16":
        return to_bf16(x).float()
    if x.dtype != torch.float32:
        raise ValueError(f"the {POLICY.quantizer_inputs} quantizer input policy takes F32 or BF16, not {x.dtype}")
    return x


def ceil_pow2(value):
    """kernel.py:22-37 fast_round_scale on a positive normal F32: 2 to the
    ceiling of log2, from the exponent and mantissa bits."""
    bits = value.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    up = ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((exponent + up) << 23).view(torch.float32)


def e4m3(values):
    """F32 to E4M3, nearest with ties to even, saturating (the vendor clamps
    to +-448 first). Returns float8_e4m3fn."""
    return values.to(torch.float8_e4m3fn)


def e2m1_codes(values):
    """F32 in [-6, 6] to E2M1 codes (sign in bit 3), nearest with ties to the
    even code: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6."""
    magnitude = values.abs()
    code = ((magnitude > 0.25).to(torch.uint8) + (magnitude >= 0.75).to(torch.uint8)
            + (magnitude > 1.25).to(torch.uint8) + (magnitude >= 1.75).to(torch.uint8)
            + (magnitude > 2.5).to(torch.uint8) + (magnitude >= 3.5).to(torch.uint8)
            + (magnitude > 5.0).to(torch.uint8))
    return code | (torch.signbit(values).to(torch.uint8) << 3)


def e2m1_values(codes):
    """E2M1 codes (one per byte) as F32."""
    return quant.table(quant.E2M1, torch.float32, codes.device).index_select(
        0, codes.reshape(-1).int()).view(codes.shape)


def pack_e2m1(codes):
    """[..., N] codes to [..., N/2] float4_e2m1fn_x2: even index in the low nibble."""
    pairs = codes.unflatten(-1, (-1, 2))
    return (pairs[..., 0] | (pairs[..., 1] << 4)).view(torch.float4_e2m1fn_x2)


def unpack_e2m1(packed):
    """[..., N/2] packed E2M1 (float4_e2m1fn_x2, int8 or uint8) to [..., N] F32."""
    codes = packed.view(torch.uint8)
    pairs = quant.table(quant.E2M1_PAIRS, torch.float32, codes.device).index_select(0, codes.reshape(-1).int())
    return pairs.view(*codes.shape[:-1], codes.shape[-1] * 2)


def scale_values(scales):
    """Stored scales as F32: E8M0 by its definition (0xFF is NaN), E4M3 and
    F32 by value."""
    if scales.dtype == torch.float8_e8m0fnu:
        codes = scales.view(torch.uint8)
        return quant.table(quant.E8M0, torch.float32, codes.device).index_select(
            0, codes.reshape(-1).int()).view(scales.shape)
    if scales.dtype == torch.float8_e4m3fn:
        return scales.to(torch.float32)
    if scales.dtype == torch.float32:
        return scales
    raise ValueError(f"no scale decoding for {scales.dtype}")


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """kernel.py:40-124. Per block of `block_size` along the last dimension:
    amax of the BF16 input in F32, floored at 1e-4; the scale amax / 448
    (F32 product with the F32 constant 1/448), rounded up to a power of two
    when `scale_fmt` is set; codes E4M3(clamp(x / scale, -448, 448)).
    Returns (codes, scales) or, with `inplace`, writes code x scale into x."""
    n = x.size(-1)
    if n % block_size:
        raise ValueError(f"act_quant: {n} columns do not split into blocks of {block_size}")
    values = quantizer_input(x).unflatten(-1, (-1, block_size))
    amax = values.abs().amax(-1).clamp_min(FP8_AMAX_FLOOR)
    scale = amax * FP8_MAX_INV
    if scale_fmt is not None:
        scale = ceil_pow2(scale)
    codes = e4m3((values / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX))
    if inplace:
        x.copy_((codes.to(torch.float32) * scale.unsqueeze(-1)).flatten(-2))
        return x
    stored = scale.to(scale_dtype) if scale_dtype != torch.float32 else scale
    return codes.flatten(-2), stored


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """kernel.py:127-204. Per block: amax of the BF16 input in F32; with E8M0
    scales floored at 6 x 2^-126 and the scale 2^ceil(log2(amax x 1/6)); with
    E4M3 scales (compressed KV) floored at 6 x 2^-9 and the scale
    E4M3(amax / 6). Codes E2M1(clamp(x / scale, -6, 6)), packed with the even
    index in the low nibble; `inplace` writes code x scale into x."""
    if scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise ValueError(f"fp4_act_quant takes E8M0 or E4M3 scales, not {scale_dtype}")
    n = x.size(-1)
    if n % block_size:
        raise ValueError(f"fp4_act_quant: {n} columns do not split into blocks of {block_size}")
    values = quantizer_input(x).unflatten(-1, (-1, block_size))
    amax = values.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        scale = e4m3(amax.clamp_min(FP4_AMAX_FLOOR_E4M3) / FP4_MAX).to(torch.float32)
    else:
        scale = ceil_pow2(amax.clamp_min(FP4_AMAX_FLOOR_E8M0) * FP4_MAX_INV)
    codes = e2m1_codes((values / scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((e2m1_values(codes) * scale.unsqueeze(-1)).flatten(-2))
        return x
    return pack_e2m1(codes.flatten(-2)), scale.to(scale_dtype)


def blocked_gemm(a, a_scale, b, b_scale, block, out_dtype):
    """sum over K blocks kb, in order, of RN(P_kb) x a_scale[:, kb] x
    b_scale[:, kb], where P_kb is block kb's exact partial product.

    `a` [M, K] and `b` [N, K] hold exactly representable values (decoded E4M3
    or E2M1) in f64, `a_scale` [M, KB] and `b_scale` [N, KB] the scale of each
    row's block. A block's partial sums 32 products of at most 4 x 4
    significant bits within 41 binades, so f64 forms it exactly in any order.
    The accumulator is F32 (f64 for an f64 output); RN is its rounding."""
    acc_dtype = accumulator(out_dtype)
    m, k = a.shape
    n = b.shape[0]
    blocks = k // block
    a_blocks = a.view(m, blocks, block).transpose(0, 1)
    b_blocks = b.view(n, blocks, block).permute(1, 2, 0)
    a_scale = a_scale.to(acc_dtype).t().contiguous()
    b_scale = b_scale.to(acc_dtype).t().contiguous()
    out = torch.zeros(m, n, dtype=acc_dtype, device=a.device)
    per_block = max(1, GEMM_PARTIAL_BYTES // (8 * n))
    rows = max(1, min(m, per_block))
    group = max(1, min(blocks, per_block // rows))
    for start in range(0, m, rows):
        stop = min(m, start + rows)
        acc = out[start:stop]
        for first in range(0, blocks, group):
            last = min(blocks, first + group)
            partial = torch.bmm(a_blocks[first:last, start:stop], b_blocks[first:last])
            fold(partial, a_scale[first:last, start:stop], b_scale[first:last], acc)
    return out.to(out_dtype)


def fold(partial, a_scale, b_scale, acc):
    """acc = RN(acc + RN(partial[g]) x a_scale[g, :, None] x b_scale[g]) for
    g in order. On CUDA one NVRTC kernel does it (a loop over the K blocks
    would launch several kernels per block); elsewhere the same IEEE
    operations run as a torch loop. The scales are powers of two, so every
    product is exact."""
    if acc.is_cuda:
        return fold_cuda(partial, a_scale, b_scale, acc)
    for index in range(partial.shape[0]):
        term = partial[index].to(acc.dtype)
        term.mul_(a_scale[index, :, None]).mul_(b_scale[index])
        acc.add_(term)
    return acc


# Each element folds its K blocks in order with explicitly rounded operations
# (no contraction into fused multiply-adds), F32 or f64.
FOLD_SOURCE = r"""
extern "C" __global__ void fold_f32(const double* partial, const float* a, const float* b, float* acc,
                                    long long rows, long long cols, int groups) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long long r = i / cols, c = i % cols;
  float total = acc[i];
  for (int g = 0; g < groups; ++g) {
    float term = __double2float_rn(partial[(long long)g * rows * cols + i]);
    term = __fmul_rn(__fmul_rn(term, a[(long long)g * rows + r]), b[(long long)g * cols + c]);
    total = __fadd_rn(total, term);
  }
  acc[i] = total;
}
extern "C" __global__ void fold_f64(const double* partial, const double* a, const double* b, double* acc,
                                    long long rows, long long cols, int groups) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long long r = i / cols, c = i % cols;
  double total = acc[i];
  for (int g = 0; g < groups; ++g) {
    double term = __dmul_rn(__dmul_rn(partial[(long long)g * rows * cols + i], a[(long long)g * rows + r]),
                            b[(long long)g * cols + c]);
    total = __dadd_rn(total, term);
  }
  acc[i] = total;
}
"""
_FOLD = {}


def checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA call failed: {error}")
    return values[0] if len(values) == 1 else values


def fold_program(capability):
    """The fold kernels compiled by NVRTC for one compute capability (cubin)."""
    key = ("cubin", capability)
    if key not in _FOLD:
        from cuda.bindings import nvrtc
        program = checked(nvrtc.nvrtcCreateProgram(FOLD_SOURCE.encode(), b"fold.cu", 0, [], []))
        options = [f"--gpu-architecture=sm_{capability[0]}{capability[1]}".encode(), b"--fmad=false"]
        error, = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(error) != 0:
            size = checked(nvrtc.nvrtcGetProgramLogSize(program))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"NVRTC failed to compile the fold kernels: {log.decode(errors='replace')}")
        size = checked(nvrtc.nvrtcGetCUBINSize(program))
        cubin = b" " * size
        checked(nvrtc.nvrtcGetCUBIN(program, cubin))
        nvrtc.nvrtcDestroyProgram(program)
        _FOLD[key] = cubin
    return _FOLD[key]


def fold_function(device, wide):
    """The loaded fold kernel on `device` (a module per device context)."""
    key = (device.index, wide)
    if key not in _FOLD:
        from cuda.bindings import driver
        cubin = fold_program(torch.cuda.get_device_capability(device))
        with torch.cuda.device(device):
            torch.cuda.current_stream().synchronize()
            module = checked(driver.cuModuleLoadData(cubin))
            _FOLD[key] = checked(driver.cuModuleGetFunction(module, b"fold_f64" if wide else b"fold_f32"))
    return _FOLD[key]


def fold_cuda(partial, a_scale, b_scale, acc):
    import ctypes
    from cuda.bindings import driver
    groups, rows, cols = partial.shape
    if not (partial.is_contiguous() and acc.is_contiguous()) or partial.dtype != torch.float64:
        raise ValueError("fold needs contiguous f64 partials and a contiguous accumulator")
    a_scale = a_scale.to(acc.dtype).contiguous()
    b_scale = b_scale.to(acc.dtype).contiguous()
    function = fold_function(acc.device, acc.dtype == torch.float64)
    block = 256
    with torch.cuda.device(acc.device):
        stream = torch.cuda.current_stream()
        arguments = ((partial.data_ptr(), a_scale.data_ptr(), b_scale.data_ptr(), acc.data_ptr(), rows, cols,
                      groups),
                     (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
                      ctypes.c_longlong, ctypes.c_int))
        checked(driver.cuLaunchKernel(function, -(-rows * cols // block), 1, 1, block, 1, 1, 0,
                                      driver.CUstream(stream.cuda_stream), arguments, 0))
    return acc


def check_contiguous(*tensors):
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("the GEMM operands and scales must be contiguous")


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """kernel.py:207-307: C[M, N] = A[M, K] B[N, K]^T with E4M3 operands,
    activation scales per row and K block, weight scales per (N block, K
    block), both of `block_size`; per-K-block accumulation (blocked_gemm)."""
    check_contiguous(a, b, a_s, b_s)
    if block_size not in (32, 128):
        raise ValueError(f"fp8_gemm takes blocks of 32 or 128, not {block_size}")
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    if k % block_size or a_s.numel() != m * (k // block_size) \
            or tuple(b_s.shape) != (-(-n // block_size), k // block_size):
        raise ValueError(f"fp8_gemm scales {list(a_s.shape)}, {list(b_s.shape)} do not fit A {list(a.shape)}, "
                         f"B {list(b.shape)} in blocks of {block_size}")
    lhs = a.view(m, k).view(torch.float8_e4m3fn).to(torch.float64)
    rhs = b.view(torch.float8_e4m3fn).to(torch.float64)
    weight_scale = scale_values(b_s).repeat_interleave(block_size, 0)[:n]
    out = blocked_gemm(lhs, scale_values(a_s.view(m, -1)), rhs, weight_scale, block_size,
                       torch.get_default_dtype())
    return out.view(*a.shape[:-1], n)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """kernel.py:477-591: C[M, N] = A_fp8[M, K] B_fp4[N, K]^T. B is packed
    E2M1 [N, K/2] (even K in the low nibble) with one E8M0 scale per 32 K;
    the vendor casts each E2M1 value to E4M3 (exact) and accumulates per
    32-K block whatever the activation block (blocked_gemm with block 32)."""
    check_contiguous(a, b, a_s, b_s)
    if act_block_size not in (32, 128):
        raise ValueError(f"fp4_gemm takes activation blocks of 32 or 128, not {act_block_size}")
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    if k % act_block_size or a_s.numel() != m * (k // act_block_size) or tuple(b_s.shape) != (n, k // 32) \
            or b.size(-1) * 2 != k:
        raise ValueError(f"fp4_gemm operands A {list(a.shape)}, B {list(b.shape)} and scales "
                         f"{list(a_s.shape)}, {list(b_s.shape)} do not fit")
    lhs = a.view(m, k).view(torch.float8_e4m3fn).to(torch.float64)
    rhs = unpack_e2m1(b).to(torch.float64)
    activation_scale = scale_values(a_s.view(m, -1)).repeat_interleave(act_block_size // 32, 1)
    out = blocked_gemm(lhs, activation_scale, rhs, scale_values(b_s), 32, torch.get_default_dtype())
    return out.view(*a.shape[:-1], n)


def exact_dot(lhs, rhs, equation, dtype):
    """einsum in f64, rounded once to `dtype`."""
    return torch.einsum(equation, lhs.double(), rhs.double()).to(dtype)


def gathered(kv, idxs):
    """kv rows [rows, topk, d] at idxs (zero where idxs is -1) and the valid mask."""
    valid = idxs != -1
    rows = kv.index_select(0, idxs.clamp_min(0).reshape(-1).long()).view(*idxs.shape, kv.size(-1))
    return rows * valid.unsqueeze(-1).to(rows.dtype), valid


def attention_tiles(q, keys, valid, sink, scale):
    """kernel.py:350-387 for a block of query rows: q [r, h, d] BF16, keys
    [r, t, d] BF16 (t a multiple of 64), valid [r, t]. Per tile of 64 keys:
    scores as an exact dot rounded to F32 (-inf where invalid), times the F32
    scale; the running max (starting at -1e30); the rescale exp(old - new);
    exp(score - max); the tile sum (exact, rounded once); sum = sum x rescale
    + tile sum; P cast to BF16; acc = acc x rescale, then acc + P V (exact,
    rounded once). Finally sum += exp(sink - max) and out = acc / sum, as BF16."""
    rows, heads = q.shape[0], q.shape[1]
    scale = f32(scale)
    maximum = torch.full((rows, heads), -1e30, dtype=torch.float32, device=q.device)
    total = torch.zeros(rows, heads, dtype=torch.float32, device=q.device)
    acc = torch.zeros(rows, heads, q.shape[2], dtype=torch.float32, device=q.device)
    for start in range(0, keys.shape[1], FP8_BLOCK_TILE):
        tile = keys[:, start:start + FP8_BLOCK_TILE]
        live = valid[:, start:start + FP8_BLOCK_TILE].unsqueeze(1)
        scores = exact_dot(q, tile, "rhd,rtd->rht", torch.float32)
        scores = torch.where(live, scores, float("-inf")) * scale
        previous = maximum
        maximum = torch.maximum(previous, scores.amax(-1))
        rescale = torch.exp(previous - maximum)
        weights = torch.exp(scores - maximum.unsqueeze(-1))
        total = total * rescale + weights.double().sum(-1).float()
        acc = acc * rescale.unsqueeze(-1)
        acc = (acc.double() + torch.einsum("rht,rtd->rhd", weights.to(torch.bfloat16).double(),
                                           tile.double())).float()
    total = total + torch.exp(sink.float().unsqueeze(0) - maximum)
    return (acc / total.unsqueeze(-1)).to(torch.bfloat16)


def attention_exact(q, keys, valid, sink, scale):
    """The F32 and f64 modes: one pass over all selected keys in the input
    dtype D: scores as an exact dot rounded to D times the scale, the row
    maximum (at least -1e30), P = exp(score - max) uncast, out = (P V) /
    (sum P + exp(sink - max)), with P V and sum P exact and rounded once."""
    dtype = q.dtype
    scores = exact_dot(q, keys, "rhd,rtd->rht", dtype) * scale
    scores = torch.where(valid.unsqueeze(1), scores, float("-inf"))
    maximum = scores.amax(-1).clamp_min(-1e30)
    weights = torch.exp(scores - maximum.unsqueeze(-1))
    total = weights.double().sum(-1).to(dtype) + torch.exp(sink.to(dtype).unsqueeze(0) - maximum)
    return exact_dot(weights, keys, "rht,rtd->rhd", dtype) / total.unsqueeze(-1)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:310-403: for each query, attention over the kv rows its
    `topk_idxs` name (-1 names none) with the per-head sink in the
    normalization. q [b, m, h, d], kv [b, n, d], attn_sink [h] F32, topk_idxs
    [b, m, topk] int32; out like q. BF16 inputs follow the vendor's tiles
    (attention_tiles); F32 and f64 inputs use attention_exact."""
    batch, length, heads, dim = q.shape
    if kv.dtype != q.dtype:
        raise ValueError(f"sparse_attn takes q and kv of one dtype, not {q.dtype} and {kv.dtype}")
    out = torch.empty_like(q)
    topk = topk_idxs.shape[-1]
    width = -(-topk // FP8_BLOCK_TILE) * FP8_BLOCK_TILE
    rows = max(1, ATTENTION_BYTES // (8 * max(width, 1) * dim))
    for b in range(batch):
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            idxs = topk_idxs[b, start:stop]
            if width > topk:
                idxs = torch.cat([idxs, idxs.new_full((stop - start, width - topk), -1)], dim=-1)
            keys, valid = gathered(kv[b], idxs)
            if q.dtype == torch.bfloat16:
                out[b, start:stop] = attention_tiles(q[b, start:stop], keys, valid, attn_sink, softmax_scale)
            else:
                out[b, start:stop] = attention_exact(q[b, start:stop], keys, valid, attn_sink, softmax_scale)
    return out


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def fma(a, b, c):
    """a x b + c rounded once to F32, as nvcc contracts the vendor's
    multiply-add: the product of two F32 values is exact in f64, and the
    f64 sum then rounds to F32 (a second rounding only when the f64 sum
    lies within 2^-29 ulp of an F32 midpoint). f64 inputs: a x b + c."""
    if a.dtype == torch.float64:
        return a * b + c
    return (a.double() * b.double() + c.double()).float()


def butterfly_sum(values, dim):
    """tilelang's AllReduce over a power-of-two width: at each offset from
    half the width down to 1, x + x[index ^ offset], so 4 values add as
    (x0 + x2) + (x1 + x3)."""
    width = values.shape[dim]
    while width > 1:
        half = width // 2
        values = values.narrow(dim, 0, half) + values.narrow(dim, half, half)
        width = half
    return values.squeeze(dim)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py:406-474: pre = sigmoid(m x s0 + base) + eps, post = 2
    sigmoid(m x s1 + base), comb = m x s2 + base as [hc, hc]; comb = row
    softmax + eps, then divided by (column sum + eps), then sinkhorn_iters - 1
    rounds of row then column normalization (sum + eps). F32 (f64 for f64
    mixes) with the generated kernel's order: each m x s + base one fused
    multiply-add, every sum over the streams tilelang's butterfly, exp and
    division IEEE (the vendor does not enable fast math); eps the F32
    literal."""
    dtype = accumulator(mixes.dtype)
    eps = f32(eps) if dtype == torch.float32 else eps
    batch, length, _ = mixes.shape
    mixes = mixes.to(dtype).reshape(-1, (2 + hc_mult) * hc_mult)
    scale, base = hc_scale.to(dtype), hc_base.to(dtype)
    pre = sigmoid(fma(mixes[:, :hc_mult], scale[0], base[:hc_mult])) + eps
    post = 2 * sigmoid(fma(mixes[:, hc_mult:2 * hc_mult], scale[1], base[hc_mult:2 * hc_mult]))
    comb = fma(mixes[:, 2 * hc_mult:], scale[2], base[2 * hc_mult:]).view(-1, hc_mult, hc_mult)
    row_max = comb.amax(-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / butterfly_sum(comb, 2).unsqueeze(-1) + eps
    comb = comb / (butterfly_sum(comb, 1).unsqueeze(1) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (butterfly_sum(comb, 2).unsqueeze(-1) + eps)
        comb = comb / (butterfly_sum(comb, 1).unsqueeze(1) + eps)
    return (pre.view(batch, length, hc_mult), post.view(batch, length, hc_mult),
            comb.view(batch, length, hc_mult, hc_mult))


LOWERING = ("the vendor's lowering may differ in the last bit of F32 values: tensor-core accumulation order and "
            "internal precision inside a block, reduction trees, fused multiply-adds and the exp and division "
            "intrinsics; the executed vendor control classifies each difference")
LEDGER = {
    "act_quant": {
        "vendor": "kernel.py:40-124",
        "reduction": "block amax (order-free)",
        "casts": "input rounded once to BF16 (the vendor kernel's input dtype); amax floored at F32 1e-4; scale = "
                 "amax x F32(1/448), rounded up to a power of two (exponent bits) when scale_fmt is set; codes "
                 "E4M3 nearest-even of clamp(x / scale, +-448); inplace writes code x scale (exact in BF16, F32 "
                 "and f64)",
        "modes": {"bf16": "the vendor's computation", "f32": "the same codes and scales from the BF16-rounded input "
                  "(trained quantizer inputs)", "f64": "as f32; the f64 input rounds to BF16 by round-to-odd to "
                  "F32 then nearest (torch's direct f64 to BF16 cast rounds twice)"},
        "claim": "bitwise on BF16 inputs",
        "vendor_control": "bitwise against the executed kernel (tilelang 0.1.12, SM120, 2026-10-17): blocks of 32 "
                          "with E8M0 scales and 128 with F32 scales, codes, scales and quantize-dequantize, over "
                          "scale boundaries, zero blocks and E4M3 midpoints",
    },
    "fp4_act_quant": {
        "vendor": "kernel.py:127-204",
        "reduction": "block amax (order-free)",
        "casts": "input rounded once to BF16; E8M0 scales: amax floored at 6 x 2^-126, scale 2^ceil(log2(amax x "
                 "F32(1/6))); E4M3 scales: amax floored at 6 x 2^-9, scale E4M3(amax / 6) (an F32 division); "
                 "codes E2M1 nearest-even of clamp(x / scale, +-6), sign kept for zeros; packed even index in "
                 "the low nibble; inplace writes code x scale",
        "modes": {"bf16": "the vendor's computation", "f32": "as bf16 on the BF16-rounded input", "f64": "as f32"},
        "claim": "bitwise on BF16 inputs",
        "vendor_control": "bitwise against the executed kernel (SM120, 2026-10-17): E8M0 scales per 32 and E4M3 "
                          "scales per 16, packed codes, scales and quantize-dequantize; the vendor's division is "
                          "IEEE (fast math off)",
    },
    "fp8_gemm": {
        "vendor": "kernel.py:207-307",
        "reduction": "per K block of block_size: the exact partial product (f64), rounded once to the accumulator; "
                     "then x activation scale x weight scale and added to a cleared accumulator, K blocks in order",
        "casts": "accumulator F32 (f64 in f64 mode); output the default dtype",
        "modes": {"bf16": "output rounded to BF16 as the vendor's kernel writes", "f32": "output kept in F32",
                  "f64": "partials and accumulator f64, output f64"},
        "claim": "bitwise where the vendor's tensor cores form each 32-K block's sum exactly and round once; "
                 + LOWERING,
        "vendor_control": "lowering (SM120, 2026-10-17): 1 of 16,896 BF16 outputs at M 33, N 512, K 5120 differs by "
                          "1 ulp (1.5e-4 of its row's largest value), the replacement the closer to the f64 "
                          "product; M 1, N 256, K 512 and M 100, N 256, K 1280 bitwise",
    },
    "fp4_gemm": {
        "vendor": "kernel.py:477-591",
        "reduction": "as fp8_gemm with 32-K blocks whatever the activation block (block_K = 32), E2M1 cast to E4M3 "
                     "exactly",
        "casts": "as fp8_gemm",
        "modes": {"bf16": "output BF16", "f32": "output F32", "f64": "f64 throughout"},
        "claim": "as fp8_gemm",
        "vendor_control": "bitwise (SM120, 2026-10-17) at M 1, N 256, K 512; M 40, N 2304, K 5120 and M 17, "
                          "N 5120, K 2304",
    },
    "sparse_attn": {
        "vendor": "kernel.py:310-403",
        "reduction": "bf16: tiles of 64 selected keys in index order with the online maximum from -1e30; each "
                     "tile's scores, sum and P V exact and rounded once to F32. f32 and f64: one pass with the row "
                     "maximum, sums exact and rounded once",
        "casts": "bf16: scores x F32 scale, unnormalized P cast to BF16 before P V, output BF16. f32 and f64: no "
                 "cast of P; the sink joins the normalization as exp(sink - max); output the input dtype",
        "modes": {"bf16": "the vendor's tiles", "f32": "one pass, P uncast", "f64": "one pass in f64"},
        "claim": "bf16 tiles as the vendor's, bitwise only where its tensor cores and reductions round as exact "
                 "sums; " + LOWERING,
        "vendor_control": "lowering, not bitwise (SM120, 2026-10-17): at 16 heads (TP4) 1,529 of 393,216 BF16 "
                          "outputs differ over 640 selected keys and 604 of 393,216 over 130, at 8 heads (TP8) "
                          "532 of 196,608; each difference is within 6.1e-3 of its output row's largest value "
                          "(under one BF16 unit there), and neither side is closer to the f64 oracle (736 against "
                          "793, 297 against 307 and 260 against 272 elements). The generated kernel chains its "
                          "MMAs over 16 dims and 16 keys and sums each tile in thread order; a replacement "
                          "following those chains with exact steps differed as often, so the tiles stay exact. "
                          "64 heads (TP1) do not run: the vendor kernel asks for 141,312 bytes of shared memory, "
                          "beyond SM120's",
    },
    "hc_split_sinkhorn": {
        "vendor": "kernel.py:406-474",
        "reduction": "every sum over the 4 streams as tilelang's butterfly, (x0 + x2) + (x1 + x3); each "
                     "m x scale + base one fused multiply-add, as nvcc contracts it",
        "casts": "F32 throughout, eps the F32 literal; sigmoid as 1 / (1 + exp(-x)); exp and division IEEE",
        "modes": {"bf16": "F32 (the vendor's mixes are F32)", "f32": "F32", "f64": "f64 on f64 mixes"},
        "claim": "bitwise on the GPU, where torch's exp is the same CUDA expf the vendor's code calls",
        "vendor_control": "bitwise (SM120, 2026-10-17) over 4,096 rows; with sums in index order and unfused "
                          "multiply-adds it differed by up to 46 F32 ulps in comb, neither side closer to the "
                          "f64 oracle",
    },
}
