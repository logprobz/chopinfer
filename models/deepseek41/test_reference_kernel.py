"""reference_kernel.py against independent NumPy transcriptions of the
vendor's tilelang kernels (kernel.py at revision 1bae0e10), on CPU.

Each transcription follows the kernel's loops element by element in NumPy
float32 (and Python floats for exact sums), with its own E4M3 and E2M1
rounding (nearest on the code grid of tools/refrun/quant.py's definitions,
ties to the even code), its own power-of-two scale (frexp) and its own BF16
rounding (on the F32 bits). Only `exp` is shared with torch, so an
elementary function's last bit, which differs between libraries, is not
what the tests compare. BF16-mode results must be bitwise equal; the F32
and f64 modes are checked against their definitions.
"""

import math
import unittest

try:
    import numpy as np
    import torch
    from tools.refrun import quant
    from models.deepseek41 import reference_kernel as kernel
except ModuleNotFoundError:
    torch = None

F32 = None if torch is None else np.float32


def e4m3_grid():
    """(values, codes) of every finite E4M3 code, sorted by value, -0 left out."""
    pairs = sorted((quant.E4M3[c], c) for c in range(256) if not math.isnan(quant.E4M3[c]) and c != 0x80)
    return np.array([v for v, _ in pairs]), np.array([c for _, c in pairs])


def e2m1_grid():
    pairs = sorted((quant.E2M1[c], c) for c in range(16) if c != 0x8)
    return np.array([v for v, _ in pairs]), np.array([c for _, c in pairs])


def nearest_even(value, values, codes):
    """The grid value nearest `value`, a tie to the even code (its mantissa's
    last bit clear), saturating at the ends."""
    value = float(value)
    if value >= values[-1]:
        return float(values[-1])
    if value <= values[0]:
        return float(values[0])
    upper = int(np.searchsorted(values, value))
    lo, hi = upper - 1, upper
    if values[hi] == value:
        return float(values[hi])
    below, above = value - values[lo], values[hi] - value
    if below < above:
        choice = lo
    elif above < below:
        choice = hi
    else:
        choice = lo if codes[lo] % 2 == 0 else hi
    result = float(values[choice])
    return -0.0 if result == 0.0 and math.copysign(1.0, value) < 0 else result


def bf16_round(x):
    """F32 to BF16 (nearest, ties to even) on the bits, as F32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def pow2_ceil(value):
    """2^ceil(log2(value)) for a positive normal value, by frexp."""
    mantissa, exponent = math.frexp(float(value))
    return math.ldexp(1.0, exponent - 1) if mantissa == 0.5 else math.ldexp(1.0, exponent)


def np_act_quant(x, block, pow2=True):
    """kernel.py:71-93 over BF16 values x [rows, n]: codes as values, scales."""
    values, codes = e4m3_grid()
    rows, n = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    scales = np.zeros((rows, n // block), dtype=np.float32)
    for r in range(rows):
        for b in range(n // block):
            seg = x[r, b * block:(b + 1) * block].astype(np.float32)
            amax = max(F32(np.abs(seg).max()), F32(1e-4))
            scale = F32(amax) * F32(1 / 448)
            if pow2:
                scale = F32(pow2_ceil(scale))
            scales[r, b] = scale
            for j in range(block):
                q = F32(seg[j]) / F32(scale)
                q = min(max(q, F32(-448.0)), F32(448.0))
                out[r, b * block + j] = nearest_even(q, values, codes)
    return out, scales


def np_fp4_quant(x, block, e4m3_scale):
    """kernel.py:155-179 over BF16 values: E2M1 values and scales."""
    values, codes = e2m1_grid()
    e4_values, e4_codes = e4m3_grid()
    rows, n = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    scales = np.zeros((rows, n // block), dtype=np.float32)
    for r in range(rows):
        for b in range(n // block):
            seg = x[r, b * block:(b + 1) * block].astype(np.float32)
            amax = F32(np.abs(seg).max())
            if e4m3_scale:
                amax = max(amax, F32(6 * 2 ** -9))
                scale = F32(nearest_even(F32(amax) / F32(6.0), e4_values, e4_codes))
            else:
                amax = max(amax, F32(6 * 2 ** -126))
                scale = F32(pow2_ceil(F32(amax) * F32(1.0 / 6.0)))
            scales[r, b] = scale
            for j in range(block):
                q = F32(seg[j]) / F32(scale)
                q = min(max(q, F32(-6.0)), F32(6.0))
                out[r, b * block + j] = nearest_even(q, values, codes)
    return out, scales


def np_blocked_gemm(a, sa, b, sb, block, acc=None):
    """kernel.py:251-270 per output: per K block the exact partial (fsum of
    exact products), rounded to the accumulator, x sa x sb, added in order."""
    acc = acc or np.float32
    m, k = a.shape
    n = b.shape[0]
    out = np.zeros((m, n), dtype=acc)
    for i in range(m):
        for j in range(n):
            total = acc(0)
            for kb in range(k // block):
                part = math.fsum(float(a[i, t]) * float(b[j, t]) for t in range(kb * block, (kb + 1) * block))
                total = acc(total + acc(acc(acc(part) * acc(sa[i, kb])) * acc(sb[j, kb])))
            out[i, j] = total
    return out


def torch_exp(x):
    return torch.exp(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()


def np_sparse_attn(q, kv, sink, idxs, scale):
    """kernel.py:350-387 for one batch: q [m, h, d] BF16 values (F32), kv
    [n, d], idxs [m, topk]; tiles of 64; exact tile dots rounded to F32."""
    m, h, d = q.shape
    topk = idxs.shape[1]
    tiles = -(-topk // 64)
    out = np.zeros((m, h, d), dtype=np.float32)
    scale = F32(scale)
    for row in range(m):
        maximum = np.full(h, F32(-1e30), dtype=np.float32)
        total = np.zeros(h, dtype=np.float32)
        acc = np.zeros((h, d), dtype=np.float32)
        for t in range(tiles):
            ids = [int(idxs[row, t * 64 + i]) if t * 64 + i < topk else -1 for i in range(64)]
            keys = np.array([kv[i] if i != -1 else np.zeros(d, dtype=np.float32) for i in ids])
            scores = np.zeros((h, 64), dtype=np.float32)
            for head in range(h):
                for i in range(64):
                    if ids[i] == -1:
                        scores[head, i] = -np.inf
                    else:
                        scores[head, i] = F32(math.fsum(float(x) * float(y) for x, y in zip(q[row, head], keys[i])))
            scores = (scores * scale).astype(np.float32)
            previous = maximum.copy()
            maximum = np.maximum(previous, scores.max(axis=1)).astype(np.float32)
            rescale = torch_exp(previous - maximum)
            weights = torch_exp(scores - maximum[:, None])
            sums = np.array([F32(math.fsum(float(w) for w in weights[head])) for head in range(h)], dtype=np.float32)
            total = (total * rescale + sums).astype(np.float32)
            acc = (acc * rescale[:, None]).astype(np.float32)
            pb = bf16_round(weights)
            for head in range(h):
                for c in range(d):
                    acc[head, c] = F32(math.fsum([float(acc[head, c])]
                                                 + [float(pb[head, i]) * float(keys[i, c]) for i in range(64)]))
        total = (total + torch_exp(sink.astype(np.float32) - maximum)).astype(np.float32)
        out[row] = bf16_round((acc / total[:, None]).astype(np.float32))
    return out


def np_sinkhorn(mixes, scale, base, hc, iters, eps):
    """kernel.py:421-460 for one row in float32, in the order of the
    generated CUDA: m x s + b fused (one rounding of the exact value, here
    through a Python float), sums over the 4 streams as tilelang's
    AllReduce, (0 + 2) + (1 + 3)."""
    eps = F32(eps)

    def sigmoid(v):
        return (F32(1) / (F32(1) + torch_exp(-v))).astype(np.float32)

    def fused(m, s, b):
        return np.array([F32(float(x) * float(s) + float(y)) for x, y in zip(m, b)], dtype=np.float32)
    m = mixes.astype(np.float32)
    pre = (sigmoid(fused(m[:hc], scale[0], base[:hc])) + eps).astype(np.float32)
    post = (F32(2) * sigmoid(fused(m[hc:2 * hc], scale[1], base[hc:2 * hc]))).astype(np.float32)
    comb = fused(m[2 * hc:], scale[2], base[2 * hc:]).reshape(hc, hc)

    def butterfly(values):
        return ((values[0] + values[2]).astype(np.float32) + (values[1] + values[3]).astype(np.float32)) \
            .astype(np.float32)

    def row_sums(c):
        return butterfly([c[:, k] for k in range(hc)])

    def col_sums(c):
        return butterfly([c[j] for j in range(hc)])
    comb = torch_exp((comb - comb.max(axis=1, keepdims=True)).astype(np.float32))
    comb = (comb / row_sums(comb)[:, None] + eps).astype(np.float32)
    comb = (comb / (col_sums(comb)[None, :] + eps)).astype(np.float32)
    for _ in range(iters - 1):
        comb = (comb / (row_sums(comb)[:, None] + eps)).astype(np.float32)
        comb = (comb / (col_sums(comb)[None, :] + eps)).astype(np.float32)
    return pre, post, comb


def bf16_inputs(generator, shape, spread=3.0):
    """BF16 values over several binades, with zeros and a few extremes."""
    x = torch.randn(shape, generator=generator) * torch.exp(torch.randn(shape, generator=generator) * spread)
    x.view(-1)[::17] = 0.0
    return x.to(torch.bfloat16)


@unittest.skipIf(torch is None, "needs torch and numpy")
class Codes(unittest.TestCase):
    def test_e8m0_scales_decode_by_definition(self):
        codes = torch.arange(256, dtype=torch.uint8)
        values = kernel.scale_values(codes.view(torch.float8_e8m0fnu))
        for code in range(255):
            self.assertEqual(values[code].item(), math.ldexp(1.0, code - 127))
        self.assertTrue(math.isnan(values[255].item()))

    def test_e2m1_nibble_order_and_values(self):
        # 0x21: low nibble 1 (0.5) is the even index, high nibble 2 (1.0) the odd one.
        packed = torch.tensor([[0x21, 0xF8, 0x7E]], dtype=torch.uint8)
        self.assertEqual(kernel.unpack_e2m1(packed).tolist(), [[0.5, 1.0, -0.0, -6.0, -4.0, 6.0]])
        codes = torch.arange(16, dtype=torch.uint8)
        for code in range(16):
            self.assertEqual(kernel.e2m1_values(codes)[code].item(), quant.E2M1[code])
        again = kernel.pack_e2m1(torch.tensor([[1, 2, 8, 15, 14, 7]], dtype=torch.uint8))
        self.assertEqual(again.view(torch.uint8).tolist(), packed.tolist())

    def test_e2m1_rounding_is_nearest_even(self):
        values, codes = e2m1_grid()
        probes = sorted({*np.linspace(-6, 6, 4801).tolist(), *(v + d for v in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
                                                              for d in (-1e-6, 0.0, 1e-6))})
        x = torch.tensor(probes, dtype=torch.float32)
        got = kernel.e2m1_values(kernel.e2m1_codes(x))
        for probe, value in zip(probes, got.tolist()):
            self.assertEqual(value, nearest_even(F32(probe), values, codes), probe)

    def test_e4m3_rounding_is_nearest_even(self):
        values, codes = e4m3_grid()
        mids = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        probes = sorted({*mids, *(m * (1 + 2 ** -20) for m in mids), *(m * (1 - 2 ** -20) for m in mids)})
        x = torch.tensor(probes, dtype=torch.float32)
        got = kernel.e4m3(x).to(torch.float32)
        for probe, value in zip(probes, got.tolist()):
            self.assertEqual(value, nearest_even(F32(probe), values, codes), probe)

    def test_ceil_pow2(self):
        for value in (1.0, 1.5, 2 ** -20, 3 * 2 ** -20, 448.0, 0.75, 2 ** 100 * 1.0000001):
            got = kernel.ceil_pow2(torch.tensor([value], dtype=torch.float32)).item()
            self.assertEqual(got, pow2_ceil(F32(value)))

    def test_f64_to_bf16_rounds_once(self):
        # 1 + 2^-8 + 2^-40 is just above a BF16 midpoint; through F32 it lands
        # on the midpoint and rounds to even (1.0).
        x = torch.tensor([1 + 2 ** -8 + 2 ** -40, 1 + 2 ** -8, -(1 + 3 * 2 ** -8 - 2 ** -45), 2 ** -130 * 1.3],
                         dtype=torch.float64)
        got = kernel.to_bf16(x).double().tolist()
        self.assertEqual(got[:3], [1 + 2 ** -7, 1.0, -(1 + 2 ** -7)])
        generator = torch.Generator().manual_seed(5)
        x = torch.randn(4096, generator=generator, dtype=torch.float64) * 1e3
        rounded = kernel.to_bf16(x)
        bits = rounded.view(torch.int16)
        for step in (-1, 1):
            neighbor = (bits + step).view(torch.bfloat16).double()
            self.assertTrue(bool(((rounded.double() - x).abs() <= (neighbor - x).abs()).all()))


@unittest.skipIf(torch is None, "needs torch and numpy")
class Quantizers(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20261017)

    def test_act_quant_matches_transcription(self):
        x = bf16_inputs(self.generator, (5, 128))
        x[2] = 0.0
        for block in (32, 128):
            codes, scales = kernel.act_quant(x, block, "ue8m0", torch.float8_e8m0fnu)
            expected, expected_scales = np_act_quant(x.float().numpy(), block)
            self.assertEqual(codes.dtype, torch.float8_e4m3fn)
            self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
            np.testing.assert_array_equal(codes.to(torch.float64).numpy(), expected)
            np.testing.assert_array_equal(kernel.scale_values(scales).numpy(), expected_scales)
            # Without scale_fmt the scale stays amax / 448 in F32.
            codes, scales = kernel.act_quant(x, block)
            expected, expected_scales = np_act_quant(x.float().numpy(), block, pow2=False)
            np.testing.assert_array_equal(codes.to(torch.float64).numpy(), expected)
            np.testing.assert_array_equal(scales.numpy(), expected_scales)

    def test_act_quant_inplace_writes_code_times_scale(self):
        x = bf16_inputs(self.generator, (3, 64))
        codes, scales = kernel.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu)
        expected = (codes.float().view(3, 2, 32) * kernel.scale_values(scales).unsqueeze(-1)).view(3, 64)
        y = x.clone()
        self.assertIs(kernel.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True), y)
        self.assertTrue(torch.equal(y.float(), expected))

    def test_fp4_act_quant_both_scale_kinds(self):
        x = bf16_inputs(self.generator, (4, 64))
        x[1, :16] = 0.0
        for e4m3_scale, block, dtype in ((False, 32, torch.float8_e8m0fnu), (True, 16, torch.float8_e4m3fn)):
            packed, scales = kernel.fp4_act_quant(x, block, scale_dtype=dtype)
            expected, expected_scales = np_fp4_quant(x.float().numpy(), block, e4m3_scale)
            self.assertEqual(packed.dtype, torch.float4_e2m1fn_x2)
            self.assertEqual(tuple(packed.shape), (4, 32))
            np.testing.assert_array_equal(kernel.unpack_e2m1(packed).double().numpy(), expected)
            np.testing.assert_array_equal(kernel.scale_values(scales).numpy(), expected_scales)
            y = x.clone()
            kernel.fp4_act_quant(y, block, True, dtype)
            scale = np.repeat(expected_scales, block, axis=1)
            np.testing.assert_array_equal(y.float().numpy(), (expected * scale).astype(np.float32))

    def test_trained_inputs_round_f32_and_f64_to_bf16(self):
        x = (torch.randn(4, 64, generator=self.generator, dtype=torch.float64) * 3)
        for value in (x, x.float()):
            reference = kernel.act_quant(kernel.to_bf16(value), 32, "ue8m0", torch.float8_e8m0fnu)
            codes, scales = kernel.act_quant(value, 32, "ue8m0", torch.float8_e8m0fnu)
            self.assertTrue(torch.equal(codes.view(torch.uint8), reference[0].view(torch.uint8)))
            self.assertTrue(torch.equal(scales.view(torch.uint8), reference[1].view(torch.uint8)))
        y = x.clone()
        kernel.fp4_act_quant(y, 32, True)
        z = kernel.to_bf16(x)
        kernel.fp4_act_quant(z, 32, True)
        self.assertTrue(torch.equal(y, z.double()))
        try:
            kernel.set_policy("mode")
            unrounded = x.float() * (1 + 2 ** -12)
            a = kernel.act_quant(unrounded, 32, "ue8m0", torch.float8_e8m0fnu)[0]
            kernel.set_policy("bfloat16")
            b = kernel.act_quant(unrounded, 32, "ue8m0", torch.float8_e8m0fnu)[0]
            self.assertFalse(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            with self.assertRaises(ValueError):
                kernel.set_policy("mode")
                kernel.act_quant(x, 32)
        finally:
            kernel.set_policy("bfloat16")


@unittest.skipIf(torch is None, "needs torch and numpy")
class Gemms(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7)

    def operands(self, m, n, k):
        a = bf16_inputs(self.generator, (m, k))
        a_codes, a_scales = kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        b_codes = torch.randint(0, 256, (n, k), generator=self.generator, dtype=torch.uint8)
        b_codes[(b_codes & 0x7F) == 0x7F] = 0x30
        b_scales = torch.randint(118, 132, (-(-n // 32), k // 32), generator=self.generator, dtype=torch.uint8)
        return a_codes, a_scales, b_codes.view(torch.float8_e4m3fn), b_scales.view(torch.float8_e8m0fnu)

    def test_fp8_gemm_matches_transcription(self):
        a, sa, b, sb = self.operands(3, 40, 128)
        default = torch.get_default_dtype()
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            try:
                got = kernel.fp8_gemm(a, sa, b, sb, torch.float8_e8m0fnu, 32)
            finally:
                torch.set_default_dtype(default)
            acc = np.float64 if dtype == torch.float64 else np.float32
            expected = np_blocked_gemm(a.double().numpy(), kernel.scale_values(sa).numpy(), b.double().numpy(),
                                       np.repeat(kernel.scale_values(sb).numpy(), 32, axis=0)[:40], 32, acc)
            self.assertEqual(got.dtype, dtype)
            if dtype == torch.bfloat16:
                expected = bf16_round(expected.astype(np.float32))
            np.testing.assert_array_equal(got.to(torch.float64).numpy(), expected.astype(np.float64))

    def test_fp8_gemm_cancellation_keeps_block_order(self):
        # Two K blocks whose partials cancel exactly, then a small one: the
        # per-block F32 accumulation keeps the small term.
        k = 96
        a = torch.zeros(1, k, dtype=torch.bfloat16)
        a[0, 0], a[0, 32], a[0, 64] = 448.0, -448.0, 2 ** -9 * 448
        codes, scales = kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        b = torch.zeros(32, k)
        b[:, 0], b[:, 32], b[:, 64] = 448.0, 448.0, 1.0
        b_codes = b.to(torch.float8_e4m3fn)
        b_scales = torch.full((1, 3), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            got = kernel.fp8_gemm(codes, scales, b_codes, b_scales, torch.float8_e8m0fnu, 32)
        finally:
            torch.set_default_dtype(default)
        self.assertEqual(got[0, 0].item(), 2 ** -9 * 448)

    def test_fp4_gemm_equals_fp8_gemm_of_the_vendor_conversion(self):
        """convert.py's cast_e2m1fn_to_e4m3fn folds each row's E8M0 scales
        into E4M3 codes over 32 x 32 blocks; the FP8 GEMM of that conversion
        must equal the FP4 GEMM of the packed weight, which checks the nibble
        order and the per-32-K block structure of both."""
        n, k = 64, 128
        a = bf16_inputs(self.generator, (3, k))
        codes, scales = kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        packed = torch.randint(0, 256, (n, k // 2), generator=self.generator, dtype=torch.uint8)
        b_scales = torch.randint(120, 128, (n, k // 32), generator=self.generator, dtype=torch.uint8)
        fp8, fp8_scales = cast_e2m1fn_to_e4m3fn(packed.view(torch.int8), b_scales.view(torch.float8_e8m0fnu))
        default = torch.get_default_dtype()
        for dtype in (torch.float32, torch.bfloat16):
            torch.set_default_dtype(dtype)
            try:
                four = kernel.fp4_gemm(codes, scales, packed.view(torch.float4_e2m1fn_x2),
                                       b_scales.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
                eight = kernel.fp8_gemm(codes, scales, fp8, fp8_scales, torch.float8_e8m0fnu, 32)
            finally:
                torch.set_default_dtype(default)
            self.assertTrue(torch.equal(four, eight))
        # A reversed nibble order would not survive the comparison.
        swapped = ((packed >> 4) | (packed << 4)).view(torch.float4_e2m1fn_x2)
        torch.set_default_dtype(torch.float32)
        try:
            other = kernel.fp4_gemm(codes, scales, swapped, b_scales.view(torch.float8_e8m0fnu),
                                    torch.float8_e8m0fnu, 32)
            eight = kernel.fp8_gemm(codes, scales, fp8, fp8_scales, torch.float8_e8m0fnu, 32)
        finally:
            torch.set_default_dtype(default)
        self.assertFalse(torch.equal(other, eight))


def cast_e2m1fn_to_e4m3fn(x, scale):
    """convert.py:13-53, the vendor's FP4 to FP8 expert conversion, copied
    from the pinned file when the checkpoint is present (else this copy)."""
    vendor = vendor_convert()
    if vendor is not None:
        return vendor.cast_e2m1fn_to_e4m3fn(x, scale)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    out_dim, in_dim = x.size()
    in_dim *= 2
    x = x.view(torch.uint8)
    x = torch.stack([table[(x & 0x0F).long()], table[((x >> 4) & 0x0F).long()]], dim=-1).flatten(2)
    b_out, b_in = out_dim // 32, in_dim // 32
    x = x.view(b_out, 32, b_in, 32).transpose(1, 2)
    scale = scale.float().view(b_out, 32, b_in, -1).transpose(1, 2).flatten(2)
    top = scale.amax(dim=-1, keepdim=True) / 64
    offset = (scale / top).unflatten(-1, (32, -1)).repeat_interleave(32, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), top.squeeze(-1).to(torch.float8_e8m0fnu)


def vendor_convert():
    import os
    import types
    from pathlib import Path
    root = os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT")
    path = Path(root or "/nonexistent") / "inference" / "convert.py"
    if not path.exists():
        return None
    import hashlib
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76":
        raise AssertionError("convert.py is not the pinned file")
    module = types.ModuleType("deepseek41_convert")
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except ModuleNotFoundError:
        return None
    return module


@unittest.skipIf(torch is None, "needs torch and numpy")
class Attention(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(11)

    def case(self, m, n, h, d, topk):
        q = bf16_inputs(self.generator, (1, m, h, d), 0.5)
        kv = bf16_inputs(self.generator, (1, n, d), 0.5)
        sink = torch.randn(h, generator=self.generator)
        idxs = torch.randint(0, n, (1, m, topk), generator=self.generator, dtype=torch.int32)
        idxs[0, 0, :] = -1          # a row with no key: all-zero output
        idxs[0, 1, 64:] = -1        # a row whose second tile is empty
        if m > 2:
            idxs[0, 2, :5] = -1
        return q, kv, sink, idxs

    def test_bf16_tiles_match_transcription(self):
        q, kv, sink, idxs = self.case(4, 150, 2, 16, 130)
        scale = 16 ** -0.5
        got = kernel.sparse_attn(q, kv, sink, idxs, scale)
        expected = np_sparse_attn(q[0].float().numpy(), kv[0].float().numpy(), sink.numpy(), idxs[0].numpy(), scale)
        self.assertEqual(got.dtype, torch.bfloat16)
        np.testing.assert_array_equal(got[0].float().numpy(), expected)
        self.assertEqual(got[0, 0].abs().max().item(), 0.0)

    def test_softmax_tile_boundary_large_scores(self):
        # A large score in the second tile rescales the first tile's sum.
        q = torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16)
        q[..., 0] = 8.0
        kv = torch.zeros(1, 128, 16, dtype=torch.bfloat16)
        kv[0, :, 0] = torch.linspace(-4, 4, 128).to(torch.bfloat16)
        kv[0, 100, 0] = 16.0
        idxs = torch.arange(128, dtype=torch.int32).view(1, 1, 128)
        sink = torch.tensor([0.5])
        got = kernel.sparse_attn(q, kv, sink, idxs, 0.25)
        expected = np_sparse_attn(q[0].float().numpy(), kv[0].float().numpy(), sink.numpy(), idxs[0].numpy(), 0.25)
        np.testing.assert_array_equal(got[0].float().numpy(), expected)

    def test_f32_and_f64_follow_their_definition(self):
        q, kv, sink, idxs = self.case(3, 70, 2, 16, 70)
        scale = 16 ** -0.5
        for dtype in (torch.float32, torch.float64):
            got = kernel.sparse_attn(q.to(dtype), kv.to(dtype), sink, idxs, scale)
            self.assertEqual(got.dtype, dtype)
            for row in range(3):
                ids = idxs[0, row]
                live = ids[ids != -1].long()
                if not live.numel():
                    self.assertEqual(got[0, row].abs().max().item(), 0.0)
                    continue
                keys = kv[0, live].double()
                scores = torch.einsum("hd,td->ht", q[0, row].double(), keys) * scale
                weights = torch.softmax(torch.cat([scores, sink.double()[:, None]], dim=-1), dim=-1)[:, :-1]
                expected = weights @ keys
                tolerance = 1e-5 if dtype == torch.float32 else 1e-13
                torch.testing.assert_close(got[0, row].double(), expected, rtol=tolerance, atol=tolerance)


@unittest.skipIf(torch is None, "needs torch and numpy")
class Sinkhorn(unittest.TestCase):
    def test_matches_transcription(self):
        generator = torch.Generator().manual_seed(3)
        mixes = torch.randn(1, 5, 24, generator=generator) * 4
        scale = torch.tensor([0.7, 1.3, 2.1])
        base = torch.randn(24, generator=generator)
        pre, post, comb = kernel.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
        for row in range(5):
            p, q, c = np_sinkhorn(mixes[0, row].numpy(), scale.numpy(), base.numpy(), 4, 20, 1e-6)
            np.testing.assert_array_equal(pre[0, row].numpy(), p)
            np.testing.assert_array_equal(post[0, row].numpy(), q)
            np.testing.assert_array_equal(comb[0, row].numpy(), c)
        wide = kernel.hc_split_sinkhorn(mixes.double(), scale, base, 4, 20, 1e-6)
        self.assertEqual(wide[2].dtype, torch.float64)
        torch.testing.assert_close(wide[2].float(), comb, rtol=1e-5, atol=1e-6)

    def test_ledger_covers_every_kernel(self):
        self.assertEqual(set(kernel.LEDGER), {"act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn",
                                              "hc_split_sinkhorn"})
        for entry in kernel.LEDGER.values():
            self.assertLessEqual({"vendor", "reduction", "casts", "modes", "claim"}, set(entry))
            self.assertLessEqual(set(entry), {"vendor", "reduction", "casts", "modes", "claim", "vendor_control"})
            self.assertEqual(set(entry["modes"]), {"bf16", "f32", "f64"})


@unittest.skipIf(torch is None, "needs torch and numpy")
class VendorControlClasses(unittest.TestCase):
    def test_discrete_and_unjudged_differences_are_semantic(self):
        from models.deepseek41 import reference_vendor_control as control
        scale = control.compare("scales", torch.tensor([127], dtype=torch.uint8).view(torch.float8_e8m0fnu),
                                torch.tensor([128], dtype=torch.uint8).view(torch.float8_e8m0fnu))
        self.assertEqual(control.classify([scale], discrete=True), "semantic")
        # Without an oracle no closeness is known, so a difference cannot be lowering.
        self.assertEqual(control.classify([scale]), "semantic")
        vendor = torch.tensor([1.0, 2.0, 3.0])
        nudged = torch.tensor([1.0, 2.0, 3.0]).view(torch.int32).add(torch.tensor([0, 1, 0], dtype=torch.int32)) \
            .view(torch.float32)
        oracle = nudged.double()
        close = control.compare("out", vendor, nudged, oracle)
        self.assertEqual(control.classify([close]), "lowering")
        self.assertEqual(control.classify([close], discrete=True), "semantic")
        far = control.compare("out", vendor, torch.tensor([1.0, 2.5, 3.0]), torch.tensor([1.0, 2.5, 3.0]).double())
        self.assertEqual(control.classify([far]), "semantic")
        same = control.compare("out", vendor, vendor.clone())
        self.assertEqual(control.classify([same], discrete=True), "bitwise")


if __name__ == "__main__":
    unittest.main()
