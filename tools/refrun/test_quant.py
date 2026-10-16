"""Exact decode and dequantization checks for tools/refrun/quant.py.

Needs torch; skipped under a Python without it. Run on CPU only.
"""

import math
import struct
import unittest

try:
    import torch
    import torch.nn.functional as F
    from tools.refrun import modes, quant
except ModuleNotFoundError:
    torch = None


def setUpModule():
    if torch is not None:
        modes.prepare_environment(cuda=False)


def minifloat(code, exponent_bits, mantissa_bits, bias):
    """IEEE-style binary interchange decoding, without infinities or NaN."""
    sign = -1.0 if code >> (exponent_bits + mantissa_bits) & 1 else 1.0
    exponent = code >> mantissa_bits & ((1 << exponent_bits) - 1)
    mantissa = code & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        return sign * mantissa / 2 ** mantissa_bits * 2.0 ** (1 - bias)
    return sign * (1 + mantissa / 2 ** mantissa_bits) * 2.0 ** (exponent - bias)


def bits(value):
    return struct.pack("<d", value)


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or bits(a) == bits(b)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class DecodeTests(unittest.TestCase):
    def test_every_e4m3_code(self):
        native = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().tolist()
        for code in range(256):
            expected = math.nan if code & 0x7F == 0x7F else minifloat(code, 4, 3, 7)
            self.assertTrue(same(quant.E4M3[code], expected), hex(code))
            self.assertTrue(same(native[code], expected), hex(code))
        self.assertEqual(max(v for v in quant.E4M3 if not math.isnan(v)), 448.0)
        self.assertEqual(quant.E4M3[1], 2.0 ** -9)
        self.assertTrue(bits(quant.E4M3[0x80]) == bits(-0.0))

    def test_every_e2m1_code(self):
        magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for code in range(16):
            expected = minifloat(code, 2, 1, 1)
            self.assertTrue(same(quant.E2M1[code], expected), code)
            self.assertEqual(abs(quant.E2M1[code]), magnitudes[code & 7])
            self.assertEqual(math.copysign(1, quant.E2M1[code]), -1 if code & 8 else 1)

    def test_every_e8m0_code(self):
        native = torch.arange(256, dtype=torch.uint8).view(torch.float8_e8m0fnu).double().tolist()
        for code in range(256):
            expected = math.nan if code == 0xFF else math.ldexp(1.0, code - 127)
            self.assertTrue(same(quant.E8M0[code], expected), code)
            self.assertTrue(same(native[code], expected), code)

    def test_native_conversion_matches_in_every_mode_dtype(self):
        quant.check_codecs("cpu")

    def test_torch_op_overrides_are_removed(self):
        registry = modes.native_overrides()
        if registry is not None:
            self.assertEqual(registry._aten_override_libs, {})


def naive(codes, scale_of, rows, cols, element):
    """f64 weight by the definitions, one element at a time."""
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if element == "e4m3":
                value = quant.E4M3[int(codes[r, c])]
            else:
                byte = int(codes[r, c // 2])
                value = quant.E2M1[byte & 0xF if c % 2 == 0 else byte >> 4]
            row.append(value * scale_of(r, c))
        out.append(row)
    return torch.tensor(out, dtype=torch.float64)


def fp8_codes(generator, shape):
    codes = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x40
    return codes


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class DequantizeTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7)

    def scales(self, shape, dtype=None):
        return (torch.rand(shape, generator=self.generator) + 0.25).to(dtype or torch.bfloat16)

    def check(self, fmt, codes, scale, shape, expected):
        for dtype in (torch.float32, torch.float64):
            tensor = quant.QuantTensor(fmt, codes, scale, shape)
            tensor.configure(dtype, "exact")
            got = tensor.dequantize(dtype, "exact")
            self.assertEqual(got.dtype, dtype)
            self.assertTrue(torch.equal(got.double(), expected), dtype)

    def test_fp8_regular_blocks(self):
        fmt = quant.fp8_block(4, 8)
        codes, scale = fp8_codes(self.generator, (12, 24)), self.scales((3, 3))
        expected = naive(codes, lambda r, c: float(scale[r // 4, c // 8]), 12, 24, "e4m3")
        self.check(fmt, codes, scale, (12, 24), expected)

    def test_fp8_partial_tail_blocks(self):
        fmt = quant.fp8_block(8, 8)
        codes, scale = fp8_codes(self.generator, (13, 20)), self.scales((2, 3))
        expected = naive(codes, lambda r, c: float(scale[r // 8, c // 8]), 13, 20, "e4m3")
        self.check(fmt, codes, scale, (13, 20), expected)

    def test_fp8_chunked_grid_restarts_per_chunk(self):
        # Two chunks of 12 rows on 8-row blocks: each chunk has a full and a
        # 4-row tail block, so 4 scale rows rather than the contiguous 3.
        fmt = quant.fp8_block(8, 8, chunk_rows=12)
        self.assertEqual(fmt.scale_shape(24, 20), [4, 3])
        self.assertEqual(quant.fp8_block(8, 8).scale_shape(24, 20), [3, 3])
        codes, scale = fp8_codes(self.generator, (24, 20)), self.scales((4, 3))
        expected = naive(codes, lambda r, c: float(scale[r // 12 * 2 + r % 12 // 8, c // 8]),
                         24, 20, "e4m3")
        self.check(fmt, codes, scale, (24, 20), expected)

    def test_mimo_qkv_grid_has_108_scale_rows(self):
        # 13,568 QKV rows in four TP chunks of 3,392 on 128-row blocks.
        self.assertEqual(quant.fp8_block(128, 128, 3392).scale_shape(13568, 4096), [108, 32])
        self.assertEqual(quant.fp8_block(128, 128).scale_shape(13568, 4096), [106, 32])
        rows = quant.fp8_block(128, 128, 3392).scale_rows(13568)
        self.assertEqual((rows[3391], rows[3392], rows[13567]), (26, 27, 107))

    def test_permuted_rows_keep_their_scales(self):
        fmt = quant.fp8_block(8, 8, chunk_rows=12)
        codes, scale = fp8_codes(self.generator, (24, 16)), self.scales((4, 2))
        order = torch.randperm(24, generator=self.generator)
        whole = quant.QuantTensor(fmt, codes, scale, (24, 16)).dequantize(torch.float64, "exact")
        tensor = quant.QuantTensor(fmt, codes, scale, (24, 16))
        tensor.permute_rows(order)
        self.assertTrue(torch.equal(tensor.dequantize(torch.float64, "exact"), whole[order]))

    def test_mxfp4_even_k_is_the_low_nibble(self):
        codes = torch.randint(0, 256, (6, 32), generator=self.generator, dtype=torch.uint8)
        scale = torch.randint(110, 140, (6, 2), generator=self.generator, dtype=torch.uint8)
        expected = naive(codes, lambda r, c: quant.E8M0[int(scale[r, c // 32])], 6, 64, "e2m1")
        self.check(quant.MXFP4, codes, scale, (6, 64), expected)
        one = quant.QuantTensor(quant.MXFP4, torch.full((1, 16), 0x62, dtype=torch.uint8),
                                torch.tensor([[127]], dtype=torch.uint8), (1, 32))
        one = one.dequantize(torch.float64, "exact")
        self.assertEqual(one[0, :2].tolist(), [1.0, 4.0])

    def test_nvfp4_global_scale(self):
        codes = torch.randint(0, 256, (4, 16), generator=self.generator, dtype=torch.uint8)
        scale = torch.randint(0x20, 0x50, (4, 2), generator=self.generator, dtype=torch.uint8)
        for global_scale in (1.0, 0.0625):
            expected = naive(codes, lambda r, c: quant.E4M3[int(scale[r, c // 16])] * global_scale,
                             4, 32, "e2m1")
            self.check(quant.nvfp4(global_scale), codes, scale.view(torch.float8_e4m3fn), (4, 32), expected)

    def test_mxfp8_by_hand(self):
        """E4M3 codes with one UE8M0 byte per 32 K: 0x38 is 1.0, 0x3C 1.5,
        0xB8 -1.0, 0x01 2^-9; bytes 127, 130, 120 and 128 are 1, 8, 2^-7
        and 2. Every product is exact in F32 and BF16."""
        fmt = quant.Format("e4m3", "e8m0", 1, 32)
        codes = torch.full((2, 64), 0x38, dtype=torch.uint8)
        codes[0, 1], codes[0, 33], codes[1, 0], codes[1, 63] = 0x3C, 0xB8, 0x01, 0x3C
        scale = torch.tensor([[127, 130], [120, 128]], dtype=torch.uint8)
        expected = torch.tensor([[1.0] * 32 + [8.0] * 32, [2.0 ** -7] * 32 + [2.0] * 32], dtype=torch.float64)
        expected[0, 1], expected[0, 33], expected[1, 0], expected[1, 63] = 1.5, -8.0, 2.0 ** -16, 3.0
        self.check(fmt, codes.view(torch.float8_e4m3fn), scale, (2, 64), expected)
        tensor = quant.QuantTensor(fmt, codes, scale, (2, 64))
        for dtype in (torch.bfloat16, torch.float32):
            self.assertTrue(torch.equal(tensor.dequantize(dtype, "hf-bf16").double(), expected), dtype)
        nan = quant.QuantTensor(fmt, codes, torch.tensor([[127, 255], [127, 127]], dtype=torch.uint8), (2, 64))
        with self.assertRaisesRegex(ValueError, "not finite"):
            nan.configure(torch.float64, "exact")

    def test_nvfp4_by_hand(self):
        """Byte 0x71 holds 0.5 (low nibble, even K) then 6.0; 0xA3 holds 1.5
        then -1.0. E4M3 0x40 is 2.0. With the F32 global g = 1/3: exact is
        element x 2 x g in f64; rounded rounds that once to F32; hf-bf16
        rounds 2g to F32 (exact), then element x 2g to F32, then to BF16."""
        third = struct.unpack("<f", struct.pack("<f", 1 / 3))[0]
        codes = torch.zeros((1, 8), dtype=torch.uint8)
        codes[0, :2] = torch.tensor([0x71, 0xA3], dtype=torch.uint8)
        scale = torch.tensor([[0x40]], dtype=torch.uint8).view(torch.float8_e4m3fn)
        tensor = quant.QuantTensor(quant.nvfp4(third), codes, scale, (1, 16))
        elements = [0.5, 6.0, 1.5, -1.0]
        exact = tensor.dequantize(torch.float64, "exact")[0, :4].tolist()
        self.assertEqual(exact, [e * 2.0 * third for e in elements])
        self.assertEqual(tensor.dequantize(torch.float64, "exact")[0, 4:].tolist(), [0.0] * 12)
        rounded = tensor.dequantize(torch.float32, "rounded")[0, :4].tolist()
        f32 = [struct.unpack("<f", struct.pack("<f", e * 2.0 * third))[0] for e in elements]
        self.assertEqual(rounded, f32)
        self.assertNotEqual(f32[1], exact[1])
        hf = tensor.dequantize(torch.bfloat16, "hf-bf16")[0, :4]
        self.assertTrue(torch.equal(hf, torch.tensor(f32, dtype=torch.float32).to(torch.bfloat16)))

    def test_rounded_nvfp4_is_one_rounding_not_staged_f32(self):
        """E2M1 1.5 (code 0x3) x E4M3 72 (0x69) x global 6.612142169615254e-05
        (an F32 value): rounded is the exact product rounded once to F32,
        0.007141113746911287; forming the scale times the global in F32 first
        and then the product in F32 (ModelOpt's staged F32 order) gives
        0.00714111328125. hf-bf16 takes the staged F32 product to BF16."""
        f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]  # noqa: E731
        scale_code, global_scale = 0x69, f32(6.612142169615254e-05)
        self.assertEqual((quant.E4M3[scale_code], global_scale), (72.0, 6.612142169615254e-05))
        tensor = quant.QuantTensor(quant.nvfp4(global_scale), torch.full((1, 8), 0x33, dtype=torch.uint8),
                                   torch.tensor([[scale_code]], dtype=torch.uint8), (1, 16))
        once = f32(1.5 * 72.0 * global_scale)
        staged = f32(1.5 * f32(72.0 * global_scale))
        self.assertEqual((once, staged), (0.007141113746911287, 0.00714111328125))
        rounded = tensor.dequantize(torch.float32, "rounded")
        self.assertEqual(rounded.unique().tolist(), [once])
        self.assertEqual(tensor.dequantize(torch.float64, "exact").unique().tolist(), [1.5 * 72.0 * global_scale])
        hf = tensor.dequantize(torch.bfloat16, "hf-bf16")
        self.assertTrue(torch.equal(hf, torch.full((1, 16), staged, dtype=torch.float32).to(torch.bfloat16)))

    def test_inexact_f32_products_are_refused(self):
        third = struct.unpack("<f", struct.pack("<f", 1 / 3))[0]
        codes = torch.randint(0, 256, (4, 16), generator=self.generator, dtype=torch.uint8)
        scale = torch.randint(0x30, 0x40, (4, 2), dtype=torch.uint8)
        tensor = quant.QuantTensor(quant.nvfp4(third), codes, scale, (4, 32))
        with self.assertRaisesRegex(ValueError, "not exact"):
            tensor.configure(torch.float32, "exact")
        tensor.configure(torch.float64, "exact")
        f32_scale = torch.full((1, 1), third, dtype=torch.float32)
        fp8 = quant.QuantTensor(quant.fp8_block(4, 16), fp8_codes(self.generator, (4, 16)), f32_scale, (4, 16))
        with self.assertRaisesRegex(ValueError, "not exact"):
            fp8.configure(torch.float32, "exact")

    def test_rounded_is_the_exact_product_rounded_once(self):
        third = struct.unpack("<f", struct.pack("<f", 1 / 3))[0]
        codes = fp8_codes(self.generator, (8, 32))
        scale = torch.tensor([[third, 0.1]], dtype=torch.float32)
        fp8 = quant.QuantTensor(quant.fp8_block(8, 16), codes, scale, (8, 32))
        nvfp4 = quant.QuantTensor(quant.nvfp4(third), torch.randint(
            0, 256, (4, 16), generator=self.generator, dtype=torch.uint8),
            torch.randint(0x30, 0x40, (4, 2), generator=self.generator, dtype=torch.uint8), (4, 32))
        for tensor in (fp8, nvfp4):
            exact = tensor.dequantize(torch.float64, "exact")
            self.assertFalse(torch.equal(exact.float().double(), exact))
            tensor.configure(torch.float32, "rounded")
            self.assertTrue(torch.equal(tensor.dequantize(torch.float32, "rounded"), exact.float()))
            self.assertTrue(torch.equal(tensor.dequantize(torch.float64, "rounded"), exact))
            with self.assertRaisesRegex(ValueError, "F32 or f64"):
                tensor.dequantize(torch.bfloat16, "rounded")
        self.assertEqual(modes.policy("f32", "rounded")["dequant"], "rounded")
        with self.assertRaisesRegex(ValueError, "cannot dequantize"):
            modes.policy("bf16", "rounded")

    def test_hf_bf16_rounds_the_f32_product(self):
        codes, scale = fp8_codes(self.generator, (8, 16)), self.scales((1, 1), torch.float32)
        tensor = quant.QuantTensor(quant.fp8_block(8, 16), codes, scale, (8, 16))
        expected = (codes.view(torch.float8_e4m3fn).float() * scale.float()).to(torch.bfloat16)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            got = tensor.dequantize(dtype, "hf-bf16")
            self.assertTrue(torch.equal(got, expected.to(dtype)), dtype)

    def test_bf16_weights_widen_exactly(self):
        weight = torch.randn((5, 7), generator=self.generator).to(torch.bfloat16)
        self.check(quant.BF16, weight, None, (5, 7), weight.double())

    def test_nonfinite_and_misshapen_weights_are_refused(self):
        nan = torch.full((4, 16), 0x7F, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "not finite"):
            quant.QuantTensor(quant.fp8_block(4, 16), nan, self.scales((1, 1)), (4, 16)).configure(
                torch.float32, "exact")
        with self.assertRaisesRegex(ValueError, "grid"):
            quant.QuantTensor(quant.fp8_block(4, 8), nan, self.scales((1, 1)), (4, 16))

    def test_quant_linear_is_f_linear_of_the_dequantized_weight(self):
        fmt = quant.fp8_block(8, 8, chunk_rows=12)
        codes, scale = fp8_codes(self.generator, (24, 16)), self.scales((4, 2))
        bias = torch.randn(24, generator=self.generator).to(torch.bfloat16)
        x = torch.randn((3, 16), generator=self.generator)
        for dtype in (torch.float32, torch.float64):
            linear = quant.QuantLinear(fmt, codes, scale, (24, 16), bias=bias)
            linear.configure(dtype, "exact")
            weight = linear.dequantize(dtype, "exact")
            self.assertTrue(torch.equal(linear(x.to(dtype)), F.linear(x.to(dtype), weight, bias.to(dtype))))

    def test_quant_stack_indexes_experts_in_separate_slots(self):
        experts = [quant.QuantTensor(quant.BF16, torch.full((2, 2), float(e), dtype=torch.bfloat16), None,
                                     (2, 2), slot="down") for e in range(3)]
        stack = quant.QuantStack(experts)
        quant.configure(stack, torch.float32, "exact")
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack[torch.tensor(2)].tolist(), [[2.0, 2.0], [2.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
