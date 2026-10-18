#pragma once

#include <math.h>
#include <stdint.h>
#include "common/convert.cuh"
#include "common/mxfp4.cuh"
#include "common/ptx.cuh"

namespace chop::gated_pooling {

// Index writes normalize the projected key before tail RoPE. Compressed
// writes receive the already normalized latent. Inputs, normalization and
// RoPE stay in F32 until the single BF16 quantizer-input rounding. Native
// and BF16 cache forms quantize identically.
template<int Width, int Rotary, int Ratio, int Page,
         bool Normalize, bool E4m3Scale, bool Packed>
__device__ void key_write(const float* key, const uint16_t* norm_weight,
    const float* cos_sin, const int32_t* positions, const int32_t* row_batch,
    const uint32_t* live_rows, const int32_t* page_tables,
    uint8_t* codes, uint8_t* scales, uint32_t items, uint32_t pages,
    uint32_t max_pages, uint64_t code_stride, uint64_t scale_stride,
    uint32_t table_rows, float eps) {
  static_assert(Width == 128 || Width == 512);
  static_assert(Rotary == 64 && Rotary <= Width);
  static_assert(Ratio == 1 || Ratio == 2);
  static_assert(Page > 0);
  constexpr int Block = E4m3Scale ? 16 : 32;
  ptx::griddep_launch_dependents();
  __shared__ float values[Width], sums[4];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t position = positions[row], item = row_batch[row];
  if (position < 0 || item < 0 || uint32_t(item) >= items) return;
  const uint64_t rotary_position = uint64_t(position) * Ratio;
  const uint32_t logical_page = uint32_t(position) / Page;
  if (rotary_position >= table_rows || logical_page >= max_pages) return;
  const int32_t page = page_tables[uint64_t(item) * max_pages + logical_page];
  if (page < 0 || uint32_t(page) >= pages) return;
  const uint32_t in_page = uint32_t(position) % Page;
  uint8_t* code_row = codes + uint64_t(page) * code_stride
      + uint64_t(in_page) * (Packed ? Width / 2 : Width * 2);
  uint8_t* scale_row = nullptr;
  if constexpr (Packed)
    scale_row = scales + uint64_t(page) * scale_stride + uint64_t(in_page) * (Width / Block);

  float square = 0.f;
  for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
    const float value = key[uint64_t(row) * Width + d];
    if (!isfinite(value)) asm volatile("trap;");
    values[d] = value;
    if constexpr (Normalize) square = __fadd_rn(square, __fmul_rn(value, value));
  }
  if constexpr (Normalize) {
    for (int offset = 16; offset; offset >>= 1)
      square = __fadd_rn(square, __shfl_down_sync(0xffffffffu, square, offset));
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = square;
    __syncthreads();
    if (threadIdx.x == 0)
      sums[0] = __fadd_rn(__fadd_rn(sums[0], sums[1]), __fadd_rn(sums[2], sums[3]));
    __syncthreads();
    const float inv = __frsqrt_rn(__fadd_rn(__fmul_rn(sums[0], 1.f / Width), eps));
    for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
      const float weight = convert::bf16_to_f32(norm_weight[d]);
      const float normalized = __fmul_rn(__fmul_rn(values[d], inv), weight);
      values[d] = normalized;
    }
  }
  __syncthreads();

  // A lane owns two adjacent channels, so each scale group occupies a
  // whole aligned subgroup of 8 or 16 lanes. Every participating warp is
  // full, including the width-128 entry's two active warps.
  for (uint32_t pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
    const uint32_t d = pair * 2;
    float x = values[d], y = values[d + 1];
    if (d >= Width - Rotary) {
      const float* cs = cos_sin + rotary_position * Rotary + d - (Width - Rotary);
      const float c = cs[0], s = cs[1];
      const float real = __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
      const float imag = __fadd_rn(__fmul_rn(x, s), __fmul_rn(y, c));
      x = real;
      y = imag;
    }
    x = convert::bf16_to_f32(convert::bf16_from_f32(x));
    y = convert::bf16_to_f32(convert::bf16_from_f32(y));
    if (!isfinite(x) || !isfinite(y)) asm volatile("trap;");
    float amax = fmaxf(fabsf(x), fabsf(y));
    for (int offset = Block / 4; offset; offset >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, Block / 2));
    float scale;
    uint8_t scale_code;
    if constexpr (E4m3Scale) {
      if (amax > 6.f * 448.f) asm volatile("trap;");
      amax = fmaxf(amax, 0x1.8p-7f);  // 6 * 2^-9
      scale_code = uint8_t(convert::pack_f32_e4m3({__fdiv_rn(amax, 6.f), 0.f}));
      scale = convert::unpack_f32_e4m3(uint16_t(scale_code)).lo;
    } else {
      amax = fmaxf(amax, 0x1.8p-124f);  // 6 * 2^-126
      const uint32_t bits = __float_as_uint(__fmul_rn(amax, 1.f / 6.f));
      const uint32_t exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
      if (exponent >= 255) asm volatile("trap;");
      scale_code = uint8_t(exponent);
      scale = __uint_as_float(exponent << 23);
    }
    const float qx = fminf(6.f, fmaxf(-6.f, __fdiv_rn(x, scale)));
    const float qy = fminf(6.f, fmaxf(-6.f, __fdiv_rn(y, scale)));
    const uint8_t code = uint8_t(convert::pack_f32_e2m1({qx, qy}));
    if constexpr (!E4m3Scale) convert::check_mxfp4(code, scale_code);
    if constexpr (Packed) {
      code_row[pair] = code;
      if (pair % (Block / 2) == 0) scale_row[d / Block] = scale_code;
    } else {
      const convert::Float2 decoded = convert::unpack_f32_e2m1(uint16_t(code));
      uint16_t* target = reinterpret_cast<uint16_t*>(code_row);
      target[d] = convert::bf16_from_f32(__fmul_rn(decoded.lo, scale));
      target[d + 1] = convert::bf16_from_f32(__fmul_rn(decoded.hi, scale));
    }
  }
}

}  // namespace chop::gated_pooling
