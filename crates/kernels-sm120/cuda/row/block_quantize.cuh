#pragma once

#include <math.h>
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/mxfp4.cuh"
#include "../common/ptx.cuh"

namespace chop::row::block_quantize {

// Inputs to every discrete decision are BF16, even when storage is F32.
template<bool F32Input>
__device__ __forceinline__ float input_value(const void* input, uint64_t at) {
  float value;
  if constexpr (F32Input) {
    value = reinterpret_cast<const float*>(input)[at];
    if (!isfinite(value)) asm volatile("trap;");
    value = convert::bf16_to_f32(convert::bf16_from_f32(value));
  } else {
    value = convert::bf16_to_f32(reinterpret_cast<const uint16_t*>(input)[at]);
  }
  if (!isfinite(value)) asm volatile("trap;");
  return value;
}

// The floor ensures a positive normal ratio. Keep both F32 roundings:
// replacing this multiply by a division can choose another exponent.
__device__ __forceinline__ uint8_t ceil_scale(float maximum, float reciprocal) {
  const uint32_t bits = __float_as_uint(__fmul_rn(maximum, reciprocal));
  return uint8_t((bits >> 23) + ((bits & 0x7fffff) != 0));
}

// Format 0: E4M3/32 with ceil UE8M0. Format 1: E2M1 with E4M3/16.
// Format 2: E2M1/32 with ceil UE8M0. Width 0 binds width at launch.
template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits>
__device__ void run(const void* input, void* output, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t launch_width = 0) {
  static_assert(Format >= 0 && Format <= 2);
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t width = Width ? Width : launch_width;
  const uint32_t group = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
  float x = 0.f, y = 0.f;
  if (lane < Group / 2) {
    x = input_value<F32Input>(input, at);
    y = input_value<F32Input>(input, at + 1);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  float scale;
  uint8_t scale_code;
  if constexpr (Format == 1) {
    if (maximum > 6.f * 448.f) asm volatile("trap;");
    maximum = fmaxf(maximum, 6.f * 0x1p-9f);
    const float ratio = __fdiv_rn(maximum, 6.f);
    scale_code = uint8_t(convert::pack_f32_e4m3({ratio, ratio}));
    scale = convert::unpack_f32_e4m3(uint16_t(scale_code)).lo;
  } else {
    maximum = fmaxf(maximum, Format == 0 ? __uint_as_float(FloorBits) : 6.f * 0x1p-126f);
    scale_code = ceil_scale(maximum, Format == 0 ? (1.f / 448.f) : (1.f / 6.f));
    scale = __uint_as_float(uint32_t(scale_code) << 23);
  }
  if constexpr (!Fake) {
    if (lane == 0) scales[uint64_t(row) * (width / Group) + group] = scale_code;
  }
  if (lane >= Group / 2) return;
  const convert::Float2 normalized{__fdiv_rn(x, scale), __fdiv_rn(y, scale)};
  uint16_t codes;
  if constexpr (Format == 0) codes = convert::pack_f32_e4m3(normalized);
  else codes = convert::pack_f32_e2m1(normalized);
  if constexpr (Fake) {
    uint32_t pair;
    if constexpr (Format == 2) {
      pair = convert::scaled_mxfp4_bf16x2(uint8_t(codes), scale_code);
    } else {
      convert::Float2 decoded;
      if constexpr (Format == 0) decoded = convert::unpack_f32_e4m3(codes);
      else decoded = convert::unpack_f32_e2m1(codes);
      decoded.lo = __fmul_rn(decoded.lo, scale);
      decoded.hi = __fmul_rn(decoded.hi, scale);
      pair = convert::bf16x2_from_f32(decoded);
      const auto exact = convert::bf16x2_to_f32(pair);
      if (!isfinite(exact.lo) || !isfinite(exact.hi) ||
          __float_as_uint(exact.lo) != __float_as_uint(decoded.lo) ||
          __float_as_uint(exact.hi) != __float_as_uint(decoded.hi)) asm volatile("trap;");
    }
    reinterpret_cast<uint16_t*>(output)[at] = uint16_t(pair);
    reinterpret_cast<uint16_t*>(output)[at + 1] = uint16_t(pair >> 16);
  } else if constexpr (Format == 0) {
    reinterpret_cast<uint8_t*>(output)[at] = uint8_t(codes);
    reinterpret_cast<uint8_t*>(output)[at + 1] = uint8_t(codes >> 8);
  } else {
    reinterpret_cast<uint8_t*>(output)[at / 2] = uint8_t(codes);
  }
}

} // namespace chop::row::block_quantize
