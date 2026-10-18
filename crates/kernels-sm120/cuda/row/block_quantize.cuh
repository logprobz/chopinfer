#pragma once

#include <math.h>
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/mxfp4.cuh"
#include "../common/ptx.cuh"

namespace chop::row::block_quantize {

constexpr uint32_t NonfiniteInput = 1;
constexpr uint32_t ScaleOverflow = 2;
constexpr uint32_t UnrepresentableOutput = 4;

// The caller clears status once before ordered launches. The atomic OR
// preserves every prior bit. It also makes independent flag updates safe.
struct StatusOr {
  __device__ static uint32_t update(uint32_t* status, uint64_t row, uint32_t flags) {
    return atomicOr(status + row, flags) | flags;
  }
};

// Inputs to every discrete decision are BF16, even when storage is F32.
// A non-finite original or rounded operand contributes a flag and a safe
// zero while the rest of its row is validated.
template<bool F32Input>
__device__ __forceinline__ float input_value(const void* input, uint64_t at, uint32_t& flags) {
  float value;
  if constexpr (F32Input) {
    value = reinterpret_cast<const float*>(input)[at];
    if (!isfinite(value)) { flags |= NonfiniteInput; return 0.f; }
    value = convert::bf16_to_f32(convert::bf16_from_f32(value));
  } else {
    value = convert::bf16_to_f32(reinterpret_cast<const uint16_t*>(input)[at]);
  }
  if (!isfinite(value)) { flags |= NonfiniteInput; return 0.f; }
  return value;
}

__device__ __forceinline__ uint8_t ceil_scale(float maximum, float reciprocal) {
  const uint32_t bits = __float_as_uint(__fmul_rn(maximum, reciprocal));
  return uint8_t((bits >> 23) + ((bits & 0x7fffff) != 0));
}

struct GroupValue {
  uint32_t flags = 0;
  uint32_t pair = 0;
  uint16_t codes = 0;
  uint8_t scale_code = 0;
};

template<int Format, bool F32Input, bool Fake, uint32_t FloorBits>
__device__ __forceinline__ GroupValue quantize_group(const void* input, uint64_t at) {
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  const uint32_t lane = threadIdx.x;
  GroupValue result;
  float x = 0.f, y = 0.f;
  if (lane < Group / 2) {
    x = input_value<F32Input>(input, at, result.flags);
    y = input_value<F32Input>(input, at + 1, result.flags);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  float scale;
  if constexpr (Format == 1) {
    maximum = fmaxf(maximum, 6.f * 0x1p-9f);
    const float ratio = __fdiv_rn(maximum, 6.f);
    result.scale_code = uint8_t(convert::pack_f32_e4m3({ratio, ratio}));
    scale = convert::unpack_f32_e4m3(uint16_t(result.scale_code)).lo;
  } else {
    maximum = fmaxf(maximum, Format == 0 ? __uint_as_float(FloorBits) : 6.f * 0x1p-126f);
    result.scale_code = ceil_scale(maximum, Format == 0 ? (1.f / 448.f) : (1.f / 6.f));
    scale = __uint_as_float(uint32_t(result.scale_code) << 23);
  }
  if (lane >= Group / 2) return result;
  convert::Float2 normalized{__fdiv_rn(x, scale), __fdiv_rn(y, scale)};
  if constexpr (Format == 1) {
    // The vendor's scalar FP8 cast is SATFINITE: scale <=448, then
    // explicit payload clipping before the E2M1 conversion.
    normalized.lo = fminf(6.f,fmaxf(-6.f,normalized.lo));
    normalized.hi = fminf(6.f,fmaxf(-6.f,normalized.hi));
  }
  if constexpr (Format == 0) result.codes = convert::pack_f32_e4m3(normalized);
  else result.codes = convert::pack_f32_e2m1(normalized);
  if constexpr (Format == 2) {
      const uint8_t codes = uint8_t(result.codes);
      const uint8_t scale_code = result.scale_code;
      const unsigned limit = scale_code == 253 ? 6 : 4;
      if (scale_code > 252 && (scale_code == 255 || (codes & 7) >= limit || ((codes >> 4) & 7) >= limit))
        result.flags |= UnrepresentableOutput;
  }
  {
    if constexpr (Format == 2) {
      const uint8_t codes = uint8_t(result.codes);
      const uint8_t scale_code = result.scale_code;
      // Check=false avoids the shared helper's trap. The row status above
      // retains its exact representability rule before any output is used.
      result.pair = convert::scaled_mxfp4_bf16x2<false>(codes, scale_code);
    } else {
      convert::Float2 decoded;
      if constexpr (Format == 0) decoded = convert::unpack_f32_e4m3(result.codes);
      else decoded = convert::unpack_f32_e2m1(result.codes);
      decoded.lo = __fmul_rn(decoded.lo, scale);
      decoded.hi = __fmul_rn(decoded.hi, scale);
      result.pair = convert::bf16x2_from_f32(decoded);
      const auto exact = convert::bf16x2_to_f32(result.pair);
      if (!isfinite(exact.lo) || !isfinite(exact.hi) ||
          __float_as_uint(exact.lo) != __float_as_uint(decoded.lo) ||
          __float_as_uint(exact.hi) != __float_as_uint(decoded.hi))
        result.flags |= UnrepresentableOutput;
    }
  }
  return result;
}

// One CTA validates an entire row before writing it. This prevents a bad
// group racing other CTAs which might otherwise write valid-looking bytes.
// Width 0 binds the host-validated width at launch. No data path traps.
template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits, class Status = StatusOr>
__device__ void run(const void* input, void* output, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset, uint32_t launch_width = 0) {
  static_assert(Format >= 0 && Format <= 2);
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const uint32_t width = Width ? Width : launch_width;
  const uint32_t lane = threadIdx.x;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = 0; group < width / Group; ++group) {
    const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
    flags |= quantize_group<Format,F32Input,Fake,FloorBits>(input,at).flags;
  }
  for (int step = 16; step; step >>= 1)
    flags |= __shfl_xor_sync(0xffffffff, flags, step);
  if (lane == 0) flags = Status::update(status,status_row_offset+row,flags);
  flags = __shfl_sync(0xffffffff,flags,0);
#pragma unroll 1
  for (uint32_t group = 0; group < width / Group; ++group) {
    const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
    GroupValue value;
    if (flags == 0) value = quantize_group<Format,F32Input,Fake,FloorBits>(input,at);
    else value.scale_code = Format == 1 ? 0x38 : 127; // scale 1
    if constexpr (!Fake) {
      if (lane == 0) scales[uint64_t(row) * (width / Group) + group] = value.scale_code;
    }
    if (lane < Group / 2) {
      if constexpr (Fake) {
        reinterpret_cast<uint16_t*>(output)[at] = uint16_t(value.pair);
        reinterpret_cast<uint16_t*>(output)[at+1] = uint16_t(value.pair >> 16);
      } else if constexpr (Format == 0) {
        reinterpret_cast<uint8_t*>(output)[at] = uint8_t(value.codes);
        reinterpret_cast<uint8_t*>(output)[at+1] = uint8_t(value.codes >> 8);
      } else reinterpret_cast<uint8_t*>(output)[at/2] = uint8_t(value.codes);
    }
  }
  ptx::griddep_launch_dependents();
}
} // namespace chop::row::block_quantize
