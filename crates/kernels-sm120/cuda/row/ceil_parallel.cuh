#pragma once
#include "block_quantize.cuh"
#include "silu_mul_quantize.cuh"

// Keep the legacy headers and their locked consumers unchanged. The group
// bodies below differ only in warp-local lane indexing; a CPU source guard
// and old-versus-new GPU tests check that invariant.
namespace chop::row::block_quantize_parallel {
using namespace block_quantize;

template<int Format, bool F32Input, bool Fake, uint32_t FloorBits>
__device__ __forceinline__ GroupValue quantize_group(const void* input, uint64_t at) {
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  const uint32_t lane = threadIdx.x & 31;
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

// Groups keep their original warp arithmetic. Only integer status ORs cross
// warps. The two barriers publish every group's flags before any output.
template<int Warps>
__device__ __forceinline__ uint32_t row_status(uint32_t flags, uint32_t* shared,
    uint32_t* status, uint64_t row) {
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int step = 16; step; step >>= 1)
    flags |= __shfl_xor_sync(0xffffffff, flags, step);
  if (lane == 0) shared[warp] = flags;
  __syncthreads();
  if (warp == 0) {
    flags = lane < Warps ? shared[lane] : 0;
    for (int step = 16; step; step >>= 1)
      flags |= __shfl_xor_sync(0xffffffff, flags, step);
    if (lane == 0) shared[0] = StatusOr::update(status, row, flags);
  }
  __syncthreads();
  return shared[0];
}
// EarlyRelease: see rmsnorm_quantize_cached_f32.cuh.
template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits,
         bool EarlyRelease = false>
__device__ void run(const void* input, void* output, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset, uint32_t launch_width = 0) {
  static_assert(Width > 0 && Format == 0 && !Fake);
  constexpr uint32_t Warps = Width / 32 < 32 ? Width / 32 : 32;
  __shared__ uint32_t shared_flags[Warps];
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  ptx::griddep_wait();
  if constexpr (EarlyRelease) ptx::griddep_launch_dependents();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) {
    if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
    return;
  }
  const uint32_t width = Width ? Width : launch_width;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = warp; group < width / Group; group += Warps) {
    const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
    flags |= quantize_group<Format,F32Input,Fake,FloorBits>(input,at).flags;
  }
  flags = row_status<Warps>(flags, shared_flags, status, status_row_offset + row);
#pragma unroll 1
  for (uint32_t group = warp; group < width / Group; group += Warps) {
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
  if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
}
} // namespace chop::row::block_quantize_parallel

namespace chop::row::silu_mul_quantize_parallel {
using silu_mul_quantize::product;

template<uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits, bool SkipInvalid>
__device__ __forceinline__ block_quantize::GroupValue quantize_group(
    const float* gate, const float* up, uint64_t at) {
  const uint32_t lane = threadIdx.x & 31;
  block_quantize::GroupValue result;
  float x = 0.f, y = 0.f;
  if (lane < 16) {
    x = product<GateMaxBits, UpLimitBits, SkipInvalid>(gate[at], up[at], result.flags);
    y = product<GateMaxBits, UpLimitBits, SkipInvalid>(gate[at + 1], up[at + 1], result.flags);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  maximum = fmaxf(maximum, __uint_as_float(FloorBits));
  result.scale_code = block_quantize::ceil_scale(maximum, 1.f / 448.f);
  const float scale = __uint_as_float(uint32_t(result.scale_code) << 23);
  if (lane >= 16) return result;
  result.codes =
      convert::pack_f32_e4m3(convert::Float2{__fdiv_rn(x, scale), __fdiv_rn(y, scale)});
  auto decoded = convert::unpack_f32_e4m3(result.codes);
  decoded.lo = __fmul_rn(decoded.lo, scale);
  decoded.hi = __fmul_rn(decoded.hi, scale);
  const auto exact = convert::bf16x2_to_f32(convert::bf16x2_from_f32(decoded));
  if (!isfinite(exact.lo) || !isfinite(exact.hi) ||
      __float_as_uint(exact.lo) != __float_as_uint(decoded.lo) ||
      __float_as_uint(exact.hi) != __float_as_uint(decoded.hi))
    result.flags |= block_quantize::UnrepresentableOutput;
  return result;
}

template<int Width, uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits,
         bool SkipInvalid = true>
__device__ void run(const float* gate, const float* up, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset) {
  static_assert(Width > 0 && Width % 32 == 0);
  constexpr uint32_t Warps = Width / 32 < 32 ? Width / 32 : 32;
  __shared__ uint32_t shared_flags[Warps];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += Warps) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    flags |= quantize_group<GateMaxBits,UpLimitBits,FloorBits,SkipInvalid>(gate,up,at).flags;
  }
  flags = block_quantize_parallel::row_status<Warps>(flags, shared_flags, status, status_row_offset + row);
#pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += Warps) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    block_quantize::GroupValue value;
    if (flags == 0)
      value = quantize_group<GateMaxBits,UpLimitBits,FloorBits,SkipInvalid>(gate,up,at);
    else value.scale_code = 127; // fixed finite scale 1
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = value.scale_code;
    if (lane < 16) {
      codes[at] = uint8_t(value.codes);
      codes[at + 1] = uint8_t(value.codes >> 8);
    }
  }
  ptx::griddep_launch_dependents();
}

} // namespace chop::row::silu_mul_quantize_parallel
