#pragma once

#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"
#include "../row/block_quantize.cuh"
#include "native_grouped.cuh"

// The serving intermediate quantizer (row/block_quantize.cuh, E4M3 per 32
// with a ceil UE8M0 scale: `run` at Format 0) fused into the native
// gate/up tiles' epilogue (native_grouped::compute's `Ceil`), with the same
// bits. The quantizer validates a whole token row (top_k routes of n
// columns) before it writes: any flag, or any status bit already set,
// zeroes the row's codes with scale 127. A tile holds only part of a row,
// so it writes each 32-column group's codes and scale as if the row were
// clean and ORs the group's flags into the token's status; `fixup`, the
// launch after the tiles, rewrites every live token whose status is then
// nonzero. A token's final status is its prior status OR every group's
// flags, as the quantizer's update, so both paths write the same codes,
// scales and status.
namespace chop::moe::native_ceil {

namespace bq = chop::row::block_quantize;

// One 32-column group of BF16 values at `input[at - 2 * lane]..`, a warp
// per group: `quantize_group<0,false,false,FloorBits>` with the lane taken
// from `lane` rather than `threadIdx.x` (the quantizer runs one warp per
// CTA), so any warp of a tile can run it.
template<uint32_t FloorBits>
__device__ __forceinline__ bq::GroupValue quantize_group(const uint16_t* input, uint64_t at,
                                                         uint32_t lane) {
  bq::GroupValue result;
  float x = 0.f, y = 0.f;
  if (lane < 16) {
    x = bq::input_value<false>(input, at, result.flags);
    y = bq::input_value<false>(input, at + 1, result.flags);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  maximum = fmaxf(maximum, __uint_as_float(FloorBits));
  result.scale_code = bq::ceil_scale(maximum, 1.f / 448.f);
  const float scale = __uint_as_float(uint32_t(result.scale_code) << 23);
  if (lane >= 16) return result;
  const convert::Float2 normalized{__fdiv_rn(x, scale), __fdiv_rn(y, scale)};
  result.codes = convert::pack_f32_e4m3(normalized);
  convert::Float2 decoded = convert::unpack_f32_e4m3(result.codes);
  decoded.lo = __fmul_rn(decoded.lo, scale);
  decoded.hi = __fmul_rn(decoded.hi, scale);
  result.pair = convert::bf16x2_from_f32(decoded);
  const auto exact = convert::bf16x2_to_f32(result.pair);
  if (!isfinite(exact.lo) || !isfinite(exact.hi) ||
      __float_as_uint(exact.lo) != __float_as_uint(decoded.lo) ||
      __float_as_uint(exact.hi) != __float_as_uint(decoded.hi))
    result.flags |= bq::UnrepresentableOutput;
  return result;
}

// `native_grouped::compute`'s `Ceil` for gate/up: `codes` and `scales` are
// the quantizer's outputs over token rows of top_k * n columns, `status`
// its row status (row offset zero).
template<uint32_t FloorBits>
struct Ceil {
  static constexpr bool enabled = true;
  uint8_t* codes;
  uint8_t* scales;
  uint32_t* status;

  // `staged` holds the tile's BF16 outputs, T::rows rows of the strip's
  // columns from `column`; rows whose route is negative are padding.
  template<class C, class T>
  __device__ void tile(const uint16_t* staged, const int* routes, int column) const {
    constexpr int Cols = native_grouped::tile_cols<C, T>, Groups = Cols / 32;
    static_assert(C::gate_up && Cols % 32 == 0 && C::n % 32 == 0, "whole 32-column groups");
    const uint32_t lane = threadIdx.x & 31;
    for (int item = threadIdx.x >> 5; item < T::rows * Groups; item += T::threads / 32) {
      const int r = item / Groups, group = item % Groups;
      const int route = routes[r], col = column + group * 32;
      if (route < 0 || col >= C::n) continue;  // warp-uniform
      const bq::GroupValue v = quantize_group<FloorBits>(
          staged, uint64_t(r) * Cols + group * 32 + 2 * lane, lane);
      uint32_t flags = v.flags;
      for (int step = 16; step; step >>= 1) flags |= __shfl_xor_sync(0xffffffff, flags, step);
      const uint64_t at = uint64_t(route) * C::n + col;
      if (lane < 16) {
        codes[at + 2 * lane] = uint8_t(v.codes);
        codes[at + 2 * lane + 1] = uint8_t(v.codes >> 8);
      }
      if (lane == 0) {
        scales[at / 32] = v.scale_code;
        if (flags) atomicOr(status + route / C::top_k, flags);
      }
    }
  }
};

// After the tiles: every live token whose status is nonzero gets the
// quantizer's flagged row, zero codes and scale 127 for every group.
template<int Width>
__device__ void fixup(const int* live_tokens, const uint32_t* status, uint8_t* codes,
                      uint8_t* scales) {
  static_assert(Width % 32 == 0);
  ptx::griddep_wait();
  const int row = blockIdx.x;
  if (row < *live_tokens && status[row] != 0) {
    for (int i = threadIdx.x; i < Width; i += blockDim.x) codes[uint64_t(row) * Width + i] = 0;
    for (int i = threadIdx.x; i < Width / 32; i += blockDim.x)
      scales[uint64_t(row) * (Width / 32) + i] = 127;
  }
  ptx::griddep_launch_dependents();
}

}  // namespace chop::moe::native_ceil
