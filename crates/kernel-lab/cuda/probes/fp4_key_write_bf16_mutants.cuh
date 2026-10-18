#pragma once

#include "row/block_quantize.cuh"

namespace chop::bf16_write_tests {

struct TruncateInput {
  __device__ static uint32_t pair(const float* values, uint64_t at,
      chop::row::block_quantize::GroupValue, uint32_t flags) {
    if (flags) return 0;
    // Store the continuous pre-quantizer value, skipping native quantization.
    return (__float_as_uint(values[at]) >> 16) | (__float_as_uint(values[at+1]) & 0xffff0000u);
  }
};
struct DropScale {
  __device__ static uint32_t pair(const float*, uint64_t,
      chop::row::block_quantize::GroupValue value, uint32_t flags) {
    if (flags) return 0;
    return convert::bf16x2_from_f32(convert::unpack_f32_e2m1(value.codes));
  }
};

}  // namespace chop::bf16_write_tests

// BEGIN mirrored writer, checked against the serving body by the family builder.


#include <math.h>
#include <stdint.h>
#include "common/convert.cuh"
#include "row/block_quantize.cuh"
#include "common/ptx.cuh"

namespace chop::bf16_write_tests {

// Index writes normalize the projected key before tail RoPE. Compressed
// writes receive the already normalized latent. Inputs, normalization and
// RoPE stay in F32 until the single BF16 quantizer-input rounding. Native
// and BF16 cache forms quantize identically.
template<int Width, int Rotary, int Ratio, int Page,
         bool Normalize, bool E4m3Scale, bool Packed, class Status, class Store>
__device__ void key_write(const float* key, const uint16_t* norm_weight,
    const float* cos_sin, const int32_t* positions, const int32_t* row_batch,
    const uint32_t* live_rows, const int32_t* page_tables,
    uint8_t* codes, uint8_t* scales, uint32_t items, uint32_t pages,
    uint32_t max_pages, uint64_t code_stride, uint64_t scale_stride,
    uint32_t table_rows, float eps, uint32_t* status, uint64_t status_row_offset) {
  static_assert(Width == 128 || Width == 512);
  static_assert(Rotary == 64 && Rotary <= Width);
  static_assert(Ratio == 1 || Ratio == 2);
  static_assert(Page > 0);
  constexpr int Block = E4m3Scale ? 16 : 32;
  __shared__ float values[Width], sums[4];
  __shared__ uint32_t row_flags;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const int32_t position = positions[row], item = row_batch[row];
  if (position < 0 || item < 0 || uint32_t(item) >= items) { ptx::griddep_launch_dependents(); return; }
  const uint64_t rotary_position = uint64_t(position) * Ratio;
  const uint32_t logical_page = uint32_t(position) / Page;
  if (rotary_position >= table_rows || logical_page >= max_pages) { ptx::griddep_launch_dependents(); return; }
  const int32_t page = page_tables[uint64_t(item) * max_pages + logical_page];
  if (page < 0 || uint32_t(page) >= pages) { ptx::griddep_launch_dependents(); return; }
  const uint32_t in_page = uint32_t(position) % Page;
  uint8_t* code_row = codes + uint64_t(page) * code_stride
      + uint64_t(in_page) * (Packed ? Width / 2 : Width * 2);
  uint8_t* scale_row = nullptr;
  if constexpr (Packed)
    scale_row = scales + uint64_t(page) * scale_stride + uint64_t(in_page) * (Width / Block);

  if (threadIdx.x == 0) row_flags = 0;
  __syncthreads();
  uint32_t flags = 0;
  float square = 0.f;
  for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
    float value = key[uint64_t(row) * Width + d];
    if (!isfinite(value)) { flags |= chop::row::block_quantize::NonfiniteInput; value = 0.f; }
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
      float weight = convert::bf16_to_f32(norm_weight[d]);
      if (!isfinite(weight)) { flags |= chop::row::block_quantize::NonfiniteInput; weight = 0.f; }
      const float normalized = __fmul_rn(__fmul_rn(values[d], inv), weight);
      values[d] = normalized;
    }
  }
  __syncthreads();

  // Rotate in F32. The shared quantizer applies the only BF16 rounding.
  for (uint32_t pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
    const uint32_t d = pair * 2;
    if (d < Width - Rotary) continue;
    const float* cs = cos_sin + rotary_position * Rotary + d - (Width - Rotary);
    float c = cs[0], s = cs[1];
    if (!isfinite(c) || !isfinite(s)) {
      flags |= chop::row::block_quantize::NonfiniteInput;
      c = s = 0.f;
    }
    const float x = values[d], y = values[d + 1];
    values[d] = __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
    values[d + 1] = __fadd_rn(__fmul_rn(x, s), __fmul_rn(y, c));
  }
  if (flags) atomicOr(&row_flags, flags);
  __syncthreads();

  // One warp validates every group before publishing any cache bytes.
  // This is the shared trained quantizer, including finite scale saturation
  // and decoder representability checks with no data-reachable traps.
  if (threadIdx.x < 32) {
    constexpr int Format = E4m3Scale ? 1 : 2;
    const uint32_t lane = threadIdx.x;
    flags = row_flags;
#pragma unroll 1
    for (uint32_t group = 0; group < Width / Block; ++group) {
      const uint64_t at = uint64_t(group) * Block + 2 * lane;
      flags |= chop::row::block_quantize::quantize_group<Format,true,!Packed,0>(values,at).flags;
    }
    for (int step = 16; step; step >>= 1)
      flags |= __shfl_xor_sync(0xffffffff, flags, step);
    if (lane == 0) flags = Status::update(status,status_row_offset+row,flags);
    flags = __shfl_sync(0xffffffff,flags,0);
#pragma unroll 1
    for (uint32_t group = 0; group < Width / Block; ++group) {
      const uint64_t at = uint64_t(group) * Block + 2 * lane;
      chop::row::block_quantize::GroupValue value;
      if (flags == 0) value = chop::row::block_quantize::quantize_group<Format,true,!Packed,0>(values,at);
      else value.scale_code = E4m3Scale ? 0x38 : 127;
      if constexpr (Packed) {
        if (lane == 0) scale_row[group] = value.scale_code;
        if (lane < Block / 2) code_row[at / 2] = uint8_t(value.codes);
      } else if (lane < Block / 2) {
        value.pair = Store::pair(values,at,value,flags);
        uint16_t* target = reinterpret_cast<uint16_t*>(code_row);
        target[at] = uint16_t(value.pair);
        target[at + 1] = uint16_t(value.pair >> 16);
      }
    }
  }
  __syncthreads();
  ptx::griddep_launch_dependents();
}

}  // namespace chop::bf16_write_tests
