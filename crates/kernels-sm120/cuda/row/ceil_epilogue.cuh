#pragma once
#include "ceil_parallel.cuh"

namespace chop::row {
// One 256-thread CTA owns a complete row. The caller publishes the input
// first and signals dependents only after all of its epilogues finish.
template<int Width, bool F32, uint32_t FloorBits>
__device__ __forceinline__ void ceil_epilogue(const void* normed, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint64_t status_row_offset) {
  const uint32_t row = blockIdx.x;
  constexpr uint32_t Warps = 8;
  __shared__ uint32_t shared_flags[Warps];
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += Warps) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    flags |= block_quantize_parallel::quantize_group<0,F32,false,FloorBits>(normed,at).flags;
  }
  flags = block_quantize_parallel::row_status<Warps>(flags,shared_flags,status,status_row_offset+row);
#pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += Warps) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    block_quantize::GroupValue value;
    if (flags == 0)
      value = block_quantize_parallel::quantize_group<0,F32,false,FloorBits>(normed,at);
    else value.scale_code = 127;
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = value.scale_code;
    if (lane < 16) {
      codes[at] = uint8_t(value.codes);
      codes[at+1] = uint8_t(value.codes >> 8);
    }
  }
 }
} // namespace chop::row
