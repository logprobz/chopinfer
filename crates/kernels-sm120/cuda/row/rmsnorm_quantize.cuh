#pragma once
#include "rmsnorm_f32.cuh"
#include "ceil_parallel.cuh"

namespace chop::row::rmsnorm_quantize {

// Keep the original norm body, its 256-thread tree, and its global output.
// Its early dependent-launch signal requires a non-PDL launch for this
// composition: consumers must wait for the quantizer epilogue as well.
template<int Width, bool F32, uint32_t FloorBits>
__device__ void run(const void* input, const uint16_t* weight, void* normed,
                    float* widened, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status,
                    uint64_t status_row_offset, float eps) {
  static_assert(Width > 0 && Width % 32 == 0);
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  if constexpr (F32)
    rmsnorm_f32<Width>(static_cast<const float*>(input),weight,
        static_cast<float*>(normed),live_rows,eps,status,status_row_offset);
  else
    rmsnorm<Width,0,0,false>(static_cast<const uint16_t*>(input),weight,
        nullptr,nullptr,static_cast<uint16_t*>(normed),live_rows,eps);
  __syncthreads();
  if constexpr (!F32) {
    if (widened)
      for (int c = threadIdx.x; c < Width; c += 256) {
        const uint64_t at = uint64_t(row) * Width + c;
        widened[at] = bf(static_cast<const uint16_t*>(normed)[at]);
      }
  }
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

} // namespace chop::row::rmsnorm_quantize
