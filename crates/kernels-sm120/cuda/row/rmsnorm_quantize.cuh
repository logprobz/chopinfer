#pragma once
#include "../common/ptx.cuh"

// This translation unit contains only fused entries. Defer the legacy
// norm's signal without changing its arithmetic or its locked header.
// Load ptx.cuh first so the macro cannot rename the real PTX helper.
namespace chop::ptx {
__device__ __forceinline__ void norm_quantize_defer_dependents() {}
}
#define griddep_launch_dependents norm_quantize_defer_dependents
#include "rmsnorm_f32.cuh"
#undef griddep_launch_dependents
#include "ceil_epilogue.cuh"

namespace chop::row::rmsnorm_quantize {

// Keep the original norm body, its 256-thread tree, and its global output.
// Consumers wait for the quantizer epilogue as well as the norm.
template<int Width, bool F32, uint32_t FloorBits>
__device__ void run(const void* input, const uint16_t* weight, void* normed,
                    float* widened, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status,
                    uint64_t status_row_offset, float eps) {
  static_assert(Width > 0 && Width % 32 == 0);
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
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
  ceil_epilogue<Width,F32,FloorBits>(normed,codes,scales,status,status_row_offset);
  ptx::griddep_launch_dependents();
}

} // namespace chop::row::rmsnorm_quantize
