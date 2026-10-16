#pragma once

// F32 residual add (a transformer's `carry` under `f32-pairs.v1`).

#include "common.cuh"

namespace chop::multimodal {

template<int Width>
__device__ void residual_add_f32(const float* input, const float* residual,
                                 float* output, const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    const uint64_t i = uint64_t(row) * Width + col;
    output[i] = __fadd_rn(input[i], residual[i]);
  }
}

} // namespace chop::multimodal
