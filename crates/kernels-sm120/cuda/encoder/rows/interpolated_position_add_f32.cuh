#pragma once

// Learned position grid resampled per image, added to F32 rows.

#include "common.cuh"

namespace chop::multimodal {

// Learned position table resampled per image: each F32 row adds
// sum_j weights[row][j] * table[indices[row][j]], HF's F32 weighted sum and
// add. The products and the sum are rounded separately (no FMA) in tap
// order. Taps outside the table contribute zero. `output` may alias `input`.
template<int Width, int TableRows>
__device__ void interpolated_position_add_f32(const float* input,
                                              const uint16_t* table,
                                              const int32_t* indices,
                                              const float* weights,
                                              float* output,
                                              const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  int32_t tap[4];
  float weight[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    tap[j] = indices[uint64_t(row) * 4 + j];
    weight[j] = weights[uint64_t(row) * 4 + j];
  }
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (tap[j] >= 0 && tap[j] < TableRows)
        sum = __fadd_rn(sum, __fmul_rn(weight[j],
            bf(table[uint64_t(tap[j]) * Width + col])));
    const uint64_t i = uint64_t(row) * Width + col;
    output[i] = __fadd_rn(input[i], sum);
  }
}

} // namespace chop::multimodal
