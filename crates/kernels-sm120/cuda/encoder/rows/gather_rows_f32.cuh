#pragma once

// Grouped row gather of F32 rows with holes (`merge.space-to-depth.v1`
// under `f32-pairs.v1`): output row r is Group input rows side by side,
// block j input row indices[r * Group + j], or zeros where that index is
// negative. The host computes the indices; every nonnegative one names an
// input row.

#include "common.cuh"

namespace chop::multimodal {

template<int Width, int Group>
__device__ void gather_rows_f32(const float* input, const int32_t* indices,
                                float* output, const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  static_assert(Width % 4 == 0 && Group > 0);
  constexpr int Columns = Width * Group;
  for (int col = threadIdx.x * 4; col < Columns; col += blockDim.x * 4) {
    const int block = col / Width;
    const int32_t src = indices[uint64_t(row) * Group + block];
    float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
    if (src >= 0) {
      value = reinterpret_cast<const float4*>(
          input + uint64_t(src) * Width + (col - block * Width))[0];
    }
    reinterpret_cast<float4*>(output + uint64_t(row) * Columns + col)[0] = value;
  }
}

} // namespace chop::multimodal
