#pragma once

// Row gather of F32 rows by index (`reorder.merge-group-columns.v1` under
// `f32-pairs.v1`).

#include "common.cuh"

namespace chop::multimodal {

template<int Width>
__device__ void permute_f32(const float* input, const int32_t* indices,
                            float* output, const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  const uint32_t live = *live_rows;
  if (row >= live) return;
  const int32_t src = indices[row];
  if (src < 0 || uint32_t(src) >= live) return;
  static_assert(Width % 4 == 0);
  for (int col = threadIdx.x * 4; col < Width; col += blockDim.x * 4) {
    const float4 value = reinterpret_cast<const float4*>(
        input + uint64_t(src) * Width + col)[0];
    reinterpret_cast<float4*>(output + uint64_t(row) * Width + col)[0] = value;
  }
}

} // namespace chop::multimodal
