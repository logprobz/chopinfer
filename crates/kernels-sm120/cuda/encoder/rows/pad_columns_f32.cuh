#pragma once

// Zero-column padding of F32 rows (`pad.zero-columns.v1`): a payload whose
// width is not a multiple of the pair GEMM's 64-wide K steps.

#include "common.cuh"

namespace chop::multimodal {

// F32 rows [Input] copied unchanged into rows [Output], zero past Input.
// Every output element of a live row is written.
template<int Input, int Output>
__device__ void pad_columns_f32(const float* input, float* output,
                                const uint32_t* live_rows) {
  static_assert(Input > 0 && Output > Input, "padding widens the rows");
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* from = input + uint64_t(row) * Input;
  float* to = output + uint64_t(row) * Output;
  for (int col = threadIdx.x; col < Output; col += blockDim.x)
    to[col] = col < Input ? from[col] : 0.f;
}

} // namespace chop::multimodal
