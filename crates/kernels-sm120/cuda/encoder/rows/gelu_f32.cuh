#pragma once

// Exact-erf GELU of F32 rows: the projector's GELU between its norm and
// its MLP (`projector.norm-gelu-mlp.v1`).

#include "common.cuh"
#include "../../common/gelu.cuh"

namespace chop::multimodal {

// F32 rows [Width] to F32 rows, in place or not, through the encoder
// GEMMs' erf epilogue (`chop::gelu_erf`).
template<int Width>
__device__ void gelu_f32(const float* input, float* output,
                         const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint64_t at = uint64_t(row) * Width;
  for (int col = threadIdx.x; col < Width; col += blockDim.x)
    output[at + col] = chop::gelu_erf(input[at + col]);
}

} // namespace chop::multimodal
