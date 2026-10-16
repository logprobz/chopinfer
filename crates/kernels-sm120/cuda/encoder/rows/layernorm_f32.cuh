#pragma once

// LayerNorm of F32 rows (`norm.layer-f32.v1`).

#include "common.cuh"

namespace chop::multimodal {

// LayerNorm of F32 rows to F32 rows: the mean, then the population variance
// of the centered values, then ((x - mean) * rstd) * weight + bias.
template<int Width>
__device__ void layernorm_f32(const float* input, const uint16_t* weight,
                              const uint16_t* bias, float* output,
                              const uint32_t* live_rows, float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float sums[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* x = input + uint64_t(row) * Width;
  float local = 0.f;
  for (int col = threadIdx.x; col < Width; col += blockDim.x) local += x[col];
  const float mean = chop::row::sum_cta(local, sums) / Width;
  float squares = 0.f;
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    const float centered = x[col] - mean;
    squares += centered * centered;
  }
  // Every warp must consume the mean before the variance reuses sums[0].
  __syncthreads();
  const float variance = chop::row::sum_cta(squares, sums) / Width;
  const float rstd = __frsqrt_rn(variance + eps);
  for (int col = threadIdx.x; col < Width; col += blockDim.x)
    output[uint64_t(row) * Width + col] =
        (x[col] - mean) * rstd * bf(weight[col]) + bf(bias[col]);
}

} // namespace chop::multimodal
