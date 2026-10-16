#pragma once

// RMSNorm of F32 rows (`norm.rms-f32.v1`).

#include "common.cuh"

namespace chop::multimodal {

// RMSNorm of F32 rows to F32 rows: the F32 mean of the squares, its
// correctly rounded reciprocal square root after eps, then (x * rstd) *
// weight.
template<int Width>
__device__ void rms_f32(const float* input, const uint16_t* weight,
                        float* output, const uint32_t* live_rows, float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float sums[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* x = input + uint64_t(row) * Width;
  float squares = 0.f;
  for (int col = threadIdx.x; col < Width; col += blockDim.x)
    squares += x[col] * x[col];
  const float mean = chop::row::sum_cta(squares, sums) / Width;
  const float rstd = __frsqrt_rn(mean + eps);
  for (int col = threadIdx.x; col < Width; col += blockDim.x)
    output[uint64_t(row) * Width + col] = x[col] * rstd * bf(weight[col]);
}

} // namespace chop::multimodal
