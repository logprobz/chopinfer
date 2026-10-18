#pragma once
#include "row.cuh"

namespace chop::row {

// Plain RMSNorm with BF16 operands and F32 output. Keep this entry separate
// from the existing BF16 specializations so their cubins remain identical.
template<int Width>
__device__ void rmsnorm_f32(const uint16_t* x, const uint16_t* weight,
                           float* y, const uint32_t* live_rows, float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float sums[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint64_t base = uint64_t(row) * Width;
  float square = 0.f;
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    for (int j = 0; j < 8; ++j) {
      const float v = bf(x[base + c + j]);
      square = __fmaf_rn(v, v, square);
    }
  }
  const float total = sum_cta(square, sums);
  const float inv = rsqrtf(total / Width + eps);
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    y[base + c] = __fmul_rn(__fmul_rn(bf(x[base + c]), inv), bf(weight[c]));
}

} // namespace chop::row
