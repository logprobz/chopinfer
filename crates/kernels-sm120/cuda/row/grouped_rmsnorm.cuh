#pragma once
#include "row/row.cuh"

namespace chop::row {
__device__ __forceinline__ float norm_weight(uint16_t x) { return bf(x); }
__device__ __forceinline__ float norm_weight(float x) { return x; }

__device__ __forceinline__ float norm_sum_xor_serial(float value, float* sums) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) sums[warp] = value;
  __syncthreads();
  float total = 0.f;
  for (int w = 0; w < int(blockDim.x / 32); ++w) total += sums[w];
  return total;
}

template<int Width, int Groups, typename Weight, bool XorSerial>
__device__ void grouped_rmsnorm(const uint16_t* x, const Weight* weight,
                               uint16_t* output, const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  __shared__ float sums[8];
  const uint64_t base = (uint64_t(blockIdx.x) * Groups + blockIdx.y) * Width;
  float square = 0.f;
  for (int c = threadIdx.x; c < Width; c += blockDim.x) {
    float v = bf(x[base + c]);
    square += v * v;
  }
  const float total = XorSerial ? norm_sum_xor_serial(square, sums) : sum_cta(square, sums);
  const float inv = rsqrtf(total / Width + 1e-6f);
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    output[base + c] = to_bf(bf(x[base + c]) * inv *
                            (1.f + norm_weight(weight[blockIdx.y * Width + c])));
}
} // namespace chop::row
