#pragma once

// Per-head gated RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w * act(z), with
// one plain weight shared by all heads. The normalized value is not rounded
// before the weight; only y is rounded to BF16.

#include "common.cuh"

namespace chop::linear {

// Grid: (row capacity). Each warp normalizes whole heads of one row.
template<int Width, int HeadDim, bool Sigmoid>
__device__ void gated_rmsnorm(const uint16_t* input, const uint16_t* gate,
                              uint32_t gate_stride, const uint16_t* weight,
                              uint16_t* output, const uint32_t* live_rows,
                              float eps) {
  static_assert(HeadDim == 128 && Width % HeadDim == 0);
  constexpr int Heads = Width / HeadDim;
  ptx::griddep_launch_dependents();
  const int lane = threadIdx.x & 31;
  // The gamma is shared by every head and independent of the input row.
  const uint2 wpack = reinterpret_cast<const uint2*>(weight)[lane];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float w[4] = {bf_lo(wpack.x), bf_hi(wpack.x), bf_lo(wpack.y), bf_hi(wpack.y)};
  for (int head = threadIdx.x >> 5; head < Heads; head += blockDim.x >> 5) {
    const uint64_t in = uint64_t(row) * Width + head * HeadDim + lane * 4;
    const uint64_t gz = uint64_t(row) * gate_stride + head * HeadDim + lane * 4;
    const uint2 xpack = *reinterpret_cast<const uint2*>(input + in);
    const uint2 zpack = *reinterpret_cast<const uint2*>(gate + gz);
    const float x[4] = {bf_lo(xpack.x), bf_hi(xpack.x), bf_lo(xpack.y), bf_hi(xpack.y)};
    const float z[4] = {bf_lo(zpack.x), bf_hi(zpack.x), bf_lo(zpack.y), bf_hi(zpack.y)};
    float squares = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) squares = fmaf(x[j], x[j], squares);
    const float inverse = rsqrtf(warp_sum(squares) / HeadDim + eps);
    uint16_t y[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float act = Sigmoid ? sigmoid(z[j]) : silu(z[j]);
      y[j] = to_bf(x[j] * inverse * w[j] * act);
    }
    *reinterpret_cast<uint2*>(output + in) =
        make_uint2(uint32_t(y[0]) | (uint32_t(y[1]) << 16),
                   uint32_t(y[2]) | (uint32_t(y[3]) << 16));
  }
}

} // namespace chop::linear
