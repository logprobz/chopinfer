#pragma once

#include "route_local_scan.cuh"

namespace chop::moe {

// One warp per token, in a one-CTA decode route or an eight-warp select
// CTA. The local-route engine keeps the same selection and metadata order.
template<int Experts>
struct SqrtSoftplusRowBias {
  static constexpr bool serial_sum = true, select_negative_infinity = true;
  const float* bias;
  const float* alternate_bias;
  const uint8_t* bias_select;
  float* own_scores;
  float scaling_factor;

  __device__ float transform(float logit) const {
    if (!isfinite(logit)) __trap();
    return sqrtf(__fadd_rn(fmaxf(logit, 0.f), log1pf(expf(-fabsf(logit)))));
  }
  __device__ float rank(float value, int expert) const {
    const int token = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t selected = bias_select[token];
    if (selected > 1) __trap();
    const float correction = (selected ? alternate_bias : bias)[expert];
    if (!isfinite(correction)) __trap();
    own_scores[uint64_t(token) * Experts + expert] = value;
    const float key = __fadd_rn(value, correction);
    if (!isfinite(key)) __trap();
    return key;
  }
  __device__ float selected(float unbiased, float, float) const { return unbiased; }
  __device__ float normalize(float weight, float sum) const {
    const float result = __fmul_rn(weight / __fadd_rn(sum, 1e-20f), scaling_factor);
    if (!isfinite(result)) __trap();
    return result;
  }
};

} // namespace chop::moe
