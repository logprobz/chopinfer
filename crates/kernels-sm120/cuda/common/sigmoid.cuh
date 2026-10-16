#pragma once

// Sigmoid and products with it for the Qwen3.8 kernels. The usual
// 1 / (1 + exp(-x)) overflows its exponential below about -88 and returns 0,
// although sigmoid(x) and a product with it can still be representable. At
// x >= kTail every helper evaluates the expression its call sites used
// before, so those results are bitwise unchanged. Below, 1 + exp(x) rounds
// to 1 even in F64 and sigmoid(x) is exp(x): a product is formed in F64 and
// rounded once, so it keeps its precision where exp(x) alone would be an F32
// subnormal or zero.

#include <math.h>

namespace chop::sigmoid {

constexpr float kTail = -80.0f;

// v * exp(x) in F64, rounded once: v * sigmoid(x) for x < kTail.
__device__ __forceinline__ float tail_times(float v, float x) {
  return float(double(v) * exp(double(x)));
}

// sigmoid(x); below kTail exp(x), which underflows gradually.
__device__ __forceinline__ float sigmoid(float x) {
  return x >= kTail ? 1.f / (1.f + expf(-x)) : expf(x);
}

// v * sigmoid(x).
__device__ __forceinline__ float times(float v, float x) {
  return x >= kTail ? v * (1.f / (1.f + expf(-x))) : tail_times(v, x);
}

// silu(x) = x / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) {
  return x >= kTail ? x / (1.f + expf(-x)) : tail_times(x, x);
}

} // namespace chop::sigmoid
