#pragma once

// `linear.pairs.v1`: the pairs GEMM's epilogue. `Gelu` is 0 (none), 1 (erf)
// or 2 (tanh) on the biased F32 accumulator; `Out` is float or uint16_t
// (BF16, rounded once at the store); `Residual` adds the F32 output to the
// rows it overwrites.

#include "../gemm/encoder_gemm_pairs.cuh"

namespace chop::encoder {

template<bool Bias, int Gelu, bool Residual, class OutType>
struct LinearPairs {
  using Out = OutType;
  static constexpr bool bias = Bias;
  static constexpr bool residual = Residual;

  __device__ __forceinline__ static float apply(float x) {
    return chop::gemm::encoder_activation<Gelu>(x);
  }
};

} // namespace chop::encoder
