#pragma once

// Clamped SwiGLU of F32 rows (`mlp.swiglu-clamped.v1` under `f32-pairs.v1`).

#include "common.cuh"

namespace chop::multimodal {

// Input rows [gate | up] of 2 * Hidden F32 values (their biases already
// added by the GEMM); output silu(min(gate, limit)) * clamp(up, -limit,
// limit), with silu(x) = x / (1 + exp(-x)) and each operation rounded in
// F32. The comparisons keep a NaN a NaN, as the reference's clamps do.
template<int Hidden>
__device__ void swiglu_clamped_f32(const float* input, float* output,
                                   const uint32_t* live_rows, float limit) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* gate = input + uint64_t(row) * 2 * Hidden;
  for (int col = threadIdx.x; col < Hidden; col += blockDim.x) {
    float x = gate[col];
    float up = gate[Hidden + col];
    x = x > limit ? limit : x;
    up = up > limit ? limit : (up < -limit ? -limit : up);
    const float silu = __fdiv_rn(x, __fadd_rn(1.f, expf(-x)));
    output[uint64_t(row) * Hidden + col] = __fmul_rn(silu, up);
  }
}

} // namespace chop::multimodal
