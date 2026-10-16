#pragma once

// SwiGLU of F32 rows (`mlp.swiglu.v1` under `f32-pairs.v1`).

#include "common.cuh"

namespace chop::multimodal {

// Input rows [gate | up] of 2 * Hidden F32 values; output silu(gate) * up,
// with silu(x) = x / (1 + exp(-x)), each operation rounded in F32.
template<int Hidden>
__device__ void swiglu_f32(const float* input, float* output,
                           const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* gate = input + uint64_t(row) * 2 * Hidden;
  for (int col = threadIdx.x; col < Hidden; col += blockDim.x) {
    const float x = gate[col];
    const float silu = __fdiv_rn(x, __fadd_rn(1.f, expf(-x)));
    output[uint64_t(row) * Hidden + col] = __fmul_rn(silu, gate[Hidden + col]);
  }
}

} // namespace chop::multimodal
