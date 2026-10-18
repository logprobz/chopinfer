#pragma once
#include "glue.cuh"

namespace chop::latent::glue {

// Lab-only control: pair the first half of the tail with the second half.
__device__ void rotate_split_half(float* values, uint16_t* native_q,
                                  const float* cos_sin,
                                  const uint32_t* live_rows, uint32_t heads) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, head = blockIdx.y;
  if (row >= *live_rows) return;
  const uint64_t base = (uint64_t(row) * heads + head) * 512;
  if (threadIdx.x < 224) {
    const uint32_t column = 2 * threadIdx.x;
    native_q[base + column] = convert::bf16_from_f32(values[base + column]);
    native_q[base + column + 1] = convert::bf16_from_f32(values[base + column + 1]);
  } else {
    const uint32_t pair = threadIdx.x - 224;
    const uint64_t first = base + 448 + pair, second = first + 32;
    const float a = values[first], b = values[second];
    const uint64_t angle = uint64_t(row) * 64 + 2 * pair;
    const float c = cos_sin[angle], s = cos_sin[angle + 1];
    const float x = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    const float y = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
    values[first] = x;
    values[second] = y;
    native_q[first] = convert::bf16_from_f32(x);
    native_q[second] = convert::bf16_from_f32(y);
  }
}

} // namespace chop::latent::glue
