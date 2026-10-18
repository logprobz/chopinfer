#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

__device__ __forceinline__ void dependencies() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ float decode(uint16_t bits) {
  return __uint_as_float(uint32_t(bits) << 16);
}

// Each thread owns one vocabulary logit. Complete the projection before
// adding base, including when base and output are the same allocation.
__device__ void markov(const uint16_t* embedding, const uint16_t* weight,
                       const float* base, float* output,
                       const uint32_t* live_rows, uint32_t rank,
                       uint32_t vocab, uint32_t stride) {
  dependencies();
  const uint32_t row = blockIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= *live_rows || col >= vocab) return;
  float sum = 0.f;
  for (uint32_t start = 0; start < rank;) {
    const uint32_t end = start + min(128u, rank - start);
    float partial = 0.f;
    for (uint32_t k = start; k < end; ++k) {
      partial = __fmaf_rn(decode(embedding[uint64_t(row) * rank + k]),
                          decode(weight[uint64_t(col) * rank + k]), partial);
    }
    sum = __fadd_rn(sum, partial);
    start = end;
  }
  const uint64_t index = uint64_t(row) * stride + col;
  output[index] = __fadd_rn(base[index], sum);
}

// A confidence row traverses the concatenation in order. Chain boundaries
// depend on the full K index, not on the boundary between its two inputs.
__device__ void confidence(const float* hidden, const uint16_t* markov,
                           const uint16_t* weight, float* scores,
                           const uint32_t* live_rows, uint32_t width,
                           uint32_t rank, uint32_t hidden_stride,
                           uint32_t score_stride) {
  dependencies();
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= *live_rows) return;
  const uint32_t total = width + rank;
  float sum = 0.f;
  for (uint32_t start = 0; start < total;) {
    const uint32_t end = start + min(128u, total - start);
    float partial = 0.f;
    for (uint32_t k = start; k < end; ++k) {
      const float x = k < width
          ? hidden[uint64_t(row) * hidden_stride + k]
          : decode(markov[uint64_t(row) * rank + k - width]);
      partial = __fmaf_rn(x, decode(weight[k]), partial);
    }
    sum = __fadd_rn(sum, partial);
    start = end;
  }
  scores[uint64_t(row) * score_stride] = sum;
}

} // namespace chop::deepseek41::dspark
