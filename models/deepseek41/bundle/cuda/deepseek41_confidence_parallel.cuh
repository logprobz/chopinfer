#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

// Threads compute independent 128-product chains. Thread zero folds their
// F32 partials in increasing K order, including across groups of 128 chains.
__device__ void confidence_parallel(const float* hidden, const uint16_t* markov,
                                    const uint16_t* weight, float* scores,
                                    const uint32_t* live_rows, uint32_t width,
                                    uint32_t rank, uint32_t hidden_stride,
                                    uint32_t score_stride) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  __shared__ float partials[128];
  const uint64_t total = uint64_t(width) + rank;
  float sum = 0.f;
  for (uint64_t base = 0; base < total; base += 128 * 128) {
    const uint64_t start = base + uint64_t(threadIdx.x) * 128;
    const uint64_t end = start + min(uint64_t(128), total > start ? total - start : 0);
    float partial = 0.f;
    for (uint64_t k = start; k < end; ++k) {
      const float x = k < width
          ? hidden[uint64_t(row) * hidden_stride + k]
          : __uint_as_float(uint32_t(markov[uint64_t(row) * rank + k - width]) << 16);
      const float w = __uint_as_float(uint32_t(weight[k]) << 16);
      partial = __fmaf_rn(x, w, partial);
    }
    partials[threadIdx.x] = partial;
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t chains = uint32_t(min(uint64_t(128), (total - base + 127) / 128));
      for (uint32_t chain = 0; chain < chains; ++chain) {
        sum = __fadd_rn(sum, partials[chain]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) scores[uint64_t(row) * score_stride] = sum;
}

} // namespace chop::deepseek41::dspark
