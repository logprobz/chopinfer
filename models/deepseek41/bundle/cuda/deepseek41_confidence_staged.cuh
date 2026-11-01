#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

constexpr uint32_t ConfidenceStagedThreads = 128;
constexpr uint32_t ConfidenceStagedChain = 128;
constexpr uint32_t ConfidenceStagedMaxChains = 64;
// One padding word per chain keeps each thread's chain reads conflict-free.
constexpr uint32_t ConfidenceStagedPitch = ConfidenceStagedChain + 1;

// One CTA per live row, one thread per 128-product chain of the
// concatenated hidden and predecessor-embedding operands; the planner admits
// only whole chains, at most 64. Each chain thread loads its 128 BF16
// weights, one contiguous 256-byte run, into registers before the dependency
// wait, since no predecessor writes the weights. After the wait the CTA
// copies the row's operands into shared memory with coalesced loads, then
// every chain runs `confidence_parallel`'s products in K order from +0 and
// thread zero folds the chains left to right from +0, so every score keeps
// its bits.
__device__ void confidence_staged(const float* hidden, const uint16_t* markov,
                                  const uint16_t* weight, float* scores,
                                  const uint32_t* live_rows, uint32_t width,
                                  uint32_t rank, uint32_t hidden_stride,
                                  uint32_t score_stride) {
  __shared__ float x[ConfidenceStagedMaxChains * ConfidenceStagedPitch];
  __shared__ float partials[ConfidenceStagedMaxChains];
  const uint32_t chains = (width + rank) / ConfidenceStagedChain;
  const uint32_t t = threadIdx.x;
  uint4 w[ConfidenceStagedChain / 8];
  if (t < chains) {
    const uint16_t* run = weight + uint64_t(t) * ConfidenceStagedChain;
    if ((reinterpret_cast<uintptr_t>(run) & 15) == 0) {
      const uint4* source = reinterpret_cast<const uint4*>(run);
#pragma unroll
      for (uint32_t i = 0; i < ConfidenceStagedChain / 8; ++i) w[i] = __ldg(source + i);
    } else {
#pragma unroll
      for (uint32_t i = 0; i < ConfidenceStagedChain / 8; ++i) {
        uint32_t words[4];
#pragma unroll
        for (uint32_t h = 0; h < 4; ++h) {
          words[h] = uint32_t(__ldg(run + i * 8 + h * 2)) |
                     (uint32_t(__ldg(run + i * 8 + h * 2 + 1)) << 16);
        }
        w[i] = make_uint4(words[0], words[1], words[2], words[3]);
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* h = hidden + uint64_t(row) * hidden_stride;
  if ((reinterpret_cast<uintptr_t>(h) & 15) == 0 && width % 4 == 0) {
    const float4* source = reinterpret_cast<const float4*>(h);
#pragma unroll 4
    for (uint32_t i = t; i < width / 4; i += ConfidenceStagedThreads) {
      const float4 v = source[i];
      const uint32_t k = i * 4;
      // Four consecutive K never cross a chain, since 128 is a multiple of 4.
      float* d = x + (k / ConfidenceStagedChain) * ConfidenceStagedPitch + k % ConfidenceStagedChain;
      d[0] = v.x;
      d[1] = v.y;
      d[2] = v.z;
      d[3] = v.w;
    }
  } else {
    for (uint32_t k = t; k < width; k += ConfidenceStagedThreads) {
      x[(k / ConfidenceStagedChain) * ConfidenceStagedPitch + k % ConfidenceStagedChain] = h[k];
    }
  }
  const uint16_t* m = markov + uint64_t(row) * rank;
  for (uint32_t k = t; k < rank; k += ConfidenceStagedThreads) {
    const uint32_t i = width + k;
    x[(i / ConfidenceStagedChain) * ConfidenceStagedPitch + i % ConfidenceStagedChain] =
        __uint_as_float(uint32_t(m[k]) << 16);
  }
  __syncthreads();
  if (t < chains) {
    const float* xs = x + t * ConfidenceStagedPitch;
    float partial = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < ConfidenceStagedChain / 8; ++i) {
      const uint32_t words[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
      for (uint32_t q = 0; q < 4; ++q) {
        partial = __fmaf_rn(xs[i * 8 + q * 2], __uint_as_float(words[q] << 16), partial);
        partial = __fmaf_rn(xs[i * 8 + q * 2 + 1], __uint_as_float(words[q] & 0xffff0000u),
                            partial);
      }
    }
    partials[t] = partial;
  }
  __syncthreads();
  if (t == 0) {
    float sum = 0.f;
    for (uint32_t chain = 0; chain < chains; ++chain) sum = __fadd_rn(sum, partials[chain]);
    scores[uint64_t(row) * score_stride] = sum;
  }
}

} // namespace chop::deepseek41::dspark
