#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

constexpr uint32_t MarkovTiledColumns = 128;
constexpr uint32_t MarkovTiledRows = 8;
constexpr uint32_t MarkovTiledChain = 128;

// Each thread owns one vocabulary logit of up to eight rows per pass. The
// CTA stages one 128-product chain of its 128 weight rows, read once and
// coalesced, for every row. Each logit keeps `markov`'s arithmetic: chains
// of 128 products in K order, folded left to right from +0, then base.
// Base is read after the projection, so base and output may alias.
__device__ void markov_tiled(const uint16_t* embedding, const uint16_t* weight,
                             const float* base, float* output,
                             const uint32_t* live_rows, uint32_t rank,
                             uint32_t vocab, uint32_t stride) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // An odd pad word keeps one column per bank when threads read across K.
  __shared__ uint32_t tile[MarkovTiledColumns][MarkovTiledChain / 2 + 1];
  __shared__ float lhs[MarkovTiledRows][MarkovTiledChain];
  const uint32_t live = *live_rows;
  const uint32_t first = blockIdx.x * MarkovTiledColumns;
  const uint32_t col = first + threadIdx.x;
  const bool pairs = (rank & 1) == 0 && (reinterpret_cast<uintptr_t>(weight) & 3) == 0;
  for (uint32_t r0 = 0; r0 < live; r0 += MarkovTiledRows) {
    const uint32_t rows = min(MarkovTiledRows, live - r0);
    float sum[MarkovTiledRows];
#pragma unroll
    for (uint32_t r = 0; r < MarkovTiledRows; ++r) sum[r] = 0.f;
    for (uint32_t start = 0; start < rank; start += MarkovTiledChain) {
      const uint32_t length = min(MarkovTiledChain, rank - start);
      __syncthreads();
      if (pairs) {
        const uint32_t words = length / 2;
        for (uint32_t i = threadIdx.x; i < MarkovTiledColumns * words; i += blockDim.x) {
          const uint32_t c = i / words, w = i % words;
          tile[c][w] = first + c < vocab
              ? reinterpret_cast<const uint32_t*>(weight + uint64_t(first + c) * rank + start)[w]
              : 0u;
        }
      } else {
        for (uint32_t i = threadIdx.x; i < MarkovTiledColumns * length; i += blockDim.x) {
          const uint32_t c = i / length, k = i % length;
          const uint32_t value = first + c < vocab ? weight[uint64_t(first + c) * rank + start + k] : 0u;
          uint16_t* words = reinterpret_cast<uint16_t*>(tile[c]);
          words[k] = uint16_t(value);
        }
      }
      for (uint32_t i = threadIdx.x; i < rows * length; i += blockDim.x) {
        const uint32_t r = i / length, k = i % length;
        lhs[r][k] = __uint_as_float(uint32_t(embedding[uint64_t(r0 + r) * rank + start + k]) << 16);
      }
      __syncthreads();
      if (col < vocab) {
        const uint16_t* column = reinterpret_cast<const uint16_t*>(tile[threadIdx.x]);
        float partial[MarkovTiledRows];
#pragma unroll
        for (uint32_t r = 0; r < MarkovTiledRows; ++r) partial[r] = 0.f;
        for (uint32_t k = 0; k < length; ++k) {
          const float w = __uint_as_float(uint32_t(column[k]) << 16);
#pragma unroll
          for (uint32_t r = 0; r < MarkovTiledRows; ++r)
            if (r < rows) partial[r] = __fmaf_rn(lhs[r][k], w, partial[r]);
        }
#pragma unroll
        for (uint32_t r = 0; r < MarkovTiledRows; ++r) sum[r] = __fadd_rn(sum[r], partial[r]);
      }
    }
    if (col < vocab) {
#pragma unroll
      for (uint32_t r = 0; r < MarkovTiledRows; ++r) {
        if (r < rows) {
          const uint64_t index = uint64_t(r0 + r) * stride + col;
          output[index] = __fadd_rn(base[index], sum[r]);
        }
      }
    }
  }
}

} // namespace chop::deepseek41::dspark
