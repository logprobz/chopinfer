#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

constexpr uint32_t MarkovRowColumns = 64;
constexpr uint32_t MarkovRowThreads = 256;
constexpr uint32_t MarkovRowChain = 128;

// The dynamic shared bytes of one CTA at `rank`: the weight tile with an odd
// pad word per column, the embedding row and each chain's partial sums.
__host__ __device__ constexpr uint32_t markov_row_shared_bytes(uint32_t rank) {
  return (MarkovRowColumns * (rank / 2 + 1) + rank +
          MarkovRowColumns * (MarkovRowThreads / MarkovRowColumns)) * 4;
}

// One drafting row. A CTA owns 64 consecutive logits, whose weight rows are
// one contiguous block; it stages them with 16-byte loads before the
// dependency wait, since the weights are never written by a predecessor.
// One thread per (logit, 128-product chain) then runs `markov`'s chain over
// the staged tile: products in K order from +0. The chains fold left to
// right from +0, then base is added, as in `markov`, so every logit keeps
// its bits. Base is read after the projection, so base and output may alias.
__device__ void markov_row(const uint16_t* embedding, const uint16_t* weight,
                           const float* base, float* output,
                           const uint32_t* live_rows, uint32_t rank,
                           uint32_t vocab, uint32_t stride) {
  (void)stride;
  extern __shared__ uint32_t markov_row_shared[];
  const uint32_t pitch = rank / 2 + 1;
  uint32_t* tile = markov_row_shared;
  float* lhs = reinterpret_cast<float*>(tile + MarkovRowColumns * pitch);
  float* partials = lhs + rank;
  const uint32_t first = blockIdx.x * MarkovRowColumns;
  const uint32_t columns = first < vocab ? min(MarkovRowColumns, vocab - first) : 0;
  const uint16_t* block = weight + uint64_t(first) * rank;
  if ((reinterpret_cast<uintptr_t>(block) & 15) == 0 && rank % 8 == 0) {
    const uint32_t vectors = rank / 8;
    const uint4* source = reinterpret_cast<const uint4*>(block);
    for (uint32_t i = threadIdx.x; i < columns * vectors; i += blockDim.x) {
      const uint4 v = source[i];
      uint32_t* destination = tile + (i / vectors) * pitch + (i % vectors) * 4;
      destination[0] = v.x;
      destination[1] = v.y;
      destination[2] = v.z;
      destination[3] = v.w;
    }
  } else {
    for (uint32_t i = threadIdx.x; i < columns * rank; i += blockDim.x) {
      reinterpret_cast<uint16_t*>(tile + (i / rank) * pitch)[i % rank] = block[i];
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const bool live = *live_rows > 0;
  for (uint32_t k = threadIdx.x; live && k < rank; k += blockDim.x) {
    lhs[k] = __uint_as_float(uint32_t(embedding[k]) << 16);
  }
  __syncthreads();
  if (!live) return;
  const uint32_t chains = (rank + MarkovRowChain - 1) / MarkovRowChain;
  const uint32_t c = threadIdx.x % MarkovRowColumns;
  const uint32_t j = threadIdx.x / MarkovRowColumns;
  if (j < chains && c < columns) {
    const uint32_t start = j * MarkovRowChain;
    const uint32_t end = min(start + MarkovRowChain, rank);
    const uint16_t* column = reinterpret_cast<const uint16_t*>(tile + c * pitch);
    float partial = 0.f;
    for (uint32_t k = start; k < end; ++k) {
      partial = __fmaf_rn(lhs[k], __uint_as_float(uint32_t(column[k]) << 16), partial);
    }
    partials[j * MarkovRowColumns + c] = partial;
  }
  __syncthreads();
  if (threadIdx.x < columns) {
    float sum = 0.f;
    for (uint32_t chain = 0; chain < chains; ++chain) {
      sum = __fadd_rn(sum, partials[chain * MarkovRowColumns + threadIdx.x]);
    }
    const uint64_t index = first + threadIdx.x;
    output[index] = __fadd_rn(base[index], sum);
  }
}

} // namespace chop::deepseek41::dspark
