#pragma once

#include <stdint.h>

namespace chop::deepseek41::dspark {

constexpr uint32_t MarkovStagedColumns = 64;
constexpr uint32_t MarkovStagedChain = 128;
constexpr uint32_t MarkovStagedChains = 2;
constexpr uint32_t MarkovStagedRank = MarkovStagedChain * MarkovStagedChains;
constexpr uint32_t MarkovStagedThreads = MarkovStagedColumns * MarkovStagedChains;
constexpr uint32_t MarkovStagedRows = 8;

// One thread per (logit, 128-product chain) of a CTA's 64 logits. Each thread
// loads its chain's 128 BF16 weights, one contiguous 256-byte run, into
// registers with 16-byte loads before the dependency wait, since no
// predecessor writes the weights. For each live row it then runs `markov`'s
// chain over them: products in K order from +0. A logit's chains fold left to
// right from +0, then base is added, as in `markov`, so every logit keeps its
// bits. Base rows are `base_stride` elements apart and output rows
// `output_stride` apart, so a position's base can be read in place from the
// item-major block logits. Base is read after the projection, so base and
// output may alias exactly.
__device__ void markov_staged(const uint16_t* embedding, const uint16_t* weight,
                              const float* base, float* output,
                              const uint32_t* live_rows, uint32_t vocab,
                              uint32_t base_stride, uint32_t output_stride) {
  __shared__ float lhs[MarkovStagedRows][MarkovStagedRank];
  __shared__ float partials[MarkovStagedRows][MarkovStagedChains][MarkovStagedColumns];
  const uint32_t c = threadIdx.x % MarkovStagedColumns;
  const uint32_t j = threadIdx.x / MarkovStagedColumns;
  const uint32_t col = blockIdx.x * MarkovStagedColumns + c;
  const bool owns = col < vocab;
  uint4 w[MarkovStagedChain / 8];
  if (owns) {
    const uint16_t* run = weight + uint64_t(col) * MarkovStagedRank + j * MarkovStagedChain;
    if ((reinterpret_cast<uintptr_t>(run) & 15) == 0) {
      const uint4* source = reinterpret_cast<const uint4*>(run);
#pragma unroll
      for (uint32_t i = 0; i < MarkovStagedChain / 8; ++i) w[i] = __ldg(source + i);
    } else {
#pragma unroll
      for (uint32_t i = 0; i < MarkovStagedChain / 8; ++i) {
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
  const uint32_t live = *live_rows;
  for (uint32_t r0 = 0; r0 < live; r0 += MarkovStagedRows) {
    const uint32_t rows = min(MarkovStagedRows, live - r0);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < rows * MarkovStagedRank; i += blockDim.x) {
      const uint32_t r = i / MarkovStagedRank, k = i % MarkovStagedRank;
      lhs[r][k] = __uint_as_float(
          uint32_t(embedding[uint64_t(r0 + r) * MarkovStagedRank + k]) << 16);
    }
    __syncthreads();
    if (owns) {
      for (uint32_t r = 0; r < rows; ++r) {
        const float* x = lhs[r] + j * MarkovStagedChain;
        float partial = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < MarkovStagedChain / 8; ++i) {
          const uint32_t words[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
          for (uint32_t h = 0; h < 4; ++h) {
            partial = __fmaf_rn(x[i * 8 + h * 2], __uint_as_float(words[h] << 16), partial);
            partial = __fmaf_rn(x[i * 8 + h * 2 + 1], __uint_as_float(words[h] & 0xffff0000u),
                                partial);
          }
        }
        partials[r][j][c] = partial;
      }
    }
    __syncthreads();
    if (owns && j == 0) {
      for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.f;
#pragma unroll
        for (uint32_t chain = 0; chain < MarkovStagedChains; ++chain) {
          sum = __fadd_rn(sum, partials[r][chain][c]);
        }
        const float b = base[uint64_t(r0 + r) * base_stride + col];
        output[uint64_t(r0 + r) * output_stride + col] = __fadd_rn(b, sum);
      }
    }
  }
}

} // namespace chop::deepseek41::dspark
