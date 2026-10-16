#pragma once
#if !defined(CHOP_MIMO_HIDDEN) || !defined(CHOP_MIMO_EXPERT_INTERMEDIATE) || !defined(CHOP_MIMO_TOP_K) || !defined(CHOP_MIMO_EXPERTS)
#error "CHOP_MIMO_HIDDEN, CHOP_MIMO_EXPERT_INTERMEDIATE, CHOP_MIMO_TOP_K and CHOP_MIMO_EXPERTS are required"
#endif
#include "moe/moe.cuh"
namespace chop::moe {
// A CTA owns one token and one 16-column output strip. Four warps cover all
// routes, then one warp sums rounded, preweighted BF16 contributions in
// the row engine's fixed route order. This is an additional decode variant.
template<int TokensCapacity>
__device__ void run_fragment_down_combine(
    const uint16_t* activation, const uint8_t* weight_bytes,
    const uint8_t* weight_scales, const int32_t* topk_ids,
    const float* topk_weights, uint16_t* output,
    const int32_t* live_tokens) {
  static_assert(TokensCapacity > 0 && TokensCapacity <= 32); static_assert(CHOP_MIMO_TOP_K % 4 == 0);
  constexpr int N = CHOP_MIMO_HIDDEN, K = CHOP_MIMO_EXPERT_INTERMEDIATE, Ktiles = K / 64;
  constexpr int Strips = N / 16;
  __shared__ uint16_t route_values[CHOP_MIMO_TOP_K][16];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  __syncthreads();
  ptx::griddep_wait();
  const int token = blockIdx.x / Strips;
  if (token >= *live_tokens || token >= TokensCapacity) return;
  const int strip_in_expert = blockIdx.x % Strips;
  for (int turn = 0; turn < CHOP_MIMO_TOP_K / 4; ++turn) {
    const int route = warp + turn * 4;
    const int routed = token * CHOP_MIMO_TOP_K + route;
    const int expert = topk_ids[routed];
    if (expert < 0 || expert >= CHOP_MIMO_EXPERTS) {
      if ((lane & 3) == 0) {
        route_values[route][lane / 4] = 0;
        route_values[route][lane / 4 + 8] = 0;
      }
      continue;
    }
    const int strip = expert * Strips + strip_in_expert;
    uint4 fragments[Ktiles];
    uint32_t scale_words[Ktiles];
#pragma unroll
    for (int kt = 0; kt < Ktiles; ++kt) {
      fragments[kt] = load_fragment(weight_bytes
          + (uint64_t(strip) * Ktiles + kt) * 512 + lane * 16); // Bytes per K64 fragment tile.
      scale_words[kt] = *reinterpret_cast<const uint32_t*>(weight_scales
          + (uint64_t(strip) * Ktiles + kt) * 32 + (lane >> 2) * 4);
    }
    // The same scaled conversion and accumulation order as the grouped
    // one-token down kernel, so both paths agree bitwise.
    ptx::D4 accum{{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int kt = 0; kt < Ktiles; ++kt) {
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        const int block = step >> 1;
        const uint32_t word = fragment_word(fragments[kt], step);
        const uint32_t low = __byte_perm(scale_words[kt], 0, 0x22 * block);
        const uint32_t high = __byte_perm(scale_words[kt], 0, 0x22 * block + 0x11);
        ptx::A4 a;
        a.v[0] = dequant(word, 0, low);
        a.v[1] = dequant(word, 1, high);
        a.v[2] = dequant(word, 2, low);
        a.v[3] = dequant(word, 3, high);
        ptx::B2 b{{0u, 0u}};
        if (lane < 4) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int k = kt * 64 + step * 16 + lane * 2 + i * 8;
            const uint16_t* pair = activation + uint64_t(routed) * K + k;
            b.v[i] = *reinterpret_cast<const uint32_t*>(pair);
          }
        }
        ptx::mma_bf16(accum, a, b);
      }
    }
    if ((lane & 3) == 0) {
      const float weight = topk_weights[routed];
      route_values[route][lane / 4] =
          convert::bf16_from_f32(accum.v[0] * weight);
      route_values[route][lane / 4 + 8] =
          convert::bf16_from_f32(accum.v[2] * weight);
    }
  }
  __syncthreads();
  if (threadIdx.x < 16) {
    float sum = 0.f;
#pragma unroll
    for (int route = 0; route < CHOP_MIMO_TOP_K; ++route)
      sum += convert::bf16_to_f32(route_values[route][threadIdx.x]);
    output[uint64_t(token) * N + strip_in_expert * 16 + threadIdx.x] =
        convert::bf16_from_f32(sum);
  }
}

} // namespace chop::moe
