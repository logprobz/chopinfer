#pragma once
#include "round_expand_mix.cuh"

namespace chop::sinkhorn_hc::round_chain {
// One CTA owns each 128-K chain; its warps divide mixes rather than K.
// Preserve the served mix-major workspace, so its finisher is unchanged.
template<class M, int Warps>
__device__ void task(const uint16_t* x, const float* sum, uint16_t* rounded,
    const float* post, const float* comb, uint16_t* expanded, const void* fn,
    float* workspace, const uint32_t* live_rows, uint32_t tile, uint32_t row,
    unsigned char* arena) {
  (void)arena;
  if (row >= *live_rows) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k = tile * kChain + lane * 4;
  if (warp == 0 && k < M::kWidth) {
    const uint64_t at = uint64_t(row) * M::kWidth + k;
    *reinterpret_cast<uint2*>(rounded + at) = make_uint2(
        round_mix::rounded_pair(sum,at),round_mix::rounded_pair(sum,at+2));
  }
  const uint32_t a = round_mix::expand_mix_pair<M>(x,sum,post,comb,row,k);
  const uint32_t b = round_mix::expand_mix_pair<M>(x,sum,post,comb,row,k+2);
  if (warp == 0)
    *reinterpret_cast<uint2*>(expanded + uint64_t(row) * M::kFlat + k) = make_uint2(a,b);
  const float xv[4] = {bf_lo(a),bf_hi(a),bf_lo(b),bf_hi(b)};
  float* parts = workspace + uint64_t(row) * (M::kMixes * M::kChains + kThreads);
  #pragma unroll
  for (int m = warp; m < M::kMixes; m += Warps) {
    float w[4];
    load_fn<M::kF32Fn>(fn,uint64_t(m) * M::kFlat + k,w);
    float value = __fmul_rn(w[0],xv[0]);
    value = fmaf(w[1],xv[1],value);
    value = fmaf(w[2],xv[2],value);
    value = fmaf(w[3],xv[3],value);
    value = warp_sum(value);
    if (lane == 0) parts[m * M::kChains + tile] = value;
  }
  // The first eight CTAs retain each original RMS lane and every FMA.
  // Recompute expansion from immutable inputs; no cross-CTA read of an
  // expanded value can precede its producer.
  if (warp == 0 && tile < kWarps) {
    const int slot = tile * 32 + lane;
    float squares = 0.f;
    for (int v = slot; v < M::kFlat / 8; v += kThreads) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t pair = round_mix::expand_mix_pair<M>(x,sum,post,comb,row,8*v+2*j);
        const float lo = bf_lo(pair), hi = bf_hi(pair);
        squares = fmaf(lo,lo,squares);
        squares = fmaf(hi,hi,squares);
      }
    }
    parts[M::kMixes * M::kChains + slot] = squares;
  }
}
} // namespace chop::sinkhorn_hc::round_chain
