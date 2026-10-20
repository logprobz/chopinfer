#pragma once
#include "round_expand_mix.cuh"

namespace chop::sinkhorn_hc::round_rows {
template<class M, int Rows> struct Arena {
  alignas(16) float weights[M::kMixes * kChain];
  alignas(16) uint16_t expanded[Rows * kChain];
};

// Reuse one weight chain across independent rows. Neither the four-term
// projection chains nor the original strided RMS lanes cross a row.
template<class M, int Rows>
__device__ void task(const uint16_t* x, const float* sum, uint16_t* rounded,
    const float* post, const float* comb, uint16_t* expanded, const void* fn,
    float* workspace, const uint32_t* live_rows, uint32_t chain,
    uint32_t row_tile, unsigned char* arena) {
  constexpr int WarpsPerRow = 4, Threads = Rows * WarpsPerRow * 32;
  static_assert(Rows == 2 || Rows == 4 || Rows == 8);
  auto& a = *reinterpret_cast<Arena<M, Rows>*>(arena);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int local_row = warp / WarpsPerRow, mix_warp = warp % WarpsPerRow;
  const uint32_t row = row_tile * Rows + local_row;
  const bool active = row < *live_rows;
  const int k = chain * kChain + lane * 4;
  for (int pair = t; pair < M::kMixes * kChain / 4; pair += Threads) {
    const int m = pair / (kChain / 4), col = pair % (kChain / 4) * 4;
    float w[4];
    load_fn<M::kF32Fn>(fn, uint64_t(m) * M::kFlat + chain * kChain + col, w);
    *reinterpret_cast<float4*>(a.weights + m * kChain + col) =
        make_float4(w[0], w[1], w[2], w[3]);
  }
  if (active && mix_warp == 0) {
    const uint32_t lo = round_mix::expand_mix_pair<M>(x, sum, post, comb, row, k);
    const uint32_t hi = round_mix::expand_mix_pair<M>(x, sum, post, comb, row, k + 2);
    *reinterpret_cast<uint2*>(a.expanded + local_row * kChain + lane * 4) =
        make_uint2(lo, hi);
    *reinterpret_cast<uint2*>(expanded + uint64_t(row) * M::kFlat + k) =
        make_uint2(lo, hi);
    if (k < M::kWidth) {
      const uint64_t at = uint64_t(row) * M::kWidth + k;
      *reinterpret_cast<uint2*>(rounded + at) = make_uint2(
          round_mix::rounded_pair(sum, at), round_mix::rounded_pair(sum, at + 2));
    }
  }
  // Inactive rows participate until all shared weights and operands publish.
  __syncthreads();
  if (!active) return;
  const uint2 packed = *reinterpret_cast<const uint2*>(
      a.expanded + local_row * kChain + lane * 4);
  const float values[4] = {
      bf_lo(packed.x), bf_hi(packed.x), bf_lo(packed.y), bf_hi(packed.y)};
  float* parts = workspace + uint64_t(row) * (M::kMixes * M::kChains + kThreads);
  #pragma unroll
  for (int m = mix_warp; m < M::kMixes; m += WarpsPerRow) {
    const float4 w = *reinterpret_cast<const float4*>(a.weights + m * kChain + lane * 4);
    float value = __fmul_rn(w.x, values[0]);
    value = fmaf(w.y, values[1], value);
    value = fmaf(w.z, values[2], value);
    value = fmaf(w.w, values[3], value);
    value = warp_sum(value);
    if (lane == 0) parts[m * M::kChains + chain] = value;
  }
  // Retain every original square FMA and its output slot. Recomputing the
  // expansion avoids reading another CTA's not-yet-published activation.
  if (mix_warp == 0 && chain < kWarps) {
    const int slot = chain * 32 + lane;
    float squares = 0.f;
    for (int v = slot; v < M::kFlat / 8; v += kThreads) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t pair = round_mix::expand_mix_pair<M>(x, sum, post, comb, row, 8*v+2*j);
        const float lo = bf_lo(pair), hi = bf_hi(pair);
        squares = fmaf(lo, lo, squares);
        squares = fmaf(hi, hi, squares);
      }
    }
    parts[M::kMixes * M::kChains + slot] = squares;
  }
}
} // namespace chop::sinkhorn_hc::round_rows
