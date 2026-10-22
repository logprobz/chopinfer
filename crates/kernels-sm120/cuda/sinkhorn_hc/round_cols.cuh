#pragma once
#include "round_expand_mix.cuh"

// Column-block rounded producer for the decode HC boundary's two-launch
// form at wide buckets. CTA (b, t) owns hidden columns [128 b, 128 b + 128)
// of all four streams for rows [Rows t, Rows t + Rows). One read of a
// column's four input streams gives its four expanded values (each output
// stream with expand_mix_pair's expression), so the CTA writes the expanded
// streams and the rounded sum of its columns and holds the four 128-K
// chains they form (chain s * H / 128 + b of stream s). Each chain sum keeps
// round_chain's lane products and butterfly, written to the served
// workspace layout. With Transposed, a warp reduces 4 * Rows chain sums at
// once by halving: at each level a lane keeps half of its sums and adds its
// partner's copy of each, so every kept value is warp_sum's value at that
// lane and level and every stored sum has warp_sum's bits. The mixer's 256
// RMS lanes are left to the finisher, which reads them from the expanded
// streams (finish_split.cuh). fn is a weight, so every CTA loads its chains'
// values into registers before the dependency wait.
namespace chop::sinkhorn_hc::round_cols {
enum class Mutation { none, reverse_levels };

template<class M, int Rows> struct Arena {
  alignas(16) uint16_t expanded[Rows][M::kStreams][kChain];
};

// Warp w sums stream w % N's chain for mixes w / N, w / N + Groups, ...
template<class M> struct Layout {
  static constexpr int kGroups = kWarps / M::kStreams;
  static constexpr int kMixesPerWarp = M::kMixes / kGroups;
  static_assert(kWarps % M::kStreams == 0 && M::kMixes % kGroups == 0);
  static_assert(kMixesPerWarp % 4 == 0, "whole batches of four mixes");
};

template<class M>
__device__ __forceinline__ void load_weights(const void* fn, uint32_t block,
    float (&w)[Layout<M>::kMixesPerWarp][4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k = (warp % M::kStreams) * M::kWidth + block * kChain + lane * 4;
  #pragma unroll
  for (int i = 0; i < Layout<M>::kMixesPerWarp; ++i) {
    const int m = warp / M::kStreams + Layout<M>::kGroups * i;
    load_fn<M::kF32Fn>(fn, uint64_t(m) * M::kFlat + k, w[i]);
  }
}

// warp_sum, or (as a lab control) its levels in the opposite order.
template<Mutation Mut>
__device__ __forceinline__ float butterfly(float value) {
  if constexpr (Mut == Mutation::reverse_levels) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
      value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
  } else {
    return warp_sum(value);
  }
}

template<class M, int Rows, bool Transposed, Mutation Mut>
__device__ void task(const uint16_t* x, const float* sum, uint16_t* rounded,
    const float* post, const float* comb, uint16_t* expanded, float* workspace,
    const uint32_t* live_rows, uint32_t block, uint32_t tile,
    const float (&w)[Layout<M>::kMixesPerWarp][4], Arena<M,Rows>& a) {
  constexpr int N = M::kStreams, H = M::kWidth, Flat = M::kFlat;
  constexpr int Words = M::kMixes * M::kChains + kThreads;
  static_assert(N == 4 && Rows >= 1 && Rows <= 8);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint32_t live = *live_rows;
  const uint32_t first = tile * Rows;
  // Item (r, q): columns 128 b + 4 q .. + 3 of the tile's row r, every
  // output stream from the same four input streams.
  for (int item = t; item < Rows * 32; item += kThreads) {
    const int r = item / 32, q = item % 32;
    const uint32_t row = first + r;
    const int col = block * kChain + 4 * q;
    uint2 out[N];
    if (row < live) {
      // Every load precedes the stores, which the compiler cannot prove
      // disjoint from the coefficients.
      const uint64_t at = uint64_t(row) * Flat + col;
      uint2 xs[N];
      #pragma unroll
      for (int i = 0; i < N; ++i)
        xs[i] = *reinterpret_cast<const uint2*>(x + at + uint64_t(i) * H);
      float c[N * N], p[N];
      #pragma unroll
      for (int k = 0; k < N * N; ++k) c[k] = comb[uint64_t(row) * N * N + k];
      #pragma unroll
      for (int j = 0; j < N; ++j) p[j] = post[uint64_t(row) * N + j];
      const uint64_t ya = uint64_t(row) * H + col;
      const uint32_t y0 = round_mix::rounded_pair(sum, ya);
      const uint32_t y1 = round_mix::rounded_pair(sum, ya + 2);
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        float l0 = 0.f, h0 = 0.f, l1 = 0.f, h1 = 0.f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
          const float w = c[i * N + j];
          l0 = fmaf(w, bf_lo(xs[i].x), l0);
          h0 = fmaf(w, bf_hi(xs[i].x), h0);
          l1 = fmaf(w, bf_lo(xs[i].y), l1);
          h1 = fmaf(w, bf_hi(xs[i].y), h1);
        }
        out[j] = make_uint2(
            pack(__fadd_rn(__fmul_rn(p[j], bf_lo(y0)), l0), __fadd_rn(__fmul_rn(p[j], bf_hi(y0)), h0)),
            pack(__fadd_rn(__fmul_rn(p[j], bf_lo(y1)), l1), __fadd_rn(__fmul_rn(p[j], bf_hi(y1)), h1)));
      }
      #pragma unroll
      for (int j = 0; j < N; ++j)
        *reinterpret_cast<uint2*>(expanded + at + uint64_t(j) * H) = out[j];
      *reinterpret_cast<uint2*>(rounded + ya) = make_uint2(y0, y1);
    } else {
      #pragma unroll
      for (int j = 0; j < N; ++j) out[j] = make_uint2(0u, 0u);
    }
    #pragma unroll
    for (int j = 0; j < N; ++j)
      *reinterpret_cast<uint2*>(&a.expanded[r][j][4 * q]) = out[j];
  }
  __syncthreads();
  const int s = warp % N;
  const int chain = s * (H / kChain) + block;
  auto values = [&](int r, float (&v)[4]) {
    const uint2 e = *reinterpret_cast<const uint2*>(&a.expanded[r][s][4 * lane]);
    v[0] = bf_lo(e.x); v[1] = bf_hi(e.x); v[2] = bf_lo(e.y); v[3] = bf_hi(e.y);
  };
  // round_chain's four products from zero for mix slot i.
  auto product = [&](int i, const float (&v)[4]) {
    float value = __fmul_rn(w[i][0], v[0]);
    value = fmaf(w[i][1], v[1], value);
    value = fmaf(w[i][2], v[2], value);
    value = fmaf(w[i][3], v[3], value);
    return value;
  };
  auto store = [&](int i, int r, float value) {
    const uint32_t row = first + r;
    const int m = warp / N + Layout<M>::kGroups * i;
    if (row < live) workspace[uint64_t(row) * Words + m * M::kChains + chain] = value;
  };
  if constexpr (!Transposed) {
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
      float v[4];
      values(r, v);
      #pragma unroll
      for (int i = 0; i < Layout<M>::kMixesPerWarp; ++i) {
        const float value = butterfly<Mut>(product(i, v));
        if (lane == 0) store(i, r, value);
      }
    }
  } else {
    static_assert(Mut == Mutation::none, "controls use the plain butterfly");
    constexpr int Batch = 4 * Rows;
    static_assert(Batch == 8 || Batch == 16 || Batch == 32);
    constexpr int Levels = Batch == 8 ? 3 : Batch == 16 ? 4 : 5;
    #pragma unroll
    for (int g = 0; g < Layout<M>::kMixesPerWarp / 4; ++g) {
      // Sum index (mix slot 4 g + mi, row r) = mi * Rows + r.
      float sums[Batch];
      #pragma unroll
      for (int r = 0; r < Rows; ++r) {
        float v[4];
        values(r, v);
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) sums[mi * Rows + r] = product(4 * g + mi, v);
      }
      // Level offsets 16, 8, ...: a lane with the offset's bit keeps the
      // upper half of its sums, others the lower half, each adding its
      // partner's copy as warp_sum adds the partner's value.
      #pragma unroll
      for (int level = 0; level < Levels; ++level) {
        const int offset = 16 >> level;
        const int half = (Batch >> level) / 2;
        const bool upper = (lane & offset) != 0;
        #pragma unroll
        for (int k = 0; k < half; ++k) {
          const float send = upper ? sums[k] : sums[k + half];
          const float keep = upper ? sums[k + half] : sums[k];
          sums[k] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, offset));
        }
      }
      // The remaining levels as warp_sum; lanes of a group share the sum
      // with index lane >> (5 - Levels).
      float value = sums[0];
      #pragma unroll
      for (int offset = 16 >> Levels; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
      const int index = lane >> (5 - Levels);
      if ((lane & ((1 << (5 - Levels)) - 1)) == 0) store(4 * g + index / Rows, index % Rows, value);
    }
  }
}
} // namespace chop::sinkhorn_hc::round_cols
