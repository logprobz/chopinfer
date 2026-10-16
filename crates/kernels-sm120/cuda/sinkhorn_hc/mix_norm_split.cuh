#pragma once

// The split Sinkhorn mix-collapse fused with the sublayer's input RMSNorm:
// bitwise mix_split.cuh's mix_collapse_split followed by row.cuh's
// HF-ordered rmsnorm<H, 0, 3, false> over the collapsed row, at every row
// count. CTA (g, row) computes its chain sums and takes a ticket exactly as
// mix_collapse_split does. The row's last CTA finishes the mix-collapse with
// mix_finish's expressions in a different schedule, then normalizes the
// collapsed row:
//
//   1. stage, rstd, each mix's chains in K order, pre and post: mix_finish;
//   2. the collapse, kept in registers and written to `collapsed`;
//   3. each collapsed value's square __fmul_rn(v, v) into shared memory
//      over the dead streams;
//   4. at once: warp 0 runs mix_finish's Sinkhorn (its barriers become warp
//      barriers, as one warp owns every cell) and writes comb, while one
//      thread of warp 1 adds the squares from zero in column order (the
//      norm's serial sum) and computes the norm's 1 / sqrtf(s / H + eps);
//   5. every thread writes its columns' bf16(bf16(x * inv) * w).
//
// Every expression is mix_finish's or rmsnorm's; only the order in which
// independent values are computed changes. See README.md ("Split
// mix-collapse with the input norm").
//
// Test-only mutants, compiled into kernel-lab's mutant cubins: with
// CHOP_MIX_NORM_TEST_LATE_WAIT the dependency wait moves after the chain
// sums' reads of the streams and the live count; with
// CHOP_MIX_NORM_TEST_REVERSE_SUM the norm adds its squares in reverse
// column order. Serving cubins leave both undefined.

#include <stdint.h>
#include <math.h>
#include "mix_split.cuh"

namespace chop::sinkhorn_hc {

template<class M>
struct MixNorm {
  // Collapsed BF16 column pairs per thread: mix_collapse's columns
  // 2 t + p kColumns.
  static constexpr int kPairs = M::kWidth / kColumns;
  // The thread that adds the norm's squares: lane 0 of warp 1, beside the
  // Sinkhorn warp.
  static constexpr int kSumThread = 32;
  static_assert(M::kWidth % kColumns == 0, "whole collapse passes");
  static_assert(M::kCells <= 32, "one warp owns every Sinkhorn cell");
  static_assert(M::kWidth * 4 <= M::kFlat * 2, "the squares fit the staged streams");
};

// The rest of the row after its chain sums (mix_finish's inputs), then the
// input RMSNorm of the collapsed row with `weight` and `eps` into `normed`.
// `collapsed`, `post`, `comb` and `normed` point at the row.
template<class M>
__device__ void mix_norm_finish(const uint16_t* x, const float* partials, const float* base,
                                const float* scale, const uint16_t* weight, float eps,
                                uint16_t* collapsed, uint16_t* post, uint16_t* comb,
                                uint16_t* normed, uint16_t* streams) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Chains = M::kChains;
  constexpr int Stride = Split<M>::kStride;
  constexpr int Pairs = MixNorm<M>::kPairs;
  // rmsnorm's group: the whole row.
  constexpr int Group = H;
  __shared__ float partials_shared[Mixes * Stride];
  __shared__ float warp_partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float pre[N];
  __shared__ float cells[Cells];
  __shared__ float norm_inv;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  // mix_collapse's staging and sum of squares.
  const uint4* source = reinterpret_cast<const uint4*>(x);
  float squares = 0.f;
  for (int v = t; v < Flat / 8; v += kThreads) {
    const uint4 packed = __ldcg(source + v);
    reinterpret_cast<uint4*>(streams)[v] = packed;
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lo = bf_lo(words[j]);
      const float hi = bf_hi(words[j]);
      squares = fmaf(lo, lo, squares);
      squares = fmaf(hi, hi, squares);
    }
  }
  for (int i = t; i < Mixes * Chains; i += kThreads) {
    partials_shared[(i / Chains) * Stride + i % Chains] = __ldcg(partials + i);
  }
  squares = warp_sum(squares);
  if (lane == 0) warp_partials[warp] = squares;
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, warp_partials[w]);
  const float rstd =
      __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));

  // Each mix's accumulator: its chains from zero in K order.
  if (t < Mixes) {
    float acc = 0.f;
    #pragma unroll 8
    for (int c = 0; c < Chains; ++c) acc = __fadd_rn(acc, partials_shared[t * Stride + c]);
    logits[t] = __fmul_rn(acc, rstd);
  }
  __syncthreads();

  // pre and post.
  const float hc_eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[t] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), hc_eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[t] = to_bf(__fmul_rn(2.f, placed));
  }
  __syncthreads();

  // The collapse: per column an FMA chain from zero in stream order, kept
  // as packed BF16 pairs.
  uint32_t pairs[Pairs];
  #pragma unroll
  for (int p = 0; p < Pairs; ++p) {
    const int col = 2 * t + p * kColumns;
    float lo = 0.f, hi = 0.f;
    #pragma unroll
    for (int s = 0; s < N; ++s) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + s * H + col);
      lo = fmaf(pre[s], bf_lo(pair), lo);
      hi = fmaf(pre[s], bf_hi(pair), hi);
    }
    pairs[p] = pack(lo, hi);
    *reinterpret_cast<uint32_t*>(collapsed + col) = pairs[p];
  }
  // Every thread has read the streams; their shared memory now holds the
  // norm's squares, rmsnorm's ordered_squares.
  __syncthreads();
  float* ordered_squares = reinterpret_cast<float*>(streams);
  #pragma unroll
  for (int p = 0; p < Pairs; ++p) {
    const int col = 2 * t + p * kColumns;
    const float lo = bf(uint16_t(pairs[p]));
    const float hi = bf(uint16_t(pairs[p] >> 16));
    ordered_squares[col] = __fmul_rn(lo, lo);
    ordered_squares[col + 1] = __fmul_rn(hi, hi);
  }
  // The weight pairs of the thread's columns, read while the sums run.
  uint32_t weights[Pairs];
  #pragma unroll
  for (int p = 0; p < Pairs; ++p) {
    weights[p] = *reinterpret_cast<const uint32_t*>(weight + 2 * t + p * kColumns);
  }
  __syncthreads();

  if (warp == 0) {
    // Sinkhorn, one lane per cell (i, j) = (lane / N, lane % N): mix_finish's
    // with its block barriers as warp barriers.
    const int i = lane / N;
    const int j = lane % N;
    const bool owner = lane < Cells;
    float cell = owner ? fmaf(logits[2 * N + lane], scale[2], base[2 * N + lane]) : 0.f;
    if (owner) cells[lane] = cell;
    __syncwarp();
    if (owner) {
      float max = cells[i * N];
      #pragma unroll
      for (int k = 1; k < N; ++k) max = fmaxf(max, cells[i * N + k]);
      cell = expf(__fadd_rn(cell, -max));
    }
    __syncwarp();
    if (owner) cells[lane] = cell;
    __syncwarp();
    if (owner) {
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[i * N + k]);
      cell = __fadd_rn(__fdiv_rn(cell, sum), hc_eps);
    }
    auto normalize = [&](bool rows) {
      __syncwarp();
      if (owner) cells[lane] = cell;
      __syncwarp();
      if (owner) {
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[rows ? i * N + k : k * N + j]);
        cell = __fdiv_rn(cell, __fadd_rn(sum, hc_eps));
      }
    };
    normalize(false);
    for (int iteration = 1; iteration < M::kIterations; ++iteration) {
      normalize(true);
      normalize(false);
    }
    if (owner) comb[lane] = to_bf(cell);
  }
  else if (t == MixNorm<M>::kSumThread) {
    // rmsnorm's ordered sum and reciprocal (Kind 3).
    float ordered = 0.f;
#ifndef CHOP_MIX_NORM_TEST_REVERSE_SUM
    for (int c = 0; c < Group; ++c) {
      ordered = __fadd_rn(ordered, ordered_squares[c]);
    }
#else
    for (int c = Group - 1; c >= 0; --c) {
      ordered = __fadd_rn(ordered, ordered_squares[c]);
    }
#endif
    const float total = ordered;
    norm_inv = 1.f / sqrtf(total / Group + eps);
  }
  __syncthreads();

  // rmsnorm's output (Kind 3): the normalized value rounded to BF16, then
  // the weight.
  const float inv = norm_inv;
  #pragma unroll
  for (int p = 0; p < Pairs; ++p) {
    const int col = 2 * t + p * kColumns;
    const uint16_t in[2] = {uint16_t(pairs[p]), uint16_t(pairs[p] >> 16)};
    const uint16_t norm_weights[2] = {uint16_t(weights[p]), uint16_t(weights[p] >> 16)};
    uint16_t out[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float w = bf(norm_weights[k]);
      float n = bf(in[k]) * inv;
      n = bf(to_bf(n));
      out[k] = to_bf(n * w);
    }
    *reinterpret_cast<uint32_t*>(normed + col) = uint32_t(out[0]) | (uint32_t(out[1]) << 16);
  }
}

// MixCollapse and the input RMSNorm split over CTAs (grid:
// (Split<M>::kGroups, row capacity), 256 threads): mix_collapse_split's ABI
// with the norm's weight [H] BF16 after `scale`, `normed` [rows, H] BF16
// after `comb`, and its eps last. CTA (g, row) computes chains g * 8 + w
// (warp w) of every mix; the row's last CTA finishes the row and its norm.
template<class M>
__device__ void mix_collapse_norm_split(const uint16_t* x, const void* fn, const float* base,
                                        const float* scale, const uint16_t* weight,
                                        uint16_t* collapsed, uint16_t* post, uint16_t* comb,
                                        uint16_t* normed, float* partials, uint32_t* tickets,
                                        const uint32_t* live_rows, float eps) {
  constexpr int Flat = M::kFlat;
  __shared__ __align__(16) uint16_t streams[Flat];
  if (blockDim.x != kThreads || gridDim.x != Split<M>::kGroups) asm volatile("trap;");
  const int group = blockIdx.x;

  ptx::griddep_launch_dependents();
  // fn and the norm weight are weights: warm L2 with this CTA's fn slices
  // and the weight's 128-byte lines before waiting on the producer.
  prefetch_group<M>(fn, group);
  if (threadIdx.x < M::kWidth * 2 / 128) {
    asm volatile("prefetch.global.L2 [%0];"
                 ::"l"(reinterpret_cast<const char*>(weight) + threadIdx.x * 128));
  }
#ifndef CHOP_MIX_NORM_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;

  const uint16_t* source = x + uint64_t(row) * Flat;
  float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
  chain_sums<M>(source, fn, group * Split<M>::kChainsPerCta + (threadIdx.x >> 5), row_partials);
#ifdef CHOP_MIX_NORM_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (!last_arrival(tickets + row, Split<M>::kGroups)) return;
  mix_norm_finish<M>(source, row_partials, base, scale, weight, eps,
                     collapsed + uint64_t(row) * M::kWidth, post + uint64_t(row) * M::kStreams,
                     comb + uint64_t(row) * M::kCells, normed + uint64_t(row) * M::kWidth,
                     streams);
}

} // namespace chop::sinkhorn_hc
