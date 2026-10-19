#pragma once

// Lagged mix-collapse with independent 128-K chains spread across CTAs.
// The last arrival folds chains in K order. Only tickets use atomics.
// The finisher reads the immutable input row directly, so the 40 KiB row
// and 15 KiB chain table need not coexist in static shared memory.
#include "mix_split.cuh"

namespace chop::sinkhorn_hc {

template<class M, bool Probe>
__device__ void mix_finish_lagged_split(const uint16_t* x, const void* fn, const float* base,
                                    const float* scale, const float* pre_in,
                                    uint16_t* collapsed, float* pre, float* post, float* comb,
                                    float* projected, float* rstd_out, float* cells_out,
                                    const float* chain_partials, uint32_t row) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Stride = M::kChains + 1;
  const uint16_t* streams = x + uint64_t(row) * Flat;
  __shared__ float sums[Mixes * Stride];
  __shared__ float partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float weights[N];
  __shared__ float cells[Cells];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  // The collapse weights the previous sublayer stored (or the one-hot
  // weights before the first), published by the barrier below.
  if (t < N) weights[t] = pre_in[uint64_t(row) * N + t];

  // Stage the row and sum its squares: mix_collapse's order.
  const uint4* source = reinterpret_cast<const uint4*>(x + uint64_t(row) * Flat);
  float squares = 0.f;
  for (int v = t; v < Flat / 8; v += kThreads) {
    const uint4 packed = __ldcg(source + v);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lo = bf_lo(words[j]);
      const float hi = bf_hi(words[j]);
      squares = fmaf(lo, lo, squares);
      squares = fmaf(hi, hi, squares);
    }
  }
  for (int i = t; i < Mixes * M::kChains; i += kThreads) {
    sums[(i / M::kChains) * Stride + i % M::kChains] = __ldcg(chain_partials + i);
  }
  squares = warp_sum(squares);
  if (lane == 0) partials[warp] = squares;
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, partials[w]);
  const float rstd =
      __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));

  // Fold each mix's chain sums in the original increasing K order.
  if (t < Mixes) {
    float acc = 0.f;
    #pragma unroll 8
    for (int step = 0; step < M::kChains; ++step) {
      #ifdef CHOP_HC_LAGGED_REVERSE_CHAINS
      const int c = M::kChains - 1 - step;
      #else
      const int c = step;
      #endif
      acc = __fadd_rn(acc, sums[t * Stride + c]);
    }
    logits[t] = __fmul_rn(acc, rstd);
    if constexpr (Probe) projected[uint64_t(row) * Mixes + t] = acc;
  }
  if constexpr (Probe) {
    if (t == 0) rstd_out[row] = rstd;
  }
  __syncthreads();

  // pre and post, stored in F32 for the next sublayer and the expand.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[uint64_t(row) * N + t] =
        __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[uint64_t(row) * N + t] = __fmul_rn(2.f, placed);
  }
  // Sinkhorn: mix_collapse's cells, sums and normalizations.
  const int i = t / N;
  const int j = t % N;
  const bool owner = t < Cells;
  float cell = owner ? fmaf(logits[2 * N + t], scale[2], base[2 * N + t]) : 0.f;
  if (owner) cells[t] = cell;
  __syncthreads();
  if (owner) {
    float max = cells[i * N];
    #pragma unroll
    for (int k = 1; k < N; ++k) max = fmaxf(max, cells[i * N + k]);
    cell = expf(__fadd_rn(cell, -max));
  }
  __syncthreads();
  if (owner) cells[t] = cell;
  __syncthreads();
  if (owner) {
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[i * N + k]);
    cell = __fadd_rn(__fdiv_rn(cell, sum), eps);
    if constexpr (Probe) cells_out[uint64_t(row) * Cells + t] = cell;
  }
  auto normalize = [&](bool rows) {
    __syncthreads();
    if (owner) cells[t] = cell;
    __syncthreads();
    if (owner) {
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[rows ? i * N + k : k * N + j]);
      cell = __fdiv_rn(cell, __fadd_rn(sum, eps));
    }
  };
  normalize(false);
  for (int iteration = 1; iteration < M::kIterations; ++iteration) {
    normalize(true);
    normalize(false);
  }
  if (owner) comb[uint64_t(row) * Cells + t] = cell;

  // The collapse with the given weights: per column an FMA chain from zero
  // in stream order.
  const uint64_t out = uint64_t(row) * H;
  for (int col = 2 * t; col < H; col += kColumns) {
    float lo = 0.f, hi = 0.f;
    #pragma unroll
    for (int s = 0; s < N; ++s) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + s * H + col);
      lo = fmaf(weights[s], bf_lo(pair), lo);
      hi = fmaf(weights[s], bf_hi(pair), hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed + out + col) = pack(lo, hi);
  }
}

template<class M, bool Probe>
__device__ void mix_collapse_lagged_split(
    const uint16_t* x, const void* fn, const float* base, const float* scale,
    const float* pre_in, uint16_t* collapsed, float* pre, float* post, float* comb,
    float* projected, float* rstd_out, float* cells_out,
    float* partials, uint32_t* tickets, const uint32_t* live_rows) {
  constexpr int Groups = M::kChains / kWarps;
  static_assert(M::kChains % kWarps == 0, "whole chain groups");
  static_assert(M::kMixes * (M::kChains + 1) * 4 <= 46 * 1024,
                "staged chain sums fit static shared memory");
  if (blockDim.x != kThreads || gridDim.x != Groups) asm volatile("trap;");
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint16_t* source = x + uint64_t(row) * M::kFlat;
  float* row_partials = partials + uint64_t(row) * M::kMixes * M::kChains;
  chain_sums<M>(source, fn, blockIdx.x * kWarps + (threadIdx.x >> 5), row_partials);
  if (!last_arrival(tickets + row, Groups)) return;
  mix_finish_lagged_split<M, Probe>(x, fn, base, scale, pre_in, collapsed, pre, post,
                                  comb, projected, rstd_out, cells_out, row_partials, row);
}

} // namespace chop::sinkhorn_hc
