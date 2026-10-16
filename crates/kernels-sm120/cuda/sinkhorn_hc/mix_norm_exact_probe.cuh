#pragma once

// Lab only: mix_norm_exact.cuh's mix_collapse_norm_exact and its finish,
// copied statement for statement, with mix_phase_probe.cuh's stamps, so the
// lab can break down the served entry's critical path (a stamp's warp
// barrier, timer read and store may move the schedule slightly; the lab
// times the unprobed entry beside it). The outputs are the served entry's.
// No serving cubin includes this header.

#include <stdint.h>
#include <math.h>
#include "mix_norm_exact.cuh"
#include "mix_phase_probe.cuh"

namespace chop::sinkhorn_hc {

// The rest of the row after its chain sums (mix_finish's inputs), then the
// input RMSNorm of the collapsed row with `weight` and `eps` into `normed`.
// `collapsed`, `post`, `comb` and `normed` point at the row.
template<class M>
__device__ void mix_norm_exact_probed_finish(const uint16_t* x, const float* partials,
                                             const float* base, const float* scale,
                                             const uint16_t* weight, float eps,
                                             uint16_t* collapsed, uint16_t* post, uint16_t* comb,
                                             uint16_t* normed, uint16_t* streams,
                                             uint64_t* probe) {
  using E = MixNormExact<M>;
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Chains = M::kChains;
  constexpr int Stride = Split<M>::kStride;
  constexpr int Workers = E::kWorkers;
  constexpr int PerWorker = E::kPairsPerWorker;
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
  if (t == 0) phase_time(probe, PhaseProbe::kStaged);
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
  if (t == 0) {
    phase_time(probe, PhaseProbe::kLogits);
    phase_clock(probe, PhaseProbe::kLogitsClock);
  }

  const float hc_eps = __uint_as_float(M::kEpsBits);
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
    phase_warp_time(probe, PhaseProbe::kSinkhorn);
    return;
  }

  // Warps 1 to 7: worker u.
  const int u = t - 32;
  auto sync = [] { chop::row::workers_sync<E::kBarrier, Workers>(); };
  // pre and post.
  if (u < N) {
    pre[u] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[u], scale[0], base[u])), hc_eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + u], scale[1], base[N + u]));
    post[u] = to_bf(__fmul_rn(2.f, placed));
  }
  sync();
  if (u == 0) phase_time(probe, PhaseProbe::kPrePost);

  // The collapse: per column an FMA chain from zero in stream order, kept
  // as packed BF16 pairs (pair u + p Workers, columns 2 pair and 2 pair + 1).
  uint32_t pairs[PerWorker];
  #pragma unroll
  for (int p = 0; p < PerWorker; ++p) {
    const int col = 2 * (u + p * Workers);
    pairs[p] = 0u;
    if (col < H) {
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
  }
  // Every worker has read the streams; their shared memory now holds the
  // norm's squares, rmsnorm's ordered_squares, then the sum's state.
  sync();
  if (u == 0) phase_time(probe, PhaseProbe::kCollapse);
  float* ordered_squares = reinterpret_cast<float*>(streams);
  #pragma unroll
  for (int p = 0; p < PerWorker; ++p) {
    const int col = 2 * (u + p * Workers);
    if (col < H) {
      const float lo = bf(uint16_t(pairs[p]));
      const float hi = bf(uint16_t(pairs[p] >> 16));
      ordered_squares[col] = __fmul_rn(lo, lo);
      ordered_squares[col + 1] = __fmul_rn(hi, hi);
    }
  }
  // The weight pairs of the worker's columns, read while the sum runs.
  uint32_t weights[PerWorker];
  #pragma unroll
  for (int p = 0; p < PerWorker; ++p) {
    const int col = 2 * (u + p * Workers);
    weights[p] = col < H ? *reinterpret_cast<const uint32_t*>(weight + col) : 0u;
  }
  sync();
  if (u == 0) phase_time(probe, PhaseProbe::kSquares);

  // rmsnorm's ordered sum (the serial adds' bits) and reciprocal (Kind 3).
  auto& state = *reinterpret_cast<typename E::Sum::Shared*>(ordered_squares + H);
  const float ordered =
      chop::row::ordered_sum<H, Workers, E::kBarrier>(ordered_squares, u, state);
  if (u == 0) {
    phase_time(probe, PhaseProbe::kSum);
    const float total = ordered;
    norm_inv = 1.f / sqrtf(total / Group + eps);
  }
  sync();
  if (u == 0) phase_time(probe, PhaseProbe::kInv);

  // rmsnorm's output (Kind 3): the normalized value rounded to BF16, then
  // the weight.
  const float inv = norm_inv;
  #pragma unroll
  for (int p = 0; p < PerWorker; ++p) {
    const int col = 2 * (u + p * Workers);
    if (col < H) {
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
  phase_warp_time(probe, PhaseProbe::kOutputWarps + warp - 1);
}

// MixCollapse and the input RMSNorm split over CTAs (grid:
// (Split<M>::kGroups, row capacity), 256 threads), with
// mix_collapse_norm_split's ABI: the norm's weight [H] BF16 after `scale`,
// `normed` [rows, H] BF16 after `comb`, and its eps last. CTA (g, row)
// computes chains g * 8 + w (warp w) of every mix; the row's last CTA
// finishes the row and its norm.
template<class M>
__device__ void mix_collapse_norm_exact_probed(const uint16_t* x, const void* fn, const float* base,
                                               const float* scale, const uint16_t* weight,
                                               uint16_t* collapsed, uint16_t* post, uint16_t* comb,
                                               uint16_t* normed, float* partials,
                                               uint32_t* tickets, const uint32_t* live_rows,
                                               float eps, uint64_t* stamps) {
  constexpr int Flat = M::kFlat;
  __shared__ __align__(16) uint16_t streams[Flat];
  if (blockDim.x != kThreads || gridDim.x != Split<M>::kGroups) asm volatile("trap;");
  const int group = blockIdx.x;
  uint64_t* probe = phase_slots(stamps, blockIdx.y, group, Split<M>::kGroups);
  if (threadIdx.x == 0) phase_time(probe, PhaseProbe::kEntry);

  ptx::griddep_launch_dependents();
  // fn and the norm weight are weights: warm L2 with this CTA's fn slices
  // and the weight's 128-byte lines before waiting on the producer.
  prefetch_group<M>(fn, group);
  if (threadIdx.x < M::kWidth * 2 / 128) {
    asm volatile("prefetch.global.L2 [%0];"
                 ::"l"(reinterpret_cast<const char*>(weight) + threadIdx.x * 128));
  }
  ptx::griddep_wait();
  if (threadIdx.x == 0) {
    phase_time(probe, PhaseProbe::kWaited);
    phase_clock(probe, PhaseProbe::kWaitedClock);
  }
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;

  const uint16_t* source = x + uint64_t(row) * Flat;
  float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
  chain_sums<M>(source, fn, group * Split<M>::kChainsPerCta + (threadIdx.x >> 5), row_partials);
  phase_warp_time(probe, PhaseProbe::kChainWarps + (threadIdx.x >> 5));
  const bool finishes = last_arrival(tickets + row, Split<M>::kGroups);
  if (threadIdx.x == 0) phase_time(probe, PhaseProbe::kTicket);
  if (!finishes) return;
  mix_norm_exact_probed_finish<M>(source, row_partials, base, scale, weight, eps,
                                  collapsed + uint64_t(row) * M::kWidth,
                                  post + uint64_t(row) * M::kStreams,
                                  comb + uint64_t(row) * M::kCells,
                                  normed + uint64_t(row) * M::kWidth, streams, probe);
}

} // namespace chop::sinkhorn_hc
