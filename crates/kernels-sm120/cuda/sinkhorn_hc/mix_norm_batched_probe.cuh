#pragma once

// Lab only: mix_norm_batched.cuh's mix_collapse_norm_batched and its finish,
// copied statement for statement (served build: no test define), with
// mix_phase_probe.cuh's stamps, so the lab can break the entry's critical
// path into phases (a stamp's warp barrier, timer read and store may move
// the schedule slightly; the lab times the unprobed entry beside it). The
// outputs are the served entry's. No serving cubin includes this header.

#include <stdint.h>
#include <math.h>
#include "mix_norm_batched.cuh"
#include "mix_phase_probe.cuh"

namespace chop::sinkhorn_hc {

// mix_norm_batched_finish with phase stamps.
template<class M>
__device__ void mix_norm_batched_probed_finish(const uint16_t* x, const float* partials,
                                               const float* base, const float* scale,
                                               const uint16_t* weight, float eps,
                                               uint16_t* collapsed, uint16_t* post,
                                               uint16_t* comb, uint16_t* normed,
                                               uint16_t* streams, uint64_t* probe) {
  using B = MixNormBatched<M>;
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Chains = M::kChains;
  constexpr int Stride = Split<M>::kStride;
  constexpr int Workers = B::kWorkers;
  constexpr int PerWorker = B::kPairsPerWorker;
  // rmsnorm's group: the whole row.
  constexpr int Group = H;
  __shared__ float partials_shared[Mixes * Stride];
  __shared__ float warp_partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float pre[N];
  __shared__ float norm_inv;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  // Worker u of warps 1 to 7.
  const int u = t - 32;

  // Every load first: the row's streams and chain sums (written by the
  // producer and this launch's CTAs, read from L2 as mix_finish does), and
  // a worker's norm weight pairs (columns 2 (u + p Workers) and the next).
  const uint4* source = reinterpret_cast<const uint4*>(x);
  uint4 packed[B::kVectors];
  #pragma unroll
  for (int v = 0; v < B::kVectors; ++v) packed[v] = issue_load_cg(source + t + v * kThreads);
  float chain_sums[B::kChainSums];
  #pragma unroll
  for (int i = 0; i < B::kChainSums; ++i) {
    chain_sums[i] = issue_load_cg(partials + t + i * kThreads);
  }
  uint32_t weights[PerWorker];
  #pragma unroll
  for (int p = 0; p < PerWorker; ++p) {
    const int col = 2 * (u + p * Workers);
    weights[p] = u >= 0 && col < H ? *reinterpret_cast<const uint32_t*>(weight + col) : 0u;
  }

  // mix_finish's staging and sum of squares: thread t's vectors t + v
  // kThreads in v order.
  float squares = 0.f;
  #pragma unroll
  for (int v = 0; v < B::kVectors; ++v) {
    reinterpret_cast<uint4*>(streams)[t + v * kThreads] = packed[v];
    const uint32_t words[4] = {packed[v].x, packed[v].y, packed[v].z, packed[v].w};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float lo = bf_lo(words[k]);
      const float hi = bf_hi(words[k]);
      squares = fmaf(lo, lo, squares);
      squares = fmaf(hi, hi, squares);
    }
  }
  #pragma unroll
  for (int i = 0; i < B::kChainSums; ++i) {
    const int at = t + i * kThreads;
    partials_shared[(at / Chains) * Stride + at % Chains] = chain_sums[i];
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
    #pragma unroll 16
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
    sinkhorn_shuffled<M>(logits, base, scale, comb);
    phase_warp_time(probe, PhaseProbe::kSinkhorn);
    return;
  }

  // Warps 1 to 7: mix_norm_exact_finish's worker u.
  auto sync = [] { chop::row::workers_sync<B::kBarrier, Workers>(); };
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
  sync();
  if (u == 0) phase_time(probe, PhaseProbe::kSquares);

  // rmsnorm's ordered sum (the serial adds' bits) and reciprocal (Kind 3).
  auto& state = *reinterpret_cast<typename B::Sum::Shared*>(ordered_squares + H);
  const float ordered =
      chop::row::ordered_sum<H, Workers, B::kBarrier>(ordered_squares, u, state);
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

// mix_collapse_norm_batched with phase stamps into `stamps`
// (mix_phase_probe.cuh).
template<class M>
__device__ void mix_collapse_norm_batched_probed(const uint16_t* x, const void* fn,
                                                 const float* base, const float* scale,
                                                 const uint16_t* weight, uint16_t* collapsed,
                                                 uint16_t* post, uint16_t* comb, uint16_t* normed,
                                                 float* partials, uint32_t* tickets,
                                                 const uint32_t* live_rows, float eps,
                                                 uint64_t* stamps) {
  using B = MixNormBatched<M>;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Chains = M::kChains;
  __shared__ __align__(16) uint16_t streams[Flat];
  if (blockDim.x != kThreads || gridDim.x != Split<M>::kGroups) asm volatile("trap;");
  const int group = blockIdx.x;
  const uint32_t row = blockIdx.y;
  const int t = threadIdx.x;
  const int lane = t & 31;
  uint64_t* probe = phase_slots(stamps, row, group, Split<M>::kGroups);
  if (t == 0) phase_time(probe, PhaseProbe::kEntry);

  ptx::griddep_launch_dependents();
  // The norm weight is a weight: warm L2 with its 128-byte lines for
  // whichever CTA finishes.
  if (t < M::kWidth * 2 / 128) {
    asm volatile("prefetch.global.L2 [%0];"
                 ::"l"(reinterpret_cast<const char*>(weight) + t * 128));
  }
  // chain_sums' chain and lane elements: k = chain * 128 + 4 lane.
  const int chain = group * Split<M>::kChainsPerCta + (t >> 5);
  const int k = chain * kChain + lane * 4;
  const uint16_t* fn_values = static_cast<const uint16_t*>(fn);
  uint2 w[Mixes];
  // fn is a weight: when the live count, read here as a hint only, covers
  // the row, its values go to registers before the wait.
  const bool early = row < __ldcg(live_rows);
  if (early) {
    #pragma unroll
    for (int m = 0; m < Mixes; ++m) {
      w[m] = *reinterpret_cast<const uint2*>(fn_values + uint64_t(m) * Flat + k);
    }
  }
  ptx::griddep_wait();
  if (t == 0) {
    phase_time(probe, PhaseProbe::kWaited);
    phase_clock(probe, PhaseProbe::kWaitedClock);
  }
  // The lane's streams are fetched into L1 beside the live count's load
  // (rows past the live count are inside the bucket), so their load after
  // the branch on it does not wait a second L2 round trip.
  const uint16_t* lane_x = x + uint64_t(row) * Flat + k;
  asm volatile("prefetch.global.L1 [%0];" ::"l"(lane_x));
  const uint32_t live = *live_rows;
  if (row >= live) return;
  const uint2 xs = *reinterpret_cast<const uint2*>(lane_x);
  if (!early) {
    #pragma unroll
    for (int m = 0; m < Mixes; ++m) {
      w[m] = *reinterpret_cast<const uint2*>(fn_values + uint64_t(m) * Flat + k);
    }
  }

  // chain_sums' lane products of every mix, then the transposed butterfly.
  const float xv[4] = {bf_lo(xs.x), bf_hi(xs.x), bf_lo(xs.y), bf_hi(xs.y)};
  float sums[Mixes];
  #pragma unroll
  for (int m = 0; m < Mixes; ++m) {
    const float wv[4] = {bf_lo(w[m].x), bf_hi(w[m].x), bf_lo(w[m].y), bf_hi(w[m].y)};
    float sum = __fmul_rn(wv[0], xv[0]);
    sum = fmaf(wv[1], xv[1], sum);
    sum = fmaf(wv[2], xv[2], sum);
    sum = fmaf(wv[3], xv[3], sum);
    sums[m] = sum;
  }
  const int first = transposed_sum<Mixes, 16>(sums, lane);
  // The lanes holding the same mixes store them in turn: copy r stores
  // the sums r, r + kCopies, ...
  const int copy = packed_bits(lane, B::kCopiesMask);
  float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
  #pragma unroll
  for (int j = 0; j < B::kHeld; ++j) {
    if (j % B::kCopies == copy) row_partials[(first + j) * Chains + chain] = sums[j];
  }
  phase_warp_time(probe, PhaseProbe::kChainWarps + (t >> 5));
  if (!last_arrival_released(tickets + row, Split<M>::kGroups)) {
    if (t == 0) phase_time(probe, PhaseProbe::kTicket);
    return;
  }
  if (t == 0) phase_time(probe, PhaseProbe::kTicket);
  mix_norm_batched_probed_finish<M>(x + uint64_t(row) * Flat, row_partials, base, scale,
                                    weight, eps, collapsed + uint64_t(row) * M::kWidth,
                                    post + uint64_t(row) * M::kStreams,
                                    comb + uint64_t(row) * M::kCells,
                                    normed + uint64_t(row) * M::kWidth, streams, probe);
}

} // namespace chop::sinkhorn_hc
