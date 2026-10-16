#pragma once

// The split Sinkhorn mix-collapse fused with the sublayer's input RMSNorm,
// bitwise mix_norm_exact.cuh's mix_collapse_norm_exact at every row count,
// with the same grid, ABI and per-element operations on a shorter critical
// path. mix_collapse_norm_exact spends most of a boundary waiting on
// memory one access at a time: each chain lane loads a mix's fn values
// only after storing the previous mix's chain sum (24 dependent L2 round
// trips), the finishing CTA stages the streams and chain sums one load per
// loop iteration (20 more), and every thread fences before the ticket.
// Here:
//
//   1. a CTA whose row the live count covers loads its lanes' fn values
//      (immutable weights) into registers before the dependency wait; the
//      live count read for this is a hint only (a stale hint loads them
//      after the wait instead), and the count read after the wait decides,
//      as before, whether the row runs;
//   2. each lane's 24 chain products are mix_collapse's (from zero, k = 4
//      lane .. 4 lane + 3); their butterfly runs transposed: at a level
//      whose count of sums is even, the lanes with the level's bit set keep
//      the upper half of the sums and the others the lower half, each adding
//      its own value and its partner's as warp_sum does, so every kept sum
//      is warp_sum's at that lane and level (27 shuffles instead of 120),
//      and every lane ends with warp_sum's bits for the mixes it holds;
//   3. the ticket is mix_split.cuh's, with the CTA's chain sums released by
//      thread 0's acq_rel atomic after the block barrier (the barrier
//      orders the other threads' stores before it) instead of a fence in
//      every thread;
//   4. the finishing CTA issues every load of the streams, chain sums and
//      norm weights before using any, then stages them with mix_finish's
//      sum of squares in mix_finish's order;
//   5. warp 0's Sinkhorn exchanges cells by shuffles instead of shared
//      memory and warp barriers; every lane runs a cell's arithmetic (lane
//      l the cell l mod N^2) so no lane branches, and lanes below N^2 write
//      comb.
//
// The chain sums, rstd, logits, pre, post, Sinkhorn, collapse, squares,
// ordered sum (row/ordered_sum.cuh), reciprocal and normed row are
// mix_norm_exact's expressions in its orders. See README.md ("Split
// mix-collapse with batched loads").
//
// Test-only mutants, compiled into kernel-lab's mutant cubins: with
// CHOP_MIX_NORM_BATCHED_TEST_LATE_WAIT the dependency wait moves after the
// chain sums' reads of the streams and the live count; with
// CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED the transposed butterfly runs
// its levels from distance 1 up; ordered_sum.cuh's
// CHOP_ORDERED_SUM_TEST_TIES_AWAY rounds the norm sum's ties away from
// zero. Serving cubins leave all three undefined.

#include <stdint.h>
#include <math.h>
#include "mix_split.cuh"
#include "mix_probe.cuh"
#include "../row/ordered_sum.cuh"

namespace chop::sinkhorn_hc {

// Sums a lane holds after the transposed butterfly's levels Distance,
// Distance / 2, .., 1 from Count sums: each level with an even count
// halves it.
__host__ __device__ constexpr int transposed_held(int count, int distance) {
  return distance == 0 ? count
                       : transposed_held(count % 2 == 0 ? count / 2 : count, distance / 2);
}

// The lane bits of the levels that keep every sum (odd counts): lanes that
// differ only in these bits hold the same mixes.
__host__ __device__ constexpr int transposed_copies_mask(int count, int distance) {
  return distance == 0 ? 0
         : count % 2 == 0
             ? transposed_copies_mask(count / 2, distance / 2)
             : distance | transposed_copies_mask(count, distance / 2);
}

__host__ __device__ constexpr int bit_count(int mask) {
  return mask == 0 ? 0 : (mask & 1) + bit_count(mask >> 1);
}

// The bits of `lane` under `mask`, packed from bit 0 up.
__device__ __forceinline__ int packed_bits(int lane, int mask) {
  int packed = 0;
  int at = 0;
  #pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
    if (mask & bit) {
      packed |= ((lane & bit) ? 1 : 0) << at;
      ++at;
    }
  }
  return packed;
}

// warp_sum's level `Distance` of each of `Count` sums, transposed when Count
// is even (see the header comment), then the levels below it. Returns the
// first mix of the lane's sums v[0, transposed_held(Count, Distance)).
template<int Count, int Distance>
__device__ __forceinline__ int transposed_sum(float* v, int lane) {
  if constexpr (Distance == 0) {
    return 0;
  } else if constexpr (Count % 2 == 0) {
    constexpr int Half = Count / 2;
    const bool upper = (lane & Distance) != 0;
    #pragma unroll
    for (int j = 0; j < Half; ++j) {
      const float keep = upper ? v[Half + j] : v[j];
      const float send = upper ? v[j] : v[Half + j];
      v[j] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, Distance));
    }
    return (upper ? Half : 0) + transposed_sum<Half, Distance / 2>(v, lane);
  } else {
    #pragma unroll
    for (int j = 0; j < Count; ++j) {
      v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffffu, v[j], Distance));
    }
    return transposed_sum<Count, Distance / 2>(v, lane);
  }
}

// Loads the compiler keeps in program order (volatile), so a batch of them
// is issued before any is used: a 16-byte vector and an F32 at L2
// (mix_finish's __ldcg).
__device__ __forceinline__ uint4 issue_load_cg(const uint4* at) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(at));
  return v;
}
__device__ __forceinline__ float issue_load_cg(const float* at) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(at));
  return v;
}

#ifdef CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED
// The mutant's butterfly: warp_sum's levels from distance 1 up, every sum
// kept.
template<int Count>
__device__ __forceinline__ int reversed_levels_sum(float* v) {
  #pragma unroll
  for (int distance = 1; distance < 32; distance <<= 1) {
    #pragma unroll
    for (int j = 0; j < Count; ++j) {
      v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffffu, v[j], distance));
    }
  }
  return 0;
}
#endif

template<class M>
struct MixNormBatched {
  // Warps 1 to 7 finish the row beside the Sinkhorn warp.
  static constexpr int kWorkers = kThreads - 32;
  static constexpr int kBarrier = 1;
  // Collapsed BF16 column pairs, and per worker.
  static constexpr int kPairs = M::kWidth / 2;
  static constexpr int kPairsPerWorker = (kPairs + kWorkers - 1) / kWorkers;
  // The finishing CTA's loads per thread: 16-byte stream vectors and chain
  // sums.
  static constexpr int kVectors = M::kFlat / 8 / kThreads;
  static constexpr int kChainSums = M::kMixes * M::kChains / kThreads;
#ifndef CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED
  // The chain sums a lane holds after the transposed butterfly, and the
  // lane bits that tell its copies apart.
  static constexpr int kHeld = transposed_held(M::kMixes, 16);
  static constexpr int kCopiesMask = transposed_copies_mask(M::kMixes, 16);
#else
  static constexpr int kHeld = M::kMixes;
  static constexpr int kCopiesMask = 31;
#endif
  static constexpr int kCopies = 1 << bit_count(kCopiesMask);
  using Sum = chop::row::OrderedSum<M::kWidth, kWorkers>;
  static_assert(!M::kF32Fn, "BF16 fn, held in registers as packed pairs");
  static_assert(M::kFlat % (8 * kThreads) == 0, "whole stream vectors per thread");
  static_assert(M::kMixes * M::kChains % kThreads == 0, "whole chain sums per thread");
  static_assert(M::kWidth % 2 == 0, "whole column pairs");
  static_assert(M::kCells <= 32, "one warp owns every Sinkhorn cell");
  static_assert(M::kStreams <= kWorkers, "one worker per stream's pre and post");
  static_assert(M::kWidth * 4 + sizeof(typename Sum::Shared) <= M::kFlat * 2,
                "the squares and the sum's state fit the staged streams");
};

// mix_finish's Sinkhorn of one row in one warp, cells exchanged by
// shuffles: lane l runs cell c = l mod N^2, (i, j) = (c / N, c % N), and
// reads its row's or column's cells from the lanes that run them, in
// mix_finish's index order. Lanes below N^2 write comb [N, N] BF16.
template<class M>
__device__ __forceinline__ void sinkhorn_shuffled(const float* logits, const float* base,
                                                  const float* scale, uint16_t* comb) {
  constexpr int N = M::kStreams;
  constexpr int Cells = M::kCells;
  const int lane = threadIdx.x & 31;
  const int c = lane % Cells;
  const int i = c / N;
  const int j = c % N;
  const float eps = __uint_as_float(M::kEpsBits);
  const auto cell_at = [](float value, int source) {
    return __shfl_sync(0xffffffffu, value, source);
  };
  float cell = fmaf(logits[2 * N + c], scale[2], base[2 * N + c]);
  {
    float max = cell_at(cell, i * N);
    #pragma unroll
    for (int k = 1; k < N; ++k) max = fmaxf(max, cell_at(cell, i * N + k));
    cell = expf(__fadd_rn(cell, -max));
  }
  {
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cell_at(cell, i * N + k));
    cell = __fadd_rn(__fdiv_rn(cell, sum), eps);
  }
  const auto normalize = [&](bool rows) {
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cell_at(cell, rows ? i * N + k : k * N + j));
    cell = __fdiv_rn(cell, __fadd_rn(sum, eps));
  };
  normalize(false);
  for (int iteration = 1; iteration < M::kIterations; ++iteration) {
    normalize(true);
    normalize(false);
  }
  if (lane < Cells) comb[lane] = to_bf(cell);
}

// The rest of the row after its chain sums (mix_norm_exact_finish's
// inputs and outputs), every load issued first.
template<class M, bool Probe>
__device__ void mix_norm_batched_finish(const uint16_t* x, const float* partials,
                                        const float* base, const float* scale,
                                        const uint16_t* weight, float eps, uint16_t* collapsed,
                                        uint16_t* post, uint16_t* comb, uint16_t* normed,
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
  if (t == 0) probe_time<Probe>(probe, MixProbe::kStaged);
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
    probe_time<Probe>(probe, MixProbe::kLogits);
    probe_clock<Probe>(probe, MixProbe::kLogitsClock);
  }

  const float hc_eps = __uint_as_float(M::kEpsBits);
  if (warp == 0) {
    sinkhorn_shuffled<M>(logits, base, scale, comb);
    if (t == 0) probe_time<Probe>(probe, MixProbe::kSinkhorn);
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
  if (u == 0) probe_time<Probe>(probe, MixProbe::kPrePost);

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
  if (u == 0) probe_time<Probe>(probe, MixProbe::kCollapse);
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
  if (u == 0) probe_time<Probe>(probe, MixProbe::kSquares);

  // rmsnorm's ordered sum (the serial adds' bits) and reciprocal (Kind 3).
  auto& state = *reinterpret_cast<typename B::Sum::Shared*>(ordered_squares + H);
  const float ordered =
      chop::row::ordered_sum<H, Workers, B::kBarrier>(ordered_squares, u, state);
  if (u == 0) {
    probe_time<Probe>(probe, MixProbe::kSum);
    const float total = ordered;
    norm_inv = 1.f / sqrtf(total / Group + eps);
  }
  sync();
  if (u == 0) probe_time<Probe>(probe, MixProbe::kInv);

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
  if (u == 0) probe_time<Probe>(probe, MixProbe::kOutput);
}

// Whether this CTA is the last of `ctas` to arrive at `ticket`, in every
// thread: mix_split.cuh's last_arrival with one release. The block barrier
// orders every thread's chain-sum stores before thread 0's acq_rel atomic,
// whose release makes them visible to the CTA that acquires the last
// ticket (the pattern of a grid barrier's arrival); that CTA's barrier then
// orders its threads' reads after the acquire.
__device__ __forceinline__ bool last_arrival_released(uint32_t* ticket, uint32_t ctas) {
  __shared__ uint32_t last;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
                 : "=r"(arrived) : "l"(ticket) : "memory");
    if (arrived >= ctas) asm volatile("trap;");
    last = arrived == ctas - 1;
    if (last) *ticket = 0;
  }
  __syncthreads();
  return last;
}

// MixCollapse and the input RMSNorm split over CTAs (grid:
// (Split<M>::kGroups, row capacity), 256 threads), with
// mix_collapse_norm_exact's ABI; a probe instantiation also takes `stamps`
// (mix_probe.cuh), which serving instantiations ignore. CTA (g, row)
// computes chains g * 8 + w (warp w) of every mix; the row's last CTA
// finishes the row and its norm.
template<class M, bool Probe>
__device__ void mix_collapse_norm_batched(const uint16_t* x, const void* fn, const float* base,
                                          const float* scale, const uint16_t* weight,
                                          uint16_t* collapsed, uint16_t* post, uint16_t* comb,
                                          uint16_t* normed, float* partials, uint32_t* tickets,
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
  uint64_t* probe = probe_slots<Probe>(stamps, row, group, Split<M>::kGroups);
  if (t == 0) probe_time<Probe>(probe, MixProbe::kEntry);

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
#ifndef CHOP_MIX_NORM_BATCHED_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (t == 0) {
    probe_time<Probe>(probe, MixProbe::kWaited);
    probe_clock<Probe>(probe, MixProbe::kWaitedClock);
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
#ifndef CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED
  const int first = transposed_sum<Mixes, 16>(sums, lane);
#else
  const int first = reversed_levels_sum<Mixes>(sums);
#endif
  // The lanes holding the same mixes store them in turn: copy r stores
  // the sums r, r + kCopies, ...
  const int copy = packed_bits(lane, B::kCopiesMask);
  float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
  #pragma unroll
  for (int j = 0; j < B::kHeld; ++j) {
    if (j % B::kCopies == copy) row_partials[(first + j) * Chains + chain] = sums[j];
  }
#ifdef CHOP_MIX_NORM_BATCHED_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (t == 0) probe_time<Probe>(probe, MixProbe::kChains);
  if (!last_arrival_released(tickets + row, Split<M>::kGroups)) {
    if (t == 0) probe_time<Probe>(probe, MixProbe::kTicket);
    return;
  }
  if (t == 0) probe_time<Probe>(probe, MixProbe::kTicket);
  mix_norm_batched_finish<M, Probe>(x + uint64_t(row) * Flat, row_partials, base, scale, weight,
                                    eps, collapsed + uint64_t(row) * M::kWidth,
                                    post + uint64_t(row) * M::kStreams,
                                    comb + uint64_t(row) * M::kCells,
                                    normed + uint64_t(row) * M::kWidth, streams, probe);
}

} // namespace chop::sinkhorn_hc
