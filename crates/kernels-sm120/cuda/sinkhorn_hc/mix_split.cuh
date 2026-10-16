#pragma once

// The Sinkhorn mix-collapse split over CTAs: bitwise sinkhorn_hc.cuh's
// one-CTA mix_collapse at every row count and grid. A row's 128-K chains
// spread over Split<M>::kGroups CTAs; each chain is computed exactly as
// mix_collapse's warps compute it (lane l FMAs k = 4 l .. 4 l + 3 of the
// chain from zero, the butterfly sums the lanes) and its sum is written to
// a per-row partials table. The CTA that takes the row's last ticket adds
// the chains into each mix from zero in K order, as mix_collapse's
// accumulators do, and finishes the row with mix_collapse's rstd, Sinkhorn
// and collapse arithmetic, copied expression for expression. The ticket
// only elects the finishing CTA: no float is added atomically, and no sum
// depends on which CTA finishes. See README.md ("Split mix-collapse").

#include <stdint.h>
#include <math.h>
#include "sinkhorn_hc.cuh"

namespace chop::sinkhorn_hc {

template<class M>
struct Split {
  // One chain per warp of a CTA.
  static constexpr int kChainsPerCta = kWarps;
  // CTAs per row.
  static constexpr int kGroups = M::kChains / kChainsPerCta;
  // F32 chain sums per row: [mixes, chains].
  static constexpr int kPartials = M::kMixes * M::kChains;
  // The finishing CTA's staged chain sums, one padding word per mix so
  // the mixes' serial sums read distinct banks.
  static constexpr int kStride = M::kChains + 1;
  static_assert(M::kChains % kChainsPerCta == 0, "whole chain groups");
  static_assert(M::kFlat * 2 + M::kMixes * kStride * 4 <= 46 * 1024,
                "streams and staged chain sums fit static shared memory");
};

// Warms L2 with the fn slices CTA group `group` reads: for every mix, the
// group's kChainsPerCta chains, in 128-byte lines.
template<class M>
__device__ __forceinline__ void prefetch_group(const void* fn, int group) {
  constexpr int ElementBytes = M::kF32Fn ? 4 : 2;
  constexpr int SliceBytes = Split<M>::kChainsPerCta * kChain * ElementBytes;
  constexpr int SliceLines = SliceBytes / 128;
  static_assert(SliceBytes % 128 == 0, "whole lines");
  const char* bytes = static_cast<const char*>(fn);
  for (int i = threadIdx.x; i < M::kMixes * SliceLines; i += kThreads) {
    const int m = i / SliceLines;
    const uint64_t at = (uint64_t(m) * M::kFlat + uint64_t(group) * Split<M>::kChainsPerCta * kChain) *
                            ElementBytes + uint64_t(i % SliceLines) * 128;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(bytes + at));
  }
}

// The chain sums of chain `chain` over `streams` (a row's N H BF16 values),
// every mix, written by lane 0 to partials [mixes, chains]: mix_collapse's
// per-chain lane products and butterfly.
template<class M>
__device__ __forceinline__ void chain_sums(const uint16_t* streams, const void* fn, int chain,
                                           float* partials) {
  const int lane = threadIdx.x & 31;
  const int k = chain * kChain + lane * 4;
  const uint2 xs = *reinterpret_cast<const uint2*>(streams + k);
  const float xv[4] = {bf_lo(xs.x), bf_hi(xs.x), bf_lo(xs.y), bf_hi(xs.y)};
  #pragma unroll
  for (int m = 0; m < M::kMixes; ++m) {
    float w[4];
    load_fn<M::kF32Fn>(fn, uint64_t(m) * M::kFlat + k, w);
    float sum = __fmul_rn(w[0], xv[0]);
    sum = fmaf(w[1], xv[1], sum);
    sum = fmaf(w[2], xv[2], sum);
    sum = fmaf(w[3], xv[3], sum);
    sum = warp_sum(sum);
    if (lane == 0) partials[m * M::kChains + chain] = sum;
  }
}

// Whether this CTA is the last of `ctas` to arrive at `ticket`, in every
// thread. The CTA's global writes are made visible first, so the last CTA
// reads every other CTA's writes. The ticket starts each launch at zero:
// it is zeroed at allocation, only this launch's CTAs of the row touch it,
// launches on one stream are serialized, and the last CTA resets it once
// every CTA has arrived. An arrival at or past `ctas` means corrupted
// state and traps.
__device__ __forceinline__ bool last_arrival(uint32_t* ticket, uint32_t ctas) {
  __shared__ uint32_t last;
  __threadfence();
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

// The rest of mix_collapse for one row, run by one CTA after every chain
// sum of the row is in `partials` ([mixes, chains], this row's): stage
// `x` (the row's N H streams) into `streams` with mix_collapse's sum of
// squares and rstd, add each mix's chains from zero in K order, then
// mix_collapse's pre, post, Sinkhorn and collapse. `collapsed`, `post` and
// `comb` point at the row.
template<class M>
__device__ void mix_finish(const uint16_t* x, const float* partials, const float* base,
                           const float* scale, uint16_t* collapsed, uint16_t* post,
                           uint16_t* comb, uint16_t* streams) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Chains = M::kChains;
  constexpr int Stride = Split<M>::kStride;
  __shared__ float partials_shared[Mixes * Stride];
  __shared__ float warp_partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float pre[N];
  __shared__ float cells[Cells];
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

  // pre, post and the comb logits.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[t] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[t] = to_bf(__fmul_rn(2.f, placed));
  }
  // Sinkhorn, one thread per cell (i, j) = (t / N, t % N).
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
  if (owner) comb[t] = to_bf(cell);

  // The collapse: per column an FMA chain from zero in stream order.
  for (int col = 2 * t; col < H; col += kColumns) {
    float lo = 0.f, hi = 0.f;
    #pragma unroll
    for (int s = 0; s < N; ++s) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + s * H + col);
      lo = fmaf(pre[s], bf_lo(pair), lo);
      hi = fmaf(pre[s], bf_hi(pair), hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed + col) = pack(lo, hi);
  }
}

// MixCollapse split over CTAs (grid: (Split<M>::kGroups, row capacity),
// 256 threads), mix_collapse's ABI plus partials [rows, mixes, chains] F32
// (rewritten before every read) and tickets [rows] u32 (zeroed at
// allocation, left zeroed by every launch). CTA (g, row) computes chains
// g * 8 + w (warp w) of every mix; the row's last CTA finishes it.
template<class M>
__device__ void mix_collapse_split(const uint16_t* x, const void* fn, const float* base,
                                   const float* scale, uint16_t* collapsed, uint16_t* post,
                                   uint16_t* comb, float* partials, uint32_t* tickets,
                                   const uint32_t* live_rows) {
  constexpr int Flat = M::kFlat;
  __shared__ __align__(16) uint16_t streams[Flat];
  if (blockDim.x != kThreads || gridDim.x != Split<M>::kGroups) asm volatile("trap;");
  const int group = blockIdx.x;

  ptx::griddep_launch_dependents();
  // fn is a weight: warm L2 with this CTA's slices before waiting on the
  // producer.
  prefetch_group<M>(fn, group);
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;

  const uint16_t* source = x + uint64_t(row) * Flat;
  float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
  chain_sums<M>(source, fn, group * Split<M>::kChainsPerCta + (threadIdx.x >> 5), row_partials);
  if (!last_arrival(tickets + row, Split<M>::kGroups)) return;
  mix_finish<M>(source, row_partials, base, scale, collapsed + uint64_t(row) * M::kWidth,
                post + uint64_t(row) * M::kStreams, comb + uint64_t(row) * M::kCells, streams);
}

} // namespace chop::sinkhorn_hc
