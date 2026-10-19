#pragma once
#include "breakdown.cuh"

namespace chop::sinkhorn_hc::finalizer_probe {
using breakdown::Arena;
// Both counters are sampled by one warp per CTA and published at the end.
// The timer gives phase microseconds without using an idle clock rate.
template<bool Sampled> struct Clock {
  uint64_t cycles[9], ns[9];
  bool enabled;
  __device__ explicit Clock(uint64_t* out) : enabled(Sampled && out != nullptr && threadIdx.x < 32) {}
  __device__ __forceinline__ void mark(int stage) {
    if constexpr (Sampled) if (enabled) {
      cycles[stage] = clock64();
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns[stage]));
    }
  }
  __device__ void end(uint64_t* out, uint32_t tile, uint32_t row, uint32_t groups,
      breakdown::Mode mode, bool last, bool finished) {
    if constexpr (Sampled) if (enabled) {
      if (!finished) for (int i = 4; i < 9; ++i) { cycles[i] = cycles[3]; ns[i] = ns[3]; }
      uint32_t sm;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      const uint64_t at = ((uint64_t(row) * groups + tile) * 32 + threadIdx.x) * 21;
      out[at] = sm; out[at+1] = last; out[at+2] = int(mode);
      #pragma unroll
      for (int i = 0; i < 9; ++i) { out[at+3+i] = cycles[i]; out[at+12+i] = ns[i]; }
    }
  }
};
// Kind 0 keeps shared-memory Sinkhorn, 1 uses warp shuffles, and 2
// converges the warp before each shuffle group. Arithmetic stays identical.
template<class M, int Kind, bool Sampled>
__device__ void finish(const uint16_t* x, const void* fn, const float* base,
                                    const float* scale, const float* pre_in,
                                    uint16_t* collapsed, float* pre, float* post, float* comb,
                                    const float* chain_partials, uint32_t row, unsigned char* arena, Clock<Sampled>& clock) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Stride = M::kChains + 1;
  const uint16_t* streams = x + uint64_t(row) * Flat;
  auto& shared = reinterpret_cast<Arena<M>*>(arena)->finish;
  auto& sums = shared.sums;
  auto& partials = shared.partials;
  auto& logits = shared.logits;
  auto& weights = shared.weights;
  auto& cells = shared.cells;
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

  clock.mark(4);

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
  }
  __syncthreads();

  clock.mark(5);

  // pre and post, stored in F32 for the next sublayer and the expand.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[uint64_t(row) * N + t] =
        __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[uint64_t(row) * N + t] = __fmul_rn(2.f, placed);
  }
  clock.mark(6);
  if constexpr (Kind == 0) {
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

  } else {
    if (warp == 0) {
      const int i = lane / N, j = lane % N;
      const bool owner = lane < Cells;
      float cell = owner ? fmaf(logits[2 * N + lane], scale[2], base[2 * N + lane]) : 0.f;
      if constexpr (Kind == 2) __syncwarp();
      float max = __shfl_sync(0xffffffff, cell, i * N);
      #pragma unroll
      for (int k = 1; k < N; ++k)
        max = fmaxf(max, __shfl_sync(0xffffffff, cell, i * N + k));
      cell = expf(__fadd_rn(cell, -max));
      if constexpr (Kind == 2) __syncwarp();
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k)
        sum = __fadd_rn(sum, __shfl_sync(0xffffffff, cell, i * N + k));
      cell = __fadd_rn(__fdiv_rn(cell, sum), eps);

      auto normalize = [&](bool rows) {
        if constexpr (Kind == 2) __syncwarp();
        float total = 0.f;
        #pragma unroll
        for (int k = 0; k < N; ++k)
          total = __fadd_rn(total, __shfl_sync(0xffffffff, cell, rows ? i * N + k : k * N + j));
        cell = __fdiv_rn(cell, __fadd_rn(total, eps));
      };
      normalize(false);
      for (int iteration = 1; iteration < M::kIterations; ++iteration) {
        normalize(true);
        normalize(false);
      }
      if (owner) comb[uint64_t(row) * Cells + lane] = cell;
    }
  }
  clock.mark(7);
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
  clock.mark(8);
}

template<class M, int Kind, bool Sampled>
__device__ void task(const uint16_t* x, const void* fn, const float* base,
    const float* scale, const float* pre_in, uint16_t* collapsed, float* pre,
    float* post, float* comb, float* partials, uint32_t* tickets,
    const uint32_t* live_rows, uint64_t* stamps, uint32_t tile, uint32_t row,
    unsigned char* arena, Clock<Sampled>& clock) {
  if (row >= *live_rows) return;
  constexpr int Groups = M::kChains / kWarps;
  float* parts = partials + uint64_t(row) * M::kMixes * M::kChains;
  chain_sums<M>(x + uint64_t(row) * M::kFlat, fn, tile * kWarps + (threadIdx.x >> 5), parts);
  clock.mark(2);
  const bool last = breakdown::arrive(tickets + row, Groups, reinterpret_cast<Arena<M>*>(arena)->last);
  clock.mark(3);
  if (last) finish<M,Kind,Sampled>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,parts,row,arena,clock);
  clock.end(stamps,tile,row,Groups,breakdown::Mode::full,last,last);
}
} // namespace chop::sinkhorn_hc::finalizer_probe
