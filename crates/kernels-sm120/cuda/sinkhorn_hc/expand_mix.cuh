#pragma once
#include "mix_split.cuh"

namespace chop::sinkhorn_hc {

// The same pair arithmetic and BF16 store boundary as expand_f32. A chain
// reads its expanded values directly, so projection needs no grid barrier.
template<class M>
__device__ __forceinline__ uint32_t expand_mix_pair(
    const uint16_t* x, const uint16_t* y, const float* post, const float* comb,
    uint32_t row, int flat) {
  constexpr int N = M::kStreams, H = M::kWidth;
  const int j = flat / H, col = flat % H;
  const uint64_t at = uint64_t(row) * N * H + col;
  float lo = 0.f, hi = 0.f;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint32_t pair = *reinterpret_cast<const uint32_t*>(x + at + uint64_t(i) * H);
    const float w = comb[uint64_t(row) * N * N + i * N + j];
    lo = fmaf(w, bf_lo(pair), lo);
    hi = fmaf(w, bf_hi(pair), hi);
  }
  const uint32_t ys = *reinterpret_cast<const uint32_t*>(y + uint64_t(row) * H + col);
  const float p = post[uint64_t(row) * N + j];
  return pack(__fadd_rn(__fmul_rn(p, bf_lo(ys)), lo),
              __fadd_rn(__fmul_rn(p, bf_hi(ys)), hi));
}

// One warp owns one complete projection chain. The first warp of each
// group also owns original RMS lanes, keeping each lane's FMA chain whole.
// Workspace is [row][mixes * chains + 256] F32, rewritten before use.
template<class M>
__device__ void expand_mix_partials(
    const uint16_t* x, const uint16_t* y, const float* post, const float* comb,
    uint16_t* expanded, const void* fn, float* workspace, const uint32_t* live_rows) {
  constexpr int Groups = M::kChains / kWarps;
  constexpr int Words = M::kMixes * M::kChains + kThreads;
  static_assert(M::kChains % kWarps == 0);
  if (blockDim.x != kThreads || gridDim.x != Groups) asm volatile("trap;");
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int chain = blockIdx.x * kWarps + warp;
  const int k = chain * kChain + lane * 4;
  const uint32_t a = expand_mix_pair<M>(x,y,post,comb,row,k);
  const uint32_t b = expand_mix_pair<M>(x,y,post,comb,row,k+2);
  *reinterpret_cast<uint2*>(expanded + uint64_t(row) * M::kFlat + k) = make_uint2(a,b);
  const float xv[4] = {bf_lo(a), bf_hi(a), bf_lo(b), bf_hi(b)};
  float* partials = workspace + uint64_t(row) * Words;
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
  if (warp == 0) {
    for (int slot = blockIdx.x * 32 + lane; slot < kThreads; slot += Groups * 32) {
      float squares = 0.f;
      for (int v = slot; v < M::kFlat / 8; v += kThreads) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t pair = expand_mix_pair<M>(x,y,post,comb,row,8*v+2*j);
          const float lo = bf_lo(pair), hi = bf_hi(pair);
          squares = fmaf(lo, lo, squares);
          squares = fmaf(hi, hi, squares);
        }
      }
      partials[M::kMixes * M::kChains + slot] = squares;
    }
  }
  // Every thread signals after its final store. The finalizer waits for
  // all producing CTAs, including the RMS lanes, before reading scratch.
  ptx::griddep_launch_dependents();
}

// Copied from the locked lagged split finisher. Only the RMS source,
// unique output owners and collapse column tile differ. All arithmetic
// expressions and the projection, butterfly and Sinkhorn orders are kept.
template<class M, bool Probe>
__device__ void expand_mix_finish_body(const uint16_t* x, const void* fn, const float* base,
                                    const float* scale, const float* pre_in,
                                    uint16_t* collapsed, float* pre, float* post, float* comb,
                                    float* projected, float* rstd_out, float* cells_out,
                                    const float* chain_partials, const float* lane_squares, uint32_t row) {
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

  // The partial stage preserves every original lane's sequential FMAs.
  float squares = __ldcg(lane_squares + t);
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
    if constexpr (Probe) if (blockIdx.x == 0) projected[uint64_t(row) * Mixes + t] = acc;
  }
  if constexpr (Probe) {
    if (t == 0 && blockIdx.x == 0) rstd_out[row] = rstd;
  }
  __syncthreads();

  // pre and post, stored in F32 for the next sublayer and the expand.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N && blockIdx.x == 0) {
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
    if constexpr (Probe) if (blockIdx.x == 0) cells_out[uint64_t(row) * Cells + t] = cell;
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
  if (owner && blockIdx.x == 0) comb[uint64_t(row) * Cells + t] = cell;

  // The collapse with the given weights: per column an FMA chain from zero
  // in stream order.
  const uint64_t out = uint64_t(row) * H;
  for (int col = blockIdx.x * kColumns + 2 * t;
       col < H && col < (blockIdx.x + 1) * kColumns; col += kColumns) {
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
__device__ void expand_mix_finish(
    const uint16_t* expanded, const float* base, const float* scale,
    const float* pre_in, uint16_t* collapsed, float* pre, float* post, float* comb,
    float* projected, float* rstd, float* cells, const float* workspace,
    const uint32_t* live_rows) {
  if (blockDim.x != kThreads || gridDim.x != (M::kWidth + kColumns - 1) / kColumns)
    asm volatile("trap;");
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  constexpr int Partials = M::kMixes * M::kChains;
  const float* row_partials = workspace + uint64_t(row) * (Partials + kThreads);
  expand_mix_finish_body<M,Probe>(expanded,nullptr,base,scale,pre_in,collapsed,pre,post,comb,
      projected,rstd,cells,row_partials,row_partials+Partials,row);
  ptx::griddep_launch_dependents();
}

} // namespace chop::sinkhorn_hc
