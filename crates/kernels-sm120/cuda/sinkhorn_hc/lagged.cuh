#pragma once

// Lagged Sinkhorn-mixed hyper-connections (kernel-core
// sinkhorn_hyper_connection, HyperConnectionPlacement::LaggedSinglePass):
// the lagged mix-collapse, the F32 expand and the sigmoid-weighted head.
// See README.md, "Lagged placement". The projection, rstd, pre, post and
// Sinkhorn are mix_collapse's expressions in its orders; only the collapse
// weights (the previous sublayer's pre, read from global memory) and the
// F32 stores differ. sinkhorn_hc.cuh is included unchanged.

#include "sinkhorn_hc.cuh"

namespace chop::sinkhorn_hc {

// Lagged MixCollapse, one row per CTA (grid: row capacity), 256 threads:
//   x [rows, N, H] BF16, fn [(2 + N) N, N H], base [(2 + N) N] F32,
//   scale [3] F32, pre_in [rows, N] F32 -> collapsed [rows, H] BF16,
//   pre [rows, N] F32, post [rows, N] F32, comb [rows, N, N] F32.
// pre_in and pre do not overlap. With Probe, also mix_collapse's raw
// projection, rstd and Sinkhorn input cells.
template<class M, bool Probe>
__device__ void mix_collapse_lagged(const uint16_t* x, const void* fn, const float* base,
                                    const float* scale, const float* pre_in,
                                    uint16_t* collapsed, float* pre, float* post, float* comb,
                                    float* projected, float* rstd_out, float* cells_out,
                                    const uint32_t* live_rows) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int ElementBytes = M::kF32Fn ? 4 : 2;
  __shared__ __align__(16) uint16_t streams[Flat];
  __shared__ float partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float weights[N];
  __shared__ float cells[Cells];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  ptx::griddep_launch_dependents();
  // fn is a weight: warm L2 with it before waiting on the producer, as
  // mix_collapse does.
  {
    constexpr uint64_t Lines = (uint64_t(Mixes) * Flat * ElementBytes + 127) / 128;
    const uint64_t line = (uint64_t(blockIdx.x) * kThreads + t) % Lines;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<const char*>(fn) + line * 128));
  }
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;

  // The collapse weights the previous sublayer stored (or the one-hot
  // weights before the first), published by the barrier below.
  if (t < N) weights[t] = pre_in[uint64_t(row) * N + t];

  // Stage the row and sum its squares: mix_collapse's order.
  const uint4* source = reinterpret_cast<const uint4*>(x + uint64_t(row) * Flat);
  float squares = 0.f;
  for (int v = t; v < Flat / 8; v += kThreads) {
    const uint4 packed = source[v];
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
  squares = warp_sum(squares);
  if (lane == 0) partials[warp] = squares;
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, partials[w]);
  const float rstd =
      __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));

  // The projection: mix_collapse's 128-K chains, butterflies and K order.
  float acc[M::kPerWarp];
  #pragma unroll
  for (int p = 0; p < M::kPerWarp; ++p) acc[p] = 0.f;
  for (int c = 0; c < M::kChains; ++c) {
    const int k = c * kChain + lane * 4;
    const uint2 xs = *reinterpret_cast<const uint2*>(streams + k);
    const float xv[4] = {bf_lo(xs.x), bf_hi(xs.x), bf_lo(xs.y), bf_hi(xs.y)};
    #pragma unroll
    for (int p = 0; p < M::kPerWarp; ++p) {
      const int m = warp + p * kWarps;
      if (m < Mixes) {
        float w[4];
        load_fn<M::kF32Fn>(fn, uint64_t(m) * Flat + k, w);
        float chain = __fmul_rn(w[0], xv[0]);
        chain = fmaf(w[1], xv[1], chain);
        chain = fmaf(w[2], xv[2], chain);
        chain = fmaf(w[3], xv[3], chain);
        acc[p] = __fadd_rn(acc[p], warp_sum(chain));
      }
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int p = 0; p < M::kPerWarp; ++p) {
      const int m = warp + p * kWarps;
      if (m < Mixes) {
        logits[m] = __fmul_rn(acc[p], rstd);
        if constexpr (Probe) projected[uint64_t(row) * Mixes + m] = acc[p];
      }
    }
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

// F32 expand (grid: (ceil(H / 512), row capacity), 256 threads), two
// columns per thread, every stream:
//   X'_j = bf16(fl(post_j * y) + sum_i comb[i][j] * X_i)
// with the stream sum an FMA chain from zero in stream order and one BF16
// rounding. x and out are [rows, N, H] BF16 (out must not alias x), y
// [rows, H] BF16, post [rows, N] and comb [rows, N, N] F32.
template<int Streams, int Width>
__device__ void expand_f32(const uint16_t* x, const uint16_t* y, const float* post,
                           const float* comb, uint16_t* out, const uint32_t* live_rows) {
  static_assert(Streams >= 2 && Streams <= 8 && Width % 2 == 0);
  constexpr int N = Streams;
  constexpr int H = Width;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const int col = blockIdx.x * kColumns + 2 * threadIdx.x;
  if (col >= H) return;
  const uint64_t base = uint64_t(row) * N * H + col;
  float lo[N], hi[N];
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint32_t pair = *reinterpret_cast<const uint32_t*>(x + base + uint64_t(i) * H);
    lo[i] = bf_lo(pair);
    hi[i] = bf_hi(pair);
  }
  const uint32_t ys = *reinterpret_cast<const uint32_t*>(y + uint64_t(row) * H + col);
  const float y_lo = bf_lo(ys);
  const float y_hi = bf_hi(ys);
  const float* weights = comb + uint64_t(row) * N * N;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    const float p = post[uint64_t(row) * N + j];
    float mixed_lo = 0.f, mixed_hi = 0.f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      const float w = weights[i * N + j];
      mixed_lo = fmaf(w, lo[i], mixed_lo);
      mixed_hi = fmaf(w, hi[i], mixed_hi);
    }
    const float out_lo = __fadd_rn(__fmul_rn(p, y_lo), mixed_lo);
    const float out_hi = __fadd_rn(__fmul_rn(p, y_hi), mixed_hi);
    *reinterpret_cast<uint32_t*>(out + base + uint64_t(j) * H) = pack(out_lo, out_hi);
  }
}

// Sigmoid-weighted head (grid: (ceil(H / 512), row capacity), 256
// threads): out = bf16(sum_i pre_i * X_i), an FMA chain from zero in stream
// order, with pre [rows, N] F32 the last lagged mix-collapse's. x is
// [rows, N, H] BF16 and out [rows, H], BF16 or unrounded F32.
template<int Streams, int Width, class Output = uint16_t>
__device__ void weighted_head(const uint16_t* x, const float* pre, Output* out,
                              const uint32_t* live_rows) {
  static_assert(Streams >= 2 && Streams <= 8 && Width % 2 == 0);
  constexpr int N = Streams;
  constexpr int H = Width;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const int col = blockIdx.x * kColumns + 2 * threadIdx.x;
  if (col >= H) return;
  const uint64_t base = uint64_t(row) * N * H + col;
  float lo = 0.f, hi = 0.f;
  #pragma unroll
  for (int step = 0; step < N; ++step) {
    #ifdef CHOP_HC_HEAD_REVERSE_STREAMS
    const int i = N - 1 - step;
    #else
    const int i = step;
    #endif
    const float w = pre[uint64_t(row) * N + i];
    const uint32_t pair = *reinterpret_cast<const uint32_t*>(x + base + uint64_t(i) * H);
    lo = fmaf(w, bf_lo(pair), lo);
    hi = fmaf(w, bf_hi(pair), hi);
  }
  if constexpr (sizeof(Output) == sizeof(float)) {
    #ifdef CHOP_HC_HEAD_ROUND_BF16
    lo = bf(to_bf(lo));
    hi = bf(to_bf(hi));
    #endif
    out[uint64_t(row) * H + col] = lo;
    out[uint64_t(row) * H + col + 1] = hi;
  } else {
    *reinterpret_cast<uint32_t*>(out + uint64_t(row) * H + col) = pack(lo, hi);
  }
}

} // namespace chop::sinkhorn_hc
