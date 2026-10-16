#pragma once

// Sinkhorn-mixed hyper-connections (kernel-core sinkhorn_hyper_connection):
// the mix-collapse before a sublayer, the expand after it and the
// mean-of-streams head. See README.md for the ABI, the schedule and the
// rounding points. All arithmetic is F32 with explicit roundings
// (__fmul_rn, __fadd_rn, fmaf) so no contraction changes a result; no
// float atomics, every reduction order is fixed.

#include <stdint.h>
#include <math.h>
#include "../common/ptx.cuh"
#include "../common/convert.cuh"
#include "../common/sigmoid.cuh"

namespace chop::sinkhorn_hc {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// K of one projection chain: a chain starts from zero and is added into
// its mix's accumulator in K order.
constexpr int kChain = 128;
// Columns of one expand or head CTA: two per thread.
constexpr int kColumns = 2 * kThreads;

__device__ __forceinline__ float bf(uint16_t x) {
  return convert::bf16_to_f32(x);
}
__device__ __forceinline__ uint16_t to_bf(float x) {
  return convert::bf16_from_f32(x);
}
// Low and high BF16 halves of a packed pair, low address first.
__device__ __forceinline__ float bf_lo(uint32_t x) {
  return __uint_as_float(x << 16);
}
__device__ __forceinline__ float bf_hi(uint32_t x) {
  return __uint_as_float(x & 0xffff0000u);
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  return uint32_t(to_bf(lo)) | (uint32_t(to_bf(hi)) << 16);
}

// The butterfly sum over a warp: every lane ends with the same bits, as
// each level adds the same two partial sums in either operand order.
__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Four consecutive fn values of a chain lane, BF16 or F32.
template<bool F32Fn>
__device__ __forceinline__ void load_fn(const void* fn, uint64_t at, float (&w)[4]) {
  if constexpr (F32Fn) {
    const float4 v = *reinterpret_cast<const float4*>(static_cast<const float*>(fn) + at);
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
  } else {
    const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(fn) + at);
    w[0] = bf_lo(v.x); w[1] = bf_hi(v.x); w[2] = bf_lo(v.y); w[3] = bf_hi(v.y);
  }
}

// One mixer: N streams of width H, `Iterations` Sinkhorn iterations, eps
// and norm eps as F32 bits, fn in F32 or BF16.
template<int Streams, int Width, int Iterations, uint32_t EpsBits, uint32_t NormEpsBits,
         bool F32Fn>
struct Mixer {
  static constexpr int kStreams = Streams;
  static constexpr int kWidth = Width;
  static constexpr bool kF32Fn = F32Fn;
  static constexpr int kIterations = Iterations;
  static constexpr uint32_t kEpsBits = EpsBits;
  static constexpr uint32_t kNormEpsBits = NormEpsBits;
  static constexpr int kFlat = Streams * Width;
  static constexpr int kMixes = (2 + Streams) * Streams;
  static constexpr int kCells = Streams * Streams;
  static constexpr int kChains = kFlat / kChain;
  // Mixes per warp: warp w owns mixes w, w + 8, ...
  static constexpr int kPerWarp = (kMixes + kWarps - 1) / kWarps;
  static_assert(Streams >= 2 && Streams <= 8, "2 to 8 streams");
  static_assert(Width % kChain == 0, "width is a multiple of 128");
  static_assert(Iterations >= 1 && Iterations <= 64, "1 to 64 Sinkhorn iterations");
  static_assert(kCells <= kThreads, "one thread per comb cell");
  // The row's streams stay in static shared memory as BF16.
  static_assert(kFlat * 2 <= 40960, "flat streams fit 40 KiB of shared memory");
};

// MixCollapse, one row per CTA (grid: row capacity), 256 threads:
//   x [rows, N, H] BF16, fn [(2 + N) N, N H], base [(2 + N) N] F32,
//   scale [3] F32 -> collapsed [rows, H] BF16, post [rows, N] BF16,
//   comb [rows, N, N] BF16.
// With Probe, also the raw projection [rows, (2 + N) N] F32 (before the RMS
// scale) and rstd [rows] F32, the long reductions the lab gates, and the
// Sinkhorn input cells [rows, N, N] F32 (the row softmax plus eps), from
// which the lab reproduces the normalizations bit for bit.
template<class M, bool Probe>
__device__ void mix_collapse(const uint16_t* x, const void* fn, const float* base,
                             const float* scale, uint16_t* collapsed, uint16_t* post,
                             uint16_t* comb, float* projected, float* rstd_out,
                             float* cells_out, const uint32_t* live_rows) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int ElementBytes = M::kF32Fn ? 4 : 2;
  __shared__ __align__(16) uint16_t streams[Flat];
  __shared__ float partials[kWarps];
  __shared__ float logits[Mixes];
  __shared__ float pre[N];
  __shared__ float cells[Cells];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  ptx::griddep_launch_dependents();
  // fn is a weight: warm L2 with it before waiting on the producer. The
  // CTAs of a launch prefetch disjoint 128-byte lines, wrapping over fn.
  {
    constexpr uint64_t Lines = (uint64_t(Mixes) * Flat * ElementBytes + 127) / 128;
    const uint64_t line = (uint64_t(blockIdx.x) * kThreads + t) % Lines;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<const char*>(fn) + line * 128));
  }
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;

  // Stage the row and sum its squares: each thread FMAs its own elements
  // in order, then a fixed tree over lanes and warps.
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

  // The projection over the raw streams: warp w owns mixes w, w + 8, ...;
  // per 128-K chain each lane FMAs four products from zero, the butterfly
  // sums the lanes, and the chain is added into the mix in K order.
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

  // pre, post and the comb logits.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[t] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[uint64_t(row) * N + t] = to_bf(__fmul_rn(2.f, placed));
  }
  // Sinkhorn: thread t < N * N owns cell (i, j) = (t / N, t % N). Every
  // sum runs in index order from zero; each normalization publishes the
  // cells, reads its line and divides by the sum plus eps.
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
  if (owner) comb[uint64_t(row) * Cells + t] = to_bf(cell);

  // The collapse: per column an FMA chain from zero in stream order.
  const uint64_t out = uint64_t(row) * H;
  for (int col = 2 * t; col < H; col += kColumns) {
    float lo = 0.f, hi = 0.f;
    #pragma unroll
    for (int s = 0; s < N; ++s) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + s * H + col);
      lo = fmaf(pre[s], bf_lo(pair), lo);
      hi = fmaf(pre[s], bf_hi(pair), hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed + out + col) = pack(lo, hi);
  }
}

// Expand (grid: (ceil(H / 512), row capacity), 256 threads), two columns
// per thread, every stream:
//   X'_j = bf16(bf16(post_j * y) + bf16(sum_i comb[i][j] * X_i))
// with the stream sum an FMA chain from zero in stream order. x and out are
// [rows, N, H], y [rows, H], post [rows, N], comb [rows, N, N], all BF16.
template<int Streams, int Width>
__device__ void expand(const uint16_t* x, const uint16_t* y, const uint16_t* post,
                       const uint16_t* comb, uint16_t* out, const uint32_t* live_rows) {
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
  const uint16_t* weights = comb + uint64_t(row) * N * N;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    const float p = bf(post[uint64_t(row) * N + j]);
    float mixed_lo = 0.f, mixed_hi = 0.f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      const float w = bf(weights[i * N + j]);
      mixed_lo = fmaf(w, lo[i], mixed_lo);
      mixed_hi = fmaf(w, hi[i], mixed_hi);
    }
    const float out_lo = __fadd_rn(bf(to_bf(__fmul_rn(p, y_lo))), bf(to_bf(mixed_lo)));
    const float out_hi = __fadd_rn(bf(to_bf(__fmul_rn(p, y_hi))), bf(to_bf(mixed_hi)));
    *reinterpret_cast<uint32_t*>(out + base + uint64_t(j) * H) = pack(out_lo, out_hi);
  }
}

// Mean-of-streams head (grid: (ceil(H / 512), row capacity), 256 threads):
//   out = bf16((sum_i X_i) / N), the sum from zero in stream order.
// x is [rows, N, H] and out [rows, H], BF16.
template<int Streams, int Width>
__device__ void head(const uint16_t* x, uint16_t* out, const uint32_t* live_rows) {
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
  for (int i = 0; i < N; ++i) {
    const uint32_t pair = *reinterpret_cast<const uint32_t*>(x + base + uint64_t(i) * H);
    lo = __fadd_rn(lo, bf_lo(pair));
    hi = __fadd_rn(hi, bf_hi(pair));
  }
  *reinterpret_cast<uint32_t*>(out + uint64_t(row) * H + col) =
      pack(__fdiv_rn(lo, float(N)), __fdiv_rn(hi, float(N)));
}

} // namespace chop::sinkhorn_hc
