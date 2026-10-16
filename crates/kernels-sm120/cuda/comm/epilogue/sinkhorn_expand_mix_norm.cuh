#pragma once

// The Sinkhorn mHC comm epilogue (kernel-core
// SinkhornHyperConnectionQuery::AllReduceExpandMixNorm) and its two
// transports. After the all-reduce of a sublayer's F32 rank partials, one
// 256-thread CTA per row
//   1. rounds the rank-order sum once to BF16 (y, written to `reduced`),
//   2. expands y into the streams with the sublayer's post and comb
//      (X'_j = bf16(bf16(post_j * y) + bf16(sum_i comb[i][j] * X_i))),
//   3. runs the next sublayer's mix-collapse over X', which overwrites post
//      and comb with the next sublayer's and writes the collapsed row, and
//   4. normalizes the collapsed row with the next sublayer's input RMSNorm.
// Every step is its unfused launch's arithmetic in the same order: the sum
// is comm.cuh's allreduce_oneshot (and the phased pair's), from zero in rank
// order; the rounding is the bundle's round row (cvt.rn); the expand is
// sinkhorn_hc's expand per element; the mix-collapse is sinkhorn_hc's own
// mix_collapse, called unchanged for this CTA's row; and the norm is
// row.cuh's HF-ordered rmsnorm (Kind 3): each square rounded alone, summed
// from zero in column order by one thread, 1 / sqrtf(mean + eps), and the
// normalized value rounded to BF16 before the weight multiply. None of
// these orders depends on the grid or the rank count, so the fused rows are
// bitwise the unfused ones at every row count. Rows at or past the live
// count are left untouched, as the unfused row kernels leave them; the
// one-shot still sums every row of its bucket, as allreduce_oneshot does.

#include <stdint.h>
#include <math.h>
#include "comm/comm.cuh"
#include "sinkhorn_hc/sinkhorn_hc.cuh"

namespace chop::comm::epilogue {

// M is a chop::sinkhorn_hc::Mixer of the fused width.
template <class M>
struct SinkhornExpandMixNorm {
  static constexpr int kWidth = M::kWidth;
  static constexpr int kStreams = M::kStreams;
  static constexpr int kThreads = chop::sinkhorn_hc::kThreads;
  // Squares the ordered norm sum stages per pass, so the mix-collapse's
  // static row of streams and this stage fit 48 KiB together.
  static constexpr int kStage = 1024;
  static_assert(kWidth == 4096, "the fused hyper-connection width (FUSED_WIDTH)");
  static_assert(kWidth % kStage == 0 && kStage % kThreads == 0, "whole stages");

  const uint16_t* streams;   // [rows, N, H] the sublayer's input streams X
  uint16_t* post;            // [rows, N]: read, then the next sublayer's
  uint16_t* comb;            // [rows, N, N]: read, then the next sublayer's
  uint16_t* reduced;         // [rows, H] y
  uint16_t* expanded;        // [rows, N, H] X'
  const void* fn;            // the next mixer's projection, as M stores it
  const float* base;
  const float* scale;
  uint16_t* collapsed;       // [rows, H]
  const uint16_t* weight;    // [H] the next sublayer's input norm weight
  uint16_t* normed;          // [rows, H]
  const uint32_t* live_rows;
  float eps;                 // the input norm's eps

  // Row `row` (this CTA's blockIdx.x, below the live count) from its F32
  // sum `sum`, which this CTA or an earlier launch wrote.
  __device__ void finish(uint32_t row, const float* sum) const {
    using namespace chop::sinkhorn_hc;
    constexpr int N = kStreams;
    constexpr int H = kWidth;
    const int t = threadIdx.x;

    // 1. The round row: four columns per thread, rounded once.
    for (int c = t * 4; c < H; c += kThreads * 4) {
      const float4 v = reinterpret_cast<const float4*>(sum + c)[0];
      uint2 bits;
      bits.x = uint32_t(to_bf(v.x)) | (uint32_t(to_bf(v.y)) << 16);
      bits.y = uint32_t(to_bf(v.z)) | (uint32_t(to_bf(v.w)) << 16);
      reinterpret_cast<uint2*>(reduced + uint64_t(row) * H + c)[0] = bits;
    }
    __syncthreads();

    // 2. The expand: per element, the stream sum an FMA chain from zero in
    // stream order, both products rounded before the add.
    for (int col = 2 * t; col < H; col += kColumns) {
      const uint64_t at = uint64_t(row) * N * H + col;
      float lo[N], hi[N];
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + at + uint64_t(i) * H);
        lo[i] = bf_lo(pair);
        hi[i] = bf_hi(pair);
      }
      const uint32_t ys = *reinterpret_cast<const uint32_t*>(reduced + uint64_t(row) * H + col);
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
        *reinterpret_cast<uint32_t*>(expanded + at + uint64_t(j) * H) = pack(out_lo, out_hi);
      }
    }
    // Every thread's reads of post and comb finish before the mix-collapse
    // overwrites them, and X' is visible to the whole CTA.
    __syncthreads();

    // 3. The next sublayer's mix-collapse, unchanged, over this row of X'.
    mix_collapse<M, false>(expanded, fn, base, scale, collapsed, post, comb, nullptr, nullptr,
                           nullptr, live_rows);
    __syncthreads();

    // 4. The HF-ordered RMSNorm of the collapsed row.
    __shared__ float squares[kStage];
    __shared__ float total;
    constexpr int Group = kWidth;
    const uint16_t* x = collapsed + uint64_t(row) * H;
    float ordered = 0.f;
    for (int first = 0; first < Group; first += kStage) {
      for (int c = t; c < kStage; c += kThreads) {
        const float v = bf(x[first + c]);
        squares[c] = __fmul_rn(v, v);
      }
      __syncthreads();
      if (t == 0) {
        for (int c = 0; c < kStage; ++c) ordered = __fadd_rn(ordered, squares[c]);
      }
      __syncthreads();
    }
    if (t == 0) total = ordered;
    __syncthreads();
    const float inv = 1.f / sqrtf(total / Group + eps);
    for (int c = t * 8; c < Group; c += kThreads * 8) {
      const uint4 packed = reinterpret_cast<const uint4*>(x + c)[0];
      const uint4 wpacked = reinterpret_cast<const uint4*>(weight + c)[0];
      const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
      const uint16_t* weights = reinterpret_cast<const uint16_t*>(&wpacked);
      uint4 result;
      uint16_t* out = reinterpret_cast<uint16_t*>(&result);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float w = bf(weights[j]);
        float n = bf(in[j]) * inv;
        n = bf(to_bf(n));
        out[j] = to_bf(n * w);
      }
      reinterpret_cast<uint4*>(normed + uint64_t(row) * H + c)[0] = result;
    }
  }
};

// The F32 one-shot all-reduce with the epilogue on each reduced row: the
// protocol, state layout and sums of allreduce_oneshot (one CTA, epoch cell
// and flag pair per 4096-element tile, which here is one row), so the two
// entries share one state and may alternate on it. Every row's sum is also
// written to `output`, as the unfused entry writes it.
//
// Test-only mutant, compiled into kernel-lab's late-wait cubin: with
// CHOP_COMM_TEST_LATE_WAIT the dependency wait moves after the reads of
// `input`, just before the epilogue, which kernel-lab's PDL partial
// producer must catch. Serving cubins leave it undefined.
template <class E, typename Inboxes, typename Flags>
__device__ void allreduce_oneshot_sinkhorn(const float* input, float* output, Inboxes inboxes,
                                           Flags flags, unsigned int* epoch, uint64_t elements,
                                           uint32_t rank, uint32_t ranks, const E& epilogue) {
  static_assert(E::kWidth == 4096, "a tile is one row");
  ptx::griddep_launch_dependents();
#ifndef CHOP_COMM_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (blockDim.x != E::kThreads || elements % E::kWidth != 0) asm volatile("trap;");
  const uint64_t tile = blockIdx.x;
  const uint64_t tiles = (elements + 4095) / 4096;
  if (tile >= tiles || rank >= ranks || ranks == 0 || ranks > kMaxRanks) return;
  const uint64_t first = tile * 4096;
  const uint64_t end = min(first + 4096, elements);
  const uint32_t next = epoch[tile] + 1;
  const uint64_t flag_index = static_cast<uint64_t>(rank) * tiles + tile;
  if (threadIdx.x == 0) {
    for (uint32_t dest = 0; dest < ranks; ++dest) {
      while (acquire_system(peer(flags, dest) + ranks * tiles + flag_index) != next - 1) {}
    }
  }
  __syncthreads();
  for (uint32_t dest = 0; dest < ranks; ++dest) {
    float* slot = peer(inboxes, dest) + static_cast<uint64_t>(rank) * elements;
    for (uint64_t i = first + threadIdx.x; i < end; i += blockDim.x)
      slot[i] = input[i];
    __syncthreads();
    if (threadIdx.x == 0)
      release_system(peer(flags, dest) + flag_index, next);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source) {
      while (acquire_system(peer(flags, rank) + static_cast<uint64_t>(source) * tiles + tile) != next) {}
    }
  }
  __syncthreads();
  for (uint64_t i = first + threadIdx.x; i < end; i += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t source = 0; source < ranks; ++source)
      sum += decode(peer(inboxes, rank)[static_cast<uint64_t>(source) * elements + i]);
    output[i] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source)
      release_system(peer(flags, rank) + ranks * tiles + static_cast<uint64_t>(source) * tiles + tile, next);
    epoch[tile] = next;
  }
#ifdef CHOP_COMM_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (tile >= *epilogue.live_rows) return;
  epilogue.finish(static_cast<uint32_t>(tile), output + first);
}

// The epilogue alone over rows another all-reduce summed into `summed` (the
// phased reduce-scatter and all-gather's F32 sums): one CTA per row.
template <class E>
__device__ void sinkhorn_expand_mix_norm_rows(const float* summed, const E& epilogue) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (blockDim.x != E::kThreads) asm volatile("trap;");
  const uint32_t row = blockIdx.x;
  if (row >= *epilogue.live_rows) return;
  epilogue.finish(row, summed + uint64_t(row) * E::kWidth);
}

} // namespace chop::comm::epilogue
