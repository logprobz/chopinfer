#pragma once

// The Sinkhorn mHC comm epilogue split over CTAs: bitwise
// sinkhorn_expand_mix_norm.cuh's one-CTA-per-row epilogue (and so the
// unfused launches) at every row count. A row of the fused width spreads
// over kSlices CTAs of kSliceColumns columns each. CTA (row, s)
//   1. rounds its columns of the F32 sum once to BF16 (y, into `reduced`),
//   2. expands them into the streams with the sublayer's post and comb,
//   3. computes the next mixer's 128-K chains over its columns of X' (one
//      chain per warp: stream w / 2, column block 2 s + w % 2) as
//      sinkhorn_hc's mix_collapse computes them, into the row's partials,
// and the CTA that takes the row's last ticket finishes it: the
// mix-collapse's rstd, chains added from zero in K order, Sinkhorn and
// collapse (sinkhorn_hc/mix_split.cuh's mix_finish), then the HF-ordered
// RMSNorm of the collapsed row. Every expression is the one-CTA
// epilogue's; the ticket only elects the finishing CTA. It runs over F32
// sums another all-reduce wrote (the one-shot or the phased pair), in a
// launch of its own: no CTA ever waits on another CTA, so the grid needs no
// co-residency. Rows at or past the live count are left untouched.

#include <stdint.h>
#include <math.h>
#include "sinkhorn_hc/mix_split.cuh"

namespace chop::comm::epilogue {

// Columns of one CTA's slice: two 128-column chain blocks per stream.
constexpr int kSliceColumns = 256;

// M is a chop::sinkhorn_hc::Mixer of the fused width.
template <class M>
struct SinkhornSplitExpandMixNorm {
  using Mixer = M;
  static constexpr int kWidth = M::kWidth;
  static constexpr int kStreams = M::kStreams;
  static constexpr int kThreads = chop::sinkhorn_hc::kThreads;
  static constexpr int kSlices = kWidth / kSliceColumns;
  // Squares the ordered norm sum stages per pass.
  static constexpr int kStage = 256;
  static_assert(kWidth == 4096, "the fused hyper-connection width (FUSED_WIDTH)");
  static_assert(kStreams * kSliceColumns / chop::sinkhorn_hc::kChain ==
                    chop::sinkhorn_hc::Split<M>::kChainsPerCta,
                "a slice holds one chain per warp");
  static_assert(chop::sinkhorn_hc::Split<M>::kGroups == kSlices, "one chain group per slice");
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
  float* partials;           // [rows, mixes, chains] F32 chain sums
  uint32_t* tickets;         // [rows], zeroed at allocation
  const uint32_t* live_rows;
  float eps;                 // the input norm's eps

  // Slice `s` of row `row` (below the live count) from the row's F32 sum
  // `sum`, which this CTA or an earlier launch wrote; then, in the row's
  // last CTA, the rest of the row.
  __device__ void run(uint32_t row, uint32_t s, const float* sum) const {
    using namespace chop::sinkhorn_hc;
    constexpr int N = kStreams;
    constexpr int H = kWidth;
    const int t = threadIdx.x;

    // 1-2. Round and expand, two columns per thread: the round row's
    // rounding and sinkhorn_hc's expand per element.
    const uint16_t* weights = comb + uint64_t(row) * N * N;
    for (int col = s * kSliceColumns + 2 * t; col < (s + 1) * kSliceColumns; col += 2 * kThreads) {
      const uint32_t ys = pack(sum[col], sum[col + 1]);
      *reinterpret_cast<uint32_t*>(reduced + uint64_t(row) * H + col) = ys;
      const uint64_t at = uint64_t(row) * N * H + col;
      float lo[N], hi[N];
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + at + uint64_t(i) * H);
        lo[i] = bf_lo(pair);
        hi[i] = bf_hi(pair);
      }
      const float y_lo = bf_lo(ys);
      const float y_hi = bf_hi(ys);
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
    // This CTA's X' columns are visible to all its warps.
    __syncthreads();
#ifdef CHOP_COMM_TEST_LATE_WAIT
    ptx::griddep_wait();
#endif

    // 3. The next mixer's chains over this slice: warp w takes stream
    // w / 2, column block 2 s + w % 2.
    const int warp = t >> 5;
    const int chain = (warp >> 1) * (H / kChain) + s * (kSliceColumns / kChain) + (warp & 1);
    const uint16_t* row_streams = expanded + uint64_t(row) * N * H;
    float* row_partials = partials + uint64_t(row) * Split<M>::kPartials;
    chain_sums<M>(row_streams, fn, chain, row_partials);
    // Every CTA of the row has read post and comb before the last one
    // overwrites them.
    if (!last_arrival(tickets + row, kSlices)) return;

    // The rest of the next sublayer's mix-collapse over the row of X'.
    __shared__ __align__(16) uint16_t staged[M::kFlat];
    mix_finish<M>(row_streams, row_partials, base, scale, collapsed + uint64_t(row) * H,
                  post + uint64_t(row) * N, comb + uint64_t(row) * M::kCells, staged);
    __syncthreads();

    // 4. The HF-ordered RMSNorm of the collapsed row (the one-CTA
    // epilogue's, staged in smaller passes).
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
      const uint16_t* norm_weights = reinterpret_cast<const uint16_t*>(&wpacked);
      uint4 result;
      uint16_t* out = reinterpret_cast<uint16_t*>(&result);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float w = bf(norm_weights[j]);
        float n = bf(in[j]) * inv;
        n = bf(to_bf(n));
        out[j] = to_bf(n * w);
      }
      reinterpret_cast<uint4*>(normed + uint64_t(row) * H + c)[0] = result;
    }
  }
};

// The epilogue alone over rows another all-reduce summed into `summed`
// (grid: (kSlices, rows)).
//
// Test-only mutant, compiled into kernel-lab's late-wait cubin: with
// CHOP_COMM_TEST_LATE_WAIT the dependency wait moves after the reads of
// `summed`, the streams, post and comb (steps 1-2 of `run`), which
// kernel-lab's PDL producers must catch. Serving cubins leave it undefined.
template <class E>
__device__ void sinkhorn_split_rows(const float* summed, const E& epilogue) {
  ptx::griddep_launch_dependents();
  chop::sinkhorn_hc::prefetch_group<typename E::Mixer>(epilogue.fn, blockIdx.x);
#ifndef CHOP_COMM_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (blockDim.x != E::kThreads || gridDim.x != E::kSlices) asm volatile("trap;");
  const uint32_t row = blockIdx.y;
  if (row >= *epilogue.live_rows) return;
  epilogue.run(row, blockIdx.x, summed + uint64_t(row) * E::kWidth);
}

} // namespace chop::comm::epilogue
