#pragma once

// The Sinkhorn mHC comm epilogue split over CTAs with the input RMSNorm's
// sum of squares in row/rmsnorm_tree.cuh's 256-lane F32 tree
// (kernel_core::rmsnorm::RmsNormNumerics::HF_TREE_256): sinkhorn_split.cuh's
// epilogue statement for statement, except step 4, whose sum is the tree's
// instead of thread 0's ordered adds. Its outputs are the unfused launches'
// with the tree norm (the round, the Sinkhorn expand, the split
// mix-collapse, then the row engine's rmsnorm_hf_tree of the collapsed
// row) bit for bit at every row count, as sinkhorn_split.cuh's are the
// unfused launches' with the ordered norm.
//
// Test-only mutants: sinkhorn_split.cuh's CHOP_COMM_TEST_LATE_WAIT and
// rmsnorm_tree.cuh's CHOP_RMSNORM_TREE_TEST_DISTANCES_REVERSED, each in
// this epilogue as in theirs. Serving cubins leave them undefined.

#include <stdint.h>
#include <math.h>
#include "sinkhorn_split.cuh"
#include "row/rmsnorm_tree.cuh"

namespace chop::comm::epilogue {

// M is a chop::sinkhorn_hc::Mixer of the fused width.
template <class M>
struct SinkhornSplitExpandMixTreeNorm : SinkhornSplitExpandMixNorm<M> {
  using Base = SinkhornSplitExpandMixNorm<M>;
  using Base::kWidth;
  using Base::kStreams;
  using Base::kThreads;
  using Base::kSlices;
  using Base::streams;
  using Base::post;
  using Base::comb;
  using Base::reduced;
  using Base::expanded;
  using Base::fn;
  using Base::base;
  using Base::scale;
  using Base::collapsed;
  using Base::weight;
  using Base::normed;
  using Base::partials;
  using Base::tickets;
  using Base::live_rows;
  using Base::eps;
  static_assert(kThreads == chop::row::kTreeLanes, "thread t is tree lane t");

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

    // 4. The input RMSNorm of the collapsed row with
    // row/rmsnorm_tree.cuh's 256-lane tree sum (thread t is tree lane t, as
    // in rmsnorm_hf_tree), then the HF form's reciprocal and output.
    __shared__ float tree_sums[chop::row::kTreeLanes / 32];
    __shared__ float tree_inv;
    constexpr int Group = kWidth;
    const uint16_t* x = collapsed + uint64_t(row) * H;
    float partial = 0.f;
    for (int c = t * 8; c < Group; c += chop::row::kTreeSpan) {
      const uint4 packed = reinterpret_cast<const uint4*>(x + c)[0];
      const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = bf(in[j]);
        partial = __fadd_rn(partial, __fmul_rn(v, v));
      }
    }
    const float warp_total = chop::row::tree_warp_sum(partial);
    if ((t & 31) == 0) tree_sums[t >> 5] = warp_total;
    __syncthreads();
    if (t < 32) {
      const float total =
          chop::row::tree_warps_sum(t < chop::row::kTreeLanes / 32 ? tree_sums[t] : 0.f);
      if (t == 0) tree_inv = 1.f / sqrtf(total / Group + eps);
    }
    __syncthreads();
    const float inv = tree_inv;
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

} // namespace chop::comm::epilogue
