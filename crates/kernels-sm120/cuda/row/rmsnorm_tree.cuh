#pragma once

// The HF RMSNorm (row.cuh's rmsnorm Kind 3: separately rounded squares,
// sqrt then divide, the normalized value rounded to BF16 before the weight)
// with its sum of squares in a fixed 256-lane F32 tree instead of one
// thread's serial adds (kernel_core::rmsnorm::RmsNormNumerics::HF_TREE_256).
// The order is a function of the width only, so every kernel that follows
// it (this row engine and the fused mixer norm of
// sinkhorn_hc/mix_norm_tree.cuh) gives a row the same bits whatever its
// batch, bucket or live count:
//
//   - lane l (0 to 255) adds the rounded squares of columns
//     8 l + 2048 i + j (i ascending, then j = 0 to 7, below the width) from
//     +0 with __fadd_rn;
//   - each warp of 32 lanes adds its lanes' partials pairwise at lane
//     distances 16, 8, 4, 2, 1 (lane m adds lane m + d to its own value);
//   - the eight warp sums are added the same way at distances 4, 2, 1.
//
// For non-negative terms (squares) the relative error of the tree's sum of
// 4096 terms is at most about 23 units of rounding (15 serial adds in a
// lane, 8 tree levels) against the serial order's 4095.
//
// Test-only mutant, compiled into kernel-lab's mutant cubins: with
// CHOP_RMSNORM_TREE_TEST_DISTANCES_REVERSED each warp adds its lanes at
// distances 1, 2, 4, 8, 16. Serving cubins leave it undefined.

#include <stdint.h>
#include <math.h>
#include "row.cuh"

namespace chop::row {

// The tree's lanes, and the columns one pass of the lanes covers.
constexpr int kTreeLanes = 256;
constexpr int kTreeSpan = kTreeLanes * 8;
#ifndef CHOP_RMSNORM_TREE_TEST_DISTANCES_REVERSED
constexpr bool kTreeDistancesReversed = false;
#else
constexpr bool kTreeDistancesReversed = true;
#endif

// Lane `lane`'s partial of `Width` squares in `squares` (shared or global
// memory).
template<int Width>
__device__ __forceinline__ float tree_lane_partial(const float* squares, int lane) {
  float partial = 0.f;
  #pragma unroll
  for (int base = 0; base < Width; base += kTreeSpan) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (base + 8 * lane + j < Width) partial = __fadd_rn(partial, squares[base + 8 * lane + j]);
    }
  }
  return partial;
}

// A warp's 32 lane partials pairwise at distances 16 to 1: the warp's sum
// in lane 0 (every lane of the warp calls this).
__device__ __forceinline__ float tree_warp_sum(float x) {
  #pragma unroll
  for (int level = 0; level < 5; ++level) {
    const int d = kTreeDistancesReversed ? 1 << level : 16 >> level;
    x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, d));
  }
  return x;
}

// The eight warp sums, held by lanes 0 to 7, pairwise at distances 4, 2, 1:
// the total in lane 0 (every lane of the warp calls this; the other lanes'
// values are unused).
__device__ __forceinline__ float tree_warps_sum(float x) {
  #pragma unroll
  for (int d = 4; d; d >>= 1) x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, d));
  return x;
}

// The RMSNorm of `Width`-wide BF16 rows (grid: row capacity, 256 threads,
// one CTA per row; row.cuh's rmsnorm ABI without a fused residual): thread
// t is tree lane t, adding its columns' squares as it loads them.
template<int Width>
__device__ void rmsnorm_hf_tree(const uint16_t* x, const uint16_t* weight, uint16_t* y,
                                const uint32_t* live_rows, float eps) {
  static_assert(Width % 8 == 0, "whole 16-byte vectors");
  ptx::griddep_launch_dependents();
  __shared__ float warp_sums[kTreeLanes / 32];
  __shared__ float inv_shared;
  if (blockDim.x != kTreeLanes) asm volatile("trap;");
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const uint16_t* in_row = x + uint64_t(row) * Width;
  float partial = 0.f;
  for (int c = t * 8; c < Width; c += kTreeSpan) {
    uint4 packed = reinterpret_cast<const uint4*>(in_row + c)[0];
    const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = bf(in[j]);
      partial = __fadd_rn(partial, __fmul_rn(v, v));
    }
  }
  const float warp_total = tree_warp_sum(partial);
  if (lane == 0) warp_sums[warp] = warp_total;
  __syncthreads();
  if (warp == 0) {
    const float total = tree_warps_sum(lane < kTreeLanes / 32 ? warp_sums[lane] : 0.f);
    if (lane == 0) inv_shared = 1.f / sqrtf(total / Width + eps);
  }
  __syncthreads();
  const float inv = inv_shared;
  for (int c = t * 8; c < Width; c += kTreeSpan) {
    uint4 packed = reinterpret_cast<const uint4*>(in_row + c)[0];
    uint4 wpacked = reinterpret_cast<const uint4*>(weight + c)[0];
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
    reinterpret_cast<uint4*>(y + uint64_t(row) * Width + c)[0] = result;
  }
}

} // namespace chop::row
