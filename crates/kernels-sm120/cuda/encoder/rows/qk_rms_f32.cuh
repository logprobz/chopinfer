#pragma once

// Per-head RMSNorm of q and k in packed F32 qkv rows
// (`attention.varlen.qk-rms.pairs.v1`), before the rotary.

#include "common.cuh"

namespace chop::multimodal {

// In place over rows [(QHeads + 2 KvHeads) * HeadDim] F32: every q head and
// every k head becomes (x * rstd) * weight, with rstd the correctly rounded
// reciprocal square root of the head's F32 mean square plus eps; v is not
// touched. One CTA per row, one warp per head at a time: lane l holds
// columns l, l + 32, ... and squares them with FMA in that order, and a
// butterfly over the 32 lanes sums them, so every lane holds the same
// total and the order is fixed.
template<int QHeads, int KvHeads, int HeadDim>
__device__ void qk_rms_f32(float* rows, const uint16_t* q_weight,
                           const uint16_t* k_weight, const uint32_t* live_rows,
                           float eps) {
  static_assert(HeadDim % 32 == 0 && HeadDim <= 256, "head_dim in 32-lane steps");
  constexpr int Width = (QHeads + 2 * KvHeads) * HeadDim;
  constexpr int PerLane = HeadDim / 32;
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  float* base = rows + uint64_t(row) * Width;
  for (int head = threadIdx.x >> 5; head < QHeads + KvHeads; head += warps) {
    float* x = base + head * HeadDim;
    const uint16_t* weight = head < QHeads ? q_weight : k_weight;
    float values[PerLane];
    float squares = 0.f;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
      values[i] = x[lane + 32 * i];
      squares = __fmaf_rn(values[i], values[i], squares);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      squares = __fadd_rn(squares, __shfl_xor_sync(0xffffffff, squares, offset));
    const float rstd = __frsqrt_rn(__fadd_rn(__fdiv_rn(squares, float(HeadDim)), eps));
#pragma unroll
    for (int i = 0; i < PerLane; ++i)
      x[lane + 32 * i] = __fmul_rn(__fmul_rn(values[i], rstd), bf(weight[lane + 32 * i]));
  }
}

} // namespace chop::multimodal
