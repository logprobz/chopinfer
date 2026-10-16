#pragma once

// Rotary of packed F32 q/k/v rows, rewritten as BF16 pair planes.

#include "common.cuh"
#include "../../common/bf16_pairs.cuh"

namespace chop::multimodal {

// One packed F32 row [q | k | v] of QHeads, KvHeads and KvHeads heads of Dim,
// in place: q and k rotate over their first Rotary dims (NeoX halves or
// GPT-J neighbours) by the token's cos and sin [Rotary / 2], as HF's
// x * cos + rotate_half(x) * sin with each product rounded; then the row is
// rewritten as the BF16 pairs of every value in six planes of the same
// bytes: q hi, q lo, k hi, k lo, v hi, v lo.
template<int QHeads, int KvHeads, int Dim, int Rotary, bool Neox>
__device__ void rope_qkv_pairs(float* rows, const float* cos, const float* sin,
                               const uint32_t* live_rows) {
  constexpr int QWidth = QHeads * Dim;
  constexpr int KvWidth = KvHeads * Dim;
  constexpr int Width = QWidth + 2 * KvWidth;
  constexpr int Pairs = Rotary / 2;
  ptx::griddep_launch_dependents();
  __shared__ float values[Width];
  ptx::griddep_wait();
  const uint32_t token = blockIdx.x;
  if (token >= *live_rows) return;
  float* row = rows + uint64_t(token) * Width;
  for (int col = threadIdx.x; col < Width; col += blockDim.x) values[col] = row[col];
  // The row is read whole before any thread overwrites it.
  __syncthreads();
  uint16_t* planes = reinterpret_cast<uint16_t*>(row);
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    float x = values[col];
    int tensor = 2, local = col - QWidth - KvWidth, width = KvWidth;
    if (col < QWidth) {
      tensor = 0;
      local = col;
      width = QWidth;
    } else if (col < QWidth + KvWidth) {
      tensor = 1;
      local = col - QWidth;
    }
    const int j = local % Dim;
    if (tensor < 2 && j < Rotary) {
      const bool first = Neox ? j < Pairs : (j & 1) == 0;
      const int pair = Neox ? j % Pairs : j / 2;
      const int partner = Neox ? (first ? j + Pairs : j - Pairs) : j ^ 1;
      const float y = values[col - j + partner];
      const float c = cos[uint64_t(token) * Pairs + pair];
      const float s = sin[uint64_t(token) * Pairs + pair];
      x = first ? __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s))
                : __fadd_rn(__fmul_rn(x, c), __fmul_rn(y, s));
    }
    // Plane starts: q hi 0, q lo QWidth, k hi 2 QWidth, then KvWidth apart.
    const int hi = tensor == 0 ? 0 : 2 * QWidth + (tensor - 1) * 2 * KvWidth;
    pairs::split(x, planes[hi + local], planes[hi + width + local]);
  }
}

} // namespace chop::multimodal
