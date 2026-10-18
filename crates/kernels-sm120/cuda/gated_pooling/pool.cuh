#pragma once

#include <math.h>
#include <stdint.h>
#include "common/convert.cuh"
#include "common/ptx.cuh"

namespace chop::gated_pooling {

// One CTA owns one completing row. Only the first live row of an item
// reads its incoming tail or updates its in-place tail, so a final tail
// write cannot race a different CTA reading the incoming group.
template<int Width, int MaxRows>
__device__ void pool2(const float* key, const float* gate,
    const int32_t* positions, const int32_t* row_batch,
    const uint32_t* live_rows, const int32_t* slot_ids,
    const int32_t* checkpoint_ids, float* state, float* pooled,
    int32_t* output_positions, uint32_t items, uint32_t entries) {
  static_assert(Width == 128 || Width == 512);
  static_assert(MaxRows >= 0 && MaxRows <= 16);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live = *live_rows;
  const uint32_t row = blockIdx.x;
  if (row >= live) return;
  if (threadIdx.x == 0) output_positions[row] = -1;
  const int32_t position = positions[row], item = row_batch[row];
  if (position < 0 || item < 0 || uint32_t(item) >= items) return;
  const int32_t source_id = slot_ids[item];
  if (source_id < 0 || uint32_t(source_id) >= entries) return;

  int64_t previous = int64_t(row) - 1;
  while (previous >= 0 && row_batch[previous] == item && positions[previous] < 0)
    --previous;
  const bool first = previous < 0 || row_batch[previous] != item;
  const float* source = state + uint64_t(source_id) * (2 * Width);

  if (position & 1) {
    // An earlier row in this launch must be the other member of this
    // exact group. Holes are ignored, never treated as projected keys.
    if (!first && positions[previous] != position - 1) return;
    for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
      const uint64_t current = uint64_t(row) * Width + d;
      const float k0 = first ? source[d] : key[uint64_t(previous) * Width + d];
      const float g0 = first ? source[Width + d] : gate[uint64_t(previous) * Width + d];
      const float k1 = key[current], g1 = gate[current];
      const float m = fmaxf(g0, g1);
      const float e0 = expf(__fsub_rn(g0, m)), e1 = expf(__fsub_rn(g1, m));
      const float den = __fadd_rn(e0, e1);
      pooled[current] = __fadd_rn(__fmul_rn(__fdiv_rn(e0, den), k0),
                                 __fmul_rn(__fdiv_rn(e1, den), k1));
    }
    if (threadIdx.x == 0) output_positions[row] = position / 2;
  }
  // All incoming-tail reads precede the first CTA's possible overwrite.
  __syncthreads();
  if constexpr (MaxRows > 0) {
    if (position & 1) return;  // the checkpoint has no live tail
    uint32_t start = row;
    while (start > 0 && row_batch[start - 1] == item) --start;
    const uint32_t offset = row - start;
    if (offset >= MaxRows) return;
    const int32_t id = checkpoint_ids[uint64_t(item) * MaxRows + offset];
    if (id < 0 || uint32_t(id) >= entries) return;
    float* target = state + uint64_t(id) * (2 * Width);
    for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
      target[d] = key[uint64_t(row) * Width + d];
      target[Width + d] = gate[uint64_t(row) * Width + d];
    }
  } else if (first) {
    uint32_t last = row;
    for (uint32_t next = row + 1; next < live && row_batch[next] == item; ++next)
      if (positions[next] >= 0) last = next;
    if (positions[last] & 1) return;
    float* target = state + uint64_t(source_id) * (2 * Width);
    for (uint32_t d = threadIdx.x; d < Width; d += blockDim.x) {
      target[d] = key[uint64_t(last) * Width + d];
      target[Width + d] = gate[uint64_t(last) * Width + d];
    }
  }
}

// The pool, norm and its output stay in F32. Masked pooling emits only
// completed groups; a ratio-one norm is stateless and emits every live row.
template<int Width, bool Masked = true>
__device__ void norm(const float* pooled, const uint16_t* weight,
    const int32_t* output_positions, float* output,
    const uint32_t* live_rows, float eps) {
  static_assert(Width == 128 || Width == 512);
  ptx::griddep_launch_dependents();
  __shared__ float sums[4];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  if constexpr (Masked) if (output_positions[row] < 0) return;
  constexpr int Columns = Width / 128;
  float values[Columns], square = 0.f;
#pragma unroll
  for (int i = 0; i < Columns; ++i) {
    const uint32_t d = threadIdx.x + i * 128;
    values[i] = pooled[uint64_t(row) * Width + d];
    square = __fadd_rn(square, __fmul_rn(values[i], values[i]));
  }
  for (int offset = 16; offset; offset >>= 1)
    square = __fadd_rn(square, __shfl_down_sync(0xffffffffu, square, offset));
  if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = square;
  __syncthreads();
  if (threadIdx.x == 0)
    sums[0] = __fadd_rn(__fadd_rn(sums[0], sums[1]), __fadd_rn(sums[2], sums[3]));
  __syncthreads();
  const float inv = __frsqrt_rn(__fadd_rn(__fmul_rn(sums[0], 1.f / Width), eps));
#pragma unroll
  for (int i = 0; i < Columns; ++i) {
    const uint32_t d = threadIdx.x + i * 128;
    const float value = __fmul_rn(__fmul_rn(values[i], inv), convert::bf16_to_f32(weight[d]));
    output[uint64_t(row) * Width + d] = value;
  }
}

}  // namespace chop::gated_pooling
