#pragma once

// F32 rows rounded to BF16 (`cast.bf16.v1`): where an encoder's F32 rows
// become the language model's BF16 rows, after a last GEMM that writes F32.

#include "common.cuh"

namespace chop::multimodal {

// Each element of F32 rows [Width] rounded once to nearest even into BF16
// rows [Width] (cvt.rn, which keeps subnormals and rounds overflow to
// infinity); every NaN becomes the canonical 0x7fff. Every element of a
// live row is written.
template<int Width>
__device__ void cast_bf16_f32(const float* input, uint16_t* output,
                              const uint32_t* live_rows) {
  static_assert(Width > 0, "a cast reads rows");
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float* from = input + uint64_t(row) * Width;
  uint16_t* to = output + uint64_t(row) * Width;
  // Not unrolled: the unrolled loop's trip count is a float division.
#pragma unroll 1
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    const float x = from[col];
    to[col] = isnan(x) ? uint16_t(0x7fff) : to_bf(x);
  }
}

} // namespace chop::multimodal
