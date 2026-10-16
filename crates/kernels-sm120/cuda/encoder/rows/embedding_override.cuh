#pragma once

// The language model's embedding rows replaced by encoder arena rows (`layout.tokens.v1`).

#include "common.cuh"

namespace chop::multimodal {

template<int Width>
__device__ void embedding_override(uint16_t* activation,
                                   const uint16_t* arena,
                                   const int64_t* source,
                                   const uint32_t* live_rows,
                                   uint64_t arena_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int64_t src = source[row];
  if (src < 0 || uint64_t(src) >= arena_rows) return;
  static_assert(Width % 8 == 0);
  for (int col = threadIdx.x * 8; col < Width; col += blockDim.x * 8) {
    const uint4 value = reinterpret_cast<const uint4*>(
        arena + uint64_t(src) * Width + col)[0];
    reinterpret_cast<uint4*>(activation + uint64_t(row) * Width + col)[0] = value;
  }
}

} // namespace chop::multimodal
