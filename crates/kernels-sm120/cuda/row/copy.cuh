#pragma once
// Row copies (`kernel_core::row_copy`): element bytes moved between row
// layouts, nothing computed. One CTA per row of the planned bucket; rows at
// or past the device live count return. The column slice is
// `precision::launch::copy_rows` over 16-bit words (`row::copy`).
#include "row.cuh"

namespace chop::row::copy {

// destination[row][s] = source[row] for each of `streams` streams, in
// 16-byte vectors: `[rows, vectors]` into `[rows, streams, vectors]`.
__device__ void broadcast_streams(const uint4* source, uint4* destination,
                                  const uint32_t* live_rows, uint32_t vectors,
                                  uint32_t streams) {
  begin();
  const uint64_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint32_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    const uint4 data = source[row * vectors + v];
    for (uint32_t s = 0; s < streams; ++s)
      destination[(row * streams + s) * vectors + v] = data;
  }
}

} // namespace chop::row::copy
