#pragma once
// The gather by index (`kernel_core::row_copy::GatherByIndex`): the source
// row each U32 row index names, copied in 16-byte vectors into a dense
// destination. One CTA per row of the planned bucket; rows at or past the
// device count return.
#include "row.cuh"

namespace chop::row::gather {

// destination[row] = source[row_indices[row]] for the first `*count` rows,
// rows of `vectors` 16-byte vectors.
__device__ void by_index(const uint4* source, const uint32_t* row_indices,
                         const uint32_t* count, uint4* destination, uint32_t vectors) {
  begin();
  const uint64_t row = blockIdx.x;
  if (row >= *count) return;
  const uint64_t from = row_indices[row];
  for (uint32_t v = threadIdx.x; v < vectors; v += blockDim.x)
    destination[row * vectors + v] = source[from * vectors + v];
}

} // namespace chop::row::gather
