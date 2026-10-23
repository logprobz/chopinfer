#pragma once
#include <cstdint>

namespace chop::gemm {

// Byte offset of 16-byte chunk c of 128-byte shared fold row r. Chunk c is
// stored at c ^ (r % 8), so the eight rows that one fragment load touches
// fall in distinct banks; unswizzled, all eight start in bank 0. Only the
// shared placement changes, never which bytes an MMA reads.
// CHOP_FOLD_PLAIN_ROWS keeps the unswizzled placement for lab references.
__device__ __forceinline__ uint32_t fold_row_chunk(uint32_t r, uint32_t c) {
#ifdef CHOP_FOLD_PLAIN_ROWS
  return r * 128 + c * 16;
#else
  return r * 128 + (c ^ (r % 8)) * 16;
#endif
}

} // namespace chop::gemm
