#pragma once
#include <stdint.h>

namespace chop::deepseek41 {
__device__ void retain_rows(const uint8_t* source, const uint32_t* segments,
                            uint8_t* destination, uint32_t row_bytes,
                            uint32_t source_rows, uint32_t destination_rows) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t* segment = segments + blockIdx.x * 3;
  const uint32_t from_row = segment[0], to_row = segment[1], rows = segment[2];
  // The host validates all segments and their disjoint destinations first.
  // These guards also keep malformed device metadata within buffer bounds.
  if (from_row > source_rows || rows > source_rows - from_row ||
      to_row > destination_rows || rows > destination_rows - to_row) return;
  source += uint64_t(from_row) * row_bytes;
  destination += uint64_t(to_row) * row_bytes;
  const uint64_t bytes = uint64_t(rows) * row_bytes;
  const uint64_t first = uint64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.y) * blockDim.x;
  if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes) & 15) == 0) {
    const uint4* from = reinterpret_cast<const uint4*>(source);
    uint4* to = reinterpret_cast<uint4*>(destination);
    for (uint64_t i = first; i < bytes / 16; i += stride) to[i] = from[i];
  } else {
    for (uint64_t i = first; i < bytes; i += stride) destination[i] = source[i];
  }
}
} // namespace chop::deepseek41
