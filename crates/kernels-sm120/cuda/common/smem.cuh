#pragma once

// TMA shared-memory swizzle addresses for 16-byte sectors. The base must be
// 128-byte aligned, and row pitch must be a multiple of the swizzle span.

#include <stdint.h>
#include "ptx.cuh"

namespace chop::smem {

template<uint32_t SpanBytes>
__device__ __forceinline__ uint32_t swizzled_offset(
    const void* base, uint32_t byte_offset, uint32_t row_pitch) {
  static_assert(SpanBytes == 32 || SpanBytes == 64 || SpanBytes == 128);
  const uint32_t row = byte_offset / row_pitch;
  const uint32_t column = byte_offset % row_pitch;
  const uint32_t sector = column / 16;
  const uint32_t base_phase = (ptx::shared_address(base) / 128) &
                              (SpanBytes / 16 - 1);
  const uint32_t swizzled_sector = sector ^
                                    ((row + base_phase) & (SpanBytes / 16 - 1));
  return row * row_pitch + swizzled_sector * 16 + column % 16;
}

__device__ __forceinline__ uint32_t swizzle_32(
    const void* base, uint32_t byte_offset, uint32_t row_pitch) {
  return swizzled_offset<32>(base, byte_offset, row_pitch);
}
__device__ __forceinline__ uint32_t swizzle_64(
    const void* base, uint32_t byte_offset, uint32_t row_pitch) {
  return swizzled_offset<64>(base, byte_offset, row_pitch);
}
__device__ __forceinline__ uint32_t swizzle_128(
    const void* base, uint32_t byte_offset, uint32_t row_pitch) {
  return swizzled_offset<128>(base, byte_offset, row_pitch);
}

} // namespace chop::smem
