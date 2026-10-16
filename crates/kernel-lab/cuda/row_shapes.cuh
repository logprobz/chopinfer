#pragma once
// Device-side poison, live counts and byte comparison for the row-stage
// shape-form sweeps (tests/projection_row_shapes.rs). A sweep runs every live
// count of a stage's capacity; filling and comparing whole outputs on the
// device keeps it from copying them to the host at each count.
#include <stdint.h>

namespace chop::lab::row_shapes {

__device__ __forceinline__ uint64_t first() {
  return uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ uint64_t stride() {
  return uint64_t(gridDim.x) * blockDim.x;
}

// Every byte of `output` holds `value`.
__device__ void fill(uint8_t* output, uint64_t bytes, uint32_t value) {
  for (uint64_t i = first(); i < bytes; i += stride()) output[i] = uint8_t(value);
}

// The u32 count a stage reads from device memory.
__device__ void store(uint32_t* output, uint32_t value) {
  if (first() == 0) *output = value;
}

// Adds the number of bytes at which `a` and `b` differ to `*count`.
__device__ void count_differences(const uint8_t* a, const uint8_t* b, uint64_t bytes,
                                  unsigned long long* count) {
  unsigned long long differing = 0;
  for (uint64_t i = first(); i < bytes; i += stride()) differing += a[i] != b[i];
  for (int offset = 16; offset; offset >>= 1)
    differing += __shfl_down_sync(0xffffffff, differing, offset);
  if ((threadIdx.x & 31) == 0 && differing) atomicAdd(count, differing);
}

} // namespace chop::lab::row_shapes
