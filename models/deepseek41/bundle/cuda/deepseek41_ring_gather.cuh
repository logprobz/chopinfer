#pragma once
#include <stdint.h>

namespace chop::deepseek41 {
__device__ void ring_gather(const uint8_t* source, const uint32_t* slots,
                            const uint32_t* live, uint8_t* destination,
                            uint32_t bytes, uint32_t source_stride,
                            uint32_t destination_stride, uint32_t source_slots) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t item = blockIdx.x;
  if (item >= *live) return;
  const uint32_t slot = slots[item];
  if (slot >= source_slots) return;
  source += uint64_t(slot) * source_stride;
  destination += uint64_t(item) * destination_stride;
  if (((uint64_t(bytes | source_stride | destination_stride) |
        reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination)) & 15) == 0) {
    const uint4* from = reinterpret_cast<const uint4*>(source);
    uint4* to = reinterpret_cast<uint4*>(destination);
    for (uint32_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) to[i] = from[i];
  } else {
    for (uint32_t i = threadIdx.x; i < bytes; i += blockDim.x) destination[i] = source[i];
  }
}
} // namespace chop::deepseek41
