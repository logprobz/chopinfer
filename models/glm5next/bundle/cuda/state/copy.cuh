#pragma once
// GLM 5.3 Flash's commit copies of layered request state: one launch copies
// `bytes` bytes in each of `gridDim.y` layers, from `source + layer *
// source_stride` to `destination + layer * destination_stride`, as one host
// copy per layer would. Byte for byte, no arithmetic: 16-byte words when
// every address, stride and the length are multiples of 16, else bytes.
// Each layer's source and destination do not overlap. Launched with
// programmatic stream serialization, as every bundle launch is: it waits
// for the stream's earlier work before reading.
#include <cstdint>

namespace chop::glm5next::state {

__device__ void copy_layers(const uint8_t* source, uint8_t* destination,
                            uint64_t source_stride, uint64_t destination_stride,
                            uint64_t bytes) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint8_t* from = source + uint64_t(blockIdx.y) * source_stride;
  uint8_t* to = destination + uint64_t(blockIdx.y) * destination_stride;
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = uint64_t(gridDim.x) * blockDim.x;
  const uint64_t aligned = reinterpret_cast<uint64_t>(source) |
                           reinterpret_cast<uint64_t>(destination) | source_stride |
                           destination_stride | bytes;
  if ((aligned & 15) == 0) {
    const uint4* words = reinterpret_cast<const uint4*>(from);
    uint4* out = reinterpret_cast<uint4*>(to);
    for (uint64_t i = first; i < bytes / 16; i += step) out[i] = words[i];
  } else {
    for (uint64_t i = first; i < bytes; i += step) to[i] = from[i];
  }
}

}  // namespace chop::glm5next::state
