#pragma once

// Lab probe for exhaustive live-count qualification (kernel-lab
// `device_check`): poisons buffers and compares outputs on the device, so a
// test can run every phase at every live count without copying each output
// to the host.
//
// `fill` writes `word`'s bytes, repeated from the first, over `bytes` bytes of
// a 16-byte aligned buffer.
// `compare` reads `vectors` 16-byte vectors of `got`: each 4-byte word below
// `live_words` must equal the same word of `want`, every live element (of
// `element_bytes`, 2 or 4) must differ from the poison element, so it was
// written, and each word at or past `live_words` must equal `poison`. It adds
// the count of differing live words to `counts[3 * slot]`, of live elements
// that still hold the poison to `counts[3 * slot + 1]` and of changed poison
// words to `counts[3 * slot + 2]`; `want` is read only below `live_words`.

#include <stdint.h>

namespace chop::probe::device_check {

__device__ void fill(uint8_t* buffer, uint64_t bytes, uint32_t word) {
  const uint4 v = make_uint4(word, word, word, word);
  const uint64_t vectors = bytes / 16;
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = first; i < vectors; i += stride) {
    reinterpret_cast<uint4*>(buffer)[i] = v;
  }
  for (uint64_t i = 16 * vectors + first; i < bytes; i += stride) {
    buffer[i] = uint8_t(word >> (8 * (i % 4)));
  }
}

__device__ void compare(const uint32_t* want, const uint4* got, uint64_t live_words,
                        uint64_t vectors, uint32_t poison, uint32_t element_bytes,
                        uint32_t* counts, uint32_t slot) {
  if (element_bytes != 2 && element_bytes != 4) __trap();
  uint32_t live_differ = 0;
  uint32_t unwritten = 0;
  uint32_t poison_changed = 0;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors; i += stride) {
    const uint4 g = got[i];
    const uint32_t words[4] = {g.x, g.y, g.z, g.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint64_t w = 4 * i + uint64_t(j);
      if (w < live_words) {
        live_differ += words[j] != want[w];
        if (element_bytes == 4) {
          unwritten += words[j] == poison;
        } else {
          unwritten += (words[j] & 0xffffu) == (poison & 0xffffu);
          unwritten += (words[j] >> 16) == (poison >> 16);
        }
      } else {
        poison_changed += words[j] != poison;
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    live_differ += __shfl_xor_sync(0xffffffffu, live_differ, offset);
    unwritten += __shfl_xor_sync(0xffffffffu, unwritten, offset);
    poison_changed += __shfl_xor_sync(0xffffffffu, poison_changed, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    if (live_differ != 0) atomicAdd(&counts[3 * slot], live_differ);
    if (unwritten != 0) atomicAdd(&counts[3 * slot + 1], unwritten);
    if (poison_changed != 0) atomicAdd(&counts[3 * slot + 2], poison_changed);
  }
}

}  // namespace chop::probe::device_check
