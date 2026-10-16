#pragma once

// Lab probe for the strip-chain stage-release race (tests/strip_chain_race.rs,
// FIX-SC1): a kernel that keeps every SM's shared-memory and global-store
// pipes busy for a while, run on a second stream beside a strip-chain launch
// so the GEMM's ldmatrix reads queue behind other traffic, as a busy GPU
// makes them.
//
// Each thread of a 128-thread CTA loops until `ns` nanoseconds have passed
// since it began: 8 shared stores and loads of its own eight words of a
// 4 KiB pad, then four 16-byte global stores into `sink` (`words` floats, a
// multiple of four, strided over the grid), so the store path stays
// saturated as the head's chain-slot stores saturate it. The values are
// never read.

#include <stdint.h>

namespace chop::probe::strip_race {

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

__device__ void contend(float* sink, uint64_t words, uint64_t ns) {
  __shared__ float storage[1024];
  // Volatile, so every access is a shared load or store.
  volatile float* pad = storage;
  const uint64_t start = now_ns();
  const uint32_t thread = threadIdx.x;
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + thread;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  float value = float(thread);
  float4* sink4 = reinterpret_cast<float4*>(sink);
  const uint64_t quads = words / 4;
  while (now_ns() - start < ns) {
#pragma unroll
    for (uint32_t r = 0; r < 8; ++r) {
      pad[r * 128 + thread] = value;
      value += pad[((r + 3) & 7) * 128 + thread];
    }
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r) {
      sink4[index % quads] = make_float4(value, value, value, value);
      index += stride;
    }
  }
}

} // namespace chop::probe::strip_race
