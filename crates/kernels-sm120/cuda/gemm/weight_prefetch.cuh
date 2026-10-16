#pragma once

// L2 prefetch of persistent byte ranges (weights) ahead of the kernels that
// read them (`gemm::prefetch`). Launched as a PDL dependent in a chain of
// kernels, the grid releases its own dependents first, asks L2 for every
// range while the kernel it depends on still runs, and only then waits for
// that kernel: so its completion still implies its predecessor's, and the
// kernel after it starts and finishes as it would without it. It writes
// nothing and reads nothing into registers; the ranges must not be written
// by any kernel of the chain (weights).
//
// The ranges are split into `chunk`-byte pieces numbered across ranges in
// order; piece g goes to thread g mod (grid threads), so the earliest
// ranges' requests issue first. `Lines`: one `prefetch.global.L2` per
// 128-byte line of a piece; otherwise one `cp.async.bulk.prefetch.L2` per
// piece (16-byte aligned starts and lengths).

#include <stdint.h>
#include "../common/ptx.cuh"

namespace chop::gemm {

constexpr int kPrefetchRanges = 8;

// Deliberate defects kernel-lab instantiates (`weight_prefetch`'s probe).
enum class PrefetchMutation : int {
  none = 0,
  no_wait = 1,  // never waits for the kernel it depends on
};

struct PrefetchRanges {
  const unsigned char* base[kPrefetchRanges];
  uint64_t bytes[kPrefetchRanges];
};

template<bool Lines, PrefetchMutation Mutation = PrefetchMutation::none>
__device__ void prefetch_ranges_l2(const PrefetchRanges& ranges, uint32_t count,
                                   uint32_t chunk) {
  ptx::griddep_launch_dependents();
  const uint64_t threads = uint64_t(gridDim.x) * blockDim.x;
  const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t start = 0;
#pragma unroll
  for (int r = 0; r < kPrefetchRanges; ++r) {
    if (uint32_t(r) < count) {
      const uint64_t bytes = ranges.bytes[r];
      const uint64_t pieces = (bytes + chunk - 1) / chunk;
      // This thread's first piece of the range: global piece start + p
      // with (start + p) mod threads == thread.
      const uint64_t skew = start % threads;
      for (uint64_t p = (thread + threads - skew) % threads; p < pieces; p += threads) {
        const uint64_t offset = p * chunk;
        const uint64_t size = bytes - offset < chunk ? bytes - offset : chunk;
        const unsigned char* address = ranges.base[r] + offset;
        if constexpr (Lines) {
          for (uint64_t line = 0; line < size; line += 128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(address + line));
        } else {
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                       :: "l"(address), "r"(uint32_t(size)) : "memory");
        }
      }
      start += pieces;
    }
  }
  if constexpr (Mutation != PrefetchMutation::no_wait) ptx::griddep_wait();
}

} // namespace chop::gemm
