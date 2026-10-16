#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

namespace chop::probe {

// A stand-in for the kernels an L2 prefetch overlaps (a collective, a
// mixer): it releases its dependents first, as those kernels do, waits for
// its own predecessor, then stays resident for `ns` nanoseconds without
// touching memory.
__device__ void idle_window(uint64_t ns) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  uint64_t start, now;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  } while (now - start < ns);
}

} // namespace chop::probe
