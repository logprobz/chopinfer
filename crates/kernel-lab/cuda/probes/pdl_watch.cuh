#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

namespace chop::probe {

// A PDL producer of one buffer that holds its write until its dependent has
// got ahead of it, and records whether it did. After its own dependency
// wait it fills `output` with NaN and reads the dependent's epoch `*epoch`
// (the dependent cannot have started: it launches only after the trigger
// below), triggers its dependents, then holds until `*flag` reaches that
// epoch + 1 or `hold_ns` passes. The flag is one the dependent releases only
// after reading `output` (a one-shot all-reduce's data flag to itself), so
// it reaching the value first means the dependent read `output` before this
// producer wrote it. record[0] is 1 then and 0 after a full hold; record[1]
// the nanoseconds held (saturated). Only then does it copy `fresh` into
// `output`.
__device__ void pdl_watch_producer(const uint32_t* fresh, uint32_t* output, uint64_t words,
                                   const unsigned int* epoch, const unsigned int* flag,
                                   uint64_t hold_ns, uint32_t* record) {
  __shared__ unsigned int expected;
  ptx::griddep_wait();
  for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) output[i] = 0x7fc17fc1u;
  if (threadIdx.x == 0) expected = *reinterpret_cast<const volatile unsigned int*>(epoch) + 1;
  __threadfence();
  __syncthreads();
  ptx::griddep_launch_dependents();
  if (threadIdx.x == 0) {
    uint64_t start, now;
    uint32_t seen = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
      uint32_t value;
      asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
      seen = value == expected;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (!seen && now - start < hold_ns);
    record[0] = seen;
    record[1] = static_cast<uint32_t>(now - start < 0xffffffffull ? now - start : 0xffffffffull);
  }
  __syncthreads();
  for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) output[i] = fresh[i];
}

} // namespace chop::probe
