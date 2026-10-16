#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

namespace chop::probe {

// A PDL producer of one buffer that holds its write until its dependent has
// written a word it writes only after reading that buffer, and records
// whether it did: pdl_watch.cuh's producer for a dependent that releases no
// counter. After its own dependency wait every CTA fills its share of
// `output` with NaN and reads `*watch` (the dependent cannot have started:
// it launches only after the trigger below), triggers its dependents, then
// holds until `*watch` differs from what it read or `hold_ns` passes. The
// watched word is one the dependent writes only from `output` (the split
// epilogue's round of its first sums), poisoned before the launch, so it
// changing first means the dependent read `output` before this producer
// wrote it. CTA 0 records: record[0] is 1 then and 0 after a full hold;
// record[1] the nanoseconds held (saturated). Only then does each CTA copy
// its share of `fresh` into `output`.
__device__ void pdl_change_watch_producer(const uint32_t* fresh, uint32_t* output, uint64_t words,
                                          const unsigned int* watch, uint64_t hold_ns,
                                          uint32_t* record) {
  __shared__ unsigned int before;
  ptx::griddep_wait();
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = first; i < words; i += stride) output[i] = 0x7fc17fc1u;
  if (threadIdx.x == 0) before = *reinterpret_cast<const volatile unsigned int*>(watch);
  __threadfence();
  __syncthreads();
  ptx::griddep_launch_dependents();
  if (threadIdx.x == 0) {
    uint64_t start, now;
    uint32_t seen = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
      uint32_t value;
      asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(watch) : "memory");
      seen = value != before;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (!seen && now - start < hold_ns);
    if (blockIdx.x == 0) {
      record[0] = seen;
      record[1] = static_cast<uint32_t>(now - start < 0xffffffffull ? now - start : 0xffffffffull);
    }
  }
  __syncthreads();
  for (uint64_t i = first; i < words; i += stride) output[i] = fresh[i];
}

} // namespace chop::probe
