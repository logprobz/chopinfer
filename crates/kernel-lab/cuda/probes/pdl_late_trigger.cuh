#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

namespace chop::probe {

// A PDL producer of one buffer that triggers its dependents late. After its
// own dependency wait it holds `lead_ns` before anything else, which leaves
// room for a dependent the host submits after it to be queued before it
// triggers; the lead does not guarantee that (the host may be slower, and
// resolving a lazily loaded entry after this launch waits it out). Then it
// fills `output` with NaN, fences, triggers its dependents, holds `tail_ns`
// and copies `fresh` into `output`. A dependent that PDL starts early and
// that reads `output` before its dependency wait reads NaN.
__device__ void pdl_late_trigger_producer(const uint32_t* fresh, uint32_t* output, uint64_t words,
                                          uint64_t lead_ns, uint64_t tail_ns) {
  ptx::griddep_wait();
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < lead_ns);
  for (uint64_t i = first; i < words; i += stride) output[i] = 0x7fc17fc1u;
  __threadfence();
  __syncthreads();
  ptx::griddep_launch_dependents();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < tail_ns);
  for (uint64_t i = first; i < words; i += stride) output[i] = fresh[i];
}

// The control's dependent: it reads `input` before its PDL wait and stores
// what it read into `seen` after it, so `seen` holds the producer's NaN
// where PDL let it start before the producer wrote.
__device__ void pdl_early_reader(const uint32_t* input, uint32_t* seen, uint64_t words) {
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint32_t read[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const uint64_t i = first + j * stride;
    read[j] = i < words ? *reinterpret_cast<const volatile uint32_t*>(input + i) : 0u;
  }
  ptx::griddep_wait();
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const uint64_t i = first + j * stride;
    if (i < words) seen[i] = read[j];
  }
}

} // namespace chop::probe
