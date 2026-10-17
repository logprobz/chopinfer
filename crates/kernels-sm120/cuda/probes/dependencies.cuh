#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"

namespace chop::probes {

// ABI of kernel_harness::poison::Rewrites.
struct Operand {
  uint32_t* fresh;
  uint32_t* input;
  uint32_t words;
  uint32_t flags;
};
static_assert(sizeof(Operand) == 24);

__device__ void copy(const uint32_t* src, uint32_t* dst, uint64_t words, uint64_t spin) {
  ptx::griddep_launch_dependents();
  const long long start = clock64();
  while (clock64() - start < (long long)spin) {}
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += uint64_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

// Publish the new inputs and live count after releasing dependent grids.
// A consumer reading either before its PDL wait observes the poison.
__device__ void live(const uint4* eviction, uint32_t* checksum, uint64_t vectors,
    const Operand* operands, uint32_t operand_count, uint32_t* count,
    uint32_t rows, uint64_t tail_ns) {
  ptx::griddep_wait();
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint32_t sum = 0;
  for (uint64_t i = first; i < vectors; i += stride) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(eviction + i) : "memory");
    sum += v.x + v.y + v.z + v.w;
  }
  checksum[first] = sum;
  for (uint32_t n = 0; n < operand_count; ++n) {
    const auto op = operands[n];
    for (uint64_t i = first; i < op.words; i += stride) {
      if (op.flags & 1u) op.fresh[i] = op.input[i];
      op.input[i] = (op.flags & 2u) ? 0x7fc17fc1u : 0u;
    }
  }
  if (first == 0) *count = 0;
  __syncthreads();
  ptx::griddep_launch_dependents();
  uint64_t start, now;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));
  do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now)); } while (now - start < tail_ns);
  for (uint32_t n = 0; n < operand_count; ++n) {
    const auto op = operands[n];
    for (uint64_t i = first; i < op.words; i += stride) op.input[i] = op.fresh[i];
  }
  if (first == 0) *count = rows;
}

} // namespace chop::probes
