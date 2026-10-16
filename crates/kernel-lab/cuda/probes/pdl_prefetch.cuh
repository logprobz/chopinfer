#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

namespace chop::probe {

// Descriptor layout is two 64-bit pointers and two 32-bit integers.
struct PdlOperand {
  uint32_t* fresh;
  uint32_t* input;
  uint32_t words;
  uint32_t flags; // bit 0: preserve evolving state; bit 1: floating poison
};
static_assert(sizeof(PdlOperand) == 24);

// Stream a buffer larger than L2, then give the consumer a producer tail in
// which to stage constants. Publish changing input and live count only after
// that tail. A consumer that moves either load before its wait sees poison.
__device__ void pdl_prefetch_producer(
    const uint4* eviction, uint32_t* checksum, uint64_t vectors,
    const PdlOperand* operands, uint32_t operand_count,
    uint32_t* live, uint32_t rows, uint64_t tail_ns) {
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
  if (first == 0) *live = 0;
  __syncthreads();
  ptx::griddep_launch_dependents();
  uint64_t start, now;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  } while (now - start < tail_ns);
  for (uint32_t n = 0; n < operand_count; ++n) {
    const auto op = operands[n];
    for (uint64_t i = first; i < op.words; i += stride) op.input[i] = op.fresh[i];
  }
  if (first == 0) *live = rows;
}

} // namespace chop::probe
