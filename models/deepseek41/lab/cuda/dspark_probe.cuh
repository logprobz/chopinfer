#pragma once
#include <stdint.h>

namespace dspark_test {

// kernel_harness::poison::LiveProducer's descriptor ABI. Only the lab
// cubin contains this producer; it is never embedded into the bundle.
struct Operand {
  uint32_t* fresh;
  uint32_t* input;
  uint32_t words;
  uint32_t flags;
};

__device__ void produce(const uint4*, uint32_t*, uint64_t,
                        const Operand* operands, uint32_t count,
                        uint32_t* live, uint32_t rows, uint64_t tail_ns) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for (uint32_t n = 0; n < count; ++n) {
    const Operand op = operands[n];
    for (uint32_t i = threadIdx.x; i < op.words; i += blockDim.x) {
      if (op.flags & 1u) op.fresh[i] = op.input[i];
      op.input[i] = (op.flags & 2u) ? 0x7fc17fc1u : 0u;
    }
  }
  if (threadIdx.x == 0) *live = 0;
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t start, now;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  } while (now - start < tail_ns);
  for (uint32_t n = 0; n < count; ++n) {
    const Operand op = operands[n];
    for (uint32_t i = threadIdx.x; i < op.words; i += blockDim.x) op.input[i] = op.fresh[i];
  }
  if (threadIdx.x == 0) *live = rows;
}

} // namespace dspark_test
