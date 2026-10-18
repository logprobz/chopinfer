#pragma once
#include <stdint.h>

// Test-only producers implementing kernel_harness::poison's public ABI.
namespace chop::deepseek41::compressor_probe {
struct Operand {
  uint32_t* fresh;
  uint32_t* input;
  uint32_t words;
  uint32_t flags;
};
static_assert(sizeof(Operand) == 24);

__device__ void live(const uint4* eviction, uint32_t* checksum,
    uint64_t vectors, const Operand* operands, uint32_t operand_count,
    uint32_t* count, uint32_t rows, uint64_t tail_ns) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t lane = threadIdx.x;
  uint32_t sum = 0;
  for (uint64_t i = lane; i < vectors; i += blockDim.x) {
    const uint4 v = eviction[i];
    sum += v.x + v.y + v.z + v.w;
  }
  checksum[lane] = sum;
  for (uint32_t n = 0; n < operand_count; ++n) {
    const Operand op = operands[n];
    for (uint64_t i = lane; i < op.words; i += blockDim.x) {
      if (op.flags & 1u) op.fresh[i] = op.input[i];
      op.input[i] = (op.flags & 2u) ? 0x7fc17fc1u : 0u;
    }
  }
  if (lane == 0) *count = 0;
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t start, now;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  } while (now - start < tail_ns);
  for (uint32_t n = 0; n < operand_count; ++n) {
    const Operand op = operands[n];
    for (uint64_t i = lane; i < op.words; i += blockDim.x)
      op.input[i] = op.fresh[i];
  }
  if (lane == 0) *count = rows;
}

__device__ void copy(const uint32_t* source, uint32_t* target,
    uint64_t words, uint64_t cycles) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint64_t start = clock64();
  while (clock64() - start < cycles) {}
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = first; i < words; i += stride) target[i] = source[i];
}
}  // namespace chop::deepseek41::compressor_probe
