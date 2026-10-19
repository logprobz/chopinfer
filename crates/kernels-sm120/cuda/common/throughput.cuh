#pragma once
#include "ptx.cuh"

// Register-only dense BF16 MMA ceiling. Eight independent accumulator
// streams expose issue throughput; no sparsity or native low-bit MMA.
namespace chop::throughput {
__device__ void bf16(float* output, const uint32_t* input, uint32_t iterations) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const ptx::A4 a{{input[0],input[1],input[2],input[3]}};
  const ptx::B2 b{{input[4],input[5]}};
  ptx::D4 d[8] = {};
  for (uint32_t i = 0; i < iterations; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) ptx::mma_bf16(d[j],a,b);
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) sum += d[j].v[0];
  output[uint64_t(blockIdx.x)*blockDim.x+threadIdx.x] = sum;
}

__device__ void read(uint32_t* output, const uint4* input, uint64_t vectors) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint64_t thread = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  uint32_t sum = 0;
  for (uint64_t i = thread; i < vectors; i += uint64_t(gridDim.x)*blockDim.x) {
    const uint4 v = input[i];
    sum ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  output[thread] = sum;
}
} // namespace chop::throughput
