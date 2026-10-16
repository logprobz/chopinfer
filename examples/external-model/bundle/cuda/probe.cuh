#pragma once
#include "common/sigmoid.cuh"

namespace example {
__device__ __forceinline__ void probe(float* result, float value) {
  if (threadIdx.x == 0) result[0] = chop::sigmoid::sigmoid(value);
}
}
