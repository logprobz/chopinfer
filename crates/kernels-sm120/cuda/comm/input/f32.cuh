#pragma once

// F32 all-reduce input (input/bf16.cuh describes the contract): four
// elements per 16-byte vector, summed as they are. The hyper-connection
// transports take it; their sums and all-gathers are F32 either way, and
// their inbox regions, strides and LL payloads follow the element size.

#include "../comm.cuh"

namespace chop::comm::input {

struct F32 {
  using Element = float;
  static constexpr uint32_t kPerVector = 4;  // elements per 16-byte vector
  static constexpr bool kStaged = false;

  __device__ __forceinline__ static float decode(float value) { return value; }
  __device__ __forceinline__ static void accumulate(Vector128 value, float (&sums)[kPerVector]) {
    sums[0] += __uint_as_float(value.x);
    sums[1] += __uint_as_float(value.y);
    sums[2] += __uint_as_float(value.z);
    sums[3] += __uint_as_float(value.w);
  }
};

} // namespace chop::comm::input
