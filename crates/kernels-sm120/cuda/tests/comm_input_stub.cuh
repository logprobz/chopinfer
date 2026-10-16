#pragma once

// An input type of 8-byte elements for compile tests
// (tests/comm_extension.rs): two per 16-byte vector, whose F32 sums do not
// fill a whole vector, so the hyper-connection transports must refuse it.

#include "comm/comm.cuh"

namespace stub {
struct F64 {
  using Element = double;
  static constexpr uint32_t kPerVector = 2;
  static constexpr bool kStaged = false;
  __device__ __forceinline__ static float decode(double value) { return float(value); }
  __device__ __forceinline__ static void accumulate(chop::comm::Vector128 value,
                                                    float (&sums)[kPerVector]) {
    sums[0] += float(__hiloint2double(value.y, value.x));
    sums[1] += float(__hiloint2double(value.w, value.z));
  }
};
}
