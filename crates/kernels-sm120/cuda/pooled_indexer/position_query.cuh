#pragma once

// F32 tail rotation. The next operation is the shared trained quantizer,
// which owns the single BF16 rounding and every discrete code decision.
#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::position_query {

template<int Heads, int Dim, int Rotary, bool IgnoreBounds = false>
__device__ void rotate(const float* input, const float2* cos_sin,
                       const int32_t* positions, const uint32_t* live_rows,
                       float* output, uint32_t* status, uint64_t status_row_offset,
                       uint32_t table_positions) {
  static_assert(Dim > 0 && Dim % 32 == 0 && Rotary > 0 && Rotary <= Dim && Rotary % 2 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  int32_t position = positions[row];
  const bool invalid = position < 0 || uint32_t(position) >= table_positions;
  const size_t head = size_t(row) * Heads + blockIdx.y;
  if (invalid) {
    if constexpr (IgnoreBounds) {
      // The lab mutant stays memory-safe but hides the invalid position.
      position = 0;
    } else {
      if (blockIdx.y == 0 && threadIdx.x == 0)
        atomicOr(status + status_row_offset + row, uint32_t(8));
      for (int channel = threadIdx.x; channel < Dim; channel += blockDim.x)
        output[head * Dim + channel] = 0.f;
      return;
    }
  }
  for (int pair = threadIdx.x; pair < Dim / 2; pair += blockDim.x) {
    const size_t at = head * Dim + 2 * pair;
    const float a = input[at];
    const float b = input[at + 1];
    if (pair < (Dim - Rotary) / 2) {
      output[at] = a;
      output[at + 1] = b;
    } else {
      const float2 cs = cos_sin[size_t(position) * (Rotary / 2) + pair - (Dim - Rotary) / 2];
      output[at] = __fsub_rn(__fmul_rn(a, cs.x), __fmul_rn(b, cs.y));
      output[at + 1] = __fadd_rn(__fmul_rn(a, cs.y), __fmul_rn(b, cs.x));
    }
  }
}

} // namespace chop::position_query
