#pragma once

#include "convert.cuh"

namespace chop::convert {

// Every E2M1 code is exact in BF16 at UE8M0 bytes 0 through 252.
// At byte 253 magnitudes 4 and 6 overflow; at 254, magnitudes 2 and above.
// Checking the ordered magnitude codes also preserves both signed zeros.
__device__ __forceinline__ void check_mxfp4(uint8_t codes, uint8_t scale) {
  if (scale > 252) {
    const unsigned limit = scale == 253 ? 6 : 4;
    if (scale == 255 || (codes & 7) >= limit || ((codes >> 4) & 7) >= limit)
      asm volatile("trap;");
  }
}

template<bool Check = true>
__device__ __forceinline__ uint32_t scaled_mxfp4_bf16x2(
    uint8_t codes, uint8_t scale) {
  uint32_t result;
  const uint16_t code_pair = codes;
  const uint16_t scale_pair = uint16_t(scale) * 0x0101;
  // The conversion is pure; only the separate trap needs to be volatile.
  asm("{ .reg .b8 c, unused; mov.b16 {c, unused}, %1; "
      "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c, %2; }"
      : "=r"(result) : "h"(code_pair), "h"(scale_pair));
  if constexpr (Check) check_mxfp4(codes, scale);
  return result;
}

} // namespace chop::convert
