#pragma once
#include "row.cuh"

namespace chop::row::mxfp8_decode {

// E4M3 times a power of two has at most four significant bits. Normal
// BF16 results are exact; only BF16 subnormals need rounding. Integer
// exponent arithmetic also preserves F32 subnormals independently of FTZ.
// Truncate is a lab negative control, never a serving entry.
template<bool Truncate = false>
__device__ __forceinline__ uint16_t value(uint8_t code, uint8_t scale) {
  const uint32_t magnitude = code & 0x7f;
  if (magnitude == 0x7f || scale == 0xff) return 0x7fc0;
  const uint16_t sign = uint16_t(code & 0x80) << 8;
  const int exponent = magnitude >> 3;
  const uint32_t significand = exponent ? 8 + (magnitude & 7) : magnitude;
  if (significand == 0) return sign;
  const int power = (exponent ? exponent - 10 : -9) + int(scale) - 127;
  const int high = 31 - __clz(significand);
  const int normalized = power + high;
  if (normalized > 127) return sign | 0x7f80;
  if (normalized >= -126)
    return sign | ((normalized + 127) << 7) | ((significand << (7 - high)) & 0x7f);
  // One BF16 subnormal unit is 2^-133. The right shift is at most three.
  const int shift = power + 133;
  if (shift >= 0) return sign | (significand << shift);
  const int right = -shift;
  uint32_t rounded = significand >> right;
  if constexpr (!Truncate) {
    const uint32_t remainder = significand & ((1u << right) - 1);
    const uint32_t half = 1u << (right - 1);
    rounded += remainder > half || (remainder == half && (rounded & 1));
  }
  return sign | rounded;
}

template<bool Truncate = false>
__device__ void rows(const uint8_t* codes, const uint8_t* scales,
                    const uint32_t* live_rows, uint16_t* output, uint32_t width) {
  begin();
  const uint64_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint64_t column = threadIdx.x; column < width; column += blockDim.x) {
    const uint64_t at = row * width + column;
    output[at] = value<Truncate>(codes[at], scales[row * (width / 32) + column / 32]);
  }
}

} // namespace chop::row::mxfp8_decode
