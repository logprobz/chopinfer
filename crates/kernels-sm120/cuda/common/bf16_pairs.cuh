#pragma once

// BF16 pairs of F32 values for BF16 tensor-core operands: hi = bf16(x) and
// lo = bf16(x - hi), both rounded to nearest even. x - hi is exact in F32,
// so hi + lo is within 2^-16 |x| of x. A product with a BF16 operand w is
// taken as hi * w + lo * w with F32 accumulation.

#include <stdint.h>

namespace chop::pairs {

// Two values as packed BF16, `low` in bits 0-15.
__device__ __forceinline__ uint32_t pack(float low, float high) {
  uint32_t bits;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(high), "f"(low));
  return bits;
}
__device__ __forceinline__ float low(uint32_t bits) {
  return __uint_as_float(bits << 16);
}
__device__ __forceinline__ float high(uint32_t bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

struct Packed { uint32_t hi, lo; };

// The pairs of two values, each half packed as `pack` does.
__device__ __forceinline__ Packed split(float x, float y) {
  const uint32_t hi = pack(x, y);
  return {hi, pack(x - low(hi), y - high(hi))};
}

// One value's pair as BF16 bit patterns.
__device__ __forceinline__ void split(float x, uint16_t& hi, uint16_t& lo) {
  const uint32_t bits = pack(x, 0.0f);
  hi = uint16_t(bits);
  lo = uint16_t(pack(x - low(bits), 0.0f));
}

} // namespace chop::pairs
