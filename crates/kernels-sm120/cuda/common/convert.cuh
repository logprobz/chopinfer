#pragma once

// Packed SM12x numeric conversions. Packed results are raw carrier bits;
// e2m1x2 uses the low byte, FP6 pairs use the low twelve bits.

#include <stdint.h>

namespace chop::convert {

struct Float2 { float lo, hi; };

__device__ __forceinline__ uint16_t bf16_from_f32(float value) {
  uint16_t bits;
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  return bits;
}
__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
  float value;
  asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(value) : "h"(bits));
  return value;
}
__device__ __forceinline__ uint32_t bf16x2_from_f32(Float2 value) {
  return uint32_t(bf16_from_f32(value.lo)) |
         (uint32_t(bf16_from_f32(value.hi)) << 16);
}
__device__ __forceinline__ Float2 bf16x2_to_f32(uint32_t bits) {
  return {bf16_to_f32(uint16_t(bits)),
          bf16_to_f32(uint16_t(bits >> 16))};
}

#define CHOP_PACKED_CONVERSION(FORMAT)                                         \
__device__ __forceinline__ uint16_t pack_f32_##FORMAT(Float2 v) {             \
  uint16_t bits;                                                               \
  asm volatile("cvt.rn.satfinite." #FORMAT "x2.f32 %0, %2, %1;"               \
               : "=h"(bits) : "f"(v.lo), "f"(v.hi));                          \
  return bits;                                                                  \
}                                                                              \
__device__ __forceinline__ uint16_t pack_f16x2_##FORMAT(uint32_t v) {         \
  uint16_t bits;                                                               \
  asm volatile("cvt.rn.satfinite." #FORMAT "x2.f16x2 %0, %1;"                 \
               : "=h"(bits) : "r"(v));                                        \
  return bits;                                                                  \
}                                                                              \
__device__ __forceinline__ uint16_t pack_bf16x2_##FORMAT(uint32_t v) {        \
  uint16_t bits;                                                               \
  asm volatile("cvt.rn.satfinite." #FORMAT "x2.bf16x2 %0, %1;"                \
               : "=h"(bits) : "r"(v));                                        \
  return bits;                                                                  \
}                                                                              \
__device__ __forceinline__ uint32_t unpack_f16x2_##FORMAT(uint16_t v) {       \
  uint32_t bits;                                                               \
  asm volatile("cvt.rn.f16x2." #FORMAT "x2 %0, %1;"                            \
               : "=r"(bits) : "h"(v));                                        \
  return bits;                                                                  \
}                                                                              \
__device__ __forceinline__ uint32_t unpack_bf16x2_##FORMAT(uint16_t v) {      \
  uint32_t bits;                                                               \
  asm volatile("cvt.rn.bf16x2." #FORMAT "x2 %0, %1;"                           \
               : "=r"(bits) : "h"(v));                                        \
  return bits;                                                                  \
}                                                                              \
__device__ __forceinline__ Float2 unpack_f32_##FORMAT(uint16_t v) {           \
  return bf16x2_to_f32(unpack_bf16x2_##FORMAT(v));                              \
}

CHOP_PACKED_CONVERSION(e4m3)
CHOP_PACKED_CONVERSION(e5m2)
CHOP_PACKED_CONVERSION(e2m3)
CHOP_PACKED_CONVERSION(e3m2)
#undef CHOP_PACKED_CONVERSION

__device__ __forceinline__ uint16_t pack_f32_e2m1(Float2 v) {
  uint16_t bits;
  asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; "
               "mov.b16 %0, {t, 0}; }"
               : "=h"(bits) : "f"(v.lo), "f"(v.hi));
  return bits;
}
__device__ __forceinline__ uint16_t pack_f16x2_e2m1(uint32_t v) {
  uint16_t bits;
  asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, %1; "
               "mov.b16 %0, {t, 0}; }"
               : "=h"(bits) : "r"(v));
  return bits;
}
__device__ __forceinline__ uint16_t pack_bf16x2_e2m1(uint32_t v) {
  uint16_t bits;
  asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.bf16x2 t, %1; "
               "mov.b16 %0, {t, 0}; }"
               : "=h"(bits) : "r"(v));
  return bits;
}
__device__ __forceinline__ uint32_t unpack_f16x2_e2m1(uint16_t v) {
  uint32_t bits;
  asm volatile("{ .reg .b8 t, unused; mov.b16 {t, unused}, %1; "
               "cvt.rn.f16x2.e2m1x2 %0, t; }"
               : "=r"(bits) : "h"(v));
  return bits;
}
__device__ __forceinline__ uint32_t unpack_bf16x2_e2m1(uint16_t v) {
  uint32_t bits;
  asm volatile("{ .reg .b8 t, unused; mov.b16 {t, unused}, %1; "
               "cvt.rn.bf16x2.e2m1x2 %0, t; }"
               : "=r"(bits) : "h"(v));
  return bits;
}
__device__ __forceinline__ Float2 unpack_f32_e2m1(uint16_t v) {
  return bf16x2_to_f32(unpack_bf16x2_e2m1(v));
}

__device__ __forceinline__ uint16_t pack_f32_ue8m0(Float2 v) {
  uint16_t bits;
  asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %2, %1;"
               : "=h"(bits) : "f"(v.lo), "f"(v.hi));
  return bits;
}
__device__ __forceinline__ uint16_t pack_f32_ue8m0_rp(Float2 v) {
  uint16_t bits;
  asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %2, %1;"
               : "=h"(bits) : "f"(v.lo), "f"(v.hi));
  return bits;
}
__device__ __forceinline__ uint16_t pack_bf16x2_ue8m0(uint32_t v) {
  uint16_t bits;
  asm volatile("cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1;"
               : "=h"(bits) : "r"(v));
  return bits;
}
__device__ __forceinline__ uint16_t pack_f16x2_ue8m0(uint32_t v) {
  // PTX 9.2 has no direct ue8m0x2.f16x2 conversion on SM120.
  uint16_t lo = uint16_t(v), hi = uint16_t(v >> 16);
  float a, b;
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
  return pack_f32_ue8m0({a, b});
}
__device__ __forceinline__ uint32_t unpack_bf16x2_ue8m0(uint16_t v) {
  uint32_t bits;
  asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;"
               : "=r"(bits) : "h"(v));
  return bits;
}
__device__ __forceinline__ uint32_t unpack_f16x2_ue8m0(uint16_t v) {
  // PTX 9.2 has no direct f16x2.ue8m0x2 conversion on SM120.
  Float2 f = bf16x2_to_f32(unpack_bf16x2_ue8m0(v));
  uint16_t lo, hi;
  asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(f.lo));
  asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(f.hi));
  return uint32_t(lo) | (uint32_t(hi) << 16);
}
__device__ __forceinline__ Float2 unpack_f32_ue8m0(uint16_t v) {
  return bf16x2_to_f32(unpack_bf16x2_ue8m0(v));
}

} // namespace chop::convert
