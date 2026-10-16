#pragma once
// GLM 5.3 Flash's payload admission: integer checks of the scale and code
// bytes its engines decode, run once per placed view at create. Each entry
// scans `count` elements grid-stride and lowers `*first` (the host sets it
// to 0xffffffff) to the lowest index of an invalid element with an integer
// atomicMin, so the result does not depend on the schedule. No floating
// point arithmetic: every rule is a test of the element's bits. Launched
// with programmatic stream serialization, as every bundle launch is: each
// waits for the stream's earlier work (the report's upload) before reading.
#include <cstdint>

namespace chop::glm5next::admission {

__device__ __forceinline__ void begin() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void report(uint32_t* first, uint64_t i) {
  atomicMin(first, uint32_t(i < 0xffffffffull ? i : 0xfffffffeull));
}

// An F32 scale (FP8 block scale or NVFP4 global) is finite and not
// negative: an exponent of all ones (infinity, NaN) or a set sign bit on a
// nonzero magnitude is invalid; zero and negative zero are admitted.
__device__ __forceinline__ bool bad_f32(uint32_t bits) {
  return (bits & 0x7f800000u) == 0x7f800000u || bits > 0x80000000u;
}

// An NVFP4 UE4M3 scale has no sign bit and 0x7f is NaN: every code is at
// most 0x7e.
__device__ __forceinline__ bool bad_ue4m3(uint8_t code) { return code > 0x7e; }

// An MXFP8 E4M3 code with its UE8M0 scale `scale`: the code is not NaN
// (0x7f, 0xff), the scale is not 255, and code times 2^(scale - 127) is a
// finite BF16 equal to it. Every finite code is exact at scales 3 to 246;
// elsewhere the code's significand M (at most four bits) times 2^k is
// exact when its lowest set bit is at least 2^-133 (BF16's smallest
// subnormal) and its highest at most 2^127.
__device__ __forceinline__ bool bad_mxfp8(uint8_t code, uint8_t scale) {
  if ((code & 0x7f) == 0x7f || scale == 0xff) return true;
  if (scale >= 3 && scale <= 246) return false;
  const uint32_t magnitude = code & 0x7f;
  if (magnitude == 0) return false;
  const uint32_t e = magnitude >> 3, f = magnitude & 7;
  // value = M * 2^E: a subnormal E4M3 code is f * 2^-9, a normal one
  // (8 + f) * 2^(e - 10).
  uint32_t m = e == 0 ? f : (8 | f);
  int k = (e == 0 ? -9 : int(e) - 10) + int(scale) - 127;
  while ((m & 1) == 0) {
    m >>= 1;
    ++k;
  }
  const int top = k + (31 - __clz(m));
  return k < -133 || top > 127;
}

__device__ __forceinline__ uint64_t stride() { return uint64_t(gridDim.x) * blockDim.x; }
__device__ __forceinline__ uint64_t start() { return uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; }

__device__ void f32_scales(const uint32_t* words, uint64_t count, uint32_t* first) {
  begin();
  for (uint64_t i = start(); i < count; i += stride())
    if (bad_f32(words[i])) report(first, i);
}

__device__ void ue4m3_scales(const uint8_t* codes, uint64_t count, uint32_t* first) {
  begin();
  for (uint64_t i = start(); i < count; i += stride())
    if (bad_ue4m3(codes[i])) report(first, i);
}

// `count` E4M3 codes, one UE8M0 scale per 32 consecutive codes.
__device__ void mxfp8_payload(const uint8_t* codes, const uint8_t* scales, uint64_t count,
                              uint32_t* first) {
  begin();
  for (uint64_t i = start(); i < count; i += stride())
    if (bad_mxfp8(codes[i], scales[i / 32])) report(first, i);
}

}  // namespace chop::glm5next::admission
