#pragma once
#include <stdint.h>

// The weight and activation code checks of the quad-lane gate queues
// (chained_lanes.cuh), four codes or scales per word, shared by the device
// code and the host test (tests/chained_lanes_checks.cpp). They accept and
// refuse exactly what the row-tiled queues' per-byte checks do.
#if defined(__CUDACC__)
#define CHOP_LANES_INLINE __host__ __device__ __forceinline__
#else
#define CHOP_LANES_INLINE inline
#endif
namespace chop::moe::lanes::checks {

// Nonzero when any byte of four UE8M0 scales is above 252: a byte of
// `~scales` below 3, found without a borrow from a lower byte.
CHOP_LANES_INLINE uint32_t extreme_scales(uint32_t scales) {
  return (~scales - 0x03030303u) & scales & 0x80808080u;
}

// Whether eight E2M1 codes of one 32-K group are valid at its UE8M0 scale,
// as streamed::check decides for the group's 32 codes: 255 is never valid,
// 253 forbids magnitudes 6 and 7, 254 the magnitude's bit 2.
CHOP_LANES_INLINE bool group_valid(uint32_t codes, uint32_t scale) {
  if (scale == 255) return false;
  if (scale <= 252) return true;
  const uint32_t bad = scale == 254 ? codes : codes & (codes << 1);
  return !(bad & 0x44444444u);
}

// Whether a lane's four words of group codes (word g holding eight codes
// of group g) are valid at the row's four group scales (byte g).
CHOP_LANES_INLINE bool weights_valid(const uint32_t (&words)[4], uint32_t scales) {
  if (!extreme_scales(scales)) return true;
  bool valid = true;
  for (int g = 0; g < 4; ++g) valid = valid && group_valid(words[g], (scales >> (8 * g)) & 255);
  return valid;
}

// Nonzero when any byte of four E4M3 codes is NaN (magnitude 0x7f).
CHOP_LANES_INLINE uint32_t e4m3_nan(uint32_t codes) {
  return ((codes & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
}

// Whether `scaled_e4m3x2` takes its exactness check at this activation
// scale, outside the bytes where every finite E4M3 code is exact.
CHOP_LANES_INLINE bool extreme_activation_scale(uint32_t scale) {
  return scale < 3 || scale > 246;
}

} // namespace chop::moe::lanes::checks
#undef CHOP_LANES_INLINE
