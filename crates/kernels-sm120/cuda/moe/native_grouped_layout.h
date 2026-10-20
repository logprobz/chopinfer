#pragma once
#include <stdint.h>

// Operand layout of the native grouped expert tiles (native_grouped.cuh),
// shared by the device code and the host emulation test
// (tests/native_grouped_layout.cpp).
//
// One m16n8k32 E2M1 x E4M3 block-scaled MMA reduces one 32-K scale group.
// Weights are the A operand (16 weight rows), routed rows the B operand (8
// rows). Lane t of a quad owns the group's K elements 8t..8t+7 in both
// operands: the instruction's K slots 4t..4t+3 hold elements 8t..8t+3 and
// slots 16+4t..16+4t+3 hold elements 8t+4..8t+7. Every product is exact,
// and this fixed permutation of K is the same in every entry, so a row's
// bits do not depend on the tile that computes it.
#if defined(__CUDACC__)
#define CHOP_NATIVE_INLINE __host__ __device__ __forceinline__
#else
#define CHOP_NATIVE_INLINE inline
#endif
namespace chop::moe::native_grouped::layout {

constexpr int StageK = 128;  // one F32 promotion per 128 K
constexpr int GroupK = 32;   // one UE8M0 scale per 32 K
constexpr int ActRowBytes = StageK;
constexpr int WeightRowBytes = StageK / 2;

CHOP_NATIVE_INLINE uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t selector) {
#if defined(__CUDA_ARCH__)
  return __byte_perm(x, y, selector);
#else
  const uint64_t bytes = uint64_t(x) | uint64_t(y) << 32;
  uint32_t result = 0;
  for (int i = 0; i < 4; ++i)
    result |= uint32_t((bytes >> (8 * ((selector >> (4 * i)) & 7))) & 0xff) << (8 * i);
  return result;
#endif
}

// The stage's 16-byte chunk c (0..7) of activation row r, swizzled so a
// warp's eight-byte fragment loads of eight rows touch every bank once.
CHOP_NATIVE_INLINE int act_chunk(int r, int c) { return c ^ ((r & 3) << 1); }
// The stage's 16-byte chunk q (0..3, one per 32-K group) of weight row w,
// swizzled so a warp's four-byte loads of eight rows touch every bank once.
CHOP_NATIVE_INLINE int weight_chunk(int w, int q) { return q ^ ((w >> 1) & 3); }

// Where the producer stores chunk c of a row, in bytes from the stage base.
CHOP_NATIVE_INLINE int act_store(int r, int c) {
  return r * ActRowBytes + act_chunk(r, c) * 16;
}
CHOP_NATIVE_INLINE int weight_store(int w, int q) {
  return w * WeightRowBytes + weight_chunk(w, q) * 16;
}
// Lane t's eight activation codes of group q in row r (one eight-byte load).
CHOP_NATIVE_INLINE int act_load(int r, int q, int t) {
  return r * ActRowBytes + act_chunk(r, 2 * q + (t >> 1)) * 16 + (t & 1) * 8;
}
// Lane t's four packed weight bytes of group q in row w (eight E2M1 codes).
CHOP_NATIVE_INLINE int weight_load(int w, int q, int t) {
  return w * WeightRowBytes + weight_chunk(w, q) * 16 + t * 4;
}

// Eight packed E2M1 codes (element 2j in the low nibble of byte j) into the
// instruction's byte containers, each code in bits 5:2: `low` holds
// elements 0..3, `high` elements 4..7.
CHOP_NATIVE_INLINE void expand_e2m1x8(uint32_t packed, uint32_t& low, uint32_t& high) {
  const uint32_t even = (packed << 2) & 0x3c3c3c3cu;
  const uint32_t odd = (packed >> 2) & 0x3c3c3c3cu;
  low = byte_perm(even, odd, 0x5140);
  high = byte_perm(even, odd, 0x7362);
}

// Nonzero when any byte of four E4M3 codes is NaN (magnitude 0x7f).
CHOP_NATIVE_INLINE uint32_t e4m3_nan(uint32_t codes) {
  return ((codes & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
}

// The four group scales of a stage from the one or two aligned words the
// producer copied: a row's scales start `shift` bytes into the first word.
CHOP_NATIVE_INLINE uint32_t scale_word(uint32_t first, uint32_t second, int shift) {
  return shift ? (first >> (8 * shift)) | (second << (32 - 8 * shift)) : first;
}

// Accumulator element i of an MMA fragment: its A row (weight row) and B
// column (routed row) within the 16x8 tile.
CHOP_NATIVE_INLINE int d_weight_row(int g, int i) { return g + 8 * (i >> 1); }
CHOP_NATIVE_INLINE int d_route(int t, int i) { return 2 * t + (i & 1); }

// Tiles of an expert's `count` rows at `rows` rows per tile.
CHOP_NATIVE_INLINE int tiles(int count, int rows) { return (count + rows - 1) / rows; }

} // namespace chop::moe::native_grouped::layout
#undef CHOP_NATIVE_INLINE
