#pragma once
#include <stdint.h>

// A sufficient domain, independent of MMA's accumulation pairing. See
// position_mma_guard.md for the integer bound and exponent limits.
#if defined(__CUDACC__)
#define CHOP_INDEX_GUARD_INLINE __host__ __device__ __forceinline__
#else
#define CHOP_INDEX_GUARD_INLINE inline
#endif
namespace chop::position_mma_guard {
CHOP_INDEX_GUARD_INLINE bool exact_scales(uint32_t query, uint32_t key) {
  int lo = 510, hi = 0;
  bool normal = true;
#if defined(__CUDACC__)
#pragma unroll
#endif
  for (int i = 0; i < 4; ++i) {
    const int q = (query >> (i * 8)) & 255;
    const int k = (key >> (i * 8)) & 255;
    normal &= q >= 2 && q <= 252 && k >= 2 && k <= 252;
    const int sum = q + k;
    lo = sum < lo ? sum : lo;
    hi = sum > hi ? sum : hi;
  }
  return normal && lo >= 130 && hi <= 369 && hi - lo <= 9;
}
} // namespace chop::position_mma_guard
#undef CHOP_INDEX_GUARD_INLINE
