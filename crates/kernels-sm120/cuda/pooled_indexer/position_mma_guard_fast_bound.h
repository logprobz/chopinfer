#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define CHOP_INDEX_BOUND_HD __host__ __device__
#else
#define CHOP_INDEX_BOUND_HD
#endif

namespace chop::position_mma_guard_fast {
struct Range {
  uint32_t lo, hi;
};
CHOP_INDEX_BOUND_HD inline Range scale_range(uint32_t word) {
  uint32_t lo = word & 255, hi = lo;
#ifdef __CUDACC__
#pragma unroll
#endif
  for (int i = 1; i < 4; ++i) {
    const uint32_t value = (word >> (8 * i)) & 255;
    lo = lo < value ? lo : value;
    hi = hi > value ? hi : value;
  }
  return {lo, hi};
}
// q contains extrema across all heads. spread is the largest within-head
// range, not the range across heads. Each head's four products therefore
// span at most spread + (k.hi - k.lo) exponent steps.
CHOP_INDEX_BOUND_HD inline bool coarse_scales(Range q, uint32_t spread, Range k) {
  return q.lo >= 2 && q.hi <= 252 && k.lo >= 2 && k.hi <= 252 &&
      q.lo + k.lo >= 130 && q.hi + k.hi <= 369 &&
      spread + k.hi - k.lo <= 9;
}
} // namespace chop::position_mma_guard_fast
#undef CHOP_INDEX_BOUND_HD
