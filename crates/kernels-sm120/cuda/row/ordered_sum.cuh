#pragma once

// The ordered sum of F32 terms, bitwise the serial sum
//
//   s = +0; for i in 0 .. Terms: s = __fadd_rn(s, terms[i]);
//
// (one round-to-nearest-even add per term, in order: row.cuh's HF-ordered
// rmsnorm sum of squares), computed by Workers threads in about Terms /
// Workers + log steps instead of Terms dependent adds. The method is exact
// for every input; non-negative finite terms take the fast path. The CPU
// emulation kernels_sm120::row::ordered_sum::Emulator follows it step for
// step and tests/ordered_sum.rs compares it with the serial sum.
//
// Construction. Index the running sums by binade class: class(s) = the
// biased exponent field of s, except that +0 and the subnormals are class
// 1. Class K (1 <= K <= 254) is the interval [lo_K, 2^(K-126)) with lo_1 =
// 0 and lo_K = 2^(K-127) otherwise, and every F32 in it is an integer
// multiple of u_K = 2^(K-150) (class 1 holds the subnormals and the first
// normal binade, all multiples of 2^-149). So s in class K is exactly
// index(s) * u_K with index(s) = bits(s) - ((K - 1) << 23) < 2^24, and
// bits(index * u_K) = ((K - 1) << 23) + index for every index < 2^24 of the
// class (for K = 1 the subnormal and first-binade encodings coincide with
// the index; for K >= 2 the index carries the implicit bit).
//
//   Step lemma. Let s be in class K and a >= 0 finite with a < 2^(K-126).
//   Write a / u_K = m + f (m integer, 0 <= f < 1) and k = m if f < 1/2,
//   m + 1 if f > 1/2. If index(s) + k' < 2^24, where k' is k, or for f =
//   1/2 the one of m, m + 1 that makes index(s) + k' even, then
//   __fadd_rn(s, a) = (index(s) + k') * u_K.
//   Proof: k' >= m and the indices are integers, so index(s) + k' < 2^24
//   gives index(s) + m + f < 2^24: s + a lies in [lo_K, 2^(K-126)), where
//   the F32 values are exactly the multiples of u_K (2^(K-126) is one too).
//   Round to nearest picks (index(s) + m) u_K or (index(s) + m + 1) u_K by
//   f, and an exact half the one whose significand is even; the
//   significand's last bit is the index's last bit. The result is at least
//   s and below 2^(K-126), so it is in class K again.
//
// So while the running sum stays in class K, every add is an integer add of
// the term's quotient on K's grid, with ties decided by the running
// index's parity. Integer adds compose: a run of terms maps an incoming
// index i to i + inc(i mod 2), and the pair (inc(even), inc(odd)) composes
// exactly ("earlier then later" is inc(p) = earlier(p) + later((p +
// earlier(p)) mod 2)). Ties to even go through that parity: a tie's result
// is always even, and every other step flips the parity by k's.
//
// Algorithm. Worker w holds terms [w Terms / Workers, (w + 1) Terms /
// Workers) (one to 128 of them).
//   1. Predictions: each worker adds its terms from zero, a warp scan and
//      the warps' totals give F32 estimates of the running sum before and
//      after its chunk. They only choose a class; no result depends on them.
//   2. Each worker whose estimates lie in one finite class K, and whose
//      terms are all below 0x7f800000 (non-negative and finite) and below
//      2^(K-126), computes its pair on K's grid: per term the quotient k
//      (bits shifted by the scale difference, the remainder against the
//      half), and the tie flag; inc(p) += k + (tie and (p + inc(p) + k)
//      odd). A worker whose pair reaches 2^24 crosses from any start. The
//      rest are serial workers.
//   3. A segmented warp scan composes the pairs of consecutive regular
//      workers of one class into runs; each run's last worker holds its
//      pair, below 2^29.
//   4. One walker goes through the runs and serial workers in order from s
//      = +0. A run of class K is applied, s = ((K - 1) << 23) + index(s) +
//      inc(index(s) mod 2), only if class(s) = K and that end is below
//      2^24; otherwise, and for serial workers, it adds the chunk's terms
//      with __fadd_rn from the current s.
//
// Why each case gives the serial sum's bits (by induction over the walker's
// items, s equal to the serial sum before the item's first term):
//   - Applied run: class(s) = K and the end index is below 2^24. Every
//     index after a step is at most the end (increments are non-negative),
//     so every intermediate sum is in class K and each step lemma applies,
//     with its tie decided by the parity the pair tracked. The run's terms
//     are all below 2^(K-126), the lemma's bound.
//   - Refused run (the sum is in another class, a prediction was off, or
//     the run crosses into the next binade, including a step that rounds up
//     to 2^(K-126) or overflows): the serial adds are the served adds.
//   - Zero start: +0 is class 1 with index 0; adding a finite non-negative
//     a gives a exactly in both. Zero terms have k = 0 and no tie;
//     __fadd_rn(s, +0) = s for s >= +0, so a zero term changes nothing.
//   - Subnormal terms: their significand has no implicit bit and their
//     scale is class 1's, so the shift is exact; on class 1's grid they are
//     integers (no rounding).
//   - Terms far below the grid: the shift saturates at 25, where every
//     significand (below 2^24) is under the half: k = 0, no tie, as the
//     round to nearest gives.
//   - Inf, NaN, negative terms (-0 included): bits at or above 0x7f800000,
//     so their worker adds serially; a NaN or an infinity in s has class
//     255 or more and matches no run, so everything after it is added
//     serially too, and NaN payloads and signs propagate exactly as the
//     served adds propagate them.
//   - Overflow: a step to 2^128 rounds past the end of class 254, the run is
//     refused and the serial adds produce the infinity.
//
// Test-only mutants, compiled into kernel-lab's mutant cubins: with
// CHOP_ORDERED_SUM_TEST_TIES_AWAY a tie always adds m + 1; with
// CHOP_ORDERED_SUM_TEST_NO_FALLBACK the walker applies every run unchecked.

#include <stdint.h>

namespace chop::row {

template<int Terms, int Workers>
struct OrderedSum {
  static constexpr int kWarps = Workers / 32;
  static constexpr int kChunk = (Terms + Workers - 1) / Workers;
  static_assert(Workers % 32 == 0 && Workers > 0 && Workers <= 1024, "whole warps of one CTA");
  static_assert(Terms >= Workers && kChunk <= 128, "one to 128 terms per worker");
  struct Shared {
    // Each worker's pair, composed at a run's last worker: the increment
    // from an even and from an odd running index.
    uint32_t even[Workers];
    uint32_t odd[Workers];
    // The worker's run class (0 for a serial worker) and, from bit 16, the
    // lane its run starts at.
    uint32_t info[Workers];
    // The warps' chunk-sum totals and their walker items (runs' last lanes
    // and serial lanes).
    float totals[kWarps];
    uint32_t items[kWarps];
  };
};

// The class of a running sum or an estimate: its biased exponent, 1 for +0
// and the subnormals, 255 and above for infinities, NaNs and negative
// values.
__device__ __forceinline__ uint32_t sum_class(uint32_t bits) {
  return max(bits >> 23, 1u);
}

// A barrier over the Workers threads (whole warps) that call ordered_sum,
// on named barrier Barrier.
template<int Barrier, int Workers>
__device__ __forceinline__ void workers_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(Barrier), "n"(Workers) : "memory");
}

// The serial sum of terms[0 .. Terms) (shared or global memory, written
// before the call and unchanged during it), computed by the Workers threads
// that call this with worker = 0 .. Workers - 1 (whole warps, every lane
// active), synchronizing on named barrier Barrier. Returns the sum in
// worker 0; the other workers return +0.
template<int Terms, int Workers, int Barrier>
__device__ float ordered_sum(const float* terms, int worker,
                            typename OrderedSum<Terms, Workers>::Shared& shared) {
  using S = OrderedSum<Terms, Workers>;
  constexpr int Chunk = S::kChunk;
  constexpr uint32_t Limit = 1u << 24;
  const int lane = worker & 31;
  const int warp = worker >> 5;
  const int lo = worker * Terms / Workers;
  const int count = (worker + 1) * Terms / Workers - lo;

  // 1. The chunk, its sum from zero, and whether every term is
  // non-negative and finite.
  uint32_t a[Chunk];
  float sigma = 0.f;
  bool regular = true;
  #pragma unroll
  for (int j = 0; j < Chunk; ++j) {
    a[j] = 0u;
    if (j < count) {
      a[j] = __float_as_uint(terms[lo + j]);
      sigma = __fadd_rn(sigma, __uint_as_float(a[j]));
      regular &= a[j] < 0x7f800000u;
    }
  }
  float inclusive = sigma;
  #pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const float other = __shfl_up_sync(0xffffffffu, inclusive, d);
    if (lane >= d) inclusive = __fadd_rn(inclusive, other);
  }
  float exclusive = __shfl_up_sync(0xffffffffu, inclusive, 1);
  if (lane == 0) exclusive = 0.f;
  if (lane == 31) shared.totals[warp] = inclusive;
  workers_sync<Barrier, Workers>();
  float base = 0.f;
  for (int v = 0; v < warp; ++v) base = __fadd_rn(base, shared.totals[v]);
  const uint32_t key = sum_class(__float_as_uint(__fadd_rn(base, exclusive)));
  regular &= key < 255u && key == sum_class(__float_as_uint(__fadd_rn(base, inclusive)));

  // 2. The pair on class key's grid.
  uint32_t even = 0u, odd = 0u;
  #pragma unroll
  for (int j = 0; j < Chunk; ++j) {
    if (j < count) {
      const uint32_t exponent = a[j] >> 23;
      const uint32_t significand = (a[j] & 0x7fffffu) | (exponent ? 0x800000u : 0u);
      const uint32_t scale = max(exponent, 1u);
      regular &= scale <= key;
      // A term at or above the class's bound wraps to a large difference:
      // shift 25, quotient 0, and the worker is serial anyway.
      const uint32_t shift = min(key - scale, 25u);
      const uint32_t rem = significand & ((1u << shift) - 1u);
      const uint32_t half = (1u << shift) >> 1;
      const uint32_t k = (significand >> shift) + (rem > half ? 1u : 0u);
      const uint32_t tie = shift != 0u && rem == half ? 1u : 0u;
#ifndef CHOP_ORDERED_SUM_TEST_TIES_AWAY
      even += k + (tie & (even + k));
      odd += k + (tie & (odd + k + 1u));
#else
      even += k + tie;
      odd += k + tie;
#endif
    }
  }
  regular &= even < Limit && odd < Limit;

  // 3. Runs: consecutive regular workers of one class in the warp,
  // composed by a segmented scan.
  const uint32_t own = regular ? key : 0u;
  const uint32_t previous = __shfl_up_sync(0xffffffffu, own, 1);
  const bool head = !regular || lane == 0 || previous != own;
  const uint32_t heads = __ballot_sync(0xffffffffu, head);
  const int start = 31 - __clz(heads & (0xffffffffu >> (31 - lane)));
  #pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t earlier_even = __shfl_up_sync(0xffffffffu, even, d);
    const uint32_t earlier_odd = __shfl_up_sync(0xffffffffu, odd, d);
    if (regular && lane - d >= start) {
      const uint32_t from_even = earlier_even + ((earlier_even & 1u) ? odd : even);
      const uint32_t from_odd = earlier_odd + ((earlier_odd & 1u) ? even : odd);
      even = from_even;
      odd = from_odd;
    }
  }
  const bool last = lane == 31 || ((heads >> (lane + 1)) & 1u);
  const uint32_t items = __ballot_sync(0xffffffffu, last);
  shared.even[worker] = even;
  shared.odd[worker] = odd;
  shared.info[worker] = own | (uint32_t(start) << 16);
  if (lane == 0) shared.items[warp] = items;
  workers_sync<Barrier, Workers>();
  if (worker != 0) return 0.f;

  // 4. The walker.
  uint32_t s = 0u;
  for (int v = 0; v < S::kWarps; ++v) {
    uint32_t pending = shared.items[v];
    while (pending) {
      const int at = v * 32 + __ffs(pending) - 1;
      pending &= pending - 1u;
      const uint32_t info = shared.info[at];
      const uint32_t run = info & 0xffffu;
      int from = at;
      if (run) {
        from = v * 32 + int(info >> 16);
        const uint32_t floor = (run - 1u) << 23;
        const uint32_t index = s - floor;
        const uint32_t end = index + ((index & 1u) ? shared.odd[at] : shared.even[at]);
#ifndef CHOP_ORDERED_SUM_TEST_NO_FALLBACK
        if (sum_class(s) == run && end < Limit)
#endif
        {
          s = floor + end;
          continue;
        }
      }
      const int stop = (at + 1) * Terms / Workers;
      float f = __uint_as_float(s);
      for (int i = from * Terms / Workers; i < stop; ++i) f = __fadd_rn(f, terms[i]);
      s = __float_as_uint(f);
    }
  }
  return __uint_as_float(s);
}

// Lab test entry (grid: rows, Workers + 32 threads): row r's Terms F32
// values from `terms`, staged in shared memory; thread 0 adds them
// serially while warps 1 .. Workers / 32 run ordered_sum; the two sums'
// bits go to serial[r] and exact[r].
template<int Terms, int Workers>
__device__ void ordered_sum_test(const float* terms, uint32_t* serial, uint32_t* exact,
                                 uint32_t rows) {
  using S = OrderedSum<Terms, Workers>;
  __shared__ float staged[Terms];
  __shared__ typename S::Shared shared;
  if (blockDim.x != Workers + 32) asm volatile("trap;");
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < Terms; i += blockDim.x) {
    staged[i] = terms[uint64_t(row) * Terms + i];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      float s = 0.f;
      for (int i = 0; i < Terms; ++i) s = __fadd_rn(s, staged[i]);
      serial[row] = __float_as_uint(s);
    }
    return;
  }
  const float sum = ordered_sum<Terms, Workers, 1>(staged, threadIdx.x - 32, shared);
  if (threadIdx.x == 32) exact[row] = __float_as_uint(sum);
}

} // namespace chop::row
