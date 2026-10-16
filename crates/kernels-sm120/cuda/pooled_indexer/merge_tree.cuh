#pragma once

// Tree merge of a selection's split lists (README.md "Tree merge"): the
// key list or per-rank pairs `select_merge` writes, bit for bit, with the
// lists merged in registers instead of through the streaming top-k. The
// candidates are distinct keys, so the kept set is the best `top_k / Group`
// keys of the union whatever the merge order. The best `max_kept` keys of
// two lists sorted descending are the elementwise maxima of one and the
// other reversed, a bitonic sequence that a bitonic merge sorts again.

#include "pooled_indexer/select.cuh"

// Lab-only mutants (README.md "Tree merge"); each is 0 when served.
#ifndef CHOP_MERGE_TREE_TEST_SKIP_LAST
#define CHOP_MERGE_TREE_TEST_SKIP_LAST 0
#endif
#ifndef CHOP_MERGE_TREE_TEST_SCORE_ONLY
#define CHOP_MERGE_TREE_TEST_SCORE_ONLY 0
#endif
#ifndef CHOP_MERGE_TREE_TEST_LATE_WAIT
#define CHOP_MERGE_TREE_TEST_LATE_WAIT 0
#endif

namespace chop::pooled_indexer {
namespace merge_tree {

// Whether key `a` ranks above key `b`.
__device__ __forceinline__ bool above(uint64_t a, uint64_t b) {
  constexpr int shift = CHOP_MERGE_TREE_TEST_SCORE_ONLY ? 32 : 0;
  return (a >> shift) > (b >> shift);
}

__device__ __forceinline__ uint64_t shuffle_xor(uint64_t x, int mask) {
  const uint32_t low = __shfl_xor_sync(0xffffffffu, uint32_t(x), mask);
  const uint32_t high = __shfl_xor_sync(0xffffffffu, uint32_t(x >> 32), mask);
  return uint64_t(high) << 32 | low;
}

// One compare-exchange stage of a bitonic network over a warp's list of
// 32 * R keys, element e = lane * R + i: e meets e ^ Stride, and the pair
// is ordered descending when e & Size is zero, ascending otherwise.
template<int R, int Size, int Stride>
__device__ __forceinline__ void stage(uint64_t (&x)[R]) {
  const int lane = threadIdx.x & 31;
  if constexpr (Stride >= R) {
    constexpr int mask = Stride / R;
    const bool lower = (lane & mask) == 0;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const bool descending = ((lane * R + i) & Size) == 0;
      const uint64_t other = shuffle_xor(x[i], mask);
      const uint64_t first = lower ? x[i] : other, second = lower ? other : x[i];
      const bool swap = descending ? above(second, first) : above(first, second);
      x[i] = lower != swap ? first : second;
    }
  } else {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      if ((i & Stride) == 0) {
        const bool descending = ((lane * R + i) & Size) == 0;
        const uint64_t first = x[i], second = x[i + Stride];
        const bool swap = descending ? above(second, first) : above(first, second);
        x[i] = swap ? second : first;
        x[i + Stride] = swap ? first : second;
      }
    }
  }
}

// The stages of strides `Stride` down to 1 at `Size`.
template<int R, int Size, int Stride>
__device__ __forceinline__ void stages(uint64_t (&x)[R]) {
  stage<R, Size, Stride>(x);
  if constexpr (Stride > 1) stages<R, Size, Stride / 2>(x);
}

// Sorts a bitonic warp list descending.
template<int R>
__device__ __forceinline__ void merge_descending(uint64_t (&x)[R]) {
  stages<R, 32 * R, 16 * R>(x);
}

// Sorts any warp list descending: bitonic sequences of `Size` and up.
template<int R, int Size = 2>
__device__ __forceinline__ void sort_descending(uint64_t (&x)[R]) {
  stages<R, Size, Size / 2>(x);
  if constexpr (Size < 32 * R) sort_descending<R, 2 * Size>(x);
}

// Folds list `y`, held reversed, into the sorted list `x`: the best 32 * R
// keys of both, sorted descending.
template<int R>
__device__ __forceinline__ void fold(uint64_t (&x)[R], const uint64_t (&y)[R]) {
#pragma unroll
  for (int i = 0; i < R; ++i) x[i] = above(y[i], x[i]) ? y[i] : x[i];
  merge_descending<R>(x);
}

// Lane l's keys of a list `[32 * R]` in global memory: elements l * R + i,
// or reversed, elements 32 * R - 1 - (l * R + i).
template<int R, bool Reversed>
__device__ __forceinline__ void load(uint64_t (&x)[R], const uint64_t* list) {
  const int lane = threadIdx.x & 31;
  const ulonglong2* source =
      reinterpret_cast<const ulonglong2*>(list + (Reversed ? 31 - lane : lane) * R);
#pragma unroll
  for (int j = 0; j < R / 2; ++j) {
    const ulonglong2 pair = source[j];
    x[Reversed ? R - 1 - 2 * j : 2 * j] = pair.x;
    x[Reversed ? R - 2 - 2 * j : 2 * j + 1] = pair.y;
  }
}

}  // namespace merge_tree

// One CTA of `Warps` warps per row over the `splits` lists `select_split`
// wrote. Warp w folds lists w, w + Warps, ... in registers; the warps then
// fold pairwise through shared memory until warp 0 holds the row's best
// `max_kept` keys sorted descending. The outputs are `select_merge`'s.
template<class T, int Warps>
__device__ void select_merge_tree(const uint64_t* workspace, const int32_t* positions,
                                  const uint32_t* live_rows, void* output, uint32_t top_k,
                                  uint32_t splits) {
  constexpr int R = T::max_kept / 32;
  static_assert(T::max_kept % 32 == 0 && (R & (R - 1)) == 0 && R >= 2,
                "a warp holds a power of two of keys per lane");
  static_assert(Warps >= 2 && (Warps & (Warps - 1)) == 0 && 32 * Warps >= T::threads,
                "the warps cover the selection's threads");
  // A list per upper warp of a fold level, each lane's keys padded by one
  // word so lanes start on distinct banks.
  constexpr int stride = 32 * (R + 1);
  __shared__ __align__(16) uint64_t exchange[Warps / 2][stride];
  __shared__ uint32_t kept_count;

  ptx::griddep_launch_dependents();
  if (!CHOP_MERGE_TREE_TEST_LATE_WAIT) ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int64_t position = positions[row];
  const uint32_t pools = visible_pools<T>(position), kept = top_k / T::group;
  if constexpr (!T::per_rank) {
    if (pools <= kept) {
      if (threadIdx.x < T::threads)
        write_list<T>(static_cast<int32_t*>(output) + size_t(row) * T::list_width, position, pools,
                      [](uint32_t j) { return j; });
      return;
    }
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint64_t* lists = workspace + size_t(row) * splits * T::max_kept;
  const uint32_t merged = splits - CHOP_MERGE_TREE_TEST_SKIP_LAST;
  uint64_t x[R];
  if (uint32_t(warp) < merged) {
    merge_tree::load<R, false>(x, lists + size_t(warp) * T::max_kept);
  } else {
#pragma unroll
    for (int i = 0; i < R; ++i) x[i] = 0;
  }
  for (uint32_t next = warp + Warps; next < merged; next += Warps) {
    uint64_t y[R];
    merge_tree::load<R, true>(y, lists + size_t(next) * T::max_kept);
    merge_tree::fold<R>(x, y);
  }
  if (CHOP_MERGE_TREE_TEST_LATE_WAIT) ptx::griddep_wait();

  // Warps past the lists hold no keys, so a level whose upper warps all
  // hold none is skipped; `splits` is uniform over the CTA.
#pragma unroll
  for (int half = Warps / 2; half >= 1; half >>= 1) {
    if (splits <= uint32_t(half)) continue;
    if (warp >= half && warp < 2 * half) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const int e = 32 * R - 1 - (lane * R + i);
        exchange[warp - half][e / R * (R + 1) + e % R] = x[i];
      }
    }
    __syncthreads();
    if (warp < half) {
      uint64_t y[R];
#pragma unroll
      for (int i = 0; i < R; ++i) y[i] = exchange[warp][lane * (R + 1) + i];
      merge_tree::fold<R>(x, y);
    }
    __syncthreads();
  }

  if (warp == 0) {
    // Nonzero keys come first, so the kept ones are the nonzero keys among
    // the first `kept`: `select_merge`'s min(live, kept).
    uint32_t mine = 0;
#pragma unroll
    for (int i = 0; i < R; ++i) mine += uint32_t(lane * R + i) < kept && x[i] != 0;
    const uint32_t count = __reduce_add_sync(0xffffffffu, mine);
    if constexpr (T::per_rank) {
      uint4* pairs = reinterpret_cast<uint4*>(static_cast<uint32_t*>(output) +
                                              (size_t(row) * T::max_kept + lane * R) * 2);
#pragma unroll
      for (int j = 0; j < R / 2; ++j) {
        uint32_t word[4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int i = 2 * j + k;
          const bool live = uint32_t(lane * R + i) < count;
          word[2 * k] = live ? __float_as_uint(candidate_score(x[i])) : 0xff800000u;
          word[2 * k + 1] = live ? candidate_pool(x[i]) : 0xffffffffu;
        }
        pairs[j] = make_uint4(word[0], word[1], word[2], word[3]);
      }
    } else {
      // Pool-major keys sort the kept pools ascending, the empty keys last.
#pragma unroll
      for (int i = 0; i < R; ++i)
        x[i] = uint32_t(lane * R + i) < count ? x[i] << 32 | x[i] >> 32 : 0;
      merge_tree::sort_descending<R>(x);
#pragma unroll
      for (int i = 0; i < R; ++i) exchange[0][lane * R + i] = x[i];
      if (lane == 0) kept_count = count;
    }
  }
  if constexpr (!T::per_rank) {
    __syncthreads();
    if (threadIdx.x < T::threads)
      write_list<T>(static_cast<int32_t*>(output) + size_t(row) * T::list_width, position,
                    kept_count,
                    [&](uint32_t j) { return ~uint32_t(exchange[0][j] >> 32); });
  }
}

}  // namespace chop::pooled_indexer
