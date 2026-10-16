#pragma once

// Radix-select merge of a global selection's split lists (README.md "Radix
// merge"): the key list `select_merge` writes, bit for bit. The candidates
// are distinct keys, so the kept set is the `top_k / Group` largest keys of
// the lists' union, whatever the order of the lists and of the keys in
// them. Split s of a row's live splits holds exactly min(pools of s, kept)
// keys first, every one nonzero. Up to `merge_radix::cache_keys` of them,
// the merge reads only those into shared memory, finds the kept-th largest
// key by a radix select from the top byte down, and sorts the kept keys
// pool-major for the list. Past that it folds the live lists, each read as
// its sorted keys then zeros, through the tree merge's register networks.

#include "pooled_indexer/merge_tree.cuh"

// Lab-only mutants (README.md "Radix merge"); each is 0 when served.
#ifndef CHOP_MERGE_RADIX_TEST_SKIP_LAST
#define CHOP_MERGE_RADIX_TEST_SKIP_LAST 0
#endif
#ifndef CHOP_MERGE_RADIX_TEST_TIES_HIGH
#define CHOP_MERGE_RADIX_TEST_TIES_HIGH 0
#endif
#ifndef CHOP_MERGE_RADIX_TEST_LATE_WAIT
#define CHOP_MERGE_RADIX_TEST_LATE_WAIT 0
#endif

namespace chop::pooled_indexer {
namespace merge_radix {

constexpr int threads = 512, warps = threads / 32;
// The most candidates the radix select takes, in shared memory; past this
// the tree folds the lists.
constexpr int cache_keys = 4096;
constexpr int digit_bits = 8, digits = 1 << digit_bits;

// The order the select ranks a key by: the key itself, so equal scores rank
// the lower pool first. The `_ties_high` mutant ranks the higher pool first.
// Either form is its own inverse.
__device__ __forceinline__ uint64_t rank(uint64_t key) {
  return CHOP_MERGE_RADIX_TEST_TIES_HIGH ? key ^ 0xffffffffull : key;
}

}  // namespace merge_radix


// One CTA of `merge_radix::threads` per row over the `splits` lists
// `select_split` wrote. The outputs are `select_merge`'s.
template<class T>
__device__ void select_merge_radix(const uint64_t* workspace, const int32_t* positions,
                                   const uint32_t* live_rows, int32_t* output, uint32_t top_k,
                                   uint32_t splits) {
  namespace mr = merge_radix;
  static_assert(!T::per_rank, "the radix merge writes a global key list");
  static_assert(mr::threads >= T::threads && mr::threads % 32 == 0);
  constexpr int R = T::max_kept / 32;
  static_assert(T::max_kept % 32 == 0 && (R & (R - 1)) == 0 && R >= 2,
                "a warp holds a power of two of keys per lane");
  // The tree's exchange: a list per upper warp of a fold level, each lane's
  // keys padded by one word so lanes start on distinct banks.
  constexpr int stride = 32 * (R + 1);
  constexpr int exchange_keys = mr::warps / 2 * stride;
  constexpr int shared_keys = exchange_keys > mr::cache_keys ? exchange_keys : mr::cache_keys;
  // The radix select's candidates, or the tree's exchange.
  __shared__ __align__(16) uint64_t keys[shared_keys];
  __shared__ __align__(16) uint64_t kept_keys[T::max_kept];
  __shared__ uint32_t histogram[mr::digits];
  __shared__ uint32_t choice[3];
  __shared__ uint32_t kept_count;

  ptx::griddep_launch_dependents();
  if (!CHOP_MERGE_RADIX_TEST_LATE_WAIT) ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int64_t position = positions[row];
  const uint32_t pools = visible_pools<T>(position), kept = top_k / T::group;
  int32_t* list = output + size_t(row) * T::list_width;
  if (pools <= kept) {
    if (threadIdx.x < T::threads) write_list<T>(list, position, pools, [](uint32_t j) { return j; });
    return;
  }

  // `split_range`'s splits: every live split but the last covers `aligned`
  // pools, and a list holds min(pools of its split, kept) keys. Their sum is
  // at least `kept`: one split of `kept` or more pools, or every pool.
  const uint32_t chunk = (pools + splits - 1) / splits;
  const uint32_t aligned = (chunk + T::tile - 1) / T::tile * T::tile;
  const uint32_t live = (pools + aligned - 1) / aligned;
  const uint32_t lists = live - CHOP_MERGE_RADIX_TEST_SKIP_LAST;
  const uint32_t per = min(aligned, kept);
  const uint32_t final_keys = min(pools - (live - 1) * aligned, kept);
  auto keys_of = [&](uint32_t split) { return split + 1 < live ? per : final_keys; };
  const uint32_t total = lists == 0 ? 0 : (lists - 1) * per + keys_of(lists - 1);
  const uint64_t* row_lists = workspace + size_t(row) * splits * T::max_kept;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  if (total > uint32_t(mr::cache_keys)) {
    // The tree merge's folds over the live lists: warp w folds lists w, w +
    // warps, ..., each read as its keys then zeros, then the warps fold
    // pairwise through shared memory until warp 0 holds the best keys.
    auto read = [&](uint64_t(&x)[R], uint32_t split, bool reversed) {
      if (reversed) {
        merge_tree::load<R, true>(x, row_lists + size_t(split) * T::max_kept);
      } else {
        merge_tree::load<R, false>(x, row_lists + size_t(split) * T::max_kept);
      }
      // A list of fewer keys than it holds reads as its keys then zeros.
      const uint32_t count = keys_of(split);
      if (count < uint32_t(32 * R)) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const uint32_t e = reversed ? 32 * R - 1 - (lane * R + i) : lane * R + i;
          if (e >= count) x[i] = 0;
        }
      }
      if (CHOP_MERGE_RADIX_TEST_TIES_HIGH) {
#pragma unroll
        for (int i = 0; i < R; ++i) x[i] = x[i] == 0 ? 0 : mr::rank(x[i]);
      }
    };
    uint64_t x[R];
    if (uint32_t(warp) < lists) {
      read(x, warp, false);
    } else {
#pragma unroll
      for (int i = 0; i < R; ++i) x[i] = 0;
    }
    for (uint32_t next = warp + mr::warps; next < lists; next += mr::warps) {
      uint64_t y[R];
      read(y, next, true);
      merge_tree::fold<R>(x, y);
    }
    if (CHOP_MERGE_RADIX_TEST_LATE_WAIT) ptx::griddep_wait();
#pragma unroll
    for (int half = mr::warps / 2; half >= 1; half >>= 1) {
      if (lists <= uint32_t(half)) continue;
      if (warp >= half && warp < 2 * half) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const int e = 32 * R - 1 - (lane * R + i);
          keys[(warp - half) * stride + e / R * (R + 1) + e % R] = x[i];
        }
      }
      __syncthreads();
      if (warp < half) {
        uint64_t y[R];
#pragma unroll
        for (int i = 0; i < R; ++i) y[i] = keys[warp * stride + lane * (R + 1) + i];
        merge_tree::fold<R>(x, y);
      }
      __syncthreads();
    }
    if (warp == 0) {
      // Nonzero keys come first; the kept ones are those among the first
      // `kept`, pool-major so that a descending sort orders their pools
      // ascending.
      uint32_t mine = 0;
#pragma unroll
      for (int i = 0; i < R; ++i) mine += uint32_t(lane * R + i) < kept && x[i] != 0;
      const uint32_t count = __reduce_add_sync(0xffffffffu, mine);
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const uint64_t original = mr::rank(x[i]);
        x[i] = uint32_t(lane * R + i) < count ? original << 32 | original >> 32 : 0;
      }
      merge_tree::sort_descending<R>(x);
#pragma unroll
      for (int i = 0; i < R; ++i) keys[lane * R + i] = x[i];
      if (lane == 0) kept_count = count;
    }
    __syncthreads();
    if (threadIdx.x < T::threads)
      write_list<T>(list, position, kept_count,
                    [&](uint32_t j) { return ~uint32_t(keys[j] >> 32); });
    return;
  }

  // The radix select over the candidates in shared memory.
  for (uint32_t i = threadIdx.x; i < total; i += mr::threads) {
    const uint32_t split = i / per;
    keys[i] = mr::rank(row_lists[size_t(split) * T::max_kept + (i - split * per)]);
  }
  if (CHOP_MERGE_RADIX_TEST_LATE_WAIT) ptx::griddep_wait();

  // The kept-th largest ranked key, a digit at a time from the top: the
  // histogram of the digit among keys that match the digits found so far,
  // then the digit whose keys hold the kept-th. `need` counts the kept keys
  // left to find among the matching ones. Keys are distinct, so the last
  // digit's bucket holds exactly the one key needed.
  uint64_t prefix = 0;
  uint32_t need = kept;
  int shift = 64 - mr::digit_bits;
  for (;; shift -= mr::digit_bits) {
    for (int i = threadIdx.x; i < mr::digits; i += mr::threads) histogram[i] = 0;
    __syncthreads();
    const uint64_t high = shift + mr::digit_bits >= 64 ? 0 : ~0ull << (shift + mr::digit_bits);
    for (uint32_t base = warp * 32; base < total; base += mr::threads) {
      const uint32_t i = base + lane;
      uint32_t bin = mr::digits;
      if (i < total) {
        const uint64_t key = keys[i];
        if ((key & high) == prefix) bin = uint32_t(key >> shift) & (mr::digits - 1);
      }
      // One add per distinct bin of the warp.
      const uint32_t peers = __match_any_sync(0xffffffffu, bin);
      if (bin < uint32_t(mr::digits) && lane == __ffs(peers) - 1)
        atomicAdd(&histogram[bin], uint32_t(__popc(peers)));
    }
    __syncthreads();
    if (warp == 0) {
      // Lane l holds digits 255 - 8 l down to 248 - 8 l, so lanes in order
      // walk the digits from the top.
      uint32_t count[mr::digits / 32], sum = 0;
#pragma unroll
      for (int q = 0; q < mr::digits / 32; ++q) {
        count[q] = histogram[mr::digits - 1 - (mr::digits / 32) * lane - q];
        sum += count[q];
      }
      uint32_t inclusive = sum;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
      }
      if (lane == 0) choice[0] = choice[1] = choice[2] = 0;
      __syncwarp();
      uint32_t before = inclusive - sum;
#pragma unroll
      for (int q = 0; q < mr::digits / 32; ++q) {
        if (before < need && need <= before + count[q]) {
          choice[0] = mr::digits - 1 - (mr::digits / 32) * lane - q;
          choice[1] = before;
          choice[2] = count[q];
        }
        before += count[q];
      }
    }
    __syncthreads();
    need -= choice[1];
    prefix |= uint64_t(choice[0]) << shift;
    if (choice[2] == need || shift == 0) break;
  }

  // The kept keys are those whose digits down to `shift` are at least the
  // threshold's, pool-major (the inverted pool, then the score bits) so
  // that a descending sort orders their pools ascending.
  const uint64_t bound = prefix >> shift;
  for (int i = threadIdx.x; i < T::max_kept; i += mr::threads) kept_keys[i] = 0;
  if (threadIdx.x == 0) kept_count = 0;
  __syncthreads();
  for (uint32_t base = warp * 32; base < total; base += mr::threads) {
    const uint32_t i = base + lane;
    const uint64_t key = i < total ? keys[i] : 0;
    const bool take = i < total && (key >> shift) >= bound;
    const uint32_t mask = __ballot_sync(0xffffffffu, take);
    uint32_t slot = 0;
    if (lane == 0 && mask != 0) slot = atomicAdd(&kept_count, uint32_t(__popc(mask)));
    slot = __shfl_sync(0xffffffffu, slot, 0) + __popc(mask & ((1u << lane) - 1));
    if (take && slot < uint32_t(T::max_kept)) {
      const uint64_t original = mr::rank(key);
      kept_keys[slot] = original << 32 | original >> 32;
    }
  }
  __syncthreads();
  const uint32_t count = min(kept_count, uint32_t(T::max_kept));
  sort_descending<T::max_kept, mr::threads>(kept_keys);
  if (threadIdx.x < T::threads)
    write_list<T>(list, position, count,
                  [&](uint32_t j) { return ~uint32_t(kept_keys[j] >> 32); });
}

}  // namespace chop::pooled_indexer
