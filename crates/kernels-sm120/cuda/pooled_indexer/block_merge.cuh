#pragma once

// The exact merge of per-rank selections written as block ids (README.md
// "Block-id merge"): `topk_merge`'s kept pools, ascending, as the i32
// `[rows, MaxTopK / Group]` list the block-sparse attention reads, then -1.

#include "pooled_indexer/head_sum.cuh"

namespace chop::pooled_indexer {

// One CTA per row over `Ranks` per-rank pair lists gathered rank-major as
// [Ranks][rank_rows][max_kept][2].
template<class T, int Ranks>
__device__ void topk_merge_blocks(const uint32_t* pairs, const uint32_t* live_rows,
                                  int32_t* blocks, uint32_t top_k, uint32_t rank_rows) {
  static_assert(!T::per_rank && Ranks >= 2 && Ranks <= 8);
  __shared__ __align__(16) uint8_t storage[T::TopKBuffer::smem];
  typename T::TopKBuffer best(storage);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  best.init();
  fold_lists<T>(best, top_k / T::group, Ranks, [&](uint32_t rank, int i) -> uint64_t {
    const uint32_t* pair = pairs + ((size_t(rank) * rank_rows + row) * T::max_kept + i) * 2;
    return int32_t(pair[1]) < 0 ? 0 : candidate_key(__uint_as_float(pair[0]), pair[1]);
  });
  best.reduce(top_k / T::group);
  // Pool-major keys sort the kept pools ascending, the empty keys last.
  const uint32_t count = *best.count;
  for (int i = threadIdx.x; i < T::max_kept; i += T::threads) {
    const uint64_t key = best.keys[i];
    best.keys[i] = uint32_t(i) < count ? key << 32 | key >> 32 : 0;
  }
  __syncthreads();
  sort_descending<T::max_kept, T::threads>(best.keys);
  int32_t* out = blocks + size_t(row) * T::max_kept;
  for (int i = threadIdx.x; i < T::max_kept; i += T::threads)
    out[i] = uint32_t(i) < count ? int32_t(~uint32_t(best.keys[i] >> 32)) : -1;
}

}  // namespace chop::pooled_indexer
