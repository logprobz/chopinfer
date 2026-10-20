#pragma once
#include "pooled_indexer/position_select_cached.cuh"

namespace chop::position_candidates_cached {
// Fuse live block maxima and ordered selection. Each thread retains its
// original score/index keys through four score-byte passes. When every
// block fits, only validation and the ordered publication remain.
template<int Block, int Kept>
__device__ void publish(
    const float* scores, const uint32_t* live_rows, const uint64_t* tags,
    const int32_t* visible, const int32_t* source_status, int32_t* ids,
    uint64_t* out_tags, int32_t* out_visible, int32_t* status,
    uint32_t width, uint32_t top_k) {
  using namespace position_live;
  static_assert(Block == 8 && Kept > 0 && Kept <= 2048);
  constexpr int Items = (2 * Kept + threads - 1) / threads;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const int32_t requested = visible[row];
  const uint32_t count = min(uint32_t(max(requested, 0)), width);
  const uint32_t active = (count + Block - 1) / Block;
  int32_t* out = ids + size_t(row) * Kept;
  for (uint32_t i = threadIdx.x; i < Kept; i += threads) out[i] = -1;
  bool invalid = source_status[row] != 0 || requested < 0
      || uint32_t(requested) > width || width > 2u * Block * Kept
      || top_k == 0 || top_k > Kept;
  uint64_t keys[Items] = {};
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const uint32_t block = uint32_t(item) * threads + threadIdx.x;
    if (block < active && !invalid) {
      float value = -__int_as_float(0x7f800000);
#pragma unroll
      for (int offset = 0; offset < Block; ++offset) {
        const uint32_t index = block * Block + offset;
        if (index < count) {
          const float score = scores[size_t(row) * width + index];
          invalid |= isnan(score) || score == __int_as_float(0x7f800000);
          value = fmaxf(value, score);
        }
      }
      // Pin only after validating the last block's scores, as before.
      if (block == (count - 1) / Block) value = __int_as_float(0x7f800000);
      if (value > -__int_as_float(0x7f800000))
        keys[item] = candidate_key(value, block);
    }
  }
  const bool failed = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    out_tags[row] = tags[row];
    out_visible[row] = int32_t(count);
    status[row] = failed;
  }
  if (failed || active == 0) return;
  __shared__ uint32_t histogram[threads], prefix, rank, warps[8], total;
  __shared__ uint32_t written, seen_ties;
  if (threadIdx.x == 0) { written = 0; seen_ties = 0; prefix = 0; rank = 0; }
  __syncthreads();
  if (active > top_k)
    position_select_cached::cutoff(keys, top_k, histogram, &prefix, &rank);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    if (uint32_t(item) * threads >= active) break;
    const uint64_t key = keys[item];
    const uint32_t score = uint32_t(key >> 32);
    const bool tied = key && rank != 0 && score == prefix;
    const uint32_t tie_offset = ballot_prefix(tied, warps, &total);
    const uint32_t tie_count = total;
    const bool keep = key && (rank == 0 || score > prefix
        || (tied && seen_ties + tie_offset < rank));
    __syncthreads();
    const uint32_t offset = ballot_prefix(keep, warps, &total);
    if (keep) out[written + offset] = int32_t(pooled_indexer::candidate_pool(key));
    __syncthreads();
    if (threadIdx.x == 0) { written += total; seen_ties += tie_count; }
    __syncthreads();
  }
}
} // namespace chop::position_candidates_cached
