#pragma once
#include "pooled_indexer/position_live.cuh"

namespace chop::position_select_cached {
// Return the boundary score and the number of its lowest-position ties to
// retain. The key array remains in registers across the four score passes.
template<int Items>
__device__ void cutoff(const uint64_t (&keys)[Items], uint32_t kept,
    uint32_t* histogram, uint32_t* prefix, uint32_t* rank) {
  if (threadIdx.x == 0) { *prefix = 0; *rank = kept; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    histogram[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t mask = shift == 24 ? 0 : (~0u << (shift + 8));
#pragma unroll
    for (int i = 0; i < Items; ++i) {
      const uint64_t key = keys[i];
      const uint32_t score = uint32_t(key >> 32);
      if (key && (score & mask) == *prefix)
        atomicAdd(histogram + ((score >> shift) & 255u),1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t remaining = *rank;
      int bucket = 255;
      while (bucket >= 0 && remaining > histogram[bucket]) remaining -= histogram[bucket--];
      if (bucket < 0) { *prefix = 0; *rank = 0; }
      else { *prefix |= uint32_t(bucket) << shift; *rank = remaining; }
    }
    __syncthreads();
    if (*rank == 0) return;
  }
}

template<int Page, int Ratio, int MaxTopK, bool Mask, bool Merge, int Items = 16>
__device__ void dense_select(float* scores, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t* mask, int32_t* output, int32_t* status, uint32_t width,
    uint32_t max_pages, uint32_t top_k, uint64_t* scratch) {
  static_assert(!Merge && Items > 0);
  using namespace position_live;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row],Ratio),width);
  const uint32_t span = (count + shards - 1) / shards;
  // Large contexts keep the existing selector instead of spilling a wider
  // register cache. This decision is uniform within the CTA and graph-safe.
  if (span > Items * threads) {
    position_live::dense_select<Page,Ratio,MaxTopK,Mask,false>(scores,tables,batches,
        positions,live_rows,mask,output,status,width,max_pages,top_k,scratch);
    return;
  }
  const uint32_t begin = min(blockIdx.x * span,count);
  const uint32_t length = min(span,count - begin);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* values = scores + size_t(row) * width;
  uint64_t keys[Items] = {};
  bool invalid = false;
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const uint32_t offset = uint32_t(item) * threads + threadIdx.x;
    const uint32_t index = begin + offset;
    if (offset < length && admitted<Page,Mask>(index,count,row_mask,table)) {
      const float value = values[index];
      if (!isfinite(value)) {
        invalid = true;
        values[index] = -__int_as_float(0x7f800000);
      } else keys[item] = candidate_key(value,index);
    }
  }
  const bool failed = __syncthreads_or(invalid);
  uint64_t* out = scratch + (size_t(row) * shards + blockIdx.x) * (MaxTopK + 1);
  for (uint32_t i = threadIdx.x; i < MaxTopK; i += threads) out[i] = 0;
  if (threadIdx.x == 0) out[MaxTopK] = failed;
  if (failed) return;
  __shared__ uint32_t histogram[threads], prefix, rank, warps[8], total;
  __shared__ uint32_t written, seen_ties;
  if (threadIdx.x == 0) { written = 0; seen_ties = 0; rank = 0; prefix = 0; }
  __syncthreads();
  if (length > top_k) cutoff(keys,top_k,histogram,&prefix,&rank);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    if (uint32_t(item) * threads >= length) break;
    const uint64_t key = keys[item];
    const uint32_t score = uint32_t(key >> 32);
    const bool tied = key && rank != 0 && score == prefix;
    const uint32_t tie_offset = ballot_prefix(tied,warps,&total);
    const uint32_t tie_count = total;
    const bool keep = key && (rank == 0 || score > prefix || (tied && seen_ties + tie_offset < rank));
    // Finish all readers before the next prefix reuses shared totals.
    __syncthreads();
    const uint32_t offset = ballot_prefix(keep,warps,&total);
    if (keep) out[written + offset] = key;
    __syncthreads();
    if (threadIdx.x == 0) { written += total; seen_ties += tie_count; }
    __syncthreads();
  }
}
} // namespace chop::position_select_cached
