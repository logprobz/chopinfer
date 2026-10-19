#pragma once
#include "pooled_indexer/position_valid.cuh"

namespace chop::position_hierarchical {
using namespace position_live;
constexpr uint32_t groups = 4, per_group = shards / groups;

// Original keys and contiguous shard groups preserve increasing-position
// emission. A global top-k key is necessarily in its group's top-k.
template<int Kept, bool Final, bool Negative = false>
__device__ void merge(uint32_t row, uint32_t count, uint32_t kept,
    const uint64_t* scratch, uint64_t* intermediate, int32_t* output, int32_t* status) {
  const uint32_t span = min(kept, (count + shards - 1) / shards);
  // Keep the original small-list merge. The group launch is a no-op here.
  if (span <= 64) {
    if constexpr (Final) position_live::merge<Kept>(row,count,kept,scratch,output,status);
    return;
  }
  __shared__ uint32_t histogram[threads], rank, warps[8], total, written;
  __shared__ uint64_t prefix;
  const uint32_t group = Final ? 0 : blockIdx.x;
  const uint64_t* input = Final
      ? intermediate + size_t(row) * groups * (Kept + 1)
      : scratch + (size_t(row) * shards + group * per_group) * (Kept + 1);
  constexpr uint32_t lists = Final ? groups : per_group;
  const uint32_t list_span = Final ? kept : span;
  const bool bad = threadIdx.x < lists && input[threadIdx.x * (Kept + 1) + Kept] != 0;
  const bool failed = __syncthreads_or(bad);
  uint64_t* out_keys = intermediate + (size_t(row) * groups + group) * (Kept + 1);
  int32_t* out_ids = output + size_t(row) * Kept;
  for (uint32_t i = threadIdx.x; i < Kept; i += threads) {
    if constexpr (Final) out_ids[i] = -1;
    else out_keys[i] = 0;
  }
  if (threadIdx.x == 0) {
    written = 0;
    if constexpr (Final) status[row] = failed;
    else out_keys[Kept] = failed;
  }
  if (failed) return;
  auto key = [&](uint32_t i) { return input[(i / list_span) * (Kept + 1) + i % list_span]; };
  const uint32_t length = lists * list_span;
  // The lab mutant loses winners concentrated in a single group.
  const uint32_t limit = Negative ? max(1u,kept / groups) : kept;
  const uint64_t threshold = length <= limit ? 0 : cutoff(length,limit,key,histogram,&prefix,&rank);
  __syncthreads();
  for (uint32_t base = 0; base < length; base += threads) {
    const uint32_t i = base + threadIdx.x;
    const uint64_t value = i < length ? key(i) : 0;
    const bool keep = value != 0 && value >= threshold;
    const uint32_t offset = ballot_prefix(keep,warps,&total);
    if (keep) {
      if constexpr (Final) out_ids[written + offset] = int32_t(pooled_indexer::candidate_pool(value));
      else out_keys[written + offset] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) written += total;
    __syncthreads();
  }
}

template<int Page, int Ratio, int MaxTopK, bool Mask, bool Final, bool Negative = false>
__device__ void dense_select(float*, const int32_t*, const int32_t*,
    const int32_t* positions, const uint32_t* live_rows, const uint8_t*,
    int32_t* output, int32_t* status, uint32_t width, uint32_t,
    uint32_t top_k, uint64_t* scratch, uint64_t* intermediate) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio),width);
  merge<MaxTopK,Final,Negative>(row,count,top_k,scratch,intermediate,output,status);
}

template<int Page, int Ratio, int MaxTopK, int Block, int Kept, bool Final, bool Negative = false>
__device__ void compact_select(float*, const int32_t*, const int32_t*,
    const int32_t* positions, const uint32_t* live_rows, const int32_t* ids,
    const uint64_t*, const uint64_t*, const int32_t*,
    const int32_t*, int32_t* output, int32_t* status,
    uint32_t width, uint32_t, uint32_t top_k, uint64_t* scratch, uint64_t* intermediate) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio),width);
  const uint32_t active = min(position_valid::retained_slots<Block,Kept>(ids + size_t(row) * Kept),
      ((count + Block - 1) / Block) * Block);
  merge<MaxTopK,Final,Negative>(row,active,top_k,scratch,intermediate,output,status);
}
} // namespace chop::position_hierarchical
