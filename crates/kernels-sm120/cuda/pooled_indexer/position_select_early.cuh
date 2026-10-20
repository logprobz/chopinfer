#pragma once

// Prefill selection with an exact early stop. The byte-radix cutoff of
// position.cuh runs all eight passes over the 64-bit score/position keys.
// After any pass whose boundary bucket holds exactly the keys still required,
// the kept set is already determined: every key at or above that bucket's
// prefix. Returning the prefix as the threshold keeps the same keys, so every
// output is unchanged. With distinct boundary scores the stop fires after the
// four score bytes and skips the four position bytes; ties and holes run the
// remaining passes as before.
//
// local, merge and dense_select below are position_live's, and compact_select
// is position_valid's, copied unchanged except that the two cutoff calls name
// stop_cutoff. A kernels-sm120 test holds the copies to their sources. This
// header is a separate source family; no existing cubin includes it.
#include "pooled_indexer/position_valid.cuh"

namespace chop::position_select_early {
using namespace position_valid;

// The cutoff's rank value once the boundary bucket is kept whole.
static constexpr uint32_t stopped = 0xffffffffu;

template<class Key>
__device__ uint64_t stop_cutoff(uint32_t width, uint32_t kept, Key key,
                                uint32_t* histogram, uint64_t* prefix, uint32_t* rank) {
  if (threadIdx.x == 0) { *prefix = 0; *rank = kept; }
  __syncthreads();
  for (int shift = 56; shift >= 0; shift -= 8) {
    histogram[threadIdx.x] = 0;
    __syncthreads();
    const uint64_t high_mask = shift == 56 ? 0 : (~uint64_t(0) << (shift + 8));
    for (uint32_t i = threadIdx.x; i < width; i += threads) {
      const uint64_t value = key(i);
      if (value != 0 && (value & high_mask) == *prefix)
        atomicAdd(histogram + ((value >> shift) & 255u), 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t remaining = *rank;
      int bucket = 255;
      while (bucket >= 0 && remaining > histogram[bucket]) remaining -= histogram[bucket--];
      if (bucket < 0) {
        // There are fewer than kept entries on the first pass.
        *prefix = 0;
        *rank = 0;
      } else {
        *prefix |= uint64_t(bucket) << shift;
        *rank = remaining == histogram[bucket] ? stopped : remaining;
      }
    }
    __syncthreads();
    const uint32_t state = *rank;
    if (state == 0) return 0;
    if (state == stopped) return *prefix;
  }
  return *prefix;
}

// A shard retains at most k keys. The global top-k is contained in the
// union of these lists. Unique score/index keys make both cutoffs exact.
template<int Kept, class Key>
__device__ void local(uint32_t row, uint32_t count, uint32_t kept, bool failed,
                      Key key, uint64_t* scratch) {
  __shared__ uint32_t histogram[threads], rank, warps[8], total, written;
  __shared__ uint64_t prefix;
  const uint32_t span = (count + shards - 1) / shards;
  const uint32_t begin = min(blockIdx.x * span, count);
  const uint32_t length = min(span, count - begin);
  uint64_t* out = scratch + (size_t(row) * shards + blockIdx.x) * (Kept + 1);
  for (uint32_t i = threadIdx.x; i < Kept; i += threads) out[i] = 0;
  if (threadIdx.x == 0) { out[Kept] = failed; written = 0; }
  if (failed) return;
  auto at = [&](uint32_t i) { return key(begin + i); };
  const uint64_t threshold = length <= kept ? 0 : stop_cutoff(length, kept, at, histogram, &prefix, &rank);
  __syncthreads();
  for (uint32_t base = 0; base < length; base += threads) {
    const uint32_t i = base + threadIdx.x;
    const uint64_t value = i < length ? at(i) : 0;
    const bool keep = value != 0 && value >= threshold;
    const uint32_t offset = ballot_prefix(keep, warps, &total);
    if (keep) out[written + offset] = value;
    __syncthreads();
    if (threadIdx.x == 0) written += total;
    __syncthreads();
  }
}

template<int Kept>
__device__ void merge(uint32_t row, uint32_t count, uint32_t kept,
                      const uint64_t* scratch, int32_t* output, int32_t* status) {
  __shared__ uint32_t histogram[threads], rank, warps[8], total, written;
  __shared__ uint64_t prefix;
  const uint64_t* lists = scratch + size_t(row) * shards * (Kept + 1);
  const uint32_t span = min(kept, (count + shards - 1) / shards);
  int32_t* out = output + size_t(row) * Kept;
  for (uint32_t i = threadIdx.x; i < Kept; i += threads) out[i] = -1;
  const bool bad = threadIdx.x < shards && lists[threadIdx.x * (Kept + 1) + Kept] != 0;
  const bool failed = __syncthreads_or(bad);
  if (threadIdx.x == 0) { status[row] = failed; written = 0; }
  if (failed || span == 0) return;
  auto key = [&](uint32_t i) { return lists[(i / span) * (Kept + 1) + i % span]; };
  const uint32_t length = shards * span;
  const uint64_t threshold = length <= kept ? 0 : stop_cutoff(length, kept, key, histogram, &prefix, &rank);
  __syncthreads();
  for (uint32_t base = 0; base < length; base += threads) {
    const uint32_t i = base + threadIdx.x;
    const uint64_t value = i < length ? key(i) : 0;
    const bool keep = value != 0 && value >= threshold;
    const uint32_t offset = ballot_prefix(keep, warps, &total);
    if (keep) out[written + offset] = int32_t(pooled_indexer::candidate_pool(value));
    __syncthreads();
    if (threadIdx.x == 0) written += total;
    __syncthreads();
  }
}

template<int Page, int Ratio, int MaxTopK, bool Mask, bool Merge>
__device__ void dense_select(
    float* scores, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const uint8_t* mask,
    int32_t* output, int32_t* status, uint32_t width, uint32_t max_pages,
    uint32_t top_k, uint64_t* scratch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  if constexpr (Merge) { merge<MaxTopK>(row, count, top_k, scratch, output, status); return; }
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* values = scores + size_t(row) * width;
  const uint32_t span = (count + shards - 1) / shards;
  const uint32_t end = min((blockIdx.x + 1) * span, count);
  bool invalid = false;
  for (uint32_t i = blockIdx.x * span + threadIdx.x; i < end; i += threads) {
    if (admitted<Page, Mask>(i, count, row_mask, table) && !isfinite(values[i])) {
      invalid = true;
      values[i] = -__int_as_float(0x7f800000);
    }
  }
  const bool failed = __syncthreads_or(invalid);
  auto key = [&](uint32_t i) {
    return admitted<Page, Mask>(i, count, row_mask, table) ? candidate_key(values[i], i) : uint64_t(0);
  };
  local<MaxTopK>(row, count, top_k, failed, key, scratch);
}

template<int Page, int Ratio, int MaxTopK, int Block, int Kept, bool Merge>
__device__ void compact_select(
    float* scores, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const int32_t* ids,
    const uint64_t* tags, const uint64_t* stored_tags, const int32_t* stored_visible,
    const int32_t* source_status, int32_t* output, int32_t* status,
    uint32_t width, uint32_t max_pages, uint32_t top_k, uint64_t* scratch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  // Validation covers every ID, including the padded tail on malformed rows.
  const uint32_t slots = Block * Kept;
  const uint32_t active = min(retained_slots<Block, Kept>(ids + size_t(row) * Kept), ((count + Block - 1) / Block) * Block);
  if constexpr (Merge) { merge<MaxTopK>(row, active, top_k, scratch, output, status); return; }
  const int32_t* row_ids = ids + size_t(row) * Kept;
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  float* values = scores + size_t(row) * slots;
  bool invalid = source_status[row] != 0 || tags[row] != stored_tags[row] || stored_visible[row] != int32_t(count);
  bool pinned = count == 0;
  for (uint32_t i = threadIdx.x; i < Kept; i += threads) {
    const int32_t id = row_ids[i];
    if (id == -1) continue;
    invalid |= id < 0 || uint64_t(id) * Block >= count;
    invalid |= i > 0 && (row_ids[i - 1] < 0 || row_ids[i - 1] >= id);
    pinned |= count > 0 && id == int32_t((count - 1) / Block);
  }
  invalid |= !__syncthreads_or(pinned);
  const bool malformed = __syncthreads_or(invalid);
  auto index = [&](uint32_t slot) {
    const int32_t id = row_ids[slot / Block];
    const uint64_t mapped = id < 0 ? uint64_t(width) : uint64_t(id) * Block + slot % Block;
    return mapped < count ? uint32_t(mapped) : width;
  };
  const uint32_t checked = malformed ? slots : active;
  const uint32_t span = (checked + shards - 1) / shards;
  const uint32_t end = min((blockIdx.x + 1) * span, checked);
  for (uint32_t i = blockIdx.x * span + threadIdx.x; i < end; i += threads) {
    if (malformed) values[i] = -__int_as_float(0x7f800000);
    else if (admitted<Page, false>(index(i), count, nullptr, table) && !isfinite(values[i])) {
      invalid = true;
      values[i] = -__int_as_float(0x7f800000);
    }
  }
  const bool failed = __syncthreads_or(invalid);
  auto key = [&](uint32_t slot) {
    const uint32_t i = index(slot);
    return admitted<Page, false>(i, count, nullptr, table) ? candidate_key(values[slot], i) : uint64_t(0);
  };
  local<MaxTopK>(row, active, top_k, failed, key, scratch);
}

// End of the copied selectors.
}  // namespace chop::position_select_early
