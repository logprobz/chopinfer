#pragma once

// Live-context variants have their own source family. The score arithmetic
// follows the immutable Position v3 headers, including FMA and tree order.
#include "pooled_indexer/position_blocks.cuh"

namespace chop::position_live {
using namespace position_indexer;
static constexpr uint32_t shards = 32;

template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void dense_scores(
    const uint16_t* queries, const float* weights, const uint8_t* codes,
    const uint8_t* scales, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const uint8_t* mask,
    float* output, uint32_t width, uint32_t max_pages, uint64_t code_stride,
    uint64_t scale_stride, float weight_scale) {
  static_assert(Dim > 0 && Dim % 32 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  const uint32_t lane = threadIdx.x & 31;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const uint32_t extent = width;
  const uint32_t active = count;
  // Clear unused score slots with coalesced stores, without capacity CTAs.
  for (uint32_t i = active + blockIdx.x * blockDim.x + threadIdx.x;
       i < extent; i += gridDim.x * blockDim.x)
    output[size_t(row) * extent + i] = -__int_as_float(0x7f800000);
  for (uint32_t index = blockIdx.x * 4 + threadIdx.x / 32;
       index < active; index += gridDim.x * 4) {
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* out = output + size_t(row) * width + index;
  if (!admitted<Page, Mask, false>(index, count, row_mask, table)) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    continue;
  }
  const uint32_t page = uint32_t(table[index / Page]);
  float key[Dim / 32];
  bool invalid = false;
  if constexpr (Bf16) {
  const uint16_t* values = reinterpret_cast<const uint16_t*>(codes + uint64_t(page) * code_stride) + size_t(index % Page) * Dim;
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    key[part] = bf16(values[part * 32 + lane]);
    invalid |= !isfinite(key[part]);
  }
  } else {
  const uint8_t* key_codes = codes + uint64_t(page) * code_stride + size_t(index % Page) * (Dim / 2);
  const uint8_t* key_scales = scales + uint64_t(page) * scale_stride + size_t(index % Page) * (Dim / 32);
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    const int channel = part * 32 + lane;
    const uint32_t pair = convert::scaled_mxfp4_bf16x2<false>(key_codes[channel / 2], key_scales[part]);
    key[part] = bf16(uint16_t(pair >> (((channel ^ (false)) & 1) * 16)));
    invalid |= key_scales[part] == 255 || !isfinite(key[part]);
  }
  }
  float result = 0.f;
#pragma unroll 1
  for (int head = 0; head < Heads; ++head) {
    const uint16_t* q = queries + (size_t(row) * Heads + head) * Dim;
    float dot = 0.f;
#pragma unroll
    for (int base = 0; base < Dim; base += 128) {
      float chain = 0.f;
#pragma unroll
      for (int part = base / 32; part < (base + 128 < Dim ? base + 128 : Dim) / 32; ++part) {
        const float value = bf16(q[part * 32 + lane]);
        invalid |= !isfinite(value);
        chain = fmaf(value, key[part], chain);
        invalid |= !isfinite(chain);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        chain += __shfl_down_sync(0xffffffffu, chain, offset);
        // Only these lanes feed the next level of the reduction tree.
        invalid |= lane < offset && !isfinite(chain);
      }
      dot += chain;
      invalid |= lane == 0 && !isfinite(dot);
    }
    if (lane == 0) {
      const int weight_head = false ? (head + 1) % Heads : head;
      const float weight = __fmul_rn(weights[size_t(row) * Heads + weight_head], weight_scale);
      invalid |= !isfinite(weight);
      result = fmaf(dot > 0.f ? dot : 0.f, weight, result);
      invalid |= !isfinite(result);
    }
  }
  const bool failed = __any_sync(0xffffffffu, invalid);
  if (lane == 0) *out = failed ? __int_as_float(0x7fc00000) : result;
  }
}

template<int Heads, int Dim, int Page, int Ratio, int Block, int Kept, bool Bf16>
__device__ void compact_scores(
    const uint16_t* queries, const float* weights, const uint8_t* codes,
    const uint8_t* scales, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const int32_t* ids,
    const uint64_t* tags, const uint64_t* stored_tags, const int32_t* stored_visible,
    const int32_t* source_status, float* output, uint32_t width,
    uint32_t max_pages, uint64_t code_stride, uint64_t scale_stride, float weight_scale) {
  static_assert(Dim > 0 && Dim % 32 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  const uint32_t lane = threadIdx.x & 31;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const uint32_t extent = Block * Kept;
  const uint32_t active = min(uint32_t(Block * Kept), ((count + Block - 1) / Block) * Block);
  // Clear unused score slots with coalesced stores, without capacity CTAs.
  for (uint32_t i = active + blockIdx.x * blockDim.x + threadIdx.x;
       i < extent; i += gridDim.x * blockDim.x)
    output[size_t(row) * extent + i] = -__int_as_float(0x7f800000);
  for (uint32_t slot = blockIdx.x * 4 + threadIdx.x / 32;
       slot < active; slot += gridDim.x * 4) {
  const int32_t id = ids[size_t(row) * Kept + slot / Block];
  const uint64_t mapped = id < 0 ? uint64_t(width) : uint64_t(id) * Block + slot % Block;
  float* out = output + size_t(row) * (Block * Kept) + slot;
  if (source_status[row] != 0 || (true && (tags[row] != stored_tags[row] || stored_visible[row] != int32_t(count))) || mapped >= count) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    continue;
  }
  const uint32_t index = uint32_t(mapped);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  if (!admitted<Page, false>(index, count, nullptr, table)) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    continue;
  }
  // Keep the admitted-position arithmetic identical to position.cuh:
  // same decoded values, 128-channel chains, trees and head FMA order.
  const uint32_t page = uint32_t(table[index / Page]);
  float key[Dim / 32];
  bool invalid = false;
  if constexpr (Bf16) {
  const uint16_t* values = reinterpret_cast<const uint16_t*>(codes + uint64_t(page) * code_stride) + size_t(index % Page) * Dim;
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    key[part] = bf16(values[part * 32 + lane]);
    invalid |= !isfinite(key[part]);
  }
  } else {
  const uint8_t* key_codes = codes + uint64_t(page) * code_stride + size_t(index % Page) * (Dim / 2);
  const uint8_t* key_scales = scales + uint64_t(page) * scale_stride + size_t(index % Page) * (Dim / 32);
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    const int channel = part * 32 + lane;
    const uint32_t pair = convert::scaled_mxfp4_bf16x2<false>(key_codes[channel / 2], key_scales[part]);
    key[part] = bf16(uint16_t(pair >> (((channel ^ false) & 1) * 16)));
    invalid |= key_scales[part] == 255 || !isfinite(key[part]);
  }
  }
  float result = 0.f;
#pragma unroll 1
  for (int head = 0; head < Heads; ++head) {
    const uint16_t* q = queries + (size_t(row) * Heads + head) * Dim;
    float dot = 0.f;
#pragma unroll
    for (int base = 0; base < Dim; base += 128) {
      float chain = 0.f;
#pragma unroll
      for (int part = base / 32; part < (base + 128 < Dim ? base + 128 : Dim) / 32; ++part) {
        const float value = bf16(q[part * 32 + lane]);
        invalid |= !isfinite(value);
        chain = fmaf(value, key[part], chain);
        invalid |= !isfinite(chain);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        chain += __shfl_down_sync(0xffffffffu, chain, offset);
        // Only these lanes feed the next level of the reduction tree.
        invalid |= lane < offset && !isfinite(chain);
      }
      dot += chain;
      invalid |= lane == 0 && !isfinite(dot);
    }
    if (lane == 0) {
      const int weight_head = head;
      const float weight = __fmul_rn(weights[size_t(row) * Heads + weight_head], weight_scale);
      invalid |= !isfinite(weight);
      result = fmaf(dot > 0.f ? dot : 0.f, weight, result);
      invalid |= !isfinite(result);
    }
  }
  const bool failed = __any_sync(0xffffffffu, invalid);
  if (lane == 0) *out = failed ? __int_as_float(0x7fc00000) : result;
  }
}

// Eight warp ballots and one shared prefix replace the 16-barrier scan.
// The prefix and the retained keys always follow increasing input indices.
__device__ uint32_t ballot_prefix(bool keep, uint32_t* warps, uint32_t* total) {
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const uint32_t bits = __ballot_sync(0xffffffffu, keep);
  if (lane == 0) warps[warp] = __popc(bits);
  __syncthreads();
  uint32_t before = 0;
  for (uint32_t w = 0; w < warp; ++w) before += warps[w];
  if (threadIdx.x == 0) {
    uint32_t sum = 0;
    for (int w = 0; w < 8; ++w) sum += warps[w];
    *total = sum;
  }
  const uint32_t prefix = before + __popc(bits & ((1u << lane) - 1));
  __syncthreads();
  return prefix;
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
  const uint64_t threshold = length <= kept ? 0 : cutoff(length, kept, at, histogram, &prefix, &rank);
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
  const uint64_t threshold = length <= kept ? 0 : cutoff(length, kept, key, histogram, &prefix, &rank);
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
  const uint32_t active = min(slots, ((count + Block - 1) / Block) * Block);
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

// Score only live blocks. Padding remains byte-identical to the old producer.
template<int Block>
__device__ void block_scores(const float* scores, const int32_t* visible_counts,
                             const uint32_t* live_rows, float* blocks, uint32_t width) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const int32_t requested = visible_counts[row];
  const uint32_t count = min(uint32_t(max(requested, 0)), width);
  const bool bad_count = requested < 0 || uint32_t(requested) > width;
  const uint32_t block_count = (width + Block - 1) / Block;
  const uint32_t active = (count + Block - 1) / Block;
  for (uint32_t block = blockIdx.x * threads + threadIdx.x; block < block_count; block += gridDim.x * threads) {
    bool invalid = bad_count;
    float value = -__int_as_float(0x7f800000);
    if (block < active) {
#pragma unroll
      for (int offset = 0; offset < Block; ++offset) {
        const uint32_t i = block * Block + offset;
        if (!invalid && i < count) {
          const float score = scores[size_t(row) * width + i];
          invalid |= isnan(score) || score == __int_as_float(0x7f800000);
          value = fmaxf(value, score);
        }
      }
      if (count != 0 && block == (count - 1) / Block) value = __int_as_float(0x7f800000);
    }
    blocks[size_t(row) * block_count + block] = invalid ? __int_as_float(0x7fc00000) : value;
  }
}

template<int Block, int Kept, bool Merge>
__device__ void gather(
    const float* blocks, const uint32_t* live_rows, const uint64_t* tags,
    const int32_t* visible_counts, const int32_t* source_status,
    int32_t* ids, uint64_t* out_tags, int32_t* out_visible, int32_t* status,
    uint32_t width, uint32_t top_k_blocks, uint64_t* scratch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) return;
  const int32_t requested = visible_counts[row];
  const uint32_t count = min(uint32_t(max(requested, 0)), width);
  const uint32_t active = (count + Block - 1) / Block;
  if constexpr (Merge) {
    if (threadIdx.x == 0) { out_tags[row] = tags[row]; out_visible[row] = int32_t(count); }
    merge<Kept>(row, active, top_k_blocks, scratch, ids, status);
    return;
  }
  const float* values = blocks + size_t(row) * ((width + Block - 1) / Block);
  const uint32_t span = (active + shards - 1) / shards;
  const uint32_t end = min((blockIdx.x + 1) * span, active);
  bool invalid = source_status[row] != 0 || requested < 0 || uint32_t(requested) > width || top_k_blocks == 0 || top_k_blocks > Kept;
  for (uint32_t i = blockIdx.x * span + threadIdx.x; i < end; i += threads) invalid |= isnan(values[i]);
  const bool failed = __syncthreads_or(invalid);
  auto key = [&](uint32_t i) {
    return values[i] > -__int_as_float(0x7f800000) ? candidate_key(values[i], i) : uint64_t(0);
  };
  local<Kept>(row, active, top_k_blocks, failed, key, scratch);
}
}  // namespace chop::position_live
