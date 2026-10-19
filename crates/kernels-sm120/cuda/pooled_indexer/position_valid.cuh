#pragma once

// Compact work follows the retained ID prefix. Full-record validation is
// still mandatory; a malformed record clears every score and fails the row.
#include "pooled_indexer/position_live.cuh"

namespace chop::position_valid {
using namespace position_live;

template<int Block, int Kept>
__device__ __forceinline__ uint32_t retained_slots(const int32_t* ids) {
  uint32_t first = 0, last = Kept;
  while (first < last) {
    const uint32_t middle = first + (last - first) / 2;
    if (ids[middle] >= 0) first = middle + 1;
    else last = middle;
  }
  return first * Block;
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
  const uint32_t active = min(retained_slots<Block, Kept>(ids + size_t(row) * Kept), ((count + Block - 1) / Block) * Block);
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

// Only live block maxima are consumed by the existing live gather. Its
// direct visible-count validation also handles an empty malformed row.
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
  for (uint32_t block = blockIdx.x * threads + threadIdx.x; block < active; block += gridDim.x * threads) {
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

}  // namespace chop::position_valid
