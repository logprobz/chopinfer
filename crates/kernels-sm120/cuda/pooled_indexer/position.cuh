#pragma once

// Position scores and exact selection. This header is outside the pooled
// BF16 engine's source closure. Floating sums have a fixed order; integer
// histograms only find an exact cutoff in the score/index total order.
#include "pooled_indexer/common.cuh"
#include "common/mxfp4.cuh"

namespace chop::position_indexer {

using pooled_indexer::candidate_key;
using pooled_indexer::bf16;
static constexpr int threads = 256;

__device__ __forceinline__ uint32_t visible(int32_t position, uint32_t ratio) {
  return position < 0 ? 0u : uint32_t((uint64_t(position) + 1) / ratio);
}

template<int Page, bool Mask, bool IgnoreMask = false>
__device__ __forceinline__ bool admitted(
    uint32_t index, uint32_t count, const uint8_t* mask, const int32_t* table) {
  if (index >= count || table[index / Page] < 0) return false;
  if constexpr (Mask && !IgnoreMask) return mask[index] != 0;
  return true;
}

// A warp scores one position. Its lanes cover channels inside each 128-K
// chain, reduced by a fixed tree. Chains fold in K order, then heads fold
// with FMA in head order. Every rank runs the whole head set.
template<int Heads, int Dim, int Page, int Ratio, bool Mask, int Control = 0>
__device__ void scores(
    const uint16_t* queries, const float* weights, const uint8_t* codes,
    const uint8_t* scales, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const uint8_t* mask,
    float* output, uint32_t width, uint32_t max_pages, uint64_t code_stride,
    uint64_t scale_stride, float weight_scale) {
  static_assert(Dim > 0 && Dim % 32 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  const uint32_t index = blockIdx.x * 4 + threadIdx.x / 32;
  const uint32_t lane = threadIdx.x & 31;
  if (row >= *live_rows || index >= width) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* out = output + size_t(row) * width + index;
  if (!admitted<Page, Mask, Control == 1>(index, count, row_mask, table)) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    return;
  }
  const uint32_t page = uint32_t(table[index / Page]);
  const uint8_t* key_codes = codes + uint64_t(page) * code_stride + size_t(index % Page) * (Dim / 2);
  const uint8_t* key_scales = scales + uint64_t(page) * scale_stride + size_t(index % Page) * (Dim / 32);
  float key[Dim / 32];
  bool invalid = false;
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    const int channel = part * 32 + lane;
    const uint32_t pair = convert::scaled_mxfp4_bf16x2<false>(key_codes[channel / 2], key_scales[part]);
    key[part] = bf16(uint16_t(pair >> (((channel ^ (Control == 5)) & 1) * 16)));
    invalid |= key_scales[part] == 255 || !isfinite(key[part]);
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
      const int weight_head = Control == 4 ? (head + 1) % Heads : head;
      const float weight = __fmul_rn(weights[size_t(row) * Heads + weight_head], weight_scale);
      invalid |= !isfinite(weight);
      result = fmaf(dot > 0.f ? dot : 0.f, weight, result);
      invalid |= !isfinite(result);
    }
  }
  const bool failed = __any_sync(0xffffffffu, invalid);
  if (lane == 0) *out = failed ? __int_as_float(0x7fc00000) : result;
}

// The kth largest nonzero key, or zero when fewer than k keys exist.
// Every byte pass counts the matching prefix. Atomic arrival order cannot
// affect an integer bucket count or the threshold. Key zero is a hole.
template<class Key>
__device__ uint64_t cutoff(uint32_t width, uint32_t kept, Key key,
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
        *rank = remaining;
      }
    }
    __syncthreads();
    if (*rank == 0) return 0;
  }
  return *prefix;
}

// Exact top-k in position order. Prefix scans assign output slots in
// increasing input order and never depend on atomic arrival order.
template<int Page, int Ratio, int MaxTopK, bool Mask, int Control = 0>
__device__ void select(
    float* scores, const int32_t* tables, const int32_t* batches,
    const int32_t* positions, const uint32_t* live_rows, const uint8_t* mask,
    int32_t* output, int32_t* status, uint32_t width, uint32_t max_pages, uint32_t top_k) {
  __shared__ uint32_t histogram[threads], rank, scan[threads], written;
  __shared__ uint64_t prefix;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* values = scores + size_t(row) * width;
  int invalid = 0;
  for (uint32_t i = threadIdx.x; i < count; i += threads) {
    if (admitted<Page, Mask, Control == 1>(i, count, row_mask, table) && !isfinite(values[i])) {
      invalid = 1;
      // Keep a following block filter inside its domain even when the
      // host checks the row status only after the whole graph finishes.
      values[i] = -__int_as_float(0x7f800000);
    }
  }
  const int failed = __syncthreads_or(invalid) != 0;
  if (threadIdx.x == 0) status[row] = failed;
  int32_t* out = output + size_t(row) * MaxTopK;
  for (int i = threadIdx.x; i < MaxTopK; i += threads) out[i] = -1;
  if (failed) return;
  auto key = [&](uint32_t i) -> uint64_t {
    if (!admitted<Page, Mask, Control == 1>(i, count, row_mask, table) || (Control == 3 && i == 0))
      return uint64_t(0);
    const uint64_t value = candidate_key(values[i], i);
    return Control == 2 ? (value & 0xffffffff00000000ull) | i : value;
  };
  const uint64_t threshold = cutoff(count, top_k, key, histogram, &prefix, &rank);
  if (threadIdx.x == 0) written = 0;
  __syncthreads();
  for (uint32_t base = 0; base < count; base += threads) {
    const uint32_t i = base + threadIdx.x;
    const uint64_t value = i < count ? key(i) : 0;
    const bool keep = value != 0 && value >= threshold;
    scan[threadIdx.x] = keep;
    __syncthreads();
    for (int distance = 1; distance < threads; distance *= 2) {
      const uint32_t add = threadIdx.x >= distance ? scan[threadIdx.x - distance] : 0;
      __syncthreads();
      scan[threadIdx.x] += add;
      __syncthreads();
    }
    if (keep) out[written + scan[threadIdx.x] - 1] = int32_t(i);
    __syncthreads();
    if (threadIdx.x == 0) written += scan[threads - 1];
    __syncthreads();
  }
}

template<int Block, bool WrongPin = false>
__device__ void block_scores(const float* scores, const int32_t* visible_counts,
                             const uint32_t* live_rows, float* blocks, uint32_t width) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  const uint32_t block = blockIdx.x * threads + threadIdx.x;
  const uint32_t block_count = (width + Block - 1) / Block;
  if (row >= *live_rows || block >= block_count) return;
  const uint32_t count = min(uint32_t(max(visible_counts[row], 0)), width);
  float value = -__int_as_float(0x7f800000);
#pragma unroll
  for (int offset = 0; offset < Block; ++offset) {
    const uint32_t i = block * Block + offset;
    if (i < count) {
      const float score = scores[size_t(row) * width + i];
      if (isnan(score) || score == __int_as_float(0x7f800000)) asm volatile("trap;");
      value = fmaxf(value, score);
    }
  }
  const uint32_t newest = WrongPin ? count / Block : (count == 0 ? 0 : (count - 1) / Block);
  if (count != 0 && block == newest) value = __int_as_float(0x7f800000);
  blocks[size_t(row) * block_count + block] = value;
}

template<int Block, int TopBlocks>
__device__ void block_mask(const float* blocks, const uint32_t* live_rows,
                           uint8_t* output, uint32_t width) {
  __shared__ uint32_t histogram[threads], rank;
  __shared__ uint64_t prefix;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t block_count = (width + Block - 1) / Block;
  const float* values = blocks + size_t(row) * block_count;
  auto key = [&](uint32_t i) {
    return values[i] > -__int_as_float(0x7f800000) ? candidate_key(values[i], i) : uint64_t(0);
  };
  const uint64_t threshold = cutoff(block_count, TopBlocks, key, histogram, &prefix, &rank);
  for (uint32_t i = threadIdx.x; i < width; i += threads) {
    const uint64_t value = key(i / Block);
    output[size_t(row) * width + i] = value != 0 && value >= threshold;
  }
}

}  // namespace chop::position_indexer
