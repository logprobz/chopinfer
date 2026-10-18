#pragma once

// Comparison storage only. Native score and selection source closures stay
// unchanged. The FMA chains below match their native counterparts exactly;
// the only production difference is the load of already-decoded BF16 keys.
#include "pooled_indexer/position_blocks.cuh"

namespace chop::position_bf16 {
using namespace position_indexer;

// Lab-only mutant: an extra BF16 rounding inside the F32 accumulator.
__device__ __forceinline__ float round_chain(float value) {
  uint16_t bits;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  return bf16(bits);
}

template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool RoundChain = false>
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
  const uint32_t index = blockIdx.x * 4 + threadIdx.x / 32;
  const uint32_t lane = threadIdx.x & 31;
  if (row >= *live_rows || index >= width) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint8_t* row_mask = Mask ? mask + size_t(row) * width : nullptr;
  float* out = output + size_t(row) * width + index;
  if (!admitted<Page, Mask, false>(index, count, row_mask, table)) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    return;
  }
  const uint32_t page = uint32_t(table[index / Page]);
  const uint16_t* values = reinterpret_cast<const uint16_t*>(codes + uint64_t(page) * code_stride) + size_t(index % Page) * Dim;
  float key[Dim / 32];
  bool invalid = false;
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    key[part] = bf16(values[part * 32 + lane]);
    invalid |= !isfinite(key[part]);
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
        if constexpr (RoundChain) chain = round_chain(chain);
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

template<int Heads, int Dim, int Page, int Ratio, int Block, int Kept, bool RoundChain = false>
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
  const uint32_t slot = blockIdx.x * 4 + threadIdx.x / 32;
  const uint32_t lane = threadIdx.x & 31;
  if (row >= *live_rows || slot >= Block * Kept) return;
  const uint32_t count = min(visible(positions[row], Ratio), width);
  const int32_t id = ids[size_t(row) * Kept + slot / Block];
  const uint64_t mapped = id < 0 ? uint64_t(width) : uint64_t(id) * Block + slot % Block;
  float* out = output + size_t(row) * (Block * Kept) + slot;
  if (source_status[row] != 0 || (true && (tags[row] != stored_tags[row] || stored_visible[row] != int32_t(count))) || mapped >= count) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    return;
  }
  const uint32_t index = uint32_t(mapped);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  if (!admitted<Page, false>(index, count, nullptr, table)) {
    if (lane == 0) *out = -__int_as_float(0x7f800000);
    return;
  }
  // Keep the admitted-position arithmetic identical to position.cuh:
  // same decoded values, 128-channel chains, trees and head FMA order.
  const uint32_t page = uint32_t(table[index / Page]);
  const uint16_t* values = reinterpret_cast<const uint16_t*>(codes + uint64_t(page) * code_stride) + size_t(index % Page) * Dim;
  float key[Dim / 32];
  bool invalid = false;
#pragma unroll
  for (int part = 0; part < Dim / 32; ++part) {
    key[part] = bf16(values[part * 32 + lane]);
    invalid |= !isfinite(key[part]);
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
        if constexpr (RoundChain) chain = round_chain(chain);
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

}  // namespace chop::position_bf16
