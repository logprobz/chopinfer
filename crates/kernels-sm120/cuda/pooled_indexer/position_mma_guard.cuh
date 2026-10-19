#pragma once
#include "pooled_indexer/position_mma_compact.cuh"
#include "pooled_indexer/position_mma_guard_bound.h"

namespace chop::position_mma_guard {
using position_indexer::bf16;
// Called by a whole warp for one admitted score. Only lane zero stores it.
__device__ __forceinline__ float simt(
    const uint16_t* queries, const float* weights, const uint8_t* codes,
    const uint8_t* scales, uint32_t row, uint32_t index, uint32_t page,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    uint32_t page_size) {
  constexpr int Heads = 32, Dim = 128;
  constexpr bool Bf16 = false;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t Page = page_size;
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
  return failed ? __int_as_float(0x7fc00000) : result;
}
template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t*, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus,
    uint32_t* guard_counts = nullptr) {
  static_assert(Heads == 32 && Dim == 128 && !Mask && !Bf16);
  __shared__ float dots[32][33];
  __shared__ float weight[32];
  __shared__ uint32_t bad[32], repair[32];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (row >= *live_rows) return;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio),width);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  for (uint32_t i = count + blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x)
    output[size_t(row) * width + i] = -__int_as_float(0x7f800000);
  if (threadIdx.x < 32) weight[threadIdx.x] = __fmul_rn(weights[size_t(row) * 32 + threadIdx.x],weight_scale);
  const uint32_t group = lane >> 2, part = lane & 3;
  ptx::A4 a[2][2];
  uint32_t sa[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    sa[m] = *reinterpret_cast<const uint32_t*>(qscales + size_t(row) * 128 + (m * 16 + group + (part & 1) * 8) * 4);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = m * 16 + group + (i % 2) * 8;
        const int c = kk * 64 + (i / 2) * 32 + part * 8;
        a[m][kk].v[i] = *reinterpret_cast<const uint32_t*>(qcodes + size_t(row) * 2048 + h * 64 + c / 2);
      }
    }
  }
  const bool qbad = qstatus[row] != 0;
  __syncthreads();
  for (uint32_t base = blockIdx.x * 32; base < count; base += gridDim.x * 32) {
    const uint32_t slot = warp * 8 + group, index = base + slot;
    const bool admitted = position_indexer::admitted<Page,false>(index,count,nullptr,table);
    ptx::B2 b[2] = {};
    uint32_t sb = 0x7f7f7f7fu;
    bool invalid = qbad;
    if (admitted) {
      const uint32_t page = uint32_t(table[index / Page]);
      const uint8_t* k = codes + uint64_t(page) * code_stride + size_t(index % Page) * 64;
      sb = *reinterpret_cast<const uint32_t*>(scales + uint64_t(page) * scale_stride + size_t(index % Page) * 4);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          b[kk].v[i] = *reinterpret_cast<const uint32_t*>(k + kk * 32 + i * 16 + part * 4);
          invalid |= position_mma_native::invalid_word(b[kk].v[i],uint8_t(sb >> ((kk * 2 + i) * 8)));
        }
      }
    }
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),1) != 0;
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),2) != 0;
    if (part == 0) bad[slot] = invalid;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ptx::D4 result = {};
      ptx::mma_mxf4nvf4<2,false,0,0,0,0>(result,a[m][0],b[0],sa[m],sb);
      ptx::mma_mxf4nvf4<2,false,2,0,2,0>(result,a[m][1],b[1],sa[m],sb);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t qs = __shfl_sync(0xffffffffu,sa[m],(lane & ~3u) + i / 2);
        const uint32_t ks = __shfl_sync(0xffffffffu,sb,(2 * part + i % 2) * 4);
        dots[warp * 8 + 2 * part + i % 2][m * 16 + group + (i / 2) * 8] =
            exact_scales(qs,ks) ? result.v[i] : __int_as_float(0x7fc00000);
      }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const uint32_t index = base + threadIdx.x;
      if (index < count) {
        repair[threadIdx.x] = 0;
        float result = -__int_as_float(0x7f800000);
        if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
          result = 0.f;
          bool invalid = bad[threadIdx.x] != 0;
#pragma unroll 1
          for (int h = 0; h < 32; ++h) {
            const float dot = dots[threadIdx.x][h];
            repair[threadIdx.x] |= isnan(dot);
            result = fmaf(dot > 0.f ? dot : 0.f,weight[h],result);
            invalid |= !isfinite(dot) || !isfinite(weight[h]) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(row) * width + index] = result;
      }
    }
    __syncthreads();
    for (uint32_t s = warp; s < 32 && base + s < count; s += 4) {
      if (repair[s]) {
        const uint32_t index = base + s;
        const float result = simt(queries,weights,codes,scales,row,index,
            uint32_t(table[index / Page]),code_stride,scale_stride,weight_scale,Page);
        if (lane == 0) output[size_t(row) * width + base + s] = result;
      }
      if (guard_counts && lane == 0 && position_indexer::admitted<Page,false>(
              base + s,count,nullptr,table)) {
        atomicAdd(guard_counts,1u);
        if (repair[s]) atomicAdd(guard_counts + 1,1u);
      }
    }
    __syncthreads();
  }
}

template<int Heads, int Dim, int Page, int Ratio, int Block, int Kept>
__device__ void compact_scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const int32_t* ids, const uint64_t* tags, const uint64_t* stored_tags,
    const int32_t* stored_visible, const int32_t* source_status,
    float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus,
    uint32_t* guard_counts = nullptr) {
  static_assert(Heads == 32 && Dim == 128 && Block == 8);
  __shared__ float dots[32][33];
  __shared__ float weight[32];
  __shared__ uint32_t bad[32], repair[32];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (row >= *live_rows) return;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio),width);
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  const uint32_t extent = Block * Kept;
  const int32_t* row_ids = ids + size_t(row) * Kept;
  const uint32_t active = min(position_valid::retained_slots<Block,Kept>(row_ids), ((count + Block - 1) / Block) * Block);
  const bool row_valid = source_status[row] == 0 && tags[row] == stored_tags[row] && stored_visible[row] == int32_t(count);
  auto mapped = [&](uint32_t slot) {
    if (slot >= active || !row_valid) return width;
    const int32_t id = row_ids[slot / Block];
    const uint64_t index = id < 0 ? uint64_t(width) : uint64_t(id) * Block + slot % Block;
    return index < count ? uint32_t(index) : width;
  };
  for (uint32_t i = active + blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += gridDim.x * blockDim.x)
    output[size_t(row) * extent + i] = -__int_as_float(0x7f800000);
  if (threadIdx.x < 32) weight[threadIdx.x] = __fmul_rn(weights[size_t(row) * 32 + threadIdx.x],weight_scale);
  const uint32_t group = lane >> 2, part = lane & 3;
  ptx::A4 a[2][2];
  uint32_t sa[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    sa[m] = *reinterpret_cast<const uint32_t*>(qscales + size_t(row) * 128 + (m * 16 + group + (part & 1) * 8) * 4);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = m * 16 + group + (i % 2) * 8;
        const int c = kk * 64 + (i / 2) * 32 + part * 8;
        a[m][kk].v[i] = *reinterpret_cast<const uint32_t*>(qcodes + size_t(row) * 2048 + h * 64 + c / 2);
      }
    }
  }
  const bool qbad = qstatus[row] != 0;
  __syncthreads();
  for (uint32_t base = blockIdx.x * 32; base < active; base += gridDim.x * 32) {
    const uint32_t slot = warp * 8 + group, index = mapped(base + slot);
    const bool admitted = position_indexer::admitted<Page,false>(index,count,nullptr,table);
    ptx::B2 b[2] = {};
    uint32_t sb = 0x7f7f7f7fu;
    bool invalid = qbad;
    if (admitted) {
      const uint32_t page = uint32_t(table[index / Page]);
      const uint8_t* k = codes + uint64_t(page) * code_stride + size_t(index % Page) * 64;
      sb = *reinterpret_cast<const uint32_t*>(scales + uint64_t(page) * scale_stride + size_t(index % Page) * 4);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          b[kk].v[i] = *reinterpret_cast<const uint32_t*>(k + kk * 32 + i * 16 + part * 4);
          invalid |= position_mma_native::invalid_word(b[kk].v[i],uint8_t(sb >> ((kk * 2 + i) * 8)));
        }
      }
    }
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),1) != 0;
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),2) != 0;
    if (part == 0) bad[slot] = invalid;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ptx::D4 result = {};
      ptx::mma_mxf4nvf4<2,false,0,0,0,0>(result,a[m][0],b[0],sa[m],sb);
      ptx::mma_mxf4nvf4<2,false,2,0,2,0>(result,a[m][1],b[1],sa[m],sb);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t qs = __shfl_sync(0xffffffffu,sa[m],(lane & ~3u) + i / 2);
        const uint32_t ks = __shfl_sync(0xffffffffu,sb,(2 * part + i % 2) * 4);
        dots[warp * 8 + 2 * part + i % 2][m * 16 + group + (i / 2) * 8] =
            exact_scales(qs,ks) ? result.v[i] : __int_as_float(0x7fc00000);
      }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const uint32_t slot = base + threadIdx.x, index = mapped(slot);
      if (slot < active) {
        repair[threadIdx.x] = 0;
        float result = -__int_as_float(0x7f800000);
        if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
          result = 0.f;
          bool invalid = bad[threadIdx.x] != 0;
#pragma unroll 1
          for (int h = 0; h < 32; ++h) {
            const float dot = dots[threadIdx.x][h];
            repair[threadIdx.x] |= isnan(dot);
            result = fmaf(dot > 0.f ? dot : 0.f,weight[h],result);
            invalid |= !isfinite(dot) || !isfinite(weight[h]) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(row) * extent + slot] = result;
      }
    }
    __syncthreads();
    for (uint32_t s = warp; s < 32 && base + s < active; s += 4) {
      if (repair[s]) {
        const uint32_t index = mapped(base + s);
        const float result = simt(queries,weights,codes,scales,row,index,
            uint32_t(table[index / Page]),code_stride,scale_stride,weight_scale,Page);
        if (lane == 0) output[size_t(row) * extent + base + s] = result;
      }
      if (guard_counts && lane == 0 && position_indexer::admitted<Page,false>(
              mapped(base + s),count,nullptr,table)) {
        atomicAdd(guard_counts,1u);
        if (repair[s]) atomicAdd(guard_counts + 1,1u);
      }
    }
    __syncthreads();
  }
}
} // namespace chop::position_mma_guard
