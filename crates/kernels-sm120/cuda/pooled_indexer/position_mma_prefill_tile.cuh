#pragma once
#include "pooled_indexer/position_mma_guard_fast.cuh"

namespace chop::position_mma_prefill_tile {
using namespace position_mma_guard_fast;
// Each CTA retains the packed query while visiting 128 compact slots per
// iteration. Every slot keeps the same MMA, exactness guard and head-FMA
// order. The caller may reduce the x grid for batches with many live rows.
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
  constexpr uint32_t Slots = 128;
  // This permutation keeps MMA stores and ordered head reads bank-disjoint.
  auto column = [](uint32_t slot, uint32_t head) {
    return head ^ (((slot & 7u) << 2) | ((slot >> 3) & 3u));
  };
  __shared__ float dots[Slots][32];
  __shared__ float weight[32];
  __shared__ uint32_t bad[Slots], repair[Slots], coarse[Slots], fine[Slots];
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
  // Every warp already owns all 32 query scale words, duplicated in pairs.
  // Reduce their bounds once, before walking the device-visible key slots.
  const Range q0 = scale_range(sa[0]), q1 = scale_range(sa[1]);
  const Range qr{__reduce_min_sync(0xffffffffu,min(q0.lo,q1.lo)),
      __reduce_max_sync(0xffffffffu,max(q0.hi,q1.hi))};
  const uint32_t qspread = __reduce_max_sync(0xffffffffu,
      max(q0.hi - q0.lo,q1.hi - q1.lo));
  const bool qbad = qstatus[row] != 0;
  __syncthreads();
  for (uint32_t base = blockIdx.x * Slots; base < active; base += gridDim.x * Slots) {
#pragma unroll 1
    for (uint32_t tile = 0; tile < 4; ++tile) {
    const uint32_t slot = tile * 32 + warp * 8 + group, index = mapped(base + slot);
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
    const bool key_safe = coarse_scales(qr,qspread,scale_range(sb));
    // The slow branch contains warp shuffles, so its predicate is warp-uniform.
    const bool fast = __all_sync(0xffffffffu,!admitted || key_safe);
    if (part == 0) {
      bad[slot] = invalid;
      if (guard_counts) { coarse[slot] = key_safe; fine[slot] = !fast; }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ptx::D4 result = {};
      ptx::mma_mxf4nvf4<2,false,0,0,0,0>(result,a[m][0],b[0],sa[m],sb);
      ptx::mma_mxf4nvf4<2,false,2,0,2,0>(result,a[m][1],b[1],sa[m],sb);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float dot = result.v[i];
        if (!fast) {
          const uint32_t qs = __shfl_sync(0xffffffffu,sa[m],(lane & ~3u) + i / 2);
          const uint32_t ks = __shfl_sync(0xffffffffu,sb,(2 * part + i % 2) * 4);
          if (!exact_scales(qs,ks)) dot = __int_as_float(0x7fc00000);
        }
        const uint32_t target = tile * 32 + warp * 8 + 2 * part + i % 2;
        dots[target][column(target,m * 16 + group + (i / 2) * 8)] = dot;
      }
    }
    }
    __syncthreads();
    bool needs_repair = false;
    if (threadIdx.x < Slots) {
      const uint32_t slot = base + threadIdx.x, index = mapped(slot);
      if (slot < active) {
        float result = -__int_as_float(0x7f800000);
        if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
          result = 0.f;
          bool invalid = bad[threadIdx.x] != 0;
#pragma unroll 1
          for (int h = 0; h < 32; ++h) {
            const float dot = dots[threadIdx.x][column(threadIdx.x,h)];
            needs_repair |= isnan(dot);
            result = fmaf(dot > 0.f ? dot : 0.f,weight[h],result);
            invalid |= !isfinite(dot) || !isfinite(weight[h]) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(row) * extent + slot] = result;
      }
      repair[threadIdx.x] = needs_repair;
    }
    // Publish the repair flags and finish shared reads before the next tile.
    const bool any_repair = __syncthreads_or(needs_repair);
    if (any_repair || guard_counts) {
    for (uint32_t s = warp; s < Slots && base + s < active; s += 4) {
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
        if (coarse[s]) atomicAdd(guard_counts + 2,1u);
        if (fine[s]) atomicAdd(guard_counts + 3,1u);
      }
    }
    __syncthreads();
    }
  }
}
} // namespace chop::position_mma_prefill_tile
