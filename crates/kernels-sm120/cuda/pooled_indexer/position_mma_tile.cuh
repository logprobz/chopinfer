#pragma once
#include "pooled_indexer/position_mma_guard.cuh"
#include "pooled_indexer/position_mma_guard_fast.cuh"

namespace chop::position_mma_tile {
using namespace position_mma_guard_fast;
// Diagnostic stamps are separate entry points. Clocks include observer cost.
__device__ __forceinline__ uint64_t stamp() {
  uint64_t value; asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory"); return value;
}
__device__ __forceinline__ uint64_t global_time() {
  uint64_t value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory"); return value;
}

using position_mma_guard::simt;
using position_mma_guard::exact_scales;
template<int Tiles, int Mode, bool Trace, int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t*, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus,
    uint64_t* trace = nullptr) {
  static_assert((Tiles == 1 || Tiles == 4) && Mode >= 0 && Mode <= 2);
  constexpr uint32_t Slots = 32 * Tiles;
  static_assert(Heads == 32 && Dim == 128 && !Mask && !Bf16);
  // Rotate the five slot bits into the bank index. For the MMA scatter,
  // group selects the low three bank bits and part the high two. For the
  // ordered epilogue, each lane has a different slot permutation.
  auto column = [](uint32_t slot, uint32_t head) {
    if constexpr (Tiles == 1) return head;
    else return head ^ (((slot & 7u) << 2) | ((slot >> 3) & 3u));
  };
  __shared__ float dots[Slots][Tiles == 1 ? 33 : 32];
  __shared__ float weight[32];
  __shared__ uint32_t bad[Slots], repair[Slots];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (row >= *live_rows) return;
  uint64_t ticks[7] = {}, last = 0, begin = 0;
  uint32_t tiles = 0;
  if constexpr (Trace) { begin = global_time(); last = stamp(); }
  auto mark = [&](int stage) {
    if constexpr (Trace) { const uint64_t now = stamp(); ticks[stage] += now - last; last = now; }
  };
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
  // Every warp already owns all 32 query scale words, duplicated in pairs.
  // Reduce their bounds once, before walking the device-visible key slots.
  const Range q0 = scale_range(sa[0]), q1 = scale_range(sa[1]);
  const Range qr{__reduce_min_sync(0xffffffffu,min(q0.lo,q1.lo)),
      __reduce_max_sync(0xffffffffu,max(q0.hi,q1.hi))};
  const uint32_t qspread = __reduce_max_sync(0xffffffffu,
      max(q0.hi - q0.lo,q1.hi - q1.lo));
  const bool qbad = qstatus[row] != 0;
  __syncthreads();
  mark(0);
  for (uint32_t base = blockIdx.x * Slots; base < count; base += gridDim.x * Slots) {
#pragma unroll 1
   for (uint32_t tile = 0; tile < Tiles; ++tile) {
    const uint32_t slot = tile * 32 + warp * 8 + group, index = base + slot;
    uint32_t checksum = 0;
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
    mark(1);
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),1) != 0;
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),2) != 0;
    const bool key_safe = coarse_scales(qr,qspread,scale_range(sb));
    // The slow branch contains warp shuffles, so its predicate is warp-uniform.
    const bool fast = Mode == 1 || __all_sync(0xffffffffu,!admitted || key_safe);
    if (part == 0) {
      bad[slot] = invalid;
    }
    mark(2);
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
        if constexpr (Mode == 2) checksum ^= __float_as_uint(dot);
        else {
          const uint32_t target = tile * 32 + warp * 8 + 2 * part + i % 2;
          dots[target][column(target,m * 16 + group + (i / 2) * 8)] = dot;
        }
      }
    }
    if constexpr (Mode == 2) {
      // All MMA results feed an observable checksum. This ablation is never
      // selected for serving and does not claim valid scores or selections.
      const uint32_t sum = __reduce_xor_sync(0xffffffffu,checksum);
      if (lane == 0 && base + tile * 32 + warp * 8 < count)
        reinterpret_cast<volatile uint32_t*>(output)[size_t(row) * width + base + tile * 32 + warp * 8] = sum;
    }
    mark(3); ++tiles;
   }
   if constexpr (Mode != 2) {
    __syncthreads(); mark(4);
    bool needs_repair = false;
    if (threadIdx.x < Slots) {
      const uint32_t index = base + threadIdx.x;
      if (index < count) {
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
        output[size_t(row) * width + index] = result;
      }
      repair[threadIdx.x] = needs_repair;
    }
    // Publish the repair flags and finish shared reads before the next tile.
    mark(5);
    const bool any_repair = __syncthreads_or(needs_repair);
    if (any_repair) {
    for (uint32_t s = warp; s < Slots && base + s < count; s += 4) {
      if (repair[s]) {
        const uint32_t index = base + s;
        const float result = simt(queries,weights,codes,scales,row,index,
            uint32_t(table[index / Page]),code_stride,scale_stride,weight_scale,Page);
        if (lane == 0) output[size_t(row) * width + base + s] = result;
      }

    }
    __syncthreads();
    }
     }
    mark(6);
  }
  if constexpr (Trace) {
    if (threadIdx.x == 0) {
      const uint64_t end = global_time();
      uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      uint64_t* record = trace + (size_t(row) * gridDim.x + blockIdx.x) * 12;
      record[0] = begin; record[1] = end; record[2] = sm; record[3] = tiles;
#pragma unroll
      for (int i = 0; i < 7; ++i) record[4+i] = ticks[i];
      record[11] = 1;
    }
  }
}
} // namespace chop::position_mma_tile
