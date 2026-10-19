#pragma once
#include "pooled_indexer/position_mma_native.cuh"
#include "pooled_indexer/position_valid.cuh"

// The qualified native arithmetic over retained candidate slots. Only
// position mapping, provenance admission and output stride differ from the
// dense scorer. The unchanged valid selector checks every ID, including
// malformed values after the retained prefix, and owns failure publication.
namespace chop::position_mma_compact {
template<int Heads, int Dim, int Page, int Ratio, int Block, int Kept>
__device__ void scores(const uint16_t*, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const int32_t* ids, const uint64_t* tags, const uint64_t* stored_tags,
    const int32_t* stored_visible, const int32_t* source_status,
    float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus) {
  static_assert(Heads == 32 && Dim == 128 && Block == 8);
  __shared__ float dots[32][33];
  __shared__ float weight[32];
  __shared__ uint32_t bad[32];
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
      for (int i = 0; i < 4; ++i)
        dots[warp * 8 + 2 * part + i % 2][m * 16 + group + (i / 2) * 8] = result.v[i];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const uint32_t slot = base + threadIdx.x, index = mapped(slot);
      if (slot < active) {
        float result = -__int_as_float(0x7f800000);
        if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
          result = 0.f;
          bool invalid = bad[threadIdx.x] != 0;
#pragma unroll 1
          for (int h = 0; h < 32; ++h) {
            const float dot = dots[threadIdx.x][h];
            result = fmaf(dot > 0.f ? dot : 0.f,weight[h],result);
            invalid |= !isfinite(dot) || !isfinite(weight[h]) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(row) * extent + slot] = result;
      }
    }
    __syncthreads();
  }
}
} // namespace chop::position_mma_compact
