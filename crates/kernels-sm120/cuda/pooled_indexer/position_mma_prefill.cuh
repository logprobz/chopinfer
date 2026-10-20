#pragma once
#include "pooled_indexer/position_mma_guard_fast.cuh"

namespace chop::position_mma_prefill {
// Four independent query rows share eight keys when they name one sequence.
// Mixed sequences retain separate key banks; every warp reaches every barrier.
template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t*, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus) {
  static_assert(Heads == 32 && Dim == 128 && !Mask && !Bf16);
  using namespace position_mma_guard_fast;
  __shared__ uint32_t key[4][8][16], scale[4][8], present[4][8];
  __shared__ float dots[32][33], weight[4][32];
  __shared__ uint32_t counts[4], pack_bad[4], bad[32], repair[32];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t first = blockIdx.y * 4, lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5, row = first + warp;
  const uint32_t live = *live_rows;
  if (first >= live) return;
  const bool active = row < live;
  const uint32_t count = active ? min(position_indexer::visible(positions[row],Ratio),width) : 0;
  const int32_t batch = active ? batches[row] : batches[first];
  const int32_t* table = tables + size_t(batch) * max_pages;
  const bool common = __syncthreads_and(batch == batches[first]);
  if (lane == 0) {
    counts[warp] = count;
    pack_bad[warp] = active ? qstatus[row] : 0;
  }
  if (active) {
    for (uint32_t i = count + blockIdx.x * 32 + lane; i < width; i += gridDim.x * 32)
      output[size_t(row) * width + i] = -__int_as_float(0x7f800000);
  }
  weight[warp][lane] = active ? __fmul_rn(weights[size_t(row) * 32 + lane],weight_scale) : 0.f;
  __syncthreads();
  const uint32_t extent = max(max(counts[0],counts[1]),max(counts[2],counts[3]));
  const bool qbad = pack_bad[warp] != 0;
  const uint32_t group = lane >> 2, part = lane & 3;
  ptx::A4 a[2][2] = {};
  uint32_t sa[2] = {0x7f7f7f7fu,0x7f7f7f7fu};
  if (active && !qbad) {
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
  }
  const Range q0 = scale_range(sa[0]), q1 = scale_range(sa[1]);
  const Range qr{__reduce_min_sync(0xffffffffu,min(q0.lo,q1.lo)),
      __reduce_max_sync(0xffffffffu,max(q0.hi,q1.hi))};
  const uint32_t qspread = __reduce_max_sync(0xffffffffu,max(q0.hi-q0.lo,q1.hi-q1.lo));
  for (uint32_t base = blockIdx.x * 8; base < extent; base += gridDim.x * 8) {
    const uint32_t bank = common ? 0 : warp;
    if (!common || warp == 0) {
      const uint32_t index = base + group;
      const bool valid = position_indexer::admitted<Page,false>(index,common ? extent : count,nullptr,table);
      const uint8_t* source = nullptr;
      uint32_t sb = 0x7f7f7f7fu;
      if (valid) {
        const uint32_t page = uint32_t(table[index / Page]);
        source = codes + uint64_t(page) * code_stride + size_t(index % Page) * 64;
        sb = *reinterpret_cast<const uint32_t*>(scales + uint64_t(page) * scale_stride + size_t(index % Page) * 4);
      }
      if (part == 0) { scale[bank][group] = sb; present[bank][group] = valid; }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        key[bank][group][i * 4 + part] = valid ? *reinterpret_cast<const uint32_t*>(source + i * 16 + part * 4) : 0;
    }
    __syncthreads();
    const bool admitted = active && base + group < count && present[bank][group];
    ptx::B2 b[2] = {};
    uint32_t sb = admitted ? scale[bank][group] : 0x7f7f7f7fu;
    bool invalid = false;
    if (admitted) {
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          b[kk].v[i] = key[bank][group][kk * 8 + i * 4 + part];
          invalid |= position_mma_native::invalid_word(b[kk].v[i],uint8_t(sb >> ((kk * 2 + i) * 8)));
        }
      }
    }
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),1) != 0;
    invalid |= __shfl_xor_sync(0xffffffffu,unsigned(invalid),2) != 0;
    if (part == 0) bad[warp * 8 + group] = invalid;
    const bool fast = __all_sync(0xffffffffu,!admitted || coarse_scales(qr,qspread,scale_range(sb)));
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ptx::D4 result = {};
      if (active && !qbad) {
        ptx::mma_mxf4nvf4<2,false,0,0,0,0>(result,a[m][0],b[0],sa[m],sb);
        ptx::mma_mxf4nvf4<2,false,2,0,2,0>(result,a[m][1],b[1],sa[m],sb);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float dot = result.v[i];
        if (!fast) {
          const uint32_t qs = __shfl_sync(0xffffffffu,sa[m],(lane & ~3u) + i / 2);
          const uint32_t ks = __shfl_sync(0xffffffffu,sb,(2 * part + i % 2) * 4);
          if (!exact_scales(qs,ks)) dot = __int_as_float(0x7fc00000);
        }
        dots[warp * 8 + 2 * part + i % 2][m * 16 + group + (i / 2) * 8] = dot;
      }
    }
    __syncthreads();
    bool needs_repair = false;
    if (warp == 0) {
      const uint32_t r = lane / 8, slot = lane % 8, target = first + r, index = base + slot;
      const uint32_t source_bank = common ? 0 : r;
      if (target < live && index < counts[r]) {
        float result = -__int_as_float(0x7f800000);
        if (present[source_bank][slot]) {
          needs_repair = pack_bad[r] != 0;
          bool invalid = bad[lane] != 0;
          result = 0.f;
#pragma unroll 1
          for (int h = 0; h < 32; ++h) {
            const float dot = dots[lane][h];
            needs_repair |= isnan(dot);
            result = fmaf(dot > 0.f ? dot : 0.f,weight[r][h],result);
            invalid |= !isfinite(dot) || !isfinite(weight[r][h]) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(target) * width + index] = result;
      }
      repair[lane] = needs_repair;
    }
    const bool any_repair = __syncthreads_or(needs_repair);
    if (any_repair) {
      for (uint32_t slot = 0; slot < 8 && base + slot < count; ++slot) {
        if (repair[warp * 8 + slot]) {
          const uint32_t index = base + slot;
          const float result = simt(queries,weights,codes,scales,row,index,uint32_t(table[index / Page]),
              code_stride,scale_stride,weight_scale,Page);
          if (lane == 0) output[size_t(row) * width + index] = result;
        }
      }
      __syncthreads();
    }
  }
}
} // namespace chop::position_mma_prefill
