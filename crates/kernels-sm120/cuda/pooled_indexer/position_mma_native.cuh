#pragma once
#include "position_live.cuh"
#include "row/ceil_parallel.cuh"

// Native FP4 lab candidate. Exact operand repacking is checked separately
// from the deliberately different tensor-core accumulation order.
namespace chop::position_mma_native {
__device__ void pack(const uint16_t* input, const uint32_t* live,
    uint8_t* codes, uint8_t* scales, uint32_t* status) {
  const uint32_t row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (row >= *live) return;
  uint32_t flags = 0;
  for (uint32_t group = warp; group < 128; group += 8) {
    const uint64_t at = size_t(row) * 4096 + group * 32 + lane * 2;
    const auto value = row::block_quantize_parallel::quantize_group<2,false,false,0>(input,at);
    if (lane < 16) {
      flags |= value.flags;
      const uint32_t original = uint32_t(input[at]) | (uint32_t(input[at + 1]) << 16);
      flags |= value.pair != original;
      codes[size_t(row) * 2048 + group * 16 + lane] = uint8_t(value.codes);
      if (lane == 0) scales[size_t(row) * 128 + group] = value.scale_code;
    }
  }
  __shared__ uint32_t failed[8];
  flags = __reduce_or_sync(0xffffffffu,flags);
  if (lane == 0) failed[warp] = flags;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) result |= failed[i];
    status[row] = result;
  }
}

__device__ __forceinline__ bool invalid_word(uint32_t word, uint8_t scale) {
  if (scale == 255) return true;
  if (scale <= 252) return false;
  const unsigned limit = scale == 253 ? 6 : 4;
#pragma unroll
  for (int i = 0; i < 8; ++i) if (((word >> (i * 4)) & 7) >= limit) return true;
  return false;
}

template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void scores(const uint16_t*, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t*, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    const uint8_t* qcodes, const uint8_t* qscales, const uint32_t* qstatus) {
  static_assert(Heads == 32 && Dim == 128 && !Mask && !Bf16);
  __shared__ float dots[32][33];
  __shared__ float weight[32];
  __shared__ uint32_t bad[32];
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
          invalid |= invalid_word(b[kk].v[i],uint8_t(sb >> ((kk * 2 + i) * 8)));
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
      const uint32_t index = base + threadIdx.x;
      if (index < count) {
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
        output[size_t(row) * width + index] = result;
      }
    }
    __syncthreads();
  }
}
} // namespace chop::position_mma_native
