#pragma once
#include "position_live.cuh"

// Lab-only block-ordered MMA experiment. It deliberately makes no bitwise
// claim: the served dot first adds strided blocks within each lane.
namespace chop::position_mma_probe {
template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t* mask, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale) {
  static_assert(Heads == 32 && Dim == 128 && !Mask && !Bf16);
  __shared__ uint16_t q[Heads * Dim];
  __shared__ uint16_t k[32 * Dim];
  __shared__ float dots[32 * Heads];
  __shared__ uint32_t bad[32];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (row >= *live_rows) return;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio), width);
  for (uint32_t i = count + blockIdx.x * blockDim.x + threadIdx.x;
       i < width; i += gridDim.x * blockDim.x)
    output[size_t(row) * width + i] = -__int_as_float(0x7f800000);
  for (int i = threadIdx.x; i < Heads * Dim; i += blockDim.x)
    q[i] = queries[size_t(row) * Heads * Dim + i];
  const int32_t* table = tables + size_t(batches[row]) * max_pages;
  __syncthreads();
  for (uint32_t base = blockIdx.x * 32; base < count; base += gridDim.x * 32) {
    for (int slot = threadIdx.x; slot < 32; slot += blockDim.x) bad[slot] = 0;
    __syncthreads();
    for (int pair = threadIdx.x; pair < 32 * Dim / 2; pair += blockDim.x) {
      const int slot = pair / (Dim / 2), channel = pair % (Dim / 2);
      const uint32_t index = base + slot;
      uint32_t value = 0;
      if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
        const uint32_t page = uint32_t(table[index / Page]);
        const uint8_t scale = scales[uint64_t(page) * scale_stride + size_t(index % Page) * 4 + channel / 16];
        const uint8_t code = codes[uint64_t(page) * code_stride + size_t(index % Page) * 64 + channel];
        value = convert::scaled_mxfp4_bf16x2<false>(code, scale);
        const auto decoded = convert::bf16x2_to_f32(value);
        if (scale == 255 || !isfinite(decoded.lo) || !isfinite(decoded.hi)) atomicOr(bad + slot,1u);
      }
      k[2 * pair] = uint16_t(value); k[2 * pair + 1] = uint16_t(value >> 16);
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ptx::D4 total = {};
#pragma unroll
      for (int block = 0; block < 4; ++block) {
        ptx::D4 part = {};
#pragma unroll
        for (int step = 0; step < 2; ++step) {
          const int c = block * 32 + step * 16 + (lane % 4) * 2;
          const int h = m * 16 + lane / 4;
          auto pair = [](const uint16_t* x) { return uint32_t(x[0]) | (uint32_t(x[1]) << 16); };
          ptx::A4 a{{pair(q+h*Dim+c), pair(q+(h+8)*Dim+c),
                     pair(q+h*Dim+c+8), pair(q+(h+8)*Dim+c+8)}};
          const int slot = warp * 8 + lane / 4;
          ptx::B2 b{{pair(k+slot*Dim+c), pair(k+slot*Dim+c+8)}};
          ptx::mma_bf16(part,a,b);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) total.v[i] = __fadd_rn(total.v[i],part.v[i]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int head = m * 16 + lane / 4 + (i / 2) * 8;
        const int slot = warp * 8 + 2 * (lane % 4) + i % 2;
        dots[slot * Heads + head] = total.v[i];
      }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const int slot = threadIdx.x;
      const uint32_t index = base + slot;
      if (index < count) {
        float result = -__int_as_float(0x7f800000);
        if (position_indexer::admitted<Page,false>(index,count,nullptr,table)) {
          result = 0.f;
          bool invalid = bad[slot] != 0;
#pragma unroll 1
          for (int h = 0; h < Heads; ++h) {
            const float dot = dots[slot * Heads + h];
            const float weight = __fmul_rn(weights[size_t(row) * Heads + h],weight_scale);
            result = fmaf(dot > 0.f ? dot : 0.f,weight,result);
            invalid |= !isfinite(dot) || !isfinite(weight) || !isfinite(result);
          }
          if (invalid) result = __int_as_float(0x7fc00000);
        }
        output[size_t(row) * width + index] = result;
      }
    }
    __syncthreads();
  }
}
} // namespace chop::position_mma_probe
