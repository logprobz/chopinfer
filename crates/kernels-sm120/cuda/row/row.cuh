#pragma once

#include <stdint.h>
#include <math.h>
#include "../common/ptx.cuh"
#include "../common/convert.cuh"

namespace chop::row {

__device__ __forceinline__ void begin() {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
}

__device__ __forceinline__ float bf(uint16_t x) {
  return convert::bf16_to_f32(x);
}
__device__ __forceinline__ uint16_t to_bf(float x) {
  return convert::bf16_from_f32(x);
}

// A complete row is assigned to one CTA. The eight BF16 values in each
// uint4 transaction are 128-bit aligned when row widths are multiples of 8.
__device__ __forceinline__ float sum_cta(float x, float* warp_sums) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_down_sync(0xffffffff, x, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = x;
  __syncthreads();
  float total = threadIdx.x < (blockDim.x >> 5) ? warp_sums[threadIdx.x] : 0.f;
  for (int offset = 16; offset; offset >>= 1)
    total += __shfl_down_sync(0xffffffff, total, offset);
  if (threadIdx.x == 0) warp_sums[0] = total;
  __syncthreads();
  return warp_sums[0];
}

template<int Width, int Head, int Kind, bool Residual>
__device__ void rmsnorm(const uint16_t* x, const uint16_t* weight,
                        const uint16_t* residual, uint16_t* residual_out,
                        uint16_t* y, const uint32_t* live_rows, float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float sums[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  constexpr int Group = Head ? Head : Width;
  constexpr int Groups = Width / Group;
  static_assert(Group % 8 == 0);
  __shared__ __align__(16) float ordered_squares[Kind == 3 ? Group : 1];
  for (int g = 0; g < Groups; ++g) {
    const uint64_t base = uint64_t(row) * Width + g * Group;
    float square = 0.f;
    for (int c = threadIdx.x * 8; c < Group; c += blockDim.x * 8) {
      uint4 packed = reinterpret_cast<const uint4*>(x + base + c)[0];
      const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
      uint4 rpacked;
      if constexpr (Residual)
        rpacked = reinterpret_cast<const uint4*>(residual + base + c)[0];
      const uint16_t* r = reinterpret_cast<const uint16_t*>(&rpacked);
      uint4 added;
      uint16_t* a = reinterpret_cast<uint16_t*>(&added);
      for (int j = 0; j < 8; ++j) {
        float v = bf(in[j]);
        if constexpr (Residual) {
          a[j] = to_bf(v + bf(r[j]));
          v = bf(a[j]);
        }
        if constexpr (Kind == 3)
          ordered_squares[c + j] = __fmul_rn(v, v);
        else
          square += v * v;
      }
      if constexpr (Residual)
        reinterpret_cast<uint4*>(residual_out + base + c)[0] = added;
    }
    float total;
    if constexpr (Kind == 3) {
      __syncthreads();
      if (threadIdx.x == 0) {
        float ordered = 0.f;
        for (int c = 0; c < Group; ++c) {
          ordered = __fadd_rn(ordered, ordered_squares[c]);
        }
        sums[0] = ordered;
      }
      __syncthreads();
      total = sums[0];
    } else {
      total = sum_cta(square, sums);
    }
    const float inv = Kind == 3 ? 1.f / sqrtf(total / Group + eps)
                                : rsqrtf(total / Group + eps);
    for (int c = threadIdx.x * 8; c < Group; c += blockDim.x * 8) {
      uint4 packed = reinterpret_cast<const uint4*>((Residual ? residual_out : x) + base + c)[0];
      uint4 wpacked = reinterpret_cast<const uint4*>(weight + g * Group + c)[0];
      const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
      const uint16_t* weights = reinterpret_cast<const uint16_t*>(&wpacked);
      uint4 result;
      uint16_t* out = reinterpret_cast<uint16_t*>(&result);
      for (int j = 0; j < 8; ++j) {
        float w = bf(weights[j]);
        if constexpr (Kind == 1) w += 1.f;
        float n = bf(in[j]) * inv;
        if constexpr (Kind == 3) n = bf(to_bf(n));
        out[j] = to_bf(n * w);
      }
      reinterpret_cast<uint4*>(y + base + c)[0] = result;
    }
    __syncthreads();
  }
}

template<int Width, bool Interleaved>
__device__ void swiglu(const uint16_t* input, uint16_t* output,
                       const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x; c < Width; c += blockDim.x) {
    const uint64_t base = uint64_t(row) * Width * 2;
    float gate = bf(input[base + (Interleaved ? 2 * c : c)]);
    float up = bf(input[base + (Interleaved ? 2 * c + 1 : Width + c)]);
    output[uint64_t(row) * Width + c] = to_bf(gate / (1.f + expf(-gate)) * up);
  }
}

template<int Width, bool Sharded>
__device__ void embedding(const uint16_t* table, const int32_t* ids,
                          uint16_t* output, const uint32_t* live_rows,
                          uint32_t vocab_offset, uint32_t shard_vocab) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  int64_t local = int64_t(ids[row]) - (Sharded ? vocab_offset : 0);
  bool valid = local >= 0 && uint64_t(local) < shard_vocab;
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    const uint64_t dst = uint64_t(row) * Width + c;
    if (c + 8 <= Width) {
      uint4 data = valid ? reinterpret_cast<const uint4*>(table + uint64_t(local) * Width + c)[0]
                         : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(output + dst)[0] = data;
    } else {
      for (int j = 0; j < 8 && c + j < Width; ++j)
        output[dst + j] = valid ? table[uint64_t(local) * Width + c + j] : 0;
    }
  }
}

template<int Width, int TopK, bool Preweighted>
__device__ void combine(const uint16_t* routed, const float* weights,
                        uint16_t* output, const uint32_t* live_rows) {
  begin();
  const uint32_t token = blockIdx.x;
  if (token >= *live_rows) return;
  static_assert(Width % 8 == 0);
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    float values[8] = {};
    #pragma unroll
    for (int k = 0; k < TopK; ++k) {
      uint4 packed = reinterpret_cast<const uint4*>(
          routed + (uint64_t(token) * TopK + k) * Width + c)[0];
      const uint16_t* parts = reinterpret_cast<const uint16_t*>(&packed);
      if constexpr (Preweighted) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) values[j] += bf(parts[j]);
      } else {
        const float weight = weights[token * TopK + k];
        #pragma unroll
        for (int j = 0; j < 8; ++j) values[j] += bf(parts[j]) * weight;
      }
    }
    uint4 result;
    uint16_t* out = reinterpret_cast<uint16_t*>(&result);
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = to_bf(values[j]);
    reinterpret_cast<uint4*>(output + uint64_t(token) * Width + c)[0] = result;
  }
}

} // namespace chop::row
