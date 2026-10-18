#pragma once

#include "../common/ptx.cuh"
#include <math.h>
#include <type_traits>

namespace chop::exact_f32 {

__device__ __forceinline__ float widen(uint16_t x) {
  return __uint_as_float(uint32_t(x) << 16);
}

// An intentionally conservative subset of f32_gemm::dot_path. Nonzero
// operands have exponents in [-30,30]. The last component has exponent at
// least -53, so every BF16 component product is normal, finite and exact
// in F32. The largest 128-product partial cannot overflow. Zero is exact.
__device__ __forceinline__ bool tensor_operand(float x) {
  const uint32_t bits = __float_as_uint(x) & 0x7fffffff;
  return bits == 0 || (bits >= (97u << 23) && bits < (158u << 23));
}

struct Components {
  __device__ __forceinline__ static void split(float x, uint16_t (&terms)[3]) {
    const uint16_t sign = (__float_as_uint(x) >> 16) & 0x8000;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      terms[c] = x == 0.f ? sign : uint16_t(__float_as_uint(x) >> 16);
      x = __fsub_rn(x, widen(terms[c]));
    }
  }
};

struct SeparateChains {
  ptx::D4 partial[3]{};
  __device__ __forceinline__ void step(const ptx::A4& a, const ptx::B2 (&b)[3]) {
#pragma unroll
    for (int c = 0; c < 3; ++c) ptx::mma_bf16(partial[c], a, b[c]);
  }
  __device__ __forceinline__ float merged(int v) const {
    return __fadd_rn(__fadd_rn(partial[0].v[v], partial[1].v[v]), partial[2].v[v]);
  }
};

// One warp owns 16 weight rows by eight token rows. Every 64-K stage is
// fully initialized, including tails. A row's path depends only on that
// row and its fixed weight tile, so batching cannot change its arithmetic.
// Split and Chains also let test-only entries exercise faulty arithmetic.
template<int N, int K, bool F32Weight, class Split = Components, class Chains = SeparateChains>
__device__ void run(const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* output, const uint32_t* live_rows, uint32_t rows) {
  ptx::griddep_wait();
  const uint32_t live = min(*live_rows, rows);
  const uint32_t token0 = blockIdx.y * 8;
  const uint32_t feature0 = blockIdx.x * 16;
  if (token0 >= live) {
    ptx::griddep_launch_dependents();
    return;
  }
  const uint32_t lane = threadIdx.x;
  __shared__ float x[8][64];
  __shared__ uint16_t w[16][64];
  __shared__ uint32_t eligible[8];
  __shared__ uint32_t weights_ok;
  bool valid_weights = true;
  for (uint64_t at = lane; at < uint64_t(16) * K; at += 32) {
    const uint32_t feature = feature0 + at / K;
    if (feature >= N) continue;
    const float value = F32Weight ? float(weight[uint64_t(feature) * K + at % K])
        : widen(uint16_t(weight[uint64_t(feature) * K + at % K]));
    valid_weights &= tensor_operand(value);
  }
  const bool weight_domain = __all_sync(0xffffffff, valid_weights);
  if (lane == 0) weights_ok = !F32Weight && weight_domain;
  for (int r = 0; r < 8; ++r) {
    bool valid = true;
    if (token0 + r < live) {
      for (int k = lane; k < K; k += 32) {
        const float value = activation[uint64_t(token0 + r) * K + k];
        valid &= tensor_operand(value);
      }
    }
    const bool domain = __all_sync(0xffffffff, valid);
    if (lane == 0) eligible[r] = domain;
  }
  __syncthreads();
  ptx::D4 total{};
  if constexpr (!F32Weight) {
    // Ineligible activation rows stage zeros and are computed below in
    // SIMT, so an exceptional row cannot contaminate an admitted row.
    if (weights_ok) {
#pragma unroll 1
      for (int base = 0; base < K; base += 128) {
        Chains chains{};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int k0 = base + half * 64;
          for (int at = lane; at < 8 * 64; at += 32) {
            const int r = at / 64;
            const int k = k0 + at % 64;
            x[r][at % 64] = token0 + r < live && eligible[r] && k < K
                ? activation[uint64_t(token0 + r) * K + k] : 0.f;
          }
          for (int at = lane; at < 16 * 64; at += 32) {
            const int f = at / 64;
            const int k = k0 + at % 64;
            w[f][at % 64] = feature0 + f < N && k < K
                ? weight[uint64_t(feature0 + f) * K + k] : 0;
          }
          __syncthreads();
#pragma unroll
          for (int step = 0; step < 4; ++step) {
            ptx::A4 a;
            ptx::B2 b[3];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
              const int f = (lane >> 2) + (h & 1) * 8;
              const int k = step * 16 + (lane & 3) * 2 + (h >> 1) * 8;
              a.v[h] = uint32_t(w[f][k]) | (uint32_t(w[f][k+1]) << 16);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int k = step * 16 + (lane & 3) * 2 + h * 8;
              uint16_t first[3], second[3];
              Split::split(x[lane >> 2][k], first);
              Split::split(x[lane >> 2][k+1], second);
#pragma unroll
              for (int c = 0; c < 3; ++c)
                b[c].v[h] = uint32_t(first[c]) | (uint32_t(second[c]) << 16);
            }
            chains.step(a,b);
          }
          __syncthreads();
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) total.v[v] = __fadd_rn(total.v[v], chains.merged(v));
      }
    }
  }
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint32_t r = (lane & 3) * 2 + (v & 1);
    const uint32_t feature = feature0 + (lane >> 2) + (v >> 1) * 8;
    const uint32_t token = token0 + r;
    if (token >= live || feature >= N) continue;
    float result = total.v[v];
    if (F32Weight || !weights_ok || !eligible[r]) {
      result = 0.f;
#pragma unroll 1
      for (int base = 0; base < K; base += 128) {
        float partial = 0.f;
#pragma unroll 1
        for (int k = base; k < min(K, base + 128); ++k) {
          const float a = activation[uint64_t(token) * K + k];
          const float b = F32Weight ? float(weight[uint64_t(feature) * K + k])
              : widen(uint16_t(weight[uint64_t(feature) * K + k]));
          partial = __fmaf_rn(a,b,partial);
        }
        result = __fadd_rn(result,partial);
      }
    }
    // Out-of-contract non-finite operands take SIMT and retain IEEE F32
    // propagation. A downstream status-reporting quantizer can flag them;
    // device data never traps or contaminates another activation row.
    output[uint64_t(token) * N + feature] = result;
  }
  ptx::griddep_launch_dependents();
}
}
