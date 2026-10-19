#pragma once
#include "exact_f32.cuh"

namespace chop::exact_f32 {

// One flag covers the same 16 output rows as the original weight scan.
template<int N, int K, bool F32Weight>
__device__ void classify_weights(
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    uint32_t* domain) {
  ptx::griddep_wait();
  const int tile = blockIdx.x;
  bool valid = !F32Weight;
  for (uint64_t at = threadIdx.x; at < uint64_t(16) * K; at += 128) {
    const int feature = tile * 16 + at / K;
    if (feature < N) {
      const float value = F32Weight ? float(weight[uint64_t(feature) * K + at % K])
          : widen(uint16_t(weight[uint64_t(feature) * K + at % K]));
      valid &= tensor_operand(value);
    }
  }
  const int accepted = __syncthreads_and(valid);
  if (threadIdx.x == 0) domain[tile] = accepted;
  ptx::griddep_launch_dependents();
}

template<int K>
__device__ void classify_rows(const float* activation, const uint32_t* live_rows,
    uint32_t rows, uint32_t* domain) {
  ptx::griddep_wait();
  const int row = blockIdx.x;
  bool valid = true;
  if (row < min(*live_rows, rows)) {
    for (int k = threadIdx.x; k < K; k += 128)
      valid &= tensor_operand(activation[uint64_t(row) * K + k]);
  }
  const int accepted = __syncthreads_and(valid);
  if (threadIdx.x == 0) domain[row] = accepted;
  ptx::griddep_launch_dependents();
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

// Four warps compute four consecutive 128-K chains independently. Each
// warp retains the original MMA lane mapping, component split, and eight
// k16 steps. No subtotal combines chains before the final ordered sum.
template<int N, int K, bool F32Weight>
__device__ void decode(const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* partial, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, const uint32_t* row_domain) {
  ptx::griddep_wait();
  constexpr int Chains = (K + 127) / 128;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chain = blockIdx.z * 4 + warp, base = chain * 128;
  const int token0 = blockIdx.y * 8, feature0 = blockIdx.x * 16;
  const uint32_t live = min(*live_rows, rows);
  if (token0 >= live) {
    ptx::griddep_launch_dependents();
    return;
  }
  __shared__ __align__(16) float x[4][8][128];
  __shared__ __align__(16) uint16_t w[4][16][128];
  const bool tensor = !F32Weight && weight_domain[blockIdx.x];
  if (tensor && chain < Chains) {
    // Whole vectors for aligned shapes; scalar tails also cover odd K
    // and arbitrary row alignment without out-of-bounds vector reads.
    for (int at = lane; at < 8 * 32; at += 32) {
      const int r = at / 32, k = (at % 32) * 4;
      const bool active = token0 + r < live && row_domain[token0 + r];
      if (K % 4 == 0 && (reinterpret_cast<uintptr_t>(activation) & 15) == 0) {
        const bool valid = active && base + k + 4 <= K;
        const float* src = valid ? activation + uint64_t(token0 + r) * K + base + k : activation;
        copy16(&x[warp][r][k], src, valid);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          x[warp][r][k+j] = active && base + k + j < K
              ? activation[uint64_t(token0+r)*K + base+k+j] : 0.f;
      }
    }
    for (int at = lane; at < 16 * 16; at += 32) {
      const int f = at / 16, k = (at % 16) * 8;
      if constexpr (!F32Weight) {
        if (K % 8 == 0 && (reinterpret_cast<uintptr_t>(weight) & 15) == 0) {
          const bool valid = feature0 + f < N && base + k + 8 <= K;
          const uint16_t* src = valid ? weight + uint64_t(feature0+f)*K + base+k : weight;
          copy16(&w[warp][f][k], src, valid);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            w[warp][f][k+j] = feature0 + f < N && base + k + j < K
                ? weight[uint64_t(feature0+f)*K + base+k+j] : 0;
        }
      }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
  }
  __syncthreads();
  if (chain < Chains) {
    SeparateChains accum{};
    if constexpr (!F32Weight) {
      if (tensor) {
#pragma unroll
        for (int step = 0; step < 8; ++step) {
          ptx::A4 a;
          ptx::B2 b[3];
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            const int f = (lane >> 2) + (h & 1) * 8;
            const int k = step * 16 + (lane & 3) * 2 + (h >> 1) * 8;
            a.v[h] = uint32_t(w[warp][f][k]) | (uint32_t(w[warp][f][k+1]) << 16);
          }
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int k = step * 16 + (lane & 3) * 2 + h * 8;
            uint16_t first[3], second[3];
            Components::split(x[warp][lane >> 2][k], first);
            Components::split(x[warp][lane >> 2][k+1], second);
#pragma unroll
            for (int c = 0; c < 3; ++c)
              b[c].v[h] = uint32_t(first[c]) | (uint32_t(second[c]) << 16);
          }
          accum.step(a, b);
        }
      }
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int token = token0 + (lane & 3) * 2 + (v & 1);
      const int feature = feature0 + (lane >> 2) + (v >> 1) * 8;
      if (token >= live || feature >= N) continue;
      float value = accum.merged(v);
      if (!tensor || !row_domain[token]) {
        value = 0.f;
#pragma unroll 1
        for (int k = base; k < min(K, base + 128); ++k) {
          const float a = activation[uint64_t(token)*K+k];
          const float b = F32Weight ? float(weight[uint64_t(feature)*K+k])
              : widen(uint16_t(weight[uint64_t(feature)*K+k]));
          value = __fmaf_rn(a, b, value);
        }
      }
      partial[(uint64_t(chain)*rows+token)*N+feature] = value;
    }
  }
  ptx::griddep_launch_dependents();
}

template<int N, int K>
__device__ void reduce(const float* partial, float* output,
    const uint32_t* live_rows, uint32_t rows) {
  ptx::griddep_wait();
  const uint64_t at = uint64_t(blockIdx.x) * 128 + threadIdx.x;
  if (at < uint64_t(min(*live_rows, rows)) * N) {
    float sum = 0.f;
#pragma unroll 1
    for (int chain = 0; chain < (K + 127) / 128; ++chain)
      sum = __fadd_rn(sum, partial[uint64_t(chain)*rows*N+at]);
    output[at] = sum;
  }
  ptx::griddep_launch_dependents();
}

// Cached prefill preserves the original tile and arithmetic. Keep the
// original entry and header unchanged for locked baseline comparisons.
template<int N, int K, bool F32Weight, class Split = Components, class Chains = SeparateChains>
__device__ void cached(const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* output, const uint32_t* live_rows, uint32_t rows, const uint32_t* weight_domain) {
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
  if (lane == 0) weights_ok = !F32Weight && weight_domain[blockIdx.x];
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
