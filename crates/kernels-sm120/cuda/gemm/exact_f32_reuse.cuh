#pragma once
#include "exact_f32_decode.cuh"

namespace chop::exact_f32_reuse {
using namespace exact_f32;
struct Task { uint32_t x, y, z; };
struct alignas(16) Arena {
  float x[4][8][128];
  uint16_t w[4][16][128];
};

// Each warp still owns one fixed 128-K chain. Reusing its immutable
// weight tile across four eight-row groups changes no per-output arithmetic.
template<int N, int K, bool F32Weight>
__device__ void task(Task task, Arena& shared, const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* partial, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, const uint32_t* row_domain) {
  constexpr int Chains = (K + 127) / 128;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chain = task.z * 4 + warp, base = chain * 128;
  const int row_base = task.y * 32, feature0 = task.x * 16;
  const uint32_t live = min(*live_rows, rows);
  if (row_base >= live) return;
  auto& x = shared.x;
  auto& w = shared.w;
  const bool tensor = !F32Weight && weight_domain[task.x];
  if (tensor && chain < Chains) {
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
  for (int group = 0; group < 4; ++group) {
    const int token0 = row_base + group * 8;
    if (token0 >= live) break;
    if (tensor && chain < Chains) {
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
    __syncthreads();
  }
}
} // namespace chop::exact_f32_reuse
