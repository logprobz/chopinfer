#pragma once
#include "exact_f32_reuse.cuh"

namespace chop::exact_f32_last {
using namespace exact_f32;
using Task = exact_f32_reuse::Task;
struct alignas(16) Arena {
  exact_f32_reuse::Arena product;
  uint32_t last;
};

// The eight-row producer retains the original operand loads, component
// split, four independent chains, MMA steps and scalar fallback.
template<int N, int K, bool F32Weight>
__device__ void produce8(Task task, exact_f32_reuse::Arena& shared,
    const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* partial, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, const uint32_t* row_domain) {
  constexpr int Chains = (K + 127) / 128;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chain = task.z * 4 + warp, base = chain * 128;
  const int token0 = task.y * 8, feature0 = task.x * 16;
  const uint32_t live = min(*live_rows, rows);
  if (token0 >= live) {
    return;
  }
  auto& x = shared.x;
  auto& w = shared.w;
  const bool tensor = !F32Weight && weight_domain[task.x];
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
}

template<int N, int K> __device__ __forceinline__ uint64_t counter_offset(uint32_t rows) {
  return ((uint64_t(32) + uint64_t((K + 127) / 128) * rows * N) * 4 + 255) / 256 * 256;
}

template<int N, int K, int TileRows>
__device__ void prepare(uint32_t cta, uint32_t ctas, const float* activation,
    const uint32_t* live_rows, uint32_t rows, uint8_t* workspace) {
  auto* domain = reinterpret_cast<uint32_t*>(workspace);
  auto* counters = reinterpret_cast<uint32_t*>(workspace + counter_offset<N,K>(rows));
  const uint32_t tiles = ((N + 15) / 16) * ((rows + TileRows - 1) / TileRows);
  for (uint32_t i = cta * blockDim.x + threadIdx.x; i < tiles; i += ctas * blockDim.x)
    counters[i] = 0;
  bool valid = true;
  if (cta < min(*live_rows, rows)) {
    for (int k = threadIdx.x; k < K; k += 128)
      valid &= tensor_operand(activation[uint64_t(cta) * K + k]);
  }
  const int accepted = __syncthreads_and(valid);
  if (threadIdx.x == 0) domain[cta] = accepted;
}

// Only arrival order chooses the executor. It never chooses the addition
// order: every output starts at +0 and adds the same global chain planes.
// Prefetch changes load scheduling, not the order of F32 additions.
template<int N, int K, int TileRows, int Prefetch, bool Reverse>
__device__ void task(Task task, Arena& shared, const float* activation,
    const uint16_t* weight, uint8_t* workspace, float* output,
    const uint32_t* live_rows, uint32_t rows, const uint32_t* weight_domain) {
  static_assert(TileRows == 8 || TileRows == 32);
  static_assert(Prefetch == 1 || Prefetch == 16);
  constexpr int Chains = (K + 127) / 128;
  constexpr int Splits = (Chains + 3) / 4;
  constexpr int Tiles = (N + 15) / 16;
  const uint32_t live = min(*live_rows, rows);
  const uint32_t row_base = task.y * TileRows;
  if (row_base >= live) return;
  auto* partial = reinterpret_cast<float*>(workspace) + 32;
  const auto* row_domain = reinterpret_cast<const uint32_t*>(workspace);
  if constexpr (TileRows == 8)
    produce8<N,K,false>(task,shared.product,activation,weight,partial,live_rows,rows,weight_domain,row_domain);
  else
    exact_f32_reuse::task<N,K,false>(task,shared.product,activation,weight,partial,live_rows,rows,weight_domain,row_domain);
  auto* counters = reinterpret_cast<uint32_t*>(workspace + counter_offset<N,K>(rows));
  const uint32_t tile = task.y * Tiles + task.x;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(counters + tile) : "memory");
    shared.last = arrived == Splits - 1;
  }
  __syncthreads();
  if (!shared.last) return;
  for (uint32_t i = threadIdx.x; i < TileRows * 16; i += blockDim.x) {
    const uint32_t row = row_base + i / 16;
    const uint32_t col = task.x * 16 + i % 16;
    if (row >= live || col >= N) continue;
    const uint64_t at = uint64_t(row) * N + col;
    float sum = 0.f;
    for (int first = 0; first < Chains; first += Prefetch) {
      float values[Prefetch];
#pragma unroll
      for (int j = 0; j < Prefetch; ++j) if (first + j < Chains) {
        const int chain = Reverse ? Chains - 1 - first - j : first + j;
        values[j] = __ldcg(partial + uint64_t(chain) * rows * N + at);
      }
#pragma unroll
      for (int j = 0; j < Prefetch; ++j) if (first + j < Chains)
        sum = __fadd_rn(sum,values[j]);
    }
    output[at] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(counters + tile,0);
}
} // namespace chop::exact_f32_last
