#pragma once
#include "exact_f32_decode.cuh"

namespace chop::exact_f32::split_v2 {

// Scratch is reset on every invocation, including empty-live graphs.
// No counter or component depends on an earlier capture or replay.
template<int N, int K>
struct Workspace {
  uint32_t* domain;
  uint32_t* tickets;
  uint16_t* components;
  float* partial;
  __device__ Workspace(void* workspace, uint32_t rows) {
    auto* bytes = reinterpret_cast<uint8_t*>(workspace);
    const uint64_t tiles = uint64_t((N+15)/16) * ((rows+7)/8);
    const uint64_t header = (4*(rows+tiles)+15) & ~uint64_t(15);
    const uint64_t expansion = (uint64_t(rows)*K*6+15) & ~uint64_t(15);
    domain = reinterpret_cast<uint32_t*>(bytes);
    tickets = domain + rows;
    components = reinterpret_cast<uint16_t*>(bytes+header);
    partial = reinterpret_cast<float*>(bytes+header+expansion);
  }
};

template<int N, int K>
__device__ void prepare(const float* activation, const uint32_t* live_rows,
    uint32_t rows, void* workspace) {
  ptx::griddep_wait();
  Workspace<N,K> work(workspace, rows);
  const uint64_t at = uint64_t(blockIdx.x)*128+threadIdx.x;
  const uint64_t tiles = uint64_t((N+15)/16)*((rows+7)/8);
  if (at < tiles) work.tickets[at] = 0;
  const uint32_t row = blockIdx.x;
  if (row < rows) {
    bool valid = true;
    if (row < min(*live_rows, rows)) {
      for (int k = threadIdx.x; k < K; k += 128) {
        const float value = activation[uint64_t(row)*K+k];
        valid &= tensor_operand(value);
        uint16_t parts[3];
        Components::split(value, parts);
#pragma unroll
        for (int c = 0; c < 3; ++c)
          work.components[(uint64_t(row)*3+c)*K+k] = parts[c];
      }
    }
    const int accepted = __syncthreads_and(valid);
    if (threadIdx.x == 0) work.domain[row] = accepted;
  }
  ptx::griddep_launch_dependents();
}

// Four warps compute four consecutive 128-K chains independently. Each
// warp retains the original MMA lane mapping, component split, and eight
// k16 steps. No subtotal combines chains before the final ordered sum.
template<int N, int K, bool F32Weight>
__device__ void compute(const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* partial, float* output, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, const uint32_t* row_domain,
    const uint16_t* components, uint32_t* tickets) {
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
  __shared__ __align__(16) uint16_t x[4][3][8][128];
  __shared__ __align__(16) uint16_t w[4][16][128];
  const bool tensor = !F32Weight && weight_domain[blockIdx.x];
  if (tensor && chain < Chains) {
    // Whole vectors for aligned shapes; scalar tails also cover odd K
    // and arbitrary row alignment without out-of-bounds vector reads.
    for (int at = lane; at < 3 * 8 * 16; at += 32) {
      const int c = at / (8 * 16), r = (at / 16) % 8, k = (at % 16) * 8;
      const bool active = token0 + r < live && row_domain[token0 + r];
      if (K % 8 == 0) {
        const bool valid = active && base + k + 8 <= K;
        const uint16_t* src = valid
            ? components + (uint64_t(token0+r)*3+c)*K + base+k : components;
        copy16(&x[warp][c][r][k], src, valid);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          x[warp][c][r][k+j] = active && base+k+j < K
              ? components[(uint64_t(token0+r)*3+c)*K+base+k+j] : 0;
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
#pragma unroll
            for (int c = 0; c < 3; ++c)
              b[c].v[h] = uint32_t(x[warp][c][lane>>2][k])
                  | (uint32_t(x[warp][c][lane>>2][k+1]) << 16);
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
  // The barrier publishes every producer thread's stores before the
  // release ticket. The final acquire observes all earlier releases;
  // a second barrier transfers that visibility to its reducer threads.
  __shared__ uint32_t last;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    uint32_t* counter = tickets + uint64_t(blockIdx.y) * gridDim.x + blockIdx.x;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(counter) : "memory");
    last = arrived == gridDim.z - 1;
  }
  __syncthreads();
  if (last) {
    const int token = token0 + threadIdx.x / 16;
    const int feature = feature0 + threadIdx.x % 16;
    if (token < live && feature < N) {
      // Bypass non-coherent L1 for values written by other CTAs. Load
      // independently, then add exactly one chain at a time in K order.
      float values[Chains];
#pragma unroll
      for (int c = 0; c < Chains; ++c)
        values[c] = __ldcg(partial + (uint64_t(c)*rows+token)*N+feature);
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < Chains; ++c) sum = __fadd_rn(sum, values[c]);
      output[uint64_t(token)*N+feature] = sum;
    }
  }
  ptx::griddep_launch_dependents();
}

}
