#pragma once

// exact_f32::decode for a consumer that would otherwise run classify_rows,
// and reset its arrival ticket, in a launch before it. The grid regroups
// the 128-K chains into Ctas CTAs of Warps warps along z, one thread-block
// cluster of Ctas CTAs per token block and feature tile, so a cluster
// covers every K of its eight rows. Each warp still owns one chain and
// computes it as decode does: the same staging, lane mapping, component
// split and SIMT fallback, into the same partial slot.
//
// The row domain is classify_rows': a live row is in it iff every one of
// its K values is a tensor operand. A tensor tile stages every live row,
// each warp checks its own staged chain, and the CTA and then the cluster
// AND the flags. Rows outside the domain are then zeroed in shared memory
// before any MMA reads them, which leaves exactly the zeros decode stages
// for them, and they take the SIMT path. A tile without tensor weights
// reads no row domain, as in decode, so it classifies nothing; that choice
// and the early exit of a token block past the live rows are uniform over
// a cluster, which shares its tile and token block. Block (0,0,0) resets
// the ticket after the dependency wait, before any exit, as prepare does.
#include "exact_f32_decode.cuh"
#include "../common/ptx_control.cuh"

namespace chop::exact_f32 {

template<int N, int K, int Warps>
__device__ void decode_classified(const float* activation, const uint16_t* weight,
    float* partial, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, uint32_t* arrived) {
  constexpr int Chains = (K + 127) / 128;
  constexpr int Ctas = (Chains + Warps - 1) / Warps;
  static_assert(Ctas <= 8 && K % 4 == 0);
  ptx::griddep_wait();
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) *arrived = 0;
  if (gridDim.z != Ctas || blockDim.x != Warps * 32 || ptx::cluster_nctarank() != Ctas) __trap();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chain = blockIdx.z * Warps + warp, base = chain * 128;
  const int token0 = blockIdx.y * 8, feature0 = blockIdx.x * 16;
  const uint32_t live = min(*live_rows, rows);
  if (token0 >= live) {
    ptx::griddep_launch_dependents();
    return;
  }
  __shared__ __align__(16) float x[Warps][8][128];
  __shared__ __align__(16) uint16_t w[Warps][16][128];
  __shared__ uint32_t local[8], domain[8];
  const bool tensor = weight_domain[blockIdx.x];
  if (tensor) {
    if (threadIdx.x < 8) local[threadIdx.x] = 1;
    if (chain < Chains) {
      for (int at = lane; at < 8 * 32; at += 32) {
        const int r = at / 32, k = (at % 32) * 4;
        const bool active = token0 + r < live;
        if ((reinterpret_cast<uintptr_t>(activation) & 15) == 0) {
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
      asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if (chain < Chains) {
      for (int r = 0; r < 8; ++r) {
        bool valid = true;
        if (token0 + r < live)
          for (int k = lane; k < 128; k += 32) valid &= tensor_operand(x[warp][r][k]);
        if (!__all_sync(0xffffffffu, valid) && lane == 0) atomicAnd(&local[r], 0u);
      }
    }
    // Every CTA's flags are complete before any is read across the cluster.
    ptx::cluster_arrive_release();
    ptx::cluster_wait_acquire();
    if (threadIdx.x < 8) {
      uint32_t all = 1;
      for (uint32_t rank = 0; rank < Ctas; ++rank) {
        uint32_t flag;
        const uint32_t remote = ptx::cluster_map_shared(&local[threadIdx.x], rank);
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(flag) : "r"(remote) : "memory");
        all &= flag;
      }
      domain[threadIdx.x] = all;
    }
    // This CTA has used every peer flag it loaded; each peer waits for
    // this before it exits. Nothing it wrote is published here.
    ptx::cluster_arrive_relaxed();
    __syncthreads();
    if (chain < Chains)
      for (int at = lane; at < 8 * 128; at += 32)
        if (!domain[at / 128]) x[warp][at / 128][at % 128] = 0.f;
  }
  __syncthreads();
  if (chain < Chains) {
    SeparateChains accum{};
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
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int r = (lane & 3) * 2 + (v & 1), token = token0 + r;
      const int feature = feature0 + (lane >> 2) + (v >> 1) * 8;
      if (token >= live || feature >= N) continue;
      float value = accum.merged(v);
      if (!tensor || !domain[r]) {
        value = 0.f;
#pragma unroll 1
        for (int k = base; k < min(K, base + 128); ++k)
          value = __fmaf_rn(activation[uint64_t(token)*K+k],
              widen(weight[uint64_t(feature)*K+k]), value);
      }
      partial[(uint64_t(chain)*rows+token)*N+feature] = value;
    }
  }
  if (tensor) ptx::cluster_wait_acquire();
  ptx::griddep_launch_dependents();
}

} // namespace chop::exact_f32
