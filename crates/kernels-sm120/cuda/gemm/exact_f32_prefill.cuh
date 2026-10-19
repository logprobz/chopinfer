#pragma once
#include "exact_f32_decode.cuh"

namespace chop::exact_f32::prefill {

template<int K>
struct Workspace {
  uint32_t* domain;
  uint16_t* components;
  __device__ Workspace(void* workspace, uint32_t rows) {
    domain = reinterpret_cast<uint32_t*>(workspace);
    components = reinterpret_cast<uint16_t*>(
        reinterpret_cast<uint8_t*>(workspace) + ((uint64_t(rows)*4+15) & ~uint64_t(15)));
  }
};

template<int K>
__device__ void prepare(const float* activation, const uint32_t* live_rows,
    uint32_t rows, void* workspace) {
  ptx::griddep_wait();
  Workspace<K> work(workspace, rows);
  const uint32_t row = blockIdx.x;
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
  ptx::griddep_launch_dependents();
}

// Four token warps per 16 features share two 128-K buffers.
// The 32/64-feature tiles use 64/80 KiB, respectively.
template<int TileN>
struct Tile {
  uint16_t (*w)[TileN][128];
  uint16_t (*x)[3][32][128];
  __device__ Tile(uint8_t* storage) {
    w = reinterpret_cast<uint16_t (*)[TileN][128]>(storage);
    x = reinterpret_cast<uint16_t (*)[3][32][128]>(storage+2*TileN*128*2);
  }
};

template<int N, int K, int TileN>
__device__ __forceinline__ void stage(Tile<TileN> tile, int buffer, int base, const uint16_t* weight,
    const uint16_t* components, const uint32_t* domain, uint32_t live) {
  const uint32_t token0 = blockIdx.y*32, feature0 = blockIdx.x*TileN;
  for (int at = threadIdx.x; at < TileN*16; at += TileN*8) {
    const int f = at/16, k = (at%16)*8;
    if (K%8 == 0 && (reinterpret_cast<uintptr_t>(weight)&15) == 0) {
      const bool valid = feature0+f < N && base+k+8 <= K;
      const uint16_t* src = valid ? weight+uint64_t(feature0+f)*K+base+k : weight;
      copy16(&tile.w[buffer][f][k], src, valid);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        tile.w[buffer][f][k+j] = feature0+f < N && base+k+j < K
            ? weight[uint64_t(feature0+f)*K+base+k+j] : 0;
    }
  }
  for (int at = threadIdx.x; at < 3*32*16; at += TileN*8) {
    const int c = at/(32*16), r = (at/16)%32, k = (at%16)*8;
    const bool active = token0+r < live && domain[token0+r];
    // Backend workspace starts on a 256-byte boundary. With K divisible
    // by eight every component row and vector retains 16-byte alignment.
    if (K%8 == 0) {
      const bool valid = active && base+k+8 <= K;
      const uint16_t* src = valid ? components+(uint64_t(token0+r)*3+c)*K+base+k : components;
      copy16(&tile.x[buffer][c][r][k], src, valid);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        tile.x[buffer][c][r][k+j] = active && base+k+j < K
            ? components[(uint64_t(token0+r)*3+c)*K+base+k+j] : 0;
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N, int K, bool F32Weight, int TileN>
__device__ void run(const float* activation,
    const typename std::conditional<F32Weight, float, uint16_t>::type* weight,
    float* output, const uint32_t* live_rows, uint32_t rows,
    const uint32_t* weight_domain, void* workspace) {
  ptx::griddep_wait();
  const uint32_t live = min(*live_rows, rows);
  const uint32_t token0 = blockIdx.y*32, feature0 = blockIdx.x*TileN;
  if (token0 >= live) {
    ptx::griddep_launch_dependents();
    return;
  }
  Workspace<K> work(workspace, rows);
  extern __shared__ __align__(16) uint8_t storage[];
  static_assert(TileN == 32 || TileN == 64);
  Tile<TileN> tile(storage);
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int wf = (warp%(TileN/16))*16, wr = (warp/(TileN/16))*8;
  const bool tensor = !F32Weight && feature0+wf < N && weight_domain[(feature0+wf)/16];
  ptx::D4 total{};
  if constexpr (!F32Weight) {
    bool any_tensor = false;
#pragma unroll
    for (int f = 0; f < TileN; f += 16)
      any_tensor |= feature0+f < N && weight_domain[(feature0+f)/16];
    if (any_tensor) {
      stage<N,K,TileN>(tile, 0, 0, weight, work.components, work.domain, live);
#pragma unroll 1
      for (int base = 0; base < K; base += 128) {
        const int buffer = (base/128)&1;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        // All current loads are visible. The same barrier finishes every
        // warp's reads of the previous tile before that buffer is reused.
        __syncthreads();
        if (base+128 < K)
          stage<N,K,TileN>(tile, buffer^1, base+128, weight, work.components, work.domain, live);
        if (tensor) {
          SeparateChains chains{};
#pragma unroll
          for (int step = 0; step < 8; ++step) {
            ptx::A4 a;
            ptx::B2 b[3];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
              const int f = wf+(lane>>2)+(h&1)*8;
              const int k = step*16+(lane&3)*2+(h>>1)*8;
              a.v[h] = uint32_t(tile.w[buffer][f][k]) | (uint32_t(tile.w[buffer][f][k+1])<<16);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int k = step*16+(lane&3)*2+h*8;
#pragma unroll
              for (int c = 0; c < 3; ++c)
                b[c].v[h] = uint32_t(tile.x[buffer][c][wr+(lane>>2)][k])
                    | (uint32_t(tile.x[buffer][c][wr+(lane>>2)][k+1])<<16);
            }
            chains.step(a,b);
          }
#pragma unroll
          for (int v = 0; v < 4; ++v) total.v[v] = __fadd_rn(total.v[v], chains.merged(v));
        }
      }
    }
  }
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint32_t token = token0+wr+(lane&3)*2+(v&1);
    const uint32_t feature = feature0+wf+(lane>>2)+(v>>1)*8;
    if (token >= live || feature >= N) continue;
    float result = total.v[v];
    if (!tensor || !work.domain[token]) {
      result = 0.f;
#pragma unroll 1
      for (int base = 0; base < K; base += 128) {
        float partial = 0.f;
#pragma unroll 1
        for (int k = base; k < min(K, base+128); ++k) {
          const float a = activation[uint64_t(token)*K+k];
          const float b = F32Weight ? float(weight[uint64_t(feature)*K+k])
              : widen(uint16_t(weight[uint64_t(feature)*K+k]));
          partial = __fmaf_rn(a,b,partial);
        }
        result = __fadd_rn(result,partial);
      }
    }
    output[uint64_t(token)*N+feature] = result;
  }
  ptx::griddep_launch_dependents();
}
}
