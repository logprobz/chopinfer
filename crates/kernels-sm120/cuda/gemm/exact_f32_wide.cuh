#pragma once
#include "exact_f32_prefill.cuh"

namespace chop::exact_f32::wide {
// Wider feature tiles share the same exact components across more warps.
// Rotate each 16-byte vector across eight banks of vectors. Whole-vector
// asynchronous copies and packed fragment reads use the same permutation.
__device__ __forceinline__ int swizzle(int row, int k) {
  return row * 64 + (k ^ ((row & 7) * 8));
}
__device__ __forceinline__ uint32_t packed(const uint16_t* p, int row, int k) {
  return *reinterpret_cast<const uint32_t*>(p + swizzle(row, k));
}

template<int N, int K, int M, int TileN>
__device__ __forceinline__ void stage_tensor(uint16_t* dst, int base,
    const uint16_t* w, const uint16_t* x, const uint32_t* domain, uint32_t live) {
  for (int at = threadIdx.x; at < (TileN + 3*M)*8; at += TileN*4) {
    const int row = at/8, k = (at%8)*8;
    const int token = blockIdx.y*M + (row-TileN)%M;
    const int col = blockIdx.x*TileN + row;
    const bool valid = base+k+8 <= K && (row < TileN ? col < N
        : token < live && domain[token]);
    const uint16_t* src = row < TileN ? w : x;
    if (valid) src = row < TileN ? w + uint64_t(col)*K + base+k
        : x + (uint64_t(token)*3+(row-TileN)/M)*K + base+k;
    copy16(dst + swizzle(row,k), src, valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N, int K, int M, int TileN>
__device__ __forceinline__ void tensor(const uint16_t* weight, float* output,
    uint32_t live, const uint32_t* weight_domain, prefill::Workspace<K> work,
    uint16_t* storage) {
  constexpr int Tiles = M/16, Plane = (TileN+3*M)*64;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int wf = (warp%(TileN/16))*16, wr = (warp/(TileN/16))*8;
  const int col = blockIdx.x*TileN + wf;
  const bool active = col < N && weight_domain[col/16];
  ptx::D4 total[Tiles]{};
  stage_tensor<N,K,M,TileN>(storage, 0, weight, work.components, work.domain, live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    SeparateChains chains[Tiles]{};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int start = base+half*64, slot = (start/64)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+64 < K)
        stage_tensor<N,K,M,TileN>(storage+(slot^1)*Plane, start+64,
            weight, work.components, work.domain, live);
      const uint16_t* tile = storage+slot*Plane;
      if (active) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          ptx::A4 a;
#pragma unroll
          for (int h = 0; h < 4; ++h)
            a.v[h] = packed(tile, wf+(lane>>2)+(h&1)*8,
                step*16+(lane&3)*2+(h>>1)*8);
#pragma unroll
          for (int t = 0; t < Tiles; ++t) {
            ptx::B2 b[3];
#pragma unroll
            for (int c = 0; c < 3; ++c)
#pragma unroll
              for (int h = 0; h < 2; ++h)
                b[c].v[h] = packed(tile, TileN+c*M+wr+t*16+(lane>>2),
                    step*16+(lane&3)*2+h*8);
            chains[t].step(a,b);
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < Tiles; ++t)
#pragma unroll
      for (int v = 0; v < 4; ++v)
        total[t].v[v] = __fadd_rn(total[t].v[v], chains[t].merged(v));
  }
#pragma unroll
  for (int t = 0; t < Tiles; ++t)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int row = blockIdx.y*M+wr+t*16+(lane&3)*2+(v&1);
      const int f = col+(lane>>2)+(v>>1)*8;
      if (active && row < live && f < N && work.domain[row])
        output[uint64_t(row)*N+f] = total[t].v[v];
    }
  __syncthreads();
}

// Each thread owns M/16 by four outputs. K is staged in 32-wide panels;
// the partial register survives all four panels of one 128-K chain.
template<int N, int K, int M, int TileN, bool Raw>
__device__ __forceinline__ void stage_simt(float* dst, int base,
    const float* x, const typename std::conditional<Raw,float,uint16_t>::type* w,
    uint32_t live) {
  for (int at = threadIdx.x; at < M*8; at += TileN*4) {
    const int r = at/8, k = (at%8)*4;
    const bool valid = blockIdx.y*M+r < live;
    const float* src = valid ? x+uint64_t(blockIdx.y*M+r)*K+base+k : x;
    copy16(dst+r*36+k, src, valid);
  }
  // Read whole 16-byte vectors, then transpose into padded K-major rows.
  // Padding also separates the banks used by the vector's scalar stores.
  constexpr int Width = Raw ? 4 : 8;
  for (int at = threadIdx.x; at < TileN*32/Width; at += TileN*4) {
    const int f = at/(32/Width), k = (at%(32/Width))*Width;
    const int col = blockIdx.x*TileN+f;
    uint4 value{};
    if (col < N) value = *reinterpret_cast<const uint4*>(w+uint64_t(col)*K+base+k);
    const uint32_t words[4] = {value.x,value.y,value.z,value.w};
#pragma unroll
    for (int j = 0; j < Width; ++j) {
      if constexpr (Raw) dst[M*36+(k+j)*(TileN+1)+f] = __uint_as_float(words[j]);
      else dst[M*36+(k+j)*(TileN+1)+f] = widen(uint16_t(words[j/2] >> ((j%2)*16)));
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N, int K, int M, int TileN, bool Raw, bool SingleChain>
__device__ __forceinline__ void simt(const float* activation,
    const typename std::conditional<Raw,float,uint16_t>::type* weight,
    float* output, uint32_t live, const uint32_t* weight_domain,
    const uint32_t* row_domain, float* storage, bool force_simt) {
  constexpr int R = M/16, Plane = M*36+32*(TileN+1);
  const int r0 = threadIdx.x/(TileN/4), f0 = threadIdx.x%(TileN/4);
  float total[R][4]{};
  stage_simt<N,K,M,TileN,Raw>(storage, 0, activation, weight, live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    float partial[R][4]{};
    if constexpr (SingleChain)
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int f = 0; f < 4; ++f) partial[r][f] = total[r][f];
#pragma unroll
    for (int panel = 0; panel < 4; ++panel) {
      const int start = base+panel*32, slot = (start/32)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+32 < K)
        stage_simt<N,K,M,TileN,Raw>(storage+(slot^1)*Plane, start+32, activation, weight, live);
      const float* tile = storage+slot*Plane;
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        float a[R], b[4];
#pragma unroll
        for (int r = 0; r < R; ++r) a[r] = tile[(r0+r*16)*36+k];
#pragma unroll
        for (int f = 0; f < 4; ++f) b[f] = tile[M*36+k*(TileN+1)+f0+f*(TileN/4)];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int f = 0; f < 4; ++f)
            partial[r][f] = __fmaf_rn(a[r],b[f],partial[r][f]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        if constexpr (SingleChain) total[r][f] = partial[r][f];
        else total[r][f] = __fadd_rn(total[r][f],partial[r][f]);
      }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int f = 0; f < 4; ++f) {
      const int row = blockIdx.y*M+r0+r*16, col = blockIdx.x*TileN+f0+f*(TileN/4);
      if (row < live && col < N && (Raw || force_simt || !weight_domain[col/16] || !row_domain[row]))
        output[uint64_t(row)*N+col] = total[r][f];
    }
}

template<int N, int K, int M, int TileN, bool Raw, bool SingleChain = false, bool NeighborFallback = false>
__device__ void run(const float* activation,
    const typename std::conditional<Raw,float,uint16_t>::type* weight,
    void* workspace, float* output, const uint32_t* live_rows,
    uint32_t rows, const uint32_t* weight_domain) {
  static_assert(N > 0 && K > 0 && K%128 == 0);
  static_assert(M == 32 && (TileN == 64 || TileN == 128));
  if (blockDim.x != TileN*4) __trap();
  ptx::griddep_wait();
  const uint32_t live = min(rows,*live_rows);
  if (blockIdx.y*M >= live) { ptx::griddep_launch_dependents(); return; }
  prefill::Workspace<K> work(workspace,rows);
  extern __shared__ __align__(16) uint8_t storage[];
  if constexpr (!Raw) {
    bool any_tensor = false, any_simt = false;
#pragma unroll
    for (int f = 0; f < TileN; f += 16)
      if (blockIdx.x*TileN+f < N) {
        any_tensor |= weight_domain[(blockIdx.x*TileN+f)/16];
        any_simt |= !weight_domain[(blockIdx.x*TileN+f)/16];
      }
    for (int r = threadIdx.x; r < M; r += TileN*4)
      if (blockIdx.y*M+r < live) any_simt |= !work.domain[blockIdx.y*M+r];
    any_simt = __syncthreads_or(any_simt);
    if (any_tensor) tensor<N,K,M,TileN>(weight,output,live,weight_domain,work,
        reinterpret_cast<uint16_t*>(storage));
    if (any_simt) simt<N,K,M,TileN,Raw,SingleChain>(activation,weight,output,live,
        weight_domain,work.domain,reinterpret_cast<float*>(storage),NeighborFallback);
  } else {
    simt<N,K,M,TileN,Raw,SingleChain>(activation,weight,output,live,
        weight_domain,work.domain,reinterpret_cast<float*>(storage),NeighborFallback);
  }
  ptx::griddep_launch_dependents();
}
}
