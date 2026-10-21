#pragma once
#include "exact_f32_certified.cuh"

// Lab: the certified prefill projection (`exact_f32_certified.cuh`) with
// E-feature tiles, E a multiple of 32. A CTA stages E weight rows beside
// its M activation rows, so each activation row block is read N / E times
// instead of N / 32. Each warp runs the served warp's 16 features and rows
// for every 32-feature group of the tile, so each output takes the same
// single chain of the same eight instructions per 128 K, the same merges
// and the same fold as the served tile. The SIMT path is the served one per
// 32-feature group, with the group's first feature in place of the CTA's.
namespace chop::exact_f32::certified_wide {

template<int N, int K, int M, int E>
__device__ __forceinline__ void stage(uint16_t* dst, int base,
    const uint16_t* w, const uint16_t* x, const uint32_t* certified, uint32_t live) {
  for (int at = threadIdx.x; at < (E+M)*8; at += 128) {
    const int r = at/8, k = (at%8)*8;
    const int row = blockIdx.y*M+r-E, col = blockIdx.x*E+r;
    const bool valid = r < E ? col < N : row < live && certified[row];
    const uint16_t* src = r < E ? w : x;
    if (valid) src = r < E ? w+uint64_t(col)*K+base+k
        : x+uint64_t(row)*K+base+k;
    copy16(dst+blocked::swizzle(r,k),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N, int K, int M, int E>
__device__ __forceinline__ void single(const uint16_t* weight, const uint16_t* normed,
    const uint32_t* certified, float* output, uint32_t live,
    const uint32_t* weight_domain, uint16_t* storage) {
  constexpr int Tiles = M/16, Groups = E/32, Plane = (E+M)*64;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int wf = (warp%2)*16, wr = (warp/2)*8;
  int col[Groups];
  bool active[Groups];
#pragma unroll
  for (int g = 0; g < Groups; ++g) {
    col[g] = blockIdx.x*E+g*32+wf;
    active[g] = col[g] < N && weight_domain[col[g]/16];
  }
  ptx::D4 total[Groups][Tiles]{};
  stage<N,K,M,E>(storage,0,weight,normed,certified,live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    ptx::D4 partial[Groups][Tiles]{};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int start = base+half*64, slot = (start/64)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+64 < K)
        stage<N,K,M,E>(storage+(slot^1)*Plane,start+64,weight,normed,certified,live);
      const uint16_t* tile = storage+slot*Plane;
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        ptx::B2 b[Tiles];
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            b[t].v[h] = blocked::packed(tile,E+wr+t*16+(lane>>2),
                step*16+(lane&3)*2+h*8);
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
          if (!active[g]) continue;
          ptx::A4 a;
#pragma unroll
          for (int h = 0; h < 4; ++h)
            a.v[h] = blocked::packed(tile,g*32+wf+(lane>>2)+(h&1)*8,
                step*16+(lane&3)*2+(h>>1)*8);
#pragma unroll
          for (int t = 0; t < Tiles; ++t) ptx::mma_bf16(partial[g][t],a,b[t]);
        }
      }
    }
    // The served tile's two explicit merges, then its fold.
#pragma unroll
    for (int g = 0; g < Groups; ++g)
#pragma unroll
      for (int t = 0; t < Tiles; ++t)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          float hi_mid, merged;
          asm volatile("add.rn.f32 %0, %1, 0f00000000;" : "=f"(hi_mid) : "f"(partial[g][t].v[v]));
          asm volatile("add.rn.f32 %0, %1, 0f00000000;" : "=f"(merged) : "f"(hi_mid));
          total[g][t].v[v] = __fadd_rn(total[g][t].v[v],merged);
        }
  }
#pragma unroll
  for (int g = 0; g < Groups; ++g)
#pragma unroll
    for (int t = 0; t < Tiles; ++t)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = blockIdx.y*M+wr+t*16+(lane&3)*2+(v&1);
        const int f = col[g]+(lane>>2)+(v>>1)*8;
        if (active[g] && row < live && f < N && certified[row])
          output[uint64_t(row)*N+f] = total[g][t].v[v];
      }
  __syncthreads();
}

// `blocked::stage_simt` (BF16 weights) with the 32 features from `col0`.
template<int N, int K, int M>
__device__ __forceinline__ void stage_simt(float* dst, int base, int col0,
    const float* x, const uint16_t* w, uint32_t live) {
  for (int at = threadIdx.x; at < M*8; at += 128) {
    const int r = at/8, k = (at%8)*4;
    const bool valid = blockIdx.y*M+r < live;
    const float* src = valid ? x+uint64_t(blockIdx.y*M+r)*K+base+k : x;
    copy16(dst+r*36+k, src, valid);
  }
  constexpr int Width = 8;
  for (int at = threadIdx.x; at < 32*32/Width; at += 128) {
    const int f = at/(32/Width), k = (at%(32/Width))*Width;
    const int col = col0+f;
    uint4 value{};
    if (col < N) value = *reinterpret_cast<const uint4*>(w+uint64_t(col)*K+base+k);
    const uint32_t words[4] = {value.x,value.y,value.z,value.w};
#pragma unroll
    for (int j = 0; j < Width; ++j)
      dst[M*36+(k+j)*33+f] = widen(uint16_t(words[j/2] >> ((j%2)*16)));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// `blocked::simt<N,K,M,false,false>` over the 32 features from `col0`.
template<int N, int K, int M>
__device__ __forceinline__ void simt(int col0, const float* activation,
    const uint16_t* weight, float* output, uint32_t live, const uint32_t* weight_domain,
    const uint32_t* row_domain, float* storage) {
  constexpr int R = M/16, Plane = M*36+32*33;
  const int r0 = threadIdx.x/8, f0 = threadIdx.x%8;
  float total[R][4]{};
  stage_simt<N,K,M>(storage, 0, col0, activation, weight, live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    float partial[R][4]{};
#pragma unroll
    for (int panel = 0; panel < 4; ++panel) {
      const int start = base+panel*32, slot = (start/32)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+32 < K)
        stage_simt<N,K,M>(storage+(slot^1)*Plane, start+32, col0, activation, weight, live);
      const float* tile = storage+slot*Plane;
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        float a[R], b[4];
#pragma unroll
        for (int r = 0; r < R; ++r) a[r] = tile[(r0+r*16)*36+k];
#pragma unroll
        for (int f = 0; f < 4; ++f) b[f] = tile[M*36+k*33+f0+f*8];
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
      for (int f = 0; f < 4; ++f) total[r][f] = __fadd_rn(total[r][f],partial[r][f]);
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int f = 0; f < 4; ++f) {
      const int row = blockIdx.y*M+r0+r*16, col = col0+f0+f*8;
      if (row < live && col < N && (!weight_domain[col/16] || !row_domain[row]))
        output[uint64_t(row)*N+col] = total[r][f];
    }
  // The next group restages the shared panels.
  __syncthreads();
}

template<int N, int K, int M, int E>
__device__ void run(const float* activation, const uint16_t* normed,
    const uint32_t* certified, const uint16_t* weight, float* output,
    const uint32_t* live_rows, uint32_t rows, const uint32_t* weight_domain) {
  static_assert(N > 0 && K > 0 && K%128 == 0);
  static_assert(M == 32 || M == 64);
  static_assert(E >= 32 && E%32 == 0);
  ptx::griddep_wait();
  const uint32_t live = min(rows,*live_rows);
  if (blockIdx.y*M >= live) { ptx::griddep_launch_dependents(); return; }
  extern __shared__ __align__(16) uint8_t storage[];
  static_assert(E <= 1024, "one SIMT bit per 32-feature group");
  bool tensor_weight = false;
  uint32_t simt_weight = 0;  // bit g: group g has a SIMT feature
#pragma unroll
  for (int f = 0; f < E; f += 16)
    if (blockIdx.x*E+f < N) {
      tensor_weight |= weight_domain[(blockIdx.x*E+f)/16];
      if (!weight_domain[(blockIdx.x*E+f)/16]) simt_weight |= 1u << (f/32);
    }
  bool fast = false, simt_rows = false;
  for (int r = threadIdx.x; r < M; r += 128)
    if (blockIdx.y*M+r < live) {
      fast |= certified[blockIdx.y*M+r];
      simt_rows |= !certified[blockIdx.y*M+r];
    }
  fast = __syncthreads_or(fast);
  simt_rows = __syncthreads_or(simt_rows);
  if (tensor_weight && fast)
    single<N,K,M,E>(weight,normed,certified,output,live,weight_domain,
        reinterpret_cast<uint16_t*>(storage));
#pragma unroll 1
  for (int g = 0; g < E/32; ++g)
    if ((simt_rows || (simt_weight >> g & 1)) && blockIdx.x*E+g*32 < N)
      simt<N,K,M>(blockIdx.x*E+g*32,activation,weight,output,live,weight_domain,certified,
          reinterpret_cast<float*>(storage));
  ptx::griddep_launch_dependents();
}
}
