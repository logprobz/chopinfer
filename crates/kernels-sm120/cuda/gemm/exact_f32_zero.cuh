#pragma once
#include "exact_f32_blocked.cuh"

namespace chop::exact_f32::zero_prefill {
// Three independent row predicates: original tensor domain, exact widened
// BF16, and tensor-domain rows that still need the residual components.
template<int K>
struct Workspace {
  prefill::Workspace<K> original;
  uint32_t *single, *residual;
  __device__ Workspace(void* p, uint32_t rows): original(p,rows) {
    single = reinterpret_cast<uint32_t*>(original.components+uint64_t(rows)*3*K);
    residual = single+rows;
  }
};

template<int K, bool ForceCertificate = false>
__device__ void prepare(const float* activation, const uint32_t* live_rows,
    uint32_t rows, void* workspace) {
  static_assert(K > 0 && K%128 == 0);
  ptx::griddep_wait();
  Workspace<K> work(workspace,rows);
  const uint32_t row = blockIdx.x;
  const bool live = row < min(rows,*live_rows);
  bool valid = live, exact = true;
  if (live)
    for (int k = threadIdx.x; k < K; k += 128) {
      const float x = activation[uint64_t(row)*K+k];
      valid &= tensor_operand(x);
      exact &= (__float_as_uint(x)&65535u) == 0;
      work.original.components[uint64_t(row)*3*K+k] = __float_as_uint(x)>>16;
    }
  const int admitted = __syncthreads_and(valid);
  const int certificate = __syncthreads_and(exact) || ForceCertificate;
  if (admitted && !certificate)
    for (int k = threadIdx.x; k < K; k += 128) {
      uint16_t parts[3];
      Components::split(activation[uint64_t(row)*K+k],parts);
#pragma unroll
      for (int c = 0; c < 3; ++c)
        work.original.components[(uint64_t(row)*3+c)*K+k] = parts[c];
    }
  if (threadIdx.x == 0) {
    work.original.domain[row] = admitted;
    work.single[row] = admitted && certificate;
    work.residual[row] = admitted && !certificate;
  }
  ptx::griddep_launch_dependents();
}

template<int N, int K, int M>
__device__ __forceinline__ void stage(uint16_t* dst, int base,
    const uint16_t* w, const uint16_t* x, const uint32_t* single, uint32_t live) {
  for (int at = threadIdx.x; at < (32+M)*8; at += 128) {
    const int r = at/8, k = (at%8)*8;
    const int row = blockIdx.y*M+r-32, col = blockIdx.x*32+r;
    const bool valid = r < 32 ? col < N : row < live && single[row];
    const uint16_t* src = r < 32 ? w : x;
    if (valid) src = r < 32 ? w+uint64_t(col)*K+base+k
        : x+uint64_t(row)*3*K+base+k;
    copy16(dst+blocked::swizzle(r,k),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// One eight-instruction chain from +0 per 128 K. Each omitted residual
// chain consists entirely of signed-zero products accumulated from +0.
// Its result is +0; retain both ordered component merges explicitly.
template<int N, int K, int M, bool OmitFirst = false>
__device__ __forceinline__ void single(const uint16_t* weight, float* output,
    uint32_t live, const uint32_t* weight_domain, Workspace<K> work,
    uint16_t* storage) {
  constexpr int Tiles = M/16, Plane = (32+M)*64;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int wf = (warp%2)*16, wr = (warp/2)*8, col = blockIdx.x*32+wf;
  const bool active = col < N && weight_domain[col/16];
  ptx::D4 total[Tiles]{};
  stage<N,K,M>(storage,0,weight,work.original.components,work.single,live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    ptx::D4 partial[Tiles]{};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int start = base+half*64, slot = (start/64)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+64 < K)
        stage<N,K,M>(storage+(slot^1)*Plane,start+64,
            weight,work.original.components,work.single,live);
      const uint16_t* tile = storage+slot*Plane;
      if (active) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          ptx::A4 a;
#pragma unroll
          for (int h = 0; h < 4; ++h)
            a.v[h] = blocked::packed(tile,wf+(lane>>2)+(h&1)*8,
                step*16+(lane&3)*2+(h>>1)*8);
#pragma unroll
          for (int t = 0; t < Tiles; ++t) {
            ptx::B2 b;
#pragma unroll
            for (int h = 0; h < 2; ++h)
              b.v[h] = blocked::packed(tile,32+wr+t*16+(lane>>2),
                  step*16+(lane&3)*2+h*8);
            if (!OmitFirst || base != 0) ptx::mma_bf16(partial[t],a,b);
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < Tiles; ++t)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        float hi_mid, merged;
        asm volatile("add.rn.f32 %0, %1, 0f00000000;" : "=f"(hi_mid) : "f"(partial[t].v[v]));
        asm volatile("add.rn.f32 %0, %1, 0f00000000;" : "=f"(merged) : "f"(hi_mid));
        total[t].v[v] = __fadd_rn(total[t].v[v],merged);
      }
  }
#pragma unroll
  for (int t = 0; t < Tiles; ++t)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int row = blockIdx.y*M+wr+t*16+(lane&3)*2+(v&1);
      const int f = col+(lane>>2)+(v>>1)*8;
      if (active && row < live && f < N && work.single[row])
        output[uint64_t(row)*N+f] = total[t].v[v];
    }
  __syncthreads();
}

template<int N, int K, int M, bool OmitFirst = false>
__device__ void run(const float* activation, const uint16_t* weight,
    void* workspace, float* output, const uint32_t* live_rows,
    uint32_t rows, const uint32_t* weight_domain) {
  static_assert(N > 0 && K > 0 && K%128 == 0);
  static_assert(M == 32 || M == 64);
  ptx::griddep_wait();
  const uint32_t live = min(rows,*live_rows);
  if (blockIdx.y*M >= live) { ptx::griddep_launch_dependents(); return; }
  Workspace<K> work(workspace,rows);
  extern __shared__ __align__(16) uint8_t storage[];
  bool tensor_weight = false, simt_weight = false;
#pragma unroll
  for (int f = 0; f < 32; f += 16)
    if (blockIdx.x*32+f < N) {
      tensor_weight |= weight_domain[(blockIdx.x*32+f)/16];
      simt_weight |= !weight_domain[(blockIdx.x*32+f)/16];
    }
  bool fast = false, general = false, simt = simt_weight;
  for (int r = threadIdx.x; r < M; r += 128)
    if (blockIdx.y*M+r < live) {
      fast |= work.single[blockIdx.y*M+r];
      general |= work.residual[blockIdx.y*M+r];
      simt |= !work.original.domain[blockIdx.y*M+r];
    }
  fast = __syncthreads_or(fast);
  general = __syncthreads_or(general);
  simt = __syncthreads_or(simt);
  if (tensor_weight && fast)
    single<N,K,M,OmitFirst>(weight,output,live,weight_domain,work,
        reinterpret_cast<uint16_t*>(storage));
  if (tensor_weight && general) {
    auto residual = work.original;
    residual.domain = work.residual;
    blocked::tensor<N,K,M>(weight,output,live,weight_domain,residual,
        reinterpret_cast<uint16_t*>(storage));
  }
  if (simt)
    blocked::simt<N,K,M,false,false>(activation,weight,output,live,
        weight_domain,work.original.domain,reinterpret_cast<float*>(storage));
  ptx::griddep_launch_dependents();
}
}
