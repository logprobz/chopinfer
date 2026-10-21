#pragma once
#include "exact_f32_zero.cuh"

// The certified prefill projection of rows the producing norm certified
// (`row/rmsnorm_prefill.cuh` with `Certify`): each row is the norm's BF16
// output, and its F32 activation is that BF16 widened, so every row is
// exact and `zero_prefill::prepare` would copy the BF16 bits to component
// plane 0 and set `single` exactly where every element is a tensor operand.
// Here the norm writes that flag (`certified`) and the tiles stage the BF16
// rows directly, so the prepare launch and its copy go away. Each output
// takes the same single chain, merges and fold as `zero_prefill::single`;
// uncertified rows take the same SIMT path from the F32 activation. No row
// needs the residual components.
namespace chop::exact_f32::certified {

template<int N, int K, int M>
__device__ __forceinline__ void stage(uint16_t* dst, int base,
    const uint16_t* w, const uint16_t* x, const uint32_t* certified, uint32_t live) {
  for (int at = threadIdx.x; at < (32+M)*8; at += 128) {
    const int r = at/8, k = (at%8)*8;
    const int row = blockIdx.y*M+r-32, col = blockIdx.x*32+r;
    const bool valid = r < 32 ? col < N : row < live && certified[row];
    const uint16_t* src = r < 32 ? w : x;
    if (valid) src = r < 32 ? w+uint64_t(col)*K+base+k
        : x+uint64_t(row)*K+base+k;
    copy16(dst+blocked::swizzle(r,k),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N, int K, int M>
__device__ __forceinline__ void single(const uint16_t* weight, const uint16_t* normed,
    const uint32_t* certified, float* output, uint32_t live,
    const uint32_t* weight_domain, uint16_t* storage) {
  constexpr int Tiles = M/16, Plane = (32+M)*64;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int wf = (warp%2)*16, wr = (warp/2)*8, col = blockIdx.x*32+wf;
  const bool active = col < N && weight_domain[col/16];
  ptx::D4 total[Tiles]{};
  stage<N,K,M>(storage,0,weight,normed,certified,live);
#pragma unroll 1
  for (int base = 0; base < K; base += 128) {
    ptx::D4 partial[Tiles]{};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int start = base+half*64, slot = (start/64)&1;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (start+64 < K)
        stage<N,K,M>(storage+(slot^1)*Plane,start+64,weight,normed,certified,live);
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
            ptx::mma_bf16(partial[t],a,b);
          }
        }
      }
    }
    // The same two explicit merges with the +0 results of the omitted
    // residual chains as `zero_prefill::single`.
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
      if (active && row < live && f < N && certified[row])
        output[uint64_t(row)*N+f] = total[t].v[v];
    }
  __syncthreads();
}

// `activation` is the norm's widened F32 rows (the SIMT path's input),
// `normed` its BF16 rows and `certified` its per-row flags.
template<int N, int K, int M>
__device__ void run(const float* activation, const uint16_t* normed,
    const uint32_t* certified, const uint16_t* weight, float* output,
    const uint32_t* live_rows, uint32_t rows, const uint32_t* weight_domain) {
  static_assert(N > 0 && K > 0 && K%128 == 0);
  static_assert(M == 32 || M == 64);
  ptx::griddep_wait();
  const uint32_t live = min(rows,*live_rows);
  if (blockIdx.y*M >= live) { ptx::griddep_launch_dependents(); return; }
  extern __shared__ __align__(16) uint8_t storage[];
  bool tensor_weight = false, simt_weight = false;
#pragma unroll
  for (int f = 0; f < 32; f += 16)
    if (blockIdx.x*32+f < N) {
      tensor_weight |= weight_domain[(blockIdx.x*32+f)/16];
      simt_weight |= !weight_domain[(blockIdx.x*32+f)/16];
    }
  bool fast = false, simt = simt_weight;
  for (int r = threadIdx.x; r < M; r += 128)
    if (blockIdx.y*M+r < live) {
      fast |= certified[blockIdx.y*M+r];
      simt |= !certified[blockIdx.y*M+r];
    }
  fast = __syncthreads_or(fast);
  simt = __syncthreads_or(simt);
  if (tensor_weight && fast)
    single<N,K,M>(weight,normed,certified,output,live,weight_domain,
        reinterpret_cast<uint16_t*>(storage));
  if (simt)
    blocked::simt<N,K,M,false,false>(activation,weight,output,live,
        weight_domain,certified,reinterpret_cast<float*>(storage));
  ptx::griddep_launch_dependents();
}
}
