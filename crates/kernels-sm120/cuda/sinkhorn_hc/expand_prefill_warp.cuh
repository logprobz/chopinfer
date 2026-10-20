#pragma once
#include "sinkhorn_hc.cuh"

namespace chop::sinkhorn_hc {

// Four columns per thread, with either one row per CTA or one row per warp.
// Coefficients may be loaded by their owner lane and broadcast. Arithmetic
// remains the ascending stream FMA chain, separate post*y add, then BF16.
template<int Streams, int Width, int Threads, bool RowWarps, bool Broadcast>
__device__ void expand_prefill_warp(const uint16_t* x, const uint16_t* y,
    const float* post, const float* comb, uint16_t* out, const uint32_t* live_rows) {
  constexpr int N=Streams, H=Width;
  static_assert(N>=2 && N<=8 && H%4==0);
  static_assert(Threads==128 || Threads==256);
  static_assert(!Broadcast || N*(N+1)<=32);
  const int t=threadIdx.x, lane=t&31;
  if (blockDim.x!=Threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row=RowWarps ? blockIdx.y*(Threads/32)+(t>>5) : blockIdx.y;
  if (row>=*live_rows) return;
  const int col=blockIdx.x*(RowWarps ? 128 : Threads*4)+(RowWarps ? lane : t)*4;
  const uint64_t at=uint64_t(row)*N*H+col;
  float values[N][4];
  #pragma unroll
  for (int i=0;i<N;++i) {
    uint2 packed=make_uint2(0,0);
    if (col<H) packed=*reinterpret_cast<const uint2*>(x+at+uint64_t(i)*H);
    values[i][0]=bf_lo(packed.x);values[i][1]=bf_hi(packed.x);
    values[i][2]=bf_lo(packed.y);values[i][3]=bf_hi(packed.y);
  }
  uint2 yp=make_uint2(0,0);
  if (col<H) yp=*reinterpret_cast<const uint2*>(y+uint64_t(row)*H+col);
  const float ys[4]={bf_lo(yp.x),bf_hi(yp.x),bf_lo(yp.y),bf_hi(yp.y)};
  float coefficient=0.f;
  if constexpr (Broadcast) {
    if (lane<N*N) coefficient=comb[uint64_t(row)*N*N+lane];
    else if (lane<N*(N+1)) coefficient=post[uint64_t(row)*N+lane-N*N];
  }
  #pragma unroll
  for (int j=0;j<N;++j) {
    const float p=Broadcast ? __shfl_sync(0xffffffffu,coefficient,N*N+j) : post[uint64_t(row)*N+j];
    float mixed[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for (int s=0;s<N;++s) {
#ifdef CHOP_HC_WARP_REVERSE_STREAMS
      const int i=N-1-s;
#else
      const int i=s;
#endif
      const float w=Broadcast ? __shfl_sync(0xffffffffu,coefficient,i*N+j) : comb[uint64_t(row)*N*N+i*N+j];
      #pragma unroll
      for (int e=0;e<4;++e) mixed[e]=fmaf(w,values[i][e],mixed[e]);
    }
    #pragma unroll
    for (int e=0;e<4;++e) mixed[e]=__fadd_rn(__fmul_rn(p,ys[e]),mixed[e]);
    if (col<H) *reinterpret_cast<uint2*>(out+at+uint64_t(j)*H)=make_uint2(pack(mixed[0],mixed[1]),pack(mixed[2],mixed[3]));
  }
}
} // namespace chop::sinkhorn_hc
