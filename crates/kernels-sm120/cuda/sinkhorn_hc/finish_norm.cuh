#pragma once
#include "round_expand_mix.cuh"
#include "../row/rmsnorm_quantize_cached.cuh"

namespace chop::sinkhorn_hc::finish_norm {
template<class M, int Warps> union Arena {
  round_mix::FinishArena<M> finish;
  chop::row::rmsnorm_quantize_cached::Arena<M::kWidth,Warps> norm;
};

// The first warp folds K chains and runs Sinkhorn. Other warps collapse
// independently using the previous pre weights. The CTA barrier publishes
// the BF16 collapse before the original norm task and permits arena reuse.
// No global scratch or ticket is added to the preceding producer.
template<class M, int Warps, uint32_t FloorBits>
__device__ void task(const uint16_t* expanded, const float* base,
    const float* scale, const float* pre_in, uint16_t* collapsed,
    float* pre, float* post, float* comb, const float* workspace,
    const uint32_t* live_rows, const uint16_t* weight, uint16_t* normed,
    float* widened, uint8_t* codes, uint8_t* scales, uint32_t* status,
    uint64_t status_row_offset, float eps_norm, uint32_t tile, uint32_t row_index,
    unsigned char* arena) {
  static_assert(M::kStreams == 4 && M::kCells == 16);
  if (row_index >= *live_rows) return;
  constexpr int N=M::kStreams, H=M::kWidth, Flat=M::kFlat;
  constexpr int Mixes=M::kMixes, Cells=M::kCells, Stride=M::kChains+1;
  constexpr int Threads=Warps*32;
  const uint32_t row = row_index;
  const int t=threadIdx.x, lane=t&31, warp=t>>5;
  auto& shared=*reinterpret_cast<Arena<M,Warps>*>(arena);
  auto& sums=shared.finish.sums;
  auto& partials=shared.finish.partials;
  auto& logits=shared.finish.logits;
  auto& weights=shared.finish.weights;
  const float* source=workspace+uint64_t(row)*(Mixes*M::kChains+256);
  if (t<N) weights[t]=pre_in[uint64_t(row)*N+t];
  for (int i=t;i<Mixes*M::kChains;i+=Threads)
    sums[(i/M::kChains)*Stride+i%M::kChains]=__ldcg(source+i);
  if (t<256) {
    float square=__ldcg(source+Mixes*M::kChains+t);
    square=warp_sum(square);
    if (lane==0) partials[warp]=square;
  }
  __syncthreads();
  if (warp==0) {
    float total=0.f;
    #pragma unroll
    for (int w=0;w<8;++w) total=__fadd_rn(total,partials[w]);
    const float rstd=__frsqrt_rn(__fadd_rn(__fdiv_rn(total,float(Flat)),__uint_as_float(M::kNormEpsBits)));
    if (lane<Mixes) {
      float acc=0.f;
      #pragma unroll 8
      for (int c=0;c<M::kChains;++c) acc=__fadd_rn(acc,sums[lane*Stride+c]);
      logits[lane]=__fmul_rn(acc,rstd);
    }
    __syncwarp();
    const float eps=__uint_as_float(M::kEpsBits);
    if (lane<N) {
      pre[uint64_t(row)*N+lane]=__fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[lane],scale[0],base[lane])),eps);
      const float placed=chop::sigmoid::sigmoid(fmaf(logits[N+lane],scale[1],base[N+lane]));
      post[uint64_t(row)*N+lane]=__fmul_rn(2.f,placed);
    }
      const int i = lane / N, j = lane % N;
      const bool owner = lane < Cells;
      float cell = owner ? fmaf(logits[2 * N + lane], scale[2], base[2 * N + lane]) : 0.f;
      float max = __shfl_sync(0xffffffff, cell, i * N);
      #pragma unroll
      for (int k = 1; k < N; ++k)
        max = fmaxf(max, __shfl_sync(0xffffffff, cell, i * N + k));
      cell = expf(__fadd_rn(cell, -max));
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k)
        sum = __fadd_rn(sum, __shfl_sync(0xffffffff, cell, i * N + k));
      cell = __fadd_rn(__fdiv_rn(cell, sum), eps);
      auto normalize = [&](bool rows) {
        float total = 0.f;
        #pragma unroll
        for (int k = 0; k < N; ++k)
          total = __fadd_rn(total, __shfl_sync(0xffffffff, cell, rows ? i * N + k : k * N + j));
        cell = __fdiv_rn(cell, __fadd_rn(total, eps));
      };
      normalize(false);
      for (int iteration = 1; iteration < M::kIterations; ++iteration) {
        normalize(true);
        normalize(false);
      }
      if (owner) comb[uint64_t(row) * Cells + lane] = cell;
  } else {
    const uint16_t* streams=expanded+uint64_t(row)*Flat;
    for (int col=2*(t-32);col<H;col+=2*(Threads-32)) {
      float lo=0.f,hi=0.f;
      #pragma unroll
      for (int s=0;s<N;++s) {
        const uint32_t pair=*reinterpret_cast<const uint32_t*>(streams+s*H+col);
        lo=fmaf(weights[s],bf_lo(pair),lo);
        hi=fmaf(weights[s],bf_hi(pair),hi);
      }
      *reinterpret_cast<uint32_t*>(collapsed+uint64_t(row)*H+col)=pack(lo,hi);
    }
  }
  __syncthreads();
  chop::row::rmsnorm_quantize_cached::task<H,Warps,FloorBits>(collapsed,weight,
      normed,widened,codes,scales,live_rows,status,status_row_offset,eps_norm,
      tile,row_index,arena);
}
} // namespace chop::sinkhorn_hc::finish_norm
