#pragma once
#include "finish_twin.cuh"

// The finish/norm tail after the column-block producer (round_cols.cuh), on
// two CTAs per row as in finish_twin. That producer leaves the mixer's 256
// RMS lanes to this launch, so the mixer CTA sums each original lane's
// squares from the expanded streams: lane l FMAs the squares of packs
// v = l + 256 i in increasing i, each pack's four BF16 pairs in order, low
// half then high, as the served producers do on the same expanded values.
// The rest of the mixer (the 256 lanes' butterfly and warp order, the fold
// of each mix's chains from zero in increasing K order, pre, post and the
// shuffled Sinkhorn) is completion::mixer's, statement for statement. The
// norm CTA is finish_twin's: the collapse with the previous pre, then the
// unchanged cached norm. Roles and their owners are finish_twin's, plus the
// mixer CTA's reads of the expanded streams, which no CTA of this launch
// writes.
namespace chop::sinkhorn_hc::finish_split {
enum class Mutation { none, clear_status, wrong_floor, reverse_fold, reverse_squares };

// The norm CTA's control, if any.
template<Mutation Mut> struct NormMutation {
  static constexpr completion::Mutation value =
      Mut == Mutation::clear_status ? completion::Mutation::clear_status
      : Mut == Mutation::wrong_floor ? completion::Mutation::wrong_floor
      : completion::Mutation::none;
};

template<class M, int Warps, Mutation Mut>
__device__ void mixer(const uint16_t* expanded, const float* base, const float* scale,
    float* pre, float* post, float* comb, const float* workspace, uint32_t row,
    round_mix::FinishArena<M>& shared) {
  static_assert(M::kStreams == 4 && M::kCells == 16 && Warps >= 8);
  constexpr int N=M::kStreams, Flat=M::kFlat;
  constexpr int Mixes=M::kMixes, Cells=M::kCells, Stride=M::kChains+1;
  constexpr int Threads=Warps*32;
  constexpr int Packs=Flat/8/kThreads;
  static_assert(Flat/8%kThreads==0, "every RMS lane owns whole packs");
  const int t=threadIdx.x, lane=t&31, warp=t>>5;
  auto& sums=shared.sums;
  auto& partials=shared.partials;
  auto& logits=shared.logits;
  const float* source=workspace+uint64_t(row)*(Mixes*M::kChains+kThreads);
  // The RMS lane's packs, loaded before the chain sums are staged.
  uint4 packs[Packs];
  if (t<kThreads) {
    const uint4* streams=reinterpret_cast<const uint4*>(expanded+uint64_t(row)*Flat);
    #pragma unroll
    for (int i=0;i<Packs;++i) packs[i]=__ldcg(streams+t+kThreads*i);
  }
  // Every chain sum load is issued before any is staged.
  constexpr int Staged=(Mixes*M::kChains+Threads-1)/Threads;
  float staged[Staged];
  #pragma unroll
  for (int s=0;s<Staged;++s)
    if (t+Threads*s<Mixes*M::kChains) staged[s]=__ldcg(source+t+Threads*s);
  #pragma unroll
  for (int s=0;s<Staged;++s) {
    const int i=t+Threads*s;
    if (i<Mixes*M::kChains) sums[(i/M::kChains)*Stride+i%M::kChains]=staged[s];
  }
  if (t<kThreads) {
    float square=0.f;
    #pragma unroll
    for (int step=0;step<Packs;++step) {
      const int i=Mut==Mutation::reverse_squares ? Packs-1-step : step;
      const uint32_t words[4]={packs[i].x,packs[i].y,packs[i].z,packs[i].w};
      #pragma unroll
      for (int j=0;j<4;++j) {
        const float lo=bf_lo(words[j]), hi=bf_hi(words[j]);
        square=fmaf(lo,lo,square);
        square=fmaf(hi,hi,square);
      }
    }
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
      for (int step=0;step<M::kChains;++step) {
        const int c=Mut==Mutation::reverse_fold ? M::kChains-1-step : step;
        acc=__fadd_rn(acc,sums[lane*Stride+c]);
      }
      logits[lane]=__fmul_rn(acc,rstd);
    }
    __syncwarp();
    const float eps=__uint_as_float(M::kEpsBits);
    if (lane<N) {
      pre[uint64_t(row)*N+lane]=__fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[lane],scale[0],base[lane])),eps);
      const float placed=chop::sigmoid::sigmoid(fmaf(logits[N+lane],scale[1],base[N+lane]));
      post[uint64_t(row)*N+lane]=__fmul_rn(2.f,placed);
    }
    const int i=lane/N, j=lane%N;
    const bool owner=lane<Cells;
    float cell=owner ? fmaf(logits[2*N+lane],scale[2],base[2*N+lane]) : 0.f;
    float max=__shfl_sync(0xffffffff,cell,i*N);
    #pragma unroll
    for (int k=1;k<N;++k)
      max=fmaxf(max,__shfl_sync(0xffffffff,cell,i*N+k));
    cell=expf(__fadd_rn(cell,-max));
    float sum=0.f;
    #pragma unroll
    for (int k=0;k<N;++k)
      sum=__fadd_rn(sum,__shfl_sync(0xffffffff,cell,i*N+k));
    cell=__fadd_rn(__fdiv_rn(cell,sum),eps);
    auto normalize=[&](bool rows) {
      float total=0.f;
      #pragma unroll
      for (int k=0;k<N;++k)
        total=__fadd_rn(total,__shfl_sync(0xffffffff,cell,rows ? i*N+k : k*N+j));
      cell=__fdiv_rn(cell,__fadd_rn(total,eps));
    };
    normalize(false);
    for (int iteration=1;iteration<M::kIterations;++iteration) {
      normalize(true);
      normalize(false);
    }
    if (owner) comb[uint64_t(row)*Cells+lane]=cell;
  }
}

template<class M, int Warps, uint32_t FloorBits, Mutation Mut>
__device__ void task(const uint16_t* expanded, const float* base,
    const float* scale, const float* pre_in, uint16_t* collapsed,
    float* pre, float* post, float* comb, const float* workspace,
    const uint32_t* live_rows, const uint16_t* weight, uint16_t* normed,
    float* widened, uint8_t* codes, uint8_t* scales, uint32_t* status,
    uint64_t status_row_offset, float eps_norm, uint32_t part, uint32_t row,
    finish_norm::Arena<M,Warps>& shared) {
  static_assert(M::kStreams == 4 && M::kCells == 16 && Warps >= 8);
  if (row >= *live_rows) return;
  if (part == 0) {
    mixer<M,Warps,Mut>(expanded,base,scale,pre,post,comb,workspace,row,shared.finish);
    return;
  }
  finish_twin::task<M,Warps,FloorBits,NormMutation<Mut>::value>(expanded,base,scale,pre_in,
      collapsed,pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,
      status,status_row_offset,eps_norm,1,row,shared);
}
} // namespace chop::sinkhorn_hc::finish_split
