#pragma once
#include "round_chain_partials.cuh"
#include "finish_norm.cuh"

// One launch for a decode HC boundary: the rounded chain producer, the
// lagged mixer finish and the cached input norm. Every float follows the
// served two-launch pair (round_chain_partials.cuh, finish_norm.cuh); only
// the launch boundary is replaced by a per-row ticket.
namespace chop::sinkhorn_hc::completion {
enum class Mutation { none, clear_status, wrong_floor, reverse_fold };

// Whether this CTA is the last of `ctas` to arrive at `ticket`. The CTA
// barrier orders every thread's prior loads and stores before thread zero's
// GPU-scope release; the last arrival acquires them all. No CTA waits for
// another, so scheduling order cannot deadlock. The last arrival returns the
// ticket to zero; tickets are zeroed at allocation and disjoint from the
// workspace, so every launch starts at zero.
__device__ __forceinline__ bool arrive(uint32_t* ticket, uint32_t ctas, uint32_t& flag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t old;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(old) : "l"(ticket) : "memory");
    if (old >= ctas) __trap();
    flag = old == ctas - 1;
    if (flag) *ticket = 0;
  }
  __syncthreads();
  return flag != 0;
}

// The finisher's mixer half, copied from finish_norm::task: stage the
// chain sums, add the 256 original RMS lanes, fold each mix's chains from
// zero in increasing K order, then pre, post and the shuffled Sinkhorn.
// Requires at least 256 threads. Writes only pre, post and comb.
template<class M, int Warps, Mutation Mut>
__device__ void mixer(const float* base, const float* scale, float* pre,
    float* post, float* comb, const float* workspace, uint32_t row,
    round_mix::FinishArena<M>& shared) {
  static_assert(M::kStreams == 4 && M::kCells == 16 && Warps >= 8);
  constexpr int N=M::kStreams, Flat=M::kFlat;
  constexpr int Mixes=M::kMixes, Cells=M::kCells, Stride=M::kChains+1;
  constexpr int Threads=Warps*32;
  const int t=threadIdx.x, lane=t&31, warp=t>>5;
  auto& sums=shared.sums;
  auto& partials=shared.partials;
  auto& logits=shared.logits;
  const float* source=workspace+uint64_t(row)*(Mixes*M::kChains+256);
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

template<class M> struct Coefficients {
  float post[M::kStreams];
  float comb[M::kCells];
  float weights[M::kStreams];
};

// The norm CTA's collapse. Each column pair recomputes the producer's four
// expanded BF16 pairs (expand_mix_pair's FMAs from zero in input-stream
// order, then the rounded sum's scaled add) from staged coefficients, and
// collapses them with the previous pre weights in stream order, exactly as
// the finisher reads the expanded streams.
template<class M, int Threads>
__device__ void collapse(const uint16_t* x, const float* y, const Coefficients<M>& c,
    uint16_t* collapsed, uint32_t row) {
  constexpr int N=M::kStreams, H=M::kWidth;
  const uint64_t at=uint64_t(row)*N*H;
  for (int col=2*threadIdx.x;col<H;col+=2*Threads) {
    uint32_t pairs[N];
    #pragma unroll
    for (int i=0;i<N;++i)
      pairs[i]=*reinterpret_cast<const uint32_t*>(x+at+col+uint64_t(i)*H);
    const uint32_t ys=round_mix::rounded_pair(y,uint64_t(row)*H+col);
    float lo=0.f,hi=0.f;
    #pragma unroll
    for (int s=0;s<N;++s) {
      float elo=0.f,ehi=0.f;
      #pragma unroll
      for (int i=0;i<N;++i) {
        const float w=c.comb[i*N+s];
        elo=fmaf(w,bf_lo(pairs[i]),elo);
        ehi=fmaf(w,bf_hi(pairs[i]),ehi);
      }
      const float p=c.post[s];
      const uint32_t expanded=pack(__fadd_rn(__fmul_rn(p,bf_lo(ys)),elo),
                                   __fadd_rn(__fmul_rn(p,bf_hi(ys)),ehi));
      lo=fmaf(c.weights[s],bf_lo(expanded),lo);
      hi=fmaf(c.weights[s],bf_hi(expanded),hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed+uint64_t(row)*H+col)=pack(lo,hi);
  }
}

template<class M, int Warps> struct TailArena {
  uint32_t last;
  alignas(16) unsigned char tail[sizeof(finish_norm::Arena<M,Warps>)];
};

template<class M, int Warps> struct SplitArena {
  uint32_t last;
  union {
    Coefficients<M> coefficients;
    round_mix::FinishArena<M> finish;
    chop::row::rmsnorm_quantize_cached::Arena<M::kWidth,Warps> norm;
  };
};

// Tail layout, grid (kChains, rows): every CTA produces one chain; the last
// arrival runs the unchanged finisher, mixer and collapse, then the norm.
template<class M, int Warps, uint32_t FloorBits, Mutation Mut>
__device__ void tail_task(uint32_t chain, uint32_t row, TailArena<M,Warps>& shared,
    const uint16_t* x, const float* sum, uint16_t* rounded,
    const float* post_in, const float* comb_in, uint16_t* expanded, const void* fn,
    const float* base, const float* scale, const float* pre_in,
    uint16_t* collapsed, float* pre, float* post, float* comb,
    float* workspace, const uint32_t* live_rows, const uint16_t* weight,
    uint16_t* normed, float* widened, uint8_t* codes, uint8_t* scales,
    uint32_t* status, uint64_t status_offset, float eps, uint32_t* tickets) {
  static_assert(Warps==8 || Warps==16 || Warps==32);
  static_assert(Mut!=Mutation::reverse_fold);
  if (row>=*live_rows) return;
  if (threadIdx.x<256)
    round_chain::task<M,8>(x,sum,rounded,post_in,comb_in,expanded,fn,
        workspace,live_rows,chain,row,nullptr);
  if (!arrive(tickets+row,M::kChains,shared.last)) return;
  if constexpr (Mut==Mutation::clear_status)
    if (threadIdx.x==0) status[status_offset+row]=0;
  __syncthreads();
  constexpr uint32_t Floor=Mut==Mutation::wrong_floor ? 0x461c4000u : FloorBits;
  finish_norm::task<M,Warps,Floor>(expanded,base,scale,pre_in,collapsed,
      pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,
      status,status_offset,eps,0,row,shared.tail);
}

// Split layout, grid (kChains + 1, rows). CTA 0 stages the row's post,
// comb and previous pre, arrives before any output write, then collapses
// and runs the unchanged cached norm. CTAs 1.. produce the chains and the
// original RMS lanes. The last of all kChains + 1 arrivals runs the mixer,
// so post and comb (which may alias post_in and comb_in) are rewritten
// only after every reader has released them. The norm never waits for the
// mixer, and the mixer never waits for the norm.
template<class M, int Warps, uint32_t FloorBits, Mutation Mut>
__device__ void split_task(uint32_t block, uint32_t row, SplitArena<M,Warps>& shared,
    const uint16_t* x, const float* sum, uint16_t* rounded,
    const float* post_in, const float* comb_in, uint16_t* expanded, const void* fn,
    const float* base, const float* scale, const float* pre_in,
    uint16_t* collapsed, float* pre, float* post, float* comb,
    float* workspace, const uint32_t* live_rows, const uint16_t* weight,
    uint16_t* normed, float* widened, uint8_t* codes, uint8_t* scales,
    uint32_t* status, uint64_t status_offset, float eps, uint32_t* tickets) {
  static_assert(Warps==8 || Warps==16);
  static_assert(M::kStreams == 4 && M::kCells == 16);
  constexpr int N=M::kStreams;
  constexpr uint32_t Arrivals=M::kChains+1;
  if (row>=*live_rows) return;
  const int t=threadIdx.x;
  bool last;
  if (block==0) {
    auto& c=shared.coefficients;
    if (t<N) {
      c.post[t]=post_in[uint64_t(row)*N+t];
      c.weights[t]=pre_in[uint64_t(row)*N+t];
    }
    if (t<M::kCells) c.comb[t]=comb_in[uint64_t(row)*M::kCells+t];
    last=arrive(tickets+row,Arrivals,shared.last);
    collapse<M,Warps*32>(x,sum,c,collapsed,row);
    __syncthreads();
    if constexpr (Mut==Mutation::clear_status)
      if (t==0) status[status_offset+row]=0;
    __syncthreads();
    constexpr uint32_t Floor=Mut==Mutation::wrong_floor ? 0x461c4000u : FloorBits;
    chop::row::rmsnorm_quantize_cached::task<M::kWidth,Warps,Floor>(collapsed,weight,
        normed,widened,codes,scales,live_rows,status,status_offset,eps,0,row,
        reinterpret_cast<unsigned char*>(&shared.norm));
    if (!last) return;
    __syncthreads();
  } else {
    if (t<256)
      round_chain::task<M,8>(x,sum,rounded,post_in,comb_in,expanded,fn,
          workspace,live_rows,block-1,row,nullptr);
    if (!arrive(tickets+row,Arrivals,shared.last)) return;
  }
  mixer<M,Warps,Mut>(base,scale,pre,post,comb,workspace,row,shared.finish);
}
} // namespace chop::sinkhorn_hc::completion
