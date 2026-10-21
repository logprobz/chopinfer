#pragma once
#include "completion.cuh"

// The served finisher and cached input norm (finish_norm::task) on two CTAs
// per row. In that task the norm reads only the collapse, and the collapse
// reads only the expanded streams and the previous pre weights, so it never
// waits for the fold or Sinkhorn. CTA 0 runs the mixer half (completion::
// mixer, the qualified copy of the finisher's warp 0). CTA 1 collapses with
// the finisher's expression, then runs the unchanged cached norm. Neither CTA
// reads a role the other writes: CTA 0 reads the partial workspace, base and
// scale and writes pre, post and comb; CTA 1 reads pre_in, expanded and the
// norm weight and writes the collapse and the norm outputs. pre and pre_in
// are distinct buffers. No global scratch or ticket is added.
namespace chop::sinkhorn_hc::finish_twin {
using completion::Mutation;

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
    completion::mixer<M,Warps,Mut>(base,scale,pre,post,comb,workspace,row,shared.finish);
    return;
  }
  constexpr int N=M::kStreams, H=M::kWidth, Flat=M::kFlat;
  constexpr int Threads=Warps*32;
  const int t=threadIdx.x;
  auto& weights=shared.finish.weights;
  if (t<N) weights[t]=pre_in[uint64_t(row)*N+t];
  __syncthreads();
  // finish_norm::task's collapse: per column an FMA chain from zero in
  // stream order; only the column-to-thread mapping differs.
  const uint16_t* streams=expanded+uint64_t(row)*Flat;
  for (int col=2*t;col<H;col+=2*Threads) {
    float lo=0.f,hi=0.f;
    #pragma unroll
    for (int s=0;s<N;++s) {
      const uint32_t pair=*reinterpret_cast<const uint32_t*>(streams+s*H+col);
      lo=fmaf(weights[s],bf_lo(pair),lo);
      hi=fmaf(weights[s],bf_hi(pair),hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed+uint64_t(row)*H+col)=pack(lo,hi);
  }
  // Publishes the collapse to the norm and permits arena reuse.
  __syncthreads();
  if constexpr (Mut==Mutation::clear_status) {
    if (t==0) status[status_row_offset+row]=0;
    __syncthreads();
  }
  constexpr uint32_t Floor=Mut==Mutation::wrong_floor ? 0x461c4000u : FloorBits;
  chop::row::rmsnorm_quantize_cached::task<H,Warps,Floor>(collapsed,weight,
      normed,widened,codes,scales,live_rows,status,status_row_offset,eps_norm,
      0,row,reinterpret_cast<unsigned char*>(&shared.norm));
}
} // namespace chop::sinkhorn_hc::finish_twin
