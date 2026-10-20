#pragma once

// Prefill-only lane mapping. The first log2(Rows) butterfly levels stay
// inside a thread; its low lane bits finish the same original tree.
// Each warp stages its own fn panel and needs no CTA barrier in the K loop.
#include "mix_tiled.cuh"

namespace chop::sinkhorn_hc {

template<class M, int Rows, bool StageFn, bool Probe>
__device__ void mix_collapse_prefill_warp(const uint16_t* x, const void* fn,
    const float* base, const float* scale, const float* pre_in, uint16_t* collapsed,
    float* pre, float* post, float* comb, float* projected, float* rstd_out,
    float* cells_out, const uint32_t* live_rows) {
  using T = Tiled<M, Rows>;
  constexpr int N=M::kStreams, H=M::kWidth, Flat=M::kFlat;
  constexpr int Mixes=M::kMixes, Cells=M::kCells, PerWarp=M::kPerWarp;
  constexpr int ElementBytes=M::kF32Fn ? 4 : 2;
  constexpr int Group=32/Rows;
  constexpr int PacksPerMix=kChain*ElementBytes/16;
  static_assert(Rows==4 || Rows==8 || Rows==16);
  __shared__ __align__(16) uint8_t fn_stage[StageFn ? 2 : 1][kWarps][PerWarp*kChain*ElementBytes];
  __shared__ float partials[kWarps][Rows];
  __shared__ float rstds[Rows];
  __shared__ float logits[Rows][Mixes];
  __shared__ float weights[Rows][N];
  const int t=threadIdx.x, lane=t&31, warp=t>>5;
  const uint32_t first=blockIdx.x*Rows;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live=*live_rows;
  if (first>=live) return;
  const int count=live-first<uint32_t(Rows) ? int(live-first) : Rows;
  for (int i=t;i<count*N;i+=kThreads) weights[i/N][i%N]=pre_in[uint64_t(first)*N+i];
  auto stage_fn=[&](int chain) {
    for (int pack=lane;pack<PerWarp*PacksPerMix;pack+=32) {
      const int p=pack/PacksPerMix;
      uint8_t* dst=fn_stage[chain&1][warp]+pack*16;
      const int mix=warp+p*kWarps;
      if (mix<Mixes) {
        const char* src=static_cast<const char*>(fn)+(uint64_t(mix)*Flat+chain*kChain)*ElementBytes+(pack%PacksPerMix)*16;
        tiled_detail::cp_async16(dst,src);
      } else *reinterpret_cast<uint4*>(dst)=make_uint4(0,0,0,0);
    }
    tiled_detail::cp_async_commit();
  };
  if constexpr (StageFn) stage_fn(0);
  float squares[Rows];
  #pragma unroll
  for (int r=0;r<Rows;++r) squares[r]=0.f;
  float acc[PerWarp];
  #pragma unroll
  for (int p=0;p<PerWarp;++p) acc[p]=0.f;
  const int own_row=lane/Group, low_lane=lane%Group;
  for (int c=0;c<M::kChains;++c) {
    if constexpr (StageFn) {
      if (c+1<M::kChains) {stage_fn(c+1);tiled_detail::cp_async_wait<1>();}
      else tiled_detail::cp_async_wait<0>();
      __syncwarp();
    }
    // Thread t still visits packs t, t+256, ... in that order.
    if (c%(kThreads/(kChain/8))==t/(kChain/8)) {
      #pragma unroll
      for (int r=0;r<Rows;++r) {
        uint4 packed=make_uint4(0,0,0,0);
        if (r<count) packed=*reinterpret_cast<const uint4*>(x+uint64_t(first+r)*Flat+c*kChain+(t%(kChain/8))*8);
        const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
        #pragma unroll
        for (int j=0;j<4;++j) {
          const float lo=bf_lo(words[j]), hi=bf_hi(words[j]);
          squares[r]=fmaf(lo,lo,squares[r]);
          squares[r]=fmaf(hi,hi,squares[r]);
        }
      }
    }
    float sums[PerWarp][Rows];
    #pragma unroll
    for (int g=0;g<Rows;++g) {
      const int original_lane=low_lane+g*Group;
      uint2 packed=make_uint2(0,0);
      if (own_row<count) packed=*reinterpret_cast<const uint2*>(x+uint64_t(first+own_row)*Flat+c*kChain+original_lane*4);
      const float xv[4]={bf_lo(packed.x),bf_hi(packed.x),bf_lo(packed.y),bf_hi(packed.y)};
      #pragma unroll
      for (int p=0;p<PerWarp;++p) {
        float w[4];
        if constexpr (StageFn) load_fn<M::kF32Fn>(fn_stage[c&1][warp],p*kChain+original_lane*4,w);
        else {
          const int mix=warp+p*kWarps;
          if (mix<Mixes) load_fn<M::kF32Fn>(fn,uint64_t(mix)*Flat+c*kChain+original_lane*4,w);
          else {w[0]=0.f;w[1]=0.f;w[2]=0.f;w[3]=0.f;}
        }
        float value=__fmul_rn(w[0],xv[0]);
        value=fmaf(w[1],xv[1],value);
        value=fmaf(w[2],xv[2],value);
        value=fmaf(w[3],xv[3],value);
        sums[p][g]=value;
      }
    }
    #pragma unroll
    for (int p=0;p<PerWarp;++p) {
      #pragma unroll
      for (int offset=Rows/2;offset>0;offset>>=1) {
        #pragma unroll
        for (int g=0;g<offset;++g) sums[p][g]=__fadd_rn(sums[p][g],sums[p][g+offset]);
      }
      float value=sums[p][0];
      #pragma unroll
      for (int step=0;step<5-T::kLevels;++step) {
#ifdef CHOP_HC_WARP_REVERSE_TREE
        const int offset=1<<step;
#else
        const int offset=(Group/2)>>step;
#endif
        value=__fadd_rn(value,__shfl_xor_sync(0xffffffffu,value,offset));
      }
      acc[p]=__fadd_rn(acc[p],value);
    }
    if constexpr (StageFn) __syncwarp();
  }
  // rstd per row: the squares' butterfly per warp, then the warps in order.
  tiled_detail::reduce<Rows, T::kLevels>(squares, lane);
  const int group = lane / T::kGroupLanes;
  const bool leader = lane % T::kGroupLanes == 0;
  if (leader) partials[warp][group] = squares[0];
  __syncthreads();
  if (t < Rows) {
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, partials[w][t]);
    rstds[t] =
        __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));
  }
  __syncthreads();
  if (leader && group < count) {
    #pragma unroll
    for (int p = 0; p < PerWarp; ++p) {
      const int m = warp + p * kWarps;
      if (m < Mixes) {
        logits[group][m] = __fmul_rn(acc[p], rstds[group]);
        if constexpr (Probe) projected[uint64_t(first + group) * Mixes + m] = acc[p];
      }
    }
  }
  if constexpr (Probe) {
    if (t < count) rstd_out[first + t] = rstds[t];
  }
  __syncthreads();

  // pre and post.
  const float eps = __uint_as_float(M::kEpsBits);
  for (int i = t; i < count * N; i += kThreads) {
    const int r = i / N;
    const int s = i % N;
    pre[uint64_t(first + r) * N + s] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[r][s], scale[0], base[s])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[r][N + s], scale[1], base[N + s]));
    post[uint64_t(first + r) * N + s] = __fmul_rn(2.f, placed);
  }
  __syncthreads();

  if (warp < T::kSinkhornWarps) {
    // Sinkhorn: lane group of kCellLanes lanes per row, lane cell (i, j) =
    // (cell / N, cell % N); every sum runs in index order from zero over
    // the cells its line holds, read from their lanes.
    constexpr int L = T::kCellLanes;
    for (int row0 = warp * T::kRowsPerWarp; row0 < Rows; row0 += kWarps * T::kRowsPerWarp) {
      const int r = row0 + lane / L;
      const int cell = lane % L;
      const int row_lane = lane - cell;
      const bool owner = cell < Cells && r < count;
      const int i = cell / N;
      const int j = cell % N;
      auto at = [&](float value, int index) {
        return __shfl_sync(0xffffffffu, value, row_lane + index);
      };
      float value = owner ? fmaf(logits[r][2 * N + cell], scale[2], base[2 * N + cell]) : 0.f;
      float max = at(value, i * N);
      #pragma unroll
      for (int k = 1; k < N; ++k) max = fmaxf(max, at(value, i * N + k));
      if (owner) value = expf(__fadd_rn(value, -max));
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, at(value, i * N + k));
      if (owner) {
        value = __fadd_rn(__fdiv_rn(value, sum), eps);
        if constexpr (Probe) cells_out[uint64_t(first + r) * Cells + cell] = value;
      }
      auto normalize = [&](bool rows) {
        float line = 0.f;
        #pragma unroll
        for (int k = 0; k < N; ++k) line = __fadd_rn(line, at(value, rows ? i * N + k : k * N + j));
        if (owner) value = __fdiv_rn(value, __fadd_rn(line, eps));
      };
      normalize(false);
      for (int iteration = 1; iteration < M::kIterations; ++iteration) {
        normalize(true);
        normalize(false);
      }
      if (owner) comb[uint64_t(first + r) * Cells + cell] = value;
    }
  }

  // The collapse: per column an FMA chain from zero in stream order.
  if (t >= T::kCollapseFirst) {
    constexpr int Packs = H / 8;
    for (int item = t - T::kCollapseFirst; item < count * Packs;
         item += kThreads - T::kCollapseFirst) {
      const int r = item / Packs;
      const int col = (item % Packs) * 8;
      const uint16_t* row = x + uint64_t(first + r) * Flat + col;
      float out[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) out[e] = 0.f;
      #pragma unroll
      for (int s = 0; s < N; ++s) {
        const uint4 packed = *reinterpret_cast<const uint4*>(row + s * H);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          out[2 * e] = fmaf(weights[r][s], bf_lo(words[e]), out[2 * e]);
          out[2 * e + 1] = fmaf(weights[r][s], bf_hi(words[e]), out[2 * e + 1]);
        }
      }
      uint4 packed;
      packed.x = pack(out[0], out[1]);
      packed.y = pack(out[2], out[3]);
      packed.z = pack(out[4], out[5]);
      packed.w = pack(out[6], out[7]);
      *reinterpret_cast<uint4*>(collapsed + uint64_t(first + r) * H + col) = packed;
    }
  }
}

} // namespace chop::sinkhorn_hc
