#pragma once
#include "mxfp8_prefill.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::gemm::mxfp8_prefill_matrix {
// Matrix loads preserve the scalar fragment lane mapping. Every output
// retains four native k32 instructions from zero, then one ordered F32 fold.
// A task owns this arena until it drains every outstanding async copy.
// Its virtual tile index is independent of the entry wrapper's grid.
template<int Rows, int Stages> struct Layout {
  static_assert(Rows == 32 || Rows == 64 || Rows == 128);
  static_assert(Stages >= 2 && Stages <= 4);
  static constexpr int threads = Rows >= 64 ? 256 : 128;
  static constexpr int row_groups = Rows >= 64 ? 2 : 1;
  static constexpr int fragments = Rows / row_groups / 16;
  static constexpr int code_bytes = (Rows + 128) * 128;
  static constexpr int stage_bytes = code_bytes + (Rows + 128) * 4;
  static constexpr int bytes = Stages * stage_bytes + Rows * 4;
};

__device__ __forceinline__ uint32_t offset(int row, int k) {
  return uint32_t(row * 128 + k) ^ uint32_t((row & 7) * 16);
}
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const uint32_t at=ptx::shared_address(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
      "r"(at),"l"(src),"r"(valid ? 16 : 0):"memory");
}
__device__ __forceinline__ void copy4(void* dst, const void* src, bool valid) {
  const uint32_t at=ptx::shared_address(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
      "r"(at),"l"(src),"r"(valid ? 4 : 0):"memory");
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Pending> __device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending):"memory");
}

template<int N,int K,int Groups,int RowStride,int GroupStride,int Rows,int Stages>
__device__ __forceinline__ void load(uint8_t* stage,int fold,uint3 tile,
    const uint8_t* input,const uint8_t* weights,const uint8_t* as,const uint8_t* ws,
    const uint32_t* admitted_rows) {
  using L=Layout<Rows,Stages>;
  const uint32_t row0=tile.y*Rows,col0=tile.x*128,group=tile.z;
  for(int index=threadIdx.x;index<(Rows+128)*8;index+=L::threads) {
    const int row=index/8,k=fold*128+(index%8)*16;
    const bool activation=row<Rows;
    const uint32_t logical=activation ? row0+row : col0+row-Rows;
    const bool valid=k<K && (activation ? admitted_rows[row]!=0 : logical<N);
    const uint64_t at=activation ? uint64_t(logical)*RowStride+uint64_t(group)*GroupStride+k
        : (uint64_t(group)*N+logical)*K+k;
    const uint8_t* base=activation ? input : weights;
    copy16(stage+offset(row,(index%8)*16),base+(valid ? at : 0),valid);
  }
  auto* scales=reinterpret_cast<uint32_t*>(stage+L::code_bytes);
  for(int row=threadIdx.x;row<Rows+128;row+=L::threads) {
    const bool activation=row<Rows;
    const uint32_t logical=activation ? row0+row : col0+row-Rows;
    const bool valid=fold*128<K && (activation ? admitted_rows[row]!=0 : logical<N);
    const uint64_t at=activation ? (uint64_t(logical)*RowStride+uint64_t(group)*GroupStride)/32+fold*4
        : (uint64_t(group)*N+logical)*(K/32)+fold*4;
    const uint8_t* base=activation ? as : ws;
    if constexpr(K%128==0 && RowStride%128==0 && GroupStride%128==0) {
      copy4(scales+row,base+(valid ? at : 0),valid);
    } else {
      uint32_t packed=0;
#pragma unroll
      for(int step=0;step<4;++step) {
        const uint32_t scale=valid && fold*128+step*32<K ? base[at+step] : 127;
        packed|=scale<<(step*8);
      }
      scales[row]=packed;
    }
  }
  commit();
}

template<int N,int K,int Groups,int RowStride,int GroupStride,bool F32Output,
    int Rows,int Stages,bool OmitFirst=false>
__device__ void task(uint3 tile,uint8_t* arena,uint32_t live,
    const uint8_t* input,const uint8_t* as,const uint8_t* weights,const uint8_t* ws,
    void* output,const uint32_t* status,uint64_t status_offset) {
  using L=Layout<Rows,Stages>;
  static_assert(K>0 && K%32==0 && Groups>0);
  static_assert(GroupStride>=K && RowStride>=(Groups-1)*GroupStride+K);
  static_assert(RowStride%32==0 && GroupStride%32==0);
  auto* active=reinterpret_cast<uint32_t*>(arena+Stages*L::stage_bytes);
  const uint32_t row0=tile.y*Rows,col0=tile.x*128;
  if(threadIdx.x<Rows) {
    const uint32_t row=row0+threadIdx.x;
    active[threadIdx.x]=row<live && status[status_offset+row]==0;
  }
  __syncthreads();
#pragma unroll
  for(int s=0;s<Stages;++s)
    load<N,K,Groups,RowStride,GroupStride,Rows,Stages>(arena+s*L::stage_bytes,s,tile,input,weights,as,ws,active);
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int wm=warp/4,wn=warp%4,quad=lane/4,lane4=lane%4;
  ptx::D4 total[L::fragments][4]{};
  for(int fold=0;fold<(K+127)/128;++fold) {
    const int slot=fold%Stages;
    // The drain has fewer outstanding groups; never rely on an empty
    // commit to keep a fixed wait depth after the last refill.
    if(fold+Stages<=(K+127)/128) wait<Stages-1>();
    else wait<0>();
    __syncthreads();
    const uint8_t* codes=arena+slot*L::stage_bytes;
    const auto* scales=reinterpret_cast<const uint32_t*>(codes+L::code_bytes);
    ptx::D4 partial[L::fragments][4]{};
#pragma unroll
    for(int step=0;step<4;++step) {
      ptx::A4 a[L::fragments];ptx::B2 b[4];
      uint32_t sa[L::fragments],sb[4];
#pragma unroll
      for(int m=0;m<L::fragments;++m) {
        const int row=wm*(Rows/L::row_groups)+m*16+quad;
        const int matrix_row=wm*(Rows/L::row_groups)+m*16+(lane&15);
        const int matrix_k=step*32+(lane>>4)*16;
        ptx::ldmatrix_b16<4,false>(a[m].v,codes+offset(matrix_row,matrix_k));
        sa[m]=(scales[row+(lane4&1)*8]>>(step*8))&255;
      }
#pragma unroll
      for(int n=0;n<4;++n) {
        const int row=Rows+wn*32+n*8+quad;
        const int matrix_row=Rows+wn*32+n*8+(lane&7);
        const int matrix_k=step*32+((lane>>3)&1)*16;
        ptx::ldmatrix_b16<2,false>(b[n].v,codes+offset(matrix_row,matrix_k));
        sb[n]=(scales[row]>>(step*8))&255;
      }
      if(!OmitFirst || fold!=0) {
#pragma unroll
        for(int m=0;m<L::fragments;++m)
#pragma unroll
          for(int n=0;n<4;++n)
            ptx::mma_mxf8f6f4<ptx::Element::e4m3,ptx::Element::e4m3>(partial[m][n],a[m],b[n],sa[m],sb[n]);
      }
    }
    __syncthreads();
    if(fold+Stages<(K+127)/128)
      load<N,K,Groups,RowStride,GroupStride,Rows,Stages>(arena+slot*L::stage_bytes,fold+Stages,tile,input,weights,as,ws,active);
#pragma unroll
    for(int m=0;m<L::fragments;++m)
#pragma unroll
      for(int n=0;n<4;++n)
#pragma unroll
        for(int v=0;v<4;++v) total[m][n].v[v]=__fadd_rn(total[m][n].v[v],partial[m][n].v[v]);
  }
  wait<0>();
  __syncthreads();
#pragma unroll
  for(int m=0;m<L::fragments;++m)
#pragma unroll
    for(int n=0;n<4;++n)
#pragma unroll
      for(int v=0;v<4;++v) {
        const int local=wm*(Rows/L::row_groups)+m*16+quad+(v/2)*8;
        const uint32_t row=row0+local,col=col0+wn*32+n*8+lane4*2+v%2;
        if(row>=live || col>=N) continue;
        const uint64_t at=(uint64_t(row)*Groups+tile.z)*N+col;
        const float value=active[local] ? total[m][n].v[v] : 0.f;
        if constexpr(F32Output) static_cast<float*>(output)[at]=value;
        else static_cast<uint16_t*>(output)[at]=convert::bf16_from_f32(value);
      }
}

template<int N,int K,int Groups,int RowStride,int GroupStride,bool F32Output,int Rows,int Stages,bool OmitFirst=false>
__device__ void run(const uint8_t* input,const uint8_t* as,const uint8_t* weights,const uint8_t* ws,
    void* output,const uint32_t* live_rows,const uint32_t* status,uint64_t status_offset,uint32_t rows) {
  extern __shared__ __align__(16) uint8_t arena[];
  if(blockDim.x!=Layout<Rows,Stages>::threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live=min(rows,*live_rows);
  if(blockIdx.y*Rows>=live) return;
  task<N,K,Groups,RowStride,GroupStride,F32Output,Rows,Stages,OmitFirst>(blockIdx,arena,live,input,as,weights,ws,output,status,status_offset);
}
} // namespace chop::gemm::mxfp8_prefill_matrix
