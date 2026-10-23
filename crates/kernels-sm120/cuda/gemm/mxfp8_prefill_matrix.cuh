#pragma once
#include "mxfp8_prefill.cuh"
#include "../common/ptx_matrix.cuh"
#include "mxfp8_fold_packed.cuh"

namespace chop::gemm::mxfp8_prefill_matrix {
// Matrix loads preserve the scalar fragment lane mapping. Every output
// retains four native k32 instructions from zero, then one ordered F32 fold.
// A task owns this arena until it drains every outstanding async copy.
// Its virtual tile index is independent of the entry wrapper's grid.
//
// PackedWeights reads weight codes stored as mxfp8-fold-packed-8x128-v1
// (mxfp8_fold_packed.cuh). A strip's fold is contiguous: each weight row's
// segment (128 bytes, or 64 in a final 64-K fold) is staged where its
// row-major chunks went, and the chunks past K stay zero-filled. A staged
// 16-byte row piece then holds one lane's words, not a row of the B
// fragment, so no ldmatrix yields the fragment: each lane takes its B words
// for two steps from one 16-byte shared load instead (lane l's words of a
// row are bytes [32l, 32l + 32) of its segment, [16l, 16l + 16) in a half
// segment). With the chunk swizzle every quarter warp's 16-byte loads hit
// distinct banks; a final half fold keeps that by staging odd rows'
// segments in the other half of their rows (half_flip). The staging copies
// do not branch on the operand, and the B loads are unguarded. Only the
// weight-code address changes.
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
// In a final 64-K fold an odd weight row's half segment and its zero half
// trade places, so the two rows each quarter warp's 16-byte B loads read
// fall in opposite halves of the swizzled pieces and share no bank.
__device__ __forceinline__ uint32_t half_flip(bool half, int row) {
  return half ? uint32_t(row & 1) * 64 : 0;
}
// Byte of packed weight row `row`'s staged segment holding K bytes
// [local, local + 4) of its fold, for a B load at step local / 32. In a
// half fold the steps past K read the zero half.
__device__ __forceinline__ uint32_t packed_at(bool half, int row, uint32_t local) {
  const uint32_t at = half
      ? mxfp8_fold_packed::segment_word(0, local % 64, true) + local / 64 * 64
      : mxfp8_fold_packed::segment_word(0, local, false);
  return at ^ half_flip(half, row);
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

template<int N,int K,int Groups,int RowStride,int GroupStride,int Rows,int Stages,
    bool PackedWeights=false>
__device__ __forceinline__ void load(uint8_t* stage,int fold,uint3 tile,
    const uint8_t* input,const uint8_t* weights,const uint8_t* as,const uint8_t* ws,
    const uint32_t* admitted_rows) {
  using L=Layout<Rows,Stages>;
  const uint32_t row0=tile.y*Rows,col0=tile.x*128,group=tile.z;
  if constexpr(PackedWeights) {
    // The row-major loop's copies in its order: activation rows, then the
    // tile's 16 strips, each fold one contiguous run, with no branch on
    // the operand. Every thread makes the same number of copies (run traps
    // any other block size). The loops stay rolled: unrolled, they cost
    // registers. A half fold stages an odd row's segment in the other half
    // of its row (see half_flip).
    static_assert(Rows*8%L::threads==0 && 128*8%L::threads==0);
#pragma unroll 1
    for(int i=0;i<Rows*8/L::threads;++i) {
      const int index=threadIdx.x+i*L::threads;
      const int row=index/8,k=fold*128+(index%8)*16;
      const bool valid=k<K && admitted_rows[row]!=0;
      const uint64_t at=uint64_t(row0+row)*RowStride+uint64_t(group)*GroupStride+k;
      copy16(stage+offset(row,(index%8)*16),input+(valid ? at : 0),valid);
    }
    const bool half=K%128!=0 && fold==K/128;
    const uint8_t* strips=weights+mxfp8_fold_packed::fold_offset((group*N+col0)/8,fold,K);
#pragma unroll 1
    for(int i=0;i<128*8/L::threads;++i) {
      const int index=threadIdx.x+i*L::threads;
      const int row=index/8,piece=index%8;
      const bool valid=fold*128+piece*16<K && col0+row<N;
      const uint32_t at=row/8*8*K+row%8*(half ? 64 : 128)+piece*16;
      copy16(stage+offset(Rows+row,piece*16^half_flip(half,row)),
             valid ? strips+at : weights,valid);
    }
  } else {
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
    int Rows,int Stages,bool OmitFirst=false,bool PackedWeights=false>
__device__ void task(uint3 tile,uint8_t* arena,uint32_t live,
    const uint8_t* input,const uint8_t* as,const uint8_t* weights,const uint8_t* ws,
    void* output,const uint32_t* status,uint64_t status_offset) {
  using L=Layout<Rows,Stages>;
  static_assert(K>0 && K%32==0 && Groups>0);
  static_assert(GroupStride>=K && RowStride>=(Groups-1)*GroupStride+K);
  static_assert(RowStride%32==0 && GroupStride%32==0);
  static_assert(!PackedWeights || (N%8==0 && K%64==0));
  auto* active=reinterpret_cast<uint32_t*>(arena+Stages*L::stage_bytes);
  const uint32_t row0=tile.y*Rows,col0=tile.x*128;
  if(threadIdx.x<Rows) {
    const uint32_t row=row0+threadIdx.x;
    active[threadIdx.x]=row<live && status[status_offset+row]==0;
  }
  __syncthreads();
#pragma unroll
  for(int s=0;s<Stages;++s)
    load<N,K,Groups,RowStride,GroupStride,Rows,Stages,PackedWeights>(arena+s*L::stage_bytes,s,tile,input,weights,as,ws,active);
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
    [[maybe_unused]] uint4 piece[4];
    [[maybe_unused]] const bool half=K%128!=0 && fold==K/128;
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
        if constexpr(PackedWeights) {
          // Unguarded: a half fold's steps 2 and 3 read the zero half, the
          // zeros row-major staging leaves past K.
          if(step%2==0)
            piece[n]=*reinterpret_cast<const uint4*>(codes
                +offset(row,packed_at(half,row,step*32+lane4*4)/16*16));
          b[n].v[0]=step%2 ? piece[n].z : piece[n].x;
          b[n].v[1]=step%2 ? piece[n].w : piece[n].y;
        } else {
          const int matrix_row=Rows+wn*32+n*8+(lane&7);
          const int matrix_k=step*32+((lane>>3)&1)*16;
          ptx::ldmatrix_b16<2,false>(b[n].v,codes+offset(matrix_row,matrix_k));
        }
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
      load<N,K,Groups,RowStride,GroupStride,Rows,Stages,PackedWeights>(arena+slot*L::stage_bytes,fold+Stages,tile,input,weights,as,ws,active);
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

template<int N,int K,int Groups,int RowStride,int GroupStride,bool F32Output,int Rows,int Stages,bool OmitFirst=false,
    bool PackedWeights=false>
__device__ void run(const uint8_t* input,const uint8_t* as,const uint8_t* weights,const uint8_t* ws,
    void* output,const uint32_t* live_rows,const uint32_t* status,uint64_t status_offset,uint32_t rows) {
  extern __shared__ __align__(16) uint8_t arena[];
  if(blockDim.x!=Layout<Rows,Stages>::threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live=min(rows,*live_rows);
  if(blockIdx.y*Rows>=live) return;
  task<N,K,Groups,RowStride,GroupStride,F32Output,Rows,Stages,OmitFirst,PackedWeights>(blockIdx,arena,live,input,as,weights,ws,output,status,status_offset);
}
} // namespace chop::gemm::mxfp8_prefill_matrix
