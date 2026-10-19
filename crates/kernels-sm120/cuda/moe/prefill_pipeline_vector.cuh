#pragma once
#include "chained_streamed.cuh"

// Lab-only vector-store experiment over the frozen exact pipeline. Arithmetic is the column kernel's eight k16
// MMAs per 128-K chain, including its zero-padded final chain.
namespace chop::moe::prefill_pipeline_vector {
constexpr int Stride = 72;
template<class C, int MaxRows> struct Queue {
  static constexpr int slots = (C::tokens_capacity*C::top_k+MaxRows-1)/MaxRows+C::experts;
};

template<class C, int MaxRows>
__device__ void pack(const int* counts, const int* offsets, const int* active,
                     const int* live_ptr, int* table) {
  const int lane = threadIdx.x, live = *live_ptr, total = active[0];
  if (blockDim.x != 32 || gridDim.x != 1 || live < 0 || live > C::tokens_capacity ||
      total < 0 || total > C::experts || total > live*C::top_k) __trap();
  int carry[4] = {};
  for (int base = 0; base < total; base += 32) {
    const bool valid = base+lane < total;
    const int expert = valid ? active[base+lane+1] : 0;
    if (expert < 0 || expert >= C::experts) __trap();
    const int count = valid ? counts[expert] : 0, first = valid ? offsets[expert] : 0;
    if (valid && (count <= 0 || count > live || first < 0 || first%64)) __trap();
    const int full = count/MaxRows, tail = count%MaxRows;
    const int sizes[4] = {int(tail > 0 && tail <= 8), int(tail > 8 && tail <= 32),
      (MaxRows == 64 ? full : 0)+int(tail > 32 && tail <= 64),
      (MaxRows == 128 ? full : 0)+int(tail > 64)};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int prefix = sizes[q];
#pragma unroll
      for (int d = 1; d < 32; d *= 2) {
        const int previous = __shfl_up_sync(0xffffffffu,prefix,d);
        if (lane >= d) prefix += previous;
      }
      const int begin = carry[q]+prefix-sizes[q];
      if (begin+sizes[q] > Queue<C,MaxRows>::slots) __trap();
      for (int tile = 0; tile < sizes[q]; ++tile) {
        const bool full_queue = (MaxRows == 64 && q == 2) || (MaxRows == 128 && q == 3);
        const int row = full_queue ? tile*MaxRows : full*MaxRows;
        const int at = 8+3*(q*Queue<C,MaxRows>::slots+begin+tile);
        table[at] = expert;
        table[at+1] = first+row;
        table[at+2] = min(MaxRows,count-row);
      }
      carry[q] += __shfl_sync(0xffffffffu,prefix,31);
    }
  }
  if (lane < 4) { table[lane] = carry[lane]; table[4+lane] = 0; }
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void copy_scale(uint32_t* dst, const uint8_t* src,
                                         uint64_t at, uint64_t bytes, bool valid) {
  const uint64_t aligned = at & ~uint64_t(3);
  const int size = valid ? int(min(uint64_t(4),bytes-aligned)) : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(src+(valid ? aligned : 0)), "r"(size) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int Pending> __device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

template<class C, int Rows, int Cols, int Stages, bool Predecoded> struct Shared {
  struct alignas(16) Raw {
    uint8_t weights[C::halves*Cols][32];
    uint8_t input[Rows][Predecoded ? 128 : 64];
    uint32_t scales[C::halves*Cols+Rows];
  } raw[Stages];
  alignas(16) uint16_t decoded[C::halves*Cols+Rows][Stride];
  int routes[Rows];
  int item;
};

template<class C, int Rows, int Cols, int Stages, bool AsyncScale, bool Predecoded>
__device__ __forceinline__ void enqueue(Shared<C,Rows,Cols,Stages,Predecoded>& s,
    int slot, int step, int expert, int column, int live,
    const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales) {
  auto& raw = s.raw[slot];
  const int k0 = step*64;
  const bool valid_k = k0 < C::k;
  for (int v = threadIdx.x; v < C::halves*Cols*2; v += blockDim.x) {
    const int r = v/2, offset = (v%2)*16;
    const int col = column+r%Cols;
    const uint64_t wr = uint64_t(expert)*C::weight_rows+(r/Cols)*C::n+col;
    const bool valid = valid_k && col < C::n;
    copy16(raw.weights[r]+offset,weight+(valid ? wr*(C::k/2)+k0/2+offset : 0),valid);
  }
  for (int v = threadIdx.x; v < Rows*(Predecoded ? 8 : 4); v += blockDim.x) {
    const int r = v/(Predecoded ? 8 : 4), offset = (v%(Predecoded ? 8 : 4))*16;
    const int route = s.routes[r], source = C::gate_up ? route/C::top_k : route;
    const bool valid = route >= 0 && valid_k;
    copy16(raw.input[r]+offset,input+(valid ? (uint64_t(source)*C::k+k0)*(Predecoded ? 2 : 1)+offset : 0),valid);
  }
  for (int r = threadIdx.x; r < C::halves*Cols+(Predecoded ? 0 : Rows); r += blockDim.x) {
    const bool w = r < C::halves*Cols;
    const int col = column+r%Cols;
    const int route = w ? 0 : s.routes[r-C::halves*Cols];
    const int source = C::gate_up ? route/C::top_k : route;
    const bool valid = valid_k && (w ? col < C::n : route >= 0);
    const uint64_t row = w ? uint64_t(expert)*C::weight_rows+(r/Cols)*C::n+col : source;
    const uint64_t at = row*(C::k/32)+k0/32;
    const uint8_t* scales = w ? weight_scales : activation_scales;
    if constexpr (AsyncScale) {
      const uint64_t total = uint64_t(w ? C::experts*C::weight_rows : live*(C::gate_up ? 1 : C::top_k))*(C::k/32);
      copy_scale(&raw.scales[r],scales,at,total,valid);
    } else raw.scales[r] = valid ? *reinterpret_cast<const uint16_t*>(scales+at) : 0x7f7f;
  }
  commit();
}

template<class C, int Rows, int Cols, int Stages, bool AsyncScale, bool Predecoded>
__device__ __forceinline__ void decode(Shared<C,Rows,Cols,Stages,Predecoded>& s,
                                      int slot, int step, int expert, int column) {
  auto& raw = s.raw[slot];
  for (int v = threadIdx.x; v < (C::halves*Cols+Rows)*8; v += blockDim.x) {
    const int r = v/8, pair0 = (v%8)*4;
    const bool w = r < C::halves*Cols;
    // Keep four consecutive words together through the shared transfer.
    // Scalar stores map this lane layout to only eight banks.
    if constexpr (Predecoded) {
      if (!w) {
        const uint4 values = *reinterpret_cast<const uint4*>(raw.input[r-C::halves*Cols]+pair0*4);
        *reinterpret_cast<uint4*>(&s.decoded[r][pair0*2]) = values;
        continue;
      }
    }
    uint32_t values[4];
    uint32_t scales = (Predecoded && !w) ? 0 : raw.scales[r];
    if constexpr (AsyncScale) {
      const int route = w ? 0 : s.routes[r-C::halves*Cols];
      const int source = C::gate_up ? route/C::top_k : route;
      const uint64_t row = w ? uint64_t(expert)*C::weight_rows+(r/Cols)*C::n+column+r%Cols : source;
      scales >>= 8*((row*(C::k/32)+step*2)&3);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int pair = pair0+i;
      const uint8_t scale = uint8_t(scales >> (pair/16*8));
      uint32_t value;
      if (w) {
        const uint8_t code = raw.weights[r][pair];
        // Match the existing admission before the nonchecking exact converter.
        convert::check_mxfp4(code,scale);
        value = convert::scaled_mxfp4_bf16x2<false>(code,scale);
      } else if constexpr (Predecoded) {
        value = *reinterpret_cast<const uint32_t*>(raw.input[r-C::halves*Cols]+pair*4);
      } else {
        const uint16_t code = *reinterpret_cast<const uint16_t*>(raw.input[r-C::halves*Cols]+pair*2);
        if ((code&0x7f)==0x7f || ((code>>8)&0x7f)==0x7f) __trap();
        value = chained::scaled_e4m3x2(code,scale);
      }
      values[i] = value;
    }
    *reinterpret_cast<uint4*>(&s.decoded[r][pair0*2]) = make_uint4(values[0],values[1],values[2],values[3]);
  }
}

// All state and scratch are supplied by the caller. Grid dependency
// operations live only in the entry wrapper, for later task execution.
template<class C, int Rows, int Cols, int Stages, int MaxRows,
         bool AsyncScale, bool Predecoded, bool Dynamic, bool Stamp, bool OmitFirst=false>
__device__ void task(int virtual_cta, int virtual_grid, void* arena,
    const uint8_t* input, const uint8_t* weight, const uint8_t* weight_scales,
    const uint8_t* activation_scales, const int* sorted_rows, const int* padded_rows,
    const int* live_tokens, int* table, const float* route_weights, void* output,
    uint64_t* stamps) {
  constexpr int H=C::halves, WarpRows=Rows > 64 ? 64 : Rows, F=WarpRows/8;
  constexpr int Threads=(Rows > 64 ? 2 : 1)*(Cols/16)*32;
  constexpr int Q=Rows==8 ? 0 : Rows==32 ? 1 : Rows==64 ? 2 : 3;
  constexpr int Steps=C::loop_k/64;
  auto& s=*static_cast<Shared<C,Rows,Cols,Stages,Predecoded>*>(arena);
  const int live=*live_tokens, padded=*padded_rows, tiles=table[Q];
  if (blockDim.x != Threads || live<0 || live>C::tokens_capacity || tiles<0 ||
      tiles>Queue<C,MaxRows>::slots || padded<0 || padded%64 ||
      padded>C::tokens_capacity*C::top_k+C::experts*64) __trap();
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5, g=lane>>2, t=lane&3;
  const int row_base=(warp/(Cols/16))*WarpRows, column_warp=warp%(Cols/16);
  const int columns=(C::n+Cols-1)/Cols, items=tiles*columns;
  uint64_t clocks[7] = {};
  int completed=0;
  for (int iteration=0;;++iteration) {
    uint64_t tick=0;
    if constexpr (Stamp) tick=clock64();
    int item;
    if constexpr (Dynamic) {
      if (threadIdx.x==0) s.item=atomicAdd(table+4+Q,1);
      __syncthreads(); item=s.item;
    } else item=virtual_cta+iteration*virtual_grid;
    if (item>=items) break;
    const int column=(item%columns)*Cols;
    const int at=8+3*(Q*Queue<C,MaxRows>::slots+item/columns);
    const int expert=table[at], first=table[at+1], count=table[at+2];
    if (expert<0 || expert>=C::experts || first<0 || first%64 || count<=0 ||
        count>Rows || count>live || first+count>padded) __trap();
    for (int r=threadIdx.x;r<Rows;r+=Threads) {
      const int route=r<count ? sorted_rows[first+r] : -1;
      if (r<count && (route<0 || route>=live*C::top_k)) __trap();
      s.routes[r]=route;
    }
    __syncthreads();
    ptx::D4 sum[H][F],chain[H][F];
#pragma unroll
    for(int h=0;h<H;++h)
#pragma unroll
      for(int f=0;f<F;++f)
#pragma unroll
        for(int i=0;i<4;++i) { sum[h][f].v[i]=0.f; chain[h][f].v[i]=0.f; }
#pragma unroll
    for(int step=0;step<Stages;++step)
      enqueue<C,Rows,Cols,Stages,AsyncScale,Predecoded>(s,step,step,expert,column,live,input,weight,weight_scales,activation_scales);
    if constexpr (Stamp) { clocks[0]+=clock64()-tick; }
#pragma unroll 1
    for(int step=0;step<Steps;++step) {
      if constexpr (Stamp) tick=clock64();
      wait<Stages-1>();
      __syncthreads();
      if constexpr (Stamp) { clocks[1]+=clock64()-tick; tick=clock64(); }
      decode<C,Rows,Cols,Stages,AsyncScale,Predecoded>(s,step%Stages,step,expert,column);
      __syncthreads();
      if constexpr (Stamp) { clocks[2]+=clock64()-tick; tick=clock64(); }
#pragma unroll
      for(int kk=0;kk<4;++kk) {
        ptx::A4 a[H];
#pragma unroll
        for(int h=0;h<H;++h) {
          const int wr=h*Cols+column_warp*16+g;
          a[h].v[0]=chained::lds32(&s.decoded[wr][kk*16+2*t]);
          a[h].v[1]=chained::lds32(&s.decoded[wr+8][kk*16+2*t]);
          a[h].v[2]=chained::lds32(&s.decoded[wr][kk*16+2*t+8]);
          a[h].v[3]=chained::lds32(&s.decoded[wr+8][kk*16+2*t+8]);
        }
#pragma unroll
        for(int f=0;f<F;++f) {
          if(row_base+f*8<count) {
            ptx::B2 b;
            b.v[0]=chained::lds32(&s.decoded[H*Cols+row_base+f*8+g][kk*16+2*t]);
            b.v[1]=chained::lds32(&s.decoded[H*Cols+row_base+f*8+g][kk*16+2*t+8]);
#pragma unroll
            for(int h=0;h<H;++h) ptx::mma_bf16(chain[h][f],a[h],b);
          }
        }
      }
      if constexpr (Stamp) { clocks[3]+=clock64()-tick; tick=clock64(); }
      if(step%2==1) {
#pragma unroll
        for(int h=0;h<H;++h)
#pragma unroll
          for(int f=0;f<F;++f)
#pragma unroll
            for(int i=0;i<4;++i) {
              if(!OmitFirst || step>1) sum[h][f].v[i]=__fadd_rn(sum[h][f].v[i],chain[h][f].v[i]);
              chain[h][f].v[i]=0.f;
            }
      }
      __syncthreads();
      if constexpr (Stamp) { clocks[4]+=clock64()-tick; tick=clock64(); }
      if(step+Stages<Steps)
        enqueue<C,Rows,Cols,Stages,AsyncScale,Predecoded>(s,step%Stages,step+Stages,expert,column,live,input,weight,weight_scales,activation_scales);
      else commit(); // Empty groups keep wait_group's pending count exact at the tail.
      if constexpr (Stamp) clocks[5]+=clock64()-tick;
    }
    wait<0>();
    if constexpr (Stamp) tick=clock64();
#pragma unroll
    for(int f=0;f<F;++f)
#pragma unroll
      for(int i=0;i<4;++i) {
        const int route=s.routes[row_base+f*8+2*t+(i&1)];
        const int col=column+column_warp*16+g+(i>=2 ? 8 : 0);
        if(route<0 || col>=C::n) continue;
        const uint64_t out=uint64_t(route)*C::n+col;
        if constexpr(C::gate_up) static_cast<uint16_t*>(output)[out]=convert::bf16_from_f32(
          chained::before_down<typename C::clamp>(sum[0][f].v[i],sum[1][f].v[i],route_weights[route]));
        else static_cast<float*>(output)[out]=sum[0][f].v[i];
      }
    __syncthreads();
    if constexpr (Stamp) clocks[6]+=clock64()-tick;
    ++completed;
  }
  if constexpr (Stamp) if(threadIdx.x==0) {
    const int base=(Q*virtual_grid+virtual_cta)*8;
    for(int i=0;i<7;++i) stamps[base+i]=clocks[i];
    stamps[base+7]=completed;
  }
}

template<class C>
__device__ void predecode(const uint8_t* input,const uint8_t* scales,
                         const int* live,uint16_t* output) {
  const int rows=*live*(C::gate_up ? 1 : C::top_k);
  if(*live<0 || *live>C::tokens_capacity) __trap();
  for(uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;p<uint64_t(rows)*C::k/2;p+=uint64_t(gridDim.x)*blockDim.x) {
    const uint16_t code=*reinterpret_cast<const uint16_t*>(input+p*2);
    if((code&0x7f)==0x7f || ((code>>8)&0x7f)==0x7f) __trap();
    reinterpret_cast<uint32_t*>(output)[p]=chained::scaled_e4m3x2(code,scales[p/16]);
  }
}
} // namespace chop::moe::prefill_pipeline_vector
