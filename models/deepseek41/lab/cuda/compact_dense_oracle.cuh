#pragma once
#include "native_down_oracle.cuh"
namespace chop::compact_dense_oracle {
template<int N,int K,int Groups,int RowStride,int GroupStride>
__device__ void run(const uint8_t* input,const uint8_t* as,const uint8_t* weight,const uint8_t* ws,
    const uint32_t* rows,double* output,double* absolute,uint32_t count) {
  const uint32_t at=blockIdx.x*blockDim.x+threadIdx.x;
  if(at>=count*Groups*N) return;
  const uint32_t row=rows[at/(Groups*N)],group=(at/N)%Groups,column=at%N;
  const uint64_t a0=uint64_t(row)*RowStride+uint64_t(group)*GroupStride;
  const uint64_t w0=(uint64_t(group)*N+column)*K;
  double sum=0.,magnitude=0.;
  for(int k=0;k<K;++k) {
    const double a=native_down_oracle::a8(input[a0+k],as[(a0+k)/32]);
    const double b=native_down_oracle::a8(weight[w0+k],ws[(w0+k)/32]);
    const double product=__dmul_rn(a,b);
    sum=__dadd_rn(sum,product);magnitude=__dadd_ru(magnitude,fabs(product));
  }
  output[at]=sum;absolute[at]=magnitude;
}
}
