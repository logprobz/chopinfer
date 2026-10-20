#pragma once
#include <stdint.h>
#include <math.h>
namespace chop::compact_dense_oracle {
// Decode the trained grid independently of the native-down candidate.
__device__ __forceinline__ double power(int exponent) {
  return __longlong_as_double(int64_t(exponent+1023)<<52);
}
__device__ __forceinline__ double a8(uint8_t code,uint8_t scale) {
  const int e=(code>>3)&15,m=code&7;
  const double v=double(e ? 8+m : m)*power(int(scale)-127+(e ? e-10 : -9));
  return code&128 ? -v : v;
}
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
    const double a=a8(input[a0+k],as[(a0+k)/32]);
    const double b=a8(weight[w0+k],ws[(w0+k)/32]);
    const double product=__dmul_rn(a,b);
    sum=__dadd_rn(sum,product);magnitude=__dadd_ru(magnitude,fabs(product));
  }
  output[at]=sum;absolute[at]=magnitude;
}
}
