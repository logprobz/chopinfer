#pragma once
#include "deepseek41_acceptance.cuh"
namespace chop::deepseek41 {
// Keep the proven reducer's counts and translate its private failure encoding
// to the optional host ABI, whose EMPTY row is relative to the item.
__device__ __forceinline__ void host_acceptance(const uint32_t* items,
    const uint32_t* proposals, const uint32_t* samples, const uint32_t* stops,
    uint32_t* results, uint32_t count, uint32_t vocab, uint32_t sample_capacity,
    uint32_t draft_capacity, uint32_t stop_capacity) {
  acceptance(items,proposals,samples,stops,results,count,vocab,sample_capacity,
      draft_capacity,stop_capacity);
  uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=count) return;
  uint32_t* out=results+index*8;
  if(out[5]==2) out[5]=3;
  else if(out[5]==3) { out[5]=2; out[6]-=items[index*10+4]; }
}
} // namespace chop::deepseek41
