#pragma once
#include <stdint.h>
namespace chop::deepseek41 {
__device__ __forceinline__ void acceptance_result(uint32_t* out, const uint32_t* item,
    uint32_t accepted, uint32_t kept, uint32_t flags, uint32_t failure, uint32_t row) {
  out[0]=item[0]; out[1]=item[1]; out[2]=accepted; out[3]=kept;
  out[4]=flags; out[5]=failure; out[6]=row; out[7]=0;
}
__device__ __forceinline__ void acceptance(const uint32_t* items,
    const uint32_t* proposals, const uint32_t* samples, const uint32_t* stops,
    uint32_t* results, uint32_t count, uint32_t vocab, uint32_t sample_capacity,
    uint32_t draft_capacity, uint32_t stop_capacity) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=count) return;
  const uint32_t* item=items+index*10;
  uint32_t* out=results+index*8;
  uint64_t remaining=uint64_t(item[2])|(uint64_t(item[3])<<32);
  uint32_t first=item[4], draft=item[5], k=item[6], offset=item[7], nstop=item[8];
  uint64_t expected=index ? uint64_t(items[(index-1)*10+4])+items[(index-1)*10+6]+1 : 0;
  uint32_t preceding=0;
  for(uint32_t i=0;i<index;++i) preceding+=items[i*10+6]>0;
  bool bad=item[9]!=0 || remaining==0 || k>5 || uint64_t(first)!=expected
      || uint64_t(first)+k+1>sample_capacity || uint64_t(offset)+nstop>stop_capacity
      || (k==0 ? draft!=0xffffffffu : (draft!=preceding || draft>=draft_capacity));
  if(bad) { acceptance_result(out,item,0,0,0,1,0xffffffffu); return; }
  for(uint32_t j=0;j<nstop;++j) {
    if(stops[offset+j]>=vocab || (j && stops[offset+j-1]>=stops[offset+j])) {
      acceptance_result(out,item,0,0,0,1,0xffffffffu); return;
    }
  }
  if(k) for(uint32_t j=0;j<5;++j) {
    if(proposals[draft*5+j]>=vocab) { acceptance_result(out,item,0,0,0,4,0xffffffffu); return; }
  }
  // Validate the whole item, even after a possible rejection. EMPTY payload
  // words are unspecified and are never inspected.
  for(uint32_t r=0;r<=k;++r) {
    const uint32_t* sample=samples+(first+r)*64;
    uint32_t status=sample[1];
    if(status==2) continue;
    if(status!=0 || sample[0]>=vocab || (sample[3]&0x7ff00000u)==0x7ff00000u) {
      acceptance_result(out,item,0,0,0,2,first+r); return;
    }
  }
  uint32_t accepted=0,kept=0,flags=0;
  for(uint32_t r=0;r<=k;++r) {
    const uint32_t* sample=samples+(first+r)*64;
    if(sample[1]==2) { acceptance_result(out,item,0,0,0,3,first+r); return; }
    uint32_t token=sample[0];
    bool matched=r<k && proposals[draft*5+r]==token;
    accepted+=matched; ++kept;
    bool terminal=false;
    for(uint32_t j=0;j<nstop;++j) terminal|=stops[offset+j]==token;
    flags=uint32_t(terminal)|(uint32_t(r<k && !matched)<<1);
    if(terminal || uint64_t(kept)==remaining || !matched) break;
  }
  acceptance_result(out,item,accepted,kept,flags,0,0xffffffffu);
}
} // namespace chop::deepseek41
