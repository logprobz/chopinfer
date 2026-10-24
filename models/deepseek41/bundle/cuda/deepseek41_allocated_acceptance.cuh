#pragma once
#include "deepseek41_acceptance.cuh"
namespace chop::deepseek41 {
// The reducer's walk over an allocated step, in the host result encoding
// (contract deepseek41.dspark-allocated-acceptance.u32.v1). Each descriptor
// keeps its reserved drafts k, which place its sample rows; the allocation's
// verified drafts v <= k bound the walk. Every reserved row is validated. A
// publishable result (OK or EMPTY) sets ALLOCATED (4) and carries v in its
// last word; a failed allocation fails every descriptor.
__device__ __forceinline__ void allocated_acceptance(const uint32_t* items,
    const uint32_t* verified, const uint32_t* status, const uint32_t* proposals,
    const uint32_t* samples, const uint32_t* stops, uint32_t* results, uint32_t count,
    uint32_t vocab, uint32_t sample_capacity, uint32_t draft_capacity,
    uint32_t stop_capacity) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=count) return;
  const uint32_t* item=items+index*10;
  uint32_t* out=results+index*8;
  uint64_t remaining=uint64_t(item[2])|(uint64_t(item[3])<<32);
  uint32_t first=item[4], draft=item[5], k=item[6], offset=item[7], nstop=item[8];
  uint32_t v=verified[index];
  uint64_t expected=index ? uint64_t(items[(index-1)*10+4])+items[(index-1)*10+6]+1 : 0;
  uint32_t preceding=0;
  for(uint32_t i=0;i<index;++i) preceding+=items[i*10+6]>0;
  bool bad=status[0]!=0 || v>k || item[9]!=0 || remaining==0 || k>5
      || uint64_t(first)!=expected || uint64_t(first)+k+1>sample_capacity
      || uint64_t(offset)+nstop>stop_capacity
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
  for(uint32_t r=0;r<=k;++r) {
    const uint32_t* sample=samples+(first+r)*64;
    uint32_t state=sample[1];
    if(state==2) continue;
    if(state!=0 || sample[0]>=vocab || (sample[3]&0x7ff00000u)==0x7ff00000u) {
      acceptance_result(out,item,0,0,0,3,first+r); return;
    }
  }
  uint32_t accepted=0,kept=0,flags=4;
  for(uint32_t r=0;r<=v;++r) {
    const uint32_t* sample=samples+(first+r)*64;
    if(sample[1]==2) { acceptance_result(out,item,0,0,4,2,r); out[7]=v; return; }
    uint32_t token=sample[0];
    bool matched=r<v && proposals[draft*5+r]==token;
    accepted+=matched; ++kept;
    bool terminal=false;
    for(uint32_t j=0;j<nstop;++j) terminal|=stops[offset+j]==token;
    flags=4u|uint32_t(terminal)|(uint32_t(r<v && !matched)<<1);
    if(terminal || uint64_t(kept)==remaining || !matched) break;
  }
  acceptance_result(out,item,accepted,kept,flags,0,0xffffffffu);
  out[7]=v;
}
} // namespace chop::deepseek41
