#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"
namespace chop::latent {
__device__ void staged_window_keys(const int32_t* positions,const int32_t* row_items,
    const int32_t* chunk_first,const int32_t* chunk_last,const int32_t* item_slots,
    const int32_t* ring_last,int32_t* output,const uint32_t* live_rows,
    uint32_t items,uint32_t sequences,uint32_t window,uint32_t ring_slots,uint32_t staged_slots) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row=blockIdx.x;
  if(row>=*live_rows) return;
  const int32_t item=row_items[row],p=positions[row];
  int32_t first=-1,last=-1,retained=-1;
  if(item>=0 && uint32_t(item)<items) {
    first=chunk_first[item]; last=chunk_last[item];
    const int32_t slot=item_slots[item];
    if(slot>=0 && uint32_t(slot)<sequences) retained=ring_last[slot];
  }
  const int64_t base=int64_t(first)-window+1;
  const int64_t stage_begin=base>0?base:0;
  const int64_t begin=int64_t(p)-window+1;
  const int64_t visible_begin=begin>0?begin:0;
  for(uint32_t i=threadIdx.x;i<window;i+=blockDim.x) {
    const int64_t position=visible_begin+i,offset=position-stage_begin;
    int32_t key=-1;
    if(first>=0 && last>=first && p>=first && p<=last && position<=p && offset>=0 && offset<staged_slots) {
      const bool prefix_live=retained>=0 && position<=retained && position+ring_slots>retained;
      if(position>=first || prefix_live) key=int32_t(offset);
    }
    output[uint64_t(row)*window+i]=key;
  }
}
} // namespace chop::latent
