#include "pooled_indexer/position_mma_guard_fast_bound.h"
#include "pooled_indexer/position_mma_guard_bound.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
using namespace chop::position_mma_guard_fast;
using chop::position_mma_guard::exact_scales;
uint32_t word(std::array<uint32_t,4> bytes) {
  uint32_t result = 0;
  for (int i = 0; i < 4; ++i) result |= bytes[i] << (8*i);
  return result;
}
struct Query { Range range; uint32_t spread; };
Query query(const std::array<uint32_t,32>& heads) {
  Query result{{255,0},0};
  for (auto head: heads) {
    const auto r = scale_range(head);
    result.range.lo = std::min(result.range.lo,r.lo);
    result.range.hi = std::max(result.range.hi,r.hi);
    result.spread = std::max(result.spread,r.hi-r.lo);
  }
  return result;
}
int main() {
  std::mt19937 random(0x189ec2);
  unsigned accepted = 0, rejected = 0;
  for (unsigned sample = 0; sample < 100000; ++sample) {
    std::array<uint32_t,32> heads;
    const uint32_t base = random()%256;
    for (auto& head: heads) {
      std::array<uint32_t,4> bytes;
      for (auto& value: bytes) value = (base+random()%10)%256;
      head = word(bytes);
    }
    std::array<uint32_t,4> key;
    const auto key_base = random()%256;
    for (auto& value: key) value = (key_base+random()%4)%256;
    const auto q = query(heads);
    if (coarse_scales(q.range,q.spread,scale_range(word(key)))) {
      for (auto head: heads) assert(exact_scales(head,word(key)));
      ++accepted;
    } else ++rejected;
  }
  assert(accepted > 1000 && rejected > 1000);
  // Uniform heads may have very different scales. Their within-head spread,
  // not the row's global range, governs the pairwise certificate.
  std::array<uint32_t,32> heads;
  for (unsigned h=0; h<32; ++h) heads[h]=word({100+h,100+h,100+h,100+h});
  auto q=query(heads);
  assert(q.spread==0 && coarse_scales(q.range,q.spread,{127,136}));
  for (auto head: heads) assert(exact_scales(head,word({127,136,129,132})));
  assert(!coarse_scales(q.range,q.spread,{127,137}));
  assert(coarse_scales({65,65},0,{65,65}));
  assert(!coarse_scales({64,65},1,{65,65}));
  assert(coarse_scales({184,184},0,{185,185}));
  assert(!coarse_scales({184,185},1,{185,185}));
  for (uint32_t x: {0,1,253,254,255}) {
    assert(!coarse_scales(scale_range(word({127,x,127,127})),0,{127,127}));
    assert(!coarse_scales({127,127},0,scale_range(word({127,x,127,127}))));
  }
  // Coarse rejection can still pass the fine certificate through cancellation
  // of scale differences. That path must retain MMA rather than forcing repair.
  heads.fill(word({127,139,127,139})); q=query(heads);
  const auto key=word({139,127,139,127});
  assert(!coarse_scales(q.range,q.spread,scale_range(key)));
  for (auto head: heads) assert(exact_scales(head,key));
  // A mutant that summarizes only the first head admits this unsafe pair.
  heads.fill(word({127,127,127,127}));
  heads[31]=word({127,151,127,103});q=query(heads);
  assert(coarse_scales(scale_range(heads[0]),0,{127,127}));
  assert(!coarse_scales(q.range,q.spread,{127,127}));
  assert(!exact_scales(heads[31],word({127,127,127,127})));
  std::cout << "GUARD_FAST_BOUND_COMPLETE accepted=" << accepted
      << " rejected=" << rejected << " negative_control_rejected=1\n";
}
