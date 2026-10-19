#include "pooled_indexer/position_mma_guard_bound.h"
#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using chop::position_mma_guard::exact_scales;
uint32_t word(std::array<int,4> a) {
  uint32_t result=0;
  for(int i=0;i<4;++i) result |= uint32_t(a[i]) << (8*i);
  return result;
}
uint32_t bits(float f) { uint32_t result; std::memcpy(&result,&f,4); return result; }
float simt(const std::array<float,128>& q,const std::array<float,128>& k) {
  std::array<float,32> lanes{};
  for(int l=0;l<32;++l) for(int g=0;g<4;++g)
    lanes[l]=std::fma(q[g*32+l],k[g*32+l],lanes[l]);
  for(int offset=16;offset;offset/=2)
    for(int l=0;l<offset;++l) lanes[l]+=lanes[l+offset];
  return 0.f+lanes[0];
}
int main() {
  const uint32_t center=word({127,127,127,127});
  assert(exact_scales(center,center));
  for(int code: {0,1,253,254,255}) for(int group=0;group<4;++group) {
    auto a=std::array<int,4>{127,127,127,127};a[group]=code;
    assert(!exact_scales(word(a),center));assert(!exact_scales(center,word(a)));
  }
  assert(exact_scales(word({65,65,65,65}),word({65,65,65,65})));
  assert(!exact_scales(word({64,65,65,65}),word({65,65,65,65})));
  assert(exact_scales(word({184,184,184,184}),word({185,185,185,185})));
  assert(!exact_scales(word({184,184,184,185}),word({185,185,185,185})));
  assert(exact_scales(center,word({127,136,129,132})));
  assert(!exact_scales(center,word({127,137,129,132})));
  std::mt19937 rng(0x4b1705);
  const int code[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
  unsigned accepted=0;
  for(int sample=0;sample<20000;++sample) {
    std::array<int,4> qs,ks;
    const int base=130+int(rng()%231);
    for(int g=0;g<4;++g) {
      const int sum=base+int(rng()%10);
      const int lo=std::max(2,sum-252), hi=std::min(252,sum-2);
      qs[g]=lo+int(rng()%unsigned(hi-lo+1)); ks[g]=sum-qs[g];
    }
    assert(exact_scales(word(qs),word(ks)));
    std::array<float,128> q,k;std::array<double,128> products;
    double oracle=0;
    for(int i=0;i<128;++i) {
      q[i]=std::ldexp(float(code[rng()%16]),qs[i/32]-128);
      k[i]=std::ldexp(float(code[rng()%16]),ks[i/32]-128);
      products[i]=double(q[i])*double(k[i]);oracle+=products[i];
    }
    assert(std::isfinite(float(oracle)));
    // Normalize zero: ReLU in the scorer discards the dot's zero sign.
    auto score_bits=[](float x){return bits(x==0.f?0.f:x);};
    assert(score_bits(simt(q,k))==score_bits(float(oracle)));
    std::shuffle(products.begin(),products.end(),rng);
    float reordered=0;
    for(double product:products) reordered+=float(product);
    assert(score_bits(reordered)==score_bits(float(oracle)));
    ++accepted;
  }
  std::array<float,128> q{},k{};q.fill(1.f);
  k[0]=0x1p24f;k[32]=1.f;k[64]=-0x1p24f;k[96]=1.f;
  assert(simt(q,k)==1.f);
  const float reordered=(k[0]+k[64])+(k[32]+k[96]);
  assert(reordered==2.f && bits(reordered)!=bits(simt(q,k)));
  assert(!exact_scales(word({125,125,125,125}),word({152,128,152,128})));
  std::cout << "GUARD_BOUND_COMPLETE accepted=" << accepted << " negative_control_rejected=1\n";
}
