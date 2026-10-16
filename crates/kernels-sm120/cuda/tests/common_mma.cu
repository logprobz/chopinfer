#include "../common/ptx.cuh"

extern "C" __global__ void common_mma(float* out, const unsigned* in) {
  chop::ptx::A4 a{{in[0], in[1], in[2], in[3]}};
  chop::ptx::B2 b{{in[4], in[5]}};
  chop::ptx::D4 d{{0, 0, 0, 0}};
  chop::ptx::mma_bf16(d, a, b);
  chop::ptx::mma_f8f6f4<chop::ptx::Element::e4m3,
                        chop::ptx::Element::e2m1>(d, a, b);
  chop::ptx::mma_mxf8f6f4<chop::ptx::Element::e4m3,
                          chop::ptx::Element::e2m1>(d, a, b, in[6], in[7]);
  chop::ptx::mma_mxf4nvf4<2>(d, a, b, in[6], in[7]);
  chop::ptx::mma_mxf4nvf4<4>(d, a, b, in[6], in[7]);
  chop::ptx::mma_mxf4nvf4<4, true>(d, a, b, in[6], in[7]);
  chop::ptx::A2 sparse_a{{in[0], in[1]}};
  chop::ptx::mma_sparse_bf16(d, sparse_a, b, in[8]);
  chop::ptx::mma_mxf4(d, a, b, in[6], in[7]);
  out[threadIdx.x & 3] = d.v[threadIdx.x & 3];
}

extern "C" __global__ void common_mma_all_formats(float* out, const unsigned* in) {
  chop::ptx::A4 a{{in[0], in[1], in[2], in[3]}};
  chop::ptx::B2 b{{in[4], in[5]}};
  chop::ptx::D4 d{{0, 0, 0, 0}};
#define MIX(A, B)                                                              \
  chop::ptx::mma_f8f6f4<chop::ptx::Element::A, chop::ptx::Element::B>(d, a, b); \
  chop::ptx::mma_mxf8f6f4<chop::ptx::Element::A, chop::ptx::Element::B>(        \
      d, a, b, in[6], in[7])
  MIX(e4m3, e4m3);
  MIX(e4m3, e5m2);
  MIX(e4m3, e3m2);
  MIX(e4m3, e2m3);
  MIX(e4m3, e2m1);
  MIX(e5m2, e4m3);
  MIX(e5m2, e5m2);
  MIX(e5m2, e3m2);
  MIX(e5m2, e2m3);
  MIX(e5m2, e2m1);
  MIX(e3m2, e4m3);
  MIX(e3m2, e5m2);
  MIX(e3m2, e3m2);
  MIX(e3m2, e2m3);
  MIX(e3m2, e2m1);
  MIX(e2m3, e4m3);
  MIX(e2m3, e5m2);
  MIX(e2m3, e3m2);
  MIX(e2m3, e2m3);
  MIX(e2m3, e2m1);
  MIX(e2m1, e4m3);
  MIX(e2m1, e5m2);
  MIX(e2m1, e3m2);
  MIX(e2m1, e2m3);
  MIX(e2m1, e2m1);
#undef MIX
  out[threadIdx.x & 3] = d.v[threadIdx.x & 3];
}
