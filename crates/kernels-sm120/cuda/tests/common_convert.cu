#include "../common/convert.cuh"

extern "C" __global__ void common_convert(
    unsigned* out, const float* input) {
  chop::convert::Float2 f{input[0], input[1]};
  uint32_t bf = chop::convert::bf16x2_from_f32(f);
  uint32_t h = bf;
  uint32_t checksum = bf;
#define CHECK(FMT)                                                             \
  {                                                                            \
    uint16_t x = chop::convert::pack_f32_##FMT(f);                             \
    x ^= chop::convert::pack_bf16x2_##FMT(bf);                                 \
    x ^= chop::convert::pack_f16x2_##FMT(h);                                   \
    checksum += chop::convert::unpack_bf16x2_##FMT(x);                         \
    checksum += chop::convert::unpack_f16x2_##FMT(x);                          \
    chop::convert::Float2 y = chop::convert::unpack_f32_##FMT(x);              \
    checksum += __float_as_uint(y.lo + y.hi);                                  \
  }
  CHECK(e4m3)
  CHECK(e5m2)
  CHECK(e2m1)
  CHECK(e2m3)
  CHECK(e3m2)
  CHECK(ue8m0)
#undef CHECK
  checksum += chop::convert::pack_f32_ue8m0_rp(f);
  out[threadIdx.x] = checksum;
}
