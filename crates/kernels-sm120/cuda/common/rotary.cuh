#pragma once

// Rotary position arithmetic shared by the row engine's RoPE with KV write
// and the decode attention kernels that fold it in. The contractions are
// explicit so that every caller rounds alike.

#include <math.h>
#include <stdint.h>
#include "convert.cuh"

namespace chop::rotary {

// Torch CPU F32 pow/reciprocal bits for MiMo text RoPE.
__device__ __constant__ uint32_t hf_inverse_bits[2][32] = {
  {0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu, 0x3ea1e89bu, 0x3e72d423u, 0x3e361887u, 0x3e088d77u, 0x3dcccccdu, 0x3d99940du, 0x3d6655c2u, 0x3d2cba15u, 0x3d0186e3u, 0x3cc2434fu, 0x3c91ad39u, 0x3c5a7bf2u, 0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u, 0x3b4f3e38u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu, 0x3a83126fu, 0x3a44948cu, 0x3a136a16u, 0x39dd1725u, 0x39a5cb60u, 0x3978a815u, 0x393a7753u, 0x390bd472u},
  {0x3f800000u, 0x3f1ab32bu, 0x3ebaf81bu, 0x3e61f835u, 0x3e088d77u, 0x3da50956u, 0x3d47763fu, 0x3cf11177u, 0x3c91ad39u, 0x3c301052u, 0x3bd4ca15u, 0x3b80967du, 0x3b1b690du, 0x3abbd3edu, 0x3a6301e2u, 0x3a092e02u, 0x39a5cb60u, 0x394860c1u, 0x38f22ce2u, 0x3892587eu, 0x3830df52u, 0x37d5c441u, 0x37812dabu, 0x371c1fc4u, 0x36bcb0c1u, 0x36640cc6u, 0x3609cf4au, 0x35a68e4cu, 0x35494c57u, 0x34f3499du, 0x3493048eu, 0x3431af44u},
};

// Cosine and sine of pair `pair` of a `Rotary`-wide rotation at `position`.
template<int Rotary>
__device__ __forceinline__ float2 generic_angle(int pair, int64_t position, float theta) {
  const float angle = float(position) * exp2f(-2.f * pair / Rotary * log2f(theta));
  const double turns = nearbyint(double(angle) * 0.15915494309189533577);
  const float reduced = float(__fma_rn(-turns, 6.2831853071795864769, double(angle)));
  return make_float2(__cosf(reduced), __sinf(reduced));
}

// MiMo text entries share these exact bits across row, DFlash and folded decode.
template<int Rotary>
__device__ __forceinline__ float2 angle(int pair, int64_t position, float theta) {
  if constexpr (Rotary == 64) {
    if (theta == 10000.f || theta == 10000000.f) {
      const float frequency = __uint_as_float(hf_inverse_bits[theta == 10000.f ? 0 : 1][pair]);
      const float rounded_angle = __fmul_rn(float(position), frequency);
      // Full-range double trig of the F32 angle, then the reference F32 cast.
      double s, c;
      sincos(double(rounded_angle), &s, &c);
      return make_float2(convert::bf16_to_f32(convert::bf16_from_f32(float(c))),
                         convert::bf16_to_f32(convert::bf16_from_f32(float(s))));
    }
  }
  return generic_angle<Rotary>(pair, position, theta);
}

// `x` rotated against its pair mate `y`: x c - y s for the first element of
// the pair, x c + y s for the second.
__device__ __forceinline__ float rotate(float x, float y, float2 cos_sin, bool second) {
  const float product = __fmul_rn(y, cos_sin.y);
  return __fmaf_rn(x, cos_sin.x, second ? product : -product);
}

// Compatibility helper for MiMo callers: the qualified rotation keeps F32
// products and FMA, with only the output store converting to BF16 or FP8.
__device__ __forceinline__ float hf_rotate(float x, float y, float2 cs, bool second) {
  return rotate(x, y, cs, second);
}

} // namespace chop::rotary
