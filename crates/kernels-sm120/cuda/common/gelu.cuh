#pragma once
#include <math.h>

namespace chop {
__device__ __forceinline__ float gelu_erf(float x) {
  // Preserve the ordinary expression near zero and on the positive branch.
  // erfc keeps representable negative tails when F32 erf rounds to -1.
  return x < -1.f ? 0.5f * x * erfcf(-x * 0.7071067811865475244f)
                  : 0.5f * x * (1.f + erff(x * 0.7071067811865475244f));
}
} // namespace chop
