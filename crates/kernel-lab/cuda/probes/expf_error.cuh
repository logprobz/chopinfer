#pragma once
#include <math.h>
#include <stdint.h>
#include "../../../kernels-sm120/cuda/row/silu_mul_quantize.cuh"

// The error of the quantized SwiGLU's exponential (row/silu_mul_quantize.cuh's
// exponential(g) = expf(-g), built with the op's flags) against exp in F64,
// over gates g whose bits run from start for count values.
// tests/silu_mul_quantize.rs launches it over every finite F32 gate up to
// the clamp: every value the op's expf can see.
//
// Per gate, t = exponential(g), e = exp(-(double)g) and r = e rounded to
// F32. The distance is the number of F32 steps between t and r (both are
// non-negative; the infinity is one step past the largest finite value). The
// error is |t - e| in ulps of e's binade, rounded up to F32, for finite t
// and e at most the largest finite F32. Class 0 holds every gate, class 1
// the gates with |g| at most the clamp. Per class, results holds SLOTS
// words: the distance histogram (the last bucket counts that distance or
// more, and NaN), then the largest error and its gate packed as
// (error bits << 32) | gate bits. Word 2 * SLOTS counts the flagged gates
// (flag_steps or more steps, NaN, or an infinity on one side only), and
// flagged receives the first flag_capacity of
// them as (gate bits << 32) | t bits, in no order, for the host to check
// against its own oracle. values, when not null, receives t per gate.

namespace chop::probe::expf_error {

constexpr int DISTANCES = 8;
constexpr int SLOTS = DISTANCES + 1;

__device__ void measure(uint32_t start, uint64_t count, uint32_t clamp_bits,
                        uint32_t flag_steps, unsigned long long* results,
                        unsigned long long* flagged, uint32_t flag_capacity, float* values) {
  uint32_t every[DISTANCES] = {};
  uint32_t clamped[DISTANCES] = {};
  unsigned long long worst[2] = {0, 0};
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
    const uint32_t bits = start + uint32_t(i);
    const float g = __uint_as_float(bits);
    const float t = chop::row::silu_mul_quantize::exponential(g);
    if (values) values[i] = t;
    const double e = exp(-double(g));
    const float r = __double2float_rn(e);
    const uint32_t a = __float_as_uint(t), b = __float_as_uint(r);
    const uint32_t steps = a > b ? a - b : b - a;
    const int bucket = isnan(t) || steps >= DISTANCES - 1 ? DISTANCES - 1 : int(steps);
    const bool inside = (bits & 0x7fffffffu) <= clamp_bits;
    if (isnan(t) || steps >= flag_steps || (a != b && (isinf(t) || isinf(r)))) {
      const unsigned long long at = atomicAdd(&results[2 * SLOTS], 1ull);
      if (at < flag_capacity)
        flagged[at] = (static_cast<unsigned long long>(bits) << 32) | a;
    }
#pragma unroll
    for (int d = 0; d < DISTANCES; ++d) {
      every[d] += bucket == d;
      clamped[d] += inside && bucket == d;
    }
    if (isfinite(t) && e <= double(__uint_as_float(0x7f7fffffu))) {
      int exponent;
      frexp(e, &exponent);
      const double ulp = ldexp(1.0, exponent - 24);
      const float error = __double2float_ru(fabs(double(t) - e) / ulp);
      const unsigned long long packed =
          (static_cast<unsigned long long>(__float_as_uint(error)) << 32) | bits;
      worst[0] = packed > worst[0] ? packed : worst[0];
      if (inside) worst[1] = packed > worst[1] ? packed : worst[1];
    }
  }
#pragma unroll
  for (int d = 0; d < DISTANCES; ++d) {
    if (every[d]) atomicAdd(&results[d], static_cast<unsigned long long>(every[d]));
    if (clamped[d]) atomicAdd(&results[SLOTS + d], static_cast<unsigned long long>(clamped[d]));
  }
  atomicMax(&results[DISTANCES], worst[0]);
  atomicMax(&results[SLOTS + DISTANCES], worst[1]);
}

} // namespace chop::probe::expf_error
