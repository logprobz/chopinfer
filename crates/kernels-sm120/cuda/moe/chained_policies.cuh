#pragma once
#include "chained_streamed.cuh"
#include "../row/ceil_parallel.cuh"

namespace chop::moe::chained_policy {

// These policies are numerical identities, separate from staging knobs.
struct ExactE4M3Activation {
  __device__ __forceinline__ static uint32_t decode(uint16_t codes, uint8_t scale) {
    return chained::scaled_e4m3x2(codes, scale);
  }
};

struct ExactMxFp4Scale {
  __device__ __forceinline__ static uint32_t decode(uint8_t codes, uint8_t scale) {
    return convert::scaled_mxfp4_bf16x2<false>(codes, scale);
  }
};

// The caller starts each 128-K chain at positive zero and issues eight
// ascending K16 instructions. No staging knob changes those boundaries.
struct Bf16Chain128 {
  __device__ __forceinline__ static void mma(ptx::D4& d, const ptx::A4& a, const ptx::B2& b) {
    ptx::mma_bf16(d, a, b);
  }
};

struct OrderedF32 {
  __device__ __forceinline__ static float add(float a, float b) {
    return __fadd_rn(a, b);
  }
};

struct BeforeQuantizeRouteWeight {
  template<class Clamp>
  __device__ __forceinline__ static float apply(float gate, float up, float weight) {
    // Clamp comparisons, SiLU and both separate products stay in the
    // established arithmetic helper, shared with existing consumers.
    return chained::before_down<Clamp>(gate, up, weight);
  }
};

struct Bf16IntermediateF32Output {
  __device__ __forceinline__ static uint16_t intermediate(float value) {
    return convert::bf16_from_f32(value);
  }
  __device__ __forceinline__ static float output(float value) { return value; }
};

template<uint32_t FloorBits>
struct CeilE4M3Group32 {
  __device__ __forceinline__ static auto quantize(const uint16_t* values, int at) {
    return row::block_quantize_parallel::quantize_group<0,false,false,FloorBits>(values, at);
  }
};

} // namespace chop::moe::chained_policy
