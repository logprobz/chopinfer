#pragma once
#include "chained_absorb.cuh"

namespace chop::moe::strip2 {
using namespace streamed;
constexpr int Rows = absorb::Rows, Strips = 2;

template<class C, class Scales = RawScales>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  absorb::compute<C, false, Scales, Strips>(input, weight, weight_scales, activation_scales, offsets, sorted_rows, padded_rows, live_tokens, scratch);
}
} // namespace chop::moe::strip2
