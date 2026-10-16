#pragma once
// Partial rounding rows (`row::rounding`): an F32 row rounded once to BF16
// and a BF16 row widened to F32, the steps a BF16-partials all-reduce takes
// around an F32 sum. One CTA per row of the planned bucket; rows at or past
// the device live count return.
#include "row.cuh"

namespace chop::row::rounding {

// output[row] = bf16(input[row]), round to nearest even.
template<int Width>
__device__ void round_rows(const float* input, uint16_t* output, const uint32_t* live_rows) {
  begin();
  static_assert(Width % 4 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x * 4; c < Width; c += blockDim.x * 4) {
    const float4 v = reinterpret_cast<const float4*>(input + uint64_t(row) * Width + c)[0];
    uint2 bits;
    bits.x = uint32_t(to_bf(v.x)) | (uint32_t(to_bf(v.y)) << 16);
    bits.y = uint32_t(to_bf(v.z)) | (uint32_t(to_bf(v.w)) << 16);
    reinterpret_cast<uint2*>(output + uint64_t(row) * Width + c)[0] = bits;
  }
}

// output[row] = f32(input[row]), exact.
template<int Width>
__device__ void widen_rows(const uint16_t* input, float* output, const uint32_t* live_rows) {
  begin();
  static_assert(Width % 4 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x * 4; c < Width; c += blockDim.x * 4) {
    const uint2 bits = reinterpret_cast<const uint2*>(input + uint64_t(row) * Width + c)[0];
    const float4 v = make_float4(bf(uint16_t(bits.x)), bf(uint16_t(bits.x >> 16)),
                                 bf(uint16_t(bits.y)), bf(uint16_t(bits.y >> 16)));
    reinterpret_cast<float4*>(output + uint64_t(row) * Width + c)[0] = v;
  }
}

} // namespace chop::row::rounding
