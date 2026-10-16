#pragma once

// BF16 all-reduce input: the element type the comm transports read from
// each rank and exchange, eight per 16-byte vector, decoded to F32 exactly
// before the rank-ordered sum. Every transport takes its input type as a
// template argument (Input): the hyper-connection transports read any
// input of this shape (input/f32.cuh is the other), the push transport
// BF16 inputs only, since it exchanges BF16 carried rows. As a push input
// policy it loads each vector as it is (kStaged false: loading costs
// nothing to repeat); input/routed_combine.cuh computes its vectors.

#include "../comm.cuh"
#include "../../common/convert.cuh"

namespace chop::comm::input {

struct Bf16 {
  using Element = uint16_t;
  static constexpr uint32_t kPerVector = 8;  // elements per 16-byte vector
  static constexpr bool kStaged = false;

  __device__ __forceinline__ static float decode(uint16_t value) {
    return convert::bf16_to_f32(value);
  }
  // Adds the vector's elements to sums[0, kPerVector) in element order.
  __device__ __forceinline__ static void accumulate(Vector128 value, float (&sums)[kPerVector]) {
    const uint32_t words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      sums[2 * word] += decode(static_cast<uint16_t>(words[word]));
      sums[2 * word + 1] += decode(static_cast<uint16_t>(words[word] >> 16));
    }
  }
  // Push input vector `index` (row * row vectors + vector) of call `c`.
  template <bool Eager, typename Call>
  __device__ __forceinline__ static Vector128 load(const Call& c, uint64_t index) {
    return reinterpret_cast<const Vector128*>(c.input)[index];
  }
};

} // namespace chop::comm::input
