#pragma once

// MiMo's MoE combine as a push input policy: `input` holds each row's TopK
// preweighted routed rows [rows][TopK][row width] in BF16, and a row's
// input vector is their sum in F32, in expert order, rounded to BF16 as
// chop_row_combine_mimo4096_v1 rounds it. Rows at or past the call's live
// count load zeros. kStaged: a vector costs TopK loads, so the push
// transport combines it once per call and keeps it for later phases.

#include "../comm.cuh"

namespace chop::comm::input {

template <uint32_t TopK>
struct RoutedCombine {
  using Element = uint16_t;
  static constexpr uint32_t kPerVector = 8;
  static constexpr bool kStaged = true;

  // Input vector `index` (row * row vectors + vector). Eager loads a dead
  // row's routed rows too (they lie inside the prepared rows) and drops the
  // sum afterwards, so the loads need not wait for the live count.
  template <bool Eager, typename E>
  __device__ __forceinline__ static Vector128 load(const push::Call<E>& c, uint64_t index) {
    constexpr uint32_t kRowVectors = push::kRowVectors<E>;
    const Vector128* input = reinterpret_cast<const Vector128*>(c.input);
    const uint64_t row = index / kRowVectors;
    if (!Eager && row >= c.live) return {0, 0, 0, 0};
    float sums[8] = {};
#pragma unroll
    for (uint32_t k = 0; k < TopK; ++k)
      push::accumulate(input[(row * TopK + k) * kRowVectors + index % kRowVectors], sums);
    uint32_t words[4];
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      uint16_t low, high;
      asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(low) : "f"(sums[2 * word]));
      asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(high) : "f"(sums[2 * word + 1]));
      words[word] = static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
    }
    if (Eager && row >= c.live) return {0, 0, 0, 0};
    return {words[0], words[1], words[2], words[3]};
  }
};

} // namespace chop::comm::input
