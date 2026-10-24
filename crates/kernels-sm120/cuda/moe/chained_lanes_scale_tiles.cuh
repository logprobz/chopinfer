#pragma once
#include "chained_lanes.cuh"
#include "scale_tiles.cuh"

// A gate/up scale tile decoded before the queue's existing row barrier.
// The bank table holds gate then up for each expert. Each matrix has N rows
// and K/32 groups; the queue item owns one 64-row, four-group tile per half.
// Each thread writes one of its 128 words. Decoder state dies before code
// loading; the quad lanes then read their words from shared memory. Codes,
// scale validation, MMA order and partial stores stay in lanes::compute.
namespace chop::moe::lanes {
template<uint32_t Width>
struct EncodedScales {
  static constexpr bool staged = true;
  template<class C>
  __device__ __forceinline__ static const uint8_t* prepare(const uint8_t* bank,
      int expert, int column, int block) {
    static_assert(C::gate_up && C::halves == 2);
    __shared__ uint32_t words[128];
    const int half = threadIdx.x / 64, row = threadIdx.x % 64;
    const auto matrix = scale_tiles::matrix(bank, 2 * expert + half, C::n, C::k / 32);
    const auto tile = scale_tiles::tile<Width>(matrix, column / 64, block);
    words[threadIdx.x] = scale_tiles::patch_row(tile, row, scale_tiles::row_word<Width>(tile, row));
    return reinterpret_cast<const uint8_t*>(words);
  }
  template<class C>
  __device__ __forceinline__ static uint32_t load(const uint8_t* words,
      int expert, int half, int column, int block, int row, uint64_t raw_tile) {
    return reinterpret_cast<const uint32_t*>(words)[half * 64 + row];
  }
};
} // namespace chop::moe::lanes
