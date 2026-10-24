#pragma once
#include "chained_absorb.cuh"
#include "chained_strip2.cuh"
#include "scale_tiles.cuh"

namespace chop::moe::encoded_down {
// One or two adjacent output strips, each one 64-row scale tile. The
// caller keeps the original zero-code and scale-127 handling for a partial
// final K block, and waits at a CTA barrier before reading these words.
template<uint32_t Width>
struct Scales {
  static constexpr bool staged = true;
  template<class C, int Strips>
  __device__ __forceinline__ static const uint32_t* prepare(const uint8_t* bank,
      int expert, int column, int block) {
    static_assert(!C::gate_up && C::halves == 1);
    static_assert(Strips == 1 || Strips == 2);
    __shared__ uint32_t words[Strips * 64];
    if (threadIdx.x < Strips * 64) {
      const int strip = threadIdx.x / 64, row = threadIdx.x % 64;
      const auto matrix = scale_tiles::matrix(bank, expert, C::n, C::k / 32);
      const auto tile = scale_tiles::tile<Width>(matrix, column / 64 + strip, block);
      words[threadIdx.x] = scale_tiles::patch_row(tile, row, scale_tiles::row_word<Width>(tile, row));
    }
    return words;
  }
};
} // namespace chop::moe::encoded_down
