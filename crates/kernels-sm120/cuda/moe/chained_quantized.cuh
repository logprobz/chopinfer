#pragma once

#include "chained_experts.cuh"

namespace chop::moe::chained::quantized {

// Stored rows keep whole 64-K tiles; only shared tiles carry the tail.
// Inherit the shared engine's geometry with whole 128-K loop chains.
// `Tiles` reads a gate/up bank stored in 64-row by 128-K tiles, `Lanes`
// those tiles in quad-lane order.
template<Projection P, int Experts, int TopK, int N, int StoredK, int RowTile,
         int Capacity, class Clamp, bool Tiles = false, bool Lanes = false>
struct Config : chained::Config<Weights::mxfp4, P, Experts, TopK, N,
                               ((StoredK + 127) / 128) * 128, RowTile, true, Clamp, Tiles,
                               Lanes> {
  static_assert(StoredK > 0 && StoredK % 64 == 0, "whole stored 64-K tiles");
  static_assert(!Tiles || StoredK % 128 == 0, "row tiles hold whole 128-K blocks");
  static_assert(Capacity > 0, "positive token capacity");
  static constexpr int k = StoredK;
  static constexpr int tokens_capacity = Capacity;
  static constexpr bool quantized_before_down = true;
};

} // namespace chop::moe::chained::quantized
