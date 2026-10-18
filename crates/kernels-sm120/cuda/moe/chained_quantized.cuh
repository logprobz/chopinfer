#pragma once

#include "chained_experts.cuh"

namespace chop::moe::chained::quantized {

// Stored rows keep whole 64-K tiles; only shared tiles carry the tail.
// Inherit the shared engine's geometry with whole 128-K loop chains.
template<Projection P, int Experts, int TopK, int N, int StoredK, int RowTile,
         int Capacity, class Clamp>
struct Config : chained::Config<Weights::mxfp4, P, Experts, TopK, N,
                               ((StoredK + 127) / 128) * 128, RowTile, true, Clamp> {
  static_assert(StoredK > 0 && StoredK % 64 == 0, "whole stored 64-K tiles");
  static_assert(Capacity > 0, "positive token capacity");
  static constexpr int k = StoredK;
  static constexpr int tokens_capacity = Capacity;
  static constexpr bool quantized_before_down = true;
};

} // namespace chop::moe::chained::quantized
