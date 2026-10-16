#pragma once

// The items of a one-item-per-CTA tile kernel over `live_m` rows in 128-row
// tiles by `tiles_b` column blocks, whose final partial wave over `Ctas`
// SMs is split into 64-row halves when they fit the SMs. Tiles are numbered
// row-major (row tile, then column block); when the 64-row halves of the
// live rows are odd, the last row tile is a lone 64-row half.
//
// The first `waves * Ctas` CTAs take the tiles of the complete waves, one
// each, in tile order, as a one-tile grid does. The final partial wave's
// tiles follow: when their halves fit `Ctas` SMs, its full tiles' halves (in
// tile order, each tile's upper half first) and then its lone halves, one
// per CTA; otherwise its tiles whole. CTAs past the items exit. The grid
// (the tiles at the row capacity plus `Ctas`) covers every live count, and
// the hardware hands each SM the next CTA as one finishes, so the halves
// fill the SMs the complete waves free first. A half runs in about two
// thirds of a tile's time, so at 2,048 rows a 1,536-column FP8 projection
// (192 tiles on 188 SMs) runs one wave of tiles and one of halves instead
// of two waves of tiles.
//
// The split depends on the live count only; no element's arithmetic
// depends on which CTA, item or half computes it.

#include <stdint.h>

namespace chop::balanced {

// One item: `rows` (64 or 128) rows from `row` by column block `block`.
struct Item {
  uint32_t row;
  uint32_t rows;
  uint32_t block;
};

template<uint32_t Ctas>
struct Tiles {
  static_assert(Ctas > 0);
  uint32_t tiles_b;
  uint32_t halves_m;   // 64-row halves of the live rows
  uint32_t row_tiles;  // 128-row tiles, the last a lone half if halves_m is odd
  uint64_t whole;      // tiles of the complete waves
  uint64_t tail;       // tiles of the final partial wave
  uint64_t tail_full;  // its full tiles (its lone halves follow them)
  bool split;          // whether its halves fit the SMs

  __device__ __forceinline__ Tiles(uint32_t live_m, uint32_t blocks) {
    tiles_b = blocks;
    halves_m = (live_m + 63) / 64;
    row_tiles = (halves_m + 1) / 2;
    const uint64_t tiles = uint64_t(row_tiles) * tiles_b;
    whole = tiles / Ctas * Ctas;
    tail = tiles - whole;
    // The lone halves are the last row tile's blocks, a suffix of the tiles.
    const uint64_t lone = halves_m % 2 == 1 ? tiles_b : 0;
    const uint64_t tail_lone = lone < tail ? lone : tail;
    tail_full = tail - tail_lone;
    split = tail > 0 && 2 * tail_full + tail_lone <= Ctas;
  }

  // Tile `tile` whole: 128 rows, or the lone half of the last row tile.
  __device__ __forceinline__ Item tile(uint64_t index) const {
    const uint32_t row_tile = uint32_t(index / tiles_b);
    const bool lone = halves_m % 2 == 1 && row_tile + 1 == row_tiles;
    return Item{row_tile * 128, lone ? 64u : 128u, uint32_t(index % tiles_b)};
  }

  // The item of CTA `cta`; false for a CTA past the items.
  __device__ __forceinline__ bool item(uint64_t cta, Item& out) const {
    if (cta < whole) {
      out = tile(cta);
      return true;
    }
    const uint64_t position = cta - whole;
    if (!split) {
      if (position >= tail) return false;
      out = tile(whole + position);
      return true;
    }
    if (position < 2 * tail_full) {
      const Item full = tile(whole + position / 2);
      out = Item{full.row + uint32_t(position % 2) * 64, 64, full.block};
      return true;
    }
    if (position >= tail + tail_full) return false;
    out = tile(whole + tail_full + (position - 2 * tail_full));
    return true;
  }
};

}  // namespace chop::balanced
