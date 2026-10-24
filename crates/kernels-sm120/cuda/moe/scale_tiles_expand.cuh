#pragma once
#include "scale_expand.cuh"
#include "scale_expand_tiles.cuh"
#include "scale_tiles.cuh"

namespace chop::moe::scale_expand {

// One CTA's row segment. Input tiles can end in partial strips and groups;
// tiled output is admitted only for a first bank of complete 64x4 tiles.
template<int Width, bool Tiled>
__device__ __forceinline__ void expand_tiles(
    Bank bank, uint32_t matrix_index, uint32_t row_first, uint32_t row_end) {
    const auto matrix = scale_tiles::matrix(bank.encoded, matrix_index, bank.rows, bank.groups);
    const uint32_t group_tiles = (bank.groups + 3) / 4;
    const uint32_t words = (row_end - row_first) * group_tiles;
    uint8_t* output = bank.output + uint64_t(matrix_index) * bank.rows * bank.groups;
    for (uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
        const uint32_t row = row_first + word / group_tiles;
        const uint32_t group = word % group_tiles;
        const auto tile = scale_tiles::tile<Width>(matrix, row / 64, group);
        const uint32_t decoded = scale_tiles::patch_row(
            tile, row % 64, scale_tiles::row_word<Width>(tile, row % 64));
        if constexpr (Tiled) {
            *reinterpret_cast<uint32_t*>(output + tiled_scale(row, group * 4, bank.groups)) = decoded;
        } else {
#pragma unroll
            for (uint32_t byte = 0; byte < 4; ++byte)
                if (group * 4 + byte < bank.groups)
                    output[uint64_t(row) * bank.groups + group * 4 + byte] = uint8_t(decoded >> (byte * 8));
        }
    }
}

// Paired expansion from tile-order encoded banks, preserving run_pair's
// nullable-bank ABI, active-list slots and wait before all device reads.
template<bool TiledFirst>
__device__ __forceinline__ void run_tiles_pair(
    Bank first_bank, Bank second_bank, const int32_t* __restrict__ active,
    uint32_t experts) {
    ptx::griddep_launch_dependents();
    ptx::griddep_wait();
    const uint32_t slot = blockIdx.z;
    uint32_t expert = slot;
    if (active) {
        const int32_t count = active[0], id = active[slot + 1];
        if (count <= 0 || uint32_t(count) > experts || slot >= uint32_t(count)) return;
        if (id < 0 || uint32_t(id) >= experts) return;
        expert = uint32_t(id);
    }
    const uint32_t first_tiles = first_bank.tiles();
    const bool second = blockIdx.x >= first_tiles;
    const Bank bank = second ? second_bank : first_bank;
    const uint32_t local = blockIdx.x - (second ? first_tiles : 0);
    if (local >= bank.tiles()) return;
    const uint32_t row_tiles = (bank.rows + bank.tile_rows - 1) / bank.tile_rows;
    const uint32_t matrix = expert * bank.projections + local / row_tiles;
    const uint32_t first = local % row_tiles * bank.tile_rows;
    const uint32_t end = min(bank.rows, first + bank.tile_rows);
#define CHOP_EXPAND_TILES(W) \
    if constexpr (TiledFirst) { \
        if (!second) expand_tiles<W, true>(bank, matrix, first, end); \
        else expand_tiles<W, false>(bank, matrix, first, end); \
    } else expand_tiles<W, false>(bank, matrix, first, end)
    if (bank.width == 1) { CHOP_EXPAND_TILES(1); }
    else if (bank.width == 2) { CHOP_EXPAND_TILES(2); }
    else if (bank.width == 4) { CHOP_EXPAND_TILES(4); }
    else __trap();
#undef CHOP_EXPAND_TILES
}
} // namespace chop::moe::scale_expand
