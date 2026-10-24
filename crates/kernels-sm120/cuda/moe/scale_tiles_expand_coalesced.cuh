#pragma once
#include "scale_tiles_expand.cuh"

namespace chop::moe::scale_expand {

// A tiled gate bank assigns consecutive output words to consecutive
// threads. Each warp then reads one tile's neighboring rows and writes
// contiguous words. The admitted first bank has complete 64x4 tiles.
template<int Width>
__device__ __forceinline__ void expand_tiles_coalesced(
    Bank bank, uint32_t matrix_index, uint32_t row_first, uint32_t row_end) {
    const auto matrix = scale_tiles::matrix(bank.encoded, matrix_index, bank.rows, bank.groups);
    const uint32_t strip_bytes = 64 * bank.groups;
    const uint32_t words = (row_end - row_first) * bank.groups / 4;
    uint8_t* output = bank.output + uint64_t(matrix_index) * bank.rows * bank.groups;
    for (uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
        const uint32_t at = word * 4;
        const uint32_t row = row_first + at / strip_bytes * 64 + at % 256 / 4;
        const uint32_t group = at % strip_bytes / 256;
        const auto tile = scale_tiles::tile<Width>(matrix, row / 64, group);
        const uint32_t decoded = scale_tiles::patch_row(
            tile, row % 64, scale_tiles::row_word<Width>(tile, row % 64));
        *reinterpret_cast<uint32_t*>(output + uint64_t(row_first) * bank.groups + at) = decoded;
    }
}

// Preserve the paired ABI, active slots and dependency ordering. The row
// output of the second bank retains the qualified tile-input decoder.
__device__ __forceinline__ void run_tiles_pair_coalesced(
    Bank first_bank, Bank second_bank, const int32_t* __restrict__ active,
    uint32_t experts) {
    if (blockIdx.x >= first_bank.tiles()) {
        run_tiles_pair<true>(first_bank, second_bank, active, experts);
        return;
    }
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
    const Bank bank = first_bank;
    const uint32_t row_tiles = (bank.rows + bank.tile_rows - 1) / bank.tile_rows;
    const uint32_t matrix = expert * bank.projections + blockIdx.x / row_tiles;
    const uint32_t first = blockIdx.x % row_tiles * bank.tile_rows;
    const uint32_t end = min(bank.rows, first + bank.tile_rows);
    if (bank.width == 1) expand_tiles_coalesced<1>(bank, matrix, first, end);
    else if (bank.width == 2) expand_tiles_coalesced<2>(bank, matrix, first, end);
    else if (bank.width == 4) expand_tiles_coalesced<4>(bank, matrix, first, end);
    else __trap();
}

} // namespace chop::moe::scale_expand
