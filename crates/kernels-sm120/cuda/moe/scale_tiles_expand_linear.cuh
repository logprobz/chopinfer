#pragma once
#include "scale_tiles_expand.cuh"

namespace chop::moe::scale_expand {

// Complete 64x4 tiles have a contiguous packed offset plane. Write every
// base word once, then distribute the tile-local exception records after
// the block barrier. No output row scans another row's exceptions.
template<int Width>
__device__ __forceinline__ void expand_tiles_linear(
    Bank bank, uint32_t matrix_index, uint32_t row_first, uint32_t row_end) {
    const uint8_t* block = scale_tiles::matrix(
        bank.encoded, matrix_index, bank.rows, bank.groups).block;
    const uint32_t strips = bank.rows / 64, tiles = bank.groups / 4;
    const auto* strip_table = reinterpret_cast<const uint32_t*>(
        block + bank.rows + uint64_t(bank.rows) * bank.groups * Width / 8);
    const auto* tile_table = reinterpret_cast<const uint16_t*>(strip_table + strips + 1);
    const auto* records = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(tile_table) + ((strips * tiles * 2 + 3) & ~3u));
    uint8_t* matrix_output = bank.output + uint64_t(matrix_index) * bank.rows * bank.groups;
    for (uint32_t first_row = row_first; first_row < row_end; first_row += 64) {
        const uint32_t strip = first_row / 64;
        const uint8_t* offsets = block + bank.rows + uint64_t(first_row) * bank.groups * Width / 8;
        uint8_t* output = matrix_output + uint64_t(first_row) * bank.groups;
        for (uint32_t word = threadIdx.x; word < tiles * 64; word += blockDim.x) {
            uint32_t packed;
            if constexpr (Width == 1) packed = offsets[word / 2] >> ((word % 2) * 4);
            else if constexpr (Width == 2) packed = offsets[word];
            else packed = reinterpret_cast<const uint16_t*>(offsets)[word];
            const uint32_t base = block[first_row + word % 64];
            uint32_t decoded = 0;
#pragma unroll
            for (uint32_t group = 0; group < 4; ++group)
                decoded |= uint32_t(uint8_t(base + ((packed >> (group * Width)) & ((1u << Width) - 1)))) << (group * 8);
            reinterpret_cast<uint32_t*>(output)[word] = decoded;
        }
        __syncthreads();
        for (uint32_t tile = threadIdx.x / 32; tile < tiles; tile += blockDim.x / 32) {
            const uint32_t start = strip_table[strip];
            const uint32_t first = start + tile_table[strip * tiles + tile];
            const uint32_t last = tile + 1 == tiles ? strip_table[strip + 1]
                : start + tile_table[strip * tiles + tile + 1];
            for (uint32_t index = first + (threadIdx.x & 31); index < last; index += 32) {
                const uint32_t record = records[index];
                output[tile * 256 + (record >> 8)] = uint8_t(record);
            }
        }
    }
}

// Preserve the paired ABI, active slots and dependency ordering. The row
// output of the second bank retains the qualified tile-input decoder.
__device__ __forceinline__ void run_tiles_pair_linear(
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
    if (bank.width == 1) expand_tiles_linear<1>(bank, matrix, first, end);
    else if (bank.width == 2) expand_tiles_linear<2>(bank, matrix, first, end);
    else if (bank.width == 4) expand_tiles_linear<4>(bank, matrix, first, end);
    else __trap();
}

} // namespace chop::moe::scale_expand
