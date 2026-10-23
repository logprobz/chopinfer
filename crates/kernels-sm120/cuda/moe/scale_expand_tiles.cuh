#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"
#include "scale_expand.cuh"

namespace chop::moe::scale_expand {

// Where the scale of `row`, `column` of a raw matrix lands in its
// `row-tiles-64x128-groups` scale section: each 64-row strip stored as its
// tiles of 4 groups in group order, every tile row-major
// (`kernel_core::transform::RowTileLayout`, 4-byte tiles).
__device__ __forceinline__ uint32_t tiled_scale(
    uint32_t row, uint32_t column, uint32_t groups) {
    return (row / 64) * 64 * groups + (column / 4) * 256 + (row % 64) * 4 + column % 4;
}

// `run_pair` with the first bank's raw matrices written as row tiles
// (`tiled_scale`); the second bank's tiles run `run_pair` unchanged. The
// host requires a first bank of whole tiles (rows a multiple of 64, groups
// a multiple of 4), and both its outputs are 4-byte aligned. A CTA's rows
// are then whole strips, its output bytes are the same contiguous range as
// in rows, and every output word holds four groups of one row: consecutive
// threads write consecutive words, reading one row each.
__device__ __forceinline__ void run_pair_tiles(
    Bank first_bank, Bank second_bank, const int32_t* __restrict__ active,
    uint32_t experts) {
    if (blockIdx.x >= first_bank.tiles()) {
        run_pair(first_bank, second_bank, active, experts);
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
    const uint32_t rows = bank.rows, groups = bank.groups, width = bank.width;
    const uint32_t row_tiles = (rows + bank.tile_rows - 1) / bank.tile_rows;
    const uint32_t matrix = expert * bank.projections + blockIdx.x / row_tiles;
    const auto* table = reinterpret_cast<const uint32_t*>(bank.encoded);
    const uint32_t start = table[matrix];
    const uint32_t end = table[matrix + 1];
    const uint8_t* block = bank.encoded + start;
    const uint32_t row_bytes = (groups * width + 7) / 8;
    const uint32_t fixed = (rows + rows * row_bytes + 3) & ~3u;
    const auto* records = reinterpret_cast<const uint32_t*>(block + fixed);
    const uint32_t count = (end - start - fixed) / 4;
    const uint32_t row_first = blockIdx.x % row_tiles * bank.tile_rows;
    const uint32_t row_end = min(rows, row_first + bank.tile_rows);
    const uint32_t first = row_first * groups;
    const uint32_t limit = row_end * groups;
    uint8_t* decoded = bank.output + uint64_t(matrix) * rows * groups;
    const uint32_t mask = (1u << width) - 1;
    const uint8_t* offsets = block + rows;
    __shared__ uint32_t exception_first, exception_end;
    if (threadIdx.x < 32) {
        const uint2 range = warp_exception_range(records, count, first, limit);
        if (threadIdx.x == 0) {
            exception_first = range.x;
            exception_end = range.y;
        }
    }
    const uint32_t strip_bytes = 64 * groups;
    const uint32_t words = (limit - first) / 4;
    for (uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
        const uint32_t at = word * 4;
        const uint32_t strip = at / strip_bytes, within = at - strip * strip_bytes;
        const uint32_t row = row_first + strip * 64 + within % 256 / 4;
        uint32_t column = within / 256 * 4, packed = 0;
#pragma unroll
        for (uint32_t byte = 0; byte < 4; ++byte, ++column) {
            const uint32_t bit = column * width;
            const uint32_t offset = (offsets[row * row_bytes + bit / 8] >> (bit % 8)) & mask;
            packed |= uint32_t(uint8_t(block[row] + offset)) << (8 * byte);
        }
        *reinterpret_cast<uint32_t*>(decoded + first + at) = packed;
    }
    // All base writes in this tile finish before any exception overwrites.
    __syncthreads();
    for (uint32_t record = exception_first + threadIdx.x;
         record < exception_end; record += blockDim.x) {
        const uint32_t value = records[record], index = value >> 8;
        const uint32_t row = index / groups;
        decoded[tiled_scale(row, index - row * groups, groups)] = uint8_t(value);
    }
}

} // namespace chop::moe::scale_expand
