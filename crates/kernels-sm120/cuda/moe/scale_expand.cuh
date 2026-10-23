#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"

namespace chop::moe::scale_expand {

__device__ __forceinline__ uint32_t lower_bound(
    const uint32_t* records, uint32_t count, uint32_t index) {
    uint32_t first = 0;
    while (first < count) {
        const uint32_t middle = first + (count - first) / 2;
        if ((records[middle] >> 8) < index) first = middle + 1;
        else count = middle;
    }
    return first;
}

// The host validates bank structure once, before upload. Active ids are
// unique and in [0, experts); a null list selects every expert. Launch with
// grid (ceil(rows / 64), projections, slots), block (256, 1, 1), with PDL.
template<int Width>
__device__ __forceinline__ void run(
    const uint8_t* __restrict__ bank,
    const int32_t* __restrict__ active,
    uint8_t* __restrict__ output,
    uint32_t rows, uint32_t groups, uint32_t projections, uint32_t experts) {
    static_assert(Width == 1 || Width == 2 || Width == 4);
    // Nothing is read or written before the predecessor completes: it may
    // write the active list, and the output's previous readers precede it.
    ptx::griddep_launch_dependents();
    ptx::griddep_wait();
    const uint32_t slot = blockIdx.z;
    uint32_t expert = slot;
    if (active) {
        const int32_t count = active[0];
        if (count <= 0 || uint32_t(count) > experts || slot >= uint32_t(count)) return;
        const int32_t id = active[slot + 1];
        if (id < 0 || uint32_t(id) >= experts) return;
        expert = uint32_t(id);
    }
    const uint32_t matrix = expert * projections + blockIdx.y;
    const auto* table = reinterpret_cast<const uint32_t*>(bank);
    const uint32_t start = table[matrix];
    const uint32_t end = table[matrix + 1];
    const uint8_t* block = bank + start;
    const uint32_t row_bytes = (groups * Width + 7) / 8;
    const uint32_t fixed = (rows + rows * row_bytes + 3) & ~3u;
    const auto* records = reinterpret_cast<const uint32_t*>(block + fixed);
    const uint32_t count = (end - start - fixed) / 4;
    const uint32_t row_first = blockIdx.x * 64;
    const uint32_t row_end = rows < row_first + 64 ? rows : row_first + 64;
    const uint32_t first = row_first * groups;
    const uint32_t limit = row_end * groups;
    // Promote before scaling a matrix id into the raw bank.
    uint8_t* decoded = output + uint64_t(matrix) * rows * groups;
    __shared__ uint32_t exception_first, exception_end;
    if (threadIdx.x == 0) {
        exception_first = lower_bound(records, count, first);
        exception_end = lower_bound(records, count, limit);
    }
    for (uint32_t index = first + threadIdx.x; index < limit; index += blockDim.x) {
        const uint32_t row = index / groups;
        const uint32_t bit = (index % groups) * Width;
        const uint32_t offset = (block[rows + row * row_bytes + bit / 8]
                                 >> (bit % 8)) & ((1u << Width) - 1);
        decoded[index] = uint8_t(block[row] + offset);
    }
    // All base writes in this tile finish before any exception overwrites.
    __syncthreads();
    for (uint32_t record = exception_first + threadIdx.x;
         record < exception_end; record += blockDim.x) {
        const uint32_t value = records[record];
        decoded[value >> 8] = uint8_t(value);
    }
}

// One bank of a paired expansion: its matrices are `rows` x `groups` bytes,
// `projections` per expert, decoded in tiles of `tile_rows` rows. A null
// bank takes no tiles.
struct Bank {
    const uint8_t* encoded;
    uint8_t* output;
    uint32_t rows, groups, projections, width, tile_rows;

    __device__ __forceinline__ uint32_t tiles() const {
        return encoded ? projections * ((rows + tile_rows - 1) / tile_rows) : 0;
    }
};

// Both routed banks of a layer in one launch: grid x covers the first
// bank's tiles, then the second's, for each active-list slot in grid z.
// The width is a runtime value. Each thread writes four bytes at a time
// where the output is word aligned, then the tile's exceptions follow a
// block barrier, as in `run`.
__device__ __forceinline__ void run_pair(
    Bank first_bank, Bank second_bank, const int32_t* __restrict__ active,
    uint32_t experts) {
    ptx::griddep_launch_dependents();
    ptx::griddep_wait();
    const uint32_t slot = blockIdx.z;
    uint32_t expert = slot;
    if (active) {
        const int32_t count = active[0];
        if (count <= 0 || uint32_t(count) > experts || slot >= uint32_t(count)) return;
        const int32_t id = active[slot + 1];
        if (id < 0 || uint32_t(id) >= experts) return;
        expert = uint32_t(id);
    }
    const uint32_t first_tiles = first_bank.tiles();
    const bool second = blockIdx.x >= first_tiles;
    const Bank bank = second ? second_bank : first_bank;
    const uint32_t local = blockIdx.x - (second ? first_tiles : 0);
    if (local >= bank.tiles()) return;
    const uint32_t rows = bank.rows, groups = bank.groups, width = bank.width;
    const uint32_t row_tiles = (rows + bank.tile_rows - 1) / bank.tile_rows;
    const uint32_t matrix = expert * bank.projections + local / row_tiles;
    const auto* table = reinterpret_cast<const uint32_t*>(bank.encoded);
    const uint32_t start = table[matrix];
    const uint32_t end = table[matrix + 1];
    const uint8_t* block = bank.encoded + start;
    const uint32_t row_bytes = (groups * width + 7) / 8;
    const uint32_t fixed = (rows + rows * row_bytes + 3) & ~3u;
    const auto* records = reinterpret_cast<const uint32_t*>(block + fixed);
    const uint32_t count = (end - start - fixed) / 4;
    const uint32_t row_first = local % row_tiles * bank.tile_rows;
    const uint32_t row_end = min(rows, row_first + bank.tile_rows);
    const uint32_t first = row_first * groups;
    const uint32_t limit = row_end * groups;
    uint8_t* decoded = bank.output + uint64_t(matrix) * rows * groups;
    const uint32_t mask = (1u << width) - 1;
    const uint8_t* offsets = block + rows;
    auto scale = [&](uint32_t row, uint32_t column) -> uint32_t {
        const uint32_t bit = column * width;
        return uint8_t(block[row] + ((offsets[row * row_bytes + bit / 8] >> (bit % 8)) & mask));
    };
    __shared__ uint32_t exception_first, exception_end;
    if (threadIdx.x == 0) {
        exception_first = lower_bound(records, count, first);
        exception_end = lower_bound(records, count, limit);
    }
    // Bytes before the first word-aligned output byte, the words, then the
    // tail.
    const uint32_t lead = (4 - uint32_t(reinterpret_cast<uintptr_t>(decoded + first) & 3)) & 3;
    const uint32_t words_first = min(limit, first + lead);
    const uint32_t words = (limit - words_first) / 4;
    const uint32_t tail = words_first + words * 4;
    for (uint32_t index = first + threadIdx.x; index < words_first; index += blockDim.x)
        decoded[index] = uint8_t(scale(index / groups, index % groups));
    for (uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
        const uint32_t index = words_first + word * 4;
        uint32_t row = index / groups, column = index - row * groups, packed = 0;
#pragma unroll
        for (uint32_t byte = 0; byte < 4; ++byte, ++column) {
            if (column == groups) {
                ++row;
                column = 0;
            }
            packed |= scale(row, column) << (8 * byte);
        }
        *reinterpret_cast<uint32_t*>(decoded + index) = packed;
    }
    for (uint32_t index = tail + threadIdx.x; index < limit; index += blockDim.x)
        decoded[index] = uint8_t(scale(index / groups, index % groups));
    __syncthreads();
    for (uint32_t record = exception_first + threadIdx.x;
         record < exception_end; record += blockDim.x) {
        const uint32_t value = records[record];
        decoded[value >> 8] = uint8_t(value);
    }
}

} // namespace chop::moe::scale_expand
