#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"

// scale-row-tiles-offset-v1, validated by the host before upload: 64-row
// strips, 4-group tiles, widths 1/2/4, at most 1024 groups. The bank is
// 16-byte aligned; its u32 offsets locate 4-byte-aligned matrix blocks.
// Bases and each tile's packed offset region are padded to 4 bytes, then
// come u32 strip starts, u16 strip-relative tile starts (padded to 4), and
// sorted u16 records (local << 8) | value, local = row * 4 + group.
// matrix reads one bank offset; tile reads strip and tile table bounds;
// row_word reads one base and the row's packed offsets (including padding
// in aligned loads); patch_row and patch_row_warp read only that tile's
// records. Indices and local rows must be in bounds, and Width must match
// the encoded block. No helper reads outside the selected matrix block.
namespace chop::moe::scale_tiles {

struct Matrix {
    const uint8_t* block;
    uint32_t rows, groups;
};

__device__ __forceinline__ Matrix matrix(
    const uint8_t* bank, uint32_t index, uint32_t rows, uint32_t groups) {
    const auto* table = reinterpret_cast<const uint32_t*>(bank);
    return {bank + table[index], rows, groups};
}

struct Tile {
    const uint8_t* bases;
    const uint8_t* offsets;
    const uint16_t* records;
    uint32_t rows, groups, first, last;
};

template <uint32_t Width>
__device__ __forceinline__ uint32_t tile_bytes(uint32_t rows, uint32_t groups) {
    static_assert(Width == 1 || Width == 2 || Width == 4);
    return ((rows * groups * Width + 7) / 8 + 3) & ~3u;
}

template <uint32_t Width>
__device__ __forceinline__ Tile tile(const Matrix& m, uint32_t strip, uint32_t k) {
    const uint32_t strips = (m.rows + 63) / 64, tiles = (m.groups + 3) / 4;
    const uint32_t rows = min(64u, m.rows - strip * 64);
    const uint32_t groups = min(4u, m.groups - k * 4);
    const uint32_t plane = (m.rows + 3) & ~3u;
    const uint32_t strip_bytes = 64 * m.groups * Width / 8;
    const uint32_t tail_bytes = m.groups / 4 * tile_bytes<Width>(m.rows % 64, 4)
        + tile_bytes<Width>(m.rows % 64, m.groups % 4);
    const auto* strip_table = reinterpret_cast<const uint32_t*>(
        m.block + plane + m.rows / 64 * strip_bytes + tail_bytes);
    const auto* tile_table = reinterpret_cast<const uint16_t*>(strip_table + strips + 1);
    const auto* records = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(tile_table) + ((strips * tiles * 2 + 3) & ~3u));
    const uint32_t first = strip_table[strip] + tile_table[strip * tiles + k];
    const uint32_t last = k + 1 == tiles ? strip_table[strip + 1]
        : strip_table[strip] + tile_table[strip * tiles + k + 1];
    return {m.block + strip * 64,
            m.block + plane + strip * strip_bytes + k * tile_bytes<Width>(rows, 4),
            records, rows, groups, first, last};
}

template <uint32_t Width>
__device__ __forceinline__ uint32_t row_word(const Tile& t, uint32_t local_row) {
    static_assert(Width == 1 || Width == 2 || Width == 4);
    uint32_t offsets;
    if (t.groups == 4) {
        if constexpr (Width == 1)
            offsets = t.offsets[local_row / 2] >> ((local_row % 2) * 4);
        else if constexpr (Width == 2)
            offsets = t.offsets[local_row];
        else
            offsets = reinterpret_cast<const uint16_t*>(t.offsets)[local_row];
    } else {
        const uint32_t bit = local_row * t.groups * Width, shift = bit % 32;
        const auto* words = reinterpret_cast<const uint32_t*>(t.offsets);
        offsets = words[bit / 32] >> shift;
        // Only a row crossing the word boundary needs the second load.
        // Its bits exist in this region, whose byte size is padded to 4.
        if (shift + t.groups * Width > 32)
            offsets |= words[bit / 32 + 1] << (32 - shift);
    }
    const uint32_t base = t.bases[local_row], mask = (1u << Width) - 1;
    uint32_t word = 0;
#pragma unroll
    for (uint32_t g = 0; g < 4; ++g)
        if (g < t.groups)
            word |= uint32_t(uint8_t(base + ((offsets >> (g * Width)) & mask))) << (8 * g);
    return word;
}

__device__ __forceinline__ uint32_t patch_record(
    uint32_t record, uint32_t local_row, uint32_t word) {
    if ((record >> 10) == local_row) {
        const uint32_t shift = ((record >> 8) & 3) * 8;
        word = (word & ~(255u << shift)) | ((record & 255) << shift);
    }
    return word;
}

__device__ __forceinline__ uint32_t patch_row(
    const Tile& t, uint32_t local_row, uint32_t word) {
    for (uint32_t i = t.first; i < t.last; ++i)
        word = patch_record(t.records[i], local_row, word);
    return word;
}

// Every lane in nonzero mask passes the same tile and mask and calls this
// convergently. The caller's lane must be in mask. Rows may differ. Packed
// lane ranks also cover sparse masks without reading from an absent lane.
__device__ __forceinline__ uint32_t patch_row_warp(
    const Tile& t, uint32_t local_row, uint32_t word, uint32_t mask) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t rank = __popc(mask & ((1u << lane) - 1));
    const uint32_t lanes = __popc(mask);
    for (uint32_t first = t.first; first < t.last; first += lanes) {
        const uint32_t count = min(lanes, t.last - first);
        const uint32_t record = rank < count ? t.records[first + rank] : 0;
        uint32_t sources = mask;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t source = __ffs(sources) - 1;
            const uint32_t value = __shfl_sync(mask, record, source);
            word = patch_record(value, local_row, word);
            sources &= sources - 1;
        }
    }
    return word;
}

// Lab entry: grid (ceil(rows / 64), matrices, 1), block (256, 1, 1).
// Each warp owns a tile and visits its two halves convergently, including
// inactive rows of a short strip. Byte stores accept unaligned raw banks.
template <uint32_t Width, bool Warp>
__device__ __forceinline__ void decode_bank(
    const uint8_t* __restrict__ bank, uint8_t* __restrict__ output,
    uint32_t rows, uint32_t groups) {
    ptx::griddep_launch_dependents();
    ptx::griddep_wait();
    const Matrix m = matrix(bank, blockIdx.y, rows, groups);
    uint8_t* decoded = output + uint64_t(blockIdx.y) * rows * groups;
    for (uint32_t k = threadIdx.x / 32; k < (groups + 3) / 4; k += blockDim.x / 32) {
        const Tile t = tile<Width>(m, blockIdx.x, k);
        for (uint32_t half = 0; half < 64; half += 32) {
            const uint32_t row = half + (threadIdx.x & 31);
            uint32_t word = row < t.rows ? row_word<Width>(t, row) : 0;
            if constexpr (Warp)
                word = patch_row_warp(t, row, word, 0xffffffffu);
            else
                word = patch_row(t, row, word);
            if (row < t.rows) {
                const uint32_t at = (blockIdx.x * 64 + row) * groups + k * 4;
#pragma unroll
                for (uint32_t g = 0; g < 4; ++g)
                    if (g < t.groups) decoded[at + g] = uint8_t(word >> (8 * g));
            }
        }
    }
}

} // namespace chop::moe::scale_tiles
