#pragma once
#include "scale_tiles_expand_linear.cuh"

namespace chop::moe::scale_expand {

// A gate-only instantiation removes the unused second decoder and makes
// tile addressing constant. The host admits one exact complete-tile bank;
// the entry keeps the paired ABI so binding preserves every argument slot.
template<uint32_t Rows, uint32_t Groups, uint32_t Projections, uint32_t Width>
__device__ __forceinline__ void run_tiles_gate(
    Bank first_bank, Bank second_bank, const int32_t* __restrict__ active,
    uint32_t experts) {
    static_assert(Rows > 0 && Rows % 64 == 0 && Groups > 0 && Groups % 4 == 0);
    static_assert(Projections > 0 && (Width == 1 || Width == 2 || Width == 4));
    constexpr uint32_t requested_rows = ((8192 / Groups + 63) / 64) * 64;
    constexpr uint32_t tile_rows = requested_rows < 64 ? 64
        : (requested_rows > Rows ? Rows : requested_rows);
    constexpr uint32_t row_tiles = (Rows + tile_rows - 1) / tile_rows;
    ptx::griddep_launch_dependents();
    ptx::griddep_wait();
    if (blockIdx.x >= Projections * row_tiles) return;
    const uint32_t slot = blockIdx.z;
    uint32_t expert = slot;
    if (active) {
        const int32_t count = active[0], id = active[slot + 1];
        if (count <= 0 || uint32_t(count) > experts || slot >= uint32_t(count)) return;
        if (id < 0 || uint32_t(id) >= experts) return;
        expert = uint32_t(id);
    }
    const Bank bank{first_bank.encoded, first_bank.output, Rows, Groups, Projections, Width, tile_rows};
    const uint32_t matrix = expert * Projections + blockIdx.x / row_tiles;
    const uint32_t first = blockIdx.x % row_tiles * tile_rows;
    const uint32_t end = Rows % tile_rows == 0 ? first + tile_rows : min(Rows, first + tile_rows);
    expand_tiles_linear<Width>(bank, matrix, first, end);
}

} // namespace chop::moe::scale_expand
