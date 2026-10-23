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

} // namespace chop::moe::scale_expand
