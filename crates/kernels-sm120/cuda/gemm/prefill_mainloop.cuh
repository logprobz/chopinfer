#pragma once

// Reusable pieces of the fragment-major prefill mainloop.
//
// Orientation is swapped: weights are the m16n8k16 A operand (16 output
// features per strip) and BF16 activations are B (tokens on n8). Each ring
// stage holds a 64-K slice: the activation tile arrives through a 2-D TMA
// load with 128-byte swizzle (128-byte rows, one row per token) and the
// weight strips arrive as contiguous fragment-major runs through 1-D bulk
// copies. Math warps read weights with conflict-free 16-byte LDS and
// activations with conflict-free ldmatrix.x4.

#include "dense.cuh"

namespace chop::gemm::prefill {

constexpr int stage_k = 64;
constexpr int swizzle_row_bytes = stage_k * 2;

__device__ __forceinline__ unsigned char* align_swizzle_storage(
    unsigned char* storage) {
  return reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(storage) + 1023) & ~uintptr_t(1023));
}

// One lane's complete A fragment (16 bytes) from a fragment-major tile.
__device__ __forceinline__ ptx::A4 load_a(uint32_t address) {
  ptx::A4 a;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(a.v[0]), "=r"(a.v[1]), "=r"(a.v[2]), "=r"(a.v[3])
      : "r"(address));
  return a;
}

// B fragments for two adjacent n8 token tiles at one k16 step. `row_address`
// is this lane's precomputed activation row base; `chunk` is the lane's
// 16-byte K chunk index before swizzling and `row` its token row.
__device__ __forceinline__ void load_b2(
    uint32_t row_address, uint32_t chunk, uint32_t row,
    ptx::B2& first, ptx::B2& second) {
  const uint32_t address = row_address + ((chunk ^ (row & 7)) << 4);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(first.v[0]), "=r"(first.v[1]),
        "=r"(second.v[0]), "=r"(second.v[1])
      : "r"(address));
}

// Two E4M3 bytes per packed conversion; one 16-byte fragment covers two
// k16 steps (`Half` selects the step).
template<int Half>
__device__ __forceinline__ ptx::A4 fp8_to_bf16(const ptx::A4& packed) {
  const uint32_t low = packed.v[Half * 2];
  const uint32_t high = packed.v[Half * 2 + 1];
  ptx::A4 a;
  a.v[0] = convert::unpack_bf16x2_e4m3(uint16_t(low));
  a.v[1] = convert::unpack_bf16x2_e4m3(uint16_t(low >> 16));
  a.v[2] = convert::unpack_bf16x2_e4m3(uint16_t(high));
  a.v[3] = convert::unpack_bf16x2_e4m3(uint16_t(high >> 16));
  return a;
}

__device__ __forceinline__ void named_barrier(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}

// Waits on the empty barrier of `slot` for the pass given by `iteration`.
// The first pass over the ring returns immediately.
__device__ __forceinline__ void wait_empty(
    uint64_t* empty, uint32_t iteration, uint32_t stages) {
  ptx::mbarrier_wait(&empty[iteration % stages],
      ((iteration / stages) & 1) ^ 1);
}

__device__ __forceinline__ void wait_full(
    uint64_t* full, uint32_t iteration, uint32_t stages) {
  ptx::mbarrier_wait(&full[iteration % stages], (iteration / stages) & 1);
}

} // namespace chop::gemm::prefill
