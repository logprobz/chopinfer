#pragma once

// Warp matrix tile transfers. The caller supplies each lane's shared address.
// All lanes in a warp must execute the same form with valid tile addresses.

#include "ptx.cuh"

namespace chop::ptx {

template<int Count, bool Transpose = false>
__device__ __forceinline__ void ldmatrix_b16(uint32_t (&out)[Count], const void* tile) {
  const uint32_t s = shared_address(tile);
  if constexpr (Count == 1) {
    if constexpr (Transpose)
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(out[0]) : "r"(s) : "memory");
    else
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                   : "=r"(out[0]) : "r"(s) : "memory");
  } else if constexpr (Count == 2) {
    if constexpr (Transpose)
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                   : "=r"(out[0]), "=r"(out[1]) : "r"(s) : "memory");
    else
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                   : "=r"(out[0]), "=r"(out[1]) : "r"(s) : "memory");
  } else if constexpr (Count == 4) {
    if constexpr (Transpose)
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                   : "r"(s) : "memory");
    else
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                   : "r"(s) : "memory");
  } else {
    static_assert(Count == 1 || Count == 2 || Count == 4);
  }
}

enum class PackedTile { b4x16_p64, b6x16_p32 };

template<int Count, PackedTile Format>
__device__ __forceinline__ void ldmatrix_b8x16(
    uint32_t (&out)[Count], const void* tile) {
  const uint32_t s = shared_address(tile);
#define CHOP_LD8(COUNT, FORMAT)                                                \
  asm volatile("ldmatrix.sync.aligned.m8n16.x" COUNT ".shared.b8x16." FORMAT   \
               " {%0,%1,%2,%3}, [%4];"                                         \
               : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])       \
               : "r"(s) : "memory")
  static_assert(Count == 4, "packed tile requires four registers");
  if constexpr (Format == PackedTile::b4x16_p64) {
    CHOP_LD8("4", "b4x16_p64");
  } else {
    CHOP_LD8("4", "b6x16_p32");
  }
#undef CHOP_LD8
}

template<PackedTile Format>
__device__ __forceinline__ void ldmatrix_b8x16_trans(
    uint32_t (&out)[4], const void* tile) {
  const uint32_t s = shared_address(tile);
  if constexpr (Format == PackedTile::b4x16_p64)
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(s) : "memory");
  else
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(s) : "memory");
}

template<int Count, bool B8 = false>
__device__ __forceinline__ void stmatrix(void* tile, const uint32_t (&in)[Count]) {
  const uint32_t s = shared_address(tile);
  static_assert(Count == 1 || Count == 2 || Count == 4);
  if constexpr (Count == 1) {
    if constexpr (B8)
      asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
                   :: "r"(s), "r"(in[0]) : "memory");
    else
      asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                   :: "r"(s), "r"(in[0]) : "memory");
  } else if constexpr (Count == 2) {
    if constexpr (B8)
      asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1,%2};"
                   :: "r"(s), "r"(in[0]), "r"(in[1]) : "memory");
    else
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};"
                   :: "r"(s), "r"(in[0]), "r"(in[1]) : "memory");
  } else {
    if constexpr (B8)
      asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1,%2,%3,%4};"
                   :: "r"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]) : "memory");
    else
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                   :: "r"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]) : "memory");
  }
}

template<int Count>
__device__ __forceinline__ void stmatrix_trans_b16(
    void* tile, const uint32_t (&in)[Count]) {
  const uint32_t s = shared_address(tile);
  if constexpr (Count == 1)
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
                 :: "r"(s), "r"(in[0]) : "memory");
  else if constexpr (Count == 2)
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1,%2};"
                 :: "r"(s), "r"(in[0]), "r"(in[1]) : "memory");
  else if constexpr (Count == 4)
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "[%0], {%1,%2,%3,%4};"
                 :: "r"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3])
                 : "memory");
  else
    static_assert(Count == 1 || Count == 2 || Count == 4);
}

} // namespace chop::ptx
