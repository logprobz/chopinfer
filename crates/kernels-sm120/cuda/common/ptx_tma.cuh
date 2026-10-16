#pragma once

// TMA policies and tensor-map operations. The map is a CUDA tensor-map
// descriptor in global memory. Multicast and gather4 are probe-gated.

#include "ptx.cuh"

namespace chop::ptx {

enum class L2Evict { first, last, normal };

template<L2Evict Priority>
__device__ __forceinline__ uint64_t l2_policy() {
  uint64_t result;
  if constexpr (Priority == L2Evict::first)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
                 : "=l"(result));
  else if constexpr (Priority == L2Evict::last)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
                 : "=l"(result));
  else
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;"
                 : "=l"(result));
  return result;
}

template<int Rank>
__device__ __forceinline__ void tma_load_hint(
    void* dst, const void* map, const int32_t (&coord)[Rank],
    uint64_t* bar, uint64_t policy) {
  const uint32_t s = shared_address(dst), m = shared_address(bar);
  const uint64_t g = reinterpret_cast<uint64_t>(map);
#define CHOP_TMA_HINT(RANK, COORDS, ...)                                       \
  asm volatile("cp.async.bulk.tensor." RANK ".shared::cta.global."             \
               "mbarrier::complete_tx::bytes.tile.L2::cache_hint "            \
               "[%0], [%1, {" COORDS "}], [%2], %3;"                         \
               :: "r"(s), "l"(g), "r"(m), "l"(policy), __VA_ARGS__ : "memory")
  if constexpr (Rank == 1)
    CHOP_TMA_HINT("1d", "%4", "r"(coord[0]));
  else if constexpr (Rank == 2)
    CHOP_TMA_HINT("2d", "%4,%5", "r"(coord[0]), "r"(coord[1]));
  else if constexpr (Rank == 3)
    CHOP_TMA_HINT("3d", "%4,%5,%6", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]));
  else if constexpr (Rank == 4)
    CHOP_TMA_HINT("4d", "%4,%5,%6,%7", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]));
  else if constexpr (Rank == 5)
    CHOP_TMA_HINT("5d", "%4,%5,%6,%7,%8", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]), "r"(coord[4]));
  else
    static_assert(Rank >= 1 && Rank <= 5);
#undef CHOP_TMA_HINT
}

// Probe-gated: cluster size > 1 and multicast need kernel-lab validation.
__device__ __forceinline__ void tma_multicast_2d(
    void* dst, const void* map, const int32_t (&coord)[2],
    uint64_t* bar, uint16_t cta_mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
               "mbarrier::complete_tx::bytes.multicast::cluster "
               "[%0], [%1, {%2,%3}], [%4], %5;"
               :: "r"(shared_address(dst)), "l"(map), "r"(coord[0]),
                  "r"(coord[1]), "r"(shared_address(bar)), "h"(cta_mask)
               : "memory");
}
// Packed 16U4_ALIGN16B maps require the shared::cluster destination form,
// even when the cluster has one CTA.
__device__ __forceinline__ void tma_load_cluster_2d(
    void* dst, const void* map, const int32_t (&coord)[2], uint64_t* bar,
    uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
               "mbarrier::complete_tx::bytes.tile.L2::cache_hint "
               "[%0], [%1, {%2,%3}], [%4], %5;"
               :: "r"(shared_address(dst)), "l"(map), "r"(coord[0]),
                  "r"(coord[1]), "r"(shared_address(bar)), "l"(policy)
               : "memory");
}
__device__ __forceinline__ void tma_multicast_2d_hint(
    void* dst, const void* map, const int32_t (&coord)[2],
    uint64_t* bar, uint16_t cta_mask, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
               "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
               "[%0], [%1, {%2,%3}], [%4], %5, %6;"
               :: "r"(shared_address(dst)), "l"(map), "r"(coord[0]),
                  "r"(coord[1]), "r"(shared_address(bar)), "h"(cta_mask),
                  "l"(policy) : "memory");
}

// Probe-gated: gathers four rows sharing one column coordinate.
__device__ __forceinline__ void tma_gather4_2d(
    void* dst, const void* map, const int32_t (&coord)[5], uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global."
               "mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
               :: "r"(shared_address(dst)), "l"(map), "r"(coord[0]),
                  "r"(coord[1]), "r"(coord[2]), "r"(coord[3]),
                  "r"(coord[4]), "r"(shared_address(bar)) : "memory");
}

template<int Rank>
__device__ __forceinline__ void tma_store_tensor(
    const void* map, const int32_t (&coord)[Rank], const void* src) {
  const uint32_t s = shared_address(src);
#define CHOP_TMA_STORE(RANK, COORDS, ...)                                      \
  asm volatile("cp.async.bulk.tensor." RANK ".global.shared::cta.tile.bulk_group " \
               "[%0, {" COORDS "}], [%1];"                                   \
               :: "l"(map), "r"(s), __VA_ARGS__ : "memory")
  if constexpr (Rank == 1)
    CHOP_TMA_STORE("1d", "%2", "r"(coord[0]));
  else if constexpr (Rank == 2)
    CHOP_TMA_STORE("2d", "%2,%3", "r"(coord[0]), "r"(coord[1]));
  else if constexpr (Rank == 3)
    CHOP_TMA_STORE("3d", "%2,%3,%4", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]));
  else if constexpr (Rank == 4)
    CHOP_TMA_STORE("4d", "%2,%3,%4,%5", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]));
  else if constexpr (Rank == 5)
    CHOP_TMA_STORE("5d", "%2,%3,%4,%5,%6", "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]), "r"(coord[4]));
  else
    static_assert(Rank >= 1 && Rank <= 5);
#undef CHOP_TMA_STORE
}

__device__ __forceinline__ void tma_reduce_add_f32(
    const void* map, const int32_t (&coord)[2], const void* src) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
               "[%0, {%1,%2}], [%3];"
               :: "l"(map), "r"(coord[0]), "r"(coord[1]),
                  "r"(shared_address(src)) : "memory");
}
__device__ __forceinline__ void tma_reduce_add_bf16(
    const void* map, const int32_t (&coord)[2], const void* src) {
  tma_reduce_add_f32(map, coord, src);
}

__device__ __forceinline__ void prefetch_tensormap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}
__device__ __forceinline__ void fence_proxy_tensormap_release() {
  asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_tensormap_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
               :: "l"(map) : "memory");
}
__device__ __forceinline__ void tensormap_replace_address(
    void* shared_map, const void* address) {
  asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 "
               "[%0], %1;" :: "r"(shared_address(shared_map)), "l"(address)
               : "memory");
}
template<int Dimension>
__device__ __forceinline__ void tensormap_replace_dim(
    void* shared_map, uint32_t extent) {
  static_assert(Dimension >= 0 && Dimension < 5);
  asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 "
               "[%0], %1, %2;"
               :: "r"(shared_address(shared_map)), "n"(Dimension), "r"(extent)
               : "memory");
}
template<int Dimension>
__device__ __forceinline__ void tensormap_replace_stride(
    void* shared_map, uint64_t bytes) {
  static_assert(Dimension >= 0 && Dimension < 5);
  asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 "
               "[%0], %1, %2;"
               :: "r"(shared_address(shared_map)), "n"(Dimension), "l"(bytes)
               : "memory");
}

} // namespace chop::ptx
