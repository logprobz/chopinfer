#pragma once

// SM12x control, cluster and global-memory primitives. Cluster and DSMEM
// helpers are probe-gated and require remote CTA addresses from mapa.

#include "ptx.cuh"

namespace chop::ptx {

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(rank));
  return rank;
}
__device__ __forceinline__ uint32_t cluster_nctarank() {
  uint32_t count;
  asm volatile("mov.u32 %0, %cluster_nctarank;" : "=r"(count));
  return count;
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_map_shared(
    const void* local, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(shared_address(local)), "r"(rank));
  return remote;
}
__device__ __forceinline__ void cluster_st_async_f32(
    uint32_t remote, float value, uint32_t remote_barrier) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 "
               "[%0], %1, [%2];"
               :: "r"(remote), "f"(value), "r"(remote_barrier) : "memory");
}
__device__ __forceinline__ void cluster_red_async_add_u32(
    uint32_t remote, uint32_t value, uint32_t remote_barrier) {
  asm volatile("red.async.relaxed.cluster.shared::cluster."
               "mbarrier::complete_tx::bytes.add.u32 [%0], %1, [%2];"
               :: "r"(remote), "r"(value), "r"(remote_barrier) : "memory");
}

// Probe-gated: response is a 16-byte aligned shared buffer. Query only after
// its 16-byte mbarrier transaction completes.
__device__ __forceinline__ void cluster_try_cancel(
    void* response, uint64_t* bar) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta."
               "mbarrier::complete_tx::bytes.b128 [%0], [%1];"
               :: "r"(shared_address(response)), "r"(shared_address(bar))
               : "memory");
}
__device__ __forceinline__ bool cluster_query_cancel(
    const uint32_t (&response)[4], uint32_t& first_cta_x) {
  uint32_t canceled;
  asm volatile("{ .reg .pred p; .reg .b128 handle; "
               "mov.b128 handle, {%2,%3,%4,%5}; "
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
               "p, handle; "
               "selp.u32 %0, 1, 0, p; "
               "clusterlaunchcontrol.query_cancel.get_first_ctaid::x."
               "b32.b128 %1, handle; }"
               : "=r"(canceled), "=r"(first_cta_x)
               : "r"(response[0]), "r"(response[1]), "r"(response[2]),
                 "r"(response[3]));
  return canceled != 0;
}

__device__ __forceinline__ uint32_t load_streaming_u32(const uint32_t* p) {
  uint32_t value;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];"
               : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ __forceinline__ void store_streaming_u32(uint32_t* p, uint32_t value) {
  asm volatile("st.global.cs.u32 [%0], %1;"
               :: "l"(p), "r"(value) : "memory");
}
__device__ __forceinline__ void red_global_add_f32(float* p, float value) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;"
               :: "l"(p), "f"(value) : "memory");
}
__device__ __forceinline__ void red_global_add_bf16x2(
    uint32_t* p, uint32_t packed) {
  asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;"
               :: "l"(p), "r"(packed) : "memory");
}
__device__ __forceinline__ uint32_t shuffle_xor(
    uint32_t value, uint32_t lane_mask) {
  uint32_t result;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
               : "=r"(result) : "r"(value), "r"(lane_mask));
  return result;
}
__device__ __forceinline__ float warp_sum(float value) {
  uint32_t bits = __float_as_uint(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    bits = __float_as_uint(__uint_as_float(bits) +
                           __uint_as_float(shuffle_xor(bits, offset)));
  }
  return __uint_as_float(bits);
}

} // namespace chop::ptx
