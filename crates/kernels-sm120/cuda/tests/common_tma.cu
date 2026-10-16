#include "../common/ptx_tma.cuh"
#include <cuda.h>

extern "C" __global__ void common_tma(
    const void* map, void* global, unsigned* out, const int* input) {
  __shared__ alignas(128) unsigned tile[256];
  __shared__ alignas(16) uint64_t bar;
  int32_t c1[1] = {input[0]};
  int32_t c2[2] = {input[0], input[1]};
  int32_t c3[3] = {input[0], input[1], input[2]};
  int32_t c4[4] = {input[0], input[1], input[2], input[3]};
  int32_t c5[5] = {input[0], input[1], input[2], input[3], input[4]};
  chop::ptx::mbarrier_init(&bar, 1);
  chop::ptx::fence_mbarrier_init();
  chop::ptx::mbarrier_arrive_expect_tx(&bar, 1024);
  chop::ptx::tma_load<1>(tile, map, c1, &bar);
  chop::ptx::tma_load<2>(tile, map, c2, &bar);
  chop::ptx::tma_load<3>(tile, map, c3, &bar);
  chop::ptx::tma_load<4>(tile, map, c4, &bar);
  chop::ptx::tma_load<5>(tile, map, c5, &bar);
  uint64_t policy = chop::ptx::l2_policy<chop::ptx::L2Evict::first>();
  policy ^= chop::ptx::l2_policy<chop::ptx::L2Evict::last>();
  policy ^= chop::ptx::l2_policy<chop::ptx::L2Evict::normal>();
  chop::ptx::tma_load_hint<1>(tile, map, c1, &bar, policy);
  chop::ptx::tma_load_hint<2>(tile, map, c2, &bar, policy);
  chop::ptx::tma_load_hint<3>(tile, map, c3, &bar, policy);
  chop::ptx::tma_load_hint<4>(tile, map, c4, &bar, policy);
  chop::ptx::tma_load_hint<5>(tile, map, c5, &bar, policy);
  chop::ptx::tma_multicast_2d(tile, map, c2, &bar, 1);
  chop::ptx::tma_load_cluster_2d(tile, map, c2, &bar, policy);
  chop::ptx::tma_multicast_2d_hint(tile, map, c2, &bar, 1, policy);
  int32_t gather[5] = {input[0], input[1], input[2], input[3], input[4]};
  chop::ptx::tma_gather4_2d(tile, map, gather, &bar);
  chop::ptx::cp_async_bulk(tile, global, 128, &bar);
  chop::ptx::cp_async_bulk_store(global, tile, 128);
  chop::ptx::cp_reduce_add_f32(static_cast<float*>(global),
                                reinterpret_cast<float*>(tile), 128);
  chop::ptx::cp_reduce_add_bf16(static_cast<unsigned*>(global), tile, 128);
  chop::ptx::tma_store_tensor<1>(map, c1, tile);
  chop::ptx::tma_store_tensor<2>(map, c2, tile);
  chop::ptx::tma_store_tensor<3>(map, c3, tile);
  chop::ptx::tma_store_tensor<4>(map, c4, tile);
  chop::ptx::tma_store_tensor<5>(map, c5, tile);
  chop::ptx::tma_reduce_add_f32(map, c2, tile);
  chop::ptx::tma_reduce_add_bf16(map, c2, tile);
  chop::ptx::tma_store(map, c2, tile);
  chop::ptx::prefetch_tensor_l2(map, c2);
  chop::ptx::prefetch_tensormap(map);
  chop::ptx::fence_proxy_tensormap_release();
  chop::ptx::fence_proxy_tensormap_acquire(map);
  chop::ptx::tensormap_replace_address(tile, global);
  chop::ptx::tensormap_replace_dim<0>(tile, 64);
  chop::ptx::tensormap_replace_stride<1>(tile, 128);
  chop::ptx::cp_bulk_commit();
  chop::ptx::cp_bulk_wait();
  chop::ptx::mbarrier_arrive(&bar);
  out[threadIdx.x] = chop::ptx::mbarrier_try_wait(&bar, 0);
  chop::ptx::mbarrier_expect_tx(&bar, 0);
  out[threadIdx.x] += chop::ptx::mbarrier_test_wait(&bar, 0);
  chop::ptx::mbarrier_inval(&bar);
}

extern "C" __global__ void common_tensormap_prefetch(
    const __grid_constant__ CUtensorMap map) {
  chop::ptx::prefetch_tensormap(&map);
}
