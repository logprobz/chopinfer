#include "../common/ptx_control.cuh"

extern "C" __global__ __launch_bounds__(288, 1) void common_control(
    uint32_t* out, float* sums, const uint32_t* input) {
  __shared__ alignas(16) uint32_t response[4];
  __shared__ alignas(16) uint64_t bar;
  __shared__ uint32_t cell;
  chop::ptx::fence_async_shared();
  chop::ptx::fence_async_global();
  uint32_t rank = chop::ptx::cluster_ctarank();
  uint32_t size = chop::ptx::cluster_nctarank();
  chop::ptx::cluster_arrive_release();
  chop::ptx::cluster_wait_acquire();
  chop::ptx::cluster_arrive_relaxed();
  uint32_t remote = chop::ptx::cluster_map_shared(&cell, rank ^ 1);
  uint32_t remote_bar = chop::ptx::cluster_map_shared(&bar, rank ^ 1);
  chop::ptx::cluster_st_async_f32(remote, 1.0f, remote_bar);
  chop::ptx::cluster_red_async_add_u32(remote, 1, remote_bar);
  chop::ptx::cluster_try_cancel(response, &bar);
  uint32_t first;
  bool canceled = chop::ptx::cluster_query_cancel(response, first);
  uint32_t v = chop::ptx::load_streaming_u32(input);
  chop::ptx::store_streaming_u32(out, v + rank + size + first + canceled);
  chop::ptx::red_global_add_f32(sums, 1.0f);
  chop::ptx::red_global_add_bf16x2(out, v);
  out[threadIdx.x] = chop::ptx::shuffle_xor(v, 1);
  sums[threadIdx.x] = chop::ptx::warp_sum(1.0f);
  uint32_t elected;
  chop::ptx::elect_sync(elected);
  out[threadIdx.x] += elected;
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  if (threadIdx.x < 32)
    chop::ptx::setmaxnreg_dec<40>();
  else
    chop::ptx::setmaxnreg_inc<232>();
}
