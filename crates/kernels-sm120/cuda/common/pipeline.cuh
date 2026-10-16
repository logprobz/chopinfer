#pragma once

// K-stage TMA to warp-MMA handoff. One producer warp and ConsumerWarps math
// warps participate. Only each warp's leader touches barriers. The caller
// synchronizes the CTA after init and before the first prime_empty.

#include "ptx.cuh"

namespace chop {

template<int Stages, int ConsumerWarps>
struct alignas(16) TmaPipeline {
  static_assert(Stages >= 2);
  static_assert(ConsumerWarps >= 1);
  uint64_t full[Stages];
  uint64_t empty[Stages];

  __device__ __forceinline__ void init() {
    for (int i = 0; i < Stages; ++i) {
      ptx::mbarrier_init(&full[i], 1);
      ptx::mbarrier_init(&empty[i], ConsumerWarps);
    }
    ptx::fence_mbarrier_init();
  }
  __device__ __forceinline__ static int slot(uint32_t step) {
    return step % Stages;
  }
  __device__ __forceinline__ static uint32_t phase(uint32_t step) {
    return (step / Stages) & 1;
  }
  __device__ __forceinline__ void prime_empty() {
    for (int i = 0; i < Stages; ++i)
      ptx::mbarrier_arrive(&empty[i]);
  }
  __device__ __forceinline__ void producer_acquire(uint32_t step) {
    ptx::mbarrier_wait(&empty[slot(step)], phase(step));
  }
  __device__ __forceinline__ uint64_t* producer_commit(
      uint32_t step, uint32_t bytes) {
    uint64_t* bar = &full[slot(step)];
    ptx::mbarrier_arrive_expect_tx(bar, bytes);
    return bar;
  }
  __device__ __forceinline__ void consumer_wait(uint32_t step) {
    ptx::mbarrier_wait(&full[slot(step)], phase(step));
  }
  __device__ __forceinline__ void consumer_release(uint32_t step) {
    ptx::mbarrier_arrive(&empty[slot(step)]);
  }
  __device__ __forceinline__ void invalidate() {
    for (int i = 0; i < Stages; ++i) {
      ptx::mbarrier_inval(&full[i]);
      ptx::mbarrier_inval(&empty[i]);
    }
  }
};

__device__ __forceinline__ void configure_producer_warp() {
  ptx::setmaxnreg_dec<40>();
}
__device__ __forceinline__ void configure_math_warp() {
  ptx::setmaxnreg_inc<232>();
}

} // namespace chop
