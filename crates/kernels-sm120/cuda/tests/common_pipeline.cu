#include "../common/pipeline.cuh"
#include "../common/smem.cuh"

extern "C" __global__ __launch_bounds__(288, 1) void common_pipeline(uint64_t* output) {
  __shared__ chop::TmaPipeline<2, 2> pipe;
  __shared__ alignas(128) uint32_t swizzle_tile[256];
  if (threadIdx.x == 0) pipe.init();
  __syncthreads();
  if ((threadIdx.x & 31) == 0 && threadIdx.x >= 32)
    pipe.prime_empty();
  if (threadIdx.x < 32) chop::configure_producer_warp();
  else chop::configure_math_warp();
  if (threadIdx.x == 0) {
    pipe.producer_acquire(0);
    uint64_t* bar = pipe.producer_commit(0, 128);
    output[0] = reinterpret_cast<uint64_t>(bar);
    pipe.consumer_wait(0);
    pipe.consumer_release(0);
    pipe.invalidate();
  }
  output[1] = chop::smem::swizzle_32(swizzle_tile, 64, 32) +
              chop::smem::swizzle_64(swizzle_tile, 64, 64) +
              chop::smem::swizzle_128(swizzle_tile, 64, 128);
}
