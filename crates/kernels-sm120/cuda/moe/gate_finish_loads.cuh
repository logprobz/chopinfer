#pragma once
#include "chained_bandwidth.cuh"

// Lab variant of the bandwidth gate finisher with the same bits and fewer
// serial L2 round trips. The served body loads the K-block partials in
// groups of four, one dependent round trip per group, then the route
// weight. Here every partial and the route weight load before the first
// add; the adds, their order and the quantized tail are unchanged.
namespace chop::moe::bandwidth_loads {
using namespace streamed;

template<class C, uint32_t FloorBits>
__device__ void gate_finish(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, uint16_t* intermediate, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  static_assert(C::gate_up && C::n % 32 == 0);
  constexpr int Blocks = Layout<C>::blocks;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, route = blockIdx.y;
  if (live < 0 || live > C::tokens_capacity || blockDim.x != 128) __trap();
  if (route >= live * C::top_k) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int column0 = blockIdx.x * 128, group = column0 / 32 + warp;
  const int token = route / C::top_k;
  __shared__ uint16_t values[128];
  __shared__ uint32_t last;
  const float* input = partials<C>(const_cast<void*>(scratch));
  const int column = column0 + threadIdx.x;
  if (column < C::n) {
    const float weight = route_weights[route];
    float value[Blocks][2];
#pragma unroll
    for (int b = 0; b < Blocks; ++b)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        value[b][h] = __ldcg(input + ((uint64_t(route) * 2 + h) * Blocks + b) * C::n + column);
    float sum[2] = {};
#pragma unroll
    for (int b = 0; b < Blocks; ++b)
#pragma unroll
      for (int h = 0; h < 2; ++h) sum[h] = __fadd_rn(sum[h], value[b][h]);
    const uint16_t out = convert::bf16_from_f32(
        chained::before_down<typename C::clamp>(sum[0], sum[1], weight));
    values[threadIdx.x] = out;
    intermediate[uint64_t(route) * C::n + column] = out;
  } else values[threadIdx.x] = 0;
  __syncthreads();
  if (group < C::n / 32) {
    auto value = row::block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
        values, warp * 32 + lane * 2);
    uint32_t flags = value.flags;
#pragma unroll
    for (int step = 16; step; step >>= 1)
      flags |= __shfl_xor_sync(0xffffffffu, flags, step);
    if (lane == 0) {
      if (flags) atomicOr(status + token, flags);
      scales[uint64_t(route) * (C::n / 32) + group] = value.scale_code;
    }
    if (lane < 16) {
      const uint64_t at = uint64_t(route) * C::n + group * 32 + lane * 2;
      codes[at] = uint8_t(value.codes);
      codes[at + 1] = uint8_t(value.codes >> 8);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + token) : "memory");
    last = arrived == ((C::n + 127) / 128) * C::top_k - 1;
  }
  __syncthreads();
  if (last && __ldcg(status + token)) {
    for (int at = threadIdx.x; at < C::top_k * C::n; at += 128)
      codes[uint64_t(token) * C::top_k * C::n + at] = 0;
    for (int at = threadIdx.x; at < C::top_k * C::n / 32; at += 128)
      scales[uint64_t(token) * (C::top_k * C::n / 32) + at] = 127;
  }
}
} // namespace chop::moe::bandwidth_loads
