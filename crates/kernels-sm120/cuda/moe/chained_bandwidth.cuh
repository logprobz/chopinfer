#pragma once
#include "chained_floor.cuh"

// Keep the qualified family locked. This candidate restores the streamed
// grid and removes the shared weight ring. The prefetch variant issues L2
// hints without consuming shared memory or holding another strip in registers.
namespace chop::moe::bandwidth {
using namespace streamed;
template<class C, int Rows, int Ahead>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  pipeline::compute<C, Rows, 0, Ahead>(input, weight, weight_scales, activation_scales, offsets, sorted_rows, padded_rows, live_tokens, scratch);
}

template<class C, uint32_t FloorBits, class Clamp = typename C::clamp,
    class Quantization = chained_policy::CeilE4M3Group32<FloorBits>,
    class RouteWeight = chained_policy::BeforeQuantizeRouteWeight,
    class Output = chained_policy::Bf16IntermediateF32Output,
    class Reduction = chained_policy::OrderedF32>
__device__ void gate_finish(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, uint16_t* intermediate, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  static_assert(C::gate_up && C::n % 32 == 0);
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
  // All lanes load consecutive columns. Stage four independent partials
  // before the ordered adds, instead of issuing dependent scalar loads
  // from half a warp twice for each output pair.
  const int column = column0 + threadIdx.x;
  if (column < C::n) {
    float sum[2] = {};
#pragma unroll 1
    for (int base = 0; base < Layout<C>::blocks; base += 4) {
      float value[4][2];
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          value[b][h] = base + b < Layout<C>::blocks ? __ldcg(input +
              ((uint64_t(route) * 2 + h) * Layout<C>::blocks + base + b) * C::n + column) : 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        if (base + b < Layout<C>::blocks)
#pragma unroll
          for (int h = 0; h < 2; ++h) sum[h] = Reduction::add(sum[h], value[b][h]);
    }
    const uint16_t value = Output::intermediate(
        RouteWeight::template apply<Clamp>(sum[0], sum[1], route_weights[route]));
    values[threadIdx.x] = value;
    intermediate[uint64_t(route) * C::n + column] = value;
  } else values[threadIdx.x] = 0;
  __syncthreads();
  if (group < C::n / 32) {
    auto value = Quantization::quantize(
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


} // namespace chop::moe::bandwidth
