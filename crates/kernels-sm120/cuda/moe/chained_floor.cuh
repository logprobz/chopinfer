#pragma once
#include "chained_pipeline.cuh"
#include "../row/ceil_parallel.cuh"

// A separate entry preserves the locked streamed consumers. Every weight
// stage is one independent 128-K chain; no ring stage changes a fold.
namespace chop::moe::floor {
using namespace streamed;
template<class C, int Rows, int Stages = 3>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  pipeline::compute<C, Rows, Stages, 0>(input, weight, weight_scales, activation_scales, offsets, sorted_rows, padded_rows, live_tokens, scratch);
}

// Each warp finishes one 32-value ceil group. The last integer ticket for
// a token clears its entire quantized row on any sticky error, just as the
// unfused row quantizer does. The ticket publishes bytes, never float sums.
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
  if (lane < 16) {
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int column = group * 32 + lane * 2 + v;
      float sum[2] = {};
      if (column < C::n) {
#pragma unroll 1
        for (int block = 0; block < Layout<C>::blocks; ++block)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            sum[h] = Reduction::add(sum[h], __ldcg(input +
                ((uint64_t(route) * 2 + h) * Layout<C>::blocks + block) * C::n + column));
        const uint16_t value = Output::intermediate(
            RouteWeight::template apply<Clamp>(sum[0], sum[1], route_weights[route]));
        values[warp * 32 + lane * 2 + v] = value;
        intermediate[uint64_t(route) * C::n + column] = value;
      } else values[warp * 32 + lane * 2 + v] = 0;
    }
  }
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

// Route weights were applied before intermediate quantization. Fold each
// down output in K order, add routes in selection order, then add shared.
// The caller places this launch after the shared-expert join.
template<class C, bool OmitFirst = false,
    class Reduction = chained_policy::OrderedF32,
    class Output = chained_policy::Bf16IntermediateF32Output>
__device__ void down_finish(const void* scratch, const int32_t* live_tokens,
    const int32_t* ids, const float* shared, float* routed, float* output) {
  static_assert(!C::gate_up);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live < 0 || live > C::tokens_capacity) __trap();
  const int token = blockIdx.y, column = blockIdx.x * 128 + threadIdx.x;
  if (token >= live || column >= C::n) return;
  const float* input = partials<C>(const_cast<void*>(scratch));
  float total = 0.f;
#pragma unroll
  for (int k = 0; k < C::top_k; ++k) {
    const int route = token * C::top_k + k;
    const int expert = ids[route];
    if (expert < 0 || expert >= C::experts) continue;
    float value = 0.f;
#pragma unroll
    for (int block = OmitFirst ? 1 : 0; block < Layout<C>::blocks; ++block)
      value = Reduction::add(value, __ldcg(input +
          (uint64_t(route) * Layout<C>::blocks + block) * C::n + column));
    routed[uint64_t(route) * C::n + column] = Output::output(value);
    total = Reduction::add(total, value);
  }
  output[uint64_t(token) * C::n + column] =
      Output::output(Reduction::add(total, shared[uint64_t(token) * C::n + column]));
}
} // namespace chop::moe::floor
