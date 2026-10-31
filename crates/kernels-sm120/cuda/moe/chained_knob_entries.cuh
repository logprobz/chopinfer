#pragma once
#include "chained_gate_tickets.cuh"
#include "chained_down_scale_tiles.cuh"
#include "gate_finish_loads.cuh"

// The finishers under a knob set (chained_knobs.cuh), each with every other
// policy of its served entry.
namespace chop::moe::knobs {
template<class C, uint32_t FloorBits, class K>
__device__ __forceinline__ void gate_finish(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, uint16_t* intermediate, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  bandwidth::gate_finish<C, FloorBits, typename C::clamp,
      chained_policy::CeilE4M3Group32<FloorBits>, chained_policy::BeforeQuantizeRouteWeight,
      chained_policy::Bf16IntermediateF32Output, chained_policy::OrderedF32, K>(
      scratch, live_tokens, route_weights, intermediate, codes, scales, status, tickets);
}

template<class C, uint32_t FloorBits, class K>
__device__ __forceinline__ void gate_finish_loads(const void* scratch,
    const int32_t* live_tokens, const float* route_weights, uint16_t* intermediate,
    uint8_t* codes, uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  bandwidth_loads::gate_finish<C, FloorBits, typename C::clamp,
      chained_policy::CeilE4M3Group32<FloorBits>, chained_policy::BeforeQuantizeRouteWeight,
      chained_policy::Bf16IntermediateF32Output, chained_policy::OrderedF32, K>(
      scratch, live_tokens, route_weights, intermediate, codes, scales, status, tickets);
}

template<class C, class K>
__device__ __forceinline__ void down_finish(const void* scratch, const int32_t* live_tokens,
    const int32_t* ids, const float* shared, float* routed, float* output) {
  floor::down_finish<C, false, chained_policy::OrderedF32,
      chained_policy::Bf16IntermediateF32Output, K>(
      scratch, live_tokens, ids, shared, routed, output);
}
} // namespace chop::moe::knobs
