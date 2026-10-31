#pragma once
#include "chained_lanes_scale_tiles.cuh"
#include "chained_down_scale_tiles.cuh"
#include "gate_finish_loads.cuh"

// The direct queues of steps above 64 rows, where an expert may take more
// than 64 routed rows (at most one per token). The generic route writes the
// metadata: expert ranges padded to whole 64-row tiles, sorted routes and
// the active list. This pack writes the streamed queue table from its
// counts and active list, as the streamed pack does, and clears the gate
// finisher's ticket of every token of the capacity, as the direct metadata
// does. The absorbing gate and two-strip down queues then run every expert
// eight rows at a time at this capacity, so each chain, partial, finisher
// and output is the 64-row direct path's.
namespace chop::moe::wide {
using namespace streamed;

template<class C>
__device__ void pack(const int32_t* counts, const int32_t* active, const int32_t* live_tokens,
    void* scratch, uint32_t* tickets) {
  streamed::pack<C>(counts, active, live_tokens, scratch);
  for (int at = threadIdx.x; at < C::tokens_capacity; at += 32) tickets[at] = 0;
}
} // namespace chop::moe::wide
