#pragma once

#include <stdint.h>
#include "route_parallel.cuh"

// Routes for any count of local experts per rank (for example 72 or 96 of
// a sharded expert set), which `route_parallel` cannot hold: it lays local
// experts on whole warps. Selection is `select_route_warp` with its scoring
// policies, comparisons, lower-expert tie breaking and serial weight sum, so
// IDs and weights equal `route_parallel`'s for the same logits. The route
// metadata is the common layout over this rank's local experts: routes of
// each expert in ascending global route index `token * TopK + k`, padded
// with -1 to a multiple of `Pad`; `active_experts[0]` is the active count,
// followed by the active experts in ascending order.
//
// The decode route is one CTA; prefill selects with one warp per row
// (`route_select_rows`), counts per local expert (`route_count_local`) and
// scatters per local expert (`route_scatter_local`). Every kernel reads the
// live count after `griddepcontrol.wait`, traps above its capacity, and
// uses no atomics; integer metadata is deterministic.
namespace chop::moe::local {

__host__ __device__ constexpr int round_up_32(int value) { return (value + 31) / 32 * 32; }

template<int Experts, int LocalExperts>
__device__ __forceinline__ int local_expert(int id, uint32_t rank) {
  return id >= 0 && id < Experts && uint32_t(id) / LocalExperts == rank ? id % LocalExperts : -1;
}

template<int Experts, int TopK, int LocalExperts, int Capacity, int Pad, class Score>
__device__ void route_decode(
    const float* logits, int32_t* ids, float* weights, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* num_padded_rows, const int32_t* live_tokens,
    int32_t* active_experts, uint32_t rank, Score score) {
  static_assert(Capacity >= 1 && Capacity <= 32 && Pad > 0);
  static_assert(LocalExperts > 0 && LocalExperts <= 1024 && Experts % LocalExperts == 0);
  constexpr int Threads = Capacity * 32 > round_up_32(LocalExperts)
      ? Capacity * 32 : round_up_32(LocalExperts);
  __shared__ int32_t route_expert[Capacity * TopK];
  __shared__ int32_t expert_count[LocalExperts], expert_first[LocalExperts];
  if (blockDim.x != Threads || gridDim.x != 1) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > Capacity || rank >= Experts / LocalExperts) __trap();
  const int tokens = max(live, 0);
  const int warp = threadIdx.x >> 5;
  if (warp < tokens) select_route_warp<Experts, TopK>(logits, warp, score, ids, weights);
  __syncthreads();
  const int routes = tokens * TopK;
  for (int r = threadIdx.x; r < routes; r += Threads)
    route_expert[r] = local_expert<Experts, LocalExperts>(ids[r], rank);
  __syncthreads();
  const int expert = threadIdx.x;
  if (expert < LocalExperts) {
    int count = 0;
    for (int r = 0; r < routes; ++r) count += route_expert[r] == expert;
    counts[expert] = count;
    expert_count[expert] = count;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0, active = 0;
    for (int e = 0; e < LocalExperts; ++e) {
      expert_first[e] = total;
      offsets[e] = total;
      total += (expert_count[e] + Pad - 1) / Pad * Pad;
      if (expert_count[e] > 0) active_experts[1 + active++] = e;
    }
    offsets[LocalExperts] = total;
    *num_padded_rows = total;
    active_experts[0] = active;
  }
  __syncthreads();
  if (expert < LocalExperts) {
    int at = expert_first[expert];
    for (int r = 0; r < routes; ++r)
      if (route_expert[r] == expert) sorted_rows[at++] = r;
    const int end = expert_first[expert] + (expert_count[expert] + Pad - 1) / Pad * Pad;
    for (; at < end; ++at) sorted_rows[at] = -1;
  }
}

// Eight row warps per 256-thread CTA; grid ceil(Capacity / 8).
template<int Experts, int TopK, int Capacity, class Score>
__device__ void route_select_rows(
    const float* logits, int32_t* ids, float* weights, const int32_t* live_tokens, Score score) {
  if (blockDim.x != 256 || gridDim.x != (Capacity + 7) / 8) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > Capacity) __trap();
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row < live) select_route_warp<Experts, TopK>(logits, row, score, ids, weights);
}

// The routes of local expert `blockIdx.x` among the live rows' IDs: thread
// t counts routes t, t + 256, ..., then a fixed-order tree sums the counts.
template<int Experts, int TopK, int LocalExperts, int Capacity>
__device__ void route_count_local(
    const int32_t* ids, int32_t* counts, const int32_t* live_tokens, uint32_t rank) {
  __shared__ int partial[256];
  if (blockDim.x != 256 || gridDim.x != LocalExperts) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > Capacity || rank >= Experts / LocalExperts) __trap();
  const int routes = max(live, 0) * TopK;
  const int expert = blockIdx.x;
  int count = 0;
  for (int r = threadIdx.x; r < routes; r += 256)
    count += local_expert<Experts, LocalExperts>(ids[r], rank) == expert;
  partial[threadIdx.x] = count;
  __syncthreads();
  for (int step = 128; step; step >>= 1) {
    if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[expert] = partial[0];
}

// Local expert `blockIdx.x` writes its offset, its active-list slot and its
// routes in ascending route order (ballots and preceding-warp counts), then
// its padding. The last expert writes the totals, the first the active count.
template<int Experts, int TopK, int LocalExperts, int Capacity, int Pad>
__device__ void route_scatter_local(
    const int32_t* ids, const int32_t* counts, int32_t* offsets, int32_t* sorted_rows,
    int32_t* num_padded_rows, const int32_t* live_tokens, int32_t* active_experts,
    uint32_t rank) {
  __shared__ int warp_counts[8];
  __shared__ int first_row, active_before;
  if (blockDim.x != 256 || gridDim.x != LocalExperts) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > Capacity || rank >= Experts / LocalExperts) __trap();
  const int expert = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    int first = 0, active = 0;
    for (int e = 0; e < expert; ++e) {
      first += (counts[e] + Pad - 1) / Pad * Pad;
      active += counts[e] > 0;
    }
    first_row = first;
    active_before = active;
    offsets[expert] = first;
    const int count = counts[expert];
    if (count > 0) active_experts[1 + active] = expert;
    if (expert == LocalExperts - 1) {
      const int total = first + (count + Pad - 1) / Pad * Pad;
      offsets[LocalExperts] = total;
      *num_padded_rows = total;
    }
    if (expert == 0) {
      int all_active = 0;
      for (int e = 0; e < LocalExperts; ++e) all_active += counts[e] > 0;
      active_experts[0] = all_active;
    }
  }
  __syncthreads();
  const int start = first_row;
  const int routes = max(live, 0) * TopK;
  int written = 0;
  for (int chunk = 0; chunk < routes; chunk += 256) {
    const int route = chunk + threadIdx.x;
    const bool matches =
        route < routes && local_expert<Experts, LocalExperts>(ids[route], rank) == expert;
    const unsigned mask = __ballot_sync(0xffffffff, matches);
    if (lane == 0) warp_counts[warp] = __popc(mask);
    __syncthreads();
    int before = 0, in_chunk = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      before += w < warp ? warp_counts[w] : 0;
      in_chunk += warp_counts[w];
    }
    if (matches) sorted_rows[start + written + before + __popc(mask & ((1u << lane) - 1))] = route;
    written += in_chunk;
    __syncthreads();
  }
  const int end = start + (counts[expert] + Pad - 1) / Pad * Pad;
  for (int at = start + written + threadIdx.x; at < end; at += 256) sorted_rows[at] = -1;
}

}  // namespace chop::moe::local
