#pragma once

#include "route_local.cuh"

// The one-CTA decode route of `route_local.cuh` (`local::route_decode`)
// with its serial metadata steps spread over warps. Selection is the same
// `select_route_warp` call per live token, so IDs and weights are the same.
// The metadata is the same layout and the same values:
// - counts: each warp of 32 routes groups its lanes by local expert
//   (`__match_any_sync`), stores each expert's count in the group, and one
//   thread per expert adds the groups in order, keeping each group's
//   starting rank;
// - offsets, the padded total and the active list: one warp scans the padded
//   counts and the active flags over 32 experts at a time (`__shfl_up_sync`);
// - sorted rows: each route writes its own slot (the expert's offset, the
//   routes of earlier groups, then the lanes before it in its group), so an
//   expert's routes stay in ascending route order, and each expert thread
//   writes its padding.
// Every kernel reads the live count after `griddepcontrol.wait`, traps above
// its capacity, and uses no atomics; integer metadata is deterministic.
//
// `M` (lab-only) is a deliberate defect kernel-lab instantiates to show its
// checks fail on it; served entries take `none`, the unmodified kernel.
namespace chop::moe::local {

enum class ScanMutation : int {
  none = 0,
  descending_lanes = 1,  // routes of a group ranked in descending lane order
  reversed_groups = 2,   // group starts summed from the last group
  no_padding = 3,        // padding slots never written
  active_shift = 4,      // active experts one slot late
  stale_carry = 5,       // offsets of each 32-expert chunk start from zero
};

template<int Experts, int TopK, int LocalExperts, int Capacity, int Pad, class Score,
         ScanMutation M = ScanMutation::none>
__device__ void route_decode_scan(
    const float* logits, int32_t* ids, float* weights, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* num_padded_rows, const int32_t* live_tokens,
    int32_t* active_experts, uint32_t rank, Score score) {
  static_assert(Capacity >= 1 && Capacity <= 32 && Pad > 0);
  static_assert(LocalExperts > 0 && LocalExperts <= 1024 && Experts % LocalExperts == 0);
  constexpr int Threads = Capacity * 32 > round_up_32(LocalExperts)
      ? Capacity * 32 : round_up_32(LocalExperts);
  constexpr int Groups = (Capacity * TopK + 31) / 32;
  static_assert(Groups <= Threads / 32, "a warp per group of 32 routes");
  static_assert(Capacity * TopK <= 65535, "group starts fit 16 bits");
  __shared__ uint16_t group_start[Groups][LocalExperts];
  __shared__ int32_t expert_count[LocalExperts], expert_first[LocalExperts];
  if (blockDim.x != Threads || gridDim.x != 1) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > Capacity || rank >= Experts / LocalExperts) __trap();
  const int tokens = max(live, 0);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < Groups * LocalExperts; i += Threads)
    (&group_start[0][0])[i] = 0;
  if (warp < tokens) select_route_warp<Experts, TopK>(logits, warp, score, ids, weights);
  __syncthreads();
  const int routes = tokens * TopK;
  // Group `warp`'s routes by local expert; invalid routes get a key no
  // valid expert or other lane shares.
  int expert = -1, before = 0;
  if (warp < Groups) {
    const int r = warp * 32 + lane;
    expert = r < routes ? local_expert<Experts, LocalExperts>(ids[r], rank) : -1;
    const unsigned same = __match_any_sync(0xffffffffu, expert >= 0 ? expert : -1 - lane);
    const unsigned earlier = M == ScanMutation::descending_lanes
        ? same & ~((2u << lane) - 1u) : same & ((1u << lane) - 1u);
    before = __popc(earlier);
    if (expert >= 0 && before == 0) group_start[warp][expert] = uint16_t(__popc(same));
  }
  __syncthreads();
  // Each expert's counts per group become the routes of the groups before.
  for (int e = threadIdx.x; e < LocalExperts; e += Threads) {
    int count = 0;
#pragma unroll
    for (int i = 0; i < Groups; ++i) {
      const int g = M == ScanMutation::reversed_groups ? Groups - 1 - i : i;
      const int n = group_start[g][e];
      group_start[g][e] = uint16_t(count);
      count += n;
    }
    expert_count[e] = count;
    counts[e] = count;
  }
  __syncthreads();
  if (warp == 0) {
    int carry = 0, carry_active = 0;
    for (int base = 0; base < LocalExperts; base += 32) {
      const int e = base + lane;
      const int count = e < LocalExperts ? expert_count[e] : 0;
      const int padded = (count + Pad - 1) / Pad * Pad;
      const int active = count > 0;
      int p = padded, a = active;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int up_p = __shfl_up_sync(0xffffffffu, p, d);
        const int up_a = __shfl_up_sync(0xffffffffu, a, d);
        if (lane >= d) {
          p += up_p;
          a += up_a;
        }
      }
      if (e < LocalExperts) {
        const int first = (M == ScanMutation::stale_carry ? 0 : carry) + p - padded;
        expert_first[e] = first;
        offsets[e] = first;
        if (active)
          active_experts[1 + carry_active + a - (M == ScanMutation::active_shift ? 0 : 1)] = e;
      }
      carry += __shfl_sync(0xffffffffu, p, 31);
      carry_active += __shfl_sync(0xffffffffu, a, 31);
    }
    if (lane == 0) {
      offsets[LocalExperts] = carry;
      *num_padded_rows = carry;
      active_experts[0] = carry_active;
    }
  }
  __syncthreads();
  if (expert >= 0)
    sorted_rows[expert_first[expert] + group_start[warp][expert] + before] = warp * 32 + lane;
  if constexpr (M != ScanMutation::no_padding) {
    for (int e = threadIdx.x; e < LocalExperts; e += Threads) {
      const int end = expert_first[e] + (expert_count[e] + Pad - 1) / Pad * Pad;
      for (int at = expert_first[e] + expert_count[e]; at < end; ++at) sorted_rows[at] = -1;
    }
  }
}

}  // namespace chop::moe::local
