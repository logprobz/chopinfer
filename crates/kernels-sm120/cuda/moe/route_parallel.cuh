#pragma once

#include <stdint.h>
#include <math.h>
#include "../common/ptx_control.cuh"

namespace chop::moe {

// Policies share one interface: transform, ranking, selected weight, and
// normalization. Reduction order and -inf tie handling are explicit because
// the qualified softmax and biased sigmoid routes have different contracts.
struct Softmax {
  static constexpr bool serial_sum = false, select_negative_infinity = false;
  __device__ float transform(float logit) const { return logit; }
  __device__ float rank(float value, int) const { return value; }
  __device__ float selected(float, float chosen, float maximum) const {
    return expf(chosen - maximum);
  }
  __device__ float normalize(float weight, float sum) const { return weight / sum; }
};
struct SigmoidBias {
  static constexpr bool serial_sum = true, select_negative_infinity = true;
  const float* bias;
  float scaling_factor;
  __device__ float transform(float logit) const { return 1.f / (1.f + expf(-logit)); }
  __device__ float rank(float value, int expert) const { return value + bias[expert]; }
  __device__ float selected(float unbiased, float, float) const { return unbiased; }
  __device__ float normalize(float weight, float sum) const {
    return weight / (sum + 1e-20f) * scaling_factor;
  }
};
struct GroupedSoftmax : Softmax {
  static constexpr bool serial_sum = true, select_negative_infinity = true;
};

__device__ __forceinline__ uint32_t ordered_route_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ __forceinline__ float route_from_ordered(uint32_t ordered) {
  return __uint_as_float(ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered);
}

// Optional grouping keeps the groups with the largest sum of their best two
// ranks, with ascending group ties. The scan order mirrors row/router.cuh.
template<int Experts, int Groups, int Take>
__device__ void mask_route_groups(float (&scores)[Experts / 32]) {
  static_assert(Groups > 0 && Experts % Groups == 0 && Take > 0 && Take <= Groups);
  if constexpr (Take < Groups) {
    constexpr int Size = Experts / Groups;
    float group_scores[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
      float first = -INFINITY, second = -INFINITY;
#pragma unroll
      for (int expert = group * Size; expert < (group + 1) * Size; ++expert) {
        const float value = __shfl_sync(0xffffffff, scores[expert / 32], expert % 32);
        if (value > first) { second = first; first = value; }
        else if (value > second) second = value;
      }
      group_scores[group] = first + (Size > 1 ? second : 0.f);
    }
#pragma unroll
    for (int i = 0; i < Take; ++i) {
      int best = 0;
      float best_score = group_scores[0];
#pragma unroll
      for (int group = 1; group < Groups; ++group) {
        if (group_scores[group] > best_score) {
          best = group;
          best_score = group_scores[group];
        }
      }
#pragma unroll
      for (int group = 0; group < Groups; ++group)
        if (group == best) group_scores[group] = -INFINITY;
    }
#pragma unroll
    for (int j = 0; j < Experts / 32; ++j) {
      const int owner = ((threadIdx.x & 31) + j * 32) / Size;
      bool retained = false;
#pragma unroll
      for (int group = 0; group < Groups; ++group)
        retained |= owner == group && group_scores[group] == -INFINITY;
      if (!retained) scores[j] = -INFINITY;
    }
  }
}

template<int Experts, int TopK, int Groups = 1, int Take = 1, class Score>
__device__ void select_route_warp(const float* logits, int token, Score score,
                                  int32_t* ids, float* weights) {
  static_assert(Experts % 32 == 0 && TopK > 0 && TopK <= 32);
  const int lane = threadIdx.x & 31;
  float scores[Experts / 32], unbiased[Experts / 32];
#pragma unroll
  for (int j = 0; j < Experts / 32; ++j) {
    const int expert = lane + j * 32;
    unbiased[j] = score.transform(logits[uint64_t(token) * Experts + expert]);
    scores[j] = score.rank(unbiased[j], expert);
  }
  mask_route_groups<Experts, Groups, Take>(scores);
  int id = INT32_MAX;
  float weight = 0.f, maximum = 0.f, selected_sum = 0.f;
#pragma unroll
  for (int k = 0; k < TopK; ++k) {
    float best = -INFINITY;
    int best_id = INT32_MAX;
#pragma unroll
    for (int j = 0; j < Experts / 32; ++j) {
      const int expert = lane + 32 * j;
      if (scores[j] > best || (Score::select_negative_infinity &&
          scores[j] == best && expert < best_id)) {
        best = scores[j];
        best_id = expert;
      }
    }
    const uint32_t winning = __reduce_max_sync(0xffffffff, ordered_route_bits(best));
    best_id = __reduce_min_sync(0xffffffff,
        ordered_route_bits(best) == winning ? best_id : INT32_MAX);
    float local_weight = 0.f;
#pragma unroll
    for (int j = 0; j < Experts / 32; ++j) {
      if (best_id == lane + 32 * j) {
        local_weight = unbiased[j];
        scores[j] = -INFINITY;
      }
    }
    // A non-finite row can select INT32_MAX. Policies never receive that
    // index: gather only matched values, using a valid shuffle lane.
    const float chosen_weight = __shfl_sync(0xffffffff, local_weight,
        best_id < Experts ? best_id & 31 : 0);
    const float chosen = route_from_ordered(winning);
    if (k == 0) maximum = chosen;
    const float selected = Score::serial_sum
        ? score.selected(chosen_weight, chosen, maximum) : chosen;
    if constexpr (Score::serial_sum) {
      if (lane == 0) selected_sum += selected;
    }
    if (lane == k) { id = best_id; weight = selected; }
  }
  if constexpr (Score::serial_sum) {
    selected_sum = __shfl_sync(0xffffffff, selected_sum, 0);
  } else {
    weight = lane < TopK ? score.selected(0.f, weight, maximum) : 0.f;
    selected_sum = weight;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      selected_sum += __shfl_xor_sync(0xffffffff, selected_sum, offset);
  }
  if (lane < TopK) {
    ids[token * TopK + lane] = id;
    weights[token * TopK + lane] = score.normalize(weight, selected_sum);
  }
}

// All prepared CTAs publish rows, including empty graph replays. The last
// builds stable local ranges. Wide capacities use eight row warps per CTA;
// narrow capacities use one. All expert metadata uses this same builder.
template<int Experts, int TopK, int LocalExperts, int Capacity, int Pad = 16,
         int Groups = 1, int Take = 1, class Score>
__device__ void route_parallel(
    const float* logits, int32_t* ids, float* weights, int32_t* counts,
    int32_t* offsets, int32_t* sorted_rows, int32_t* num_padded_rows,
    const int32_t* live_tokens, int32_t* active_experts, uint32_t rank,
    uint32_t* ticket, Score score) {
  static_assert(Capacity > 0 && Capacity <= 256 && Pad > 0);
  static_assert(LocalExperts >= 32 && LocalExperts <= 512 && LocalExperts % 32 == 0);
  static_assert(Experts % LocalExperts == 0);
  constexpr bool Wide = Capacity > 32;
  constexpr int RowsPerCta = Wide ? 8 : 1;
  constexpr int kCtas = (Capacity + RowsPerCta - 1) / RowsPerCta;
  constexpr int kGroups = (Capacity * TopK + 31) / 32;
  constexpr int Threads = Wide ? 1024 : (LocalExperts > kGroups * 32 ? LocalExperts : kGroups * 32);
  constexpr int Warps = Threads / 32;
  constexpr int kGroupsPerWarp = (kGroups + Warps - 1) / Warps;
  constexpr int kParts = Wide ? 1024 / LocalExperts : 1;
  constexpr int kGroupsPerPart = (kGroups + kParts - 1) / kParts;
  static_assert(!Wide || 1024 % LocalExperts == 0);
  __shared__ uint16_t group_counts[kGroups][LocalExperts];
  __shared__ int32_t part_base[kParts][LocalExperts];
  __shared__ int32_t expert_first[LocalExperts], expert_count[LocalExperts];
  __shared__ int32_t warp_padded[LocalExperts / 32], warp_active[LocalExperts / 32];
  __shared__ uint32_t last;
  if (blockDim.x != Threads || gridDim.x != kCtas) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (*live_tokens > Capacity || rank >= Experts / LocalExperts) __trap();
  const int tokens = max(*live_tokens, 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // Warps without a row clear the group table while the row warps select,
  // in case this CTA arrives last; with no spare warp every thread clears.
  constexpr int kClearFirst = Warps > RowsPerCta ? RowsPerCta * 32 : 0;
  static_assert(LocalExperts % 2 == 0);
  if (threadIdx.x >= kClearFirst) {
    uint32_t* words = reinterpret_cast<uint32_t*>(&group_counts[0][0]);
    for (int i = threadIdx.x - kClearFirst; i < kGroups * LocalExperts / 2;
         i += Threads - kClearFirst)
      words[i] = 0;
  }
  if (warp < RowsPerCta) {
    const int row = blockIdx.x * RowsPerCta + warp;
    if (row < tokens)
      select_route_warp<Experts, TopK, Groups, Take>(logits, row, score, ids, weights);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(ticket) : "memory");
    // The ticket starts every launch at zero: it is zeroed at create, only
    // this route writes it, its launches are serialized on one stream, and
    // a launch that does not finish ends the context. An arrival of at
    // least kCtas therefore means corrupted state. This check does not
    // catch a corrupted count below kCtas: the last CTA resets the count
    // on election, so a late CTA can arrive after the reset. Detecting that
    // in the kernel needs the last CTA to wait for every arrival, which
    // measured about 6 us per launch in the parallel-route benchmark.
    if (arrived >= kCtas) __trap();
    last = arrived == kCtas - 1;
    // Every CTA has arrived, so the last one resets the replay state now,
    // off the metadata's critical path.
    if (last) *ticket = 0;
  }
  __syncthreads();
  if (!last) return;
  const int routes = tokens * TopK;
  int route_id[kGroupsPerWarp], rank_in_group[kGroupsPerWarp];
#pragma unroll
  for (int i = 0; i < kGroupsPerWarp; ++i) {
    const int group = warp + i * Warps;
    const int route = group * 32 + lane;
    const int global_id = group < kGroups && route < routes ? __ldcg(ids + route) : -1;
    const int id = global_id >= 0 && global_id < Experts && uint32_t(global_id) / LocalExperts == rank
        ? global_id % LocalExperts : -1;
    const bool valid = id >= 0 && id < LocalExperts;
    // Invalid routes get a key no valid expert or other lane shares.
    const unsigned same = __match_any_sync(0xffffffff, valid ? id : -1 - lane);
    rank_in_group[i] = __popc(same & ((1u << lane) - 1));
    if (valid && rank_in_group[i] == 0) group_counts[group][id] = __popc(same);
    route_id[i] = valid ? id : -1;
  }
  __syncthreads();
  const int expert = threadIdx.x % LocalExperts;
  const int part = threadIdx.x / LocalExperts;
  if (part < kParts) {
  int sum = 0;
#pragma unroll
  for (int g = part * kGroupsPerPart; g < min((part + 1) * kGroupsPerPart, kGroups); ++g) {
    const int count = group_counts[g][expert];
    group_counts[g][expert] = sum;
    sum += count;
  }
  part_base[part][expert] = sum;
  }
  __syncthreads();
  // One thread per local expert scans the part totals.
  int count = 0, padded = 0, padded_scan = 0, active_scan = 0;
  if (part == 0) {
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      const int total = part_base[p][expert];
      part_base[p][expert] = count;
      count += total;
    }
    counts[expert] = count;
    expert_count[expert] = count;
    padded = (count + Pad - 1) / Pad * Pad;
    padded_scan = padded;
    active_scan = count > 0;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous_padded =
          __shfl_up_sync(0xffffffff, padded_scan, delta);
      const int previous_active =
          __shfl_up_sync(0xffffffff, active_scan, delta);
      if (lane >= delta) {
        padded_scan += previous_padded;
        active_scan += previous_active;
      }
    }
    if (lane == 31) {
      warp_padded[warp] = padded_scan;
      warp_active[warp] = active_scan;
    }
  }
  __syncthreads();
  if (part == 0) {
    int before_padded = 0, before_active = 0;
    for (int w = 0; w < warp; ++w) {
      before_padded += warp_padded[w];
      before_active += warp_active[w];
    }
    const int first = before_padded + padded_scan - padded;
    offsets[expert] = first;
    expert_first[expert] = first;
    if (count > 0) active_experts[before_active + active_scan] = expert;
    if (expert == LocalExperts - 1) {
      const int total = before_padded + padded_scan;
      offsets[LocalExperts] = total;
      *num_padded_rows = total;
      active_experts[0] = before_active + active_scan;
    }
  }
  __syncthreads();
  if (part < kParts) {
    // Padding follows the expert's routed rows; four threads share it.
    const int first = expert_first[expert];
    const int routed = expert_count[expert];
    const int end = first + ((routed + Pad - 1) / Pad * Pad);
    for (int at = first + routed + part; at < end; at += kParts)
      sorted_rows[at] = -1;
  }
  // Routes of an expert keep ascending order: earlier parts, earlier groups
  // of the part, then earlier lanes of the same group.
#pragma unroll
  for (int i = 0; i < kGroupsPerWarp; ++i) {
    const int id = route_id[i];
    if (id < 0) continue;
    const int group = warp + i * Warps;
    sorted_rows[expert_first[id] + part_base[group / kGroupsPerPart][id] +
        group_counts[group][id] + rank_in_group[i]] = group * 32 + lane;
  }
}

} // namespace chop::moe
