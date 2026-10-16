#pragma once

#include <stdint.h>

namespace chop::moe {

// One CTA owns the complete decode permutation. Either the standalone route
// entry or the router GEMV's final CTA calls this after logits are ready.
template<int Threads>
__device__ void route_decode_mimo_body(
    const float* logits, const float* bias, int32_t* ids, float* weights,
    int32_t* counts, int32_t* offsets, int32_t* sorted_rows,
    int32_t* num_padded_rows, int tokens,
    int32_t* active_experts, float scaling_factor) {
  static_assert(256 == 8 * 32);
  // Each thread below 256 owns one expert and at most one route.
  static_assert(Threads == 256);
  if (blockDim.x < 256) __trap();
  __shared__ int32_t selected_ids[256];
  __shared__ float selected_weights[256];
  __shared__ int32_t warp_padded[8], warp_active[8];
  // One route per thread of 256 holds at most 32 tokens of eight routes. A
  // batch above that must fail loudly, never lose tokens.
  if (tokens > 32) __trap();
  tokens = max(tokens, 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int token = warp; token < tokens; token += blockDim.x / 32) {
    // The bias only ranks experts; the route weight is the unbiased sigmoid,
    // as in the row router and the oracle.
    float scores[8], unbiased[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int expert = lane + j * 32;
      const float sigmoid = 1.f /
          (1.f + expf(-logits[uint64_t(token) * 256 + expert]));
      scores[j] = sigmoid + bias[expert];
      unbiased[j] = sigmoid;
    }
    float selected_sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float best_score = -INFINITY;
      int best_id = INT32_MAX;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int expert = lane + j * 32;
        if (scores[j] > best_score ||
            (scores[j] == best_score && expert < best_id)) {
          best_score = scores[j];
          best_id = expert;
        }
      }
      const uint32_t bits = __float_as_uint(best_score);
      const uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
      const uint32_t winning = __reduce_max_sync(0xffffffff, ordered);
      best_id = __reduce_min_sync(
          0xffffffff, ordered == winning ? best_id : INT32_MAX);
      float local_weight = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (best_id == lane + j * 32) {
          local_weight = unbiased[j];
          scores[j] = -INFINITY;
        }
      }
      const float chosen_weight =
          __shfl_sync(0xffffffff, local_weight, best_id & 31);
      if (lane == 0) {
        const int route = token * 8 + k;
        ids[route] = best_id;
        selected_ids[route] = best_id;
        selected_weights[route] = chosen_weight;
        selected_sum += chosen_weight;
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
        weights[token * 8 + k] =
            selected_weights[token * 8 + k] /
            (selected_sum + 1e-20f) * scaling_factor;
    }
  }
  __syncthreads();
  if (tokens == 1) {
    // The top eight IDs are unique for one token. Each expert computes its
    // rank among those IDs directly, avoiding the general prefix scan and
    // repeated routed-row scatter.
    for (int expert = threadIdx.x; expert < 256; expert += blockDim.x) {
      int rank = 0;
      int routed = -1;
#pragma unroll
      for (int route = 0; route < 8; ++route) {
        const int id = selected_ids[route];
        rank += id < expert;
        if (id == expert) routed = route;
      }
      counts[expert] = routed >= 0;
      offsets[expert] = rank * 16;
      if (routed >= 0) {
        active_experts[rank + 1] = expert;
        sorted_rows[rank * 16] = routed;
#pragma unroll
        for (int slot = 1; slot < 16; ++slot)
          sorted_rows[rank * 16 + slot] = -1;
      }
    }
    if (threadIdx.x == 0) {
      // Non-finite logits select no valid expert; such routes own no rows.
      int valid = 0;
#pragma unroll
      for (int route = 0; route < 8; ++route)
        valid += selected_ids[route] >= 0 && selected_ids[route] < 256;
      offsets[256] = valid * 16;
      *num_padded_rows = valid * 16;
      active_experts[0] = valid;
    }
    return;
  }
  // Up to 32 tokens give at most 256 routes, eight groups of 32. Each group
  // counts its routes per expert with a match mask, so an expert's count and
  // a route's stable rank within its expert are short sums over groups.
  __shared__ int32_t group_counts[8][256];
  __shared__ int32_t expert_first[256];
  const int routes = tokens * 8;
  for (int i = threadIdx.x; i < 8 * 256; i += blockDim.x) group_counts[i >> 8][i & 255] = 0;
  __syncthreads();
  int route_id = -1, rank_in_group = 0;
  for (int route = threadIdx.x; route < 256; route += blockDim.x) {
    const int id = route < routes ? selected_ids[route] : -1;
    const bool valid = id >= 0 && id < 256;
    // Invalid routes get a key no valid expert or other lane shares.
    const unsigned same = __match_any_sync(0xffffffff, valid ? id : -1 - lane);
    rank_in_group = __popc(same & ((1u << lane) - 1));
    if (valid && rank_in_group == 0) group_counts[route >> 5][id] = __popc(same);
    route_id = valid ? id : -1;
  }
  __syncthreads();
  int count_local[256 / Threads], padded_local[256 / Threads];
  int active_local[256 / Threads], padded_prefix[256 / Threads];
  int active_prefix[256 / Threads];
#pragma unroll
  for (int pass = 0; pass < 256 / Threads; ++pass) {
    const int expert = threadIdx.x + pass * Threads;
    if (expert >= 256) continue;
    int count = 0;
#pragma unroll
    for (int group = 0; group < 8; ++group) count += group_counts[group][expert];
    counts[expert] = count;
    count_local[pass] = count;
    padded_local[pass] = (count + 15) & ~15;
    active_local[pass] = count > 0;
    int padded_scan = padded_local[pass];
    int active_scan = active_local[pass];
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
    padded_prefix[pass] = padded_scan;
    active_prefix[pass] = active_scan;
    if (lane == 31) {
      warp_padded[expert >> 5] = padded_scan;
      warp_active[expert >> 5] = active_scan;
    }
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 256 / Threads; ++pass) {
    const int expert = threadIdx.x + pass * Threads;
    if (expert >= 256) continue;
    int before_padded = 0, before_active = 0;
    for (int w = 0; w < expert / 32; ++w) {
      before_padded += warp_padded[w];
      before_active += warp_active[w];
    }
    const int first = before_padded + padded_prefix[pass] - padded_local[pass];
    offsets[expert] = first;
    expert_first[expert] = first;
    if (active_local[pass])
      active_experts[before_active + active_prefix[pass]] = expert;
    if (expert == 255) {
      const int total = before_padded + padded_prefix[pass];
      offsets[256] = total;
      *num_padded_rows = total;
      active_experts[0] = before_active + active_prefix[pass];
    }
    // Padding follows the expert's routed rows.
    for (int at = first + count_local[pass]; at < first + padded_local[pass]; ++at)
      sorted_rows[at] = -1;
  }
  __syncthreads();
  // Routes of an expert keep ascending order: earlier groups, then earlier
  // lanes of the same group.
  for (int route = threadIdx.x; route < 256; route += blockDim.x) {
    if (route_id < 0) continue;
    int rank = rank_in_group;
    for (int group = 0; group < (route >> 5); ++group) rank += group_counts[group][route_id];
    sorted_rows[expert_first[route_id] + rank] = route;
  }
}

// Route outputs a warm-up pass writes and nobody reads: the most routes
// (32 tokens of eight), their 16-row padding, and room past the active list.
struct RouteSink {
  int32_t ids[256];
  float weights[256];
  int32_t counts[256];
  int32_t offsets[257];
  int32_t sorted_rows[256 + 256 * 15];
  int32_t num_padded_rows;
  int32_t active_experts[1024 + 16];
};

__device__ void route_decode_mimo(
    const float* logits, const float* bias, int32_t* ids, float* weights,
    int32_t* counts, int32_t* offsets, int32_t* sorted_rows,
    int32_t* num_padded_rows, const int32_t* live_tokens,
    int32_t* active_experts, float scaling_factor) {
  ptx::griddep_launch_dependents();
  // The route starts under PDL while the router GEMM still runs, and its
  // DRAM-cold bias and instruction-cold code cost microseconds after the
  // wait. Before the wait it stages the bias (a model constant) in shared
  // memory and runs the whole route once on the logits buffer as it stands,
  // writing only a shared sink. The pass after the wait is the same code on
  // the same bias values, so its outputs are unchanged.
  __shared__ float bias_shared[256];
  __shared__ RouteSink sink;
  for (int expert = threadIdx.x; expert < 256; expert += blockDim.x)
    bias_shared[expert] = bias[expert];
  __syncthreads();
  route_decode_mimo_body<256>(logits, bias_shared, sink.ids, sink.weights,
      sink.counts, sink.offsets, sink.sorted_rows, &sink.num_padded_rows,
      *live_tokens, sink.active_experts, scaling_factor);
  __syncthreads();
  // Live counts are never negative: this store keeps the warm-up pass from
  // being removed as dead code without ever running.
  if (*live_tokens < 0 && threadIdx.x < 8)
    ids[threadIdx.x] = sink.ids[threadIdx.x] + sink.counts[threadIdx.x] +
        sink.offsets[threadIdx.x] + sink.sorted_rows[threadIdx.x * 16] +
        sink.active_experts[threadIdx.x] + sink.num_padded_rows +
        int32_t(sink.weights[threadIdx.x]);
  ptx::griddep_wait();
  route_decode_mimo_body<256>(logits, bias_shared, ids, weights, counts,
      offsets, sorted_rows, num_padded_rows, *live_tokens, active_experts,
      scaling_factor);
}

} // namespace chop::moe
