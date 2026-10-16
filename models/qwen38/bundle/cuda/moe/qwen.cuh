#pragma once

#include <stdint.h>
#include "moe/prefill_staging.cuh"
#include "common/sigmoid.cuh"

// Qwen3.8 Flash expert-sharded MoE routing (decision 0084). The router
// selects the top 10 of 512 experts per token from a softmax and
// renormalizes their weights. Each rank owns the kExperts experts [rank *
// kExperts, rank * kExperts + kExperts) and runs them for every token routed
// there; activations are replicated. The build defines kExperts,
// CHOP_QWEN38_LOCAL_EXPERTS: 256, 128 or 64 at TP2, TP4 or TP8. A routed row
// keeps its global route index `r = token * 10 + k`. This header holds the
// routes, the permutation and the combine; the expert GEMMs are instances
// of the grouped-MoE template (`grouped*.cuh`).
namespace chop::moe::qwen {

#ifndef CHOP_QWEN38_LOCAL_EXPERTS
#error "CHOP_QWEN38_LOCAL_EXPERTS is required"
#endif

constexpr int kGlobalExperts = CHOP_QWEN38_GLOBAL_EXPERTS;
constexpr int kExperts = CHOP_QWEN38_LOCAL_EXPERTS;
// A permutation tile's 256 threads cover the local experts, and the scans
// run whole warps of them.
static_assert(kExperts >= 32 && kExperts <= 256 && (kExperts & (kExperts - 1)) == 0);
constexpr int kTopK = CHOP_QWEN38_TOP_K;
constexpr int kDecodeBlock = 16;

__device__ __forceinline__ int local_expert(int32_t id, uint32_t rank) {
  return id >= 0 && id < kGlobalExperts && uint32_t(id) / kExperts == rank
      ? id % kExperts : -1;
}

__device__ __forceinline__ uint32_t ordered_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

__device__ __forceinline__ float from_ordered(uint32_t ordered) {
  return __uint_as_float(ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered);
}

// One warp selects a token's ten experts: the largest logits, ties to the
// lower expert. Lane k < 10 returns route k's expert and renormalized
// softmax weight; the softmax total cancels in the renormalization.
__device__ __forceinline__ void select_top10(const float* row, int& id, float& weight) {
  const int lane = threadIdx.x & 31;
  float scores[kGlobalExperts / 32];
#pragma unroll
  for (int j = 0; j < kGlobalExperts / 32; ++j) scores[j] = row[lane + 32 * j];
  float maximum = 0.f, chosen = 0.f;
  id = INT32_MAX;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int best_id = INT32_MAX;
#pragma unroll
    for (int j = 0; j < kGlobalExperts / 32; ++j)
      if (scores[j] > best) {
        best = scores[j];
        best_id = lane + 32 * j;
      }
    const uint32_t winning = __reduce_max_sync(0xffffffff, ordered_bits(best));
    best_id = __reduce_min_sync(
        0xffffffff, ordered_bits(best) == winning ? best_id : INT32_MAX);
#pragma unroll
    for (int j = 0; j < kGlobalExperts / 32; ++j)
      if (best_id == lane + 32 * j) scores[j] = -INFINITY;
    const float value = from_ordered(winning);
    if (k == 0) maximum = value;
    if (lane == k) {
      id = best_id;
      chosen = value;
    }
  }
  const float e = lane < kTopK ? expf(chosen - maximum) : 0.f;
  float total = e;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    total += __shfl_xor_sync(0xffffffff, total, offset);
  weight = e / total;
}

// Prefill router: one warp per token writes the global expert IDs and
// weights `[tokens, 10]`.
__device__ void router(const float* logits, int32_t* ids, float* weights,
                       const int32_t* live_tokens) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int token = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (token >= *live_tokens) return;
  int id;
  float weight;
  select_top10(logits + uint64_t(token) * kGlobalExperts, id, weight);
  const int lane = threadIdx.x & 31;
  if (lane < kTopK) {
    ids[uint64_t(token) * kTopK + lane] = id;
    weights[uint64_t(token) * kTopK + lane] = weight;
  }
}

// Exclusive CTA scan for up to 32 warps; warp_bases holds blockDim / 32.
__device__ __forceinline__ int block_exclusive_scan(int value, int* warp_bases) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int inclusive = value;
#pragma unroll
  for (int step = 1; step < 32; step <<= 1) {
    const int prior = __shfl_up_sync(0xffffffff, inclusive, step);
    if (lane >= step) inclusive += prior;
  }
  if (lane == 31) warp_bases[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int total = lane < warps ? warp_bases[lane] : 0;
    int prefix = total;
#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
      const int prior = __shfl_up_sync(0xffffffff, prefix, step);
      if (lane >= step) prefix += prior;
    }
    if (lane < warps) warp_bases[lane] = prefix - total;
  }
  __syncthreads();
  const int result = warp_bases[warp] + inclusive - value;
  __syncthreads();
  return result;
}

// Decode route: one CTA selects every token's experts, keeps this rank's
// routes and builds the stable, 16-row padded expert ranges. The active
// expert list is ascending; `active_experts[0]` is its length.
template<int TokensCapacity>
__device__ void route_decode(
    const float* logits, int32_t* ids, float* weights, int32_t* counts,
    int32_t* offsets, int32_t* sorted_rows, int32_t* num_padded_rows,
    const int32_t* live_tokens, int32_t* active_experts, uint32_t rank) {
  constexpr int Routes = TokensCapacity * kTopK;
  constexpr int Groups = (Routes + 31) / 32;
  __shared__ int32_t local_ids[Groups * 32];
  __shared__ int32_t group_counts[Groups][kExperts];
  __shared__ int32_t expert_first[kExperts];
  __shared__ int32_t warp_padded[kExperts / 32], warp_active[kExperts / 32];
  if (blockDim.x < kExperts || blockDim.x < Groups * 32) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int tokens = min(max(*live_tokens, 0), TokensCapacity);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < Groups * 32; i += blockDim.x) local_ids[i] = -1;
  for (int i = threadIdx.x; i < Groups * kExperts; i += blockDim.x)
    group_counts[i / kExperts][i % kExperts] = 0;
  __syncthreads();
  for (int token = warp; token < tokens; token += blockDim.x / 32) {
    int id;
    float weight;
    select_top10(logits + uint64_t(token) * kGlobalExperts, id, weight);
    if (lane < kTopK) {
      const int route = token * kTopK + lane;
      ids[route] = id;
      weights[route] = weight;
      local_ids[route] = local_expert(id, rank);
    }
  }
  __syncthreads();
  // Each 32-route group counts its routes per local expert with a match
  // mask, so a route's stable rank within its expert is a short sum.
  int route_expert = -1, rank_in_group = 0;
  if (threadIdx.x < Groups * 32) {
    route_expert = local_ids[threadIdx.x];
    const unsigned same = __match_any_sync(
        0xffffffff, route_expert >= 0 ? route_expert : -1 - lane);
    rank_in_group = __popc(same & ((1u << lane) - 1));
    if (route_expert >= 0 && rank_in_group == 0)
      group_counts[warp][route_expert] = __popc(same);
  }
  __syncthreads();
  int count = 0, padded = 0, active = 0, padded_scan = 0, active_scan = 0;
  if (threadIdx.x < kExperts) {
#pragma unroll
    for (int group = 0; group < Groups; ++group)
      count += group_counts[group][threadIdx.x];
    counts[threadIdx.x] = count;
    padded = (count + kDecodeBlock - 1) / kDecodeBlock * kDecodeBlock;
    active = count > 0;
    padded_scan = padded;
    active_scan = active;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous_padded = __shfl_up_sync(0xffffffff, padded_scan, delta);
      const int previous_active = __shfl_up_sync(0xffffffff, active_scan, delta);
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
  if (threadIdx.x < kExperts) {
    const int expert = threadIdx.x;
    int before_padded = 0, before_active = 0;
    for (int w = 0; w < warp; ++w) {
      before_padded += warp_padded[w];
      before_active += warp_active[w];
    }
    const int first = before_padded + padded_scan - padded;
    offsets[expert] = first;
    expert_first[expert] = first;
    if (active) active_experts[before_active + active_scan] = expert;
    if (expert == kExperts - 1) {
      offsets[kExperts] = before_padded + padded_scan;
      *num_padded_rows = before_padded + padded_scan;
      active_experts[0] = before_active + active_scan;
    }
    for (int at = first + count; at < first + padded; ++at) sorted_rows[at] = -1;
  }
  __syncthreads();
  if (threadIdx.x < Groups * 32 && route_expert >= 0) {
    int position = rank_in_group;
    for (int group = 0; group < warp; ++group)
      position += group_counts[group][route_expert];
    sorted_rows[expert_first[route_expert] + position] = threadIdx.x;
  }
}

// Prefill permutation over 256-route tiles, as the row engine's tile
// kernels, restricted to this rank's experts. Counting may use atomics
// because only totals are stored; scatter ranks each route stably.
__device__ void permute_count(const int32_t* ids, int32_t* tile_counts,
                              const int32_t* live_tokens, uint32_t rank) {
  ptx::griddep_launch_dependents();
  __shared__ int histogram[kExperts];
  ptx::griddep_wait();
  const uint32_t tile = blockIdx.x;
  const uint32_t routes = uint32_t(max(*live_tokens, 0)) * kTopK;
  if (tile * 256 >= routes) return;
  const uint32_t route = tile * 256 + threadIdx.x;
  const int expert = route < routes ? local_expert(ids[route], rank) : -1;
  if (threadIdx.x < kExperts) histogram[threadIdx.x] = 0;
  __syncthreads();
  if (expert >= 0) atomicAdd(&histogram[expert], 1);
  __syncthreads();
  if (threadIdx.x < kExperts)
    tile_counts[uint64_t(tile) * kExperts + threadIdx.x] = histogram[threadIdx.x];
}

// One CTA per expert scans its live tile counts; blockDim covers the tiles.
__device__ void permute_prefix(const int32_t* tile_counts, int32_t* counts,
                               int32_t* tile_offsets, const int32_t* live_tokens) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_bases[32];
  ptx::griddep_wait();
  const int expert = blockIdx.x;
  const int tile = threadIdx.x;
  const int live_tiles = (max(*live_tokens, 0) * kTopK + 255) / 256;
  if (live_tiles > int(blockDim.x)) __trap();
  const int count = tile < live_tiles ? tile_counts[uint64_t(tile) * kExperts + expert] : 0;
  const int before = block_exclusive_scan(count, warp_bases);
  if (tile < live_tiles) tile_offsets[uint64_t(tile) * kExperts + expert] = before;
  if (tile == int(blockDim.x) - 1) counts[expert] = before + count;
}

// kExperts threads: padded expert offsets, the active list and route padding.
template<int Block>
__device__ void permute_scan(const int32_t* counts, int32_t* offsets,
                             int32_t* sorted_rows, int32_t* num_padded_rows,
                             int32_t* active_experts) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_bases[kExperts / 32];
  ptx::griddep_wait();
  const int expert = threadIdx.x;
  const int count = counts[expert];
  const int padded = (count + Block - 1) / Block * Block;
  const int first = block_exclusive_scan(padded, warp_bases);
  const int before_active = block_exclusive_scan(count > 0, warp_bases);
  offsets[expert] = first;
  if (count > 0) active_experts[1 + before_active] = expert;
  if (expert == kExperts - 1) {
    offsets[kExperts] = first + padded;
    *num_padded_rows = first + padded;
    active_experts[0] = before_active + (count > 0);
  }
  for (int slot = first + count; slot < first + padded; ++slot) sorted_rows[slot] = -1;
}

__device__ void permute_scatter(const int32_t* ids, const int32_t* tile_offsets,
                                const int32_t* offsets, int32_t* sorted_rows,
                                const int32_t* live_tokens, uint32_t rank) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_counts[8][kExperts];
  ptx::griddep_wait();
  const uint32_t tile = blockIdx.x;
  const uint32_t routes = uint32_t(max(*live_tokens, 0)) * kTopK;
  if (tile * 256 >= routes) return;
  const uint32_t route = tile * 256 + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int e = lane; e < kExperts; e += 32) warp_counts[warp][e] = 0;
  __syncthreads();
  const int expert = route < routes ? local_expert(ids[route], rank) : -1;
  if (expert >= 0) atomicAdd(&warp_counts[warp][expert], 1);
  __syncthreads();
  const unsigned matches = __match_any_sync(0xffffffff, expert >= 0 ? expert : -1 - lane);
  if (expert < 0) return;
  int before_warp = 0;
  for (int w = 0; w < warp; ++w) before_warp += warp_counts[w][expert];
  const int before_lane = __popc(matches & ((1u << lane) - 1));
  sorted_rows[uint64_t(offsets[expert])
              + tile_offsets[uint64_t(tile) * kExperts + expert]
              + before_warp + before_lane] = int32_t(route);
}

// This rank's partial of the MoE output for one token: its routed rows
// (preweighted BF16, added in route order) plus
// `sigmoid(x . gate_weight) * shared`. 320 threads own eight columns each.
template<int Width>
__device__ void combine(const uint16_t* routed, const int32_t* ids,
                        const uint16_t* shared_out, const uint16_t* x,
                        const uint16_t* gate_weight, uint16_t* output,
                        const int32_t* live_tokens, uint32_t rank) {
  static_assert(Width % 256 == 0);
  constexpr int Threads = Width / 8;
  __shared__ float warp_sums[Threads / 32];
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int token = blockIdx.x;
  if (token >= *live_tokens) return;
  const int column = threadIdx.x * 8;
  const uint4 x_bits = *reinterpret_cast<const uint4*>(x + uint64_t(token) * Width + column);
  const uint4 w_bits = *reinterpret_cast<const uint4*>(gate_weight + column);
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(&x_bits);
  const uint32_t* ww = reinterpret_cast<const uint32_t*>(&w_bits);
  float dot = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const convert::Float2 a = convert::bf16x2_to_f32(xw[i]);
    const convert::Float2 b = convert::bf16x2_to_f32(ww[i]);
    dot = fmaf(a.lo, b.lo, dot);
    dot = fmaf(a.hi, b.hi, dot);
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    dot += __shfl_xor_sync(0xffffffff, dot, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = dot;
  float sum[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) sum[i] = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const int route = token * kTopK + k;
    if (local_expert(ids[route], rank) < 0) continue;
    const uint4 bits = *reinterpret_cast<const uint4*>(routed + uint64_t(route) * Width + column);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&bits);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const convert::Float2 v = convert::bf16x2_to_f32(words[i]);
      sum[2 * i] += v.lo;
      sum[2 * i + 1] += v.hi;
    }
  }
  __syncthreads();
  float logit = 0.f;
#pragma unroll
  for (int w = 0; w < Threads / 32; ++w) logit += warp_sums[w];
  // Below the sigmoid's tail the gated product is formed in F64.
  const float gate = chop::sigmoid::sigmoid(logit);
  auto combine = [&](float shared, float routed) {
    return logit >= chop::sigmoid::kTail ? fmaf(gate, shared, routed)
                                         : routed + chop::sigmoid::tail_times(shared, logit);
  };
  const uint4 shared_bits =
      *reinterpret_cast<const uint4*>(shared_out + uint64_t(token) * Width + column);
  const uint32_t* shared_words = reinterpret_cast<const uint32_t*>(&shared_bits);
  uint4 result;
  uint32_t* out = reinterpret_cast<uint32_t*>(&result);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const convert::Float2 s = convert::bf16x2_to_f32(shared_words[i]);
    out[i] = convert::bf16x2_from_f32({combine(s.lo, sum[2 * i]), combine(s.hi, sum[2 * i + 1])});
  }
  *reinterpret_cast<uint4*>(output + uint64_t(token) * Width + column) = result;
}

} // namespace chop::moe::qwen
