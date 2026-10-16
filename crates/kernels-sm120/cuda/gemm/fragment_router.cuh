#pragma once

#include "fragment_decode.cuh"

namespace chop::gemm {

// Route metadata for up to 32 tokens from the selected expert IDs, in the
// layout of chop::moe::route_decode_mimo_body: per-expert counts, offsets
// padded to 16 rows, the active expert list in expert order, and sorted rows
// in route order with -1 padding. One thread per expert (256 threads).
__device__ __forceinline__ void route_metadata_256(
    const int32_t* selected_ids, int routes, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* num_padded_rows, int32_t* active_experts) {
  __shared__ int32_t warp_padded[8], warp_active[8];
  const int lane = threadIdx.x & 31;
  const int expert = threadIdx.x;
  __shared__ int32_t warp_counts[8][256];
#pragma unroll
  for (int w = 0; w < 8; ++w) warp_counts[w][expert] = 0;
  __syncthreads();
  if (expert < routes) atomicAdd(&warp_counts[expert / 32][selected_ids[expert]], 1);
  __syncthreads();
  int count = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) count += warp_counts[w][expert];
  counts[expert] = count;
  const int padded = (count + 15) & ~15;
  const int active = count > 0;
  int padded_scan = padded, active_scan = active;
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
    warp_padded[expert >> 5] = padded_scan;
    warp_active[expert >> 5] = active_scan;
  }
  __syncthreads();
  int before_padded = 0, before_active = 0;
  for (int w = 0; w < expert / 32; ++w) {
    before_padded += warp_padded[w];
    before_active += warp_active[w];
  }
  const int first = before_padded + padded_scan - padded;
  offsets[expert] = first;
  if (active) {
    active_experts[before_active + active_scan] = expert;
    for (int at = first + count; at < first + padded; ++at) sorted_rows[at] = -1;
  }
  if (expert == 255) {
    offsets[256] = before_padded + padded_scan;
    *num_padded_rows = before_padded + padded_scan;
    active_experts[0] = before_active + active_scan;
  }
  __syncthreads();
  const uint32_t live_mask = __ballot_sync(0xffffffff, expert < routes);
  if (expert < routes) {
    const int destination = selected_ids[expert];
    const uint32_t peers = __match_any_sync(live_mask, destination);
    int rank = __popc(peers & ((1u << lane) - 1));
    for (int w = 0; w < expert / 32; ++w) rank += warp_counts[w][destination];
    sorted_rows[offsets[destination] + rank] = expert;
  }
}

// Orders scores as route_decode_mimo_body does: higher score first, lower
// expert ID on equal scores.
__device__ __forceinline__ uint32_t route_key(float score) {
  const uint32_t bits = __float_as_uint(score);
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

// Top-8 selection and route metadata for up to 32 tokens, bit-identical
// to chop::moe::route_decode_mimo_body on the same logits (in shared
// memory), with 256 threads. One token ranks packed candidate keys;
// more tokens use one warp per token with integer maximum reductions.
template<int Tokens>
__device__ void fused_route_tail(
    const float* logits, const float* bias, uint32_t tokens, int32_t* ids,
    float* weights, int32_t* counts, int32_t* offsets, int32_t* sorted_rows,
    int32_t* num_padded_rows, int32_t* active_experts, float scaling_factor) {
  __shared__ int32_t selected_ids[Tokens * 8];
  // A batch above the token tile must fail loudly, never lose tokens.
  if (tokens > uint32_t(Tokens)) __trap();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (tokens == 1) {
    // Key: score order in the high word, then the lower expert ID, so
    // keys are unique. Each warp sorts its 32 keys with a shuffle bitonic
    // network and keeps its top eight; the global top eight are among
    // those 64 candidates, ranked by four threads each.
    __shared__ __align__(16) uint64_t candidates[64];
    __shared__ float chosen[8];
    const int expert = threadIdx.x;
    const float sigmoid = 1.f / (1.f + expf(-logits[expert * (Tokens + 1)]));
    const float score = sigmoid + bias[expert];
    uint64_t key = uint64_t(route_key(score)) << 32 | (255 - expert);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
#pragma unroll
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        const uint64_t other = __shfl_xor_sync(0xffffffff, key, stride);
        const bool descending = (lane & size) == 0;
        const bool lower = (lane & stride) == 0;
        key = lower == descending ? max(key, other) : min(key, other);
      }
    if (lane < 8) candidates[warp * 8 + lane] = key;
    __syncthreads();
    const int candidate = threadIdx.x >> 2;
    const uint64_t mine = candidates[candidate];
    int rank = 0;
#pragma unroll
    for (int other = (threadIdx.x & 3) * 16; other < (threadIdx.x & 3) * 16 + 16;
         other += 2) {
      const ulonglong2 pair =
          *reinterpret_cast<const ulonglong2*>(candidates + other);
      rank += (pair.x > mine) + (pair.y > mine);
    }
    rank += __shfl_xor_sync(0xffffffff, rank, 1);
    rank += __shfl_xor_sync(0xffffffff, rank, 2);
    if ((threadIdx.x & 3) == 0 && rank < 8) {
      const int selected = 255 - int(mine & 0xff);
      ids[rank] = selected;
      selected_ids[rank] = selected;
    }
    __syncthreads();
    // The selected expert's own thread supplies its weight.
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (selected_ids[k] == expert) chosen[k] = sigmoid;
    __syncthreads();
    if (threadIdx.x < 8) {
      float selected_sum = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) selected_sum += chosen[k];
      weights[threadIdx.x] =
          chosen[threadIdx.x] / (selected_sum + 1e-20f) * scaling_factor;
    }
    // The eight IDs are unique: each expert's rank among them fixes its
    // padded 16-row range.
    int below = 0, routed = -1;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      below += selected_ids[k] < expert;
      if (selected_ids[k] == expert) routed = k;
    }
    counts[expert] = routed >= 0;
    offsets[expert] = below * 16;
    if (routed >= 0) {
      active_experts[below + 1] = expert;
      sorted_rows[below * 16] = routed;
#pragma unroll
      for (int slot = 1; slot < 16; ++slot)
        sorted_rows[below * 16 + slot] = -1;
    }
    if (threadIdx.x == 0) {
      offsets[256] = 128;
      *num_padded_rows = 128;
      active_experts[0] = 8;
    }
    return;
  }
  for (uint32_t token = warp; token < tokens; token += 8) {
    float sigmoids[8];
    uint32_t keys[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int expert = lane + j * 32;
      sigmoids[j] = 1.f /
          (1.f + expf(-logits[expert * (Tokens + 1) + token]));
      keys[j] = route_key(sigmoids[j] + bias[expert]);
    }
    float chosen[8];
    float selected_sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint32_t local_key = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) local_key = max(local_key, keys[j]);
      const uint32_t winning = __reduce_max_sync(0xffffffff, local_key);
      int local_id = INT32_MAX;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (keys[j] == winning) local_id = min(local_id, lane + j * 32);
      const int best_id = __reduce_min_sync(0xffffffff, local_id);
      float local_weight = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (best_id == lane + j * 32) {
          local_weight = sigmoids[j];
          keys[j] = 0;
        }
      }
      chosen[k] = __shfl_sync(0xffffffff, local_weight, best_id & 31);
      if (lane == 0) {
        ids[token * 8 + k] = best_id;
        selected_ids[token * 8 + k] = best_id;
        selected_sum += chosen[k];
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
        weights[token * 8 + k] =
            chosen[k] / (selected_sum + 1e-20f) * scaling_factor;
    }
  }
  __syncthreads();
  route_metadata_256(selected_ids, int(tokens) * 8, counts, offsets,
      sorted_rows, num_padded_rows, active_experts);
}

// K splits per 16-row router strip; the planner's grid target matches.
constexpr int kRouterSplits = 4;

// Each CTA handles one 16-row router strip and a short K slice.
// Four warps share that K slice, then publish one F32 partial. The last CTA
// for a strip sums its split partials in order and resets the semaphore.
template<int TileN, bool FusedRoute>
__device__ void run_fragment_router_impl(
    const uint8_t* weight, const uint16_t* activation,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const int32_t* live_tokens, const float* bias, int32_t* topk_ids,
    float* topk_weights, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* num_padded_rows,
    int32_t* active_experts, float scaling_factor) {
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  (void)scales;
  (void)residual;
  constexpr int kSplits = kRouterSplits;
  if (n != 256 || k != 4096 || splits != kSplits || !output_f32) return;
  constexpr int warps = 8;
  constexpr int tile_rows = 16;
  constexpr int k_tiles = 4096 / 16;
  // Each warp owns consecutive K blocks of its split.
  constexpr int warp_blocks = 32 / (kSplits * warps);
  const uint32_t strip = blockIdx.x / kSplits;
  const uint32_t split = blockIdx.x % kSplits;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t first_block = (split * warps + warp) * warp_blocks;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* warp_partials = reinterpret_cast<float*>(storage);
  __shared__ uint32_t last;
  __shared__ float bias_shared[FusedRoute ? 256 : 1];

  ptx::A4 weights[warp_blocks * 8];
  ptx::griddep_launch_dependents();
  {
    const uint8_t* lane_base = weight +
        (uint64_t(strip) * k_tiles + first_block * 8) * 512 + lane * 16;
#pragma unroll
    for (int group = 0; group < warp_blocks * 2; ++group)
      load_weight_group4(lane_base + group * 2048, weights[group * 4],
          weights[group * 4 + 1], weights[group * 4 + 2],
          weights[group * 4 + 3]);
  }
  if constexpr (FusedRoute) {
    // The bias is a model constant, so it loads before the PDL wait.
    for (uint32_t expert = threadIdx.x; expert < 256; expert += blockDim.x)
      bias_shared[expert] = bias[expert];
  }
  ptx::griddep_wait();
  // Rows above the tile stay unread here; the reducing CTA traps on them
  // before route selection.
  if constexpr (FusedRoute)
    live_m = uint32_t(min(max(*live_tokens, 0), TileN));

  ptx::D4 acc0{}, acc1{}, acc2{}, acc3{};
#pragma unroll
  for (int local = 0; local < warp_blocks; ++local) {
    const uint32_t kb = first_block + local;
    ptx::B2 activation_b[8 * (TileN / 8)];
#pragma unroll
    for (int step = 0; step < 8; ++step) {
#pragma unroll
      for (int token_tile = 0; token_tile < TileN / 8; ++token_tile) {
        const uint32_t token = token_tile * 8 + (lane >> 2);
        const uint32_t lane4 = lane & 3;
        ptx::B2& b = activation_b[step * (TileN / 8) + token_tile];
        if (token < live_m) {
          const auto* src = reinterpret_cast<const uint32_t*>(
              activation + uint64_t(token) * k + kb * 128 +
              step * 16 + lane4 * 2);
          b.v[0] = src[0];
          b.v[1] = src[4];
        } else {
          b.v[0] = 0;
          b.v[1] = 0;
        }
      }
    }
    ptx::D4 unused0{}, unused1{}, unused2{}, unused3{};
#define CHOP_ROUTER_MMA(SUB)                                               \
    consume_weight_fragment<WeightKind::bf16, TileN, SUB, true>(            \
        weights[local * 8 + SUB], nullptr, acc0, acc1, acc2, acc3,         \
        unused0, unused1, unused2, unused3, activation_b)
    CHOP_ROUTER_MMA(0);
    CHOP_ROUTER_MMA(1);
    CHOP_ROUTER_MMA(2);
    CHOP_ROUTER_MMA(3);
    CHOP_ROUTER_MMA(4);
    CHOP_ROUTER_MMA(5);
    CHOP_ROUTER_MMA(6);
    CHOP_ROUTER_MMA(7);
#undef CHOP_ROUTER_MMA
  }

  const uint32_t group = lane >> 2;
  const uint32_t lane4 = lane & 3;
#define CHOP_ROUTER_WRITE(I, ACC)                                           \
  if constexpr (TileN / 8 > I) {                                            \
    for (int v = 0; v < 4; ++v) {                                            \
      const uint32_t row = group + (v / 2) * 8;                             \
      const uint32_t token = I * 8 + lane4 * 2 + (v % 2);                   \
      warp_partials[(warp * tile_rows + row) * TileN + token] = ACC.v[v];   \
    }                                                                       \
  }
  CHOP_ROUTER_WRITE(0, acc0)
  CHOP_ROUTER_WRITE(1, acc1)
  CHOP_ROUTER_WRITE(2, acc2)
  CHOP_ROUTER_WRITE(3, acc3)
#undef CHOP_ROUTER_WRITE
  __syncthreads();

  const uint64_t split_base = (uint64_t(strip) * kSplits + split) *
      tile_rows * TileN;
  for (uint32_t index = threadIdx.x; index < tile_rows * TileN;
       index += blockDim.x) {
    if (index % TileN >= live_m) continue;
    float value = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < warps; ++source_warp)
      value += warp_partials[(source_warp * tile_rows) * TileN + index];
    partial[split_base + index] = value;
  }
  __syncthreads();
  // Small fused buckets reduce in the grid's last CTA. Other buckets
  // distribute reduction across the last split of each expert strip.
  if (threadIdx.x == 0) {
    // The release publishes this CTA's partial (ordered by the barrier);
    // the acquire lets the last CTA read every other partial.
    uint32_t* counter = &semaphores[FusedRoute && TileN <= 8 ? 0 : strip];
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(counter) : "memory");
    last = arrived == (FusedRoute && TileN <= 8 ? gridDim.x : kSplits) - 1;
    if (last) *counter = 0;
  }
  __syncthreads();
  const auto split_partial = [&](uint32_t row, uint32_t token, int s) {
    return partial + ((uint64_t(row / tile_rows) * kSplits + s) * tile_rows +
        row % tile_rows) * TileN + token;
  };
  // Sums one output over its strip's splits in split order, with every
  // split's load in flight before the first add.
  const auto reduce = [&](uint32_t row, uint32_t token) {
    float parts[kSplits];
#pragma unroll
    for (int s = 0; s < kSplits; ++s)
      parts[s] = __ldcg(split_partial(row, token, s));
    float value = 0.0f;
#pragma unroll
    for (int s = 0; s < kSplits; ++s) value += parts[s];
    return value;
  };
  if constexpr (!FusedRoute) {
    if (!last) return;
    for (uint32_t index = threadIdx.x; index < tile_rows * TileN;
         index += blockDim.x) {
      const uint32_t token = index % TileN;
      if (token >= live_m) continue;
      static_cast<float*>(output)[uint64_t(token) * n + strip * tile_rows +
          index / TileN] = reduce(strip * tile_rows + index / TileN, token);
    }
    if (threadIdx.x == 0) semaphores[strip] = 0;
  } else {
    if (!last) return;
    // Larger buckets let the last split of each strip reduce in parallel.
    // A second ticket publishes the complete logits before route selection.
    if constexpr (TileN > 8) {
      for (uint32_t index = threadIdx.x; index < tile_rows * TileN;
           index += blockDim.x) {
        const uint32_t token = index % TileN;
        if (token < live_m)
          static_cast<float*>(output)[uint64_t(token) * n + strip * tile_rows +
              index / TileN] = reduce(strip * tile_rows + index / TileN, token);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t arrived;
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
            : "=r"(arrived) : "l"(&semaphores[16]) : "memory");
        last = arrived == 15;
        if (last) semaphores[16] = 0;
      }
      __syncthreads();
      if (!last) return;
    }
    // Padding makes both token-major writes and expert-major reads conflict free.
    __shared__ float logits_shared[(TileN + 1) * 256];
    if constexpr (TileN > 8) {
      for (uint32_t index = threadIdx.x; index < live_m * 256; index += blockDim.x)
        logits_shared[(index % 256) * (TileN + 1) + index / 256] =
            __ldcg(static_cast<float*>(output) + index);
    } else {
      if (live_m <= 4) {
        for (uint32_t token = 0; token < live_m; ++token) {
          const float value = reduce(threadIdx.x, token);
          static_cast<float*>(output)[uint64_t(token) * n + threadIdx.x] = value;
          logits_shared[threadIdx.x * (TileN + 1) + token] = value;
        }
      } else {
        for (uint32_t index = threadIdx.x; index < 256 * TileN; index += blockDim.x) {
          const uint32_t token = index % TileN;
          if (token < live_m)
            logits_shared[(index / TileN) * (TileN + 1) + token] = reduce(index / TileN, token);
        }
      }
    }
    __syncthreads();
    if constexpr (TileN <= 8) {
      if (live_m > 4)
        for (uint32_t index = threadIdx.x; index < live_m * 256; index += blockDim.x)
          static_cast<float*>(output)[index] =
              logits_shared[(index % 256) * (TileN + 1) + index / 256];
    }
    // A batch above the token tile must fail loudly, never lose tokens.
    // Checking here, once in the reducing CTA, keeps the GEMV's registers.
    if (*live_tokens > TileN) __trap();
    fused_route_tail<TileN>(logits_shared, bias_shared, live_m, topk_ids,
        topk_weights, counts, offsets, sorted_rows, num_padded_rows,
        active_experts, scaling_factor);
  }
}

template<int TileN>
__device__ void run_fragment_router(
    const uint8_t* weight, const uint16_t* activation,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual) {
  run_fragment_router_impl<TileN, false>(weight, activation, output, partial,
      semaphores, live_m, n, k, splits, output_f32, scales, residual,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, 1.0f);
}

template<int TileN>
__device__ void run_fragment_router_route(
    const uint8_t* weight, const uint16_t* activation, float* logits,
    float* partial, uint32_t* semaphores, const int32_t* live_tokens,
    const float* bias, int32_t* topk_ids, float* topk_weights,
    int32_t* counts, int32_t* offsets, int32_t* sorted_rows,
    int32_t* num_padded_rows, int32_t* active_experts,
    float scaling_factor) {
  run_fragment_router_impl<TileN, true>(weight, activation, logits, partial,
      semaphores, TileN, 256, 4096, kRouterSplits, 1, nullptr, nullptr,
      live_tokens,
      bias, topk_ids, topk_weights, counts, offsets, sorted_rows,
      num_padded_rows, active_experts, scaling_factor);
}

} // namespace chop::gemm
