#pragma once
#include "router_finish.cuh"

// Lab variants of the router epilogue with the same bits and less serial
// latency. Variant bits: 1 issues every chain's partial load before the
// first add (same adds, same order); 2 scans expert counts in every warp,
// then adds the earlier warps' totals (exact integers, same order); 4
// computes each expert's transform and rank once in its own thread, then
// selects on warp 0 with the unchanged rounds. Variant 0 is the served
// body. Stamped builds record clock64 at phase boundaries and globaltimer
// at entry and exit in thread 0 of every CTA: 8 words per CTA.
namespace chop::moe::router_finish_r2 {

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// select_route_warp after its transform loop, reading each expert's
// unbiased value and rank from shared memory.
template<int Experts, int TopK, class Score>
__device__ void select_scored(const float* unbiased_shared, const float* rank_shared, int token,
                              Score score, int32_t* ids, float* weights) {
  const int lane = threadIdx.x & 31;
  float scores[Experts / 32], unbiased[Experts / 32];
#pragma unroll
  for (int j = 0; j < Experts / 32; ++j) {
    unbiased[j] = unbiased_shared[lane + j * 32];
    scores[j] = rank_shared[lane + j * 32];
  }
  mask_route_groups<Experts, 1, 1>(scores);
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

// router_finish::metadata_after_acquire with the offset, active and queue
// scan split over the expert warps.
template<int Experts, int TopK, int Capacity>
__device__ void metadata_parallel(const int32_t* ids, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* padded, const int32_t* live_tokens,
    int32_t* active, int32_t* table, uint32_t* gate_tickets) {
  static_assert(Experts % 32 == 0 && Capacity <= 64 && TopK <= 8);
  constexpr int Groups = (Capacity * TopK + 31) / 32;
  constexpr int Threads = Experts > Groups * 32 ? Experts : Groups * 32;
  constexpr int ExpertWarps = Experts / 32;
  __shared__ uint16_t group_start[Groups][Experts];
  __shared__ int32_t expert_count[Experts], expert_first[Experts], active_ids[Experts];
  __shared__ int32_t warp_total[ExpertWarps][3];
  __shared__ int active_count;
  if (blockDim.x != Threads) __trap();
  const int live = *live_tokens;
  if (live < 0 || live > Capacity) __trap();
  const int routes = live * TopK, groups = (routes + 31) / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int at = threadIdx.x; at < groups * Experts; at += Threads)
    (&group_start[0][0])[at] = 0;
  for (int at = threadIdx.x; at < Capacity; at += Threads) gate_tickets[at] = 0;
  __syncthreads();
  int expert = -1, before = 0;
  if (warp < groups) {
    const int route = warp * 32 + lane;
    expert = route < routes ? __ldcg(ids + route) : -1;
    if (expert >= Experts || expert < -1) __trap();
    const unsigned same = __match_any_sync(0xffffffffu, expert >= 0 ? expert : -1 - lane);
    before = __popc(same & ((1u << lane) - 1u));
    if (expert >= 0 && before == 0) group_start[warp][expert] = uint16_t(__popc(same));
  }
  __syncthreads();
  for (int e = threadIdx.x; e < Experts; e += Threads) {
    int count = 0;
    for (int g = 0; g < groups; ++g) {
      const int n = group_start[g][e];
      group_start[g][e] = uint16_t(count);
      count += n;
    }
    if (count > live) __trap();
    expert_count[e] = count;
    counts[e] = count;
  }
  __syncthreads();
  int a = 0, s = 0, w = 0, valid = 0, is_small = 0, is_wide = 0, count = 0;
  const int e = warp * 32 + lane;
  if (warp < ExpertWarps) {
    count = expert_count[e];
    valid = count > 0; is_small = valid && count <= 8; is_wide = count > 8;
    a = valid; s = is_small; w = is_wide;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const int ua = __shfl_up_sync(0xffffffffu, a, d);
      const int us = __shfl_up_sync(0xffffffffu, s, d);
      const int uw = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) { a += ua; s += us; w += uw; }
    }
    if (lane == 31) { warp_total[warp][0] = a; warp_total[warp][1] = s; warp_total[warp][2] = w; }
  }
  __syncthreads();
  if (warp < ExpertWarps) {
    int carry = 0, small = 0, wide = 0;
    for (int u = 0; u < warp; ++u) {
      carry += warp_total[u][0]; small += warp_total[u][1]; wide += warp_total[u][2];
    }
    expert_first[e] = offsets[e] = (carry + a - valid) * 64;
    if (valid) active[1 + carry + a - 1] = active_ids[carry + a - 1] = e;
    if (is_small) {
      const int at = 2 + 2 * (small + s - 1);
      table[at] = e; table[at + 1] = count;
    }
    if (is_wide) {
      const int at = 2 + 2 * Experts + 2 * (wide + w - 1);
      table[at] = e; table[at + 1] = count;
    }
    if (warp == ExpertWarps - 1 && lane == 31) {
      const int total = carry + a;
      offsets[Experts] = *padded = total * 64;
      active[0] = active_count = total;
      table[0] = small + s; table[1] = wide + w;
    }
  }
  __syncthreads();
  if (expert >= 0)
    sorted_rows[expert_first[expert] + group_start[warp][expert] + before] = warp * 32 + lane;
  for (int at = threadIdx.x; at < active_count * 64; at += Threads) {
    const int x = active_ids[at / 64], row = at % 64;
    if (row >= expert_count[x]) sorted_rows[expert_first[x] + row] = -1;
  }
}

template<int Experts, int TopK, int K, int Variant, bool Stamped>
__device__ void finish(const float* workspace, float* logits, const int32_t* live_tokens,
    uint32_t rows, const float* bias, const float* alternate_bias,
    const uint8_t* bias_select, float* scores, int32_t* ids, float* weights,
    float scaling, int32_t* counts, int32_t* offsets, int32_t* sorted,
    int32_t* padded, int32_t* active, void* scratch, uint32_t* gate_tickets,
    uint32_t* arrived, uint64_t* stamps) {
  constexpr int Threads = Experts > 64 * TopK ? Experts : 64 * TopK;
  constexpr int Chains = (K + 127) / 128;
  constexpr bool AllLoads = Variant & 1, ParallelScan = Variant & 2, ParallelScores = Variant & 4;
  static_assert(Experts % 32 == 0 && TopK <= 8 && Variant >= 0 && Variant < 8);
  if (rows == 0 || rows > 32 || gridDim.x != rows || blockDim.x != Threads) __trap();
  ptx::griddep_wait();
  uint64_t stamp[5] = {0, 0, 0, 0, 0}, entered = 0;
  if constexpr (Stamped) {
    if (threadIdx.x == 0) { entered = globaltimer(); stamp[0] = clock64(); }
  }
  const int live = *live_tokens, token = blockIdx.x;
  if (live < 0 || live > int(rows)) __trap();
  __shared__ float row_logits[Experts];
  __shared__ float row_unbiased[ParallelScores ? Experts : 1], row_rank[ParallelScores ? Experts : 1];
  __shared__ uint32_t last;
  const float* partial = workspace + 32;
  if (token < live) {
    for (int expert = threadIdx.x; expert < Experts; expert += Threads) {
      float sum = 0.f;
      if constexpr (AllLoads) {
        float value[Chains];
#pragma unroll
        for (int b = 0; b < Chains; ++b)
          value[b] = partial[(uint64_t(b)*rows+token)*Experts+expert];
#pragma unroll
        for (int b = 0; b < Chains; ++b) sum = __fadd_rn(sum, value[b]);
      } else {
#pragma unroll 1
        for (int base = 0; base < Chains; base += 4) {
          float value[4];
#pragma unroll
          for (int b = 0; b < 4; ++b)
            value[b] = base + b < Chains
                ? partial[(uint64_t(base+b)*rows+token)*Experts+expert] : 0.f;
#pragma unroll
          for (int b = 0; b < 4; ++b)
            if (base + b < Chains) sum = __fadd_rn(sum, value[b]);
        }
      }
      row_logits[expert] = sum;
      logits[uint64_t(token)*Experts+expert] = sum;
    }
  }
  __syncthreads();
  if constexpr (Stamped) {
    if (threadIdx.x == 0) stamp[1] = clock64();
  }
  floor::RowScore<Experts> score{{bias,alternate_bias,bias_select,scores,scaling},token};
  if constexpr (ParallelScores) {
    if (token < live && threadIdx.x < Experts) {
      const float value = score.transform(row_logits[threadIdx.x]);
      row_unbiased[threadIdx.x] = value;
      row_rank[threadIdx.x] = score.rank(value, threadIdx.x);
    }
    __syncthreads();
    if (token < live && threadIdx.x < 32)
      select_scored<Experts,TopK>(row_unbiased,row_rank,0,score,ids + token * TopK,
          weights + token * TopK);
  } else {
    if (token < live && threadIdx.x < 32)
      select_route_warp<Experts,TopK>(row_logits,0,score,ids + token * TopK,
          weights + token * TopK);
  }
  __syncthreads();
  if constexpr (Stamped) {
    if (threadIdx.x == 0) stamp[2] = clock64();
  }
  if (threadIdx.x == 0) {
    uint32_t previous;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(previous) : "l"(arrived) : "memory");
    last = previous == rows - 1;
  }
  __syncthreads();
  if constexpr (Stamped) {
    if (threadIdx.x == 0) stamp[3] = clock64();
  }
  if (last) {
    if constexpr (ParallelScan)
      metadata_parallel<Experts,TopK,64>(ids,counts,offsets,sorted,padded,
          live_tokens,active,static_cast<int32_t*>(scratch),gate_tickets);
    else
      router_finish::metadata_after_acquire<Experts,TopK,64>(ids,counts,offsets,sorted,padded,
          live_tokens,active,static_cast<int32_t*>(scratch),gate_tickets);
  }
  if constexpr (Stamped) {
    __syncthreads();
    if (threadIdx.x == 0) {
      stamp[4] = clock64();
      uint64_t* out = stamps + uint64_t(blockIdx.x) * 8;
#pragma unroll
      for (int i = 0; i < 5; ++i) out[i] = stamp[i];
      out[5] = entered;
      out[6] = globaltimer();
      out[7] = last;
    }
  }
  // No early launch signal: consumers may read the completed queue header
  // before their PDL wait. Empty-live graphs still publish an empty header.
}
} // namespace chop::moe::router_finish_r2
